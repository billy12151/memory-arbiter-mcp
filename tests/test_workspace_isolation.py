"""Workspace three-tier isolation + alias canonicalization (none/weak/strict).

Design (see memory: workspace-isolation-design):
  - none   : an explicit workspace scopes that query; omission spans the library;
             no ranking effect; new ws silent.
  - weak   : recall always whole-DB (never filtered); passing ws only soft-reranks
             (same-ws boost, cross-ws penalty). New ws → write_hint.
  - strict : write requires ws (empty → error); recall without ws → error;
             recall with ws → hard-filter to same canonical. New ws → blocked as
             status=pending + action_required=confirm_new_workspace; activated via
             memory_activate(authorized=true).

Alias canonicalization (double-store): memories.workspace (raw) +
memories.workspace_canonical (resolved). Runs in every isolation mode; without an
embedder it degrades to exact string identity.
"""
from pathlib import Path
import hashlib

import pytest

from memory_arbiter.acl import raw_workspace
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryStatus
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path, isolation: str = "none", *, vec: bool = False) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "iso.sqlite3",
        backup_jsonl=tmp_path / "iso.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        enable_sqlite_vec=vec,
        vec_dim=2,
        isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write(tools: MemoryTools, content: str, workspace: str = "default", subject: str = "test", **kw) -> dict:
    return tools.memory_write(content=content, workspace=workspace, source_type="agent_generated", subject=subject, **kw)


def _results(search: dict) -> list:
    return (search.get("data") or {}).get("results") or []


def _confirm_pending(tools: MemoryTools, memory_id: int) -> dict:
    record = tools.db.get_memory(memory_id)
    if record["status"] != MemoryStatus.PENDING.value:
        return {"ok": True, "data": {"record": record}}
    return tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id,
        "canonical": record["workspace_canonical"] or record["workspace"],
        "authorized": True,
    })


def _active_write(tools: MemoryTools, content: str, workspace: str, subject: str = "test", **kw) -> int:
    mid = _write(tools, content, workspace, subject=subject, **kw)["data"]["id"]
    assert _confirm_pending(tools, mid)["ok"] is True
    return mid


def _record_group(tools: MemoryTools, member_ids: list[int], *, conflict_point: str = "database") -> int:
    members = []
    groups = []
    for index, memory_id in enumerate(member_ids):
        record = tools.db.get_memory(memory_id)
        value = f"value-{index}"
        quote = record["content"]
        members.append(ConflictMember(
            memory_id=memory_id, version=record["version"], attribute_raw="database", value_raw=value,
            normalized_attribute="database", normalized_value=value, evidence_quote=quote,
            evidence_span=(0, len(quote)), content_hash=hashlib.sha256(quote.encode()).hexdigest(),
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ))
        groups.append(ConflictValueGroup(value, value, (f"{memory_id}@{record['version']}",)))
    result = tools.db.record_conflict_group(
        workspace_canonical=raw_workspace(tools.db.get_memory(member_ids[0])),
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=members, value_groups=groups, detection_reason="different values",
        source="scan", detector_version="d1", conflict_point=conflict_point,
    )
    return result["conflict_id"]


# ── v0.12.5 strict read ACL ────────────────────────────────────────────────


def test_strict_memory_get_by_id_filters_workspace(tmp_path):
    tools = make_tools(tmp_path, "strict")
    a_id = _active_write(tools, "alpha private", "projA")
    b_id = _active_write(tools, "beta private", "projB")

    ok = tools.memory_get(memory_id=a_id, workspace="projA")
    denied = tools.memory_get(memory_id=b_id, workspace="projA")

    assert ok["ok"] is True and ok["data"]["memory"]["id"] == a_id
    assert denied["ok"] is False
    assert denied["data"]["workspace_source"] == "explicit"


def test_strict_history_filters_workspace(tmp_path):
    tools = make_tools(tmp_path, "strict")
    b_id = _active_write(tools, "beta private", "projB")

    r = tools.memory_history(memory_id=b_id, workspace="projA")

    assert r["ok"] is False
    assert r["data"]["workspace_source"] == "explicit"


def test_strict_conflict_creation_rejects_cross_workspace_members(tmp_path):
    tools = make_tools(tmp_path, "strict")
    a_id = _active_write(tools, "alpha visible text", "projA")
    b_id = _active_write(tools, "beta hidden secret", "projB")
    before = tools.db.list_conflicts("open", 100)

    members = []
    groups = []
    for index, memory_id in enumerate((a_id, b_id)):
        record = tools.db.get_memory(memory_id)
        value = f"value-{index}"
        quote = record["content"]
        members.append(ConflictMember(
            memory_id=memory_id, version=record["version"], attribute_raw="database", value_raw=value,
            normalized_attribute="database", normalized_value=value, evidence_quote=quote,
            evidence_span=(0, len(quote)), content_hash=hashlib.sha256(quote.encode()).hexdigest(),
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ))
        groups.append(ConflictValueGroup(value, value, (f"{memory_id}@{record['version']}",)))
    result = tools.db.record_conflict_group(
        workspace_canonical="projA", slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=members, value_groups=groups, detection_reason="secret point",
        source="scan", detector_version="d1",
    )
    assert result["outcome"] == "workspace_mismatch"
    assert tools.db.list_conflicts("open", 100) == before


def test_strict_console_memory_detail_uses_workspace_acl(tmp_path):
    from memory_arbiter.console_api import ConsoleAPI

    tools = make_tools(tmp_path, "strict")
    b_id = _active_write(tools, "beta console secret", "projB")
    api = ConsoleAPI(tools=tools)

    denied = api.memory_detail(b_id, workspace="projA")
    ok = api.memory_detail(b_id, workspace="projB")

    assert denied["_http_status"] == 404
    assert "beta console secret" not in str(denied)
    assert ok["memory"]["id"] == b_id


def test_strict_search_conflict_signal_hides_cross_workspace_group_existence(tmp_path):
    tools = make_tools(tmp_path, "strict")
    a_id = _active_write(tools, "alpha searchable conflict", "projA", subject="same")
    b_id = _active_write(tools, "beta hidden conflict", "projA", subject="same")
    _record_group(tools, [a_id, b_id])
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET workspace='projB',workspace_canonical='projB' WHERE id=?", (b_id,))

    r = tools.memory_search(query="searchable", workspace="projA", limit=10)
    assert "conflict_signal" not in r["data"]["results"][0]
    assert "beta hidden conflict" not in str(r)


def test_strict_conflict_signal_redacts_legacy_empty_workspace_peer(tmp_path):
    tools = make_tools(tmp_path, "strict")
    a_id = _active_write(tools, "alpha legacy peer search", "projA", subject="same")
    b_id = _active_write(tools, "beta legacy secret peer", "projA", subject="same")
    _record_group(tools, [a_id, b_id], conflict_point="LEGACY_SECRET_POINT")
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET workspace='', workspace_canonical=NULL WHERE id=?", (b_id,))

    r = tools.memory_search(query="legacy peer search", workspace="projA", limit=10)
    assert "conflict_signal" not in r["data"]["results"][0]
    assert "LEGACY_SECRET_POINT" not in str(r)
    assert "beta legacy secret" not in str(r)


def test_strict_expired_search_does_not_attach_conflict_signal(tmp_path):
    tools = make_tools(tmp_path, "strict")
    a_id = _active_write(tools, "alpha expired searchable", "projA", subject="same")
    b_id = _active_write(tools, "beta expired hidden", "projA", subject="same")
    _record_group(tools, [a_id, b_id], conflict_point="EXPIRED_SECRET_POINT")
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET workspace='projB',workspace_canonical='projB' WHERE id=?", (b_id,))
    tools.db.update_memory(a_id, {"status": "superseded"})

    r = tools.memory_search_expired(query="expired searchable", workspace="projA", limit=10)
    result = r["data"]["results"][0]

    assert "conflict_signal" not in result
    assert "EXPIRED_SECRET_POINT" not in str(r)



def test_strict_memory_recent_and_compare_filter_workspace(tmp_path):
    tools = make_tools(tmp_path, "strict")
    a_id = _active_write(tools, "alpha recent", "projA")
    b_id = _active_write(tools, "beta recent", "projB")

    recent = tools.memory_recent(workspace="projA", limit=10)
    compare = tools.memory_compare(left_id=a_id, right_id=b_id, workspace="projA")

    assert [m["id"] for m in recent["data"]["results"]] == [a_id]
    assert compare["ok"] is False
    assert "beta recent" not in str(compare)



def test_strict_memory_edit_filters_workspace_before_write(tmp_path):
    tools = make_tools(tmp_path, "strict")
    b_id = _active_write(tools, "beta edit secret", "projB")

    denied = tools.memory_edit(memory_id=b_id, new_content="attacker edit", workspace="projA")
    unchanged = tools.memory_get(memory_id=b_id, workspace="projB")

    assert denied["ok"] is False
    assert unchanged["data"]["memory"]["content"] == "beta edit secret"
    assert "attacker edit" not in str(unchanged)



def test_strict_memory_arbitrate_requires_both_visible(tmp_path):
    tools = make_tools(tmp_path, "strict")
    a_id = _active_write(tools, "alpha arbitrate", "projA")
    b_id = _active_write(tools, "beta hidden arbitrate", "projB")

    r = tools.memory_arbitrate(a_id, b_id, mark_conflict=False, workspace="projA")

    assert r["ok"] is False
    assert "beta hidden arbitrate" not in str(r)



# ── Config ────────────────────────────────────────────────────────────────

def test_isolation_default_is_none(tmp_path):
    assert make_tools(tmp_path).settings.isolation == "none"


def test_isolation_invalid_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("MEMORY_ARBITER_ISOLATION", "bogus")
    s = Settings.from_env()
    assert s.isolation == "none"
    assert any("isolation=" in w for w in s.config_warnings)


def test_memory_status_echoes_isolation(tmp_path):
    tools = make_tools(tmp_path, "weak")
    assert tools.memory_status()["data"]["isolation"] == "weak"


# ── none: normalization without ACL; explicit filter still scopes ──────────

def test_none_write_normalizes_with_nonblocking_new_workspace_notice(tmp_path):
    """none stays ACL-free but makes first-time workspace registration visible."""
    tools = make_tools(tmp_path, "none")
    r = _write(tools, "alpha content", "projA")
    # Canonical falls back to raw and does not block the write.
    assert r["data"].get("action_required") is None
    hint = (r["data"].get("write_hints") or {}).get("new_workspace_detected")
    assert hint == {"canonical": "projA", "similar_workspaces": []}
    notice = next(item for item in r["notices"] if item["type"] == "workspace_review")
    assert notice["workspace"] == "projA"
    assert notice["action_required"] == "review_workspace_registry"
    assert notice["review_call"] == {
        "tool": "memory_review", "view": "doctor", "data": {},
    }
    assert notice["authorization_required"] is True
    assert "authorized" not in notice["confirm_call"]["data"]

    repeated = _write(tools, "second content", "projA")
    assert not any(item.get("type") == "workspace_review" for item in repeated.get("notices", []))


def test_none_explicit_workspace_filter_scopes_results(tmp_path):
    """Spec §15.6: an explicit none-mode workspace filter canonicalizes then
    filters; an omitted workspace still spans all workspaces."""
    tools = make_tools(tmp_path, "none")
    _write(tools, "alpha marketing content", "projA")
    _write(tools, "beta marketing content", "projB")
    with_ws = _results(tools.memory_search(query="marketing", workspace="projA", limit=10))
    without = _results(tools.memory_search(query="marketing", limit=10))
    # Explicit filter scopes to that workspace only; no filter spans all.
    assert [m["workspace"] for m in with_ws] == ["projA"]
    assert sorted(m["workspace"] for m in without) == ["projA", "projB"]


def test_none_explicit_workspace_filter_scopes_empty_query_pagination(tmp_path):
    tools = make_tools(tmp_path, "none")
    for index in range(3):
        _write(tools, f"alpha browse {index}", "projA")
        _write(tools, f"beta browse {index}", "projB")

    first = tools.memory_search(query="", workspace="projA", limit=2)
    second = tools.memory_search(query="", workspace="projA", limit=2, offset=2)

    assert [m["workspace"] for m in _results(first)] == ["projA", "projA"]
    assert [m["workspace"] for m in _results(second)] == ["projA"]
    assert first["data"]["total_estimate"] == 3
    assert first["data"]["has_more"] is True
    assert second["data"]["has_more"] is False


def test_none_explicit_workspace_filter_scopes_recent_fallback(tmp_path):
    tools = make_tools(tmp_path, "none")
    _write(tools, "alpha recent", "projA")
    _write(tools, "beta recent", "projB")

    result = tools.memory_search(query="no-direct-hit-zzzz", workspace="projA", limit=10)

    assert result["data"]["retrieval_mode"] == "recent_fallback"
    assert [m["workspace"] for m in _results(result)] == ["projA"]
    assert result["data"]["total_estimate"] == 1


def test_none_explicit_workspace_filter_scopes_expired_paths(tmp_path):
    tools = make_tools(tmp_path, "none")
    alpha_ids = [_write(tools, f"alpha archived {index}", "projA")["data"]["id"] for index in range(3)]
    beta_id = _write(tools, "beta archived", "projB")["data"]["id"]
    for memory_id in [*alpha_ids, beta_id]:
        tools.memory_supersede(memory_id=memory_id, reason="archived", authorized=True)

    direct = tools.memory_search_expired(query="archived", workspace="projA", limit=10)
    fallback = tools.memory_search_expired(query="no-direct-hit-zzzz", workspace="projA", limit=2, offset=1)

    assert [m["workspace"] for m in _results(direct)] == ["projA", "projA", "projA"]
    assert [m["workspace"] for m in _results(fallback)] == ["projA", "projA"]
    assert fallback["data"]["retrieval_mode"] == "recent_fallback"
    assert fallback["data"]["total_estimate"] == 3
    assert fallback["data"]["has_more"] is False


# ── strict: mandatory ws, hard filter, blocking new ws ──────────────────────

def test_strict_write_without_workspace_errors(tmp_path):
    tools = make_tools(tmp_path, "strict")
    r = tools.memory_write(content="no ws", source_type="agent_generated", workspace="", subject="test")
    assert r["ok"] is False
    assert "strict" in (r["data"].get("error") or "").lower()


def test_strict_recall_without_workspace_uses_settings_workspace(tmp_path):
    tools = make_tools(tmp_path, "strict")
    tools.memory_write(content="x", workspace="projA", source_type="agent_generated", subject="test")
    r = tools.memory_search(query="x", limit=10)
    assert r["ok"] is True
    assert r["data"]["workspace_source"] == "settings"
    assert r["data"]["caller_workspace"] == "default"
    assert r["data"]["results"] == []


def test_strict_expired_recall_without_workspace_uses_settings_workspace(tmp_path):
    tools = make_tools(tmp_path, "strict")
    _write(tools, "pending alpha", "projA")

    r = tools.memory_search_expired(query="pending", limit=10)

    assert r["ok"] is True
    assert r["data"]["workspace_source"] == "settings"
    assert r["data"]["caller_workspace"] == "default"
    assert r["data"]["results"] == []


def test_strict_expired_recall_hard_filters_workspace(tmp_path):
    tools = make_tools(tmp_path, "strict")
    a_id = _write(tools, "pending shared alpha", "projA")["data"]["id"]
    _write(tools, "pending shared beta", "projB")

    r = tools.memory_search_expired(query="pending shared", workspace="projA", limit=10)

    assert r["ok"] is True
    assert [(m["id"], m["workspace"]) for m in _results(r)] == [(a_id, "projA")]


def test_strict_new_workspace_blocks_as_pending(tmp_path):
    tools = make_tools(tmp_path, "strict")
    r = _write(tools, "first in new ws", "projA")
    d = r["data"]
    assert d.get("action_required") == "confirm_new_workspace"
    assert d.get("verification_status") == "pending_user"
    assert (d.get("record") or {}).get("status") == MemoryStatus.PENDING.value
    assert not any(item.get("type") == "workspace_review" for item in r.get("notices", []))


def test_strict_pending_requires_workspace_confirmation(tmp_path):
    tools = make_tools(tmp_path, "strict")
    r = _write(tools, "pending content", "projA")
    mid = r["data"]["id"]
    assert _results(tools.memory_search(query="pending", workspace="projA")) == []
    blocked = tools.memory_activate(memory_id=mid, authorized=True, workspace="projA")
    assert blocked["ok"] is False
    assert blocked["data"]["action_required"] == "confirm_new_workspace"
    confirmed = _confirm_pending(tools, mid)
    assert confirmed["ok"] is True
    assert tools.db.get_memory(mid)["status"] == MemoryStatus.ACTIVE.value
    assert len(_results(tools.memory_search(query="pending", workspace="projA"))) == 1


def test_memory_activate_requires_authorized(tmp_path):
    tools = make_tools(tmp_path, "strict")
    mid = _write(tools, "c", "projA")["data"]["id"]
    r = tools.memory_activate(memory_id=mid, authorized=False)
    assert r["ok"] is False and r["data"]["activated"] is False


def test_memory_activate_rejects_non_pending(tmp_path):
    tools = make_tools(tmp_path, "none")
    mid = _write(tools, "active mem", "projA")["data"]["id"]
    r = tools.memory_activate(memory_id=mid, authorized=True)
    assert r["ok"] is False


def test_memory_activate_rechecks_status_inside_write_transaction(tmp_path, monkeypatch):
    tools = make_tools(tmp_path, "none")
    mid = _write(tools, "pending", "projA", status="pending")["data"]["id"]
    stale_pending = tools.db.get_memory(mid)
    assert stale_pending["status"] == MemoryStatus.PENDING.value
    assert tools.db.update_memory(mid, {"status": "superseded"}) is True

    # Old code trusted this pre-transaction snapshot and could reactivate it.
    monkeypatch.setattr(tools, "_get_memory_visible", lambda *_args, **_kwargs: stale_pending)
    rejected = tools.memory_activate(memory_id=mid, authorized=True)

    assert rejected["ok"] is False
    assert "not pending" in rejected["data"]["error"]
    assert tools.db.get_memory(mid)["status"] == "superseded"


def test_strict_mutations_recheck_workspace_inside_write_transaction(tmp_path, monkeypatch):
    tools = make_tools(tmp_path, "strict")
    ids = [
        _active_write(tools, f"content {index}", "projA", subject=f"s{index}")
        for index in range(5)
    ]
    history_id = ids[4]
    assert tools.memory_edit(
        memory_id=history_id, new_content="content 4 edited", workspace="projA",
    )["ok"] is True
    stale = {memory_id: tools.db.get_memory(memory_id) for memory_id in ids}
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET workspace_canonical='projB' WHERE id IN (?,?,?,?,?)",
            ids,
        )

    monkeypatch.setattr(
        tools, "_get_memory_visible",
        lambda memory_id, _caller=None: stale.get(int(memory_id)),
    )
    outcomes = [
        tools.memory_edit(memory_id=ids[0], new_content="forbidden", workspace="projA"),
        tools.memory_edit(
            memory_id=ids[1], tags_only=True, add_tags=["forbidden"], workspace="projA",
        ),
        tools.memory_set_entity(memory_id=ids[2], entity="forbidden", workspace="projA"),
        tools.memory_supersede(
            memory_id=ids[3], reason="forbidden", authorized=True, workspace="projA",
        ),
        tools.memory_cleanup_history(
            memory_id=history_id, authorized=True, workspace="projA",
        ),
    ]

    assert all(result["ok"] is False for result in outcomes)
    assert all(result["data"]["error"] == "forbidden_strict_workspace" for result in outcomes)
    assert tools.db.get_memory(ids[0])["content"] == "content 0"
    assert "forbidden" not in tools.db.get_memory(ids[1])["tags"]
    assert "entity" not in tools.db.get_memory(ids[2])["metadata"]
    assert tools.db.get_memory(ids[3])["status"] == "active"
    assert len(tools.db.list_history(history_id)) == 1


def test_strict_hard_filter_same_canonical_only(tmp_path):
    tools = make_tools(tmp_path, "strict")
    # activate both distinct workspaces
    for ws, content in [("projA", "apple content"), ("projB", "apple content two")]:
        mid = _write(tools, content, ws)["data"]["id"]
        assert _confirm_pending(tools, mid)["ok"] is True
    res = _results(tools.memory_search(query="apple", workspace="projA", limit=10))
    assert {m["workspace"] for m in res} == {"projA"}


def test_strict_no_leak_when_query_hits_other_workspace(tmp_path):
    """v0.9.7 regression: searching ws-A for a term that only exists in ws-B
    must return nothing — NOT fall back to whole-DB recent memories.

    Before the fix, the strict pool post-filter left `pool` empty, so
    search_memories fell through to _recent_fallback WITHOUT ws_canonical and
    leaked ws-B's memory into a ws-A search (retrieval_mode=recent_fallback).
    """
    tools = make_tools(tmp_path, "strict")
    a_mid = _write(tools, "alpha deployment notes", "projA")["data"]["id"]
    b_mid = _write(tools, "beta release notes uniquebeta", "projB")["data"]["id"]
    assert _confirm_pending(tools, a_mid)["ok"] is True
    assert _confirm_pending(tools, b_mid)["ok"] is True
    # 'uniquebeta' exists only in projB; a projA search must stay empty.
    res = _results(tools.memory_search(query="uniquebeta", workspace="projA", limit=10))
    assert res == [], f"strict leaked cross-workspace memory: {[m['workspace'] for m in res]}"


def test_strict_no_leak_when_query_matches_nothing(tmp_path):
    """v0.9.7 regression: a query matching nothing in any workspace must not
    fall back to whole-DB recent memories under strict isolation.

    Before the fix, an empty pool triggered _recent_fallback with
    ws_canonical=None, returning every workspace's recent memories.
    """
    tools = make_tools(tmp_path, "strict")
    a_mid = _write(tools, "alpha deployment notes", "projA")["data"]["id"]
    b_mid = _write(tools, "beta release notes", "projB")["data"]["id"]
    assert _confirm_pending(tools, a_mid)["ok"] is True
    assert _confirm_pending(tools, b_mid)["ok"] is True
    res = _results(tools.memory_search(query="nomatchterm_zzz", workspace="projA", limit=10))
    assert res == [], f"strict leaked cross-workspace memory on no-match: {[m['workspace'] for m in res]}"


def test_strict_attention_summary_names_confirm_pending_workspace(tmp_path):
    tools = make_tools(tmp_path, "strict")
    r = _write(tools, "first in new ws", "projA")
    summary = r["data"].get("attention_summary") or ""
    assert "confirm_pending_workspace" in summary, f"attention_summary points at wrong tool: {summary!r}"


def test_strict_bm25_direct_hits_are_workspace_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_ARBITER_RANKING_MODE", "bm25")
    tools = make_tools(tmp_path, "strict")
    for ws, content in [("projA", "apple alpha"), ("projB", "apple beta")]:
        mid = _write(tools, content, ws)["data"]["id"]
        assert _confirm_pending(tools, mid)["ok"] is True

    res = _results(tools.memory_search(query="apple", workspace="projA", limit=10))

    assert [(m["workspace"], m["content"]) for m in res] == [("projA", "apple alpha")]


def test_strict_recall_pool_saturation_does_not_hide_same_workspace_hit(tmp_path):
    tools = make_tools(tmp_path, "strict")
    tools.settings.recall_pool_cap = 50
    for i in range(60):
        mid = _write(tools, f"apple shared beta {i:02d}", "projB")["data"]["id"]
        assert _confirm_pending(tools, mid)["ok"] is True
    expected_id = _write(tools, "apple shared alpha should be found", "projA")["data"]["id"]
    assert _confirm_pending(tools, expected_id)["ok"] is True

    res = _results(tools.memory_search(query="apple", workspace="projA", limit=10))

    assert [m["id"] for m in res] == [expected_id]
    assert {m["workspace"] for m in res} == {"projA"}


def test_strict_linked_open_items_are_workspace_scoped(tmp_path):
    tools = make_tools(tmp_path, "strict")
    main_id = _write(tools, "auth bug main", "projA", tags=["auth"])["data"]["id"]
    assert _confirm_pending(tools, main_id)["ok"] is True
    beta_todo_id = _write(tools, "todo in beta", "projB", tags=["todo", "auth"])["data"]["id"]
    assert _confirm_pending(tools, beta_todo_id)["ok"] is True

    res = tools.memory_search(query="auth", workspace="projA", limit=10)

    assert [(m["id"], m["workspace"]) for m in res["data"]["results"]] == [(main_id, "projA")]
    assert res["data"]["linked_open_items"] == []


def test_strict_filter_counts_are_workspace_scoped(tmp_path):
    tools = make_tools(tmp_path, "strict")
    alpha_id = _write(tools, "apple alpha", "projA", tags=["fruit"])["data"]["id"]
    assert _confirm_pending(tools, alpha_id)["ok"] is True
    for i in range(3):
        mid = _write(tools, f"banana beta {i}", "projB", tags=["fruit"])["data"]["id"]
        assert _confirm_pending(tools, mid)["ok"] is True

    res = tools.memory_search(query="apple", workspace="projA", tags_filter=["fruit"], limit=10)

    assert [(m["id"], m["workspace"]) for m in res["data"]["results"]] == [(alpha_id, "projA")]
    assert res["data"]["total_estimate"] == 1
    assert res["data"]["has_more"] is False


# ── weak: whole-DB recall, soft rerank only ─────────────────────────────────

def test_weak_recall_never_filters(tmp_path):
    tools = make_tools(tmp_path, "weak")
    _write(tools, "gamma marketing here", "projA")
    _write(tools, "delta marketing here", "projB")
    # passing projA still returns BOTH (whole-DB), just reranked
    res = _results(tools.memory_search(query="marketing", workspace="projA", limit=10))
    assert len(res) == 2


def test_weak_same_workspace_ranks_first(tmp_path):
    tools = make_tools(tmp_path, "weak")
    # identical relevance content in two workspaces
    _write(tools, "same marketing text", "projB")
    _write(tools, "same marketing text", "projA")
    res = _results(tools.memory_search(query="marketing", workspace="projA", limit=10))
    assert len(res) == 2
    # the projA memory should sort first due to same-ws boost
    assert res[0]["workspace"] == "projA"


def test_weak_new_workspace_emits_hint(tmp_path):
    tools = make_tools(tmp_path, "weak")
    r = _write(tools, "new ws content", "projA")
    hint = (r["data"].get("write_hints") or {}).get("new_workspace_detected")
    assert hint is not None
    assert hint["canonical"] == "projA"
    notice = next(item for item in r["notices"] if item["type"] == "workspace_review")
    assert notice["workspace"] == "projA"


def test_product_remember_preserves_new_workspace_notice(tmp_path):
    tools = make_tools(tmp_path, "none")
    response = tools.memory("remember", {
        "content": "new workspace through product wrapper",
        "subject": "workspace notice",
        "workspace": "projA",
        "source_type": "agent_generated",
    })
    notice = next(item for item in response["notices"] if item["type"] == "workspace_review")
    assert notice["workspace"] == "projA"
    assert notice["action_required"] == "review_workspace_registry"


# ── alias canonicalization degradation (no embedder) ────────────────────────

def test_alias_no_embedder_exact_match(tmp_path):
    """Without GGUF, resolution degrades to exact string identity."""
    tools = make_tools(tmp_path, "weak")
    db = tools.db
    r1 = db.resolve_workspace_canonical("金科营销项目", embedder=None)
    assert r1["canonical"] == "金科营销项目" and r1["is_new"] is True
    # exact repeat is not new
    r2 = db.resolve_workspace_canonical("金科营销项目", embedder=None)
    assert r2["is_new"] is False and r2["matched_by"] == "exact"
    # a different string is a distinct new canonical (no vector merge)
    r3 = db.resolve_workspace_canonical("金营项目", embedder=None)
    assert r3["is_new"] is True


# ── v0.9.7 third-round adversarial: channels the first two rounds didn't cover ──

def test_strict_offset_beyond_same_workspace_set_does_not_leak(tmp_path):
    """strict + offset past the same-ws result window must return empty, not
    spill cross-workspace memories into the page (pagination boundary).

    Tests the search_memories core directly — the memory_search *tool* does
    not expose offset (only memory_search_expired does), but search_memories
    itself supports it, so this guards the slicing + strict-filter interplay."""
    from memory_arbiter.search import search_memories
    tools = make_tools(tmp_path, "strict")
    db = tools.db
    for i in range(3):
        mid = _write(tools, f"apple alpha {i}", "projA")["data"]["id"]
        assert _confirm_pending(tools, mid)["ok"] is True
    for i in range(5):
        mid = _write(tools, f"apple beta {i}", "projB")["data"]["id"]
        assert _confirm_pending(tools, mid)["ok"] is True
    # offset beyond projA's 3 matches — page must be empty, not projB.
    outcome = search_memories(
        db, "apple", "projA", None, 10,
        status_filter="active", ws_canonical="projA", isolation="strict", offset=3,
    )
    ws = {m["workspace"] for m in outcome.results}
    assert ws == set(), f"offset boundary leaked cross-workspace: {ws}"


def test_strict_alias_canonicalization_keeps_distinct_workspaces_separate(tmp_path):
    """strict + alias resolution (no embedder → exact-string): two distinct
    workspace strings stay distinct; searching one does not leak the other.
    Also covers double-store raw/canonical values and repeated pending writes."""
    tools = make_tools(tmp_path, "strict")
    db = tools.db
    r1 = _write(tools, "apple alpha one", "projA")
    r2 = _write(tools, "apple alpha two", "projA")
    # Every write remains pending until the canonical is explicitly confirmed;
    # an unconfirmed first write does not register a canonical that retries can bypass.
    assert r1["data"].get("action_required") == "confirm_new_workspace"
    assert r2["data"].get("action_required") == "confirm_new_workspace"
    assert _confirm_pending(tools, r1["data"]["id"])["ok"] is True
    r3 = _write(tools, "apple beta three", "projB")
    assert _confirm_pending(tools, r3["data"]["id"])["ok"] is True
    # strict search projA returns only projA's memories.
    res = _results(tools.memory_search(query="apple", workspace="projA", limit=10))
    assert {m["workspace"] for m in res} == {"projA"}
    # double-store: raw and canonical agree for exact-string workspaces.
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT workspace, workspace_canonical FROM memories ORDER BY id"
        ).fetchall()
    assert all(r["workspace"] == r["workspace_canonical"] for r in rows)


def test_weak_linked_open_items_not_over_scoped_by_strict_fix(tmp_path):
    """weak must keep linked_open_items whole-DB (the weak contract). The
    strict ws_canonical scoping must NOT leak into weak mode. A cross-workspace
    todo sharing a tag with the result — but NOT itself matched by the query —
    must still surface as a linked_open_item under weak."""
    tools = make_tools(tmp_path, "weak")
    main_id = _write(tools, "deploy main result notes", "projA", tags=["release", "feature"])["data"]["id"]
    # projB todo: text does NOT contain "deploy", but shares the "release" tag.
    beta = _write(tools, "unrelated text zzz", "projB", tags=["todo", "release"])
    beta_id = beta["data"]["id"]
    res = tools.memory_search(query="deploy", workspace="projA", limit=10)
    assert [m["id"] for m in _results(res)] == [main_id], "weak should still filter results by relevance"
    linked_ids = [li["id"] for li in (res["data"].get("linked_open_items") or [])]
    assert beta_id in linked_ids, (
        f"weak linked_open_items missing cross-workspace todo {beta_id}; got {linked_ids}"
    )


try:
    import sqlite_vec  # type: ignore  # noqa: F401
    _VEC_AVAILABLE = True
except Exception:
    _VEC_AVAILABLE = False


@pytest.mark.skipif(not _VEC_AVAILABLE, reason="sqlite-vec not installed")
def test_strict_evidence_knn_excludes_closer_cross_workspace_vector(tmp_path):
    """Adversarial vector channel: a cross-workspace memory whose vector is
    CLOSER to the query than the same-workspace hit must still be excluded
    under strict. Verifies vec_knn's workspace_predicate is wired and its
    parameter order is correct across all three SQL branches."""
    tools = make_tools(tmp_path, "strict", vec=True)
    db = tools.db
    # query vector [1.0, 0.0]; projA same-ws [0.9, 0.1] (L2~0.14);
    # projB cross-ws [1.0, 0.0] (L2=0.0, exact match — closer!).
    a_mid = _write(tools, "alpha same ws", "projA")["data"]["id"]
    assert _confirm_pending(tools, a_mid)["ok"] is True
    b_mid = _write(tools, "beta cross ws exact", "projB")["data"]["id"]
    assert _confirm_pending(tools, b_mid)["ok"] is True
    from memory_arbiter.evidence import EvidenceUnit, evidence_content_hash
    db.evidence.publish(a_mid, 2, evidence_content_hash("alpha same ws"), [EvidenceUnit("text", "alpha same ws", 0, 13, 0)], [[0.9, 0.1]])
    db.evidence.publish(b_mid, 2, evidence_content_hash("beta cross ws exact"), [EvidenceUnit("text", "beta cross ws exact", 0, 19, 0)], [[1.0, 0.0]])
    res = tools.memory_search(query="x", workspace="projA", limit=10, query_embedding=[1.0, 0.0])
    rows = _results(res)
    assert {r["workspace"] for r in rows} == {"projA"}, (
        f"vec_knn leaked closer cross-workspace vector: {[r['workspace'] for r in rows]}"
    )
    # Direct evidence KNN confirms the predicate (not just the search wrapper).
    knn_a = db.evidence_knn([1.0, 0.0], k=10, parent_status_filter="active", workspace="projA")
    assert all(r.get("workspace") == "projA" for r in knn_a)
    knn_b = db.evidence_knn([1.0, 0.0], k=10, parent_status_filter="active", workspace="projB")
    assert all(r.get("workspace") == "projB" for r in knn_b)


# ── empty/default workspace placement suggestion (2026-08-21) ────────────────
#
# A real-library A/B chose "default + non-binding subject hint": the memory
# still lands in default, but the response suggests the workspace of the nearest
# existing memory by subject. Never auto-assigns (thought-A distance is not
# comparable to the name-vector threshold; global memories belong in default).

class _StubEmbedder:
    embedding_space_id = "stub"
    last_encode_error = None

    @staticmethod
    def embed_text(prefix, body, max_body_chars=None):
        from memory_arbiter.embedder import EmbedResult
        return EmbedResult([1.0, 0.0], False, len(body), len(body))


def _placement_tools(tmp_path):
    t = make_tools(tmp_path, "none")
    t._embedder = _StubEmbedder()
    t._embedder_loaded = True
    t.db.state.sqlite_vec_available = True
    return t


def test_placement_suggestion_for_empty_workspace(tmp_path):
    from memory_arbiter.models import MemoryRecord
    t = _placement_tools(tmp_path)
    # Nearest neighbor lives in a real workspace.
    t.db.evidence_knn = lambda *a, **k: [{"memory_id": 42, "distance": 5.0}]  # type: ignore
    t.db.get_memory = lambda mid: {"id": 42, "status": "active", "workspace": "金营项目",  # type: ignore
                                   "workspace_canonical": "金营项目"} if mid == 42 else None
    rec = MemoryRecord.from_input(
        {"content": "排期正文", "subject": "金营项目 Q3 排期", "workspace": ""},
        t.settings.defaults(),
    )
    ws = t._write_pipeline._resolve_write_workspace(rec)
    assert str(ws["canonical"]).casefold() in {"", "default"}   # NOT auto-assigned
    sug = ws["placement_suggestion"]
    assert sug and sug["suggested_workspace"] == "金营项目"
    assert sug["from_memory_id"] == 42


def test_no_placement_suggestion_when_neighbor_is_default(tmp_path):
    from memory_arbiter.models import MemoryRecord
    t = _placement_tools(tmp_path)
    # Only default-workspace neighbors → global memory stays in default, no hint.
    t.db.evidence_knn = lambda *a, **k: [{"memory_id": 7, "distance": 5.0}]  # type: ignore
    t.db.get_memory = lambda mid: {"id": 7, "status": "active", "workspace": "default",  # type: ignore
                                   "workspace_canonical": "default"} if mid == 7 else None
    rec = MemoryRecord.from_input(
        {"content": "偏好正文", "subject": "今天想吃什么", "workspace": ""},
        t.settings.defaults(),
    )
    ws = t._write_pipeline._resolve_write_workspace(rec)
    assert ws["placement_suggestion"] is None


def test_no_placement_suggestion_without_subject(tmp_path):
    from memory_arbiter.models import MemoryRecord
    t = _placement_tools(tmp_path)
    t.db.evidence_knn = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not embed"))  # type: ignore
    rec = MemoryRecord.from_input(
        {"content": "正文", "subject": "", "workspace": ""},
        t.settings.defaults(),
    )
    ws = t._write_pipeline._resolve_write_workspace(rec)
    assert ws["placement_suggestion"] is None


def test_non_default_workspace_gets_no_placement_suggestion(tmp_path):
    from memory_arbiter.models import MemoryRecord
    t = _placement_tools(tmp_path)
    t.db.evidence_knn = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run for a real ws"))  # type: ignore
    rec = MemoryRecord.from_input(
        {"content": "正文", "subject": "某主题", "workspace": "some-real-project"},
        t.settings.defaults(),
    )
    ws = t._write_pipeline._resolve_write_workspace(rec)
    assert "placement_suggestion" not in ws or ws.get("placement_suggestion") is None
