# ── from test_workspace_isolation.py ──

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
    # vec=True points at a (fake) GGUF model — since 0.15.0 the model path IS
    # the intent to embed — and mirrors the first successful embedder build by
    # creating the lazy vec0 tables at dim 2.
    model = tmp_path / "fake.gguf"
    settings = Settings(
        db_path=tmp_path / "iso.sqlite3",
        backup_jsonl=tmp_path / "iso.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        embedding_model_path=model if vec else None,
        isolation=isolation,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    if vec:
        model.write_bytes(b"fake")
        assert db.ensure_vec_tables(2) == []
    return tools


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


def test_isolation_invalid_falls_back_to_none(tmp_path, monkeypatch):
    # isolation is file-only since 0.15.0 (the env var is gone); an invalid
    # file value still falls back to none with a warning.
    import json

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"isolation": "bogus"}), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg))
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


def test_blank_workspace_write_lands_in_default(tmp_path):
    # from_input does not strip: a whitespace-only workspace is truthy and
    # passes through the whole main path. Resolution must still collapse it to
    # the default pool instead of storing an empty/blank canonical.
    from memory_arbiter.models import MemoryRecord

    t = make_tools(tmp_path)
    rec = MemoryRecord.from_input(
        {"content": "空白 workspace 正文", "subject": "空白边界", "workspace": "   "},
        t.settings.defaults(),
    )
    assert rec.workspace == "   "  # truthy blank survives from_input
    memory_id, warnings = t.db.insert_memory(rec, "")
    assert memory_id is not None
    with t.db.connection() as conn:
        row = conn.execute(
            "SELECT workspace, workspace_canonical FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
    assert row["workspace"] == "   "
    assert row["workspace_canonical"] == "default"


def test_insert_memory_empty_canonical_falls_back_to_default(tmp_path):
    # canonical 空串兜底: an explicit empty canonical with a blank raw workspace
    # must become DEFAULT_WORKSPACE_NAME, never an empty string.
    from memory_arbiter.models import MemoryRecord

    t = make_tools(tmp_path)
    rec = MemoryRecord.from_input(
        {"content": "空 canonical 正文", "subject": "空串兜底", "workspace": ""},
        t.settings.defaults(),
    )
    memory_id, _warnings = t.db.insert_memory(rec, "  ")
    assert memory_id is not None
    with t.db.connection() as conn:
        row = conn.execute(
            "SELECT workspace_canonical FROM memories WHERE id=?", (memory_id,),
        ).fetchone()
        registered = conn.execute(
            "SELECT name FROM workspace_canonicals"
        ).fetchall()
    assert row["workspace_canonical"] == "default"
    assert all(str(r["name"]).strip() for r in registered)


# ── from test_workspace_strict_admission.py ──
# helper make_tools renamed: strict_admission_make_tools (collision with test_workspace_isolation.py)

"""strict 向量准入 (mema 721 期3).

The plan's core trap: strict recall is locked at the SQL layer, so changing
only a Python post-filter is a no-op that still passes naive tests. These
tests therefore assert BOTH halves of the contract:

  - recall widens (a query in workspace A returns memories of the in-radius
    workspace B), AND
  - the same memory is readable by id (memory_read/recent/conflict paths) —
    proving search and ACL share one admitted set, no "搜得到读不到".

Plus the guardrails: default flood insulation, short-name and generic-word
guards, the mema/abbreviation backstop, COUNT/pagination consistency, and the
`workspace_recall_admission=off` full rollback to exact-equality behavior.
"""
import json
from pathlib import Path

import pytest

from memory_arbiter.acl import CallerWorkspace, scope_names, workspace_scope_sql
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import MemoryStatus
from memory_arbiter.tools import MemoryTools

# Two-dimensional unit vectors whose cosine distances are known exactly, so the
# tests pin admission behavior without depending on a real embedding model.
VEC_SELF = [1.0, 0.0]
VEC_NEAR = [0.99, 0.141]      # cosine distance ≈ 0.01 → inside the 0.25 cutoff
VEC_FAR = [0.0, 1.0]          # cosine distance 1.0 → outside any sane cutoff


def strict_admission_make_tools(
    tmp_path: Path,
    *,
    admission: bool = True,
    isolation: str = "strict",
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> MemoryTools:
    if not admission:
        if monkeypatch is None:
            raise TypeError(
                "admission=False requires the monkeypatch fixture "
                "(workspace_recall_admission is a frozen constant since 0.15.0)"
            )
        # The off-rollback is simulated by flipping the frozen constant in the
        # modules that read it at call time.
        monkeypatch.setattr("memory_arbiter.tools.WORKSPACE_RECALL_ADMISSION", False)
        monkeypatch.setattr(
            "memory_arbiter.pipeline.operations.WORKSPACE_RECALL_ADMISSION", False,
        )
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "adm.sqlite3",
        backup_jsonl=tmp_path / "adm.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        embedding_model_path=model,
        isolation=isolation,
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    if not tools.db.state.sqlite_vec_available:  # pragma: no cover - env guard
        pytest.skip("sqlite-vec unavailable")
    # Lazy vec0 tables (0.15.0): created at the model's dim, mirroring the
    # first successful embedder build.
    assert tools.db.ensure_vec_tables(2) == []
    return tools


def publish(tools: MemoryTools, name: str, vector: list[float]) -> None:
    """Register a canonical with a published vector (no embedder needed)."""
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
            (name,),
        )
        row = conn.execute("SELECT id FROM workspace_canonicals WHERE name = ?", (name,)).fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
            (int(row["id"]), json.dumps(vector)),
        )


def active_write(tools: MemoryTools, content: str, workspace: str, subject: str = "test") -> int:
    mid = tools.memory_write(
        content=content, workspace=workspace, subject=subject,
        source_type="agent_generated",
    )["data"]["id"]
    record = tools.db.get_memory(mid)
    if record["status"] == MemoryStatus.PENDING.value:
        confirmed = tools.memory_govern("confirm_pending_workspace", {
            "memory_id": mid,
            "canonical": record["workspace_canonical"] or record["workspace"],
            "authorized": True,
        })
        assert confirmed["ok"] is True
    return mid


def results(search: dict) -> list[int]:
    return [r["id"] for r in (search.get("data") or {}).get("results") or []]


# ── shared SQL/scope helpers ────────────────────────────────────────────────

def test_workspace_scope_sql_collapses_to_equality_for_one_name():
    sql, params = workspace_scope_sql("WS", ["only"])
    assert sql == "WS = ?"
    assert params == ["only"]


def test_workspace_scope_sql_in_clause_and_dedup():
    sql, params = workspace_scope_sql("WS", ["a", "b", "a", "  ", "c"])
    assert sql == "WS IN (?,?,?)"
    assert params == ["a", "b", "c"]
    assert workspace_scope_sql("WS", []) == ("", [])
    assert workspace_scope_sql("WS", None) == ("", [])
    # a bare string is a one-element scope
    assert workspace_scope_sql("WS", "solo") == ("WS = ?", ["solo"])


def test_scope_names_normalizes():
    assert scope_names(None) == []
    assert scope_names("x") == ["x"]
    assert scope_names(["x", " x ", "", "y"]) == ["x", "y"]


def test_caller_workspace_scope_defaults_to_own_canonical():
    caller = CallerWorkspace(isolation="strict", workspace="w", canonical="proj", source="explicit")
    assert caller.scope_canonicals() == ("proj",)
    widened = CallerWorkspace(
        isolation="strict", workspace="w", canonical="proj", source="explicit",
        admitted=("proj", "proj-sibling"),
    )
    assert widened.scope_canonicals() == ("proj", "proj-sibling")


# ── admitted-set computation ────────────────────────────────────────────────

def test_admitted_canonicals_includes_near_excludes_far(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)

    admitted = tools.db.workspaces.admitted_canonicals(
        "agent-lane", cutoff=0.25, min_name_len=3,
    )
    assert admitted[0] == "agent-lane"          # own canonical always first
    assert "agent-rail" in admitted
    assert "unrelated-ws" not in admitted


def test_admitted_canonicals_do_not_truncate_valid_neighbors(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "tenant-root", VEC_SELF)
    for index in range(25):
        publish(tools, f"neighbor-{index:03d}", VEC_SELF)
    admitted = tools.db.workspaces.admitted_canonicals("tenant-root", cutoff=0.25)
    assert len(admitted) == 26
    assert "neighbor-024" in admitted


def test_evidence_knn_does_not_starve_scoped_hit_after_global_2048(tmp_path):
    """A scoped evidence hit after >2048 closer out-of-scope units remains
    reachable; admission is not a fixed global-window post-filter."""
    tools = strict_admission_make_tools(tmp_path)
    far_memory = active_write(tools, "far evidence owner", "unrelated-ws", "far evidence")
    target_memory = active_write(tools, "target admitted evidence", "agent-rail", "target evidence")
    with tools.db.write_transaction() as conn:
        for index in range(2050):
            cur = conn.execute(
                "INSERT INTO memory_evidence(memory_id,memory_version,content_hash,unit_index,kind,text,start_offset,end_offset,created_at) "
                "VALUES(?,1,?,?,?,?,0,1,'2026-01-01T00:00:00Z')",
                (far_memory, "f" * 64, index, "sentence", f"far-{index}"),
            )
            conn.execute(
                "INSERT INTO memory_evidence_vec(id,parent_status,embedding) VALUES(?,'active',?)",
                (int(cur.lastrowid), json.dumps(VEC_SELF)),
            )
        cur = conn.execute(
            "INSERT INTO memory_evidence(memory_id,memory_version,content_hash,unit_index,kind,text,start_offset,end_offset,created_at) "
            "VALUES(?,1,?,0,'sentence','target',0,6,'2026-01-01T00:00:00Z')",
            (target_memory, "t" * 64),
        )
        conn.execute(
            "INSERT INTO memory_evidence_vec(id,parent_status,embedding) VALUES(?,'active',?)",
            (int(cur.lastrowid), json.dumps(VEC_FAR)),
        )
    hits = tools.db.evidence_knn(
        VEC_SELF, k=1, workspace=("agent-lane", "agent-rail"),
    )
    assert [int(hit["memory_id"]) for hit in hits] == [target_memory]


def test_admitted_canonicals_degrades_to_own_canonical(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    # No published vector for the query canonical → degraded to exact scope.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES ('vecless', '2026-01-01T00:00:00Z')",
        )
    publish(tools, "agent-rail", VEC_NEAR)
    assert tools.db.workspaces.admitted_canonicals("vecless", cutoff=0.25) == ("vecless",)
    # default is insulated in both directions
    publish(tools, "default", VEC_SELF)
    assert tools.db.workspaces.admitted_canonicals("default", cutoff=0.25) == ("default",)


def test_admitted_canonicals_applies_short_name_guard(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "w", VEC_SELF)               # 1 char — below min_name_len
    publish(tools, "claw", VEC_NEAR)            # near in vector space
    assert tools.db.workspaces.admitted_canonicals("w", cutoff=0.25) == ("w",)
    # Control: the same near vectors DO admit between two long, unrelated names.
    publish(tools, "alpha-proj", VEC_SELF)
    assert "claw" in tools.db.workspaces.admitted_canonicals("alpha-proj", cutoff=0.25)


def test_admitted_canonicals_applies_generic_substring_guard(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    # Identical vectors (distance 0) but the names share only the substring
    # "main" — the 721 §3d hazard the guard exists for.
    publish(tools, "main", VEC_SELF)
    publish(tools, "openclaw-main", VEC_SELF)
    assert tools.db.workspaces.admitted_canonicals("main", cutoff=0.25) == ("main",)


def test_admitted_canonicals_excludes_default_terms_as_targets(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "默认", VEC_NEAR)             # a legacy synonym canonical
    admitted = tools.db.workspaces.admitted_canonicals("agent-lane", cutoff=0.25)
    assert admitted == ("agent-lane",)


# ── the core adversarial pair: recall AND read widen together ───────────────

def test_strict_admission_widens_recall_and_read(tmp_path):
    """Plan §3 核心验收: strict 下查 agent-lane 能召回 agent-rail 的记忆,
    且该记忆 memory_read 也放行（证明 ACL 同步、非「搜到读不到」）。"""
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "release checklist lives here", "agent-rail", "release checklist")
    lane_id = active_write(tools, "lane local note", "agent-lane", "lane note")

    found = results(tools.memory_search(query="release checklist", workspace="agent-lane"))
    assert rail_id in found, "recall did not widen — SQL admission is a no-op"

    read = tools.memory_get(memory_id=rail_id, workspace="agent-lane")
    assert read["ok"] is True, "search widened but read did not — ACL out of sync"
    assert read["data"]["memory"]["id"] == rail_id

    recent = tools.memory_recent(workspace="agent-lane", limit=50)
    recent_ids = [r["id"] for r in recent["data"]["results"]]
    assert rail_id in recent_ids and lane_id in recent_ids


def test_admission_off_restores_exact_isolation(tmp_path, monkeypatch):
    """开关 off 时完全回退精确等值旧行为（recall + read 都不放宽）。"""
    tools = strict_admission_make_tools(tmp_path, admission=False, monkeypatch=monkeypatch)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "release checklist lives here", "agent-rail", "release checklist")

    assert rail_id not in results(tools.memory_search(query="release checklist", workspace="agent-lane"))
    assert tools.memory_get(memory_id=rail_id, workspace="agent-lane")["ok"] is False
    recent = tools.memory_recent(workspace="agent-lane", limit=50)
    assert rail_id not in [r["id"] for r in recent["data"]["results"]]


def test_far_workspace_never_admitted(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "unrelated-ws", VEC_FAR)
    far_id = active_write(tools, "unrelated secret content", "unrelated-ws", "unrelated subject")

    assert far_id not in results(tools.memory_search(query="unrelated secret", workspace="agent-lane"))
    assert tools.memory_get(memory_id=far_id, workspace="agent-lane")["ok"] is False


def test_default_pool_never_flooded_into_strict_recall(tmp_path):
    """default 洪水测试: strict 下任意查询不召回 default 池记忆。"""
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "default", VEC_SELF)  # even at distance 0
    # default writes are not new workspaces, so they are active immediately.
    default_id = tools.memory_write(
        content="global preference note", workspace="default", subject="global preference",
        source_type="agent_generated",
    )["data"]["id"]

    assert default_id not in results(tools.memory_search(query="global preference", workspace="agent-lane"))
    assert tools.memory_get(memory_id=default_id, workspace="agent-lane")["ok"] is False


def test_short_name_workspace_stays_isolated_end_to_end(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "w", VEC_SELF)
    publish(tools, "claw", VEC_NEAR)
    claw_id = active_write(tools, "claw project content", "claw", "claw subject")

    assert claw_id not in results(tools.memory_search(query="claw project", workspace="w"))
    assert tools.memory_get(memory_id=claw_id, workspace="w")["ok"] is False


def test_generic_substring_workspace_stays_isolated_end_to_end(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "main", VEC_SELF)
    publish(tools, "openclaw-main", VEC_NEAR)
    other_id = active_write(tools, "openclaw main content", "openclaw-main", "openclaw subject")

    assert other_id not in results(tools.memory_search(query="openclaw main", workspace="main"))
    assert tools.memory_get(memory_id=other_id, workspace="main")["ok"] is False


def test_over_cutoff_abbreviation_uses_workspace_migration(tmp_path):
    """Over-cutoff names stay separate until the user merges their workspaces."""
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "memory-arbiter-mcp", VEC_SELF)
    publish(tools, "mema", VEC_FAR)  # too far to admit
    mema_id = active_write(tools, "mema abbreviation content", "mema", "mema subject")

    assert mema_id not in results(
        tools.memory_search(query="mema abbreviation", workspace="memory-arbiter-mcp")
    )
    merged = tools.memory_govern("migrate_workspace", {
        "from": "mema", "to": "memory-arbiter-mcp", "authorized": True,
    })
    assert merged["ok"] is True
    assert tools.memory_get(memory_id=mema_id, workspace="mema")["ok"] is True


# ── consistency: COUNT / pagination / expired / filters ────────────────────

def test_counts_and_pagination_match_admitted_scope(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    for index in range(3):
        active_write(tools, f"shared tagged item {index}", "agent-lane", f"lane item {index}")
    for index in range(2):
        active_write(tools, f"shared tagged item rail {index}", "agent-rail", f"rail item {index}")
    active_write(tools, "shared tagged item far", "unrelated-ws", "far item")

    # filter-driven path (empty query + tags/source filter) is SQL-exact
    page = tools.memory_search(
        query="", workspace="agent-lane", source_type="agent_generated", limit=100,
    )
    ids = results(page)
    assert len(ids) == 5, ids                      # 3 lane + 2 rail, never the far one
    assert page["data"]["total_estimate"] == 5     # COUNT agrees with the page

    first = tools.memory_search(
        query="", workspace="agent-lane", source_type="agent_generated", limit=2, offset=0,
    )
    second = tools.memory_search(
        query="", workspace="agent-lane", source_type="agent_generated", limit=2, offset=2,
    )
    assert first["data"]["total_estimate"] == second["data"]["total_estimate"] == 5
    assert not set(results(first)) & set(results(second))


def test_expired_recall_uses_same_admitted_scope(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "retired rail decision", "agent-rail", "retired rail")
    # Retire it from a caller that can see it, then read the expired domain
    # from the admitted neighbour — both halves must use the same scope.
    retired = tools.memory_supersede(
        memory_id=rail_id, reason="superseded for test", authorized=True, workspace="agent-rail",
    )
    assert retired["ok"] is True, retired

    expired = tools.memory_search_expired(query="retired rail", workspace="agent-lane")
    assert rail_id in [r["id"] for r in expired["data"]["results"]]


def test_supersede_authorization_widens_with_admission(tmp_path):
    """An admitted neighbour may govern the memory it can now read (unified
    admission: read and governance share one set)."""
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "rail governance target", "agent-rail", "rail governance")

    retired = tools.memory_supersede(
        memory_id=rail_id, reason="admitted neighbour retires it", authorized=True,
        workspace="agent-lane",
    )
    assert retired["ok"] is True, retired
    assert tools.db.get_memory(rail_id)["status"] == "superseded"


def test_recent_fallback_scopes_to_admitted_set(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    rail_id = active_write(tools, "rail body", "agent-rail", "rail subject")
    far_id = active_write(tools, "far body", "unrelated-ws", "far subject")

    # empty query + no filters → recent browse, scoped in SQL
    browse = tools.memory_search(query="", workspace="agent-lane", limit=100)
    ids = results(browse)
    assert rail_id in ids
    assert far_id not in ids


# ── governance/aggregate consistency ───────────────────────────────────────

def test_audit_summary_reports_each_admitted_workspace(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    active_write(tools, "lane content", "agent-lane", "lane subject")
    active_write(tools, "rail content", "agent-rail", "rail subject")
    active_write(tools, "far content", "unrelated-ws", "far subject")

    summary = tools.memory_audit_summary(workspace="agent-lane")["data"]
    assert set(summary["workspaces"]) == {"agent-lane", "agent-rail"}
    assert summary["total_memories"] == 2


def test_audit_summary_preserves_empty_caller_bucket_with_admission_off(tmp_path, monkeypatch):
    tools = strict_admission_make_tools(tmp_path, admission=False, monkeypatch=monkeypatch)
    summary = tools.memory_audit_summary(workspace="empty-project")["data"]
    assert summary["workspaces"] == {
        "empty-project": {
            "count": 0, "oldest": None, "newest": None,
            "open_conflicts": 0, "by_source_type": {},
        }
    }
    assert summary["total_memories"] == 0


def test_strict_rebuild_evidence_scopes_discovery_flag_off_and_on(tmp_path, monkeypatch):
    """Round-1 high-severity regression: a strict scope tuple must expand to
    SQL parameters, never bind as one scalar; admission widens discovery only
    when enabled."""
    tools = strict_admission_make_tools(tmp_path, admission=True)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    lane = active_write(tools, "index lane body", "agent-lane", "index lane")
    rail = active_write(tools, "index rail body", "agent-rail", "index rail")
    far = active_write(tools, "index far body", "unrelated-ws", "index far")

    widened = tools.memory_repair(
        "rebuild_evidence", {"dry_run": True, "workspace": "agent-lane", "batch_size": 50},
    )
    assert widened["ok"] is True, widened
    assert set(widened["data"]["memory_ids"]) == {lane, rail}
    assert far not in widened["data"]["memory_ids"]

    # admission froze into a constant (0.15.0); flip the module binding the
    # caller resolver reads to simulate the off rollback.
    monkeypatch.setattr("memory_arbiter.tools.WORKSPACE_RECALL_ADMISSION", False)
    exact = tools.memory_repair(
        "rebuild_evidence", {"dry_run": True, "workspace": "agent-lane", "batch_size": 50},
    )
    assert exact["ok"] is True, exact
    assert exact["data"]["memory_ids"] == [lane]


def test_entities_listing_scopes_to_admitted_set(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    lane_id = active_write(tools, "lane content", "agent-lane", "lane subject")
    rail_id = active_write(tools, "rail content", "agent-rail", "rail subject")
    far_id = active_write(tools, "far content", "unrelated-ws", "far subject")
    tools.memory_set_entity(memory_id=lane_id, entity="lane-entity", workspace="agent-lane")
    tools.memory_set_entity(memory_id=rail_id, entity="rail-entity", workspace="agent-rail")
    tools.memory_set_entity(memory_id=far_id, entity="far-entity", workspace="unrelated-ws")

    data = tools.memory_list_entities(workspace="agent-lane")["data"]
    listed = json.dumps(data, ensure_ascii=False)
    assert "lane-entity" in listed and "rail-entity" in listed
    assert "far-entity" not in listed


def test_console_browse_and_status_use_admitted_scope(tmp_path):
    from memory_arbiter.console_api import ConsoleAPI

    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    rail_id = active_write(tools, "rail console content", "agent-rail", "rail subject")
    rail_peer = active_write(tools, "rail console peer", "agent-rail", "rail peer")
    far_id = active_write(tools, "far console content", "unrelated-ws", "far subject")
    notice = tools.db.record_semantic_notice(
        memory_id=rail_id, peer_id=rail_peer, severity="normal",
        notice_type="semantic_evidence", title="console notice", message="console",
        payload={}, left_version=1, right_version=1,
    )
    assert notice["outcome"] == "created"

    api = ConsoleAPI(tools)
    browse = api.memories(query=None, workspace="agent-lane", limit=100, offset=0)
    ids = [item["id"] for item in (browse.get("items") or [])]
    assert rail_id in ids
    assert far_id not in ids

    counts = api._status_counts(workspace="agent-lane")
    assert counts["total"] == 2  # the two admitted rail rows; never the far one
    assert counts["active"] == 2
    overview = api.overview(workspace="agent-lane")
    assert overview["status"]["semantic_conflict"]["notices"].get("open") == 1


def test_conflict_detail_visible_across_admitted_workspaces(tmp_path):
    """A conflict recorded in an admitted neighbour is inspectable, and its
    members read back — the authorization path shares the admitted set."""
    import hashlib
    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    first = active_write(tools, "rail says sqlite", "agent-rail", "rail database")
    second = active_write(tools, "rail says mysql", "agent-rail", "rail database alt")

    members = []
    groups = []
    for index, memory_id in enumerate((first, second)):
        record = tools.db.get_memory(memory_id)
        value = f"value-{index}"
        quote = record["content"]
        members.append(ConflictMember(
            memory_id=memory_id, version=record["version"], attribute_raw="database",
            value_raw=value, normalized_attribute="database", normalized_value=value,
            evidence_quote=quote, evidence_span=(0, len(quote)),
            content_hash=hashlib.sha256(quote.encode()).hexdigest(),
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ))
        groups.append(ConflictValueGroup(value, value, (f"{memory_id}@{record['version']}",)))
    conflict_id = tools.db.record_conflict_group(
        workspace_canonical="agent-rail",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=members, value_groups=groups, detection_reason="different values",
        source="scan", detector_version="d1", conflict_point="database",
    )["conflict_id"]

    detail = tools.memory_review("conflict_detail", {"conflict_id": conflict_id, "workspace": "agent-lane"})
    assert detail["ok"] is True, detail
    assert detail["data"]["next_executable_call"]["data"]["workspace"] == "agent-lane"
    listing = tools.memory_list_conflicts(status="open", limit=50, workspace="agent-lane")
    assert conflict_id in [c["id"] for c in listing["data"]["conflicts"]]

    judged = tools.memory("judge", {
        "workspace": "agent-lane", "conflict_id": conflict_id,
        "expected_revision": 1, "chosen_value": "value-1",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "authorized": True,
        "apply_plan": [
            {"memory_id": first, "action": "preserve_historical_record"},
            {"memory_id": second, "action": "use_as_resolution"},
        ],
        "resolution_memory_id": second,
    })
    assert judged["ok"] is True, judged
    assert judged["data"]["next_action"]["data"]["workspace"] == "agent-lane"


def test_conflict_outside_admitted_set_stays_hidden(tmp_path):
    import hashlib
    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "unrelated-ws", VEC_FAR)
    first = active_write(tools, "far says sqlite", "unrelated-ws", "far database")
    second = active_write(tools, "far says mysql", "unrelated-ws", "far database alt")

    members = []
    groups = []
    for index, memory_id in enumerate((first, second)):
        record = tools.db.get_memory(memory_id)
        value = f"value-{index}"
        quote = record["content"]
        members.append(ConflictMember(
            memory_id=memory_id, version=record["version"], attribute_raw="database",
            value_raw=value, normalized_attribute="database", normalized_value=value,
            evidence_quote=quote, evidence_span=(0, len(quote)),
            content_hash=hashlib.sha256(quote.encode()).hexdigest(),
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ))
        groups.append(ConflictValueGroup(value, value, (f"{memory_id}@{record['version']}",)))
    conflict_id = tools.db.record_conflict_group(
        workspace_canonical="unrelated-ws",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"},
        members=members, value_groups=groups, detection_reason="different values",
        source="scan", detector_version="d1", conflict_point="database",
    )["conflict_id"]

    detail = tools.memory_review("conflict_detail", {"conflict_id": conflict_id, "workspace": "agent-lane"})
    assert detail["ok"] is False
    listing = tools.memory_list_conflicts(status="open", limit=50, workspace="agent-lane")
    assert conflict_id not in [c["id"] for c in listing["data"]["conflicts"]]


def test_admitted_neighbor_notice_auto_delivery_and_retry_calls(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    left = active_write(tools, "rail notice left", "agent-rail", "rail notice")
    right = active_write(tools, "rail notice right", "agent-rail", "rail notice peer")
    created = tools.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal",
        notice_type="semantic_evidence", title="rail candidate", message="review rail",
        payload={}, left_version=1, right_version=1,
    )
    assert created["outcome"] == "created", created

    delivered = tools.memory(action="help", data={"workspace": "agent-lane"})
    notice_stub = next(n for n in delivered.get("notices", []) if n.get("notice_id") == created["notice_id"])
    assert notice_stub["read_call"]["data"]["workspace"] == "agent-lane"
    read = tools.memory_repair(
        notice_stub["read_call"]["task"], notice_stub["read_call"]["data"],
    )
    assert read["ok"] is True, read
    calls = read["data"]["notice"]["read_calls"]
    assert len(calls) == 2
    assert all(call["data"]["workspace"] == "agent-lane" for call in calls)
    assert all(isinstance(call["data"]["workspace"], str) for call in calls)


def test_notice_delivery_reuses_operation_scope(tmp_path, monkeypatch):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    calls = 0
    original = tools.db.workspaces.admitted_canonicals

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tools.db.workspaces, "admitted_canonicals", counted)
    response = tools.memory("find", {"query": "nothing", "workspace": "agent-lane"})
    assert response["ok"] is True
    assert calls == 1


def test_notice_claim_skips_full_hidden_page_in_one_call(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    left = active_write(tools, "hidden notice left", "agent-rail", "hidden left")
    right = active_write(tools, "hidden notice right", "agent-rail", "hidden right")
    for index in range(25):
        created = tools.db.record_semantic_notice(
            memory_id=left, peer_id=right, severity="normal",
            notice_type="semantic_evidence", title=f"hidden-{index}", message="hidden",
            payload={}, dedupe_key=f"hidden-{index}", left_version=1, right_version=1,
        )
        assert created["outcome"] == "created"
    assert tools.db.set_memory_workspace_canonical(right, "unrelated-ws")[0] is True
    valid_left = active_write(tools, "valid notice left", "agent-rail", "valid left")
    valid_right = active_write(tools, "valid notice right", "agent-rail", "valid right")
    valid = tools.db.record_semantic_notice(
        memory_id=valid_left, peer_id=valid_right, severity="normal",
        notice_type="semantic_evidence", title="valid", message="valid",
        payload={}, dedupe_key="valid-after-hidden", left_version=1, right_version=1,
    )
    delivered = tools.memory("help", {"workspace": "agent-lane"})
    assert any(n.get("notice_id") == valid["notice_id"] for n in delivered.get("notices", []))


def test_notice_counts_refresh_version_stale_before_read(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    left = active_write(tools, "stale count left", "agent-rail", "count left")
    right = active_write(tools, "stale count right", "agent-rail", "count right")
    created = tools.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal",
        notice_type="semantic_evidence", title="stale-count", message="count",
        payload={}, left_version=1, right_version=1,
    )
    assert created["outcome"] == "created"
    edited = tools.memory_edit(
        memory_id=left, new_content="stale count left edited", reason="version bump",
        workspace="agent-rail",
    )
    assert edited["ok"] is True
    status = tools.memory_status(workspace="agent-lane")
    assert status["data"]["semantic_conflict"]["notices"].get("stale") == 1
    assert status["data"]["semantic_conflict"]["notices"].get("open", 0) == 0


def test_stale_notice_list_fills_past_hidden_rows(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    visible_left = active_write(tools, "visible stale left", "agent-rail", "visible left")
    visible_right = active_write(tools, "visible stale right", "agent-rail", "visible right")
    visible = tools.db.record_semantic_notice(
        memory_id=visible_left, peer_id=visible_right, severity="normal",
        notice_type="semantic_evidence", title="visible-stale", message="visible",
        payload={}, dedupe_key="visible-stale", left_version=1, right_version=1,
    )
    tools.memory_edit(
        memory_id=visible_left, new_content="visible stale edited", reason="version bump",
        workspace="agent-rail",
    )
    hidden_left = active_write(tools, "hidden stale left", "agent-rail", "hidden stale left")
    hidden_right = active_write(tools, "hidden stale right", "agent-rail", "hidden stale right")
    for index in range(12):
        tools.db.record_semantic_notice(
            memory_id=hidden_left, peer_id=hidden_right, severity="normal",
            notice_type="semantic_evidence", title=f"hidden-stale-{index}", message="hidden",
            payload={}, dedupe_key=f"hidden-stale-{index}", left_version=1, right_version=1,
        )
    tools.db.set_memory_workspace_canonical(hidden_right, "unrelated-ws")
    listed = tools.memory_repair("notice", {
        "action": "list", "status": "stale", "limit": 1, "workspace": "agent-lane",
    })
    assert [n["id"] for n in listed["data"]["notices"]] == [visible["notice_id"]]


def test_stale_notice_hides_member_moved_outside_scope(tmp_path):
    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    publish(tools, "unrelated-ws", VEC_FAR)
    left = active_write(tools, "rail public member", "agent-rail", "rail public")
    right = active_write(tools, "SECRET MOVED MEMBER", "agent-rail", "rail secret")
    created = tools.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal",
        notice_type="semantic_evidence", title="secret candidate", message="SECRET MESSAGE",
        payload={}, left_version=1, right_version=1,
    )
    assert created["outcome"] == "created", created
    assert tools.db.set_memory_workspace_canonical(right, "unrelated-ws")[0] is True

    read = tools.memory_repair("notice", {
        "action": "read", "notice_id": created["notice_id"], "workspace": "agent-lane",
    })
    assert read["ok"] is False
    listed = tools.memory_repair("notice", {
        "action": "list", "status": "stale", "limit": 10, "workspace": "agent-lane",
    })
    encoded = json.dumps(listed, ensure_ascii=False)
    assert str(created["notice_id"]) not in json.dumps(
        [n.get("id") for n in listed["data"]["notices"]]
    )
    assert "SECRET MOVED MEMBER" not in encoded
    assert "SECRET MESSAGE" not in encoded


def test_record_conflict_in_admitted_neighbor_uses_member_workspace(tmp_path):
    import hashlib

    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    ids = [
        active_write(tools, "rail value sqlite", "agent-rail", "rail sqlite"),
        active_write(tools, "rail value mysql", "agent-rail", "rail mysql"),
    ]
    members = []
    groups = []
    for index, memory_id in enumerate(ids):
        record = tools.db.get_memory(memory_id)
        value = ("sqlite", "mysql")[index]
        quote = record["content"]
        members.append({
            "memory_id": memory_id, "version": record["version"],
            "attribute_raw": "database", "value_raw": value,
            "normalized_attribute": "database", "normalized_value": value,
            "evidence_quote": quote, "evidence_span": [0, len(quote)],
            "content_hash": hashlib.sha256(quote.encode()).hexdigest(),
            "direction": "a_to_b", "prompt_version": "p1", "detector_version": "d1",
        })
        groups.append({
            "normalized_value": value, "display_value": value,
            "members": [f"{memory_id}@{record['version']}"],
        })
    result = tools.memory_repair("record_conflict", {
        "workspace": "agent-lane",
        "slot_key": {"entity": "project", "attribute": "database", "scope": "global"},
        "members": members, "value_groups": groups,
        "detector_version": "d1", "source": "scan", "reason": "different values",
        "conflict_point": "database",
    })
    assert result["ok"] is True, result
    conflict = tools.db.get_conflict(result["data"]["conflict_id"])
    assert conflict["workspace_canonical"] == "agent-rail"

    audit = tools.memory_audit_summary(workspace="agent-lane")["data"]
    assert audit["workspaces"]["agent-rail"]["open_conflicts"] == 1
    assert audit["workspaces"]["agent-lane"]["open_conflicts"] == 0
    assert audit["total_open_conflicts"] == 1


# ── anti-no-op: SQL layer must be the one doing the scoping ────────────────

def test_wide_recall_sql_scopes_to_admitted_set_directly(tmp_path):
    """Directly exercise the recall SQL with an admitted set: if a future change
    reverted the SQL to single-canonical equality (leaving only a Python
    post-filter), this fails — the plan's 空操作 detector."""
    from memory_arbiter.search import _wide_recall

    tools = strict_admission_make_tools(tmp_path)
    publish(tools, "agent-lane", VEC_SELF)
    publish(tools, "agent-rail", VEC_NEAR)
    rail_id = active_write(tools, "sql level admission probe", "agent-rail", "probe subject")

    pool = _wide_recall(
        tools.db, "sql level admission probe", None, None,
        "m.status = 'active'", "status = 'active'",
        ws_canonical=("agent-lane", "agent-rail"),
    )
    assert rail_id in [row["id"] for row in pool]

    narrow = _wide_recall(
        tools.db, "sql level admission probe", None, None,
        "m.status = 'active'", "status = 'active'",
        ws_canonical=("agent-lane",),
    )
    assert rail_id not in [row["id"] for row in narrow]


# ── from test_workspace_default_insulation.py ──
# helpers renamed: default_insulation_make_tools, _default_insulation_write (collisions)

"""default 双向绝缘 (mema 721 期0).

The reserved default pool ("", default/默认/none/null/unknown/未知, case-
insensitive) must be insulated from the vector/alias system in BOTH
directions: no workspace may be merged INTO default by KNN AUTO-merge, no
canonical vector may ever be published for a default term, default synonyms
resolve to the single global pool, and alias governance (accept/reject/
rename/migrate) refuses any pair touching default.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import (
    DEFAULT_TERMS,
    DEFAULT_WORKSPACE_NAME,
    is_default_workspace_term,
)
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools

NON_EMPTY_DEFAULT_TERMS = sorted(t for t in DEFAULT_TERMS if t)

try:
    import sqlite_vec  # type: ignore  # noqa: F401
    _VEC_AVAILABLE = True
except Exception:
    _VEC_AVAILABLE = False

requires_vec = pytest.mark.skipif(not _VEC_AVAILABLE, reason="sqlite-vec not installed")


def default_insulation_make_tools(tmp_path: Path, isolation: str = "none", *, vec: bool = False) -> MemoryTools:
    # vec=True points at a (fake) GGUF model — the model path IS the intent
    # since 0.15.0 — and mirrors the first successful embedder build by
    # creating the lazy vec0 tables at dim 2.
    model = tmp_path / "fake.gguf"
    settings = Settings(
        db_path=tmp_path / "ins.sqlite3",
        backup_jsonl=tmp_path / "ins.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        embedding_model_path=model if vec else None,
        isolation=isolation,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    if vec:
        model.write_bytes(b"fake")
        assert db.ensure_vec_tables(2) == []
    return tools


class FixedEmbedder:
    def __init__(self, vector: list[float]):
        self.vector = vector

    def embed_text(self, prefix: str = "", body: str = ""):
        return SimpleNamespace(embedding=list(self.vector))


def _default_insulation_write(tools: MemoryTools, content: str, workspace: str = "", subject: str = "test") -> dict:
    return tools.memory_write(
        content=content, workspace=workspace, subject=subject,
        source_type="agent_generated",
    )


def _register_canonical_with_vector(tools: MemoryTools, name: str, vector: list[float]) -> None:
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
            (name,),
        )
        row = conn.execute(
            "SELECT id FROM workspace_canonicals WHERE name = ?", (name,)
        ).fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
            (int(row["id"]), json.dumps(vector)),
        )


def _canonical_vec_row(tools: MemoryTools, name: str):
    with tools.db.connection() as conn:
        return conn.execute(
            "SELECT v.id FROM workspace_canonicals c "
            "JOIN workspace_canonicals_vec v ON v.id = c.id WHERE c.name = ?",
            (name,),
        ).fetchone()


# ── KNN exclusion (改动1) ────────────────────────────────────────────────────

@requires_vec
def test_knn_never_merges_into_default_even_with_published_vector(tmp_path):
    tools = default_insulation_make_tools(tmp_path, vec=True)
    assert tools.db.state.sqlite_vec_available
    # Simulate a legacy/foreign backfill that force-published a default vector.
    _register_canonical_with_vector(tools, "default", [1.0, 0.0])
    assert _canonical_vec_row(tools, "default") is not None

    # Control: a normal canonical at distance 0 does attract the AUTO merge —
    # proving the embedder/distances work and default is excluded by name.
    _register_canonical_with_vector(tools, "claw", [0.0, 1.0])
    merged = tools.db.resolve_workspace_canonical(
        "clawproj", FixedEmbedder([0.0, 1.0]), register_new=False,
    )
    assert merged["matched_by"] == "vector"
    assert merged["canonical"] == "claw"

    # A name embedding to distance ~0 FROM DEFAULT must stay NEW: the only
    # ≤cutoff neighbour is the excluded default row (claw sits at 1.0).
    resolved = tools.db.resolve_workspace_canonical(
        "defaultproj", FixedEmbedder([1.0, 0.0]), register_new=False,
    )
    assert resolved["matched_by"] == "new"
    assert resolved["canonical"] == "defaultproj"
    assert resolved["is_new"] is True
    assert "default" not in [s["name"] for s in resolved["similar"]]


@requires_vec
@pytest.mark.parametrize("term", NON_EMPTY_DEFAULT_TERMS)
def test_knn_excludes_every_default_synonym_canonical(tmp_path, term):
    tools = default_insulation_make_tools(tmp_path, vec=True)
    # Even a legacy DB that registered a synonym as its own canonical (with a
    # vector) can never attract merges: the candidate SQL excludes all terms.
    _register_canonical_with_vector(tools, term, [1.0, 0.0])
    resolved = tools.db.resolve_workspace_canonical(
        f"{term}project", FixedEmbedder([1.0, 0.0]), register_new=False,
    )
    assert resolved["matched_by"] == "new"
    assert term not in [s["name"] for s in resolved["similar"]]
# ── synonym resolution (2c) ──────────────────────────────────────────────────

@pytest.mark.parametrize("term", NON_EMPTY_DEFAULT_TERMS + ["Default", "DEFAULT", " None ", "未知 "])
def test_default_synonyms_resolve_to_single_pool(tmp_path, term):
    tools = default_insulation_make_tools(tmp_path)
    resolved = tools.db.resolve_workspace_canonical(term, None, register_new=True)
    assert resolved["canonical"] == DEFAULT_WORKSPACE_NAME
    assert resolved["matched_by"] == "fallback"
    assert resolved["is_new"] is False
    with tools.db.connection() as conn:
        names = [r["name"] for r in conn.execute("SELECT name FROM workspace_canonicals").fetchall()]
    assert term.strip() not in names  # no phantom synonym canonical


# ── vector publish insulation (改动2) ────────────────────────────────────────

@requires_vec
def test_vector_publish_paths_skip_default_terms(tmp_path):
    tools = default_insulation_make_tools(tmp_path, vec=True)
    store = tools.db.workspaces
    embedder = FixedEmbedder([1.0, 0.0])
    with tools.db.write_transaction() as conn:
        for name in ("default", "默认", "null", "projx"):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
                (name,),
            )
    result = {"warnings": [], "vector_publish_pending": False}

    store._publish_missing_workspace_canonical_vector("default", embedder, result)
    assert _canonical_vec_row(tools, "default") is None
    assert result["vector_publish_pending"] is False

    assert store.prepare_missing_workspace_canonical_embedding("默认", embedder) is None
    assert store.prepare_workspace_canonical_embedding("null", embedder) is None
    assert store.publish_workspace_canonical_vector("default", [1.0, 0.0]) == []
    for name in ("default", "默认", "null"):
        assert _canonical_vec_row(tools, name) is None

    # Control: a normal canonical publishes through the same path.
    store._publish_missing_workspace_canonical_vector("projx", embedder, result)
    assert _canonical_vec_row(tools, "projx") is not None


# ── governance guards (改动3) ────────────────────────────────────────────────

@pytest.mark.parametrize("term", NON_EMPTY_DEFAULT_TERMS)
def test_rename_refuses_default_in_both_directions(tmp_path, term):
    tools = default_insulation_make_tools(tmp_path)
    _default_insulation_write(tools, "projX memory", "projX")

    updated, warnings = tools.db.rename_workspace_canonical("projX", term)
    assert updated == 0
    assert warnings and "reserved" in warnings[0]
    updated, warnings = tools.db.rename_workspace_canonical(term, "projY")
    assert updated == 0
    assert warnings and "reserved" in warnings[0]

    r = tools.memory_govern("rename_workspace_canonical", {
        "old": "projX", "new": term, "reason": "try merge into default", "authorized": True,
    })
    assert r["ok"] is False
    assert r["data"]["renamed"] is False
    assert any("reserved" in w for w in r["warnings"])


@pytest.mark.parametrize("term", NON_EMPTY_DEFAULT_TERMS)
def test_removed_pairwise_actions_and_internal_decisions_never_touch_default(tmp_path, term):
    tools = default_insulation_make_tools(tmp_path)
    _default_insulation_write(tools, "projX memory", "projX")

    for action, alias, canonical in (
        ("accept_workspace_alias", term, "projX"),
        ("accept_workspace_alias", "projX", term),
        ("reject_workspace_alias", term, "projX"),
    ):
        result = tools.memory_govern(action, {
            "alias": alias, "canonical": canonical, "authorized": True,
        })
        assert result["ok"] is False
        assert result["data"]["error_code"] == "workspace_alias_action_removed"

    assert tools.db.record_workspace_decision(term, "projX", status="confirmed")[0] is False
    assert tools.db.record_workspace_decision("projX", term, status="confirmed")[0] is False
    with tools.db.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases WHERE canonical = ? OR alias_workspace = ?",
            (term, term),
        ).fetchone()[0]
    assert count == 0


def test_migrate_refuses_default_in_both_directions(tmp_path):
    tools = default_insulation_make_tools(tmp_path)
    _default_insulation_write(tools, "projX memory", "projX")
    updated, warnings = tools.db.migrate_workspace("projX", "default")
    assert updated == 0 and warnings
    updated, warnings = tools.db.migrate_workspace("默认", "projX")
    assert updated == 0 and warnings


# ── regression: default writes/recall keep working ──────────────────────────

def test_default_write_and_synonym_recall_regression(tmp_path):
    tools = default_insulation_make_tools(tmp_path)
    w1 = _default_insulation_write(tools, "global fact one", "")
    w2 = _default_insulation_write(tools, "synonym fact two", "默认")
    assert w1["data"]["workspace_canonical"] == DEFAULT_WORKSPACE_NAME
    assert w2["data"]["workspace_canonical"] == DEFAULT_WORKSPACE_NAME

    found = tools.memory_search(query="fact", workspace="默认")
    ids = [r["id"] for r in found["data"]["results"]]
    assert w1["data"]["id"] in ids
    assert w2["data"]["id"] in ids


@requires_vec
def test_placement_hint_still_fires_for_default_synonym(tmp_path, monkeypatch):
    tools = default_insulation_make_tools(tmp_path, vec=True)  # placement hint requires sqlite_vec
    proj = _default_insulation_write(tools, "projA memory", "projA", subject="projA subject")
    monkeypatch.setattr(tools, "_ensure_embedder", lambda: (FixedEmbedder([1.0, 0.0]), []))
    monkeypatch.setattr(
        tools.db, "evidence_knn",
        lambda emb, k=8: [{"memory_id": proj["data"]["id"], "distance": 0.1}],
    )

    w = _default_insulation_write(tools, "editor preference", "默认", subject="editor preference")
    assert w["data"]["workspace_canonical"] == DEFAULT_WORKSPACE_NAME
    hints = w["data"].get("write_hints") or {}
    assert hints["placement_suggestion"]["suggested_workspace"] == "projA"


# ── regression: full-width IME spellings fold into the global pool ──────────

def test_full_width_default_spellings_are_default_terms():
    # NFKC folds full-width IME spellings onto their ASCII twins before the
    # synonym comparison; without it ｄｅｆａｕｌｔ registers a phantom second
    # default pool instead of landing in the global one.
    assert is_default_workspace_term("ｄｅｆａｕｌｔ")
    assert is_default_workspace_term("ＮＵＬＬ")
    assert is_default_workspace_term("　默认　")  # U+3000 spaces are stripped
    # boundary: full-width PROJECT names are still real workspaces, and
    # supersets of "default" are not synonyms
    assert not is_default_workspace_term("ｐｒｏｊ")
    assert not is_default_workspace_term("defaulted")


def test_full_width_default_write_lands_in_global_pool(tmp_path):
    tools = default_insulation_make_tools(tmp_path)
    written = _default_insulation_write(tools, "full-width default write", "ｄｅｆａｕｌｔ", subject="nfkc")
    assert written["ok"]
    assert written["data"]["workspace_canonical"] == DEFAULT_WORKSPACE_NAME
    with tools.db.connection() as conn:
        names = [str(row["name"]) for row in conn.execute("SELECT name FROM workspace_canonicals")]
    assert "ｄｅｆａｕｌｔ" not in names


def test_move_refuses_full_width_default_destination(tmp_path):
    tools = default_insulation_make_tools(tmp_path)
    memory_id = int(_default_insulation_write(tools, "x", "proj-a", subject="s")["data"]["id"])
    outcome = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [memory_id], "new_workspace": "ＮＵＬＬ", "authorized": True,
    })
    assert not outcome["ok"]
    assert "reserved global pool" in outcome["data"]["error"]
    assert tools.db.get_memory(memory_id)["workspace"] == "proj-a"


# ── from test_workspace_weak_vector_weight.py ──
# helpers renamed: weak_vector_weight_make_tools, _weak_vector_weight_write (collisions)

"""weak 连续向量加权 (mema 721 期1) + §2a 共享准入 helper.

Covers the pure helpers (guarded distance, admission predicate, weight
curve), the _workspace_bonus curve/binary fallbacks, and the search-path
wiring: distance_map precompute behind the workspace_weak_vector_weight flag,
ranking lift for a near workspace, flag-off binary regression, and magnitude
discipline (the nudge never overrides a subject/tags hit).
"""
from pathlib import Path

import pytest

from memory_arbiter import workspace_rules as wr
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.search import _workspace_bonus
from memory_arbiter.tools import MemoryTools


def weak_vector_weight_make_tools(
    tmp_path: Path,
    isolation: str = "weak",
    *,
    weak_vector: bool = True,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> MemoryTools:
    if monkeypatch is not None:
        # workspace_weak_vector_weight froze into a constant (0.15.0, default
        # off); the continuous-weight path is exercised by flipping the module
        # binding the search wiring reads.
        monkeypatch.setattr(
            "memory_arbiter.search.WORKSPACE_WEAK_VECTOR_WEIGHT", weak_vector,
        )
    elif weak_vector:
        raise TypeError(
            "weak_vector=True requires the monkeypatch fixture "
            "(workspace_weak_vector_weight is a frozen constant since 0.15.0)"
        )
    settings = Settings(
        db_path=tmp_path / "wv.sqlite3",
        backup_jsonl=tmp_path / "wv.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _weak_vector_weight_write(tools: MemoryTools, content: str, workspace: str, subject: str) -> int:
    return tools.memory_write(
        content=content, workspace=workspace, subject=subject,
        source_type="agent_generated",
    )["data"]["id"]


# ── §2a pure helpers ─────────────────────────────────────────────────────────

def test_workspace_vector_distance_guards():
    dmap = {"agent-rail": 0.142}
    assert wr.workspace_vector_distance("agent-lane", "agent-lane", dmap) == 0.0
    assert wr.workspace_vector_distance("agent-lane", "agent-rail", dmap) == pytest.approx(0.142)
    # no map / missing entry → exact-equality fallback for the caller
    assert wr.workspace_vector_distance("agent-lane", "agent-rail", None) is None
    assert wr.workspace_vector_distance("agent-lane", "agent-rail", {}) is None
    assert wr.workspace_vector_distance("agent-lane", "unknown-ws", dmap) is None
    # default insulation (both sides, every synonym)
    for term in ("default", "默认", "none", "null", "unknown", "未知"):
        assert wr.workspace_vector_distance(term, "agent-rail", dmap) is None
        assert wr.workspace_vector_distance("agent-lane", term, dmap) is None
    # short-name guard
    assert wr.workspace_vector_distance("w", "agent-rail", dmap) is None
    assert wr.workspace_vector_distance("agent-lane", "w", dmap) is None
    # substring / generic-token proximity guard (721 §3d hazards)
    assert wr.workspace_vector_distance(
        "main", "openclaw-main", {"openclaw-main": 0.132},
    ) is None
    assert wr.workspace_vector_distance(
        "project-alpha", "project-beta", {"project-beta": 0.1},
    ) is None
    # real same-project pair is NOT suppressed (no containment, no generic-only overlap)
    assert wr.workspace_vector_distance(
        "agent-rail", "agent-lane", {"agent-lane": 0.142},
    ) == pytest.approx(0.142)
    assert wr.workspace_vector_distance(
        "金营项目", "金科营销项目", {"金科营销项目": 0.16},
    ) == pytest.approx(0.16)
    # configurable min_name_len
    assert wr.workspace_vector_distance("abc", "wxyz", {"wxyz": 0.2}, min_name_len=4) is None
    assert wr.workspace_vector_distance("abcd", "wxyz", {"wxyz": 0.2}, min_name_len=4) == pytest.approx(0.2)


def test_workspace_admit_cutoff():
    dmap = {"agent-rail": 0.142, "openclaw-main": 0.132, "far-ws": 0.364}
    assert wr.workspace_admit("agent-lane", "agent-rail", dmap, 0.25) is True
    assert wr.workspace_admit("agent-lane", "agent-lane", dmap, 0.25) is True
    assert wr.workspace_admit("agent-lane", "far-ws", dmap, 0.25) is False
    assert wr.workspace_admit("main", "openclaw-main", dmap, 0.25) is False
    assert wr.workspace_admit("default", "agent-rail", dmap, 0.25) is False
    assert wr.workspace_admit("agent-lane", "w", dmap, 0.25) is False
    assert wr.workspace_admit("agent-lane", "missing", dmap, 0.25) is False


def test_weak_curve_anchor_points():
    assert wr.weak_workspace_vector_weight(0.0) == pytest.approx(0.30)
    assert wr.weak_workspace_vector_weight(0.10) == pytest.approx(0.30)
    assert wr.weak_workspace_vector_weight(0.142) == pytest.approx(0.30)  # full-bonus zone
    assert wr.weak_workspace_vector_weight(0.15) == pytest.approx(0.30)
    assert wr.weak_workspace_vector_weight(0.20) == pytest.approx(0.20)  # decayed, in (0, 0.30)
    assert wr.weak_workspace_vector_weight(0.225) == pytest.approx(0.15)
    assert wr.weak_workspace_vector_weight(0.30) == pytest.approx(0.0)
    assert wr.weak_workspace_vector_weight(0.364) == 0.0
    assert wr.weak_workspace_vector_weight(0.9) == 0.0


def test_weak_curve_cap_below_subject_medium():
    # 满分锚 0.30 不盖过 subject-medium (6.0)
    assert wr.WEAK_VECTOR_WEIGHT_MAX < 6.0 / 10


# ── _workspace_bonus curve + fallbacks ───────────────────────────────────────

def test_workspace_bonus_curve_and_fallbacks():
    dmap = {"agent-rail": 0.20, "far-ws": 0.40}
    rec_rail = {"workspace_canonical": "agent-rail"}
    rec_far = {"workspace_canonical": "far-ws"}
    rec_def = {"workspace_canonical": "default"}

    assert _workspace_bonus(rec_rail, "agent-lane", "weak", distance_map=dmap) == pytest.approx(0.20)
    # known-far cross-workspace: ≈0, no -0.15 hard penalty
    assert _workspace_bonus(rec_far, "agent-lane", "weak", distance_map=dmap) == 0.0
    # no map → exact v0.9.7 binary step
    assert _workspace_bonus(rec_rail, "agent-lane", "weak") == -0.15
    assert _workspace_bonus(rec_rail, "agent-rail", "weak") == 0.30
    # default insulation → binary fallback even with a map entry
    assert _workspace_bonus(rec_def, "agent-lane", "weak", distance_map={"default": 0.0}) == -0.15
    assert _workspace_bonus(rec_def, "default", "weak", distance_map={"default": 0.0}) == 0.30
    # guarded pair inside the map → binary fallback
    assert _workspace_bonus(
        {"workspace_canonical": "openclaw-main"}, "main", "weak",
        distance_map={"openclaw-main": 0.132},
    ) == -0.15
    # 0 outside weak mode
    assert _workspace_bonus(rec_rail, "agent-lane", "strict", distance_map=dmap) == 0.0
    assert _workspace_bonus(rec_rail, "agent-lane", "none", distance_map=dmap) == 0.0
    assert _workspace_bonus(rec_rail, None, "weak", distance_map=dmap) == 0.0


# ── search wiring ────────────────────────────────────────────────────────────

def test_search_lifts_near_workspace_with_vector_weight(tmp_path, monkeypatch):
    tools = weak_vector_weight_make_tools(tmp_path, "weak", weak_vector=True, monkeypatch=monkeypatch)
    near = _weak_vector_weight_write(tools, "release notes near", "agent-rail", "shared release notes")
    _weak_vector_weight_write(tools, "release notes far", "unrelated-ws", "shared release notes")
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map",
        lambda query, names: {"agent-rail": 0.20, "unrelated-ws": 0.9},
    )

    r = tools.memory_search(query="shared release notes", workspace="agent-lane", debug_ranking=True)
    results = r["data"]["results"]
    assert results[0]["id"] == near
    boosted = next(x for x in results if x["id"] == near)
    assert boosted["_workspace_bonus"] == pytest.approx(0.20)  # decayed positive weight


def test_search_full_bonus_zone_uses_max(tmp_path, monkeypatch):
    tools = weak_vector_weight_make_tools(tmp_path, "weak", weak_vector=True, monkeypatch=monkeypatch)
    near = _weak_vector_weight_write(tools, "rail memory", "agent-rail", "shared release notes")
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map",
        lambda query, names: {"agent-rail": 0.142},
    )
    r = tools.memory_search(query="shared release notes", workspace="agent-lane", debug_ranking=True)
    boosted = next(x for x in r["data"]["results"] if x["id"] == near)
    assert boosted["_workspace_bonus"] == pytest.approx(0.30)


def test_search_never_computes_map_when_flag_off(tmp_path, monkeypatch):
    tools = weak_vector_weight_make_tools(tmp_path, "weak", weak_vector=False, monkeypatch=monkeypatch)
    _weak_vector_weight_write(tools, "release notes", "agent-rail", "shared release notes")
    calls: list[str] = []

    def spy(query, names):
        calls.append(str(query))
        return {"agent-rail": 0.2}

    monkeypatch.setattr(tools.db.workspaces, "canonical_distance_map", spy)
    r = tools.memory_search(query="shared release notes", workspace="agent-lane", debug_ranking=True)
    assert calls == []  # off = exact binary behaviour, map never requested
    boosted = next(x for x in r["data"]["results"])
    assert boosted["_workspace_bonus"] in (0.30, -0.15)


def test_degraded_map_falls_back_to_binary(tmp_path, monkeypatch):
    tools = weak_vector_weight_make_tools(tmp_path, "weak", weak_vector=True, monkeypatch=monkeypatch)
    _weak_vector_weight_write(tools, "release notes", "agent-rail", "shared release notes")
    # Flag on but sqlite-vec degraded → canonical_distance_map returns {} →
    # per-record guard fallback to the binary step.
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map", lambda query, names: {},
    )
    r = tools.memory_search(query="shared release notes", workspace="agent-lane", debug_ranking=True)
    assert r["data"]["results"]
    assert r["data"]["results"][0]["_workspace_bonus"] == -0.15


def test_default_query_never_uses_vector_weight(tmp_path, monkeypatch):
    tools = weak_vector_weight_make_tools(tmp_path, "weak", weak_vector=True, monkeypatch=monkeypatch)
    _weak_vector_weight_write(tools, "global memory", "", "shared release notes")
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map",
        lambda query, names: pytest.fail("default query must not enter the vector system"),
    )
    r = tools.memory_search(query="shared release notes", workspace="默认")
    assert r["data"]["results"]
    assert r["data"]["results"][0]["id"]


def test_vector_weight_never_overrides_subject_hit(tmp_path, monkeypatch):
    tools = weak_vector_weight_make_tools(tmp_path, "weak", weak_vector=True, monkeypatch=monkeypatch)
    subject_hit = _weak_vector_weight_write(tools, "plain content", "unrelated-ws", "deploy checklist")
    content_only = _weak_vector_weight_write(tools, "mentions deploy checklist deep inside", "agent-rail", "misc notes")
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map",
        lambda query, names: {"agent-rail": 0.10, "unrelated-ws": 0.9},
    )
    r = tools.memory_search(query="deploy checklist", workspace="agent-lane")
    ids = [x["id"] for x in r["data"]["results"]]
    assert subject_hit in ids and content_only in ids
    assert ids.index(subject_hit) < ids.index(content_only)


# ── canonical_distance_map (2b) against a real vec table ────────────────────

def test_canonical_distance_map_one_query(tmp_path):
    import json as _json

    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "dm.sqlite3",
        backup_jsonl=tmp_path / "dm.jsonl",
        embedding_model_path=model,
        client="codex",
        agent_id="agent-a",
        workspace="default",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    if not tools.db.state.sqlite_vec_available:  # pragma: no cover - env without sqlite-vec
        pytest.skip("sqlite-vec unavailable")
    assert tools.db.ensure_vec_tables(2) == []
    with tools.db.write_transaction() as conn:
        for name, vector in (
            ("agent-lane", [1.0, 0.0]),
            ("agent-rail", [0.99, 0.141]),   # cosine distance ≈ 0.01-0.02
            ("orthogonal-ws", [0.0, 1.0]),   # cosine distance = 1.0
        ):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
                (name,),
            )
            row = conn.execute("SELECT id FROM workspace_canonicals WHERE name = ?", (name,)).fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                (int(row["id"]), _json.dumps(vector)),
            )

    dmap = tools.db.workspaces.canonical_distance_map("agent-lane", ["agent-rail", "orthogonal-ws"])
    assert set(dmap) == {"agent-rail", "orthogonal-ws"}
    assert dmap["agent-rail"] == pytest.approx(0.0, abs=0.05)
    assert dmap["orthogonal-ws"] == pytest.approx(1.0, abs=0.05)
    # query canonical without a vector → degraded {}
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES ('vecless', '2026-01-01T00:00:00Z')",
        )
    assert tools.db.workspaces.canonical_distance_map("vecless", ["agent-rail"]) == {}
    # default query never enters
    assert tools.db.workspaces.canonical_distance_map("default", ["agent-rail"]) == {}


def test_distance_map_skips_null_distance_instead_of_raising(tmp_path, monkeypatch):
    """Round-1 review fix: degenerate (all-zero) vectors make sqlite-vec return
    SQL NULL for vec_distance_cosine — the map must skip that canonical
    (vectorless → binary fallback), never raise TypeError through search."""
    import json as _json

    monkeypatch.setattr("memory_arbiter.search.WORKSPACE_WEAK_VECTOR_WEIGHT", True)
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "null.sqlite3",
        backup_jsonl=tmp_path / "null.jsonl",
        embedding_model_path=model,
        client="codex",
        agent_id="agent-a",
        workspace="default",
        isolation="weak",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    if not tools.db.state.sqlite_vec_available:  # pragma: no cover
        pytest.skip("sqlite-vec unavailable")
    assert tools.db.ensure_vec_tables(2) == []
    with tools.db.write_transaction() as conn:
        for name, vector in (
            ("agent-lane", [1.0, 0.0]),
            ("zero-ws", [0.0, 0.0]),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
                (name,),
            )
            row = conn.execute("SELECT id FROM workspace_canonicals WHERE name = ?", (name,)).fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                (int(row["id"]), _json.dumps(vector)),
            )

    dmap = tools.db.workspaces.canonical_distance_map("agent-lane", ["zero-ws"])
    assert dmap == {}  # NULL distance skipped, not raised

    tools.memory_write(
        content="zero workspace memory", workspace="zero-ws",
        subject="null distance subject", source_type="agent_generated",
    )
    result = tools.memory_search(query="null distance subject", workspace="agent-lane")
    assert result["ok"] is True  # search never raises; zero-ws record still reachable
