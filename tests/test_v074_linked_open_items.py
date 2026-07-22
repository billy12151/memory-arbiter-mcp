"""v0.7.4 tests: linked_open_items (M1-M4) + memory_complete_open_item (M5).

Covers the acceptance checklist from the v0.7.4 design (id=242):
  A. linked_open_items main flow (dirty tags, generic-tag stoplist, modes, …)
  B. retrieval_mode / SearchOutcome contract
  C. server wrapper parameter pass-through
  D. memory_complete_open_item outcomes + transaction safety
  plus the M3 workspace-no-filter guarantee at the search layer.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def _tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "b.jsonl",
        client="test",
        agent_id="tester",
        workspace="repo-a",
        enable_sqlite_vec=False,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write(
    tools: MemoryTools,
    *,
    content: str,
    subject: str,
    tags: list[str],
    workspace: str = "ws",
    source_type: str = "agent_generated",
    protection_level: str = "normal",
    ingest_time: str | None = None,
) -> int:
    payload: dict[str, Any] = {
        "content": content,
        "subject": subject,
        "tags": tags,
        "workspace": workspace,
        "source_type": source_type,
        "protection_level": protection_level,
        "agent_id": "tester",
    }
    if ingest_time is not None:
        payload["ingest_time"] = ingest_time
    res = tools.memory_write(**payload)
    assert res["ok"], f"write failed: {res}"
    return res["data"]["id"]


# ──────────────────────────────────────────────────────────────────────────
#  A. linked_open_items main flow
# ──────────────────────────────────────────────────────────────────────────

def test_linked_open_items_basic(tmp_path: Path) -> None:
    """A todo sharing a meaningful tag with the result set is attached.

    The todo's content deliberately does NOT contain the search term, so it is
    not itself recalled into results — only the tag links it. The search term
    is chosen so its trigrams don't appear inside any tag either.
    """
    tools = _tools(tmp_path)
    _write(tools, content="quantum mechanics overview", subject="physics", tags=["feature-auth"])
    todo_id = _write(
        tools, content="follow up next week", subject="reminder",
        tags=["todo", "feature-auth"], ingest_time="2026-07-01T00:00:00Z",
    )
    res = tools.memory_search(query="quantum")
    assert res["data"]["retrieval_mode"] == "direct"
    linked = res["data"]["linked_open_items"]
    assert len(linked) == 1
    assert linked[0]["id"] == todo_id
    assert "feature-auth" in linked[0]["reason"]


def test_linked_open_items_disabled(tmp_path: Path) -> None:
    """include_linked_open_items=False suppresses the side query entirely."""
    tools = _tools(tmp_path)
    _write(tools, content="how auth works", subject="auth design", tags=["feature-auth"])
    _write(tools, content="follow up", subject="reminder", tags=["todo", "feature-auth"])
    res = tools.memory_search(query="auth", include_linked_open_items=False)
    assert res["data"]["linked_open_items"] == []


def test_linked_open_items_empty_when_no_todo(tmp_path: Path) -> None:
    """No active todo anywhere → linked=[] (L1 EXISTS short-circuit)."""
    tools = _tools(tmp_path)
    _write(tools, content="how auth works", subject="auth design", tags=["feature-auth"])
    res = tools.memory_search(query="auth")
    assert res["data"]["linked_open_items"] == []


def test_linked_open_items_excludes_main_results(tmp_path: Path) -> None:
    """A todo that also matched the query must not appear twice."""
    tools = _tools(tmp_path)
    _write(tools, content="auth documentation", subject="auth doc", tags=["feature-auth"])
    todo_id = _write(
        tools, content="auth implementation detail", subject="auth todo",
        tags=["todo", "feature-auth"],
    )
    res = tools.memory_search(query="auth")
    result_ids = [r["id"] for r in res["data"]["results"]]
    linked_ids = [l["id"] for l in res["data"]["linked_open_items"]]
    # The todo was recalled into results (its content matches), so it must not
    # also appear in linked_open_items.
    assert todo_id in result_ids
    assert todo_id not in linked_ids


def test_linked_open_items_max_five(tmp_path: Path) -> None:
    """More than 5 matching todos → only 5 returned.

    Uses a distinct tag per todo (df=2 each: hub + one todo) so none hit the
    df>=3 branch of the stoplist — isolating the truncation check from the
    generic-tag filter.
    """
    tools = _tools(tmp_path)
    _write(
        tools, content="central hub document", subject="hub",
        tags=["t0", "t1", "t2", "t3", "t4", "t5", "t6"],
    )
    for i in range(7):
        _write(
            tools, content=f"task number {i}", subject=f"task{i}",
            tags=["todo", f"t{i}"], ingest_time=f"2026-07-0{i+1}T00:00:00Z",
        )
    res = tools.memory_search(query="hub")
    assert len(res["data"]["linked_open_items"]) == 5


def test_linked_open_items_generic_tag_filtered(tmp_path: Path) -> None:
    """A tag appearing in ≥20% of active memories (df≥3) is stoplisted."""
    tools = _tools(tmp_path)
    # 5 memories all tagged "common" → df=5, active_count=5, 5/5=1.0 ≥ 0.20.
    for i in range(5):
        _write(tools, content=f"common item {i}", subject=f"c{i}", tags=["common"])
    # todo that ONLY shares the generic tag — no meaningful overlap.
    _write(tools, content="a follow up", subject="todo", tags=["todo", "common"])
    res = tools.memory_search(query="common")
    # "common" is stoplisted, so the todo has zero meaningful overlap → [].
    assert res["data"]["linked_open_items"] == []


def test_linked_open_items_only_direct_mode(tmp_path: Path) -> None:
    """Fallback / browse modes never trigger linked_open_items."""
    tools = _tools(tmp_path)
    _write(tools, content="follow up", subject="reminder", tags=["todo", "feature-auth"])
    # No real match → recent_fallback; linked must stay empty.
    res_fb = tools.memory_search(query="zzz_no_match_xyz")
    assert res_fb["data"]["retrieval_mode"] == "recent_fallback"
    assert res_fb["data"]["linked_open_items"] == []
    # Empty query → recent_browse; linked must stay empty.
    res_br = tools.memory_search(query="")
    assert res_br["data"]["retrieval_mode"] == "recent_browse"
    assert res_br["data"]["linked_open_items"] == []


def test_linked_open_items_dirty_tags_silent(tmp_path: Path) -> None:
    """A memory with malformed JSON tags is silently skipped — no warning."""
    tools = _tools(tmp_path)
    _write(tools, content="auth doc", subject="auth", tags=["feature-auth"])
    # Corrupt one memory's tags directly (bypasses memory_write validation).
    bad_id = _write(tools, content="other", subject="other", tags=["ok"])
    with tools.db.connection() as conn:
        conn.execute("UPDATE memories SET tags=? WHERE id=?", ("not-json", bad_id))
        conn.commit()
    res = tools.memory_search(query="auth")
    assert res["ok"]
    # The dirty row did not crash the side query and produced no warning.
    assert not any("linked_open_items" in w for w in res["warnings"])


def test_linked_open_items_db_failure_degrades(tmp_path: Path) -> None:
    """A real DB error in the side query → linked=[] + degradation warning.

    Exercises the helper directly so the main search path is unaffected.
    """
    from memory_arbiter.search import _linked_open_items_for_search

    tools = _tools(tmp_path)
    _write(tools, content="auth doc", subject="auth", tags=["feature-auth"])
    results = [{"id": 1, "tags": ["feature-auth"]}]
    warnings: list[str] = []

    def boom() -> sqlite3.Connection:
        raise sqlite3.Error("simulated")

    original = tools.db._new_connection
    tools.db._new_connection = boom  # type: ignore[method-assign]
    try:
        out = _linked_open_items_for_search(tools.db, results, warnings)
    finally:
        tools.db._new_connection = original  # type: ignore[method-assign]
    assert out == []
    assert any("linked_open_items" in w for w in warnings)


def test_linked_open_items_tag_exact_match(tmp_path: Path) -> None:
    """A memory tagged 'done' (not 'todo') is never picked up as a todo."""
    tools = _tools(tmp_path)
    _write(tools, content="auth doc", subject="auth", tags=["feature-auth"])
    _write(tools, content="finished item", subject="done", tags=["done", "feature-auth"])
    res = tools.memory_search(query="auth")
    assert res["data"]["linked_open_items"] == []


def test_coerce_tags_handles_all_shapes() -> None:
    """_coerce_tags normalises list/JSON-string/malformed/scalar/null safely."""
    from memory_arbiter.search import _coerce_tags

    assert _coerce_tags(["a", "b", "a"]) == ["a", "b"]
    assert _coerce_tags('["x", "y"]') == ["x", "y"]
    assert _coerce_tags("not-json") == []
    assert _coerce_tags(None) == []
    assert _coerce_tags(42) == []
    assert _coerce_tags({"k": "v"}) == []
    assert _coerce_tags(["a", 1, None, "a"]) == ["a"]


# ──────────────────────────────────────────────────────────────────────────
#  B. retrieval_mode / SearchOutcome contract
# ──────────────────────────────────────────────────────────────────────────

def test_retrieval_mode_direct_and_empty(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    _write(tools, content="hello world", subject="greet", tags=[])
    direct = tools.memory_search(query="hello")
    assert direct["data"]["retrieval_mode"] == "direct"
    # tags_filter that matches nothing → empty.
    empty = tools.memory_search(query="hello", tags_filter=["nonexistent-tag"])
    assert empty["data"]["retrieval_mode"] == "empty"
    assert empty["data"]["results"] == []


# ──────────────────────────────────────────────────────────────────────────
#  C. server wrapper parameter pass-through
# ──────────────────────────────────────────────────────────────────────────

def test_search_result_envelope_has_linked_field(tmp_path: Path) -> None:
    """The tools-layer response always carries the linked_open_items key."""
    tools = _tools(tmp_path)
    _write(tools, content="doc", subject="d", tags=["x"])
    res = tools.memory_search(query="doc")
    assert "linked_open_items" in res["data"]
    assert "retrieval_mode" in res["data"]


# ──────────────────────────────────────────────────────────────────────────
#  D. memory_complete_open_item (M5)
# ──────────────────────────────────────────────────────────────────────────

def test_complete_open_item_removes_todo(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    mem_id = _write(tools, content="do it", subject="task", tags=["todo", "feature"])
    res = tools.memory_complete_open_item(memory_id=mem_id)
    assert res["ok"]
    assert res["data"]["completed"] is True
    assert res["data"]["already_completed"] is False
    assert "todo" not in res["data"]["tags"]
    assert "feature" in res["data"]["tags"]
    # Version bumped, content unchanged.
    mem = tools.db.get_memory(mem_id)
    assert mem["version"] == 2
    assert mem["content"] == "do it"


def test_complete_open_item_already_completed(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    mem_id = _write(tools, content="done", subject="task", tags=["feature"])
    res = tools.memory_complete_open_item(memory_id=mem_id)
    assert res["ok"]
    assert res["data"]["completed"] is False
    assert res["data"]["already_completed"] is True
    # Zero writes: version unchanged.
    assert tools.db.get_memory(mem_id)["version"] == 1


def test_complete_open_item_idempotent(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    mem_id = _write(tools, content="do it", subject="task", tags=["todo"])
    first = tools.memory_complete_open_item(memory_id=mem_id)
    assert first["data"]["completed"] is True
    second = tools.memory_complete_open_item(memory_id=mem_id)
    assert second["data"]["completed"] is False
    assert second["data"]["already_completed"] is True
    # Only one version bump total.
    assert tools.db.get_memory(mem_id)["version"] == 2


def test_complete_open_item_not_found(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    res = tools.memory_complete_open_item(memory_id=99999)
    assert res["ok"] is False
    assert "not found" in res["data"]["error"]


def test_complete_open_item_not_active(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    mem_id = _write(tools, content="do it", subject="task", tags=["todo"])
    tools.memory_supersede(memory_id=mem_id, reason="done elsewhere", authorized=True)
    res = tools.memory_complete_open_item(memory_id=mem_id)
    assert res["ok"] is False
    assert "not active" in res["data"]["error"]


def test_complete_open_item_forbidden_without_authorization(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    mem_id = _write(
        tools, content="protected todo", subject="pt", tags=["todo"],
        protection_level="locked", source_type="user_confirmed",
    )
    res = tools.memory_complete_open_item(memory_id=mem_id)
    assert res["ok"] is False
    assert "authorized=True" in res["data"]["error"]
    # Zero writes: tag unchanged.
    assert "todo" in tools.db.get_memory(mem_id)["tags"]


def test_complete_open_item_authorized_succeeds(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    mem_id = _write(
        tools, content="protected todo", subject="pt", tags=["todo", "keep"],
        protection_level="locked", source_type="user_confirmed",
    )
    res = tools.memory_complete_open_item(memory_id=mem_id, authorized=True)
    assert res["ok"]
    assert res["data"]["completed"] is True
    mem = tools.db.get_memory(mem_id)
    assert "todo" not in mem["tags"]
    assert "keep" in mem["tags"]


def test_complete_open_item_removes_from_linked(tmp_path: Path) -> None:
    """After completion, the memory no longer appears in linked_open_items."""
    tools = _tools(tmp_path)
    _write(tools, content="quantum documentation", subject="physics", tags=["feature-auth"])
    todo_id = _write(
        tools, content="follow up later", subject="reminder", tags=["todo", "feature-auth"],
    )
    before = tools.memory_search(query="quantum")
    assert any(l["id"] == todo_id for l in before["data"]["linked_open_items"])
    tools.memory_complete_open_item(memory_id=todo_id)
    after = tools.memory_search(query="auth")
    assert not any(l["id"] == todo_id for l in after["data"]["linked_open_items"])


def test_complete_open_item_preserves_sections_and_content(tmp_path: Path) -> None:
    """Completing a todo must not alter content/subject/sections/split_status."""
    tools = _tools(tmp_path)
    mem_id = _write(tools, content="body text here", subject="s", tags=["todo"])
    tools.memory_complete_open_item(memory_id=mem_id)
    mem = tools.db.get_memory(mem_id)
    assert mem["content"] == "body text here"
    assert mem["subject"] == "s"
    assert mem["split_status"] is None


# ──────────────────────────────────────────────────────────────────────────
#  M3: workspace is a reserved no-op at the search layer too
# ──────────────────────────────────────────────────────────────────────────

def test_search_ignores_workspace_filter(tmp_path: Path) -> None:
    """Passing workspace= does not hide memories from other workspaces."""
    tools = _tools(tmp_path)
    _write(tools, content="auth in repo-a", subject="a", tags=[], workspace="repo-a")
    _write(tools, content="auth in repo-b", subject="b", tags=[], workspace="repo-b")
    res = tools.memory_search(query="auth", workspace="repo-a")
    subjects = {r["subject"] for r in res["data"]["results"]}
    # Both visible — workspace="repo-a" must NOT hide repo-b's memory.
    assert "a" in subjects
    assert "b" in subjects


# ──────────────────────────────────────────────────────────────────────────
#  Gap coverage: tests the design checklist called for but the original
#  v0.7.4 suite omitted (items 10, 13, 17, 22).
# ──────────────────────────────────────────────────────────────────────────

def test_complete_open_item_fts_failure_rolls_back(tmp_path: Path) -> None:
    """Design item 22: a mid-transaction FTS failure must roll back EVERYTHING.

    history INSERT, memories UPDATE, and FTS re-sync all run in one
    write_transaction(); if any statement raises, no partial write leaks
    (no history row, tags/version unchanged). Drops memories_fts to force
    the FTS delete/insert inside the transaction to raise.
    """
    tools = _tools(tmp_path)
    mem_id = _write(tools, content="do it", subject="task", tags=["todo", "feature"])
    tags_before = tools.db.get_memory(mem_id)["tags"]
    version_before = tools.db.get_memory(mem_id)["version"]
    history_before = len(tools.db.list_history(mem_id))
    # Sabotage FTS so the in-transaction re-sync raises sqlite3.Error.
    with tools.db.connection() as conn:
        conn.execute("DROP TABLE memories_fts")
        conn.commit()
    # complete_open_item still believes FTS is available (state is probed once
    # at startup), so it enters the FTS branch and hits the dropped table.
    result = tools.db.complete_open_item(mem_id, reason="fts probe")
    assert result["outcome"] == "error"
    mem_after = tools.db.get_memory(mem_id)
    # Zero partial writes: tags, version, and history all unchanged.
    assert mem_after["tags"] == tags_before
    assert mem_after["version"] == version_before
    assert len(tools.db.list_history(mem_id)) == history_before


def test_linked_open_items_sort_stability(tmp_path: Path) -> None:
    """Design item 10: score DESC → ingest_time DESC → id DESC.

    Uses TWO distinctive hub tags so two todos can each share a different
    meaningful tag (df stays 1 hub + 1 todo = 2 each, well under stoplist),
    letting us hold score equal while varying ingest_time and id.
    """
    tools = _tools(tmp_path)
    # Hub carries two tags; each todo shares exactly one (df=2 each, meaningful).
    _write(tools, content="hub doc", subject="hub", tags=["soloA", "soloB"])
    t_early = _write(tools, content="todo early", subject="te",
                     tags=["todo", "soloA"], ingest_time="2026-07-01T00:00:00Z")
    t_late = _write(tools, content="todo late", subject="tl",
                    tags=["todo", "soloB"], ingest_time="2026-07-10T00:00:00Z")
    res = tools.memory_search(query="hub")
    linked = res["data"]["linked_open_items"]
    ids = [item["id"] for item in linked]
    assert len(linked) == 2
    # Equal score (1 meaningful tag each); ingest_time DESC → late before early.
    assert ids[0] == t_late
    assert ids[1] == t_early

    # id DESC tier: two todos with equal score AND equal ingest_time, sharing
    # the same tag (df=3: hub + 2 todos; 3/4 = 0.75 is stoplisted, so use two
    # DISTINCT shared tags via a second hub to keep df=2). Simpler: directly
    # verify the sort key tuple ordering with a third todo at t_late's time.
    tools2 = _tools(tmp_path)
    _write(tools2, content="hub2", subject="hub2", tags=["zeta", "gamma"])
    ta = _write(tools2, content="a", subject="a", tags=["todo", "zeta"],
                ingest_time="2026-07-10T00:00:00Z")
    tb = _write(tools2, content="b", subject="b", tags=["todo", "gamma"],
                ingest_time="2026-07-10T00:00:00Z")
    res2 = tools2.memory_search(query="hub2")
    ids2 = [item["id"] for item in res2["data"]["linked_open_items"]]
    # Same score, same ingest_time → id DESC: higher id first.
    assert ids2[0] == max(ta, tb)


def test_linked_open_items_duplicate_tag_no_inflation(tmp_path: Path) -> None:
    """Design item 13: a tag repeated within one memory must not inflate score.

    A todo whose tags repeat the matched tag (e.g. ['todo','solo','solo','solo'])
    must score the same as one with a single occurrence (['todo','solo']):
    matched_meaningful is a set, and _coerce_tags dedupes. We assert the
    deduped tag set and that the reason names each tag only once.
    """
    from memory_arbiter.search import _coerce_tags

    # _coerce_tags dedupes: repeated tags collapse to one.
    assert _coerce_tags(["todo", "solo", "solo", "solo"]).count("solo") == 1
    tools = _tools(tmp_path)
    for i in range(14):
        _write(tools, content=f"filler {i}", subject=f"f{i}", tags=["filler"],
               ingest_time="2026-06-01T00:00:00Z")
    _write(tools, content="hub doc", subject="hub", tags=["solo"])
    dup_id = _write(tools, content="dup tags", subject="dup",
                    tags=["todo", "solo", "solo", "solo"],
                    ingest_time="2026-07-01T00:00:00Z")
    single_id = _write(tools, content="single tag", subject="single",
                       tags=["todo", "solo"], ingest_time="2026-07-01T00:00:00Z")
    res = tools.memory_search(query="hub")
    linked = res["data"]["linked_open_items"]
    by_id = {item["id"]: item for item in linked}
    # Both must appear (each shares 'solo', df=3/17 < 0.20, meaningful).
    assert dup_id in by_id and single_id in by_id
    # reason names 'solo' exactly once — no duplicate-tag inflation in output.
    assert by_id[dup_id]["reason"].count("solo") == 1
    assert by_id[dup_id]["reason"] == by_id[single_id]["reason"]


def test_server_wrapper_passes_include_linked_open_items(tmp_path: Path) -> None:
    """Design item 17: include_linked_open_items must travel through the MCP
    server wrapper (@app.tool()), not just when calling tools.* directly.

    Builds the real FastMCP server, extracts the registered memory_search tool
    function, and calls it with include_linked_open_items=False/True to confirm
    the param reaches the search layer.
    """
    from unittest.mock import patch
    from memory_arbiter import server as srv

    settings = Settings(
        db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl",
        client="test", agent_id="tester", workspace="ws", enable_sqlite_vec=False,
    )
    with patch("memory_arbiter.server.Settings.from_env", return_value=settings):
        app = srv.build_server()
    tools_reg = app._tool_manager._tools  # type: ignore[attr-defined]
    search_fn = tools_reg["memory_search"].fn
    # Signature exposes the param.
    import inspect
    assert "include_linked_open_items" in str(inspect.signature(search_fn))
    # Seed data: a doc plus a linked todo sharing a distinctive tag.
    my_tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    my_tools.memory_write(content="probe doc", subject="probe", tags=["solo"],
                          workspace="ws", source_type="agent_generated", agent_id="t")
    my_tools.memory_write(content="reminder", subject="td", tags=["todo", "solo"],
                          workspace="ws", source_type="agent_generated", agent_id="t")
    disabled = search_fn(query="probe", include_linked_open_items=False)
    enabled = search_fn(query="probe", include_linked_open_items=True)
    assert disabled["data"]["linked_open_items"] == []
    assert len(enabled["data"]["linked_open_items"]) == 1

