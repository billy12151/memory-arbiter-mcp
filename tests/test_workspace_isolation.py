"""Workspace three-tier isolation + alias canonicalization (none/weak/strict).

Design (see memory: workspace-isolation-design):
  - none   : workspace ignored on recall; no ordering effect; new ws silent.
  - weak   : recall always whole-DB (never filtered); passing ws only soft-reranks
             (same-ws boost, cross-ws penalty). New ws → write_hint.
  - strict : write requires ws (empty → error); recall without ws → error;
             recall with ws → hard-filter to same canonical. New ws → blocked as
             status=pending + action_required=confirm_new_workspace; activated via
             memory_activate(authorized=true).

Alias canonicalization (double-store): memories.workspace (raw) +
memories.workspace_canonical (resolved). Runs only when isolation != none.
Without an embedder it degrades to exact string identity.
"""
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import MemoryStatus
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


def _write(tools: MemoryTools, content: str, workspace: str = "default", **kw) -> dict:
    return tools.memory_write(content=content, workspace=workspace, source_type="agent_generated", **kw)


def _results(search: dict) -> list:
    return (search.get("data") or {}).get("results") or []


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


# ── none: workspace has no effect ───────────────────────────────────────────

def test_none_workspace_no_canonical_computed(tmp_path):
    """none must not invoke the embedder or write a canonical (silent)."""
    tools = make_tools(tmp_path, "none")
    r = _write(tools, "alpha content", "projA")
    # canonical column falls back to raw (insert_memory never NULLs it), but
    # no alias resolution ran and no hint/action is attached.
    assert r["data"].get("action_required") is None
    assert (r["data"].get("write_hints") or {}).get("new_workspace_detected") is None


def test_none_recall_ignores_workspace(tmp_path):
    tools = make_tools(tmp_path, "none")
    _write(tools, "alpha marketing content", "projA")
    _write(tools, "beta marketing content", "projB")
    with_ws = _results(tools.memory_search(query="marketing", workspace="projA", limit=10))
    without = _results(tools.memory_search(query="marketing", limit=10))
    # Same result set + same order regardless of workspace.
    assert [m["id"] for m in with_ws] == [m["id"] for m in without]
    assert len(with_ws) == 2


# ── strict: mandatory ws, hard filter, blocking new ws ──────────────────────

def test_strict_write_without_workspace_errors(tmp_path):
    tools = make_tools(tmp_path, "strict")
    r = tools.memory_write(content="no ws", source_type="agent_generated", workspace="")
    assert r["ok"] is False
    assert "strict" in (r["data"].get("error") or "").lower()


def test_strict_recall_without_workspace_errors(tmp_path):
    tools = make_tools(tmp_path, "strict")
    tools.memory_write(content="x", workspace="projA", source_type="agent_generated")
    r = tools.memory_search(query="x", limit=10)
    assert r["ok"] is False
    assert "strict" in (r["data"].get("error") or "").lower()


def test_strict_new_workspace_blocks_as_pending(tmp_path):
    tools = make_tools(tmp_path, "strict")
    r = _write(tools, "first in new ws", "projA")
    d = r["data"]
    assert d.get("action_required") == "confirm_new_workspace"
    assert d.get("verification_status") == "pending_user"
    assert (d.get("record") or {}).get("status") == MemoryStatus.PENDING.value


def test_strict_pending_excluded_from_recall_until_activated(tmp_path):
    tools = make_tools(tmp_path, "strict")
    r = _write(tools, "pending content", "projA")
    mid = r["data"]["id"]
    # pending → excluded from same-ws recall
    assert _results(tools.memory_search(query="pending", workspace="projA")) == []
    act = tools.memory_activate(memory_id=mid, authorized=True)
    assert act["data"]["activated"] is True
    assert (act["data"]["record"]).get("status") == MemoryStatus.ACTIVE.value
    # now recallable
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


def test_strict_hard_filter_same_canonical_only(tmp_path):
    tools = make_tools(tmp_path, "strict")
    # activate both distinct workspaces
    for ws, content in [("projA", "apple content"), ("projB", "apple content two")]:
        mid = _write(tools, content, ws)["data"]["id"]
        tools.memory_activate(memory_id=mid, authorized=True)
    res = _results(tools.memory_search(query="apple", workspace="projA", limit=10))
    assert {m["workspace"] for m in res} == {"projA"}


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
