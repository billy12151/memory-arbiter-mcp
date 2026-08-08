"""Workspace alias governance (design 637): confirmed/rejected current-state
table + append-only event log, no CAS. Covers resolver short-circuit, rejected
suppression, rename/migrate, confirm-pending, and audit-trail append-only.
"""
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB, _normalize_alias_key
from memory_arbiter.models import MemoryStatus
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path, isolation: str = "weak") -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "gov.sqlite3",
        backup_jsonl=tmp_path / "gov.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        enable_sqlite_vec=False,
        vec_dim=2,
        isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


# ── normalization ───────────────────────────────────────────────────────────

def test_normalize_alias_key_casefold_and_whitespace():
    assert _normalize_alias_key("  金营项目 ") == _normalize_alias_key("金营项目")
    assert _normalize_alias_key("Project  X") == _normalize_alias_key("project x")
    assert _normalize_alias_key("") == ""
    assert _normalize_alias_key(None) == ""


# ── accept → confirmed short-circuit ─────────────────────────────────────────

def test_accept_alias_short_circuits_resolver(tmp_path):
    t = make_tools(tmp_path)
    r = t.memory_govern("accept_workspace_alias",
                        {"alias": "金营二期", "canonical": "金营项目", "reason": "same project"})
    assert r["ok"] is True
    resolved = t.db.resolve_workspace_canonical("金营二期", None, register_new=False)
    assert resolved["canonical"] == "金营项目"
    assert resolved["matched_by"] == "confirmed_alias"


def test_accept_alias_is_casefold_stable(tmp_path):
    t = make_tools(tmp_path)
    t.memory_govern("accept_workspace_alias", {"alias": "Project X", "canonical": "project-x"})
    resolved = t.db.resolve_workspace_canonical("  project  x ", None, register_new=False)
    assert resolved["canonical"] == "project-x"
    assert resolved["matched_by"] == "confirmed_alias"


# ── reject → suppression ─────────────────────────────────────────────────────

def test_reject_alias_records_rejected_canonical(tmp_path):
    t = make_tools(tmp_path)
    r = t.memory_govern("reject_workspace_alias",
                        {"alias": "金营培训", "canonical": "金营项目", "reason": "distinct"})
    assert r["ok"] is True
    resolved = t.db.resolve_workspace_canonical("金营培训", None, register_new=False)
    # not merged to the rejected canonical
    assert resolved["canonical"] != "金营项目"
    assert "金营项目" in resolved["rejected_canonicals"]


# ── validation ───────────────────────────────────────────────────────────────

def test_accept_alias_requires_alias_and_canonical(tmp_path):
    t = make_tools(tmp_path)
    assert t.memory_govern("accept_workspace_alias", {"alias": "x"})["ok"] is False
    assert t.memory_govern("accept_workspace_alias", {"canonical": "y"})["ok"] is False


def test_unknown_govern_action(tmp_path):
    t = make_tools(tmp_path)
    assert t.memory_govern("frobnicate", {})["ok"] is False


# ── audit trail is append-only ───────────────────────────────────────────────

def test_events_are_appended_not_overwritten(tmp_path):
    t = make_tools(tmp_path)
    t.memory_govern("accept_workspace_alias", {"alias": "a", "canonical": "c1"})
    t.memory_govern("reject_workspace_alias", {"alias": "a", "canonical": "c1"})
    t.memory_govern("accept_workspace_alias", {"alias": "a", "canonical": "c2"})
    events = t.db.list_workspace_alias_events("a")
    # three governance actions → three events, current state = last write
    assert len(events) == 3
    actions = [e["action"] for e in events]
    assert set(actions) == {"accept", "reject"}
    cur = t.db.get_workspace_alias("a")
    assert cur["canonical"] == "c2" and cur["status"] == "confirmed"
    # earlier event still records the old snapshot
    assert any(e["new_canonical"] == "c1" for e in events)


# ── rename ───────────────────────────────────────────────────────────────────

def test_rename_canonical_updates_memories(tmp_path):
    t = make_tools(tmp_path)
    t.memory_write(content="alpha note", workspace="OldName", source_type="agent_generated")
    r = t.memory_govern("rename_workspace_canonical", {"old": "OldName", "new": "NewName"})
    assert r["ok"] is True
    assert r["data"]["memories_updated"] >= 1
    events = t.db.list_workspace_alias_events()
    assert any(e["action"] == "rename" for e in events)


# ── migrate ──────────────────────────────────────────────────────────────────

def test_migrate_moves_memories_and_records_alias(tmp_path):
    t = make_tools(tmp_path)
    t.memory_write(content="beta note", workspace="Sub2", source_type="agent_generated")
    r = t.memory_govern("migrate_workspace", {"from": "Sub2", "to": "Main"})
    assert r["ok"] is True
    assert r["data"]["memories_updated"] >= 1
    # alias recorded so future writes short-circuit
    resolved = t.db.resolve_workspace_canonical("Sub2", None, register_new=False)
    assert resolved["canonical"] == "Main"
    assert resolved["matched_by"] == "confirmed_alias"


# ── confirm pending workspace (strict flow) ──────────────────────────────────

def test_confirm_pending_workspace_activates_and_aliases(tmp_path):
    t = make_tools(tmp_path, isolation="strict")
    w = t.memory_write(content="gamma note", workspace="BrandNew", source_type="agent_generated")
    mid = w["data"]["id"]
    # strict + new workspace → pending
    assert t.db.get_memory(mid)["status"] == MemoryStatus.PENDING.value
    r = t.memory_govern("confirm_pending_workspace", {"memory_id": mid, "canonical": "BrandNew"})
    assert r["ok"] is True
    assert r["data"]["activated"] is True
    assert t.db.get_memory(mid)["status"] == MemoryStatus.ACTIVE.value
    # raw workspace now confirmed-aliased to canonical
    resolved = t.db.resolve_workspace_canonical("BrandNew", None, register_new=False)
    assert resolved["matched_by"] == "confirmed_alias"


# ── review hardening: non-string inputs must not crash (tools.py 698/2474/2499)

def test_nonstring_alias_returns_structured_error_not_crash(tmp_path):
    t = make_tools(tmp_path)
    # int / list / dict must degrade to a structured ok=False, never AttributeError
    for bad in (5, ["x"], {"a": 1}):
        r = t.memory_govern("accept_workspace_alias", {"alias": bad, "canonical": "c"})
        assert isinstance(r, dict)  # did not raise
        r2 = t.memory_govern("reject_workspace_alias", {"alias": "a", "canonical": bad})
        assert isinstance(r2, dict)


def test_nonstring_rename_migrate_do_not_crash(tmp_path):
    t = make_tools(tmp_path)
    assert isinstance(t.memory_govern("rename_workspace_canonical", {"old": {"a": 1}, "new": "p"}), dict)
    assert isinstance(t.memory_govern("migrate_workspace", {"from": ["x"], "to": "p"}), dict)
    assert isinstance(t.memory_govern("confirm_pending_workspace", {"memory_id": 1, "canonical": ["p"]}), dict)


# ── review hardening: rename into an existing canonical must merge, not orphan

def test_rename_into_existing_canonical_merges(tmp_path):
    t = make_tools(tmp_path)
    t.memory_write(content="a", workspace="OldName", source_type="agent_generated")
    t.memory_write(content="b", workspace="NewName", source_type="agent_generated")
    r = t.memory_govern("rename_workspace_canonical", {"old": "OldName", "new": "NewName"})
    assert r["ok"] is True
    # old canonical row must be gone (merged), not silently left behind
    with t.db.connection() as conn:
        rows = [row["name"] for row in conn.execute("SELECT name FROM workspace_canonicals")]
    assert "OldName" not in rows
    assert "NewName" in rows


def test_rename_repoints_confirmed_alias(tmp_path):
    t = make_tools(tmp_path)
    t.memory_write(content="a", workspace="OldName", source_type="agent_generated")
    t.memory_govern("accept_workspace_alias", {"alias": "jinying", "canonical": "OldName"})
    t.memory_govern("rename_workspace_canonical", {"old": "OldName", "new": "NewName"})
    # alias must now resolve to the renamed canonical, not the dead old name
    resolved = t.db.resolve_workspace_canonical("jinying", None, register_new=False)
    assert resolved["canonical"] == "NewName"


# ── review hardening: migrate records alias atomically (same call)

def test_migrate_records_alias_in_same_call(tmp_path):
    t = make_tools(tmp_path)
    t.memory_write(content="a", workspace="Sub2", source_type="agent_generated")
    t.memory_govern("migrate_workspace", {"from": "Sub2", "to": "Main"})
    # alias present immediately (no separate-transaction gap)
    cur = t.db.get_workspace_alias("Sub2")
    assert cur is not None and cur["canonical"] == "Main" and cur["status"] == "confirmed"
    events = t.db.list_workspace_alias_events("Sub2")
    assert any(e["action"] == "migrate" for e in events)


# ── round-2 review: confirm_pending actually writes the canonical column ──────

def test_confirm_pending_actually_sets_canonical_column(tmp_path):
    t = make_tools(tmp_path, isolation="strict")
    w = t.memory_write(content="g", workspace="金营二期", source_type="agent_generated")
    mid = w["data"]["id"]
    r = t.memory_govern("confirm_pending_workspace", {"memory_id": mid, "canonical": "金营项目"})
    assert r["ok"] is True and r["data"]["confirmed"] is True
    # the column must actually be written (update_memory whitelist used to drop it)
    assert t.db.get_memory(mid)["workspace_canonical"] == "金营项目"


# ── round-2 review: non-string workspace fields → structured error, not garbage

def test_nonstring_workspace_rejected_not_stringified(tmp_path):
    t = make_tools(tmp_path)
    r = t.memory_govern("accept_workspace_alias", {"alias": "realproj", "canonical": ["x"]})
    assert r["ok"] is False  # rejected, not stored as "['x']"
    # nothing got written under the garbage canonical
    resolved = t.db.resolve_workspace_canonical("realproj", None, register_new=False)
    assert resolved["matched_by"] != "confirmed_alias"

    r2 = t.memory_govern("migrate_workspace", {"from": "SubProj", "to": ["Main"]})
    assert r2["ok"] is False


# ── round-2 review: rename inserts forwarding alias, no re-split ──────────────

def test_rename_inserts_forwarding_alias_no_resplit(tmp_path):
    t = make_tools(tmp_path)
    t.db.resolve_workspace_canonical("Foo", None, register_new=True)
    t.memory_govern("rename_workspace_canonical", {"old": "Foo", "new": "Bar"})
    # re-submitting the OLD raw workspace must resolve to the new canonical,
    # not re-register "Foo" as a fresh split canonical.
    resolved = t.db.resolve_workspace_canonical("Foo", None, register_new=True)
    assert resolved["canonical"] == "Bar"
    assert resolved["matched_by"] == "confirmed_alias"
    assert resolved["is_new"] is False


