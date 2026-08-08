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
    # Two confirmed writes to the same key (no rejection in between) → 2 events,
    # append-only, current state = last write.
    t.memory_govern("accept_workspace_alias", {"alias": "a", "canonical": "c1"})
    t.memory_govern("accept_workspace_alias", {"alias": "a", "canonical": "c2"})
    events = t.db.list_workspace_alias_events("a")
    assert len(events) == 2
    cur = t.db.get_workspace_alias("a")
    assert cur["canonical"] == "c2" and cur["status"] == "confirmed"
    # earlier event still records the old snapshot (append-only, not overwritten)
    assert any(e["new_canonical"] == "c1" for e in events)


def test_accept_after_reject_refused_without_authorization(tmp_path):
    # A prior rejection must not be silently reversed by a later accept.
    t = make_tools(tmp_path)
    t.memory_govern("accept_workspace_alias", {"alias": "a", "canonical": "c1"})
    t.memory_govern("reject_workspace_alias", {"alias": "a", "canonical": "c1"})
    r = t.memory_govern("accept_workspace_alias", {"alias": "a", "canonical": "c2"})
    assert r["ok"] is False  # refused — a is rejected
    assert t.db.get_workspace_alias("a")["status"] == "rejected"
    # explicit authorized override reverses it deliberately
    r2 = t.memory_govern("accept_workspace_alias", {"alias": "a", "canonical": "c2", "authorized": True})
    assert r2["ok"] is True
    cur = t.db.get_workspace_alias("a")
    assert cur["status"] == "confirmed" and cur["canonical"] == "c2"


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


# ── round-3 review: rename must not clobber a rejected alias with the same key

def test_rename_preserves_rejected_alias(tmp_path):
    t = make_tools(tmp_path)
    # Register canonicals so the rejected alias targets a real one.
    t.db.resolve_workspace_canonical("Foo", None, register_new=True)
    t.db.resolve_workspace_canonical("BarBaz", None, register_new=True)
    # User explicitly rejects: "Foo is NOT BarBaz"
    r = t.memory_govern("reject_workspace_alias", {"alias": "Foo", "canonical": "BarBaz"})
    assert r["ok"] is True
    # Now admin renames the canonical Foo -> NewFoo. The rejection concerns a
    # DIFFERENT canonical (BarBaz) and must survive — silently flipping it to
    # confirmed=NewFoo would reverse the user's decision.
    t.memory_govern("rename_workspace_canonical", {"old": "Foo", "new": "NewFoo"})
    row = t.db.get_workspace_alias("Foo")
    assert row is not None
    assert row["status"] == "rejected", f"rejection was silently flipped: {row}"
    assert row["canonical"] == "BarBaz"


def test_rename_chain_forwards_correctly(tmp_path):
    # Foo -> Bar -> Baz : the Foo forwarding alias must chain to Baz, not stay at Bar.
    t = make_tools(tmp_path)
    t.db.resolve_workspace_canonical("Foo", None, register_new=True)
    t.memory_govern("rename_workspace_canonical", {"old": "Foo", "new": "Bar"})
    t.memory_govern("rename_workspace_canonical", {"old": "Bar", "new": "Baz"})
    resolved = t.db.resolve_workspace_canonical("Foo", None, register_new=False)
    assert resolved["canonical"] == "Baz"
    assert resolved["matched_by"] == "confirmed_alias"


# ── round-3 review: confirm_pending must NOT activate on canonical-write failure

def test_confirm_pending_fails_cleanly_on_missing_memory(tmp_path):
    t = make_tools(tmp_path, isolation="strict")
    # No such memory id → early ok=False, memory never activated.
    r = t.memory_govern("confirm_pending_workspace", {"memory_id": 9999, "canonical": "X"})
    assert r["ok"] is False
    assert not r["data"].get("activated")


# ── round-4 review: migrate registers to_ws + repoints aliases ───────────────

def test_migrate_registers_destination_canonical(tmp_path):
    t = make_tools(tmp_path)
    t.memory_write(content="a", workspace="A", source_type="agent_generated")
    t.memory_govern("migrate_workspace", {"from": "A", "to": "NewProj"})
    # destination must be a registered canonical, not a phantom
    with t.db.connection() as conn:
        names = [r["name"] for r in conn.execute("SELECT name FROM workspace_canonicals")]
    assert "NewProj" in names
    # a later write to the destination is NOT treated as brand-new
    resolved = t.db.resolve_workspace_canonical("NewProj", None, register_new=False)
    assert resolved["is_new"] is False


def test_migrate_repoints_existing_alias(tmp_path):
    t = make_tools(tmp_path)
    t.memory_write(content="a", workspace="Sub2", source_type="agent_generated")
    t.memory_govern("accept_workspace_alias", {"alias": "X", "canonical": "Sub2"})
    t.memory_govern("migrate_workspace", {"from": "Sub2", "to": "Main"})
    # alias X must now forward to Main, not dangle at the migrated-away Sub2
    resolved = t.db.resolve_workspace_canonical("X", None, register_new=False)
    assert resolved["canonical"] == "Main"


# ── round-4 review: confirm_pending must not silently reverse a rejection ─────

def test_confirm_pending_refuses_rejected_alias(tmp_path):
    t = make_tools(tmp_path, isolation="strict")
    t.db.resolve_workspace_canonical("金科营销项目", None, register_new=True)
    # user rejects 项目 == 金科营销项目
    t.memory_govern("reject_workspace_alias", {"alias": "项目", "canonical": "金科营销项目"})
    w = t.memory_write(content="x", workspace="项目", source_type="agent_generated")
    mid = w["data"]["id"]
    r = t.memory_govern("confirm_pending_workspace", {"memory_id": mid, "canonical": "金科营销项目"})
    assert r["ok"] is False  # refused — rejection not silently reversed
    assert t.db.get_workspace_alias("项目")["status"] == "rejected"
    # authorized override succeeds
    r2 = t.memory_govern("confirm_pending_workspace",
                         {"memory_id": mid, "canonical": "金科营销项目", "authorized": True})
    assert r2["ok"] is True
    assert t.db.get_workspace_alias("项目")["status"] == "confirmed"


# ── full-review round: canonical registration & rejection integrity ─────────

def test_accept_alias_registers_target_canonical(tmp_path):
    # A confirmed alias must ensure its target canonical is registered so the
    # resolver's KNN can fuzzy-merge later near-misses. Otherwise a near-string
    # falls through to a fresh sibling canonical, defeating the alias intent.
    t = make_tools(tmp_path)
    r = t.memory_govern("accept_workspace_alias", {"alias": "foo", "canonical": "Bar"})
    assert r["ok"] is True
    with t.db.connection() as c:
        names = [row["name"] for row in c.execute("SELECT name FROM workspace_canonicals")]
    assert "Bar" in names


def test_string_false_authorized_does_not_bypass_rejection_guard(tmp_path):
    # bool("false") is True in Python — a client sending the JSON string
    # "false" for authorized must NOT be treated as a truthy override.
    t = make_tools(tmp_path)
    t.memory_govern("accept_workspace_alias", {"alias": "a", "canonical": "c1"})
    t.memory_govern("reject_workspace_alias", {"alias": "a", "canonical": "c1"})
    r = t.memory_govern("accept_workspace_alias",
                        {"alias": "a", "canonical": "c2", "authorized": "false"})
    assert r["ok"] is False  # not overridden
    assert t.db.get_workspace_alias("a")["status"] == "rejected"


def test_unrecognized_authorized_string_does_not_override(tmp_path):
    # An authorization flag uses an allow-list: only true/1/yes/on grant it.
    # "null"/"maybe"/"" must NOT be treated as an override.
    t = make_tools(tmp_path)
    t.memory_govern("accept_workspace_alias", {"alias": "a", "canonical": "c1"})
    t.memory_govern("reject_workspace_alias", {"alias": "a", "canonical": "c1"})
    for bad in ("null", "maybe", "", "0", "no"):
        r = t.memory_govern("accept_workspace_alias",
                            {"alias": "a", "canonical": "c2", "authorized": bad})
        assert r["ok"] is False, f"authorized={bad!r} wrongly granted override"
    # genuine true tokens work
    r = t.memory_govern("accept_workspace_alias",
                        {"alias": "a", "canonical": "c2", "authorized": "true"})
    assert r["ok"] is True


def test_rename_repoints_rejected_alias_targeting_old(tmp_path):
    # rename(old→new) means the canonical formerly-called-`old` IS now `new`.
    # A rejection "foo is not old" must FOLLOW to "foo is not new" — otherwise
    # the rejected row is stranded on a name the resolver never returns from
    # KNN, and a later write auto-merges foo→new via the vector path,
    # silently reversing the user's decision.
    t = make_tools(tmp_path)
    t.db.resolve_workspace_canonical("old", None, register_new=True)
    t.memory_govern("reject_workspace_alias", {"alias": "foo", "canonical": "old"})
    t.memory_govern("rename_workspace_canonical", {"old": "old", "new": "new"})
    row = t.db.get_workspace_alias("foo")
    # rejection now targets the renamed canonical, still status=rejected
    assert row["status"] == "rejected"
    assert row["canonical"] == "new"
    # and the resolver's rejected filter correctly suppresses `new` for `foo`
    resolved = t.db.resolve_workspace_canonical("foo", None, register_new=False)
    assert "new" in (resolved.get("rejected_canonicals") or [])


def test_merge_rename_repoints_rejected_alias(tmp_path):
    # rename's MERGE branch (new already exists → old canonical row deleted)
    # must also carry the rejection to `new`, not strand it on the deleted `old`.
    t = make_tools(tmp_path)
    t.db.resolve_workspace_canonical("old", None, register_new=True)
    t.db.resolve_workspace_canonical("new", None, register_new=True)
    t.memory_govern("reject_workspace_alias", {"alias": "foo", "canonical": "old"})
    t.memory_govern("rename_workspace_canonical", {"old": "old", "new": "new"})
    row = t.db.get_workspace_alias("foo")
    assert row["status"] == "rejected" and row["canonical"] == "new"


def test_migrate_repoints_rejected_alias(tmp_path):
    t = make_tools(tmp_path)
    t.memory_write(content="a", workspace="Sub2", source_type="agent_generated")
    t.memory_govern("reject_workspace_alias", {"alias": "foo", "canonical": "Sub2"})
    t.memory_govern("migrate_workspace", {"from": "Sub2", "to": "Main"})
    row = t.db.get_workspace_alias("foo")
    assert row["status"] == "rejected" and row["canonical"] == "Main"


def test_resolver_similar_excludes_rejected(tmp_path):
    # Downstream write-hints must never re-surface a rejected pair. The
    # resolver's `similar` list must be filtered.
    from memory_arbiter.config import Settings
    from memory_arbiter.db import MemoryDB
    # This test doesn't need vec — the filter runs even when vec is off.
    t = make_tools(tmp_path)
    t.memory_govern("reject_workspace_alias", {"alias": "queried", "canonical": "SomeProj"})
    resolved = t.db.resolve_workspace_canonical("queried", None, register_new=False)
    names = [s.get("name") for s in resolved.get("similar", [])]
    assert "SomeProj" not in names


def test_qwen_related_relation_does_not_silent_merge(tmp_path):
    # Weak-mode Qwen merge must gate on relation ∈ {alias,typo,same_project}.
    # A high-confidence 'related' or 'unrelated' must NOT silently merge.
    from memory_arbiter.semantic_conflict import WorkspaceCandidateSignal

    class StubBackend:
        def __init__(self, sig): self._sig = sig
        def suggest_workspace_candidate(self, ws, ev, cs): return self._sig

    t = make_tools(tmp_path, isolation="weak")

    def fake_resolve(ws_raw, embedder=None, *, match_distance=None, register_new=True):
        return {"canonical": ws_raw, "is_new": True, "matched_by": "new",
                "distance": None, "similar": [{"name": "金营项目", "distance": 0.4}],
                "rejected_canonicals": []}
    t.db.resolve_workspace_canonical = fake_resolve  # type: ignore
    t._ensure_semantic_backend = lambda: StubBackend(  # type: ignore
        WorkspaceCandidateSignal("金营项目", "related", 0.95, "topical only")
    )
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated")
    # NOT merged — related@0.95 is not identity-grade
    assert r["data"]["workspace_canonical"] != "金营项目"
    assert r["data"]["workspace_decision"] == "ASK"





