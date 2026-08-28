"""Unit contract tests for the unified alias-decision primitive.

``WorkspaceStore._apply_alias_decision_on_conn`` is the single primitive behind
``record_workspace_decision_on_conn`` and ``_install_workspace_redirect_on_conn``.
Covered contract points (PR-C1):
  * guard union: non-empty inputs, default-term refused in both directions,
    status enum {confirmed, rejected};
  * self-pair (and mechanical-twin self-pair) no-op writes nothing;
  * first-seen orthography: a registered mechanical twin spelling wins
    (AgentLane vs agent-lane); rejected targets never register;
  * the post-substitution default recheck: a mechanical variant of the
    registered default pool ('de-fault' -> 'default') still refuses in both
    directions, so the twin fold cannot bypass the reserved-pool guard;
  * rejected pairs block by mechanical key, so a ghost spelling variant
    (GhostB vs ghost-b) cannot bypass the refusal; the match spans
    alias-side spelling variants too (rejected 'agent_lane'→'X' blocks
    confirm 'agent-lane'→'X'), and force clears every matched row under its
    own stored alias spelling before confirming;
  * a confirmation unconditionally clears other confirmed rows of the alias;
  * ON CONFLICT(alias_workspace, canonical) upsert;
  * read side: resolve_workspace_canonical expands rejected_canonicals with
    registered canonicals sharing a rejected target's mechanical key.
"""
from pathlib import Path
from typing import Optional

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB


def make_db(tmp_path: Path) -> MemoryDB:
    settings = Settings(
        db_path=tmp_path / "prim.sqlite3",
        backup_jsonl=tmp_path / "prim.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        enable_sqlite_vec=False, vec_dim=2, isolation="weak",
    )
    return MemoryDB(settings)


def apply_decision(
    db: MemoryDB, workspace: str, canonical: str,
    *, status: str = "confirmed", force: bool = False,
) -> "tuple[bool, list[str]]":
    with db.write_transaction() as conn:
        return db.workspaces._apply_alias_decision_on_conn(
            conn, workspace, canonical, status=status, force=force,
        )


def alias_rows(db: MemoryDB) -> "list[tuple[str, str, str]]":
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT alias_workspace,canonical,status FROM workspace_aliases "
            "ORDER BY alias_workspace,canonical"
        ).fetchall()
    return [tuple(row) for row in rows]


def canonical_names(db: MemoryDB) -> list:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT name FROM workspace_canonicals ORDER BY name"
        ).fetchall()
    return [str(row["name"]) for row in rows]


def register_canonical(db: MemoryDB, name: str) -> None:
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES(?,?)",
            (name, "2026-01-01T00:00:00+00:00"),
        )


# ---------------------------------------------------------------------------
#  Guard union
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workspace", ["", "   "])
def test_guard_rejects_empty_workspace_name(tmp_path: Path, workspace: str) -> None:
    db = make_db(tmp_path)
    ok, errors = apply_decision(db, workspace, "proj")
    assert ok is False
    assert errors == ["workspace name must be non-empty."]
    assert alias_rows(db) == []


@pytest.mark.parametrize("canonical", ["", "   "])
def test_guard_rejects_empty_canonical(tmp_path: Path, canonical: str) -> None:
    db = make_db(tmp_path)
    ok, errors = apply_decision(db, "proj", canonical)
    assert ok is False
    assert errors == ["canonical must be a non-empty workspace string."]
    assert alias_rows(db) == []


@pytest.mark.parametrize("workspace,canonical", [
    ("default", "proj"),
    ("proj", "default"),
    ("默认", "proj"),
    ("proj", "NONE"),  # default terms are matched case-insensitively
])
def test_default_term_refused_in_both_directions(
    tmp_path: Path, workspace: str, canonical: str,
) -> None:
    db = make_db(tmp_path)
    ok, errors = apply_decision(db, workspace, canonical)
    assert ok is False
    assert "reserved global pool" in errors[0]
    assert alias_rows(db) == []


@pytest.mark.parametrize("status", ["confirmed", "rejected"])
def test_mechanical_twin_of_registered_default_still_refused(
    tmp_path: Path, status: str,
) -> None:
    db = make_db(tmp_path)
    # Any default-pool write registers the literal 'default' canonical row.
    # 'de-fault' is not a verbatim reserved term, so it passes the front
    # guard — but it folds mechanically onto the registered 'default', and the
    # post-substitution recheck must refuse it with the same default message.
    register_canonical(db, "default")
    ok, errors = apply_decision(db, "proj-x", "de-fault", status=status)
    assert ok is False
    assert "reserved global pool" in errors[0]
    assert alias_rows(db) == []
    assert canonical_names(db) == ["default"]


def test_invalid_status_rejected(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    ok, errors = apply_decision(db, "proj", "target", status="pending")
    assert ok is False
    assert "status='pending' invalid" in errors[0]
    assert alias_rows(db) == []


# ---------------------------------------------------------------------------
#  Self-pair no-op
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["confirmed", "rejected"])
def test_self_pair_is_noop_and_writes_nothing(tmp_path: Path, status: str) -> None:
    db = make_db(tmp_path)
    ok, errors = apply_decision(db, "ProjX", " projx ", status=status)
    assert (ok, errors) == (True, [])
    assert alias_rows(db) == []
    assert canonical_names(db) == []


def test_mechanical_twin_self_pair_is_noop(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    register_canonical(db, "AgentLane")
    # 'agentlane' resolves mechanically to the registered 'AgentLane', which
    # normalizes back to the alias key — a self-pair after resolution.
    ok, errors = apply_decision(db, "agentlane", "agent-lane")
    assert (ok, errors) == (True, [])
    assert alias_rows(db) == []
    assert canonical_names(db) == ["AgentLane"]


# ---------------------------------------------------------------------------
#  First-seen orthography / mechanical twins
# ---------------------------------------------------------------------------

def test_confirm_reuses_registered_mechanical_twin_spelling(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    register_canonical(db, "AgentLane")
    ok, errors = apply_decision(db, "x-alias", "agent-lane")
    assert (ok, errors) == (True, [])
    assert alias_rows(db) == [("x-alias", "AgentLane", "confirmed")]
    # The ghost spelling is never stored or registered.
    assert canonical_names(db) == ["AgentLane"]


def test_rejected_never_registers_workspace_canonical(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    ok, errors = apply_decision(db, "ws", "GhostName", status="rejected")
    assert (ok, errors) == (True, [])
    assert alias_rows(db) == [("ws", "GhostName", "rejected")]
    assert canonical_names(db) == []


# ---------------------------------------------------------------------------
#  Rejected-pair blocking by mechanical key + force override
# ---------------------------------------------------------------------------

def test_rejected_ghost_spelling_variant_blocks_confirm(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    ok, errors = apply_decision(db, "projx", "GhostB", status="rejected")
    assert (ok, errors) == (True, [])
    # 'ghost-b' shares the mechanical key of the rejected 'GhostB'.
    ok, errors = apply_decision(db, "projx", "ghost-b")
    assert ok is False
    assert "kept separate" in errors[0]
    assert "GhostB" in errors[0]
    assert alias_rows(db) == [("projx", "GhostB", "rejected")]


def test_force_clears_every_rejected_spelling_then_confirms(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    apply_decision(db, "projx", "GhostB", status="rejected")
    apply_decision(db, "projx", "ghost_b", status="rejected")
    assert sorted(alias_rows(db)) == [
        ("projx", "GhostB", "rejected"), ("projx", "ghost_b", "rejected"),
    ]
    ok, errors = apply_decision(db, "projx", "ghost-b", force=True)
    assert (ok, errors) == (True, [])
    assert alias_rows(db) == [("projx", "ghost-b", "confirmed")]
    assert canonical_names(db) == ["ghost-b"]


# ---------------------------------------------------------------------------
#  Confirmed-row replacement + upsert
# ---------------------------------------------------------------------------

def test_confirmed_unconditionally_clears_other_confirmed_rows(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    apply_decision(db, "a", "B")
    # No force: a newer confirmation still wins over the stale confirmed row.
    ok, errors = apply_decision(db, "a", "C")
    assert (ok, errors) == (True, [])
    assert alias_rows(db) == [("a", "C", "confirmed")]


def test_same_pair_upserts_on_conflict(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    apply_decision(db, "a", "b")
    apply_decision(db, "a", "b")
    assert alias_rows(db) == [("a", "b", "confirmed")]
    # Re-recording the identical pair with the opposite status updates the
    # existing row in place instead of adding a second row.
    ok, errors = apply_decision(db, "a", "b", status="rejected")
    assert (ok, errors) == (True, [])
    assert alias_rows(db) == [("a", "b", "rejected")]


# ---------------------------------------------------------------------------
#  Read side: rejected_canonicals expansion with registered ghosts
# ---------------------------------------------------------------------------

def test_resolve_expands_rejected_with_registered_mechanical_twins(
    tmp_path: Path,
) -> None:
    db = make_db(tmp_path)
    # Reject BEFORE any twin canonical is registered: with no registered twin
    # the ghost spelling stays verbatim (rejected targets never register).
    ok, errors = apply_decision(db, "myws", "projectb", status="rejected")
    assert (ok, errors) == (True, [])
    assert alias_rows(db) == [("myws", "projectb", "rejected")]
    register_canonical(db, "ProjectB")
    resolved = db.resolve_workspace_canonical("myws", None, register_new=False)
    # Verbatim rejected row plus the registered spelling of the same
    # mechanical key.
    assert set(resolved["rejected_canonicals"]) == {"projectb", "ProjectB"}
    assert resolved["matched_by"] != "confirmed_alias"


def test_reject_after_twin_registration_stores_registered_spelling(
    tmp_path: Path,
) -> None:
    db = make_db(tmp_path)
    register_canonical(db, "ProjectB")
    # Twin substitution is unconditional (confirmed AND rejected): with the
    # twin already registered the rejected row is stored under the registered
    # spelling, and resolution suppresses that spelling directly.
    ok, errors = apply_decision(db, "myws", "projectb", status="rejected")
    assert (ok, errors) == (True, [])
    assert alias_rows(db) == [("myws", "ProjectB", "rejected")]
    resolved = db.resolve_workspace_canonical("myws", None, register_new=False)
    assert set(resolved["rejected_canonicals"]) == {"ProjectB"}


# ---------------------------------------------------------------------------
#  Rejected matching across alias spellings (dual-side mechanical keys)
# ---------------------------------------------------------------------------

def test_rejected_match_spans_alias_spelling_variants(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    # The rejection is recorded under one alias spelling; confirming the same
    # pair under a mechanical-twin ALIAS spelling must not slip past the
    # refusal just because the two alias keys differ verbatim.
    ok, errors = apply_decision(db, "agent_lane", "TargetX", status="rejected")
    assert (ok, errors) == (True, [])
    ok, errors = apply_decision(db, "agent-lane", "TargetX")
    assert ok is False
    assert "kept separate" in errors[0]
    assert "TargetX" in errors[0]
    assert alias_rows(db) == [("agent_lane", "TargetX", "rejected")]


def test_force_clears_cross_spelling_rejected_rows(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    apply_decision(db, "agent_lane", "TargetX", status="rejected")
    apply_decision(db, "AgentLane", "TargetX", status="rejected")
    # alias_workspace stores the normalized key, so the two rejections live
    # under 'agent_lane' and 'agentlane' — both verbatim-different from the
    # caller's 'agent-lane' key below.
    assert sorted(alias_rows(db)) == [
        ("agent_lane", "TargetX", "rejected"), ("agentlane", "TargetX", "rejected"),
    ]
    # force clears the matched rows under THEIR OWN stored alias spellings...
    ok, errors = apply_decision(db, "agent-lane", "TargetX", force=True)
    assert (ok, errors) == (True, [])
    # ...then confirms under the caller's own alias key.
    assert alias_rows(db) == [("agent-lane", "TargetX", "confirmed")]
    assert canonical_names(db) == ["TargetX"]
