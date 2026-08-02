"""v0.9.2: supersede cascade-deletes vectors + inactive-vector cleanup tool.

Root cause: superseded/deleted vectors stayed in memories_vec / memory_sections_vec
forever, occupying vec0 KNN top-k slots and forcing every KNN onto the exact-L2
slow path. v0.9.2 (a) cascade-deletes a memory's vectors on supersede/arbitrate,
and (b) adds memory_cleanup_inactive_vectors for existing backlogs.
"""
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def make_vec_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    settings = Settings(
        db_path=tmp_path / "memory-vec.sqlite3",
        backup_jsonl=tmp_path / "backup-vec.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=2,
        split_threshold=1,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write(tools: MemoryTools, content: str, subject: str, event_time: str = "2026-02-01T00:00:00Z") -> int:
    res = tools.memory_write(
        content=content,
        subject=subject,
        source_type="agent_generated",
        event_time=event_time,
    )
    return res["data"]["id"]


def _vec_counts(db: MemoryDB) -> dict:
    with db.connection() as conn:
        mem = conn.execute("SELECT COUNT(*) AS c FROM memories_vec").fetchone()["c"]
        sec = conn.execute("SELECT COUNT(*) AS c FROM memory_sections_vec").fetchone()["c"]
    return {"mem": int(mem), "sec": int(sec)}


def _embed(tools: MemoryTools, memory_id: int) -> None:
    tools.memory_store_embedding(memory_id=memory_id, embedding=[0.1, 0.2])


# =====================================================================
#  1. cascade delete on supersede / arbitrate
# =====================================================================

def test_supersede_deletes_memory_vector(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    winner = _write(tools, "release v2 current", "release-spec")
    loser = _write(tools, "release v1 stale", "release-spec")
    _embed(tools, winner)
    _embed(tools, loser)
    assert _vec_counts(tools.db)["mem"] == 2

    res = tools.memory_supersede(memory_id=loser, reason="replaced", superseded_by=winner, authorized=True)
    assert res["data"]["superseded"] is True
    # Winner keeps its vector; loser's vector is gone.
    assert _vec_counts(tools.db)["mem"] == 1
    with tools.db.connection() as conn:
        ids = {r["id"] for r in conn.execute("SELECT id FROM memories_vec").fetchall()}
    assert ids == {winner}


def test_supersede_deletes_section_vectors(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    loser = _write(tools, "stale long doc", "stale-doc")
    _embed(tools, loser)
    # Manually attach a section + section vector to the loser.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO memory_sections(memory_id, section_index, title, summary, start_offset, end_offset, provenance, created_at) "
            "VALUES (?, 0, 't', 's', 0, 1, 'agent', '2026-02-01T00:00:00Z')",
            (loser,),
        )
        sec_id = conn.execute("SELECT id FROM memory_sections WHERE memory_id=?", (loser,)).fetchone()["id"]
        conn.execute("INSERT INTO memory_sections_vec(id, embedding) VALUES (?, '[0.3,0.4]')", (sec_id,))
    assert _vec_counts(tools.db)["sec"] == 1

    tools.memory_supersede(memory_id=loser, reason="replaced", authorized=True)
    counts = _vec_counts(tools.db)
    assert counts["mem"] == 0
    assert counts["sec"] == 0, "section vectors must be cascade-deleted with the memory"


def test_arbitrate_loser_vector_deleted(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    a = _write(tools, "fact A old", "fact", event_time="2026-01-01T00:00:00Z")
    b = _write(tools, "fact B new authoritative", "fact", event_time="2026-03-01T00:00:00Z")
    _embed(tools, a)
    _embed(tools, b)
    res = tools.memory_arbitrate(left_id=a, right_id=b, authorized=True)
    assert res["ok"] is True
    assert res["data"]["applied"] is True, res["data"]["comparison"]["reasons"]
    # Exactly one vector survives (the winner's).
    assert _vec_counts(tools.db)["mem"] == 1


def test_supersede_content_retained_for_audit(tmp_path: Path) -> None:
    """Deleting vectors must NOT delete the memory's content/FTS (audit history)."""
    tools = make_vec_tools(tmp_path)
    loser = _write(tools, "stale audit content marker", "audit-doc")
    _embed(tools, loser)
    tools.memory_supersede(memory_id=loser, reason="replaced", authorized=True)
    rec = tools.db.get_memory(loser)
    assert rec is not None
    assert rec["status"] == "superseded"
    assert "stale audit content marker" in rec["content"]


# =====================================================================
#  2. memory_cleanup_inactive_vectors tool
# =====================================================================

def _seed_inactive(tools: MemoryTools) -> None:
    """One active + one superseded memory, both embedded; one orphan section vec."""
    active = _write(tools, "active doc", "active-doc")
    stale = _write(tools, "stale doc", "stale-doc")
    _embed(tools, active)
    _embed(tools, stale)
    tools.db.update_memory(stale, {"status": "superseded"})
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO memory_sections(memory_id, section_index, title, summary, start_offset, end_offset, provenance, created_at) "
            "VALUES (?, 0, 't', 's', 0, 1, 'agent', '2026-02-01T00:00:00Z')",
            (stale,),
        )
        sec_id = conn.execute("SELECT id FROM memory_sections WHERE memory_id=?", (stale,)).fetchone()["id"]
        conn.execute("INSERT INTO memory_sections_vec(id, embedding) VALUES (?, '[0.5,0.6]')", (sec_id,))
        # Physical orphan: section vec with no section row at all.
        conn.execute("INSERT INTO memory_sections_vec(id, embedding) VALUES (999, '[0.7,0.8]')")


def test_cleanup_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    _seed_inactive(tools)
    res = tools.memory_cleanup_inactive_vectors(dry_run=True)
    assert res["data"]["dry_run"] is True
    assert res["data"]["would_delete"]["inactive_memory_vectors"] == 1
    assert res["data"]["would_delete"]["inactive_section_vectors"] == 2
    # Nothing actually deleted.
    assert _vec_counts(tools.db) == {"mem": 2, "sec": 2}


def test_cleanup_requires_authorized(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    _seed_inactive(tools)
    res = tools.memory_cleanup_inactive_vectors(dry_run=False, authorized=False)
    assert res["ok"] is False
    assert _vec_counts(tools.db) == {"mem": 2, "sec": 2}


def test_cleanup_deletes_only_inactive(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    _seed_inactive(tools)
    res = tools.memory_cleanup_inactive_vectors(dry_run=False, authorized=True)
    assert res["ok"] is True
    assert res["data"]["deleted"]["deleted_memory_vectors"] == 1
    assert res["data"]["deleted"]["deleted_section_vectors"] == 2
    # Active memory's vector survives; superseded + orphan vectors gone.
    counts = _vec_counts(tools.db)
    assert counts == {"mem": 1, "sec": 0}
    with tools.db.connection() as conn:
        ids = {r["id"] for r in conn.execute("SELECT id FROM memories_vec").fetchall()}
    assert len(ids) == 1


def test_cleanup_restores_fast_path(tmp_path: Path) -> None:
    """After cleanup, the KNN eligibility probe finds all rows eligible → vec0 fast path."""
    tools = make_vec_tools(tmp_path)
    _seed_inactive(tools)
    # Before cleanup: an inactive vector exists → KNN would take the slow path.
    before = tools.memory_cleanup_inactive_vectors(dry_run=True)["data"]["would_delete"]
    assert before["inactive_memory_vectors"] + before["inactive_section_vectors"] > 0
    tools.memory_cleanup_inactive_vectors(dry_run=False, authorized=True)
    after = tools.memory_cleanup_inactive_vectors(dry_run=True)["data"]["would_delete"]
    assert after == {"inactive_memory_vectors": 0, "inactive_section_vectors": 0}
    # vec_knn now returns the active memory via the fast path without error.
    hits = tools.db.vec_knn([0.1, 0.2], k=5)
    assert isinstance(hits, list)
