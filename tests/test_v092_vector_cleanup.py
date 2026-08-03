"""v0.9.4: supersede MARKS vectors + inactive-vector cleanup (resync + orphan purge).

v0.9.2 cascade-deleted a memory's vectors on supersede/arbitrate and purged all
inactive vectors. v0.9.4 reversed that (design id=483, decision C + §3.7):
  - supersede/arbitrate now MARK the loser's vectors ``parent_status='superseded'``
    (retained for ``memory_search_expired`` vec-hybrid recall) instead of deleting.
  - ``memory_cleanup_inactive_vectors`` RESYNCS ``parent_status`` mismatches
    (drift from direct DB edits) and PURGES only true orphan vec rows (no parent
    memory/section row). Superseded vectors are never purged.
  - Direct INSERTs into vec0 tables must supply ``parent_status`` (NOT NULL).
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


def _vec_parent_status(db: MemoryDB, table: str, vec_id: int):
    with db.connection() as conn:
        row = conn.execute(
            f"SELECT parent_status AS ps FROM {table} WHERE id = ?", (vec_id,)
        ).fetchone()
    return row["ps"] if row else None


# =====================================================================
#  1. supersede / arbitrate MARK vectors (v0.9.4 decision C)
# =====================================================================

def test_supersede_marks_memory_vector(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    winner = _write(tools, "release v2 current", "release-spec")
    loser = _write(tools, "release v1 stale", "release-spec")
    _embed(tools, winner)
    _embed(tools, loser)
    assert _vec_counts(tools.db)["mem"] == 2

    res = tools.memory_supersede(memory_id=loser, reason="replaced", superseded_by=winner, authorized=True)
    assert res["data"]["superseded"] is True
    # v0.9.4: loser's vector is MARKED 'superseded', NOT deleted. Both kept.
    assert _vec_counts(tools.db)["mem"] == 2
    assert _vec_parent_status(tools.db, "memories_vec", winner) == "active"
    assert _vec_parent_status(tools.db, "memories_vec", loser) == "superseded"
    # Active KNN sees only the winner; expired KNN sees only the loser.
    active_ids = {h["id"] for h in tools.db.vec_knn([0.1, 0.2], k=5)}
    assert active_ids == {winner}
    expired_ids = {h["id"] for h in tools.db.vec_knn([0.1, 0.2], k=5, parent_status_filter="expired")}
    assert expired_ids == {loser}


def test_supersede_marks_section_vectors(tmp_path: Path) -> None:
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
        conn.execute(
            "INSERT INTO memory_sections_vec(id, parent_status, embedding) VALUES (?, 'active', '[0.3,0.4]')",
            (sec_id,),
        )
    assert _vec_counts(tools.db)["sec"] == 1

    tools.memory_supersede(memory_id=loser, reason="replaced", authorized=True)
    # v0.9.4: section vectors are MARKED 'superseded' alongside the memory's vec.
    counts = _vec_counts(tools.db)
    assert counts["mem"] == 1, "memory vector must be kept (marked, not deleted)"
    assert counts["sec"] == 1, "section vectors must be kept (marked, not deleted)"
    assert _vec_parent_status(tools.db, "memories_vec", loser) == "superseded"
    assert _vec_parent_status(tools.db, "memory_sections_vec", sec_id) == "superseded"


def test_arbitrate_marks_loser_vector(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    a = _write(tools, "fact A old", "fact", event_time="2026-01-01T00:00:00Z")
    b = _write(tools, "fact B new authoritative", "fact", event_time="2026-03-01T00:00:00Z")
    _embed(tools, a)
    _embed(tools, b)
    res = tools.memory_arbitrate(left_id=a, right_id=b, authorized=True)
    assert res["ok"] is True
    assert res["data"]["applied"] is True, res["data"]["comparison"]["reasons"]
    # v0.9.4: both vectors survive; loser is MARKED 'superseded'.
    assert _vec_counts(tools.db)["mem"] == 2
    loser_id = res["data"]["comparison"]["loser_id"]
    winner_id = res["data"]["comparison"]["winner_id"]
    assert _vec_parent_status(tools.db, "memories_vec", loser_id) == "superseded"
    assert _vec_parent_status(tools.db, "memories_vec", winner_id) == "active"


def test_update_memory_status_syncs_vector_parent_status(tmp_path: Path) -> None:
    """Generic status updates must keep v0.9.4 vec metadata in sync."""
    tools = make_vec_tools(tmp_path)
    mid = _write(tools, "confirmable memory", "confirmable")
    _embed(tools, mid)
    assert _vec_parent_status(tools.db, "memories_vec", mid) == "active"

    assert tools.db.update_memory(mid, {"status": "superseded"}) is True
    assert _vec_parent_status(tools.db, "memories_vec", mid) == "superseded"

    assert tools.db.update_memory(mid, {"status": "active"}) is True
    assert _vec_parent_status(tools.db, "memories_vec", mid) == "active"


def test_update_memory_status_succeeds_when_vec_parent_status_column_missing(tmp_path: Path) -> None:
    """Status writes must not be blocked by a failed/old vec0 metadata schema."""
    tools = make_vec_tools(tmp_path)
    mid = _write(tools, "old schema fallback", "old-schema")
    _embed(tools, mid)
    with tools.db.write_transaction() as conn:
        conn.execute("DROP TABLE memories_vec")
        conn.execute("DROP TABLE memory_sections_vec")
        conn.execute("CREATE VIRTUAL TABLE memories_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[2])")
        conn.execute("CREATE VIRTUAL TABLE memory_sections_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[2])")
        conn.execute("INSERT INTO memories_vec(id, embedding) VALUES (?, '[0.1,0.2]')", (mid,))

    assert tools.db.update_memory(mid, {"status": "superseded"}) is True
    assert tools.db.get_memory(mid)["status"] == "superseded"


def test_supersede_content_and_vector_retained_for_audit(tmp_path: Path) -> None:
    """Supersede must NOT delete the memory's content/FTS (audit history) nor its
    vector (v0.9.4 keeps it marked for expired recall)."""
    tools = make_vec_tools(tmp_path)
    loser = _write(tools, "stale audit content marker", "audit-doc")
    _embed(tools, loser)
    tools.memory_supersede(memory_id=loser, reason="replaced", authorized=True)
    rec = tools.db.get_memory(loser)
    assert rec is not None
    assert rec["status"] == "superseded"
    assert "stale audit content marker" in rec["content"]
    # Vector retained and marked superseded (available for expired recall).
    assert _vec_counts(tools.db)["mem"] == 1
    assert _vec_parent_status(tools.db, "memories_vec", loser) == "superseded"


def test_expired_recall_includes_conflicted_and_pending(tmp_path: Path) -> None:
    """v0.9.4 点2: memory_search_expired recalls all non-active non-deleted
    memories (superseded + conflicted + pending), not just superseded.

    Before v0.9.4, conflicted/pending vectors had parent_status set to those
    literal statuses by update_memory, but vec_knn only filtered on
    'active'/'superseded' — so those vectors were dead (invisible to both
    active and expired recall). The 'expired' filter now matches
    parent_status NOT IN ('active','deleted').
    """
    tools = make_vec_tools(tmp_path)
    active = _write(tools, "release-domain active record", "release-domain")
    conflicted = _write(tools, "release-domain conflicted record", "release-domain")
    pending = _write(tools, "release-domain pending record", "release-domain")
    _embed(tools, active)
    _embed(tools, conflicted)
    _embed(tools, pending)
    # Flip two memories to conflicted/pending via update_memory (which syncs
    # vec parent_status to the new status).
    assert tools.db.update_memory(conflicted, {"status": "conflicted"}) is True
    assert tools.db.update_memory(pending, {"status": "pending"}) is True
    assert _vec_parent_status(tools.db, "memories_vec", conflicted) == "conflicted"
    assert _vec_parent_status(tools.db, "memories_vec", pending) == "pending"

    # Active recall (memory_search) sees only the active record.
    found = tools.memory_search(query="release-domain")
    active_ids = [r["id"] for r in found["data"]["results"]]
    assert active in active_ids
    assert conflicted not in active_ids
    assert pending not in active_ids

    # Expired recall (memory_search_expired) sees conflicted + pending.
    expired = tools.memory_search_expired(query="release-domain")
    expired_ids = [r["id"] for r in expired["data"]["results"]]
    assert conflicted in expired_ids, "conflicted memory not recalled by memory_search_expired"
    assert pending in expired_ids, "pending memory not recalled by memory_search_expired"
    assert active not in expired_ids, "active memory leaked into expired recall"

    # Direct vec_knn with the 'expired' filter also returns both.
    expired_vec_ids = {h["id"] for h in tools.db.vec_knn([0.1, 0.2], k=5, parent_status_filter="expired")}
    assert conflicted in expired_vec_ids
    assert pending in expired_vec_ids
    assert active not in expired_vec_ids


def test_search_all_vec_channel_matches_fts(tmp_path: Path) -> None:
    """v0.9.4 P4: status_filter="all" vec channel returns non-deleted vectors,
    matching the FTS channel semantics (previously vec only returned active)."""
    from memory_arbiter.search import search_memories

    tools = make_vec_tools(tmp_path)
    active = _write(tools, "release-spec-v2 active authoritative", "release-spec")
    stale = _write(tools, "release-spec-v1 stale superseded", "release-spec")
    _embed(tools, active)
    _embed(tools, stale)
    assert tools.memory_supersede(memory_id=stale, reason="replaced", superseded_by=active, authorized=True)["ok"] is True

    # vec_knn with "all" returns both active and superseded vectors.
    all_vec_ids = {h["id"] for h in tools.db.vec_knn([0.1, 0.2], k=5, parent_status_filter="all")}
    assert active in all_vec_ids, "active vector missing from 'all' vec recall"
    assert stale in all_vec_ids, "superseded vector missing from 'all' vec recall (P4 regression)"

    # search_memories(status_filter="all") vec channel surfaces superseded too.
    outcome = search_memories(tools.db, "release-spec", limit=10, status_filter="all")
    found_ids = {r["id"] for r in outcome.results}
    assert active in found_ids
    assert stale in found_ids, "superseded record missing from status_filter='all' (vec channel was active-only pre-P4)"


def test_memory_search_expired_offset_pagination(tmp_path: Path) -> None:
    """v0.9.4 P3: memory_search_expired supports offset cursor pagination on
    the exact-count empty-query+filters path."""
    tools = make_vec_tools(tmp_path)
    # Seed 5 superseded memories sharing a tag so the filter path has a
    # precise count and deterministic ingest_time ordering.
    ids = []
    for i in range(5):
        mid = _write(tools, f"expired doc {i}", "audit-tag", event_time=f"2026-01-0{i+1}T00:00:00Z")
        ids.append(mid)
        # Tag it so tags_filter drives the exact-count filter path.
        tools.memory_edit(memory_id=mid, tags_only=True, add_tags=["audit"], authorized=True)
        tools.memory_supersede(memory_id=mid, reason="replaced", authorized=True)

    # Page 1: limit=2, offset=0 → 2 results, has_more=True.
    page1 = tools.memory_search_expired(query="", tags_filter=["audit"], limit=2, offset=0)
    p1_ids = [r["id"] for r in page1["data"]["results"]]
    assert len(p1_ids) == 2
    assert page1["data"]["has_more"] is True
    assert page1["data"]["total_estimate"] == 5
    assert page1["data"]["offset"] == 0
    assert page1["data"]["effective_limit"] == 2
    assert page1["data"]["next_offset"] == 2
    assert page1["data"]["pagination_precision"] == "exact"

    # Page 2: limit=2, offset=2 → next 2, no overlap with page 1.
    page2 = tools.memory_search_expired(query="", tags_filter=["audit"], limit=2, offset=2)
    p2_ids = [r["id"] for r in page2["data"]["results"]]
    assert len(p2_ids) == 2
    assert page2["data"]["has_more"] is True
    assert page2["data"]["next_offset"] == 4
    assert set(p1_ids).isdisjoint(set(p2_ids)), "offset pages must not overlap"

    # Page 3: limit=2, offset=4 → last 1, has_more=False.
    page3 = tools.memory_search_expired(query="", tags_filter=["audit"], limit=2, offset=4)
    p3_ids = [r["id"] for r in page3["data"]["results"]]
    assert len(p3_ids) == 1
    assert page3["data"]["has_more"] is False
    assert page3["data"]["next_offset"] is None

    # Out-of-range manual offset preserves the true total for callers.
    beyond = tools.memory_search_expired(query="", tags_filter=["audit"], limit=2, offset=10)
    assert beyond["data"]["results"] == []
    assert beyond["data"]["total_estimate"] == 5
    assert beyond["data"]["has_more"] is False
    assert any("offset beyond result set" in w for w in beyond["warnings"])

    # Union of all pages covers all 5 superseded memories.
    assert set(p1_ids) | set(p2_ids) | set(p3_ids) == set(ids)


def test_memory_search_expired_browse_offset_without_filters(tmp_path: Path) -> None:
    """Browsing all expired rows with an empty query must honor offset too."""
    tools = make_vec_tools(tmp_path)
    ids = []
    for i in range(5):
        mid = _write(tools, f"expired browse doc {i}", "audit-browse", event_time=f"2026-01-0{i+1}T00:00:00Z")
        ids.append(mid)
        tools.memory_supersede(memory_id=mid, reason="replaced", authorized=True)

    page1 = tools.memory_search_expired(query="", limit=2, offset=0)
    page2 = tools.memory_search_expired(query="", limit=2, offset=2)
    p1_ids = [r["id"] for r in page1["data"]["results"]]
    p2_ids = [r["id"] for r in page2["data"]["results"]]

    assert len(p1_ids) == 2
    assert len(p2_ids) == 2
    assert set(p1_ids).isdisjoint(p2_ids)
    assert page1["data"]["has_more"] is True
    assert page1["data"]["total_estimate"] == 5
    assert page2["data"]["has_more"] is True
    assert page2["data"]["offset"] == 2
    assert page2["data"]["pagination_precision"] == "exact"
    assert set(p1_ids + p2_ids).issubset(set(ids))


def test_vec_search_defensively_filters_status_drift(tmp_path: Path) -> None:
    """Current memories.status must still guard domains if vec.parent_status drifts."""
    tools = make_vec_tools(tmp_path)
    active = _write(tools, "drift-domain active", "drift-domain")
    stale = _write(tools, "drift-domain stale", "drift-domain")
    wrong_expired = _write(tools, "drift-domain wrong-expired", "drift-domain")
    _embed(tools, active)
    _embed(tools, stale)
    _embed(tools, wrong_expired)
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status='superseded' WHERE id=?", (stale,))
        conn.execute("UPDATE memories_vec SET parent_status='active' WHERE id=?", (stale,))
        conn.execute("UPDATE memories_vec SET parent_status='superseded' WHERE id=?", (wrong_expired,))

    active_hits = tools.memory_search(query="semantically close", query_embedding=[0.1, 0.2])
    active_ids = {r["id"] for r in active_hits["data"]["results"]}
    assert active in active_ids
    assert stale not in active_ids

    expired_hits = tools.memory_search_expired(query="semantically close", query_embedding=[0.1, 0.2])
    expired_ids = {r["id"] for r in expired_hits["data"]["results"]}
    assert active not in expired_ids
    assert wrong_expired not in expired_ids


# =====================================================================
#  2. memory_cleanup_inactive_vectors tool (resync + orphan purge)
# =====================================================================

def _seed_inactive(tools: MemoryTools) -> dict:
    """Seed one synced active + one drifted superseded memory + one orphan section vec.

    - active: embedded, parent_status='active' (synced).
    - stale: embedded while active, then status flipped via direct SQL →
      vec.parent_status still 'active' while memory.status='superseded'
      (DRIFT / mismatch from an external DB edit).
    - stale's section vec: inserted with parent_status='active' (mismatch vs the
      superseded memory) — another drift row for the section channel.
    - orphan section vec id=999: no parent section row at all (true orphan).
    """
    active = _write(tools, "active doc", "active-doc")
    stale = _write(tools, "stale doc", "stale-doc")
    _embed(tools, active)
    _embed(tools, stale)
    # Direct status flip bypasses MemoryDB.update_memory's vector sync → creates drift.
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status='superseded' WHERE id=?", (stale,))
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO memory_sections(memory_id, section_index, title, summary, start_offset, end_offset, provenance, created_at) "
            "VALUES (?, 0, 't', 's', 0, 1, 'agent', '2026-02-01T00:00:00Z')",
            (stale,),
        )
        sec_id = conn.execute("SELECT id FROM memory_sections WHERE memory_id=?", (stale,)).fetchone()["id"]
        conn.execute(
            "INSERT INTO memory_sections_vec(id, parent_status, embedding) VALUES (?, 'active', '[0.5,0.6]')",
            (sec_id,),
        )
        # Physical orphan: section vec with no section row at all.
        conn.execute(
            "INSERT INTO memory_sections_vec(id, parent_status, embedding) VALUES (999, 'active', '[0.7,0.8]')"
        )
    return {"active": active, "stale": stale, "stale_sec": sec_id}


def test_cleanup_dry_run_reports_without_changing(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    _seed_inactive(tools)
    res = tools.memory_cleanup_inactive_vectors(dry_run=True)
    assert res["data"]["dry_run"] is True
    assert res["data"]["vec_parent_status_mismatches"] == {
        "memory_vec_mismatch": 1,
        "section_vec_mismatch": 1,
    }
    assert res["data"]["orphan_vectors"] == {
        "orphan_memory_vectors": 0,
        "orphan_section_vectors": 1,
    }
    # Nothing actually changed.
    assert _vec_counts(tools.db) == {"mem": 2, "sec": 2}


def test_cleanup_requires_authorized(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    _seed_inactive(tools)
    res = tools.memory_cleanup_inactive_vectors(dry_run=False, authorized=False)
    assert res["ok"] is False
    # Nothing changed.
    assert _vec_counts(tools.db) == {"mem": 2, "sec": 2}


def test_cleanup_resyncs_mismatches_and_purges_orphans(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    ids = _seed_inactive(tools)
    res = tools.memory_cleanup_inactive_vectors(dry_run=False, authorized=True)
    assert res["ok"] is True
    # Phase 1: resync flipped stale's mem vec + section vec to 'superseded'.
    assert res["data"]["resynced"] == {
        "resynced_memory_vecs": 1,
        "resynced_section_vecs": 1,
    }
    # Phase 2: only the orphan section vec (999) is purged; superseded vecs kept.
    assert res["data"]["purged"] == {
        "purged_memory_orphans": 0,
        "purged_section_orphans": 1,
    }
    # active + stale mem vecs both kept; orphan section purged, stale's section kept.
    assert _vec_counts(tools.db) == {"mem": 2, "sec": 1}
    # stale's vecs are now resynced to 'superseded' (no longer drift); active untouched.
    assert _vec_parent_status(tools.db, "memories_vec", ids["stale"]) == "superseded"
    assert _vec_parent_status(tools.db, "memory_sections_vec", ids["stale_sec"]) == "superseded"
    assert _vec_parent_status(tools.db, "memories_vec", ids["active"]) == "active"


def test_cleanup_restores_clean_state(tmp_path: Path) -> None:
    """After cleanup, no mismatches and no orphans remain; KNN runs clean."""
    tools = make_vec_tools(tmp_path)
    _seed_inactive(tools)
    before = tools.memory_cleanup_inactive_vectors(dry_run=True)["data"]
    assert (before["vec_parent_status_mismatches"]["memory_vec_mismatch"]
            + before["vec_parent_status_mismatches"]["section_vec_mismatch"]
            + before["orphan_vectors"]["orphan_memory_vectors"]
            + before["orphan_vectors"]["orphan_section_vectors"]) > 0
    tools.memory_cleanup_inactive_vectors(dry_run=False, authorized=True)
    after = tools.memory_cleanup_inactive_vectors(dry_run=True)["data"]
    assert after["vec_parent_status_mismatches"] == {
        "memory_vec_mismatch": 0,
        "section_vec_mismatch": 0,
    }
    assert after["orphan_vectors"] == {
        "orphan_memory_vectors": 0,
        "orphan_section_vectors": 0,
    }
    # vec_knn runs without error on the cleaned store.
    hits = tools.db.vec_knn([0.1, 0.2], k=5)
    assert isinstance(hits, list)


# =====================================================================
#  4. vec0 parent_status migration persists on pre-existing databases
# =====================================================================

def test_migration_persists_on_existing_old_schema_db(tmp_path: Path) -> None:
    """Booting v0.9.4 against a pre-existing database whose vec0 tables still
    use the old (no parent_status) schema must migrate them in-place and
    PERSIST it.

    Regression: ``_migrate_vec_parent_status`` originally lacked a commit, so
    the DROP+CREATE+re-insert was rolled back when the init connection closed,
    silently leaving the vec table on the old schema. The fixtures elsewhere
    always create fresh vec tables, so this path — an existing table with real
    rows — was never exercised.
    """
    import json
    import sqlite3

    sqlite_vec = pytest.importorskip("sqlite_vec")

    # 1. Boot once so the full (non-vec) schema exists, then seed real memories.
    tools = make_vec_tools(tmp_path)
    db_path = tools.settings.db_path
    active_id = _write(tools, "active body", "active subj")
    superseded_id = _write(tools, "superseded body", "superseded subj")
    tools.db.update_memory(superseded_id, {"status": "superseded"})

    def _raw() -> sqlite3.Connection:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    # 2. Force the vec tables back to the OLD (no parent_status) schema with
    #    committed rows, simulating a database written by v0.9.x.
    conn = _raw()
    conn.execute("DROP TABLE IF EXISTS memories_vec")
    conn.execute("DROP TABLE IF EXISTS memory_sections_vec")
    conn.execute("CREATE VIRTUAL TABLE memories_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[2])")
    conn.execute("INSERT INTO memories_vec(id, embedding) VALUES(?, ?)", (active_id, json.dumps([0.1, 0.2])))
    conn.execute("INSERT INTO memories_vec(id, embedding) VALUES(?, ?)", (superseded_id, json.dumps([0.3, 0.4])))
    conn.execute("CREATE VIRTUAL TABLE memory_sections_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[2])")
    conn.commit()
    conn.close()

    # 3. Boot a second MemoryDB → triggers _migrate_vec_parent_status.
    settings = Settings(
        db_path=db_path,
        backup_jsonl=tmp_path / "backup-vec.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=2,
        split_threshold=1,
    )
    MemoryDB(settings)

    # 4. Reopen fresh and assert the migration persisted.
    conn = _raw()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(memories_vec)")}
    sec_cols = {r["name"] for r in conn.execute("PRAGMA table_info(memory_sections_vec)")}
    n = conn.execute("SELECT COUNT(*) AS c FROM memories_vec").fetchone()["c"]
    statuses = {
        int(r["id"]): r["parent_status"]
        for r in conn.execute("SELECT id, parent_status FROM memories_vec")
    }
    conn.close()

    assert "parent_status" in cols, "memories_vec migration did not persist"
    assert "parent_status" in sec_cols, "memory_sections_vec migration did not persist"
    assert int(n) == 2, "migration dropped existing embedding rows"
    assert statuses == {active_id: "active", superseded_id: "superseded"}, \
        "parent_status not backfilled from memories.status"
