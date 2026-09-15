"""0.16.6 content-sha dedup migration tests (owner spec 2026-09-14: partial
unique over ACTIVE rows only — "只管活的"), plus the conflicts.overflow
retirement and the semantic-notice dedupe-key backfill that rides the same
boot pass."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.db import additive
from memory_arbiter.semantic_conflict import notice_dedupe_key

from test_scan_pipeline import make_tools


def _index_exists(conn: sqlite3.Connection, name: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (name,)
    ).fetchone())


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return additive.has_column(conn, table, column)


def _raw_memories_row(conn: sqlite3.Connection, **cols: object) -> int:
    """Insert a memories row with arbitrary columns via SQL (legacy shapes)."""
    base = {
        "content": "x", "agent_id": "a", "workspace": "ws", "workspace_canonical": "ws",
        "tags": "[]", "source_type": "agent_generated", "event_time": "2026-01-01T00:00:00+00:00",
        "ingest_time": "2026-01-01T00:00:00+00:00", "confidence": 0.5,
        "protection_level": "normal", "status": "active", "subject": "s",
        "metadata": "{}", "version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    base.update(cols)
    keys = ",".join(base)
    marks = ",".join("?" for _ in base)
    cur = conn.execute(f"INSERT INTO memories({keys}) VALUES({marks})", tuple(base.values()))
    return int(cur.lastrowid)


def test_fresh_db_has_column_and_partial_index(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    with tools.db.connection() as conn:
        assert _column_exists(conn, "memories", "content_sha")
        assert _index_exists(conn, "idx_memories_content_sha")
        # partial index: same sha twice is fine once one row is not active
        sha = "a" * 64
        _raw_memories_row(conn, content="same", content_sha=sha)
        _raw_memories_row(conn, content="same", content_sha=sha, status="superseded")
        # a second ACTIVE row with the same (workspace, sha) must violate
        with pytest.raises(sqlite3.IntegrityError):
            _raw_memories_row(conn, content="same", content_sha=sha)
        # a different workspace never conflicts
        _raw_memories_row(conn, content="same", content_sha=sha, workspace="ws2",
                          workspace_canonical="ws2")
        # NULL shas (legacy rows pending backfill) never conflict either
        _raw_memories_row(conn, content="same")
        _raw_memories_row(conn, content="same")


def test_migration_backfills_sha_and_normalises_canonical(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    with tools.db.connection() as conn:
        # simulate legacy rows written before the gate: sha NULL, one row
        # with a NULL canonical that would otherwise sit outside the index
        row1 = _raw_memories_row(conn, content="hello 世界", workspace_canonical=None)
        row2 = _raw_memories_row(conn, content="second", status="superseded")
        conn.execute("DELETE FROM migration_state WHERE key='content_sha_dedupe_v1'")
        conn.execute("UPDATE memories SET content_sha=NULL")
        applied = additive.ensure_additive_structures(conn)
    assert any("content_sha_dedupe" in item for item in applied), applied
    with tools.db.connection() as conn:
        import hashlib
        want = hashlib.sha256("hello 世界".encode("utf-8")).hexdigest()
        got1 = conn.execute("SELECT content_sha, workspace_canonical FROM memories WHERE id=?",
                            (row1,)).fetchone()
        assert got1["content_sha"] == want
        assert got1["workspace_canonical"] == "ws", "NULL canonical must be normalised"
        # non-active rows are backfilled too (observability), they just hold no slot
        assert conn.execute("SELECT content_sha FROM memories WHERE id=?",
                            (row2,)).fetchone()["content_sha"] is not None
        # idempotent re-run is a no-op entry
        applied2 = additive.ensure_additive_structures(conn)
        assert not any("content_sha_dedupe" in item for item in applied2)


def test_migration_aborts_on_active_duplicate_pairs(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    with tools.db.connection() as conn:
        left = _raw_memories_row(conn, content="dup", subject="L")
        right = _raw_memories_row(conn, content="dup", subject="R")
        conn.execute("DELETE FROM migration_state WHERE key='content_sha_dedupe_v1'")
        conn.execute("UPDATE memories SET content_sha=NULL")
        with pytest.raises(RuntimeError, match=r"active duplicate pairs"):
            additive.ensure_additive_structures(conn)
        # governance retires one of the pair → migration now completes
        conn.execute("UPDATE memories SET status='superseded' WHERE id=?", (right,))
        applied = additive.ensure_additive_structures(conn)
    assert any("content_sha_dedupe" in item for item in applied)


def test_conflicts_overflow_column_retired(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    with tools.db.connection() as conn:
        assert not _column_exists(conn, "conflicts", "overflow")
        applied2 = additive.ensure_additive_structures(conn)
        assert not any("conflicts_overflow_dropped" in item for item in applied2)


def _unique_hash(left: int, right: int, tag: str) -> str:
    import hashlib
    return hashlib.sha256(f"{left}:{right}:{tag}".encode()).hexdigest()


def _notice_row(conn: sqlite3.Connection, *, status: str, key: str | None,
                left: int, right: int, notice_type: str = "semantic_evidence") -> int:
    import json as _json
    members = _json.dumps([
        {"memory_id": left, "version": 2}, {"memory_id": right, "version": 3},
    ])
    cur = conn.execute(
        """INSERT INTO conflicts(workspace_canonical,candidate_key,candidate_key_hash,
             member_versions,member_fingerprint,value_groups,detection_reason,source,
             detector_version,status,notice_type,notice_delivery_status,notice_dedupe_key,
             created_at,refreshed_at)
           VALUES('ws','{}','%(ck)s','%(mv)s','%(fp)s','[]','semantic','scan','v1',
                  'not_a_conflict',?,?,%(dk)s,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')"""
        % {"ck": _unique_hash(left, right, status), "fp": "f" * 64, "mv": members,
           "dk": "NULL" if key is None else "?"},
        ((notice_type, status) + ((key,) if key is not None else ())),
    )
    return int(cur.lastrowid)


def test_notice_dedupe_backfill_and_pair_closed(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    with tools.db.connection() as conn:
        # three legacy rows without keys: dismissed, resolved, pending
        d = _notice_row(conn, status="dismissed", key=None, left=11, right=12)
        r = _notice_row(conn, status="resolved", key=None, left=21, right=22)
        _notice_row(conn, status="pending", key=None, left=31, right=32)
        conn.execute("DELETE FROM migration_state WHERE key='semantic_notice_dedupe_backfill_v1'")
        applied = additive.ensure_additive_structures(conn)
    assert any("notice_dedupe_backfill" in item for item in applied), applied
    store = tools.db.semantic_notices if hasattr(tools.db, "semantic_notices") else tools.db
    closed = store.is_semantic_pair_closed
    # decided rows are closed at their pinned versions, via the index now
    assert closed(11, 12, 2, 3) is True
    assert closed(21, 22, 2, 3) is True
    # pending rows must NOT suppress re-detection
    assert closed(31, 32, 2, 3) is False
    # version drift re-opens the pair (new key)
    assert closed(11, 12, 2, 4) is False
    # a different notice type has its own key space
    assert closed(11, 12, 2, 3, notice_type="semantic_other") is False
    # self-pair guard
    assert closed(11, 11, 2, 3) is False
    # version-less fallback (legacy test surface): same-notice proof by ids
    assert closed(11, 12) is True
    assert closed(11, 32) is False
    # fresh keyed rows are found without any backfill involvement
    with tools.db.connection() as conn:
        key = notice_dedupe_key(41, 42, 1, 1, "semantic_evidence")
        _notice_row(conn, status="dismissed", key=key, left=41, right=42)
        conn.commit()  # connection() does not auto-commit
    assert closed(41, 42, 1, 1) is True
    assert closed(41, 42, 1, 2) is False


def test_pair_closed_query_uses_index(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    with tools.db.connection() as conn:
        key = notice_dedupe_key(51, 52, 1, 1, "semantic_evidence")
        _notice_row(conn, status="dismissed", key=key, left=51, right=52)
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM conflicts WHERE notice_dedupe_key=? "
            "AND notice_delivery_status IN ('dismissed','resolved') LIMIT 1",
            (key,),
        ).fetchall()
    detail = " ".join(str(row[-1]) for row in plan).lower()
    assert "scan" not in detail.replace("covering index", "") or "index" in detail
