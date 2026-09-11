# ── is_semantic_pair_closed direct-query regression (review P1-4) ──
"""The pair-closed check must not window through list_semantic_notices.

The list path clamps ``limit`` to 100 (newest first), so a closed pair older
than the newest 100 dismissed/resolved notices wrongly looked open and was
re-detected. The direct query has no such window.

Rows are created via the production record_conflict_group/notice paths
(status='not_a_conflict' rows carry candidate status transitions), then moved
to a terminal delivery status to close them.
"""

from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord


def _db(tmp_path: Path) -> MemoryDB:
    return MemoryDB(Settings(
        db_path=tmp_path / "memory.db", backup_jsonl=tmp_path / "backup.jsonl",
    ))


def _memory(db: MemoryDB, content: str) -> int:
    memory_id, _ = db.insert_memory(
        MemoryRecord(content=content, subject="pair", agent_id="a", workspace="w"),
        "w",
    )
    assert memory_id is not None
    return memory_id


def _member(memory_id: int, value: str) -> ConflictMember:
    quote = f"value is {value}"
    return ConflictMember(
        memory_id=memory_id, version=1, attribute_raw="attr", value_raw=value,
        normalized_attribute="attr", normalized_value=value.casefold(),
        evidence_quote=quote, evidence_span=(0, len(quote)),
        content_hash=(str(memory_id) * 64)[:64], direction="a_to_b",
        prompt_version="p1", detector_version="d1",
    )


def _groups(*members: ConflictMember) -> list[ConflictValueGroup]:
    by_value: dict[str, list[str]] = {}
    for member in members:
        by_value.setdefault(member.normalized_value, []).append(f"{member.memory_id}@1")
    return [ConflictValueGroup(v, v.title(), tuple(refs)) for v, refs in by_value.items()]


def _close_pair(db: MemoryDB, left: int, right: int, *, delivery: str = "dismissed", notice_type: str = "semantic_evidence") -> int:
    """Record a not_a_conflict event for the pair, then flip its delivery
    status — exactly the shape a dismissed/resolved notice leaves behind."""
    members = [_member(left, "alpha"), _member(right, "beta")]
    outcome = db.record_conflict_group(
        workspace_canonical="w", slot_key=None, members=members,
        value_groups=_groups(*members), detection_reason="closed test",
        source="scan", detector_version="d1", status="not_a_conflict",
        conflict_point="attr", prompt_version="p1",
    )
    assert outcome["outcome"] == "inserted", outcome
    conflict_id = int(outcome["conflict_id"])
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE conflicts SET notice_delivery_status=?, notice_type=? WHERE id=?",
            (delivery, notice_type, conflict_id),
        )
    return conflict_id


def test_closed_pair_beyond_the_100_notice_window_still_found(tmp_path: Path) -> None:
    db = _db(tmp_path)
    # Pair of interest: memories 1/2, dismissed as the OLDEST closed row;
    # 150 newer dismissed rows pile on top so the pair is far beyond the
    # clamped 100-row list window of the old implementation.
    left, right = _memory(db, "alpha fact"), _memory(db, "beta fact")
    _close_pair(db, left, right)
    for i in range(150):
        a, b = _memory(db, f"noise {i} a"), _memory(db, f"noise {i} b")
        _close_pair(db, a, b)

    assert db.is_semantic_pair_closed(left, right) is True
    # Version-pinned variant must also find it.
    assert db.is_semantic_pair_closed(left, right, 1, 1) is True
    # A different version of either side is NOT covered by the old dismissal.
    assert db.is_semantic_pair_closed(left, right, 1, 2) is False
    # An unrelated pair stays open.
    x, y = _memory(db, "unrelated a"), _memory(db, "unrelated b")
    assert db.is_semantic_pair_closed(x, y) is False


def test_resolved_delivery_also_closes_pair(tmp_path: Path) -> None:
    db = _db(tmp_path)
    left, right = _memory(db, "alpha fact"), _memory(db, "beta fact")
    _close_pair(db, left, right, delivery="resolved")

    assert db.is_semantic_pair_closed(left, right) is True


def test_notice_type_mismatch_does_not_close(tmp_path: Path) -> None:
    db = _db(tmp_path)
    left, right = _memory(db, "alpha fact"), _memory(db, "beta fact")
    _close_pair(db, left, right, notice_type="write_time_duplicate")

    assert db.is_semantic_pair_closed(left, right) is False
    assert db.is_semantic_pair_closed(left, right, notice_type="write_time_duplicate") is True


def test_pending_delivery_does_not_close_pair(tmp_path: Path) -> None:
    db = _db(tmp_path)
    left, right = _memory(db, "alpha fact"), _memory(db, "beta fact")
    _close_pair(db, left, right, delivery="delivered")

    assert db.is_semantic_pair_closed(left, right) is False


def test_self_pair_never_closed_by_single_member_row(tmp_path: Path) -> None:
    db = _db(tmp_path)
    # A single-member dismissed row pinning only id N must not close the
    # (N, N) self-pair: for equal ids the two EXISTS clauses collapse into
    # one, so a one-member row would wrongly satisfy both. (The production
    # recorder refuses duplicate members outright; a legacy/imported row is
    # the only way this shape exists.)
    solo, peer = _memory(db, "solo fact"), _memory(db, "peer fact")
    _close_pair(db, solo, peer)
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE conflicts SET member_versions=? WHERE id IN "
            "(SELECT MIN(id) FROM conflicts)",
            ('[{"memory_id": %d, "version": 1}]' % solo,),
        )

    assert db.is_semantic_pair_closed(solo, solo) is False
    assert db.is_semantic_pair_closed(peer, peer) is False


def test_malformed_member_json_row_is_skipped_not_fatal(tmp_path: Path) -> None:
    db = _db(tmp_path)
    left, right = _memory(db, "alpha fact"), _memory(db, "beta fact")
    _close_pair(db, left, right)
    # Corrupt one unrelated closed row's member JSON; the check must not
    # raise and must still find the well-formed pair.
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE conflicts SET member_versions=json('{\"bad\": true}') "
            "WHERE id <> (SELECT MIN(id) FROM conflicts)"
        )

    assert db.is_semantic_pair_closed(left, right) is True
