"""v0.7.5 tests: conflict scan (id=243 path-B).

Covers:
  A. scan_conflict_candidates — vector recall, incremental, same-ws filter,
     max_distance, truncation, vec-unavailable branch, scan_log write
  B. record_conflict_enriched — idempotency, pair canonicalisation, new columns
  C. resolve_conflict — single-row close, not_open on already-resolved
  D. get_embedding — struct.unpack round-trip
  E. doctor _check_conflicts_open — three-state sentinel (no scan / stale / fresh)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB, _coerce_tags_db
from memory_arbiter.tools import MemoryTools

try:
    import sqlite_vec  # type: ignore  # noqa: F401
    _VEC_AVAILABLE = True
except Exception:
    _VEC_AVAILABLE = False


def _tools(tmp_path: Path, *, vec: bool = False, dim: int = 4) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "b.jsonl",
        client="test",
        agent_id="tester",
        workspace="ws",
        enable_sqlite_vec=vec,
        vec_dim=dim,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write(
    tools: MemoryTools,
    *,
    content: str,
    subject: str,
    tags: list[str] | None = None,
    workspace: str = "ws",
    memory_id_offset: int = 0,
) -> int:
    res = tools.memory_write(
        content=content, subject=subject, tags=tags or [],
        workspace=workspace, source_type="agent_generated",
    )
    assert res["ok"], f"write failed: {res}"
    return res["data"]["id"]


# ──────────────────────────────────────────────────────────────────────────
#  A. scan_conflict_candidates
# ──────────────────────────────────────────────────────────────────────────

def test_scan_vec_unavailable_returns_scanned_false(tmp_path: Path) -> None:
    """Vector not enabled -> scanned=False with hint, NOT an error (ok=True)."""
    tools = _tools(tmp_path, vec=False)
    _write(tools, content="alpha", subject="s1")
    result = tools.memory_scan_conflict_candidates()
    assert result["ok"] is True
    data = result["data"]
    assert data["scanned"] is False
    assert data["reason"] == "sqlite_vec_unavailable"
    assert "hint" in data


def test_scan_no_candidates_returns_empty(tmp_path: Path) -> None:
    """Scan with memories but no close neighbours -> empty candidates, scan_log written."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    a = _write(tools, content="alpha", subject="s1", workspace="ws")
    # store orthogonal embeddings so they are far apart
    tools.memory_store_embedding(memory_id=a, embedding=[1.0, 0.0, 0.0, 0.0])
    result = tools.memory_scan_conflict_candidates(max_distance=0.5)
    assert result["data"]["scanned"] is True
    assert result["data"]["candidates"] == []
    assert result["data"]["scan_log_written"] is True


def test_scan_finds_close_pair(tmp_path: Path) -> None:
    """Two memories with near-identical embeddings in the same ws -> recalled."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    a = _write(tools, content="revenue grew 20 percent", subject="revenue-q1", workspace="ws")
    b = _write(tools, content="revenue increased twenty percent", subject="revenue-q1-v2", workspace="ws")
    tools.memory_store_embedding(memory_id=a, embedding=[0.9, 0.1, 0.0, 0.0])
    tools.memory_store_embedding(memory_id=b, embedding=[0.91, 0.09, 0.0, 0.0])
    result = tools.memory_scan_conflict_candidates(max_distance=5.0, top_k=5)
    data = result["data"]
    assert data["scanned"] is True
    assert len(data["candidates"]) >= 1
    # pair must be canonicalised left < right
    c = data["candidates"][0]
    assert c["left_id"] < c["right_id"]
    assert {c["left_id"], c["right_id"]} == {a, b}
    assert c["workspace"] == "ws"


def test_scan_same_workspace_filter(tmp_path: Path) -> None:
    """Close embeddings but different workspaces -> pair filtered out."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    a = _write(tools, content="config port 8080", subject="cfg", workspace="ws-a")
    b = _write(tools, content="config port 8080", subject="cfg-dup", workspace="ws-b")
    tools.memory_store_embedding(memory_id=a, embedding=[1.0, 0.0, 0.0, 0.0])
    tools.memory_store_embedding(memory_id=b, embedding=[1.0, 0.0, 0.0, 0.0])
    result = tools.memory_scan_conflict_candidates(max_distance=5.0)
    # Different workspace — no pair should survive
    for c in result["data"]["candidates"]:
        assert c["left_id"] == c["right_id"] or c["workspace"] in ("ws-a", "ws-b")
    # The pair (a,b) must not appear since they differ in ws
    pair_ids = {frozenset({c["left_id"], c["right_id"]}) for c in result["data"]["candidates"]}
    assert frozenset({a, b}) not in pair_ids


def test_scan_truncation(tmp_path: Path) -> None:
    """More pairs than max_pairs -> truncated=True, list capped."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    # Create many near-identical memories
    ids = []
    for i in range(10):
        mid = _write(tools, content=f"dup content {i}", subject=f"s{i}", workspace="ws")
        tools.memory_store_embedding(memory_id=mid, embedding=[1.0, 0.0, 0.0, 0.0])
        ids.append(mid)
    result = tools.memory_scan_conflict_candidates(max_pairs=3, max_distance=5.0, top_k=10)
    data = result["data"]
    assert data["truncated"] is True
    assert len(data["candidates"]) == 3


def test_scan_incremental_watermark(tmp_path: Path) -> None:
    """Second scan with incremental=True only sees new memories since watermark."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    a = _write(tools, content="first", subject="s1", workspace="ws")
    tools.memory_store_embedding(memory_id=a, embedding=[1.0, 0.0, 0.0, 0.0])
    r1 = tools.memory_scan_conflict_candidates(max_distance=5.0)
    assert r1["data"]["scanned"] is True
    assert r1["data"]["max_memory_id"] == a
    # Second scan: no new memories -> candidates empty (watermark blocks re-scan)
    r2 = tools.memory_scan_conflict_candidates(max_distance=5.0)
    assert r2["data"]["candidates"] == []


def test_scan_log_written_and_read(tmp_path: Path) -> None:
    """scan_log.jsonl is written and _scan_log_last_completed reads it back."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    _write(tools, content="x", subject="s1", workspace="ws")
    # No embedding -> scan runs but finds nothing; still writes scan_log
    tools.memory_scan_conflict_candidates()
    last = tools.db._scan_log_last_completed()
    assert last is not None
    assert last["status"] == "completed"
    assert "max_memory_id" in last


# ──────────────────────────────────────────────────────────────────────────
#  B. record_conflict_enriched
# ──────────────────────────────────────────────────────────────────────────

def test_record_conflict_inserts_with_enrichment(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    a = _write(tools, content="a", subject="sa")
    b = _write(tools, content="b", subject="sb")
    result = tools.memory_record_conflict(
        left_id=b, right_id=a,  # deliberately reversed
        reason="factual contradiction",
        conflict_type="contradiction",
        conflict_point="revenue figure differs",
        suggested_winner=a,
        confidence_hint="high",
        source="llm_informed",
    )
    assert result["data"]["outcome"] == "inserted"
    conflicts = tools.memory_list_conflicts()["data"]["conflicts"]
    assert len(conflicts) == 1
    c = conflicts[0]
    # pair canonicalised
    assert c["left_id"] == a and c["right_id"] == b
    assert c["conflict_type"] == "contradiction"
    assert c["conflict_point"] == "revenue figure differs"
    assert c["suggested_winner"] == a
    assert c["source"] == "llm_informed"


def test_record_conflict_idempotent(tmp_path: Path) -> None:
    """Same pair recorded twice -> second call returns deduped=True, no new row."""
    tools = _tools(tmp_path)
    a = _write(tools, content="a", subject="sa")
    b = _write(tools, content="b", subject="sb")
    r1 = tools.memory_record_conflict(left_id=a, right_id=b, reason="r1")
    r2 = tools.memory_record_conflict(left_id=b, right_id=a, reason="r2")  # reversed
    assert r1["data"]["outcome"] == "inserted"
    assert r2["data"]["outcome"] == "deduped"
    assert tools.memory_list_conflicts()["data"]["count"] == 1


# ──────────────────────────────────────────────────────────────────────────
#  C. resolve_conflict
# ──────────────────────────────────────────────────────────────────────────

def test_resolve_conflict_closes_single(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    a = _write(tools, content="a", subject="sa")
    b = _write(tools, content="b", subject="sb")
    c = _write(tools, content="c", subject="sc")
    # two open conflicts
    r1 = tools.memory_record_conflict(left_id=a, right_id=b, reason="r1")
    r2 = tools.memory_record_conflict(left_id=a, right_id=c, reason="r2")
    cid1 = r1["data"]["conflict_id"]
    # resolve only the first
    res = tools.memory_resolve_conflict(conflict_id=cid1, reason="false positive")
    assert res["data"]["outcome"] == "resolved"
    # the second conflict is still open
    open_conflicts = tools.memory_list_conflicts(status="open")["data"]["conflicts"]
    assert len(open_conflicts) == 1
    assert open_conflicts[0]["id"] == r2["data"]["conflict_id"]


def test_resolve_conflict_not_open_idempotent(tmp_path: Path) -> None:
    """Resolving an already-resolved conflict returns not_open, no error."""
    tools = _tools(tmp_path)
    a = _write(tools, content="a", subject="sa")
    b = _write(tools, content="b", subject="sb")
    r = tools.memory_record_conflict(left_id=a, right_id=b, reason="r")
    cid = r["data"]["conflict_id"]
    tools.memory_resolve_conflict(conflict_id=cid)
    r2 = tools.memory_resolve_conflict(conflict_id=cid)
    assert r2["data"]["outcome"] == "not_open"


# ──────────────────────────────────────────────────────────────────────────
#  D. get_embedding
# ──────────────────────────────────────────────────────────────────────────

def test_get_embedding_roundtrip(tmp_path: Path) -> None:
    """store_embedding(json list) -> get_embedding returns same floats via struct.unpack."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    mid = _write(tools, content="x", subject="s")
    emb = [0.1, 0.2, 0.3, 0.4]
    tools.memory_store_embedding(memory_id=mid, embedding=emb)
    got = tools.db.get_embedding(mid)
    assert got is not None
    assert len(got) == 4
    for a, b in zip(got, emb):
        assert abs(a - b) < 1e-5


def test_get_embedding_none_when_no_vec(tmp_path: Path) -> None:
    tools = _tools(tmp_path, vec=False)
    mid = _write(tools, content="x", subject="s")
    assert tools.db.get_embedding(mid) is None


# ──────────────────────────────────────────────────────────────────────────
#  E. doctor three-state sentinel
# ──────────────────────────────────────────────────────────────────────────

def test_doctor_conflicts_warns_when_never_scanned(tmp_path: Path) -> None:
    """Vec available but no scan_log -> WARN 'never scanned'."""
    from memory_arbiter.doctor import _check_conflicts_open, doctor_overview_mcp
    tools = _tools(tmp_path, vec=True, dim=4)
    _write(tools, content="x", subject="s")
    report = doctor_overview_mcp(tools.db, tools.db.settings, runtime_state=tools.db.state)
    finding = next(f for f in report.findings if f.check_id == "capacity.conflicts_open")
    assert finding.severity.value == "warning"
    assert "从未" in finding.title or "never" in finding.title.lower()


def test_doctor_conflicts_pass_when_fresh_scan(tmp_path: Path) -> None:
    """Recent scan_log + no open conflicts -> INFO pass."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    from memory_arbiter.doctor import doctor_overview_mcp
    tools = _tools(tmp_path, vec=True, dim=4)
    _write(tools, content="x", subject="s")
    # run a scan to create scan_log
    tools.memory_scan_conflict_candidates()
    report = doctor_overview_mcp(tools.db, tools.db.settings, runtime_state=tools.db.state)
    finding = next(f for f in report.findings if f.check_id == "capacity.conflicts_open")
    assert finding.severity.value == "info"


def test_doctor_conflicts_vec_off_falls_back_to_count(tmp_path: Path) -> None:
    """Vec off -> falls back to table count (legacy behaviour), not 'never scanned'."""
    from memory_arbiter.doctor import doctor_overview_mcp
    tools = _tools(tmp_path, vec=False)
    _write(tools, content="x", subject="s")
    report = doctor_overview_mcp(tools.db, tools.db.settings, runtime_state=tools.db.state)
    finding = next(f for f in report.findings if f.check_id == "capacity.conflicts_open")
    # No conflicts -> INFO pass, NOT warning about "never scanned"
    assert finding.severity.value == "info"
    assert "从未" not in finding.title


# ──────────────────────────────────────────────────────────────────────────
#  F. _coerce_tags_db unit
# ──────────────────────────────────────────────────────────────────────────

def test_coerce_tags_db_variants() -> None:
    assert _coerce_tags_db(None) == []
    assert _coerce_tags_db("[]") == []
    assert _coerce_tags_db('["a","b"]') == ["a", "b"]
    assert _coerce_tags_db(["x", "y", "x"]) == ["x", "y"]  # dedup
    assert _coerce_tags_db("not-json") == []
    assert _coerce_tags_db(123) == []


# ──────────────────────────────────────────────────────────────────────────
#  G. scan edge cases (review-2: coverage gaps)
# ──────────────────────────────────────────────────────────────────────────

def test_scan_catches_edited_memory_incrementally(tmp_path: Path) -> None:
    """An old memory edited after the first scan is caught on the second scan."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    # write two memories, scan once (establishes watermark + scan_time)
    a = _write(tools, content="alpha", subject="s1", workspace="ws")
    b = _write(tools, content="beta", subject="s2", workspace="ws")
    tools.memory_store_embedding(memory_id=a, embedding=[1.0, 0.0, 0.0, 0.0])
    tools.memory_store_embedding(memory_id=b, embedding=[0.0, 1.0, 0.0, 0.0])
    tools.memory_scan_conflict_candidates(max_distance=5.0)
    # now edit memory A (creates a memory_history row with changed_at > scan_time)
    tools.memory_edit(memory_id=a, new_content="alpha revised", authorized=True)
    # second incremental scan should catch the edited memory
    r2 = tools.memory_scan_conflict_candidates(max_distance=5.0)
    # a was edited and b is its neighbour — pair (a,b) should appear
    pair_ids = {frozenset({c["left_id"], c["right_id"]}) for c in r2["data"]["candidates"]}
    assert frozenset({a, b}) in pair_ids, (
        f"edited memory {a} was not caught on second scan; "
        f"pair_ids={pair_ids}, checked_pairs={r2['data'].get('checked_pairs')}"
    )


def test_scan_excludes_superseded_edited_memory(tmp_path: Path) -> None:
    """A memory edited then superseded must NOT appear in scan candidates."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    a = _write(tools, content="alpha", subject="s1", workspace="ws")
    tools.memory_store_embedding(memory_id=a, embedding=[1.0, 0.0, 0.0, 0.0])
    tools.memory_scan_conflict_candidates(max_distance=5.0)
    # supersede memory a
    tools.memory_supersede(memory_id=a, reason="obsolete", authorized=True)
    r2 = tools.memory_scan_conflict_candidates(max_distance=5.0)
    # a is superseded — must not appear in any candidate pair
    for c in r2["data"]["candidates"]:
        assert a not in (c["left_id"], c["right_id"])


def test_scan_max_distance_filters_far_pairs(tmp_path: Path) -> None:
    """Pairs beyond max_distance are filtered out."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    a = _write(tools, content="near", subject="s1", workspace="ws")
    b = _write(tools, content="far", subject="s2", workspace="ws")
    # orthogonal embeddings -> large distance
    tools.memory_store_embedding(memory_id=a, embedding=[1.0, 0.0, 0.0, 0.0])
    tools.memory_store_embedding(memory_id=b, embedding=[0.0, 0.0, 0.0, 1.0])
    # tight max_distance -> pair filtered
    r = tools.memory_scan_conflict_candidates(max_distance=0.5)
    assert all(c["distance"] <= 0.5 for c in r["data"]["candidates"])
    pair_ids = {frozenset({c["left_id"], c["right_id"]}) for c in r["data"]["candidates"]}
    assert frozenset({a, b}) not in pair_ids


def test_scan_backfills_neighbour_meta_not_in_scan_set(tmp_path: Path) -> None:
    """When a scan memory recalls an old neighbour (id <= watermark), its meta
    (subject/excerpt/tags) is bulk-fetched correctly, not left blank."""
    if not _VEC_AVAILABLE:
        pytest.skip("sqlite-vec not installed")
    tools = _tools(tmp_path, vec=True, dim=4)
    # old memory (will be below watermark after first scan)
    old = _write(tools, content="legacy config port 8080", subject="legacy-cfg",
                 tags=["config"], workspace="ws")
    tools.memory_store_embedding(memory_id=old, embedding=[1.0, 0.0, 0.0, 0.0])
    # first scan establishes watermark = old
    tools.memory_scan_conflict_candidates(max_distance=5.0)
    # new memory near the old one
    new = _write(tools, content="new config port 8080", subject="new-cfg",
                 tags=["config"], workspace="ws")
    tools.memory_store_embedding(memory_id=new, embedding=[0.99, 0.01, 0.0, 0.0])
    # second incremental scan: new is in scan set, old is a recalled neighbour
    r = tools.memory_scan_conflict_candidates(max_distance=5.0)
    candidates = r["data"]["candidates"]
    assert len(candidates) >= 1
    c = candidates[0]
    pair = {c["left_id"], c["right_id"]}
    assert pair == {old, new}
    # the old neighbour's meta must be populated (not blank/default)
    old_side = "left" if c["left_id"] == old else "right"
    old_subject = c[f"{old_side}_subject"]
    assert old_subject == "legacy-cfg"
    assert "legacy config" in c[f"{old_side}_excerpt"]
    assert "config" in c[f"{old_side}_tags"]
