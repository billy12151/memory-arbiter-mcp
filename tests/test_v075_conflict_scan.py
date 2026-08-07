"""v0.7.5 legacy conflict infrastructure tests.

The old vector conflict candidate scan is deprecated. This file keeps tests for
conflict persistence/resolve and vector storage helpers that still support active
features, while scan-specific behaviour is covered only by a migration-hint test.
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
#  A. legacy vector conflict candidate scan removed from tool surface
# ──────────────────────────────────────────────────────────────────────────

# The old scan tool is intentionally no longer registered/callable.


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


