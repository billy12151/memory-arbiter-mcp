"""C3b (0.15.13): workspace-grouped conflict enumeration + suspected-bucket sweep.

Cross-bucket pairs used to be enumerated globally but could never land in
record_conflict (its group identity derives one workspace from members) —
a weekly dead loop. Pairing is now per anchor bucket; suspected misplaced
memories (active workspace_review notices) are additionally swept against
their suspected bucket, with those pairs surfaced as reference-only
cross_bucket_references.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.tools import MemoryTools


@pytest.fixture()
def vec_tools(tmp_path: Path):
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    yield tools


def _bucket_of(tools: MemoryTools, memory_id: int) -> str:
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS w "
            "FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
    return str(row["w"])


def _scan(tools: MemoryTools, **extra: object) -> dict:
    result = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10,
        "include_check": True, "include_quotes": True, **extra,
    })
    assert result["ok"] is True, result
    return result["data"]


def _library_with_misplacement(tools: MemoryTools) -> dict:
    """apisvc: 2 timeout notes + 1 misplaced PostgreSQL memory (id returned);
    dbpgsql: 9 PostgreSQL brothers + 1 planted real conflict partner."""
    for i in range(2):
        tools.memory_write(
            content=f"接口超时配置记录 {i}。", subject=f"t-{i}", tags=["timeout"],
            workspace="apisvc",
        )
    for i in range(9):
        tools.memory_write(
            content=f"生产环境数据库使用 PostgreSQL，兄弟条目 {i}。", subject=f"db-{i}",
            tags=["db"], workspace="dbpgsql",
        )
    misplaced = tools.memory_write(
        content="生产环境数据库使用 PostgreSQL，属于 dbpgsql 的错位记忆，端口是 5432。",
        subject="misplaced", tags=["db"], workspace="apisvc",
    )["data"]
    partner = tools.memory_write(
        content="生产环境数据库使用 PostgreSQL，端口是 5433。", subject="port-b",
        tags=["db"], workspace="dbpgsql",
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=10)
    return {"misplaced": misplaced, "partner": partner}


def test_no_cross_bucket_pairs_in_candidates(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _library_with_misplacement(tools)
    # No anomaly notice yet: global scan pairs only within buckets.
    data = _scan(tools)
    for cand in data["candidates"]:
        left_bucket = _bucket_of(tools, int(cand["left_id"]))
        right_bucket = _bucket_of(tools, int(cand["right_id"]))
        assert left_bucket == right_bucket, (
            f"cross-bucket pair leaked into candidates: "
            f"{cand['left_id']}({left_bucket}) x {cand['right_id']}({right_bucket})"
        )
    assert data.get("cross_bucket_references") == []


def test_suspect_sweep_surfaces_planted_conflict_same_week(vec_tools: MemoryTools) -> None:
    """C6 step 4 shape: after the anomaly check flags the misplaced memory,
    the SAME round's conflict scan must surface its conflict with the
    dbpgsql partner as a cross-bucket reference (recordable only post-move)."""
    tools = vec_tools
    lib = _library_with_misplacement(tools)
    anomaly = tools.memory_repair("scan_workspace_anomalies", {})
    assert anomaly["data"]["findings"], "misplaced memory must be flagged"

    data = _scan(tools)
    pair = tuple(sorted((lib["misplaced"]["id"], lib["partner"]["id"])))
    refs = {
        tuple(sorted((r["left_id"], r["right_id"]))): r
        for r in data.get("cross_bucket_references", [])
    }
    assert pair in refs, f"planted conflict must surface via the suspected-bucket sweep: {sorted(refs)}"
    ref = refs[pair]
    assert "numeric_value_candidate" in ref["reasons"]
    assert ref["suspected_workspace"] == "dbpgsql"
    assert "record_conflict" in ref["note"] or "workspace_review" in ref["note"]

    # And it must NOT be in candidates (it cannot land in record_conflict).
    cand_pairs = {
        tuple(sorted((c["left_id"], c["right_id"]))) for c in data["candidates"]
    }
    assert pair not in cand_pairs


def test_post_move_pair_becomes_recordable_candidate(vec_tools: MemoryTools) -> None:
    """C6 step 5 shape: after the move, the memory participates normally in
    its home bucket and the conflict pair lands in candidates."""
    tools = vec_tools
    lib = _library_with_misplacement(tools)
    tools.memory_repair("scan_workspace_anomalies", {})
    moved = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [int(lib["misplaced"]["id"])], "new_workspace": "dbpgsql",
        "reason": "confirmed placement", "authorized": True,
    })
    assert moved["ok"] is True, moved

    data = _scan(tools)
    pair = tuple(sorted((lib["misplaced"]["id"], lib["partner"]["id"])))
    cand_pairs = {
        tuple(sorted((c["left_id"], c["right_id"]))) for c in data["candidates"]
    }
    assert pair in cand_pairs, "post-move, the conflict pair must be a regular candidate"


def test_strict_scan_not_widened_by_suspect_sweep(vec_tools: MemoryTools) -> None:
    """A strict caller's page must not gain cross-bucket references: the
    suspected bucket usually sits outside its admitted set and the reference
    snippets would leak."""
    tools = vec_tools
    _library_with_misplacement(tools)
    tools.memory_repair("scan_workspace_anomalies", {})
    tools.settings.isolation = "strict"
    tools.settings.workspace = "apisvc"
    data = _scan(tools, workspace="apisvc")
    assert data.get("cross_bucket_references") == []
    for cand in data["candidates"]:
        # Strict scope: every candidate member must stay in apisvc.
        assert _bucket_of(tools, int(cand["left_id"])) == "apisvc"
        assert _bucket_of(tools, int(cand["right_id"])) == "apisvc"


def test_lightweight_page_keeps_cross_refs_slim(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _library_with_misplacement(tools)
    tools.memory_repair("scan_workspace_anomalies", {})
    page = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10, "include_check": True,
    })
    refs = page["data"].get("cross_bucket_references", [])
    assert refs, "suspected-bucket sweep must surface references on the slim page too"
    for ref in refs:
        assert len(str(ref.get("left_snippet") or "")) <= 200
        assert len(str(ref.get("right_snippet") or "")) <= 200
        assert "members" not in ref
