"""memory_repair(task="scan_duplicates") — bounded full-library duplicate sweep.

Decided in mema 825 (problem 2): an independent repair task separate from the
conflict scan, aggregating the per-page duplicates_pool server-side under one
global cap with lightweight entries by default and include_quotes opt-in.
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


def _write_for_scan(tools: MemoryTools, content: str, workspace: str = "w") -> int:
    result = tools.memory_write(content=content, subject="scan", tags=[], workspace=workspace)
    return int(result["data"]["id"])


def _sweep(tools: MemoryTools, data: dict | None = None) -> dict:
    result = tools.memory_repair("scan_duplicates", data or {})
    assert result["ok"] is True, result
    return result["data"]


def test_scan_duplicates_full_sweep_lightweight(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _write_for_scan(tools, "alpha duplicate fact statement")
    _write_for_scan(tools, "alpha duplicate fact statement")
    _write_for_scan(tools, "beta duplicate fact statement")
    _write_for_scan(tools, "beta duplicate fact statement")
    assert tools.wait_evidence_worker_drained(timeout=5)

    data = _sweep(tools)
    assert data["total_pairs"] == 2
    assert data["truncated"] is False
    assert data["anchors_scanned"] == 4
    assert len(data["duplicates"]) == 2
    for entry in data["duplicates"]:
        assert set(entry) == {
            "left_id", "right_id", "left_subject", "right_subject",
            "workspace", "reason", "distance", "candidate_key_hash",
        }
        assert entry["left_subject"] == "scan" and entry["right_subject"] == "scan"
        assert entry["workspace"] == "w"
        assert entry["reason"] in {"equivalent_value", "compatible_evidence"}
        assert entry["left_id"] < entry["right_id"]

    quoted = _sweep(tools, {"include_quotes": True})
    for entry in quoted["duplicates"]:
        assert entry["left_quote"] and entry["right_quote"]


def test_scan_duplicates_suppresses_recorded_pairs(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _write_for_scan(tools, "gamma duplicate fact statement")
    _write_for_scan(tools, "gamma duplicate fact statement")
    assert tools.wait_evidence_worker_drained(timeout=5)

    page = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10, "include_duplicates": True,
    })
    pool = page["data"]["duplicates_pool"]
    assert len(pool) == 1
    recorded = tools.memory_repair("record_conflict", {
        "slot_key": None,
        "members": pool[0]["members"],
        "value_groups": [
            {"normalized_value": "None", "display_value": "same value",
             "members": [f"{pool[0]['members'][0]['memory_id']}@{pool[0]['members'][0]['version']}",
                         f"{pool[0]['members'][1]['memory_id']}@{pool[0]['members'][1]['version']}"]},
        ],
        "status": "not_a_conflict",
        "detector_version": pool[0]["members"][0]["detector_version"],
        "prompt_version": None,
        "source": "scan",
        "reason": "same value, not a conflict",
        "workspace": "w",
        "authorized": True,
    })
    assert recorded["ok"] is True, recorded["data"]

    data = _sweep(tools)
    assert data["duplicates"] == [], "recorded not_a_conflict pairs must not re-surface"


def test_scan_duplicates_global_cap_and_truncation(
    vec_tools: MemoryTools, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = vec_tools
    for _ in range(2):
        _write_for_scan(tools, "alpha duplicate fact statement")
        _write_for_scan(tools, "beta duplicate fact statement")
    assert tools.wait_evidence_worker_drained(timeout=5)

    monkeypatch.setattr("memory_arbiter.surfaces.SCAN_DUPLICATES_MAX_RESULTS", 1)
    data = _sweep(tools)
    assert len(data["duplicates"]) == 1
    assert data["truncated"] is True
    assert data["max_results"] == 1


def test_scan_duplicates_aggregates_across_pages(
    vec_tools: MemoryTools, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = vec_tools
    for _ in range(2):
        _write_for_scan(tools, "alpha duplicate fact statement")
        _write_for_scan(tools, "beta duplicate fact statement")
    assert tools.wait_evidence_worker_drained(timeout=5)

    monkeypatch.setattr("memory_arbiter.surfaces.SCAN_DUPLICATES_BATCH", 1)
    data = _sweep(tools)
    assert data["total_pairs"] == 2
    assert data["truncated"] is False


def test_scan_duplicates_strict_scope_does_not_leak(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _write_for_scan(tools, "alpha duplicate fact statement", workspace="apisvc")
    _write_for_scan(tools, "alpha duplicate fact statement", workspace="apisvc")
    _write_for_scan(tools, "outside duplicate fact statement", workspace="dbpgsql")
    _write_for_scan(tools, "outside duplicate fact statement", workspace="dbpgsql")
    assert tools.wait_evidence_worker_drained(timeout=5)

    tools.settings.isolation = "strict"
    tools.settings.workspace = "apisvc"
    data = _sweep(tools)
    pairs = {(entry["left_id"], entry["right_id"]) for entry in data["duplicates"]}
    assert pairs, "expected at least one apisvc pair"
    with tools.db.connection() as conn:
        for left_id, right_id in pairs:
            for memory_id in (left_id, right_id):
                row = conn.execute(
                    "SELECT workspace, workspace_canonical FROM memories WHERE id=?",
                    (memory_id,),
                ).fetchone()
                canonical = str(row["workspace_canonical"] or row["workspace"])
                assert canonical == "apisvc", f"strict sweep leaked memory {memory_id}"


def test_scan_duplicates_rejects_unknown_fields(vec_tools: MemoryTools) -> None:
    result = vec_tools.memory_repair("scan_duplicates", {"include_quotes": False, "bogus": 1})
    assert result["ok"] is True
    assert any("unknown field" in warning for warning in result.get("warnings") or [])


def test_scan_duplicates_writes_no_scan_log_and_advances_no_progress(
    vec_tools: MemoryTools, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = vec_tools
    _write_for_scan(tools, "alpha duplicate fact statement")
    _write_for_scan(tools, "alpha duplicate fact statement")
    assert tools.wait_evidence_worker_drained(timeout=5)

    def _no_progress(**kwargs):
        raise AssertionError("scan_duplicates must not advance conflict-scan progress")

    monkeypatch.setattr(tools.db, "record_conflict_scan_page", _no_progress)
    monkeypatch.setattr(tools.db, "complete_conflict_scan", _no_progress)
    data = _sweep(tools)
    assert data["total_pairs"] == 1
    scan_log = tools.settings.db_path.parent / "scan_log.jsonl"
    assert not scan_log.exists(), "scan_duplicates must not write scan_log.jsonl"


def test_scan_candidates_zero_anchors_write_no_scan_log(vec_tools: MemoryTools) -> None:
    result = vec_tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50})
    assert result["ok"] is True
    assert result["data"]["anchors_scanned"] == 0
    assert result["data"]["next_anchor_memory_id"] is None
    scan_log = vec_tools.settings.db_path.parent / "scan_log.jsonl"
    assert not scan_log.exists(), (
        "a zero-anchor boundary on an empty library is no evidence a task exists"
    )


def test_scan_duplicates_page_truncation_propagates(
    vec_tools: MemoryTools, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = vec_tools
    for _ in range(4):
        _write_for_scan(tools, "shared identical duplicate fact statement")
    assert tools.wait_evidence_worker_drained(timeout=5)

    monkeypatch.setattr("memory_arbiter.surfaces.SCAN_DUPLICATES_BATCH", 1)
    data = _sweep(tools)
    # First anchor alone pairs with 3 peers but its page pool caps at 2*1.
    assert data["truncated"] is True
