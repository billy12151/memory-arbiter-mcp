"""C1 (0.15.13): scan_candidates response slimming.

workbuddy's first spec-shaped round sized a batch=200 page at ~12MB and
exploded its session. The default page now carries only the lightweight
triage identity per candidate; include_quotes=true restores the full
member/slot envelope that record_conflict consumes (the spec sample call
passes it for exactly that reason).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter.tools import MemoryTools


@pytest.fixture()
def vec_tools(tmp_path: Path):
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    yield tools


def _write_pair(tools: MemoryTools, content: str, workspace: str = "w") -> None:
    tools.memory_write(content=content, subject="scan-c1", tags=[], workspace=workspace)
    tools.memory_write(content=content + " but the value is 7 now", subject="scan-c1", tags=[], workspace=workspace)


def _scan(tools: MemoryTools, data: dict | None = None) -> dict:
    result = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10,
        **(data or {}),
    })
    assert result["ok"] is True, result
    return result["data"]


def test_default_page_is_lightweight(vec_tools: MemoryTools) -> None:
    _write_pair(vec_tools, "deploy region is eu-west-1")
    assert vec_tools.wait_evidence_worker_drained(timeout=5)

    page = _scan(vec_tools)
    candidates = page["candidates"]
    assert candidates, "expected at least one candidate from the numeric pair"
    for item in candidates:
        # Lightweight identity only: no members/slot_key/value_groups/deep_read.
        assert set(item) <= {
            "left_id", "right_id", "workspace", "state", "route",
            "reasons", "distance", "left_quote", "right_quote", "qwen_signal",
        }
        assert "members" not in item and "value_groups" not in item
        assert "slot_key" not in item and "deep_read" not in item
        assert len(str(item.get("left_quote") or "")) <= 60
        assert len(str(item.get("right_quote") or "")) <= 60


def test_include_quotes_restores_full_envelope(vec_tools: MemoryTools) -> None:
    _write_pair(vec_tools, "deploy region is eu-west-1")
    assert vec_tools.wait_evidence_worker_drained(timeout=5)

    page = _scan(vec_tools, {"include_quotes": True})
    candidates = page["candidates"]
    assert candidates
    item = candidates[0]
    assert item.get("members"), "include_quotes must restore full members"
    for member in item["members"]:
        for field in (
            "memory_id", "version", "evidence_quote", "evidence_span",
            "content_hash", "detector_version",
        ):
            assert field in member, f"member missing {field}"
    # value_groups/slot_groups appear only after a successful Qwen enhancement;
    # the deterministic baseline must never drop the member envelope.


def test_lightweight_page_size_bound(vec_tools: MemoryTools) -> None:
    # 12 clean memories: the lightweight page must stay far below the old
    # ~12MB-per-200-candidate shape even at the spec's batch size.
    for index in range(12):
        vec_tools.memory_write(
            content=f"clean library row {index} with padding text " * 40,
            subject=f"row-{index}", tags=[], workspace="w",
        )
    assert vec_tools.wait_evidence_worker_drained(timeout=10)

    page = _scan(vec_tools, {"batch": 50})
    size = len(json.dumps(page, ensure_ascii=False).encode("utf-8"))
    assert size < 200_000, f"lightweight page too large: {size} bytes"


def test_spec_sample_carries_include_quotes() -> None:
    from memory_arbiter.scan_tasks import SCHEDULED_TASKS_SPEC

    conflict_scan = next(
        task for task in SCHEDULED_TASKS_SPEC["tasks"] if task["name"] == "conflict_scan"
    )
    scan_call = next(
        call for call in conflict_scan["calls"] if call.get("task") == "scan_candidates"
    )
    assert scan_call["data"]["batch"] == 50
    assert scan_call["data"].get("include_quotes") is True, (
        "the spec sample must pass include_quotes=true: triage builds "
        "record_conflict members from the page payload"
    )
