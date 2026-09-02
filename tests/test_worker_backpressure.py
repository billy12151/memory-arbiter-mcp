"""Backpressure for the evidence-index worker (review decision #825 issue 4)."""
from __future__ import annotations

from pathlib import Path

import tests.test_vnext_evidence as tv
from memory_arbiter.constants import EVIDENCE_QUEUE_MAX_SIZE


def test_evidence_worker_enqueue_rejects_when_queue_full(tmp_path: Path) -> None:
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    worker = tools._evidence_worker

    # Saturate the worker by marking the cap as inflight; this avoids waiting
    # for 200 real jobs while still exercising the capacity check.
    with worker._cond:
        worker._inflight = set(range(EVIDENCE_QUEUE_MAX_SIZE))
    try:
        result = worker.enqueue(9999, {"version": 1, "task_id": "semantic:9999@1"})
        assert result["status"] == "busy"
        assert result["reason"] == "evidence_worker_queue_full"
        assert result["queue_depth"] == 0
        assert len(worker._pending) <= EVIDENCE_QUEUE_MAX_SIZE
        # The new item must not have been queued.
        assert 9999 not in worker._pending
    finally:
        with worker._cond:
            worker._inflight.clear()
            worker._cond.notify_all()


def test_memory_write_warns_when_evidence_worker_busy(tmp_path: Path) -> None:
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    worker = tools._evidence_worker

    with worker._cond:
        worker._inflight = set(range(EVIDENCE_QUEUE_MAX_SIZE))
    try:
        resp = tools.memory_write(
            content="queued write", subject="s", tags=[], workspace="w",
        )
        assert resp["ok"] is True, "write must succeed even when indexer is busy"
        memory_id = resp["data"]["id"]
        assert memory_id is not None
        assert resp["data"]["evidence_index"]["status"] == "busy"
        depth = resp["data"]["evidence_index"].get("queue_depth", 0)
        assert any(
            "evidence indexer is busy" in w and str(depth) in w
            for w in resp["warnings"]
        )
        # The memory itself is persisted regardless of queue pressure.
        fetched = tools.db.get_memory(memory_id)
        assert fetched is not None
        assert fetched["subject"] == "s"
    finally:
        with worker._cond:
            worker._inflight.clear()
            worker._cond.notify_all()
