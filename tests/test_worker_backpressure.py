"""Backpressure for the semantic worker (R2-S1: the retired evidence-index
worker's own capacity test died with the class — the merged queue's
queue_full rejection is covered by
test_vnext_evidence.test_queue_full_drop_completes_exact_reserved_task_end_to_end)."""
from __future__ import annotations

from pathlib import Path

import tests.test_vnext_evidence as tv


def test_memory_write_warns_when_evidence_worker_busy(tmp_path: Path) -> None:
    """C2：索引繁忙告警随写入路径搬到语义队列（busy 文案沿用）。"""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    worker = tools._semantic_worker

    with worker._cond:
        worker._pending = {i: {"version": 1, "task_id": f"semantic:{i}@1"} for i in range(1000, 1200)}
        worker._cond.notify_all()
    try:
        resp = tools.memory_write(
            content="queued write", subject="s", tags=[], workspace="w",
        )
        assert resp["ok"] is True, "write must succeed even when indexer is busy"
        memory_id = resp["data"]["id"]
        assert memory_id is not None
        assert resp["data"]["evidence_index"]["status"] == "incomplete"
        assert resp["data"]["evidence_index"]["reason"] == "queue_full"
        # The memory itself is persisted regardless of queue pressure.
        fetched = tools.db.get_memory(memory_id)
        assert fetched is not None
        assert fetched["subject"] == "s"
    finally:
        with worker._cond:
            worker._inflight.clear()
            worker._cond.notify_all()
