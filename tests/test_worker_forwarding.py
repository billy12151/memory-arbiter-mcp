"""Regression for mema #813: the evidence-index worker forwards snapshots
across a thread boundary as plain dicts, so a trusted_applying_context stored
as a dict must be rehydrated into the frozen dataclass before the semantic
enqueue (0.14.9 typing refactor left this one forwarding point passing the
raw dict, and .to_dict() inside the enqueue crashed with AttributeError,
polluting worker last_error and skipping the post-apply semantic recheck).
"""
from __future__ import annotations

from pathlib import Path

import tests.test_vnext_evidence as tv
from memory_arbiter.models import TrustedApplyingContext
from memory_arbiter.tools import MemoryTools


def test_evidence_worker_forwarding_deserializes_trusted_context(
    tmp_path: Path, monkeypatch,
) -> None:
    tools: MemoryTools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    memory_id = tools.memory_write(
        content="database is sqlite", subject="a", tags=[], workspace="w",
    )["data"]["id"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    # C2 (worker merge): the forwarding chain is gone — the write path
    # enqueues the semantic job DIRECTLY, and the trusted context must
    # survive as a plain dict on the snapshot (crossing the thread boundary),
    # rehydrating at its consumer (process_conflicts §15.3). Pin the
    # pass-through, the mema #813 lineage of this test.
    context_dict = {
        "conflict_id": 1, "revision": 2, "memory_id": memory_id,
        "action": "update_current_claim", "chosen_value": "sqlite",
    }
    receipt = tools._post_commit(
        memory_id, None, recheck_conflicts=False,
        trusted_applying_context=TrustedApplyingContext(**context_dict),
    )[0]
    assert receipt["status"] in {"queued", "completed"}
    task_id = receipt["semantic_task_id"]
    completed = tools._semantic_worker.wait_task(task_id, 5.0)
    assert completed is not None
    assert tools._semantic_worker.status()["last_error"] is None


def test_malformed_trusted_context_degrades_to_context_free(
    tmp_path: Path, monkeypatch,
) -> None:
    # R2-S1 移植：C2 单队列后 trusted context 随快照直接进语义 job，重水化
    # 点在 job 内的 §15.3 门（evidence._conflicts_deterministic_collect）。
    # on_write 须 ≠ off，否则 job 降级 index_only 根本不触达重水化；后端钉
    # None 只走确定性相位。
    tools: MemoryTools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "sync"
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: None)
    memory_id = tools.memory_write(
        content="another statement", subject="b", tags=[], workspace="w",
    )["data"]["id"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    from memory_arbiter.models import TrustedApplyingContext
    captured: dict[str, object] = {}
    real_from_dict = TrustedApplyingContext.from_dict.__func__

    def spy_from_dict(cls, raw):
        rehydrated = real_from_dict(cls, raw)
        captured["context"] = rehydrated
        return rehydrated

    monkeypatch.setattr(TrustedApplyingContext, "from_dict", classmethod(spy_from_dict))

    snapshot = tv._job_snapshot(tools, memory_id)
    # 专用 task id：不得与写路径 job 的 semantic:{id}@{version} 撞 completed
    # 去重（撞上会直接回放旧回执，job 根本不入队）。
    snapshot["task_id"] = f"semantic-test:malformed:{memory_id}"
    snapshot["trusted_applying_context"] = {"conflict_id": "not-an-int"}  # malformed
    tools._semantic_worker.enqueue(memory_id, snapshot)
    completed = tools._semantic_worker.wait_task(snapshot["task_id"], 5.0)
    assert completed is not None and completed["status"] == "completed"

    assert captured.get("context") is None, "malformed context must fail open to context-free"
    assert tools._semantic_worker.status()["last_error"] is None
