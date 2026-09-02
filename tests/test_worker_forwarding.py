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
    assert tools.wait_evidence_worker_drained(timeout=5)

    captured: dict[str, object] = {}

    def fake_enqueue(mid, record, *, after_evidence: bool = False, trusted_applying_context=None):
        captured["context"] = trusted_applying_context
        return {"status": "ok"}

    monkeypatch.setattr(tools, "_enqueue_semantic_conflict_check", fake_enqueue)

    snapshot = tv._job_snapshot(tools, memory_id)
    snapshot["task_id"] = f"semantic:{memory_id}@{snapshot['version']}"
    snapshot["trusted_applying_context"] = {
        "conflict_id": 1, "revision": 2, "memory_id": memory_id,
        "action": "update_current_claim", "chosen_value": "sqlite",
    }
    tools._evidence_worker.enqueue(memory_id, snapshot)
    assert tools.wait_evidence_worker_drained(timeout=5)

    context = captured.get("context")
    assert isinstance(context, TrustedApplyingContext), (
        "worker must rehydrate the dict snapshot into the dataclass (mema #813)"
    )
    assert context.conflict_id == 1 and context.action == "update_current_claim"
    assert tools._evidence_worker.status()["last_error"] is None


def test_malformed_trusted_context_degrades_to_context_free(
    tmp_path: Path, monkeypatch,
) -> None:
    tools: MemoryTools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    memory_id = tools.memory_write(
        content="another statement", subject="b", tags=[], workspace="w",
    )["data"]["id"]
    assert tools.wait_evidence_worker_drained(timeout=5)

    captured: dict[str, object] = {}

    def fake_enqueue(mid, record, *, after_evidence: bool = False, trusted_applying_context=None):
        captured["context"] = trusted_applying_context
        return {"status": "ok"}

    monkeypatch.setattr(tools, "_enqueue_semantic_conflict_check", fake_enqueue)

    snapshot = tv._job_snapshot(tools, memory_id)
    snapshot["task_id"] = f"semantic:{memory_id}@{snapshot['version']}"
    snapshot["trusted_applying_context"] = {"conflict_id": "not-an-int"}  # malformed
    tools._evidence_worker.enqueue(memory_id, snapshot)
    assert tools.wait_evidence_worker_drained(timeout=5)

    assert captured.get("context") is None, "malformed context must fail open to context-free"
    assert tools._evidence_worker.status()["last_error"] is None
