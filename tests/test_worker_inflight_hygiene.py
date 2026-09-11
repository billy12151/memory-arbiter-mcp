# ── Evidence worker inflight hygiene (review P1-7) ──
"""No code path may strand an _inflight entry — wait_drained hangs forever
on a leaked entry (it is the shutdown gate). The finally-discard covers every
exception path through _run's processing section.
"""

from __future__ import annotations

from pathlib import Path

import tests.test_vnext_evidence as tv


def test_processing_exception_still_drains_inflight(tmp_path: Path) -> None:
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    worker = tools._evidence_worker
    tools.memory_write(
        content="seed fact", subject="seed", workspace="w",
        source_type="agent_generated",
    )
    with tools.db.connection() as conn:
        memory_id = int(conn.execute(
            "SELECT MIN(id) AS id FROM memories"
        ).fetchone()["id"])

    # Make the very first fallible step inside the try blow up; the finally
    # must still discard the inflight entry.
    def boom(*args, **kwargs):
        raise RuntimeError("index exploded")
    tools._index_local_text_evidence = boom  # type: ignore[method-assign]

    worker.enqueue(memory_id, {"version": 1, "task_id": f"semantic:{memory_id}@1"})
    assert worker.wait_drained(timeout=10.0) is True
    with worker._cond:
        assert worker._inflight == set()
    assert worker._last_error == "index exploded"


def test_shutdown_after_clean_processing_drains(tmp_path: Path) -> None:
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    worker = tools._evidence_worker
    result = tools.memory_write(
        content="clean fact", subject="clean", workspace="w",
        source_type="agent_generated",
    )
    assert result["ok"] is True

    worker.shutdown()
    assert worker.wait_drained(timeout=10.0) is True
    with worker._cond:
        assert worker._inflight == set()
        assert worker._pending == {}
