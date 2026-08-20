"""Background workers for MemoryTools (Phase 4 extraction)."""
from __future__ import annotations

import threading
import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import MemoryTools


class LocalTextIndexWorker:
    """Coalescing evidence-index worker keyed by memory id."""

    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools
        self._pending: dict[int, dict[str, Any]] = {}
        self._inflight: set[int] = set()
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._shutdown = False
        self._processed = 0
        self._last_error: Optional[str] = None

    def start(self) -> None:
        self._ensure_thread()

    def enqueue(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        displaced: Optional[dict[str, Any]] = None
        with self._cond:
            if self._shutdown:
                return {"status": "shutdown"}
            displaced = self._pending.get(int(memory_id))
            self._pending[int(memory_id)] = dict(snapshot)
            self._cond.notify_all()
        if displaced is not None:
            self._tools._semantic_worker.complete(
                str(displaced["task_id"]),
                {"status": "incomplete", "reason": "coalesced_by_newer_snapshot", "notices_created": 0},
            )
        self._ensure_thread()
        return {"status": "queued"}

    def status(self) -> dict[str, Any]:
        with self._cond:
            return {
                "queue_depth": len(self._pending), "inflight": sorted(self._inflight),
                "processed": self._processed, "last_error": self._last_error,
                "shutdown": self._shutdown,
            }

    def shutdown(self, discard_pending: bool = False) -> dict[str, Any]:
        with self._cond:
            self._shutdown = True
            discarded = len(self._pending) if discard_pending else 0
            if discard_pending:
                self._pending.clear()
            self._cond.notify_all()
            return {"status": "shutdown", "discarded_pending": discarded}

    def wait_drained(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            while self._pending or self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="memory-arbiter-local-text-index", daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._shutdown:
                    self._cond.wait()
                if self._shutdown and not self._pending:
                    return
                memory_id = next(iter(self._pending))
                snapshot = self._pending.pop(memory_id)
                self._inflight.add(memory_id)
            task_id = str(snapshot.get("task_id") or f"semantic:{memory_id}@{int(snapshot.get('version') or 1)}")
            try:
                current = self._tools.db.get_memory(memory_id)
                if current is None or int(current.get("version") or 1) != int(snapshot.get("version") or 1):
                    result = {"status": "skipped", "reason": "stale_or_missing"}
                else:
                    result = self._tools._index_local_text_evidence(memory_id, current)
                    if result.get("status") == "indexed":
                        context = snapshot.get("trusted_applying_context")
                        if context is None:
                            self._tools._enqueue_semantic_conflict_check(
                                memory_id, current, after_evidence=True,
                            )
                        else:
                            self._tools._enqueue_semantic_conflict_check(
                                memory_id, current, after_evidence=True,
                                trusted_applying_context=context,
                            )
                if result.get("status") != "indexed":
                    self._tools._semantic_worker.complete(
                        task_id,
                        {
                            "status": "incomplete",
                            "reason": f"evidence_index_{result.get('reason') or result.get('status') or 'failed'}",
                            "notices_created": 0,
                        },
                    )
                with self._cond:
                    # A stale publish race (edit landed between fetch and
                    # publish) is routine coalescing, not an error.
                    clean = result.get("status") in {"indexed", "skipped"} or result.get(
                        "outcome"
                    ) in {"stale_snapshot"}
                    self._last_error = None if clean else str(result)
                    self._processed += 1
            except Exception as exc:
                self._tools._semantic_worker.complete(
                    task_id,
                    {"status": "incomplete", "reason": "evidence_index_error", "error": str(exc), "notices_created": 0},
                )
                with self._cond:
                    self._last_error = str(exc)
            finally:
                with self._cond:
                    self._inflight.discard(memory_id)
                    self._cond.notify_all()


class SemanticConflictWorker:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools
        self._pending: dict[int, dict[str, Any]] = {}
        self._inflight: set[int] = set()
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._paused = False
        self._runtime_disabled = False
        self._shutdown = False
        self._last_error: Optional[str] = None
        self._error_seq = 0
        self._processed = 0
        self._skipped = 0
        # Overflow drops kill the whole conflict job (notify included); the
        # drop must stay observable instead of silently vanishing.
        self._dropped_queue_full = 0
        self._expected: set[str] = set()
        self._completed: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        if self._tools.settings.semantic_conflict_on_write == "off":
            return
        self._ensure_thread()
        if self._tools.settings.semantic_conflict_preload:
            threading.Thread(
                target=self._preload_backend,
                name="memory-arbiter-semantic-preload",
                daemon=True,
            ).start()

    def _preload_backend(self) -> None:
        try:
            backend = self._tools._ensure_semantic_backend()
            if backend is not None:
                backend.load()
        except Exception as exc:
            self.set_error(str(exc))

    def reserve(self, task_id: str) -> None:
        with self._cond:
            if task_id not in self._completed:
                self._expected.add(task_id)

    def complete(self, task_id: str, result: dict[str, Any]) -> None:
        with self._cond:
            if task_id in self._completed:
                return
            self._completed[task_id] = {**result, "task_id": task_id, "dedupe_key": task_id}
            self._expected.discard(task_id)
            if len(self._completed) > 1000:
                self._completed.pop(next(iter(self._completed)))
            self._cond.notify_all()

    def enqueue(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        task_id = str(snapshot.get("task_id") or f"semantic:{int(memory_id)}@{int(snapshot.get('version') or 1)}")
        snapshot = {**snapshot, "task_id": task_id, "dedupe_key": task_id}
        with self._cond:
            if task_id in self._completed:
                return {"status": "completed", "task_id": task_id, "dedupe_key": task_id}
            rejected: Optional[str] = None
            if self._shutdown:
                rejected = "shutdown"
            elif self._runtime_disabled:
                rejected = "runtime_disabled"
            elif self._paused:
                rejected = "paused"
            max_size = int(getattr(self._tools.settings, "semantic_conflict_queue_max_size", 100))
            if rejected is None and len(self._pending) >= max_size and int(memory_id) not in self._pending:
                self._dropped_queue_full += 1
                rejected = "queue_full"
            if rejected is not None:
                self._completed[task_id] = {
                    "status": "incomplete", "reason": rejected, "notices_created": 0,
                    "task_id": task_id, "dedupe_key": task_id,
                }
                self._expected.discard(task_id)
                self._cond.notify_all()
                return dict(self._completed[task_id])
            displaced = self._pending.get(int(memory_id))
            if displaced is not None:
                displaced_task_id = str(displaced["task_id"])
                self._completed[displaced_task_id] = {
                    "status": "incomplete", "reason": "coalesced_by_newer_snapshot", "notices_created": 0,
                    "task_id": displaced_task_id, "dedupe_key": displaced_task_id,
                }
                self._expected.discard(displaced_task_id)
            self._pending[int(memory_id)] = snapshot
            self._expected.add(task_id)
            self._cond.notify_all()
        self._ensure_thread()
        return {"status": "queued", "task_id": task_id, "dedupe_key": task_id}

    def wait_task(self, task_id: str, timeout: float) -> Optional[dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            while task_id not in self._completed:
                if task_id not in self._expected:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return dict(self._completed[task_id])

    def status(self) -> dict[str, Any]:
        with self._cond:
            if self._shutdown:
                state = "shutdown"
            elif self._tools.settings.semantic_conflict_on_write == "off":
                state = "on_write_off"
            elif self._runtime_disabled:
                state = "disabled"
            elif self._paused:
                state = "paused"
            else:
                state = "running"
            return {
                "runtime_state": state,
                "queue_depth": len(self._pending),
                "inflight": sorted(self._inflight),
                "processed": self._processed,
                "skipped": self._skipped,
                "dropped_queue_full": self._dropped_queue_full,
                "shutdown": self._shutdown,
                "last_error": self._last_error,
            }

    def pause(self) -> None:
        with self._cond:
            self._paused = True

    def shutdown(self, discard_pending: bool = True) -> dict[str, Any]:
        with self._cond:
            self._shutdown = True
            self._paused = True
            skipped = 0
            if discard_pending:
                skipped = len(self._pending)
                for snapshot in self._pending.values():
                    task_id = str(snapshot["task_id"])
                    self._completed[task_id] = {
                        "status": "incomplete", "reason": "shutdown", "notices_created": 0,
                        "task_id": task_id, "dedupe_key": task_id,
                    }
                    self._expected.discard(task_id)
                self._pending.clear()
                self._skipped += skipped
            self._cond.notify_all()
            return {"status": "shutdown", "discarded_pending": skipped, "inflight": len(self._inflight)}

    def wait_drained(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            while self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def disable_runtime(self) -> None:
        with self._cond:
            self._runtime_disabled = True
            self._paused = True
            skipped = len(self._pending)
            if skipped:
                for snapshot in self._pending.values():
                    task_id = str(snapshot["task_id"])
                    self._completed[task_id] = {
                        "status": "incomplete", "reason": "runtime_disabled", "notices_created": 0,
                        "task_id": task_id, "dedupe_key": task_id,
                    }
                    self._expected.discard(task_id)
                self._pending.clear()
                self._skipped += skipped
            self._cond.notify_all()

    def enable_runtime(self) -> None:
        with self._cond:
            if self._shutdown:
                return
            self._runtime_disabled = False
            self._paused = False
            self._cond.notify_all()

    def resume(self) -> None:
        with self._cond:
            if self._runtime_disabled or self._shutdown:
                return
            self._paused = False
            self._cond.notify_all()
        self._ensure_thread()

    def set_error(self, message: str) -> None:
        # All other reads/writes of _last_error happen under _cond (status() and
        # the _run finally branch). Route external writers through the same lock
        # instead of reaching into the private attribute from the caller.
        with self._cond:
            self._last_error = str(message) if message is not None else None
            self._error_seq += 1

    def _ensure_thread(self) -> None:
        with self._lock:
            with self._cond:
                if self._shutdown:
                    return
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="memory-arbiter-semantic-conflict", daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._cond:
                while (not self._pending or self._paused) and not self._shutdown:
                    self._cond.wait()
                if self._shutdown and not self._pending:
                    return
                if self._shutdown and self._paused:
                    return
                memory_id = next(iter(self._pending))
                snapshot = self._pending.pop(memory_id)
                self._inflight.add(memory_id)
            error_message: Optional[str] = None
            job_result: dict[str, Any] = {"status": "incomplete", "reason": "worker_error"}
            task_id = str(snapshot.get("task_id") or f"semantic:{memory_id}@{int(snapshot.get('version') or 1)}")
            with self._cond:
                error_seq_before = self._error_seq
            try:
                result = self._tools._process_semantic_conflict_job(memory_id, snapshot)
                if isinstance(result, dict):
                    job_result = result
            except Exception as exc:
                error_message = str(exc)
                job_result = {"status": "incomplete", "reason": "worker_error", "error": error_message}
            finally:
                with self._cond:
                    self._completed[task_id] = {**job_result, "task_id": task_id, "dedupe_key": task_id}
                    self._expected.discard(task_id)
                    if len(self._completed) > 1000:
                        self._completed.pop(next(iter(self._completed)))
                    self._inflight.discard(memory_id)
                    if error_message is not None:
                        self._last_error = error_message
                        self._error_seq += 1
                    else:
                        # A clean run clears a prior transient error so a stale
                        # last_error does not linger forever in worker status --
                        # UNLESS this same job recorded its own diagnostic via
                        # set_error (timeout / min-pair-budget early stop) and
                        # then returned normally. In that case _error_seq moved
                        # while the job ran, so we must preserve the message the
                        # job just set instead of clobbering it to None.
                        if self._error_seq == error_seq_before:
                            self._last_error = None
                        self._processed += 1
                    self._cond.notify_all()
