"""Background workers for MemoryTools (Phase 4 extraction)."""
from __future__ import annotations

import threading
import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import MemoryTools


class LocalTextIndexWorker:
    """Coalescing vNext evidence-index worker keyed by memory id."""

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
        if self._tools._vnext_storage_enabled():
            self._ensure_thread()

    def enqueue(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        if not self._tools._vnext_storage_enabled():
            return {"status": "skipped", "reason": "legacy_storage_profile"}
        with self._cond:
            if self._shutdown:
                return {"status": "shutdown"}
            self._pending[int(memory_id)] = dict(snapshot)
            self._cond.notify_all()
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
            try:
                current = self._tools.db.get_memory(memory_id)
                if current is None or int(current.get("version") or 1) != int(snapshot.get("version") or 1):
                    result = {"status": "skipped", "reason": "stale_or_missing"}
                else:
                    result = self._tools._index_local_text_evidence(memory_id, current)
                    if result.get("status") == "indexed":
                        self._tools._enqueue_semantic_conflict_check(
                            memory_id, current, after_evidence=True,
                        )
                with self._cond:
                    self._last_error = None if result.get("status") in {"indexed", "skipped"} else str(result)
                    self._processed += 1
            except Exception as exc:
                with self._cond:
                    self._last_error = str(exc)
            finally:
                with self._cond:
                    self._inflight.discard(memory_id)
                    self._cond.notify_all()


class SplitReindexWorker:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools
        self._pending: dict[int, dict[str, Any]] = {}
        self._inflight: set[int] = set()
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def start(self) -> None:
        self._ensure_thread()

    def enqueue(self, memory_id: int, snapshot: dict[str, Any]) -> None:
        with self._cond:
            self._pending[int(memory_id)] = snapshot
            self._cond.notify_all()
        self._ensure_thread()

    def pending_ids(self) -> list[int]:
        with self._cond:
            return sorted(set(self._pending.keys()) | self._inflight)

    def wait_drained(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
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
                target=self._run,
                name="memory-arbiter-split-reindex",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending:
                    self._cond.wait()
                memory_id = next(iter(self._pending))
                snapshot = self._pending.pop(memory_id)
                self._inflight.add(memory_id)
            try:
                self._process_one(memory_id, snapshot)
            except Exception as exc:
                self._mark_failed_from_snapshot(snapshot, "worker", str(exc))
            finally:
                with self._cond:
                    self._inflight.discard(memory_id)
                    self._cond.notify_all()

    def _process_one(self, memory_id: int, snapshot: dict[str, Any]) -> None:
        plan = snapshot.get("plan") or []
        if not plan:
            return
        result = self._tools._publish_sections(
            memory_id,
            str(snapshot.get("content") or ""),
            plan,
            str(snapshot.get("content_hash") or ""),
            int(snapshot.get("memory_version") or 1),
            snapshot.get("split_status"),
            int(snapshot.get("split_revision") or 0),
            decision_kind="split",
            provenance="parser",
        )
        if not result.get("ok"):
            error = (result.get("data") or {}).get("error")
            if error and error not in {"memory_changed", "split_revision_conflict", "vec_space_changed"}:
                self._mark_failed_from_snapshot(snapshot, "publish", str(error))

    def _mark_failed_from_snapshot(self, snapshot: dict[str, Any], stage: str, message: str) -> None:
        try:
            memory_id = snapshot.get("memory_id")
            if memory_id is None:
                raise ValueError("missing memory_id")
            self._tools._mark_split_failed(
                int(memory_id),
                str(snapshot.get("content_hash") or ""),
                int(snapshot.get("memory_version") or 1),
                int(snapshot.get("split_revision") or 0),
                snapshot.get("split_status"),
                stage,
                message,
            )
        except Exception:
            pass


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

    def start(self) -> None:
        if (
            not self._tools.settings.semantic_conflict_enabled
            and not self._tools._vnext_storage_enabled()
        ):
            return
        # When on_write is "off", writes never enqueue (see _enqueue_semantic_conflict_check),
        # so spinning up the worker thread and preloading the model would just burn
        # resources for a queue that stays empty. Treat "off" like config-disabled here;
        # resume()/enable_runtime() can still revive the worker if the caller flips
        # on_write at runtime later and re-enqueues.
        if (
            getattr(self._tools.settings, "semantic_conflict_on_write", "async") == "off"
            and not self._tools._vnext_storage_enabled()
        ):
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

    def enqueue(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        if (
            not self._tools.settings.semantic_conflict_enabled
            and not self._tools._vnext_storage_enabled()
        ):
            return {"status": "disabled"}
        with self._cond:
            if self._shutdown:
                return {"status": "shutdown"}
            if self._runtime_disabled:
                return {"status": "runtime_disabled"}
            if self._paused:
                return {"status": "paused"}
            max_size = int(getattr(self._tools.settings, "semantic_conflict_queue_max_size", 100))
            if len(self._pending) >= max_size and int(memory_id) not in self._pending:
                return {"status": "queue_full"}
            self._pending[int(memory_id)] = snapshot
            self._cond.notify_all()
        self._ensure_thread()
        return {"status": "queued"}

    def status(self) -> dict[str, Any]:
        with self._cond:
            if (
                not self._tools.settings.semantic_conflict_enabled
                and not self._tools._vnext_storage_enabled()
            ):
                state = "config_disabled"
            elif self._shutdown:
                state = "shutdown"
            elif (
                getattr(self._tools.settings, "semantic_conflict_on_write", "async") == "off"
                and not self._tools._vnext_storage_enabled()
            ):
                # enabled but on_write=off: writes never enqueue and start() does
                # not spin up the thread, so report disabled rather than "running".
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
            with self._cond:
                error_seq_before = self._error_seq
            try:
                self._tools._process_semantic_conflict_job(memory_id, snapshot)
            except Exception as exc:
                error_message = str(exc)
            finally:
                with self._cond:
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
