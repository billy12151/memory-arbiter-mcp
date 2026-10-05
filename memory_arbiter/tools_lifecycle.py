"""生命周期 mixin：shutdown + boot 回填三件套（从 tools.py 搬出，拆分批 ⑥ 纯移动；__init__ 装配顺序契约留守不动）。"""
from __future__ import annotations

import json
import time
from typing import Any, TYPE_CHECKING
from .config import Settings
from .db import MemoryDB
from .scan_pipeline import ScanPipeline
from .constants import EMBED_PREFIX_STS

if TYPE_CHECKING:
    import threading
    from .embedder import ManagedEmbedder
    from .semantic_conflict import SemanticBackend
    from .workers import SemanticConflictWorker
    from .config import Settings
    from .db import MemoryDB
    from .scan_pipeline import ScanPipeline

class _ToolsLifecycle:
    if TYPE_CHECKING:
        db: "MemoryDB"
        settings: "Settings"
        _scan_pipeline: "ScanPipeline"
        _embedder: "ManagedEmbedder | None"
        _embedder_lock: "Any"
        _embedder_warnings: "list[str]"
        _semantic_backend: "SemanticBackend | None"
        _semantic_backend_lock: "Any"
        _semantic_runtime_disabled: bool
        _shutdown_lock: "Any"
        _shutdown_started: bool
        _shutdown_complete: bool
        _backfill_thread: "Any"
        _backfill_done: "threading.Event"
        _subject_tags_backfilled: bool
        _summary_vec_backfilled: bool
        _semantic_worker: "SemanticConflictWorker"
        def _get_semantic_backend_ref(self, *args: Any, **kwargs: Any) -> "SemanticBackend | None": ...
        def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _post_commit(self, *args: Any, **kwargs: Any) -> Any: ...
        def _record_check_degradation(self, *args: Any, **kwargs: Any) -> Any: ...

    def shutdown(self, timeout: float = 30.0) -> dict[str, Any]:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return {"ok": True, "already_shutdown": True}
            if self._shutdown_started:
                return {"ok": False, "already_shutdown": False, "shutdown_in_progress": True}
            self._shutdown_started = True
        timeout = max(0.0, float(timeout))
        deadline = time.monotonic() + timeout
        # 0.16.12 P2-T4: give the boot-backfill daemon a bounded chance to
        # leave its current encode before the process tears down the embedder
        # (llama-cpp Metal teardown mid-inference is the known exit-crash
        # path; bounded wait keeps shutdown latency predictable).
        if self._backfill_thread is not None and self._backfill_thread.is_alive():
            self._backfill_done.wait(timeout=min(30.0, timeout))
        worker_shutdown = self._semantic_worker.shutdown(discard_pending=True)
        # Shutdown also closes synchronous workspace-suggestion admission before
        # waiting; otherwise a new call can race the worker drain/unload phase.
        with self._semantic_backend_lock:
            self._semantic_runtime_disabled = True
            admitted_backend = self._semantic_backend
            if admitted_backend is not None:
                admitted_backend.set_disabled(True)
        remaining = max(0.0, deadline - time.monotonic())
        semantic_drained = self._semantic_worker.wait_drained(remaining)
        backend = self._get_semantic_backend_ref()
        unload_result: dict[str, Any] = {"ok": True, "unloaded": False, "reason": "no_backend"}
        if backend is not None:
            remaining = max(0.0, deadline - time.monotonic())
            unload_result = backend.unload(timeout=remaining, disable=True)
            if not unload_result.get("ok"):
                force_terminate = getattr(backend, "force_terminate", None)
                if callable(force_terminate):
                    unload_result = force_terminate()
        # Free the embedder's llama-cpp instance after the workers that use
        # it have drained: interpreter-teardown finalization of its Metal
        # buffers trips a ggml device-free assert (llama-cpp-python 0.3.34,
        # Apple Silicon) that crashes one-shot processes after their work is
        # done (see ManagedEmbedder.close). Best-effort hygiene — a close
        # failure never fails shutdown.
        with self._embedder_lock:
            embedder = self._embedder
        embedder_closed = False
        if embedder is not None:
            embedder.close()
            embedder_closed = True
        ok = bool(semantic_drained and unload_result.get("ok", False))
        with self._shutdown_lock:
            self._shutdown_complete = ok
            self._shutdown_started = False
        return {
            "ok": ok,
            "already_shutdown": False,
            "semantic_worker": worker_shutdown,
            "semantic_drained": semantic_drained,
            "backend_unload": unload_result,
            "embedder_closed": embedder_closed,
        }

    def _run_boot_backfills(self, embedder: "ManagedEmbedder") -> None:
        try:
            try:
                self._backfill_subject_tags_vectors(embedder)
            except Exception:
                pass
            try:
                self._backfill_memory_summary_vectors(embedder)
            except Exception:
                pass
            try:
                # 0.17.0 P2-2.5: row-level vectors for the conflict channel
                # (per-memory short transactions, embed outside; failures
                # leave the memory pending for the next restart).
                self._backfill_memory_row_vectors(embedder)
            except Exception:
                pass
        finally:
            self._backfill_done.set()

    def _backfill_memory_row_vectors(self, embedder: "ManagedEmbedder") -> int:
        """Row-vector backfill: segment + embed + publish_rows per active
        memory that has evidence units but no rows yet. publish_rows rechecks
        version and content hash under its own transaction, so a concurrent
        edit simply yields stale_snapshot and the memory stays pending."""
        from .evidence import evidence_content_hash
        from .pipeline.evidence import EvidencePipeline, _filter_exempted_segments  # noqa: F401  (类型注释用)
        from .rowseg import segment_rows

        rows = self.db.missing_row_vector_rows()
        written = 0
        for row in rows:
            try:
                content = str(row.get("content") or "")
                row_sha = str(row.get("content_sha") or "") or evidence_content_hash(content)
                segments = segment_rows(str(row.get("subject") or ""), content)
                if not segments:
                    continue
                # A3（0.17.1 修复批）：B3 超长表格段豁免——boot backfill 此前
                # 未接线，存量巨表会在这里重新嵌入并复活（每次启动白烧 GPU）。
                # 全豁免（空 subject + 全表）时不 publish（否则清空行会让
                # missing_row_vector_rows 每次重选该记忆，造成回填死循环）。
                segments, _exempted = _filter_exempted_segments(segments)
                if not segments:
                    continue
                # C1: one batched embed per memory instead of a per-segment
                # loop (14.5→8.4ms/item wall clock on the GPU worker).
                results = embedder.embed_texts(
                    [segment.text for segment in segments],
                    prefix=EMBED_PREFIX_STS,
                )
                vectors: list[list[float]] = []
                ok = True
                for result in results:
                    if not result or not result.embedding:
                        ok = False
                        break
                    vectors.append([float(x) for x in result.embedding])
                if not ok:
                    continue
                outcome = self.db.evidence.publish_rows(
                    int(row["id"]), int(row["version"] or 1), row_sha,
                    segments, vectors,
                )
                if outcome.get("published"):
                    written += 1
            except Exception:
                continue
        return written

    def wait_boot_backfills(self, timeout: float = 120.0) -> bool:
        """Block until the boot backfill thread finished (test/verification
        wait point; the thread itself is a daemon and never blocks shutdown)."""
        return self._backfill_done.wait(timeout)

    SUMMARY_SEGMENT_CHARS = 40
    SUMMARY_TOTAL_CHARS = 800

    @classmethod
    def _summary_embed_text(cls, subject: Any, tags: Any, content: Any) -> str:
        """C3a canonical summary text: subject + sorted tags + each body
        segment's first 40 chars, capped ~800 chars total.

        Segments split on blank lines (local_text_units' paragraph shape);
        ownership voting needs topical signal, not full bodies — a misplaced
        memory keeps its subject/tag vocabulary even when bodies drift.
        """
        cleaned = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        parts = [f"{str(subject or '').strip()}\n{' '.join(sorted(cleaned))}".strip()]
        body = str(content or "")
        for segment in body.split("\n\n"):
            segment = segment.strip()
            if segment:
                parts.append(segment[:cls.SUMMARY_SEGMENT_CHARS])
        return "\n".join(parts)[:cls.SUMMARY_TOTAL_CHARS]

    def _backfill_memory_summary_vectors(self, embedder: "ManagedEmbedder") -> int:
        """Embed the C3a summary vector for every active memory missing one.

        Same chunked embed-outside-transaction contract as the
        subject_tags_vec backfill (see _backfill_subject_tags_vectors):
        prepare per chunk, commit in one short transaction, failures leave
        rows missing for the next restart.
        """
        from .pipeline.operations import _embed_input_profile

        rows = self.db.missing_summary_vec_rows()
        written = 0
        try:
            with self.db.write_transaction() as conn:
                conn.execute(
                    "DELETE FROM memory_summary_vec WHERE id NOT IN "
                    "(SELECT id FROM memories WHERE status='active')"
                )
        except Exception:
            pass
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            # prepared carries the snapshot's derived-input profile: under the
            # chunk's write lock we re-check status AND that the row's
            # subject/tags/content still match the snapshot — a concurrent
            # edit's refresh_*_vector may already have written the NEW vector,
            # and overwriting it with the snapshot's embedding would regress
            # the row until its next edit (first-round review finding).
            prepared: list[tuple[int, list[float], tuple[str, str, str]]] = []
            for row in chunk:
                try:
                    profile = _embed_input_profile(row)
                    er = embedder.embed_text(
                        prefix=EMBED_PREFIX_STS,
                        body=self._summary_embed_text(
                            row.get("subject"), row.get("tags"), row.get("content"),
                        ),
                    )
                    if er and er.embedding:
                        prepared.append((int(row["id"]), [float(x) for x in er.embedding], profile))
                except Exception:
                    continue
            # 0.16.12 P2-T4: ONE short write transaction per 64-row chunk
            # (mirroring _backfill_subject_tags_vectors) — the embed above ran
            # outside the transaction; status/inputs re-check under the lock.
            try:
                chunk_written = 0
                with self.db.write_transaction() as conn:
                    for memory_id, vector, profile in prepared:
                        fresh = conn.execute(
                            "SELECT status, subject, tags, content FROM memories WHERE id = ?",
                            (memory_id,),
                        ).fetchone()
                        if fresh is None or str(fresh["status"]) != "active":
                            conn.execute(
                                "DELETE FROM memory_summary_vec WHERE id = ?", (memory_id,),
                            )
                            continue
                        if _embed_input_profile(dict(fresh)) != profile:
                            # Inputs moved since the snapshot — a fresher
                            # refresh already owns this row; skip, don't regress.
                            continue
                        conn.execute(
                            "DELETE FROM memory_summary_vec WHERE id = ?", (memory_id,),
                        )
                        conn.execute(
                            "INSERT INTO memory_summary_vec(id, embedding) VALUES (?, ?)",
                            (memory_id, json.dumps(vector)),
                        )
                        chunk_written += 1
                written += chunk_written  # count only what committed
            except Exception:
                continue
        return written

    def _backfill_subject_tags_vectors(self, embedder: "ManagedEmbedder") -> int:
        """Embed subject+tags for every active memory missing a hint vector.

        Covers libraries created before 0.15.3 and any vector whose publish
        was skipped by a failed embed or a vec-table rebuild. Embedding
        happens OUTSIDE the write transactions: an embed under BEGIN
        IMMEDIATE would hold the write lock for the whole chunk's model time
        and starve concurrent writers past the busy timeout, so each chunk
        is prepared first and committed in one short transaction. Rows whose
        embedding fails (empty sentinel) are left missing and retried on the
        next process start. Returns the number of vectors actually written
        (not the number of candidates).
        """
        from .pipeline.operations import _embed_input_profile
        from .pipeline.write import WritePipeline

        rows = self.db.missing_subject_tags_rows()
        written = 0
        # Stale rows for memories that left active (a retire committed
        # between a snapshot and its publish) waste KNN window slots and
        # have no other cleanup path — purge them while we are here.
        try:
            with self.db.write_transaction() as conn:
                conn.execute(
                    "DELETE FROM subject_tags_vec WHERE id NOT IN "
                    "(SELECT id FROM memories WHERE status='active')"
                )
        except Exception:
            pass
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            # Snapshot profile (subject, tags) guards the same race the
            # summary backfill guards: a concurrent edit's refresh may have
            # already written the NEW vector — skip instead of regressing.
            prepared: list[tuple[int, str, tuple[str, str]]] = []
            for row in chunk:
                try:
                    er = embedder.embed_text(
                        prefix=EMBED_PREFIX_STS,
                        body=WritePipeline._subject_tags_embed_text(
                            row.get("subject"), row.get("tags"),
                        ),
                    )
                    if er and er.embedding:
                        prepared.append((
                            int(row["id"]),
                            json.dumps([float(x) for x in er.embedding]),
                            _embed_input_profile(row)[:2],
                        ))
                except Exception:
                    continue
            with self.db.write_transaction() as conn:
                for memory_id, blob, profile in prepared:
                    # Re-check under the write lock: a retire that committed
                    # after the snapshot must not leave a stale vector.
                    fresh = conn.execute(
                        "SELECT status, subject, tags FROM memories WHERE id = ?", (memory_id,)
                    ).fetchone()
                    if fresh is None or str(fresh["status"]) != "active":
                        conn.execute(
                            "DELETE FROM subject_tags_vec WHERE id = ?", (memory_id,),
                        )
                        continue
                    if _embed_input_profile(dict(fresh))[:2] != profile:
                        continue
                    # vec0 rejects conflict clauses; the delete keeps the
                    # statement safe against a concurrent publish.
                    conn.execute(
                        "DELETE FROM subject_tags_vec WHERE id = ?", (memory_id,),
                    )
                    conn.execute(
                        "INSERT INTO subject_tags_vec(id, embedding) VALUES (?, ?)",
                        (memory_id, blob),
                    )
                    written += 1
        return written
