"""语义判定后端/状态/控制/遥测 mixin（从 tools.py 搬出，拆分批 ⑥ 纯移动）。"""
from __future__ import annotations

import math
import time
from typing import Any, TYPE_CHECKING
from .config import Settings
from .db import MemoryDB
from .scan_pipeline import ScanPipeline
from .acl import WorkspaceScope
from .constants import SEMANTIC_INFERENCE_TIMEOUT_MS, SEMANTIC_LOAD_TIMEOUT_MS, SEMANTIC_N_THREADS
from .semantic_conflict import SemanticBackend

if TYPE_CHECKING:
    from collections import deque
    from .acl import CallerWorkspace
    from .pipeline.evidence import EvidencePipeline
    from .semantic_conflict import SemanticBackend
    from .workers import SemanticConflictWorker
    from .config import Settings
    from .db import MemoryDB
    from .scan_pipeline import ScanPipeline

class _ToolsSemantic:
    if TYPE_CHECKING:
        db: "MemoryDB"
        settings: "Settings"
        _scan_pipeline: "ScanPipeline"
        _semantic_backend: "SemanticBackend | None"
        _semantic_backend_lock: "Any"
        _semantic_runtime_disabled: bool
        _semantic_worker: "SemanticConflictWorker"
        _evidence: "EvidencePipeline"
        _check_degradation_reason: "str | None"
        _check_degradation_count: int
        _check_degradation_at: "str | None"
        _check_degradation_samples: "list[dict[str, str]]"
        _notice_claim_error_count: int
        _notice_claim_last_error: "str | None"
        _notice_claim_last_error_at: "str | None"
        _pair_samples: "deque[dict[str, Any]]"
        def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace": ...
        def _get_semantic_backend_ref(self, *args: Any, **kwargs: Any) -> "SemanticBackend | None": ...

    def _record_check_degradation(self, reason: str, sample: str | None = None) -> None:
        from .models import utc_now_iso
        self._check_degradation_reason = str(reason)
        self._check_degradation_count += 1
        self._check_degradation_at = utc_now_iso()
        if sample:
            text = " ".join(str(sample).split())[:300]
            if text:
                # Reason/timestamp ride along so a stale sample is never
                # misread as belonging to the current last_reason.
                self._check_degradation_samples.append(
                    {"reason": str(reason), "at": self._check_degradation_at, "sample": text}
                )
                del self._check_degradation_samples[:-3]

    def _check_degradation_status(self) -> dict[str, Any]:
        return {
            "last_reason": self._check_degradation_reason,
            "count": self._check_degradation_count,
            "last_at": self._check_degradation_at,
            "recent_samples": [dict(item) for item in self._check_degradation_samples],
            "note": (
                "check-route candidates are fail-open while the judge is "
                "unavailable (judge_unavailable/judge_backend_error) or times "
                "out (judge_timeout); the check is truncated (rows_capped: the "
                "memory exceeds the 256-row conflict-channel cap, 0.17.0; "
                "pairs_examined_capped: the 500-pair examined "
                "cap; notice_budget_exhausted: the fair job deadline hit). "
                "Pairs beyond a truncation land in the conflict backlog "
                "(0.17.0, bounded and visible) or are covered by scheduled "
                "scan. Since 0.15.14 the former notice-count early stop is "
                "gone: every examined pair may surface its notice. Semantic-"
                "worker queue overflow shows as worker.dropped_queue_full."
            ),
        }

    def _record_pair_sample(self, *, pair_ms: int) -> None:
        """A1 ring: one sample per examined pair. The Qwen decode telemetry
        (prompt/generated tokens, retry flag) retired with the slot-extraction
        engine — the ring carries wall-clock pair duration only."""
        self._pair_samples.append({"pair_ms": int(pair_ms), "at": time.time()})

    def _pair_timing_summary(self) -> dict[str, Any]:
        """Aggregates over the recent ring: mean/p95 wall-clock pair duration."""
        samples = list(self._pair_samples)
        if not samples:
            return {"samples": 0}
        durations = sorted(int(item["pair_ms"]) for item in samples)
        return {
            "samples": len(samples),
            "mean_pair_ms": round(sum(durations) / len(durations)),
            "p95_pair_ms": durations[max(0, math.ceil(0.95 * len(durations)) - 1)],
        }

    def _semantic_configured(self) -> bool:
        # 0.17.1: the judge is the mDeBERTa checkpoint — configured → enabled.
        return (
            bool(self.settings.semantic_conflict_enabled)
            and self.settings.semantic_conflict_mdeberta_ckpt is not None
        )

    def _ensure_semantic_backend(self) -> SemanticBackend | None:
        if not self._semantic_configured():
            return None
        with self._semantic_backend_lock:
            if self._semantic_runtime_disabled:
                return None
            if self._semantic_backend is not None:
                return self._semantic_backend
            assert self.settings.semantic_conflict_mdeberta_ckpt is not None
            from .semantic_judge import IsolatedMDeBERTaBackend, device_default_batch

            configured_batch = int(self.settings.semantic_conflict_mdeberta_batch or 0)
            self._semantic_backend = IsolatedMDeBERTaBackend(
                self.settings.semantic_conflict_mdeberta_ckpt,
                self.settings.semantic_conflict_mdeberta_model_dir
                or (self.settings.semantic_conflict_mdeberta_ckpt.parent / "mdeberta-base"),
                batch_size=configured_batch if configured_batch > 0 else device_default_batch(),
                n_threads=SEMANTIC_N_THREADS,
                hard_timeout_ms=SEMANTIC_INFERENCE_TIMEOUT_MS,
                load_timeout_ms=SEMANTIC_LOAD_TIMEOUT_MS,
            )
            return self._semantic_backend

    # 0.17.1 (owner 拍板): _suggest_workspace_candidate retired — the model
    # judge has no suggester; workspace normalization falls back to the
    # existing ASK path (write.py), keeping the human-in-the-loop design.

    def _semantic_notice_workspace_scope(self, workspace: Any = None) -> "WorkspaceScope":
        """Use the shared read-only caller resolver for notice API/count scope.

        returns the admitted canonical set so strict notice reads widen
        with the same vector admission as search/conflict (off → single canonical).
        """
        if self.settings.isolation != "strict":
            return None
        return self._caller_workspace(workspace).scope_canonicals()

    def _lightweight_scan_candidates(self, result: dict[str, Any]) -> dict[str, Any]:
        return self._scan_pipeline._lightweight_scan_candidates(result)

    def memory_scan_workspace_anomalies(self, **_: Any) -> dict[str, Any]:
        return self._scan_pipeline.memory_scan_workspace_anomalies(**_)

    def _semantic_status(self, workspace_canonical: WorkspaceScope = None) -> dict[str, Any]:
        backend = self._get_semantic_backend_ref()
        ckpt = self.settings.semantic_conflict_mdeberta_ckpt
        backend_status: dict[str, Any] = (
            backend.status()
            if backend is not None else
            {
                "ckpt": str(ckpt or ""),
                "ckpt_sha8": None,
                "model_dir": str(self.settings.semantic_conflict_mdeberta_model_dir or ""),
                "model_version": None,
                "device": "cpu",
                "batch_size": int(self.settings.semantic_conflict_mdeberta_batch),
                "last_error": None,
            }
        )
        backend_status.setdefault("notice_min_prob", float(self.settings.semantic_conflict_mdeberta_notice_min_prob))
        return {
            "engine": "mdeberta",
            "enabled": bool(self.settings.semantic_conflict_enabled),
            "configured": self._semantic_configured(),
            "on_write": self.settings.semantic_conflict_on_write,
            # Effective write-response wait (semantic_conflict.notice_sync_wait_ms,
            # default 3000; 0 = batch mode, never block the write response).
            "notice_sync_wait_ms": int(self.settings.semantic_conflict_notice_sync_wait_ms),
            "max_concurrency": 1,
            "max_concurrency_note": "reserved; the semantic worker is single-threaded",
            "last_pair_duration_ms": (
                int(self._pair_samples[-1]["pair_ms"]) if self._pair_samples else None
            ),
            "pair_timing": self._pair_timing_summary(),
            "check_degradation": self._check_degradation_status(),
            "job_deadline_behavior": (
                "The job budget activates only while another semantic job is queued and "
                "gates between pairs. An inference already in flight is governed only by "
                "the inference timeout; a timed-out child is terminated and the next request "
                "starts a new generation. Job/inference/load timeouts are frozen constants "
                "since 0.15.0 (memory_arbiter.constants); the write-response wait is "
                "configurable again since 0.15.8 (semantic_conflict.notice_sync_wait_ms, "
                "default 3000, 0 = never block the write response)."
            ),
            "worker": self._semantic_worker.status(),
            "backend": backend_status,
            "notices": self.db.semantic_notice_counts(workspace_canonical),
            "notice_delivery": {
                "claim_error_count": self._notice_claim_error_count,
                "last_claim_error": self._notice_claim_last_error,
                "last_claim_error_at": self._notice_claim_last_error_at,
            },
        }

    def _semantic_control_with_timeout(
        self, action: str, timeout: float = 30.0, workspace: Any = None,
    ) -> dict[str, Any]:
        action = str(action or "status").strip().lower()
        timeout = max(0.0, float(timeout))
        notice_scope = self._semantic_notice_workspace_scope(workspace)
        if action == "status":
            return self._semantic_status(notice_scope)
        if action == "pause":
            self._semantic_worker.pause()
            return {"outcome": "paused", "semantic_conflict": self._semantic_status(notice_scope)}
        if action == "resume":
            worker_state = self._semantic_worker.status().get("runtime_state")
            if worker_state == "disabled":
                return {
                    "outcome": "runtime_disabled_use_enable",
                    "semantic_conflict": self._semantic_status(notice_scope),
                }
            self._semantic_worker.resume()
            return {"outcome": "resumed", "semantic_conflict": self._semantic_status(notice_scope)}
        if action == "enable":
            with self._semantic_backend_lock:
                self._semantic_runtime_disabled = False
                backend = self._semantic_backend
                if backend is not None:
                    backend.set_disabled(False)
            self._semantic_worker.enable_runtime()
            return {"outcome": "enabled", "semantic_conflict": self._semantic_status(notice_scope)}
        if action == "unload":
            backend = self._get_semantic_backend_ref()
            unload_result = (
                backend.unload(timeout=timeout, disable=False)
                if backend is not None else
                {"ok": True, "unloaded": False, "timeout": False, "inflight": 0, "retry_hint": None, "generation": None, "reason": "no_backend"}
            )
            outcome = "unloaded" if unload_result.get("ok") else "unload_timeout"
            result: dict[str, Any] = {"outcome": outcome, "unload": unload_result, "semantic_conflict": self._semantic_status(notice_scope)}
            if unload_result.get("timeout"):
                result["warnings"] = ["semantic backend still has in-flight inference; model was not unloaded"]
            return result
        if action == "disable":
            # Close both admissions before waiting for an in-flight request. The
            # backend gate covers synchronous workspace suggestions, which do not
            # pass through the semantic worker queue.
            self._semantic_worker.disable_runtime()
            with self._semantic_backend_lock:
                self._semantic_runtime_disabled = True
                backend = self._semantic_backend
                if backend is not None:
                    backend.set_disabled(True)
            unload_result = (
                backend.unload(timeout=timeout, disable=True)
                if backend is not None else
                {"ok": True, "unloaded": False, "timeout": False, "inflight": 0, "retry_hint": None, "generation": None, "reason": "no_backend"}
            )
            outcome = "runtime_disabled" if unload_result.get("ok") else "runtime_disabled_unload_timeout"
            disable_result: dict[str, Any] = {
                "outcome": outcome,
                "unload": unload_result,
                "note": "This disables the current runtime only; set semantic_conflict.enabled=false in config to persist it.",
                "semantic_conflict": self._semantic_status(notice_scope),
            }
            if unload_result.get("timeout"):
                disable_result["warnings"] = ["semantic backend disabled for new jobs, but current inference is still in flight"]
            return disable_result
        return {"outcome": "invalid_action", "valid_actions": ["status", "pause", "resume", "enable", "unload", "disable"]}

    def _process_semantic_conflict_job(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        # C2 (0.17.0 worker merge): the job IS the indexer now. index_only
        # snapshots (conflict-apply edits §15.3, replay postprocess, the
        # global off switch — stamped at enqueue time) run the index phase
        # only: segment + batch-embed + publish, no detection, receipt says so.
        if snapshot.get("index_only"):
            return self._evidence.index_rows_in_job(memory_id, snapshot)
        # 0.17.0 Q1 相分裂 (owner plan §3.1, D1/D7)：确定性相 → internal
        # 判定相 → A-cross 派发相，共享 job 全局判定预算池（ctx["budget"]：
        # internal 保护帽 ≤3 → A-cross 余量派发，耗尽 continue 不 break）。
        # 0.17.1：claims 通道 B/C 退役，相序只剩两段。
        ev = self._evidence
        ctx = ev.conflicts_deterministic_phase(memory_id, snapshot)
        terminal: "dict[str, Any] | None" = ctx.get("terminal")
        if terminal is not None and terminal.get("status") == "skipped":
            return terminal
        # 0.17.1 重标合一 (owner 拍板 2026-10-03，方案 §2.2)：确定性相 →
        # 合一判定相（internal keepers 与 A-cross 一批发出，单一总池
        # internal-first 排序，internal 帽撤销）。0.17.1：claims 通道 B/C
        # 退役，冲突检测回归 A-cross 句子对 + internal + 确定性直出。
        if terminal is None:
            # R1-2: a deterministic-phase truncation terminal skips the
            # judge phase entirely (the pre-split semantics).
            ev.conflicts_judge_phase(ctx)
            result = ev.conflicts_finalize_receipt(ctx)
        else:
            result = terminal
        # 0.17.1 owner A-4: job-level top-5 — conflict(normal) + possible(info)
        # notices this write produced compete in ONE ranked pool; the tail is
        # demoted to info (visible in the judgment page, out of the feed).
        if terminal is None:
            ev.conflicts_job_level_notice_cap(ctx)
        # r2s-08: the receipt tail (judge_budget/pairs_examined/elapsed_ms) is
        # stamped in ONE place — the truncation-terminal branch previously
        # re-assembled it by hand and forgot elapsed_ms.
        result = ev.conflicts_receipt_tail(ctx, result)
        return result
