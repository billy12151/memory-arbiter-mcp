from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from contextvars import ContextVar
from typing import Any, Callable, cast

from .acl import CallerWorkspace, WorkspaceScope, forbidden_payload, memory_public_stub, raw_workspace, redacted_conflict_shell, visible_memory
from .arbitration import compare_memories
from .config import Settings
from .constants import (
    EMBEDDING_MAX_SECTION_CHARS,
    EMBEDDING_N_CTX,
    EMBEDDING_RESERVED_TOKENS,
    NOTICE_SYNC_WAIT_MS,
    QWEN_BUDGET_MS,
    QWEN_CANDIDATE_DISTANCE,
    QWEN_CANDIDATE_TOP_K,
    SEMANTIC_INFERENCE_TIMEOUT_MS,
    SEMANTIC_LOAD_TIMEOUT_MS,
    SEMANTIC_N_BATCH,
    SEMANTIC_N_CTX,
    SEMANTIC_N_THREADS,
    SEMANTIC_SCAN_BUDGET_MS,
    SEMANTIC_SCAN_ENHANCE,
    SEMANTIC_SCAN_MAX_PAIRS,
    WORKSPACE_MIN_NAME_LEN,
    WORKSPACE_RECALL_ADMISSION,
    WORKSPACE_RECALL_CUTOFF,
    strict_ws,
)
from .db import MemoryDB
from .embedder import ManagedEmbedder
from .text import canon_entity as _canon_entity, canon_scope as _canon_scope
from .models import MemoryRecord, MemoryStatus, ProtectionLevel, SourceType, TrustedApplyingContext
from .search import search_memories, _linked_open_items_for_search
from .semantic_conflict import (
    IsolatedGGUFSemanticBackend,
    SemanticBackend,
)
from .update_monitor import UpdateMonitor
from .request_identity import get_request_identity
from . import __version__
from . import workspace_rules
from .workers import LocalTextIndexWorker, SemanticConflictWorker
from .surfaces import ProductSurfaces
from .pipeline.signals import ConflictSignalPipeline
from .pipeline.write import WritePipeline
from .pipeline.read import ReadPipeline
from .pipeline.operations import OperationsPipeline
from .pipeline.evidence import EvidencePipeline


# Backup-replay notice signatures compared in _consume_notices. EMPTY means
# the replay inbox holds no pending work (and resets the dedup state).
# DEGRADED is emitted when inspection itself raises: the -1 sentinels mark
# the counts as unknown (real counts are always >= 0) with has_more forced
# True so the degraded signature can never collide with the empty one.
_BACKUP_NOTICE_EMPTY_SIGNATURE = (0, 0, 0, False)
_BACKUP_NOTICE_DEGRADED_SIGNATURE = (-1, -1, -1, True)


class MemoryTools:
    def __init__(self, settings: Settings | None = None, db: MemoryDB | None = None):
        self.settings = settings or Settings.from_env()
        self.db = db or MemoryDB(self.settings)
        self._embedder: ManagedEmbedder | None = None
        self._embedder_loaded = False
        self._embedder_lock = threading.Lock()
        # Config-time warnings (removed env vars, deprecated file keys) are NOT
        # seeded into read/search responses — they are surfaced by doctor,
        # console settings, and memory status instead (0.15.0 behavior change).
        self._embedder_warnings: list[str] = []
        self._update_monitor: UpdateMonitor | None = None
        self._evidence_worker = LocalTextIndexWorker(self)
        self._surfaces = ProductSurfaces(self)
        self._signals = ConflictSignalPipeline(self)
        self._write_pipeline = WritePipeline(self)
        self._read_pipeline = ReadPipeline(self)
        self._operations = OperationsPipeline(self)
        self._evidence = EvidencePipeline(self)
        self._semantic_backend: SemanticBackend | None = None
        self._semantic_backend_lock = threading.Lock()
        self._semantic_runtime_disabled = False
        self._semantic_worker = SemanticConflictWorker(self)
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_complete = False
        # Per-product-call caller cache. ContextVar isolates concurrent MCP
        # tasks; the product wrapper clears it before dispatch and automatic
        # notice delivery reuses the scope computed by the operation.
        self._product_caller: ContextVar[CallerWorkspace | None] = ContextVar(
            "memory_arbiter_product_caller", default=None,
        )
        self._last_pair_duration_ms: int | None = None
        # Spec §7: check-route fail-closed degradation must stay observable
        # (qwen unavailable / per-job budget exhausted), not silently skipped.
        self._check_degradation_reason: str | None = None
        self._check_degradation_count = 0
        self._check_degradation_at: str | None = None
        # Last few offending raw model outputs (whitespace-folded, truncated,
        # each tagged with its reason/timestamp), so a degradation spike is
        # debuggable from status instead of guesswork. Samples can embed quote
        # fragments from the checked pair — process-global diagnostics, same
        # single-trust-domain exposure as the rest of semantic status.
        self._check_degradation_samples: list[dict[str, str]] = []
        self._notice_claim_error_count = 0
        self._notice_claim_last_error: str | None = None
        self._notice_claim_last_error_at: str | None = None
        self._last_backup_notice_signature: tuple[int, int, int, bool] | None = None
        self._last_backup_source_signature: tuple[int, int, int] | None = None

    def start_update_monitor(self, monitor: UpdateMonitor | None = None) -> None:
        # Product notice delivery is owned by the four outer product wrappers,
        # not DegradeState.response(): nested responses must never consume it.
        self.db.state.notice_provider = None
        try:
            self._update_monitor = monitor or UpdateMonitor(enabled=self.settings.update_check_enabled)
        except Exception:
            self._update_monitor = None
            return
        try:
            self._update_monitor.maybe_start_check_if_due()
        except Exception:
            pass

    def start_evidence_worker(self) -> None:
        self._evidence_worker.start()

    def wait_evidence_worker_drained(self, timeout: float = 30.0) -> bool:
        return self._evidence_worker.wait_drained(timeout)

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
                "check-route candidates are fail-closed (no notice) while Qwen "
                "is unavailable (qwen_unavailable/qwen_backend_error), times out "
                "(qwen_timeout), returns invalid output (qwen_invalid_output), "
                "or the job budget is exhausted. While any of those hold, the "
                "write-time check route creates no notices at all — including "
                "pairs the deterministic rules classified as notify — and "
                "recall is guaranteed only by scheduled scan. Semantic-worker "
                "queue overflow shows as worker.dropped_queue_full."
            ),
        }

    def _enqueue_local_text_index(
        self, memory_id: int, record: dict[str, Any] | None = None,
        *, trusted_applying_context: TrustedApplyingContext | None = None,
    ) -> dict[str, Any]:
        current = record or self.db.get_memory(int(memory_id)) or {}
        version = int(current.get("version") or 1)
        task_id = f"semantic:{int(memory_id)}@{version}"
        self._semantic_worker.reserve(task_id)
        snapshot: dict[str, Any] = {"version": version, "task_id": task_id}
        if trusted_applying_context is not None:
            snapshot["trusted_applying_context"] = trusted_applying_context.to_dict()
        result = self._evidence_worker.enqueue(int(memory_id), snapshot)
        if result.get("status") != "queued":
            self._semantic_worker.complete(
                task_id,
                {"status": "incomplete", "reason": f"evidence_index_{result.get('status') or 'rejected'}", "notices_created": 0},
            )
        return {**result, "semantic_task_id": task_id, "semantic_dedupe_key": task_id}

    def _enqueue_content_postcommit(
        self, memory_id: int, record: dict[str, Any] | None = None,
        *, trusted_applying_context: TrustedApplyingContext | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        index = self._enqueue_local_text_index(
            memory_id, record, trusted_applying_context=trusted_applying_context,
        )
        task_id = index.get("semantic_task_id")
        wait_ms = max(0, NOTICE_SYNC_WAIT_MS)
        can_check = bool(self._embedding_configured()) and self.settings.semantic_conflict_on_write != "off"
        completed = (
            self._semantic_worker.wait_task(str(task_id), wait_ms / 1000.0)
            if can_check and task_id and wait_ms > 0 else None
        )
        if completed is not None:
            return index, completed
        status = "deferred" if not can_check else "async"
        check: dict[str, Any] = {"status": status, "task_id": task_id, "dedupe_key": task_id}
        if not can_check:
            check["reason"] = "waiting_for_evidence_index"
        return index, check

    def _post_commit(
        self, memory_id: int, record: dict[str, Any] | None = None,
        *, recheck_conflicts: bool,
        trusted_applying_context: TrustedApplyingContext | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Single write-path post-commit entry: index always, recheck explicitly.

        Every writer states whether the semantic-conflict check re-enters for
        this write (recheck_conflicts) instead of each call site hand-picking
        between the two enqueue helpers; the trusted context is only valid on
        the apply flow's own committed edits (§15.3). Returns
        (evidence_index, semantic_conflict_check); the check slot is
        skipped/recheck_disabled when the writer opted out.
        """
        if not recheck_conflicts:
            return (
                self._enqueue_local_text_index(memory_id, record),
                {"status": "skipped", "reason": "recheck_disabled"},
            )
        return self._enqueue_content_postcommit(
            memory_id, record, trusted_applying_context=trusted_applying_context,
        )

    def start_semantic_worker(self) -> None:
        self._semantic_worker.start()

    def current_client(self) -> str | None:
        # Trusted identity comes only from the request scope (HTTP headers or
        # the stdio process identity established in server.build_runtime).
        # Never fall back to process-level settings here: a missing trusted
        # source must surface as None, not silently claim the env identity.
        identity = get_request_identity()
        return identity.client if identity is not None else None

    def current_agent_id(self) -> str | None:
        identity = get_request_identity()
        return identity.agent_id if identity is not None else None

    def _consume_notices(self) -> list[dict[str, Any]]:
        notices: list[dict[str, Any]] = []
        if self._update_monitor is not None:
            notices.extend(self._update_monitor.consume_agent_onboarding_notice(self.current_agent_id()))
            notices.extend(self._update_monitor.consume_notices())
        try:
            source_signature = self.db.backup_replay.state_signature()
            if source_signature == self._last_backup_source_signature:
                return notices
            self._last_backup_source_signature = source_signature
            inspection = self.db.backup_replay.inspect(limit=10_000, offset=0)
            signature = (
                int(inspection.get("importable") or 0),
                int(inspection.get("invalid") or 0),
                int(inspection.get("conflicts") or 0),
                bool(inspection.get("has_more")),
            )
            if signature != _BACKUP_NOTICE_EMPTY_SIGNATURE and signature != self._last_backup_notice_signature:
                notices.append({
                    "type": "backup_replay_pending",
                    "severity": "warning",
                    "pending_records": signature[0],
                    "invalid_records": signature[1],
                    "conflicting_receipts": signature[2],
                    "additional_pages": signature[3],
                    "action_required": "preview_backup_replay",
                    "suggested_call": {
                        "tool": "memory_repair", "task": "replay_backup",
                        "data": {"dry_run": True},
                    },
                })
                self._last_backup_notice_signature = signature
            elif signature == _BACKUP_NOTICE_EMPTY_SIGNATURE:
                self._last_backup_notice_signature = None
        except Exception as exc:
            if self._last_backup_notice_signature != _BACKUP_NOTICE_DEGRADED_SIGNATURE:
                notices.append({
                    "type": "backup_replay_notice_degraded",
                    "severity": "warning",
                    "reason": str(exc),
                    "action_required": "inspect_backup_replay_manually",
                    "suggested_call": {
                        "tool": "memory_repair", "task": "replay_backup",
                        "data": {"dry_run": True, "limit": 200, "offset": 0},
                    },
                })
                self._last_backup_notice_signature = _BACKUP_NOTICE_DEGRADED_SIGNATURE
        return notices

    def wait_semantic_worker_drained(self, timeout: float = 30.0) -> bool:
        return self._semantic_worker.wait_drained(timeout)

    def _get_semantic_backend_ref(self) -> SemanticBackend | None:
        with self._semantic_backend_lock:
            return self._semantic_backend

    def shutdown(self, timeout: float = 30.0) -> dict[str, Any]:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return {"ok": True, "already_shutdown": True}
            if self._shutdown_started:
                return {"ok": False, "already_shutdown": False, "shutdown_in_progress": True}
            self._shutdown_started = True
        timeout = max(0.0, float(timeout))
        deadline = time.monotonic() + timeout
        worker_shutdown = self._semantic_worker.shutdown(discard_pending=True)
        evidence_shutdown = self._evidence_worker.shutdown(discard_pending=False)
        # Shutdown also closes synchronous workspace-suggestion admission before
        # waiting; otherwise a new call can race the worker drain/unload phase.
        with self._semantic_backend_lock:
            self._semantic_runtime_disabled = True
            admitted_backend = self._semantic_backend
            if admitted_backend is not None:
                admitted_backend.set_disabled(True)
        remaining = max(0.0, deadline - time.monotonic())
        semantic_drained = self._semantic_worker.wait_drained(remaining)
        remaining = max(0.0, deadline - time.monotonic())
        evidence_drained = self._evidence_worker.wait_drained(remaining)
        backend = self._get_semantic_backend_ref()
        unload_result: dict[str, Any] = {"ok": True, "unloaded": False, "reason": "no_backend"}
        if backend is not None:
            remaining = max(0.0, deadline - time.monotonic())
            unload_result = backend.unload(timeout=remaining, disable=True)
            if not unload_result.get("ok"):
                force_terminate = getattr(backend, "force_terminate", None)
                if callable(force_terminate):
                    unload_result = force_terminate()
        ok = bool(semantic_drained and evidence_drained and unload_result.get("ok", False))
        with self._shutdown_lock:
            self._shutdown_complete = ok
            self._shutdown_started = False
        return {
            "ok": ok,
            "already_shutdown": False,
            "semantic_worker": worker_shutdown,
            "evidence_worker": evidence_shutdown,
            "semantic_drained": semantic_drained,
            "evidence_drained": evidence_drained,
            "backend_unload": unload_result,
        }

    def _allowed(self, agent_id: str | None = None, client: str | None = None) -> tuple[bool, list[str]]:
        actual_agent = agent_id or self.current_agent_id()
        actual_client = client or self.current_client()
        if self.settings.policy.enabled_for(actual_client, actual_agent):
            return True, []
        return False, [f"Memory arbiter disabled by policy for client={actual_client}, agent_id={actual_agent}."]

    @staticmethod
    def _payload_dict(data: dict[str, Any] | None) -> dict[str, Any]:
        return dict(data) if isinstance(data, dict) else {}

    def _judge_constraints(self) -> dict[str, Any]:
        return self._surfaces._judge_constraints()

    def _product_help(self, surface: str, topic: str | None = None) -> dict[str, Any]:
        return self._surfaces._product_help(surface, topic)

    def _invalid_product_call(self, surface: str, message: str, topic: str | None = None) -> dict[str, Any]:
        return self._surfaces._invalid_product_call(surface, message, topic)

    @staticmethod
    def _help_topic(payload: dict[str, Any], fallback_key: str) -> str | None:
        return ProductSurfaces._help_topic(payload, fallback_key)

    def _forward(
        self, surface: str, topic: str | None, fn: Callable[..., dict[str, Any]], **payload: Any,
    ) -> dict[str, Any]:
        return self._surfaces._forward(surface, topic, fn, **payload)

    @staticmethod
    def _alias_id(payload: dict[str, Any], target: str) -> None:
        return ProductSurfaces._alias_id(payload, target)

    def _int_product_arg(
        self, surface: str, value: Any, name: str, topic: str | None = None,
    ) -> int | dict[str, Any] | None:
        return self._surfaces._int_product_arg(surface, value, name, topic)

    def _require_id(
        self, surface: str, payload: dict[str, Any], name: str, topic: str | None = None,
    ) -> dict[str, Any] | None:
        return self._surfaces._require_id(surface, payload, name, topic)

    def _coerce_product_id(
        self, surface: str, payload: dict[str, Any], name: str, topic: str | None = None,
    ) -> dict[str, Any] | None:
        return self._surfaces._coerce_product_id(surface, payload, name, topic)

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        return ProductSurfaces._is_truthy(value)

    def _require_ws_strings(
        self, payload: dict[str, Any], names: tuple[str, ...], surface: str, topic: str | None = None,
    ) -> dict[str, Any] | None:
        return self._surfaces._require_ws_strings(payload, names, surface, topic)

    def memory(self, action: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory(action, data, **_)

    def memory_review(self, view: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory_review(view, data, **_)

    def memory_govern(self, action: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory_govern(action, data, **_)

    def memory_repair(self, task: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory_repair(task, data, **_)

    def _embedding_configured(self) -> bool:
        # Pointing at a GGUF model IS the intent to embed (no provider or
        # vec.enabled knob since 0.15.0).
        return self.settings.embedding_model_path is not None

    def _index_local_text_evidence(self, memory_id: int, record: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._evidence.index_memory(memory_id, record)

    def _ensure_embedder(self) -> tuple[ManagedEmbedder | None, list[str]]:
        if self._embedder_loaded:
            return self._embedder, []
        with self._embedder_lock:
            if self._embedder_loaded:
                return self._embedder, []
            if not self._embedding_configured():
                self._embedder_loaded = True  # deterministic config state; safe to cache
                return None, []
            from .embedder import build_embedder

            assert self.settings.embedding_model_path is not None
            embedder, warnings = build_embedder(
                str(self.settings.embedding_model_path),
                n_ctx=EMBEDDING_N_CTX,
                reserved_tokens=EMBEDDING_RESERVED_TOKENS,
                max_section_chars=EMBEDDING_MAX_SECTION_CHARS,
            )
            self._embedder_warnings.extend(warnings)
            if embedder is None:
                # Build failed (missing model / load error). Do NOT cache —
                # a later retry (e.g. model installed) should still be able to succeed.
                return None, warnings
            self._embedder = embedder
            self._embedder_loaded = True  # cache only on successful build
            try:
                # Lazy vec-table creation: the derived vec0 tables are built
                # here (first successful embedder load), not at schema init,
                # with the dim the model itself reported. The same dim is
                # recorded in _vec_index_meta as the library's active dim.
                table_warnings = self.db.ensure_vec_tables(embedder.dim)
                self._embedder_warnings.extend(table_warnings)
                warnings.extend(table_warnings)
                self.db.init_vec_index_state(
                    embedder.embedding_space_id, True, active_dim=embedder.dim,
                )
            except Exception as exc:
                warning = f"vector space state initialization failed: {exc}"
                self._embedder_warnings.append(warning)
                warnings.append(warning)
            return self._embedder, warnings

    def _ensure_active_embedder(self) -> tuple[ManagedEmbedder | None, list[str]]:
        """Load the configured embedder, but expose it only for a ready index."""
        embedder, warnings = self._ensure_embedder()
        if embedder is None:
            return None, warnings
        state = self.db.get_vec_index_state().get("state")
        if state in {"mismatch", "failed"}:
            reason = (
                "embedding_space_mismatch" if state == "mismatch"
                else "embedding_migration_failed"
            )
            warning = f"vec_disabled={reason}"
            if warning not in warnings:
                warnings.append(warning)
            return None, warnings
        return embedder, warnings

    def _caller_workspace(self, explicit_workspace: str | None = None) -> CallerWorkspace:
        """Resolve the caller workspace for strict read ACLs.

        Explicit payload/query/console workspace wins. Without one, fall back to
        settings.workspace and surface the source/warning in responses.
        """
        isolation = getattr(self.settings, "isolation", "none")
        explicit = str(explicit_workspace or "").strip()
        if explicit:
            source = "explicit"
            workspace = explicit
        else:
            source = "settings"
            workspace = str(getattr(self.settings, "workspace", "") or "").strip()
        warnings: list[str] = []
        canonical: str | None = None
        admitted: tuple[str, ...] = ()
        if workspace:
            # Explicit filters are canonicalized in every isolation mode. In
            # none an explicit filter scopes the query (canonicalize-then-
            # filter, spec §15.6); it is never an ACL boundary — an omitted
            # workspace still spans all workspaces.
            embedder, ensure_warnings = self._ensure_active_embedder()
            warnings.extend(ensure_warnings)
            try:
                resolved = self.db.resolve_workspace_canonical(workspace, embedder, register_new=False)
                canonical = str(resolved.get("canonical") or workspace)
            except Exception:
                canonical = workspace
            # under strict isolation the readable set is the caller's
            # own canonical PLUS any within the recall cutoff (vector
            # admission). Off / degraded → (canonical,), i.e. the single-canonical
            # scope. Only strict consults it; none/weak never hard-scope by it.
            if isolation == "strict" and canonical:
                if WORKSPACE_RECALL_ADMISSION:
                    try:
                        admitted = self.db.workspaces.admitted_canonicals(
                            canonical,
                            cutoff=WORKSPACE_RECALL_CUTOFF,
                            min_name_len=WORKSPACE_MIN_NAME_LEN,
                        )
                    except Exception:
                        admitted = (canonical,)
                else:
                    admitted = (canonical,)
        elif isolation == "strict":
            warnings.append("isolation=strict has no caller workspace; read denied")
        if isolation == "strict" and source == "settings":
            warnings.append(f"strict read ACL using settings.workspace={workspace or '<empty>'!r}")
        caller = CallerWorkspace(
            isolation=isolation,
            workspace=workspace or None,
            canonical=canonical,
            source=source,
            warnings=tuple(warnings),
            admitted=admitted,
        )
        self._product_caller.set(caller)
        return caller

    def _strict_acl_unavailable(self, caller: CallerWorkspace) -> dict[str, Any] | None:
        if caller.isolation == "strict" and not caller.canonical:
            return self.db.state.response(
                forbidden_payload("workspace", workspace=caller, reason="missing_caller_workspace"),
                ok=False,
                extra_warnings=list(caller.warnings),
            )
        return None

    def _get_memory_visible(self, memory_id: int, caller: CallerWorkspace | None = None) -> dict[str, Any] | None:
        caller = caller or self._caller_workspace(None)
        if caller.isolation == "strict":
            if not caller.canonical:
                return None
            return self.db.get_memory_for_workspace(
                int(memory_id), caller.canonical, caller.scope_canonicals(),
            )
        return self.db.get_memory(int(memory_id))

    def _memory_acl_response_fields(self, caller: CallerWorkspace) -> dict[str, Any]:
        return caller.response_fields() if caller.isolation == "strict" else {}

    def _strict_filter_records(self, records: list[dict[str, Any]], caller: CallerWorkspace) -> list[dict[str, Any]]:
        if caller.isolation != "strict" or not caller.canonical:
            return records
        allowed = {str(a or "").strip() for a in caller.scope_canonicals() if str(a or "").strip()}
        return [r for r in records if raw_workspace(r) in allowed]

    @staticmethod
    def _conflict_next_call(
        conflict: dict[str, Any], workspace: str | None = None,
    ) -> dict[str, Any] | None:
        conflict_id = int(conflict["id"])
        revision = int(conflict["revision"])
        status = conflict.get("status")

        def data(**values: Any) -> dict[str, Any]:
            payload = dict(values)
            if workspace:
                payload["workspace"] = workspace
            return payload

        if status == "open":
            return {
                "tool": "memory", "action": "judge",
                "data": data(conflict_id=conflict_id, expected_revision=revision),
            }
        if status == "applying":
            plan = (conflict.get("apply_summary") or {}).get("plan") or []
            pending = next((item for item in plan if item.get("status") == "pending"), None)
            if pending is not None:
                return {
                    "tool": "memory_govern", "action": "apply_conflict_action",
                    "data": data(
                        conflict_id=conflict_id, expected_revision=revision,
                        memory_id=pending.get("memory_id"), action=pending.get("action"),
                    ),
                    "authorization_required": True,
                }
            if any(item.get("status") not in {"pending", "completed"} for item in plan):
                return {
                    "tool": "memory_govern", "action": "replan_conflict",
                    "data": data(conflict_id=conflict_id, expected_revision=revision),
                    "authorization_required": True,
                }
            return {
                "tool": "memory_govern", "action": "resolve_conflict",
                "data": data(conflict_id=conflict_id, expected_revision=revision),
                "authorization_required": True,
            }
        return None

    def _conflict_detail_for_workspace(self, conflict_id: int, caller: CallerWorkspace | None = None) -> dict[str, Any] | None:
        """Return group detail only when every member passes strict ACL."""
        caller = caller or self._caller_workspace(None)
        conflict = self.db.get_conflict(int(conflict_id))
        if conflict is None:
            return None
        if (
            caller.isolation == "none" and caller.source == "explicit"
            and str(conflict.get("workspace_canonical") or "") != str(caller.canonical or "")
        ):
            return None
        member_ids = sorted({int(member["memory_id"]) for member in conflict.get("member_versions") or []})
        resolution_id = conflict.get("resolution_memory_id")
        lookup_ids = member_ids + ([int(resolution_id)] if resolution_id is not None else [])
        memories = {
            memory_id: memory
            for memory_id in lookup_ids
            if (memory := self.db.get_memory(memory_id)) is not None
        }
        if caller.isolation == "strict":
            visible = {
                memory_id: visible_memory(
                    memories.get(memory_id), caller.canonical, caller.scope_canonicals(),
                )
                for memory_id in lookup_ids
            }
            visible_member_count = sum(bool(visible.get(memory_id, False)) for memory_id in member_ids)
            if not member_ids or visible_member_count == 0:
                return None
            # Strict callers either see the complete correlated snapshot or no
            # conflict at all. Even a redacted shell leaks lifecycle/existence.
            if visible_member_count != len(member_ids):
                return None
            if resolution_id is not None and not visible.get(int(resolution_id), False):
                return None
        members = [
            memory_public_stub(memory_id, visible=True, memory=memories.get(memory_id))
            for memory_id in member_ids
        ]
        resolution = (
            memory_public_stub(resolution_id, visible=True, memory=memories.get(int(resolution_id)))
            if resolution_id is not None else None
        )
        detail = {
            "conflict": conflict,
            "revision": conflict.get("revision"),
            "slot": conflict.get("slot_key"),
            "member_versions": conflict.get("member_versions") or [],
            "value_groups": conflict.get("value_groups") or [],
            "members": members,
            "resolution_memory": resolution,
            "resolution_memory_version": conflict.get("resolution_memory_version"),
            "apply_summary": conflict.get("apply_summary") or {"plan": []},
            "next_executable_call": self._conflict_next_call(
                conflict,
                caller.workspace if caller.isolation == "strict" else None,
            ),
            "all_members_visible": True,
        }
        if caller.isolation == "strict":
            detail.update(caller.response_fields())
        return detail

    def _conflict_visible(self, conflict_id: int, caller: CallerWorkspace | None = None) -> bool:
        return self._conflict_detail_for_workspace(conflict_id, caller) is not None

    @staticmethod
    def _embedding_text(record: dict[str, Any]) -> str:
        subject = record.get("subject") or ""
        content = record.get("content") or ""
        return f"{subject}\n{content}".strip()

    def _semantic_configured(self) -> bool:
        # The backend is always the local GGUF engine now — the former
        # semantic_conflict.backend knob was dead configuration.
        return (
            bool(self.settings.semantic_conflict_enabled)
            and self.settings.semantic_conflict_model_path is not None
        )

    def _ensure_semantic_backend(self) -> SemanticBackend | None:
        if not self._semantic_configured():
            return None
        with self._semantic_backend_lock:
            if self._semantic_runtime_disabled:
                return None
            if self._semantic_backend is not None:
                return self._semantic_backend
            assert self.settings.semantic_conflict_model_path is not None
            self._semantic_backend = IsolatedGGUFSemanticBackend(
                self.settings.semantic_conflict_model_path,
                n_ctx=SEMANTIC_N_CTX,
                n_threads=SEMANTIC_N_THREADS,
                n_batch=SEMANTIC_N_BATCH,
                hard_timeout_ms=SEMANTIC_INFERENCE_TIMEOUT_MS,
                load_timeout_ms=SEMANTIC_LOAD_TIMEOUT_MS,
            )
            return self._semantic_backend

    def _suggest_workspace_candidate(
        self, ws_raw: str, evidence: dict[str, Any], similar: list[dict[str, Any]],
    ) -> Any:
        """Ask the local model to suggest a workspace normalization candidate.

        Returns a WorkspaceCandidateSignal, or None if no backend is configured
        (caller then falls back to ASK). Never raises — the backend degrades to
        an uncertain signal on any error (636 §6: suggester only, never arbiter).
        """
        backend = self._ensure_semantic_backend()
        if backend is None or not hasattr(backend, "suggest_workspace_candidate"):
            return None
        # Spec §11: Qwen only arbitrates among candidates the vector already
        # brought within range. Bounding the pool by distance stops the model
        # from resurrecting an over-distance name (a real-library dry-run had
        # Qwen "same_project@0.95" merge openclaw into proto-test at cosine
        # 0.357, far past the 0.25 threshold). Cap at top-K (A/B: 3 beats 5).
        candidates = [
            s["name"] for s in (similar or [])
            if s.get("name") and float(s.get("distance", 9.0)) <= QWEN_CANDIDATE_DISTANCE
        ][:QWEN_CANDIDATE_TOP_K]
        if not candidates:
            return None
        budget_ms = max(0, QWEN_BUDGET_MS)
        if budget_ms <= 0:
            return None
        try:
            suggestion = backend.suggest_workspace_candidate(
                ws_raw, evidence, candidates,
                deadline_monotonic=time.monotonic() + budget_ms / 1000.0,
            )
        except TypeError:
            # Compatibility for injected/test backends implementing the original
            # protocol. Production scheduling is deadline-aware below.
            suggestion = backend.suggest_workspace_candidate(ws_raw, evidence, candidates)
        except Exception:
            suggestion = None
        # Frozen resident=true: the model stays loaded, so there is no idle
        # unload path any more.
        return suggestion

    def _semantic_notice_workspace_scope(self, workspace: Any = None) -> "WorkspaceScope":
        """Use the shared read-only caller resolver for notice API/count scope.

        returns the admitted canonical set so strict notice reads widen
        with the same vector admission as search/conflict (off → single canonical).
        """
        if self.settings.isolation != "strict":
            return None
        return self._caller_workspace(workspace).scope_canonicals()

    @staticmethod
    def _scan_envelope(memory: dict[str, Any], quote: str) -> dict[str, Any]:
        metadata_value = memory.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, dict) else {}
        return {
            "quote": str(quote)[:1000], "subject": str(memory.get("subject") or "")[:200],
            "tags": list(memory.get("tags") or [])[:20],
            "workspace_canonical": memory.get("workspace_canonical") or memory.get("workspace"),
            "memory_id": int(memory.get("id") or 0), "version": int(memory.get("version") or 1),
            "event_time": memory.get("event_time"),
            "metadata": {key: metadata.get(key) for key in ("entity", "scope") if metadata.get(key)},
        }

    def _enhance_scan_candidates(self, result: dict[str, Any]) -> dict[str, Any]:
        """Spec §7.1 wide gate: bounded Qwen pair enhancement over one scan page.

        Deterministic rule candidates are enriched with extracted
        attribute/value member fields and value_groups; similarity-pool pairs
        whose extraction yields a legal same-attribute/different-value in
        either direction are unioned into the candidate list. Verified
        candidates whose memories agree on metadata entity/scope are
        aggregated by canonical slot. Fail-open by contract: backend absence,
        timeout, or invalid output leaves the deterministic candidate set
        unchanged and never removes a base candidate.
        """
        import json as _json

        from .semantic_conflict import evaluate_pair_extractions, normalize_value as _normalize_value, signal_extraction, PAIR_PROMPT_VERSION

        pool = result.pop("similarity_pool", None) or []
        max_pairs = max(0, SEMANTIC_SCAN_MAX_PAIRS)
        if not SEMANTIC_SCAN_ENHANCE or max_pairs <= 0:
            result["qwen_enhancement"] = {"status": "disabled", "similarity_pool_size": len(pool)}
            return result
        candidates = result.get("candidates") or []
        if not candidates and not pool:
            result["qwen_enhancement"] = {"status": "ok", "pairs_evaluated": 0, "enhanced": 0}
            return result
        backend = self._ensure_semantic_backend()
        if backend is None:
            result["qwen_enhancement"] = {"status": "skipped_unavailable", "similarity_pool_size": len(pool)}
            return result
        deadline = time.monotonic() + max(1.0, SEMANTIC_SCAN_BUDGET_MS / 1000.0)
        memory_cache: dict[int, dict[str, Any] | None] = {}

        def memory(mid: int) -> dict[str, Any] | None:
            if mid not in memory_cache:
                memory_cache[mid] = self.db.get_memory(int(mid))
            return memory_cache[mid]

        def classify(left_env: dict[str, Any], right_env: dict[str, Any]) -> Any:
            try:
                return backend.classify_pair(left_env, right_env, deadline_monotonic=deadline)
            except TypeError:
                # Test/legacy backends implementing the original two-arg protocol.
                return backend.classify_pair(left_env, right_env)

        state = {"evaluated": 0, "enhanced": 0}
        slot_groups: dict[str, dict[str, Any]] = {}

        def enhance(item: dict[str, Any]) -> str:
            left_mem = memory(int(item.get("left_id") or 0))
            right_mem = memory(int(item.get("right_id") or 0))
            if (
                not left_mem or not right_mem
                or left_mem.get("status") != "active" or right_mem.get("status") != "active"
            ):
                return "skipped_inactive"
            left_env = self._scan_envelope(left_mem, str(item.get("left_snippet") or ""))
            right_env = self._scan_envelope(right_mem, str(item.get("right_snippet") or ""))
            forward_signal = classify(left_env, right_env)
            reverse_signal = classify(right_env, left_env)
            state["evaluated"] += 1
            gate = evaluate_pair_extractions(
                signal_extraction(forward_signal), signal_extraction(reverse_signal),
                left_env, right_env, require_bidirectional=False,
            )
            item["qwen_signal"] = {
                "state": gate.state, "reason": gate.reason,
                "forward_type": forward_signal.candidate_type,
                "reverse_type": reverse_signal.candidate_type,
                "prompt_version": PAIR_PROMPT_VERSION,
            }
            positive = gate.state == "notice_ready" or gate.reason == "single_direction_only"
            if not positive:
                return gate.reason
            state["enhanced"] += 1
            forward_parsed = forward_signal.parsed if isinstance(forward_signal.parsed, dict) else {}
            reverse_parsed = reverse_signal.parsed if isinstance(reverse_signal.parsed, dict) else {}
            if forward_parsed:
                display_a, display_b = forward_parsed.get("value_a"), forward_parsed.get("value_b")
                attribute_raw = forward_parsed.get("attribute_a")
            else:
                display_a, display_b = reverse_parsed.get("value_b"), reverse_parsed.get("value_a")
                attribute_raw = reverse_parsed.get("attribute_b")
            # gate.value_a/value_b follow the surviving extraction's OWN input
            # order (reverse = B->A), but display_a/display_b and members[0/1]
            # are in left/right order. Normalize each side's own display value so
            # the stored value is grounded to that member (not its peer).
            norm_a = _normalize_value(str(display_a)) if display_a else gate.value_a
            norm_b = _normalize_value(str(display_b)) if display_b else gate.value_b
            members = item.get("members") or []
            for index, member in enumerate(members):
                member["attribute_raw"] = str(attribute_raw or gate.attribute)
                member["value_raw"] = str(display_a if index == 0 else display_b)
                member["normalized_attribute"] = gate.attribute
                member["normalized_value"] = norm_a if index == 0 else norm_b
                member["direction"] = "a_to_b" if forward_parsed else "b_to_a"
                member["prompt_version"] = PAIR_PROMPT_VERSION
            refs = [f"{int(m['memory_id'])}@{int(m['version'])}" for m in members]
            item["value_groups"] = [
                {"normalized_value": norm_a, "display_value": str(display_a or norm_a),
                 "members": [refs[0]] if refs else []},
                {"normalized_value": norm_b, "display_value": str(display_b or norm_b),
                 "members": [refs[1]] if len(refs) > 1 else []},
            ]
            if gate.state == "notice_ready":
                item["state"] = item["route"] = "notice_ready"
            left_raw_meta = left_mem.get("metadata")
            right_raw_meta = right_mem.get("metadata")
            left_meta: dict[str, Any] = left_raw_meta if isinstance(left_raw_meta, dict) else {}
            right_meta: dict[str, Any] = right_raw_meta if isinstance(right_raw_meta, dict) else {}
            entity = left_meta.get("entity") if left_meta.get("entity") == right_meta.get("entity") else None
            scope = left_meta.get("scope") if left_meta.get("scope") == right_meta.get("scope") else None
            if entity and scope:
                # B-C4: slot keys are built with canonicalised entity/scope
                # (the comparison-side counterpart of the storage-side canon
                # in db/conflicts.py _normalize_slot) so lexical variants like
                # "MyProject"/"myproject" aggregate into one slot group.
                slot_key = {
                    "entity": _canon_entity(entity), "attribute": gate.attribute,
                    "scope": _canon_scope(scope),
                }
                item["slot_key"] = slot_key
                item["slot_provenance"] = {
                    "entity": "metadata", "scope": "metadata",
                    "attribute": "bidirectional_extraction" if gate.state == "notice_ready"
                    else "single_direction_extraction",
                }
                slot_json = _json.dumps(slot_key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                group = slot_groups.setdefault(slot_json, {
                    "slot_key": slot_key, "members": {}, "value_groups": {},
                    "candidate_pairs": [], "value_conflict": False,
                })
                for index, member in enumerate(members):
                    ref = f"{int(member['memory_id'])}@{int(member['version'])}"
                    value = norm_a if index == 0 else norm_b
                    display = str(display_a if index == 0 else display_b) or value
                    prior = group["members"].get(ref)
                    if prior is not None and prior != value:
                        # The same memory extracted a different value in another
                        # same-slot pair: an un-recordable payload. Flag it for
                        # agent deep-read instead of emitting a bad group.
                        group["value_conflict"] = True
                    group["members"][ref] = value
                    entry = group["value_groups"].setdefault(
                        value, {"normalized_value": value, "display_value": display, "members": set()},
                    )
                    entry["members"].add(ref)
                group["candidate_pairs"].append([int(item["left_id"]), int(item["right_id"])])
            return "notice_ready" if gate.state == "notice_ready" else "single_direction_only"

        def enhance_guarded(item: dict[str, Any]) -> str:
            # Fail-open by contract: a raising backend must never abort the page
            # or discard the deterministic baseline candidate set.
            try:
                return enhance(item)
            except Exception:
                return "error"

        last_reason: str | None = None
        for item in candidates:
            if state["evaluated"] >= max_pairs or time.monotonic() >= deadline:
                break
            last_reason = enhance_guarded(item) or last_reason
        added: list[dict[str, Any]] = []
        for item in pool:
            if state["evaluated"] >= max_pairs or time.monotonic() >= deadline:
                break
            tag = enhance_guarded(item)
            last_reason = last_reason or tag
            if tag in {"notice_ready", "single_direction_only"}:
                added.append(item)
        if added:
            result["candidates"] = candidates + added
        counts = result.setdefault("counts", {})
        counts["qwen_union_added"] = len(added)
        counts["qwen_enhanced"] = state["enhanced"]
        result["qwen_enhancement"] = {
            "status": "ok",
            "pairs_evaluated": state["evaluated"],
            "enhanced": state["enhanced"],
            "budget_exhausted": state["evaluated"] >= max_pairs or time.monotonic() >= deadline,
            "last_reason": last_reason,
        }
        if slot_groups:
            result["slot_groups"] = [
                {
                    "slot_key": group["slot_key"],
                    "members": [
                        {"member": ref, "normalized_value": value}
                        for ref, value in sorted(group["members"].items())
                    ],
                    "value_groups": [
                        {**entry, "members": sorted(entry["members"])}
                        for entry in sorted(group["value_groups"].values(), key=lambda e: e["normalized_value"])
                    ],
                    "candidate_pairs": group["candidate_pairs"],
                    # A member with disagreeing values across same-slot pairs
                    # cannot be recorded as one group: hand it to deep-read.
                    "value_conflict": bool(group.get("value_conflict")),
                    "route": "review_candidate" if group.get("value_conflict") else "recordable",
                }
                for group in slot_groups.values()
            ]
        return result

    def _semantic_status(self, workspace_canonical: WorkspaceScope = None) -> dict[str, Any]:
        backend = self._get_semantic_backend_ref()
        backend_status = (
            backend.status()
            if backend is not None else
            {
                "model_path": str(self.settings.semantic_conflict_model_path or ""),
                "model_exists": bool(
                    self.settings.semantic_conflict_model_path
                    and self.settings.semantic_conflict_model_path.exists()
                ),
                "model_state": "unloaded",
                "last_error": None,
            }
        )
        return {
            "enabled": bool(self.settings.semantic_conflict_enabled),
            "configured": self._semantic_configured(),
            "on_write": self.settings.semantic_conflict_on_write,
            "max_concurrency": 1,
            "max_concurrency_note": "reserved; the semantic worker is single-threaded",
            "last_pair_duration_ms": self._last_pair_duration_ms,
            "check_degradation": self._check_degradation_status(),
            "job_deadline_behavior": (
                "The job budget activates only while another semantic job is queued and "
                "gates between pairs. An inference already in flight is governed only by "
                "the inference timeout; a timed-out child is terminated and the next request "
                "starts a new generation. Timeouts are frozen constants since 0.15.0 "
                "(memory_arbiter.constants)."
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

    def _semantic_control(self, action: str) -> dict[str, Any]:
        return self._semantic_control_with_timeout(action, timeout=30.0)

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

    def _enqueue_semantic_conflict_check(
        self, memory_id: int | None, record: Any, *, after_evidence: bool = False,
        trusted_applying_context: TrustedApplyingContext | None = None,
    ) -> dict[str, Any]:
        if memory_id is None:
            return {"status": "skipped", "reason": "backup_only"}
        if not after_evidence:
            return {"status":"deferred","reason":"waiting_for_evidence_index"}
        if self.settings.semantic_conflict_on_write == "off":
            return {"status": "off"}
        stored = self.db.get_memory(int(memory_id)) or {}
        content = (record.get("content") if isinstance(record, dict) else getattr(record, "content", None))
        if content is None:
            content = stored.get("content") or ""
        version = int(stored.get("version") or self.db.get_memory_version(int(memory_id)) or 1)
        task_id = f"semantic:{int(memory_id)}@{version}"
        snapshot = {
            "memory_id": int(memory_id),
            "version": version,
            "content_hash": hashlib.sha256(str(content or "").encode("utf-8")).hexdigest(),
            "task_id": task_id,
            "dedupe_key": task_id,
        }
        if trusted_applying_context is not None:
            snapshot["trusted_applying_context"] = trusted_applying_context.to_dict()
        return self._semantic_worker.enqueue(int(memory_id), snapshot)

    def _process_semantic_conflict_job(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        return self._evidence.process_conflicts(memory_id, snapshot)

    def memory_write(self, **payload: Any) -> dict[str, Any]:
        return self._write_pipeline.memory_write(**payload)

    def memory_search(self, query: str = "", workspace: str | None = None, tags: list[str] | None = None, limit: int = 10, offset: int = 0, debug_ranking: bool = False, query_embedding: list[float] | None = None, tags_filter: list[str] | None = None, after_time: str | None = None, before_time: str | None = None, source_type: str | None = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True, **_: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_search(
            query=query, workspace=workspace, tags=tags, limit=limit, offset=offset,
            debug_ranking=debug_ranking, query_embedding=query_embedding,
            tags_filter=tags_filter, after_time=after_time, before_time=before_time,
            source_type=source_type, include_linked_open_items=include_linked_open_items,
            include_conflict_signal=include_conflict_signal, **_,
        )

    def memory_search_expired(
        self,
        query: str = "",
        workspace: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        debug_ranking: bool = False,
        query_embedding: list[float] | None = None,
        tags_filter: list[str] | None = None,
        after_time: str | None = None,
        before_time: str | None = None,
        source_type: str | None = None,
        include_conflict_signal: bool = True,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        return self._read_pipeline.memory_search_expired(
            query=query, workspace=workspace, tags=tags, limit=limit,
            debug_ranking=debug_ranking, query_embedding=query_embedding,
            tags_filter=tags_filter, after_time=after_time, before_time=before_time,
            source_type=source_type, include_conflict_signal=include_conflict_signal,
            offset=offset, **_,
        )

    def memory_get(
        self,
        memory_id: int,
        sections: str = "none",
        section_ids: list[int] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._read_pipeline.memory_get(
            memory_id=memory_id, sections=sections, section_ids=section_ids, **_,
        )

    def memory_recent(self, workspace: str | None = None, limit: int = 20, **_: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_recent(workspace, limit, **_)

    def memory_compare(self, left_id: int | None = None, right_id: int | None = None, left: dict[str, Any] | None = None, right: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_compare(left_id, right_id, left, right, **_)

    def memory_arbitrate(self, left_id: int, right_id: int, mark_conflict: bool = True, authorized: bool = False, **_: Any) -> dict[str, Any]:
        return self._operations.memory_arbitrate(
            left_id, right_id, mark_conflict, self._is_truthy(authorized), **_,
        )

    def _with_resolution_guidance(self, conflict: dict[str, Any]) -> dict[str, Any]:
        return self._operations._with_resolution_guidance(conflict)

    def memory_list_conflicts(self, status: str = "open", limit: int = 50, source: str | None = None, **_: Any) -> dict[str, Any]:
        return self._operations.memory_list_conflicts(status, limit, source, **_)

    def memory_resolve_conflict(
        self, conflict_id: int, reason: str = "", status: str = "resolved", **_: Any,
    ) -> dict[str, Any]:
        resolve_conflict = cast(Callable[..., dict[str, Any]], self._operations.memory_resolve_conflict)
        return resolve_conflict(conflict_id, reason, status, **_)

    def memory_confirm(self, memory_id: int, source_ref: str | None = None, confidence: float = 1.0, authorized: bool = False, **_: Any) -> dict[str, Any]:
        return self._operations.memory_confirm(
            memory_id, source_ref, confidence, self._is_truthy(authorized), **_,
        )

    def memory_rename_workspace_canonical(
        self, old: str, new: str, reason: str | None = None, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_rename_workspace_canonical(old, new, reason, **_)

    def memory_migrate_workspace(
        self, reason: str | None = None, **payload: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_migrate_workspace(reason, **payload)

    def memory_move_memories_workspace(
        self,
        memory_ids: list[int] | None = None,
        new_workspace: str = "",
        reason: str | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_move_memories_workspace(
            memory_ids or [], new_workspace, reason, self._is_truthy(authorized), **_,
        )

    def memory_confirm_pending_workspace(
        self, memory_id: int, canonical: str, reason: str | None = None,
        authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_confirm_pending_workspace(
            memory_id, canonical, reason, self._is_truthy(authorized), **_,
        )

    def memory_confirm_workspaces(
        self,
        workspaces: list[str] | None = None,
        reason: str | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_confirm_workspaces(
            workspaces, reason, self._is_truthy(authorized), **_,
        )

    def memory_activate(
        self, memory_id: int, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_activate(memory_id, self._is_truthy(authorized), **_)

    def memory_supersede(
        self,
        memory_id: int,
        reason: str,
        superseded_by: int | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_supersede(
            memory_id, reason, superseded_by, self._is_truthy(authorized), **_,
        )

    def _update_check_status(self) -> dict[str, Any]:
        return self._operations._update_check_status()

    def memory_status(self, **_: Any) -> dict[str, Any]:
        return self._operations.memory_status(**_)

    def memory_doctor_overview(self, deep: bool = False, **_: Any) -> dict[str, Any]:
        return self._operations.memory_doctor_overview(deep, **_)

    def memory_set_entity(
        self, memory_id: int, entity: str | None = None, scope: str | None = None,
        clear: bool = False, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_set_entity(
            memory_id, entity, scope, clear, self._is_truthy(authorized), **_,
        )

    def memory_list_entities(
        self, limit: int = 50, include_unassigned: bool = True, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_list_entities(limit, include_unassigned, **_)

    def memory_rebuild_evidence(
        self, memory_ids: list[int] | None = None, dry_run: bool = True,
        batch_size: int = 50, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_rebuild_evidence(memory_ids, dry_run, batch_size, **_)

    def memory_audit_summary(self, **_: Any) -> dict[str, Any]:
        return self._operations.memory_audit_summary(**_)

    def memory_edit(
        self,
        memory_id: int,
        new_content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        reason: str = "",
        authorized: bool = False,
        tags_only: bool = False,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_edit(
            memory_id, new_content, old_text, new_text, new_subject, new_tags,
            reason, self._is_truthy(authorized), tags_only, add_tags, remove_tags, **_,
        )

    def memory_history(self, memory_id: int, **_: Any) -> dict[str, Any]:
        return self._operations.memory_history(memory_id, **_)

    def memory_cleanup_history(
        self,
        memory_id: int | None = None,
        older_than_days: int | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_cleanup_history(
            memory_id, older_than_days, self._is_truthy(authorized), **_,
        )

    def memory_replay_backup(
        self, dry_run: bool = True, authorized: bool = False,
        limit: int = 1_000, offset: int = 0, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_replay_backup(
            dry_run, self._is_truthy(authorized), limit, offset, **_,
        )

    def memory_normalize_workspaces(
        self, dry_run: bool = True, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        """Fold registered workspace spelling variants into first-seen canonicals."""
        dry_run = self._is_truthy(dry_run)
        if not dry_run and not self._is_truthy(authorized):
            # Same caller-confirmation gate as replay_backup: executing the
            # merge re-points memories and drops canonical rows, so it needs
            # explicit user authorization.
            return {
                "ok": False,
                "dry_run": False,
                "error": "authorized=True is required to execute workspace normalization",
                "action_required": "ask_user_for_authorization",
                "groups": [],
                "merged": [],
                "rejected_normalized": [],
                "skipped": [],
                "warnings": [],
            }
        return self.db.workspaces.normalize_workspace_canonicals(dry_run=dry_run)

    @staticmethod
    def _confidence_rank(hint: str | None) -> int:
        return ConflictSignalPipeline._confidence_rank(hint)

    def _attach_conflict_signals(
        self,
        results: list[dict[str, Any]],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        return self._signals._attach_conflict_signals(results, warnings)

    def _build_open_table_signal(
        self,
        memory_id: int,
        conflicts: list[dict[str, Any]],
        summaries: dict[int, dict[str, Any]],
        result_id_set: set[int],
    ) -> dict[str, Any] | None:
        return self._signals._build_open_table_signal(memory_id, conflicts, summaries, result_id_set)
