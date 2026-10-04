from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict, deque
from contextvars import ContextVar
from typing import Any

from .acl import CallerWorkspace, forbidden_payload, memory_public_stub, raw_workspace, visible_memory
from .tools_forwards import _ToolsForwards
from .tools_lifecycle import _ToolsLifecycle
from .tools_semantic import _ToolsSemantic
from .tools_notices import _ToolsNotices
from .tools_governance import _ToolsGovern  # noqa: F401 (monkeypatch seam, see pipeline/read.py:226)
from .config import Settings
from .constants import EMBEDDING_MAX_SECTION_CHARS, EMBEDDING_N_CTX, EMBEDDING_RESERVED_TOKENS, SEMANTIC_PAIR_RING_SIZE, WORKSPACE_MIN_NAME_LEN, WORKSPACE_RECALL_ADMISSION, WORKSPACE_RECALL_CUTOFF
from .db import MemoryDB
from .embedder import ManagedEmbedder
from .models import TrustedApplyingContext
from .arbitration import compare_memories  # noqa: F401 (monkeypatch seam, see pipeline/operations.py)
from .search import search_memories, _linked_open_items_for_search  # noqa: F401 (monkeypatch seam, see pipeline/read.py:226)
from .semantic_conflict import (
    SemanticBackend,
)
from .update_monitor import UpdateMonitor
from .request_identity import get_request_identity
from .workers import SemanticConflictWorker
from .surfaces import ProductSurfaces
from .pipeline.signals import ConflictSignalPipeline
from .pipeline.write import WritePipeline
from .pipeline.read import ReadPipeline
from .pipeline.operations import OperationsPipeline
from .pipeline.evidence import EvidencePipeline
from .scan_pipeline import ScanPipeline
from .queue_protocol import QueueProtocol


# Backup-replay notice signatures compared in _consume_notices. EMPTY means
# the replay inbox holds no pending work (and resets the dedup state).
# DEGRADED is emitted when inspection itself raises: the -1 sentinels mark
# the counts as unknown (real counts are always >= 0) with has_more forced
# True so the degraded signature can never collide with the empty one.
_BACKUP_NOTICE_EMPTY_SIGNATURE = (0, 0, 0, False)
_BACKUP_NOTICE_DEGRADED_SIGNATURE = (-1, -1, -1, True)


class MemoryTools(_ToolsForwards, _ToolsLifecycle, _ToolsSemantic, _ToolsNotices, _ToolsGovern):
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
        # 0.16.12 P1-T1 query-embed LRU cache (capacity 128): identical
        # (space, embedder lineage, query) triples return the stored vector
        # instead of re-embedding. Keyed by the vec index's active space id,
        # the process build counter (embedder rebuild) and the embedder's own
        # device epoch (GPU→CPU degrade), so vectors never cross lineages.
        self._embedder_builds = 0
        self._query_embed_cache: "OrderedDict[tuple[Any, ...], list[float]]" = OrderedDict()
        self._query_embed_cache_lock = threading.Lock()
        self._query_embed_cache_capacity = 128
        self._update_monitor: UpdateMonitor | None = None
        # R2-S1：唯一 worker 提前到管线构造前（原 LocalTextIndexWorker 的
        # 位置）——ReadPipeline/OperationsPipeline 在 __init__ 里就要引用它。
        self._semantic_worker = SemanticConflictWorker(self)
        self._surfaces = ProductSurfaces(self)
        self._signals = ConflictSignalPipeline(self)
        self._write_pipeline = WritePipeline(self)
        self._read_pipeline = ReadPipeline(self)
        self._operations = OperationsPipeline(self)
        self._evidence = EvidencePipeline(self)
        self._scan_pipeline = ScanPipeline(self)
        self._queue_protocol = QueueProtocol(self)
        self._semantic_backend: SemanticBackend | None = None
        self._semantic_backend_lock = threading.Lock()
        self._semantic_runtime_disabled = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_complete = False
        # Per-product-call caller cache. ContextVar isolates concurrent MCP
        # tasks; the product wrapper clears it before dispatch and automatic
        # notice delivery reuses the scope computed by the operation.
        self._product_caller: ContextVar[CallerWorkspace | None] = ContextVar(
            "memory_arbiter_product_caller", default=None,
        )
        # A1 ring (0.15.14): recent examined-pair samples {pair_ms,
        # prompt_tokens, generated_tokens, retried, at} replacing the single
        # last_pair_duration_ms scalar, so status/doctor aggregates can
        # separate queue competition from long decodes from retries.
        self._pair_samples: deque[dict[str, Any]] = deque(maxlen=SEMANTIC_PAIR_RING_SIZE)
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
        # 0.16.0 §6⑪: detector-epoch arm runs once at boot — a running
        # CONFLICT_DETECTOR_VERSION newer than the persisted arm re-arms a
        # full pipeline round (watermark-NULL) and records the reason for
        # doctor + the first-call side-channel notice.
        self._arm_scan_epoch_if_needed()
        self._notice_claim_last_error_at: str | None = None
        self._last_backup_notice_signature: tuple[int, int, int, bool] | None = None
        self._last_backup_source_signature: tuple[int, int, int] | None = None
        # One-time-per-process subject_tags_vec backfill (0.15.3): the write-
        # time duplicate-hint recall index must cover pre-existing active
        # memories, not only rows written after the upgrade.
        self._subject_tags_backfilled = False
        # Same contract for the C3a summary vectors (0.15.13): one per active
        # memory, workspace-anomaly voting index.
        self._summary_vec_backfilled = False
        # 0.16.12 P2-T4: boot backfills run on this daemon thread (flags above
        # flip to "scheduled" when it starts).
        self._backfill_thread: threading.Thread | None = None
        self._backfill_done = threading.Event()
        self._backfill_done.set()
        banner = self._setup_capability_banner()
        if banner is not None:
            # Persistent (deduped) — rides every response's warnings until the
            # capability is installed and the process restarts.
            self.db.state.warn(banner)

    def _setup_health(self) -> dict[str, Any]:
        """Capability health for the first-call onboarding notice (P1)."""
        embedding = self.settings.embedding_model_path
        embedding_state = "ok" if (embedding is not None and embedding.is_file()) else "missing"
        semantic_ckpt = self.settings.semantic_conflict_mdeberta_ckpt
        if self.settings.semantic_conflict_enabled:
            semantic_state = (
                "ok" if (semantic_ckpt is not None and semantic_ckpt.is_file()) else "missing"
            )
        elif semantic_ckpt is not None:
            # enabled=false with a configured checkpoint: deliberate opt-out.
            semantic_state = "disabled"
        else:
            semantic_state = "missing"
        try:
            import sqlite_vec  # noqa: F401
            vec_state = "ok"
        except Exception:
            vec_state = "missing"
        health: dict[str, Any] = {
            "sqlite_vec": vec_state,
            "embedding_model": embedding_state,
            "semantic_model": semantic_state,
        }
        if "missing" in health.values():
            hints = ["mema 正在以降级模式运行：缺失能力见上。"]
            if health["embedding_model"] == "missing":
                hints.append(
                    "向量模型：运行 mema setup --install 补齐"
                    "（自动装依赖+下载模型+回写 config，支持断点续传与国内镜像）。"
                )
            if health["semantic_model"] == "missing":
                hints.append(
                    "判定模型不在 setup --install 范围：装 [mdeberta] extra、"
                    "下载 V4m ckpt 后在 config 设 semantic_conflict.mdeberta_ckpt。"
                )
            health["hint"] = " ".join(hints)
        return health

    def _setup_capability_banner(self) -> str | None:
        """The loud degraded-mode banner, or None when the install is full.

        Gated on config_file_loaded: real installs load Settings from an
        on-disk config via from_env; directly-constructed Settings (tests,
        embedded use) never nag. An explicit semantic_conflict.enabled=false
        with a configured model is a deliberate minimal install and stays
        quiet; enabled=false with NO model path means "never installed" and
        nags like any other missing capability.
        """
        if not self.settings.config_file_loaded:
            return None
        missing: list[str] = []
        embedding = self.settings.embedding_model_path
        if embedding is None or not embedding.is_file():
            missing.append("✗ 向量召回未启用（embedding 模型未找到）")
        if self.settings.semantic_conflict_enabled:
            ckpt = self.settings.semantic_conflict_mdeberta_ckpt
            semantic_missing = ckpt is None or not ckpt.is_file()
            if semantic_missing:
                missing.append("✗ 冲突检测未启用（mdeberta 判定模型未配置：装 [mdeberta] extra、下载 V4m ckpt、配 semantic_conflict.mdeberta_ckpt）")
        if not missing:
            return None
        return "\n".join([
            "mema 正在以【降级模式】运行：",
            *missing,
            "当前只有基础全文搜索/写入可用，这不是 mema 的完整能力。",
            "→ 向量模型缺失：运行 mema setup --install 补齐（自动装依赖+下载模型+回写 config）",
            "→ 判定模型缺失：不在 setup --install 范围——装 [mdeberta] extra、下载 V4m ckpt、配 semantic_conflict.mdeberta_ckpt",
        ])

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

    def wait_evidence_worker_drained(self, timeout: float = 30.0) -> bool:
        # R2-S1（C2 后 LocalTextIndexWorker 退役）：历史方法名保留给 eval/
        # replay 调用方，语义 worker 就是唯一的存活 worker——索引与检测同队列。
        return self._semantic_worker.wait_drained(timeout)

    def _enqueue_local_text_index(
        self, memory_id: int, record: dict[str, Any] | None = None,
        *, trusted_applying_context: TrustedApplyingContext | None = None,
        recheck_conflicts: bool = True,
    ) -> dict[str, Any]:
        """C2 (0.17.0 worker merge): enqueue the ONE semantic job that does
        indexing AND detection. The old two-queue chain (evidence worker
        indexes → forwards to the semantic worker) is gone; the local-text
        index worker is retired with it (R2-S1) — boot backfill / repair
        paths call the same single semantic queue.

        ``recheck_conflicts=False`` (conflict-apply edits §15.3, replay
        postprocess) enqueues an index_only job: segment + batch-embed +
        publish, no detection — same queue, same ordering guarantees.
        ``semantic_conflict_on_write == "off"`` degrades EVERY job to
        index_only the same way (checked again at job time, so toggling the
        setting needs no queue surgery).
        """
        current = record or self.db.get_memory(int(memory_id)) or {}
        version = int(current.get("version") or 1)
        # C2: index-only jobs (recheck opt-out OR the global off switch —
        # one predicate, both the flag and the id derive from it) carry
        # their own monotonic task id: the queue's completed-dedupe must not
        # swallow repair retries, and an off-period job must never occupy
        # the normal task slot a later on-period enqueue waits on.
        index_only = (
            not recheck_conflicts
            or self.settings.semantic_conflict_on_write == "off"
        )
        task_id = (
            f"semantic:{int(memory_id)}@{version}!index{time.monotonic_ns()}"
            if index_only
            else f"semantic:{int(memory_id)}@{version}"
        )
        content = str(current.get("content") or "")
        content_hash = (
            str(current.get("content_sha") or "")
            or hashlib.sha256(content.encode("utf-8")).hexdigest()
        )
        snapshot: dict[str, Any] = {
            "memory_id": int(memory_id),
            "version": version,
            "content_hash": content_hash,
            "task_id": task_id,
            "dedupe_key": task_id,
        }
        if trusted_applying_context is not None:
            snapshot["trusted_applying_context"] = trusted_applying_context.to_dict()
        # C2: index_only is decided AT ENQUEUE TIME (recheck opt-out or the
        # global off switch). The job body itself only honours the snapshot
        # flag, so direct _process_semantic_conflict_job calls (tests, the
        # harness) always run the full detection regardless of the switch.
        if index_only:
            snapshot["index_only"] = True
        result = self._semantic_worker.enqueue(int(memory_id), snapshot)
        return {**result, "semantic_task_id": task_id, "semantic_dedupe_key": task_id}

    def _enqueue_content_postcommit(
        self, memory_id: int, record: dict[str, Any] | None = None,
        *, trusted_applying_context: TrustedApplyingContext | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        index = self._enqueue_local_text_index(
            memory_id, record, trusted_applying_context=trusted_applying_context,
        )
        task_id = index.get("semantic_task_id")
        # 0 (semantic_conflict.notice_sync_wait_ms=0) = never block the write
        # response on the post-commit check: batch ingestion still gets the
        # job run and notices deliver on a later response. C2: the sync wait
        # now covers the WHOLE job (embed+publish+detect) — the window is
        # unchanged, the chain link it used to wait behind no longer exists.
        wait_ms = max(0, int(self.settings.semantic_conflict_notice_sync_wait_ms))
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
            check["reason"] = "semantic_conflict_off"
        return index, check

    def _post_commit(
        self, memory_id: int, record: dict[str, Any] | None = None,
        *, recheck_conflicts: bool,
        trusted_applying_context: TrustedApplyingContext | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Single write-path post-commit entry: one job, recheck explicitly.

        Every writer states whether the semantic-conflict check re-enters for
        this write (recheck_conflicts) instead of each call site hand-picking
        between the two enqueue helpers; the trusted context is only valid on
        the apply flow's own committed edits (§15.3). Returns
        (evidence_index, semantic_conflict_check); the check slot is
        skipped/recheck_disabled when the writer opted out.
        """
        if not recheck_conflicts:
            return (
                self._enqueue_local_text_index(
                    memory_id, record,
                    trusted_applying_context=trusted_applying_context,
                    recheck_conflicts=False,
                ),
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

    def wait_semantic_worker_drained(self, timeout: float = 30.0) -> bool:
        return self._semantic_worker.wait_drained(timeout)

    def _get_semantic_backend_ref(self) -> SemanticBackend | None:
        with self._semantic_backend_lock:
            return self._semantic_backend

    def _embedding_configured(self) -> bool:
        # Pointing at a GGUF model IS the intent to embed (no provider or
        # vec.enabled knob since 0.15.0).
        return self.settings.embedding_model_path is not None

    def _index_local_text_evidence(self, memory_id: int, record: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._evidence.index_memory(memory_id, record)

    def _query_embed_cache_key(
        self, embedder: ManagedEmbedder, query: str, active_space_id: Any,
    ) -> tuple[Any, ...]:
        """Cache key: (space, build lineage, device epoch, query)."""
        return (
            str(active_space_id), int(self._embedder_builds),
            int(getattr(embedder, "embed_epoch", 0)), query,
        )

    def _query_embed_cache_get(self, key: tuple[Any, ...]) -> "list[float] | None":
        with self._query_embed_cache_lock:
            cached = self._query_embed_cache.get(key)
            if cached is None:
                return None
            self._query_embed_cache.move_to_end(key)
            # Stored list is shared: callers must treat it as read-only
            # (downstream only json-serialises it into SQL).
            return cached

    def _query_embed_cache_put(self, key: tuple[Any, ...], embedding: "list[float]") -> None:
        with self._query_embed_cache_lock:
            self._query_embed_cache[key] = embedding
            self._query_embed_cache.move_to_end(key)
            while len(self._query_embed_cache) > self._query_embed_cache_capacity:
                self._query_embed_cache.popitem(last=False)

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
            # 0.16.12 query-embed LRU: a fresh embedder instance invalidates
            # every cached query vector (different lineage) — bump the build
            # counter that keys the cache and drop the stale entries outright.
            self._embedder_builds += 1
            with self._query_embed_cache_lock:
                self._query_embed_cache.clear()
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
            if not (self._subject_tags_backfilled and self._summary_vec_backfilled):
                # 0.16.12 P2-T4 step 2: the boot backfills move OFF the
                # _embedder_lock onto a daemon thread — the first write/find
                # that built the embedder no longer blocks on re-embedding
                # every pre-existing active row. Embedding stays serial: the
                # backfill's embed_text calls take the embedder's own
                # _embed_lock, the same lock the evidence worker's embeds
                # take. Flag semantics change from "ran" to "scheduled";
                # failure retries on the next process restart (same
                # fail-open, non-repeating contract as before).
                self._subject_tags_backfilled = True
                self._summary_vec_backfilled = True
                if self._backfill_thread is None:
                    self._backfill_done.clear()
                    self._backfill_thread = threading.Thread(
                        target=self._run_boot_backfills, args=(embedder,),
                        name="mema-boot-backfill", daemon=True,
                    )
                    self._backfill_thread.start()
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
            from .pipeline.read import vec_disabled_warning

            warning = vec_disabled_warning(reason)
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
                resolved = self.db.resolve_workspace_canonical(workspace, embedder)
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

    @staticmethod
    def _memory_visible(record: dict[str, Any] | None, caller: CallerWorkspace) -> bool:
        """Shared id-driven visibility predicate (0.16.12 P1-T4).

        The single-id path and the batch_read prefetch MUST apply the same
        rule: strict callers see only rows inside their admitted canonical set
        (mirrors get_memory_for_workspace's SQL scope); non-strict callers see
        any existing row; a strict caller without a canonical sees nothing.
        """
        if caller.isolation != "strict":
            return record is not None
        if not caller.canonical:
            return False
        return visible_memory(record, caller.canonical, caller.scope_canonicals())

    def _get_memory_visible(self, memory_id: int, caller: CallerWorkspace | None = None) -> dict[str, Any] | None:
        caller = caller or self._caller_workspace(None)
        record = self.db.get_memory(int(memory_id))
        return record if self._memory_visible(record, caller) else None

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
    def _confidence_rank(hint: str | None) -> int:
        return ConflictSignalPipeline._confidence_rank(hint)

    def _attach_conflict_signals(
        self,
        results: list[dict[str, Any]],
        warnings: list[str],
        precomputed_groups: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return self._signals._attach_conflict_signals(results, warnings, precomputed_groups)

    def _build_open_table_signal(
        self,
        memory_id: int,
        conflicts: list[dict[str, Any]],
        summaries: dict[int, dict[str, Any]],
        result_id_set: set[int],
    ) -> dict[str, Any] | None:
        return self._signals._build_open_table_signal(memory_id, conflicts, summaries, result_id_set)
