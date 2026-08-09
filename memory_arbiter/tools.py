from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from importlib import resources
from typing import Any, Callable, Optional, Tuple

from .arbitration import compare_memories
from .claims import extract_claims
from .conflict_judgments import ConflictJudgmentStore
from .config import Settings
from .constants import strict_ws
from .db import MemoryDB, _canon_entity, _canon_scope
from .embedder import ManagedEmbedder
from .models import MemoryRecord, MemoryStatus, ProtectionLevel, SourceType
from .search import search_memories, _linked_open_items_for_search
from .semantic_conflict import (
    LocalGGUFSemanticBackend,
    SemanticBackend,
    ModelSignal,
    notice_dedupe_key,
    pair_text_gate,
)
from .update_monitor import UpdateMonitor
from . import __version__
from . import workspace_rules
from .workers import SemanticConflictWorker, SplitReindexWorker
from .surfaces import ProductSurfaces
from .pipeline.signals import ConflictSignalPipeline
from .pipeline.sections import SectionPipeline
from .pipeline.write import WritePipeline
from .pipeline.read import ReadPipeline
from .pipeline.operations import OperationsPipeline


AGENT_ONBOARDING_TOPIC = "agent_onboarding"


def _agent_onboarding_guide() -> str:
    try:
        return resources.files("memory_arbiter").joinpath("AGENT_ONBOARDING.md").read_text(encoding="utf-8")
    except Exception:
        return (
            "mema / Memory Arbiter: use MCP tools for memory operations and governance. "
            "Save this compact rule to your persistent local agent memory file when possible. "
            "Full guide topic: memory(action='help', data={'topic': 'agent_onboarding'})."
        )


class MemoryTools:
    _V08_SPLIT_STATUSES = (None, "active", "failed", "declined")
    _TRUST_RANK: dict[str, int] = {
        "user_confirmed": 4,
        "locked": 4,
        "document_extracted": 3,
        "protected": 3,
        "agent_generated": 2,
        "unknown": 1,
    }

    def __init__(self, settings: Optional[Settings] = None, db: Optional[MemoryDB] = None):
        self.settings = settings or Settings.from_env()
        self.db = db or MemoryDB(self.settings)
        self._embedder: Optional[ManagedEmbedder] = None
        self._embedder_loaded = False
        self._embedder_lock = threading.Lock()
        self._embedder_warnings: list[str] = list(self.settings.config_warnings)
        self._update_monitor: Optional[UpdateMonitor] = None
        self._split_worker = SplitReindexWorker(self)
        self._surfaces = ProductSurfaces(self)
        self._signals = ConflictSignalPipeline(self)
        self._sections = SectionPipeline(self)
        self._write_pipeline = WritePipeline(self)
        self._read_pipeline = ReadPipeline(self)
        self._operations = OperationsPipeline(self)
        self._semantic_backend: Optional[SemanticBackend] = None
        self._semantic_backend_lock = threading.Lock()
        self._semantic_worker = SemanticConflictWorker(self)
        self._last_pair_duration_ms: Optional[int] = None
        # v0.6.0: initialise vec index state on startup
        self._init_vec_state()

    def start_update_monitor(self, monitor: Optional[UpdateMonitor] = None) -> None:
        try:
            self._update_monitor = monitor or UpdateMonitor(enabled=self.settings.update_check_enabled)
            self.db.state.notice_provider = self._consume_notices
            self._update_monitor.maybe_start_check_if_due()
        except Exception:
            self._update_monitor = None

    def start_split_worker(self) -> None:
        self._split_worker.start()
        self._enqueue_pending_rule_splits(limit=100)

    def start_semantic_worker(self) -> None:
        self._semantic_worker.start()

    def _consume_notices(self) -> list[dict[str, Any]]:
        if self._update_monitor is None:
            return []
        notices: list[dict[str, Any]] = []
        notices.extend(self._update_monitor.consume_agent_onboarding_notice(self.settings.agent_id))
        notices.extend(self._update_monitor.consume_notices())
        return notices

    def _enqueue_pending_rule_splits(self, limit: int = 100) -> None:
        threshold = self.settings.split_threshold
        max_sections = self.settings.max_sections
        max_section_chars = self.settings.max_section_chars
        if self.db.get_vec_index_state().get("state") != "ready":
            return
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, content, version, split_status, split_revision "
                    "FROM memories "
                    "WHERE status = 'active' AND split_status IS NULL AND length(content) >= ? "
                    "ORDER BY id LIMIT ?",
                    (threshold, int(limit)),
                ).fetchall()
        except Exception:
            return
        for row in rows:
            content = row["content"] or ""
            plan, _reason = self._rule_plan_sections(content, max_sections, max_section_chars)
            if plan is None:
                continue
            memory_id = int(row["id"])
            self._split_worker.enqueue(memory_id, {
                "memory_id": memory_id,
                "content": content,
                "plan": plan,
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "memory_version": int(row["version"] or 1),
                "split_status": row["split_status"],
                "split_revision": int(row["split_revision"] or 0),
            })

    def wait_split_worker_drained(self, timeout: float = 5.0) -> bool:
        return self._split_worker.wait_drained(timeout)

    def _init_vec_state(self) -> None:
        """Initialise _vec_index_meta based on current embedder availability."""
        space_id = None
        has_managed = False
        if self._embedding_configured() and self.settings.enable_sqlite_vec:
            embedder, _ = self._ensure_embedder()
            if embedder is not None:
                space_id = embedder.embedding_space_id
                has_managed = True
        try:
            self.db.init_vec_index_state(space_id, has_managed)
        except Exception:
            pass  # non-fatal: state init failure shouldn't block startup

    def _allowed(self, agent_id: Optional[str] = None, client: Optional[str] = None) -> Tuple[bool, list[str]]:
        actual_agent = agent_id or self.settings.agent_id
        actual_client = client or self.settings.client
        if self.settings.policy.enabled_for(actual_client, actual_agent):
            return True, []
        return False, [f"Memory arbiter disabled by policy for client={actual_client}, agent_id={actual_agent}."]

    @staticmethod
    def _payload_dict(data: Optional[dict[str, Any]]) -> dict[str, Any]:
        return dict(data) if isinstance(data, dict) else {}

    def _judge_constraints(self) -> dict[str, Any]:
        return self._surfaces._judge_constraints()

    def _product_help(self, surface: str, topic: Optional[str] = None) -> dict[str, Any]:
        return self._surfaces._product_help(surface, topic)

    def _invalid_product_call(self, surface: str, message: str, topic: Optional[str] = None) -> dict[str, Any]:
        return self._surfaces._invalid_product_call(surface, message, topic)

    @staticmethod
    def _help_topic(payload: dict[str, Any], fallback_key: str) -> Optional[str]:
        return ProductSurfaces._help_topic(payload, fallback_key)

    def _forward(
        self, surface: str, topic: Optional[str], fn: Callable[..., dict[str, Any]], **payload: Any,
    ) -> dict[str, Any]:
        return self._surfaces._forward(surface, topic, fn, **payload)

    @staticmethod
    def _alias_id(payload: dict[str, Any], target: str) -> None:
        return ProductSurfaces._alias_id(payload, target)

    def _int_product_arg(
        self, surface: str, value: Any, name: str, topic: Optional[str] = None,
    ) -> Optional[int | dict[str, Any]]:
        return self._surfaces._int_product_arg(surface, value, name, topic)

    def _require_id(
        self, surface: str, payload: dict[str, Any], name: str, topic: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        return self._surfaces._require_id(surface, payload, name, topic)

    def _coerce_product_id(
        self, surface: str, payload: dict[str, Any], name: str, topic: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        return self._surfaces._coerce_product_id(surface, payload, name, topic)

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        return ProductSurfaces._is_truthy(value)

    def _require_ws_strings(
        self, payload: dict[str, Any], names: tuple[str, ...], surface: str, topic: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        return self._surfaces._require_ws_strings(payload, names, surface, topic)

    def memory(self, action: str = "help", data: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory(action, data, **_)

    def memory_review(self, view: str = "help", data: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory_review(view, data, **_)

    def memory_govern(self, action: str = "help", data: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory_govern(action, data, **_)

    def memory_repair(self, task: str = "help", data: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory_repair(task, data, **_)

    def _embedding_configured(self) -> bool:
        return self.settings.embedding_provider == "gguf" and self.settings.embedding_model_path is not None

    def _ensure_embedder(self) -> Tuple[Optional[ManagedEmbedder], list[str]]:
        if self._embedder_loaded:
            return self._embedder, []
        with self._embedder_lock:
            if self._embedder_loaded:
                return self._embedder, []
            if not self._embedding_configured():
                self._embedder_loaded = True  # deterministic config state; safe to cache
                return None, []
            if not self.settings.enable_sqlite_vec:
                warning = "embedding configured but vec.enabled=false; auto-embedding disabled. Set vec.enabled=true to enable."
                self._embedder_warnings.append(warning)
                self._embedder_loaded = True  # deterministic config state
                return None, [warning]
            from .embedder import build_embedder

            assert self.settings.embedding_model_path is not None
            embedder, warnings = build_embedder(
                str(self.settings.embedding_model_path),
                self.settings.vec_dim,
                n_ctx=self.settings.embedding_n_ctx,
                reserved_tokens=self.settings.embedding_reserved_tokens,
                max_section_chars=self.settings.max_section_chars,
            )
            self._embedder_warnings.extend(warnings)
            if embedder is None:
                # Build failed (missing model / dim mismatch / load error). Do NOT cache —
                # a later retry (e.g. model installed) should still be able to succeed.
                return None, warnings
            self._embedder = embedder
            self._embedder_loaded = True  # cache only on successful build
            return self._embedder, warnings

    @staticmethod
    def _embedding_text(record: dict[str, Any]) -> str:
        subject = record.get("subject") or ""
        content = record.get("content") or ""
        return f"{subject}\n{content}".strip()

    def _semantic_configured(self) -> bool:
        return (
            bool(self.settings.semantic_conflict_enabled)
            and self.settings.semantic_conflict_backend == "local_gguf"
            and self.settings.semantic_conflict_model_path is not None
        )

    def _ensure_semantic_backend(self) -> Optional[SemanticBackend]:
        if not self._semantic_configured():
            return None
        with self._semantic_backend_lock:
            if self._semantic_backend is not None:
                return self._semantic_backend
            assert self.settings.semantic_conflict_model_path is not None
            self._semantic_backend = LocalGGUFSemanticBackend(
                self.settings.semantic_conflict_model_path,
                n_ctx=self.settings.semantic_conflict_n_ctx,
                n_threads=self.settings.semantic_conflict_n_threads,
                n_batch=self.settings.semantic_conflict_n_batch,
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
        candidates = [s["name"] for s in (similar or []) if s.get("name")][:5]
        if not candidates:
            return None
        try:
            return backend.suggest_workspace_candidate(ws_raw, evidence, candidates)
        except Exception:
            return None

    def _semantic_status(self) -> dict[str, Any]:
        backend_status = (
            self._semantic_backend.status()
            if self._semantic_backend is not None else
            {
                "backend": self.settings.semantic_conflict_backend,
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
            "pair_text_gate": self.settings.semantic_conflict_pair_text_gate,
            "resident": bool(self.settings.semantic_conflict_resident),
            "max_concurrency": 1,
            "max_concurrency_note": "reserved; MVP semantic worker is single-threaded (configured values are clamped to 1)",
            "job_timeout_ms": int(self.settings.semantic_conflict_job_timeout_ms),
            "min_pair_budget_ms": int(self.settings.semantic_conflict_min_pair_budget_ms),
            "last_pair_duration_ms": self._last_pair_duration_ms,
            "job_deadline_behavior": (
                "The deadline gates BETWEEN pairs only. A model call (synchronous C inference) "
                "cannot be interrupted mid-call; when remaining budget drops below "
                "min_pair_budget_ms the job stops early instead of starting another inference. "
                "A single stuck call still stalls the worker until process-level isolation lands."
            ),
            "worker": self._semantic_worker.status(),
            "backend": backend_status,
            "notices": self.db.semantic_notice_counts(),
        }

    def _semantic_control(self, action: str) -> dict[str, Any]:
        action = str(action or "status").strip().lower()
        if action == "status":
            return self._semantic_status()
        if action == "pause":
            self._semantic_worker.pause()
            return {"outcome": "paused", "semantic_conflict": self._semantic_status()}
        if action == "resume":
            worker_state = self._semantic_worker.status().get("runtime_state")
            if worker_state == "disabled":
                return {
                    "outcome": "runtime_disabled_use_enable",
                    "semantic_conflict": self._semantic_status(),
                }
            self._semantic_worker.resume()
            return {"outcome": "resumed", "semantic_conflict": self._semantic_status()}
        if action == "enable":
            self._semantic_worker.enable_runtime()
            return {"outcome": "enabled", "semantic_conflict": self._semantic_status()}
        if action == "unload":
            if self._semantic_backend is not None:
                self._semantic_backend.unload()
            return {"outcome": "unloaded", "semantic_conflict": self._semantic_status()}
        if action == "disable":
            self._semantic_worker.disable_runtime()
            if self._semantic_backend is not None:
                self._semantic_backend.unload()
            return {
                "outcome": "runtime_disabled",
                "note": "This disables the current runtime only; set semantic_conflict.enabled=false in config to persist it.",
                "semantic_conflict": self._semantic_status(),
            }
        return {"outcome": "invalid_action", "valid_actions": ["status", "pause", "resume", "enable", "unload", "disable"]}

    def _enqueue_semantic_conflict_check(self, memory_id: Optional[int], record: Any) -> dict[str, Any]:
        if memory_id is None:
            return {"status": "skipped", "reason": "backup_only"}
        if not self.settings.semantic_conflict_enabled:
            return {"status": "disabled"}
        if self.settings.semantic_conflict_on_write == "off":
            return {"status": "off"}
        stored = self.db.get_memory(int(memory_id)) or {}
        content = (record.get("content") if isinstance(record, dict) else getattr(record, "content", None))
        if content is None:
            content = stored.get("content") or ""
        snapshot = {
            "memory_id": int(memory_id),
            "version": int(stored.get("version") or self.db.get_memory_version(int(memory_id)) or 1),
            "claim_revision": int(stored.get("claim_revision") or 1),
            "content_hash": hashlib.sha256(str(content or "").encode("utf-8")).hexdigest(),
        }
        return self._semantic_worker.enqueue(int(memory_id), snapshot)

    def _semantic_candidate_memories(self, memory_id: int, record: dict[str, Any]) -> list[dict[str, Any]]:
        tags = record.get("tags") or []
        if isinstance(tags, str):
            try:
                parsed = json.loads(tags)
                tags = parsed if isinstance(parsed, list) else []
            except Exception:
                tags = []
        candidates = self.db.find_metadata_overlap_candidates(
            subject=record.get("subject"),
            tags=[str(t) for t in tags if isinstance(t, str)],
            exclude_id=int(memory_id),
            limit=int(self.settings.semantic_conflict_candidate_limit),
        )
        return candidates[: int(self.settings.semantic_conflict_pair_limit)]

    def _process_semantic_conflict_job(self, memory_id: int, snapshot: dict[str, Any]) -> None:
        if not self._semantic_configured():
            return
        record = self.db.get_memory(int(memory_id))
        if not record or record.get("status") != "active":
            return
        if int(record.get("version") or 1) != int(snapshot.get("version") or 1):
            return
        if int(record.get("claim_revision") or 1) != int(snapshot.get("claim_revision") or 1):
            return
        content_hash = hashlib.sha256((record.get("content") or "").encode("utf-8")).hexdigest()
        if content_hash != snapshot.get("content_hash"):
            return
        backend = self._ensure_semantic_backend()
        if backend is None:
            return
        timeout_ms = float(self.settings.semantic_conflict_job_timeout_ms)
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        # The model call is a synchronous, non-interruptible C call; the
        # deadline can only gate *between* pairs, not abort one in flight.
        # Reserve a floor so we never start a fresh inference with only a few
        # ms left (which would blow the budget and stall the worker). 1s is a
        # conservative lower bound for a 0.5B local model on a single short pair.
        min_pair_budget = float(self.settings.semantic_conflict_min_pair_budget_ms) / 1000.0
        isolation = self.settings.isolation
        for peer in self._semantic_candidate_memories(memory_id, record):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._semantic_worker.set_error("semantic conflict job timed out before next pair")
                return
            if remaining < min_pair_budget:
                # Not enough budget to safely start another inference; stop
                # cleanly so the worker loop stays responsive instead of
                # launching a call that may overrun and stall it.
                self._semantic_worker.set_error(
                    f"semantic conflict job stopped early: remaining budget {remaining:.2f}s below {min_pair_budget:.2f}s floor"
                )
                return
            peer_record = self.db.get_memory(int(peer["id"]))
            if not peer_record or peer_record.get("status") != "active":
                continue
            if isolation != "none":
                left_ws = record.get("workspace_canonical") or record.get("workspace")
                right_ws = peer_record.get("workspace_canonical") or peer_record.get("workspace")
                if left_ws and right_ws and left_ws != right_ws:
                    continue
            left_version = self.db.get_memory_version(int(memory_id)) or 1
            right_version = self.db.get_memory_version(int(peer_record["id"])) or 1
            try:
                if self.db.is_pair_dismissed(int(memory_id), int(peer_record["id"])):
                    continue
                if self.db.is_semantic_pair_closed(int(memory_id), int(peer_record["id"]), left_version, right_version):
                    continue
            except Exception:
                pass
            call_start = time.monotonic()
            signal = backend.classify_pair(record, peer_record)
            self._last_pair_duration_ms = int((time.monotonic() - call_start) * 1000)
            if not self.settings.semantic_conflict_resident and self._semantic_backend is not None:
                self._semantic_backend.unload()
            if not signal.candidate:
                continue
            gate = pair_text_gate(
                f"subject: {record.get('subject') or ''}\ntags: {', '.join(record.get('tags') or []) if isinstance(record.get('tags'), list) else ''}\ncontent: {record.get('content') or ''}",
                f"subject: {peer_record.get('subject') or ''}\ntags: {', '.join(peer_record.get('tags') or []) if isinstance(peer_record.get('tags'), list) else ''}\ncontent: {peer_record.get('content') or ''}",
                mode=self.settings.semantic_conflict_pair_text_gate,
            )
            if not gate.passed:
                continue
            dedupe = notice_dedupe_key(
                int(memory_id), int(peer_record["id"]), left_version, right_version, "semantic_pair"
            )
            title = f"Possible semantic memory conflict with #{peer_record['id']}"
            message = "; ".join(gate.reasons) or signal.candidate_type
            self.db.record_semantic_notice(
                memory_id=int(memory_id),
                peer_id=int(peer_record["id"]),
                severity=gate.severity,
                notice_type="semantic_pair",
                title=title,
                message=message,
                payload={
                    "model_signal": {
                        "candidate_type": signal.candidate_type,
                        "confidence": signal.confidence,
                        "parsed": signal.parsed,
                        "error": signal.error,
                    },
                    "gate": {
                        "mode": gate.mode,
                        "severity": gate.severity,
                        "reasons": gate.reasons,
                        "evidence": gate.evidence.__dict__,
                    },
                    "left": {"id": int(memory_id), "subject": record.get("subject")},
                    "right": {"id": int(peer_record["id"]), "subject": peer_record.get("subject")},
                },
                dedupe_key=dedupe,
                left_version=left_version,
                right_version=right_version,
                left_claim_revision=int(record.get("claim_revision") or 1),
                right_claim_revision=int(peer_record.get("claim_revision") or 1),
            )

    def memory_write(self, **payload: Any) -> dict[str, Any]:
        return self._write_pipeline.memory_write(**payload)

    def _enrich_write_response(
        self, data: dict[str, Any], memory_id: Optional[int], record: MemoryRecord,
    ) -> dict[str, Any]:
        return self._write_pipeline._enrich_write_response(data, memory_id, record)

    @staticmethod
    def _structured_conflict_point(claims: list[dict[str, Any]]) -> str:
        return WritePipeline._structured_conflict_point(claims)

    @staticmethod
    def _apply_structured_gate(data: dict[str, Any], structured: dict[str, Any]) -> None:
        return WritePipeline._apply_structured_gate(data, structured)

    def _index_and_reconcile_claims(self, memory_id: int) -> dict[str, Any]:
        return self._write_pipeline._index_and_reconcile_claims(memory_id)

    def _index_and_reconcile_claims_impl(
        self, memory_id: int, started: float,
    ) -> dict[str, Any]:
        return self._write_pipeline._index_and_reconcile_claims_impl(memory_id, started)

    def _write_duplicate_hints(
        self, memory_id: int, record: MemoryRecord,
    ) -> Optional[dict[str, Any]]:
        return self._write_pipeline._write_duplicate_hints(memory_id, record)

    def memory_search(self, query: str = "", workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, offset: int = 0, debug_ranking: bool = False, query_embedding: Optional[list[float]] = None, tags_filter: Optional[list[str]] = None, after_time: Optional[str] = None, before_time: Optional[str] = None, source_type: Optional[str] = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True, **_: Any) -> dict[str, Any]:
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
        workspace: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 20,
        debug_ranking: bool = False,
        query_embedding: Optional[list[float]] = None,
        tags_filter: Optional[list[str]] = None,
        after_time: Optional[str] = None,
        before_time: Optional[str] = None,
        source_type: Optional[str] = None,
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
        sections: str = "catalog",
        section_ids: Optional[list[int]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._read_pipeline.memory_get(
            memory_id=memory_id, sections=sections, section_ids=section_ids, **_,
        )

    def memory_store_embedding(self, memory_id: int, embedding: list[float], **_: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_store_embedding(memory_id, embedding, **_)

    def memory_recent(self, workspace: Optional[str] = None, limit: int = 20, **_: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_recent(workspace, limit, **_)

    def memory_compare(self, left_id: Optional[int] = None, right_id: Optional[int] = None, left: Optional[dict[str, Any]] = None, right: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_compare(left_id, right_id, left, right, **_)

    def memory_arbitrate(self, left_id: int, right_id: int, mark_conflict: bool = True, authorized: bool = False, **_: Any) -> dict[str, Any]:
        return self._operations.memory_arbitrate(left_id, right_id, mark_conflict, authorized, **_)

    def _with_resolution_guidance(self, conflict: dict[str, Any]) -> dict[str, Any]:
        return self._operations._with_resolution_guidance(conflict)

    def memory_list_conflicts(self, status: str = "open", limit: int = 50, source: Optional[str] = None, **_: Any) -> dict[str, Any]:
        return self._operations.memory_list_conflicts(status, limit, source, **_)

    def memory_record_conflict(
        self, left_id: int, right_id: int, reason: str,
        conflict_type: Optional[str] = None, conflict_point: Optional[str] = None,
        suggested_winner: Optional[int] = None, confidence_hint: Optional[str] = None,
        source: Optional[str] = None, refresh: bool = False,
        left_version: Optional[int] = None, right_version: Optional[int] = None,
        scan_prompt_version: Optional[str] = None, scan_model: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_record_conflict(
            left_id, right_id, reason, conflict_type, conflict_point, suggested_winner,
            confidence_hint, source, refresh, left_version, right_version,
            scan_prompt_version, scan_model, **_,
        )

    def memory_resolve_conflict(
        self, conflict_id: int, reason: str = "", status: str = "resolved", **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_resolve_conflict(conflict_id, reason, status, **_)

    def memory_confirm(self, memory_id: int, source_ref: Optional[str] = None, confidence: float = 1.0, authorized: bool = False, **_: Any) -> dict[str, Any]:
        return self._operations.memory_confirm(memory_id, source_ref, confidence, authorized, **_)

    def memory_accept_workspace_alias(
        self, alias: str, canonical: str, relation: str = "alias",
        reason: Optional[str] = None, source: str = "user",
        authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_accept_workspace_alias(
            alias, canonical, relation, reason, source, authorized, **_,
        )

    def memory_reject_workspace_alias(
        self, alias: str, canonical: str, reason: Optional[str] = None,
        source: str = "user", **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_reject_workspace_alias(alias, canonical, reason, source, **_)

    def memory_rename_workspace_canonical(
        self, old: str, new: str, reason: Optional[str] = None, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_rename_workspace_canonical(old, new, reason, **_)

    def memory_migrate_workspace(
        self, reason: Optional[str] = None, **payload: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_migrate_workspace(reason, **payload)

    def memory_confirm_pending_workspace(
        self, memory_id: int, canonical: str, reason: Optional[str] = None,
        authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_confirm_pending_workspace(memory_id, canonical, reason, authorized, **_)

    def memory_activate(
        self, memory_id: int, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_activate(memory_id, authorized, **_)

    def memory_supersede(
        self,
        memory_id: int,
        reason: str,
        superseded_by: Optional[int] = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_supersede(memory_id, reason, superseded_by, authorized, **_)

    def _split_capability(self, vec_state: dict[str, Any]) -> dict[str, Any]:
        return self._operations._split_capability(vec_state)

    def _update_check_status(self) -> dict[str, Any]:
        return self._operations._update_check_status()

    def memory_status(self, **_: Any) -> dict[str, Any]:
        return self._operations.memory_status(**_)

    def memory_doctor_overview(self, deep: bool = False, **_: Any) -> dict[str, Any]:
        return self._operations.memory_doctor_overview(deep, **_)

    def memory_set_entity(
        self, memory_id: int, entity: Optional[str] = None, scope: Optional[str] = None,
        clear: bool = False, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_set_entity(memory_id, entity, scope, clear, authorized, **_)

    def memory_list_entities(
        self, limit: int = 50, include_unassigned: bool = True, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_list_entities(limit, include_unassigned, **_)

    def memory_rebuild_claims(
        self,
        memory_ids: Optional[list[int]] = None,
        dry_run: bool = True,
        batch_size: int = 50,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_rebuild_claims(memory_ids, dry_run, batch_size, **_)

    def memory_submit_conflict_judgment(
        self,
        conflict_id: int,
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        confidence_hint: Optional[str],
        reason: str,
        affects_current_output: bool,
        usage_context: str,
        judge_ref: Optional[str] = None,
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_submit_conflict_judgment(
            conflict_id, expected_left_version, expected_right_version,
            expected_left_claim_revision, expected_right_claim_revision,
            verdict, recommended_use, suggested_winner, confidence_hint,
            reason, affects_current_output, usage_context, judge_ref,
            resolution_kind, conflict_scope, **_,
        )

    def memory_correct_conflict_judgment(
        self,
        conflict_id: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        reason: str,
        expected_judgment_id: int,
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
        authorized: bool = False,
        judge_ref: Optional[str] = None,
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_correct_conflict_judgment(
            conflict_id, verdict, recommended_use, suggested_winner, reason,
            expected_judgment_id, expected_left_version, expected_right_version,
            expected_left_claim_revision, expected_right_claim_revision,
            authorized, judge_ref, resolution_kind, conflict_scope, **_,
        )

    def memory_list_conflict_judgments(self, conflict_id: int, **_: Any) -> dict[str, Any]:
        return self._operations.memory_list_conflict_judgments(conflict_id, **_)

    def memory_audit_summary(self, **_: Any) -> dict[str, Any]:
        return self._operations.memory_audit_summary(**_)

    def memory_edit(
        self,
        memory_id: int,
        new_content: Optional[str] = None,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        new_subject: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
        reason: str = "",
        authorized: bool = False,
        tags_only: bool = False,
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_edit(
            memory_id, new_content, old_text, new_text, new_subject, new_tags,
            reason, authorized, tags_only, add_tags, remove_tags, **_,
        )

    def memory_history(self, memory_id: int, **_: Any) -> dict[str, Any]:
        return self._operations.memory_history(memory_id, **_)

    def memory_cleanup_history(
        self,
        memory_id: Optional[int] = None,
        older_than_days: Optional[int] = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_cleanup_history(memory_id, older_than_days, authorized, **_)

    def memory_cleanup_inactive_vectors(
        self,
        dry_run: bool = True,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_cleanup_inactive_vectors(dry_run, authorized, **_)

    def _count_orphan_vectors(self) -> dict[str, int]:
        return self._operations._count_orphan_vectors()

    def _count_vec_parent_status_mismatch(self) -> dict[str, int]:
        return self._operations._count_vec_parent_status_mismatch()

    def memory_resync_vec_parent_status(
        self,
        dry_run: bool = True,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_resync_vec_parent_status(dry_run, authorized, **_)

    def _trust_score(self, record: dict[str, Any]) -> int:
        return self._signals._trust_score(record)

    @staticmethod
    def _confidence_rank(hint: Optional[str]) -> int:
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
    ) -> Optional[dict[str, Any]]:
        return self._signals._build_open_table_signal(memory_id, conflicts, summaries, result_id_set)

    def _compute_runtime_hint(
        self,
        memory_id: int,
        rec: dict[str, Any],
        all_results: list[dict[str, Any]],
        result_id_set: set[int],
        dismissed_pairs: Optional[set] = None,
    ) -> Optional[dict[str, Any]]:
        return self._signals._compute_runtime_hint(
            memory_id, rec, all_results, result_id_set, dismissed_pairs,
        )

    @staticmethod
    def _catalog_entry(s: dict) -> dict:
        return SectionPipeline._catalog_entry(s)

    def _attach_sections(
        self,
        results: list[dict[str, Any]],
        query_embedding: Optional[list[float]],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        return self._sections._attach_sections(results, query_embedding, warnings)

    @staticmethod
    def _find_nth_occurrence(text: str, anchor: str, occurrence: int) -> int:
        return SectionPipeline._find_nth_occurrence(text, anchor, occurrence)

    def _compute_offsets(
        self,
        content: str,
        sections_data: list[dict[str, Any]],
        trust_planner_offsets: bool = False,
    ) -> Optional[list[dict[str, Any]]]:
        return self._sections._compute_offsets(content, sections_data, trust_planner_offsets)

    @staticmethod
    def _rule_plan_sections(content: str, max_sections: int, max_section_chars: int) -> tuple[Optional[list[dict[str, Any]]], str]:
        return SectionPipeline._rule_plan_sections(content, max_sections, max_section_chars)

    @staticmethod
    def _split_snapshot_error(
        memory: dict[str, Any],
        decision_content_hash: Optional[str],
        decision_memory_version: Optional[int],
        decision_split_status: Optional[str],
        decision_split_revision: Optional[int],
        allowed_split_statuses: tuple[Optional[str], ...],
    ) -> Optional[str]:
        return SectionPipeline._split_snapshot_error(
            memory, decision_content_hash, decision_memory_version,
            decision_split_status, decision_split_revision, allowed_split_statuses,
        )

    def _publish_sections(
        self,
        memory_id: int,
        content: str,
        sections_data: list[dict[str, Any]],
        decision_content_hash: str,
        decision_memory_version: int,
        decision_split_status: Optional[str],
        decision_split_revision: int,
        decision_kind: str,
        provenance: str,
    ) -> dict[str, Any]:
        return self._sections._publish_sections(
            memory_id, content, sections_data, decision_content_hash,
            decision_memory_version, decision_split_status, decision_split_revision,
            decision_kind, provenance,
        )

    def _after_write_split(
        self, memory_id: int,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]], list[str]]:
        return self._sections._after_write_split(memory_id)

    def memory_split(
        self,
        memory_id: int,
        split_decision: Optional[str] = None,
        decision_content_hash: Optional[str] = None,
        decision_memory_version: Optional[int] = None,
        decision_split_status: Optional[str] = None,
        decision_split_revision: Optional[int] = None,
        sections: Optional[list[dict]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._sections.memory_split(
            memory_id, split_decision, decision_content_hash, decision_memory_version,
            decision_split_status, decision_split_revision, sections, **_,
        )

    def _mark_split_failed(
        self,
        mid: int,
        content_hash: str,
        version: int,
        revision: int,
        expected_status: Optional[str],
        stage: str,
        message: str,
    ) -> None:
        return self._sections._mark_split_failed(
            mid, content_hash, version, revision, expected_status, stage, message,
        )

    def memory_rebuild_embeddings(
        self,
        memory_ids: Optional[list[int]] = None,
        dry_run: bool = True,
        batch_size: Optional[int] = 50,
        **_: Any,
    ) -> dict[str, Any]:
        return self._sections.memory_rebuild_embeddings(memory_ids, dry_run, batch_size, **_)
