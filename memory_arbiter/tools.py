from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional, Tuple

from .arbitration import compare_memories
from .config import Settings
from .db import MemoryDB
from .embedder import ManagedEmbedder
from .models import MemoryRecord, ProtectionLevel, SourceType
from .search import search_memories, _linked_open_items_for_search


class MemoryTools:
    def __init__(self, settings: Optional[Settings] = None, db: Optional[MemoryDB] = None):
        self.settings = settings or Settings.from_env()
        self.db = db or MemoryDB(self.settings)
        self._embedder: Optional[ManagedEmbedder] = None
        self._embedder_loaded = False
        self._embedder_warnings: list[str] = list(self.settings.config_warnings)
        # v0.6.0: initialise vec index state on startup
        self._init_vec_state()

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

    def _embedding_configured(self) -> bool:
        return self.settings.embedding_provider == "gguf" and self.settings.embedding_model_path is not None

    def _ensure_embedder(self) -> Tuple[Optional[ManagedEmbedder], list[str]]:
        if self._embedder_loaded:
            return self._embedder, []
        self._embedder_loaded = True
        if not self._embedding_configured():
            return None, []
        if not self.settings.enable_sqlite_vec:
            warning = "embedding configured but vec.enabled=false; auto-embedding disabled. Set vec.enabled=true to enable."
            self._embedder_warnings.append(warning)
            return None, [warning]
        from .embedder import build_embedder

        assert self.settings.embedding_model_path is not None
        self._embedder, warnings = build_embedder(
            str(self.settings.embedding_model_path),
            self.settings.vec_dim,
            n_ctx=getattr(self.settings, "embedding_n_ctx", 2048),
            reserved_tokens=getattr(self.settings, "embedding_reserved_tokens", 64),
            max_section_chars=getattr(self.settings, "max_section_chars", 3600),
        )
        self._embedder_warnings.extend(warnings)
        return self._embedder, warnings

    @staticmethod
    def _embedding_text(record: dict[str, Any]) -> str:
        subject = record.get("subject") or ""
        content = record.get("content") or ""
        return f"{subject}\n{content}".strip()

    def memory_write(self, **payload: Any) -> dict[str, Any]:
        allowed, warnings = self._allowed(payload.get("agent_id"), payload.get("client"))
        if not allowed:
            return self.db.state.response({"written": False}, ok=False, extra_warnings=warnings)
        try:
            record = MemoryRecord.from_input(payload, self.settings.defaults())
            memory_id, write_warnings = self.db.insert_memory(record)
            data = {"id": memory_id, "backup_only": memory_id is None, "record": {**record.__dict__, "id": memory_id}}
            embedding_warnings: list[str] = []
            if memory_id is not None and self.settings.embedding_auto_write and self._embedding_configured():
                data["embedding_stored"] = False
                embedder, ensure_warnings = self._ensure_embedder()
                embedding_warnings.extend(ensure_warnings)
                if embedder is not None:
                    try:
                        er = embedder.embed_text(
                            prefix=record.subject or "",
                            body=record.content,
                        )
                        if not er.embedding:
                            raise RuntimeError(
                                f"encode returned empty embedding: {getattr(embedder, 'last_encode_error', None) or 'unknown'}"
                            )
                        data["embedding_stored"], store_warnings = self.db.store_embedding(memory_id, er.embedding)
                        embedding_warnings.extend(store_warnings)
                        if er.truncated:
                            embedding_warnings.append(
                                f"memory embedding truncated: {er.used_tokens}/{er.original_tokens} tokens"
                            )
                    except Exception as exc:
                        embedding_warnings.append(f"auto-embedding write failed: {exc}")
            # v0.6.0: split_hint
            if (
                memory_id is not None
                and getattr(self.settings, "split_enabled", False)
                and len(record.content) > getattr(self.settings, "split_threshold", 4000)
            ):
                vec_state = self.db.get_vec_index_state()
                if vec_state.get("state") == "ready":
                    data["split_hint"] = {
                        "char_count": len(record.content),
                        "split_threshold": getattr(self.settings, "split_threshold", 4000),
                        "prompt": (
                            f"已保存。该记忆 {len(record.content)} 字符，"
                            "分段可提升检索精度。如需分段，调用 memory_split(memory_id="
                            f"{memory_id})。"
                        ),
                        "memory_id": memory_id,
                    }
            return self.db.state.response(self._enrich_write_response(data, memory_id, record), extra_warnings=warnings + write_warnings + embedding_warnings)
        except Exception as exc:
            return self.db.state.response({"error": str(exc)}, ok=False, extra_warnings=warnings)

    def _enrich_write_response(
        self, data: dict[str, Any], memory_id: Optional[int], record: MemoryRecord,
    ) -> dict[str, Any]:
        """v0.7.6: post-write enrichment — attach write_hints if duplicates found.

        Never raises; hint failures are silently swallowed (hint is advisory).
        """
        if memory_id is None:
            return data
        try:
            hints = self._write_duplicate_hints(memory_id, record)
            if hints:
                data["write_hints"] = hints
        except Exception:
            pass
        return data

    def _write_duplicate_hints(
        self, memory_id: int, record: MemoryRecord,
    ) -> Optional[dict[str, Any]]:
        """Detect possible duplicates/evolution of the just-written memory.

        Returns ``{possible_supersede_targets: [...]}`` or None if no
        candidates found. Uses DB candidate recall + Python overlap scoring.
        """
        candidates = self.db.find_metadata_overlap_candidates(
            subject=record.subject,
            tags=record.tags,
            exclude_id=memory_id,
        )
        if not candidates:
            return None
        new_content = record.content or ""
        targets: list[dict[str, Any]] = []
        my_tags = set(record.tags or [])
        my_subject_tokens = set((record.subject or "").lower().split())
        for cand in candidates:
            cand_tags = set(cand.get("tags") or [])
            cand_subject_tokens = set((cand.get("subject") or "").lower().split())
            # Tags Jaccard.
            if my_tags and cand_tags:
                common_tags = my_tags & cand_tags
                if len(common_tags) < 2:
                    continue
                tags_jaccard = len(common_tags) / len(my_tags | cand_tags)
                if tags_jaccard < 0.8:
                    continue
            else:
                continue
            # Subject overlap.
            if my_subject_tokens and cand_subject_tokens:
                subj_overlap = len(my_subject_tokens & cand_subject_tokens) / len(my_subject_tokens | cand_subject_tokens)
                if subj_overlap < 0.7:
                    continue
            # Determine hint type.
            cand_content = cand.get("content") or ""
            hint_type = "possible_duplicate"
            reason = f"tags Jaccard {len(my_tags & cand_tags)}/{len(my_tags | cand_tags)} + subject overlap"
            if len(new_content) >= len(cand_content) * 1.3:
                hint_type = "possible_evolution_of"
                reason += "; new content ≥1.3× candidate"
            targets.append({
                "id": int(cand["id"]),
                "subject": cand.get("subject"),
                "reason": reason,
                "hint_type": hint_type,
            })
            if len(targets) >= 3:
                break
        if not targets:
            return None
        return {"possible_supersede_targets": targets}

    def memory_search(self, query: str = "", workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, include_superseded: bool = False, debug_ranking: bool = False, query_embedding: Optional[list[float]] = None, tags_filter: Optional[list[str]] = None, after_time: Optional[str] = None, before_time: Optional[str] = None, source_type: Optional[str] = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True, **_: Any) -> dict[str, Any]:
        extra_warnings = list(self._embedder_warnings)
        vec_state = self.db.get_vec_index_state()
        vec_disabled = vec_state.get("state") in {"mismatch", "failed"}
        if vec_disabled and (query_embedding is not None or (query and self.settings.embedding_auto_query)):
            disabled_reason = (
                "embedding_space_mismatch"
                if vec_state.get("state") == "mismatch"
                else "embedding_migration_failed"
            )
            extra_warnings.append(f"vec_disabled={disabled_reason}")
            query_embedding = None
        elif query_embedding is None and query and self.settings.embedding_auto_query:
            embedder, ensure_warnings = self._ensure_embedder()
            extra_warnings.extend(ensure_warnings)
            if embedder is not None:
                try:
                    er = embedder.embed_text(prefix="", body=query)
                    if er.embedding:
                        query_embedding = er.embedding
                    else:
                        extra_warnings.append(
                            f"auto-embedding query failed: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
                        )
                except Exception as exc:
                    extra_warnings.append(f"auto-embedding query failed: {exc}")
        # v0.7.4 (M2): search_memories now returns a SearchOutcome dataclass.
        outcome = search_memories(
            self.db, query, workspace, tags, limit,
            include_superseded=include_superseded,
            debug_ranking=debug_ranking,
            query_embedding=query_embedding,
            tags_filter=tags_filter,
            after_time=after_time,
            before_time=before_time,
            source_type=source_type,
        )
        results = outcome.results
        warnings = outcome.warnings
        has_more = outcome.has_more
        total_estimate = outcome.total_estimate
        retrieval_mode = outcome.retrieval_mode
        # v0.6.0: attach section enhancement to active-split results
        results = self._attach_sections(results, query_embedding, extra_warnings)
        # v0.7.6: attach conflict signals (open_table + runtime_metadata_hint),
        # only on genuine query hits (direct mode). Failures degrade silently.
        if include_conflict_signal and retrieval_mode == "direct" and results:
            results = self._attach_conflict_signals(results, extra_warnings)
        # v0.7.4: linked_open_items — only on genuine query hits (direct mode),
        # never on browse/fallback/empty. Failures degrade to [] + warning.
        linked: list[dict[str, Any]] = []
        if include_linked_open_items and retrieval_mode == "direct" and results:
            linked = _linked_open_items_for_search(self.db, results, extra_warnings)
        return self.db.state.response(
            {
                "results": results,
                "count": len(results),
                # v0.7.3: exhaustive-query support (design §3.6)
                "has_more": has_more,
                "total_estimate": total_estimate,
                # v0.7.4 (M2): expose retrieval_mode so callers know how rows were produced.
                "retrieval_mode": retrieval_mode,
                # v0.7.4: related active todos, separated from the ranking engine.
                "linked_open_items": linked,
            },
            extra_warnings=extra_warnings + warnings,
        )

    def memory_get(self, memory_id: int, **_: Any) -> dict[str, Any]:
        """通过 ID 直接获取一条记忆的完整信息。只读，不修改任何数据。"""
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer"}, ok=False)
        memory = self.db.get_memory(memory_id_int)
        if not memory:
            return self.db.state.response({"error": f"memory id {memory_id_int} not found"}, ok=False)
        return self.db.state.response({"memory": memory})

    def memory_store_embedding(self, memory_id: int, embedding: list[float], **_: Any) -> dict[str, Any]:
        """Store or replace an embedding for a memory (v0.3.1 semantic recall).

        The caller is responsible for generating the embedding with any model
        of matching dimension. memory-arbiter does not bundle an embedding
        model by design (local-first, zero cloud, no heavy deps). See
        docs/semantic_example.py for a backfill script using sentence-transformers.
        """
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer"}, ok=False)
        if not isinstance(embedding, list) or not embedding:
            return self.db.state.response({"error": "embedding must be a non-empty list of floats"}, ok=False)
        if not self.db.get_memory(memory_id_int):
            return self.db.state.response({"error": f"memory id {memory_id_int} not found"}, ok=False)
        ok, store_warnings = self.db.store_embedding(memory_id_int, embedding)
        return self.db.state.response({"stored": ok, "memory_id": memory_id_int, "dimensions": len(embedding)}, ok=ok, extra_warnings=store_warnings)

    def memory_recent(self, workspace: Optional[str] = None, limit: int = 20, **_: Any) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        # v0.7.4 (M3): workspace is reserved metadata; memory_recent lists
        # across the whole library (shared memory layer). The parameter stays
        # in the signature for interface stability but does not filter.
        results = self.db.list_memories(limit=limit)
        return self.db.state.response({"results": results, "count": len(results)})

    def memory_compare(self, left_id: Optional[int] = None, right_id: Optional[int] = None, left: Optional[dict[str, Any]] = None, right: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
        left_record = left or (self.db.get_memory(int(left_id)) if left_id is not None else None)
        right_record = right or (self.db.get_memory(int(right_id)) if right_id is not None else None)
        if not left_record or not right_record:
            return self.db.state.response({"error": "left and right records are required"}, ok=False)
        return self.db.state.response({"comparison": compare_memories(left_record, right_record), "left": left_record, "right": right_record})

    def memory_arbitrate(self, left_id: int, right_id: int, mark_conflict: bool = True, apply: bool = False, **_: Any) -> dict[str, Any]:
        left = self.db.get_memory(int(left_id))
        right = self.db.get_memory(int(right_id))
        if not left or not right:
            return self.db.state.response({"error": "memory id not found"}, ok=False)
        comparison = compare_memories(left, right)
        conflict_id = None
        if mark_conflict:
            reason = "; ".join(comparison["reasons"])
            conflict_id = self.db.record_conflict(int(left_id), int(right_id), left.get("subject") or right.get("subject"), reason, comparison["winner_id"])
        applied = False
        if apply and comparison["winner_id"] and comparison["loser_id"] and not comparison["manual_review"]:
            loser = self.db.get_memory(int(comparison["loser_id"]))
            if loser and loser.get("protection_level") != ProtectionLevel.LOCKED.value and loser.get("source_type") != SourceType.USER_CONFIRMED.value:
                applied = self.db.update_memory(int(comparison["loser_id"]), {"status": "superseded"})
        return self.db.state.response({"comparison": comparison, "conflict_id": conflict_id, "applied": applied})

    def memory_list_conflicts(self, status: str = "open", limit: int = 50, **_: Any) -> dict[str, Any]:
        conflicts = self.db.list_conflicts(status=status, limit=int(limit))
        return self.db.state.response({"conflicts": conflicts, "count": len(conflicts)})

    def memory_scan_conflict_candidates(
        self,
        workspace: Optional[str] = None,
        top_k: int = 8,
        max_pairs: int = 200,
        max_distance: float = 12.0,
        incremental: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        """v0.7.5 (id=243): vector-recall candidate conflict pairs (no LLM).

        Returns up to ``max_pairs`` candidate pairs ranked by vector distance.
        Pairs are canonicalised (left<right), filtered to same workspace, and
        truncated. Writes a ``scan_log.jsonl`` entry for doctor freshness
        tracking. When sqlite-vec is unavailable, returns a normal
        ``scanned=False`` with a hint (config state, not an error). The agent
        is expected to run LLM comparison on each pair, then call
        ``memory_record_conflict`` to persist the verdict.
        """
        result = self.db.scan_conflict_candidates(
            workspace=workspace,
            top_k=int(top_k),
            max_pairs=int(max_pairs),
            max_distance=float(max_distance),
            incremental=bool(incremental),
        )
        return self.db.state.response(result)

    def memory_record_conflict(
        self,
        left_id: int,
        right_id: int,
        reason: str,
        conflict_type: Optional[str] = None,
        conflict_point: Optional[str] = None,
        suggested_winner: Optional[int] = None,
        confidence_hint: Optional[str] = None,
        source: Optional[str] = None,
        refresh: bool = False,
        left_version: Optional[int] = None,
        right_version: Optional[int] = None,
        scan_prompt_version: Optional[str] = None,
        scan_model: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """v0.7.5/v0.7.6: persist a conflict with scan-enrichment fields.

        Pairs are canonicalised (left<right). Idempotent: if an open conflict
        on the same pair already exists, returns ``deduped`` without writing.
        Pass ``refresh=True`` to update the existing row's enrichment fields
        in place (returns ``refreshed``); use when the scan task re-runs LLM
        after a memory version or model change. The ``source`` field (e.g.
        ``"llm_informed"``) records whether the suggestion came from an LLM
        that read the content or from a metadata heuristic. ``conflict_type``
        can be ``contradiction``, ``evolution`` (stale_active_memory — should
        supersede but both still active), or other.
        """
        result = self.db.record_conflict_enriched(
            int(left_id), int(right_id),
            conflict_type=conflict_type,
            conflict_point=conflict_point,
            reason=reason,
            suggested_winner=int(suggested_winner) if suggested_winner is not None else None,
            confidence_hint=confidence_hint,
            source=source,
            refresh=refresh,
            left_version=left_version,
            right_version=right_version,
            scan_prompt_version=scan_prompt_version,
            scan_model=scan_model,
        )
        return self.db.state.response(result)

    def memory_resolve_conflict(
        self,
        conflict_id: int,
        reason: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        """v0.7.5 (id=243): close a single open conflict by id (dismiss).

        Unlike ``memory_supersede`` (which resolves all conflicts touching a
        memory via ``resolve_conflicts_for``), this targets exactly one
        conflict row — used to dismiss a false positive without touching
        either memory.
        """
        result = self.db.resolve_conflict(int(conflict_id), reason=reason)
        return self.db.state.response(result)

    def memory_confirm(self, memory_id: int, source_ref: Optional[str] = None, confidence: float = 1.0, **_: Any) -> dict[str, Any]:
        memory = self.db.get_memory(int(memory_id))
        if not memory:
            return self.db.state.response({"error": "memory id not found"}, ok=False)
        metadata = dict(memory.get("metadata") or {})
        metadata["confirmed_from"] = source_ref or "manual"
        ok = self.db.update_memory(
            int(memory_id),
            {
                "source_type": SourceType.USER_CONFIRMED.value,
                "confidence": float(confidence),
                "protection_level": ProtectionLevel.LOCKED.value,
                "status": "active",
                "metadata": metadata,
            },
        )
        updated = self.db.get_memory(int(memory_id)) if ok else memory
        return self.db.state.response({"confirmed": ok, "record": updated})

    def memory_supersede(
        self,
        memory_id: int,
        reason: str,
        superseded_by: Optional[int] = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Explicitly supersede a memory, bypassing the user-confirmed/locked
        protection that blocks ``memory_arbitrate``. Requires ``authorized=True``.

        Side effects: status -> superseded, protection_level -> normal, all open
        conflicts involving this memory are resolved, and an audit row is appended
        to the conflicts table (reason prefixed with ``USER-AUTHORIZED SUPERSEDE``).
        """
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to supersede a memory", "superseded": False},
                ok=False,
            )
        memory = self.db.get_memory(int(memory_id))
        if not memory:
            return self.db.state.response({"error": "memory id not found", "superseded": False}, ok=False)
        if memory.get("status") in {"superseded", "deleted"}:
            return self.db.state.response(
                {"error": f"memory already {memory.get('status')}", "superseded": False},
                ok=False,
            )
        if superseded_by is not None:
            replacement = self.db.get_memory(int(superseded_by))
            if not replacement:
                return self.db.state.response(
                    {"error": "superseded_by memory id not found", "superseded": False},
                    ok=False,
                )
            # Guard against supersede-chain breakage: starting in v0.2.6,
            # memory_search filters out superseded records by default. If the
            # replacement target is itself deleted/superseded, the new default
            # would leave the chain pointing at a record that search can't see
            # — the user would lose both the old and the new view. Reject early
            # with an explicit error so the caller picks a live replacement.
            if replacement.get("status") != "active":
                return self.db.state.response(
                    {"error": f"superseded_by target is not active (status={replacement.get('status')}); pick a live replacement to avoid a broken chain", "superseded": False},
                    ok=False,
                )

        self.db.update_memory(
            int(memory_id),
            {"status": "superseded", "protection_level": ProtectionLevel.NORMAL.value},
        )
        resolved = self.db.resolve_conflicts_for(int(memory_id))
        audit_reason = f"USER-AUTHORIZED SUPERSEDE: {reason}"
        conflict_id = self.db.record_conflict(
            int(memory_id),
            int(superseded_by) if superseded_by is not None else int(memory_id),
            memory.get("subject"),
            audit_reason,
            int(superseded_by) if superseded_by is not None else None,
            status="resolved",
        )
        updated = self.db.get_memory(int(memory_id))
        return self.db.state.response(
            {
                "superseded": True,
                "memory_id": int(memory_id),
                "linked_conflicts_resolved": resolved,
                "conflict_id": conflict_id,
                "record": updated,
            }
        )

    def memory_status(self, **_: Any) -> dict[str, Any]:
        vec_state = self.db.get_vec_index_state()
        return self.db.state.response(
            {
                "db_path": str(self.settings.db_path),
                "backup_jsonl": str(self.settings.backup_jsonl),
                "sqlite_vec_available": self.db.state.sqlite_vec_available,
                "fts5_available": self.db.state.fts5_available,
                "sqlite_writable": self.db.state.sqlite_writable,
                "jsonl_backup_active": self.db.state.jsonl_backup_active,
                "client": self.settings.client,
                "agent_id": self.settings.agent_id,
                "workspace": self.settings.workspace,
                "config_warnings": self.settings.config_warnings,
                "embedding_configured": self._embedding_configured(),
                "embedding_auto_query": self.settings.embedding_auto_query,
                "embedding_auto_write": self.settings.embedding_auto_write,
                "split_enabled": getattr(self.settings, "split_enabled", False),
                "vec_index_state": vec_state,
                "policy": {
                    "client_defaults": self.settings.policy.client_defaults,
                    "default_enabled": self.settings.policy.default_enabled,
                    "allow_agents": self.settings.policy.allow_agents,
                    "deny_agents": self.settings.policy.deny_agents,
                },
            },
            extra_warnings=self.settings.config_warnings,
        )

    def memory_doctor_overview(self, deep: bool = False, **_: Any) -> dict[str, Any]:
        """Run a read-only health check and return a graded diagnostic report.

        Covers config integrity, the vector-enablement chain, split, data
        consistency, and capacity. Each finding carries a severity and a
        fix_hint tailored to the current config.json. Read-only: never writes,
        never changes schema. ``deep=true`` additionally loads the GGUF model
        for a dimension probe (seconds-level cost); MCP reuses an
        already-loaded embedder at zero cost.
        """
        from .doctor import doctor_overview_mcp, report_to_dict

        report = doctor_overview_mcp(
            self.db, self.settings, deep,
            embedder_probe=self._ensure_embedder,
            runtime_state=self.db.state,
        )
        return self.db.state.response(report_to_dict(report))

    def memory_audit_summary(self, **_: Any) -> dict[str, Any]:
        summary = self.db.audit_summary()
        return self.db.state.response(summary)

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
        """In-place edit a memory's content or tags.

        Edit modes:
          * tags-only (v0.7.6): pass ``tags_only=True`` with
            ``add_tags``/``remove_tags`` to update tags without touching
            content, memory_history, version, embeddings, or sections.
            FTS is re-synced because tags are indexed in FTS5.
          * full replace: pass ``new_content`` (old_text/new_text must be empty)
          * partial replace: pass ``old_text`` + ``new_text`` for an exact
            substring substitution (new_content must be empty)

        Authorization (layered): normal records edit freely; ``locked`` /
        ``user_confirmed`` records require ``authorized=True`` (mirrors
        ``memory_supersede``). Records already superseded/deleted are rejected.
        """
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer", "edited": False}, ok=False)

        # ---- tags-only fast path (v0.7.6) ----
        if tags_only:
            result = self.db.update_tags_low_side_effect(
                memory_id_int,
                add_tags=add_tags or [],
                remove_tags=remove_tags or [],
                authorized=authorized,
            )
            outcome = result.get("outcome")
            if outcome == "updated":
                updated_mem = self.db.get_memory(memory_id_int)
                return self.db.state.response({
                    "edited": True,
                    "tags_only": True,
                    "memory_id": memory_id_int,
                    "tags": result.get("tags"),
                    "record": updated_mem,
                })
            if outcome == "no_change":
                return self.db.state.response({
                    "edited": False,
                    "tags_only": True,
                    "already_completed": True,
                    "memory_id": memory_id_int,
                    "tags": result.get("tags"),
                })
            if outcome == "forbidden":
                return self.db.state.response({
                    "error": (
                        f"memory is protected (protection_level={result.get('protection_level')}, "
                        f"source_type={result.get('source_type')}); authorized=True required to edit tags"
                    ),
                    "edited": False,
                }, ok=False)
            if outcome == "not_found":
                return self.db.state.response({"error": f"memory id {memory_id_int} not found", "edited": False}, ok=False)
            if outcome == "not_active":
                return self.db.state.response({
                    "error": f"memory is not active (status={result.get('status')}); cannot edit tags",
                    "edited": False,
                }, ok=False)
            if outcome == "unavailable":
                return self.db.state.response({"error": "database not available", "edited": False}, ok=False)
            # outcome == "error"
            return self.db.state.response(
                {"error": "tags-only edit failed; transaction rolled back, no changes applied", "edited": False},
                ok=False,
            )

        # ---- full / partial content edit (existing path) ----
        memory = self.db.get_memory(memory_id_int)
        if not memory:
            return self.db.state.response({"error": "memory id not found", "edited": False}, ok=False)
        if memory.get("status") in {"superseded", "deleted"}:
            return self.db.state.response(
                {"error": f"memory already {memory.get('status')}", "edited": False},
                ok=False,
            )
        is_protected = (
            memory.get("protection_level") == ProtectionLevel.LOCKED.value
            or memory.get("source_type") == SourceType.USER_CONFIRMED.value
        )
        if is_protected and not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to edit a locked/user_confirmed memory", "edited": False},
                ok=False,
            )
        # Resolve the resulting content from the two edit modes.
        current_content = memory.get("content") or ""
        if new_content is not None and (old_text or new_text):
            return self.db.state.response(
                {"error": "pass either new_content (full replace) or old_text+new_text (partial), not both", "edited": False},
                ok=False,
            )
        if new_content is not None:
            resolved_content = new_content
        elif old_text is not None and new_text is not None:
            if old_text not in current_content:
                return self.db.state.response(
                    {"error": "old_text not found in current content", "edited": False},
                    ok=False,
                )
            resolved_content = current_content.replace(old_text, new_text, 1)
        else:
            return self.db.state.response(
                {"error": "provide new_content for full replace, or old_text+new_text for partial replace, or tags_only=true", "edited": False},
                ok=False,
            )
        history_id = self.db.edit_memory(
            memory_id_int,
            resolved_content,
            new_subject=new_subject,
            new_tags=new_tags,
            reason=reason or None,
        )
        if history_id is None:
            return self.db.state.response({"error": "edit failed (db not writable)", "edited": False}, ok=False)
        updated = self.db.get_memory(memory_id_int)
        embedding_warnings: list[str] = []
        embedding_stored: Optional[bool] = None
        if self.settings.embedding_auto_write and self._embedding_configured():
            embedding_stored = False
            embedder, ensure_warnings = self._ensure_embedder()
            embedding_warnings.extend(ensure_warnings)
            if embedder is None:
                _deleted, delete_warnings = self.db.delete_embedding(memory_id_int)
                embedding_warnings.extend(delete_warnings)
                embedding_warnings.append("re-embedding on edit skipped because embedder unavailable; deleted stale embedding to avoid dirty recall.")
            elif updated is not None:
                try:
                    embedding_result = embedder.embed_text(
                        prefix=updated.get("subject") or "",
                        body=updated.get("content") or "",
                    )
                    if not embedding_result.embedding:
                        raise RuntimeError(
                            f"encode returned empty embedding: {getattr(embedder, 'last_encode_error', None) or 'unknown'}"
                        )
                    embedding_stored, store_warnings = self.db.store_embedding(memory_id_int, embedding_result.embedding)
                    embedding_warnings.extend(store_warnings)
                    if not embedding_stored:
                        _deleted, delete_warnings = self.db.delete_embedding(memory_id_int)
                        embedding_warnings.extend(delete_warnings)
                        embedding_warnings.append("re-embedding on edit failed; deleted stale embedding to avoid dirty recall.")
                except Exception as exc:
                    _deleted, delete_warnings = self.db.delete_embedding(memory_id_int)
                    embedding_warnings.extend(delete_warnings)
                    embedding_warnings.append(f"re-embedding on edit failed: {exc}; deleted stale embedding to avoid dirty recall.")
        data = {
            "edited": True,
            "memory_id": memory_id_int,
            "new_version": int(updated.get("version") or 1) if updated else None,
            "history_id": history_id,
            "record": updated,
        }
        if embedding_stored is not None:
            data["embedding_stored"] = embedding_stored
        return self.db.state.response(
            data,
            extra_warnings=embedding_warnings,
        )

    def memory_history(self, memory_id: int, **_: Any) -> dict[str, Any]:
        """View the version-chain (historical snapshots) of a memory, newest
        version first. Read-only; does not modify any table.
        """
        memory = self.db.get_memory(int(memory_id))
        if not memory:
            return self.db.state.response({"error": "memory id not found"}, ok=False)
        history = self.db.list_history(int(memory_id))
        return self.db.state.response(
            {
                "memory_id": int(memory_id),
                "current_version": int(memory.get("version") or 1),
                "history": history,
                "count": len(history),
            }
        )

    def memory_cleanup_history(
        self,
        memory_id: Optional[int] = None,
        older_than_days: Optional[int] = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Delete historical snapshots from ``memory_history``.

        Scope:
          * ``memory_id`` set: clean only that memory's history
          * ``older_than_days`` set: clean only snapshots older than N days
          * both set: both filters apply
          * neither set (full cleanup): **requires ``authorized=True``** as an
            explicit confirmation gate

        SAFETY: this tool only ever deletes from memory_history. The memories
        table (active records) is never touched, regardless of arguments.
        """
        full_cleanup = memory_id is None and older_than_days is None
        if older_than_days is not None and int(older_than_days) < 0:
            return self.db.state.response(
                {"error": "older_than_days must be >= 0", "cleaned": 0},
                ok=False,
            )
        if full_cleanup and not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required for full history cleanup (no memory_id / older_than_days filter)", "cleaned": 0},
                ok=False,
            )
        cleaned = self.db.cleanup_history(memory_id=memory_id, older_than_days=older_than_days)
        scope = "full" if full_cleanup else ("memory" if memory_id is not None else "by_age")
        return self.db.state.response(
            {
                "cleaned": cleaned,
                "scope": scope,
                "memory_id": memory_id,
                "older_than_days": older_than_days,
            }
        )

    # ==================================================================
    #  v0.7.6: Conflict-signal attachment for search results
    # ==================================================================

    # Trust rank for runtime_metadata_hint (higher = more authoritative).
    _TRUST_RANK: dict[str, int] = {
        "locked": 100,
        "user_confirmed": 100,
        "document_extracted": 70,
        "agent_generated": 45,
        "pending": 20,
        "unknown": 10,
    }

    def _trust_score(self, record: dict[str, Any]) -> int:
        """Composite trust rank from source_type + protection_level."""
        st = self._TRUST_RANK.get(record.get("source_type", ""), 0)
        pl = self._TRUST_RANK.get(record.get("protection_level", ""), 0)
        return max(st, pl)

    @staticmethod
    def _confidence_rank(hint: Optional[str]) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(hint or "", 0)

    def _attach_conflict_signals(
        self,
        results: list[dict[str, Any]],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """v0.7.6: attach conflict_signal to each direct-mode search result.

        Two sources, strongly distinguished by ``conflict_source``:
          * ``open_table``: conflict already in the conflicts table (written by
            scan + record_conflict). Carries structured fields.
          * ``runtime_metadata_hint``: computed on-the-fly from subject/tags
            overlap + trust disparity. **Not LLM-verified** — advisory only.

        open_table takes priority. Both attach a ``conflict_peer`` summary so
        the caller knows who the conflict is with, even if the peer was cut by
        ``limit``. Never raises; failures degrade to no signal.
        """
        if not results:
            return results
        try:
            result_ids = [int(r["id"]) for r in results if r.get("id") is not None]
            if not result_ids:
                return results

            # Batch-fetch open conflicts for all result IDs (one SQL, no N+1).
            conflicts = self.db.list_open_conflicts_for_memory_ids(result_ids)
            # Build memory_id → list of conflicts.
            conflicts_by_mem: dict[int, list[dict[str, Any]]] = {}
            all_peer_ids: set[int] = set()
            for c in conflicts:
                left = int(c.get("left_id"))
                right = int(c.get("right_id"))
                conflicts_by_mem.setdefault(left, []).append(c)
                conflicts_by_mem.setdefault(right, []).append(c)
                all_peer_ids.add(left)
                all_peer_ids.add(right)

            # Batch-fetch summaries for all IDs that appear in any conflict.
            summaries: dict[int, dict[str, Any]] = {}
            if all_peer_ids:
                summaries = self.db.get_memory_summaries(list(all_peer_ids))

            # Attach signals.
            result_id_set = set(result_ids)
            for rec in results:
                mid = int(rec["id"])
                if mid in conflicts_by_mem:
                    signal = self._build_open_table_signal(
                        mid, conflicts_by_mem[mid], summaries, result_id_set,
                    )
                    if signal:
                        rec["conflict_signal"] = signal
                        continue
                # No open_table signal → try runtime_metadata_hint.
                hint = self._compute_runtime_hint(mid, rec, results, result_id_set)
                if hint:
                    rec["conflict_signal"] = hint
        except Exception as exc:
            warnings.append(f"conflict_signal attachment failed: {exc}")
        return results

    def _build_open_table_signal(
        self,
        memory_id: int,
        conflicts: list[dict[str, Any]],
        summaries: dict[int, dict[str, Any]],
        result_id_set: set[int],
    ) -> Optional[dict[str, Any]]:
        """Build an open_table conflict_signal for a memory with open conflicts.

        If a memory has multiple open conflicts, pick the primary one by
        confidence_hint > created_at > conflict_id.
        """
        def conflict_sort_key(c: dict[str, Any]) -> tuple:
            return (
                self._confidence_rank(c.get("confidence_hint")),
                str(c.get("created_at", "")),
                int(c.get("id", 0)),
            )

        primary = max(conflicts, key=conflict_sort_key)
        peer_id = int(primary["right_id"]) if primary["left_id"] == memory_id else int(primary["left_id"])
        peer_summary = summaries.get(peer_id, {})
        return {
            "conflict_source": "open_table",
            "conflict_id": int(primary["id"]),
            "conflict_type": primary.get("conflict_type"),
            "conflict_point": primary.get("conflict_point"),
            "suggested_winner": primary.get("suggested_winner"),
            "confidence_hint": primary.get("confidence_hint"),
            "source": primary.get("source"),
            "open_conflict_count": len(conflicts),
            "conflict_peer": {
                "id": peer_id,
                "subject": peer_summary.get("subject"),
                "status": peer_summary.get("status"),
                "snippet": peer_summary.get("snippet"),
            },
        }

    def _compute_runtime_hint(
        self,
        memory_id: int,
        rec: dict[str, Any],
        all_results: list[dict[str, Any]],
        result_id_set: set[int],
    ) -> Optional[dict[str, Any]]:
        """Compute a runtime_metadata_hint by comparing this result against
        other results in the same result set (bounded to first 20).

        Only fires on high subject/tags overlap + trust disparity.
        """
        my_tags = set(rec.get("tags") or [])
        my_subject = (rec.get("subject") or "").lower()
        my_trust = self._trust_score(rec)
        # Cap to avoid O(n²) blowup on large result sets.
        candidates = [r for r in all_results[:20] if int(r.get("id", 0)) != memory_id]
        best_peer: Optional[dict[str, Any]] = None
        best_score = 0.0
        for peer in candidates:
            peer_id = int(peer.get("id", 0))
            peer_tags = set(peer.get("tags") or [])
            peer_subject = (peer.get("subject") or "").lower()
            # Tags overlap.
            if my_tags and peer_tags:
                common = my_tags & peer_tags
                if len(common) >= 2:
                    overlap_ratio = len(common) / len(my_tags | peer_tags)
                    if overlap_ratio >= 0.8:
                        trust_gap = abs(my_trust - self._trust_score(peer))
                        if trust_gap > 0:
                            score = overlap_ratio + trust_gap * 0.01
                            if score > best_score:
                                best_score = score
                                best_peer = peer
            # Subject overlap.
            if my_subject and peer_subject and best_peer is None:
                # Simple token overlap for ASCII.
                my_tokens = set(my_subject.split())
                peer_tokens = set(peer_subject.split())
                if my_tokens and peer_tokens:
                    overlap = len(my_tokens & peer_tokens) / len(my_tokens | peer_tokens)
                    if overlap >= 0.7:
                        trust_gap = abs(my_trust - self._trust_score(peer))
                        if trust_gap > 0:
                            score = overlap + trust_gap * 0.01
                            if score > best_score:
                                best_score = score
                                best_peer = peer
        if best_peer is None:
            return None
        peer_id = int(best_peer.get("id", 0))
        return {
            "conflict_source": "runtime_metadata_hint",
            "conflict_type": "metadata_overlap",
            "confidence_hint": "low",
            "conflict_point": "subject/tags overlap; not LLM-verified",
            "conflict_peer": {
                "id": peer_id,
                "subject": best_peer.get("subject"),
                "status": best_peer.get("status"),
                "snippet": (best_peer.get("content") or "")[:200],
            },
        }

    # ==================================================================
    #  v0.6.0: Section split tools
    # ==================================================================

    @staticmethod
    def _catalog_entry(s: dict) -> dict:
        """Unified section-catalog schema (same shape in zero-match and partial branches)."""
        return {
            "section_id": s["id"],
            "title": s.get("title"),
            "title_path": s.get("title_path"),
            "summary": s.get("summary"),
            "embedding_truncated": bool(s.get("embedding_truncated")),
            "embedding_original_tokens": s.get("embedding_original_tokens", 0),
            "embedding_used_tokens": s.get("embedding_used_tokens", 0),
        }

    def _attach_sections(
        self,
        results: list[dict[str, Any]],
        query_embedding: Optional[list[float]],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """Post-process search results: attach section enhancement for active-split memories.

        v0.8.0 protocol (design doc §6.3). Each result carries a top-level
        ``content`` that is the directly-consumable complete content unit, and
        a ``content_scope`` tag the caller must use to interpret it:

          * full_memory      — the whole memory content
          * matched_sections — the matched sections' full text, joined

        Branch matrix:
          | coverage ≥ threshold            | full_memory   | (matched refs)      | –            |
          | 0 < coverage < threshold        | matched_sections | full section bodies | catalog(unmatched) |
          | memory hit, section zero-match  | full_memory   | –                   | –            |
          | invariant broken / vec gate down| full_memory   | –                   | optional     |

        Ordinary ``matched_sections`` never carry embedding budget diagnostics;
        those live only in catalog/get/doctor (or debug_ranking).
        """
        if not results or not self.db.db_available:
            return results

        vec_state = self.db.get_vec_index_state()
        vec_gate_open = (
            vec_state.get("state") == "ready"
            and query_embedding is not None
            and self.db.state.sqlite_vec_available
        )
        if not query_embedding:
            vec_disabled_reason = "no_query_embedding"
        elif vec_state.get("state") in {"mismatch", "failed"}:
            vec_disabled_reason = (
                "embedding_space_mismatch"
                if vec_state.get("state") == "mismatch"
                else "embedding_migration_failed"
            )
        elif vec_state.get("state") != "ready":
            vec_disabled_reason = "gate_closed_state_not_ready"
        else:
            vec_disabled_reason = "vec_extension_unavailable"
        threshold = getattr(self.settings, "section_vec_distance_threshold", 0.42)
        fulltext_threshold = getattr(self.settings, "section_fulltext_threshold", 0.8)

        active_ids = [
            r.get("id") for r in results
            if r.get("split_status") == "active"
        ]
        if not active_ids:
            return results

        # Read all sections + section vec IDs in one snapshot
        sections_map: dict[int, list[dict]] = {}
        section_vec_ids_map: dict[int, set[int]] = {}
        current_mem_map: dict[int, dict] = {}
        try:
            with self.db.connection() as conn:
                for mid in active_ids:
                    mem = MemoryDB._fetch_memory(conn, mid)
                    if mem is None or mem.get("status") == "deleted":
                        continue
                    if mem.get("split_status") != "active":
                        continue
                    current_mem_map[mid] = mem
                    sections_map[mid] = MemoryDB._get_sections(conn, mid)
                    section_vec_ids_map[mid] = MemoryDB._get_section_vec_ids(conn, mid)
        except Exception as exc:
            warnings.append(f"attach_sections read failed: {exc}")
            return results

        for result in results:
            mid = result.get("id")
            if mid not in current_mem_map:
                continue

            # Normalise content: Channel-6 candidates carry content="" upstream,
            # but current_mem_map has the full text. Ensure the result content is
            # the real full text before branching.
            real_content = current_mem_map[mid].get("content") or ""
            if real_content and not result.get("content"):
                result["content"] = real_content
            full_content = result.get("content") or ""

            sections = sections_map.get(mid, [])
            total_sections = len(sections)
            sec_by_id: dict[int, dict] = {s["id"]: s for s in sections}

            # ---- Invariant guards: always return full memory, but flag the
            # corruption so it is detectable regardless of the vec gate.
            if total_sections == 0:
                result.setdefault("warnings", []).append("split_invariant_broken_empty_sections")
                result["content_scope"] = "full_memory"
                result["section_enhancement_applied"] = False
                result["content"] = full_content
                continue
            if total_sections == 1:
                result.setdefault("warnings", []).append("split_invariant_broken_too_few_sections")
                result["content_scope"] = "full_memory"
                result["section_enhancement_applied"] = False
                result["content"] = full_content
                continue
            section_ids = {s["id"] for s in sections}
            vec_ids = section_vec_ids_map.get(mid, set())
            if section_ids - vec_ids:
                result.setdefault("warnings", []).append("split_invariant_broken_missing_section_vec")
                result["content_scope"] = "full_memory"
                result["section_enhancement_applied"] = False
                result["content"] = full_content
                continue

            # ---- Vec gate closed → return full memory (explicit degrade).
            if not vec_gate_open:
                result.setdefault("warnings", []).append(f"vec_disabled={vec_disabled_reason}")
                result["content_scope"] = "full_memory"
                result["section_enhancement_applied"] = False
                result["content"] = full_content
                continue

            # ---- Section vec matching
            try:
                vec_hits = self.db.section_vec_distance_match(mid, query_embedding, threshold)
            except Exception:
                vec_hits = []

            matched_ids = {h["section_id"] for h in vec_hits}
            matched_count = len(matched_ids)

            # Build matched_sections with FULL section bodies, ordered by index.
            # Embedding diagnostics are deliberately omitted from ordinary search.
            def _matched_entry(h: dict) -> dict:
                s = sec_by_id.get(h["section_id"], {})
                body = full_content[s.get("start_offset", 0):s.get("end_offset", 0)] if s else ""
                return {
                    "section_id": h["section_id"],
                    "section_index": s.get("section_index"),
                    "title": s.get("title"),
                    "title_path": s.get("title_path"),
                    "summary": s.get("summary"),
                    "content": body,
                    "char_count": len(body),
                }

            if matched_count == 0:
                # Zero section match → return the FULL memory (design §6.3).
                # No preview, no truncation.
                result["content_scope"] = "full_memory"
                result["content"] = full_content
                result["section_enhancement_applied"] = True
                result["zero_section_match"] = True
                result["hint"] = (
                    f"已拆分为 {total_sections} 段，零段落命中阈值，已返回完整全文"
                )
            elif matched_count / total_sections >= fulltext_threshold:
                # Coverage ≥ threshold → return full memory.
                ordered = sorted(vec_hits, key=lambda h: (sec_by_id.get(h["section_id"], {}).get("section_index", 0)))
                result["content_scope"] = "full_memory"
                result["content"] = full_content
                result["section_enhancement_applied"] = True
                result["matched_sections"] = [_matched_entry(h) for h in ordered]
                pct = round(100 * matched_count / total_sections)
                result["hint"] = f"{pct}% 段落命中，建议直接看全文"
            else:
                # Partial match → join matched sections' full text by index.
                ordered = sorted(vec_hits, key=lambda h: (sec_by_id.get(h["section_id"], {}).get("section_index", 0)))
                matched = [_matched_entry(h) for h in ordered]
                joined = "\n\n".join(m["content"] for m in matched)
                result["content_scope"] = "matched_sections"
                result["content"] = joined
                result["section_enhancement_applied"] = True
                result["matched_sections"] = matched
                result["matched_section_count"] = len(matched)
                result["total_section_count"] = total_sections
                # Catalog of UNMATCHED sections (diagnostic fields allowed here).
                result["section_catalog"] = [
                    self._catalog_entry(s) for s in sections if s["id"] not in matched_ids
                ]
                result["hint"] = "已返回命中段落完整原文；未命中段落见 section_catalog"

        return results

    @staticmethod
    def _find_nth_occurrence(text: str, anchor: str, occurrence: int) -> int:
        """Find the start position of the n-th (0-based) occurrence of anchor in text."""
        start = 0
        for i in range(occurrence + 1):
            pos = text.find(anchor, start)
            if pos == -1:
                return -1
            if i == occurrence:
                return pos
            start = pos + 1
        return -1

    def _compute_offsets(
        self,
        content: str,
        sections_data: list[dict[str, Any]],
    ) -> Optional[list[dict[str, Any]]]:
        """Compute global offsets from LLM-provided anchors.

        Returns list of {start_offset, end_offset, ...section_data} or None on
        failure.  v0.6.0 is single-batch: every anchor is located in the full
        content and its offset is already global.  The caller must not supply
        start_offset/end_offset — only anchors.
        """
        result: list[dict[str, Any]] = []
        for i, sec in enumerate(sections_data):
            if i == 0:
                local_start = 0
            else:
                anchor = sec.get("anchor_text")
                occ = sec.get("occurrence_index", 0)
                if not anchor:
                    return None
                local_start = self._find_nth_occurrence(content, anchor, occ)
                if local_start == -1:
                    return None
            result.append({**sec, "start_offset": local_start})

        # Derive end_offsets
        for i in range(len(result)):
            if i < len(result) - 1:
                result[i]["end_offset"] = result[i + 1]["start_offset"]
            else:
                result[i]["end_offset"] = len(content)

        # Validate
        offsets = [(r["start_offset"], r["end_offset"]) for r in result]
        for i in range(len(offsets)):
            if offsets[i][0] >= offsets[i][1]:
                return None
            if i > 0 and offsets[i][0] != offsets[i - 1][1]:
                return None
        if offsets[0][0] != 0 or offsets[-1][1] != len(content):
            return None
        # Check strict increase
        starts = [o[0] for o in offsets]
        if starts != sorted(set(starts)):
            return None
        # Coverage
        if "".join(content[s:e] for s, e in offsets) != content:
            return None

        return result

    @staticmethod
    def _detect_markdown_headings(content: str) -> list[tuple[int, str]]:
        """Detect ATX headings outside fenced code blocks.

        Returns list of (char_offset, heading_text) where heading_text is the
        heading text with the ``#`` prefix stripped (e.g. ``"标题"`` not
        ``"## 标题"``). Consumers compare against section titles/anchors.
        """
        in_fence = False
        fence_marker = None
        headings: list[tuple[int, str]] = []
        pos = 0
        for line in content.splitlines(keepends=True):
            stripped = line.lstrip()
            # Fence tracking
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                    fence_marker = None
            elif not in_fence:
                m = re.match(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$", line.rstrip())
                if m:
                    headings.append((pos, m.group(2).strip()))
            pos += len(line)
        return headings

    @staticmethod
    def _split_snapshot_error(
        memory: dict[str, Any],
        decision_content_hash: Optional[str],
        decision_memory_version: Optional[int],
        decision_split_status: Optional[str],
        decision_split_revision: Optional[int],
        allowed_split_statuses: tuple[Optional[str], ...],
    ) -> Optional[str]:
        """Validate a caller's prepare snapshot against the current row."""
        if (
            not decision_content_hash
            or decision_memory_version is None
            or decision_split_revision is None
        ):
            return "decision snapshot fields are required"
        current_hash = hashlib.sha256(
            str(memory.get("content") or "").encode("utf-8")
        ).hexdigest()
        if (
            memory.get("status") != "active"
            or current_hash != decision_content_hash
            or int(memory.get("version") or 1) != int(decision_memory_version)
        ):
            return "memory_changed"
        if (
            memory.get("split_status") != decision_split_status
            or int(memory.get("split_revision") or 0) != int(decision_split_revision)
            or decision_split_status not in allowed_split_statuses
        ):
            return "split_revision_conflict"
        return None

    # ------------------------------------------------------------------
    #  v0.8.0: Unified publish helper (design doc §9.1)
    #
    #  Shared by the rules path (memory_write/edit auto-split) and the Agent
    #  path (memory_split publish/rebuild). Replaces the inline validate-then-
    #  write block that previously lived only inside memory_split.
    #
    #  Provenance is now an explicit caller argument ("parser" for rules,
    #  "agent" for memory_split) instead of inferred from anchor text — the
    #  old heuristic guessed "parser" when an anchor happened to equal a
    #  heading string, which conflated the two paths.
    #
    #  Failure semantics (design doc §5.3 / §9.2):
    #    * decision_kind="split": a real failure marks split_status=failed
    #      via CAS (_mark_split_failed), and returns an error.
    #    * decision_kind="rebuild": failures NEVER touch split_status — the
    #      old active sections stay intact. Only an error is returned.
    # ------------------------------------------------------------------

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
        """Validate, embed, and atomically publish sections + section vectors.

        Returns a state.response() dict. On failure, the original content is
        untouched. ``decision_kind`` is "split" (initial publish from
        NULL/failed/declined) or "rebuild" (replace existing active sections).
        """
        mid = memory_id
        max_sections = getattr(self.settings, "max_sections", 50)
        max_section_chars = getattr(self.settings, "max_section_chars", 3600)

        # 1) Count gate.
        if len(sections_data) < 2 or len(sections_data) > max_sections:
            return self.db.state.response({
                "error": f"sections count must be 2..{max_sections}, got {len(sections_data)}",
            }, ok=False)

        # 2) Compute + validate offsets (anchor→offset, continuity, coverage).
        #    Pure text computation — run it before touching the embedder so a
        #    bad anchor fails fast and (for split) records the failure reason.
        offset_result = self._compute_offsets(content, sections_data)
        if offset_result is None:
            if decision_kind == "split":
                self._mark_split_failed(
                    mid, decision_content_hash, decision_memory_version,
                    decision_split_revision, decision_split_status,
                    "validation", "offset computation failed",
                )
            return self.db.state.response({"error": "offset validation failed"}, ok=False)

        # 3) Section-size hard gate (design doc §6.2): a section slice that
        #    exceeds max_section_chars would embed only its front portion,
        #    producing a misleading section vector. Reject before embedding.
        for i, sec in enumerate(offset_result):
            slice_len = sec["end_offset"] - sec["start_offset"]
            if slice_len > max_section_chars:
                if decision_kind == "split":
                    self._mark_split_failed(
                        mid, decision_content_hash, decision_memory_version,
                        decision_split_revision, decision_split_status,
                        "validation", f"section {i} too large: {slice_len}>{max_section_chars}",
                    )
                return self.db.state.response({
                    "error": f"section_too_large: section {i} is {slice_len} chars (max {max_section_chars})",
                }, ok=False)

        # 4) Vec state + embedder must be ready before the expensive embedding.
        vec_state = self.db.get_vec_index_state()
        if vec_state.get("state") != "ready":
            return self.db.state.response({
                "error": "vec index not ready, complete migration first",
                "vec_index_state": vec_state,
            }, ok=False)
        embedder, _ = self._ensure_embedder()
        if embedder is None:
            return self.db.state.response({"error": "embedder unavailable"}, ok=False)

        # 5) Generate section embeddings (outside the write transaction).
        section_embeddings: list[tuple[int, list[float], int, int, int, bool]] = []
        for i, sec in enumerate(offset_result):
            title_path = sec.get("title_path") or sec.get("title") or ""
            body = content[sec["start_offset"]:sec["end_offset"]]
            try:
                er = embedder.embed_text(prefix=title_path, body=body, max_body_chars=max_section_chars)
                if not er.embedding:
                    raise RuntimeError(
                        f"section {i}: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
                    )
                section_embeddings.append((i, er.embedding, int(er.truncated), er.original_tokens, er.used_tokens, True))
            except Exception as exc:
                if decision_kind == "split":
                    self._mark_split_failed(
                        mid, decision_content_hash, decision_memory_version,
                        decision_split_revision, decision_split_status,
                        "embedding", f"section {i}: {exc}",
                    )
                return self.db.state.response({"error": f"section embedding failed at {i}: {exc}"}, ok=False)

        # 6) Atomic publish: re-CAS inside the write transaction, then swap.
        try:
            with self.db.write_transaction() as conn:
                cur = conn.execute(
                    "SELECT status, content, version, split_status, split_revision FROM memories WHERE id = ?",
                    (mid,),
                ).fetchone()
                if cur is None:
                    raise ValueError("memory_changed")
                if cur["status"] != "active":
                    raise ValueError("memory_changed")
                if hashlib.sha256(str(cur["content"]).encode("utf-8")).hexdigest() != decision_content_hash:
                    raise ValueError("memory_changed")
                if int(cur["version"]) != int(decision_memory_version):
                    raise ValueError("memory_changed")
                if cur["split_status"] != decision_split_status:
                    raise ValueError("split_revision_conflict")
                if int(cur["split_revision"]) != int(decision_split_revision):
                    raise ValueError("split_revision_conflict")
                if MemoryDB._get_meta(conn, "state") != "ready":
                    raise ValueError("vec_space_changed")
                active_space = MemoryDB._get_meta(conn, "active_space_id")
                if active_space != embedder.embedding_space_id:
                    raise ValueError("vec_space_changed")

                MemoryDB._delete_sections_for_memory(conn, mid)
                for i, sec in enumerate(offset_result):
                    em = section_embeddings[i]
                    section_id = MemoryDB._insert_section(
                        conn, mid, i,
                        title=sec.get("title"),
                        title_path=sec.get("title_path"),
                        summary=sec.get("summary"),
                        anchor_text=sec.get("anchor_text"),
                        occurrence_index=sec.get("occurrence_index", 0),
                        start_offset=sec["start_offset"],
                        end_offset=sec["end_offset"],
                        provenance=provenance,
                        embedding_truncated=em[2],
                        embedding_original_tokens=em[3],
                        embedding_used_tokens=em[4],
                    )
                    MemoryDB._store_section_vec(conn, section_id, em[1])
                updated = conn.execute(
                    "UPDATE memories SET split_status = 'active', "
                    "split_revision = split_revision + 1 "
                    "WHERE id = ? AND split_revision = ?",
                    (mid, int(decision_split_revision)),
                )
                if updated.rowcount != 1:
                    raise ValueError("split_revision_conflict")
        except ValueError as e:
            return self.db.state.response({"error": str(e)}, ok=False)

        return self.db.state.response({
            "split_active": True,
            "memory_id": mid,
            "section_count": len(offset_result),
        })

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
        """Section split: prepare (return content for LLM) or publish (validate + atomically write).

        v0.6.0 is single-batch: prepare returns the full content in one go.  For
        ultra-long documents that exceed an external LLM's context, the caller
        should pre-chunk before ``memory_write`` (split across multiple
        memories) rather than relying on a server-side batch protocol.
        """
        mid = int(memory_id)
        memory = self.db.get_memory(mid)
        if not memory:
            return self.db.state.response({"error": "memory not found"}, ok=False)

        content = memory.get("content") or ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        memory_version = int(memory.get("version") or 1)
        split_status = memory.get("split_status")
        split_revision = int(memory.get("split_revision") or 0)

        # ---- PREPARE ----
        if split_decision is None:
            # Prerequisites
            if not getattr(self.settings, "split_enabled", False):
                return self.db.state.response({"error": "split_enabled is false"}, ok=False)
            vec_state = self.db.get_vec_index_state()
            if vec_state.get("state") != "ready":
                return self.db.state.response({
                    "error": "vec index not ready",
                    "vec_index_state": vec_state,
                }, ok=False)
            if split_status == "active":
                return self.db.state.response({
                    "error": "already active, use split_decision='rebuild' to rebuild",
                    "split_status": split_status,
                }, ok=False)
            if len(content) <= getattr(self.settings, "split_threshold", 4000):
                return self.db.state.response({"error": "content below threshold, no need to split"})

            # Detect headings
            headings = self._detect_markdown_headings(content)
            parser_detected = len(headings) >= 2

            # Single-batch prepare: returns the full content for one LLM pass.
            # For ultra-long docs the caller should pre-chunk before memory_write.
            return self.db.state.response({
                "requires_user_confirmation": True,
                "content": content,
                "content_hash": content_hash,
                "memory_version": memory_version,
                "split_status": split_status,
                "split_revision": split_revision,
                "char_count": len(content),
                "parser_detected": parser_detected,
                "split_prompt": (
                    f"该记忆 {len(content)} 字符。分段为单批：原文一次性返回。"
                    "原文已完整保存，分段仅影响检索精度。是否分段？"
                ),
                "split_schema": {
                    "sections": [{
                        "title": "str",
                        "summary": "str",
                        "anchor_text": "str (除第一段外必填)",
                        "occurrence_index": "int (0-based)",
                        "title_path": "str (可选)",
                    }],
                },
            })

        # ---- DECLINE ----
        if split_decision == "decline":
            snapshot_error = self._split_snapshot_error(
                memory,
                decision_content_hash,
                decision_memory_version,
                decision_split_status,
                decision_split_revision,
                (None, "failed", "declined"),
            )
            if snapshot_error:
                return self.db.state.response({"error": snapshot_error}, ok=False)
            with self.db.write_transaction() as conn:
                cur = conn.execute(
                    "SELECT status, content, version, split_status, split_revision "
                    "FROM memories WHERE id = ?", (mid,)
                ).fetchone()
                current = dict(cur) if cur is not None else {}
                snapshot_error = self._split_snapshot_error(
                    current,
                    decision_content_hash,
                    decision_memory_version,
                    decision_split_status,
                    decision_split_revision,
                    (None, "failed", "declined"),
                )
                if snapshot_error:
                    return self.db.state.response({"error": snapshot_error}, ok=False)
                updated = conn.execute(
                    "UPDATE memories SET split_status = 'declined', "
                    "split_revision = split_revision + 1 "
                    "WHERE id = ? AND split_revision = ?",
                    (mid, int(decision_split_revision)),
                )
                if updated.rowcount != 1:
                    return self.db.state.response({"error": "split_revision_conflict"}, ok=False)
            return self.db.state.response({"declined": True, "memory_id": mid})

        # ---- PUBLISH (split or rebuild) ----
        if split_decision in ("split", "rebuild"):
            allowed_statuses: tuple[Optional[str], ...] = (
                ("active",)
                if split_decision == "rebuild"
                else (None, "failed", "declined")
            )
            snapshot_error = self._split_snapshot_error(
                memory,
                decision_content_hash,
                decision_memory_version,
                decision_split_status,
                decision_split_revision,
                allowed_statuses,
            )
            if snapshot_error:
                return self.db.state.response({"error": snapshot_error}, ok=False)
            if not sections:
                return self.db.state.response({"error": "sections required for publish"}, ok=False)

            # Delegate to the unified publish helper (v0.8.0). The Agent
            # continuation/repair path is always provenance="agent"; the rules
            # path (memory_write/edit) calls the same helper with "parser".
            return self._publish_sections(
                mid, content, sections,
                str(decision_content_hash), int(decision_memory_version),
                decision_split_status, int(decision_split_revision),
                decision_kind=split_decision,
                provenance="agent",
            )

        return self.db.state.response({"error": f"unknown split_decision: {split_decision}"}, ok=False)

    def _mark_split_failed(
        self, mid: int, content_hash: str, version: int, revision: int,
        expected_status: Optional[str], stage: str, message: str,
    ) -> None:
        """Mark split as failed using CAS (best-effort)."""
        try:
            with self.db.write_transaction() as conn:
                cur = conn.execute(
                    "SELECT status, content, version, split_status, split_revision, metadata "
                    "FROM memories WHERE id = ?",
                    (mid,),
                ).fetchone()
                if cur is None:
                    return
                if cur["status"] != "active":
                    return
                if hashlib.sha256(str(cur["content"]).encode("utf-8")).hexdigest() != content_hash:
                    return
                if int(cur["version"]) != version or int(cur["split_revision"]) != revision:
                    return
                if str(cur["split_status"]) != str(expected_status):
                    return
                # Merge metadata
                try:
                    meta = json.loads(cur["metadata"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                split_meta = meta.get("_split", {})
                split_meta["last_split_error"] = {"stage": stage, "message": message}
                meta["_split"] = split_meta
                conn.execute(
                    "UPDATE memories SET split_status = 'failed', "
                    "split_revision = split_revision + 1, "
                    "metadata = ? WHERE id = ?",
                    (json.dumps(meta, ensure_ascii=False), mid),
                )
        except Exception:
            pass

    def get_sections(
        self,
        memory_id: int,
        section_ids: Optional[list[int]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Get section content + metadata by IDs."""
        mid = int(memory_id)
        memory = self.db.get_memory(mid)
        if not memory:
            return self.db.state.response({"error": "memory not found"}, ok=False)
        content = memory.get("content") or ""
        if section_ids is None:
            sections = self.db.get_sections_by_memory(mid)
        else:
            sections, missing = self.db.get_sections_by_ids(mid, section_ids)
            if missing:
                return self.db.state.response({
                    "memory_id": mid,
                    "sections": [
                        {**s, "content": content[s["start_offset"]:s["end_offset"]]}
                        for s in sections
                    ],
                    "found_count": len(sections),
                    "missing_section_ids": missing,
                })
        return self.db.state.response({
            "memory_id": mid,
            "sections": [
                {**s, "content": content[s["start_offset"]:s["end_offset"]]}
                for s in sections
            ],
            "found_count": len(sections),
            "missing_section_ids": [],
        })

    def memory_split_status(self, memory_id: int, **_: Any) -> dict[str, Any]:
        """Check split status of a memory."""
        mid = int(memory_id)
        memory = self.db.get_memory(mid)
        if not memory:
            return self.db.state.response({"error": "memory not found"}, ok=False)
        sections = self.db.get_sections_by_memory(mid)
        vec_state = self.db.get_vec_index_state()
        return self.db.state.response({
            "memory_id": mid,
            "split_status": memory.get("split_status"),
            "split_revision": memory.get("split_revision", 0),
            "content_hash": hashlib.sha256(
                (memory.get("content") or "").encode("utf-8")
            ).hexdigest(),
            "sections": [
                {"section_id": s["id"], "title": s.get("title"),
                 "title_path": s.get("title_path"), "summary": s.get("summary")}
                for s in sections
            ],
            "section_count": len(sections),
            "vec_index_state": vec_state,
        })

    def memory_rebuild_embeddings(
        self,
        memory_ids: Optional[list[int]] = None,
        dry_run: bool = True,
        batch_size: Optional[int] = 50,
        **_: Any,
    ) -> dict[str, Any]:
        """Rebuild embeddings after model switch or for repair."""
        vec_state = self.db.get_vec_index_state()
        state = vec_state.get("state", "unmanaged")

        if state == "unmanaged":
            return self.db.state.response({"error": "unmanaged: no managed embedder"}, ok=False)

        embedder, _ = self._ensure_embedder()
        if embedder is None:
            return self.db.state.response({"error": "embedder unavailable"}, ok=False)

        # Determine target memories
        migration_mode = state in ("mismatch", "failed")
        migration_cursor = vec_state.get("migration_cursor")
        cursor_value = int(migration_cursor) if migration_cursor is not None else -1
        if migration_mode:
            if vec_state.get("target_space_id") != embedder.embedding_space_id:
                return self.db.state.response({
                    "error": "current embedder does not match migration target",
                    "target_space_id": vec_state.get("target_space_id"),
                    "current_space_id": embedder.embedding_space_id,
                }, ok=False)
            # Migration mode: continue after the persisted contiguous cursor.
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT m.id AS id FROM memories m "
                    "LEFT JOIN memories_vec v ON v.id = m.id "
                    "LEFT JOIN memory_sections s ON s.memory_id = m.id "
                    "WHERE (v.id IS NOT NULL OR s.id IS NOT NULL) AND m.id > ? "
                    "ORDER BY m.id",
                    (cursor_value,),
                ).fetchall()
                target_ids = [int(r["id"]) for r in rows]
        elif state == "ready":
            if not memory_ids:
                return self.db.state.response({
                    "error": "ready state: specify memory_ids for local repair"
                }, ok=False)
            target_ids = [int(mid) for mid in memory_ids]
        else:
            return self.db.state.response({"error": f"unexpected state: {state}"}, ok=False)

        if batch_size is not None:
            batch_size = max(1, int(batch_size))
            target_ids = target_ids[:batch_size]

        if dry_run:
            return self.db.state.response({
                "dry_run": True,
                "target_memory_ids": target_ids,
                "target_count": len(target_ids),
                "current_space_id": embedder.embedding_space_id,
                "active_space_id": vec_state.get("active_space_id"),
                "global_state": state,
            })

        # Execute rebuild
        succeeded = 0
        failed = 0
        processed = 0
        errors: list[dict] = []
        max_section_chars = getattr(self.settings, "max_section_chars", 3600)

        for mid in target_ids:
            processed += 1
            try:
                with self.db.connection() as conn:
                    mem = MemoryDB._fetch_memory(conn, mid)
                    if mem is None or mem.get("status") == "deleted":
                        # Cleanup
                        with self.db.write_transaction() as wconn:
                            MemoryDB._delete_sections_for_memory(wconn, mid)
                            wconn.execute("DELETE FROM memories_vec WHERE id = ?", (mid,))
                            if migration_mode:
                                MemoryDB._set_meta(wconn, "migration_cursor", str(mid))
                        succeeded += 1
                        continue

                    # Memory embedding
                    er = embedder.embed_text(
                        prefix=mem.get("subject") or "",
                        body=mem.get("content") or "",
                    )
                    if not er.embedding:
                        raise RuntimeError(
                            f"memory {mid}: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
                        )

                    # Section embeddings
                    sections = MemoryDB._get_sections(conn, mid)
                    sec_embeddings = []
                    content = mem.get("content") or ""
                    for sec in sections:
                        title_path = sec.get("title_path") or sec.get("title") or ""
                        body = content[sec["start_offset"]:sec["end_offset"]]
                        sec_er = embedder.embed_text(prefix=title_path, body=body, max_body_chars=max_section_chars)
                        if not sec_er.embedding:
                            raise RuntimeError(
                                f"section {sec['id']}: {getattr(embedder, 'last_encode_error', None) or 'encode returned empty embedding'}"
                            )
                        sec_embeddings.append((sec["id"], sec_er))

                    # Write
                    with self.db.write_transaction() as wconn:
                        wconn.execute("DELETE FROM memories_vec WHERE id = ?", (mid,))
                        wconn.execute(
                            "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                            (mid, json.dumps(er.embedding)),
                        )
                        for sid, ser in sec_embeddings:
                            wconn.execute("DELETE FROM memory_sections_vec WHERE id = ?", (sid,))
                            wconn.execute(
                                "INSERT INTO memory_sections_vec(id, embedding) VALUES (?, ?)",
                                (sid, json.dumps(ser.embedding)),
                            )
                            wconn.execute(
                                "UPDATE memory_sections SET embedding_truncated = ?, "
                                "embedding_original_tokens = ?, embedding_used_tokens = ? "
                                "WHERE id = ?",
                                (int(ser.truncated), ser.original_tokens, ser.used_tokens, sid),
                            )
                        if migration_mode:
                            MemoryDB._set_meta(wconn, "migration_cursor", str(mid))
                    succeeded += 1
            except Exception as exc:
                failed += 1
                errors.append({"memory_id": mid, "error": str(exc)})
                if migration_mode:
                    try:
                        with self.db.write_transaction() as conn:
                            MemoryDB._set_meta(conn, "state", "failed")
                            MemoryDB._set_meta(conn, "last_error", f"memory_id={mid}: {exc}")
                    except Exception:
                        pass
                    break  # preserve a contiguous cursor for the next resume

        # Update vec state only after checking for targets beyond the cursor
        # produced by this batch.  Re-querying without a cursor would select
        # the vectors just rebuilt and make migration impossible to finish.
        if migration_mode and not errors:
            try:
                with self.db.write_transaction() as conn:
                    current_cursor_raw = MemoryDB._get_meta(conn, "migration_cursor")
                    current_cursor = int(current_cursor_raw) if current_cursor_raw is not None else -1
                    remaining = conn.execute(
                        "SELECT 1 FROM memories m "
                        "LEFT JOIN memories_vec v ON v.id = m.id "
                        "LEFT JOIN memory_sections s ON s.memory_id = m.id "
                        "WHERE (v.id IS NOT NULL OR s.id IS NOT NULL) AND m.id > ? "
                        "LIMIT 1",
                        (current_cursor,),
                    ).fetchone()
                    if remaining is None:
                        MemoryDB._set_meta(conn, "state", "ready")
                        MemoryDB._set_meta(conn, "active_space_id", embedder.embedding_space_id)
                        for key in (
                            "target_space_id", "migration_cursor", "migration_epoch",
                            "migration_lease_owner", "migration_lease_expires_at",
                            "last_error",
                        ):
                            MemoryDB._delete_meta(conn, key)
            except Exception:
                pass

        return self.db.state.response({
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "errors": errors,
            "global_state": self.db.get_vec_index_state().get("state"),
        })
