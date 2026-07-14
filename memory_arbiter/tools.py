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
from .search import search_memories


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
            str(self.settings.embedding_model_path), self.settings.vec_dim
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
            return self.db.state.response(data, extra_warnings=warnings + write_warnings + embedding_warnings)
        except Exception as exc:
            return self.db.state.response({"error": str(exc)}, ok=False, extra_warnings=warnings)

    def memory_search(self, query: str = "", workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, include_superseded: bool = False, debug_ranking: bool = False, query_embedding: Optional[list[float]] = None, **_: Any) -> dict[str, Any]:
        extra_warnings = list(self._embedder_warnings)
        if query_embedding is None and query and self.settings.embedding_auto_query:
            embedder, ensure_warnings = self._ensure_embedder()
            extra_warnings.extend(ensure_warnings)
            if embedder is not None:
                try:
                    er = embedder.embed_text(prefix="", body=query)
                    query_embedding = er.embedding
                except Exception as exc:
                    extra_warnings.append(f"auto-embedding query failed: {exc}")
        results, warnings = search_memories(self.db, query, workspace or self.settings.workspace, tags, limit, include_superseded=include_superseded, debug_ranking=debug_ranking, query_embedding=query_embedding)
        # v0.6.0: attach section enhancement to active-split results
        results = self._attach_sections(results, query_embedding, extra_warnings)
        return self.db.state.response({"results": results, "count": len(results)}, extra_warnings=extra_warnings + warnings)

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
        results = self.db.list_memories(workspace=workspace or self.settings.workspace, limit=limit)
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
        **_: Any,
    ) -> dict[str, Any]:
        """In-place edit a memory's content, archiving the prior version to
        ``memory_history`` (version chain) and syncing the FTS index.

        Two edit modes:
          * full replace: pass ``new_content`` (old_text/new_text must be empty)
          * partial replace: pass ``old_text`` + ``new_text`` for an exact
            substring substitution (new_content must be empty)

        Authorization (layered): normal records edit freely; ``locked`` /
        ``user_confirmed`` records require ``authorized=True`` (mirrors
        ``memory_supersede``). Records already superseded/deleted are rejected.
        """
        memory = self.db.get_memory(int(memory_id))
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
                {"error": "provide new_content for full replace, or old_text+new_text for partial replace", "edited": False},
                ok=False,
            )
        history_id = self.db.edit_memory(
            int(memory_id),
            resolved_content,
            new_subject=new_subject,
            new_tags=new_tags,
            reason=reason or None,
        )
        if history_id is None:
            return self.db.state.response({"error": "edit failed (db not writable)", "edited": False}, ok=False)
        updated = self.db.get_memory(int(memory_id))
        embedding_warnings: list[str] = []
        embedding_stored: Optional[bool] = None
        if self.settings.embedding_auto_write and self._embedding_configured():
            embedding_stored = False
            embedder, ensure_warnings = self._ensure_embedder()
            embedding_warnings.extend(ensure_warnings)
            if embedder is None:
                _deleted, delete_warnings = self.db.delete_embedding(int(memory_id))
                embedding_warnings.extend(delete_warnings)
                embedding_warnings.append("re-embedding on edit skipped because embedder unavailable; deleted stale embedding to avoid dirty recall.")
            elif updated is not None:
                try:
                    embedding_result = embedder.embed_text(
                        prefix=updated.get("subject") or "",
                        body=updated.get("content") or "",
                    )
                    embedding_stored, store_warnings = self.db.store_embedding(int(memory_id), embedding_result.embedding)
                    embedding_warnings.extend(store_warnings)
                    if not embedding_stored:
                        _deleted, delete_warnings = self.db.delete_embedding(int(memory_id))
                        embedding_warnings.extend(delete_warnings)
                        embedding_warnings.append("re-embedding on edit failed; deleted stale embedding to avoid dirty recall.")
                except Exception as exc:
                    _deleted, delete_warnings = self.db.delete_embedding(int(memory_id))
                    embedding_warnings.extend(delete_warnings)
                    embedding_warnings.append(f"re-embedding on edit failed: {exc}; deleted stale embedding to avoid dirty recall.")
        data = {
            "edited": True,
            "memory_id": int(memory_id),
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
    #  v0.6.0: Section split tools
    # ==================================================================

    def _attach_sections(
        self,
        results: list[dict[str, Any]],
        query_embedding: Optional[list[float]],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """Post-process search results: attach section enhancement for active-split memories.

        Uses a short read snapshot so all reads are consistent (design doc §4.3).
        """
        if not results or not self.db.db_available:
            return results

        vec_state = self.db.get_vec_index_state()
        vec_gate_open = (
            vec_state.get("state") == "ready"
            and query_embedding is not None
            and self.db.state.sqlite_vec_available
        )
        threshold = getattr(self.settings, "section_vec_distance_threshold", 0.7)
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

            sections = sections_map.get(mid, [])
            total_sections = len(sections)

            # Invariant guards
            if total_sections == 0:
                result.setdefault("warnings", []).append("split_invariant_broken_empty_sections")
                continue
            if total_sections == 1:
                result.setdefault("warnings", []).append("split_invariant_broken_too_few_sections")
                continue

            # Vec gate closed → return full text
            if not vec_gate_open:
                result.setdefault("warnings", []).append(
                    f"vec_disabled={'gate_closed' if query_embedding else 'no_query_embedding'}"
                )
                continue

            # Check section vec completeness
            section_ids = {s["id"] for s in sections}
            vec_ids = section_vec_ids_map.get(mid, set())
            if section_ids - vec_ids:
                result.setdefault("warnings", []).append("split_invariant_broken_missing_section_vec")
                continue

            # Section Vec matching
            try:
                vec_hits = self.db.section_vec_distance_match(mid, query_embedding, threshold)
            except Exception:
                vec_hits = []

            matched_ids = {h["section_id"] for h in vec_hits}
            matched_count = len(matched_ids)

            if matched_count == 0:
                # True zero match
                result["content"] = None
                result["content_omitted"] = True
                result["section_enhancement_applied"] = True
                result["section_catalog"] = [
                    {"section_id": s["id"], "title": s.get("title"), "title_path": s.get("title_path"),
                     "summary": s.get("summary"), "embedding_truncated": bool(s.get("embedding_truncated")),
                     "embedding_original_tokens": s.get("embedding_original_tokens", 0),
                     "embedding_used_tokens": s.get("embedding_used_tokens", 0)}
                    for s in sections
                ]
                result["hint"] = f"已拆分为 {total_sections} 段，可用 get_sections 获取"
            elif matched_count / total_sections >= fulltext_threshold:
                # Most sections matched → return full text
                result["content_omitted"] = False
                result["section_enhancement_applied"] = True
                result["matched_sections"] = [
                    {"section_id": h["section_id"], "title": h.get("title"),
                     "title_path": h.get("title_path"), "summary": h.get("summary")}
                    for h in vec_hits
                ]
                pct = round(100 * matched_count / total_sections)
                result["hint"] = f"{pct}% 段落命中，建议直接看全文"
            else:
                # Partial match
                result["content"] = None
                result["content_omitted"] = True
                result["section_enhancement_applied"] = True
                result["matched_sections"] = [
                    {"section_id": h["section_id"], "title": h.get("title"),
                     "title_path": h.get("title_path"), "summary": h.get("summary")}
                    for h in vec_hits
                ]
                result["section_catalog"] = [
                    {"section_id": s["id"], "title": s.get("title"), "title_path": s.get("title_path"),
                     "summary": s.get("summary")}
                    for s in sections if s["id"] not in matched_ids
                ]
                result["hint"] = "已返回命中段落元数据，用 get_sections 获取段落原文"

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
        batch_start: int = 0,
    ) -> Optional[list[dict[str, Any]]]:
        """Compute global offsets from LLM-provided anchors.

        Returns list of {start_offset, end_offset, ...section_data} or None on failure.
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
                local_start = self._find_nth_occurrence(
                    content[batch_start:batch_start + len(content)], anchor, occ
                )
                # Actually search in the full content from batch_start
                # Simplified: search in the remaining content from batch_start
                search_text = content
                local_start = self._find_nth_occurrence(search_text, anchor, occ)
                if local_start == -1:
                    return None
                local_start = local_start  # already global if batch_start=0
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

        Returns list of (char_offset, heading_text).
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
                    headings.append((pos, line.rstrip()))
            pos += len(line)
        return headings

    def memory_split(
        self,
        memory_id: int,
        split_decision: Optional[str] = None,
        decision_content_hash: Optional[str] = None,
        decision_memory_version: Optional[int] = None,
        decision_split_status: Optional[str] = None,
        decision_split_revision: Optional[int] = None,
        sections: Optional[list[dict]] = None,
        prepare_batch_index: int = 0,
        llm_batch_chars: int = 12000,
        **_: Any,
    ) -> dict[str, Any]:
        """Section split: prepare (return content for LLM) or publish (validate + atomically write)."""
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

            # Simple single-batch prepare (multi-batch can be added later)
            return self.db.state.response({
                "requires_user_confirmation": True,
                "content": content,
                "content_hash": content_hash,
                "memory_version": memory_version,
                "split_status": split_status,
                "split_revision": split_revision,
                "char_count": len(content),
                "parser_detected": parser_detected,
                "llm_batch_chars": llm_batch_chars,
                "batch_count": 1,
                "split_prompt": (
                    f"该记忆 {len(content)} 字符。分段需 1 个 LLM 批次。"
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
            if decision_content_hash != content_hash:
                return self.db.state.response({"error": "content_hash mismatch"}, ok=False)
            with self.db.write_transaction() as conn:
                cur = conn.execute(
                    "SELECT split_status, split_revision FROM memories WHERE id = ?", (mid,)
                ).fetchone()
                if cur["split_status"] != split_status or cur["split_revision"] != split_revision:
                    return self.db.state.response({"error": "split_revision_conflict"}, ok=False)
                conn.execute(
                    "UPDATE memories SET split_status = 'declined', "
                    "split_revision = split_revision + 1 WHERE id = ?",
                    (mid,),
                )
            return self.db.state.response({"declined": True, "memory_id": mid})

        # ---- PUBLISH (split or rebuild) ----
        if split_decision in ("split", "rebuild"):
            if not sections:
                return self.db.state.response({"error": "sections required for publish"}, ok=False)

            # Validate count
            max_sections = getattr(self.settings, "max_sections", 50)
            if len(sections) < 2 or len(sections) > max_sections:
                return self.db.state.response({
                    "error": f"sections count must be 2..{max_sections}, got {len(sections)}",
                }, ok=False)

            # Vec state check
            vec_state = self.db.get_vec_index_state()
            if vec_state.get("state") != "ready":
                return self.db.state.response({
                    "error": "vec index not ready, complete migration first",
                    "vec_index_state": vec_state,
                }, ok=False)

            # Compute offsets
            offset_result = self._compute_offsets(content, sections)
            if offset_result is None:
                # Mark as failed
                self._mark_split_failed(mid, content_hash, memory_version, split_revision,
                                        split_status, "validation", "offset computation failed")
                return self.db.state.response({"error": "offset validation failed"}, ok=False)

            # Generate section embeddings
            embedder, _ = self._ensure_embedder()
            if embedder is None:
                return self.db.state.response({"error": "embedder unavailable"}, ok=False)

            max_section_chars = getattr(self.settings, "max_section_chars", 3600)
            section_embeddings: list[tuple[int, list[float], int, int, int, bool]] = []
            for i, sec in enumerate(offset_result):
                title_path = sec.get("title_path") or sec.get("title") or ""
                body = content[sec["start_offset"]:sec["end_offset"]]
                try:
                    er = embedder.embed_text(prefix=title_path, body=body, max_body_chars=max_section_chars)
                    section_embeddings.append((i, er.embedding, int(er.truncated), er.original_tokens, er.used_tokens, True))
                except Exception as exc:
                    self._mark_split_failed(mid, content_hash, memory_version, split_revision,
                                            split_status, "embedding", f"section {i}: {exc}")
                    return self.db.state.response({"error": f"section embedding failed at {i}: {exc}"}, ok=False)

            # Atomic publish
            expected_status = "active" if split_decision == "rebuild" else split_status
            try:
                with self.db.write_transaction() as conn:
                    # CAS
                    cur = conn.execute(
                        "SELECT status, content, version, split_status, split_revision FROM memories WHERE id = ?",
                        (mid,),
                    ).fetchone()
                    if cur is None:
                        raise ValueError("memory disappeared")
                    if hashlib.sha256(str(cur["content"]).encode()).hexdigest() != content_hash:
                        raise ValueError("memory_changed")
                    if int(cur["version"]) != memory_version:
                        raise ValueError("memory_changed")
                    if str(cur["split_status"]) != str(expected_status):
                        raise ValueError("split_revision_conflict")
                    if int(cur["split_revision"]) != split_revision:
                        raise ValueError("split_revision_conflict")

                    # Vec space check
                    active_space = MemoryDB._get_meta(conn, "active_space_id")
                    if active_space and active_space != embedder.embedding_space_id:
                        raise ValueError("vec_space_changed")

                    # Delete old sections
                    MemoryDB._delete_sections_for_memory(conn, mid)

                    # Insert new sections + vecs
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
                            provenance="llm",
                            embedding_truncated=em[2],
                            embedding_original_tokens=em[3],
                            embedding_used_tokens=em[4],
                        )
                        MemoryDB._store_section_vec(conn, section_id, em[1])

                    # Update status
                    conn.execute(
                        "UPDATE memories SET split_status = 'active', "
                        "split_revision = split_revision + 1 WHERE id = ?",
                        (mid,),
                    )
            except ValueError as e:
                return self.db.state.response({"error": str(e)}, ok=False)

            return self.db.state.response({
                "split_active": True,
                "memory_id": mid,
                "section_count": len(offset_result),
            })

        return self.db.state.response({"error": f"unknown split_decision: {split_decision}"}, ok=False)

    def _mark_split_failed(
        self, mid: int, content_hash: str, version: int, revision: int,
        expected_status: Optional[str], stage: str, message: str,
    ) -> None:
        """Mark split as failed using CAS (best-effort)."""
        try:
            with self.db.write_transaction() as conn:
                cur = conn.execute(
                    "SELECT content, version, split_status, split_revision FROM memories WHERE id = ?",
                    (mid,),
                ).fetchone()
                if cur is None:
                    return
                if hashlib.sha256(str(cur["content"]).encode()).hexdigest() != content_hash:
                    return
                if int(cur["version"]) != version or int(cur["split_revision"]) != revision:
                    return
                if str(cur["split_status"]) != str(expected_status):
                    return
                # Merge metadata
                row = cur
                meta = json.loads(row["metadata"] or "{}") if "metadata" in row.keys() else {}
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
        if state in ("mismatch", "failed"):
            # Migration mode: all memories with vectors
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT m.id AS id FROM memories m "
                    "LEFT JOIN memories_vec v ON v.id = m.id "
                    "LEFT JOIN memory_sections s ON s.memory_id = m.id "
                    "WHERE v.id IS NOT NULL OR s.id IS NOT NULL "
                    "ORDER BY m.id"
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
        errors: list[dict] = []
        max_section_chars = getattr(self.settings, "max_section_chars", 3600)

        for mid in target_ids:
            try:
                with self.db.connection() as conn:
                    mem = MemoryDB._fetch_memory(conn, mid)
                    if mem is None or mem.get("status") == "deleted":
                        # Cleanup
                        with self.db.write_transaction() as wconn:
                            MemoryDB._delete_sections_for_memory(wconn, mid)
                            wconn.execute("DELETE FROM memories_vec WHERE id = ?", (mid,))
                        succeeded += 1
                        continue

                    # Memory embedding
                    er = embedder.embed_text(
                        prefix=mem.get("subject") or "",
                        body=mem.get("content") or "",
                    )

                    # Section embeddings
                    sections = MemoryDB._get_sections(conn, mid)
                    sec_embeddings = []
                    content = mem.get("content") or ""
                    for sec in sections:
                        title_path = sec.get("title_path") or sec.get("title") or ""
                        body = content[sec["start_offset"]:sec["end_offset"]]
                        sec_er = embedder.embed_text(prefix=title_path, body=body, max_body_chars=max_section_chars)
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
                    succeeded += 1
            except Exception as exc:
                failed += 1
                errors.append({"memory_id": mid, "error": str(exc)})

        # Update vec state if migration complete
        if state in ("mismatch", "failed") and not errors and not target_ids:
            try:
                with self.db.write_transaction() as conn:
                    MemoryDB._set_meta(conn, "state", "ready")
                    MemoryDB._set_meta(conn, "active_space_id", embedder.embedding_space_id)
                    MemoryDB._delete_meta(conn, "target_space_id")
                    MemoryDB._delete_meta(conn, "migration_cursor")
                    MemoryDB._delete_meta(conn, "last_error")
            except Exception:
                pass

        return self.db.state.response({
            "processed": len(target_ids),
            "succeeded": succeeded,
            "failed": failed,
            "errors": errors,
            "global_state": self.db.get_vec_index_state().get("state"),
        })
