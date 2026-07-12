from __future__ import annotations

from typing import Any, Optional, Tuple

from .arbitration import compare_memories
from .config import Settings
from .db import MemoryDB
from .embedder import EncodeFn
from .models import MemoryRecord, ProtectionLevel, SourceType
from .search import search_memories


class MemoryTools:
    def __init__(self, settings: Optional[Settings] = None, db: Optional[MemoryDB] = None):
        self.settings = settings or Settings.from_env()
        self.db = db or MemoryDB(self.settings)
        self._embedder: Optional[EncodeFn] = None
        self._embedder_loaded = False
        self._embedder_warnings: list[str] = list(self.settings.config_warnings)

    def _allowed(self, agent_id: Optional[str] = None, client: Optional[str] = None) -> Tuple[bool, list[str]]:
        actual_agent = agent_id or self.settings.agent_id
        actual_client = client or self.settings.client
        if self.settings.policy.enabled_for(actual_client, actual_agent):
            return True, []
        return False, [f"Memory arbiter disabled by policy for client={actual_client}, agent_id={actual_agent}."]

    def _embedding_configured(self) -> bool:
        return self.settings.embedding_provider == "gguf" and self.settings.embedding_model_path is not None

    def _ensure_embedder(self) -> Tuple[Optional[EncodeFn], list[str]]:
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
        self._embedder, warnings = build_embedder(str(self.settings.embedding_model_path), self.settings.vec_dim)
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
                        embedding = embedder(self._embedding_text(data["record"]))
                        data["embedding_stored"], store_warnings = self.db.store_embedding(memory_id, embedding)
                        embedding_warnings.extend(store_warnings)
                    except Exception as exc:
                        embedding_warnings.append(f"auto-embedding write failed: {exc}")
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
                    query_embedding = embedder(query)
                except Exception as exc:
                    extra_warnings.append(f"auto-embedding query failed: {exc}")
        results, warnings = search_memories(self.db, query, workspace or self.settings.workspace, tags, limit, include_superseded=include_superseded, debug_ranking=debug_ranking, query_embedding=query_embedding)
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
                    embedding = embedder(self._embedding_text(updated))
                    embedding_stored, store_warnings = self.db.store_embedding(int(memory_id), embedding)
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
