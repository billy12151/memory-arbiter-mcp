from __future__ import annotations

from typing import Any, Optional, Tuple

from .arbitration import compare_memories
from .config import Settings
from .db import MemoryDB
from .models import MemoryRecord, ProtectionLevel, SourceType
from .search import search_memories


class MemoryTools:
    def __init__(self, settings: Optional[Settings] = None, db: Optional[MemoryDB] = None):
        self.settings = settings or Settings.from_env()
        self.db = db or MemoryDB(self.settings)

    def _allowed(self, agent_id: Optional[str] = None, client: Optional[str] = None) -> Tuple[bool, list[str]]:
        actual_agent = agent_id or self.settings.agent_id
        actual_client = client or self.settings.client
        if self.settings.policy.enabled_for(actual_client, actual_agent):
            return True, []
        return False, [f"Memory arbiter disabled by policy for client={actual_client}, agent_id={actual_agent}."]

    def memory_write(self, **payload: Any) -> dict[str, Any]:
        allowed, warnings = self._allowed(payload.get("agent_id"), payload.get("client"))
        if not allowed:
            return self.db.state.response({"written": False}, ok=False, extra_warnings=warnings)
        try:
            record = MemoryRecord.from_input(payload, self.settings.defaults())
            memory_id, write_warnings = self.db.insert_memory(record)
            data = {"id": memory_id, "backup_only": memory_id is None, "record": {**record.__dict__, "id": memory_id}}
            return self.db.state.response(data, extra_warnings=warnings + write_warnings)
        except Exception as exc:
            return self.db.state.response({"error": str(exc)}, ok=False, extra_warnings=warnings)

    def memory_search(self, query: str = "", workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, **_: Any) -> dict[str, Any]:
        results, warnings = search_memories(self.db, query, workspace or self.settings.workspace, tags, limit)
        return self.db.state.response({"results": results, "count": len(results)}, extra_warnings=warnings)

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
                "policy": {
                    "client_defaults": self.settings.policy.client_defaults,
                    "default_enabled": self.settings.policy.default_enabled,
                    "allow_agents": self.settings.policy.allow_agents,
                    "deny_agents": self.settings.policy.deny_agents,
                },
            }
        )
