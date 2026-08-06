from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .config import Settings, _find_config_file
from .config_registry import CONFIG_DESCRIPTORS, grouped_descriptors
from .conflict_judgments import ConflictJudgmentStore
from .tools import MemoryTools


class ConsoleAPI:
    """Read-only data adapter for the local Console HTTP server."""

    def __init__(self, tools: Optional[MemoryTools] = None, settings: Optional[Settings] = None):
        self.tools = tools or MemoryTools(settings or Settings.from_env())
        self.settings = self.tools.settings

    @staticmethod
    def _payload(response: dict[str, Any]) -> dict[str, Any]:
        return response.get("data") if isinstance(response, dict) and isinstance(response.get("data"), dict) else response

    @staticmethod
    def _ok(response: dict[str, Any]) -> bool:
        return bool(response.get("ok", True)) if isinstance(response, dict) else True

    @staticmethod
    def _limit(value: Any, default: int, max_value: int = 100) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(1, min(parsed, max_value))

    @staticmethod
    def _offset(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, parsed)

    def health(self) -> dict[str, Any]:
        return {"ok": True, "version": __version__, "read_only": True, "brand": {"en": "mema", "zh": "迷码"}}

    def overview(self) -> dict[str, Any]:
        status = self._payload(self.tools.memory_status())
        audit = self._payload(self.tools.memory_audit_summary())
        doctor = self._payload(self.tools.memory_doctor_overview(deep=False))
        counts = self._status_counts()
        by_workspace = {k: v.get("count", 0) for k, v in (audit.get("workspaces") or {}).items()}
        by_source_type: dict[str, int] = {}
        for ws_data in (audit.get("workspaces") or {}).values():
            for source, count in (ws_data.get("by_source_type") or {}).items():
                by_source_type[source] = by_source_type.get(source, 0) + int(count)
        last_scan = self.tools.db._scan_log_last_completed()
        open_conflicts = int(audit.get("total_open_conflicts") or counts.get("open_conflicts") or 0)
        counts["open_conflicts"] = open_conflicts
        return {
            "version": __version__,
            "brand": {"en": "mema", "zh": "迷码", "full": "Memory Arbiter"},
            "read_only": True,
            "local_only": True,
            "db_path": status.get("db_path"),
            "backup_jsonl": status.get("backup_jsonl"),
            "counts": counts,
            "by_workspace": by_workspace,
            "by_source_type": by_source_type,
            "doctor_overall": doctor.get("overall"),
            "doctor_summary": doctor.get("summary"),
            "last_scan": last_scan,
            "config_warnings": status.get("config_warnings") or [],
            "update_check": status.get("update_check"),
            "status": status,
        }

    def _status_counts(self) -> dict[str, int]:
        counts = {"total": 0, "active": 0, "superseded": 0, "conflicted": 0, "pending": 0, "deleted": 0, "expired": 0, "open_conflicts": 0}
        if not self.tools.db.db_available:
            return counts
        try:
            with self.tools.db.connection() as conn:
                rows = conn.execute("SELECT status, COUNT(*) AS count FROM memories GROUP BY status").fetchall()
                for row in rows:
                    status = row["status"] or "unknown"
                    count = int(row["count"] or 0)
                    counts[status] = count
                    counts["total"] += count
                conflict_row = conn.execute("SELECT COUNT(*) AS count FROM conflicts WHERE status='open'").fetchone()
                counts["open_conflicts"] = int(conflict_row["count"] or 0) if conflict_row else 0
        except sqlite3.Error:
            return counts
        counts["expired"] = counts.get("superseded", 0) + counts.get("conflicted", 0) + counts.get("pending", 0)
        return counts

    def conflicts(self, status: str = "open", limit: Any = 50) -> dict[str, Any]:
        response = self.tools.memory_list_conflicts(status=status or "open", limit=self._limit(limit, 50, 200))
        data = self._payload(response)
        items = data.get("conflicts") or []
        return {"items": items, "count": len(items), "status": status or "open"}

    def conflict_detail(self, conflict_id: int) -> dict[str, Any]:
        conflict = self._get_conflict_row(conflict_id)
        if not conflict:
            return {"error": f"conflict id {conflict_id} not found", "_http_status": 404}
        resolution_kind = conflict.get("resolution_kind") or conflict.get("judgment_resolution_kind")
        conflict["resolution_kind"] = resolution_kind
        conflict["conflict_scope"] = conflict.get("conflict_scope") or conflict.get("judgment_conflict_scope")
        conflict["recommended_resolution_action"] = ConflictJudgmentStore.resolution_action(resolution_kind)
        conflict["supersede_candidate"] = ConflictJudgmentStore.is_supersede_candidate(resolution_kind)
        left = self._memory_or_error(conflict.get("left_id"), sections="all")
        right = self._memory_or_error(conflict.get("right_id"), sections="all")
        judgments = self._payload(self.tools.memory_list_conflict_judgments(conflict_id)).get("judgments", [])
        winner = conflict.get("suggested_winner") or conflict.get("winner_id") or conflict.get("judgment_suggested_winner")
        winner_side = None
        if winner is not None:
            try:
                winner_int = int(winner)
                if winner_int == int(conflict.get("left_id")):
                    winner_side = "left"
                elif winner_int == int(conflict.get("right_id")):
                    winner_side = "right"
            except (TypeError, ValueError):
                winner_side = None
        return {"conflict": conflict, "left": left, "right": right, "winner_side": winner_side, "judgments": judgments}

    def _get_conflict_row(self, conflict_id: int) -> Optional[dict[str, Any]]:
        if not self.tools.db.db_available:
            return None
        select = (
            "SELECT c.*, j.verdict AS judgment_verdict, "
            "j.recommended_use AS judgment_recommended_use, "
            "j.suggested_winner AS judgment_suggested_winner, "
            "j.confidence_hint AS judgment_confidence_hint, "
            "j.reason AS judgment_reason, j.judge_type AS judgment_judge_type, "
            "j.judge_ref AS judgment_judge_ref, j.resolution_kind AS judgment_resolution_kind, "
            "j.conflict_scope AS judgment_conflict_scope, j.created_at AS judged_at "
            "FROM conflicts c LEFT JOIN conflict_judgments j ON j.id=c.active_judgment_id "
            "WHERE c.id=?"
        )
        try:
            with self.tools.db.connection() as conn:
                row = conn.execute(select, (int(conflict_id),)).fetchone()
                if row is None:
                    return None
                return {k: row[k] for k in row.keys()}
        except sqlite3.Error:
            return None

    def memories(
        self,
        query: str = "",
        status: str = "active",
        workspace: Optional[str] = None,
        source_type: Optional[str] = None,
        tags: Optional[str] = None,
        limit: Any = 30,
        offset: Any = 0,
    ) -> dict[str, Any]:
        normalized_status = (status or "active").strip().lower()
        if normalized_status not in {"active", "expired"}:
            return {"error": "status must be active or expired", "_http_status": 400}
        tag_filter = [t.strip() for t in (tags or "").split(",") if t.strip()] or None
        page_size = self._limit(limit, 30, 100)
        page_offset = self._offset(offset)
        if normalized_status == "active":
            if page_offset:
                return {"error": "offset is not supported for active search; narrow with a more specific query or tags_filter", "_http_status": 400}
            response = self.tools.memory_search(
                query=query or "",
                workspace=workspace or None,
                source_type=source_type or None,
                tags_filter=tag_filter,
                limit=page_size,
                include_linked_open_items=False,
                include_conflict_signal=True,
            )
        else:
            response = self.tools.memory_search_expired(
                query=query or "",
                workspace=workspace or None,
                source_type=source_type or None,
                tags_filter=tag_filter,
                limit=page_size,
                offset=page_offset,
            )
        data = self._payload(response)
        if not self._ok(response):
            return {"error": data.get("error") or "memory search failed", "_http_status": 400}
        return {
            "items": data.get("results") or [],
            "count": data.get("count", len(data.get("results") or [])),
            "has_more": data.get("has_more", False),
            "status": normalized_status,
            "query_domain": data.get("query_domain"),
            "warnings": response.get("warnings", []) if isinstance(response, dict) else [],
        }

    def memory_detail(self, memory_id: int, sections: str = "catalog") -> dict[str, Any]:
        return self._memory_or_error(memory_id, sections=sections if sections in {"none", "catalog", "all"} else "catalog")

    def _memory_or_error(self, memory_id: Any, sections: str = "catalog") -> dict[str, Any]:
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return {"error": "memory_id must be an integer", "_http_status": 400}
        response = self.tools.memory_get(memory_id=memory_id_int, sections=sections)
        data = self._payload(response)
        if not self._ok(response):
            return {"error": data.get("error") or f"memory id {memory_id_int} not found", "_http_status": 404}
        return data

    def doctor(self) -> dict[str, Any]:
        return self._payload(self.tools.memory_doctor_overview(deep=False))

    def settings_view(self) -> dict[str, Any]:
        warnings: list[str] = []
        config_path = _find_config_file(warnings)
        config_path_str = str(config_path) if config_path is not None else None
        config_exists = config_path.exists() if config_path is not None else False
        current = self._settings_values()
        groups = []
        for group in grouped_descriptors():
            items = []
            for item in group["items"]:
                value = current.get(item.get("settings_attr") or item["path"])
                if isinstance(value, Path):
                    value = str(value)
                enriched = {**item, "current": value, "source": "effective", "editable": False}
                items.append(enriched)
            groups.append({k: v for k, v in group.items() if k != "items"} | {"items": items})
        return {
            "config_file": {
                "path": config_path_str,
                "exists": config_exists,
                "warnings": list(dict.fromkeys(warnings + list(self.settings.config_warnings))),
            },
            "groups": groups,
            "read_only": True,
            "message_en": "Read-only in this version",
            "message_zh": "当前版本只读",
        }

    def _settings_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for item in CONFIG_DESCRIPTORS:
            path = item["path"]
            attr = item.get("settings_attr")
            if attr:
                value = getattr(self.settings, attr, None)
            elif "." in path:
                value = self._nested_setting(path)
            else:
                value = getattr(self.settings, path, None)
            if isinstance(value, Path):
                value = str(value)
            values[path] = value
            if attr:
                values[attr] = value
        return values

    def _nested_setting(self, path: str) -> Any:
        mapping = {
            "vec.enabled": self.settings.enable_sqlite_vec,
            "vec.dim": self.settings.vec_dim,
            "embedding.provider": self.settings.embedding_provider,
            "embedding.model_path": self.settings.embedding_model_path,
            "embedding.auto_query": self.settings.embedding_auto_query,
            "embedding.auto_write": self.settings.embedding_auto_write,
            "embedding.n_ctx": self.settings.embedding_n_ctx,
            "embedding.reserved_tokens": self.settings.embedding_reserved_tokens,
            "split.threshold": self.settings.split_threshold,
            "split.section_vec_distance_threshold": self.settings.section_vec_distance_threshold,
            "split.section_fulltext_threshold": self.settings.section_fulltext_threshold,
            "split.max_sections": self.settings.max_sections,
            "split.max_section_chars": self.settings.max_section_chars,
            "update_check.enabled": self.settings.update_check_enabled,
        }
        value = mapping.get(path)
        if isinstance(value, Path):
            return str(value)
        return value
