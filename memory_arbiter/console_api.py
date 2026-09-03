from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from . import __version__
from .acl import workspace_scope_sql
from .config import Settings, _find_config_file
from .config_registry import CONFIG_DESCRIPTORS, grouped_descriptors
from .tools import MemoryTools


SUPPORT_REPO_URL = "https://github.com/billy12151/memory-arbiter-mcp"
SUPPORT_NEW_ISSUE_URL = f"{SUPPORT_REPO_URL}/issues/new"


class ConsoleAPI:
    """Read-only data adapter for the local Console HTTP server."""

    def __init__(self, tools: MemoryTools | None = None, settings: Settings | None = None):
        self.tools = tools or MemoryTools(settings or Settings.from_env())
        self.settings = self.tools.settings

    @staticmethod
    def _payload(response: dict[str, Any]) -> dict[str, Any]:
        data = response.get("data") if isinstance(response, dict) else None
        return data if isinstance(data, dict) else response

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

    def _strict_workspace_required(self, workspace: str | None) -> dict[str, Any] | None:
        if getattr(self.tools.settings, "isolation", "none") == "strict" and not str(workspace or "").strip():
            return {"error": "isolation=strict requires an explicit workspace query", "_http_status": 400}
        return None

    def overview(self, workspace: str | None = None) -> dict[str, Any]:
        missing_ws = self._strict_workspace_required(workspace)
        if missing_ws is not None:
            return missing_ws
        caller = self.tools._caller_workspace(workspace)
        denied = self.tools._strict_acl_unavailable(caller)
        if denied is not None:
            payload = self._payload(denied)
            return {"error": payload.get("error") or "forbidden_strict_workspace", "_http_status": 403, **caller.response_fields()}
        status = self._payload(self.tools.memory_status(workspace=workspace))
        audit = self._payload(self.tools.memory_audit_summary(workspace=workspace))
        doctor = self._payload(self.tools.memory_doctor_overview(deep=False))
        counts = self._status_counts(workspace=workspace)
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
            "support": {
                "repo_url": SUPPORT_REPO_URL,
                "new_issue_url": SUPPORT_NEW_ISSUE_URL,
            },
            "status": status,
        }

    def _status_counts(self, workspace: str | None = None) -> dict[str, int]:
        # Conflict counters follow the doctor "unresolved" definition: open + applying.
        counts = {"total": 0, "active": 0, "superseded": 0, "conflicted": 0, "pending": 0, "deleted": 0, "expired": 0, "open_conflicts": 0, "applying_conflicts": 0}
        if not self.tools.db.db_available:
            return counts
        try:
            caller = self.tools._caller_workspace(workspace)
            explicit_none_scope = (
                caller.isolation == "none" and bool(str(workspace or "").strip())
            )
            if (caller.isolation == "strict" or explicit_none_scope) and caller.canonical:
                scope_sql, scope_params = workspace_scope_sql(
                    "COALESCE(NULLIF(workspace_canonical, ''), workspace)",
                    caller.scope_canonicals() if caller.isolation == "strict" else caller.canonical,
                )
                with self.tools.db.connection() as conn:
                    rows = conn.execute(
                        f"SELECT status, COUNT(*) AS count FROM memories WHERE {scope_sql} GROUP BY status",
                        scope_params,
                    ).fetchall()
                    for row in rows:
                        status = row["status"] or "unknown"
                        count = int(row["count"] or 0)
                        counts[status] = count
                        counts["total"] += count
                if caller.isolation == "strict":
                    open_rows = self._payload(self.tools.memory_list_conflicts(
                        status="open", limit=10000, workspace=caller.workspace,
                    )).get("conflicts") or []
                    applying_rows = self._payload(self.tools.memory_list_conflicts(
                        status="applying", limit=10000, workspace=caller.workspace,
                    )).get("conflicts") or []
                else:
                    open_rows = self.tools.db.list_conflicts(
                        status="open", limit=10000, workspace=caller.canonical,
                    )
                    applying_rows = self.tools.db.list_conflicts(
                        status="applying", limit=10000, workspace=caller.canonical,
                    )
                counts["applying_conflicts"] = len(applying_rows)
                counts["open_conflicts"] = len(open_rows) + len(applying_rows)
            else:
                with self.tools.db.connection() as conn:
                    rows = conn.execute("SELECT status, COUNT(*) AS count FROM memories GROUP BY status").fetchall()
                    for row in rows:
                        status = row["status"] or "unknown"
                        count = int(row["count"] or 0)
                        counts[status] = count
                        counts["total"] += count
                    conflict_rows = conn.execute(
                        "SELECT status, COUNT(*) AS count FROM conflicts "
                        "WHERE status IN ('open','applying') GROUP BY status"
                    ).fetchall()
                    for row in conflict_rows:
                        count = int(row["count"] or 0)
                        counts["open_conflicts"] += count
                        if row["status"] == "applying":
                            counts["applying_conflicts"] = count
        except sqlite3.Error:
            return counts
        counts["expired"] = counts.get("superseded", 0) + counts.get("conflicted", 0) + counts.get("pending", 0)
        return counts

    def conflicts(self, status: str = "open", limit: Any = 50, workspace: str | None = None) -> dict[str, Any]:
        missing_ws = self._strict_workspace_required(workspace)
        if missing_ws is not None:
            return missing_ws
        response = self.tools.memory_list_conflicts(status=status or "open", limit=self._limit(limit, 50, 200), workspace=workspace)
        if not self._ok(response):
            data = self._payload(response)
            return {"error": data.get("error") or "conflict list failed", "_http_status": 400}
        data = self._payload(response)
        items = data.get("conflicts") or []
        return {"items": items, "count": len(items), "status": status or "open"}

    def conflict_detail(self, conflict_id: int, workspace: str | None = None) -> dict[str, Any]:
        missing_ws = self._strict_workspace_required(workspace)
        if missing_ws is not None:
            return missing_ws
        caller = self.tools._caller_workspace(workspace)
        denied = self.tools._strict_acl_unavailable(caller)
        if denied is not None:
            return {"error": (denied.get("data") or {}).get("error", "forbidden_strict_workspace"), "_http_status": 403}
        detail = self.tools._conflict_detail_for_workspace(conflict_id, caller)
        if not detail:
            return {"error": f"conflict id {conflict_id} not found", "_http_status": 404}
        return detail

    def _get_conflict_row(self, conflict_id: int) -> dict[str, Any] | None:
        if not self.tools.db.db_available:
            return None
        try:
            with self.tools.db.connection() as conn:
                row = conn.execute("SELECT * FROM conflicts WHERE id=?", (int(conflict_id),)).fetchone()
                if row is None:
                    return None
                conflict = {key: row[key] for key in row.keys()}
                for key in (
                    "slot_key", "candidate_key", "member_versions", "value_groups",
                    "apply_summary", "notice_payload", "notice_slot_provenance",
                ):
                    if isinstance(conflict.get(key), str):
                        try:
                            conflict[key] = json.loads(conflict[key])
                        except json.JSONDecodeError:
                            conflict[key] = None
                return conflict
        except sqlite3.Error:
            return None

    def memories(
        self,
        query: str = "",
        status: str = "active",
        workspace: str | None = None,
        source_type: str | None = None,
        tags: str | None = None,
        limit: Any = 30,
        offset: Any = 0,
    ) -> dict[str, Any]:
        normalized_status = (status or "active").strip().lower()
        missing_ws = self._strict_workspace_required(workspace)
        if missing_ws is not None:
            return missing_ws
        if normalized_status not in {"active", "expired"}:
            return {"error": "status must be active or expired", "_http_status": 400}
        tag_filter = [t.strip() for t in (tags or "").split(",") if t.strip()] or None
        page_size = self._limit(limit, 30, 100)
        page_offset = self._offset(offset)
        if normalized_status == "active":
            # Empty query + no filters → browse by recency (not memory_search,
            # whose _recent_fallback uses a multi-level
            # status→protection→source_type→confidence→time sort that buries
            # recent memories behind locked/user_confirmed ones). Direct
            # ORDER BY ingest_time DESC gives the user what they expect when
            # browsing the memories page: newest first, paginated.
            if not query and not tag_filter and not source_type:
                return self._recent_browse(page_size, page_offset, workspace=workspace)
            # Active search with query/filters: memory_search supports offset as
            # best-effort query-recall pagination (deep pages may return empty
            # while has_more is still true, because total_estimate is an estimate).
            # `count` in the response is the page size, not a total — the UI must
            # drive paging off has_more, not a page count.
            response = self.tools.memory_search(
                query=query or "",
                workspace=workspace or None,
                source_type=source_type or None,
                tags_filter=tag_filter,
                limit=page_size,
                offset=page_offset,
                include_linked_open_items=False,
                include_conflict_signal=True,
                # Console is the human-facing channel: keep full content,
                # find's index-page preview is for agents.
                include_content=True,
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
        # `count` from memory_search(_expired) is the page size (len results),
        # NOT a total. `total_estimate` is the engine's total signal:
        #   - expired without query: exact (SQL COUNT) → pagination_precision="exact"
        #   - active search / expired with query: best-effort estimate (query-recall)
        #   - v0.15.4: unfiltered query-recall reports None (no exact total
        #     exists) — fall back to the page size so the UI shows an item count
        # `total_precise` tells the UI whether to show "共 N 条" vs "约 N 条".
        total = data.get("total_estimate")
        if total is None:
            total = len(data.get("results") or [])
        precision = data.get("pagination_precision")
        if precision is not None:
            total_precise = precision == "exact"
        else:
            # active search path has no pagination_precision field; it is always
            # best-effort query-recall (memory_search deliberately unpaginated).
            total_precise = False
        return {
            "items": data.get("results") or [],
            "count": data.get("count", len(data.get("results") or [])),
            "total": total,
            "total_precise": total_precise,
            "has_more": data.get("has_more", False),
            "status": normalized_status,
            "query_domain": data.get("query_domain"),
            "warnings": response.get("warnings", []) if isinstance(response, dict) else [],
        }

    def _recent_browse(self, limit: int, offset: int, workspace: str | None = None) -> dict[str, Any]:
        """Browse active memories by recency (newest first), paginated.

        Bypasses memory_search so the memories page shows the actual newest
        memories instead of a relevance/safety-net sort that buries recent
        agent_generated memories behind locked/user_confirmed ones.
        """
        db = self.tools.db
        if not db.db_available:
            return {"items": [], "count": 0, "total": 0, "total_precise": True, "has_more": False, "status": "active"}
        # strict isolation requires a workspace on every recall — browsing
        # without one would leak cross-workspace memories. Reject the same way
        # memory_search does, so the console surfaces the error rather than
        # silently bypassing isolation.
        isolation = getattr(self.tools.settings, "isolation", "none")
        caller = self.tools._caller_workspace(workspace)
        if isolation == "strict" and not caller.canonical:
            return {"error": "forbidden_strict_workspace", "_http_status": 400, **caller.response_fields()}
        try:
            with db.connection() as conn:
                explicit_none_scope = isolation == "none" and bool(str(workspace or "").strip())
                if isolation == "strict" or explicit_none_scope:
                    scope_sql, scope_params = workspace_scope_sql(
                        "COALESCE(NULLIF(workspace_canonical, ''), workspace)",
                        caller.scope_canonicals() if isolation == "strict" else caller.canonical,
                    )
                    total = int(conn.execute(
                        f"SELECT COUNT(*) FROM memories WHERE status='active' AND {scope_sql}",
                        scope_params,
                    ).fetchone()[0] or 0)
                    rows = conn.execute(
                        f"SELECT * FROM memories WHERE status='active' AND {scope_sql} "
                        "ORDER BY ingest_time DESC, id DESC LIMIT ? OFFSET ?",
                        (*scope_params, limit, offset),
                    ).fetchall()
                else:
                    total = int(conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE status='active'"
                    ).fetchone()[0] or 0)
                    rows = conn.execute(
                        "SELECT * FROM memories WHERE status='active' "
                        "ORDER BY ingest_time DESC, id DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    ).fetchall()
        except sqlite3.Error as exc:
            return {"error": f"browse failed: {exc}", "_http_status": 500}
        items = [dict(r) for r in rows]
        has_more = total > offset + len(items)
        out = {
            "items": items,
            "count": total,
            "total": total,
            "total_precise": True,
            "has_more": has_more,
            "status": "active",
        }
        if isolation == "strict":
            out.update(caller.response_fields())
        return out

    def memory_detail(self, memory_id: int, workspace: str | None = None) -> dict[str, Any]:
        missing_ws = self._strict_workspace_required(workspace)
        if missing_ws is not None:
            return missing_ws
        return self._memory_or_error(memory_id, workspace=workspace)

    def _memory_or_error(self, memory_id: Any, workspace: str | None = None) -> dict[str, Any]:
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return {"error": "memory_id must be an integer", "_http_status": 400}
        response = self.tools.memory_get(memory_id=memory_id_int, workspace=workspace)
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
        # Nested file keys of the 0.15.0 slim config face; every top-level
        # Settings attribute resolves through _settings_values' getattr path.
        mapping = {
            "embedding.model_path": self.settings.embedding_model_path,
            "embedding.auto_query": self.settings.embedding_auto_query,
            "embedding.auto_write": self.settings.embedding_auto_write,
            "semantic_conflict.enabled": self.settings.semantic_conflict_enabled,
            "semantic_conflict.model_path": self.settings.semantic_conflict_model_path,
            "semantic_conflict.on_write": self.settings.semantic_conflict_on_write,
            "semantic_conflict.max_notice_pairs": self.settings.semantic_conflict_max_notice_pairs,
            "mcp.http.host": self.settings.mcp_http_host,
            "mcp.http.port": self.settings.mcp_http_port,
            "update_check.enabled": self.settings.update_check_enabled,
        }
        value = mapping.get(path)
        if isinstance(value, Path):
            return str(value)
        return value
