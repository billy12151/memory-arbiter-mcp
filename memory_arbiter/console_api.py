from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from . import __version__
from .acl import raw_workspace, visible_memory, workspace_scope_sql
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
        # by_workspace counts every non-deleted memory (audit_summary semantics:
        # active + superseded + retired + …), which confuses next to the
        # "active" metric — pair it with per-workspace active counts.
        by_workspace_active: dict[str, int] = {}
        try:
            explicit_none_scope = (
                caller.isolation == "none" and bool(str(workspace or "").strip())
            )
            if (caller.isolation == "strict" or explicit_none_scope) and caller.canonical:
                active_scope_sql, active_scope_params = workspace_scope_sql(
                    "COALESCE(NULLIF(workspace_canonical, ''), workspace)",
                    caller.scope_canonicals() if caller.isolation == "strict" else caller.canonical,
                )
                active_sql = (
                    "SELECT COALESCE(NULLIF(workspace_canonical, ''), workspace) AS ws, COUNT(*) AS c "
                    f"FROM memories WHERE status='active' AND {active_scope_sql} GROUP BY ws"
                )
                active_params: list[str] = active_scope_params
            else:
                active_sql = (
                    "SELECT COALESCE(NULLIF(workspace_canonical, ''), workspace) AS ws, COUNT(*) AS c "
                    "FROM memories WHERE status='active' GROUP BY ws"
                )
                active_params = []
            with self.tools.db.connection() as conn:
                by_workspace_active = {
                    str(row["ws"]): int(row["c"]) for row in conn.execute(active_sql, active_params).fetchall()
                }
        except sqlite3.Error:
            by_workspace_active = {}
        by_source_type: dict[str, int] = {}
        for ws_data in (audit.get("workspaces") or {}).values():
            for source, count in (ws_data.get("by_source_type") or {}).items():
                by_source_type[source] = by_source_type.get(source, 0) + int(count)
        last_scan = self.tools.db._scan_log_last_completed()
        open_conflicts = int(audit.get("total_open_conflicts") or counts.get("open_conflicts") or 0)
        counts["open_conflicts"] = open_conflicts
        # C4 (0.15.13): triage counter visibility for the console metrics row
        # (doctor's conflicts.backlog carries the same numbers with weekly
        # detail + latest time).
        try:
            with self.tools.db.connection() as conn:
                counts["dismissed_conflicts"] = int(conn.execute(
                    "SELECT COUNT(*) FROM conflicts WHERE status='not_a_conflict'"
                ).fetchone()[0])
        except sqlite3.Error:
            counts["dismissed_conflicts"] = 0
        # 0.16.0 §6㉑⑦: scan-queue backlog visible on the console metrics row
        # (same source as doctor's conflicts.scan_queue_backlog finding —
        # counts only, never the items; suspected items are mostly noise).
        try:
            with self.tools.db.connection() as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS c FROM scan_queue GROUP BY status"
                ).fetchall()
            q = {str(r["status"]): int(r["c"]) for r in rows}
            counts["scan_queue_backlog"] = q.get("pending", 0)
        except sqlite3.Error:
            counts["scan_queue_backlog"] = 0
        return {
            "version": __version__,
            "brand": {"en": "mema", "zh": "迷码", "full": "Memory Arbiter"},
            "read_only": True,
            "local_only": True,
            "db_path": status.get("db_path"),
            "backup_jsonl": status.get("backup_jsonl"),
            "counts": counts,
            "by_workspace": by_workspace,
            "by_workspace_active": by_workspace_active,
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
            # whose recent-browse path (_recent_fallback, browse-only since
            # v0.15.9 — query recall no longer falls back) uses a multi-level
            # status→protection→source_type→confidence→time sort that buries
            # recent memories behind locked/user_confirmed ones). Direct
            # ORDER BY ingest_time DESC gives the user what they expect when
            # browsing the memories page: newest first, paginated.
            if not query and not tag_filter and not source_type:
                return self._recent_browse(page_size, page_offset, workspace=workspace)
            # Active search with query/filters: memory_search supports offset as
            # best-effort pagination. Since 0.15.4 the unfiltered query path
            # reports total_estimate=None and has_more=False (no exact total
            # exists); filtered paths keep exact has_more/total. `count` in the
            # response is the page size, not a total — the UI must drive paging
            # off has_more, not a page count.
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
                content_mode="full",
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

    # Memory relations graph (memory detail page) — per-source caps keep the
    # canvas readable; GRAPH_MAX_NEIGHBORS caps the union.
    GRAPH_CONFLICT_LIMIT = 5
    GRAPH_CONFLICT_MEMBERS_LIMIT = 4
    GRAPH_CANDIDATE_LIMIT = 8
    GRAPH_SAME_SUBJECT_LIMIT = 8
    GRAPH_SAME_TAG_LIMIT = 6
    GRAPH_MAX_NEIGHBORS = 24

    def memory_graph(self, memory_id: int, workspace: str | None = None) -> dict[str, Any]:
        """One-hop relations graph for one memory (read-only).

        Edge sources, in priority order: open conflict groups
        (``conflicts.member_versions``), pending ``conflict_backlog`` pairs
        (``pair_score`` as weight), active memories sharing the subject, and
        active memories sharing at least one tag. Entity edges were dropped
        on purpose: ``metadata.entity`` is retired (0.17.0 G3 — new writes
        strip it), so an entity edge would only light up for legacy rows.
        """
        missing_ws = self._strict_workspace_required(workspace)
        if missing_ws is not None:
            return missing_ws
        detail = self._memory_or_error(memory_id, workspace=workspace)
        if "error" in detail:
            return detail
        memory = detail.get("memory") or {}
        mid = int(memory.get("id") or memory_id)
        tags = memory.get("tags")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = []
        tags = [str(tag) for tag in (tags or []) if str(tag).strip()][:8]

        edges: list[dict[str, Any]] = []
        truncated = {"conflict": False, "candidate": False, "same_subject": False, "same_tag": False}
        neighbor_ids: set[int] = set()

        caller = self.tools._caller_workspace(workspace)
        isolation = getattr(self.tools.settings, "isolation", "none")
        explicit_none_scope = isolation == "none" and bool(str(workspace or "").strip())
        scope_sql: str = ""
        scope_params: list[str] = []
        scope_sql_m: str = ""
        scope_params_m: list[str] = []
        if isolation == "strict" or explicit_none_scope:
            scope = caller.scope_canonicals() if isolation == "strict" else caller.canonical
            scope_sql, scope_params = workspace_scope_sql(
                "COALESCE(NULLIF(workspace_canonical, ''), workspace)", scope
            )
            scope_sql_m, scope_params_m = workspace_scope_sql(
                "COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace)", scope
            )

        def _neighbor_visible(row: dict[str, Any] | None) -> bool:
            # Same read-ACL shape as memory_get (center) and
            # _conflict_detail_for_workspace (members): strict → admitted-set
            # membership; none + explicit workspace → single-canonical match.
            if row is None:
                return False
            if isolation == "strict":
                return visible_memory(row, caller.canonical, caller.scope_canonicals())
            if explicit_none_scope:
                return raw_workspace(row) == str(caller.canonical or "")
            return True

        conflicts = self.tools.db.list_open_conflicts_for_memory_ids([mid])
        if len(conflicts) > self.GRAPH_CONFLICT_LIMIT:
            truncated["conflict"] = True
            conflicts = conflicts[: self.GRAPH_CONFLICT_LIMIT]
        conflict_others: list[tuple[dict[str, Any], list[int]]] = []
        backlog_pairs: list[tuple[int, float]] = []
        involved: set[int] = set()
        for conflict in conflicts:
            others: list[int] = []
            for member in conflict.get("member_versions") or []:
                try:
                    member_id = int(member.get("memory_id"))
                except (TypeError, ValueError):
                    continue
                if member_id != mid and member_id not in others:
                    others.append(member_id)
            if len(others) > self.GRAPH_CONFLICT_MEMBERS_LIMIT:
                truncated["conflict"] = True
            conflict_others.append((conflict, others[: self.GRAPH_CONFLICT_MEMBERS_LIMIT]))
            involved.update(others)

        try:
            with self.tools.db.connection() as conn:
                backlog_rows = conn.execute(
                    "SELECT left_memory_id, right_memory_id, pair_score FROM conflict_backlog "
                    "WHERE status='pending' AND (left_memory_id=? OR right_memory_id=?) "
                    "ORDER BY pair_score DESC, id DESC LIMIT ?",
                    (mid, mid, self.GRAPH_CANDIDATE_LIMIT + 1),
                ).fetchall()
        except sqlite3.Error:
            backlog_rows = []
        if len(backlog_rows) > self.GRAPH_CANDIDATE_LIMIT:
            truncated["candidate"] = True
            backlog_rows = backlog_rows[: self.GRAPH_CANDIDATE_LIMIT]
        for row in backlog_rows:
            left, right = int(row["left_memory_id"]), int(row["right_memory_id"])
            other = right if left == mid else left
            if other == mid:
                continue
            backlog_pairs.append((other, float(row["pair_score"] or 0.0)))
            involved.add(other)

        rows_by_id: dict[int, dict[str, Any]] = {}
        if involved:
            id_placeholders = ",".join("?" for _ in sorted(involved))
            try:
                with self.tools.db.connection() as conn:
                    rows_by_id = {
                        int(row["id"]): dict(row)
                        for row in conn.execute(
                            f"SELECT id, workspace, workspace_canonical FROM memories "
                            f"WHERE id IN ({id_placeholders})",
                            tuple(sorted(involved)),
                        ).fetchall()
                    }
            except sqlite3.Error:
                rows_by_id = {}

        for conflict, others in conflict_others:
            if explicit_none_scope and str(conflict.get("workspace_canonical") or "") != str(caller.canonical or ""):
                continue  # same gate as _conflict_detail_for_workspace (none + explicit)
            if isolation == "strict" and not all(
                _neighbor_visible(rows_by_id.get(member_id)) for member_id in others
            ):
                # strict sees the complete correlated snapshot or nothing —
                # a partial group would leak members' lifecycle/existence.
                continue
            for other_id in others:
                edges.append({
                    "source": mid, "target": other_id, "type": "conflict",
                    "conflict_id": conflict.get("id"), "status": conflict.get("status"),
                })
                neighbor_ids.add(other_id)

        for other, score in backlog_pairs:
            if not _neighbor_visible(rows_by_id.get(other)):
                continue
            edges.append({"source": mid, "target": other, "type": "candidate", "weight": score})
            neighbor_ids.add(other)

        subject = str(memory.get("subject") or "")
        subject_rows: list[Any] = []
        if subject:
            scope_clause = f" AND {scope_sql}" if scope_sql else ""
            try:
                with self.tools.db.connection() as conn:
                    subject_rows = conn.execute(
                        f"SELECT id FROM memories WHERE subject=? AND id<>? AND status='active'{scope_clause} "
                        "ORDER BY ingest_time DESC, id DESC LIMIT ?",
                        (subject, mid, *scope_params, self.GRAPH_SAME_SUBJECT_LIMIT + 1),
                    ).fetchall()
            except sqlite3.Error:
                subject_rows = []
        if len(subject_rows) > self.GRAPH_SAME_SUBJECT_LIMIT:
            truncated["same_subject"] = True
            subject_rows = subject_rows[: self.GRAPH_SAME_SUBJECT_LIMIT]
        for row in subject_rows:
            other = int(row["id"])
            edges.append({"source": mid, "target": other, "type": "same_subject"})
            neighbor_ids.add(other)

        tag_rows: list[Any] = []
        if tags:
            tag_placeholders = ",".join("?" for _ in tags)
            scope_clause_m = f" AND {scope_sql_m}" if scope_sql_m else ""
            try:
                with self.tools.db.connection() as conn:
                    tag_rows = conn.execute(
                        f"SELECT m.id AS id, COUNT(*) AS overlap FROM memories m "
                        f"JOIN json_each(m.tags) t ON t.value IN ({tag_placeholders}) "
                        f"WHERE m.id<>? AND m.status='active'{scope_clause_m} "
                        "GROUP BY m.id ORDER BY overlap DESC, m.id DESC LIMIT ?",
                        (*tags, mid, *scope_params_m, self.GRAPH_SAME_TAG_LIMIT + 1),
                    ).fetchall()
            except sqlite3.Error:
                tag_rows = []
        if len(tag_rows) > self.GRAPH_SAME_TAG_LIMIT:
            truncated["same_tag"] = True
            tag_rows = tag_rows[: self.GRAPH_SAME_TAG_LIMIT]
        for row in tag_rows:
            other = int(row["id"])
            edges.append({"source": mid, "target": other, "type": "same_tag", "weight": int(row["overlap"])})
            neighbor_ids.add(other)

        if len(neighbor_ids) > self.GRAPH_MAX_NEIGHBORS:
            keep: set[int] = set()
            for edge in edges:  # priority order: conflict → candidate → subject → tag
                target = int(edge["target"])
                if target in keep or len(keep) < self.GRAPH_MAX_NEIGHBORS:
                    keep.add(target)
                else:
                    truncated[edge["type"]] = True
            edges = [edge for edge in edges if int(edge["target"]) in keep]
            neighbor_ids = keep

        nodes: list[dict[str, Any]] = [{
            "id": mid, "kind": "self", "subject": memory.get("subject"),
            "status": memory.get("status"), "source_type": memory.get("source_type"),
            "version": memory.get("version"),
        }]
        if neighbor_ids:
            id_placeholders = ",".join("?" for _ in sorted(neighbor_ids))
            scope_clause = f" AND {scope_sql}" if scope_sql else ""
            try:
                with self.tools.db.connection() as conn:
                    node_rows = conn.execute(
                        f"SELECT id, subject, status, source_type, version FROM memories "
                        f"WHERE id IN ({id_placeholders}){scope_clause}",
                        (*sorted(neighbor_ids), *scope_params),
                    ).fetchall()
            except sqlite3.Error:
                node_rows = []
            visible_ids: set[int] = set()
            for row in node_rows:
                nodes.append({
                    "id": int(row["id"]), "kind": "memory", "subject": row["subject"],
                    "status": row["status"], "source_type": row["source_type"],
                    "version": row["version"],
                })
                visible_ids.add(int(row["id"]))
            # Drop edges whose target vanished (deleted memory, or outside the
            # caller's workspace scope) so the canvas never draws to nowhere.
            edges = [edge for edge in edges if int(edge["target"]) in visible_ids]
            neighbor_ids = visible_ids

        history = [
            {"version": row.get("version"), "changed_at": row.get("changed_at"), "reason": row.get("reason")}
            for row in self.tools.db.list_history(mid)
        ]
        try:
            internal_pairs = int(self.tools.db.internal_conflicts.pending_pair_count(mid))
        except (sqlite3.Error, AttributeError):
            internal_pairs = 0

        return {
            "memory": {
                "id": mid, "subject": memory.get("subject"), "status": memory.get("status"),
                "source_type": memory.get("source_type"), "tags": tags, "version": memory.get("version"),
            },
            "nodes": nodes,
            "edges": edges,
            "history": history,
            "internal_conflict_pairs": internal_pairs,
            "truncated": truncated,
        }

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
            "semantic_conflict.n_gpu_layers": self.settings.semantic_conflict_gpu_layers,
            "mcp.http.host": self.settings.mcp_http_host,
            "mcp.http.port": self.settings.mcp_http_port,
            "update_check.enabled": self.settings.update_check_enabled,
        }
        value = mapping.get(path)
        if isinstance(value, Path):
            return str(value)
        return value
