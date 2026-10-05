"""状态与运维组（status/doctor_overview/list_workspaces/audit_summary/rebuild_evidence/replay 两方法/_update_check_status；WORKSPACE_RECALL_ADMISSION 读取点随迁本模块，patch 缝路径变更见方案 §6-1，从 operations.py 搬出，拆分批 ④ 纯移动）。
"""
from __future__ import annotations


from typing import Any, TYPE_CHECKING

from .. import __version__
from ..acl import CallerWorkspace, WorkspaceScope, workspace_scope_sql
from ..constants import (
    WORKSPACE_MIN_NAME_LEN,
    WORKSPACE_RECALL_ADMISSION,
    WORKSPACE_RECALL_CUTOFF,
    WORKSPACE_WEAK_VECTOR_WEIGHT,
)
from ..embedder import ManagedEmbedder

if TYPE_CHECKING:
    from ..config import Settings
    from ..db import MemoryDB
    from ..update_monitor import UpdateMonitor
    from ..tools import MemoryTools

class _OpsStatus:
    if TYPE_CHECKING:
        db: "MemoryDB"
        settings: "Settings"
        _tools: "MemoryTools"

        # 主类委托薄层/跨 mixin 成员的 mypy strict 声明（attr-defined）
        @property
        def _update_monitor(self) -> "UpdateMonitor | None": ...
        def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace": ...
        def _conflict_detail_for_workspace(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _embedding_configured(self) -> bool: ...
        def _post_commit(
            self, *args: Any, **kwargs: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]: ...
        def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _get_memory_visible(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _is_truthy(self, *args: Any, **kwargs: Any) -> bool: ...
        def _semantic_notice_workspace_scope(self, *args: Any, **kwargs: Any) -> "WorkspaceScope": ...
        def _semantic_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
        def _strict_acl_unavailable(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def current_agent_id(self) -> "str | None": ...
        def current_client(self) -> "str | None": ...
        def wait_semantic_worker_drained(self, *args: Any, **kwargs: Any) -> bool: ...
        @staticmethod
        def _compare_memories(*args: Any, **kwargs: Any) -> Any: ...
        def memory_list_conflicts(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def _update_check_status(self) -> dict[str, Any]:
        if self._update_monitor is None:
            status = "disabled" if not self.settings.update_check_enabled else "not_started"
            return {"enabled": self.settings.update_check_enabled, "status": status, "current_version": __version__}
        return self._update_monitor.update_status()

    def memory_status(self, **_: Any) -> dict[str, Any]:
        evidence_status: dict[str, Any] = {
            "available": False,
        }
        try:
            evidence_status.update({"available": True, **self.db.evidence.coverage()})
            with self.db.connection() as conn:
                rows = conn.execute("SELECT key,value FROM migration_state").fetchall()
            evidence_status["migration"] = {str(row["key"]): str(row["value"]) for row in rows}
        except Exception as exc:
            evidence_status["error"] = str(exc)
        conflict_scan = self.db.conflict_scan_state()
        return self.db.state.response(
            {
                "arbiter_version": __version__,
                "db_path": str(self.settings.db_path),
                "backup_jsonl": str(self.settings.backup_jsonl),
                "sqlite_vec_available": self.db.state.sqlite_vec_available,
                "fts5_available": self.db.state.fts5_available,
                "sqlite_writable": self.db.state.sqlite_writable,
                "jsonl_backup_active": self.db.state.jsonl_backup_active,
                # B5（0.17.1 优化批）：最后一次降级写入时间（观测；缺省 null）
                "jsonl_backup_last_used_at": self.db.state.jsonl_backup_last_used_at,
                "client": self.current_client(),
                "agent_id": self.current_agent_id(),
                "workspace": self.settings.workspace,
                "config_warnings": self.settings.config_warnings,
                "embedding_configured": self._embedding_configured(),
                "embedding_auto_query": self.settings.embedding_auto_query,
                "embedding_auto_write": self.settings.embedding_auto_write,
                "local_text_evidence": evidence_status,
                "conflict_scan": conflict_scan,
                "conflict_scan_required": conflict_scan["required"],
                # R2-S1：local_text_index_worker 随 LocalTextIndexWorker 退役
                # 删除——worker 观测在同一回执的 semantic_conflict.worker。
                "isolation": self.settings.isolation,
                "workspace_recall": {
                    "admission_enabled": WORKSPACE_RECALL_ADMISSION,
                    "cutoff": WORKSPACE_RECALL_CUTOFF,
                    "min_name_len": WORKSPACE_MIN_NAME_LEN,
                    "weak_vector_weight": WORKSPACE_WEAK_VECTOR_WEIGHT,
                    "strict_scope_behavior": (
                        "guarded_vector_neighbors"
                        if WORKSPACE_RECALL_ADMISSION
                        else "exact_canonical"
                    ),
                },
                "tool_surface": {
                    "profile": "product",
                    "default_profile": "product",
                    "product_tools": ["memory", "memory_review", "memory_govern", "memory_repair"],
                },
                "update_check": self._update_check_status(),
                "vec_index_state": self.db.get_vec_index_state(),
                "semantic_conflict": self._semantic_status(
                    self._semantic_notice_workspace_scope(_.get("workspace")),
                ),
            },
            extra_warnings=self.settings.config_warnings,
        )

    def memory_doctor_overview(self, deep: bool = False, **_: Any) -> dict[str, Any]:
        """Run a read-only health check and return a graded diagnostic report.

        Covers config integrity, evidence indexing, data consistency, and
        capacity. Each finding carries a severity and a
        fix_hint tailored to the current config.json. Read-only: never writes,
        never changes schema. ``deep=true`` additionally loads the GGUF model
        for a dimension probe (seconds-level cost); MCP reuses an
        already-loaded embedder at zero cost.
        """
        from ..doctor import doctor_overview_mcp, report_to_dict

        report = doctor_overview_mcp(
            self.db, self.settings, deep,
            embedder_probe=self._ensure_embedder,
            runtime_state=self.db.state,
        )
        if self._update_monitor is not None:
            self._update_monitor.record_doctor_run()
        data = report_to_dict(report)
        data["update_check"] = self._update_check_status()
        # A1 (0.15.14): the pair-timing ring rides on doctor so the
        # slow-vs-clean question (competition / long decode / retry) is
        # answerable from one health view without a separate status call.
        data["semantic_pair_timing"] = self._tools._pair_timing_summary()
        return self.db.state.response(data)
    def memory_rebuild_evidence(
        self,
        memory_ids: list[int] | None = None,
        dry_run: bool = True,
        batch_size: int = 50,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            batch = max(1, min(500, int(batch_size)))
        except (TypeError, ValueError):
            return self.db.state.response({"error": "batch_size must be an integer"}, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        # Vec tables are created lazily at the first embedder load, so their
        # absence only counts as "missing" once the vec state says the index
        # should be live (state=ready with embedding configured) — a fresh
        # library in its pre-first-embed window is healthy, not broken.
        missing_vector_tables = (
            self.db.missing_vector_tables()
            if (
                self.settings.embedding_model_path is not None
                and self.db.get_vec_index_state().get("state") == "ready"
            )
            else []
        )
        table_repair: dict[str, Any] = {
            "required": bool(missing_vector_tables),
            "missing_tables": missing_vector_tables,
            "recreated": False,
            "warnings": [],
        }
        if not dry_run and self.settings.embedding_model_path is not None:
            recreated, repair_warnings = self.db.ensure_vector_tables_for_repair()
            table_repair.update({"recreated": recreated, "warnings": repair_warnings})
            if repair_warnings:
                return self.db.state.response(
                    {"error": "vector table repair failed", "vector_table_repair": table_repair},
                    ok=False, extra_warnings=list(caller.warnings) + repair_warnings,
                )
            if recreated:
                embedder, embedder_warnings = self._ensure_embedder()
                if embedder is None:
                    return self.db.state.response(
                        {"error": "embedding runtime unavailable after vector table repair",
                         "vector_table_repair": table_repair},
                        ok=False, extra_warnings=list(caller.warnings) + embedder_warnings,
                    )
                self.db.require_space_rebuild(
                    embedder.embedding_space_id, "derived vector tables were recreated",
                )
        mismatch_rebuild = False
        workspace_rebuild: dict[str, Any] = {"ok": True, "rebuilt": 0}
        if memory_ids is None:
            # Embedding-space mismatch is a whole-index condition: existing
            # rows may be healthy-looking but live in the old space, so the
            # rebuild must republish EVERYTHING (not just stale rows) before
            # the vec channel can be re-enabled. The pending set is keyed on
            # the evidence-id epoch, so republished rows drop out and repeated
            # calls paginate forward instead of re-selecting the first batch
            # forever. Strict-isolation callers only see their own workspace
            # and therefore cannot complete a global rebuild; they keep the
            # stale-only selection. The epoch mark itself is written only on
            # the execute path so dry runs stay side-effect-free.
            vec_state = self.db.get_vec_index_state()
            mismatch_rebuild = (
                (
                    bool(missing_vector_tables)
                    or (
                        vec_state.get("state") == "mismatch"
                        and vec_state.get("target_space_id") is not None
                    )
                )
                and caller.isolation != "strict"
            )
            if mismatch_rebuild:
                if not dry_run:
                    embedder, embedder_warnings = self._ensure_embedder()
                    if embedder is None:
                        return self.db.state.response(
                            {
                                "error": "embedding runtime unavailable for space rebuild",
                                "dry_run": False,
                                "queued": 0,
                                "failed": 0,
                                "results": [],
                                "workspace_vector_rebuild": {
                                    "ok": False,
                                    "error": "embedding_runtime_unavailable",
                                },
                                "vec_index_state": self.db.get_vec_index_state(),
                            },
                            ok=False,
                            extra_warnings=list(caller.warnings) + list(embedder_warnings),
                        )
                    workspace_rebuild = self.db.rebuild_workspace_canonical_vectors(
                        embedder, embedder.embedding_space_id,
                    )
                    if not workspace_rebuild.get("ok"):
                        return self.db.state.response(
                            {
                                "error": "workspace vector rebuild failed",
                                "workspace_vector_rebuild": workspace_rebuild,
                                "dry_run": False,
                                "vec_index_state": self.db.get_vec_index_state(),
                            },
                            ok=False,
                            extra_warnings=list(caller.warnings),
                        )
                    self.db.mark_space_rebuild_started()
                ids = (
                    self.db.stale_index_ids(batch)
                    if dry_run and missing_vector_tables
                    else self.db.space_rebuild_pending_ids(batch)
                )
                if not dry_run and not ids:
                    # Nothing pending: settle the flip now instead of waiting
                    # for an unrelated write to trigger the completion check.
                    embedder, _w = self._ensure_embedder()
                    if embedder is not None:
                        self.db.maybe_complete_space_rebuild(embedder.embedding_space_id)
            else:
                ids = self.db.stale_index_ids(
                    batch,
                    workspace=caller.scope_canonicals() if caller.isolation == "strict" else None,
                )
        else:
            try:
                # An explicitly enumerated repair set is not batch-capped:
                # silently truncating it under-repairs (the discovery path
                # below paginates with `batch` instead).
                requested = list(dict.fromkeys(int(value) for value in memory_ids))
            except (TypeError, ValueError):
                return self.db.state.response({"error": "memory_ids must contain integers"}, ok=False)
            ids = [mid for mid in requested if caller.isolation != "strict" or self._get_memory_visible(mid, caller)]
        if dry_run:
            return self.db.state.response(
                {
                    "dry_run": True,
                    "memory_ids": ids,
                    "count": len(ids),
                    "workspace_vectors": "rebuild_required" if mismatch_rebuild else "unchanged",
                    "vector_table_repair": table_repair,
                    "vec_index_state": self.db.get_vec_index_state(),
                },
                extra_warnings=list(caller.warnings),
            )
        results = [
            {"memory_id": mid, **self._post_commit(mid, self.db.get_memory(mid), recheck_conflicts=False)[0]}
            for mid in ids
        ]
        # C2: the queue dedupes by task_id — "completed" means this version's
        # job already ran (rows persisted in the CURRENT space here, since a
        # mismatch rebuild re-queues after the epoch mark); it counts as
        # success, not failure.
        failed = sum(item.get("status") not in {"queued", "completed"} for item in results)
        # A table repair / dim swap recreates subject_tags_vec empty; the
        # idempotent backfill restores hint vectors now instead of waiting
        # for a process restart (fail-open, no-rows-missing is one cheap
        # scan).
        try:
            embedder, _w = self._ensure_embedder()
            if embedder is not None:
                self._tools._backfill_subject_tags_vectors(embedder)
        except Exception:
            pass
        worker_snapshot = self._tools._semantic_worker.snapshot()
        return self.db.state.response(
            {
                "dry_run": False,
                "queued": len(results) - failed,
                "failed": failed,
                "results": results,
                "semantic_worker": worker_snapshot,
                "workspace_vector_rebuild": (
                    workspace_rebuild if mismatch_rebuild else {"ok": True, "rebuilt": 0}
                ),
                "vector_table_repair": table_repair,
                # Surfaces a skipped flip (e.g. embedder unavailable with an
                # empty pending set): without it, "queued=0" is
                # indistinguishable from a settled mismatch->ready flip.
                "vec_index_state": self.db.get_vec_index_state(),
            },
            ok=failed == 0,
            extra_warnings=list(caller.warnings),
        )
    def memory_list_workspaces(
        self,
        limit: int = 50,
        workspace: str | None = None,
        *,
        admitted: "frozenset[str] | set[str] | None" = None,
        **_: Any,
    ) -> dict[str, Any]:
        """C2（owner 2026-10-03 拍板）：已有桶列表——agent 选桶前的发现接口。

        单 SQL 聚合（owner SQL 单查询纪律）：canonical×active/pending 计数×
        最近写入 + 别名数并查。排序最近写入倒序。

        A8（0.17.1 修复批）：strict 调用者的 admitted 集**进 SQL**（此前
        LIMIT 先于 surfaces 的事后过滤：limit=1 时自有桶被更晚更新的外来桶
        挤出窗口，实测返回空列表；count 还被重算成过滤后长度）。调用方
        （surfaces）只对 strict 传 admitted；none/weak 保持全量。count 现在
        就是 SQL 返回行数（= 本次可见桶数）。
        """
        limit = max(1, min(int(limit or 50), 200))
        scope_sql = ""
        scope_params: list[Any] = []
        if admitted is not None:
            if not admitted:
                return self.db.state.response(
                    {"view": "workspaces", "count": 0, "workspaces": []}
                )
            scope_sql, scope_params = workspace_scope_sql("c.name", sorted(admitted))
            scope_sql = f"WHERE {scope_sql}"
        with self.db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.name AS canonical,
                       COALESCE(a.active_count, 0) AS active_count,
                       COALESCE(p.pending_count, 0) AS pending_count,
                       COALESCE(al.alias_count, 0) AS alias_count,
                       COALESCE(a.last_write, p.last_write) AS last_write_at
                FROM workspace_canonicals c
                LEFT JOIN (
                    SELECT COALESCE(NULLIF(workspace_canonical, ''), workspace) AS bucket,
                           COUNT(*) AS active_count, MAX(created_at) AS last_write
                    FROM memories WHERE status = 'active' GROUP BY 1
                ) a ON a.bucket = c.name
                LEFT JOIN (
                    SELECT COALESCE(NULLIF(workspace_canonical, ''), workspace) AS bucket,
                           COUNT(*) AS pending_count, MAX(created_at) AS last_write
                    FROM memories WHERE status = 'pending' GROUP BY 1
                ) p ON p.bucket = c.name
                LEFT JOIN (
                    SELECT canonical, COUNT(*) AS alias_count
                    FROM workspace_aliases GROUP BY canonical
                ) al ON al.canonical = c.name
                {scope_sql}
                ORDER BY last_write_at DESC, c.name ASC
                LIMIT ?
                """,
                (*scope_params, limit),
            ).fetchall()
        buckets = []
        for row in rows:
            active, pending = int(row["active_count"]), int(row["pending_count"])
            buckets.append({
                "canonical": str(row["canonical"]),
                "active_count": active,
                "pending_count": pending,
                "alias_count": int(row["alias_count"]),
                "last_write_at": row["last_write_at"],
                "empty": (active + pending) == 0,
            })
        return self.db.state.response({"view": "workspaces", "count": len(buckets), "workspaces": buckets})

    def memory_audit_summary(self, **_: Any) -> dict[str, Any]:
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        scoped = caller.isolation == "strict" or (
            caller.isolation == "none" and caller.source == "explicit"
        )
        if not scoped:
            summary = self.db.audit_summary()
            # P2 #7: the audit view tail carries the newest governance rows
            # (rename/migrate/confirm_pending reasons).
            summary["governance_audit"] = self.db.audit.recent_governance_actions(limit=20)
            return self.db.state.response(summary)
        workspace_scope = (
            caller.scope_canonicals() if caller.isolation == "strict" else caller.canonical
        )
        with self.db.connection() as conn:
            scope_sql, scope_params = workspace_scope_sql(
                "COALESCE(NULLIF(workspace_canonical, ''), workspace)", workspace_scope,
            )
            mem_rows = conn.execute(
                "SELECT COALESCE(NULLIF(workspace_canonical, ''), workspace) AS workspace, "
                "COUNT(*) AS count, MIN(event_time) AS oldest, MAX(event_time) AS newest, source_type "
                f"FROM memories WHERE status != 'deleted' AND {scope_sql} "
                "GROUP BY workspace, source_type",
                scope_params,
            ).fetchall()
        def _empty_workspace_bucket() -> dict[str, Any]:
            return {
                "count": 0, "oldest": None, "newest": None,
                "open_conflicts": 0, "by_source_type": {},
            }

        # Pre-seed every admitted workspace. This preserves the single-canonical strict
        # response shape when the caller owns zero memories and makes each
        # admitted workspace explicit rather than synthesizing buckets only
        # when rows happen to exist.
        scope_names_value = (
            caller.scope_canonicals()
            if caller.isolation == "strict"
            else ((caller.canonical,) if caller.canonical else ())
        )
        workspaces: dict[str, dict[str, Any]] = {
            name: _empty_workspace_bucket() for name in scope_names_value
        }
        total_count = 0
        for row in mem_rows:
            ws_name = str(row["workspace"] or caller.canonical or "")
            bucket = workspaces.setdefault(ws_name, _empty_workspace_bucket())
            count = int(row["count"] or 0)
            bucket["count"] = int(bucket["count"] or 0) + count
            total_count += count
            if row["oldest"] is not None and (bucket["oldest"] is None or row["oldest"] < bucket["oldest"]):
                bucket["oldest"] = row["oldest"]
            if row["newest"] is not None and (bucket["newest"] is None or row["newest"] > bucket["newest"]):
                bucket["newest"] = row["newest"]
            if row["source_type"] is not None:
                source_type = str(row["source_type"])
                by_source_type = bucket["by_source_type"]
                by_source_type[source_type] = by_source_type.get(source_type, 0) + count
        if caller.isolation == "strict":
            conflicts = self.memory_list_conflicts(
                status="open", limit=10000, workspace=caller.workspace,
            ).get("data", {}).get("conflicts", [])
            conflicts += self.memory_list_conflicts(
                status="applying", limit=10000, workspace=caller.workspace,
            ).get("data", {}).get("conflicts", [])
        else:
            conflicts = self.db.list_conflicts(
                status="open", limit=10000, workspace=workspace_scope,
            )
            conflicts += self.db.list_conflicts(
                status="applying", limit=10000, workspace=workspace_scope,
            )
        for conflict in conflicts:
            ws_name = str(conflict.get("workspace_canonical") or "").strip()
            if ws_name in workspaces:
                bucket = workspaces[ws_name]
                bucket["open_conflicts"] = int(bucket["open_conflicts"] or 0) + 1
        summary = {
            "workspaces": workspaces,
            "total_memories": total_count,
            "total_open_conflicts": len(conflicts),
            # P2 #7: scoped callers see only governance rows touching their
            # own scope (strict ACL — no cross-bucket rename history leak).
            "governance_audit": self.db.audit.recent_governance_actions(
                limit=20, scope=set(scope_names_value),
            ),
            **caller.response_fields(),
        }
        return self.db.state.response(summary, extra_warnings=list(caller.warnings))
