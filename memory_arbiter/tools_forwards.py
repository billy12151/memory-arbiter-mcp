"""产品参数转发壳与 memory_* 转发群 mixin（从 tools.py 搬出，拆分批 ⑥ 纯移动）。"""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING, cast
from .config import Settings
from .db import MemoryDB
from .scan_pipeline import ScanPipeline
from .search import search_memories, _linked_open_items_for_search  # noqa: F401 (monkeypatch seam, see pipeline/read.py:226)
from .surfaces import ProductSurfaces

if TYPE_CHECKING:
    from .pipeline.operations import OperationsPipeline
    from .pipeline.read import ReadPipeline
    from .pipeline.write import WritePipeline
    from .surfaces import ProductSurfaces
    from .config import Settings
    from .db import MemoryDB
    from .scan_pipeline import ScanPipeline

class _ToolsForwards:
    if TYPE_CHECKING:
        db: "MemoryDB"
        settings: "Settings"
        _scan_pipeline: "ScanPipeline"
        _operations: "OperationsPipeline"
        _read_pipeline: "ReadPipeline"
        _surfaces: "ProductSurfaces"
        _write_pipeline: "WritePipeline"

    @staticmethod
    def _payload_dict(data: dict[str, Any] | None) -> dict[str, Any]:
        return dict(data) if isinstance(data, dict) else {}

    def _judge_constraints(self) -> dict[str, Any]:
        return self._surfaces._judge_constraints()

    def _product_help(self, surface: str, topic: str | None = None) -> dict[str, Any]:
        return self._surfaces._product_help(surface, topic)

    def _invalid_product_call(self, surface: str, message: str, topic: str | None = None) -> dict[str, Any]:
        return self._surfaces._invalid_product_call(surface, message, topic)

    @staticmethod
    def _help_topic(payload: dict[str, Any], fallback_key: str) -> str | None:
        return ProductSurfaces._help_topic(payload, fallback_key)

    def _forward(
        self, surface: str, topic: str | None, fn: Callable[..., dict[str, Any]], **payload: Any,
    ) -> dict[str, Any]:
        return self._surfaces._forward(surface, topic, fn, **payload)

    @staticmethod
    def _alias_id(payload: dict[str, Any], target: str) -> None:
        return ProductSurfaces._alias_id(payload, target)

    def _int_product_arg(
        self, surface: str, value: Any, name: str, topic: str | None = None,
    ) -> int | dict[str, Any] | None:
        return self._surfaces._int_product_arg(surface, value, name, topic)

    def _require_id(
        self, surface: str, payload: dict[str, Any], name: str, topic: str | None = None,
    ) -> dict[str, Any] | None:
        return self._surfaces._require_id(surface, payload, name, topic)

    def _coerce_product_id(
        self, surface: str, payload: dict[str, Any], name: str, topic: str | None = None,
    ) -> dict[str, Any] | None:
        return self._surfaces._coerce_product_id(surface, payload, name, topic)

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        return ProductSurfaces._is_truthy(value)

    def _require_ws_strings(
        self, payload: dict[str, Any], names: tuple[str, ...], surface: str, topic: str | None = None,
    ) -> dict[str, Any] | None:
        return self._surfaces._require_ws_strings(payload, names, surface, topic)

    def memory(self, action: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory(action, data, **_)

    def memory_review(self, view: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory_review(view, data, **_)

    def memory_govern(self, action: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory_govern(action, data, **_)

    def memory_repair(self, task: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._surfaces.memory_repair(task, data, **_)

    def memory_write(self, **payload: Any) -> dict[str, Any]:
        return self._write_pipeline.memory_write(**payload)

    def memory_search(self, query: str = "", workspace: str | None = None, tags: list[str] | None = None, limit: int = 10, offset: int = 0, debug_ranking: bool = False, query_embedding: list[float] | None = None, tags_filter: list[str] | None = None, after_time: str | None = None, before_time: str | None = None, source_type: str | None = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True, content_mode: str = "preview", hit_window: int = 0, **_: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_search(
            query=query, workspace=workspace, tags=tags, limit=limit, offset=offset,
            debug_ranking=debug_ranking, query_embedding=query_embedding,
            tags_filter=tags_filter, after_time=after_time, before_time=before_time,
            source_type=source_type, include_linked_open_items=include_linked_open_items,
            include_conflict_signal=include_conflict_signal, content_mode=content_mode,
            hit_window=hit_window, **_,
        )

    def memory_batch_find(self, **payload: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_batch_find(**payload)

    def memory_batch_read(self, **payload: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_batch_read(**payload)

    def memory_search_expired(
        self,
        query: str = "",
        workspace: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        debug_ranking: bool = False,
        query_embedding: list[float] | None = None,
        tags_filter: list[str] | None = None,
        after_time: str | None = None,
        before_time: str | None = None,
        source_type: str | None = None,
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
        sections: str = "none",
        section_ids: list[int] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._read_pipeline.memory_get(
            memory_id=memory_id, sections=sections, section_ids=section_ids, **_,
        )

    def memory_recent(self, workspace: str | None = None, limit: int = 20, **_: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_recent(workspace, limit, **_)

    def memory_compare(self, left_id: int | None = None, right_id: int | None = None, left: dict[str, Any] | None = None, right: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        return self._read_pipeline.memory_compare(left_id, right_id, left, right, **_)

    def memory_arbitrate(self, left_id: int, right_id: int, mark_conflict: bool = True, authorized: bool = False, **_: Any) -> dict[str, Any]:
        return self._operations.memory_arbitrate(
            left_id, right_id, mark_conflict, self._is_truthy(authorized), **_,
        )

    def _with_resolution_guidance(self, conflict: dict[str, Any]) -> dict[str, Any]:
        return self._operations._with_resolution_guidance(conflict)

    def memory_list_conflicts(self, status: str = "open", limit: int = 50, source: str | None = None, **_: Any) -> dict[str, Any]:
        return self._operations.memory_list_conflicts(status, limit, source, **_)

    def memory_resolve_conflict(
        self, conflict_id: int, reason: str = "", status: str = "resolved", **_: Any,
    ) -> dict[str, Any]:
        resolve_conflict = cast(Callable[..., dict[str, Any]], self._operations.memory_resolve_conflict)
        return resolve_conflict(conflict_id, reason, status, **_)

    def memory_confirm(self, memory_id: int, source_ref: str | None = None, confidence: float = 1.0, authorized: bool = False, **_: Any) -> dict[str, Any]:
        return self._operations.memory_confirm(
            memory_id, source_ref, confidence, self._is_truthy(authorized), **_,
        )

    def memory_rename_workspace_canonical(
        self, old: str, new: str, reason: str | None = None, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_rename_workspace_canonical(old, new, reason, **_)

    def memory_migrate_workspace(
        self, reason: str | None = None, **payload: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_migrate_workspace(reason, **payload)

    def memory_move_memories_workspace(
        self,
        memory_ids: list[int] | None = None,
        new_workspace: str = "",
        reason: str | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_move_memories_workspace(
            memory_ids or [], new_workspace, reason, self._is_truthy(authorized), **_,
        )

    def memory_confirm_pending_workspace(
        self, memory_id: int, canonical: str, reason: str | None = None,
        authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_confirm_pending_workspace(
            memory_id, canonical, reason, self._is_truthy(authorized), **_,
        )

    def memory_confirm_workspaces(
        self,
        workspaces: list[str] | None = None,
        reason: str | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_confirm_workspaces(
            workspaces, reason, self._is_truthy(authorized), **_,
        )

    def memory_activate(
        self, memory_id: int, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_activate(memory_id, self._is_truthy(authorized), **_)

    def memory_supersede(
        self,
        memory_id: int,
        reason: str,
        superseded_by: int | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_supersede(
            memory_id, reason, superseded_by, self._is_truthy(authorized), **_,
        )

    def _update_check_status(self) -> dict[str, Any]:
        return self._operations._update_check_status()

    def memory_status(self, **_: Any) -> dict[str, Any]:
        return self._operations.memory_status(**_)

    def memory_doctor_overview(self, deep: bool = False, **_: Any) -> dict[str, Any]:
        return self._operations.memory_doctor_overview(deep, **_)

    def memory_set_entity(
        self, memory_id: int, entity: str | None = None, scope: str | None = None,
        clear: bool = False, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_set_entity(
            memory_id, entity, scope, clear, self._is_truthy(authorized), **_,
        )

    def memory_list_entities(
        self, limit: int = 50, include_unassigned: bool = True, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_list_entities(limit, include_unassigned, **_)

    def memory_list_workspaces(
        self, limit: int = 50, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_list_workspaces(limit, **_)

    def memory_rebuild_evidence(
        self, memory_ids: list[int] | None = None, dry_run: bool = True,
        batch_size: int = 50, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_rebuild_evidence(memory_ids, dry_run, batch_size, **_)

    def memory_audit_summary(self, **_: Any) -> dict[str, Any]:
        return self._operations.memory_audit_summary(**_)

    def memory_edit(
        self,
        memory_id: int,
        new_content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        patches: list[dict[str, Any]] | None = None,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        reason: str = "",
        authorized: bool = False,
        tags_only: bool = False,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_edit(
            memory_id,
            new_content=new_content,
            old_text=old_text,
            new_text=new_text,
            patches=patches,
            new_subject=new_subject,
            new_tags=new_tags,
            reason=reason,
            authorized=self._is_truthy(authorized),
            tags_only=tags_only,
            add_tags=add_tags,
            remove_tags=remove_tags,
            **_,
        )

    def memory_history(self, memory_id: int, **_: Any) -> dict[str, Any]:
        return self._operations.memory_history(memory_id, **_)

    def memory_cleanup_history(
        self,
        memory_id: int | None = None,
        older_than_days: int | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_cleanup_history(
            memory_id, older_than_days, self._is_truthy(authorized), **_,
        )

    def memory_replay_backup(
        self, dry_run: bool = True, authorized: bool = False,
        limit: int = 1_000, offset: int = 0, **_: Any,
    ) -> dict[str, Any]:
        return self._operations.memory_replay_backup(
            dry_run, self._is_truthy(authorized), limit, offset, **_,
        )

    def memory_normalize_workspaces(
        self, dry_run: bool = True, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        """Fold registered workspace spelling variants into first-seen canonicals."""
        dry_run = self._is_truthy(dry_run)
        if not dry_run and not self._is_truthy(authorized):
            # Same caller-confirmation gate as replay_backup: executing the
            # merge re-points memories and drops canonical rows, so it needs
            # explicit user authorization.
            return {
                "ok": False,
                "dry_run": False,
                "error": "authorized=True is required to execute workspace normalization",
                "action_required": "ask_user_for_authorization",
                "groups": [],
                "merged": [],
                "rejected_normalized": [],
                "skipped": [],
                "warnings": [],
            }
        return self.db.workspaces.normalize_workspace_canonicals(dry_run=dry_run)
