"""MemoryDB 委托墙 mixin（从 core.py 搬出，拆分批 ③ 纯移动）。

一行委托按 store 分组、保持原顺序；连接/事务生命周期与 scan watermark
六方法留守 core.py（watermark 与 store 职责重叠属拍板项，本批不动）。
"""
from __future__ import annotations

import sqlite3
from typing import Any, TYPE_CHECKING

from .memories import MemoriesStore  # runtime: _fetch_memory staticmethod 转发目标

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path
    from ..acl import WorkspaceScope
    from ..models import MemoryRecord
    from .conflicts import ConflictStore
    from .audit import AuditStore
    from .evidence_store import EvidenceStore
    from .meta import MetaStore
    from .schema import SchemaStore
    from .scan_queue import ScanQueueStore
    from .semantic_notices import SemanticNoticeStore
    from .workspaces import WorkspaceStore


class _CoreDelegatesMixin:
    if TYPE_CHECKING:
        schema: "SchemaStore"
        workspaces: "WorkspaceStore"
        memories: "MemoriesStore"
        conflicts: "ConflictStore"
        audit: "AuditStore"
        semantic_notices: "SemanticNoticeStore"
        meta: "MetaStore"
        evidence: "EvidenceStore"
        scan_queue: "ScanQueueStore"

    def ensure_workspace_vec_table(self, conn: sqlite3.Connection, dim: int) -> None:
        return self.schema.ensure_workspace_vec_table(conn, dim)

    def ensure_vec_tables(self, dim: int) -> list[str]:
        """Lazily create the derived vec0 tables at the model-reported dim."""
        return self.schema.ensure_vec_tables(dim)

    def ensure_vector_tables_for_repair(self) -> tuple[bool, list[str]]:
        return self.schema.ensure_vector_tables_for_repair()

    def missing_vector_tables(self) -> list[str]:
        return self.schema.missing_vector_tables()

    def resolve_workspace_canonical(
        self,
        ws_raw: str | None,
        embedder: Any = None,
        *,
        match_distance: float | None = None,
    ) -> dict[str, Any]:
        return self.workspaces.resolve_workspace_canonical(
            ws_raw, embedder, match_distance=match_distance,
        )

    def record_workspace_decision(
        self,
        workspace_name: str,
        canonical: str,
        *,
        status: str = "confirmed",
        force: bool = False,
    ) -> tuple[bool, list[str]]:
        return self.workspaces.record_workspace_decision(
            workspace_name, canonical, status=status, force=force,
        )

    def record_workspace_decision_on_conn(
        self,
        conn: sqlite3.Connection,
        workspace_name: str,
        canonical: str,
        *,
        status: str = "confirmed",
        force: bool = False,
    ) -> tuple[bool, list[str]]:
        return self.workspaces.record_workspace_decision(
            workspace_name, canonical, status=status, force=force, conn=conn,
        )

    def get_workspace_decision(self, workspace_name: str) -> dict[str, Any] | None:
        return self.workspaces.get_workspace_decision(workspace_name)

    def rename_workspace_canonical(
        self, old: str, new: str,
    ) -> tuple[int, list[str], bool]:
        return self.workspaces.rename_workspace_canonical(old, new)

    def migrate_workspace(
        self, from_ws: str, to_ws: str, *, embedder: Any = None,
    ) -> tuple[int, list[str], bool]:
        return self.workspaces.migrate_workspace(from_ws, to_ws, embedder=embedder)

    def prepare_workspace_canonical_embedding(
        self, canonical: str, embedder: Any = None,
    ) -> list[float] | None:
        return self.workspaces.prepare_workspace_canonical_embedding(canonical, embedder)

    def rebuild_workspace_canonical_vectors(
        self, embedder: Any, embedding_space_id: str,
    ) -> dict[str, Any]:
        return self.workspaces.rebuild_workspace_canonical_vectors(
            embedder, embedding_space_id,
        )

    def set_memory_workspace_canonical(
        self,
        memory_id: int,
        canonical: str,
        embedder: Any = None,
    ) -> tuple[bool, list[str]]:
        return self.workspaces.set_memory_workspace_canonical(memory_id, canonical, embedder)

    def set_memory_workspace_canonical_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        canonical: str,
        precomputed_embedding: list[float] | None = None,
    ) -> tuple[bool, list[str]]:
        return self.workspaces.set_memory_workspace_canonical(
            memory_id,
            canonical,
            None,
            conn=conn,
            precomputed_embedding=precomputed_embedding,
        )

    # (0.17.0 C5: db.evidence_knn — the unit-table KNN forward — was retired
    # with the unit channel; callers use row_knn.)

    def row_knn(
        self, query_embedding: list[float], *, k: int = 5, parent_status_filter: str = "active",
        workspace: WorkspaceScope = None, exclude_memory_id: int | None = None,
        exclude_workspaces: "list[str] | set[str] | frozenset[str] | None" = None,
        conn: sqlite3.Connection | None = None,
        include_subject_rows: bool = True,
        include_memory_ids: "list[int] | set[int] | None" = None,
        subject_rows_only: bool = False,
    ) -> list[dict[str, Any]]:
        """0.17.0 P2-2.4: row-level KNN convenience (conflict channel)."""
        return self.evidence.row_knn(query_embedding, k=k, parent_status_filter=parent_status_filter, workspace=workspace, exclude_memory_id=exclude_memory_id, exclude_workspaces=exclude_workspaces, conn=conn, include_subject_rows=include_subject_rows, include_memory_ids=include_memory_ids, subject_rows_only=subject_rows_only)

    def insert_memory(
        self,
        record: MemoryRecord,
        workspace_canonical: str | None = None,
        workspace_embedding: list[float] | None = None,
        *,
        register_workspace_canonical: bool = True,
    ) -> tuple[int | None, list[str]]:
        return self.memories.insert_memory(
            record, workspace_canonical, workspace_embedding,
            register_workspace_canonical=register_workspace_canonical,
        )

    def insert_memory_on_conn(
        self, conn: sqlite3.Connection, record: MemoryRecord,
        workspace_canonical: str | None = None,
    ) -> int:
        return self.memories.insert_memory_on_conn(conn, record, workspace_canonical)

    def _append_backup(self, record: MemoryRecord) -> None:
        return self.memories._append_backup(record)

    @staticmethod
    def _fetch_memory(conn: sqlite3.Connection, memory_id: int) -> dict[str, Any] | None:
        return MemoriesStore._fetch_memory(conn, memory_id)

    def get_memory_on_conn(self, conn: sqlite3.Connection, memory_id: int) -> dict[str, Any] | None:
        return self.memories.get_memory_on_conn(conn, memory_id)

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        return self.memories.get_memory(memory_id)

    def get_memories_by_ids(
        self, ids: list[int], *, conn: sqlite3.Connection | None = None,
    ) -> dict[int, dict[str, Any]]:
        return self.memories.get_memories_by_ids(ids, conn=conn)

    def get_memory_for_workspace(
        self, memory_id: int, ws_canonical: str, admitted: "WorkspaceScope" = None,
    ) -> dict[str, Any] | None:
        return self.memories.get_memory_for_workspace(memory_id, ws_canonical, admitted)

    def list_memories_for_workspace(
        self, ws_canonical: str, limit: int = 50, admitted: "WorkspaceScope" = None,
    ) -> list[dict[str, Any]]:
        return self.memories.list_memories_for_workspace(ws_canonical, limit, admitted)

    def update_memory(self, memory_id: int, updates: dict[str, Any]) -> bool:
        return self.memories.update_memory(memory_id, updates)

    def update_memory_on_conn(self, conn: sqlite3.Connection, memory_id: int, updates: dict[str, Any]) -> bool:
        return self.memories.update_memory(memory_id, updates, conn=conn)

    def list_memories(self, workspace: str | None = None, subject: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self.memories.list_memories(subject=subject, limit=limit)

    def active_subject_tag_rows(
        self, exclude_memory_id: int, workspace_canonical: str | None,
        *, limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.memories.active_subject_tag_rows(
            exclude_memory_id, workspace_canonical, limit=limit,
        )

    def memory_summary_knn(
        self, query_embedding: list[float], *, k: int, exclude_memory_id: int,
        workspace_canonical: str | None,
    ) -> list[dict[str, Any]]:
        """0.17.0 P2-7 convenience."""
        return self.memories.memory_summary_knn(
            query_embedding, k=k, exclude_memory_id=exclude_memory_id,
            workspace_canonical=workspace_canonical,
        )

    def upsert_subject_tags_vector(self, memory_id: int, embedding: list[float]) -> bool:
        return self.memories.upsert_subject_tags_vector(memory_id, embedding)

    def delete_subject_tags_vector(self, memory_id: int) -> bool:
        return self.memories.delete_subject_tags_vector(memory_id)

    def missing_subject_tags_rows(self) -> list[dict[str, Any]]:
        return self.memories.missing_subject_tags_rows()

    def missing_summary_vec_rows(self) -> list[dict[str, Any]]:
        return self.memories.missing_summary_vec_rows()

    def missing_row_vector_rows(self) -> list[dict[str, Any]]:
        return self.memories.missing_row_vector_rows()

    def all_summary_vectors(self) -> dict[int, tuple[str, list[float]]]:
        return self.memories.all_summary_vectors()

    def upsert_summary_vector(self, memory_id: int, embedding: list[float]) -> bool:
        return self.memories.upsert_summary_vector(memory_id, embedding)

    def delete_summary_vector(self, memory_id: int) -> bool:
        return self.memories.delete_summary_vector(memory_id)

    def count_filtered_memories(
        self,
        like_status_clause: str,
        tags_filter: list[str] | None,
        after_dt: datetime | None,
        before_dt: datetime | None,
        source_type: str | None,
        ws_canonical: WorkspaceScope = None,
    ) -> int:
        return self.memories.count_filtered_memories(
            like_status_clause, tags_filter, after_dt, before_dt, source_type, ws_canonical,
        )

    def recall_by_filters(
        self,
        like_status_clause: str,
        tags_filter: list[str] | None,
        after_dt: datetime | None,
        before_dt: datetime | None,
        source_type: str | None,
        limit: int,
        offset: int = 0,
        ws_canonical: WorkspaceScope = None,
    ) -> list[dict[str, Any]]:
        return self.memories.recall_by_filters(
            like_status_clause, tags_filter, after_dt, before_dt, source_type,
            limit, offset, ws_canonical,
        )

    def record_conflict_group(self, **kwargs: Any) -> dict[str, Any]:
        return self.conflicts.record_conflict_group(**kwargs)

    def void_conflicts(self, memory_ids: list[int], *, reason: str) -> int:
        return self.conflicts.void_conflicts(memory_ids, reason=reason)

    def get_conflict(self, conflict_id: int) -> dict[str, Any] | None:
        return self.conflicts.get_conflict(conflict_id)

    def escalate_structured_notice(self, notice_id: int, **kwargs: Any) -> dict[str, Any]:
        return self.conflicts.escalate_structured_notice(notice_id, **kwargs)

    def judge_conflict(self, conflict_id: int, **kwargs: Any) -> dict[str, Any]:
        return self.conflicts.judge_conflict(conflict_id, **kwargs)

    def resolve_conflicts_for(self, memory_id: int) -> int:
        return self.conflicts.resolve_conflicts_for(memory_id)

    def resolve_conflicts_for_on_conn(self, conn: sqlite3.Connection, memory_id: int) -> int:
        return self.conflicts.resolve_conflicts_for_on_conn(conn, memory_id)

    def list_conflicts(
        self,
        status: str = "open",
        limit: int = 50,
        source: str | None = None,
        workspace: WorkspaceScope = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.conflicts.list_conflicts(status, limit, source, workspace, offset)

    def list_open_conflicts_for_memory_ids(
        self, memory_ids: list[int], *, include_applying: bool = False,
    ) -> list[dict[str, Any]]:
        return self.conflicts.list_open_conflicts_for_memory_ids(memory_ids, include_applying=include_applying)

    def get_memory_summaries(
        self, memory_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        return self.audit.get_memory_summaries(memory_ids)

    def resolve_conflict(
        self, conflict_id: int, reason: str = "", status: str = "resolved",
        *, expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return self.conflicts.resolve_conflict(
            conflict_id, reason=reason, status=status, expected_revision=expected_revision,
        )

    def is_pair_dismissed(self, left_id: int, right_id: int) -> bool:
        return self.conflicts.is_pair_dismissed(left_id, right_id)

    def get_memory_version(self, memory_id: int) -> int | None:
        return self.conflicts.get_memory_version(memory_id)

    def dismissed_pairs_for(self, memory_ids: list[int]) -> set[tuple[int, int]]:
        return self.conflicts.dismissed_pairs_for(memory_ids)

    # ------------------------------------------------------------------
    #  Semantic write-time notices
    # ------------------------------------------------------------------

    def record_semantic_notice(
        self,
        *,
        memory_id: int,
        peer_id: int | None,
        severity: str,
        notice_type: str,
        title: str,
        message: str,
        payload: dict[str, Any],
        dedupe_key: str | None = None,
        conflict_id: int | None = None,
        left_version: int | None = None,
        right_version: int | None = None,
        source: str = "semantic_evidence",
    ) -> dict[str, Any]:
        return self.semantic_notices.record_semantic_notice(
            memory_id=memory_id,
            peer_id=peer_id,
            severity=severity,
            notice_type=notice_type,
            title=title,
            message=message,
            payload=payload,
            dedupe_key=dedupe_key,
            conflict_id=conflict_id,
            left_version=left_version,
            right_version=right_version,
            source=source,
        )

    def claim_next_semantic_notice(self, workspace_canonical: WorkspaceScope = None) -> dict[str, Any] | None:
        return self.semantic_notices.claim_next_semantic_notice(workspace_canonical)

    def recent_semantic_notices_for_memory(self, memory_id: int, limit: int = 64) -> list[dict[str, Any]]:
        return self.semantic_notices.recent_semantic_notices_for_memory(memory_id, limit=limit)

    def demote_semantic_notice_to_info(self, notice_id: int) -> bool:
        return self.semantic_notices.demote_semantic_notice_to_info(notice_id)

    def read_semantic_notice(
        self, notice_id: int, workspace_canonical: WorkspaceScope = None,
    ) -> dict[str, Any] | None:
        return self.semantic_notices.read_semantic_notice(notice_id, workspace_canonical)

    def list_semantic_notices(
        self, status: str = "open", limit: int = 10, workspace_canonical: WorkspaceScope = None,
    ) -> list[dict[str, Any]]:
        return self.semantic_notices.list_semantic_notices(
            status=status, limit=limit, workspace_canonical=workspace_canonical,
        )

    def semantic_notice_counts(
        self, workspace_canonical: WorkspaceScope = None,
    ) -> dict[str, int]:
        return self.semantic_notices.semantic_notice_counts(workspace_canonical)

    def is_semantic_pair_closed(
        self,
        left_id: int,
        right_id: int,
        left_version: int | None = None,
        right_version: int | None = None,
        notice_type: str = "semantic_evidence",
    ) -> bool:
        return self.semantic_notices.is_semantic_pair_closed(
            left_id, right_id, left_version, right_version, notice_type,
        )

    def update_semantic_notice_status(
        self,
        notice_id: int,
        status: str,
        reason: str = "",
        workspace_canonical: WorkspaceScope = None,
        conflict_id: int | None = None,
    ) -> dict[str, Any]:
        return self.semantic_notices.update_semantic_notice_status(
            notice_id, status, reason, workspace_canonical, conflict_id,
        )

    @property
    def scan_log_path(self) -> Path:
        return self.audit.scan_log_path

    @property
    def attention_log_path(self) -> Path:
        return self.audit.attention_log_path

    def log_attention(self, *, trigger: str, source: str, memory_ids: list[int]) -> None:
        return self.audit.log_attention(trigger=trigger, source=source, memory_ids=memory_ids)

    def log_scan(
        self,
        *,
        duration_sec: float,
        client: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        return self.audit.log_scan(
            duration_sec=duration_sec, client=client, agent_id=agent_id,
        )

    def _scan_log_last_completed(self) -> dict[str, Any] | None:
        return self.audit.scan_log_last_completed()

    # ------------------------------------------------------------------
    #  Edit / History
    # ------------------------------------------------------------------

    def update_tags_low_side_effect(
        self,
        memory_id: int,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        authorized: bool = False,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        return self.memories.update_tags_low_side_effect(
            memory_id, add_tags, remove_tags, authorized, conn=conn,
        )

    def update_metadata_fields_low_side_effect(
        self,
        memory_id: int,
        set_fields: dict[str, Any] | None = None,
        clear_fields: list[str] | None = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        return self.memories.update_metadata_fields_low_side_effect(
            memory_id, set_fields=set_fields, clear_fields=clear_fields, authorized=authorized,
        )

    def update_metadata_fields_low_side_effect_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        set_fields: dict[str, Any] | None = None,
        clear_fields: list[str] | None = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        return self.memories.update_metadata_fields_low_side_effect_on_conn(
            conn, memory_id, set_fields=set_fields,
            clear_fields=clear_fields, authorized=authorized,
        )

    def list_entities(
        self,
        limit: int = 50,
        include_unassigned: bool = True,
    ) -> dict[str, Any]:
        return self.memories.list_entities(limit, include_unassigned)

    def find_metadata_overlap_candidates(
        self,
        subject: str | None,
        tags: list[str],
        exclude_id: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.memories.find_metadata_overlap_candidates(subject, tags, exclude_id, limit)

    def find_semantic_overlap_candidates(
        self,
        subject: str | None,
        tags: list[str],
        exclude_id: int,
        limit: int = 50,
        canonical_workspace: str | None = None,
        isolation: str = "none",
    ) -> list[dict[str, Any]]:
        return self.memories.find_semantic_overlap_candidates(
            subject, tags, exclude_id, limit,
            canonical_workspace=canonical_workspace,
            isolation=isolation,
        )

    def edit_memory(
        self,
        memory_id: int,
        new_content: str,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        reason: str | None = None,
    ) -> int | None:
        return self.memories.edit_memory(memory_id, new_content, new_subject, new_tags, reason)


    def edit_memory_intent(
        self,
        memory_id: int,
        *,
        new_content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        patches: list[dict[str, Any]] | None = None,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        reason: str | None = None,
        authorized: bool = False,
        expected_version: int | None = None,
        expected_content_hash: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        return self.memories.edit_memory_intent(
            memory_id,
            new_content=new_content,
            old_text=old_text,
            new_text=new_text,
            patches=patches,
            new_subject=new_subject,
            new_tags=new_tags,
            add_tags=add_tags,
            remove_tags=remove_tags,
            reason=reason,
            authorized=authorized,
            expected_version=expected_version,
            expected_content_hash=expected_content_hash,
            conn=conn,
        )

    def list_history(self, memory_id: int) -> list[dict[str, Any]]:
        return self.memories.list_history(memory_id)

    def cleanup_history(
        self, memory_id: int | None = None, older_than_days: int | None = None,
        *, conn: sqlite3.Connection | None = None,
    ) -> int:
        return self.memories.cleanup_history(memory_id, older_than_days, conn=conn)

    def audit_summary(self) -> dict[str, Any]:
        return self.audit.audit_summary()

    # ==================================================================
    #  v0.6.0: _vec_index_meta + vec-index state machine
    # ==================================================================

    # ---- _vec_index_meta CRUD ----

    def conflict_scan_state(self) -> dict[str, Any]:
        return self.meta.conflict_scan_state()

    def scan_page_progress_state(self) -> dict[str, Any] | None:
        return self.meta.scan_page_progress_state()

    def record_scan_page_progress(
        self, *,
        after_memory_id: int,
        next_anchor_memory_id: int | None,
        anchor_buckets: list[dict[str, Any]] | None,
        client: str | None,
    ) -> bool:
        return self.meta.record_scan_page_progress(
            after_memory_id=after_memory_id,
            next_anchor_memory_id=next_anchor_memory_id,
            anchor_buckets=anchor_buckets,
            client=client,
        )

    def rearm_conflict_scan_if_drifted(self) -> bool:
        return self.meta.rearm_conflict_scan_if_drifted()

    def record_conflict_scan_page(
        self,
        *,
        epoch: str,
        detector_version: str,
        boundary: dict[str, Any],
        after_memory_id: int,
        next_anchor_memory_id: int | None,
        anchors_scanned: int,
        workspace: WorkspaceScope = None,
    ) -> bool:
        return self.meta.record_conflict_scan_page(
            epoch=epoch,
            detector_version=detector_version,
            boundary=boundary,
            after_memory_id=after_memory_id,
            next_anchor_memory_id=next_anchor_memory_id,
            anchors_scanned=anchors_scanned,
            workspace=workspace,
        )

    def complete_conflict_scan(
        self, *, epoch: str, detector_version: str, boundary: dict[str, Any]
    ) -> bool:
        return self.meta.complete_conflict_scan(
            epoch=epoch, detector_version=detector_version, boundary=boundary,
        )

    def get_vec_index_state(self) -> dict[str, Any]:
        return self.meta.get_vec_index_state()

    def get_active_dim(self) -> int | None:
        return self.meta.get_active_dim()

    def set_active_dim(self, dim: int) -> None:
        return self.meta.set_active_dim(dim)

    def mark_space_rebuild_started(self) -> None:
        return self.meta.mark_space_rebuild_started()

    def space_rebuild_pending_ids(self, limit: int) -> list[int]:
        return self.meta.space_rebuild_pending_ids(limit)

    def stale_index_ids(self, limit: int, workspace: WorkspaceScope = None) -> list[int]:
        return self.meta.stale_index_ids(limit, workspace)

    def maybe_complete_space_rebuild(self, embedding_space_id: str) -> bool:
        return self.meta.maybe_complete_space_rebuild(embedding_space_id)

    def require_space_rebuild(self, embedding_space_id: str, reason: str) -> None:
        return self.meta.require_space_rebuild(embedding_space_id, reason)

    def init_vec_index_state(
        self,
        embedding_space_id: str | None,
        has_managed_embedder: bool,
        active_dim: int | None = None,
    ) -> None:
        return self.meta.init_vec_index_state(
            embedding_space_id, has_managed_embedder, active_dim=active_dim,
        )
