"""conflicts 读查询与 void（从 conflicts.py 搬出，拆分批 ②c 纯移动）。"""
from __future__ import annotations

import sqlite3
from typing import Any, TYPE_CHECKING

from ..degrade import DegradeState
from ..models import utc_now_iso

from ._conflicts_helpers import (
    _canonical_json,
    _decode_row,
)
from ..acl import WorkspaceScope, workspace_scope_sql

if TYPE_CHECKING:
    from .core import MemoryDB


class _ConflictsReadMixin:
    """conflicts 读查询与 void（从 conflicts.py 搬出，拆分批 ②c 纯移动）。_MAX_MEMBERS patch 缝在留守的 record/escalate 侧（§6-7），本组无读点。"""

    # 拆分批 ②c：声明式注解（mypy strict；形态对齐主类）
    if TYPE_CHECKING:
        _db: "MemoryDB"
        from typing import Any as _Any
        @property
        def _db_available(self) -> bool: ...
        @property
        def state(self) -> "DegradeState": ...
        def connection(self) -> "_Any": ...
        def write_transaction(self) -> "_Any": ...

    def get_conflict(self, conflict_id: int) -> dict[str, Any] | None:
        if not self._db_available:
            return None
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM conflicts WHERE id=?", (int(conflict_id),)).fetchone()
        return _decode_row(row) if row else None

    def list_conflicts(
        self,
        status: str = "open",
        limit: int = 50,
        source: str | None = None,
        workspace: "WorkspaceScope" = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not self._db_available:
            return []
        sql = "SELECT * FROM conflicts WHERE status=?"
        params: list[Any] = [status]
        if source is not None:
            sql += " AND source=?"
            params.append(source)
        scope_sql, scope_params = workspace_scope_sql("workspace_canonical", workspace)
        if scope_sql:
            sql += f" AND {scope_sql}"
            params.extend(scope_params)
        sql += " ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self.connection() as conn:
            return [_decode_row(row) for row in conn.execute(sql, params).fetchall()]

    def list_open_conflicts_for_memory_ids(
        self, memory_ids: list[int], *, include_applying: bool = False,
    ) -> list[dict[str, Any]]:
        wanted = sorted({int(value) for value in memory_ids})
        if not wanted or not self._db_available:
            return []
        statuses = "('open','applying')" if include_applying else "('open')"
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT c.* FROM conflicts AS c "
                "JOIN json_each(c.member_versions) AS member "
                "JOIN json_each(?) AS wanted "
                "ON CAST(json_extract(member.value,'$.memory_id') AS INTEGER)=CAST(wanted.value AS INTEGER) "
                f"WHERE c.status IN {statuses} ORDER BY c.created_at DESC,c.id DESC",
                (_canonical_json(wanted),),
            ).fetchall()
        return [_decode_row(row) for row in rows]

    def resolve_conflicts_for_on_conn(self, conn: sqlite3.Connection, memory_id: int) -> int:
        # Generic memory mutation cannot complete a revisioned application plan.
        return 0

    def void_conflicts_on_conn(
        self, conn: sqlite3.Connection, memory_ids: list[int], *, reason: str,
    ) -> int:
        """Void every non-terminal conflict row involving any of ``memory_ids``.

        0.16.0 §6⑯ (作废重立) — the move/auto-move companion: a moved memory's
        old-bucket tickets must die, not linger as unfreshable rows (judge →
        stale_member, dismiss → workspace_mismatch, resolve → not_applying).
        Implementation constraints, each owner-verified:
        - terminal status = ``resolved``: NOT in the suppression loader's
          status set (open/applying/not_a_conflict), so nothing is suppressed
          and the pair can re-establish in the new bucket (§6⑯②);
        - ``candidate_key_hash`` is rewritten (UNIQUE index spans ALL
          statuses) so the candidate identity is released for re-recording
          (§6⑯③);
        - ``member_fingerprint`` is rewritten so the event-snapshot index
          (workspace, slot, fingerprint — all statuses) releases too;
        - the active-slot index releases itself: its partial predicate only
          covers open/applying (§6⑯④).
        The caller owns the write transaction (move + void must be atomic).
        """
        if not memory_ids:
            return 0
        from .additive import voided_identity_hash

        placeholders = ",".join("?" for _ in memory_ids)
        rows = conn.execute(
            f"""SELECT c.id, c.candidate_key_hash, c.member_fingerprint
                FROM conflicts AS c
                JOIN json_each(c.member_versions) AS member
                WHERE CAST(json_extract(member.value,'$.memory_id') AS INTEGER)
                      IN ({placeholders})
                  AND c.status IN ('open','applying','candidate')""",
            tuple(int(value) for value in memory_ids),
        ).fetchall()
        now = utc_now_iso()
        voided = 0
        for row in rows:
            row_id = int(row["id"])
            base_hash = str(row["candidate_key_hash"] or "")
            base_fp = str(row["member_fingerprint"] or "")
            conn.execute(
                """UPDATE conflicts SET status='resolved',
                     candidate_key_hash=?, member_fingerprint=?,
                     decided_by='agent', decision_reason=?, decided_at=?, resolved_at=?,
                     notice_delivery_status=CASE
                       WHEN notice_delivery_status IN ('pending','delivered') THEN 'stale'
                       ELSE notice_delivery_status END,
                     revision=revision+1, refreshed_at=?
                   WHERE id=? AND status IN ('open','applying','candidate')""",
                (
                    voided_identity_hash(base_hash, row_id),
                    voided_identity_hash(base_fp or f"fp-missing:{row_id}", row_id),
                    f"voided: {reason}", now, now, now, row_id,
                ),
            )
            voided += 1
        return voided

    def void_conflicts(self, memory_ids: list[int], *, reason: str) -> int:
        """Standalone (own-transaction) variant of ``void_conflicts_on_conn``."""
        if not memory_ids or not self._db_available or not self.state.sqlite_writable:
            return 0
        with self.write_transaction() as conn:
            return self.void_conflicts_on_conn(conn, memory_ids, reason=reason)

    def resolve_conflicts_for(self, memory_id: int, *, conn: sqlite3.Connection | None = None) -> int:
        return 0

    def get_memory_version(self, memory_id: int) -> int | None:
        memory = self._db.get_memory(int(memory_id))
        return int(memory["version"]) if memory else None

    def is_pair_dismissed(self, left_id: int, right_id: int) -> bool:
        pair = sorted({int(left_id), int(right_id)})
        if len(pair) != 2 or not self._db_available:
            return False
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM conflicts AS c WHERE c.status='not_a_conflict' "
                "AND json_array_length(c.member_versions)=2 "
                "AND EXISTS (SELECT 1 FROM json_each(c.member_versions) WHERE json_extract(value,'$.memory_id')=?) "
                "AND EXISTS (SELECT 1 FROM json_each(c.member_versions) WHERE json_extract(value,'$.memory_id')=?) LIMIT 1",
                (pair[0], pair[1]),
            ).fetchone()
        return row is not None

    def dismissed_pairs_for(self, memory_ids: list[int]) -> set[tuple[int, int]]:
        wanted = sorted({int(value) for value in memory_ids})
        if not wanted or not self._db_available:
            return set()
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT c.id,CAST(json_extract(member.value,'$.memory_id') AS INTEGER) AS memory_id "
                "FROM conflicts AS c JOIN json_each(c.member_versions) AS member "
                "WHERE c.status='not_a_conflict' AND json_array_length(c.member_versions)=2 "
                "AND EXISTS (SELECT 1 FROM json_each(c.member_versions) AS linked "
                "JOIN json_each(?) AS wanted ON json_extract(linked.value,'$.memory_id')=wanted.value) "
                "ORDER BY c.id,memory_id",
                (_canonical_json(wanted),),
            ).fetchall()
        by_conflict: dict[int, set[int]] = {}
        for row in rows:
            by_conflict.setdefault(int(row["id"]), set()).add(int(row["memory_id"]))
        return {
            tuple(sorted(ids))  # type: ignore[misc]
            for ids in by_conflict.values() if len(ids) == 2
        }
