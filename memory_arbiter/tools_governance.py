"""治理 mixin：rollback_auto_move/expire_relocated + scan/queue 面板（从 tools.py 搬出，拆分批 ⑥ 纯移动）。"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
from .config import Settings
from .db import MemoryDB
from .scan_pipeline import ScanPipeline
from .constants import is_default_workspace_term
from .models import utc_now_iso

if TYPE_CHECKING:
    from .queue_protocol import QueueProtocol
    from .config import Settings
    from .db import MemoryDB
    from .scan_pipeline import ScanPipeline

class _ToolsGovern:
    if TYPE_CHECKING:
        db: "MemoryDB"
        settings: "Settings"
        _scan_pipeline: "ScanPipeline"
        _queue_protocol: "QueueProtocol"
        def _is_truthy(self, *args: Any, **kwargs: Any) -> bool: ...

    def scan_pipeline_kick(self, **payload: Any) -> dict[str, Any]:
        return self._scan_pipeline.kick(
            max_memories=payload.get("max_memories") or 400,
            time_budget_s=payload.get("time_budget_s") or 45.0,
            neighbor_k=payload.get("neighbor_k") or 10,
            # P2 #2: loosely-typed JSON flag — bool("false") is True in Python
            # and would silently re-enable the slow lane; use the same
            # allow-list truthiness as every other surface flag.
            slow_lane=self._is_truthy(payload.get("slow_lane", True)),
        )

    def scan_pipeline_status(self) -> dict[str, Any]:
        return self._scan_pipeline.status()

    def scan_queue_page(self, caller: Any = None, **payload: Any) -> dict[str, Any]:
        return self._queue_protocol.page(
            page_size=payload.get("page_size") or 10,
            page_token=payload.get("page_token") or 0,
            caller=caller,
        )

    def scan_queue_submit(self, caller: Any = None, **payload: Any) -> dict[str, Any]:
        return self._queue_protocol.submit(payload.get("decisions") or [], caller=caller)

    def memory_rollback_auto_move(self, audit_id: int = 0, reason: str = "", **_: Any) -> dict[str, Any]:
        """0.16.0 §6⑫: reverse ONE autonomous normalization move by audit id.

        Restores both workspace columns (default pool allowed as the restore
        target — unlike a manual move), voids the new-bucket conflict tickets
        (§6⑯), invalidates the scan watermark, and marks the audit row
        rolled_back (审计反写). Manual moves are out of scope by design.
        """
        if not self.db.db_available or not self.db.state.sqlite_writable:
            return self.db.state.response({"moved": False, "error": "database_not_writable"}, ok=False)
        audit_id = int(audit_id or 0)
        if audit_id <= 0:
            return self.db.state.response(
                {"moved": False, "error": "rollback_auto_move requires audit_id"}, ok=False)
        try:
            with self.db.write_transaction() as conn:
                row = conn.execute(
                    "SELECT * FROM normalize_audit WHERE id=? AND status='applied'",
                    (audit_id,),
                ).fetchone()
                if row is None:
                    return self.db.state.response(
                        {"moved": False, "error": "audit entry not found or not applied"},
                        ok=False,
                    )
                memory_id = int(row["memory_id"])
                from_ws = str(row["from_workspace"])
                to_ws = str(row["to_workspace"])
                current = conn.execute(
                    "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS bucket "
                    "FROM memories WHERE id=? AND status='active'", (memory_id,),
                ).fetchone()
                if current is None or str(current["bucket"] or "") != to_ws:
                    return self.db.state.response(
                        {"moved": False,
                         "error": f"memory no longer sits in {to_ws!r}; rollback refused",
                         "current_bucket": str(current["bucket"]) if current else None},
                        ok=False,
                    )
                # 0.16.6 dedup gate (before any ticket voiding, same ordering
                # as move_memory_workspace_on_conn): restoring the memory into
                # a bucket that now holds its byte-identical ACTIVE twin
                # refuses with the colliding id instead of a bare constraint.
                sha_collision = self.db.workspaces._content_sha_collision_warning_on_conn(
                    conn, from_ws, only_id=memory_id,
                )
                if sha_collision is not None:
                    return self.db.state.response({"moved": False, "error": sha_collision}, ok=False)
                # Same §6⑯ discipline as any move: old tickets die, watermark
                # invalidates, the pipeline re-pairs in the restored bucket.
                self.db.conflicts.void_conflicts_on_conn(
                    conn, [memory_id], reason=f"rollback_auto_move #{audit_id}",
                )
                conn.execute(
                    "UPDATE memories SET workspace=?, workspace_canonical=?, scan_watermark=NULL "
                    "WHERE id=?",
                    (from_ws, from_ws, memory_id),
                )
                # Rollback moves a row across scopes without COUNT/version
                # movement — the linked-df fingerprint cannot see it.
                self.db.invalidate_linked_df_cache()
                if from_ws and not is_default_workspace_term(from_ws):
                    # A9（0.17.1 修复批）：回滚同样不得注册保护桶变体。
                    from .twin_redirect import protected_bucket_variant

                    if protected_bucket_variant(from_ws):
                        raise ValueError(
                            f"workspace {from_ws!r} is a protected-bucket spelling "
                            "variant and cannot be registered"
                        )
                    conn.execute(
                        "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
                        (from_ws, utc_now_iso()),
                    )
                conn.execute(
                    "UPDATE normalize_audit SET status='rolled_back', rolled_back_at=? WHERE id=?",
                    (utc_now_iso(), audit_id),
                )
            return self.db.state.response({
                "moved": True, "audit_id": audit_id, "memory_id": memory_id,
                "restored_to": from_ws, "reason": reason or None,
            })
        except Exception as exc:
            return self.db.state.response({"moved": False, "error": str(exc)}, ok=False)

    def _expire_relocated_workspace_rows(self) -> None:
        """Retire every pending kind='workspace' row whose subject memory no
        longer sits in the row's pinned current bucket — resolved by a move,
        exactly the notice channel's lazy staleness, carried over for the
        queue (0.16.2 §1.3)."""
        import json as _json
        from .models import utc_now_iso

        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, member_versions, detail FROM scan_queue "
                    "WHERE kind='workspace' AND status='pending'"
                ).fetchall()
                stale_ids: list[int] = []
                for row in rows:
                    try:
                        memory_id = int(
                            _json.loads(str(row["member_versions"] or "[]"))[0]["memory_id"]
                        )
                        pinned = str(
                            _json.loads(str(row["detail"] or "{}")).get("current_workspace")
                            or ""
                        ).strip()
                    except (IndexError, KeyError, TypeError, ValueError):
                        continue
                    if not pinned:
                        continue
                    current = conn.execute(
                        "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS ws "
                        "FROM memories WHERE id=? AND status='active'",
                        (memory_id,),
                    ).fetchone()
                    if current is None or str(current["ws"] or "").strip() != pinned:
                        stale_ids.append(int(row["id"]))
            if not stale_ids:
                return
            now = utc_now_iso()
            with self.db.write_transaction() as conn:
                # P2 #16: CAS on status='pending' — a row the agent already
                # decided between the SELECT above and this write (submit
                # lands confirmed/dismissed) must keep that decision; the
                # executemany form gets no rowcount per row, so skips are
                # silent by design.
                conn.executemany(
                    "UPDATE scan_queue SET status='expired', "
                    "decided_reason='resolved by move (subject left the pinned workspace)', "
                    "decided_at=?, updated_at=? WHERE id=? AND status='pending'",
                    [(now, now, row_id) for row_id in stale_ids],
                )
        except Exception:
            pass
