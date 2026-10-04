"""Judgment-queue protocol (0.16.0 plan §6㉑/§2 commit 5).

The agent processes the scan_queue page by page — "handle page 1, submit its
dispositions with the next page fetch" — while the server does all the
transport work:

- Page assembly builds TRANSITIVE-CLOSURE GROUPS from pair rows (A↔B + A↔C →
  {A,B,C}) purely as judgment units (§6⑥): groups are never persisted; the
  landed state is per-pair. Oversized components (>N members) are split back
  into their constituent edges.
- Submission is server-side land-from-reference: the agent submits
  ``{candidate_key_hash, status, reason}`` plus — only for confirms — the
  slot key and per-group display values; the server resolves the frozen
  envelope from the queue row and drives the existing ``record_conflict``
  (confirm → open, dismiss → not_a_conflict suppression source). Queue rows
  NEVER enter ``conflicts`` themselves.
- Page boundaries are natural breakpoints: a decision that arrives after a
  crash resumes from wherever the queue stands; nothing is lost because
  nothing was held in memory.
"""
from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from .tools import MemoryTools

# §1.5: 10-30 groups per page (owner-pinned band; default at the band floor).
# §6⑬: a closure component with more than N members is split back into its
# edges — one judgment item per pair — instead of one mega-group.
# Fetch window for group assembly per page call (rows, not groups).
ASSEMBLY_WINDOW = 400
# 0.16.4 §3: per-memory internal aggregation — pairs preview cap per page
# item. The FULL count travels as pair_count; a memory-level dismissal of a
# pair_count beyond this cap requires expanded=true (the agent read every
# pair via batch_read hits or the per-row channel first).
# 0.16.4 live-judgment review: group-level hash preview cap — full hash
# lists dominated real page bytes; group_token dispositions re-assemble the
# complete group server-side (see _submit_group), so the cap is display-only.
from ._queue_consts import (  # noqa: F401
    DEFAULT_PAGE_SIZE as DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE as MAX_PAGE_SIZE,
    GROUP_MEMBER_CAP as GROUP_MEMBER_CAP,
    INTERNAL_PAIRS_CAP as INTERNAL_PAIRS_CAP,
    GROUP_HASHES_CAP as GROUP_HASHES_CAP,
    _decision_truthy as _decision_truthy,
)
from ._queue_page import _QueuePage
from ._queue_items import _QueueItems
from ._queue_submit import _QueueSubmit


class QueueProtocol(_QueuePage, _QueueItems, _QueueSubmit):
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools
        self.db = tools.db

    @staticmethod
    def _scope_sql(workspace_canonical_column: str, scope: Any) -> "tuple[str, list[Any]]":
        from .acl import workspace_scope_sql

        return workspace_scope_sql(workspace_canonical_column, scope)

    # ── page fetch ──────────────────────────────────────────────────────────

    def _fetch_conflict_rows(self, after_id: int, scope: Any = None) -> list[dict[str, Any]]:
        if not self.db.db_available:
            return []
        scope_sql, scope_params = self._scope_sql("workspace_canonical", scope)
        if scope_sql:
            scope_sql = " AND " + scope_sql
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """SELECT id,kind,workspace_canonical,status,candidate_key_hash,
                              member_versions,evidence,reason,severity,source,priority,detail
                       FROM scan_queue WHERE status='pending' AND kind='conflict' AND id>?
                       """ + scope_sql + " ORDER BY id LIMIT ?",
                    (int(after_id), *scope_params, ASSEMBLY_WINDOW),
                ).fetchall()
        except Exception:
            return []
        decoded: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("member_versions", "evidence", "detail"):
                if isinstance(item.get(key), str):
                    try:
                        item[key] = json.loads(item[key])
                    except (TypeError, json.JSONDecodeError):
                        item[key] = None
            decoded.append(item)
        return decoded

    def _fetch_all_conflict_rows(self, scope: Any = None) -> list[dict[str, Any]]:
        """Every pending conflict row, paged past the ASSEMBLY_WINDOW.

        Group-level dispositions must re-assemble the COMPLETE closure even
        on libraries deeper than one fetch window — a token-only submit on a
        63-pair group in a large library must not strand the tail. Defensive
        ceiling keeps a pathological queue bounded.
        """
        rows: list[dict[str, Any]] = []
        cursor = 0
        ceiling = 50  # 50 × ASSEMBLY_WINDOW(400) = 20k rows defensive cap
        for _ in range(ceiling):
            batch = self._fetch_conflict_rows(cursor, scope)
            rows.extend(batch)
            if len(batch) < ASSEMBLY_WINDOW:
                return rows
            cursor = int(batch[-1]["id"])
        return rows

    def _fetch_workspace_rows(self, after_id: int = 0, scope: Any = None) -> list[dict[str, Any]]:
        if not self.db.db_available:
            return []
        scope_sql, scope_params = self._scope_sql("workspace_canonical", scope)
        if scope_sql:
            scope_sql = " AND " + scope_sql
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """SELECT id,kind,workspace_canonical,status,candidate_key_hash,
                              member_versions,evidence,reason,severity,source,detail
                       FROM scan_queue WHERE status='pending' AND kind='workspace'
                           AND id>? """ + scope_sql + " ORDER BY id LIMIT ?",
                    (int(after_id), *scope_params, ASSEMBLY_WINDOW),
                ).fetchall()
        except Exception:
            return []
        decoded: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("member_versions", "evidence", "detail"):
                if isinstance(item.get(key), str):
                    try:
                        item[key] = json.loads(item[key])
                    except (TypeError, json.JSONDecodeError):
                        item[key] = None
            decoded.append(item)
        return decoded
