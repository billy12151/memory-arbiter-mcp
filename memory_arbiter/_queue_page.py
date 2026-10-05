"""queue 分页 mixin：page/组件 token/backlog/组装配/detector 版本（从 queue_protocol.py 搬出，拆分批 ⑦ 纯移动）。_fetch_*_rows 两窗口读取方法留守（ASSEMBLY_WINDOW 模块 patch 缝）。"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ._queue_consts import DEFAULT_PAGE_SIZE, GROUP_MEMBER_CAP, INTERNAL_PAIRS_CAP, MAX_PAGE_SIZE

if TYPE_CHECKING:
    from .config import Settings
    from .db import MemoryDB
    from .tools import MemoryTools


class _QueuePage:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"
        settings: "Settings"
        def _fetch_conflict_rows(self, *args: Any, **kwargs: Any) -> "list[dict[str, Any]]": ...
        def _fetch_workspace_rows(self, *args: Any, **kwargs: Any) -> "list[dict[str, Any]]": ...
        def _fetch_all_conflict_rows(self, *args: Any, **kwargs: Any) -> "list[dict[str, Any]]": ...
        def _scope_sql(self, *args: Any, **kwargs: Any) -> Any: ...
        def _pair_item(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...
        def _group_item(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...
        def _internal_memory_item(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...
        def _workspace_item(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...
        def _member_visible(self, *args: Any, **kwargs: Any) -> bool: ...
        def _row_visible(self, *args: Any, **kwargs: Any) -> bool: ...
        def _member_meta(self, *args: Any, **kwargs: Any) -> Any: ...

    def page(self, *, page_size: int = DEFAULT_PAGE_SIZE, page_token: int = 0, caller: Any = None) -> dict[str, Any]:
        self.db.internal_conflicts.expire_stale()
        scope = caller.scope_canonicals() if caller is not None and caller.isolation == "strict" else None
        page_size = max(1, min(int(page_size or DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE))
        page_token = max(0, int(page_token or 0))
        items: list[dict[str, Any]] = []
        meta_cache: dict[int, dict[str, Any]] = {}
        last_id = page_token
        # Priority: workspace suspects first (rare, cheap to judge, gate
        # autonomous moves), then internal contradictions, then conflict
        # groups — a big conflict backlog must not starve the other kinds.
        for row in self._fetch_workspace_rows(page_token, scope):
            if len(items) >= page_size:
                break
            if not self._row_visible(row, caller):
                continue
            items.append(self._workspace_item(row, meta_cache))
            # The cursor advances through workspace rows too: a page whose
            # items are all workspace/internal used to echo the caller's
            # token back unchanged (0 on the first page), which a
            # defensive agent reads as a pagination loop and stops early
            # (live repro 2026-09-13: the weekly task judged 67 of 125
            # suspects, never reached a single conflict pair, and exited
            # believing it was done). Judgement-free rows no longer block
            # the page either — the next page moves past them instead of
            # re-serving the same head forever.
            last_id = max(last_id, int(row["id"]))
        if len(items) < page_size:
            # Internal items are capped per page (0.16.4 §3: one aggregated
            # row per MEMORY — information-dense items, and the cap keeps a
            # fragment-noise flood from starving conflict judgment).
            internal_cap = min(3, page_size)
            for group in self.db.internal_conflicts.list_pending_grouped(
                limit_memories=internal_cap, pairs_cap=INTERNAL_PAIRS_CAP,
            ):
                if len(items) >= page_size:
                    break
                if not self._member_visible(int(group["memory_id"]), caller):
                    continue
                items.append(self._internal_memory_item(group, meta_cache))
        if len(items) < page_size:
            rows = self._fetch_conflict_rows(page_token, scope)
            for group in self._assemble_groups(rows):
                if len(items) >= page_size:
                    # Page full: the cursor must stay on the last DISPLAYED
                    # group — advancing past the boundary here would strand
                    # the undisplayed group between pages forever (adversarial
                    # review repro #3).
                    break
                if not all(self._row_visible(edge, caller) for edge in group["pairs"]):
                    # Fail closed: hide the whole component from a caller who
                    # cannot read every member (matches conflict_detail).
                    continue
                if len(group["member_ids"]) > GROUP_MEMBER_CAP:
                    # §6⑬: oversize component → split back into edges.
                    for edge in group["pairs"]:
                        if len(items) >= page_size:
                            break
                        items.append(self._pair_item(edge, meta_cache))
                        last_id = max(last_id, int(edge["id"]))
                else:
                    items.append(self._group_item(group, meta_cache))
                    last_id = max(last_id, group["last_queue_id"])
        # P2 #17: remaining is the CALLER-SCOPED backlog. A strict caller must
        # neither learn other buckets' pending counts (queue_backlog leaks
        # existence) nor wrap/continue paging on rows it can never see — the
        # scan_queue segment reuses the same workspace_canonical scope SQL as
        # _fetch_conflict_rows; the internal segment applies the same scope by
        # JOINing memories (one SQL, same pending predicate as list_pending).
        remaining = self._scoped_backlog(scope)
        if not items and page_token > 0 and remaining > 0:
            # Cursor wrap-around: the pass advanced past every row but some
            # backlog remains (rows the caller skipped or failed to judge).
            # Returning an empty page with the echoed token would signal a
            # pagination loop; restart from the head instead so the agent
            # gets another pass over the survivors. Terminates: the wrapped
            # call runs with page_token=0 and cannot re-enter this branch.
            return self.page(page_size=page_size, page_token=0, caller=caller)
        response: dict[str, Any] = {
            "ok": True,
            "items": items[:page_size],
            "count": len(items[:page_size]),
            "queue_backlog": remaining,
            "detector_version": self._detector_version(),
            "instruction": (
                "Judge each item from the evidence quotes; upgrade an uncertain item with "
                "memory(action='batch_read', content_mode='hits', spans={...evidence_span...}) "
                "before deciding, and read full texts before any confirm-driven edit. Submit "
                "dispositions with memory_repair(task='scan_queue', action='submit'): per-pair "
                "{candidate_key_hash, status: confirmed|dismissed, reason} + (confirms) "
                "slot_key + value_groups with display_value per member group. A whole group "
                "that is noise can be dismissed in one entry with just the group's group_token "
                "(pair_hashes are a preview; the server re-assembles the whole group). "
                "internal_memory items (aggregated per memory): one "
                "{kind:'internal_memory', memory_id, status: dismissed|resolved, reason} "
                "clears the whole memory — pair_count beyond the pairs preview requires "
                "reading every pair first and resubmitting with expanded=true. Workspace "
                "suspects: confirmed with target_workspace + "
                "conf (server re-runs the vote gate); if NO suitable bucket exists, "
                "confirmed with target_workspace='default' AND fallback=true + reason "
                "(vote gate waived, audited, reported to the user — use sparingly)."
            ),
        }
        # has_more: any SCOPED backlog beyond what this page displayed — the
        # scoped backlog is authoritative (workspace/internal rows live in
        # separate tables and may not all fit this page).
        has_more = remaining > len(items[:page_size])
        if has_more:
            response["next_page_token"] = last_id
        response["has_more"] = bool(has_more)
        if self.db.settings.include_size and items:
            from .tokens import meter_payloads

            response["size"] = meter_payloads(items[:page_size])
        return response

    @staticmethod

    def _component_token(pairs: list[dict[str, Any]]) -> str:
        import hashlib

        joined = ":".join(sorted(str(pair["candidate_key_hash"]) for pair in pairs))
        return hashlib.sha256(joined.encode()).hexdigest()[:16]

    def _scoped_backlog(self, scope: Any = None) -> int:
        """P2 #17: pending backlog in the caller's scope.

        scope=None (none/weak callers) keeps the global caliber. Strict scope:
        the scan_queue segment counts pending rows through the same
        workspace_canonical scope SQL as _fetch_conflict_rows; the internal
        segment JOINs memories and applies the identical pending predicate as
        list_pending (version-current + active) plus the same scope — one SQL
        each.
        """
        if scope is None:
            return self.db.scan_queue_backlog() + len(
                self.db.internal_conflicts.list_pending(limit=10**6)
            )
        try:
            with self.db.connection() as conn:
                queue_scope_sql, queue_params = self._scope_sql("workspace_canonical", scope)
                queue_count = int(conn.execute(
                    "SELECT COUNT(*) FROM scan_queue WHERE status='pending'"
                    + (f" AND {queue_scope_sql}" if queue_scope_sql else ""),
                    tuple(queue_params),
                ).fetchone()[0])
                mem_scope_sql, mem_params = self._scope_sql("m.workspace_canonical", scope)
                internal_count = int(conn.execute(
                    "SELECT COUNT(*) FROM internal_conflicts i "
                    "JOIN memories m ON m.id=i.memory_id "
                    "WHERE i.status='pending' AND m.version=i.memory_version "
                    "AND m.status='active'"
                    + (f" AND {mem_scope_sql}" if mem_scope_sql else ""),
                    tuple(mem_params),
                ).fetchone()[0])
            return queue_count + internal_count
        except Exception:
            return 0

    def _assemble_groups(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Union-find pair rows into closure groups by shared member refs."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        row_refs: dict[int, list[str]] = {}
        for row in rows:
            refs = [
                f"{int(member['memory_id'])}@{int(member.get('version') or 1)}"
                for member in (row.get("member_versions") or [])
                if member.get("memory_id") is not None
            ]
            row_refs[int(row["id"])] = refs
            for ref in refs:
                union(ref, refs[0])

        components: dict[str, dict[str, Any]] = {}
        for row in rows:
            refs = row_refs[int(row["id"])]
            root = find(refs[0]) if refs else f"row:{row['id']}"
            component = components.setdefault(root, {"pairs": [], "member_ids": set(), "last_queue_id": 0})
            component["pairs"].append(row)
            component["last_queue_id"] = max(component["last_queue_id"], int(row["id"]))
            for member in row.get("member_versions") or []:
                if member.get("memory_id") is not None:
                    component["member_ids"].add(int(member["memory_id"]))
        # 0.17.0 review R2：判定页窗口（ASSEMBLY_WINDOW）内按组最高 pair
        # 优先级降序展示（tie-break 保 id 游标稳定）——数值对立/低 cos 高分
        # 对先到 agent 手里，不再被入队顺序埋没。
        return sorted(
            components.values(),
            key=lambda c: (
                -max(
                    (float(p.get("priority") or 0.0) for p in c["pairs"]),
                    default=0.0,
                ),
                c["last_queue_id"],
            ),
        )

    def _detector_version(self) -> str:
        from .db_generation import CONFLICT_DETECTOR_VERSION

        return CONFLICT_DETECTOR_VERSION

    # ── submission (server-side land-from-reference) ───────────────────────
