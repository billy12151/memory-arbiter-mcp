"""queue 渲染 item 层 mixin：成员可见性/pair/group/internal/workspace item 构造（从 queue_protocol.py 搬出，拆分批 ⑦ 纯移动）。"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ._queue_consts import GROUP_HASHES_CAP, INTERNAL_PAIRS_CAP

if TYPE_CHECKING:
    from .config import Settings
    from .db import MemoryDB
    from .tools import MemoryTools


class _QueueItems:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"
        settings: "Settings"
        def _caller_workspace(self, *args: Any, **kwargs: Any) -> Any: ...
        def _get_memory_visible(self, *args: Any, **kwargs: Any) -> Any: ...
        def _detector_version(self) -> str: ...
        def _component_token(self, *args: Any, **kwargs: Any) -> str: ...

    def _member_visible(self, memory_id: int, caller: Any = None) -> bool:
        """Strict-isolation fail-closed check (adversarial review #2): a
        member the caller cannot read takes its whole item off the page and
        blocks its dispositions. P2 #15: the caller is threaded through as a
        parameter (no shared self._caller field left between page/submit)."""
        if caller is None or caller.isolation != "strict":
            return True
        return self._tools._get_memory_visible(int(memory_id), caller) is not None

    def _row_visible(self, row: dict[str, Any], caller: Any = None) -> bool:
        return all(
            self._member_visible(int(member["memory_id"]), caller)
            for member in (row.get("member_versions") or [])
            if member.get("memory_id") is not None
        )

    def _member_meta(self, memory_id: int, cache: dict[int, Any]) -> dict[str, Any] | None:
        if memory_id not in cache:
            record = self.db.get_memory(memory_id)
            if not record:
                cache[memory_id] = None
            else:
                tags = record.get("tags")
                cache[memory_id] = {
                    "subject": str(record.get("subject") or "")[:200],
                    "tags": (tags if isinstance(tags, list) else [])[:20],
                    "event_time": record.get("event_time"),
                    "workspace": record.get("workspace_canonical") or record.get("workspace"),
                    "version": int(record.get("version") or 1),
                }
        cached: dict[str, Any] | None = cache[memory_id]
        return cached

    def _pair_item(self, row: dict[str, Any], cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        members = row.get("member_versions") or []
        raw_detail = row.get("detail")
        detail: dict[str, Any] = raw_detail if isinstance(raw_detail, dict) else {}
        return {
            "kind": "conflict",
            "group_token": f"pair:{row['candidate_key_hash'][:16]}",
            "pair_count": 1,
            "member_ids": sorted({int(m["memory_id"]) for m in members if m.get("memory_id") is not None}),
            "member_meta": {
                str(m["memory_id"]): self._member_meta(int(m["memory_id"]), cache)
                for m in members if m.get("memory_id") is not None
            },
            "pairs": [{
                "queue_id": int(row["id"]),
                "candidate_key_hash": row["candidate_key_hash"],
                "severity": row.get("severity"),
                "reason": row.get("reason"),
                "workspace": row.get("workspace_canonical"),
                "evidence": row.get("evidence") or [],
                "candidate_key": detail.get("candidate_key"),
            }],
        }

    def _group_item(self, group: dict[str, Any], cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        members = group["pairs"][0].get("member_versions") or []
        return {
            "kind": "conflict",
            "group_token": f"group:{self._component_token(group['pairs'])}",
            "pair_count": len(group["pairs"]),
            "member_ids": sorted(group["member_ids"]),
            "member_meta": {
                str(mid): self._member_meta(mid, cache) for mid in sorted(group["member_ids"])
            },
            "pair_hashes": [str(row["candidate_key_hash"]) for row in group["pairs"][:GROUP_HASHES_CAP]],
            # 0.16.4 live-judgment review: the hash list is a PREVIEW (byte
            # budget — full lists dominated real pages); pair_count stays the
            # full total and a group_token disposition re-assembles the whole
            # group server-side, so the cap cannot strand pairs.
            "pairs": [{
                "queue_id": int(row["id"]),
                "candidate_key_hash": row["candidate_key_hash"],
                "severity": row.get("severity"),
                "reason": row.get("reason"),
                "workspace": row.get("workspace_canonical"),
                "evidence": row.get("evidence") or [],
                "candidate_key": (row.get("detail") or {}).get("candidate_key")
                if isinstance(row.get("detail"), dict) else None,
            } for row in group["pairs"]],
        }

    def _internal_memory_item(self, group: dict[str, Any], cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        """0.16.4 §3: one judgment item per MEMORY — pairs cap 8 preview,
        full pair_count, reason distribution. The judgment unit is the
        memory, so no dedicated cursor: a disposition rolls the page."""
        memory_id = int(group["memory_id"])
        pairs = [
            {
                "internal_id": int(row["id"]),
                "quote_a": row.get("quote_a"),
                "quote_b": row.get("quote_b"),
                "span_a": row.get("span_a"),
                "span_b": row.get("span_b"),
                "reason": row.get("reason"),
            }
            for row in group["pairs"]
        ]
        return {
            "kind": "internal_memory",
            "memory_id": memory_id,
            "pair_count": int(group["pair_count"]),
            "pairs": pairs,
            "reasons_summary": group.get("reasons_summary") or {},
            "member_meta": {str(memory_id): self._member_meta(memory_id, cache)},
            "instruction": (
                "One memory contradicting itself (aggregated: the pairs list is a "
                f"preview of up to {INTERNAL_PAIRS_CAP}, pair_count is the full total). "
                "Dismiss the whole memory with {kind:'internal_memory', memory_id, "
                "status:'dismissed', reason} — if pair_count exceeds the preview, first "
                "read EVERY pair (memory action='batch_read', content_mode='hits', spans "
                "from the pairs' span_a/span_b) and resubmit with expanded=true. Resolve "
                "only for confirmed real contradictions. Fixing the memory "
                "(memory action='update') also works — rows auto-stale on version lift."
            ),
        }

    def _workspace_item(self, row: dict[str, Any], cache: dict[int, dict[str, Any]]) -> dict[str, Any]:
        member = (row.get("member_versions") or [{}])[0]
        memory_id = int(member.get("memory_id") or 0)
        raw_detail = row.get("detail")
        detail: dict[str, Any] = raw_detail if isinstance(raw_detail, dict) else {}
        outline: list[dict[str, Any]] = []
        content_chars = 0
        record = self.db.get_memory(memory_id)
        if record:
            from .pipeline.read import _content_outline

            content = str(record.get("content") or "")
            content_chars = len(content)
            outline = _content_outline(str(record.get("subject") or ""), content)
        return {
            "kind": "workspace",
            "queue_id": int(row["id"]),
            "candidate_key_hash": row["candidate_key_hash"],
            "memory_id": memory_id,
            "memory_version": member.get("version"),
            "current_workspace": detail.get("current_workspace") or row.get("workspace_canonical"),
            "suspected_workspace": detail.get("suspected_workspace"),
            "votes": detail.get("votes") or {},
            "protected_involved": bool(detail.get("protected_involved")),
            "member_meta": {str(memory_id): self._member_meta(memory_id, cache)},
            "content_chars": content_chars,
            "outline": outline,
            "reason": row.get("reason"),
            "instruction": (
                "E5 summary input: judge placement from subject/tags/outline. Confirm with "
                "{kind:'workspace', memory_id, status:'confirmed', target_workspace, conf} — "
                "the server re-runs the vector vote and only moves when the two signals agree "
                "(protected buckets are never moved autonomously)."
            ),
        }
