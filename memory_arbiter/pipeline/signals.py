"""Attach member-linked open/applying conflict-group state to search results."""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from ..acl import raw_workspace

if TYPE_CHECKING:
    from ..tools import MemoryTools


class ConflictSignalPipeline:
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    @staticmethod
    def _confidence_rank(hint: Optional[str]) -> int:
        # Retained for callers of the old private helper; group signals do not
        # rank by legacy confidence judgments.
        return {"high": 3, "medium": 2, "low": 1}.get(hint or "", 0)

    def _attach_conflict_signals(
        self, results: list[dict[str, Any]], warnings: list[str],
    ) -> list[dict[str, Any]]:
        if not results:
            return results
        try:
            ids = [int(row["id"]) for row in results if row.get("id") is not None]
            wanted = set(ids)
            conflicts = self.db.conflicts.list_open_conflicts_for_memory_ids(
                ids, include_applying=True,
            )
            by_memory: dict[int, list[dict[str, Any]]] = {}
            all_ids: set[int] = set(ids)
            for conflict in conflicts:
                member_ids = {int(member["memory_id"]) for member in conflict.get("member_versions") or []}
                all_ids.update(member_ids)
                for memory_id in member_ids & wanted:
                    by_memory.setdefault(memory_id, []).append(conflict)
            summaries = self.db.get_memory_summaries(list(all_ids))
            for row in results:
                memory_id = int(row["id"])
                linked = by_memory.get(memory_id)
                if linked:
                    signal = self._build_open_table_signal(memory_id, linked, summaries, wanted)
                    if signal:
                        row["conflict_signal"] = signal
        except Exception as exc:
            warnings.append(f"conflict_signal attachment failed: {exc}")
        return results

    def _build_open_table_signal(
        self,
        memory_id: int,
        conflicts: list[dict[str, Any]],
        summaries: dict[int, dict[str, Any]],
        result_id_set: set[int],
    ) -> Optional[dict[str, Any]]:
        del result_id_set  # Membership, not pair orientation, determines linkage.
        primary = max(
            conflicts,
            key=lambda row: (
                1 if row.get("status") == "applying" else 0,
                str(row.get("refreshed_at") or row.get("created_at") or ""),
                int(row.get("id") or 0),
            ),
        )
        member_ids = sorted({
            int(member["memory_id"])
            for member in primary.get("member_versions") or []
        })
        caller_workspace = raw_workspace(summaries.get(memory_id, {}))
        all_visible = True
        if self.settings.isolation == "strict":
            all_visible = bool(member_ids) and all(
                bool(summaries.get(member_id))
                and raw_workspace(summaries[member_id]) == caller_workspace
                for member_id in member_ids
            )
        signal: dict[str, Any] = {
            "conflict_source": "conflict_group",
            "conflict_id": int(primary["id"]),
            "conflict_revision": int(primary.get("revision") or 0),
            "conflict_status": primary.get("status"),
            "related_conflict_count": len(conflicts),
            "member_count": len(member_ids),
            "action_required": "apply_conflict_action" if primary.get("status") == "applying" else "judge_conflict",
            "next_executable_call": self._conflict_next_call(primary),
        }
        if not all_visible:
            # Do not reveal that a hidden cross-workspace snapshot exists.
            return None
        signal.update({
            "slot": primary.get("slot_key"),
            "conflict_point": primary.get("conflict_point"),
            "value_groups": primary.get("value_groups") or [],
            "members": [
                {
                    "id": member_id,
                    "subject": summaries.get(member_id, {}).get("subject"),
                    "status": summaries.get(member_id, {}).get("status"),
                    "snippet": summaries.get(member_id, {}).get("snippet"),
                }
                for member_id in member_ids
            ],
            "apply_summary": primary.get("apply_summary") if primary.get("status") == "applying" else None,
        })
        return signal
