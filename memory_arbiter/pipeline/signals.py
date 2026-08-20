"""Attach formally recorded conflict state to search results."""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from ..acl import raw_workspace
from ..conflict_judgments import ConflictJudgmentStore

if TYPE_CHECKING:
    from ..tools import MemoryTools


class ConflictSignalPipeline:
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    @staticmethod
    def _confidence_rank(hint: Optional[str]) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(hint or "", 0)

    def _attach_conflict_signals(
        self, results: list[dict[str, Any]], warnings: list[str],
    ) -> list[dict[str, Any]]:
        if not results:
            return results
        try:
            ids = [int(row["id"]) for row in results if row.get("id") is not None]
            conflicts = self.db.list_open_conflicts_for_memory_ids(ids)
            by_memory: dict[int, list[dict[str, Any]]] = {}
            all_ids: set[int] = set(ids)
            for conflict in conflicts:
                left, right = int(conflict["left_id"]), int(conflict["right_id"])
                by_memory.setdefault(left, []).append(conflict)
                by_memory.setdefault(right, []).append(conflict)
                all_ids.update((left, right))
            summaries = self.db.get_memory_summaries(list(all_ids))
            result_ids = set(ids)
            for row in results:
                memory_id = int(row["id"])
                if memory_id in by_memory:
                    signal = self._build_open_table_signal(
                        memory_id, by_memory[memory_id], summaries, result_ids,
                    )
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
        primary = max(
            conflicts,
            key=lambda row: (
                self._confidence_rank(row.get("confidence_hint")),
                str(row.get("created_at") or ""), int(row.get("id") or 0),
            ),
        )
        peer_id = int(primary["right_id"]) if int(primary["left_id"]) == memory_id else int(primary["left_id"])
        peer = summaries.get(peer_id, {})
        peer_visible = True
        if self.settings.isolation == "strict":
            peer_visible = bool(peer) and raw_workspace(peer) == raw_workspace(summaries.get(memory_id, {}))
        resolution_kind = primary.get("resolution_kind") or primary.get("judgment_resolution_kind")
        conflict_scope = primary.get("conflict_scope") or primary.get("judgment_conflict_scope")
        guidance = primary.get("active_judgment_id") is not None
        source = "conflict_guidance" if guidance else "open_table"
        signal: dict[str, Any] = {
            "conflict_source": source,
            "conflict_id": int(primary["id"]),
            "conflict_status": primary.get("status"),
            "conflict_type": primary.get("conflict_type"),
            "conflict_point": primary.get("conflict_point") if peer_visible else None,
            "resolution_kind": resolution_kind,
            "conflict_scope": conflict_scope,
            "recommended_resolution_action": ConflictJudgmentStore.resolution_action(resolution_kind),
            "supersede_candidate": ConflictJudgmentStore.is_supersede_candidate(resolution_kind),
            "confidence_hint": primary.get("confidence_hint"),
            "verification_status": primary.get("judgment_status") or "recorded",
            "action_required": "ask_user" if primary.get("judgment_status") == "pending_user" else None,
            "related_conflict_count": len(conflicts),
            "conflict_peer": (
                {"id": peer_id, "subject": peer.get("subject"), "status": peer.get("status"), "snippet": peer.get("snippet")}
                if peer_visible else {"id": None, "redaction_reason": "workspace_acl"}
            ),
        }
        if not peer_visible:
            signal["redacted_fields"] = [
                "conflict_point", "suggested_winner", "judgment", "conflict_peer",
            ]
        if guidance:
            signal["judgment"] = {
                "verdict": primary.get("judgment_verdict"),
                "recommended_use": primary.get("judgment_recommended_use"),
                "suggested_winner": primary.get("judgment_suggested_winner"),
                "reason": primary.get("judgment_reason"),
                "judge_type": primary.get("judgment_judge_type"),
                "judged_at": primary.get("judged_at"),
            }
        return signal
