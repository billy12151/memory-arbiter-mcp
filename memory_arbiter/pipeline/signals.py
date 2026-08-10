"""Conflict signal attachment pipeline for MemoryTools (Phase 4 extraction)."""
# mypy: disable-error-code=type-arg
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from ..acl import raw_workspace
from ..conflict_judgments import ConflictJudgmentStore

if TYPE_CHECKING:
    from ..tools import MemoryTools


class ConflictSignalPipeline:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    def _trust_score(self, record: dict[str, Any]) -> int:
        """Composite trust rank from source_type + protection_level."""
        st = self._tools._TRUST_RANK.get(record.get("source_type", ""), 0)
        pl = self._tools._TRUST_RANK.get(record.get("protection_level", ""), 0)
        return max(st, pl)

    @staticmethod
    def _confidence_rank(hint: Optional[str]) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(hint or "", 0)

    def _attach_conflict_signals(
        self,
        results: list[dict[str, Any]],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """v0.7.6: attach conflict_signal to each direct-mode search result.

        Two sources, strongly distinguished by ``conflict_source``:
          * ``open_table``: conflict already in the conflicts table (written by
            scan + record_conflict). Carries structured fields.
          * ``runtime_metadata_hint``: computed on-the-fly from subject/tags
            overlap + trust disparity. **Not LLM-verified** — advisory only.

        open_table takes priority. Both attach a ``conflict_peer`` summary so
        the caller knows who the conflict is with, even if the peer was cut by
        ``limit``. Never raises; failures degrade to no signal.
        """
        if not results:
            return results
        try:
            result_ids = [int(r["id"]) for r in results if r.get("id") is not None]
            if not result_ids:
                return results

            # Batch-fetch open conflicts for all result IDs (one SQL, no N+1).
            conflicts = self.db.list_open_conflicts_for_memory_ids(result_ids)
            if self.settings.structured_claim_mode == "off":
                conflicts = [c for c in conflicts if c.get("left_claim_revision") is None]
            # Build memory_id → list of conflicts.
            conflicts_by_mem: dict[int, list[dict[str, Any]]] = {}
            all_peer_ids: set[int] = set()
            for c in conflicts:
                left = int(c.get("left_id"))
                right = int(c.get("right_id"))
                conflicts_by_mem.setdefault(left, []).append(c)
                conflicts_by_mem.setdefault(right, []).append(c)
                all_peer_ids.add(left)
                all_peer_ids.add(right)

            # Batch-fetch summaries for all IDs that appear in any conflict.
            summaries: dict[int, dict[str, Any]] = {}
            if all_peer_ids:
                summaries = self.db.get_memory_summaries(list(all_peer_ids))
            for rec in results:
                try:
                    summaries.setdefault(int(rec["id"]), rec)
                except (KeyError, TypeError, ValueError):
                    pass

            # Attach signals.
            result_id_set = set(result_ids)
            # v0.8.8 Layer 0: pairs already dismissed (not_a_conflict, version
            # match) — the computed-overlap advisory path must skip them.
            dismissed_pairs = self.db.dismissed_pairs_for(result_ids)
            for rec in results:
                mid = int(rec["id"])
                if mid in conflicts_by_mem:
                    signal = self._build_open_table_signal(
                        mid, conflicts_by_mem[mid], summaries, result_id_set,
                    )
                    if signal:
                        rec["conflict_signal"] = signal
                        continue
                # No open_table signal → try runtime_metadata_hint.
                hint = self._compute_runtime_hint(mid, rec, results, result_id_set, dismissed_pairs)
                if hint:
                    rec["conflict_signal"] = hint
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
        """Build an open_table conflict_signal for a memory with open conflicts.

        If a memory has multiple open conflicts, pick the primary one by
        confidence_hint > created_at > conflict_id.
        """
        def conflict_sort_key(c: dict[str, Any]) -> tuple[Any, ...]:
            if c.get("judgment_status") == "pending_user":
                state_priority = 5
            elif c.get("judgment_status") == "pending_llm":
                state_priority = 4
            elif c.get("source") == "metadata_write_hint":
                state_priority = 1
            elif c.get("judgment_status") in {"llm_assessed", "human_confirmed"}:
                state_priority = 3
            else:
                state_priority = 4  # verified scan/record conflict
            return (
                state_priority,
                self._confidence_rank(c.get("confidence_hint")),
                str(c.get("created_at", "")),
                int(c.get("id", 0)),
            )

        primary = max(conflicts, key=conflict_sort_key)
        peer_id = int(primary["right_id"]) if primary["left_id"] == memory_id else int(primary["left_id"])
        peer_summary = summaries.get(peer_id, {})
        src = primary.get("source")
        advisory = (src == "metadata_write_hint")
        judgment_status = primary.get("judgment_status")
        structured_origin = primary.get("left_claim_revision") is not None
        if advisory:
            conflict_source = "runtime_metadata_hint"
        elif structured_origin and judgment_status == "pending_llm":
            conflict_source = "structured_claim_candidate"
        elif (
            primary.get("status") == "resolved"
            and primary.get("active_judgment_id") is not None
        ) or (structured_origin and judgment_status in {"llm_assessed", "human_confirmed"}):
            conflict_source = "conflict_guidance"
        else:
            conflict_source = "open_table"
        judgment_request = None
        peer_visible = True
        if self.settings.isolation == "strict":
            result_summary = summaries.get(memory_id, {})
            caller_ws = raw_workspace(result_summary)
            # Result records themselves are already workspace-scoped under strict.
            # Summaries include the peer workspace; redact the peer if it differs.
            peer_visible = bool(peer_summary) and raw_workspace(peer_summary) == caller_ws
        if conflict_source == "structured_claim_candidate" and peer_visible:
            judgment_request = self.db.judgments.build_conflict_judgment_request(int(primary["id"]))
        resolution_kind = primary.get("resolution_kind") or primary.get("judgment_resolution_kind")
        conflict_scope = primary.get("conflict_scope") or primary.get("judgment_conflict_scope")
        recommended_resolution_action = ConflictJudgmentStore.resolution_action(resolution_kind)
        supersede_candidate = ConflictJudgmentStore.is_supersede_candidate(resolution_kind)
        if self.settings.isolation == "strict" and not peer_visible:
            return {
                "conflict_source": conflict_source,
                "conflict_id": int(primary["id"]),
                "conflict_status": primary.get("status"),
                "conflict_type": primary.get("conflict_type"),
                "resolution_kind": resolution_kind,
                "conflict_scope": conflict_scope,
                "recommended_resolution_action": recommended_resolution_action,
                "supersede_candidate": supersede_candidate,
                "verification_status": "redacted_workspace_acl",
                "action_required": None,
                "conflict_judgment_request": None,
                "judgment": None,
                "redacted_fields": [
                    "conflict_point", "suggested_winner", "confidence_hint",
                    "judgment", "judge_ref", "conflict_peer",
                ],
                "related_conflict_count": len(conflicts),
                "open_conflict_count": sum(1 for c in conflicts if c.get("status") == "open"),
                "conflict_peer": {
                    "id": None,
                    "subject": None,
                    "status": None,
                    "snippet": None,
                    "redaction_reason": "workspace_acl",
                },
            }
        return {
            # v0.8.8: route by row source — write-hint rows are advisory, NOT
            # verified; presenting them as open_table would cry-wolf. (漏洞#1)
            "conflict_source": conflict_source,
            "conflict_id": int(primary["id"]),
            "conflict_status": primary.get("status"),
            "conflict_type": primary.get("conflict_type"),
            "conflict_point": primary.get("conflict_point"),
            "suggested_winner": primary.get("suggested_winner"),
            "resolution_kind": resolution_kind,
            "conflict_scope": conflict_scope,
            "recommended_resolution_action": recommended_resolution_action,
            "supersede_candidate": supersede_candidate,
            "confidence_hint": "low" if advisory else primary.get("confidence_hint"),
            "source": src,
            "verification_status": judgment_status or ("pending_llm" if structured_origin else "verified"),
            "action_required": (
                "judge_conflict_before_use" if conflict_source == "structured_claim_candidate"
                else ("ask_user" if judgment_status == "pending_user" else None)
            ),
            "conflict_judgment_request": judgment_request,
            "judgment": ({
                "verdict": primary.get("judgment_verdict"),
                "recommended_use": primary.get("judgment_recommended_use"),
                "suggested_winner": primary.get("judgment_suggested_winner"),
                "confidence_hint": primary.get("judgment_confidence_hint"),
                "reason": primary.get("judgment_reason"),
                "resolution_kind": resolution_kind,
                "conflict_scope": conflict_scope,
                "recommended_resolution_action": recommended_resolution_action,
                "supersede_candidate": supersede_candidate,
                "judge_type": primary.get("judgment_judge_type"),
                "judge_ref": primary.get("judgment_judge_ref"),
                "judged_at": primary.get("judged_at"),
            } if primary.get("active_judgment_id") is not None else None),
            "related_conflict_count": len(conflicts),
            "open_conflict_count": sum(1 for c in conflicts if c.get("status") == "open"),
            "conflict_peer": ({
                "id": peer_id,
                "subject": peer_summary.get("subject"),
                "status": peer_summary.get("status"),
                "snippet": peer_summary.get("snippet"),
            } if peer_visible else {
                "id": None,
                "subject": None,
                "status": None,
                "snippet": None,
                "redaction_reason": "workspace_acl",
            }),
        }

    def _compute_runtime_hint(
        self,
        memory_id: int,
        rec: dict[str, Any],
        all_results: list[dict[str, Any]],
        result_id_set: set[int],
        dismissed_pairs: Optional[set] = None,
    ) -> Optional[dict[str, Any]]:
        """Compute a runtime_metadata_hint by comparing this result against
        other results in the same result set (bounded to first 20).

        Only fires on high subject/tags overlap + trust disparity. ``dismissed_pairs``
        (v0.8.8 Layer 0): canonical (a,b) pairs the user already judged
        not-a-conflict — skipped so a dismissed pair can't nag via this path.
        """
        dismissed_pairs = dismissed_pairs or set()
        my_tags = set(rec.get("tags") or [])
        my_subject = (rec.get("subject") or "").lower()
        my_trust = self._trust_score(rec)
        # Cap to avoid O(n²) blowup on large result sets.
        candidates = [r for r in all_results[:20] if int(r.get("id", 0)) != memory_id]
        best_peer: Optional[dict[str, Any]] = None
        best_score = 0.0
        for peer in candidates:
            peer_id = int(peer.get("id", 0))
            if (min(memory_id, peer_id), max(memory_id, peer_id)) in dismissed_pairs:
                continue  # v0.8.8 Layer 0: user already dismissed this pair.
            peer_tags = set(peer.get("tags") or [])
            peer_subject = (peer.get("subject") or "").lower()
            # Tags overlap.
            if my_tags and peer_tags:
                common = my_tags & peer_tags
                if len(common) >= 2:
                    overlap_ratio = len(common) / len(my_tags | peer_tags)
                    if overlap_ratio >= 0.8:
                        trust_gap = abs(my_trust - self._trust_score(peer))
                        if trust_gap > 0:
                            score = overlap_ratio + trust_gap * 0.01
                            if score > best_score:
                                best_score = score
                                best_peer = peer
            # Subject overlap.
            if my_subject and peer_subject and best_peer is None:
                # Simple token overlap for ASCII.
                my_tokens = set(my_subject.split())
                peer_tokens = set(peer_subject.split())
                if my_tokens and peer_tokens:
                    overlap = len(my_tokens & peer_tokens) / len(my_tokens | peer_tokens)
                    if overlap >= 0.7:
                        trust_gap = abs(my_trust - self._trust_score(peer))
                        if trust_gap > 0:
                            score = overlap + trust_gap * 0.01
                            if score > best_score:
                                best_score = score
                                best_peer = peer
        if best_peer is None:
            return None
        peer_id = int(best_peer.get("id", 0))
        return {
            "conflict_source": "runtime_metadata_hint",
            "conflict_type": "metadata_overlap",
            "confidence_hint": "low",
            "conflict_point": "subject/tags overlap; not LLM-verified",
            "conflict_peer": {
                "id": peer_id,
                "subject": best_peer.get("subject"),
                "status": best_peer.get("status"),
                "snippet": (best_peer.get("content") or "")[:200],
            },
        }

    # ==================================================================
    #  v0.6.0: Section split tools
    # ==================================================================
