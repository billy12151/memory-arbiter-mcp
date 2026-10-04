"""冲突治理组（arbitrate/list/resolve/apply/replan/judge + guidance/mismatch helper，从 operations.py 搬出，拆分批 ④ 纯移动）。
"""
from __future__ import annotations


import hashlib
import json
import sqlite3
import time
from typing import Any, Mapping, cast, TYPE_CHECKING

from ..acl import CallerWorkspace, WorkspaceScope, forbidden_payload
from ..embedder import ManagedEmbedder
from ..models import ProtectionLevel, SourceType, TrustedApplyingContext
from ..semantic_conflict import normalize_value, value_is_grounded

if TYPE_CHECKING:
    from ..config import Settings
    from ..db import MemoryDB
    from ..update_monitor import UpdateMonitor
    from ..tools import MemoryTools

class _OpsConflicts:
    if TYPE_CHECKING:
        db: "MemoryDB"
        settings: "Settings"
        _tools: "MemoryTools"

        # 主类委托薄层/跨 mixin 成员的 mypy strict 声明（attr-defined）
        @property
        def _update_monitor(self) -> "UpdateMonitor | None": ...
        def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace": ...
        def _conflict_detail_for_workspace(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _embedding_configured(self) -> bool: ...
        def _post_commit(
            self, *args: Any, **kwargs: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]: ...
        def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _get_memory_visible(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _is_truthy(self, *args: Any, **kwargs: Any) -> bool: ...
        def _semantic_notice_workspace_scope(self, *args: Any, **kwargs: Any) -> "WorkspaceScope": ...
        def _semantic_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
        def _strict_acl_unavailable(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def current_agent_id(self) -> "str | None": ...
        def current_client(self) -> "str | None": ...
        def wait_semantic_worker_drained(self, *args: Any, **kwargs: Any) -> bool: ...
        @staticmethod
        def _compare_memories(*args: Any, **kwargs: Any) -> Any: ...

    def memory_arbitrate(self, left_id: int, right_id: int, mark_conflict: bool = True, authorized: bool = False, **_: Any) -> dict[str, Any]:
        authorized = self._is_truthy(authorized)
        if _.get("apply") is not None:
            return self.db.state.response(
                {"error": "the 'apply' parameter was renamed to 'authorized' in v0.8.5 and no longer takes effect; pass authorized=True to auto-supersede the non-protected loser", "applied": False},
                ok=False,
            )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        left = self._get_memory_visible(int(left_id), caller)
        right = self._get_memory_visible(int(right_id), caller)
        if not left or not right:
            missing_data = {"error": "memory id not found"}
            if caller.isolation == "strict":
                missing_data.update(caller.response_fields())
            return self.db.state.response(missing_data, ok=False, extra_warnings=list(caller.warnings))
        comparison = self._compare_memories(left, right)
        conflict_id = None
        conflict_recording = "requires_structured_group" if mark_conflict else "not_requested"
        applied = False
        resolved = 0
        if authorized and comparison["winner_id"] and comparison["loser_id"] and not comparison["manual_review"]:
            loser = self.db.get_memory(int(comparison["loser_id"]))
            if loser and loser.get("protection_level") != ProtectionLevel.LOCKED.value and loser.get("source_type") != SourceType.USER_CONFIRMED.value:
                try:
                    with self.db.write_transaction() as conn:
                        applied = self.db.update_memory_on_conn(
                            conn, int(comparison["loser_id"]), {"status": "superseded"}
                        )
                        if applied:
                            resolved = self.db.resolve_conflicts_for_on_conn(conn, int(comparison["loser_id"]))
                except sqlite3.Error:
                    applied = False
                    resolved = 0
        result_data = {"comparison": comparison, "conflict_id": conflict_id, "conflict_recording": conflict_recording, "applied": applied, "linked_conflicts_resolved": resolved}
        if caller.isolation == "strict":
            result_data.update(caller.response_fields())
        return self.db.state.response(result_data, extra_warnings=list(caller.warnings))

    @staticmethod
    def _with_resolution_guidance(conflict: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(conflict)
        plan = (enriched.get("apply_summary") or {}).get("plan", [])
        pending = next((item for item in plan if item.get("status") == "pending"), None)
        if pending is not None:
            enriched["next_action"] = {
                "tool": "memory_govern", "action": "apply_conflict_action",
                "data": {
                    "conflict_id": enriched.get("id"), "expected_revision": enriched.get("revision"),
                    "memory_id": pending.get("memory_id"), "action": pending.get("action"),
                    "authorized": True,
                },
            }
        elif enriched.get("status") == "applying":
            if any(item.get("status") not in {"pending", "completed"} for item in plan):
                # Failed step, nothing pending: guide to an authorized replan
                # instead of a resolve_conflict that would fail apply_incomplete.
                enriched["next_action"] = {
                    "tool": "memory_govern", "action": "replan_conflict",
                    "data": {"conflict_id": enriched.get("id"), "expected_revision": enriched.get("revision"), "authorized": True},
                }
            else:
                enriched["next_action"] = {
                    "tool": "memory_govern", "action": "resolve_conflict",
                    "data": {"conflict_id": enriched.get("id"), "expected_revision": enriched.get("revision"), "authorized": True},
                }
        return enriched

    def memory_list_conflicts(self, status: str = "open", limit: int = 50, source: str | None = None, **_: Any) -> dict[str, Any]:
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        conflicts: list[dict[str, Any]] = []
        raw_limit = max(int(limit), 1)
        explicit_none_scope = caller.isolation == "none" and caller.source == "explicit"
        if caller.isolation != "strict":
            conflicts = [
                self._with_resolution_guidance(c)
                for c in self.db.conflicts.list_conflicts(
                    status=status, limit=raw_limit, source=source,
                    workspace=caller.canonical if explicit_none_scope else None,
                )
            ]
        else:
            # Scope in SQL BEFORE LIMIT so large out-of-scope backlogs cannot
            # hide an admitted workspace's older conflicts. Page through rows
            # because member revalidation may still discard stale/malformed
            # groups after SQL scoping.
            scope = caller.scope_canonicals()
            page_size = min(max(raw_limit, 50), 1000)
            offset = 0
            while len(conflicts) < raw_limit:
                rows = self.db.conflicts.list_conflicts(
                    status=status, limit=page_size, source=source,
                    workspace=scope, offset=offset,
                )
                if not rows:
                    break
                for c in rows:
                    detail = self._conflict_detail_for_workspace(int(c.get("id") or 0), caller)
                    if detail is not None:
                        conflicts.append(detail["conflict"])
                        if len(conflicts) >= raw_limit:
                            break
                offset += len(rows)
                if len(rows) < page_size:
                    break
        data = {"conflicts": conflicts, "count": len(conflicts)}
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

    def memory_resolve_conflict(
        self, conflict_id: int, expected_revision: int, reason: str = "", **_: Any,
    ) -> dict[str, Any]:
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        conflict = self.db.get_conflict(int(conflict_id))
        if conflict is None or (
            caller.isolation == "strict"
            and str(conflict.get("workspace_canonical") or "") not in set(caller.scope_canonicals())
        ):
            return self.db.state.response(forbidden_payload("conflict", workspace=caller), ok=False, extra_warnings=list(caller.warnings))
        result = self.db.conflicts.resolve_conflict(
            int(conflict_id), reason=reason, expected_revision=int(expected_revision),
            strict_workspace=caller.scope_canonicals() if caller.isolation == "strict" else None,
        )
        return self.db.state.response(
            result, ok=result.get("outcome") == "resolved", extra_warnings=list(caller.warnings),
        )

    @staticmethod
    def _member_value_mismatches(
        conn: sqlite3.Connection, conflict: dict[str, Any], target_id: int, chosen: str,
    ) -> bool:
        """Whether the use_as_resolution target's own value group differs from
        the chosen value (P1 #970 adversarial pass).

        Uses the frozen member snapshot pinned on the conflict row, not the
        live memory: apply semantics operate on the recorded group identity.
        """
        chosen_norm = normalize_value(chosen)
        for member in conflict.get("member_versions") or []:
            if int(member.get("memory_id") or 0) == int(target_id):
                return normalize_value(str(member.get("normalized_value") or "")) != chosen_norm
        return True  # target not among the frozen members: judge already rejects this

    def memory_apply_conflict_action(
        self, conflict_id: int, expected_revision: int, memory_id: int, action: str,
        content: str | None = None, old_text: str | None = None,
        new_text: str | None = None, reason: str = "", authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Apply one planned member action and update its result in one write transaction."""
        if not self._is_truthy(authorized):
            return self.db.state.response({"error": "authorized=True is required"}, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        try:
            conflict_id_int, revision, target_id = int(conflict_id), int(expected_revision), int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "conflict_id, expected_revision, and memory_id must be integers"}, ok=False)
        try:
            with self.db.write_transaction() as conn:
                row = conn.execute("SELECT * FROM conflicts WHERE id=?", (conflict_id_int,)).fetchone()
                if row is None:
                    raise ValueError("not_found")
                conflict = dict(row)
                for json_field in ("slot_key", "candidate_key", "member_versions", "value_groups", "apply_summary"):
                    if isinstance(conflict.get(json_field), str):
                        conflict[json_field] = json.loads(conflict[json_field])
                if conflict.get("status") != "applying":
                    raise ValueError("not_applying")
                if int(conflict.get("revision") or 0) != revision:
                    raise ValueError(f"stale_conflict:{conflict.get('revision')}")
                if caller.isolation == "strict" and not self.db.conflicts._active_members_match_workspace(
                    conn, conflict, caller.scope_canonicals(),
                ):
                    raise PermissionError("forbidden")
                plan = (conflict.get("apply_summary") or {}).get("plan", [])
                step = next((item for item in plan if int(item.get("memory_id") or 0) == target_id), None)
                if step is None or step.get("action") != action or step.get("status") != "pending":
                    raise ValueError("invalid_action")
                current = self.db.get_memory_on_conn(conn, target_id)
                if current is None or int(current.get("version") or 0) != int(step.get("expected_version") or 0):
                    raise ValueError("stale_member")
                if action in {"preserve_historical_record", "use_as_resolution"}:
                    edited = {"outcome": "no_change", "record": current}
                elif action == "needs_authorization":
                    # Not executable by apply: mark it blocked (committed below)
                    # so the guidance surfaces route to replan instead of looping
                    # back to a call that always fails (spec 10.6 keeps applying
                    # and preserves the remaining plan).
                    edited = {"outcome": "blocked", "record": current}
                else:
                    edited = self.db.edit_memory_intent(
                        target_id, new_content=content, old_text=old_text, new_text=new_text,
                        reason=reason or f"Apply conflict #{conflict_id_int}: {action}",
                        authorized=True, expected_version=int(step["expected_version"]), conn=conn,
                    )
                    if edited.get("outcome") != "edited":
                        raise ValueError(str(edited.get("outcome") or "edit_failed"))
                updated = edited.get("record") or current
                if not isinstance(updated, dict):
                    updated = dict(cast(Mapping[str, Any], updated))
                chosen = str(conflict.get("chosen_value") or "")
                updated_content = str(updated.get("content") or "")
                if edited.get("outcome") == "blocked":
                    # needs_authorization: recorded blocked so guidance routes to
                    # replan; no edit happened.
                    step.update(status="blocked", result_version=None, result_hash=None,
                                error="needs_authorization")
                elif action == "update_current_claim" and not value_is_grounded(
                    chosen, updated_content
                ):
                    # The memory edit (if any) and failure bookkeeping must commit
                    # together: applying remains retryable/replannable and history
                    # accurately records that this attempt did not establish the
                    # chosen value. orphaned_edit flags that the content actually
                    # changed (update_current_claim) so a replan accounts for it.
                    # use_as_resolution does not ground here (D2, #970): judge
                    # already guarantees its value-group membership, and the
                    # whole-content grounding of a long normalized value was
                    # structurally impossible.
                    edit_committed = edited.get("outcome") == "edited"
                    step.update(status="failed", result_version=int(updated.get("version") or 0),
                                result_hash=hashlib.sha256(updated_content.encode("utf-8")).hexdigest(),
                                error="chosen_value_not_grounded", orphaned_edit=edit_committed)
                elif action == "use_as_resolution" and chosen and self._member_value_mismatches(
                    conn, conflict, target_id, chosen,
                ):
                    # use_as_resolution must land on a member that actually
                    # holds the chosen value group (P1 #970 adversarial pass):
                    # without this check a plan could mark a mysql holder as
                    # the sqlite resolution and resolve the group, leaving the
                    # wrong data in place. Group membership, not content
                    # grounding (D2): the member-vs-group equality is an
                    # intake invariant (D1), so this check is always
                    # satisfiable — unlike the removed grounding check it
                    # cannot recreate the #970 deadlock.
                    step.update(status="failed", result_version=int(updated.get("version") or 0),
                                result_hash=hashlib.sha256(updated_content.encode("utf-8")).hexdigest(),
                                error="resolution_member_value_mismatch", orphaned_edit=False)
                else:
                    step.update(status="completed", result_version=int(updated.get("version") or step["expected_version"]),
                                result_hash=hashlib.sha256(updated_content.encode("utf-8")).hexdigest(), error=None)
                result_version = int(updated.get("version") or step["expected_version"])
                result_hash = hashlib.sha256(updated_content.encode("utf-8")).hexdigest()
                summary: dict[str, Any] = {"plan": plan}
                prior_history = (conflict.get("apply_summary") or {}).get("history")
                if prior_history:
                    # replan_conflict preserved prior plan snapshots; applying a
                    # step must not silently drop that history.
                    summary["history"] = prior_history
                cur = conn.execute(
                    "UPDATE conflicts SET revision=revision+1,apply_summary=?,refreshed_at=? WHERE id=? AND status='applying' AND revision=?",
                    (json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), conflict_id_int, revision),
                )
                if cur.rowcount != 1:
                    raise ValueError("stale_conflict")
        except PermissionError:
            return self.db.state.response(forbidden_payload("conflict", workspace=caller), ok=False, extra_warnings=list(caller.warnings))
        except ValueError as exc:
            error_code, _separator, current_revision = str(exc).partition(":")
            data: dict[str, Any] = {"outcome": error_code, "error": error_code}
            if current_revision:
                data["revision"] = int(current_revision)
            if error_code in {"stale_conflict", "stale_member"}:
                data["action_required"] = "replan_conflict"
                data["note"] = (
                    "re-read memory_review(view='conflict_detail') and the member memories, then call "
                    "authorized memory_govern(action='replan_conflict') with the current revision"
                )
            return self.db.state.response(data, ok=False, extra_warnings=list(caller.warnings))
        successful = step.get("status") == "completed"
        result = {"outcome": "completed" if successful else "apply_failed", "conflict_id": conflict_id_int, "revision": revision + 1, "memory_id": target_id, "action": action, "apply_summary": summary}
        if not successful and step.get("error") == "chosen_value_not_grounded":
            # D3 (#970): a grounding failure used to dead-end the applying
            # group because replan could not touch chosen_value. Name the two
            # real recoveries so the caller does not loop on a retry that can
            # never succeed.
            result["note"] = (
                "The committed member content does not establish the chosen value. Recover via "
                "memory_govern(action='replan_conflict'): re-edit so the chosen value (the stored "
                "machine-normalized form, not its display casing) appears verbatim in the member "
                "content, or pass a replacement chosen_value drawn from the conflict's "
                "value_groups together with a fresh plan."
            )
        if successful and action not in {"preserve_historical_record", "use_as_resolution"}:
            # Post-commit only: index the committed result, then let the normal
            # semantic worker re-enter. Its task remains allowed to discover
            # unrelated facts; conflict-plan trust is not exposed to callers as
            # a general suppression switch.
            result["evidence_index"], result["semantic_conflict_check"] = self._post_commit(
                target_id, self.db.get_memory(target_id), recheck_conflicts=True,
                trusted_applying_context=TrustedApplyingContext(
                    conflict_id=conflict_id_int, revision=revision + 1,
                    memory_id=target_id, action=action,
                    chosen_value=str(conflict.get("chosen_value") or "") or None,
                ),
            )
        elif not successful and step.get("error") == "chosen_value_not_grounded" and step.get("orphaned_edit"):
            # A grounding-failed edit committed without a post-commit re-index
            # (edit_memory_intent already dropped the old evidence rows). Re-index
            # the orphaned edit — untrusted, so the semantic worker re-checks it —
            # and surface the flag so a replan knows the member content changed.
            # The no_change path (use_as_resolution) has no committed edit to re-index.
            result["orphaned_edit"] = True
            result["evidence_index"], result["semantic_conflict_check"] = self._post_commit(
                target_id, self.db.get_memory(target_id), recheck_conflicts=True,
            )
        next_step = next((item for item in plan if item.get("status") == "pending"), None)
        if successful:
            next_data: dict[str, Any] = {
                "conflict_id": conflict_id_int,
                "expected_revision": revision + 1,
                "authorized": True,
            }
            if caller.isolation == "strict" and caller.workspace:
                next_data["workspace"] = caller.workspace
            if next_step:
                next_data.update({
                    "memory_id": next_step["memory_id"], "action": next_step["action"],
                })
                result["next_action"] = {
                    "tool": "memory_govern", "action": "apply_conflict_action", "data": next_data,
                }
            else:
                result["next_action"] = {
                    "tool": "memory_govern", "action": "resolve_conflict", "data": next_data,
                }
        else:
            replan_data: dict[str, Any] = {
                "conflict_id": conflict_id_int,
                "expected_revision": revision + 1,
                "authorized": True,
            }
            if caller.isolation == "strict" and caller.workspace:
                replan_data["workspace"] = caller.workspace
            result["action_required"] = "replan_conflict"
            result["replan"] = {
                "tool": "memory_govern", "action": "replan_conflict", "data": replan_data,
            }
        return self.db.state.response(result, ok=successful, extra_warnings=list(caller.warnings))

    def memory_replan_conflict(
        self, conflict_id: int, expected_revision: int, apply_plan: list[dict[str, Any]],
        resolution_memory_id: int | None = None, chosen_value: str | None = None,
        authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        if not self._is_truthy(authorized):
            return self.db.state.response({"error": "authorized=True is required"}, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        conflict = self.db.get_conflict(int(conflict_id))
        if conflict is None or (
            caller.isolation == "strict"
            and str(conflict.get("workspace_canonical") or "") not in set(caller.scope_canonicals())
        ):
            return self.db.state.response(
                forbidden_payload("conflict", workspace=caller), ok=False,
                extra_warnings=list(caller.warnings),
            )
        result = self.db.conflicts.replan_conflict(
            int(conflict_id), expected_revision=int(expected_revision), apply_plan=apply_plan,
            resolution_memory_id=resolution_memory_id, chosen_value=chosen_value,
            strict_workspace=caller.scope_canonicals() if caller.isolation == "strict" else None,
        )
        return self.db.state.response(result, ok=result.get("outcome") == "replanned", extra_warnings=list(caller.warnings))
    def memory_judge_conflict(
        self, conflict_id: int, expected_revision: int, chosen_value: str,
        decided_by: str, ref: str | None, reason: str,
        apply_plan: list[dict[str, Any]], resolution_memory_id: int | None,
        authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        try:
            conflict_id_int = int(conflict_id)
            revision = int(expected_revision)
            resolution_id = int(resolution_memory_id) if resolution_memory_id is not None else None
        except (TypeError, ValueError):
            return self.db.state.response({"error": "conflict_id, expected_revision, and resolution_memory_id must be integers"}, ok=False)
        if not isinstance(apply_plan, list):
            return self.db.state.response({"error": "apply_plan must be an array"}, ok=False)
        # Authorization gate (B-E3): a decision attributed to a human
        # (decided_by="user") must be explicitly confirmed by the caller,
        # matching the memory_govern authorization contract. Agent-attributed
        # decisions stay ungated — they are the detector's routine flow.
        if str(decided_by).strip().lower() == "user" and not self._is_truthy(authorized):
            return self.db.state.response(
                {
                    "error": "explicit user authorization required",
                    "action_required": "ask_user_for_authorization",
                    "governance_action": "judge",
                    "impact": "Records a user-attributed conflict decision and starts applying its plan.",
                    "authorized": False,
                    "retry": {
                        "tool": "memory",
                        "action": "judge",
                        "set_after_user_confirmation": {"authorized": True},
                    },
                },
                ok=False,
            )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        conflict = self.db.get_conflict(conflict_id_int)
        if conflict is None or (
            caller.isolation == "strict"
            and str(conflict.get("workspace_canonical") or "") not in set(caller.scope_canonicals())
        ):
            return self.db.state.response(forbidden_payload("conflict", workspace=caller), ok=False, extra_warnings=list(caller.warnings))
        result = self.db.judge_conflict(
            conflict_id_int, expected_revision=revision, chosen_value=str(chosen_value),
            decided_by=str(decided_by), decided_ref=ref, decision_reason=str(reason),
            apply_plan=apply_plan, resolution_memory_id=resolution_id,
            strict_workspace=caller.scope_canonicals() if caller.isolation == "strict" else None,
        )
        if result.get("outcome") == "applying":
            plan = (result.get("apply_summary") or {}).get("plan", [])
            pending = next((item for item in plan if item.get("status") == "pending"), None)
            if pending is not None:
                next_data: dict[str, Any] = {
                    "conflict_id": conflict_id_int,
                    "expected_revision": result["revision"],
                    "memory_id": pending["memory_id"],
                    "action": pending["action"],
                    "authorized": True,
                }
                if caller.isolation == "strict" and caller.workspace:
                    next_data["workspace"] = caller.workspace
                result["next_action"] = {
                    "tool": "memory_govern",
                    "action": "apply_conflict_action",
                    "data": next_data,
                }
        elif result.get("outcome") in {"stale_conflict", "stale_member"}:
            result["note"] = (
                "re-read memory_review(view='conflict_detail') and the member memories, then retry judge "
                "with the current revision. For stale_member a member was edited outside the plan: judge "
                "pins versions from the recorded group snapshot, so first register the member's new version "
                "via memory_repair(task='record_conflict') with status='open' and the current "
                "expected_revision (or plan around that member) before retrying."
            )
        return self.db.state.response(result, ok=result.get("outcome") == "applying", extra_warnings=list(caller.warnings))
