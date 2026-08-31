"""Governance, edit, status, and maintenance operations for MemoryTools (Phase 4 extraction)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping, cast, TYPE_CHECKING

from .. import __version__
from ..acl import CallerWorkspace, WorkspaceScope, scope_names, workspace_scope_sql, forbidden_payload, raw_workspace
from ..arbitration import compare_memories
from ..constants import (
    DEFAULT_WORKSPACE_NAME,
    WORKSPACE_MIN_NAME_LEN,
    WORKSPACE_RECALL_ADMISSION,
    WORKSPACE_RECALL_CUTOFF,
    WORKSPACE_WEAK_VECTOR_WEIGHT,
    is_default_workspace_term,
)
from ..db import MemoryDB, _normalize_alias_key
from ..embedder import ManagedEmbedder
from ..db_generation import database_startup_lock
from ..db.workspaces import _coerce_ws, _mechanical_ws_key
from ..validation import MAX_BATCH_IDS, _controlled_integer
from ..models import MemoryRecord, MemoryStatus, ProtectionLevel, SourceType, TrustedApplyingContext
from ..semantic_conflict import normalize_value, value_is_grounded
from ..text import canon_entity as _canon_entity, canon_scope as _canon_scope

if TYPE_CHECKING:
    from ..update_monitor import UpdateMonitor
    from ..tools import MemoryTools


class OperationsPipeline:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools
        self.db = tools.db
        self.settings = tools.settings
        self._evidence_worker = tools._evidence_worker

    @property
    def _update_monitor(self) -> "UpdateMonitor | None":
        # Assigned on MemoryTools after pipeline construction: resolve lazily.
        return self._tools._update_monitor

    def _allowed(self, *args: Any, **kwargs: Any) -> "tuple[bool, list[str]]":
        return self._tools._allowed(*args, **kwargs)

    def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace":
        return self._tools._caller_workspace(*args, **kwargs)

    def _conflict_detail_for_workspace(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._conflict_detail_for_workspace(*args, **kwargs)

    def _embedding_configured(self) -> bool:
        return self._tools._embedding_configured()

    def _post_commit(
        self, *args: Any, **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._tools._post_commit(*args, **kwargs)

    def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_active_embedder()

    def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_embedder()

    def _get_memory_visible(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._get_memory_visible(*args, **kwargs)

    def _is_truthy(self, *args: Any, **kwargs: Any) -> bool:
        return self._tools._is_truthy(*args, **kwargs)

    def _semantic_notice_workspace_scope(self, *args: Any, **kwargs: Any) -> "WorkspaceScope":
        return self._tools._semantic_notice_workspace_scope(*args, **kwargs)

    def _semantic_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._tools._semantic_status(*args, **kwargs)

    def _strict_acl_unavailable(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._strict_acl_unavailable(*args, **kwargs)

    def current_agent_id(self) -> "str | None":
        return self._tools.current_agent_id()

    def current_client(self) -> "str | None":
        return self._tools.current_client()

    def wait_evidence_worker_drained(self, *args: Any, **kwargs: Any) -> bool:
        return self._tools.wait_evidence_worker_drained(*args, **kwargs)

    @staticmethod
    def _compare_memories(*args: Any, **kwargs: Any) -> Any:
        # Preserve legacy patch seam for memory_arbiter.tools.compare_memories.
        from .. import tools as tools_mod
        return getattr(tools_mod, "compare_memories")(*args, **kwargs)

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
                elif action in {"update_current_claim", "use_as_resolution"} and not value_is_grounded(
                    chosen, updated_content
                ):
                    # The memory edit (if any) and failure bookkeeping must commit
                    # together: applying remains retryable/replannable and history
                    # accurately records that this attempt did not establish the
                    # chosen value. orphaned_edit flags that the content actually
                    # changed (update_current_claim) so a replan accounts for it;
                    # use_as_resolution takes the no_change path and is not orphaned.
                    edit_committed = edited.get("outcome") == "edited"
                    step.update(status="failed", result_version=int(updated.get("version") or 0),
                                result_hash=hashlib.sha256(updated_content.encode("utf-8")).hexdigest(),
                                error="chosen_value_not_grounded", orphaned_edit=edit_committed)
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
        resolution_memory_id: int | None = None, authorized: bool = False, **_: Any,
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
            resolution_memory_id=resolution_memory_id,
            strict_workspace=caller.scope_canonicals() if caller.isolation == "strict" else None,
        )
        return self.db.state.response(result, ok=result.get("outcome") == "replanned", extra_warnings=list(caller.warnings))

    def memory_confirm(self, memory_id: int, source_ref: str | None = None, confidence: float = 1.0, authorized: bool = False, **_: Any) -> dict[str, Any]:
        authorized = self._is_truthy(authorized)
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to confirm a memory", "confirmed": False},
                ok=False,
            )
        if isinstance(confidence, bool):
            return self.db.state.response(
                {"error": "confidence must be a finite number between 0 and 1", "confirmed": False},
                ok=False,
            )
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            return self.db.state.response(
                {"error": "confidence must be a finite number between 0 and 1", "confirmed": False},
                ok=False,
            )
        if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
            return self.db.state.response(
                {"error": "confidence must be a finite number between 0 and 1", "confirmed": False},
                ok=False,
            )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        updated: dict[str, Any] | None = None
        error: str | None = None
        try:
            with self.db.write_transaction() as conn:
                memory = self.db.get_memory_on_conn(conn, int(memory_id))
                if not memory:
                    error = "memory id not found"
                elif caller.isolation == "strict" and raw_workspace(memory) not in set(caller.scope_canonicals()):
                    error = "memory id not found"
                elif memory.get("status") != "active":
                    error = f"memory is not active (status={memory.get('status')}); cannot confirm inactive memory"
                else:
                    metadata = dict(memory.get("metadata") or {})
                    metadata["confirmed_from"] = source_ref or "manual"
                    ok = self.db.update_memory_on_conn(
                        conn,
                        int(memory_id),
                        {
                            "source_type": SourceType.USER_CONFIRMED.value,
                            "confidence": confidence_value,
                            "protection_level": ProtectionLevel.LOCKED.value,
                            "metadata": metadata,
                        },
                    )
                    if not ok:
                        error = "failed to confirm memory"
                    else:
                        updated = self.db.get_memory_on_conn(conn, int(memory_id))
        except sqlite3.Error as exc:
            error = f"confirm failed; transaction rolled back: {exc}"
        if error is not None:
            data: dict[str, Any] = {"error": error, "confirmed": False}
            if caller.isolation == "strict":
                data.update(caller.response_fields())
            return self.db.state.response(data, ok=False, extra_warnings=list(caller.warnings))
        ok = updated is not None
        data = {"confirmed": ok, "record": updated}
        if ok:
            data["evidence_index"], data["semantic_conflict_check"] = self._post_commit(
                int(memory_id), updated, recheck_conflicts=False,
            )
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

    # ------------------------------------------------------------------
    #  Workspace rename/migration and strict pending confirmation.
    # ------------------------------------------------------------------
    def memory_rename_workspace_canonical(
        self, old: str, new: str, reason: str | None = None, **_: Any,
    ) -> dict[str, Any]:
        """Rename a canonical workspace and maintain internal forwarding."""
        updated, warnings = self.db.rename_workspace_canonical(old, new)
        return self.db.state.response(
            {
                "renamed": not warnings,
                "old": old,
                "new": new,
                "memories_updated": updated,
            },
            ok=not warnings, extra_warnings=warnings,
        )

    def memory_migrate_workspace(
        self, reason: str | None = None, **payload: Any,
    ) -> dict[str, Any]:
        """Merge one workspace into another and maintain internal forwarding.

        `from`/`to` are reserved words so they arrive via **payload.
        """
        from_ws = str(payload.get("from") or "")
        to_ws = str(payload.get("to") or "")
        embedder, ensure_warnings = self._ensure_active_embedder()
        updated, warnings = self.db.migrate_workspace(
            from_ws, to_ws, embedder=embedder,
        )
        vector_publish_pending = any("workspace canonical vector publish failed" in warning for warning in warnings)
        operation_warnings = [
            warning for warning in warnings
            if "workspace canonical vector publish failed" not in warning
        ]
        data: dict[str, Any] = {
            "migrated": not operation_warnings,
            "from": from_ws,
            "to": to_ws,
            "memories_updated": updated,
        }
        if vector_publish_pending:
            data["workspace_vector_publish"] = {
                "status": "pending_retry",
                "canonical": to_ws,
                "retry": "After sqlite-vec and embedding configuration recover, write another memory using this workspace to retry publication.",
                "repair_task_available": False,
            }
        # Embedder-init and vector-publication warnings are observable degraded
        # indexing, not a rollback of the completed workspace migration.
        return self.db.state.response(
            data,
            ok=not operation_warnings,
            extra_warnings=list(ensure_warnings) + list(warnings),
        )

    def memory_move_memories_workspace(
        self,
        memory_ids: list[int],
        new_workspace: str,
        reason: str | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Move selected memories by id to a different workspace bucket.

        Companion to migrate_workspace: migrate merges one canonical
        workspace into another by name and reroutes the alias; move
        reassigns individual memories — the raw bucket column and
        workspace_canonical together — and leaves alias/normalization rules
        untouched. Divergent rows (workspace_canonical already pointing
        somewhere other than the raw bucket, e.g. rows written through a
        confirmed alias) are refused unless the call is authorized, in which
        case they are re-anchored to the destination with an explicit
        response note. The source bucket may be default; default is never a
        valid destination. Moving does not change memory status: pending
        memories stay pending, and superseded/deleted rows keep their status
        (reported via moved_non_active).
        """
        authorized = self._is_truthy(authorized)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        warnings: list[str] = list(caller.warnings)
        admitted: set[str] = set()
        if caller.isolation == "strict":
            admitted = {
                str(name or "").strip()
                for name in caller.scope_canonicals()
                if str(name or "").strip()
            }

        ids: list[int] = []
        for value in list(memory_ids or []):
            # Same strict coercion as the product surface (_controlled_integer):
            # a direct call must not let bool True / float 1.5 silently become
            # memory id 1.
            memory_id = _controlled_integer(value)
            if memory_id is None or memory_id <= 0:
                return self.db.state.response(
                    {
                        "moved": False,
                        "error": "memory_ids must contain positive integer ids",
                        "field": "memory_ids",
                    },
                    ok=False, extra_warnings=warnings,
                )
            if memory_id not in ids:
                ids.append(memory_id)
        if len(ids) > MAX_BATCH_IDS:
            return self.db.state.response(
                {
                    "moved": False,
                    "error": f"memory_ids must be a list with at most {MAX_BATCH_IDS} items",
                    "field": "memory_ids",
                },
                ok=False, extra_warnings=warnings,
            )

        requested = str(new_workspace or "").strip()
        target = _coerce_ws(new_workspace)
        if not target:
            return self.db.state.response(
                {
                    "moved": False,
                    "error": "new_workspace must be a non-empty workspace string",
                    "field": "new_workspace",
                },
                ok=False, extra_warnings=warnings,
            )

        def destination_error(name: str) -> dict[str, Any] | None:
            if is_default_workspace_term(name):
                return {
                    "moved": False,
                    "error": (
                        "default is a reserved global pool and cannot be a move "
                        "destination; choose a non-default workspace"
                    ),
                    "field": "new_workspace",
                }
            if caller.isolation == "strict" and name not in admitted:
                return {
                    "moved": False,
                    "error": (
                        "forbidden_strict_workspace: new_workspace is outside "
                        "the caller workspace scope"
                    ),
                    "field": "new_workspace",
                    **caller.response_fields(),
                }
            return None

        def destination_block_response(blocked: dict[str, Any]) -> dict[str, Any]:
            # Reconciliation contract: ids are already parsed when a
            # destination is rejected, so the response must account for them.
            blocked.update({
                "moved_ids": [],
                "failed_ids": list(ids),
                "errors": [
                    {"memory_id": memory_id, "reason": "destination_rejected"}
                    for memory_id in ids
                ],
            })
            return self.db.state.response(blocked, ok=False, extra_warnings=warnings)

        bad_destination = destination_error(target)
        if bad_destination is not None:
            return destination_block_response(bad_destination)

        # Destination orthography, mirroring what a write using the same name
        # would do: follow ONE confirmed alias hop to its decision canonical
        # (checked first, matching the write path's alias short-circuit), else
        # land on the registered spelling of a mechanical twin (migrate's
        # destination fold). Never a second alias hop — a write using the
        # same name would not take one either.
        def fold_destination(name: str) -> str:
            alias_target = self.db.workspaces.confirmed_alias_canonical(name)
            if alias_target:
                return alias_target
            return self.db.workspaces.registered_mechanical_canonical(name) or name

        target = fold_destination(target)
        bad_destination = destination_error(target)
        if bad_destination is not None:
            return destination_block_response(bad_destination)

        # Advisory pre-validation (fast-fail before any transaction): per-id
        # visibility plus the divergence gate. The write transaction re-checks
        # each row on its own connection snapshot, so a row that diverges,
        # vanishes, or leaves the caller scope in the window between the two
        # is failed per id instead of silently re-anchored.
        failures: list[dict[str, Any]] = []
        movable: list[int] = []
        for memory_id in ids:
            memory = self._get_memory_visible(memory_id, caller)
            if not memory:
                failures.append({
                    "memory_id": memory_id,
                    "reason": "not_found_or_forbidden",
                })
                continue
            bucket = str(memory.get("workspace") or "").strip()
            canonical = str(memory.get("workspace_canonical") or "").strip()
            if canonical and bucket and canonical != bucket and not authorized:
                failures.append({
                    "memory_id": memory_id,
                    "reason": "canonical_diverged",
                    "workspace": bucket,
                    "workspace_canonical": canonical,
                })
                continue
            movable.append(memory_id)

        if not movable:
            return self.db.state.response(
                {
                    "moved": False,
                    "moved_ids": [],
                    "failed_ids": [failure["memory_id"] for failure in failures],
                    "errors": failures,
                    "new_workspace": target,
                },
                ok=False, extra_warnings=warnings,
            )

        # Canonical embedding is prepared outside the write transaction, like
        # every other canonical registration path.
        embedder, ensure_warnings = self._ensure_active_embedder()
        warnings.extend(ensure_warnings)
        target_embedding = self.db.workspaces.prepare_missing_workspace_canonical_embedding(
            target, embedder,
        )

        moved: list[int] = []
        forced: list[dict[str, Any]] = []
        non_active: list[dict[str, Any]] = []
        bucket_was_new = False

        def aborted(exc: BaseException) -> dict[str, Any]:
            # The transaction rolled back: nothing moved, so every candidate
            # id is accounted for in failed_ids (reconciliation contract:
            # moved_ids + failed_ids covers the request). The vector-publish
            # warning is dropped — after rollback the canonical row it names
            # was never committed, so the retry guidance would be misleading.
            reason_text = f"aborted: {exc}"
            abort_warnings = [
                warning for warning in warnings
                if "workspace canonical vector publish failed" not in warning
            ]
            return self.db.state.response(
                {
                    "moved": False,
                    "moved_ids": [],
                    "failed_ids": [
                        failure["memory_id"] for failure in failures
                    ] + list(movable),
                    "errors": failures + [
                        {"memory_id": memory_id, "reason": reason_text}
                        for memory_id in movable
                    ],
                    "new_workspace": target,
                },
                ok=False, extra_warnings=abort_warnings,
            )

        try:
            # Same advisory flock as rename/migrate/normalize, always taken
            # before the write transaction so every registry-mutating path
            # serializes in one order; a concurrent merge can no longer
            # resurrect a spelling between the fold read and the commit.
            with database_startup_lock(self.settings.db_path), self.db.write_transaction() as conn:
                # Re-fold from the ORIGINAL request, single hop, on the
                # transaction's own snapshot. Re-folding the already-folded
                # target would follow a second alias hop that a write using
                # the same name would never take.
                locked = (
                    self.db.workspaces._mechanical_canonical_on_conn(conn, requested)
                    or requested
                )
                alias_row = conn.execute(
                    "SELECT canonical FROM workspace_aliases "
                    "WHERE alias_workspace=? AND status='confirmed' "
                    "ORDER BY updated_at DESC, canonical ASC LIMIT 1",
                    (_normalize_alias_key(requested),),
                ).fetchone()
                if alias_row is not None:
                    locked = str(alias_row["canonical"])
                if locked != target:
                    blocked = destination_error(locked)
                    if blocked is not None:
                        # Reconciliation contract: after ids were resolved the
                        # response must still account for every request id.
                        blocked.update({
                            "new_workspace": target,
                            "moved_ids": [],
                            "failed_ids": [
                                failure["memory_id"] for failure in failures
                            ] + list(movable),
                            "errors": failures + [
                                {
                                    "memory_id": memory_id,
                                    "reason": "destination_changed_after_recheck",
                                }
                                for memory_id in movable
                            ],
                        })
                        if requested != target:
                            blocked["requested_new_workspace"] = requested
                        return self.db.state.response(
                            blocked, ok=False, extra_warnings=warnings,
                        )
                    target = locked
                    # The prepared embedding belongs to the stale spelling;
                    # drop it rather than publish a mismatched vector under
                    # the rechecked canonical (a later write republishes it).
                    target_embedding = None
                bucket_was_new = conn.execute(
                    "SELECT 1 FROM workspace_canonicals WHERE name = ?", (target,),
                ).fetchone() is None
                for memory_id in list(movable):
                    row = conn.execute(
                        "SELECT workspace, workspace_canonical, status FROM memories "
                        "WHERE id = ?",
                        (memory_id,),
                    ).fetchone()
                    if row is None:
                        failures.append({
                            "memory_id": memory_id,
                            "reason": "not_found_or_forbidden",
                        })
                        movable.remove(memory_id)
                        continue
                    bucket = str(row["workspace"] or "").strip()
                    canonical = str(row["workspace_canonical"] or "").strip()
                    if caller.isolation == "strict":
                        effective = canonical or bucket
                        if effective not in admitted:
                            failures.append({
                                "memory_id": memory_id,
                                "reason": "not_found_or_forbidden",
                            })
                            movable.remove(memory_id)
                            continue
                    if canonical and bucket and canonical != bucket:
                        if not authorized:
                            failures.append({
                                "memory_id": memory_id,
                                "reason": "canonical_diverged",
                                "workspace": bucket,
                                "workspace_canonical": canonical,
                            })
                            movable.remove(memory_id)
                            continue
                        forced.append({
                            "memory_id": memory_id,
                            "workspace": bucket,
                            "workspace_canonical": canonical,
                        })
                    ok_move, move_warnings = self.db.workspaces.move_memory_workspace_on_conn(
                        conn, memory_id, target,
                        precomputed_embedding=target_embedding,
                    )
                    if not ok_move:
                        failures.append({
                            "memory_id": memory_id,
                            "reason": "not_found_or_forbidden",
                        })
                        movable.remove(memory_id)
                        continue
                    for warning in move_warnings:
                        if warning not in warnings:
                            warnings.append(warning)
                    status = str(row["status"] or "")
                    if status not in (MemoryStatus.ACTIVE.value, MemoryStatus.PENDING.value):
                        non_active.append({"memory_id": memory_id, "status": status})
                    moved.append(memory_id)
        except OSError as exc:
            abort_warnings = [
                warning for warning in warnings
                if "workspace canonical vector publish failed" not in warning
            ]
            return self.db.state.response(
                {
                    "moved": False,
                    "error": f"workspace move lock unavailable: {exc}",
                    "moved_ids": [],
                    "failed_ids": [
                        failure["memory_id"] for failure in failures
                    ] + list(movable),
                    "errors": failures + [
                        {"memory_id": memory_id, "reason": f"aborted: {exc}"}
                        for memory_id in movable
                    ],
                    "new_workspace": target,
                },
                ok=False, extra_warnings=abort_warnings,
            )
        except sqlite3.Error as exc:
            return aborted(exc)

        # Open/applying conflicts keep their NOT NULL scope key: a by-id move
        # must not rewrite conflict scopes wholesale (only migrate, which owns
        # a whole canonical, does). Moved members whose conflict is scoped
        # elsewhere are reported for a scan instead.
        conflict_scope_ids: list[int] = []
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, workspace_canonical, member_versions FROM conflicts "
                    "WHERE status IN ('open','applying')"
                ).fetchall()
            moved_set = set(moved)
            for row in rows:
                if str(row["workspace_canonical"] or "") == target:
                    continue
                try:
                    members = json.loads(str(row["member_versions"] or "[]"))
                except (TypeError, ValueError):
                    continue
                member_ids: set[int] = set()
                for member in members if isinstance(members, list) else []:
                    if not isinstance(member, dict):
                        continue
                    raw_member_id = member.get("memory_id")
                    if raw_member_id is None:
                        continue
                    try:
                        member_ids.add(int(raw_member_id))
                    except (TypeError, ValueError):
                        continue
                if member_ids & moved_set:
                    conflict_scope_ids.append(int(row["id"]))
        except sqlite3.Error:
            conflict_scope_ids = []

        data: dict[str, Any] = {
            "moved": not failures,
            "moved_ids": moved,
            "failed_ids": [failure["memory_id"] for failure in failures],
            "errors": failures,
            "new_workspace": target,
        }
        if requested != target:
            data["requested_new_workspace"] = requested
        if forced:
            data["forced_reanchored"] = {
                "memory_ids": [entry["memory_id"] for entry in forced],
                "details": forced,
                "note": (
                    "These rows had workspace_canonical pointing somewhere other than "
                    "their workspace bucket; both columns were re-anchored to the "
                    "destination. Normalization rules are unchanged: future writes "
                    "using the old workspace name still resolve to the old registered "
                    "canonical. To reroute that name, use migrate_workspace or "
                    "rename_workspace_canonical."
                ),
                "suggested_call": {
                    "tool": "memory_govern",
                    "action": "migrate_workspace",
                    "data": {"from": forced[0]["workspace_canonical"], "to": target},
                },
                "authorization_required": True,
            }
        if non_active:
            data["moved_non_active"] = non_active
        if conflict_scope_ids:
            data["conflict_scope_note"] = {
                "conflict_ids": conflict_scope_ids,
                "note": (
                    "These open/applying conflicts are scoped to a different "
                    "workspace_canonical than the destination; their scope keys were "
                    "left unchanged on purpose. Re-check them via "
                    "memory_repair(task='scan_candidates')."
                ),
            }
        if any("workspace canonical vector publish failed" in warning for warning in warnings):
            data["workspace_vector_publish"] = {
                "status": "pending_retry",
                "canonical": target,
                "retry": (
                    "After sqlite-vec and embedding configuration recover, write "
                    "another memory using this workspace to retry publication."
                ),
                "repair_task_available": False,
            }
        response = self.db.state.response(data, ok=not failures, extra_warnings=warnings)
        if bucket_was_new and moved:
            # Deliberately delivered even on partial failure: the bucket row
            # IS committed in that case, and dropping the review notice would
            # hide a registered-but-unreviewed workspace.
            response.setdefault("notices", []).append({
                "type": "workspace_review",
                "severity": "info",
                "workspace": target,
                "message": (
                    f"New workspace {target!r} was registered. "
                    "Review the workspace registry for duplicates before confirming it."
                ),
                "action_required": "review_workspace_registry",
                "review_call": {"tool": "memory_review", "view": "doctor", "data": {}},
                "confirm_call": {"tool": "memory_govern", "action": "confirm_workspaces", "data": {}},
                "authorization_required": True,
            })
        return response

    def memory_confirm_pending_workspace(
        self, memory_id: int, canonical: str, reason: str | None = None,
        authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        """Assign a pending memory's canonical workspace and activate it.

        When the raw and selected canonical names differ, an internal redirect
        is recorded so future writes using the old raw name do not re-split.
        """
        authorized = self._is_truthy(authorized)
        explicit_workspace = _.get("workspace")
        caller = self._caller_workspace(explicit_workspace) if explicit_workspace else None
        if caller is not None:
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
        warnings: list[str] = list(caller.warnings) if caller is not None else []

        def visible_error_record() -> dict[str, Any] | None:
            if caller is not None:
                return self._get_memory_visible(int(memory_id), caller)
            return self.db.get_memory(int(memory_id))

        # Confirmation assigns an already user-selected canonical. It must not
        # invoke embedding/model work; a later ordinary write can publish the
        # canonical vector through the normal retry path.
        activated = False
        updated: dict[str, Any] | None = None
        try:
            with self.db.write_transaction() as conn:
                memory = self.db.get_memory_on_conn(conn, int(memory_id))
                if not memory:
                    raise ValueError("memory id not found")
                if memory.get("status") != MemoryStatus.PENDING.value:
                    raise ValueError(
                        f"memory is not pending (status={memory.get('status')}); only pending memories can be confirmed"
                    )
                raw_ws = memory.get("workspace") or ""
                if is_default_workspace_term(raw_ws):
                    if not is_default_workspace_term(canonical):
                        raise ValueError(
                            "reserved default workspace cannot be confirmed into a project canonical"
                        )
                    canonical = DEFAULT_WORKSPACE_NAME
                else:
                    if _mechanical_ws_key(raw_ws) == _mechanical_ws_key(canonical):
                        canonical = str(raw_ws).strip()
                    else:
                        resolved_canonical = self.db.resolve_workspace_canonical(
                            canonical, None, register_new=False,
                        )
                        canonical = str(
                            resolved_canonical.get("canonical") or canonical
                        ).strip()
                if caller is not None and caller.isolation == "strict":
                    memory_workspace = raw_workspace(memory)
                    if not memory_workspace or memory_workspace != caller.canonical:
                        raise ValueError("forbidden_strict_workspace: pending memory is outside caller workspace")
                    if str(canonical or "").strip() != caller.canonical:
                        raise ValueError("forbidden_strict_workspace: canonical must match caller workspace")
                alias_warnings: list[str]
                if is_default_workspace_term(raw_ws):
                    # a reserved default synonym raw is already the global
                    # pool; no internal redirect is meaningful for it. Fold
                    # a default-term canonical to the one true spelling so a
                    # synonym can never be re-persisted as a phantom canonical
                    # (round-2 review: the bypass must not drop the
                    # canonical-side default-term guard either).
                    if is_default_workspace_term(canonical):
                        canonical = DEFAULT_WORKSPACE_NAME
                    ok_alias, alias_warnings = True, []
                elif _normalize_alias_key(raw_ws) == _normalize_alias_key(canonical):
                    ok_alias, alias_warnings = True, []
                else:
                    ok_alias, alias_warnings = self.db.record_workspace_decision_on_conn(
                        conn, raw_ws, canonical, status="confirmed",
                        force=self._is_truthy(authorized),
                    )
                if not ok_alias:
                    raise ValueError("; ".join(alias_warnings) or "workspace redirect not written")
                warnings.extend(alias_warnings)
                canonical_set, canonical_warnings = self.db.set_memory_workspace_canonical_on_conn(
                    conn, int(memory_id), canonical,
                )
                if not canonical_set:
                    raise ValueError("; ".join(canonical_warnings) or "workspace_canonical not set")
                warnings.extend(canonical_warnings)
                activated = self.db.update_memory_on_conn(
                    conn, int(memory_id), {"status": MemoryStatus.ACTIVE.value},
                )
                if not activated:
                    raise ValueError("failed to activate pending memory")
                updated = self.db.get_memory_on_conn(conn, int(memory_id))
        except ValueError as exc:
            data = {
                "confirmed": False,
                "activated": False,
                "canonical": canonical,
                "record": visible_error_record(),
                "error": str(exc),
            }
            return self.db.state.response(data, ok=False, extra_warnings=warnings)
        except sqlite3.Error as exc:
            data = {
                "confirmed": False,
                "activated": False,
                "canonical": canonical,
                "record": visible_error_record(),
                "error": f"confirm pending workspace failed: {exc}",
            }
            return self.db.state.response(data, ok=False, extra_warnings=warnings)
        except Exception as exc:
            data = {
                "confirmed": False,
                "activated": False,
                "canonical": canonical,
                "record": visible_error_record(),
                "error": f"confirm pending workspace failed: {exc}",
            }
            return self.db.state.response(data, ok=False, extra_warnings=warnings)
        data = {
            "confirmed": True,
            "activated": activated,
            "canonical": canonical,
            "record": updated,
        }
        if any("workspace canonical vector publish failed" in warning for warning in warnings):
            data["workspace_vector_publish"] = {
                "status": "pending_retry",
                "canonical": canonical,
                "retry": "After sqlite-vec and embedding configuration recover, write another memory using this workspace to retry publication.",
                "repair_task_available": False,
            }
        if caller is not None and caller.isolation == "strict":
            data.update(caller.response_fields())
        if activated:
            data["record"] = self.db.get_memory(int(memory_id))
            data["evidence_index"], data["semantic_conflict_check"] = self._post_commit(
                int(memory_id), data["record"], recheck_conflicts=False,
            )
        return self.db.state.response(data, ok=True, extra_warnings=warnings)

    def memory_confirm_workspaces(
        self,
        workspaces: list[str] | None = None,
        reason: str | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Record the user-confirmed workspace registry snapshot.

        Writes workspace_review.json next to the DB — the baseline doctor's
        workspace.review diffs against. Explicit governance action only:
        doctor never refreshes the snapshot itself, otherwise a routine run
        would silently mark unreviewed workspaces confirmed. Default is to
        snapshot the CURRENT workspace_canonicals registry — run it after
        rename/migrate merges so the final normalized set is recorded; an
        explicit ``workspaces`` list overrides (e.g. to confirm a subset).
        Reserved default terms are never stored in the snapshot.
        """
        if not self._is_truthy(authorized):
            return self.db.state.response(
                {"error": "authorized=True is required to confirm workspaces", "confirmed": False},
                ok=False,
            )
        from ..doctor import WORKSPACE_REVIEW_SIDECAR
        from ..models import utc_now_iso

        if workspaces is None:
            try:
                with self.db.connection() as conn:
                    rows = conn.execute(
                        "SELECT name FROM workspace_canonicals ORDER BY name"
                    ).fetchall()
                names = [str(row["name"]) for row in rows]
            except sqlite3.Error as exc:
                return self.db.state.response(
                    {"confirmed": False, "error": f"workspace registry unreadable: {exc}"},
                    ok=False,
                )
        else:
            # Authoritative guard for direct callers that bypass the product
            # surface (validation.py + surfaces dispatch type-check too): a
            # non-list input must be refused, not str()-iterated into
            # single-character names. Bounded like every other product list
            # field (≤100 items × 2000 chars).
            if not isinstance(workspaces, list) or any(
                not isinstance(name, str) for name in workspaces
            ):
                return self.db.state.response(
                    {"confirmed": False, "error": "workspaces must be a list of workspace name strings"},
                    ok=False,
                )
            names = list(workspaces)
            if len(names) > 100 or any(len(name) > 2000 for name in names):
                return self.db.state.response(
                    {"confirmed": False, "error": "workspaces must be at most 100 items of at most 2000 characters each"},
                    ok=False,
                )
        confirmed = sorted({
            name.strip() for name in names
            if name.strip() and not is_default_workspace_term(name)
        })
        sidecar = Path(self.settings.db_path).parent / WORKSPACE_REVIEW_SIDECAR
        snapshot = {
            "confirmed_workspaces": confirmed,
            "confirmed_at": utc_now_iso(),
            "version": 1,
        }
        reason_text = str(reason or "").strip()
        if reason_text:
            # Accepted-and-bounded by validation; persist it so the snapshot
            # answers "who confirmed what, why" like the alias audit trail.
            snapshot["reason"] = reason_text[:2000]
        try:
            # Unique tmp name + os.replace so neither a concurrent confirm
            # (same fixed tmp path would collide) nor a concurrent doctor read
            # can observe a torn file (which would degrade to a spurious full
            # re-review).
            tmp_path = sidecar.with_name(
                f"{sidecar.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            tmp_path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_path, sidecar)
        except OSError as exc:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return self.db.state.response(
                {"confirmed": False, "error": f"workspace review snapshot write failed: {exc}"},
                ok=False,
            )
        return self.db.state.response({
            "confirmed": True,
            "confirmed_workspaces": confirmed,
            "count": len(confirmed),
            "sidecar": str(sidecar),
        })

    def memory_activate(
        self, memory_id: int, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        """Activate a pending memory blocked by strict workspace isolation.

        strict isolation writes brand-new workspaces as status=pending (excluded
        from active recall) until the user confirms the workspace name. This
        flips it to active — without the trust/protection promotion that
        memory_confirm applies. Requires authorized=true.
        """
        authorized = self._is_truthy(authorized)
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to activate a pending memory", "activated": False},
                ok=False,
            )
        explicit_workspace = _.get("workspace")
        caller = (
            self._caller_workspace(explicit_workspace)
            if explicit_workspace or self.settings.isolation == "strict" else None
        )
        if caller is not None:
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
        updated: dict[str, Any] | None = None
        error: str | None = None
        action_required = False
        canonical = ""
        try:
            with self.db.write_transaction() as conn:
                memory = self.db.get_memory_on_conn(conn, int(memory_id))
                if not memory:
                    error = "memory id not found"
                elif caller is not None and caller.isolation == "strict" and raw_workspace(memory) not in set(caller.scope_canonicals()):
                    error = "memory id not found"
                elif memory.get("status") != MemoryStatus.PENDING.value:
                    error = f"memory is not pending (status={memory.get('status')}); only pending memories can be activated"
                else:
                    canonical = raw_workspace(memory)
                    if caller is not None and caller.isolation == "strict":
                        registered = conn.execute(
                            "SELECT 1 FROM workspace_canonicals WHERE name=?",
                            (canonical,),
                        ).fetchone()
                        if registered is None:
                            error = "strict new workspaces require confirm_pending_workspace"
                            action_required = True
                    if error is None:
                        ok = self.db.update_memory_on_conn(
                            conn, int(memory_id), {"status": MemoryStatus.ACTIVE.value},
                        )
                        if not ok:
                            error = "failed to activate pending memory"
                        else:
                            updated = self.db.get_memory_on_conn(conn, int(memory_id))
        except sqlite3.Error as exc:
            error = f"activate pending failed; transaction rolled back: {exc}"
        if error is not None:
            data: dict[str, Any] = {"error": error, "activated": False}
            if action_required and caller is not None:
                data.update({
                    "action_required": "confirm_new_workspace",
                    "next_call": {
                        "tool": "memory_govern",
                        "action": "confirm_pending_workspace",
                        "data": {
                            "memory_id": int(memory_id), "canonical": canonical,
                            "workspace": caller.workspace,
                        },
                        "authorization_required": True,
                    },
                })
            if caller is not None and caller.isolation == "strict":
                data.update(caller.response_fields())
            return self.db.state.response(
                data, ok=False,
                extra_warnings=list(caller.warnings) if caller is not None else [],
            )
        ok = updated is not None
        data = {"activated": ok, "record": updated}
        warnings: list[str] = list(caller.warnings) if caller is not None else []
        if caller is not None and caller.isolation == "strict":
            data.update(caller.response_fields())
        if ok:
            data["record"] = self.db.get_memory(int(memory_id))
            data["evidence_index"], data["semantic_conflict_check"] = self._post_commit(
                int(memory_id), data["record"], recheck_conflicts=False,
            )
        return self.db.state.response(data, extra_warnings=warnings)

    def memory_supersede(
        self,
        memory_id: int,
        reason: str,
        superseded_by: int | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Explicitly supersede a memory, bypassing the user-confirmed/locked
        protection that blocks ``memory_arbitrate``. Requires ``authorized=True``.

        Side effects: status -> superseded, protection_level -> normal, all open
        conflicts involving this memory are resolved, and an audit row is appended
        to the conflicts table (reason prefixed with ``USER-AUTHORIZED SUPERSEDE``).
        """
        authorized = self._is_truthy(authorized)
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to supersede a memory", "superseded": False},
                ok=False,
            )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if superseded_by is not None and int(superseded_by) == int(memory_id):
            return self.db.state.response(
                {"error": "superseded_by must identify a different active memory", "superseded": False},
                ok=False, extra_warnings=list(caller.warnings),
            )
        if caller.isolation == "strict" and self._get_memory_visible(int(memory_id), caller) is None:
            return self.db.state.response(
                forbidden_payload("memory", workspace=caller),
                ok=False,
                extra_warnings=list(caller.warnings),
            )
        if superseded_by is not None and caller.isolation == "strict" and self._get_memory_visible(int(superseded_by), caller) is None:
            return self.db.state.response(
                forbidden_payload("memory", workspace=caller, reason="replacement_workspace_acl"),
                ok=False,
                extra_warnings=list(caller.warnings),
            )
        resolved = 0
        updated: dict[str, Any] | None = None
        try:
            with self.db.write_transaction() as conn:
                memory = self.db.get_memory_on_conn(conn, int(memory_id))
                if not memory:
                    raise ValueError("memory id not found")
                if caller.isolation == "strict" and raw_workspace(memory) not in set(caller.scope_canonicals()):
                    raise PermissionError("forbidden")
                if memory.get("status") in {"superseded", "deleted"}:
                    raise ValueError(f"memory already {memory.get('status')}")
                if superseded_by is not None:
                    replacement = self.db.get_memory_on_conn(conn, int(superseded_by))
                    if not replacement:
                        raise ValueError("superseded_by memory id not found")
                    if caller.isolation == "strict" and raw_workspace(replacement) not in set(caller.scope_canonicals()):
                        raise PermissionError("replacement_workspace_acl")
                    if replacement.get("status") != "active":
                        raise ValueError(
                            f"superseded_by target is not active (status={replacement.get('status')}); pick a live replacement to avoid a broken chain"
                        )
                status_updated = self.db.update_memory_on_conn(
                    conn,
                    int(memory_id),
                    {"status": "superseded", "protection_level": ProtectionLevel.NORMAL.value},
                )
                if not status_updated:
                    raise ValueError("failed to update memory status")
                resolved = self.db.resolve_conflicts_for_on_conn(conn, int(memory_id))
                updated = self.db.get_memory_on_conn(conn, int(memory_id))
        except PermissionError as exc:
            return self.db.state.response(
                forbidden_payload(
                    "memory", workspace=caller,
                    reason="replacement_workspace_acl" if str(exc) == "replacement_workspace_acl" else "workspace_acl",
                ),
                ok=False, extra_warnings=list(caller.warnings),
            )
        except ValueError as exc:
            return self.db.state.response({"error": str(exc), "superseded": False}, ok=False)
        except sqlite3.Error as exc:
            return self.db.state.response(
                {
                    "error": f"supersede failed; transaction rolled back: {exc}",
                    "superseded": False,
                    "memory_id": int(memory_id),
                },
                ok=False,
            )
        except Exception as exc:
            return self.db.state.response(
                {
                    "error": f"supersede failed; transaction rolled back: {exc}",
                    "superseded": False,
                    "memory_id": int(memory_id),
                },
                ok=False,
            )
        resp = {
            "superseded": True,
            "memory_id": int(memory_id),
            "linked_conflicts_resolved": resolved,
            "record": updated,
        }
        if caller.isolation == "strict":
            resp.update(caller.response_fields())
        return self.db.state.response(resp, extra_warnings=list(caller.warnings))


    def _update_check_status(self) -> dict[str, Any]:
        if self._update_monitor is None:
            status = "disabled" if not self.settings.update_check_enabled else "not_started"
            return {"enabled": self.settings.update_check_enabled, "status": status, "current_version": __version__}
        return self._update_monitor.update_status()

    def memory_status(self, **_: Any) -> dict[str, Any]:
        evidence_status: dict[str, Any] = {
            "available": False,
        }
        try:
            evidence_status.update({"available": True, **self.db.evidence.coverage()})
            with self.db.connection() as conn:
                rows = conn.execute("SELECT key,value FROM migration_state").fetchall()
            evidence_status["migration"] = {str(row["key"]): str(row["value"]) for row in rows}
        except Exception as exc:
            evidence_status["error"] = str(exc)
        conflict_scan = self.db.conflict_scan_state()
        return self.db.state.response(
            {
                "arbiter_version": __version__,
                "db_path": str(self.settings.db_path),
                "backup_jsonl": str(self.settings.backup_jsonl),
                "sqlite_vec_available": self.db.state.sqlite_vec_available,
                "fts5_available": self.db.state.fts5_available,
                "sqlite_writable": self.db.state.sqlite_writable,
                "jsonl_backup_active": self.db.state.jsonl_backup_active,
                "client": self.current_client(),
                "agent_id": self.current_agent_id(),
                "workspace": self.settings.workspace,
                "config_warnings": self.settings.config_warnings,
                "embedding_configured": self._embedding_configured(),
                "embedding_auto_query": self.settings.embedding_auto_query,
                "embedding_auto_write": self.settings.embedding_auto_write,
                "local_text_evidence": evidence_status,
                "conflict_scan": conflict_scan,
                "conflict_scan_required": conflict_scan["required"],
                "local_text_index_worker": self._evidence_worker.status(),
                "isolation": self.settings.isolation,
                "workspace_recall": {
                    "admission_enabled": WORKSPACE_RECALL_ADMISSION,
                    "cutoff": WORKSPACE_RECALL_CUTOFF,
                    "min_name_len": WORKSPACE_MIN_NAME_LEN,
                    "weak_vector_weight": WORKSPACE_WEAK_VECTOR_WEIGHT,
                    "strict_scope_behavior": (
                        "guarded_vector_neighbors"
                        if WORKSPACE_RECALL_ADMISSION
                        else "exact_canonical"
                    ),
                },
                "tool_surface": {
                    "profile": "product",
                    "default_profile": "product",
                    "product_tools": ["memory", "memory_review", "memory_govern", "memory_repair"],
                },
                "update_check": self._update_check_status(),
                "vec_index_state": self.db.get_vec_index_state(),
                "semantic_conflict": self._semantic_status(
                    self._semantic_notice_workspace_scope(_.get("workspace")),
                ),
                # Policy echo is reduced to "is the current caller allowed":
                # the trusted request identity is evaluated, and the raw
                # allow/deny lists no longer leak through the status surface.
                "policy": {"caller_allowed": self._allowed()[0]},
            },
            extra_warnings=self.settings.config_warnings,
        )

    def memory_doctor_overview(self, deep: bool = False, **_: Any) -> dict[str, Any]:
        """Run a read-only health check and return a graded diagnostic report.

        Covers config integrity, evidence indexing, data consistency, and
        capacity. Each finding carries a severity and a
        fix_hint tailored to the current config.json. Read-only: never writes,
        never changes schema. ``deep=true`` additionally loads the GGUF model
        for a dimension probe (seconds-level cost); MCP reuses an
        already-loaded embedder at zero cost.
        """
        from ..doctor import doctor_overview_mcp, report_to_dict

        report = doctor_overview_mcp(
            self.db, self.settings, deep,
            embedder_probe=self._ensure_embedder,
            runtime_state=self.db.state,
        )
        if self._update_monitor is not None:
            self._update_monitor.record_doctor_run()
        data = report_to_dict(report)
        data["update_check"] = self._update_check_status()
        return self.db.state.response(data)

    def memory_set_entity(
        self,
        memory_id: int,
        entity: str | None = None,
        scope: str | None = None,
        clear: bool = False,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Set canonical metadata.entity/scope without creating content history."""
        authorized = self._is_truthy(authorized)
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer"}, ok=False)
        if not clear and not _canon_entity(entity):
            return self.db.state.response({"error": "entity is required unless clear=true"}, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation == "strict" and self._get_memory_visible(memory_id_int, caller) is None:
            return self.db.state.response(
                forbidden_payload("memory", workspace=caller),
                ok=False,
                extra_warnings=list(caller.warnings),
            )
        set_fields: dict[str, Any] = {}
        clear_fields: list[str] = []
        if clear:
            clear_fields.append("entity")
        else:
            set_fields["entity"] = _canon_entity(entity)
        if scope is not None:
            canonical_scope = _canon_scope(scope)
            if canonical_scope:
                set_fields["scope"] = canonical_scope
            else:
                clear_fields.append("scope")
        try:
            with self.db.write_transaction() as conn:
                current = self.db.get_memory_on_conn(conn, memory_id_int)
                if current is None:
                    result = {"outcome": "not_found", "memory_id": memory_id_int}
                elif caller.isolation == "strict" and raw_workspace(current) not in set(caller.scope_canonicals()):
                    result = {"outcome": "workspace_mismatch", "memory_id": memory_id_int}
                else:
                    result = self.db.update_metadata_fields_low_side_effect_on_conn(
                        conn, memory_id_int, set_fields=set_fields,
                        clear_fields=clear_fields, authorized=authorized,
                    )
        except sqlite3.Error:
            result = {"outcome": "error", "memory_id": memory_id_int}
        outcome = result.get("outcome")
        if outcome not in {"updated", "no_change"}:
            ok = False
            error = {
                "forbidden": "authorized=True is required for locked/user_confirmed memory",
                "not_found": "memory id not found",
                "not_active": "memory is not active",
                "unavailable": "database not available",
                "workspace_mismatch": "forbidden_strict_workspace",
            }.get(str(outcome), "entity update failed")
            if outcome == "workspace_mismatch":
                return self.db.state.response(
                    forbidden_payload("memory", workspace=caller), ok=False,
                    extra_warnings=list(caller.warnings),
                )
            return self.db.state.response({"error": error, **result}, ok=ok)
        data: dict[str, Any] = {
            "updated": outcome == "updated", "outcome": outcome,
            "memory_id": memory_id_int, "metadata": result.get("metadata"),
        }
        data["record"] = self.db.get_memory(memory_id_int)
        if outcome == "updated":
            data["evidence_index"], data["semantic_conflict_check"] = self._post_commit(
                memory_id_int, data["record"], recheck_conflicts=False,
            )
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

    def memory_list_entities(
        self, limit: int = 50, include_unassigned: bool = True, **_: Any,
    ) -> dict[str, Any]:
        try:
            limit_int = max(1, min(500, int(limit)))
        except (TypeError, ValueError):
            return self.db.state.response({"error": "limit must be an integer"}, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation != "strict":
            return self.db.state.response(
                self.db.list_entities(limit=limit_int, include_unassigned=bool(include_unassigned))
            )
        counts: dict[str, int] = {}
        sample: dict[str, int] = {}
        unassigned: list[int] = []
        total = 0
        with self.db.connection() as conn:
            scope_sql, scope_params = workspace_scope_sql(
                "COALESCE(NULLIF(workspace_canonical, ''), workspace)", caller.scope_canonicals(),
            )
            rows = conn.execute(
                f"SELECT id, metadata FROM memories WHERE status='active' AND {scope_sql} ORDER BY id",
                scope_params,
            ).fetchall()
        for row in rows:
            total += 1
            try:
                md = json.loads(row["metadata"] or "{}")
                if not isinstance(md, dict):
                    md = {}
            except Exception:
                md = {}
            entity = _canon_entity(md.get("entity"))
            if entity:
                counts[entity] = counts.get(entity, 0) + 1
                sample.setdefault(entity, int(row["id"]))
            elif include_unassigned:
                unassigned.append(int(row["id"]))
        data = {
            "entities": [
                {"entity": entity, "count": counts[entity], "sample_memory_id": sample[entity]}
                for entity in sorted(counts, key=lambda key: (-counts[key], key))[:limit_int]
            ],
            "distinct_entities": len(counts),
            "assigned_count": sum(counts.values()),
            "total_active": total,
            "unassigned_count": total - sum(counts.values()),
            "unassigned_ids": unassigned[:limit_int],
            **caller.response_fields(),
        }
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

    def memory_rebuild_evidence(
        self,
        memory_ids: list[int] | None = None,
        dry_run: bool = True,
        batch_size: int = 50,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            batch = max(1, min(500, int(batch_size)))
        except (TypeError, ValueError):
            return self.db.state.response({"error": "batch_size must be an integer"}, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        # Vec tables are created lazily at the first embedder load, so their
        # absence only counts as "missing" once the vec state says the index
        # should be live (state=ready with embedding configured) — a fresh
        # library in its pre-first-embed window is healthy, not broken.
        missing_vector_tables = (
            self.db.missing_vector_tables()
            if (
                self.settings.embedding_model_path is not None
                and self.db.get_vec_index_state().get("state") == "ready"
            )
            else []
        )
        table_repair: dict[str, Any] = {
            "required": bool(missing_vector_tables),
            "missing_tables": missing_vector_tables,
            "recreated": False,
            "warnings": [],
        }
        if not dry_run and self.settings.embedding_model_path is not None:
            recreated, repair_warnings = self.db.ensure_vector_tables_for_repair()
            table_repair.update({"recreated": recreated, "warnings": repair_warnings})
            if repair_warnings:
                return self.db.state.response(
                    {"error": "vector table repair failed", "vector_table_repair": table_repair},
                    ok=False, extra_warnings=list(caller.warnings) + repair_warnings,
                )
            if recreated:
                embedder, embedder_warnings = self._ensure_embedder()
                if embedder is None:
                    return self.db.state.response(
                        {"error": "embedding runtime unavailable after vector table repair",
                         "vector_table_repair": table_repair},
                        ok=False, extra_warnings=list(caller.warnings) + embedder_warnings,
                    )
                self.db.require_space_rebuild(
                    embedder.embedding_space_id, "derived vector tables were recreated",
                )
        mismatch_rebuild = False
        workspace_rebuild: dict[str, Any] = {"ok": True, "rebuilt": 0}
        if memory_ids is None:
            # Embedding-space mismatch is a whole-index condition: existing
            # rows may be healthy-looking but live in the old space, so the
            # rebuild must republish EVERYTHING (not just stale rows) before
            # the vec channel can be re-enabled. The pending set is keyed on
            # the evidence-id epoch, so republished rows drop out and repeated
            # calls paginate forward instead of re-selecting the first batch
            # forever. Strict-isolation callers only see their own workspace
            # and therefore cannot complete a global rebuild; they keep the
            # stale-only selection. The epoch mark itself is written only on
            # the execute path so dry runs stay side-effect-free.
            vec_state = self.db.get_vec_index_state()
            mismatch_rebuild = (
                (
                    bool(missing_vector_tables)
                    or (
                        vec_state.get("state") == "mismatch"
                        and vec_state.get("target_space_id") is not None
                    )
                )
                and caller.isolation != "strict"
            )
            if mismatch_rebuild:
                if not dry_run:
                    embedder, embedder_warnings = self._ensure_embedder()
                    if embedder is None:
                        return self.db.state.response(
                            {
                                "error": "embedding runtime unavailable for space rebuild",
                                "dry_run": False,
                                "queued": 0,
                                "failed": 0,
                                "results": [],
                                "workspace_vector_rebuild": {
                                    "ok": False,
                                    "error": "embedding_runtime_unavailable",
                                },
                                "vec_index_state": self.db.get_vec_index_state(),
                            },
                            ok=False,
                            extra_warnings=list(caller.warnings) + list(embedder_warnings),
                        )
                    workspace_rebuild = self.db.rebuild_workspace_canonical_vectors(
                        embedder, embedder.embedding_space_id,
                    )
                    if not workspace_rebuild.get("ok"):
                        return self.db.state.response(
                            {
                                "error": "workspace vector rebuild failed",
                                "workspace_vector_rebuild": workspace_rebuild,
                                "dry_run": False,
                                "vec_index_state": self.db.get_vec_index_state(),
                            },
                            ok=False,
                            extra_warnings=list(caller.warnings),
                        )
                    self.db.mark_space_rebuild_started()
                ids = (
                    self.db.stale_index_ids(batch)
                    if dry_run and missing_vector_tables
                    else self.db.space_rebuild_pending_ids(batch)
                )
                if not dry_run and not ids:
                    # Nothing pending: settle the flip now instead of waiting
                    # for an unrelated write to trigger the completion check.
                    embedder, _w = self._ensure_embedder()
                    if embedder is not None:
                        self.db.maybe_complete_space_rebuild(embedder.embedding_space_id)
            else:
                ids = self.db.stale_index_ids(
                    batch,
                    workspace=caller.scope_canonicals() if caller.isolation == "strict" else None,
                )
        else:
            try:
                # An explicitly enumerated repair set is not batch-capped:
                # silently truncating it under-repairs (the discovery path
                # below paginates with `batch` instead).
                requested = list(dict.fromkeys(int(value) for value in memory_ids))
            except (TypeError, ValueError):
                return self.db.state.response({"error": "memory_ids must contain integers"}, ok=False)
            ids = [mid for mid in requested if caller.isolation != "strict" or self._get_memory_visible(mid, caller)]
        if dry_run:
            return self.db.state.response(
                {
                    "dry_run": True,
                    "memory_ids": ids,
                    "count": len(ids),
                    "workspace_vectors": "rebuild_required" if mismatch_rebuild else "unchanged",
                    "vector_table_repair": table_repair,
                    "vec_index_state": self.db.get_vec_index_state(),
                },
                extra_warnings=list(caller.warnings),
            )
        results = [
            {"memory_id": mid, **self._post_commit(mid, self.db.get_memory(mid), recheck_conflicts=False)[0]}
            for mid in ids
        ]
        failed = sum(item.get("status") != "queued" for item in results)
        return self.db.state.response(
            {
                "dry_run": False,
                "queued": len(results) - failed,
                "failed": failed,
                "results": results,
                "workspace_vector_rebuild": (
                    workspace_rebuild if mismatch_rebuild else {"ok": True, "rebuilt": 0}
                ),
                "vector_table_repair": table_repair,
                # Surfaces a skipped flip (e.g. embedder unavailable with an
                # empty pending set): without it, "queued=0" is
                # indistinguishable from a settled mismatch->ready flip.
                "vec_index_state": self.db.get_vec_index_state(),
            },
            ok=failed == 0,
            extra_warnings=list(caller.warnings),
        )

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

    def memory_audit_summary(self, **_: Any) -> dict[str, Any]:
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        scoped = caller.isolation == "strict" or (
            caller.isolation == "none" and caller.source == "explicit"
        )
        if not scoped:
            summary = self.db.audit_summary()
            return self.db.state.response(summary)
        workspace_scope = (
            caller.scope_canonicals() if caller.isolation == "strict" else caller.canonical
        )
        with self.db.connection() as conn:
            scope_sql, scope_params = workspace_scope_sql(
                "COALESCE(NULLIF(workspace_canonical, ''), workspace)", workspace_scope,
            )
            mem_rows = conn.execute(
                "SELECT COALESCE(NULLIF(workspace_canonical, ''), workspace) AS workspace, "
                "COUNT(*) AS count, MIN(event_time) AS oldest, MAX(event_time) AS newest, source_type "
                f"FROM memories WHERE status != 'deleted' AND {scope_sql} "
                "GROUP BY workspace, source_type",
                scope_params,
            ).fetchall()
        def _empty_workspace_bucket() -> dict[str, Any]:
            return {
                "count": 0, "oldest": None, "newest": None,
                "open_conflicts": 0, "by_source_type": {},
            }

        # Pre-seed every admitted workspace. This preserves the single-canonical strict
        # response shape when the caller owns zero memories and makes each
        # admitted workspace explicit rather than synthesizing buckets only
        # when rows happen to exist.
        scope_names_value = (
            caller.scope_canonicals()
            if caller.isolation == "strict"
            else ((caller.canonical,) if caller.canonical else ())
        )
        workspaces: dict[str, dict[str, Any]] = {
            name: _empty_workspace_bucket() for name in scope_names_value
        }
        total_count = 0
        for row in mem_rows:
            ws_name = str(row["workspace"] or caller.canonical or "")
            bucket = workspaces.setdefault(ws_name, _empty_workspace_bucket())
            count = int(row["count"] or 0)
            bucket["count"] = int(bucket["count"] or 0) + count
            total_count += count
            if row["oldest"] is not None and (bucket["oldest"] is None or row["oldest"] < bucket["oldest"]):
                bucket["oldest"] = row["oldest"]
            if row["newest"] is not None and (bucket["newest"] is None or row["newest"] > bucket["newest"]):
                bucket["newest"] = row["newest"]
            if row["source_type"] is not None:
                source_type = str(row["source_type"])
                by_source_type = bucket["by_source_type"]
                by_source_type[source_type] = by_source_type.get(source_type, 0) + count
        if caller.isolation == "strict":
            conflicts = self.memory_list_conflicts(
                status="open", limit=10000, workspace=caller.workspace,
            ).get("data", {}).get("conflicts", [])
            conflicts += self.memory_list_conflicts(
                status="applying", limit=10000, workspace=caller.workspace,
            ).get("data", {}).get("conflicts", [])
        else:
            conflicts = self.db.list_conflicts(
                status="open", limit=10000, workspace=workspace_scope,
            )
            conflicts += self.db.list_conflicts(
                status="applying", limit=10000, workspace=workspace_scope,
            )
        for conflict in conflicts:
            ws_name = str(conflict.get("workspace_canonical") or "").strip()
            if ws_name in workspaces:
                bucket = workspaces[ws_name]
                bucket["open_conflicts"] = int(bucket["open_conflicts"] or 0) + 1
        summary = {
            "workspaces": workspaces,
            "total_memories": total_count,
            "total_open_conflicts": len(conflicts),
            **caller.response_fields(),
        }
        return self.db.state.response(summary, extra_warnings=list(caller.warnings))

    def memory_edit(
        self,
        memory_id: int,
        new_content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        reason: str = "",
        authorized: bool = False,
        tags_only: bool = False,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """In-place edit a memory's content or tags.

        Edit modes:
          * tags-only (v0.7.6): pass ``tags_only=True`` with
            ``add_tags``/``remove_tags`` to update tags without touching
            content, memory_history, version, or the evidence index
            (content is unchanged, so no re-embedding is needed).
            FTS is re-synced because tags are indexed in FTS5.
          * full replace: pass ``new_content`` (old_text/new_text must be empty)
          * partial replace: pass ``old_text`` + ``new_text`` for an exact
            substring substitution (new_content must be empty)

        Tags in content modes: ``add_tags``/``remove_tags`` also work in
        full/partial mode — they overlay on top of ``new_tags`` (if given)
        else the current tags, mirroring the tags-only path's
        order-preserving dedup (remove first, then add). ``new_tags`` alone
        is a full replace.

        Authorization (layered): normal records edit freely; ``locked`` /
        ``user_confirmed`` records require ``authorized=True`` (mirrors
        ``memory_supersede``). Records already superseded/deleted are rejected.
        """
        authorized = self._is_truthy(authorized)
        tags_only = self._is_truthy(tags_only)
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer", "edited": False}, ok=False)

        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation == "strict" and self._get_memory_visible(memory_id_int, caller) is None:
            return self.db.state.response(
                forbidden_payload("memory", workspace=caller),
                ok=False,
                extra_warnings=list(caller.warnings),
            )

        # ---- tags-only fast path (v0.7.6) ----
        if tags_only:
            tag_result: dict[str, Any]
            try:
                with self.db.write_transaction() as conn:
                    current = self.db.get_memory_on_conn(conn, memory_id_int)
                    if current is None:
                        tag_result = {"outcome": "not_found", "memory_id": memory_id_int}
                    elif caller.isolation == "strict" and raw_workspace(current) not in set(caller.scope_canonicals()):
                        tag_result = {"outcome": "workspace_mismatch", "memory_id": memory_id_int}
                    else:
                        tag_result = self.db.update_tags_low_side_effect(
                            memory_id_int,
                            add_tags=add_tags or [],
                            remove_tags=remove_tags or [],
                            authorized=authorized,
                            conn=conn,
                        )
            except sqlite3.Error:
                tag_result = {"outcome": "error", "memory_id": memory_id_int}
            outcome = tag_result.get("outcome")
            if outcome == "workspace_mismatch":
                return self.db.state.response(
                    forbidden_payload("memory", workspace=caller), ok=False,
                    extra_warnings=list(caller.warnings),
                )
            if outcome == "updated":
                updated_mem = self.db.get_memory(memory_id_int)
                data = {
                    "edited": True,
                    "tags_only": True,
                    "memory_id": memory_id_int,
                    "tags": tag_result.get("tags"),
                    "record": updated_mem,
                }
                return self.db.state.response(data)
            if outcome == "no_change":
                return self.db.state.response({
                    "edited": False,
                    "tags_only": True,
                    "already_completed": True,
                    "memory_id": memory_id_int,
                    "tags": tag_result.get("tags"),
                })
            if outcome == "forbidden":
                return self.db.state.response({
                    "error": (
                        f"memory is protected (protection_level={tag_result.get('protection_level')}, "
                        f"source_type={tag_result.get('source_type')}); authorized=True required to edit tags"
                    ),
                    "edited": False,
                }, ok=False)
            if outcome == "not_found":
                return self.db.state.response({"error": f"memory id {memory_id_int} not found", "edited": False}, ok=False)
            if outcome == "not_active":
                return self.db.state.response({
                    "error": f"memory is not active (status={tag_result.get('status')}); cannot edit tags",
                    "edited": False,
                }, ok=False)
            if outcome == "unavailable":
                return self.db.state.response({"error": "database not available", "edited": False}, ok=False)
            # outcome == "error"
            return self.db.state.response(
                {"error": "tags-only edit failed; transaction rolled back, no changes applied", "edited": False},
                ok=False,
            )

        # ---- full / partial content edit ----
        edit_result: dict[str, Any]
        expected_version_raw = _.get("expected_version")
        expected_hash = _.get("expected_content_hash") or _.get("content_hash")
        try:
            expected_version = int(expected_version_raw) if expected_version_raw is not None else None
        except (TypeError, ValueError):
            return self.db.state.response({"error": "expected_version must be an integer", "edited": False}, ok=False)
        try:
            with self.db.write_transaction() as conn:
                current = self.db.get_memory_on_conn(conn, memory_id_int)
                if current is None:
                    edit_result = {"outcome": "not_found", "memory_id": memory_id_int}
                elif caller.isolation == "strict" and raw_workspace(current) not in set(caller.scope_canonicals()):
                    edit_result = {"outcome": "workspace_mismatch", "memory_id": memory_id_int}
                else:
                    edit_result = self.db.edit_memory_intent(
                        memory_id_int,
                        new_content=new_content,
                        old_text=old_text,
                        new_text=new_text,
                        new_subject=new_subject,
                        new_tags=new_tags,
                        add_tags=add_tags,
                        remove_tags=remove_tags,
                        reason=reason or None,
                        authorized=authorized,
                        expected_version=expected_version,
                        expected_content_hash=str(expected_hash) if expected_hash is not None else None,
                        conn=conn,
                    )
        except sqlite3.Error:
            edit_result = {"outcome": "error", "memory_id": memory_id_int}
        outcome = edit_result.get("outcome")
        if outcome != "edited":
            if outcome == "workspace_mismatch":
                return self.db.state.response(
                    forbidden_payload("memory", workspace=caller), ok=False,
                    extra_warnings=list(caller.warnings),
                )
            if outcome == "not_found":
                error = "memory id not found"
            elif outcome == "not_active":
                status = edit_result.get("status")
                error = f"memory already {status}" if status in {"superseded", "deleted"} else f"memory is not active (status={status}); cannot edit"
            elif outcome == "forbidden":
                error = "authorized=True is required to edit a locked/user_confirmed memory"
            elif outcome == "stale_edit":
                error = edit_result.get("error") or f"stale_edit: {edit_result.get('reason') or 'current memory changed'}"
            elif outcome == "unavailable":
                error = "database not available"
            else:
                error = edit_result.get("error") or "edit failed (db not writable)"
            return self.db.state.response({"error": error, "edited": False, **edit_result}, ok=False)
        history_id = int(edit_result["history_id"])
        updated = edit_result.get("record") or self.db.get_memory(memory_id_int)
        data = {
            "edited": True,
            "memory_id": memory_id_int,
            "new_version": int(updated.get("version") or 1) if updated else None,
            "history_id": history_id,
            "record": updated,
        }
        data["record"] = self.db.get_memory(memory_id_int)
        data["evidence_index"], data["semantic_conflict_check"] = (
            self._post_commit(memory_id_int, data["record"], recheck_conflicts=True)
        )
        unresolved = self.db.conflicts.list_open_conflicts_for_memory_ids(
            [memory_id_int], include_applying=True,
        )
        if unresolved:
            # Content edits bump the version, staling the group's pinned member
            # snapshot (open) or the in-flight plan's expected_version
            # (applying). Prompt synchronously — same contract the search path
            # signals with — so the caller handles the group instead of
            # leaving it wedged. tags-only edits keep the version and stay
            # silent; the apply flow's own writes go through memory_govern.
            ids = ", ".join(f"#{int(group.get('id') or 0)}" for group in unresolved)
            data["attention_required"] = True
            data["action_required"] = "review_unresolved_conflicts"
            data["unresolved_conflicts"] = [
                {
                    "conflict_id": group.get("id"),
                    "revision": group.get("revision"),
                    "status": group.get("status"),
                    "conflict_point": group.get("conflict_point"),
                }
                for group in unresolved
            ]
            data["attention_summary"] = (
                f"edited memory is a member of unresolved conflict group(s) {ids}; "
                "this edit may stale their pinned member versions — review via "
                "memory_review(view='conflict_detail') and judge / replan / "
                "resolve as appropriate"
            )
        return self.db.state.response(data)

    def memory_history(self, memory_id: int, **_: Any) -> dict[str, Any]:
        """View the version-chain (historical snapshots) of a memory, newest
        version first. Read-only; does not modify any table.
        """
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        memory = self._get_memory_visible(int(memory_id), caller)
        if not memory:
            data: dict[str, Any] = {"error": "memory id not found"}
            if caller.isolation == "strict":
                data.update(caller.response_fields())
            return self.db.state.response(data, ok=False, extra_warnings=list(caller.warnings))
        history = self.db.list_history(int(memory_id))
        history_data: dict[str, Any] = {
            "memory_id": int(memory_id),
            "current_version": int(memory.get("version") or 1),
            "history": history,
            "count": len(history),
        }
        if caller.isolation == "strict":
            history_data.update(caller.response_fields())
        return self.db.state.response(history_data, extra_warnings=list(caller.warnings))

    def memory_cleanup_history(
        self,
        memory_id: int | None = None,
        older_than_days: int | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Delete historical snapshots from ``memory_history``.

        Scope:
          * ``memory_id`` set: clean only that memory's history
          * ``older_than_days`` set: clean only snapshots older than N days
          * both set: both filters apply
          * neither set (full cleanup): **requires ``authorized=True``** as an
            explicit confirmation gate

        SAFETY: this tool only ever deletes from memory_history. The memories
        table (active records) is never touched, regardless of arguments.
        """
        authorized = self._is_truthy(authorized)
        full_cleanup = memory_id is None and older_than_days is None
        if older_than_days is not None and int(older_than_days) < 0:
            return self.db.state.response(
                {"error": "older_than_days must be >= 0", "cleaned": 0},
                ok=False,
            )
        if full_cleanup and not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required for full history cleanup (no memory_id / older_than_days filter)", "cleaned": 0},
                ok=False,
            )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation == "strict":
            if memory_id is None:
                return self.db.state.response(
                    forbidden_payload("memory_history", workspace=caller, reason="workspace_scoped_cleanup_requires_memory_id"),
                    ok=False,
                    extra_warnings=list(caller.warnings),
                )
            if self._get_memory_visible(int(memory_id), caller) is None:
                return self.db.state.response(
                    forbidden_payload("memory_history", workspace=caller),
                    ok=False,
                    extra_warnings=list(caller.warnings),
                )
        try:
            with self.db.write_transaction() as conn:
                if caller.isolation == "strict" and memory_id is not None:
                    current = self.db.get_memory_on_conn(conn, int(memory_id))
                    if current is None or raw_workspace(current) not in set(caller.scope_canonicals()):
                        return self.db.state.response(
                            forbidden_payload("memory_history", workspace=caller),
                            ok=False, extra_warnings=list(caller.warnings),
                        )
                cleaned = self.db.cleanup_history(
                    memory_id=memory_id, older_than_days=older_than_days, conn=conn,
                )
        except sqlite3.Error as exc:
            return self.db.state.response(
                {"error": f"history cleanup failed; transaction rolled back: {exc}", "cleaned": 0},
                ok=False, extra_warnings=list(caller.warnings),
            )
        scope = "full" if full_cleanup else ("memory" if memory_id is not None else "by_age")
        data = {
            "cleaned": cleaned,
            "scope": scope,
            "memory_id": memory_id,
            "older_than_days": older_than_days,
        }
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

    @staticmethod
    def _replay_stage_done(value: Any) -> bool:
        return str(value or "") in {"complete", "skipped", "warning", "queued"}

    def _checkpoint_replay_stage(
        self,
        replay_key: str,
        stages: dict[str, str],
        stage: str,
        outcome: str,
        error_code: str | None = None,
    ) -> None:
        stages[stage] = outcome
        self.db.backup_replay.set_postprocess_state(
            replay_key, "pending", stages, error_code,
        )

    def _postprocess_replayed_memory(
        self,
        replay_key: str,
        memory_id: int,
        prior_stages: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, str], str | None, list[str]]:
        stages = {
            str(key): str(value)
            for key, value in (prior_stages or {}).items()
            if isinstance(key, str)
        }
        record = self.db.get_memory(memory_id) or {}
        result, _check = self._post_commit(memory_id, record, recheck_conflicts=False)
        outcome = str(result.get("status") or "unknown")
        if outcome == "queued":
            stages["evidence"] = "queued"
            status, retry_code = "complete", None
        elif outcome == "skipped" and not self._embedding_configured():
            stages["evidence"] = "skipped"
            status, retry_code = "complete", None
        else:
            stages["evidence"] = "retry_pending"
            status, retry_code = "pending", f"evidence_{outcome}"
        self.db.backup_replay.set_postprocess_state(
            replay_key, status, stages, retry_code,
        )
        return status, stages, retry_code, []

    def memory_replay_backup(
        self,
        dry_run: bool = True,
        authorized: bool = False,
        limit: int = 1_000,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        """Inspect or deterministically replay backup-only memory records."""
        dry_run = self._is_truthy(dry_run)
        authorized = self._is_truthy(authorized)
        requested_limit = max(1, min(int(limit), 10_000))
        page_limit = requested_limit if dry_run else min(requested_limit, 200)
        inspection = self.db.backup_replay.inspect(
            limit=page_limit, offset=max(0, int(offset)),
        )
        public = {key: value for key, value in inspection.items() if key != "entries"}
        if dry_run:
            return self.db.state.response({"dry_run": True, **public})
        if not authorized:
            return self.db.state.response(
                {
                    "error": "authorized=True is required to replay backup records",
                    "action_required": "ask_user_for_authorization",
                    "dry_run": False,
                    **public,
                },
                ok=False,
            )
        imported: list[dict[str, Any]] = []
        already_replayed: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        warnings: list[str] = []
        for entry in inspection["entries"]:
            if entry["status"] == "already_replayed" and entry.get("postprocess_status") in {"complete", "warning"}:
                already_replayed.append({
                    "replay_key": entry["replay_key"],
                    "memory_id": entry["memory_id"],
                    "postprocess_status": entry.get("postprocess_status"),
                    "postprocess_stages": entry.get("postprocess_stages") or {},
                    "postprocess_error_code": entry.get("postprocess_error_code"),
                })
                continue
            if entry["status"] not in {"importable", "already_replayed"}:
                conflicts.append({"replay_key": entry["replay_key"], "outcome": entry["status"]})
                continue
            try:
                replayed = self.db.backup_replay.replay_one(entry)
            except Exception as exc:
                conflicts.append({"replay_key": entry["replay_key"], "outcome": "error", "reason": str(exc)})
                continue
            outcome = replayed.get("outcome")
            needs_postprocess = outcome == "imported" or (
                outcome == "already_replayed"
                and replayed.get("postprocess_status") not in {"complete", "warning"}
            )
            if needs_postprocess:
                memory_id = int(replayed["memory_id"])
                if outcome == "imported":
                    receipt_result = {"replay_key": entry["replay_key"], "memory_id": memory_id, "postprocess_status": "pending"}
                    imported.append(receipt_result)
                else:
                    receipt_result = {"replay_key": entry["replay_key"], "memory_id": memory_id, "postprocess_status": replayed.get("postprocess_status")}
                    already_replayed.append(receipt_result)
                try:
                    final_status, stages, error_code, postprocess_warnings = (
                        self._postprocess_replayed_memory(
                            entry["replay_key"], memory_id,
                            replayed.get("postprocess_stages"),
                        )
                    )
                    receipt_result["postprocess_status"] = final_status
                    receipt_result["postprocess_stages"] = stages
                    if error_code is not None:
                        receipt_result["postprocess_error_code"] = error_code
                    warnings.extend(postprocess_warnings)
                except Exception as exc:
                    # Earlier stage checkpoints may already be durable. Keep
                    # them and only mark the receipt retryable here.
                    self.db.backup_replay.mark_postprocess_retry(
                        entry["replay_key"], "postprocess_exception",
                    )
                    receipt_result["postprocess_status"] = "pending"
                    receipt_result["postprocess_stages"] = replayed.get("postprocess_stages") or {}
                    receipt_result["postprocess_error_code"] = "postprocess_exception"
                    warnings.append(f"replayed memory {memory_id} committed; derived post-processing failed and will retry: {exc}")
            elif outcome == "already_replayed":
                already_replayed.append({"replay_key": entry["replay_key"], "memory_id": replayed.get("memory_id"), "postprocess_status": "complete"})
            else:
                conflicts.append({"replay_key": entry["replay_key"], "outcome": outcome})
        # Drain the evidence worker before responding (B-D2): replay
        # post-processing enqueues local-text indexing on the background
        # worker, so without the drain a "complete" receipt could be observed
        # before the evidence index is actually persisted.
        drained = self.wait_evidence_worker_drained(timeout=30.0)
        if not drained:
            warnings.append(
                "evidence worker did not drain within 30s; complete receipts may still have text indexing in flight"
            )
        return self.db.state.response(
            {
                "dry_run": False,
                "evidence_worker_drained": drained,
                "imported": imported,
                "imported_count": len(imported),
                "already_replayed": already_replayed,
                "already_replayed_count": len(already_replayed),
                "invalid_entries": inspection["invalid_entries"],
                "conflicts": conflicts,
                "offset": inspection["offset"],
                "next_offset": inspection["next_offset"],
                "has_more": inspection["has_more"],
                "processed": len(inspection["entries"]) + len(inspection["invalid_entries"]),
                "remaining": inspection["has_more"],
            },
            ok=not conflicts,
            extra_warnings=warnings,
        )
