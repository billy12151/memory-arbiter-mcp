"""Governance, edit, status, and maintenance operations for MemoryTools (Phase 4 extraction)."""
# mypy: disable-error-code=no-any-return
from __future__ import annotations

import json
import time
from typing import Any, Optional, TYPE_CHECKING

from .. import __version__
from ..arbitration import compare_memories
from ..conflict_judgments import ConflictJudgmentStore
from ..db import MemoryDB, _canon_entity, _canon_scope
from ..models import MemoryRecord, MemoryStatus, ProtectionLevel, SourceType

if TYPE_CHECKING:
    from ..tools import MemoryTools


class OperationsPipeline:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    @staticmethod
    def _compare_memories(*args: Any, **kwargs: Any) -> Any:
        # Preserve legacy patch seam for memory_arbiter.tools.compare_memories.
        from .. import tools as tools_mod
        return getattr(tools_mod, "compare_memories")(*args, **kwargs)

    def memory_arbitrate(self, left_id: int, right_id: int, mark_conflict: bool = True, authorized: bool = False, **_: Any) -> dict[str, Any]:
        if _.get("apply") is not None:
            return self.db.state.response(
                {"error": "the 'apply' parameter was renamed to 'authorized' in v0.8.5 and no longer takes effect; pass authorized=True to auto-supersede the non-protected loser", "applied": False},
                ok=False,
            )
        left = self.db.get_memory(int(left_id))
        right = self.db.get_memory(int(right_id))
        if not left or not right:
            return self.db.state.response({"error": "memory id not found"}, ok=False)
        comparison = self._compare_memories(left, right)
        conflict_id = None
        if mark_conflict:
            reason = "; ".join(comparison["reasons"])
            conflict_id = self.db.record_conflict(int(left_id), int(right_id), left.get("subject") or right.get("subject"), reason, comparison["winner_id"])
        applied = False
        if authorized and comparison["winner_id"] and comparison["loser_id"] and not comparison["manual_review"]:
            loser = self.db.get_memory(int(comparison["loser_id"]))
            if loser and loser.get("protection_level") != ProtectionLevel.LOCKED.value and loser.get("source_type") != SourceType.USER_CONFIRMED.value:
                applied = self.db.update_memory(int(comparison["loser_id"]), {"status": "superseded"})
                # v0.9.4: vec parent_status sync happens inside update_memory
                # (it writes both memories_vec + memory_sections_vec in the
                # same transaction as the status flip). The redundant
                # mark_vectors_for_memory call is gone — same value, twice.
        return self.db.state.response({"comparison": comparison, "conflict_id": conflict_id, "applied": applied})

    def _with_resolution_guidance(self, conflict: dict[str, Any]) -> dict[str, Any]:
        resolution_kind = conflict.get("resolution_kind") or conflict.get("judgment_resolution_kind")
        conflict_scope = conflict.get("conflict_scope") or conflict.get("judgment_conflict_scope")
        enriched = dict(conflict)
        enriched["resolution_kind"] = resolution_kind
        enriched["conflict_scope"] = conflict_scope
        enriched["recommended_resolution_action"] = ConflictJudgmentStore.resolution_action(resolution_kind)
        enriched["supersede_candidate"] = ConflictJudgmentStore.is_supersede_candidate(resolution_kind)
        if conflict.get("active_judgment_id") is not None:
            enriched["active_judgment"] = {
                "id": conflict.get("active_judgment_id"),
                "verdict": conflict.get("judgment_verdict"),
                "recommended_use": conflict.get("judgment_recommended_use"),
                "suggested_winner": conflict.get("judgment_suggested_winner"),
                "confidence_hint": conflict.get("judgment_confidence_hint"),
                "reason": conflict.get("judgment_reason"),
                "resolution_kind": resolution_kind,
                "conflict_scope": conflict_scope,
                "recommended_resolution_action": enriched["recommended_resolution_action"],
                "supersede_candidate": enriched["supersede_candidate"],
                "judge_type": conflict.get("judgment_judge_type"),
                "judge_ref": conflict.get("judgment_judge_ref"),
                "judged_at": conflict.get("judged_at"),
            }
        return enriched

    def memory_list_conflicts(self, status: str = "open", limit: int = 50, source: Optional[str] = None, **_: Any) -> dict[str, Any]:
        conflicts = [
            self._with_resolution_guidance(c)
            for c in self.db.list_conflicts(status=status, limit=int(limit), source=source)
        ]
        return self.db.state.response({"conflicts": conflicts, "count": len(conflicts)})

    def memory_record_conflict(
        self,
        left_id: int,
        right_id: int,
        reason: str,
        conflict_type: Optional[str] = None,
        conflict_point: Optional[str] = None,
        suggested_winner: Optional[int] = None,
        confidence_hint: Optional[str] = None,
        source: Optional[str] = None,
        refresh: bool = False,
        left_version: Optional[int] = None,
        right_version: Optional[int] = None,
        scan_prompt_version: Optional[str] = None,
        scan_model: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """v0.7.5/v0.7.6: persist a conflict with scan-enrichment fields.

        Pairs are canonicalised (left<right). Idempotent: if an open conflict
        on the same pair already exists, returns ``deduped`` without writing.
        Pass ``refresh=True`` to update the existing row's enrichment fields
        in place (returns ``refreshed``); use when the scan task re-runs LLM
        after a memory version or model change. The ``source`` field (e.g.
        ``"llm_informed"``) records whether the suggestion came from an LLM
        that read the content or from a metadata heuristic. ``conflict_type``
        can be ``contradiction``, ``evolution`` (same-topic change over time;
        not necessarily a whole-memory supersede), or other.
        """
        result = self.db.record_conflict_enriched(
            int(left_id), int(right_id),
            conflict_type=conflict_type,
            conflict_point=conflict_point,
            reason=reason,
            suggested_winner=int(suggested_winner) if suggested_winner is not None else None,
            confidence_hint=confidence_hint,
            source=source,
            refresh=refresh,
            left_version=left_version,
            right_version=right_version,
            scan_prompt_version=scan_prompt_version,
            scan_model=scan_model,
            detection_channel="scan",
        )
        return self.db.state.response(result)

    def memory_resolve_conflict(
        self,
        conflict_id: int,
        reason: str = "",
        status: str = "resolved",
        **_: Any,
    ) -> dict[str, Any]:
        """v0.7.5 (id=243): close a single open conflict by id (dismiss).

        ``status``: ``'resolved'`` (default; arbitrated, has winner) or
        ``'not_a_conflict'`` (v0.8.8; pair judged NOT a real conflict — advisory
        dismissal; write/search then skip it via Layer 0 until a version change
        invalidates it). Use ``'not_a_conflict'`` when the user confirms an
        advisory/duplicate hint was a false positive.

        Unlike ``memory_supersede`` (which resolves all conflicts touching a
        memory via ``resolve_conflicts_for``), this targets exactly one
        conflict row — used to dismiss a false positive without touching
        either memory.
        """
        result = self.db.resolve_conflict(int(conflict_id), reason=reason, status=status)
        return self.db.state.response(result)

    def memory_confirm(self, memory_id: int, source_ref: Optional[str] = None, confidence: float = 1.0, authorized: bool = False, **_: Any) -> dict[str, Any]:
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to confirm a memory", "confirmed": False},
                ok=False,
            )
        memory = self.db.get_memory(int(memory_id))
        if not memory:
            return self.db.state.response({"error": "memory id not found"}, ok=False)
        if memory.get("status") != "active":
            return self.db.state.response(
                {
                    "error": f"memory is not active (status={memory.get('status')}); cannot confirm inactive memory",
                    "confirmed": False,
                },
                ok=False,
            )
        metadata = dict(memory.get("metadata") or {})
        metadata["confirmed_from"] = source_ref or "manual"
        ok = self.db.update_memory(
            int(memory_id),
            {
                "source_type": SourceType.USER_CONFIRMED.value,
                "confidence": float(confidence),
                "protection_level": ProtectionLevel.LOCKED.value,
                "status": "active",
                "metadata": metadata,
            },
        )
        updated = self.db.get_memory(int(memory_id)) if ok else memory
        data: dict[str, Any] = {"confirmed": ok, "record": updated}
        warnings: list[str] = []
        if ok:
            structured = self._index_and_reconcile_claims(int(memory_id))
            data["realtime_conflict_check"] = structured["diagnostic"]
            data["claim_indexed"] = bool(structured["diagnostic"].get("claim_indexed"))
            data["claim_reconciled"] = bool(
                structured["diagnostic"].get("claim_reconciled")
            )
            warnings.extend(structured.get("warnings") or [])
            self._apply_structured_gate(data, structured)
        return self.db.state.response(data, extra_warnings=warnings)

    # ------------------------------------------------------------------
    #  Workspace alias governance (design 636 §8 / 637). User-authorized
    #  accept/reject/rename/migrate/confirm-pending.
    # ------------------------------------------------------------------
    def memory_accept_workspace_alias(
        self, alias: str, canonical: str, relation: str = "alias",
        reason: Optional[str] = None, source: str = "user",
        authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        """Confirm that `alias` is the same workspace as `canonical`.

        Future writes/queries with `alias` resolve straight to `canonical` — no
        vector/Qwen round-trip, no repeat prompt (636 §8). If the pair was
        previously rejected, pass authorized=true to deliberately reverse it.
        """
        ok, warnings = self.db.upsert_workspace_alias(
            alias, canonical, relation=str(relation or "alias"), status="confirmed",
            source=str(source or "user"), action="accept", judge_type="user",
            reason=reason, force=self._is_truthy(authorized),
        )
        return self.db.state.response(
            {"accepted": ok, "alias": alias, "canonical": canonical, "status": "confirmed"},
            ok=ok, extra_warnings=warnings,
        )

    def memory_reject_workspace_alias(
        self, alias: str, canonical: str, reason: Optional[str] = None,
        source: str = "user", **_: Any,
    ) -> dict[str, Any]:
        """Record that `alias` is NOT `canonical`.

        Suppresses this pair in future candidate ranking so it stops being
        proposed (636 §4,8).
        """
        ok, warnings = self.db.upsert_workspace_alias(
            alias, canonical, relation="unrelated", status="rejected",
            source=str(source or "user"), action="reject", judge_type="user",
            reason=reason,
        )
        return self.db.state.response(
            {"rejected": ok, "alias": alias, "canonical": canonical, "status": "rejected"},
            ok=ok, extra_warnings=warnings,
        )

    def memory_rename_workspace_canonical(
        self, old: str, new: str, reason: Optional[str] = None, **_: Any,
    ) -> dict[str, Any]:
        """Rename a canonical workspace everywhere (memories + registry + audit)."""
        updated, warnings = self.db.rename_workspace_canonical(old, new, judge_type="user", reason=reason)
        return self.db.state.response(
            {"renamed": True, "old": old, "new": new, "memories_updated": updated},
            ok=not warnings, extra_warnings=warnings,
        )

    def memory_migrate_workspace(
        self, reason: Optional[str] = None, **payload: Any,
    ) -> dict[str, Any]:
        """Bulk-move memories from one workspace to another; record the alias.

        `from`/`to` are reserved words so they arrive via **payload.
        """
        from_ws = str(payload.get("from") or "")
        to_ws = str(payload.get("to") or "")
        embedder, ensure_warnings = self._ensure_embedder()
        updated, warnings = self.db.migrate_workspace(
            from_ws, to_ws, judge_type="user", reason=reason, embedder=embedder,
        )
        # Only migrate's own warnings gate ok; embedder-init warnings (e.g. vec
        # disabled) are informational and flow through extra_warnings.
        return self.db.state.response(
            {"migrated": True, "from": from_ws, "to": to_ws, "memories_updated": updated},
            ok=not warnings, extra_warnings=list(ensure_warnings) + list(warnings),
        )

    def memory_confirm_pending_workspace(
        self, memory_id: int, canonical: str, reason: Optional[str] = None,
        authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        """Confirm a strict-blocked pending memory's workspace and activate it.

        Records a confirmed alias (raw workspace -> canonical), sets the
        memory's canonical, and flips status pending -> active. If the raw
        workspace was previously rejected as an alias of `canonical`, this call
        fails (ok=False) unless authorized=true — a rejection is a user
        decision and must not be silently reversed by a confirm-pending flow.
        """
        memory = self.db.get_memory(int(memory_id))
        if not memory:
            return self.db.state.response({"error": "memory id not found", "confirmed": False}, ok=False)
        raw_ws = memory.get("workspace") or ""
        warnings: list[str] = []
        ok_alias, alias_warnings = self.db.upsert_workspace_alias(
            raw_ws, canonical, relation="alias", status="confirmed",
            source="user", action="accept", judge_type="user", reason=reason,
            force=self._is_truthy(authorized),
        )
        warnings.extend(alias_warnings)
        # Point the memory at the confirmed canonical. update_memory's whitelist
        # doesn't include workspace_canonical (it would bypass claim_revision
        # semantics), so use the dedicated helper. Pass the embedder so the
        # canonical also gets its vec row (else strict-isolation KNN re-splits).
        embedder, ensure_warnings = self._ensure_embedder()
        warnings.extend(ensure_warnings)
        canonical_set, canonical_warnings = self.db.set_memory_workspace_canonical(
            int(memory_id), canonical, embedder,
        )
        warnings.extend(canonical_warnings)
        # Do NOT activate a pending memory if the canonical write failed — that
        # would leave the memory active while still pointing at the raw pending
        # workspace, defeating the point of confirmation.
        activated = False
        if canonical_set and memory.get("status") == MemoryStatus.PENDING.value:
            activated = self.db.update_memory(int(memory_id), {"status": MemoryStatus.ACTIVE.value})
        updated = self.db.get_memory(int(memory_id))
        confirmed_all = ok_alias and canonical_set
        data = {
            "confirmed": confirmed_all,
            "activated": activated,
            "canonical": canonical,
            "record": updated,
        }
        if activated:
            data["semantic_conflict_check"] = self._enqueue_semantic_conflict_check(int(memory_id), updated or {})
        # Response ok reflects the FULL operation, not just the alias step.
        return self.db.state.response(data, ok=confirmed_all, extra_warnings=warnings)

    def memory_activate(
        self, memory_id: int, authorized: bool = False, **_: Any,
    ) -> dict[str, Any]:
        """Activate a pending memory blocked by strict workspace isolation.

        strict isolation writes brand-new workspaces as status=pending (excluded
        from active recall) until the user confirms the workspace name. This
        flips it to active — without the trust/protection promotion that
        memory_confirm applies. Requires authorized=true.
        """
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to activate a pending memory", "activated": False},
                ok=False,
            )
        memory = self.db.get_memory(int(memory_id))
        if not memory:
            return self.db.state.response({"error": "memory id not found", "activated": False}, ok=False)
        if memory.get("status") != MemoryStatus.PENDING.value:
            return self.db.state.response(
                {
                    "error": f"memory is not pending (status={memory.get('status')}); only pending memories can be activated",
                    "activated": False,
                },
                ok=False,
            )
        ok = self.db.update_memory(int(memory_id), {"status": MemoryStatus.ACTIVE.value})
        updated = self.db.get_memory(int(memory_id)) if ok else memory
        data = {"activated": ok, "record": updated}
        if ok:
            data["semantic_conflict_check"] = self._enqueue_semantic_conflict_check(int(memory_id), updated or {})
        return self.db.state.response(data)

    def memory_supersede(
        self,
        memory_id: int,
        reason: str,
        superseded_by: Optional[int] = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Explicitly supersede a memory, bypassing the user-confirmed/locked
        protection that blocks ``memory_arbitrate``. Requires ``authorized=True``.

        Side effects: status -> superseded, protection_level -> normal, all open
        conflicts involving this memory are resolved, and an audit row is appended
        to the conflicts table (reason prefixed with ``USER-AUTHORIZED SUPERSEDE``).
        """
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to supersede a memory", "superseded": False},
                ok=False,
            )
        memory = self.db.get_memory(int(memory_id))
        if not memory:
            return self.db.state.response({"error": "memory id not found", "superseded": False}, ok=False)
        if memory.get("status") in {"superseded", "deleted"}:
            return self.db.state.response(
                {"error": f"memory already {memory.get('status')}", "superseded": False},
                ok=False,
            )
        if superseded_by is not None:
            replacement = self.db.get_memory(int(superseded_by))
            if not replacement:
                return self.db.state.response(
                    {"error": "superseded_by memory id not found", "superseded": False},
                    ok=False,
                )
            # Guard against supersede-chain breakage: starting in v0.2.6,
            # memory_search filters out superseded records by default. If the
            # replacement target is itself deleted/superseded, the new default
            # would leave the chain pointing at a record that search can't see
            # — the user would lose both the old and the new view. Reject early
            # with an explicit error so the caller picks a live replacement.
            if replacement.get("status") != "active":
                return self.db.state.response(
                    {"error": f"superseded_by target is not active (status={replacement.get('status')}); pick a live replacement to avoid a broken chain", "superseded": False},
                    ok=False,
                )

        status_updated = self.db.update_memory(
            int(memory_id),
            {"status": "superseded", "protection_level": ProtectionLevel.NORMAL.value},
        )
        if not status_updated:
            return self.db.state.response(
                {
                    "error": "failed to update memory status",
                    "superseded": False,
                    "memory_id": int(memory_id),
                },
                ok=False,
            )
        # v0.9.4: vec parent_status sync happens inside update_memory (it
        # writes both memories_vec + memory_sections_vec in the same
        # transaction as the status flip, retaining vectors for audit recall
        # via memory_search_expired). Content/FTS kept for audit; vectors are
        # a derivative and can be recomputed from content if ever needed.
        resolved = self.db.resolve_conflicts_for(int(memory_id))
        audit_reason = f"USER-AUTHORIZED SUPERSEDE: {reason}"
        conflict_id = self.db.record_conflict(
            int(memory_id),
            int(superseded_by) if superseded_by is not None else int(memory_id),
            memory.get("subject"),
            audit_reason,
            int(superseded_by) if superseded_by is not None else None,
            status="resolved",
        )
        updated = self.db.get_memory(int(memory_id))
        resp = {
            "superseded": True,
            "memory_id": int(memory_id),
            "linked_conflicts_resolved": resolved,
            "conflict_id": conflict_id,
            "record": updated,
        }
        return self.db.state.response(resp)

    def _split_capability(self, vec_state: dict[str, Any]) -> dict[str, Any]:
        """v0.8 §6.5: whether the server can split, and why/why not."""
        if vec_state.get("state") == "ready":
            return {"available": True, "reason": "vec_ready"}
        if self.db.state.sqlite_vec_available and not self._embedding_configured():
            return {"available": False, "reason": "embedder_unavailable"}
        return {"available": False, "reason": "vec_not_ready"}

    def _update_check_status(self) -> dict[str, Any]:
        if self._update_monitor is None:
            status = "disabled" if not self.settings.update_check_enabled else "not_started"
            return {"enabled": self.settings.update_check_enabled, "status": status, "current_version": __version__}
        return self._update_monitor.update_status()

    def memory_status(self, **_: Any) -> dict[str, Any]:
        vec_state = self.db.get_vec_index_state()
        return self.db.state.response(
            {
                "arbiter_version": __version__,
                "db_path": str(self.settings.db_path),
                "backup_jsonl": str(self.settings.backup_jsonl),
                "sqlite_vec_available": self.db.state.sqlite_vec_available,
                "fts5_available": self.db.state.fts5_available,
                "sqlite_writable": self.db.state.sqlite_writable,
                "jsonl_backup_active": self.db.state.jsonl_backup_active,
                "client": self.settings.client,
                "agent_id": self.settings.agent_id,
                "workspace": self.settings.workspace,
                "config_warnings": self.settings.config_warnings,
                "embedding_configured": self._embedding_configured(),
                "embedding_auto_query": self.settings.embedding_auto_query,
                "embedding_auto_write": self.settings.embedding_auto_write,
                "structured_claim_mode": self.settings.structured_claim_mode,
                "isolation": self.settings.isolation,
                "tool_surface": {
                    "profile": self.settings.tool_profile,
                    "default_profile": "product",
                    "product_tools": ["memory", "memory_review", "memory_govern", "memory_repair"],
                    "legacy_tools_hidden_by_default": True,
                    "legacy_restore_hint": "Set MEMORY_ARBITER_TOOL_PROFILE=legacy_full for the old low-level MCP tool surface.",
                },
                "update_check": self._update_check_status(),
                "split_reindex_pending": self._split_worker.pending_ids(),
                # v0.8: split capability is bound to vec readiness, not a toggle.
                "split_capability": self._split_capability(vec_state),
                "vec_index_state": vec_state,
                "semantic_conflict": self._semantic_status(),
                "policy": {
                    "client_defaults": self.settings.policy.client_defaults,
                    "default_enabled": self.settings.policy.default_enabled,
                    "allow_agents": self.settings.policy.allow_agents,
                    "deny_agents": self.settings.policy.deny_agents,
                },
            },
            extra_warnings=self.settings.config_warnings,
        )

    def memory_doctor_overview(self, deep: bool = False, **_: Any) -> dict[str, Any]:
        """Run a read-only health check and return a graded diagnostic report.

        Covers config integrity, the vector-enablement chain, split, data
        consistency, and capacity. Each finding carries a severity and a
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
            inflight_ids=set(self._split_worker.pending_ids()),
        )
        if self._update_monitor is not None:
            self._update_monitor.record_doctor_run()
        data = report_to_dict(report)
        data["update_check"] = self._update_check_status()
        return self.db.state.response(data)

    def memory_set_entity(
        self,
        memory_id: int,
        entity: Optional[str] = None,
        scope: Optional[str] = None,
        clear: bool = False,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Set canonical metadata.entity/scope without creating content history."""
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer"}, ok=False)
        if not clear and not _canon_entity(entity):
            return self.db.state.response({"error": "entity is required unless clear=true"}, ok=False)
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
        result = self.db.update_metadata_fields_low_side_effect(
            memory_id_int, set_fields=set_fields, clear_fields=clear_fields,
            authorized=authorized,
        )
        outcome = result.get("outcome")
        if outcome not in {"updated", "no_change"}:
            ok = False
            error = {
                "forbidden": "authorized=True is required for locked/user_confirmed memory",
                "not_found": "memory id not found",
                "not_active": "memory is not active",
                "unavailable": "database not available",
            }.get(str(outcome), "entity update failed")
            return self.db.state.response({"error": error, **result}, ok=ok)
        warnings: list[str] = []
        data: dict[str, Any] = {
            "updated": outcome == "updated", "outcome": outcome,
            "memory_id": memory_id_int, "metadata": result.get("metadata"),
        }
        if result.get("claim_semantics_changed"):
            structured = self._index_and_reconcile_claims(memory_id_int)
            data["realtime_conflict_check"] = structured["diagnostic"]
            data["claim_indexed"] = bool(structured["diagnostic"].get("claim_indexed"))
            data["claim_reconciled"] = bool(
                structured["diagnostic"].get("claim_reconciled")
            )
            warnings.extend(structured.get("warnings") or [])
            self._apply_structured_gate(data, structured)
        data["record"] = self.db.get_memory(memory_id_int)
        return self.db.state.response(data, extra_warnings=warnings)

    def memory_list_entities(
        self, limit: int = 50, include_unassigned: bool = True, **_: Any,
    ) -> dict[str, Any]:
        try:
            limit_int = max(1, min(500, int(limit)))
        except (TypeError, ValueError):
            return self.db.state.response({"error": "limit must be an integer"}, ok=False)
        return self.db.state.response(
            self.db.list_entities(limit=limit_int, include_unassigned=bool(include_unassigned))
        )

    def memory_rebuild_claims(
        self,
        memory_ids: Optional[list[int]] = None,
        dry_run: bool = True,
        batch_size: int = 50,
        **_: Any,
    ) -> dict[str, Any]:
        """Idempotently rebuild the deterministic claim index."""
        if self.settings.structured_claim_mode == "off" and not dry_run:
            return self.db.state.response(
                {"error": "structured_claim_mode=off; enable beta_all before rebuilding"}, ok=False,
            )
        try:
            batch = max(1, min(500, int(batch_size)))
        except (TypeError, ValueError):
            return self.db.state.response({"error": "batch_size must be an integer"}, ok=False)
        if memory_ids is None:
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT id FROM memories WHERE status='active' "
                    "AND (claims_indexed_revision IS NULL "
                    "OR claims_indexed_revision<>claim_revision "
                    "OR claims_reconciled_revision IS NULL "
                    "OR claims_reconciled_revision<>claim_revision) "
                    "ORDER BY id LIMIT ?", (batch,)
                ).fetchall()
                ids = [int(row["id"]) for row in rows]
        else:
            try:
                ids = list(dict.fromkeys(int(value) for value in memory_ids))[:batch]
            except (TypeError, ValueError):
                return self.db.state.response({"error": "memory_ids must contain integers"}, ok=False)
        if dry_run:
            return self.db.state.response({"dry_run": True, "memory_ids": ids, "count": len(ids)})
        results = []
        for memory_id in ids:
            results.append({"memory_id": memory_id, **self._index_and_reconcile_claims(memory_id)})
        failed = sum(
            1 for item in results
            if not item["diagnostic"].get("claim_indexed")
            or not item["diagnostic"].get("claim_reconciled")
        )
        return self.db.state.response({
            "dry_run": False, "processed": len(results), "failed": failed, "results": results,
        }, ok=failed == 0)

    def memory_submit_conflict_judgment(
        self,
        conflict_id: int,
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        confidence_hint: Optional[str],
        reason: str,
        affects_current_output: bool,
        usage_context: str,
        judge_ref: Optional[str] = None,
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            conflict_id_int = int(conflict_id)
            left_version = int(expected_left_version)
            right_version = int(expected_right_version)
            left_revision = int(expected_left_claim_revision)
            right_revision = int(expected_right_claim_revision)
            winner = int(suggested_winner) if suggested_winner is not None else None
        except (TypeError, ValueError):
            return self.db.state.response(
                {"error": "conflict id, snapshot pins, and winner must be integers"}, ok=False,
            )
        request_before = self.db.judgments.build_conflict_judgment_request(conflict_id_int)
        result = self.db.judgments.submit_conflict_judgment(
            conflict_id_int, left_version, right_version, left_revision, right_revision,
            verdict, recommended_use, winner,
            confidence_hint, reason, bool(affects_current_output), usage_context,
            judge_ref=judge_ref,
            resolution_kind=resolution_kind,
            conflict_scope=conflict_scope,
        )
        if result.get("outcome") == "judged":
            event = "user_escalated" if result.get("user_action_required") else "llm_assessed"
            ids = []
            if request_before:
                ids = [
                    int(request_before["left"]["id"]),
                    int(request_before["right"]["id"]),
                ]
            self.db.log_attention(trigger="structured_claim", source=event, memory_ids=ids)
        return self.db.state.response(result, ok=result.get("outcome") == "judged")

    def memory_correct_conflict_judgment(
        self,
        conflict_id: int,
        verdict: str,
        recommended_use: str,
        suggested_winner: Optional[int],
        reason: str,
        expected_judgment_id: int,
        expected_left_version: int,
        expected_right_version: int,
        expected_left_claim_revision: int,
        expected_right_claim_revision: int,
        authorized: bool = False,
        judge_ref: Optional[str] = None,
        resolution_kind: Optional[str] = None,
        conflict_scope: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required for human judgment correction"}, ok=False,
            )
        try:
            conflict_id_int = int(conflict_id)
            winner = int(suggested_winner) if suggested_winner is not None else None
            judgment_id = int(expected_judgment_id)
            left_version = int(expected_left_version)
            right_version = int(expected_right_version)
            left_revision = int(expected_left_claim_revision)
            right_revision = int(expected_right_claim_revision)
        except (TypeError, ValueError):
            return self.db.state.response(
                {"error": "conflict id, judgment id, snapshot pins, and winner must be integers"},
                ok=False,
            )
        result = self.db.judgments.correct_conflict_judgment(
            conflict_id_int, verdict, recommended_use, winner,
            reason, judgment_id, left_version, right_version,
            left_revision, right_revision, judge_ref=judge_ref,
            resolution_kind=resolution_kind,
            conflict_scope=conflict_scope,
        )
        return self.db.state.response(result, ok=result.get("outcome") == "corrected")

    def memory_list_conflict_judgments(self, conflict_id: int, **_: Any) -> dict[str, Any]:
        try:
            conflict_id_int = int(conflict_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "conflict_id must be an integer"}, ok=False)
        rows = self.db.judgments.list_conflict_judgments(conflict_id_int)
        return self.db.state.response({
            "conflict_id": conflict_id_int, "judgments": rows, "count": len(rows),
        })

    def memory_audit_summary(self, **_: Any) -> dict[str, Any]:
        summary = self.db.audit_summary()
        return self.db.state.response(summary)

    def memory_edit(
        self,
        memory_id: int,
        new_content: Optional[str] = None,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        new_subject: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
        reason: str = "",
        authorized: bool = False,
        tags_only: bool = False,
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """In-place edit a memory's content or tags.

        Edit modes:
          * tags-only (v0.7.6): pass ``tags_only=True`` with
            ``add_tags``/``remove_tags`` to update tags without touching
            content, memory_history, version, embeddings, or sections.
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
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer", "edited": False}, ok=False)

        # ---- tags-only fast path (v0.7.6) ----
        if tags_only:
            result = self.db.update_tags_low_side_effect(
                memory_id_int,
                add_tags=add_tags or [],
                remove_tags=remove_tags or [],
                authorized=authorized,
            )
            outcome = result.get("outcome")
            if outcome == "updated":
                updated_mem = self.db.get_memory(memory_id_int)
                data = {
                    "edited": True,
                    "tags_only": True,
                    "memory_id": memory_id_int,
                    "tags": result.get("tags"),
                    "record": updated_mem,
                }
                claim_warnings: list[str] = []
                if result.get("claim_semantics_changed"):
                    structured = self._index_and_reconcile_claims(memory_id_int)
                    data["realtime_conflict_check"] = structured["diagnostic"]
                    data["claim_indexed"] = bool(structured["diagnostic"].get("claim_indexed"))
                    data["claim_reconciled"] = bool(
                        structured["diagnostic"].get("claim_reconciled")
                    )
                    claim_warnings.extend(structured.get("warnings") or [])
                    self._apply_structured_gate(data, structured)
                    data["semantic_conflict_check"] = self._enqueue_semantic_conflict_check(memory_id_int, updated_mem or {})
                else:
                    data["semantic_conflict_check"] = {
                        "status": "skipped",
                        "reason": "tags_only_no_semantic_change",
                    }
                return self.db.state.response(data, extra_warnings=claim_warnings)
            if outcome == "no_change":
                return self.db.state.response({
                    "edited": False,
                    "tags_only": True,
                    "already_completed": True,
                    "memory_id": memory_id_int,
                    "tags": result.get("tags"),
                })
            if outcome == "forbidden":
                return self.db.state.response({
                    "error": (
                        f"memory is protected (protection_level={result.get('protection_level')}, "
                        f"source_type={result.get('source_type')}); authorized=True required to edit tags"
                    ),
                    "edited": False,
                }, ok=False)
            if outcome == "not_found":
                return self.db.state.response({"error": f"memory id {memory_id_int} not found", "edited": False}, ok=False)
            if outcome == "not_active":
                return self.db.state.response({
                    "error": f"memory is not active (status={result.get('status')}); cannot edit tags",
                    "edited": False,
                }, ok=False)
            if outcome == "unavailable":
                return self.db.state.response({"error": "database not available", "edited": False}, ok=False)
            # outcome == "error"
            return self.db.state.response(
                {"error": "tags-only edit failed; transaction rolled back, no changes applied", "edited": False},
                ok=False,
            )

        # ---- full / partial content edit (existing path) ----
        memory = self.db.get_memory(memory_id_int)
        if not memory:
            return self.db.state.response({"error": "memory id not found", "edited": False}, ok=False)
        if memory.get("status") in {"superseded", "deleted"}:
            return self.db.state.response(
                {"error": f"memory already {memory.get('status')}", "edited": False},
                ok=False,
            )
        is_protected = (
            memory.get("protection_level") == ProtectionLevel.LOCKED.value
            or memory.get("source_type") == SourceType.USER_CONFIRMED.value
        )
        if is_protected and not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to edit a locked/user_confirmed memory", "edited": False},
                ok=False,
            )
        # Resolve the resulting content from the two edit modes.
        current_content = memory.get("content") or ""
        if new_content is not None and (old_text or new_text):
            return self.db.state.response(
                {"error": "pass either new_content (full replace) or old_text+new_text (partial), not both", "edited": False},
                ok=False,
            )
        if new_content is not None and not str(new_content).strip():
            return self.db.state.response(
                {"error": "new_content is empty; refusing to wipe memory content (use memory_supersede to retire it, or pass real content)", "edited": False},
                ok=False,
            )
        if new_subject is not None and not str(new_subject).strip():
            return self.db.state.response(
                {"error": "new_subject is empty; refusing to wipe subject (pass None to keep current)", "edited": False},
                ok=False,
            )
        if new_content is not None:
            resolved_content = new_content
        elif old_text is not None and new_text is not None:
            if old_text not in current_content:
                return self.db.state.response(
                    {"error": "old_text not found in current content", "edited": False},
                    ok=False,
                )
            resolved_content = current_content.replace(old_text, new_text, 1)
        else:
            return self.db.state.response(
                {"error": "provide new_content for full replace, or old_text+new_text for partial replace, or tags_only=true", "edited": False},
                ok=False,
            )
        # Resolve tags: new_tags (full replace) + add_tags/remove_tags (delta).
        # add/remove overlay on top of new_tags (if given) else current tags,
        # mirroring update_tags_low_side_effect's order-preserving dedup
        # (remove first, then add). Previously the content path dropped
        # add_tags/remove_tags entirely (silent no-op) — fixed.
        current_tags_raw = memory.get("tags") or []
        if isinstance(current_tags_raw, str):
            try:
                current_tags_raw = json.loads(current_tags_raw)
            except (json.JSONDecodeError, ValueError):
                current_tags_raw = []
        resolved_new_tags: Optional[list[str]]
        if new_tags is not None:
            resolved_new_tags = list(new_tags)
        else:
            resolved_new_tags = list(current_tags_raw)
        if add_tags or remove_tags:
            tag_set = set(resolved_new_tags)
            for t in (remove_tags or []):
                if t in tag_set:
                    tag_set.discard(t)
                    resolved_new_tags = [x for x in resolved_new_tags if x != t]
            for t in (add_tags or []):
                if t not in tag_set:
                    tag_set.add(t)
                    resolved_new_tags.append(t)
        else:
            # No delta: pass new_tags through as-is, or None to keep current.
            resolved_new_tags = new_tags

        history_id = self.db.edit_memory(
            memory_id_int,
            resolved_content,
            new_subject=new_subject,
            new_tags=resolved_new_tags,
            reason=reason or None,
        )
        if history_id is None:
            return self.db.state.response({"error": "edit failed (db not writable)", "edited": False}, ok=False)
        updated = self.db.get_memory(memory_id_int)
        embedding_warnings: list[str] = []
        embedding_stored: Optional[bool] = None
        if self.settings.embedding_auto_write and self._embedding_configured():
            embedding_stored = False
            embedder, ensure_warnings = self._ensure_embedder()
            embedding_warnings.extend(ensure_warnings)
            if embedder is None:
                _deleted, delete_warnings = self.db.delete_embedding(memory_id_int)
                embedding_warnings.extend(delete_warnings)
                embedding_warnings.append("re-embedding on edit skipped because embedder unavailable; deleted stale embedding to avoid dirty recall.")
            elif updated is not None:
                try:
                    embedding_result = embedder.embed_text(
                        prefix=updated.get("subject") or "",
                        body=updated.get("content") or "",
                    )
                    if not embedding_result.embedding:
                        raise RuntimeError(
                            f"encode returned empty embedding: {getattr(embedder, 'last_encode_error', None) or 'unknown'}"
                        )
                    embedding_stored, store_warnings = self.db.store_embedding(memory_id_int, embedding_result.embedding)
                    embedding_warnings.extend(store_warnings)
                    if not embedding_stored:
                        _deleted, delete_warnings = self.db.delete_embedding(memory_id_int)
                        embedding_warnings.extend(delete_warnings)
                        embedding_warnings.append("re-embedding on edit failed; deleted stale embedding to avoid dirty recall.")
                except Exception as exc:
                    _deleted, delete_warnings = self.db.delete_embedding(memory_id_int)
                    embedding_warnings.extend(delete_warnings)
                    embedding_warnings.append(f"re-embedding on edit failed: {exc}; deleted stale embedding to avoid dirty recall.")
        data = {
            "edited": True,
            "memory_id": memory_id_int,
            "new_version": int(updated.get("version") or 1) if updated else None,
            "history_id": history_id,
            "record": updated,
        }
        if embedding_stored is not None:
            data["embedding_stored"] = embedding_stored
        # v0.8: a content edit re-runs the post-write split decision. The DB
        # layer already cleared old sections + reset split_status to NULL and
        # bumped split_revision (db.edit_memory). If the new content has a
        # publishable rule plan it is re-split synchronously (revision bumps
        # again); otherwise a split_request is returned. tags-only edits
        # return early above and never reach here.
        split_block, split_request, split_warnings = self._after_write_split(memory_id_int)
        data["split"] = split_block
        if split_request is not None:
            data["split_request"] = split_request
        embedding_warnings.extend(split_warnings)
        structured = self._index_and_reconcile_claims(memory_id_int)
        data["realtime_conflict_check"] = structured["diagnostic"]
        data["claim_indexed"] = bool(structured["diagnostic"].get("claim_indexed"))
        data["claim_reconciled"] = bool(
            structured["diagnostic"].get("claim_reconciled")
        )
        embedding_warnings.extend(structured.get("warnings") or [])
        self._apply_structured_gate(data, structured)
        data["record"] = self.db.get_memory(memory_id_int)
        data["semantic_conflict_check"] = self._enqueue_semantic_conflict_check(memory_id_int, data["record"] or {})
        return self.db.state.response(
            data,
            extra_warnings=embedding_warnings,
        )

    def memory_history(self, memory_id: int, **_: Any) -> dict[str, Any]:
        """View the version-chain (historical snapshots) of a memory, newest
        version first. Read-only; does not modify any table.
        """
        memory = self.db.get_memory(int(memory_id))
        if not memory:
            return self.db.state.response({"error": "memory id not found"}, ok=False)
        history = self.db.list_history(int(memory_id))
        return self.db.state.response(
            {
                "memory_id": int(memory_id),
                "current_version": int(memory.get("version") or 1),
                "history": history,
                "count": len(history),
            }
        )

    def memory_cleanup_history(
        self,
        memory_id: Optional[int] = None,
        older_than_days: Optional[int] = None,
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
        cleaned = self.db.cleanup_history(memory_id=memory_id, older_than_days=older_than_days)
        scope = "full" if full_cleanup else ("memory" if memory_id is not None else "by_age")
        return self.db.state.response(
            {
                "cleaned": cleaned,
                "scope": scope,
                "memory_id": memory_id,
                "older_than_days": older_than_days,
            }
        )

    def memory_cleanup_inactive_vectors(
        self,
        dry_run: bool = True,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Purge orphan vec rows (v0.9.4) and optionally resync parent_status.

        v0.9.4: ``_purge_inactive_vectors`` only removes true orphans (vec rows
        whose parent memory/section row no longer exists in the DB). Superseded
        vectors are KEPT for ``memory_search_expired`` vec-hybrid recall.

        Resync mode: if ``parent_status`` mismatches are detected (vec.status !=
        memories.status), ``dry_run`` reports them. Use
        ``memory_resync_vec_parent_status(dry_run=False)`` to repair drift
        without authorization; cleanup execution requires ``authorized=True``
        because it may purge orphan rows after resync.

        ``dry_run=True`` (default) only reports counts.
        Actual orphan purge requires ``dry_run=False`` AND ``authorized=True``.
        """
        mismatches = self._count_vec_parent_status_mismatch()
        orphan_counts = self._count_orphan_vectors()
        if dry_run:
            resp = {
                "dry_run": True,
                "vec_parent_status_mismatches": mismatches,
                "orphan_vectors": orphan_counts,
            }
            if mismatches.get("memory_vec_mismatch", 0) > 0 or mismatches.get("section_vec_mismatch", 0) > 0:
                resp["hint"] = (
                    "For non-destructive drift repair, run memory_resync_vec_parent_status(dry_run=False) "
                    "without authorized. To also purge orphan vector rows, re-run cleanup with "
                    "dry_run=False and authorized=True."
                )
            else:
                resp["hint"] = "re-run with dry_run=False and authorized=True to purge orphan vector rows"
            return self.db.state.response(resp)
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to purge orphan vector rows via cleanup",
                 "orphan_vectors": orphan_counts,
                 "vec_parent_status_mismatches": mismatches},
                ok=False,
            )
        # Phase 1: resync parent_status mismatches if any
        resync_counts = {}
        if mismatches.get("memory_vec_mismatch", 0) > 0 or mismatches.get("section_vec_mismatch", 0) > 0:
            resync_counts = self.db._resync_vec_parent_status()
        # Phase 2: purge orphans
        purged, warnings = self.db._purge_inactive_vectors()
        resp = {
            "dry_run": False,
            "purged": purged,
            "resynced": resync_counts,
        }
        if warnings:
            resp["warnings"] = warnings
        return self.db.state.response(resp, ok=not warnings)

    def _count_orphan_vectors(self) -> dict[str, int]:
        """Read-only count of truly orphan vec rows (no parent row in DB)."""
        if not self.db._db_available or not self.db.state.sqlite_vec_available:
            return {"orphan_memory_vectors": 0, "orphan_section_vectors": 0}
        try:
            with self.db.connection() as conn:
                mem = conn.execute(
                    "SELECT COUNT(*) AS c FROM memories_vec v "
                    "WHERE v.id NOT IN (SELECT id FROM memories)"
                ).fetchone()["c"]
                sec = conn.execute(
                    "SELECT COUNT(*) AS c FROM memory_sections_vec v "
                    "WHERE v.id NOT IN (SELECT id FROM memory_sections)"
                ).fetchone()["c"]
                return {"orphan_memory_vectors": int(mem), "orphan_section_vectors": int(sec)}
        except Exception:
            return {"orphan_memory_vectors": 0, "orphan_section_vectors": 0}

    def _count_vec_parent_status_mismatch(self) -> dict[str, int]:
        """Delegate to db layer (v0.9.4)."""
        return self.db._count_vec_parent_status_mismatch() if self.db._db_available and self.db.state.sqlite_vec_available else {}

    def memory_resync_vec_parent_status(
        self,
        dry_run: bool = True,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Repair vec.parent_status to match memories.status (v0.9.4, design §3.5 N16).

        Scans memories_vec and memory_sections_vec for rows where
        ``parent_status != COALESCE(memories.status, 'deleted')`` and updates
        them in-place. This fixes drift caused by direct DB edits, migration
        bugs, or failed transactions.

        ``dry_run=True`` (default) only reports how many rows would be updated.
        Per design §3.5 N16, resync is a **non-destructive UPDATE** (it only
        aligns ``parent_status`` to the already-existing ``memories.status``;
        no memory content is rewritten, no vectors are deleted), so it does
        NOT require ``authorized=True``. The ``authorized`` parameter is kept
        as a no-op compatibility placeholder so the signature matches the other
        repair tools (``memory_cleanup_inactive_vectors`` still requires it,
        because that path also purges orphan rows).

        Returns:
            dict with keys:
              - dry_run: bool
              - mismatches: dict[str, int] with memory_vec_mismatch, section_vec_mismatch
              - resynced: dict[str, int] (only if dry_run=False) with resynced_memory_vecs, resynced_section_vecs
              - warnings: list[str] if any errors occurred
        """
        mismatches = self._count_vec_parent_status_mismatch()
        if dry_run:
            return self.db.state.response(
                {
                    "dry_run": True,
                    "mismatches": mismatches,
                    "hint": "re-run with dry_run=False to resync mismatched rows (non-destructive; authorized not required)",
                }
            )
        # N16: non-destructive UPDATE — authorized accepted as compat placeholder, not enforced.
        resync_counts = self.db._resync_vec_parent_status()
        return self.db.state.response(
            {
                "dry_run": False,
                "mismatches": mismatches,
                "resynced": resync_counts,
            }
        )

    # ==================================================================
    #  v0.7.6: Conflict-signal attachment for search results
    # ==================================================================

    # Trust rank for runtime_metadata_hint (higher = more authoritative).
    _TRUST_RANK: dict[str, int] = {
        "locked": 100,
        "user_confirmed": 100,
        "document_extracted": 70,
        "agent_generated": 45,
        "pending": 20,
        "unknown": 10,
    }
