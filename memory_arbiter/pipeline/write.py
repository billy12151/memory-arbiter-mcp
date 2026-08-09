"""Write + structured-claim reconciliation pipeline for MemoryTools (Phase 4 extraction)."""
# mypy: disable-error-code=no-any-return
from __future__ import annotations

import json
import time
from typing import Any, Optional, TYPE_CHECKING

from ..claims import extract_claims
from ..models import MemoryRecord, MemoryStatus
from ..db import _canon_entity, _canon_scope
from .. import workspace_rules

if TYPE_CHECKING:
    from ..tools import MemoryTools


class WritePipeline:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    @staticmethod
    def _extract_claims(*args: Any, **kwargs: Any) -> Any:
        # Preserve the legacy monkeypatch seam: tests/external diagnostics patch
        # memory_arbiter.tools.extract_claims, so resolve that binding at call
        # time instead of using this module's import cache (R4).
        from .. import tools as tools_mod
        return getattr(tools_mod, "extract_claims", extract_claims)(*args, **kwargs)

    def memory_write(self, **payload: Any) -> dict[str, Any]:
        allowed, warnings = self._allowed(payload.get("agent_id"), payload.get("client"))
        if not allowed:
            return self.db.state.response({"written": False}, ok=False, extra_warnings=warnings)
        try:
            # subject is required at write time, BEFORE any workspace side effect.
            # A missing/empty subject must fail fast here — otherwise a failed write
            # would still register the workspace canonical (resolve_workspace_canonical
            # below runs with register_new=True), letting a retry skip the strict
            # new-workspace pending gate.
            if not payload.get("subject") or not str(payload["subject"]).strip():
                return self.db.state.response(
                    {"written": False, "error": "subject is required"},
                    ok=False,
                    extra_warnings=warnings,
                )
            # ── Workspace isolation (strict/weak/none) ──
            isolation = self.settings.isolation
            # strict: workspace is mandatory on the write path. Inspect the RAW
            # payload — MemoryRecord.from_input substitutes "default" for a
            # missing/empty workspace, which would mask the omission here.
            if isolation == "strict" and not str(payload.get("workspace") or "").strip():
                return self.db.state.response(
                    {
                        "written": False,
                        "error": "isolation=strict requires a workspace on every write; "
                                 "an empty workspace would cause silent recall failure.",
                    },
                    ok=False,
                    extra_warnings=warnings,
                )
            record = MemoryRecord.from_input(payload, self.settings.defaults())
            ws_raw = record.workspace
            ws_canonical = ws_raw
            ws_is_new = False
            ws_matched_by = "fallback"
            ws_similar: list[dict[str, Any]] = []
            # Resolve canonical alias only under weak/strict — in `none` the
            # workspace is fully ignored (no embedder call, no canonical write),
            # so we don't perturb the write path or its embedder-call invariants.
            # Resolution runs even without an embedder: it degrades to exact
            # string identity (still detects new-vs-existing via the canonical
            # table), which is what drives strict-block / weak-hint.
            embedding_warnings: list[str] = []
            ws_rule_decision: Optional[dict[str, Any]] = None
            ws_candidate_suggestion: Any = None
            if isolation != "none":
                embedder, ensure_warnings = self._ensure_embedder()
                embedding_warnings.extend(ensure_warnings)
                resolved = self.db.resolve_workspace_canonical(
                    ws_raw, embedder, register_new=True,
                )
                ws_canonical = resolved["canonical"]
                ws_is_new = resolved["is_new"]
                ws_matched_by = resolved["matched_by"]
                ws_similar = resolved.get("similar") or []
                # Rule-first decision layer (636 §5): vector produced candidates;
                # rules decide AUTO/KEEP/ASK. KEEP overrides a vector merge back
                # to the raw workspace; ASK is surfaced as a hint below.
                evidence = workspace_rules.extract_evidence(record)
                ws_rule_decision = workspace_rules.rule_decision(ws_raw, resolved, evidence)
                if ws_rule_decision["decision"] == "KEEP" and ws_matched_by == "vector":
                    # Undo the vector merge: keep this memory in its own workspace.
                    ws_canonical = (ws_raw or "").strip() or ws_canonical
                    ws_is_new = True
                    ws_matched_by = "rule_keep"
                    # Register the kept-separate workspace as its own canonical
                    # so a later write with the same raw string doesn't fall
                    # through to "new" and re-run this KEEP decision every time.
                    self.db.resolve_workspace_canonical(ws_canonical, embedder, register_new=True)
                elif ws_rule_decision["decision"] is None:
                    # Rules undecided → model layer suggests a candidate (636 §6).
                    # The model is a *suggester*: it can promote a weak-mode merge
                    # but can never override confirmed/rejected/strict policy.
                    cand_sig = self._suggest_workspace_candidate(ws_raw, evidence, ws_similar)
                    ws_candidate_suggestion = cand_sig
                    if (
                        cand_sig is not None
                        and cand_sig.candidate
                        and cand_sig.relation in {"alias", "typo", "same_project"}
                        and isolation == "weak"
                        and (cand_sig.confidence or 0.0) >= 0.85
                        and cand_sig.candidate not in (resolved.get("rejected_canonicals") or [])
                    ):
                        # High-confidence weak-mode silent merge — only for
                        # identity-grade relations. related/same_family/unrelated/
                        # uncertain are NOT the same workspace even at high conf.
                        ws_canonical = cand_sig.candidate
                        ws_is_new = False
                        ws_matched_by = "qwen"
                        ws_rule_decision = {"decision": "AUTO", "reason": "qwen_high_conf", "canonical": ws_canonical}
                    else:
                        # Mid/low confidence, or strict: don't silently merge.
                        # weak → write active + surface hint; strict → pending.
                        ws_rule_decision = {"decision": "ASK", "reason": "qwen_low_conf", "canonical": None}
            # strict + brand-new canonical → block activation (status=pending)
            # until the user confirms the workspace name.
            strict_block = isolation == "strict" and ws_is_new
            if strict_block:
                record.status = MemoryStatus.PENDING.value
            memory_id, write_warnings = self.db.insert_memory(record, ws_canonical)
            data = {"id": memory_id, "backup_only": memory_id is None, "record": {**record.__dict__, "id": memory_id}}
            data["workspace_canonical"] = ws_canonical
            data["workspace_matched_by"] = ws_matched_by
            if strict_block:
                data.update({
                    "attention_required": True,
                    "action_required": "confirm_new_workspace",
                    "verification_status": "pending_user",
                    "workspace_is_new": True,
                    "pending_workspace": {
                        "canonical": ws_canonical,
                        "similar_workspaces": ws_similar,
                    },
                    "attention_summary": (
                        f"strict isolation: workspace {ws_canonical!r} is new. "
                        "Memory written as pending and excluded from active recall "
                        "until activated via memory_activate."
                    ),
                })
            elif isolation == "weak" and ws_is_new:
                hints = data.get("write_hints") or {}
                hints["new_workspace_detected"] = {
                    "canonical": ws_canonical,
                    "similar_workspaces": ws_similar,
                }
                data["write_hints"] = hints
            # Surface the rule decision. ASK → the write succeeds (weak keeps it
            # active) but the agent/user is nudged to confirm the workspace via
            # memory_govern accept/reject_workspace_alias (636 §5,8).
            if ws_rule_decision is not None:
                data["workspace_decision"] = ws_rule_decision["decision"]
                data["workspace_decision_reason"] = ws_rule_decision["reason"]
                if ws_candidate_suggestion is not None and ws_candidate_suggestion.candidate:
                    data["workspace_candidate"] = {
                        "candidate": ws_candidate_suggestion.candidate,
                        "relation": ws_candidate_suggestion.relation,
                        "confidence": ws_candidate_suggestion.confidence,
                        "evidence": ws_candidate_suggestion.evidence,
                    }
                if ws_rule_decision["decision"] == "ASK" and not strict_block:
                    hints = data.get("write_hints") or {}
                    hints["workspace_review"] = {
                        "raw": ws_raw,
                        "reason": ws_rule_decision["reason"],
                        "similar_workspaces": ws_similar,
                        "how_to_confirm": "memory_govern accept_workspace_alias / reject_workspace_alias",
                    }
                    data["write_hints"] = hints
            if memory_id is not None and self.settings.embedding_auto_write and self._embedding_configured():
                data["embedding_stored"] = False
                embedder, ensure_warnings = self._ensure_embedder()
                embedding_warnings.extend(ensure_warnings)
                if embedder is not None:
                    try:
                        er = embedder.embed_text(
                            prefix=record.subject or "",
                            body=record.content,
                        )
                        if not er.embedding:
                            raise RuntimeError(
                                f"encode returned empty embedding: {getattr(embedder, 'last_encode_error', None) or 'unknown'}"
                            )
                        data["embedding_stored"], store_warnings = self.db.store_embedding(memory_id, er.embedding)
                        embedding_warnings.extend(store_warnings)
                        if er.truncated:
                            embedding_warnings.append(
                                f"memory embedding truncated: {er.used_tokens}/{er.original_tokens} tokens"
                            )
                    except Exception as exc:
                        embedding_warnings.append(f"auto-embedding write failed: {exc}")
            # v0.8.0: post-write split (design §6.1). Replaces the v0.6 split_hint.
            # vec ready + long enough + safe heading plan → rules auto-split;
            # otherwise a full split_request is returned for the Agent to
            # continue with its own LLM. Never gated on split_enabled.
            if memory_id is not None:
                split_block, split_request, split_warnings = self._after_write_split(memory_id)
                data["split"] = split_block
                if split_request is not None:
                    data["split_request"] = split_request
                embedding_warnings.extend(split_warnings)
            data = self._enrich_write_response(data, memory_id, record)
            if memory_id is not None:
                data["semantic_conflict_check"] = self._enqueue_semantic_conflict_check(memory_id, record)
            claim_warnings = list(data.pop("_claim_warnings", []))
            return self.db.state.response(
                data,
                extra_warnings=warnings + write_warnings + embedding_warnings + claim_warnings,
            )
        except Exception as exc:
            return self.db.state.response({"error": str(exc)}, ok=False, extra_warnings=warnings)

    def _enrich_write_response(
        self, data: dict[str, Any], memory_id: Optional[int], record: MemoryRecord,
    ) -> dict[str, Any]:
        """Post-write enrichment: v0.9 structured claims first, then metadata hints.

        Never raises; failures are silently swallowed (advisory is best-effort —
        the insert already committed before this runs, so it can never lose the
        write). v0.8.8: each duplicate candidate is recorded as a dismissable
        ``metadata_write_hint`` conflict row; pairs the user already dismissed
        (not_a_conflict, version match) are skipped (Layer 0). ``open`` pairs
        still re-ring on repeat writes (an unresolved conflict is worth reminding).
        """
        if memory_id is None:
            return data
        try:
            structured = self._index_and_reconcile_claims(memory_id)
            data["realtime_conflict_check"] = structured["diagnostic"]
            data["claim_indexed"] = bool(structured["diagnostic"].get("claim_indexed"))
            data["claim_reconciled"] = bool(
                structured["diagnostic"].get("claim_reconciled")
            )
            data["record"] = self.db.get_memory(memory_id) or data.get("record")
            if structured.get("warnings"):
                data["_claim_warnings"] = list(structured["warnings"])
            conflicts = structured.get("conflicts") or []
            structured_pair_ids = set(structured.get("peer_ids") or [])
            self._apply_structured_gate(data, structured)
            if conflicts:
                first = conflicts[0]
                data["attention_summary"] = (
                    f"Structured claim conflict with memory #{first['peer_id']}"
                    + (f" and {len(conflicts) - 1} more" if len(conflicts) > 1 else "")
                )
            elif structured.get("pending_user_conflicts"):
                pending = structured["pending_user_conflicts"]
                data["attention_summary"] = (
                    f"Structured conflict with memory #{pending[0]['peer_id']} requires user confirmation"
                    + (f" and {len(pending) - 1} more" if len(pending) > 1 else "")
                )
            hints = self._write_duplicate_hints(memory_id, record)
            if not hints:
                return data
            data["write_hints"] = hints
            targets = hints.get("possible_supersede_targets") or []
            new_ver = self.db.get_memory_version(memory_id) or 1
            rang = False
            for t in targets:
                cand_id = int(t["id"])
                if cand_id in structured_pair_ids:
                    continue
                # Layer 0: pair already dismissed → skip. Fail-open: if the check
                # itself errors, treat as NOT dismissed (ring) — a diagnostic
                # failure must never silently suppress the duplicate prompt.
                try:
                    dismissed = self.db.is_pair_dismissed(memory_id, cand_id)
                except Exception:
                    dismissed = False
                if dismissed:
                    continue
                # v0.8.8: record as a dismissable advisory conflict row (idempotent
                # if a row already exists for this pair).
                self.db.record_conflict_enriched(
                    memory_id, cand_id,
                    conflict_type="metadata_overlap",
                    conflict_point=t.get("reason") or "write metadata overlap",
                    reason=t.get("reason") or "possible duplicate/evolution",
                    confidence_hint="low",
                    source="metadata_write_hint", status="open",
                    left_version=new_ver,
                    right_version=self.db.get_memory_version(cand_id) or 1,
                    detection_channel="metadata",
                )
                # Ring + log once, for the first non-dismissed target.
                if not rang and not conflicts:
                    rang = True
                    summary = f"Possible duplicate/evolution of memory #{cand_id}"
                    if t.get("subject"):
                        summary += f" ({t['subject']})"
                    if len(targets) > 1:
                        summary += f" and {len(targets) - 1} more"
                    data["attention_required"] = True
                    data["attention_summary"] = summary
                    self.db.log_attention(
                        trigger="write",
                        source=str(t.get("hint_type", "possible_duplicate")),
                        memory_ids=[memory_id, cand_id],
                    )
        except Exception as exc:
            data["claim_indexed"] = False
            data["claim_reconciled"] = False
            data.setdefault("_claim_warnings", []).append(
                f"structured claim enrichment failed: {type(exc).__name__}: {exc}"
            )
        return data

    @staticmethod
    def _structured_conflict_point(claims: list[dict[str, Any]]) -> str:
        parts = [
            f"{c['entity']}.{c['attribute']}: {c['left_value']} vs {c['right_value']}"
            for c in claims[:5]
        ]
        if len(claims) > 5:
            parts.append(f"and {len(claims) - 5} more")
        return "; ".join(parts)

    @staticmethod
    def _apply_structured_gate(data: dict[str, Any], structured: dict[str, Any]) -> None:
        """Promote the current structured state to one unambiguous top-level gate."""
        pending_llm = structured.get("conflicts") or []
        pending_user = structured.get("pending_user_conflicts") or []
        if pending_llm:
            data.update({
                "attention_required": True,
                "action_required": "judge_conflict_before_use",
                "verification_status": "pending_llm",
                "conflict_source": "structured_claim_candidate",
                "conflict_judgment_requests": [
                    item["judgment_request"] for item in pending_llm
                ],
            })
        elif pending_user:
            data.update({
                "attention_required": True,
                "action_required": "ask_user",
                "verification_status": "pending_user",
                "conflict_source": "open_table",
                "pending_user_conflicts": pending_user,
            })

    def _index_and_reconcile_claims(self, memory_id: int) -> dict[str, Any]:
        """Extract, publish, and durably reconcile structured conflicts."""
        started = time.perf_counter()
        try:
            return self._index_and_reconcile_claims_impl(memory_id, started)
        except Exception as exc:
            try:
                record = self.db.get_memory(int(memory_id))
            except Exception:
                record = None
            current_revision = (
                int(record.get("claim_revision") or 1) if record else None
            )
            indexed = bool(
                record
                and record.get("claims_indexed_revision") == current_revision
            )
            reconciled = bool(
                indexed
                and record.get("claims_reconciled_revision") == current_revision
            )
            return {
                "diagnostic": {
                    "claim_indexed": indexed,
                    "claim_reconciled": reconciled,
                    "skipped_reason": "structured_enrichment_error",
                    "error": str(exc),
                    "structured_enrich_ms": round(
                        (time.perf_counter() - started) * 1000, 3
                    ),
                },
                "conflicts": [],
                "pending_user_conflicts": [],
                "peer_ids": [],
                "warnings": [
                    f"structured claim enrichment failed: "
                    f"{type(exc).__name__}: {exc}; rebuild will retry"
                ],
            }

    def _index_and_reconcile_claims_impl(
        self, memory_id: int, started: float,
    ) -> dict[str, Any]:
        """Implementation behind the fail-open structured-enrichment boundary."""

        def finish(payload: dict[str, Any]) -> dict[str, Any]:
            diagnostic = payload.setdefault("diagnostic", {})
            diagnostic["structured_enrich_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            return payload

        if self.settings.structured_claim_mode == "off":
            return finish({
                "diagnostic": {
                    "claim_indexed": False,
                    "claim_reconciled": False,
                    "skipped_reason": "structured_claim_mode_off",
                },
                "conflicts": [], "peer_ids": [], "warnings": [],
            })

        diagnostics: dict[str, Any] = {}
        publish: dict[str, Any] = {}
        expected_revision = 1
        for _attempt in range(2):
            record = self.db.get_memory(int(memory_id))
            if not record:
                return finish({
                    "diagnostic": {
                        "claim_indexed": False,
                        "claim_reconciled": False,
                        "skipped_reason": "memory_not_found",
                    },
                    "conflicts": [], "peer_ids": [],
                    "warnings": ["claim indexing skipped: memory not found"],
                })
            expected_revision = int(record.get("claim_revision") or 1)
            if record.get("status") != "active":
                publish = self.db.claims.publish_memory_claims(
                    int(memory_id), [], 0,
                    expected_claim_revision=expected_revision,
                )
                if publish.get("outcome") == "stale_snapshot":
                    continue
                indexed = publish.get("outcome") == "skipped_inactive"
                return finish({
                    "diagnostic": {
                        "claim_indexed": indexed,
                        "claim_reconciled": indexed,
                        "skipped_reason": "inactive",
                    },
                    "conflicts": [], "peer_ids": [], "warnings": [],
                })

            diagnostics = {}
            try:
                claims = self._extract_claims(record, diagnostics)
                publish = self.db.claims.publish_memory_claims(
                    int(memory_id), claims,
                    int(diagnostics.get("ambiguous_key_count") or 0),
                    expected_claim_revision=expected_revision,
                )
            except Exception as exc:
                self.db.claims.mark_claim_index_failed(
                    int(memory_id), expected_claim_revision=expected_revision,
                )
                return finish({
                    "diagnostic": {
                        "claim_indexed": False,
                        "claim_reconciled": False,
                        "error": str(exc),
                    },
                    "conflicts": [], "peer_ids": [],
                    "warnings": [
                        f"claim extraction failed: {type(exc).__name__}: {exc}"
                    ],
                })
            if publish.get("outcome") == "stale_snapshot":
                continue
            break
        else:
            return finish({
                "diagnostic": {
                    **diagnostics,
                    "claim_indexed": False,
                    "claim_reconciled": False,
                    "skipped_reason": "concurrent_revision_change",
                    "publish": publish,
                },
                "conflicts": [], "peer_ids": [],
                "warnings": [
                    "claim indexing deferred: memory changed during both publish attempts"
                ],
            })

        if publish.get("outcome") != "indexed":
            self.db.claims.mark_claim_index_failed(
                int(memory_id), expected_claim_revision=expected_revision,
            )
            return finish({
                "diagnostic": {
                    **diagnostics,
                    "claim_indexed": False,
                    "claim_reconciled": False,
                    "publish": publish,
                },
                "conflicts": [], "peer_ids": [],
                "warnings": [
                    f"claim index publish failed: "
                    f"{publish.get('error') or publish.get('outcome')}"
                ],
            })

        detection = self.db.claims.find_structured_claim_pairs(int(memory_id))
        if detection.get("stale_index") or detection.get("error"):
            return finish({
                "diagnostic": {
                    **diagnostics,
                    "claim_indexed": True,
                    "claim_reconciled": False,
                    "skipped_reason": (
                        "concurrent_revision_change"
                        if detection.get("stale_index")
                        else "collision_query_failed"
                    ),
                    "claim_revision": publish.get("claim_revision"),
                    "claim_count": publish.get("claim_count", 0),
                    "mode": self.settings.structured_claim_mode,
                },
                "conflicts": [], "peer_ids": [],
                "warnings": [
                    "claim collision reconciliation deferred; rebuild will retry"
                ],
            })

        current_pairs = {
            (int(pair["left_id"]), int(pair["right_id"])): pair
            for pair in detection.get("pairs", [])
        }
        reconciliation_errors: list[str] = []

        existing_state = self.db.claims.read_structured_open_conflicts_for_memory(
            int(memory_id)
        )
        if existing_state.get("error"):
            reconciliation_errors.append(
                f"read existing structured conflicts: {existing_state['error']}"
            )
        existing_open = existing_state.get("rows") or []
        for existing in existing_open:
            key = (int(existing["left_id"]), int(existing["right_id"]))
            if key not in current_pairs:
                resolved = self.db.resolve_conflict(
                    int(existing["id"]),
                    reason="claims aligned after reindex",
                    status="resolved",
                )
                if resolved.get("outcome") not in {"resolved", "not_open"}:
                    reconciliation_errors.append(
                        f"resolve conflict #{existing['id']}: "
                        f"{resolved.get('outcome')}"
                    )

        gate_states = self.db.claims.structured_pair_gate_states(
            int(memory_id), list(current_pairs.values())
        )
        if gate_states.get("error"):
            reconciliation_errors.append("structured pair gate query failed")
        dismissed = gate_states.get("dismissed") or set()
        closed = gate_states.get("closed") or set()

        peer_ids: list[int] = []
        recorded: list[tuple[int, int]] = []
        for key, pair in current_pairs.items():
            peer_id = (
                pair["right_id"]
                if pair["left_id"] == int(memory_id)
                else pair["left_id"]
            )
            peer_ids.append(int(peer_id))
            if key in dismissed or key in closed:
                continue
            point = self._structured_conflict_point(pair["claims"])
            result = self.db.record_conflict_enriched(
                pair["left_id"], pair["right_id"],
                conflict_type="contradiction",
                conflict_point=point,
                reason=(
                    "deterministic structured claims share "
                    "entity/attribute but differ in value"
                ),
                confidence_hint="high",
                source="structured_claim",
                status="open",
                refresh=True,
                left_version=pair["left_version"],
                right_version=pair["right_version"],
                left_claim_revision=pair["left_claim_revision"],
                right_claim_revision=pair["right_claim_revision"],
                judgment_status="pending_llm",
                detection_channel="structured",
                structured_details=[{
                    "entity": claim["entity"],
                    "attribute": claim["attribute"],
                    "scope": claim.get("scope") or "",
                    "left_value": claim["left_value"],
                    "right_value": claim["right_value"],
                    "extractor_rule": claim.get("extractor_rule"),
                } for claim in pair["claims"]],
            )
            conflict_id = result.get("conflict_id")
            if conflict_id is None:
                reconciliation_errors.append(
                    f"record pair {key}: {result.get('outcome') or 'missing conflict id'}"
                )
                continue
            recorded.append((int(conflict_id), int(peer_id)))

        conflicts: list[dict[str, Any]] = []
        pending_user_conflicts: list[dict[str, Any]] = []
        if recorded:
            current_state = self.db.claims.read_structured_open_conflicts_for_memory(
                int(memory_id)
            )
            if current_state.get("error"):
                reconciliation_errors.append(
                    f"read current structured conflicts: {current_state['error']}"
                )
            current_rows = {
                int(row["id"]): row
                for row in current_state.get("rows") or []
            }
            pending_ids = [
                conflict_id for conflict_id, _peer_id in recorded
                if (current_rows.get(conflict_id) or {}).get("judgment_status")
                in {None, "pending_llm"}
            ]
            requests = self.db.judgments.build_conflict_judgment_requests(pending_ids)
            for conflict_id, peer_id in recorded:
                current_row = current_rows.get(conflict_id)
                if current_row is None:
                    reconciliation_errors.append(
                        f"recorded conflict #{conflict_id} is not open"
                    )
                    continue
                current_status = current_row.get("judgment_status")
                if current_status in {None, "pending_llm"}:
                    request = requests.get(conflict_id)
                    if request is None:
                        reconciliation_errors.append(
                            f"conflict #{conflict_id} judgment request unavailable"
                        )
                        continue
                    conflicts.append({
                        "conflict_id": conflict_id,
                        "peer_id": peer_id,
                        "judgment_request": request,
                    })
                    self.db.log_attention(
                        trigger="structured_claim",
                        source="candidate_surfaced",
                        memory_ids=[int(memory_id), peer_id],
                    )
                elif current_status == "pending_user":
                    pending_user_conflicts.append({
                        "conflict_id": conflict_id,
                        "peer_id": peer_id,
                        "suggested_winner": current_row.get("suggested_winner"),
                        "reason": current_row.get("reason"),
                    })

        candidate_count = len(current_pairs)
        if reconciliation_errors:
            return finish({
                "diagnostic": {
                    **diagnostics,
                    "claim_indexed": True,
                    "claim_reconciled": False,
                    "skipped_reason": "conflict_reconciliation_failed",
                    "claim_revision": publish.get("claim_revision"),
                    "claim_count": publish.get("claim_count", 0),
                    "candidate_count": candidate_count,
                    "mode": self.settings.structured_claim_mode,
                },
                "conflicts": conflicts,
                "pending_user_conflicts": pending_user_conflicts,
                "peer_ids": peer_ids,
                "warnings": reconciliation_errors,
            })

        elapsed_before_marker = (time.perf_counter() - started) * 1000
        marker = self.db.claims.mark_claim_reconciled(
            int(memory_id),
            expected_revision,
            elapsed_before_marker,
            candidate_count,
        )
        if marker.get("outcome") != "reconciled":
            return finish({
                "diagnostic": {
                    **diagnostics,
                    "claim_indexed": True,
                    "claim_reconciled": False,
                    "skipped_reason": "reconciliation_marker_failed",
                    "claim_revision": publish.get("claim_revision"),
                    "claim_count": publish.get("claim_count", 0),
                    "candidate_count": candidate_count,
                    "mode": self.settings.structured_claim_mode,
                },
                "conflicts": conflicts,
                "pending_user_conflicts": pending_user_conflicts,
                "peer_ids": peer_ids,
                "warnings": [
                    f"claim reconciliation marker failed: "
                    f"{marker.get('error') or marker.get('outcome')}"
                ],
            })

        diagnostic = {
            **diagnostics,
            "claim_indexed": True,
            "claim_reconciled": True,
            "claim_revision": publish.get("claim_revision"),
            "claims_reconciled_revision": expected_revision,
            "claim_count": publish.get("claim_count", 0),
            "candidate_count": candidate_count,
            "pending_llm_count": len(conflicts),
            "pending_user_count": len(pending_user_conflicts),
            "evolution_pair_count": int(detection.get("evolution_pairs") or 0),
            "mode": self.settings.structured_claim_mode,
        }
        return finish({
            "diagnostic": diagnostic,
            "conflicts": conflicts,
            "pending_user_conflicts": pending_user_conflicts,
            "peer_ids": peer_ids,
            "warnings": [],
        })

    def _write_duplicate_hints(
        self, memory_id: int, record: MemoryRecord,
    ) -> Optional[dict[str, Any]]:
        """Detect possible duplicates/evolution of the just-written memory.

        Returns ``{possible_supersede_targets: [...]}`` or None if no
        candidates found. Uses DB candidate recall + Python overlap scoring.
        """
        candidates = self.db.find_metadata_overlap_candidates(
            subject=record.subject,
            tags=record.tags,
            exclude_id=memory_id,
        )
        if not candidates:
            return None
        new_content = record.content or ""
        targets: list[dict[str, Any]] = []
        my_tags = set(record.tags or [])
        my_subject_tokens = set((record.subject or "").lower().split())
        for cand in candidates:
            cand_tags = set(cand.get("tags") or [])
            cand_subject_tokens = set((cand.get("subject") or "").lower().split())
            # Tags Jaccard.
            if my_tags and cand_tags:
                common_tags = my_tags & cand_tags
                if len(common_tags) < 2:
                    continue
                tags_jaccard = len(common_tags) / len(my_tags | cand_tags)
                if tags_jaccard < 0.8:
                    continue
            else:
                continue
            # Subject overlap.
            if my_subject_tokens and cand_subject_tokens:
                subj_overlap = len(my_subject_tokens & cand_subject_tokens) / len(my_subject_tokens | cand_subject_tokens)
                if subj_overlap < 0.7:
                    continue
            # Determine hint type.
            cand_content = cand.get("content") or ""
            hint_type = "possible_duplicate"
            reason = f"tags Jaccard {len(my_tags & cand_tags)}/{len(my_tags | cand_tags)} + subject overlap"
            if len(new_content) >= len(cand_content) * 1.3:
                hint_type = "possible_evolution_of"
                reason += "; new content ≥1.3× candidate"
            targets.append({
                "id": int(cand["id"]),
                "subject": cand.get("subject"),
                "reason": reason,
                "hint_type": hint_type,
            })
            if len(targets) >= 3:
                break
        if not targets:
            return None
        return {"possible_supersede_targets": targets}
