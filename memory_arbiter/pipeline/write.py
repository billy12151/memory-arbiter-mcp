"""Memory write path for the local-text evidence architecture."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..embedder import ManagedEmbedder

from .. import workspace_rules
from ..constants import is_default_workspace_term
from ..models import MemoryRecord, MemoryStatus
from ..validation import validate_product_payload

if TYPE_CHECKING:
    from ..tools import MemoryTools


class WritePipeline:
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools: "MemoryTools" = tools
        self.db = tools.db
        self.settings = tools.settings

    def _allowed(self, *args: Any, **kwargs: Any) -> "tuple[bool, list[str]]":
        return self._tools._allowed(*args, **kwargs)

    def _post_commit(
        self, *args: Any, **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._tools._post_commit(*args, **kwargs)

    def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_active_embedder()

    def _suggest_workspace_candidate(self, *args: Any, **kwargs: Any) -> Any:
        return self._tools._suggest_workspace_candidate(*args, **kwargs)

    def current_agent_id(self) -> str | None:
        return self._tools.current_agent_id()

    def memory_write(self, **payload: Any) -> dict[str, Any]:
        payload = dict(payload)
        validation = validate_product_payload(
            "memory", "remember", payload, vec_dim=int(self.settings.vec_dim),
        )
        if validation.error is not None:
            error = dict(validation.error)
            if error.get("field") in {"content", "subject"} and str(error.get("reason") or "").startswith("is required"):
                error["error"] = f"{error['field']} is required"
            return self._tools.db.state.response(
                {"written": False, **error}, ok=False,
            )
        # Policy is evaluated against the trusted request identity, never
        # caller-supplied payload fields (agent_id/client are not write inputs).
        allowed, policy_warnings = self._allowed()
        if not allowed:
            return self._tools.db.state.response(
                {"written": False}, ok=False, extra_warnings=policy_warnings,
            )
        if not str(payload.get("subject") or "").strip():
            return self._tools.db.state.response(
                {"written": False, "error": "subject is required"},
                ok=False, extra_warnings=policy_warnings,
            )
        if self.settings.isolation == "strict" and not str(payload.get("workspace") or "").strip():
            return self._tools.db.state.response(
                {"written": False, "error": "isolation=strict requires a workspace on every write"},
                ok=False, extra_warnings=policy_warnings,
            )
        # Authoritative guard for callers that bypass product-surface validation
        # (console API, direct MemoryTools use): superseded/conflicted/deleted
        # are lifecycle outcomes, never caller-supplied write inputs.
        status_value = payload.get("status")
        if status_value is not None and status_value not in (
            MemoryStatus.ACTIVE.value, MemoryStatus.PENDING.value,
        ):
            return self._tools.db.state.response(
                {
                    "written": False,
                    "error": "invalid_input",
                    "field": "status",
                    "reason": "must be 'active' (default) or 'pending'; superseded/conflicted/deleted are lifecycle outcomes, not write inputs",
                },
                ok=False, extra_warnings=policy_warnings,
            )

        try:
            record = MemoryRecord.from_input(payload, self.settings.defaults())
            # Attribution channel: the trusted request identity (HTTP headers or
            # the stdio process identity) owns agent_id. Payload/agent input can
            # no longer smuggle provenance; when no trusted identity exists
            # (direct MemoryTools use), from_input's env/config fallback stands.
            record.agent_id = self.current_agent_id() or record.agent_id
            workspace = self._resolve_write_workspace(record)
            if workspace["strict_block"]:
                record.status = MemoryStatus.PENDING.value

            memory_id, write_warnings = self.db.insert_memory(
                record, workspace["canonical"], workspace.get("canonical_embedding"),
                register_workspace_canonical=not workspace["strict_block"],
            )
            if any("workspace canonical vector publish failed" in warning for warning in write_warnings):
                workspace["vector_publish_pending"] = True
            data: dict[str, Any] = {
                "id": memory_id,
                "backup_only": memory_id is None,
                "record": {**record.__dict__, "id": memory_id},
                "workspace_canonical": workspace["canonical"],
                "workspace_matched_by": workspace["matched_by"],
            }
            self._apply_workspace_response(data, workspace)
            if memory_id is not None:
                data["evidence_index"], data["semantic_conflict_check"] = (
                    self._post_commit(memory_id, data["record"], recheck_conflicts=True)
                )
            response = self._tools.db.state.response(
                data,
                extra_warnings=(
                    policy_warnings + validation.warnings + write_warnings + workspace["warnings"]
                ),
            )
            if workspace["is_new"] and not workspace["strict_block"]:
                response.setdefault("notices", []).append({
                    "type": "workspace_review",
                    "severity": "info",
                    "workspace": workspace["canonical"],
                    "message": (
                        f"New workspace {workspace['canonical']!r} was registered. "
                        "Review the workspace registry for duplicates before confirming it."
                    ),
                    "action_required": "review_workspace_registry",
                    "review_call": {
                        "tool": "memory_review",
                        "view": "doctor",
                        "data": {},
                    },
                    "confirm_call": {
                        "tool": "memory_govern",
                        "action": "confirm_workspaces",
                        "data": {},
                    },
                    "authorization_required": True,
                })
            return response
        except Exception as exc:
            return self._tools.db.state.response(
                {"written": False, "error": str(exc)},
                ok=False, extra_warnings=policy_warnings + validation.warnings,
            )

    def _resolve_write_workspace(self, record: MemoryRecord) -> dict[str, Any]:
        raw = record.workspace
        result: dict[str, Any] = {
            "canonical": raw,
            "is_new": False,
            "matched_by": "fallback",
            "similar": [],
            "warnings": [],
            "decision": None,
            "decision_reason": None,
            "candidate": None,
            "vector_publish_pending": False,
            "strict_block": False,
            "canonical_embedding": None,
        }
        isolation = self.settings.isolation

        embedder, warnings = self._ensure_active_embedder()
        result["warnings"].extend(warnings)
        # Resolution is read-only. Registration of only the final policy result
        # happens atomically in insert_memory.
        resolved = self.db.resolve_workspace_canonical(raw, embedder, register_new=False)
        result.update({
            "canonical": resolved["canonical"],
            "is_new": bool(resolved["is_new"]),
            "matched_by": resolved["matched_by"],
            "similar": resolved.get("similar") or [],
            "vector_publish_pending": bool(resolved.get("vector_publish_pending")),
        })
        candidate_embedding = resolved.get("candidate_embedding")
        if result["canonical"] == raw and candidate_embedding:
            result["canonical_embedding"] = candidate_embedding
        result["warnings"].extend(resolved.get("warnings") or [])
        # Confirmed aliases short-circuit before embedding their canonical text.
        # Backfill a missing canonical vector once, outside the memory write
        # transaction; insert_memory publishes this prepared vector post-commit.
        # Exact matches take the same repair path so the "retry a write using
        # this workspace" guidance actually republishes a missing vector.
        if result["matched_by"] in {"confirmed_alias", "exact"}:
            result["canonical_embedding"] = (
                self.db.workspaces.prepare_missing_workspace_canonical_embedding(
                    result["canonical"], embedder,
                )
            )
        evidence = workspace_rules.extract_evidence(record)
        rule = workspace_rules.rule_decision(raw, resolved, evidence)
        result["decision"] = rule["decision"]
        result["decision_reason"] = rule["reason"]
        if isolation == "strict" and result["matched_by"] == "vector":
            # Vector similarity is a candidate, not a safe mechanical identity.
            # Strict mode must never silently merge it.
            result["canonical"] = raw.strip() or result["canonical"]
            result["is_new"] = True
            result["matched_by"] = "strict_candidate"
            result["decision"] = "ASK"
            result["decision_reason"] = "strict_requires_confirmation"
            result["canonical_embedding"] = candidate_embedding
        elif rule["decision"] == "KEEP" and result["matched_by"] == "vector":
            result["canonical"] = raw.strip() or result["canonical"]
            result["is_new"] = True
            result["matched_by"] = "rule_keep"
            result["canonical_embedding"] = candidate_embedding
        elif rule["decision"] is None:
            suggestion = self._suggest_workspace_candidate(raw, evidence, result["similar"])
            result["candidate"] = suggestion
            rejected = bool(
                suggestion is not None and suggestion.candidate
                and suggestion.candidate in (resolved.get("rejected_canonicals") or [])
            )
            if (
                suggestion is not None and suggestion.candidate
                and suggestion.relation in {"alias", "typo", "same_project"}
                and isolation in {"none", "weak"} and (suggestion.confidence or 0.0) >= 0.85
                and not rejected
            ):
                result["canonical"] = suggestion.candidate
                result["is_new"] = False
                result["matched_by"] = "qwen"
                result["decision"] = "AUTO"
                result["decision_reason"] = "qwen_high_conf"
                # The selected candidate already exists; never publish the raw
                # query embedding under the chosen canonical.
                result["canonical_embedding"] = None
            else:
                # Behavior is unchanged (keep raw canonical + workspace_review);
                # the reason distinguishes model-absent/timeout/uncertain from a
                # genuine low-confidence model output (spec §8 diagnostics).
                result["decision"] = "ASK"
                err = str(suggestion.error).lower() if (suggestion and suggestion.error) else ""
                if suggestion is None:
                    result["decision_reason"] = "qwen_unavailable" if result["similar"] else "no_similar_candidates"
                elif rejected:
                    result["decision_reason"] = "qwen_rejected_candidate"
                elif "timeout" in err or "deadline" in err:
                    # Admission/inference deadlines surface as "... deadline
                    # expired ..." not the literal "timeout"; both are technical.
                    result["decision_reason"] = "qwen_timeout"
                elif err:
                    # Any other backend error (disabled/crashed/invalid child)
                    # is a technical failure, not a low-confidence model output.
                    result["decision_reason"] = "qwen_backend_error"
                elif not suggestion.candidate and suggestion.relation == "unrelated":
                    result["decision_reason"] = "qwen_unrelated"
                else:
                    result["decision_reason"] = "qwen_low_conf"
        # Empty/default workspace: offer a NON-binding placement suggestion from
        # the memory's own subject (thought A: nearest existing memory's
        # workspace). A real-library A/B showed this is accurate but its
        # distance scale is not comparable to the name-vector threshold, so it
        # must never auto-assign — global memories (user prefs/identity) belong
        # in default. Read-only: suggest, let the agent/user decide.
        # all reserved default synonyms resolve to the same canonical
        # ("default"), so the hint fires for 默认/none/未知 writes too.
        if is_default_workspace_term(str(result["canonical"] or "")):
            result["placement_suggestion"] = self._suggest_placement_for_default(record)
        result["strict_block"] = isolation == "strict" and result["is_new"]
        return result

    def _suggest_placement_for_default(self, record: MemoryRecord) -> dict[str, Any] | None:
        """Read-only subject-based placement hint for a default/empty workspace.

        Embeds the subject and finds the nearest existing memory that lives in a
        real (non-default) workspace. Returns a suggestion dict or None. Never
        writes, never auto-assigns; global memories intentionally stay in
        default when no confident neighbor exists.
        """
        # Under strict isolation the unscoped KNN would leak the existence and
        # canonical name of a foreign workspace's memory to a caller whose read
        # ACL hides it; the hint is a none/weak convenience only.
        if getattr(self.settings, "isolation", "none") == "strict":
            return None
        subject = str(getattr(record, "subject", None) or "").strip()
        if not subject:
            return None
        embedder, _ = self._ensure_active_embedder()
        if embedder is None or not self.db.state.sqlite_vec_available:
            return None
        try:
            er = embedder.embed_text(prefix="", body=subject)
        except Exception:
            return None
        if not er or not er.embedding:
            return None
        try:
            hits = self.db.evidence_knn(list(er.embedding), k=8)
        except Exception:
            return None
        for hit in hits:
            peer = self.db.get_memory(int(hit.get("memory_id") or 0))
            if not peer or peer.get("status") != "active":
                continue
            ws = str(peer.get("workspace_canonical") or peer.get("workspace") or "").strip()
            if ws and not is_default_workspace_term(ws):
                return {
                    "suggested_workspace": ws,
                    "from_memory_id": int(peer.get("id") or 0),
                    "basis": "subject_nearest_memory",
                    "note": (
                        "Not auto-assigned: this memory is in 'default'. Its subject is "
                        f"closest to memory #{peer.get('id')} in workspace {ws!r}. If it belongs "
                        "there, rewrite with that workspace or use governance to move it."
                    ),
                }
        return None

    def _apply_workspace_response(self, data: dict[str, Any], workspace: dict[str, Any]) -> None:
        if workspace["vector_publish_pending"]:
            data["workspace_vector_publish"] = {
                "status": "pending_retry",
                "canonical": workspace["canonical"],
                "repair_task_available": False,
                "retry": "Write another memory using this workspace after sqlite-vec recovers.",
            }
        if workspace["strict_block"]:
            data.update({
                "attention_required": True,
                "action_required": "confirm_new_workspace",
                "verification_status": "pending_user",
                "workspace_is_new": True,
                "pending_workspace": {
                    "canonical": workspace["canonical"],
                    "similar_workspaces": workspace["similar"],
                },
                "attention_summary": (
                    f"strict isolation: workspace {workspace['canonical']!r} is new; "
                    "confirm it with memory_govern(action='confirm_pending_workspace')"
                ),
            })
        elif workspace["is_new"]:
            data.setdefault("write_hints", {})["new_workspace_detected"] = {
                "canonical": workspace["canonical"],
                "similar_workspaces": workspace["similar"],
            }
        if workspace["decision"] is not None:
            data["workspace_decision"] = workspace["decision"]
            data["workspace_decision_reason"] = workspace["decision_reason"]
        suggestion = workspace.get("candidate")
        if suggestion is not None and suggestion.candidate:
            data["workspace_candidate"] = {
                "candidate": suggestion.candidate,
                "relation": suggestion.relation,
                "confidence": suggestion.confidence,
                "evidence": suggestion.evidence,
            }
        if workspace["decision"] == "ASK" and not workspace["strict_block"]:
            similar = workspace["similar"]
            options: list[dict[str, Any]] = [
                {"decision": "keep_separate", "action": None},
            ]
            if similar:
                options.insert(0, {
                    "decision": "merge",
                    "call": {
                        "tool": "memory_govern",
                        "action": "migrate_workspace",
                        "data": {
                            "from": data["record"]["workspace"],
                            "to": similar[0].get("name"),
                        },
                    },
                    "authorization_required": True,
                })
            data.setdefault("write_hints", {})["workspace_review"] = {
                "raw": data["record"]["workspace"],
                "reason": workspace["decision_reason"],
                "similar_workspaces": similar,
                "options": options,
                "note": "Merge only after user confirmation; otherwise keep the workspace separate.",
            }
        placement = workspace.get("placement_suggestion")
        if placement:
            # Non-binding: the memory was still written to default. Surface a
            # subject-based placement hint so the agent/user can move it if it
            # truly belongs elsewhere.
            data.setdefault("write_hints", {})["placement_suggestion"] = placement
