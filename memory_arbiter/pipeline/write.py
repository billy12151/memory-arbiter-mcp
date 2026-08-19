"""Memory write path for the local-text evidence architecture."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .. import workspace_rules
from ..models import MemoryRecord, MemoryStatus

if TYPE_CHECKING:
    from ..tools import MemoryTools


class WritePipeline:
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools: "MemoryTools" = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    def memory_write(self, **payload: Any) -> dict[str, Any]:
        allowed, policy_warnings = self._allowed(payload.get("agent_id"), payload.get("client"))
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

        try:
            record = MemoryRecord.from_input(payload, self.settings.defaults())
            workspace = self._resolve_write_workspace(record)
            if workspace["strict_block"]:
                record.status = MemoryStatus.PENDING.value

            memory_id, write_warnings = self.db.insert_memory(record, workspace["canonical"])
            data: dict[str, Any] = {
                "id": memory_id,
                "backup_only": memory_id is None,
                "record": {**record.__dict__, "id": memory_id},
                "workspace_canonical": workspace["canonical"],
                "workspace_matched_by": workspace["matched_by"],
            }
            self._apply_workspace_response(data, workspace)
            if memory_id is not None:
                data["evidence_index"] = self._enqueue_local_text_index(memory_id, data["record"])
                data["semantic_conflict_check"] = {
                    "status": "deferred", "reason": "waiting_for_evidence_index",
                }
            return self._tools.db.state.response(
                data,
                extra_warnings=policy_warnings + write_warnings + workspace["warnings"],
            )
        except Exception as exc:
            return self._tools.db.state.response(
                {"written": False, "error": str(exc)},
                ok=False, extra_warnings=policy_warnings,
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
        }
        isolation = self.settings.isolation
        if isolation == "none":
            return result

        embedder, warnings = self._ensure_embedder()
        result["warnings"].extend(warnings)
        resolved = self.db.resolve_workspace_canonical(raw, embedder, register_new=True)
        result.update({
            "canonical": resolved["canonical"],
            "is_new": bool(resolved["is_new"]),
            "matched_by": resolved["matched_by"],
            "similar": resolved.get("similar") or [],
            "vector_publish_pending": bool(resolved.get("vector_publish_pending")),
        })
        result["warnings"].extend(resolved.get("warnings") or [])
        evidence = workspace_rules.extract_evidence(record)
        rule = workspace_rules.rule_decision(raw, resolved, evidence)
        result["decision"] = rule["decision"]
        result["decision_reason"] = rule["reason"]
        if rule["decision"] == "KEEP" and result["matched_by"] == "vector":
            result["canonical"] = raw.strip() or result["canonical"]
            result["is_new"] = True
            result["matched_by"] = "rule_keep"
            self.db.resolve_workspace_canonical(result["canonical"], embedder, register_new=True)
        elif rule["decision"] is None:
            suggestion = self._suggest_workspace_candidate(raw, evidence, result["similar"])
            result["candidate"] = suggestion
            if (
                suggestion is not None and suggestion.candidate
                and suggestion.relation in {"alias", "typo", "same_project"}
                and isolation == "weak" and (suggestion.confidence or 0.0) >= 0.85
                and suggestion.candidate not in (resolved.get("rejected_canonicals") or [])
            ):
                result["canonical"] = suggestion.candidate
                result["is_new"] = False
                result["matched_by"] = "qwen"
                result["decision"] = "AUTO"
                result["decision_reason"] = "qwen_high_conf"
            else:
                result["decision"] = "ASK"
                result["decision_reason"] = "qwen_low_conf"
        result["strict_block"] = isolation == "strict" and result["is_new"]
        return result

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
        elif self.settings.isolation == "weak" and workspace["is_new"]:
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
            data.setdefault("write_hints", {})["workspace_review"] = {
                "raw": data["record"]["workspace"],
                "reason": workspace["decision_reason"],
                "similar_workspaces": workspace["similar"],
                "how_to_confirm": "memory_govern accept_workspace_alias / reject_workspace_alias",
            }
