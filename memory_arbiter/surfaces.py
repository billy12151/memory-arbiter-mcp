"""Product-surface routing helpers for MemoryTools (Phase 4 extraction)."""
from __future__ import annotations

import time
from importlib import resources
from typing import Any, Callable, TYPE_CHECKING

from .acl import CallerWorkspace, WorkspaceScope, forbidden_payload, raw_workspace
from .constants import SEMANTIC_SCAN_ENHANCE, SEMANTIC_SCAN_MAX_PAIRS
from .db_generation import CONFLICT_DETECTOR_VERSION
from .models import MemoryStatus, ProtectionLevel, SourceType
from .request_identity import get_request_identity
from .scan_tasks import SCHEDULED_TASKS_TOPIC, scheduled_tasks_help
from .validation import PRODUCT_FIELD_REGISTRY, _controlled_integer, validate_product_payload

if TYPE_CHECKING:
    from .tools import MemoryTools


AGENT_ONBOARDING_TOPIC = "agent_onboarding"


def _agent_onboarding_guide() -> str:
    try:
        return resources.files("memory_arbiter").joinpath("AGENT_ONBOARDING.md").read_text(encoding="utf-8")
    except Exception:
        return (
            "mema / Memory Arbiter: use MCP tools for memory operations and governance. "
            "Use memory(action='remember'|'find'|'read'|'update'|'judge'), memory_review for read-only inspection, "
            "memory_govern only for explicit user-authorized governance, and memory_repair for maintenance. "
            "Do not infer conflicts away; if a response says attention_required or action_required "
            "(e.g. read_semantic_notice, ask_user_for_authorization, confirm_new_workspace), handle it before relying on the memory."
        )


def _memory_value_reference() -> dict[str, Any]:
    return {
        "source_type": [item.value for item in SourceType],
        "protection_level": [item.value for item in ProtectionLevel],
        "memory_status_note": (
            "Lifecycle values a record can carry (seen on reads). On remember, "
            "status accepts only 'active' (the default - omit it) or 'pending'; "
            "strict isolation sets pending internally until confirm_pending_workspace "
            "activates the memory. superseded/conflicted/deleted are rejected as write inputs."
        ),
        "memory_status": [item.value for item in MemoryStatus],
        "update_modes": {
            "replace_content": "memory_id plus new_content, optionally new_subject/new_tags/add_tags/remove_tags/reason.",
            "replace_text": "memory_id plus old_text and new_text, optionally new_subject/new_tags/add_tags/remove_tags/reason.",
            "tags_only": "memory_id plus tags_only=true with add_tags and/or remove_tags; content is unchanged.",
        },
    }


# One shared help-document instance (#9): the literal's only dynamic input is
# the (now module-level) _memory_value_reference, so it is built once here
# instead of on every _product_help call. Every mutation path in
# _product_help copies via dict() before adding keys, so callers can never
# mutate this shared instance.
_PRODUCT_HELPS: dict[str, Any] = {
    "memory": {
        "description": "Daily memory operations: remember, find, read, update, judge, status.",
        "actions": ["remember", "find", "read", "update", "judge", "status", "help"],
        "examples": {
            "remember": {"action": "remember", "data": {"content": "Fact to remember", "subject": "Short subject", "tags": ["project"]}},
            "find": {"action": "find", "data": {"query": "project decision", "limit": 5}},
            "read": {"action": "read", "data": {"memory_id": 123}},
            "update": {"action": "update", "data": {"memory_id": 123, "new_content": "Updated current fact", "reason": "User provided a newer source-of-truth."}},
            "judge": {"action": "judge", "data": {"conflict_id": 1, "expected_revision": 1, "chosen_value": "SQLite", "decided_by": "user", "ref": "chat", "reason": "User confirmed the current database.", "apply_plan": [{"memory_id": 12, "action": "update_current_claim"}, {"memory_id": 34, "action": "use_as_resolution"}], "resolution_memory_id": 34}},
        },
        "source_of_truth_rule": "When a user says a new document replaces the current source of truth, find/read the existing current memory and update it; do not create a second active memory or retire the old one unless the user explicitly asks for whole-memory retirement.",
        "write_duplicate_hint": (
            "remember responses may carry a similar_active_memory notice when the new subject/tags "
            "closely match an existing active memory (subject ratio >=0.95 AND tag Jaccard >=0.8). "
            "Triage it silently: ignore deliberate series entries, prefer updating the original on "
            "true duplicates, and ask the user only when retiring/merging (governance) is needed."
        ),
        "find_size_metering": (
            "find responses carry a size block (returned_chars/returned_count, tokens_estimate, "
            "matched_beyond_limit_*) plus the caller-scope unresolved_conflict_count; disable with "
            "include_size=false. tokens_estimate uses a deterministic bucket estimator "
            "(heuristic_v1, Qwen2.5-calibrated): pure Chinese prose runs ~30% high and pure English "
            "~17% high — the estimate and the estimated share one yardstick, so savings comparisons stay valid."
        ),
        "value_reference": _memory_value_reference(),
    },
    "memory_review": {
        "description": "Read-only inspection. Never changes memory state.",
        "views": ["overview", "doctor", "conflicts", "conflict_detail", "history", "expired", "audit", "entities", "help"],
        "examples": {
            "conflicts": {"view": "conflicts", "data": {"status": "open", "limit": 20}},
            "history": {"view": "history", "data": {"memory_id": 123}},
            "expired": {"view": "expired", "data": {"query": "old decision", "limit": 10}},
        },
    },
    "memory_govern": {
        "description": "Explicit user-authorized governance. Every state-changing action requires authorized=true after the user confirms that specific action. Do not use for ordinary source-of-truth updates; use memory(action='update') instead.",
        "actions": ["retire", "merge_memories", "apply_conflict_action", "replan_conflict", "resolve_conflict", "confirm", "rename_workspace_canonical", "migrate_workspace", "move_memories_workspace", "separate_workspace_alias", "confirm_pending_workspace", "confirm_workspaces", "help"],
        "examples": {
            "retire": {"action": "retire", "data": {"memory_id": 123, "superseded_by": 456, "reason": "User explicitly requested retiring the old whole memory.", "authorized": True}},
            "merge_memories": {"action": "merge_memories", "data": {"survivor_id": 456, "loser_ids": [123, 124], "reason": "Same fact recorded twice; keeping the newer record.", "authorized": True}},
            "merge_memories_with_content": {"action": "merge_memories", "data": {"survivor_id": 456, "loser_ids": [123], "merged_content": "Combined statement retaining every unique detail from both records.", "reason": "User confirmed the combined wording.", "authorized": True}},
            "separate_workspace_alias": {"action": "separate_workspace_alias", "data": {"alias": "旧项目名", "canonical": "新项目名", "reason": "User confirmed the two workspaces must stay separate.", "authorized": True}},
            "apply_conflict_action": {"action": "apply_conflict_action", "data": {"conflict_id": 1, "expected_revision": 2, "memory_id": 12, "action": "update_current_claim", "content": "The database is SQLite.", "reason": "Apply the confirmed conflict decision.", "authorized": True}},
            "resolve_conflict": {"action": "resolve_conflict", "data": {"conflict_id": 1, "expected_revision": 4, "reason": "All planned member actions completed.", "authorized": True}},
            "rename_workspace_canonical": {"action": "rename_workspace_canonical", "data": {"old": "旧项目名", "new": "新项目名", "reason": "User confirmed the rename.", "authorized": True}},
            "migrate_workspace": {"action": "migrate_workspace", "data": {"from": "金营二期", "to": "金营项目", "reason": "User confirmed the merge.", "authorized": True}},
            "move_memories_workspace": {"action": "move_memories_workspace", "data": {"memory_ids": [123, 124], "new_workspace": "金营项目", "reason": "User confirmed these memories belong to the project bucket.", "authorized": True}},
            "confirm_pending_workspace": {"action": "confirm_pending_workspace", "data": {"memory_id": 123, "canonical": "金营项目", "authorized": True}},
            "confirm_workspaces": {"action": "confirm_workspaces", "data": {"reason": "Reviewed the registry after renaming duplicates; snapshots the current registry.", "authorized": True}},
        },
        "safety_note": "Set authorized=true only after the user explicitly confirms the specific governance action. Retire only whole memories; for partial updates or current-document replacement, update the existing memory instead.",
        "workspace_move_vs_migrate": (
            "migrate_workspace merges one whole canonical workspace into another by "
            "name and reroutes the old name through an alias; move_memories_workspace "
            "moves selected memories by id to another workspace bucket (both workspace "
            "columns) and leaves alias/normalization rules untouched. Moving does not "
            "change memory status: pending memories stay pending until activated via "
            "confirm_pending_workspace, and superseded/deleted rows keep their status "
            "(reported via moved_non_active)."
        ),
        "authorization_rule": "All state-changing actions require authorized=true. Without it, the response returns action_required=ask_user_for_authorization and an impact description.",
        "confirm_actions": {
            "confirm": "Promote one memory to user_confirmed and lock it against ordinary changes.",
            "confirm_pending_workspace": (
                "Confirm a new canonical workspace under strict isolation and activate its pending memory."
            ),
            "confirm_workspaces": (
                "Record the reviewed workspace snapshot after rename/merge cleanup. "
                "Omit workspaces to snapshot the current registry and clear workspace.review; "
                "an explicit subset confirms only those names, so other current names remain warnings."
            ),
        },
    },
    "memory_repair": {
        "description": "Maintenance and repair operations. Prefer dry_run first; cleanup, activation, and protected-memory metadata changes still require authorized=true when the underlying operation requires it.",
        "tasks": ["rebuild_evidence", "scan_candidates", "cleanup_history", "set_entity", "activate_pending", "replay_backup", "normalize_workspaces", "semantic_control", "notice", "record_conflict", "help"],
        "examples": {
            "rebuild_evidence": {"task": "rebuild_evidence", "data": {"dry_run": True, "memory_ids": [123]}},
            "set_entity": {"task": "set_entity", "data": {"memory_id": 123, "entity": "project-x", "scope": "charter"}},
            "semantic_control": {"task": "semantic_control", "data": {"action": "status"}},
            "replay_backup": {"task": "replay_backup", "data": {"dry_run": True}},
            "normalize_workspaces": {"task": "normalize_workspaces", "data": {"dry_run": True}},
            "normalize_workspaces_apply": {"task": "normalize_workspaces", "data": {"dry_run": False, "authorized": True}},
            "record_conflict": {"task": "record_conflict", "data": {"slot_key": {"entity": "project-x", "attribute": "database", "scope": "production"}, "members": [{"memory_id": 12, "version": 1, "attribute_raw": "database", "value_raw": "MySQL", "normalized_attribute": "database", "normalized_value": "mysql", "evidence_quote": "database is MySQL", "evidence_span": [0, 17], "content_hash": "0000000000000000000000000000000000000000000000000000000000000000", "direction": "a_to_b", "prompt_version": "p1", "detector_version": "d1"}, {"memory_id": 34, "version": 1, "attribute_raw": "database", "value_raw": "SQLite", "normalized_attribute": "database", "normalized_value": "sqlite", "evidence_quote": "database is SQLite", "evidence_span": [0, 18], "content_hash": "1111111111111111111111111111111111111111111111111111111111111111", "direction": "b_to_a", "prompt_version": "p1", "detector_version": "d1"}], "value_groups": [{"normalized_value": "mysql", "display_value": "MySQL", "members": ["12@1"]}, {"normalized_value": "sqlite", "display_value": "SQLite", "members": ["34@1"]}], "status": "open", "detector_version": "d1", "prompt_version": "p1", "source": "scheduled_scan", "reason": "Reviewed conflicting values."}},
            "scan_candidates": {"task": "scan_candidates", "data": {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_check": False}},
            "scan_candidates_duplicates": {"task": "scan_candidates", "data": {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_duplicates": True}},
            "notice": {"task": "notice", "data": {"action": "list", "status": "open", "limit": 5}},
            "notice_read": {"task": "notice", "data": {"action": "read", "notice_id": 1}},
            "notice_dismiss": {"task": "notice", "data": {"action": "dismiss", "notice_id": 1, "reason": "Reviewed; not actionable."}},
            "notice_resolve": {"task": "notice", "data": {"action": "resolve", "notice_id": 1, "reason": "Reviewed and handled."}},
            "notice_escalate": {"task": "notice", "data": {"action": "escalate", "notice_id": 1, "reason": "Verified against both memories: real contradiction needing governance."}},
        },
        "semantic_notice_delivery": "Notices progress pending -> delivered while open, then dismissed/resolved, or stale when any frozen member is no longer active at its pinned version. Read requires freshness.fresh=true and executing every read_calls entry for complete memories before triage; two-member notices also expose optional left/right aliases. Dismiss a false positive, resolve a handled one, or escalate a verified contradiction into a formal conflict.",
        "checked_no_notice": "A completed semantic task with outcome=checked_no_notice examined its eligible candidates and emitted zero notices; it is not a claim that no conflict can exist outside that task snapshot or candidate budget.",
        "normalize_workspaces_scope": (
            "normalize_workspaces is a GLOBAL registry operation: it takes no workspace "
            "filter and folds spelling-variant canonicals across the whole registry. "
            "Under strict isolation it still requires a resolvable caller workspace "
            "(settings.workspace), the same ACL gate as scan_candidates/record_conflict."
        ),
        "semantic_control_actions": [
            "status", "pause", "resume", "enable", "unload", "disable",
        ],
    },
}


class ProductSurfaces:
    _GOVERNANCE_IMPACTS: dict[str, str] = {
        "retire": "Marks a whole memory superseded and removes it from active recall.",
        "resolve_conflict": "Marks an applying conflict resolved after every planned member action completed.",
        "apply_conflict_action": "Atomically applies one planned member change and records its result in the conflict.",
        "replan_conflict": "CAS-replaces a stale or failed applying plan while preserving prior plan history.",
        "confirm": "Promotes the memory to user_confirmed and locks it against ordinary changes.",
        "rename_workspace_canonical": "Renames a canonical workspace and reroutes all affected memories.",
        "migrate_workspace": "Bulk-moves memories to another canonical workspace and records the alias.",
        "move_memories_workspace": "Moves the selected memories by id to another workspace bucket (both workspace columns); alias and normalization rules are not changed. Rows whose canonical already diverges from their bucket (e.g. rows written through a confirmed alias) are re-anchored to the destination when authorized.",
        "confirm_pending_workspace": "Assigns the canonical workspace and activates the pending memory for recall.",
        "confirm_workspaces": "Records the reviewed workspace registry snapshot that doctor's workspace.review diffs against; unconfirmed new workspaces keep the check warning.",
        "record_conflict": "Records a not_a_conflict disposition that suppresses future detection of the same candidate; ordinary open conflict intake does not require authorization.",
        "merge_memories": "Merges near-duplicate memories: keeps the survivor (optionally replacing its content with merged_content), supersedes the losers with a persistent merged_into pointer, and leaves conflict groups untouched (losers that are members of open/applying groups are rejected per-id).",
        "separate_workspace_alias": "Undoes an installed alias redirect or records a keep-separate decision for two workspaces; reversing it later requires the confirm side to pass force=true.",
    }

    def __init__(self, tools: "MemoryTools"):
        self._tools = tools
        self.db = tools.db
        self.settings = tools.settings

    def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace":
        return self._tools._caller_workspace(*args, **kwargs)

    def _conflict_detail_for_workspace(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._conflict_detail_for_workspace(*args, **kwargs)

    def _get_memory_visible(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._get_memory_visible(*args, **kwargs)

    def _payload_dict(self, data: "dict[str, Any] | None") -> dict[str, Any]:
        return self._tools._payload_dict(data)

    def _semantic_control_with_timeout(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._tools._semantic_control_with_timeout(*args, **kwargs)

    def _strict_acl_unavailable(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None":
        return self._tools._strict_acl_unavailable(*args, **kwargs)

    def memory_audit_summary(self, **kwargs: Any) -> dict[str, Any]:
        return self._tools.memory_audit_summary(**kwargs)

    def memory_history(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._tools.memory_history(*args, **kwargs)

    def memory_status(self, **kwargs: Any) -> dict[str, Any]:
        return self._tools.memory_status(**kwargs)

    @staticmethod
    def _judge_constraints() -> dict[str, Any]:
        return {
            "decided_by": ["user", "agent"],
            "apply_actions": [
                "update_current_claim", "append_superseded_context",
                "preserve_historical_record", "use_as_resolution", "needs_authorization",
            ],
            "rules": [
                "expected_revision must match the current conflict revision.",
                "apply_plan contains each planned member at most once.",
                "resolution_memory_id is required and must identify an active memory; prefer the existing correct member as the resolution.",
                "Each plan step is applied sequentially with memory_govern(action='apply_conflict_action').",
            ],
        }

    @staticmethod
    def _judge_required_fields() -> list[str]:
        return [
            "conflict_id", "expected_revision", "chosen_value", "decided_by",
            "ref", "reason", "apply_plan", "resolution_memory_id",
        ]

    @staticmethod
    def _action_required_paths() -> dict[str, str]:
        return {
            "read_semantic_notice": (
                "Read the notice, require freshness.fresh=true, execute every returned read_calls entry, and "
                "assess every complete frozen member before triage. Dismiss a false positive; after the credible "
                "notice has been handled, resolve the notice. A notice is not a formal conflict and cannot be "
                "judged or passed to resolve_conflict."
            ),
            "ask_user": (
                "Governance-impacting decisions should be confirmed with the user. decided_by="
                "'agent' is a valid recorded provenance when the agent has authorization, but a "
                "formal conflict judgment with real impact should still be presented to the user."
            ),
            "judge_conflict": (
                "Read memory_review(view='conflict_detail') with every member memory, present the common "
                "slot and value groups, then call memory(action='judge') with the current revision and a "
                "per-member apply plan."
            ),
            "replan_conflict": (
                "A plan step failed or a member changed mid-apply. Re-read the group and members, then "
                "call authorized memory_govern(action='replan_conflict') with the current revision and a "
                "replacement plan; prior plan history is preserved."
            ),
            "confirm_new_workspace": (
                "Explain the proposed canonical workspace and ask the user to authorize confirmation. After "
                "approval call memory_govern(action='confirm_pending_workspace') with authorized=true."
            ),
            "review_workspace_registry": (
                "Run the notice's memory_review(view='doctor') call, inspect the complete workspace "
                "registry for duplicates, and rename or migrate any duplicates first. Then ask the user "
                "to authorize memory_govern(action='confirm_workspaces'); only after approval add "
                "authorized=true to the notice's confirm_call."
            ),
            "ask_user_for_authorization": (
                "Explain the returned impact and ask the user to authorize that specific governance action. "
                "Authorization is mandatory; only after approval add authorized=true to the returned retry call."
            ),
            "apply_conflict_action": (
                "Inspect conflict_detail and the pending plan step. Obtain explicit user authorization, add "
                "authorized=true, then execute next_executable_call."
            ),
            "preview_backup_replay": "Execute suggested_call to preview pending backup records; do not apply them during preview.",
            "inspect_backup_replay_manually": "Execute suggested_call and inspect the dry-run result because automatic notice inspection degraded.",
        }

    @staticmethod
    def _field_reference(surface: str) -> dict[str, list[str]]:
        return {
            operation: sorted(fields)
            for (registered_surface, operation), fields in PRODUCT_FIELD_REGISTRY.items()
            if registered_surface == surface and not operation.startswith("_")
        }

    def _product_help(self, surface: str, topic: str | None = None) -> dict[str, Any]:
        # The document bodies live in the module-level _PRODUCT_HELPS constant
        # (#9); every mutation path below copies via dict() first.
        helps = _PRODUCT_HELPS
        if topic == AGENT_ONBOARDING_TOPIC:
            return {
                "description": "Agent onboarding guide for using mema / Memory Arbiter correctly.",
                "topic": AGENT_ONBOARDING_TOPIC,
                "notice": "agent-onboarding:v1",
                "guide_file": "memory_arbiter/AGENT_ONBOARDING.md",
                "content": _agent_onboarding_guide(),
            }
        if topic == SCHEDULED_TASKS_TOPIC:
            return scheduled_tasks_help()
        help_doc = helps.get(surface, {"description": "Unknown product surface."})
        if surface in {"memory", "memory_govern"} and isinstance(help_doc, dict):
            help_doc = dict(help_doc)
            help_doc["judge_constraints"] = self._judge_constraints()
        if surface == "memory" and isinstance(help_doc, dict):
            help_doc = dict(help_doc)
            help_doc["judge_required_fields"] = self._judge_required_fields()
        if isinstance(help_doc, dict):
            help_doc = dict(help_doc)
            help_doc["action_required_paths"] = self._action_required_paths()
            help_doc["accepted_fields"] = self._field_reference(surface)
        if surface == "memory_repair" and isinstance(help_doc, dict):
            help_doc = dict(help_doc)
            help_doc.setdefault("semantic_control_actions", [
                "status", "pause", "resume", "enable", "unload", "disable",
            ])
        # helps.get returns Any-typed values from the module constant; narrow
        # once so the return contract stays dict for strict mypy.
        if isinstance(help_doc, dict):
            if topic:
                narrowed = dict(help_doc)
                narrowed["requested_topic"] = topic
                return narrowed
            return help_doc
        return {"description": str(help_doc)}

    def _invalid_product_call(self, surface: str, message: str, topic: str | None = None) -> dict[str, Any]:
        return self.db.state.response(
            {"error": message, "help": self._product_help(surface, topic)},
            ok=False,
        )

    @staticmethod
    def _help_topic(payload: dict[str, Any], fallback_key: str) -> str | None:
        return payload.get("topic") or payload.get(fallback_key)

    def _forward(
        self, surface: str, topic: str | None, fn: Callable[..., dict[str, Any]], **payload: Any,
    ) -> dict[str, Any]:
        """Forward ``**payload`` to a low-level method with a product-surface guard.

        Low-level methods coerce their own int/bool args (``limit``, ``superseded_by``,
        ``older_than_days``, …). Some validate and return ``ok=False``; others let
        ``int()`` raise ``ValueError`` / ``TypeError`` straight through. An MCP
        client sending loosely-typed JSON (``"limit": "5"``) must never get a raw
        exception, so catch those two and surface a structured ``ok=False`` error
        instead. The primary-id guard (``_require_id`` / ``_coerce_product_id``)
        still runs first so the id-specific message is preserved.
        """
        try:
            return fn(**payload)
        except (TypeError, ValueError) as exc:
            # Turn Python's "invalid literal for int() with base 10: 'x'" into a
            # compact, agent-readable message. The offending value is already in
            # the message; we just drop the implementation noise.
            detail = str(exc).replace("invalid literal for int() with base 10: ", "not an integer: ")
            return self._invalid_product_call(
                surface, f"invalid argument — {detail}", topic,
            )

    @staticmethod
    def _alias_id(payload: dict[str, Any], target: str) -> None:
        if target not in payload and "id" in payload:
            payload[target] = payload.pop("id")

    def _int_product_arg(
        self, surface: str, value: Any, name: str, topic: str | None = None,
    ) -> int | dict[str, Any] | None:
        parsed = _controlled_integer(value)
        if parsed is None:
            return self._invalid_product_call(surface, f"{name} must be an integer", topic)
        return parsed

    def _require_id(
        self, surface: str, payload: dict[str, Any], name: str, topic: str | None = None,
    ) -> dict[str, Any] | None:
        """Guard a forward whose target has a required positional ``name``.

        Product tools forward ``**payload`` to low-level methods. When the agent
        omits the id entirely, the underlying signature (e.g. ``memory_get(memory_id: int)``)
        raises ``TypeError: missing required argument`` before its own validation
        runs. Return an ``ok=False`` payload there instead, so the contract stays
        the same as every other bad-input path.
        """
        if name not in payload:
            return self._invalid_product_call(surface, f"{topic or surface} requires {name}", topic)
        return None

    def _coerce_product_id(
        self, surface: str, payload: dict[str, Any], name: str, topic: str | None = None,
    ) -> dict[str, Any] | None:
        self._alias_id(payload, name)
        missing = self._require_id(surface, payload, name, topic)
        if missing is not None:
            return missing
        coerced = self._int_product_arg(surface, payload.get(name), name, topic)
        if isinstance(coerced, dict):
            return coerced
        payload[name] = coerced
        return None

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        """Robust truthiness for a loosely-typed JSON authorization flag.

        A JSON client may send the *string* "false" for a boolean; bool("false")
        is True in Python, which would silently grant an override. For an
        authorization flag the safe default is an ALLOW-LIST: only genuine
        booleans and explicit true-tokens grant it. Any other string
        ("false", "null", "maybe", "") → False.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        if isinstance(value, (int, float)):
            return value != 0
        return False

    def _require_ws_strings(
        self, payload: dict[str, Any], names: tuple[str, ...], surface: str, topic: str | None = None,
    ) -> dict[str, Any] | None:
        """Reject non-string workspace fields with a structured error.

        Loosely-typed MCP JSON can pass a list/dict/int; str()-coercing those
        would silently store a garbage canonical like "['x']". A workspace name
        must be a genuine string — anything else is a client error.
        """
        for name in names:
            val = payload.get(name)
            if val is not None and not isinstance(val, str):
                return self._invalid_product_call(
                    surface,
                    f"{name} must be a string workspace name, got {type(val).__name__}",
                    topic,
                )
        return None

    def _normalize_boolean_fields(
        self, payload: dict[str, Any], *names: str,
    ) -> None:
        """Normalize loosely typed MCP booleans with an explicit allow-list."""
        for name in names:
            if name in payload:
                payload[name] = self._is_truthy(payload[name])

    def _validated_product_call(
        self,
        surface: str,
        operation: str,
        data: dict[str, Any] | None,
        dispatch: Callable[[str, dict[str, Any] | None], dict[str, Any]],
    ) -> dict[str, Any]:
        if data is not None and not isinstance(data, dict):
            return dispatch(operation, data)
        payload = dict(data or {})
        validation = validate_product_payload(surface, operation, payload)
        if validation.error is not None:
            return self.db.state.response(validation.error, ok=False)
        response = dispatch(operation, payload)
        if validation.warnings:
            warnings = response.setdefault("warnings", [])
            for warning in validation.warnings:
                if warning not in warnings:
                    warnings.append(warning)
            response["degraded"] = bool(warnings) or response.get("mode") != "sqlite_vec"
        return response

    def _notice_workspace_scope(
        self, response: dict[str, Any], data: dict[str, Any] | None,
    ) -> "WorkspaceScope":
        """Scope automatic notice delivery like every other strict read.

        delivery uses the full admitted set, not only the response's
        single ``caller_workspace_canonical`` field. The returned notice's retry
        payload still echoes one valid workspace string (the caller canonical).
        """
        if self.settings.isolation != "strict":
            return None
        cached = self._tools._product_caller.get()
        if cached is not None and cached.isolation == "strict":
            return cached.scope_canonicals()
        raw = data.get("workspace") if isinstance(data, dict) else None
        return self._caller_workspace(raw).scope_canonicals()

    def _deliver_product_notices(
        self, response: dict[str, Any], data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Attach notices only after success, scoped to this product caller."""
        if not response.get("ok"):
            return response
        notices: list[dict[str, Any]] = []
        # Consume existing update/onboarding/backup notices without slicing them;
        # semantic delivery has its own one-per-response limit.
        try:
            notices.extend(self._tools._consume_notices())
        except Exception:
            pass
        try:
            semantic = self.db.claim_next_semantic_notice(
                self._notice_workspace_scope(response, data),
            )
        except Exception as exc:
            from .models import utc_now_iso
            semantic = None
            self._tools._notice_claim_error_count += 1
            self._tools._notice_claim_last_error = str(exc)
            self._tools._notice_claim_last_error_at = utc_now_iso()
            warning = f"semantic_notice_claim_failed: {exc}"
            if warning not in response.setdefault("warnings", []):
                response["warnings"].append(warning)
            response["degraded"] = True
        if semantic is not None:
            notices.append(semantic)
        if notices:
            response.setdefault("notices", []).extend(notices)
        return response

    def memory(self, action: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        operation = str(action or "help").strip().lower()
        self._tools._product_caller.set(None)
        response = self._validated_product_call("memory", operation, data, self._memory)
        return self._deliver_product_notices(response, data)

    def memory_review(self, view: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        operation = str(view or "help").strip().lower()
        self._tools._product_caller.set(None)
        response = self._validated_product_call("memory_review", operation, data, self._memory_review)
        return self._deliver_product_notices(response, data)

    def memory_govern(self, action: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        operation = str(action or "help").strip().lower()
        self._tools._product_caller.set(None)
        response = self._validated_product_call("memory_govern", operation, data, self._memory_govern)
        return self._deliver_product_notices(response, data)

    def memory_repair(self, task: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        operation = str(task or "help").strip().lower()
        self._tools._product_caller.set(None)
        response = self._validated_product_call("memory_repair", operation, data, self._memory_repair)
        return self._deliver_product_notices(response, data)

    def _governance_authorization_required(
        self, action: str, *, tool: str = "memory_govern", retry_field: str = "action",
    ) -> dict[str, Any]:
        return self.db.state.response(
            {
                "error": "explicit user authorization required",
                "action_required": "ask_user_for_authorization",
                "governance_action": action,
                "impact": self._GOVERNANCE_IMPACTS[action],
                "authorized": False,
                "retry": {
                    "tool": tool,
                    retry_field: action,
                    "set_after_user_confirmation": {"authorized": True},
                },
            },
            ok=False,
        )

    def _governance_authorization_error(
        self, action: str, payload: dict[str, Any],
        *, tool: str = "memory_govern", retry_field: str = "action",
    ) -> dict[str, Any] | None:
        if not payload.get("authorized"):
            return self._governance_authorization_required(action, tool=tool, retry_field=retry_field)
        return None

    def _memory(self, action: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        """Task-oriented daily memory tool: remember/find/read/update/judge/status.

        Use help when unsure about fields. For current source-of-truth updates,
        find/read the existing memory and update it; do not create a duplicate
        active memory or retire the old one unless the user explicitly requests
        whole-memory retirement.
        """
        payload = self._payload_dict(data)
        if data is not None and not isinstance(data, dict):
            return self._invalid_product_call("memory", "data must be a JSON object", action)
        action = str(action or "help").strip().lower()
        self._normalize_boolean_fields(
            payload, "authorized", "tags_only", "debug_ranking",
            "include_linked_open_items", "include_conflict_signal",
            "include_size",
            "affects_current_output",
        )
        if action == "help":
            return self.db.state.response(self._product_help("memory", self._help_topic(payload, "action")))
        if action == "remember":
            return self._forward("memory", action, self._tools.memory_write, **payload)
        if action == "find":
            return self._forward("memory", action, self._tools.memory_search, **payload)
        if action == "read":
            self._alias_id(payload, "memory_id")
            missing = self._require_id("memory", payload, "memory_id", action)
            if missing is not None:
                return missing
            return self._forward("memory", action, self._tools.memory_get, **payload)
        if action == "update":
            self._alias_id(payload, "memory_id")
            missing = self._require_id("memory", payload, "memory_id", action)
            if missing is not None:
                return missing
            return self._forward("memory", action, self._tools.memory_edit, **payload)
        if action == "judge":
            required = self._judge_required_fields()
            if "conflict_id" not in payload and "id" in payload:
                payload["conflict_id"] = payload.pop("id")
            missing_fields = [name for name in required if name not in payload]
            if missing_fields:
                help_doc: dict[str, Any] = self._product_help("memory", "judge")
                help_doc["required_fields"] = required
                help_doc["missing_fields"] = missing_fields
                return self.db.state.response({"error": "judge missing required fields", "help": help_doc}, ok=False)
            # Coerce conflict_id after the missing-fields check so a non-integer
            # id gets its own clear error instead of a generic invalid_input
            # deep inside submit_conflict_judgment.
            invalid_id = self._coerce_product_id("memory", payload, "conflict_id", action)
            if invalid_id is not None:
                return invalid_id
            return self._forward("memory", action, self._tools._operations.memory_judge_conflict, **payload)
        if action == "status":
            return self.memory_status(workspace=payload.get("workspace"))
        return self._invalid_product_call("memory", f"unknown action: {action}", action)

    def _memory_review(self, view: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        """Read-only memory inspection: health, conflicts, history, expired recall, audit, and entities."""
        payload = self._payload_dict(data)
        if data is not None and not isinstance(data, dict):
            return self._invalid_product_call("memory_review", "data must be a JSON object", view)
        view = str(view or "help").strip().lower()
        self._normalize_boolean_fields(
            payload, "deep", "debug_ranking", "include_unassigned",
            "include_conflict_signal",
        )
        if view == "help":
            return self.db.state.response(self._product_help("memory_review", self._help_topic(payload, "view")))
        if view == "overview":
            return self.db.state.response({
                "status": self.memory_status(workspace=payload.get("workspace")).get("data"),
                "audit": self.memory_audit_summary(**payload).get("data"),
            })
        if view == "doctor":
            return self._forward("memory_review", view, self._tools.memory_doctor_overview, **payload)
        if view == "audit":
            return self._forward("memory_review", view, self.memory_audit_summary, **payload)
        if view == "conflicts":
            return self._forward("memory_review", view, self._tools.memory_list_conflicts, **payload)
        if view == "conflict_detail":
            conflict_id = payload.get("conflict_id") or payload.get("id")
            if conflict_id is None:
                return self._invalid_product_call("memory_review", "conflict_detail requires conflict_id", view)
            conflict_id_int = self._int_product_arg("memory_review", conflict_id, "conflict_id", view)
            if isinstance(conflict_id_int, dict):
                return conflict_id_int
            caller = self._caller_workspace(payload.get("workspace"))
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
            detail = self._conflict_detail_for_workspace(conflict_id_int, caller)
            if detail is None:
                data = {"error": "conflict id not found"}
                if caller.isolation == "strict":
                    data.update(caller.response_fields())
                return self.db.state.response(data, ok=False, extra_warnings=list(caller.warnings))
            return self.db.state.response(detail, extra_warnings=list(caller.warnings))
        if view == "history":
            memory_id = payload.get("memory_id") or payload.get("id")
            if memory_id is None:
                return self._invalid_product_call("memory_review", "history requires memory_id", view)
            memory_id_int = self._int_product_arg("memory_review", memory_id, "memory_id", view)
            if isinstance(memory_id_int, dict):
                return memory_id_int
            return self.memory_history(memory_id=memory_id_int, workspace=payload.get("workspace"))
        if view == "expired":
            return self._forward("memory_review", view, self._tools.memory_search_expired, **payload)
        if view == "entities":
            return self._forward("memory_review", view, self._tools.memory_list_entities, **payload)
        return self._invalid_product_call("memory_review", f"unknown view: {view}", view)

    def _memory_govern(self, action: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        """Explicit user-authorized governance: retire, merge near-duplicates, apply/replan/resolve conflict plans, confirm, or manage workspaces.

        Do not use this for ordinary updates or current source-of-truth replacement;
        use memory(action="update") for those. Retire is only for whole-memory
        retirement after explicit user authorization; merge_memories is for
        near-duplicate whole memories, with losers rejected per-id when they sit
        in an open/applying conflict group.
        """
        payload = self._payload_dict(data)
        if data is not None and not isinstance(data, dict):
            return self._invalid_product_call("memory_govern", "data must be a JSON object", action)
        action = str(action or "help").strip().lower()
        self._normalize_boolean_fields(payload, "authorized")
        if action == "help":
            return self.db.state.response(self._product_help("memory_govern", self._help_topic(payload, "action")))
        if action == "retire":
            invalid_id = self._coerce_product_id("memory_govern", payload, "memory_id", action)
            if invalid_id is not None:
                return invalid_id
            if not payload.get("reason"):
                return self._invalid_product_call("memory_govern", "retire requires reason and authorized=true", action)
            if payload.get("superseded_by") is not None:
                superseded_by_int = self._int_product_arg("memory_govern", payload.get("superseded_by"), "superseded_by", action)
                if isinstance(superseded_by_int, dict):
                    return superseded_by_int
                payload["superseded_by"] = superseded_by_int
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward("memory_govern", action, self._tools.memory_supersede, **payload)
        if action in {"apply_conflict_action", "replan_conflict", "resolve_conflict"}:
            invalid_id = self._coerce_product_id("memory_govern", payload, "conflict_id", action)
            if invalid_id is not None:
                return invalid_id
            if payload.get("expected_revision") is None:
                response = self._invalid_product_call(
                    "memory_govern", f"{action} requires expected_revision", action,
                )
                response["data"]["outcome"] = "invalid_input"
                response["data"]["field"] = "expected_revision"
                return response
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            if action == "apply_conflict_action":
                return self._forward(
                    "memory_govern", action,
                    self._tools._operations.memory_apply_conflict_action, **payload,
                )
            if action == "replan_conflict":
                if not isinstance(payload.get("apply_plan"), list):
                    return self._invalid_product_call("memory_govern", "replan_conflict requires apply_plan", action)
                return self._forward(
                    "memory_govern", action,
                    self._tools._operations.memory_replan_conflict, **payload,
                )
            return self._forward(
                "memory_govern", action,
                self._tools._operations.memory_resolve_conflict, **payload,
            )
        if action == "confirm":
            invalid_id = self._coerce_product_id("memory_govern", payload, "memory_id", action)
            if invalid_id is not None:
                return invalid_id
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward("memory_govern", action, self._tools.memory_confirm, **payload)
        if action == "merge_memories":
            invalid_id = self._coerce_product_id("memory_govern", payload, "survivor_id", action)
            if invalid_id is not None:
                return invalid_id
            raw_losers = payload.get("loser_ids")
            if not isinstance(raw_losers, list) or not raw_losers:
                return self._invalid_product_call(
                    "memory_govern", "merge_memories requires loser_ids: a non-empty list of memory ids", action,
                )
            coerced_losers: list[int] = []
            for item in raw_losers:
                coerced = self._int_product_arg("memory_govern", item, "loser_ids", action)
                if isinstance(coerced, dict):
                    return coerced
                if coerced is None:
                    return self._invalid_product_call(
                        "memory_govern", "loser_ids must contain integers only", action,
                    )
                coerced_losers.append(coerced)
            # Bound the DEDUPED set, mirroring the pipeline's own accounting:
            # 60 ids that collapse to 40 unique losers are one legal call.
            payload["loser_ids"] = sorted(set(coerced_losers))
            if len(payload["loser_ids"]) > 50:
                return self._invalid_product_call(
                    "memory_govern", "merge_memories accepts at most 50 unique loser_ids per call", action,
                )
            if not str(payload.get("reason") or "").strip():
                return self._invalid_product_call(
                    "memory_govern", "merge_memories requires reason and authorized=true", action,
                )
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward(
                "memory_govern", action,
                self._tools._operations.memory_merge_memories, **payload,
            )
        if action == "separate_workspace_alias":
            if not str(payload.get("alias") or "").strip() or not str(payload.get("canonical") or "").strip():
                return self._invalid_product_call(
                    "memory_govern", "separate_workspace_alias requires alias and canonical", action,
                )
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward(
                "memory_govern", action,
                self._tools._operations.memory_separate_workspace_alias, **payload,
            )
        if action in {"accept_workspace_alias", "reject_workspace_alias"}:
            alias = payload.get("alias")
            canonical = payload.get("canonical")
            replacements: list[dict[str, Any]] = []
            if action == "accept_workspace_alias":
                if isinstance(alias, str) and alias.strip() and isinstance(canonical, str) and canonical.strip():
                    replacements.extend([
                        {
                            "use_when": "merge the source workspace into the canonical and forward its old name",
                            "suggested_call": {
                                "tool": "memory_govern", "action": "migrate_workspace",
                                "data": {"from": alias, "to": canonical},
                            },
                            "authorization_required": True,
                        },
                        {
                            "use_when": "rename or merge a canonical workspace",
                            "suggested_call": {
                                "tool": "memory_govern", "action": "rename_workspace_canonical",
                                "data": {"old": alias, "new": canonical},
                            },
                            "authorization_required": True,
                        },
                    ])
                replacements.append({
                    "use_when": "activate a strict pending memory under the selected canonical",
                    "suggested_call": {
                        "tool": "memory_govern", "action": "confirm_pending_workspace",
                        "data": {"canonical": canonical} if isinstance(canonical, str) else {},
                    },
                    "required_input": ["memory_id"],
                    "authorization_required": True,
                })
            else:
                replacements.append({
                    "use_when": "keep the workspaces separate",
                    "suggested_call": None,
                    "note": "No pairwise governance call is needed.",
                })
            return self.db.state.response(
                {
                    "outcome": "removed",
                    "error_code": "workspace_alias_action_removed",
                    "removed_action": action,
                    "error": (
                        "pairwise workspace alias actions were removed; use workspace "
                        "rename/migration or pending confirmation instead"
                    ),
                    "replacements": replacements,
                },
                ok=False,
            )
        if action == "rename_workspace_canonical":
            if not payload.get("old") or not payload.get("new"):
                return self._invalid_product_call("memory_govern", "rename_workspace_canonical requires old and new", action)
            bad = self._require_ws_strings(payload, ("old", "new"), "memory_govern", action)
            if bad is not None:
                return bad
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward("memory_govern", action, self._tools.memory_rename_workspace_canonical, **payload)
        if action == "migrate_workspace":
            if not payload.get("from") or not payload.get("to"):
                return self._invalid_product_call("memory_govern", "migrate_workspace requires from and to", action)
            bad = self._require_ws_strings(payload, ("from", "to"), "memory_govern", action)
            if bad is not None:
                return bad
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward("memory_govern", action, self._tools.memory_migrate_workspace, **payload)
        if action == "move_memories_workspace":
            raw_ids = payload.get("memory_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                return self._invalid_product_call(
                    "memory_govern",
                    "move_memories_workspace requires memory_ids as a non-empty list of memory ids",
                    action,
                )
            if not payload.get("new_workspace"):
                return self._invalid_product_call(
                    "memory_govern", "move_memories_workspace requires new_workspace", action,
                )
            bad = self._require_ws_strings(payload, ("new_workspace",), "memory_govern", action)
            if bad is not None:
                return bad
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward("memory_govern", action, self._tools.memory_move_memories_workspace, **payload)
        if action == "confirm_pending_workspace":
            invalid_id = self._coerce_product_id("memory_govern", payload, "memory_id", action)
            if invalid_id is not None:
                return invalid_id
            if not payload.get("canonical"):
                return self._invalid_product_call("memory_govern", "confirm_pending_workspace requires memory_id and canonical", action)
            bad = self._require_ws_strings(payload, ("canonical",), "memory_govern", action)
            if bad is not None:
                return bad
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward("memory_govern", action, self._tools.memory_confirm_pending_workspace, **payload)
        if action == "confirm_workspaces":
            raw_list = payload.get("workspaces")
            if raw_list is not None and (
                not isinstance(raw_list, list)
                or not raw_list
                or any(not isinstance(item, str) or not item.strip() for item in raw_list)
            ):
                return self._invalid_product_call(
                    "memory_govern",
                    "confirm_workspaces workspaces must be a non-empty list of workspace "
                    "name strings (omit it to confirm the current registry snapshot)",
                    action,
                )
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward("memory_govern", action, self._tools.memory_confirm_workspaces, **payload)
        return self._invalid_product_call("memory_govern", f"unknown action: {action}", action)

    def _memory_repair(self, task: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        """Maintenance and repair for evidence, history, backup, notices, and runtime state."""
        payload = self._payload_dict(data)
        if data is not None and not isinstance(data, dict):
            return self._invalid_product_call("memory_repair", "data must be a JSON object", task)
        task = str(task or "help").strip().lower()
        self._normalize_boolean_fields(payload, "authorized", "dry_run", "clear")
        if task == "help":
            return self.db.state.response(self._product_help("memory_repair", self._help_topic(payload, "task")))
        if task == "rebuild_evidence":
            return self._forward("memory_repair", task, self._tools.memory_rebuild_evidence, **payload)
        if task == "scan_candidates":
            caller = self._caller_workspace(payload.get("workspace"))
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
            try:
                batch_value = int(payload["batch"]) if payload.get("batch") is not None else 50
                k_value = int(payload["k"]) if payload.get("k") is not None else 10
                anchor_value = (
                    int(payload["anchor_memory_id"])
                    if payload.get("anchor_memory_id") is not None else 0
                )
                distance_value = payload.get("max_distance")
                if distance_value is not None:
                    distance_value = float(distance_value)
            except (TypeError, ValueError):
                return self._invalid_product_call("memory_repair", "scan_candidates batch/k/anchor_memory_id must be integers and max_distance a number", task)
            if not (1 <= batch_value <= 200) or not (1 <= k_value <= 20) or anchor_value < 0:
                return self._invalid_product_call("memory_repair", "scan_candidates requires 1<=batch<=200, 1<=k<=20, anchor_memory_id>=0", task)
            scan_workspace = caller.scope_canonicals() if caller.isolation == "strict" else None
            scan_enhance = SEMANTIC_SCAN_ENHANCE
            scan_started = time.perf_counter()
            result = self.db.scan_rule_candidates(
                after_memory_id=anchor_value,
                anchor_batch=batch_value,
                neighbor_k=k_value,
                include_check=self._is_truthy(payload.get("include_check")),
                max_distance=distance_value,
                workspace=scan_workspace,
                similarity_pool_limit=(max(0, SEMANTIC_SCAN_MAX_PAIRS) if scan_enhance else 0),
                include_duplicates=self._is_truthy(payload.get("include_duplicates")),
            )
            if "error" not in result:
                # Spec §7.1 wide gate: bounded Qwen enhancement over the page.
                result = self._tools._enhance_scan_candidates(result)
            ok = "error" not in result
            if ok:
                # Unconditional activity record: the newest completed line in
                # scan_log.jsonl is the evidence that a scheduled scan task
                # exists (the guidance notice clears on it). Success-only by
                # design — a failing scan is not proof a task is running.
                identity = get_request_identity()
                scan_counts = result.get("counts") or {}
                self.db.log_scan(
                    duration_sec=time.perf_counter() - scan_started,
                    anchors_scanned=int(result.get("anchors_scanned") or 0),
                    candidates=len(result.get("candidates") or []),
                    knn_pairs=int(scan_counts.get("knn_pairs") or 0),
                    rule_pass=int(scan_counts.get("rule_pass") or 0),
                    next_anchor_memory_id=result.get("next_anchor_memory_id"),
                    truncated=bool(result.get("duplicates_truncated")),
                    client=(identity.client if identity else None),
                    agent_id=(identity.agent_id if identity else None),
                )
            scan_state = self.db.conflict_scan_state()
            if ok and scan_state.get("required"):
                # Compare the PERSISTED requirement against the RUNNING
                # detector identity, never the persisted echo: an old-detector
                # scan must not be able to clear the flag (spec §15.7/§15.8.24).
                progress_ok = self.db.record_conflict_scan_page(
                    epoch=str(scan_state.get("epoch") or ""),
                    detector_version=CONFLICT_DETECTOR_VERSION,
                    boundary=scan_state.get("boundary") or {},
                    after_memory_id=anchor_value,
                    next_anchor_memory_id=result.get("next_anchor_memory_id"),
                    anchors_scanned=int(result.get("anchors_scanned") or 0),
                    workspace=scan_workspace,
                )
                if progress_ok:
                    result["conflict_scan_progress"] = self.db.conflict_scan_state().get("progress")
                    if result.get("next_anchor_memory_id") is None:
                        result["conflict_scan_completed"] = self.db.complete_conflict_scan(
                            epoch=str(scan_state.get("epoch") or ""),
                            detector_version=CONFLICT_DETECTOR_VERSION,
                            boundary=scan_state.get("boundary") or {},
                        )
                else:
                    # A write between upgrade and scan completion drifts the
                    # live boundary and would otherwise wedge the flag. Re-arm
                    # against the current live set so a fresh full scan clears.
                    if self.db.rearm_conflict_scan_if_drifted():
                        result["conflict_scan_rearmed"] = True
                        result["conflict_scan_state"] = self.db.conflict_scan_state()
                    result["conflict_scan_progress_rejected"] = True
            return self.db.state.response(result, ok=ok, extra_warnings=list(caller.warnings))
        if task == "cleanup_history":
            if "id" in payload or "memory_id" in payload:
                invalid_id = self._coerce_product_id("memory_repair", payload, "memory_id", task)
                if invalid_id is not None:
                    return invalid_id
            return self._forward("memory_repair", task, self._tools.memory_cleanup_history, **payload)
        if task == "set_entity":
            invalid_id = self._coerce_product_id("memory_repair", payload, "memory_id", task)
            if invalid_id is not None:
                return invalid_id
            return self._forward("memory_repair", task, self._tools.memory_set_entity, **payload)
        if task == "activate_pending":
            invalid_id = self._coerce_product_id("memory_repair", payload, "memory_id", task)
            if invalid_id is not None:
                return invalid_id
            return self._forward("memory_repair", task, self._tools.memory_activate, **payload)
        if task == "replay_backup":
            return self._forward("memory_repair", task, self._tools.memory_replay_backup, **payload)
        if task == "normalize_workspaces":
            # normalize is a GLOBAL registry operation (see the help note):
            # the payload carries no workspace filter, so the strict-ACL gate
            # resolves the caller from settings only — the same two-line gate
            # as scan_candidates/record_conflict.
            caller = self._caller_workspace(None)
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
            return self._forward("memory_repair", task, self._tools.memory_normalize_workspaces, **payload)
        if task == "record_conflict":
            caller = self._caller_workspace(payload.get("workspace"))
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
            required = ["members", "value_groups", "detector_version", "source", "reason"]
            missing = [name for name in required if name not in payload]
            if missing:
                return self._invalid_product_call(
                    "memory_repair", f"record_conflict missing required fields: {', '.join(missing)}", task,
                )
            if not isinstance(payload.get("members"), list) or not isinstance(payload.get("value_groups"), list):
                return self._invalid_product_call("memory_repair", "members and value_groups must be arrays", task)
            # Authorization gate (B-C3): a not_a_conflict disposition suppresses
            # future detection of the same candidate, so it requires explicit
            # user authorization; ordinary open intake stays ungated — it is
            # the external reviewer's routine flow.
            if str(payload.get("status") or "open").strip().lower() == "not_a_conflict":
                auth_error = self._governance_authorization_error(
                    "record_conflict", payload, tool="memory_repair", retry_field="task",
                )
                if auth_error is not None:
                    return auth_error
            if caller.isolation == "strict":
                try:
                    member_ids = [int(member["memory_id"]) for member in payload["members"]]
                except (TypeError, ValueError, KeyError):
                    return self._invalid_product_call(
                        "memory_repair", "every conflict member requires an integer memory_id", task,
                    )
                visible_members = [
                    self._get_memory_visible(memory_id, caller) for memory_id in member_ids
                ]
                if not member_ids or any(memory is None for memory in visible_members):
                    return self.db.state.response(
                        forbidden_payload("conflict_members", workspace=caller),
                        ok=False, extra_warnings=list(caller.warnings),
                    )
                member_workspaces = {
                    raw_workspace(memory) for memory in visible_members if memory is not None
                }
                if len(member_workspaces) != 1:
                    return self._invalid_product_call(
                        "memory_repair",
                        "record_conflict members must belong to one admitted canonical workspace",
                        task,
                    )
                workspace_canonical = next(iter(member_workspaces))
            else:
                workspace_canonical = caller.canonical or str(
                    payload.get("workspace") or self.settings.workspace or ""
                ).strip()
            result = self.db.record_conflict_group(
                workspace_canonical=workspace_canonical,
                slot_key=payload.get("slot_key"),
                members=payload["members"],
                value_groups=payload["value_groups"],
                candidate_key=payload.get("candidate_key"),
                status=str(payload.get("status") or "open").strip().lower(),
                detector_version=str(payload["detector_version"]),
                prompt_version=payload.get("prompt_version"),
                source=str(payload["source"]),
                detection_reason=str(payload["reason"]),
                conflict_point=payload.get("conflict_point"),
                expected_revision=payload.get("expected_revision"),
            )
            ok = result.get("outcome") in {"inserted", "appended", "deduped"}
            if not ok:
                result.setdefault("error", f"record_conflict failed: {result.get('outcome')}")
            return self.db.state.response(result, ok=ok, extra_warnings=list(caller.warnings))
        if task == "semantic_control":
            timeout_raw = payload.get("timeout")
            timeout_value = 30.0 if timeout_raw is None else float(timeout_raw)
            result = self._semantic_control_with_timeout(
                str(payload.get("action") or "status"),
                timeout=timeout_value,
                workspace=payload.get("workspace"),
            )
            if result.get("outcome") == "invalid_action":
                result["error"] = "invalid semantic_control action"
                result["help"] = self._product_help("memory_repair", "semantic_control")
                return self.db.state.response(result, ok=False)
            return self.db.state.response(result)
        if task == "notice":
            action_value = payload.get("action", "list")
            if not isinstance(action_value, str):
                return self._invalid_product_call("memory_repair", "notice action must be a string", task)
            action = action_value.strip().lower()
            if action not in {"list", "read", "dismiss", "resolve", "escalate"}:
                return self._invalid_product_call("memory_repair", f"unknown notice action: {action}", task)
            caller = self._caller_workspace(payload.get("workspace"))
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
            workspace = caller.scope_canonicals() if caller.isolation == "strict" else None
            if action == "list":
                try:
                    notice_limit = int(payload.get("limit") or 10)
                except (TypeError, ValueError):
                    return self._invalid_product_call("memory_repair", "notice limit must be an integer", task)
                status_value = payload.get("status", "open")
                if not isinstance(status_value, str):
                    return self._invalid_product_call("memory_repair", "notice status must be a string", task)
                status = status_value.strip().lower()
                if status not in {"open", "dismissed", "resolved", "stale"}:
                    return self._invalid_product_call("memory_repair", f"invalid notice status: {status}", task)
                notices = self.db.list_semantic_notices(
                    status=status, limit=notice_limit, workspace_canonical=workspace,
                )
                return self.db.state.response({"notices": notices}, extra_warnings=list(caller.warnings))
            invalid_id = self._coerce_product_id("memory_repair", payload, "notice_id", task)
            if invalid_id is not None:
                return invalid_id
            if action == "read":
                notice = self.db.read_semantic_notice(int(payload["notice_id"]), workspace)
                if notice is None:
                    return self.db.state.response({"outcome": "not_found"}, ok=False)
                return self.db.state.response({"notice": notice}, extra_warnings=list(caller.warnings))
            if action == "escalate":
                agent_reason = str(payload.get("reason") or "").strip()
                escalate_reason = (
                    f"Escalated from semantic notice #{int(payload['notice_id'])}"
                    + (f" — {agent_reason}" if agent_reason else "")
                )
                created = self.db.escalate_structured_notice(
                    int(payload["notice_id"]), workspace_canonical=workspace, reason=escalate_reason,
                )
                if created.get("outcome") == "stale_snapshot":
                    return self.db.state.response(
                        {
                            "outcome": "stale_notice",
                            "error": "notice pins no longer match current memory versions; read both memories and judge the current state instead",
                            "freshness": created.get("freshness"),
                        }, ok=False, extra_warnings=list(caller.warnings),
                    )
                if created.get("outcome") not in {"promoted", "appended", "linked"}:
                    outcome = created.get("outcome")
                    return self.db.state.response(
                        {"outcome": "escalate_failed", "detail": created},
                        ok=False, extra_warnings=list(caller.warnings),
                    ) if outcome == "structured_group_required" else self.db.state.response(
                        created, ok=False, extra_warnings=list(caller.warnings),
                    )
                return self.db.state.response(
                    {
                        "outcome": "escalated", "conflict_outcome": created.get("outcome"),
                        "conflict_id": created["conflict_id"], "revision": created["revision"],
                        "member_versions": created.get("member_versions"),
                        "value_groups": created.get("value_groups"),
                        "next_step": (
                            "Credible contradiction is now linked to a formal conflict. Use "
                            "memory_review(view='conflict_detail') to inspect it, then "
                            "memory(action='judge') with the pinned revision."
                        ),
                    }, extra_warnings=list(caller.warnings),
                )
            status = "dismissed" if action == "dismiss" else "resolved"
            result = self.db.update_semantic_notice_status(
                int(payload["notice_id"]), status, str(payload.get("reason") or ""), workspace,
            )
            return self.db.state.response(
                result, ok=result.get("outcome") == "updated", extra_warnings=list(caller.warnings),
            )
        return self._invalid_product_call("memory_repair", f"unknown task: {task}", task)
