"""Product-surface routing helpers for MemoryTools (Phase 4 extraction)."""
# mypy: disable-error-code=no-any-return
from __future__ import annotations

from importlib import resources
from typing import Any, Callable, Optional, TYPE_CHECKING

from .conflict_judgments import ConflictJudgmentStore

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
            "Do not infer conflicts away; if a response says attention_required or action_required=judge_conflict_before_use, handle it before relying on the memory."
        )


class ProductSurfaces:
    def __init__(self, tools: "MemoryTools"):
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    def _judge_constraints(self) -> dict[str, Any]:
        """Allowed values and cross-field rules for conflict judgments.

        Pulled live from ``ConflictJudgmentStore`` (the single source of truth
        for resolution_kind / conflict_scope) plus the verdict/recommendation/
        context/hint enums validated in ``submit_conflict_judgment``. Surfacing
        them in help lets an agent fill a judgment request on the first try
        instead of iterating against ``invalid_*`` outcomes.
        """
        store = ConflictJudgmentStore
        return {
            "verdict": ["contradiction", "evolution", "compatible", "uncertain"],
            "recommended_use": ["left", "right", "contextual", "merge", "ask_user", "none"],
            "usage_context": ["answer", "code", "config", "memory_write", "external_action", "unrelated", "unknown"],
            "confidence_hint": ["low", "medium", "high"],
            "resolution_kind": sorted(store.RESOLUTION_KINDS),
            "conflict_scope": sorted(store.CONFLICT_SCOPES),
            "rules": [
                "resolution_kind=partial_update|merge requires recommended_use in [merge, contextual, ask_user].",
                "resolution_kind=near_duplicate|full_replacement requires recommended_use in [left, right] and a suggested_winner, with conflict_scope in [record, whole_memory].",
                "resolution_kind=contextual_keep_both requires recommended_use=contextual and no suggested_winner.",
                "resolution_kind=not_a_conflict requires recommended_use=none and no suggested_winner.",
                "conflict_scope in [field, section] cannot pair with near_duplicate|full_replacement.",
            ],
        }

    def _product_help(self, surface: str, topic: Optional[str] = None) -> dict[str, Any]:
        helps: dict[str, Any] = {
            "memory": {
                "description": "Daily memory operations: remember, find, read, update, judge, status.",
                "actions": ["remember", "find", "read", "update", "judge", "status", "help"],
                "examples": {
                    "remember": {"action": "remember", "data": {"content": "Fact to remember", "subject": "Short subject", "tags": ["project"]}},
                    "find": {"action": "find", "data": {"query": "project decision", "limit": 5}},
                    "read": {"action": "read", "data": {"memory_id": 123, "sections": "catalog"}},
                    "update": {"action": "update", "data": {"memory_id": 123, "new_content": "Updated current fact", "reason": "User provided a newer source-of-truth."}},
                    "judge": {"action": "judge", "data": {"conflict_id": 1, "verdict": "evolution", "recommended_use": "merge", "resolution_kind": "partial_update", "conflict_scope": "field", "reason": "Only one field changed."}},
                },
                "source_of_truth_rule": "When a user says a new document replaces the current source of truth, find/read the existing current memory and update it; do not create a second active memory or retire the old one unless the user explicitly asks for whole-memory retirement.",
            },
            "memory_review": {
                "description": "Read-only inspection. Never changes memory state.",
                "views": ["overview", "doctor", "conflicts", "conflict_detail", "judgments", "history", "expired", "audit", "entities", "help"],
                "examples": {
                    "conflicts": {"view": "conflicts", "data": {"status": "open", "limit": 20}},
                    "history": {"view": "history", "data": {"memory_id": 123}},
                    "expired": {"view": "expired", "data": {"query": "old decision", "limit": 10}},
                },
            },
            "memory_govern": {
                "description": "Explicit user-authorized governance. Do not use for ordinary source-of-truth updates; use memory(action='update') instead.",
                "actions": ["retire", "resolve_conflict", "confirm", "correct_judgment", "accept_workspace_alias", "reject_workspace_alias", "rename_workspace_canonical", "migrate_workspace", "confirm_pending_workspace", "help"],
                "examples": {
                    "retire": {"action": "retire", "data": {"memory_id": 123, "superseded_by": 456, "reason": "User explicitly requested retiring the old whole memory.", "authorized": True}},
                    "resolve_conflict": {"action": "resolve_conflict", "data": {"conflict_id": 1, "status": "not_a_conflict", "reason": "User confirmed this is not a conflict."}},
                    "accept_workspace_alias": {"action": "accept_workspace_alias", "data": {"alias": "金营二期", "canonical": "金营项目", "reason": "User confirmed these are the same project."}},
                    "reject_workspace_alias": {"action": "reject_workspace_alias", "data": {"alias": "金营培训", "canonical": "金营项目", "reason": "User confirmed these are distinct workspaces."}},
                    "migrate_workspace": {"action": "migrate_workspace", "data": {"from": "金营二期", "to": "金营项目", "reason": "Merge subprojects."}},
                    "confirm_pending_workspace": {"action": "confirm_pending_workspace", "data": {"memory_id": 123, "canonical": "金营项目"}},
                },
                "safety_note": "Retire only whole memories after explicit user authorization. For partial updates or current-document replacement, update the existing memory instead.",
            },
            "memory_repair": {
                "description": "Maintenance and repair operations. Prefer dry_run first; cleanup, activation, and protected-memory metadata changes still require authorized=true when the underlying operation requires it.",
                "tasks": ["split", "rebuild_claims", "rebuild_embeddings", "cleanup_history", "cleanup_vectors", "resync_vectors", "set_entity", "activate_pending", "semantic_control", "notice", "help"],
                "examples": {
                    "rebuild_claims": {"task": "rebuild_claims", "data": {"dry_run": True, "memory_ids": [123]}},
                    "cleanup_vectors": {"task": "cleanup_vectors", "data": {"dry_run": True}},
                    "set_entity": {"task": "set_entity", "data": {"memory_id": 123, "entity": "project-x", "scope": "charter"}},
                    "semantic_control": {"task": "semantic_control", "data": {"action": "status"}},
                    "notice": {"task": "notice", "data": {"action": "list", "limit": 5}},
                },
            },
        }
        if topic == AGENT_ONBOARDING_TOPIC:
            return {
                "description": "Agent onboarding guide for using mema / Memory Arbiter correctly.",
                "topic": AGENT_ONBOARDING_TOPIC,
                "notice": "agent-onboarding:v1",
                "guide_file": "memory_arbiter/AGENT_ONBOARDING.md",
                "content": _agent_onboarding_guide(),
            }
        help_doc = helps.get(surface, {"description": "Unknown product surface."})
        # judge / correct_judgment carry enum + cross-field constraints that the
        # agent cannot otherwise discover without iterating invalid_* outcomes.
        if surface in {"memory", "memory_govern"} and isinstance(help_doc, dict):
            help_doc = dict(help_doc)
            help_doc["judge_constraints"] = self._judge_constraints()
        if topic and isinstance(help_doc, dict):
            narrowed = dict(help_doc)
            narrowed["requested_topic"] = topic
            return narrowed
        return help_doc

    def _invalid_product_call(self, surface: str, message: str, topic: Optional[str] = None) -> dict[str, Any]:
        return self.db.state.response(
            {"error": message, "help": self._product_help(surface, topic)},
            ok=False,
        )

    @staticmethod
    def _help_topic(payload: dict[str, Any], fallback_key: str) -> Optional[str]:
        return payload.get("topic") or payload.get(fallback_key)

    def _forward(
        self, surface: str, topic: Optional[str], fn: Callable[..., dict[str, Any]], **payload: Any,
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
        self, surface: str, value: Any, name: str, topic: Optional[str] = None,
    ) -> Optional[int | dict[str, Any]]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return self._invalid_product_call(surface, f"{name} must be an integer", topic)

    def _require_id(
        self, surface: str, payload: dict[str, Any], name: str, topic: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
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
        self, surface: str, payload: dict[str, Any], name: str, topic: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
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
        self, payload: dict[str, Any], names: tuple[str, ...], surface: str, topic: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
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

    def memory(self, action: str = "help", data: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
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
        if action == "help":
            return self.db.state.response(self._product_help("memory", self._help_topic(payload, "action")))
        if action == "remember":
            return self._forward("memory", action, self.memory_write, **payload)
        if action == "find":
            return self._forward("memory", action, self.memory_search, **payload)
        if action == "read":
            self._alias_id(payload, "memory_id")
            missing = self._require_id("memory", payload, "memory_id", action)
            if missing is not None:
                return missing
            return self._forward("memory", action, self.memory_get, **payload)
        if action == "update":
            self._alias_id(payload, "memory_id")
            missing = self._require_id("memory", payload, "memory_id", action)
            if missing is not None:
                return missing
            return self._forward("memory", action, self.memory_edit, **payload)
        if action == "judge":
            required = [
                "conflict_id", "expected_left_version", "expected_right_version",
                "expected_left_claim_revision", "expected_right_claim_revision",
                "verdict", "recommended_use", "suggested_winner",
                "confidence_hint", "reason", "affects_current_output", "usage_context",
            ]
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
            return self._forward("memory", action, self.memory_submit_conflict_judgment, **payload)
        if action == "status":
            return self.memory_status()
        return self._invalid_product_call("memory", f"unknown action: {action}", action)

    def memory_review(self, view: str = "help", data: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
        """Read-only memory inspection: health, conflicts, history, expired recall, audit, and entities."""
        payload = self._payload_dict(data)
        if data is not None and not isinstance(data, dict):
            return self._invalid_product_call("memory_review", "data must be a JSON object", view)
        view = str(view or "help").strip().lower()
        if view == "help":
            return self.db.state.response(self._product_help("memory_review", self._help_topic(payload, "view")))
        if view == "overview":
            return self.db.state.response({"status": self.memory_status().get("data"), "audit": self.memory_audit_summary().get("data")})
        if view == "doctor":
            return self._forward("memory_review", view, self.memory_doctor_overview, **payload)
        if view == "audit":
            return self.memory_audit_summary()
        if view == "conflicts":
            return self._forward("memory_review", view, self.memory_list_conflicts, **payload)
        if view == "conflict_detail":
            conflict_id = payload.get("conflict_id") or payload.get("id")
            if conflict_id is None:
                return self._invalid_product_call("memory_review", "conflict_detail requires conflict_id", view)
            conflict_id_int = self._int_product_arg("memory_review", conflict_id, "conflict_id", view)
            if isinstance(conflict_id_int, dict):
                return conflict_id_int
            limit_int = self._int_product_arg("memory_review", payload.get("limit", 200), "limit", view)
            if isinstance(limit_int, dict):
                return limit_int
            row = self.db.list_conflicts(status=payload.get("status", "open"), limit=limit_int)
            conflict = next((c for c in row if int(c.get("id")) == conflict_id_int), None)
            if conflict is None:
                try:
                    with self.db.connection() as conn:
                        raw = conn.execute("SELECT * FROM conflicts WHERE id=?", (conflict_id_int,)).fetchone()
                        conflict = {k: raw[k] for k in raw.keys()} if raw else None
                except Exception:
                    conflict = None
            if conflict is None:
                return self.db.state.response({"error": "conflict id not found"}, ok=False)
            return self.db.state.response({"conflict": self._with_resolution_guidance(conflict)})
        if view == "judgments":
            conflict_id = payload.get("conflict_id") or payload.get("id")
            if conflict_id is None:
                return self._invalid_product_call("memory_review", "judgments requires conflict_id", view)
            conflict_id_int = self._int_product_arg("memory_review", conflict_id, "conflict_id", view)
            if isinstance(conflict_id_int, dict):
                return conflict_id_int
            return self.memory_list_conflict_judgments(conflict_id=conflict_id_int)
        if view == "history":
            memory_id = payload.get("memory_id") or payload.get("id")
            if memory_id is None:
                return self._invalid_product_call("memory_review", "history requires memory_id", view)
            memory_id_int = self._int_product_arg("memory_review", memory_id, "memory_id", view)
            if isinstance(memory_id_int, dict):
                return memory_id_int
            return self.memory_history(memory_id=memory_id_int)
        if view == "expired":
            return self._forward("memory_review", view, self.memory_search_expired, **payload)
        if view == "entities":
            return self._forward("memory_review", view, self.memory_list_entities, **payload)
        return self._invalid_product_call("memory_review", f"unknown view: {view}", view)

    def memory_govern(self, action: str = "help", data: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
        """Explicit user-authorized governance: retire, resolve conflicts, confirm, or correct judgments.

        Do not use this for ordinary updates or current source-of-truth replacement;
        use memory(action="update") for those. Retire is only for whole-memory
        retirement after explicit user authorization.
        """
        payload = self._payload_dict(data)
        if data is not None and not isinstance(data, dict):
            return self._invalid_product_call("memory_govern", "data must be a JSON object", action)
        action = str(action or "help").strip().lower()
        if action == "help":
            return self.db.state.response(self._product_help("memory_govern", self._help_topic(payload, "action")))
        if action == "retire":
            invalid_id = self._coerce_product_id("memory_govern", payload, "memory_id", action)
            if invalid_id is not None:
                return invalid_id
            if not payload.get("reason"):
                return self._invalid_product_call("memory_govern", "retire requires reason and authorized=true", action)
            return self._forward("memory_govern", action, self.memory_supersede, **payload)
        if action == "resolve_conflict":
            invalid_id = self._coerce_product_id("memory_govern", payload, "conflict_id", action)
            if invalid_id is not None:
                return invalid_id
            return self._forward("memory_govern", action, self.memory_resolve_conflict, **payload)
        if action == "confirm":
            invalid_id = self._coerce_product_id("memory_govern", payload, "memory_id", action)
            if invalid_id is not None:
                return invalid_id
            return self._forward("memory_govern", action, self.memory_confirm, **payload)
        if action == "correct_judgment":
            if "conflict_id" not in payload and "id" in payload:
                payload["conflict_id"] = payload.pop("id")
            required = [
                "conflict_id", "verdict", "recommended_use", "suggested_winner",
                "reason", "expected_judgment_id", "expected_left_version",
                "expected_right_version", "expected_left_claim_revision",
                "expected_right_claim_revision", "authorized",
            ]
            missing_fields = [name for name in required if name not in payload]
            if missing_fields:
                help_doc = self._product_help("memory_govern", "correct_judgment")
                help_doc["required_fields"] = required
                help_doc["missing_fields"] = missing_fields
                return self.db.state.response({"error": "correct_judgment missing required fields", "help": help_doc}, ok=False)
            # Coerce conflict_id AFTER the missing-fields check so a non-integer
            # id is reported as its own error, not as "conflict_id must be an
            # integer" while other required fields are also missing.
            invalid_id = self._coerce_product_id("memory_govern", payload, "conflict_id", action)
            if invalid_id is not None:
                return invalid_id
            return self._forward("memory_govern", action, self.memory_correct_conflict_judgment, **payload)
        if action == "accept_workspace_alias":
            if not payload.get("alias") or not payload.get("canonical"):
                return self._invalid_product_call("memory_govern", "accept_workspace_alias requires alias and canonical", action)
            bad = self._require_ws_strings(payload, ("alias", "canonical"), "memory_govern", action)
            if bad is not None:
                return bad
            return self._forward("memory_govern", action, self.memory_accept_workspace_alias, **payload)
        if action == "reject_workspace_alias":
            if not payload.get("alias") or not payload.get("canonical"):
                return self._invalid_product_call("memory_govern", "reject_workspace_alias requires alias and canonical", action)
            bad = self._require_ws_strings(payload, ("alias", "canonical"), "memory_govern", action)
            if bad is not None:
                return bad
            return self._forward("memory_govern", action, self.memory_reject_workspace_alias, **payload)
        if action == "rename_workspace_canonical":
            if not payload.get("old") or not payload.get("new"):
                return self._invalid_product_call("memory_govern", "rename_workspace_canonical requires old and new", action)
            bad = self._require_ws_strings(payload, ("old", "new"), "memory_govern", action)
            if bad is not None:
                return bad
            return self._forward("memory_govern", action, self.memory_rename_workspace_canonical, **payload)
        if action == "migrate_workspace":
            if not payload.get("from") or not payload.get("to"):
                return self._invalid_product_call("memory_govern", "migrate_workspace requires from and to", action)
            bad = self._require_ws_strings(payload, ("from", "to"), "memory_govern", action)
            if bad is not None:
                return bad
            return self._forward("memory_govern", action, self.memory_migrate_workspace, **payload)
        if action == "confirm_pending_workspace":
            invalid_id = self._coerce_product_id("memory_govern", payload, "memory_id", action)
            if invalid_id is not None:
                return invalid_id
            if not payload.get("canonical"):
                return self._invalid_product_call("memory_govern", "confirm_pending_workspace requires memory_id and canonical", action)
            bad = self._require_ws_strings(payload, ("canonical",), "memory_govern", action)
            if bad is not None:
                return bad
            return self._forward("memory_govern", action, self.memory_confirm_pending_workspace, **payload)
        return self._invalid_product_call("memory_govern", f"unknown action: {action}", action)

    def memory_repair(self, task: str = "help", data: Optional[dict[str, Any]] = None, **_: Any) -> dict[str, Any]:
        """Maintenance and repair: split, rebuild indexes, cleanup, vector resync, entity backfill."""
        payload = self._payload_dict(data)
        if data is not None and not isinstance(data, dict):
            return self._invalid_product_call("memory_repair", "data must be a JSON object", task)
        task = str(task or "help").strip().lower()
        if task == "help":
            return self.db.state.response(self._product_help("memory_repair", self._help_topic(payload, "task")))
        if task == "split":
            invalid_id = self._coerce_product_id("memory_repair", payload, "memory_id", task)
            if invalid_id is not None:
                return invalid_id
            return self._forward("memory_repair", task, self.memory_split, **payload)
        if task == "rebuild_claims":
            return self._forward("memory_repair", task, self.memory_rebuild_claims, **payload)
        if task == "rebuild_embeddings":
            return self._forward("memory_repair", task, self.memory_rebuild_embeddings, **payload)
        if task == "cleanup_history":
            if "id" in payload or "memory_id" in payload:
                invalid_id = self._coerce_product_id("memory_repair", payload, "memory_id", task)
                if invalid_id is not None:
                    return invalid_id
            return self._forward("memory_repair", task, self.memory_cleanup_history, **payload)
        if task == "cleanup_vectors":
            return self._forward("memory_repair", task, self.memory_cleanup_inactive_vectors, **payload)
        if task == "resync_vectors":
            return self._forward("memory_repair", task, self.memory_resync_vec_parent_status, **payload)
        if task == "set_entity":
            invalid_id = self._coerce_product_id("memory_repair", payload, "memory_id", task)
            if invalid_id is not None:
                return invalid_id
            return self._forward("memory_repair", task, self.memory_set_entity, **payload)
        if task == "activate_pending":
            invalid_id = self._coerce_product_id("memory_repair", payload, "memory_id", task)
            if invalid_id is not None:
                return invalid_id
            return self._forward("memory_repair", task, self.memory_activate, **payload)
        if task == "semantic_control":
            return self.db.state.response(self._semantic_control(str(payload.get("action") or "status")))
        if task == "notice":
            action = str(payload.get("action") or "list").strip().lower()
            if action == "list":
                try:
                    notice_limit = int(payload.get("limit") or 10)
                except (TypeError, ValueError):
                    notice_limit = 10
                return self.db.state.response({"notices": self.db.list_semantic_notices(limit=notice_limit)})
            if action in {"dismiss", "resolve"}:
                invalid_id = self._coerce_product_id("memory_repair", payload, "notice_id", task)
                if invalid_id is not None:
                    return invalid_id
                status = "dismissed" if action == "dismiss" else "resolved"
                result = self.db.update_semantic_notice_status(int(payload["notice_id"]), status, str(payload.get("reason") or ""))
                return self.db.state.response(result, ok=result.get("outcome") == "updated")
        return self._invalid_product_call("memory_repair", f"unknown task: {task}", task)
