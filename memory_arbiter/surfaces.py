"""Product-surface routing helpers for MemoryTools (Phase 4 extraction)."""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

from .acl import CallerWorkspace, WorkspaceScope
from .constants import SCAN_DUPLICATES_BATCH as SCAN_DUPLICATES_BATCH, SCAN_DUPLICATES_MAX_PAGES as SCAN_DUPLICATES_MAX_PAGES, SCAN_DUPLICATES_MAX_RESULTS as SCAN_DUPLICATES_MAX_RESULTS  # noqa: F401
from .product_helps import (  # noqa: F401
    AGENT_ONBOARDING_TOPIC as AGENT_ONBOARDING_TOPIC,
    _PRODUCT_HELPS as _PRODUCT_HELPS,
    _agent_onboarding_guide as _agent_onboarding_guide,
    _memory_value_reference as _memory_value_reference,
)
from .surfaces_govern import _SurfacesGovern
from .surfaces_repair import _SurfacesRepair
from .scan_tasks import (
    SCHEDULED_TASKS_TOPIC,
    scheduled_tasks_help,
)
from .validation import PRODUCT_FIELD_REGISTRY, _controlled_integer, validate_product_payload

if TYPE_CHECKING:
    from .tools import MemoryTools


class ProductSurfaces(_SurfacesGovern, _SurfacesRepair):
    _GOVERNANCE_IMPACTS: dict[str, str] = {
        "retire": "Marks a whole memory superseded and removes it from active recall.",
        "resolve_conflict": "Marks an applying conflict resolved after every planned member action completed.",
        "apply_conflict_action": "Atomically applies one planned member change and records its result in the conflict.",
        "replan_conflict": "CAS-replaces a stale or failed applying plan while preserving prior plan history.",
        "confirm": "Promotes the memory to user_confirmed and locks it against ordinary changes.",
        "rename_workspace_canonical": "Renames a canonical workspace and reroutes all affected memories.",
        "migrate_workspace": "Bulk-moves memories to another canonical workspace and records the alias.",
        "move_memories_workspace": "Moves the selected memories by id to another workspace bucket (both workspace columns); alias and normalization rules are not changed. Rows whose canonical already diverges from their bucket (e.g. rows written through a confirmed alias) are re-anchored to the destination when authorized. default_fallback=true + a reason moves memories BACK to the global default pool when no suitable bucket exists (audited + user-notified; use sparingly).",
        "rollback_auto_move": "Reverses ONE autonomous normalization move by its normalize_audit id (0.16.0): restores both workspace columns, voids new-bucket conflict tickets, invalidates the scan watermark, and marks the audit row rolled_back. Manual moves are out of scope; protected-bucket moves never happened autonomously. ACL note (owner 2026-10-04, deliberate exemption): exempt from strict caller-workspace gating — this undoes the server's own autonomous move under the authorized=true gate; its accepted workspace key is forward-compat only and unconsumed.",
        "confirm_pending_workspace": "Assigns the canonical workspace and activates the pending memory for recall.",
        "confirm_workspaces": "Records the reviewed workspace registry snapshot that doctor's workspace.review diffs against; unconfirmed new workspaces keep the check warning. Confirmed pairs also stop generating workspace-move proposals in the scan pipeline (0.17.1 prompt suppression): a suspect whose current AND suspected buckets are both confirmed is never enqueued, and pending rows of such pairs are expired on confirm / at kick self-heal.",
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

    def _active_workspace_anomalies(self) -> "dict[int, str]":
        """C3b: {memory_id: suspected_bucket} from pending kind='workspace'
        scan_queue rows.

        0.16.2 §1.3: the weekly anomaly check no longer produces
        workspace_review notices — the pipeline's incremental suspects and the
        weekly backstop's findings share the judgment queue, so the sweep
        reads queue detail instead. Only rows whose subject memory STILL sits
        in the row's pinned current_workspace participate — a post-move
        suspect that has not been judged/expired yet must not re-inject a
        sweep against its old suspicion.
        """
        import json as _json

        out: dict[int, str] = {}
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT member_versions, detail, workspace_canonical "
                    "FROM scan_queue WHERE kind='workspace' AND status='pending'"
                ).fetchall()
                if not rows:
                    return out
                wanted: dict[int, tuple[str, str]] = {}
                for row in rows:
                    try:
                        detail = _json.loads(str(row["detail"] or "{}"))
                    except (TypeError, ValueError):
                        continue
                    suspected = str(detail.get("suspected_workspace") or "").strip()
                    pinned = (
                        str(detail.get("current_workspace") or "").strip()
                        or str(row["workspace_canonical"] or "")
                    )
                    if not suspected or not pinned:
                        continue
                    try:
                        members = _json.loads(str(row["member_versions"] or "[]"))
                        memory_id = int(members[0]["memory_id"])
                    except (IndexError, KeyError, TypeError, ValueError):
                        continue
                    wanted[memory_id] = (suspected, pinned)
                if not wanted:
                    return out
                placeholders = ",".join("?" for _ in wanted)
                current = {
                    int(r["id"]): str(r["workspace"] or "")
                    for r in conn.execute(
                        "SELECT id, COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                        f"FROM memories WHERE id IN ({placeholders}) AND status='active'",
                        list(wanted),
                    )
                }
                for memory_id, (suspected, pinned) in wanted.items():
                    if current.get(memory_id) == pinned:
                        out[memory_id] = suspected
        except Exception:
            return {}
        return out

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
                "The default judgment corrects the wrong data in memory: plan update_current_claim or append_superseded_context for members holding superseded claims; use preserve_historical_record only when the user explicitly asks to keep the historical record.",
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
                "replacement plan; a grounding-failed update_current_claim is also recovered by passing "
                "chosen_value (drawn from the conflict's value_groups) to replace the deadlocked choice. "
                "Prior plan history is preserved."
            ),
            "confirm_new_workspace": (
                "Explain the proposed canonical workspace and ask the user to authorize confirmation. After "
                "approval call memory_govern(action='confirm_pending_workspace') with authorized=true."
            ),
            "review_default_fallback": (
                "Memories were parked in the default pool because no suitable bucket was "
                "found (default_fallback). Present the list to the user: re-home each via "
                "authorized memory_govern(action='move_memories_workspace'), or confirm the "
                "global placement is right. The count is on doctor's normalize board."
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
                "notice": "agent-onboarding:v2",
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

    def _bind_product_id(
        self, surface: str, payload: dict[str, Any], name: str,
        topic: str | None = None, *, required: bool = True,
    ) -> dict[str, Any] | None:
        """Alias ``id``→``name`` and guard presence ONLY.

        Numeric coercion already happened in ``validate_product_payload``
        (_v_id_fields coerces in place and runs for every product call), so a
        second numeric pass here would be dead work — the 0.16.6 audit
        removed exactly that double coercion. The judge dispatcher keeps its
        own strict coerce by design (validation skips judge ids to preserve
        missing-receipt-fields-first error ordering)."""
        self._alias_id(payload, name)
        if required and name not in payload:
            return self._invalid_product_call(surface, f"{topic or surface} requires {name}", topic)
        return None

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
