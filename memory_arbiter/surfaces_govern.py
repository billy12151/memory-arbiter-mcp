"""治理面 mixin：授权门 + _memory_govern（从 surfaces.py 搬出，拆分批 ⑦ 纯移动）。_GOVERNANCE_IMPACTS 类常量留守主类（3 测试按类名直取）。"""
from __future__ import annotations
from __future__ import annotations

from typing import Any, TYPE_CHECKING



from .acl import CallerWorkspace

if TYPE_CHECKING:
    from .config import Settings
    from .db import MemoryDB
    from .tools import MemoryTools



class _SurfacesGovern:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"
        settings: "Settings"
        _GOVERNANCE_IMPACTS: "dict[str, Any]"

        # 主类留守成员的 mypy strict 声明
        def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace": ...
        def _conflict_detail_for_workspace(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _get_memory_visible(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _payload_dict(self, data: "dict[str, Any] | None") -> "dict[str, Any]": ...
        def _semantic_control_with_timeout(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
        def _strict_acl_unavailable(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _active_workspace_anomalies(self) -> "dict[int, str]": ...
        def memory_audit_summary(self, **kwargs: Any) -> dict[str, Any]: ...
        def memory_history(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
        def memory_status(self, **kwargs: Any) -> dict[str, Any]: ...
        @staticmethod
        def _judge_required_fields() -> "list[str]": ...
        def _product_help(self, surface: str, topic: "str | None" = None) -> "dict[str, Any]": ...
        def _invalid_product_call(self, surface: str, message: str, topic: "str | None" = None) -> "dict[str, Any]": ...

        def _forward(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...
        @staticmethod
        def _alias_id(payload: "dict[str, Any]", target: str) -> None: ...
        def _int_product_arg(self, *args: Any, **kwargs: Any) -> Any: ...
        def _bind_product_id(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _require_id(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _coerce_product_id(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        @staticmethod
        def _is_truthy(value: Any) -> bool: ...
        def _require_ws_strings(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _normalize_boolean_fields(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        @staticmethod
        def _help_topic(payload: "dict[str, Any]", fallback_key: str) -> "str | None": ...
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
            "include_size", "deduplicate",
            "affects_current_output",
        )
        if action == "help":
            return self.db.state.response(self._product_help("memory", self._help_topic(payload, "action")))
        if action == "remember":
            return self._forward("memory", action, self._tools.memory_write, **payload)
        if action == "find":
            return self._forward("memory", action, self._tools.memory_search, **payload)
        if action == "batch_find":
            return self._forward("memory", action, self._tools.memory_batch_find, **payload)
        if action == "read":
            self._alias_id(payload, "memory_id")
            missing = self._require_id("memory", payload, "memory_id", action)
            if missing is not None:
                return missing
            return self._forward("memory", action, self._tools.memory_get, **payload)
        if action == "batch_read":
            if not payload.get("memory_ids"):
                return self._invalid_product_call(
                    "memory", "batch_read requires memory_ids (non-empty list of ids)", action,
                )
            return self._forward("memory", action, self._tools.memory_batch_read, **payload)
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
            conflict_id_int = int(conflict_id)  # already coerced by validation
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
            memory_id_int = int(memory_id)  # already coerced by validation
            return self.memory_history(memory_id=memory_id_int, workspace=payload.get("workspace"))
        if view == "expired":
            return self._forward("memory_review", view, self._tools.memory_search_expired, **payload)
        if view == "entities":
            return self._forward("memory_review", view, self._tools.memory_list_entities, **payload)
        if view == "workspaces":
            # C2（owner 2026-10-03）：选桶发现接口。strict 调用者先过 denied 门
            # （无 canonical=denied 优先于空集，对齐全库口径），再看 admitted 集；
            # none/weak 全量。
            #
            # A8（0.17.1 修复批）：admitted 进 SQL（keyword-only 显式传递，
            # validation 白名单保证 payload 无法注入）——此前 LIMIT 先于此处
            # 事后过滤：limit=1 时自有桶被更晚更新的外来桶挤出窗口（实测
            # 返回 []），且 count 被重算为过滤后长度。现在 count 即 SQL 行数。
            caller = self._caller_workspace(payload.get("workspace"))
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
            if caller.isolation == "strict":
                return self._tools.memory_list_workspaces(
                    **payload, admitted=set(caller.admitted),
                )
            return self._tools.memory_list_workspaces(**payload)
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
            bad_id = self._bind_product_id("memory_govern", payload, "memory_id", action)
            if bad_id is not None:
                return bad_id
            if not payload.get("reason"):
                return self._invalid_product_call("memory_govern", "retire requires reason and authorized=true", action)
            # superseded_by arrives already int-coerced from validation
            # (numeric-string compatible); None stays None there by design.
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward("memory_govern", action, self._tools.memory_supersede, **payload)
        if action in {"apply_conflict_action", "replan_conflict", "resolve_conflict"}:
            bad_id = self._bind_product_id("memory_govern", payload, "conflict_id", action)
            if bad_id is not None:
                return bad_id
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
            bad_id = self._bind_product_id("memory_govern", payload, "memory_id", action)
            if bad_id is not None:
                return bad_id
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._forward("memory_govern", action, self._tools.memory_confirm, **payload)
        if action == "merge_memories":
            bad_id = self._bind_product_id("memory_govern", payload, "survivor_id", action)
            if bad_id is not None:
                return bad_id
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
        if action == "rollback_auto_move":
            bad_id = self._bind_product_id("memory_govern", payload, "audit_id", action)
            if bad_id is not None:
                return bad_id
            auth_error = self._governance_authorization_error(action, payload)
            if auth_error is not None:
                return auth_error
            return self._tools.memory_rollback_auto_move(**payload)
        if action == "confirm_pending_workspace":
            bad_id = self._bind_product_id("memory_govern", payload, "memory_id", action)
            if bad_id is not None:
                return bad_id
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
