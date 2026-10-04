"""修复面 mixin：_memory_repair + _scan_duplicates_task（从 surfaces.py 搬出，拆分批 ⑦ 纯移动）。"""
from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from .acl import CallerWorkspace, forbidden_payload, raw_workspace
from .config import Settings
from .db import MemoryDB
from .db_generation import CONFLICT_DETECTOR_VERSION
from .request_identity import get_request_identity
from .scan_tasks import SCHEDULED_TASKS_SPEC_VERSION

if TYPE_CHECKING:
    from .tools import MemoryTools


class _SurfacesRepair:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"
        settings: "Settings"

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
        def _governance_authorization_error(
            self, action: str, payload: "dict[str, Any]",
            *, tool: str = "memory_govern", retry_field: str = "action",
        ) -> "dict[str, Any] | None": ...
        @staticmethod
        def _judge_required_fields() -> "list[str]": ...
        def _product_help(self, surface: str, topic: "str | None" = None) -> "dict[str, Any]": ...
        def _invalid_product_call(self, surface: str, message: str, topic: "str | None" = None) -> "dict[str, Any]": ...
        def _forward(self, *args: Any, **kwargs: Any) -> "dict[str, Any]": ...
        @staticmethod
        def _alias_id(payload: "dict[str, Any]", target: str) -> None: ...
        def _int_product_arg(self, *args: Any, **kwargs: Any) -> Any: ...
        def _bind_product_id(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _require_id(self, *args: Any, **kwargs: Any) -> Any: ...
        def _coerce_product_id(self, *args: Any, **kwargs: Any) -> Any: ...
        @staticmethod
        def _is_truthy(value: Any) -> bool: ...
        def _require_ws_strings(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _normalize_boolean_fields(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        @staticmethod
        def _help_topic(payload: "dict[str, Any]", fallback_key: str) -> "str | None": ...
    def _memory_repair(self, task: str = "help", data: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        """Maintenance and repair for evidence, history, backup, notices, and runtime state."""
        payload = self._payload_dict(data)
        if data is not None and not isinstance(data, dict):
            return self._invalid_product_call("memory_repair", "data must be a JSON object", task)
        task = str(task or "help").strip().lower()
        # "expanded" (0.16.4): the internal_memory blind-judge guard flag.
        self._normalize_boolean_fields(payload, "authorized", "dry_run", "clear", "expanded")
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
            # C3b: the suspected-bucket sweep for misplaced memories. Global
            # scans only — a strict caller's admitted set usually excludes the
            # suspected bucket, and the cross-bucket reference snippets must
            # not leak outside the caller's scope.
            suspected_anomalies = (
                self._active_workspace_anomalies() if scan_workspace is None else None
            )
            scan_started = time.perf_counter()
            result = self._tools._scan_pipeline.scan_rule_candidates(
                after_memory_id=anchor_value,
                anchor_batch=batch_value,
                neighbor_k=k_value,
                include_check=self._is_truthy(payload.get("include_check")),
                max_distance=distance_value,
                workspace=scan_workspace,
                include_duplicates=self._is_truthy(payload.get("include_duplicates")),
                suspected_anomalies=suspected_anomalies,
            )
            if "error" not in result and not self._is_truthy(payload.get("include_quotes")):
                # C1 response slimming: the default page carries only the
                # lightweight triage identity; include_quotes=true restores
                # the full member/slot envelope record_conflict consumes.
                result = self._tools._lightweight_scan_candidates(result)
            ok = "error" not in result
            if ok:
                # 0.16.0 §6⑩/§3: scan_candidates is now the manual/diagnostic
                # channel; every page echoes the served spec so a still-v1
                # scheduled task sees the drift hint on its next run.
                result["scheduled_tasks_spec"] = {
                    "spec_version": SCHEDULED_TASKS_SPEC_VERSION,
                    "drift": (
                        "the page-driven triage loop is the v1 scheduled-task "
                        "contract; v2 tasks kick memory_repair(task='scan_pipeline') "
                        "and clear the scan_queue — rebuild your task (help topic "
                        "scheduled_tasks)"
                    ),
                }
            if ok and scan_workspace is None and int(result.get("anchors_scanned") or 0) > 0:
                # C5: per-group pacing record for the ROUTINE scan. Global
                # pages only — a strict workspace-scoped page sees part of the
                # library and must not advance the global walk (same scoping
                # rule as the upgrade-gate cursor below). Doctor's
                # broken-chain alarm reads this kv.
                identity = get_request_identity()
                self.db.record_scan_page_progress(
                    after_memory_id=anchor_value,
                    next_anchor_memory_id=result.get("next_anchor_memory_id"),
                    anchor_buckets=result.get("anchor_buckets"),
                    client=(identity.client if identity else None),
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
            if ok and result.get("next_anchor_memory_id") is None and int(
                result.get("anchors_scanned") or 0
            ) > 0:
                # Audit evidence: only completed full-scan boundaries write a
                # line; intermediate pages stay silent, and a zero-anchor page
                # (empty library) is no evidence a scheduled task exists.
                identity = get_request_identity()
                self.db.log_scan(
                    duration_sec=time.perf_counter() - scan_started,
                    client=(identity.client if identity else None),
                    agent_id=(identity.agent_id if identity else None),
                )
            return self.db.state.response(result, ok=ok, extra_warnings=list(caller.warnings))
        if task == "scan_pipeline":
            # 0.16.0 §6⑦: the scheduled task KICKS the pipeline; full vs
            # incremental, breakpoints, and completion are server-decided.
            # Global operation (like scan_candidates): strict callers resolve
            # from settings only.
            caller = self._caller_workspace(None)
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
            action_value = str(payload.get("action") or "kick").strip().lower()
            if action_value == "status":
                return self.db.state.response(self._tools.scan_pipeline_status())
            if action_value != "kick":
                return self._invalid_product_call(
                    "memory_repair", f"unknown scan_pipeline action: {action_value}", task,
                )
            result = self._forward("memory_repair", task, self._tools.scan_pipeline_kick, **payload)
            if result.get("ok"):
                # Expire queue rows whose pinned versions drifted before the
                # next judgment page is built (§6㉑④: refresh, never drop).
                self.db.scan_queue_refresh_stale_pins()
            # A10（0.17.1 修复批）：_forward 的错误路径返回的是**完整信封**
            # （_invalid_product_call → state.response(..., ok=False)）。再包
            # 一层会把被拒的 kick 变成顶层 ok=True + data.ok=False（通用
            # ok 契约误读为成功，实测 neighbor_k="abc" 即此形态）。识别
            # 信封形状后原样透传；其余（kick 自身失败 dict / 成功回执）
            # 按 result["ok"] 组装。
            if "mode" in result and "data" in result:
                return result
            return self.db.state.response(result, ok=bool(result.get("ok", True)))
        if task == "scan_queue":
            # 0.16.0 §6㉑: agent-facing judgment queue. Page = "handle page 1,
            # submit its dispositions with the next page fetch"; submit lands
            # dispositions server-side (confirm → open, dismiss → suppression
            # source). Queue contents never reach the user-facing surfaces.
            caller = self._caller_workspace(payload.get("workspace"))
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
            action_value = str(payload.get("action") or "page").strip().lower()
            if action_value == "status":
                return self.db.state.response({
                    "ok": True,
                    "queue": self.db.scan_queue_counts(),
                    "queue_backlog": self.db.scan_queue_backlog(),
                    "internal_conflicts": self.db.internal_conflicts.counts(),
                })
            if action_value == "page":
                try:
                    result = self._tools.scan_queue_page(caller=caller, **payload)
                except (TypeError, ValueError) as exc:
                    # int coercion of a loosely-typed page_size/page_token must
                    # not leak a bare traceback — align with _forward's
                    # loose-JSON posture (structured invalid_input, ok=False).
                    return self.db.state.response(
                        {"outcome": "invalid_input", "error": str(exc)}, ok=False,
                    )
                return self.db.state.response(result, ok=result.get("ok", True))
            if action_value == "submit":
                result = self._tools.scan_queue_submit(caller=caller, **payload)
                return self.db.state.response(result, ok=result.get("ok", True))
            return self._invalid_product_call(
                "memory_repair", f"unknown scan_queue action: {action_value} (page|submit|status)", task,
            )
        if task == "scan_duplicates":
            return self._scan_duplicates_task(task, payload)
        if task == "cleanup_history":
            if "id" in payload or "memory_id" in payload:
                bad_id = self._bind_product_id("memory_repair", payload, "memory_id", task, required=False)
                if bad_id is not None:
                    return bad_id
            return self._forward("memory_repair", task, self._tools.memory_cleanup_history, **payload)
        if task == "set_entity":
            bad_id = self._bind_product_id("memory_repair", payload, "memory_id", task)
            if bad_id is not None:
                return bad_id
            return self._forward("memory_repair", task, self._tools.memory_set_entity, **payload)
        if task == "activate_pending":
            bad_id = self._bind_product_id("memory_repair", payload, "memory_id", task)
            if bad_id is not None:
                return bad_id
            return self._forward("memory_repair", task, self._tools.memory_activate, **payload)
        if task == "replay_backup":
            return self._forward("memory_repair", task, self._tools.memory_replay_backup, **payload)
        if task == "scan_workspace_anomalies":
            # C3a: the anomaly check reads the FULL summary-vector matrix —
            # it is inherently global (voting needs cross-bucket neighbours),
            # so the strict-ACL gate resolves the caller from settings only,
            # same as normalize_workspaces/scan_candidates.
            caller = self._caller_workspace(None)
            denied = self._strict_acl_unavailable(caller)
            if denied is not None:
                return denied
            return self._forward("memory_repair", task, self._tools.memory_scan_workspace_anomalies, **payload)
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
            bad_id = self._bind_product_id("memory_repair", payload, "notice_id", task)
            if bad_id is not None:
                return bad_id
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
                            "Credible contradiction is now filed as a formal conflict "
                            "(escalate never edits memory content). Use "
                            "memory_review(view='conflict_detail') to inspect it, then "
                            "memory(action='judge') with the pinned revision and "
                            "apply_plan to land the correction and close the group; the "
                            "judge records who decided what for the audit trail."
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

    def _scan_duplicates_task(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        from . import surfaces as _sf  # 拆分批 ⑦：常量读取经 surfaces 模块属性（测试 patch 缝同命名空间）
        """Full-library near-duplicate sweep as one bounded response (0.15.3).

        Separated from the conflict scan on purpose: an agent session must
        never have to walk scan_candidates pages with include_duplicates to
        enumerate duplicates. The pages are aggregated server-side under one
        global pair cap; entries are lightweight by default (ids/subjects/
        workspace/reason/distance) and include_quotes=true adds the evidence
        quotes. Pairs already dismissed (not_a_conflict) or inside open/
        applying groups are suppressed by the same candidate-hash contract
        as scan_candidates pages. This task does NOT advance conflict-scan
        progress or write scan_log — those belong to scan_candidates.
        """
        caller = self._caller_workspace(payload.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        include_quotes = self._is_truthy(payload.get("include_quotes"))
        scan_workspace = caller.scope_canonicals() if caller.isolation == "strict" else None
        collected: dict[tuple[int, int], dict[str, Any]] = {}
        truncated = False
        anchors_scanned = 0
        pages_scanned = 0
        anchor = 0
        while True:
            page = self._tools._scan_pipeline.scan_rule_candidates(
                after_memory_id=anchor,
                anchor_batch=_sf.SCAN_DUPLICATES_BATCH,
                include_duplicates=True,
                workspace=scan_workspace,
            )
            if "error" in page:
                return self.db.state.response(
                    {"task": task, "error": page["error"]}, ok=False,
                    extra_warnings=list(caller.warnings),
                )
            pages_scanned += 1
            anchors_scanned += int(page.get("anchors_scanned") or 0)
            for entry in page.get("duplicates_pool") or []:
                pair_key = (int(entry["left_id"]), int(entry["right_id"]))
                if pair_key not in collected and len(collected) >= _sf.SCAN_DUPLICATES_MAX_RESULTS:
                    # Enforced inside the page too: a single page's pool (cap
                    # 2*batch) can otherwise push past the global cap.
                    truncated = True
                    continue
                # Re-hitting an already-collected pair is a dict replace.
                collected[pair_key] = entry
            if page.get("duplicates_truncated"):
                truncated = True
            next_anchor = page.get("next_anchor_memory_id")
            if next_anchor is None:
                break
            if len(collected) >= _sf.SCAN_DUPLICATES_MAX_RESULTS:
                # Bounded by design: stop at the cap instead of shipping an
                # unbounded enumeration into one agent session.
                truncated = True
                break
            if pages_scanned >= _sf.SCAN_DUPLICATES_MAX_PAGES:
                # Hard work ceiling (pages × batch anchors): no call can run
                # unbounded sweep work even on an unexpectedly huge library.
                truncated = True
                break
            anchor = int(next_anchor)
        subject_rows: dict[int, dict[str, Any]] = {}
        wanted_ids = sorted({memory_id for pair in collected for memory_id in pair})
        if wanted_ids:
            with self.db.connection() as conn:
                for start in range(0, len(wanted_ids), 500):
                    chunk = wanted_ids[start:start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    for row in conn.execute(
                        "SELECT id, subject, workspace, workspace_canonical FROM memories "
                        f"WHERE id IN ({placeholders})",
                        chunk,
                    ):
                        subject_rows[int(row["id"])] = dict(row)
        duplicates: list[dict[str, Any]] = []
        for (left_id, right_id), entry in collected.items():
            left_row = subject_rows.get(left_id) or {}
            right_row = subject_rows.get(right_id) or {}
            item: dict[str, Any] = {
                "left_id": left_id,
                "right_id": right_id,
                "left_subject": str(left_row.get("subject") or ""),
                "right_subject": str(right_row.get("subject") or ""),
                "workspace": str(
                    left_row.get("workspace_canonical") or left_row.get("workspace") or ""
                ),
                "reason": entry.get("reason"),
                "distance": entry.get("distance"),
                "candidate_key_hash": entry.get("candidate_key_hash"),
            }
            if include_quotes:
                item["left_quote"] = entry.get("left_snippet")
                item["right_quote"] = entry.get("right_snippet")
            duplicates.append(item)
        result: dict[str, Any] = {
            "task": task,
            "duplicates": duplicates,
            "total_pairs": len(duplicates),
            "truncated": truncated,
            "anchors_scanned": anchors_scanned,
            "pages_scanned": pages_scanned,
            "max_results": _sf.SCAN_DUPLICATES_MAX_RESULTS,
            "next_steps": (
                "Triage silently: merge true duplicates via memory_govern(action="
                "'merge_memories'); verify a pair's full evidence (members contract) "
                "with scan_candidates(include_duplicates=true) before recording a "
                "dismissal via record_conflict(status='not_a_conflict'); ask the user "
                "only when the merge set is ambiguous."
            ),
        }
        return self.db.state.response(result, extra_warnings=list(caller.warnings))
