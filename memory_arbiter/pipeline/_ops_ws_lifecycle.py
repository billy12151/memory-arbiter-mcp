"""workspace 生命周期治理组（confirm/rename/migrate/confirm_pending/confirm_workspaces/activate/supersede/separate_alias，从 operations.py 搬出，拆分批 ④ 纯移动）。
"""
from __future__ import annotations


import json
import math
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..acl import CallerWorkspace, WorkspaceScope, forbidden_payload, raw_workspace
from ..constants import (
    DEFAULT_WORKSPACE_NAME,
    is_default_workspace_term,
)
from ..db import _normalize_alias_key
from ..embedder import ManagedEmbedder
from ..db.workspaces import _mechanical_ws_key
from ..validation import WORKSPACE_GUIDANCE
from ..models import MemoryStatus, ProtectionLevel, SourceType

if TYPE_CHECKING:
    from ..config import Settings
    from ..db import MemoryDB
    from ..update_monitor import UpdateMonitor
    from ..tools import MemoryTools

class _OpsWsLifecycle:
    if TYPE_CHECKING:
        db: "MemoryDB"
        settings: "Settings"
        _tools: "MemoryTools"

        # 主类委托薄层/跨 mixin 成员的 mypy strict 声明（attr-defined）
        @property
        def _update_monitor(self) -> "UpdateMonitor | None": ...
        def _caller_workspace(self, *args: Any, **kwargs: Any) -> "CallerWorkspace": ...
        def _conflict_detail_for_workspace(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _embedding_configured(self) -> bool: ...
        def _post_commit(
            self, *args: Any, **kwargs: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]: ...
        def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _get_memory_visible(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def _is_truthy(self, *args: Any, **kwargs: Any) -> bool: ...
        def _semantic_notice_workspace_scope(self, *args: Any, **kwargs: Any) -> "WorkspaceScope": ...
        def _semantic_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
        def _strict_acl_unavailable(self, *args: Any, **kwargs: Any) -> "dict[str, Any] | None": ...
        def current_agent_id(self) -> "str | None": ...
        def current_client(self) -> "str | None": ...
        def wait_semantic_worker_drained(self, *args: Any, **kwargs: Any) -> bool: ...
        @staticmethod
        def _compare_memories(*args: Any, **kwargs: Any) -> Any: ...

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
        """Rename a canonical workspace and maintain internal forwarding.

        A5（0.17.1 修复批）：审计与 ok 以 ``committed``（事务是否越过前置
        守卫并提交）为准，不再用 ``not warnings``——repoint 警告形态下
        UPDATE 已提交，旧口径会同时报失败且零审计（实测：库内已改名、
        响应 renamed=False、审计 0 行；agent 会重试）。
        """
        updated, warnings, committed = self.db.rename_workspace_canonical(old, new)
        if committed:
            # P2 #7: bucket-level reason lands in governance_audit (additive
            # trail; normalize_audit consumers never see these rows).
            # A5: 无条件落库（已提交即有痕），warnings 一并入 detail。
            self.db.audit.record_governance_action(
                "rename_workspace",
                reason,
                {"old_canonical": old, "new_canonical": new,
                 "memories_updated": updated, "warnings": list(warnings)},
            )
        return self.db.state.response(
            {
                "renamed": committed,
                "old": old,
                "new": new,
                "memories_updated": updated,
            },
            ok=committed, extra_warnings=warnings,
        )

    def memory_migrate_workspace(
        self, reason: str | None = None, **payload: Any,
    ) -> dict[str, Any]:
        """Merge one workspace into another and maintain internal forwarding.

        `from`/`to` are reserved words so they arrive via **payload.

        A5（0.17.1 修复批）：同 rename——``committed`` 判定 ok 与审计。
        """
        from_ws = str(payload.get("from") or "")
        to_ws = str(payload.get("to") or "")
        embedder, ensure_warnings = self._ensure_active_embedder()
        updated, warnings, committed = self.db.migrate_workspace(
            from_ws, to_ws, embedder=embedder,
        )
        vector_publish_pending = any("workspace canonical vector publish failed" in warning for warning in warnings)
        operation_warnings = [
            warning for warning in warnings
            if "workspace canonical vector publish failed" not in warning
        ]
        if committed:
            # P2 #7: same governance trail as rename (migration committed;
            # vector-publication degradation is not a migration failure).
            # A5: 无条件落库（已提交即有痕），warnings 一并入 detail。
            self.db.audit.record_governance_action(
                "migrate_workspace",
                reason,
                {"from": from_ws, "to": to_ws, "memories_updated": updated,
                 "warnings": list(operation_warnings)},
            )
        data: dict[str, Any] = {
            "migrated": committed,
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
            ok=committed,
            extra_warnings=list(ensure_warnings) + list(warnings),
        )
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
        # owner 2026-10-03 拍板（P1-2，方案 A2+C1 两层定序）：validation 层
        # 是唯一产品门（缺 workspace 一律 invalid_input+指路）；本层保留
        # 轻量兜底——strict+缺参/空串 → ValueError，供绕过 validation 的
        # 直调路径。此前省略参数会让 caller=None：两道 strict 校验整体
        # 短路（可确认他人桶的 pending 行），错误路径还回退 db.get_memory
        # 泄漏外来记录。非 strict 维持原行为（全局池，无 ACL 可跳过）。
        isolation = str(getattr(self.settings, "isolation", "none") or "none")
        if isolation == "strict" and not str(explicit_workspace or "").strip():
            raise ValueError("workspace_required_strict: " + WORKSPACE_GUIDANCE)
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
                            canonical, None,
                        )
                        canonical = str(
                            resolved_canonical.get("canonical") or canonical
                        ).strip()
                if caller is not None and caller.isolation == "strict":
                    memory_workspace = raw_workspace(memory)
                    if not memory_workspace or memory_workspace != caller.canonical:
                        raise ValueError("forbidden_strict_workspace: pending memory is outside caller workspace")
                    if str(canonical or "").strip() != caller.canonical:
                        # twin 等价豁免（A2×A3 交汇，2026-10-03）：twin 改道
                        # pending 行的 canonical 列已是 mema-twin-dev，安全
                        # confirm（canonical=原名 mema-twin）会被本校验误拒。
                        # canonical 会被 redirect tail 落进调用方桶（=caller.
                        # canonical）时二者等价合规——0.16.2 §1.2「非 twin
                        # 调用方确认改道行入 -dev」的既有流程保持可达。
                        from ..twin_redirect import twin_redirect_target

                        _equiv = twin_redirect_target(
                            str(canonical or "").strip(),
                            client=self._tools.current_client(),
                            agent_id=self._tools.current_agent_id(),
                        )
                        if _equiv != caller.canonical:
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
                    # P1-3 防毒守卫（R2 对抗轮改宽形态，owner 2026-10-03 方案
                    # A3）：raw 是会触发 twin 改道的保护键时，只许 no-op（键
                    # 相等，上一分支）走通；canonical 无论 -dev 还是任意第三桶
                    # 一律拦——毒化的资产是 raw 这个 alias 键本身，任何以它为
                    # 键的 confirmed 行都会让 alias 命中先于 identity 改道，
                    # 整体劫持 twin 本体的写入。twin 自身（redirect 判定返回
                    # None）不在拦截面，属可信主体。
                    from ..twin_redirect import twin_redirect_target

                    _redirect = twin_redirect_target(
                        str(raw_ws or ""), client=self._tools.current_client(),
                        agent_id=self._tools.current_agent_id(),
                    )
                    if _redirect is not None:
                        raise ValueError(
                            "protected_bucket_redirect: a redirected workspace key "
                            "confirms only into itself; use the original name"
                        )
                    ok_alias, alias_warnings = self.db.record_workspace_decision_on_conn(
                        conn, raw_ws, canonical, status="confirmed",
                        force=self._is_truthy(authorized),
                    )
                if not ok_alias:
                    raise ValueError("; ".join(alias_warnings) or "workspace redirect not written")
                warnings.extend(alias_warnings)
                # 0.16.2 §1.2: pending activation honors the twin write
                # routing — a non-twin caller cannot activate a row INTO
                # mema-twin; the confirmed canonical lands in mema-twin-dev.
                # Deliberately AFTER the alias chain: recording the alias
                # with the redirected canonical would write a confirmed
                # mema-twin→mema-twin-dev reroute and silently break the
                # twin's own future writes (the write-path redirect keys on
                # caller identity, an alias would bypass it for everyone).
                # With raw='mema-twin' the alias chain above already
                # no-ops (raw == canonical); a differing raw keeps its own
                # alias to the persona bucket untouched.
                from ..twin_redirect import twin_redirect_target

                redirect_target = twin_redirect_target(
                    str(canonical or ""),
                    client=self._tools.current_client(),
                    agent_id=self._tools.current_agent_id(),
                )
                if redirect_target is not None:
                    canonical = redirect_target
                    warnings.append(
                        "protected_bucket_redirect: canonical mema-twin is the "
                        "twin agent's bucket; activated into mema-twin-dev "
                        "instead (owner rule #976)."
                    )
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
                # P2 #7: the confirmation's reason joins the governance trail
                # atomically with the activation (same transaction, so an
                # aborted confirm leaves no phantom audit row).
                self.db.audit.record_governance_action_on_conn(
                    conn, "confirm_pending_workspace", reason,
                    {
                        "memory_id": int(memory_id),
                        "raw_workspace": raw_ws,
                        "canonical": canonical,
                    },
                )
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
        response = self.db.state.response(data, ok=True, extra_warnings=warnings)
        if activated:
            record = self.db.get_memory(int(memory_id))
            if record is None:
                raise RuntimeError("activated record disappeared")
            data["record"] = record
            data["evidence_index"], data["semantic_conflict_check"] = self._post_commit(
                int(memory_id), record, recheck_conflicts=False,
            )
            from .operations import _SubjectTagView
            similar_notice = self._tools._write_pipeline._similar_active_notice(
                int(memory_id),
                _SubjectTagView(record.get("subject"), record.get("tags"), record.get("content")),
                raw_workspace(record),
            )
            if similar_notice is not None:
                response.setdefault("notices", []).append(similar_notice)
        return response

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
        # 0.17.1 prompt suppression 存量清场：快照已持久（os.replace 成功），
        # 把两端都确认的 pending workspace 提议行即时了结——它们再也不可能
        # 被合法判成 move。清场失败绝不回滚快照：降级 warning，下次 kick
        # 前置自愈兜底（suppressed=-1 在响应里可见）。
        suppressed = 0
        try:
            confirmed_set = set(confirmed)
            with self.db.write_transaction() as conn:
                pend = conn.execute(
                    """SELECT id, detail, workspace_canonical FROM scan_queue
                       WHERE kind='workspace' AND status='pending'"""
                ).fetchall()
                stale: list[int] = []
                for row in pend:
                    try:
                        detail = json.loads(str(row["detail"] or "{}"))
                    except json.JSONDecodeError:
                        detail = {}
                    if not isinstance(detail, dict):
                        detail = {}
                    own = str(detail.get("current_workspace") or row["workspace_canonical"] or "")
                    top = str(detail.get("suspected_workspace") or "")
                    if own and top and own in confirmed_set and top in confirmed_set:
                        stale.append(int(row["id"]))
                if stale:
                    now = utc_now_iso()
                    conn.executemany(
                        """UPDATE scan_queue SET status='expired',
                             decided_reason='confirmed pair suppressed', decided_at=?, updated_at=?
                           WHERE id=? AND status='pending'""",  # CAS（B10 纪律）
                        [(now, now, rid) for rid in stale],
                    )
            suppressed = len(stale)
        except Exception:
            suppressed = -1
        extra_warnings: list[str] = []
        if suppressed < 0:
            extra_warnings.append(
                "workspace 存量清场失败（确认快照已生效）：双确认对的 pending 提议行未被清退，"
                "下次 scan kick 前置自愈会重试"
            )
        return self.db.state.response({
            "confirmed": True,
            "confirmed_workspaces": confirmed,
            "count": len(confirmed),
            "sidecar": str(sidecar),
            "suppressed_pending": suppressed,
        }, extra_warnings=extra_warnings or None)

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
        except (sqlite3.Error, ValueError) as exc:
            # ValueError: 0.16.6 DuplicateActiveContentError from the
            # pending->active flip (same-content twin) — same rollback story.
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
        response = self.db.state.response(data, extra_warnings=warnings)
        if ok:
            record = self.db.get_memory(int(memory_id))
            if record is None:
                raise RuntimeError("activated record disappeared")
            data["record"] = record
            data["evidence_index"], data["semantic_conflict_check"] = self._post_commit(
                int(memory_id), record, recheck_conflicts=False,
            )
            from .operations import _SubjectTagView
            similar_notice = self._tools._write_pipeline._similar_active_notice(
                int(memory_id),
                _SubjectTagView(record.get("subject"), record.get("tags"), record.get("content")),
                raw_workspace(record),
            )
            if similar_notice is not None:
                response.setdefault("notices", []).append(similar_notice)
        return response

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

        Side effects: status -> superseded, protection_level -> normal, and the
        derived evidence rows follow the new version (vec0 parent_status
        propagates, so the memory leaves active recall immediately). Open
        conflict groups are NOT auto-resolved — members are version-pinned and
        generic mutation cannot complete a revisioned plan; the response
        reports ``linked_conflicts_resolved`` as 0 for that reason and groups
        must be handled through the judge/apply lifecycle. There is no
        separate audit row: the status change itself plus this call's
        ``reason`` are the record. ``superseded_by`` is validated but not
        persisted — use merge_memories when a durable survivor pointer is
        needed.
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

    def memory_separate_workspace_alias(
        self,
        alias: str,
        canonical: str,
        reason: str = "",
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Undo an installed alias redirect or record a keep-separate decision.

        Deliberately named "separate", not "reject" (owner decision
        2026-09-02): reject reads like refusing a memory write, while this
        action states that two workspaces must stay independent. Reversing a
        recorded separation later requires the confirm side to pass force=true
        (the keep-separate guard in _apply_alias_decision_on_conn).
        """
        authorized = self._is_truthy(authorized)
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to separate workspace aliases", "separated": False},
                ok=False,
            )
        alias_str = str(alias or "").strip()
        canonical_str = str(canonical or "").strip()
        if not alias_str or not canonical_str:
            return self.db.state.response(
                {"error": "separate_workspace_alias requires alias and canonical", "separated": False},
                ok=False,
            )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        errors: list[str] = []
        try:
            with self.db.write_transaction() as conn:
                ok, errors = self.db.workspaces.record_workspace_decision_on_conn(
                    conn, alias_str, canonical_str, status="rejected",
                )
        except sqlite3.Error as exc:
            return self.db.state.response(
                {"error": f"separate_workspace_alias failed; transaction rolled back: {exc}", "separated": False},
                ok=False, extra_warnings=list(caller.warnings),
            )
        if not ok:
            return self.db.state.response(
                {"error": "; ".join(errors), "separated": False},
                ok=False, extra_warnings=list(caller.warnings),
            )
        data: dict[str, Any] = {
            "separated": True,
            "alias": alias_str,
            "canonical": canonical_str,
            "decision": "rejected",
            "note": (
                "the alias no longer redirects; a later merge/confirm targeting the same "
                "pair requires force=true to reverse this decision"
            ),
        }
        if reason:
            data["reason"] = str(reason)
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))
