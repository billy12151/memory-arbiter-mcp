"""workspace 批量搬迁组（move_memories + 冲突成员籍合并 helper，从 operations.py 搬出，拆分批 ④ 纯移动）。
"""
from __future__ import annotations


import json
import sqlite3
from typing import Any, TYPE_CHECKING

from ..acl import CallerWorkspace, WorkspaceScope
from ..constants import (
    DEFAULT_WORKSPACE_NAME,
    is_default_workspace_term,
)
from ..db import _normalize_alias_key
from ..embedder import ManagedEmbedder
from ..db_generation import database_startup_lock
from ..db.workspaces import _coerce_ws
from ..validation import MAX_BATCH_IDS, _controlled_integer
from ..models import MemoryStatus, utc_now_iso

if TYPE_CHECKING:
    from ..config import Settings
    from ..db import MemoryDB
    from ..update_monitor import UpdateMonitor
    from ..tools import MemoryTools

class _OpsWsMove:
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

    def memory_move_memories_workspace(
        self,
        memory_ids: list[int],
        new_workspace: str,
        reason: str | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Move selected memories by id to a different workspace bucket.

        Companion to migrate_workspace: migrate merges one canonical
        workspace into another by name and reroutes the alias; move
        reassigns individual memories — the raw bucket column and
        workspace_canonical together — and leaves alias/normalization rules
        untouched. Divergent rows (workspace_canonical already pointing
        somewhere other than the raw bucket, e.g. rows written through a
        confirmed alias) are refused unless the call is authorized, in which
        case they are re-anchored to the destination with an explicit
        response note. The source bucket may be default; default is never a
        valid destination. Moving does not change memory status: pending
        memories stay pending, and superseded/deleted rows keep their status
        (reported via moved_non_active).
        """
        authorized = self._is_truthy(authorized)
        # 0.16.3 default fallback (owner rule): when an agent genuinely
        # cannot find a suitable bucket, moving the memories BACK to the
        # global default pool is allowed as an explicitly declared escape
        # hatch — under strict isolation default is the only bucket outside
        # the caller's own that still participates in recall, so a memory
        # parked in a wrong bucket is invisible to everyone who could fix it.
        # Guard rails (all must hold): explicit default_fallback=true, a
        # non-empty reason, and the audit trail + user-facing notice below —
        # default must never become a dumping ground by accident.
        default_fallback = self._is_truthy(_.get("default_fallback"))
        if default_fallback:
            if not is_default_workspace_term(str(new_workspace or "")):
                return self.db.state.response({
                    "moved": False,
                    "error": (
                        "default_fallback=true requires new_workspace to be "
                        "the default pool; for a project bucket drop the flag"
                    ),
                    "field": "new_workspace",
                }, ok=False)
            if not str(reason or "").strip():
                return self.db.state.response({
                    "moved": False,
                    "error": (
                        "default_fallback=true requires a non-empty reason "
                        "(why no suitable bucket exists)"
                    ),
                    "field": "reason",
                }, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        warnings: list[str] = list(caller.warnings)
        admitted: set[str] = set()
        if caller.isolation == "strict":
            admitted = {
                str(name or "").strip()
                for name in caller.scope_canonicals()
                if str(name or "").strip()
            }

        ids: list[int] = []
        for value in list(memory_ids or []):
            # Same strict coercion as the product surface (_controlled_integer):
            # a direct call must not let bool True / float 1.5 silently become
            # memory id 1.
            memory_id = _controlled_integer(value)
            if memory_id is None or memory_id <= 0:
                return self.db.state.response(
                    {
                        "moved": False,
                        "error": "memory_ids must contain positive integer ids",
                        "field": "memory_ids",
                    },
                    ok=False, extra_warnings=warnings,
                )
            if memory_id not in ids:
                ids.append(memory_id)
        if len(ids) > MAX_BATCH_IDS:
            return self.db.state.response(
                {
                    "moved": False,
                    "error": f"memory_ids must be a list with at most {MAX_BATCH_IDS} items",
                    "field": "memory_ids",
                },
                ok=False, extra_warnings=warnings,
            )

        requested = str(new_workspace or "").strip()
        target = _coerce_ws(new_workspace)
        if default_fallback:
            # Fold any reserved default synonym to the one true spelling; no
            # canonical registration and no name embedding for the global
            # pool (it is not a project bucket). Both variables: the
            # in-transaction re-fold re-derives from `requested` and would
            # otherwise resurrect the synonym spelling.
            requested = DEFAULT_WORKSPACE_NAME
            target = DEFAULT_WORKSPACE_NAME
        if not target:
            return self.db.state.response(
                {
                    "moved": False,
                    "error": "new_workspace must be a non-empty workspace string",
                    "field": "new_workspace",
                },
                ok=False, extra_warnings=warnings,
            )
        # 0.16.2 §1.2: the mema-twin bucket is the twin agent's write domain —
        # a move by any other caller (governance moves included, no exception)
        # lands in mema-twin-dev instead. Rewritten here on BOTH variables so
        # the transaction's re-fold from the original request stays
        # consistent; twin_redirect_from feeds the response report. Content
        # stays reachable in the twin family; this is the default write
        # destination, not an E6 change.
        from ..twin_redirect import twin_redirect_target

        twin_redirect_from = twin_redirect_target(
            target,
            client=self._tools.current_client(),
            agent_id=self._tools.current_agent_id(),
        )
        if twin_redirect_from is not None:
            requested = twin_redirect_from
            target = twin_redirect_from
            warnings.append(
                "protected_bucket_redirect: destination mema-twin is the twin "
                f"agent's bucket; moved to {twin_redirect_from} instead (owner rule #976)."
            )

        def destination_error(name: str) -> dict[str, Any] | None:
            if is_default_workspace_term(name):
                if default_fallback:
                    # Explicitly declared escape hatch — see the guard rails
                    # at the top of this method.
                    return None
                return {
                    "moved": False,
                    "error": (
                        "default is a reserved global pool and cannot be a move "
                        "destination; pass default_fallback=true with a reason "
                        "only when no suitable bucket exists"
                    ),
                    "field": "new_workspace",
                }
            if caller.isolation == "strict" and name not in admitted:
                if default_fallback and is_default_workspace_term(name):
                    return None
                return {
                    "moved": False,
                    "error": (
                        "forbidden_strict_workspace: new_workspace is outside "
                        "the caller workspace scope"
                    ),
                    "field": "new_workspace",
                    **caller.response_fields(),
                }
            return None

        def destination_block_response(blocked: dict[str, Any]) -> dict[str, Any]:
            # Reconciliation contract: ids are already parsed when a
            # destination is rejected, so the response must account for them.
            blocked.update({
                "moved_ids": [],
                "failed_ids": list(ids),
                "errors": [
                    {"memory_id": memory_id, "reason": "destination_rejected"}
                    for memory_id in ids
                ],
            })
            return self.db.state.response(blocked, ok=False, extra_warnings=warnings)

        bad_destination = destination_error(target)
        if bad_destination is not None:
            return destination_block_response(bad_destination)

        # Destination orthography, mirroring what a write using the same name
        # would do: follow ONE confirmed alias hop to its decision canonical
        # (checked first, matching the write path's alias short-circuit), else
        # land on the registered spelling of a mechanical twin (migrate's
        # destination fold). Never a second alias hop — a write using the
        # same name would not take one either.
        def fold_destination(name: str) -> str:
            alias_target = self.db.workspaces.confirmed_alias_canonical(name)
            if alias_target:
                return alias_target
            return self.db.workspaces.registered_mechanical_canonical(name) or name

        target = fold_destination(target)
        bad_destination = destination_error(target)
        if bad_destination is not None:
            return destination_block_response(bad_destination)

        # Advisory pre-validation (fast-fail before any transaction): per-id
        # visibility plus the divergence gate. The write transaction re-checks
        # each row on its own connection snapshot, so a row that diverges,
        # vanishes, or leaves the caller scope in the window between the two
        # is failed per id instead of silently re-anchored.
        failures: list[dict[str, Any]] = []
        movable: list[int] = []
        for memory_id in ids:
            memory = self._get_memory_visible(memory_id, caller)
            if not memory:
                failures.append({
                    "memory_id": memory_id,
                    "reason": "not_found_or_forbidden",
                })
                continue
            bucket = str(memory.get("workspace") or "").strip()
            canonical = str(memory.get("workspace_canonical") or "").strip()
            if canonical and bucket and canonical != bucket and not authorized:
                failures.append({
                    "memory_id": memory_id,
                    "reason": "canonical_diverged",
                    "workspace": bucket,
                    "workspace_canonical": canonical,
                })
                continue
            movable.append(memory_id)

        if not movable:
            return self.db.state.response(
                {
                    "moved": False,
                    "moved_ids": [],
                    "failed_ids": [failure["memory_id"] for failure in failures],
                    "errors": failures,
                    "new_workspace": target,
                },
                ok=False, extra_warnings=warnings,
            )

        # Canonical embedding is prepared outside the write transaction, like
        # every other canonical registration path.
        embedder, ensure_warnings = self._ensure_active_embedder()
        warnings.extend(ensure_warnings)
        target_embedding = None
        if not default_fallback:
            target_embedding = self.db.workspaces.prepare_missing_workspace_canonical_embedding(
                target, embedder,
            )

        moved: list[int] = []
        forced: list[dict[str, Any]] = []
        non_active: list[dict[str, Any]] = []
        bucket_was_new = False

        def aborted(exc: BaseException) -> dict[str, Any]:
            # The transaction rolled back: nothing moved, so every candidate
            # id is accounted for in failed_ids (reconciliation contract:
            # moved_ids + failed_ids covers the request). The vector-publish
            # warning is dropped — after rollback the canonical row it names
            # was never committed, so the retry guidance would be misleading.
            # The voided_conflict_tickets sentinel is dropped too (P2 #10):
            # the voids rolled back with the transaction, and the sentinel is
            # a structured counter the success path consumes into response
            # data — never a raw warning.
            reason_text = f"aborted: {exc}"
            abort_warnings = [
                warning for warning in warnings
                if "workspace canonical vector publish failed" not in warning
                and not warning.startswith("voided_conflict_tickets:")
            ]
            return self.db.state.response(
                {
                    "moved": False,
                    "moved_ids": [],
                    "failed_ids": [
                        failure["memory_id"] for failure in failures
                    ] + list(movable),
                    "errors": failures + [
                        {"memory_id": memory_id, "reason": reason_text}
                        for memory_id in movable
                    ],
                    "new_workspace": target,
                },
                ok=False, extra_warnings=abort_warnings,
            )

        try:
            # Same advisory flock as rename/migrate/normalize, always taken
            # before the write transaction so every registry-mutating path
            # serializes in one order; a concurrent merge can no longer
            # resurrect a spelling between the fold read and the commit.
            with database_startup_lock(self.settings.db_path), self.db.write_transaction() as conn:
                # Re-fold from the ORIGINAL request, single hop, on the
                # transaction's own snapshot. Re-folding the already-folded
                # target would follow a second alias hop that a write using
                # the same name would never take.
                locked = (
                    self.db.workspaces._mechanical_canonical_on_conn(conn, requested)
                    or requested
                )
                alias_row = conn.execute(
                    "SELECT canonical FROM workspace_aliases "
                    "WHERE alias_workspace=? AND status='confirmed' "
                    "ORDER BY updated_at DESC, canonical ASC LIMIT 1",
                    (_normalize_alias_key(requested),),
                ).fetchone()
                if alias_row is not None:
                    locked = str(alias_row["canonical"])
                if locked != target:
                    blocked = destination_error(locked)
                    if blocked is not None:
                        # Reconciliation contract: after ids were resolved the
                        # response must still account for every request id.
                        blocked.update({
                            "new_workspace": target,
                            "moved_ids": [],
                            "failed_ids": [
                                failure["memory_id"] for failure in failures
                            ] + list(movable),
                            "errors": failures + [
                                {
                                    "memory_id": memory_id,
                                    "reason": "destination_changed_after_recheck",
                                }
                                for memory_id in movable
                            ],
                        })
                        if requested != target:
                            blocked["requested_new_workspace"] = requested
                        return self.db.state.response(
                            blocked, ok=False, extra_warnings=warnings,
                        )
                    target = locked
                    # The prepared embedding belongs to the stale spelling;
                    # drop it rather than publish a mismatched vector under
                    # the rechecked canonical (a later write republishes it).
                    target_embedding = None
                bucket_was_new = conn.execute(
                    "SELECT 1 FROM workspace_canonicals WHERE name = ?", (target,),
                ).fetchone() is None
                # P2 #10: one batched prefetch per request on this transaction's
                # snapshot (no TOCTOU drift) replaces the per-id re-read and the
                # per-id EXISTS probes — rows, the destination sha-collision
                # set, and the pending queue rows by member id. Same-request
                # sha siblings stay gated sequentially: an ACTIVE row that
                # lands in the target makes a same-sha later id refuse exactly
                # as the interleaved per-id flow did.
                request_ids = [int(value) for value in movable]
                prefetched: dict[int, Any] = {}
                if request_ids:
                    prefetch_ph = ",".join("?" * len(request_ids))
                    prefetched = {
                        int(row["id"]): row
                        for row in conn.execute(
                            f"SELECT id, workspace, workspace_canonical, status, content_sha "
                            f"FROM memories WHERE id IN ({prefetch_ph})",
                            request_ids,
                        ).fetchall()
                    }
                colliding_ids = (
                    self.db.workspaces._content_sha_collision_ids_on_conn(
                        conn, target, request_ids,
                    )
                )
                queue_rows_by_id = (
                    self.db.workspaces._pending_queue_rows_by_memory_id_on_conn(
                        conn, request_ids,
                    )
                )
                target_active_shas: set[str] = set()
                for memory_id in list(movable):
                    row = prefetched.get(int(memory_id))
                    if row is None:
                        failures.append({
                            "memory_id": memory_id,
                            "reason": "not_found_or_forbidden",
                        })
                        movable.remove(memory_id)
                        continue
                    bucket = str(row["workspace"] or "").strip()
                    canonical = str(row["workspace_canonical"] or "").strip()
                    if caller.isolation == "strict":
                        effective = canonical or bucket
                        if effective not in admitted:
                            failures.append({
                                "memory_id": memory_id,
                                "reason": "not_found_or_forbidden",
                            })
                            movable.remove(memory_id)
                            continue
                    if canonical and bucket and canonical != bucket:
                        if not authorized:
                            failures.append({
                                "memory_id": memory_id,
                                "reason": "canonical_diverged",
                                "workspace": bucket,
                                "workspace_canonical": canonical,
                            })
                            movable.remove(memory_id)
                            continue
                        forced.append({
                            "memory_id": memory_id,
                            "workspace": bucket,
                            "workspace_canonical": canonical,
                        })
                    row_status = str(row["status"] or "")
                    row_sha = str(row["content_sha"] or "")
                    sha_hit = (
                        int(memory_id) in colliding_ids
                        or (
                            row_status == MemoryStatus.ACTIVE.value
                            and bool(row_sha)
                            and row_sha in target_active_shas
                        )
                    )
                    current_bucket_value = (
                        str(row["workspace_canonical"])
                        if row["workspace_canonical"] not in (None, "")
                        else str(row["workspace"] or "")
                    )
                    ok_move, move_warnings = self.db.workspaces.move_memory_workspace_on_conn(
                        conn, memory_id, target,
                        precomputed_embedding=target_embedding,
                        allow_default=default_fallback,
                        current_bucket=current_bucket_value,
                        sha_collision=sha_hit,
                        queue_rows=queue_rows_by_id.get(int(memory_id)) or [],
                    )
                    if not ok_move:
                        reason = "; ".join(move_warnings) or "not_found_or_forbidden"
                        if "memory id not found" in reason:
                            reason = "not_found_or_forbidden"
                        failures.append({"memory_id": memory_id, "reason": reason})
                        movable.remove(memory_id)
                        continue
                    if default_fallback:
                        # Every fallback landing is auditable (owner-visible
                        # via doctor's normalize board) — default must never
                        # become an untraceable dumping ground.
                        conn.execute(
                            """INSERT INTO normalize_audit(
                                 memory_id, from_workspace, to_workspace, gate, status, created_at)
                               VALUES(?,?,?,?, 'manual_move', ?)""",
                            (int(memory_id), str(row["workspace"] or ""), "default",
                             json.dumps({"default_fallback": True, "reason": reason},
                                        ensure_ascii=False),
                             utc_now_iso()),
                        )
                    for warning in move_warnings:
                        if warning not in warnings:
                            warnings.append(warning)
                    if row_status not in (MemoryStatus.ACTIVE.value, MemoryStatus.PENDING.value):
                        non_active.append({"memory_id": memory_id, "status": row_status})
                    if row_status == MemoryStatus.ACTIVE.value and row_sha:
                        # P2 #10: sequential dedup-gate parity — this ACTIVE row
                        # now lives in the target bucket, so a same-sha sibling
                        # later in the same request must refuse.
                        target_active_shas.add(row_sha)
                    moved.append(memory_id)
        except OSError as exc:
            # Same rollback-time filtering as aborted(): the publish-failure
            # guidance and the voided-ticket sentinel both describe work that
            # the rollback undid (P2 #10).
            abort_warnings = [
                warning for warning in warnings
                if "workspace canonical vector publish failed" not in warning
                and not warning.startswith("voided_conflict_tickets:")
            ]
            return self.db.state.response(
                {
                    "moved": False,
                    "error": f"workspace move lock unavailable: {exc}",
                    "moved_ids": [],
                    "failed_ids": [
                        failure["memory_id"] for failure in failures
                    ] + list(movable),
                    "errors": failures + [
                        {"memory_id": memory_id, "reason": f"aborted: {exc}"}
                        for memory_id in movable
                    ],
                    "new_workspace": target,
                },
                ok=False, extra_warnings=abort_warnings,
            )
        except sqlite3.Error as exc:
            return aborted(exc)

        # 0.16.0 §6⑯: a move VOIDS the moved memory's non-terminal conflict
        # tickets (releasing slot/candidate identities, suppressing nothing);
        # the pipeline re-establishes them in the new bucket on its next pass.
        # The store reports the count via a structured sentinel warning —
        # surface it as response data, never as a raw warning.
        voided_ticket_total = 0
        for warning in list(warnings):
            if warning.startswith("voided_conflict_tickets:"):
                try:
                    voided_ticket_total += int(warning.split(":", 1)[1])
                except ValueError:
                    pass
                warnings.remove(warning)

        data: dict[str, Any] = {
            "moved": not failures,
            "moved_ids": moved,
            "failed_ids": [failure["memory_id"] for failure in failures],
            "errors": failures,
            "new_workspace": target,
        }
        if voided_ticket_total:
            data["conflict_tickets_voided"] = {
                "count": voided_ticket_total,
                "note": (
                    "The moved memories' non-terminal conflict tickets were voided "
                    "(released, not suppressed). The scan pipeline re-establishes "
                    "them inside the new bucket on its next pass."
                ),
            }
        if requested != target:
            data["requested_new_workspace"] = requested
        elif twin_redirect_from is not None:
            data["requested_new_workspace"] = "mema-twin"
        if forced:
            data["forced_reanchored"] = {
                "memory_ids": [entry["memory_id"] for entry in forced],
                "details": forced,
                "note": (
                    "These rows had workspace_canonical pointing somewhere other than "
                    "their workspace bucket; both columns were re-anchored to the "
                    "destination. Normalization rules are unchanged: future writes "
                    "using the old workspace name still resolve to the old registered "
                    "canonical. To reroute that name, use migrate_workspace or "
                    "rename_workspace_canonical."
                ),
                "suggested_call": {
                    "tool": "memory_govern",
                    "action": "migrate_workspace",
                    "data": {"from": forced[0]["workspace_canonical"], "to": target},
                },
                "authorization_required": True,
            }
        if non_active:
            data["moved_non_active"] = non_active
        if default_fallback and moved:
            data["default_fallback"] = {
                "count": len(moved),
                "reason": reason,
                "note": (
                    "These memories were parked in the global default pool "
                    "because no suitable project bucket was found. Tell the "
                    "user: they should re-home them via memory_govern("
                    "action='move_memories_workspace') when a bucket is decided."
                ),
            }
        if any("workspace canonical vector publish failed" in warning for warning in warnings):
            data["workspace_vector_publish"] = {
                "status": "pending_retry",
                "canonical": target,
                "retry": (
                    "After sqlite-vec and embedding configuration recover, write "
                    "another memory using this workspace to retry publication."
                ),
                "repair_task_available": False,
            }
        response = self.db.state.response(data, ok=not failures, extra_warnings=warnings)
        if default_fallback and moved:
            # The user-facing prompt the owner rule requires: a default
            # fallback landing must never pass silently.
            response.setdefault("notices", []).append({
                "type": "default_fallback",
                "severity": "info",
                "workspace": "default",
                "message": (
                    f"{len(moved)} memorie(s) moved back to the default pool "
                    "(no suitable bucket found). This needs the user's "
                    "attention: re-home or confirm the placement."
                ),
                "action_required": "review_default_fallback",
                "review_call": {"tool": "memory_review", "view": "doctor", "data": {}},
            })
        if bucket_was_new and moved:
            # Deliberately delivered even on partial failure: the bucket row
            # IS committed in that case, and dropping the review notice would
            # hide a registered-but-unreviewed workspace.
            response.setdefault("notices", []).append({
                "type": "workspace_review",
                "severity": "info",
                "workspace": target,
                "message": (
                    f"New workspace {target!r} was registered. "
                    "Review the workspace registry for duplicates before confirming it."
                ),
                "action_required": "review_workspace_registry",
                "review_call": {"tool": "memory_review", "view": "doctor", "data": {}},
                "confirm_call": {"tool": "memory_govern", "action": "confirm_workspaces", "data": {}},
                "authorization_required": True,
            })
        return response
    @staticmethod
    def _merge_conflict_membership_on_conn(
        conn: sqlite3.Connection, memory_ids: list[int],
    ) -> dict[int, list[dict[str, Any]]]:
        """Group open/applying conflicts whose pinned members include any id.

        Uses the caller-owned transaction connection: reading through a second
        connection inside write_transaction would observe the pre-transaction
        snapshot. Mirrors the SQL of list_open_conflicts_for_memory_ids.
        """
        membership: dict[int, list[dict[str, Any]]] = {}
        wanted = sorted({int(value) for value in memory_ids})
        if not wanted:
            return membership
        rows = conn.execute(
            "SELECT DISTINCT c.* FROM conflicts AS c "
            "JOIN json_each(c.member_versions) AS member "
            "JOIN json_each(?) AS wanted "
            "ON CAST(json_extract(member.value,'$.memory_id') AS INTEGER)=CAST(wanted.value AS INTEGER) "
            "WHERE c.status IN ('open','applying') ORDER BY c.created_at DESC,c.id DESC",
            (json.dumps(wanted, ensure_ascii=False, sort_keys=True, separators=(",", ":")),),
        ).fetchall()
        for row in rows:
            group = dict(row)
            try:
                member_ids = {
                    int(member["memory_id"])
                    for member in json.loads(str(group.get("member_versions") or "[]"))
                }
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
            for memory_id in member_ids & set(wanted):
                membership.setdefault(memory_id, []).append(group)
        return membership
