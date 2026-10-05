"""内容域组（merge_memories/entity 两方法/edit/history/cleanup_history，从 operations.py 搬出，拆分批 ④ 纯移动）。
"""
from __future__ import annotations


import json
import sqlite3
from typing import Any, TYPE_CHECKING

from ..acl import CallerWorkspace, WorkspaceScope, workspace_scope_sql, forbidden_payload, raw_workspace
from ..embedder import ManagedEmbedder
from ..models import ProtectionLevel
from ..text import canon_entity as _canon_entity, canon_scope as _canon_scope

if TYPE_CHECKING:
    from ..config import Settings
    from ..db import MemoryDB
    from ..update_monitor import UpdateMonitor
    from ..tools import MemoryTools

class _OpsContent:
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
        @staticmethod
        def _merge_conflict_membership_on_conn(*args: Any, **kwargs: Any) -> dict[int, list[dict[str, Any]]]: ...

    def memory_merge_memories(
        self,
        survivor_id: int,
        loser_ids: list[int],
        reason: str,
        merged_content: str | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Merge near-duplicate memories: keep the survivor, supersede losers.

        Deliberately NOT the conflict state machine: near-duplicates are not
        conflicts, so losers are superseded through update_memory_on_conn
        (version bump, evidence pinning, and vec0 parent_status propagation all
        follow automatically) and a persistent ``merged_into`` pointer is
        written into each loser's metadata (read-modify-write: the update
        primitive replaces metadata wholesale). This is the library's first
        persisted successor pointer — retire's ``superseded_by`` is validated
        but never stored.

        Losers that are members of an open/applying conflict group are
        rejected per-id: members are pinned at id@version and superseding one
        cannot close the group, so a hard merge would wedge it. The survivor
        may sit in a group — without merged_content nothing changes for the
        group; with merged_content the edit may stale pinned members and the
        edit-path attention_required contract applies.
        """
        authorized = self._is_truthy(authorized)
        if not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required to merge memories", "merged": False},
                ok=False,
            )
        try:
            survivor_id_int = int(survivor_id)
        except (TypeError, ValueError):
            return self.db.state.response(
                {"error": "survivor_id must be an integer", "merged": False}, ok=False,
            )
        raw_losers = loser_ids if isinstance(loser_ids, list) else []
        loser_ints: list[int] = []
        for value in raw_losers:
            try:
                loser_ints.append(int(value))
            except (TypeError, ValueError):
                return self.db.state.response(
                    {"error": "loser_ids must contain integers only", "merged": False}, ok=False,
                )
        loser_ints = sorted(set(loser_ints))
        if not loser_ints:
            return self.db.state.response(
                {"error": "loser_ids requires 1-50 positive integer ids", "merged": False}, ok=False,
            )
        if len(loser_ints) > 50:
            return self.db.state.response(
                {"error": "loser_ids requires 1-50 positive integer ids", "merged": False}, ok=False,
            )
        if survivor_id_int in loser_ints:
            return self.db.state.response(
                {"error": "survivor_id must not appear in loser_ids", "merged": False}, ok=False,
            )
        if not str(reason or "").strip():
            return self.db.state.response(
                {"error": "merge requires reason and authorized=true", "merged": False}, ok=False,
            )
        if merged_content is not None and not str(merged_content).strip():
            # A provided-but-blank value is a caller mistake; the edit
            # primitive would refuse to wipe content, so fail loudly here
            # instead of silently treating it as "no edit requested".
            return self.db.state.response(
                {"error": "merged_content must be non-empty when provided", "merged": False}, ok=False,
            )

        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied

        merged_ids: list[int] = []
        deduped_ids: list[int] = []
        failed_ids: list[dict[str, Any]] = []
        survivor_edited = False
        attention_groups: list[dict[str, Any]] = []
        survivor_record: dict[str, Any] | None = None
        survivor_history_id: int | None = None
        try:
            with self.db.write_transaction() as conn:
                survivor = self.db.get_memory_on_conn(conn, survivor_id_int)
                if caller.isolation == "strict":
                    if not survivor or raw_workspace(survivor) not in set(caller.scope_canonicals()):
                        raise ValueError(
                            f"survivor memory id {survivor_id_int} not found or not accessible"
                        )
                elif not survivor:
                    raise ValueError(f"survivor memory id {survivor_id_int} not found")
                if survivor.get("status") != "active":
                    raise ValueError(
                        f"survivor is not active (status={survivor.get('status')}); pick a live memory to keep"
                    )
                survivor_canonical = raw_workspace(survivor)
                conflict_membership = self._merge_conflict_membership_on_conn(
                    conn, loser_ints + [survivor_id_int],
                )
                # Survivor-side group check only matters when the survivor is
                # about to be edited (merged_content is validated non-blank
                # above); a no-op merge does not stale any pin.
                if merged_content is not None:
                    attention_groups = conflict_membership.get(survivor_id_int, [])
                for loser_id in loser_ints:
                    loser = self.db.get_memory_on_conn(conn, loser_id)
                    if caller.isolation == "strict":
                        if not loser or raw_workspace(loser) not in set(caller.scope_canonicals()):
                            failed_ids.append({
                                "memory_id": loser_id, "error": "not_found_or_forbidden",
                            })
                            continue
                    elif not loser:
                        failed_ids.append({"memory_id": loser_id, "error": "not_found"})
                        continue
                    status = str(loser.get("status") or "")
                    if status in {"superseded", "deleted"}:
                        raw_metadata = loser.get("metadata")
                        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
                        pointer = metadata.get("merged_into")
                        try:
                            same_target = pointer is not None and int(pointer) == survivor_id_int
                        except (TypeError, ValueError):
                            same_target = False
                        if status == "superseded" and same_target:
                            deduped_ids.append(loser_id)
                        elif status == "superseded":
                            if pointer is None:
                                # Superseded/retired outside merge carries no
                                # merged_into pointer — name the actual state
                                # instead of claiming a merge that never ran.
                                failed_ids.append({
                                    "memory_id": loser_id, "error": "already_superseded",
                                    "merged_into": None, "status": status,
                                })
                            else:
                                failed_ids.append({
                                    "memory_id": loser_id, "error": "already_merged",
                                    "merged_into": pointer,
                                })
                        else:
                            failed_ids.append({
                                "memory_id": loser_id, "error": "not_active", "status": status,
                            })
                        continue
                    if raw_workspace(loser) != survivor_canonical:
                        failed_ids.append({
                            "memory_id": loser_id, "error": "workspace_mismatch",
                            "survivor_workspace": survivor_canonical,
                            "loser_workspace": raw_workspace(loser),
                        })
                        continue
                    groups = conflict_membership.get(loser_id, [])
                    if groups:
                        # Members are pinned at id@version; superseding one
                        # leaves the group open forever. Reject and steer the
                        # caller to the conflict flow instead.
                        failed_ids.append({
                            "memory_id": loser_id, "error": "conflict_member",
                            "attention_required": True,
                            "conflicts": [
                                {
                                    "conflict_id": group.get("id"),
                                    "revision": group.get("revision"),
                                    "status": group.get("status"),
                                    "conflict_point": group.get("conflict_point"),
                                }
                                for group in groups
                            ],
                            "hint": (
                                "resolve the group first via judge/replan/resolve_conflict "
                                "or record_conflict(status='not_a_conflict'), then merge"
                            ),
                        })
                        continue
                    # metadata is replaced wholesale by update_memory_on_conn,
                    # so merge the pointer into the existing dict first.
                    raw_metadata = loser.get("metadata")
                    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
                    merged_metadata = dict(metadata)
                    merged_metadata["merged_into"] = survivor_id_int
                    merged_metadata["merge_reason"] = str(reason).strip()
                    status_updated = self.db.update_memory_on_conn(
                        conn,
                        loser_id,
                        {
                            "status": "superseded",
                            "protection_level": ProtectionLevel.NORMAL.value,
                            "metadata": merged_metadata,
                        },
                    )
                    if not status_updated:
                        failed_ids.append({"memory_id": loser_id, "error": "update_failed"})
                        continue
                    merged_ids.append(loser_id)
                if not merged_ids and not deduped_ids:
                    raise ValueError("no loser could be merged; see failed_ids")
                if merged_content is not None:
                    edit_result = self.db.edit_memory_intent(
                        survivor_id_int,
                        new_content=str(merged_content),
                        reason=f"merge survivors: {str(reason).strip()}",
                        authorized=True,
                        conn=conn,
                    )
                    if edit_result.get("outcome") != "edited":
                        raise ValueError(
                            f"survivor content merge failed: {edit_result.get('error') or edit_result.get('outcome')}"
                        )
                    survivor_edited = True
                    survivor_history_id = int(edit_result.get("history_id") or 0) or None
                survivor_record = self.db.get_memory_on_conn(conn, survivor_id_int)
        except ValueError as exc:
            return self.db.state.response(
                {"error": str(exc), "merged": False, "failed_ids": failed_ids}, ok=False,
                extra_warnings=list(caller.warnings),
            )
        except sqlite3.Error as exc:
            return self.db.state.response(
                {
                    "error": f"merge failed; transaction rolled back: {exc}",
                    "merged": False, "survivor_id": survivor_id_int,
                },
                ok=False, extra_warnings=list(caller.warnings),
            )
        except Exception as exc:
            return self.db.state.response(
                {
                    "error": f"merge failed; transaction rolled back: {exc}",
                    "merged": False, "survivor_id": survivor_id_int,
                },
                ok=False, extra_warnings=list(caller.warnings),
            )
        data: dict[str, Any] = {
            "merged": True,
            "survivor_id": survivor_id_int,
            "merged_ids": merged_ids,
            "deduped_ids": deduped_ids,
            "failed_ids": failed_ids,
            "survivor_edited": survivor_edited,
            "history_id": survivor_history_id,
            "record": survivor_record,
        }
        if survivor_edited:
            data["evidence_index"], data["semantic_conflict_check"] = (
                self._post_commit(survivor_id_int, survivor_record, recheck_conflicts=True)
            )
            data["post_commit"] = {"status": "recheck_conflicts"}
        else:
            # Superseding losers does not change the survivor's content, so no
            # conflict recheck is owed; flag it explicitly for API stability.
            data["post_commit"] = {"status": "skipped", "reason": "no_survivor_change"}
        if attention_groups:
            ids = ", ".join(f"#{int(group.get('id') or 0)}" for group in attention_groups)
            data["attention_required"] = True
            data["action_required"] = "review_unresolved_conflicts"
            data["unresolved_conflicts"] = [
                {
                    "conflict_id": group.get("id"),
                    "revision": group.get("revision"),
                    "status": group.get("status"),
                    "conflict_point": group.get("conflict_point"),
                }
                for group in attention_groups
            ]
            data["attention_summary"] = (
                f"edited survivor is a member of unresolved conflict group(s) {ids}; "
                "this edit may stale their pinned member versions — review via "
                "memory_review(view='conflict_detail') and judge / replan / "
                "resolve as appropriate"
            )
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))
    def memory_set_entity(
        self,
        memory_id: int,
        entity: str | None = None,
        scope: str | None = None,
        clear: bool = False,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Set canonical metadata.entity/scope without creating content history."""
        authorized = self._is_truthy(authorized)
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer"}, ok=False)
        if not clear and not _canon_entity(entity):
            return self.db.state.response({"error": "entity is required unless clear=true"}, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation == "strict" and self._get_memory_visible(memory_id_int, caller) is None:
            return self.db.state.response(
                forbidden_payload("memory", workspace=caller),
                ok=False,
                extra_warnings=list(caller.warnings),
            )
        set_fields: dict[str, Any] = {}
        clear_fields: list[str] = []
        if clear:
            clear_fields.append("entity")
        else:
            set_fields["entity"] = _canon_entity(entity)
        if scope is not None:
            canonical_scope = _canon_scope(scope)
            if canonical_scope:
                set_fields["scope"] = canonical_scope
            else:
                clear_fields.append("scope")
        try:
            with self.db.write_transaction() as conn:
                current = self.db.get_memory_on_conn(conn, memory_id_int)
                if current is None:
                    result = {"outcome": "not_found", "memory_id": memory_id_int}
                elif caller.isolation == "strict" and raw_workspace(current) not in set(caller.scope_canonicals()):
                    result = {"outcome": "workspace_mismatch", "memory_id": memory_id_int}
                else:
                    result = self.db.update_metadata_fields_low_side_effect_on_conn(
                        conn, memory_id_int, set_fields=set_fields,
                        clear_fields=clear_fields, authorized=authorized,
                    )
        except sqlite3.Error:
            result = {"outcome": "error", "memory_id": memory_id_int}
        outcome = result.get("outcome")
        if outcome not in {"updated", "no_change"}:
            ok = False
            error = {
                "forbidden": "authorized=True is required for locked/user_confirmed memory",
                "not_found": "memory id not found",
                "not_active": "memory is not active",
                "unavailable": "database not available",
                "workspace_mismatch": "forbidden_strict_workspace",
            }.get(str(outcome), "entity update failed")
            if outcome == "workspace_mismatch":
                return self.db.state.response(
                    forbidden_payload("memory", workspace=caller), ok=False,
                    extra_warnings=list(caller.warnings),
                )
            return self.db.state.response({"error": error, **result}, ok=ok)
        data: dict[str, Any] = {
            "updated": outcome == "updated", "outcome": outcome,
            "memory_id": memory_id_int, "metadata": result.get("metadata"),
        }
        data["record"] = self.db.get_memory(memory_id_int)
        if outcome == "updated":
            data["evidence_index"], data["semantic_conflict_check"] = self._post_commit(
                memory_id_int, data["record"], recheck_conflicts=False,
            )
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        # Gate-v2 G3: entity/scope are retired — the storage strip turns any
        # set/clear into a metadata no-op, so say so instead of silently
        # doing nothing.
        warnings = list(caller.warnings)
        warnings.append("metadata entity/scope 已废弃（0.17 门 v2），无需再传")
        return self.db.state.response(data, extra_warnings=warnings)

    def memory_list_entities(
        self, limit: int = 50, include_unassigned: bool = True, **_: Any,
    ) -> dict[str, Any]:
        try:
            limit_int = max(1, min(500, int(limit)))
        except (TypeError, ValueError):
            return self.db.state.response({"error": "limit must be an integer"}, ok=False)
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation != "strict":
            return self.db.state.response(
                self.db.list_entities(limit=limit_int, include_unassigned=bool(include_unassigned))
            )
        counts: dict[str, int] = {}
        sample: dict[str, int] = {}
        unassigned: list[int] = []
        total = 0
        with self.db.connection() as conn:
            scope_sql, scope_params = workspace_scope_sql(
                "COALESCE(NULLIF(workspace_canonical, ''), workspace)", caller.scope_canonicals(),
            )
            rows = conn.execute(
                f"SELECT id, metadata FROM memories WHERE status='active' AND {scope_sql} ORDER BY id",
                scope_params,
            ).fetchall()
        for row in rows:
            total += 1
            try:
                md = json.loads(row["metadata"] or "{}")
                if not isinstance(md, dict):
                    md = {}
            except Exception:
                md = {}
            entity = _canon_entity(md.get("entity"))
            if entity:
                counts[entity] = counts.get(entity, 0) + 1
                sample.setdefault(entity, int(row["id"]))
            elif include_unassigned:
                unassigned.append(int(row["id"]))
        data = {
            "entities": [
                {"entity": entity, "count": counts[entity], "sample_memory_id": sample[entity]}
                for entity in sorted(counts, key=lambda key: (-counts[key], key))[:limit_int]
            ],
            "distinct_entities": len(counts),
            "assigned_count": sum(counts.values()),
            "total_active": total,
            "unassigned_count": total - sum(counts.values()),
            "unassigned_ids": unassigned[:limit_int],
            **caller.response_fields(),
        }
        return self.db.state.response(data, extra_warnings=list(caller.warnings))
    def memory_edit(
        self,
        memory_id: int,
        new_content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        patches: list[dict[str, Any]] | None = None,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        reason: str = "",
        authorized: bool = False,
        tags_only: bool = False,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """In-place edit a memory's content or tags.

        Edit modes:
          * tags-only (v0.7.6): pass ``tags_only=True`` with
            ``add_tags``/``remove_tags`` to update tags without touching
            content, memory_history, version, or the evidence index
            (content is unchanged, so no re-embedding is needed).
            FTS is re-synced because tags are indexed in FTS5.
          * full replace: pass ``new_content`` (old_text/new_text/patches
            must be empty)
          * single partial replace: pass ``old_text`` + ``new_text`` for an
            exact substring substitution (new_content/patches must be empty)
          * sequential partial replace (v0.15.12): pass ``patches=[{old_text,
            new_text}, ...]`` (1..8 pairs, new_content/old_text/new_text
            must be empty) — applied in order, each match computed on the
            result of the previous patches, first occurrence replaced. Any
            miss rejects the whole call atomically (stale_edit with
            patch_index); one call = one version bump + one history row +
            one evidence republish + one post-commit check.

        Tags in content modes: ``add_tags``/``remove_tags`` also work in
        full/partial mode — they overlay on top of ``new_tags`` (if given)
        else the current tags, mirroring the tags-only path's
        order-preserving dedup (remove first, then add). ``new_tags`` alone
        is a full replace.

        Authorization (layered): normal records edit freely; ``locked`` /
        ``user_confirmed`` records require ``authorized=True`` (mirrors
        ``memory_supersede``). Records already superseded/deleted are rejected.
        """
        authorized = self._is_truthy(authorized)
        tags_only = self._is_truthy(tags_only)
        try:
            memory_id_int = int(memory_id)
        except (TypeError, ValueError):
            return self.db.state.response({"error": "memory_id must be an integer", "edited": False}, ok=False)

        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation == "strict" and self._get_memory_visible(memory_id_int, caller) is None:
            return self.db.state.response(
                forbidden_payload("memory", workspace=caller),
                ok=False,
                extra_warnings=list(caller.warnings),
            )

        # ---- tags-only fast path (v0.7.6) ----
        # P2 #8: the refusal set is the content parameters ∪ the field
        # replaces (new_subject/new_tags) — a silent ignore here would eat
        # the caller's edit intent; tag changes go through add_tags/remove_tags.
        if tags_only and (
            new_content is not None
            or old_text is not None
            or new_text is not None
            or patches is not None
            or new_subject is not None
            or new_tags is not None
        ):
            # A silent ignore here would eat the caller's edit intent: the
            # tags-only path never touches content, so a combined call must
            # be rejected loudly instead of half-executed.
            return self.db.state.response(
                {
                    "error": (
                        "tags_only=true cannot be combined with content edits "
                        "(new_content / old_text+new_text / patches) or field replaces "
                        "(new_subject / new_tags); use add_tags/remove_tags for tag "
                        "changes, or make two calls"
                    ),
                    "edited": False,
                },
                ok=False,
            )
        if tags_only:
            tag_result: dict[str, Any]
            try:
                with self.db.write_transaction() as conn:
                    current = self.db.get_memory_on_conn(conn, memory_id_int)
                    if current is None:
                        tag_result = {"outcome": "not_found", "memory_id": memory_id_int}
                    elif caller.isolation == "strict" and raw_workspace(current) not in set(caller.scope_canonicals()):
                        tag_result = {"outcome": "workspace_mismatch", "memory_id": memory_id_int}
                    else:
                        tag_result = self.db.update_tags_low_side_effect(
                            memory_id_int,
                            add_tags=add_tags or [],
                            remove_tags=remove_tags or [],
                            authorized=authorized,
                            conn=conn,
                        )
            except sqlite3.Error:
                tag_result = {"outcome": "error", "memory_id": memory_id_int}
            outcome = tag_result.get("outcome")
            if outcome == "workspace_mismatch":
                return self.db.state.response(
                    forbidden_payload("memory", workspace=caller), ok=False,
                    extra_warnings=list(caller.warnings),
                )
            if outcome == "updated":
                updated_mem = self.db.get_memory(memory_id_int)
                # Tags shape the duplicate-hint recall vector even though the
                # content (and version) is unchanged.
                self._tools._write_pipeline.refresh_subject_tags_vector(memory_id_int)
                # ...and the C3a summary vector likewise (tags are its input).
                self._tools._write_pipeline.refresh_summary_vector(memory_id_int)
                data = {
                    "edited": True,
                    "tags_only": True,
                    "memory_id": memory_id_int,
                    "tags": tag_result.get("tags"),
                    "record": updated_mem,
                }
                return self.db.state.response(data)
            if outcome == "no_change":
                return self.db.state.response({
                    "edited": False,
                    "tags_only": True,
                    "already_completed": True,
                    "memory_id": memory_id_int,
                    "tags": tag_result.get("tags"),
                })
            if outcome == "tags_over_limit":
                # 0.16.0 §6⑮: the whole tags-only call is refused; the error
                # carries the merged total and the remove-first hint.
                return self.db.state.response({
                    "error": tag_result.get("error"),
                    "current_total": tag_result.get("current_total"),
                    "cap": tag_result.get("cap"),
                    "edited": False,
                }, ok=False)
            if outcome == "forbidden":
                return self.db.state.response({
                    "error": (
                        f"memory is protected (protection_level={tag_result.get('protection_level')}, "
                        f"source_type={tag_result.get('source_type')}); authorized=True required to edit tags"
                    ),
                    "edited": False,
                }, ok=False)
            if outcome == "not_found":
                return self.db.state.response({"error": f"memory id {memory_id_int} not found", "edited": False}, ok=False)
            if outcome == "not_active":
                return self.db.state.response({
                    "error": f"memory is not active (status={tag_result.get('status')}); cannot edit tags",
                    "edited": False,
                }, ok=False)
            if outcome == "unavailable":
                return self.db.state.response({"error": "database not available", "edited": False}, ok=False)
            # outcome == "error"
            return self.db.state.response(
                {"error": "tags-only edit failed; transaction rolled back, no changes applied", "edited": False},
                ok=False,
            )

        # ---- full / partial content edit ----
        edit_result: dict[str, Any]
        expected_version_raw = _.get("expected_version")
        expected_hash = _.get("expected_content_hash") or _.get("content_hash")
        try:
            expected_version = int(expected_version_raw) if expected_version_raw is not None else None
        except (TypeError, ValueError):
            return self.db.state.response({"error": "expected_version must be an integer", "edited": False}, ok=False)
        try:
            with self.db.write_transaction() as conn:
                current = self.db.get_memory_on_conn(conn, memory_id_int)
                if current is None:
                    edit_result = {"outcome": "not_found", "memory_id": memory_id_int}
                elif caller.isolation == "strict" and raw_workspace(current) not in set(caller.scope_canonicals()):
                    edit_result = {"outcome": "workspace_mismatch", "memory_id": memory_id_int}
                else:
                    edit_result = self.db.edit_memory_intent(
                        memory_id_int,
                        new_content=new_content,
                        old_text=old_text,
                        new_text=new_text,
                        patches=patches,
                        new_subject=new_subject,
                        new_tags=new_tags,
                        add_tags=add_tags,
                        remove_tags=remove_tags,
                        reason=reason or None,
                        authorized=authorized,
                        expected_version=expected_version,
                        expected_content_hash=str(expected_hash) if expected_hash is not None else None,
                        conn=conn,
                    )
        except sqlite3.Error:
            edit_result = {"outcome": "error", "memory_id": memory_id_int}
        outcome = edit_result.get("outcome")
        if outcome != "edited":
            if outcome == "workspace_mismatch":
                return self.db.state.response(
                    forbidden_payload("memory", workspace=caller), ok=False,
                    extra_warnings=list(caller.warnings),
                )
            if outcome == "not_found":
                error = "memory id not found"
            elif outcome == "not_active":
                status = edit_result.get("status")
                error = f"memory already {status}" if status in {"superseded", "deleted"} else f"memory is not active (status={status}); cannot edit"
            elif outcome == "forbidden":
                error = "authorized=True is required to edit a locked/user_confirmed memory"
            elif outcome == "stale_edit":
                error = edit_result.get("error") or f"stale_edit: {edit_result.get('reason') or 'current memory changed'}"
            elif outcome == "tags_over_limit":
                # 0.16.0 §6⑮: over-cap tag merges refuse the whole edit.
                return self.db.state.response({
                    "error": edit_result.get("error"),
                    "current_total": edit_result.get("current_total"),
                    "cap": edit_result.get("cap"),
                    "edited": False,
                }, ok=False, extra_warnings=list(caller.warnings))
            elif outcome == "unavailable":
                error = "database not available"
            else:
                error = edit_result.get("error") or "edit failed (db not writable)"
            return self.db.state.response({"error": error, "edited": False, **edit_result}, ok=False)
        history_id = int(edit_result["history_id"])
        updated = edit_result.get("record") or self.db.get_memory(memory_id_int)
        data = {
            "edited": True,
            "memory_id": memory_id_int,
            "new_version": int(updated.get("version") or 1) if updated else None,
            "history_id": history_id,
            "record": updated,
        }
        # 0.17.1：claims 数据层全退（owner 拍板连表删）——编辑落库/继承钩子
        # 一并退役，update 的 claims 参数出 schema（未知键软着陆：警告+忽略）。
        data["evidence_index"], data["semantic_conflict_check"] = (
            self._post_commit(memory_id_int, updated, recheck_conflicts=True)
        )
        # 0.16.12 P2-T1: the two recall-vector refreshes re-embed from the
        # row's derived inputs — skip whichever re-embed cannot change the
        # vector. The pre-edit row read inside the same write transaction
        # (:current above) is the exact baseline edit_memory_intent applied
        # its deltas to, so this comparison is race-free. The separate
        # post-commit re-read of the row is gone (P2-T3): edit_result always
        # carries the transaction's own post-edit record.
        from .operations import _embed_input_profile
        pre_inputs = _embed_input_profile(current)
        post_inputs = _embed_input_profile(updated)
        if pre_inputs[:2] != post_inputs[:2]:
            # Subject/tags shape the duplicate-hint recall vector.
            self._tools._write_pipeline.refresh_subject_tags_vector(memory_id_int)
        if pre_inputs != post_inputs:
            # C3a summary vector: subject+tags+content segments (edits bump
            # the version, which is the anomaly vote's refresh point — but a
            # metadata-only edit with identical derived inputs re-embeds the
            # same text for nothing).
            self._tools._write_pipeline.refresh_summary_vector(memory_id_int)
        unresolved = self.db.conflicts.list_open_conflicts_for_memory_ids(
            [memory_id_int], include_applying=True,
        )
        if unresolved:
            # Content edits bump the version, staling the group's pinned member
            # snapshot (open) or the in-flight plan's expected_version
            # (applying). Prompt synchronously — same contract the search path
            # signals with — so the caller handles the group instead of
            # leaving it wedged. tags-only edits keep the version and stay
            # silent; the apply flow's own writes go through memory_govern.
            ids = ", ".join(f"#{int(group.get('id') or 0)}" for group in unresolved)
            data["attention_required"] = True
            data["action_required"] = "review_unresolved_conflicts"
            data["unresolved_conflicts"] = [
                {
                    "conflict_id": group.get("id"),
                    "revision": group.get("revision"),
                    "status": group.get("status"),
                    "conflict_point": group.get("conflict_point"),
                }
                for group in unresolved
            ]
            data["attention_summary"] = (
                f"edited memory is a member of unresolved conflict group(s) {ids}; "
                "this edit may stale their pinned member versions — review via "
                "memory_review(view='conflict_detail') and judge / replan / "
                "resolve as appropriate"
            )
        return self.db.state.response(data)

    def memory_history(self, memory_id: int, **_: Any) -> dict[str, Any]:
        """View the version-chain (historical snapshots) of a memory, newest
        version first. Read-only; does not modify any table.
        """
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        memory = self._get_memory_visible(int(memory_id), caller)
        if not memory:
            data: dict[str, Any] = {"error": "memory id not found"}
            if caller.isolation == "strict":
                data.update(caller.response_fields())
            return self.db.state.response(data, ok=False, extra_warnings=list(caller.warnings))
        history = self.db.list_history(int(memory_id))
        history_data: dict[str, Any] = {
            "memory_id": int(memory_id),
            "current_version": int(memory.get("version") or 1),
            "history": history,
            "count": len(history),
        }
        if self.settings.include_size:
            # v0.15.6: the shared size block (find/read/expired/history one
            # switch). list_history is an unbounded SELECT * of full version
            # snapshots — long chains are exactly where the token number
            # matters. The display_hint carries the number as a
            # report-this-cost instruction, silent on empty history.
            from ..tokens import meter_payloads

            size_block = meter_payloads(history)
            display_hint = None
            if history:
                display_hint = (
                    f"history (~{size_block['tokens_estimate']} tokens returned for "
                    f"{len(history)} version snapshot{'s' if len(history) != 1 else ''}): "
                    "report this recall cost when citing it."
                )
            history_data["size"] = {**size_block, "display_hint": display_hint}
        if caller.isolation == "strict":
            history_data.update(caller.response_fields())
        return self.db.state.response(history_data, extra_warnings=list(caller.warnings))

    def memory_cleanup_history(
        self,
        memory_id: int | None = None,
        older_than_days: int | None = None,
        authorized: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Delete historical snapshots from ``memory_history``.

        Scope:
          * ``memory_id`` set: clean only that memory's history
          * ``older_than_days`` set: clean only snapshots older than N days
          * both set: both filters apply
          * neither set (full cleanup): **requires ``authorized=True``** as an
            explicit confirmation gate

        SAFETY: this tool only ever deletes from memory_history. The memories
        table (active records) is never touched, regardless of arguments.
        """
        authorized = self._is_truthy(authorized)
        full_cleanup = memory_id is None and older_than_days is None
        if older_than_days is not None and int(older_than_days) < 0:
            return self.db.state.response(
                {"error": "older_than_days must be >= 0", "cleaned": 0},
                ok=False,
            )
        if full_cleanup and not authorized:
            return self.db.state.response(
                {"error": "authorized=True is required for full history cleanup (no memory_id / older_than_days filter)", "cleaned": 0},
                ok=False,
            )
        caller = self._caller_workspace(_.get("workspace"))
        denied = self._strict_acl_unavailable(caller)
        if denied is not None:
            return denied
        if caller.isolation == "strict":
            if memory_id is None:
                return self.db.state.response(
                    forbidden_payload("memory_history", workspace=caller, reason="workspace_scoped_cleanup_requires_memory_id"),
                    ok=False,
                    extra_warnings=list(caller.warnings),
                )
            if self._get_memory_visible(int(memory_id), caller) is None:
                return self.db.state.response(
                    forbidden_payload("memory_history", workspace=caller),
                    ok=False,
                    extra_warnings=list(caller.warnings),
                )
        try:
            with self.db.write_transaction() as conn:
                if caller.isolation == "strict" and memory_id is not None:
                    current = self.db.get_memory_on_conn(conn, int(memory_id))
                    if current is None or raw_workspace(current) not in set(caller.scope_canonicals()):
                        return self.db.state.response(
                            forbidden_payload("memory_history", workspace=caller),
                            ok=False, extra_warnings=list(caller.warnings),
                        )
                cleaned = self.db.cleanup_history(
                    memory_id=memory_id, older_than_days=older_than_days, conn=conn,
                )
        except sqlite3.Error as exc:
            return self.db.state.response(
                {"error": f"history cleanup failed; transaction rolled back: {exc}", "cleaned": 0},
                ok=False, extra_warnings=list(caller.warnings),
            )
        scope = "full" if full_cleanup else ("memory" if memory_id is not None else "by_age")
        data = {
            "cleaned": cleaned,
            "scope": scope,
            "memory_id": memory_id,
            "older_than_days": older_than_days,
        }
        if caller.isolation == "strict":
            data.update(caller.response_fields())
        return self.db.state.response(data, extra_warnings=list(caller.warnings))

    def _postprocess_replayed_memory(
        self,
        replay_key: str,
        memory_id: int,
        prior_stages: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, str], str | None, list[str]]:
        stages = {
            str(key): str(value)
            for key, value in (prior_stages or {}).items()
            if isinstance(key, str)
        }
        record = self.db.get_memory(memory_id) or {}
        result, _check = self._post_commit(memory_id, record, recheck_conflicts=False)
        # Replayed active rows carry no subject_tags_vec vector (replay
        # restores base rows, not derived indexes); publish one so the
        # write-time duplicate hint can recall them before the next restart.
        self._tools._write_pipeline.refresh_subject_tags_vector(memory_id)
        # Same for the C3a summary vector (anomaly voting index).
        self._tools._write_pipeline.refresh_summary_vector(memory_id)
        outcome = str(result.get("status") or "unknown")
        if outcome in {"queued", "completed"}:
            # C2: the semantic queue dedupes by task_id — "completed" means
            # this version's job (index+detect) already ran, so the receipt's
            # text index is persisted and the stage is complete all the same.
            stages["evidence"] = outcome
            status, retry_code = "complete", None
        elif outcome == "skipped" and not self._embedding_configured():
            stages["evidence"] = "skipped"
            status, retry_code = "complete", None
        else:
            stages["evidence"] = "retry_pending"
            status, retry_code = "pending", f"evidence_{outcome}"
        self.db.backup_replay.set_postprocess_state(
            replay_key, status, stages, retry_code,
        )
        return status, stages, retry_code, []

    def memory_replay_backup(
        self,
        dry_run: bool = True,
        authorized: bool = False,
        limit: int = 1_000,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        """Inspect or deterministically replay backup-only memory records."""
        dry_run = self._is_truthy(dry_run)
        authorized = self._is_truthy(authorized)
        requested_limit = max(1, min(int(limit), 10_000))
        page_limit = requested_limit if dry_run else min(requested_limit, 200)
        inspection = self.db.backup_replay.inspect(
            limit=page_limit, offset=max(0, int(offset)),
        )
        public = {key: value for key, value in inspection.items() if key != "entries"}
        if dry_run:
            return self.db.state.response({"dry_run": True, **public})
        if not authorized:
            return self.db.state.response(
                {
                    "error": "authorized=True is required to replay backup records",
                    "action_required": "ask_user_for_authorization",
                    "dry_run": False,
                    **public,
                },
                ok=False,
            )
        imported: list[dict[str, Any]] = []
        already_replayed: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        warnings: list[str] = []
        for entry in inspection["entries"]:
            if entry["status"] == "already_replayed" and entry.get("postprocess_status") in {"complete", "warning"}:
                already_replayed.append({
                    "replay_key": entry["replay_key"],
                    "memory_id": entry["memory_id"],
                    "postprocess_status": entry.get("postprocess_status"),
                    "postprocess_stages": entry.get("postprocess_stages") or {},
                    "postprocess_error_code": entry.get("postprocess_error_code"),
                })
                continue
            if entry["status"] not in {"importable", "already_replayed"}:
                conflicts.append({"replay_key": entry["replay_key"], "outcome": entry["status"]})
                continue
            try:
                replayed = self.db.backup_replay.replay_one(entry)
            except Exception as exc:
                conflicts.append({"replay_key": entry["replay_key"], "outcome": "error", "reason": str(exc)})
                continue
            outcome = replayed.get("outcome")
            needs_postprocess = outcome == "imported" or (
                outcome == "already_replayed"
                and replayed.get("postprocess_status") not in {"complete", "warning"}
            )
            if needs_postprocess:
                memory_id = int(replayed["memory_id"])
                if outcome == "imported":
                    receipt_result = {"replay_key": entry["replay_key"], "memory_id": memory_id, "postprocess_status": "pending"}
                    imported.append(receipt_result)
                else:
                    receipt_result = {"replay_key": entry["replay_key"], "memory_id": memory_id, "postprocess_status": replayed.get("postprocess_status")}
                    already_replayed.append(receipt_result)
                try:
                    final_status, stages, error_code, postprocess_warnings = (
                        self._postprocess_replayed_memory(
                            entry["replay_key"], memory_id,
                            replayed.get("postprocess_stages"),
                        )
                    )
                    receipt_result["postprocess_status"] = final_status
                    receipt_result["postprocess_stages"] = stages
                    if error_code is not None:
                        receipt_result["postprocess_error_code"] = error_code
                    warnings.extend(postprocess_warnings)
                except Exception as exc:
                    # Earlier stage checkpoints may already be durable. Keep
                    # them and only mark the receipt retryable here.
                    self.db.backup_replay.mark_postprocess_retry(
                        entry["replay_key"], "postprocess_exception",
                    )
                    receipt_result["postprocess_status"] = "pending"
                    receipt_result["postprocess_stages"] = replayed.get("postprocess_stages") or {}
                    receipt_result["postprocess_error_code"] = "postprocess_exception"
                    warnings.append(f"replayed memory {memory_id} committed; derived post-processing failed and will retry: {exc}")
            elif outcome in {"already_replayed", "duplicate_content"}:
                # duplicate_content (0.16.6 dedup gate) is the same idempotent
                # family: the line imported nothing because the content
                # already lives in an ACTIVE row — not a replay conflict.
                already_replayed.append({"replay_key": entry["replay_key"], "memory_id": replayed.get("memory_id"), "postprocess_status": "complete", **({"outcome": outcome} if outcome != "already_replayed" else {})})
            else:
                conflicts.append({"replay_key": entry["replay_key"], "outcome": outcome})
        # Drain the semantic worker before responding (B-D2, rewired C2):
        # replay post-processing now enqueues index_only jobs on the semantic
        # queue (the index worker no longer serves the write path), so the
        # "complete receipt => text index persisted" guarantee drains the
        # semantic queue instead. The wait may cover unrelated detect jobs
        # (Qwen-bearing, up to seconds each) — the merge's known cost for
        # replay batches, recorded here.
        drained = self.wait_semantic_worker_drained(timeout=30.0)
        if not drained:
            warnings.append(
                "semantic worker did not drain within 30s; complete receipts may still have text indexing in flight"
            )
        return self.db.state.response(
            {
                "dry_run": False,
                "semantic_worker_drained": drained,
                "imported": imported,
                "imported_count": len(imported),
                "already_replayed": already_replayed,
                "already_replayed_count": len(already_replayed),
                "invalid_entries": inspection["invalid_entries"],
                "conflicts": conflicts,
                "offset": inspection["offset"],
                "next_offset": inspection["next_offset"],
                "has_more": inspection["has_more"],
                "processed": len(inspection["entries"]) + len(inspection["invalid_entries"]),
                "remaining": inspection["has_more"],
            },
            ok=not conflicts,
            extra_warnings=warnings,
        )
