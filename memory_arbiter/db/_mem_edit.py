"""Memory row CRUD, filters, edit/history operations for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import hashlib
import json
import sqlite3

from typing import Any, Callable, TYPE_CHECKING

from ..config import Settings
from ..degrade import DegradeState

from ..constants import MAX_MEMORY_TOTAL_TAGS
from ..models import utc_now_iso
from ._mem_helpers import DuplicateActiveContentError, _row_to_dict, content_sha, _strip_retired_metadata_keys

if TYPE_CHECKING:
    from .core import MemoryDB
class _MemEditMixin:
    """memories 编辑与历史（从 memories.py 搬出，拆分批 ②b 纯移动）。"""

    # 拆分批 ②b：声明式注解（mypy strict；形态对齐主类）
    if TYPE_CHECKING:
        _db: "MemoryDB"
        from typing import Any as _Any
        @property
        def _db_available(self) -> bool: ...
        @property
        def settings(self) -> "Settings": ...
        @property
        def state(self) -> "DegradeState": ...
        def connection(self) -> "_Any": ...
        def write_transaction(self) -> "_Any": ...
        _fetch_memory: "Callable[..., Any]"
        active_content_twin_on_conn: "Callable[..., Any]"
    def update_memory_on_conn(self, conn: sqlite3.Connection, memory_id: int, updates: dict[str, Any]) -> bool:
        """Update allowed memory fields using caller-owned transaction.

        External-transaction mode intentionally does not catch sqlite3.Error:
        callers need an exception to roll back the whole unit of work.
        """
        allowed = {"source_type", "confidence", "protection_level", "status", "metadata"}
        # Gate-v2 G3: retired keys leave the metadata BEFORE the change
        # probe — an update that only moves entity/scope is a no_change,
        # never a spurious version bump or rewrite.
        pairs = [
            (key, _strip_retired_metadata_keys(value) if key == "metadata" and isinstance(value, dict) else value)
            for key, value in (item for item in updates.items() if item[0] in allowed)
        ]
        if not pairs:
            return True
        current = self._fetch_memory(conn, int(memory_id))
        if not current:
            return False
        changed = any(current.get(key) != value for key, value in pairs)
        if not changed:
            return True
        status_changed = any(
            key == "status" and current.get(key) != value
            for key, value in pairs
        )
        new_status = next((value for key, value in pairs if key == "status"), None)
        snapshot_semantics_changed = any(
            key in {"source_type", "confidence", "protection_level", "status"}
            and current.get(key) != value
            for key, value in pairs
        )
        # Gate-v2 G3: the old "entity/scope changed → snapshot semantics
        # changed → version bump" leg is gone — the keys are retired, they
        # never reach storage (stripped below), so they can no longer drive
        # snapshot semantics.
        if status_changed and str(new_status) == "active":
            # 0.16.6 dedup gate (owner: 只管活的): the pending row itself sat
            # outside the partial index, so the flip INTO active is the one
            # moment a same-content twin can collide. Check before the UPDATE
            # so the caller's rollback contract sees a named cause instead
            # of a raw constraint failure.
            twin = self.active_content_twin_on_conn(
                conn,
                current.get("workspace_canonical") or current.get("workspace"),
                current.get("content_sha"),
                exclude_id=int(memory_id),
            )
            if twin is not None:
                raise DuplicateActiveContentError(
                    f"duplicate_active_content: active memory #{int(twin['id'])} already holds "
                    "byte-identical content in this workspace; govern it (merge/retire) "
                    "before activating this one"
                )
        sql = ", ".join(f"{key} = ?" for key, _ in pairs)
        if snapshot_semantics_changed:
            sql += ", version = version + 1"
        values = [
            json.dumps(
                _strip_retired_metadata_keys(v) if key == "metadata" and isinstance(v, dict) else v,
                ensure_ascii=False,
            ) if isinstance(v, (dict, list)) else v
            for key, v in pairs
        ]
        values.append(int(memory_id))
        conn.execute(f"UPDATE memories SET {sql} WHERE id = ?", values)
        if snapshot_semantics_changed:
            # These updates do not change evidence text. Keep the derived rows
            # pinned to the new authoritative memory version in the same
            # transaction instead of making doctor report false staleness.
            # 0.17.0 C5: the unit version-pin leg's ROW counterpart — a
            # status flip does NOT republish rows (only content edits do),
            # so pin memory_row the same way or doctor reports false
            # staleness and version checks drift.
            conn.execute(
                "UPDATE memory_row SET memory_version=memory_version+1 "
                "WHERE memory_id=?",
                (int(memory_id),),
            )
        if status_changed and self.state.sqlite_vec_available:
            try:
                # (0.17.0 C5: the memory_evidence_vec parent_status flip
                # retired with the unit tables.)
                # 0.17.0 P2-2.2: the row-level conflict store mirrors the
                # evidence lifecycle exactly (same parent_status flip).
                try:
                    conn.execute(
                        "UPDATE memory_row_vec SET parent_status=? WHERE id IN "
                        "(SELECT id FROM memory_row WHERE memory_id=?)",
                        (str(new_status or "deleted"), int(memory_id)),
                    )
                except sqlite3.OperationalError as exc:
                    if "no such table" not in str(exc):
                        raise
                if str(new_status or "deleted") != "active":
                    # The duplicate-hint recall index tracks the ACTIVE set;
                    # leaving a stale row would only waste KNN window slots
                    # (the join filters it anyway, but the domain should not
                    # drift). Re-activation paths re-publish on activation.
                    conn.execute(
                        "DELETE FROM subject_tags_vec WHERE id = ?", (int(memory_id),)
                    )
                    # 0.17.0 (adversarial review P2-8): the summary vec feeds
                    # the write-time duplicate-hint recall (P2-7) — a retired
                    # row must stop being a candidate/voter immediately, not
                    # at the next boot backfill purge.
                    conn.execute(
                        "DELETE FROM memory_summary_vec WHERE id = ?", (int(memory_id),),
                    )
            except sqlite3.Error:
                # Governance must remain available while the derived index is
                # temporarily unavailable; rebuild_evidence repairs it later.
                pass
        return True

    def update_memory(
        self,
        memory_id: int,
        updates: dict[str, Any],
        *,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        if conn is not None:
            return self.update_memory_on_conn(conn, memory_id, updates)
        if not self._db_available or not self.state.sqlite_writable:
            return False
        try:
            with self.write_transaction() as txn_conn:
                return self.update_memory_on_conn(txn_conn, memory_id, updates)
        except sqlite3.Error:
            return False

    def update_tags_low_side_effect(
        self,
        memory_id: int,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        authorized: bool = False,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        """v0.7.6: low-side-effect tag-only update.

        Unlike ``edit_memory``, this does NOT write ``memory_history``,
        does NOT bump ``version`` and does not rebuild evidence vectors.
        It only updates the ``tags`` column and re-syncs FTS (tags are
        indexed in FTS5).

        Uses ``write_transaction()`` (BEGIN IMMEDIATE) so the re-read +
        protection check + writes share the write lock (TOCTOU-safe).

        Returns an outcome dict:
          ``updated``       — tags changed, FTS re-synced.
          ``no_change``     — add/remove yielded no difference; zero writes.
          ``not_found``     — memory_id absent.
          ``not_active``    — superseded/deleted.
          ``forbidden``     — protected and not authorized.
          ``unavailable``   — DB not writable.
          ``error``         — sqlite3.Error mid-transaction; fully rolled back.
        """
        if conn is not None:
            return self.update_tags_low_side_effect_on_conn(
                conn, memory_id, add_tags=add_tags,
                remove_tags=remove_tags, authorized=authorized,
            )
        if not self._db_available or not self.state.sqlite_writable:
            return {"outcome": "unavailable", "memory_id": memory_id}
        try:
            with self.write_transaction() as conn:
                return self.update_tags_low_side_effect_on_conn(
                    conn, memory_id, add_tags=add_tags,
                    remove_tags=remove_tags, authorized=authorized,
                )
        except sqlite3.Error:
            return {"outcome": "error", "memory_id": memory_id}

    def update_tags_low_side_effect_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """Update tags using a caller-owned write transaction."""
        current = self._fetch_memory(conn, memory_id)
        if not current:
            return {"outcome": "not_found", "memory_id": memory_id}
        status = current.get("status")
        if status != "active":
            return {"outcome": "not_active", "memory_id": memory_id, "status": status}
        raw_tags = current.get("tags")
        if isinstance(raw_tags, list):
            old_tags = raw_tags
        elif isinstance(raw_tags, str):
            try:
                parsed = json.loads(raw_tags)
                old_tags = parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                old_tags = []
        else:
            old_tags = []
        protection = current.get("protection_level")
        source_type = current.get("source_type")
        is_protected = protection == "locked" or source_type == "user_confirmed"
        if is_protected and not authorized:
            return {
                "outcome": "forbidden", "memory_id": memory_id,
                "protection_level": protection, "source_type": source_type,
            }
        current_set: set[str] = set(old_tags)
        new_tags_list = list(old_tags)
        for tag in remove_tags or []:
            if tag in current_set:
                current_set.discard(tag)
                new_tags_list = [item for item in new_tags_list if item != tag]
        for tag in add_tags or []:
            if tag not in current_set:
                current_set.add(tag)
                new_tags_list.append(tag)
        if new_tags_list == old_tags:
            return {"outcome": "no_change", "memory_id": memory_id, "tags": old_tags}
        # 0.16.0 §6⑮: persisted tag total cap (remove+add in one call is
        # legal; the cap applies to the merged result). Net shrinks always
        # pass so over-cap stock stays trimmable.
        if len(new_tags_list) > len(old_tags) and len(new_tags_list) > MAX_MEMORY_TOTAL_TAGS:
            return {
                "outcome": "tags_over_limit",
                "memory_id": memory_id,
                "error": (
                    f"tags would total {len(new_tags_list)} (cap {MAX_MEMORY_TOTAL_TAGS}); "
                    "remove_tags first — tags are a retrieval dimension, not an event log "
                    "(one-off state belongs in metadata)"
                ),
                "current_total": len(new_tags_list),
                "cap": MAX_MEMORY_TOTAL_TAGS,
            }
        conn.execute(
            "UPDATE memories SET tags=? WHERE id=?",
            (json.dumps(new_tags_list, ensure_ascii=False), memory_id),
        )
        # Tags changed without a version bump — the linked-df fingerprint
        # (COUNT+SUM(version)) cannot see this; drop the cache explicitly.
        self._db.invalidate_linked_df_cache()
        if self.state.fts5_available:
            old_content = current["content"]
            old_subject = current.get("subject")
            conn.execute(
                "INSERT INTO memories_fts(memories_fts, rowid, content, tags, subject) "
                "VALUES('delete', ?, ?, ?, ?)",
                (memory_id, old_content, " ".join(old_tags), old_subject or ""),
            )
            conn.execute(
                "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                (memory_id, old_content, " ".join(new_tags_list), old_subject or ""),
            )
        return {
            "outcome": "updated", "memory_id": memory_id,
            "tags": new_tags_list, "semantic_content_changed": False,
        }

    def update_metadata_fields_low_side_effect(
        self,
        memory_id: int,
        set_fields: dict[str, Any] | None = None,
        clear_fields: list[str] | None = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if not self._db_available or not self.state.sqlite_writable:
            return {"outcome": "unavailable", "memory_id": memory_id}
        with self.write_transaction() as conn:
            return self.update_metadata_fields_low_side_effect_on_conn(
                conn, memory_id, set_fields=set_fields,
                clear_fields=clear_fields, authorized=authorized,
            )

    def update_metadata_fields_low_side_effect_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        set_fields: dict[str, Any] | None = None,
        clear_fields: list[str] | None = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """Update metadata using a caller-owned write transaction."""
        current = self._fetch_memory(conn, int(memory_id))
        if not current:
            return {"outcome": "not_found", "memory_id": memory_id}
        if current.get("status") != "active":
            return {"outcome": "not_active", "memory_id": memory_id}
        if (current.get("protection_level") == "locked" or current.get("source_type") == "user_confirmed") and not authorized:
            return {"outcome": "forbidden", "memory_id": memory_id}
        metadata = dict(current.get("metadata") or {})
        before = dict(metadata)
        for key in clear_fields or []:
            metadata.pop(str(key), None)
        metadata.update(set_fields or {})
        # Gate-v2 G3: strip BEFORE the change probe — set_entity/clear on the
        # retired keys lands as no_change instead of an empty rewrite that
        # would bump the version forever (third serialization point; the
        # owner's "any path" ruling covers the repair tool too).
        metadata = _strip_retired_metadata_keys(metadata)
        before = _strip_retired_metadata_keys(before)
        if metadata == before:
            return {"outcome": "no_change", "memory_id": memory_id, "metadata": metadata}
        conn.execute(
            "UPDATE memories SET metadata=?,version=version+1 WHERE id=?",
            (json.dumps(metadata, ensure_ascii=False), int(memory_id)),
        )
        return {"outcome": "updated", "memory_id": memory_id, "metadata": metadata}

    def edit_memory_intent_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        *,
        new_content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        patches: list[dict[str, Any]] | None = None,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        reason: str | None = None,
        authorized: bool = False,
        expected_version: int | None = None,
        expected_content_hash: str | None = None,
        require_active: bool = True,
    ) -> dict[str, Any]:
        """Apply a full/partial edit intent inside caller-owned transaction.

        This helper re-reads the row after BEGIN IMMEDIATE, re-checks protection
        and status, validates optional CAS pins, computes partial content and tag
        overlays from the current row, then writes history + memory update. It
        deliberately does not catch sqlite3.Error so outer transactions roll back.
        """
        current = self._fetch_memory(conn, int(memory_id))
        if not current:
            return {"outcome": "not_found", "memory_id": int(memory_id)}
        status = current.get("status")
        if require_active and status != "active":
            return {"outcome": "not_active", "memory_id": int(memory_id), "status": status}
        protection = current.get("protection_level")
        source_type = current.get("source_type")
        is_protected = protection == "locked" or source_type == "user_confirmed"
        if is_protected and not authorized:
            return {
                "outcome": "forbidden",
                "memory_id": int(memory_id),
                "protection_level": protection,
                "source_type": source_type,
            }
        old_version = int(current.get("version") or 1)
        if expected_version is not None and old_version != int(expected_version):
            return {
                "outcome": "stale_edit",
                "memory_id": int(memory_id),
                "reason": "version_mismatch",
                "current_version": old_version,
                "expected_version": int(expected_version),
            }
        old_content = current.get("content") or ""
        current_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()
        if expected_content_hash is not None and current_hash != str(expected_content_hash):
            return {
                "outcome": "stale_edit",
                "memory_id": int(memory_id),
                "reason": "content_hash_mismatch",
                "current_version": old_version,
            }
        # CAS pins are validated against the pre-edit row; content application
        # happens after, in every mode.
        content_modes = sum(
            1 for flag in (
                new_content is not None,
                (old_text is not None or new_text is not None),
                patches is not None,
            ) if flag
        )
        if content_modes > 1:
            return {"outcome": "invalid", "memory_id": int(memory_id), "error": "pass exactly one content mode: new_content (full replace), old_text+new_text (single partial), or patches (sequential partial), not a combination"}
        if new_content is not None:
            if not str(new_content).strip():
                return {"outcome": "invalid", "memory_id": int(memory_id), "error": "new_content is empty; refusing to wipe memory content (use memory_supersede to retire it, or pass real content)"}
            resolved_content = str(new_content)
        elif old_text is not None and new_text is not None:
            if str(old_text) not in old_content:
                return {"outcome": "stale_edit", "memory_id": int(memory_id), "reason": "old_text_not_found", "error": "old_text not found in current content"}
            resolved_content = old_content.replace(str(old_text), str(new_text), 1)
        elif patches is not None:
            # Defensive shape re-check (the validation boundary already
            # normalized the list): a direct pipeline caller bypassing the
            # product surface must not reach the replace loop with a
            # malformed batch.
            if (
                not isinstance(patches, list) or not 1 <= len(patches) <= 8
                or any(
                    not isinstance(patch, dict) or set(patch) != {"old_text", "new_text"}
                    or not isinstance(patch.get("old_text"), str) or not patch["old_text"]
                    or not isinstance(patch.get("new_text"), str)
                    for patch in patches
                )
            ):
                return {"outcome": "invalid", "memory_id": int(memory_id), "error": "patches must be a list of 1..8 objects with exactly old_text (non-empty string) and new_text (string)"}
            resolved_content = old_content
            for patch_index, patch in enumerate(patches):
                old_piece = str(patch["old_text"])
                if old_piece not in resolved_content:
                    # Atomic all-or-nothing: any miss rejects the whole call
                    # with zero side effects (this helper has written nothing
                    # yet; the caller's transaction stays pristine).
                    return {
                        "outcome": "stale_edit", "memory_id": int(memory_id),
                        "reason": "old_text_not_found", "patch_index": patch_index,
                        "error": f"patches[{patch_index}].old_text not found in current content (after applying patches 0..{patch_index - 1})",
                    }
                resolved_content = resolved_content.replace(old_piece, str(patch["new_text"]), 1)
            if not resolved_content.strip():
                # Same guard as the new_content path: deletion-only patch
                # batches must not silently wipe the memory.
                return {"outcome": "invalid", "memory_id": int(memory_id), "error": "patches would empty the content; refusing to wipe memory content (use memory_supersede to retire it)"}
        else:
            return {"outcome": "invalid", "memory_id": int(memory_id), "error": "provide new_content for full replace, or old_text+new_text for partial replace, or patches for sequential partial replace, or tags_only=true"}
        if new_subject is not None and not str(new_subject).strip():
            return {"outcome": "invalid", "memory_id": int(memory_id), "error": "new_subject is empty; refusing to wipe subject (pass None to keep current)"}
        old_subject = current.get("subject")
        subject_value = new_subject if new_subject is not None else old_subject
        old_tags = current.get("tags") or []
        if isinstance(old_tags, str):
            try:
                parsed_tags = json.loads(old_tags)
                old_tags = parsed_tags if isinstance(parsed_tags, list) else []
            except (json.JSONDecodeError, ValueError):
                old_tags = []
        resolved_tags: list[str]
        if new_tags is not None:
            resolved_tags = list(new_tags)
        else:
            resolved_tags = list(old_tags)
        tag_set = set(resolved_tags)
        for tag in (remove_tags or []):
            if tag in tag_set:
                tag_set.discard(tag)
                resolved_tags = [existing for existing in resolved_tags if existing != tag]
        for tag in (add_tags or []):
            if tag not in tag_set:
                tag_set.add(tag)
                resolved_tags.append(tag)
        # 0.16.0 §6⑮: the PERSISTED tag total is capped — tags are a
        # retrieval dimension, not an event log. remove-then-add in one call
        # is legal (the cap applies to the merged result); over-limit rejects
        # the WHOLE edit with the current count so the agent can remove first.
        # Net SHRINKS are always allowed: pre-cap stock rows (doctor
        # tags.over_limit) must stay trimmable, not frozen.
        if len(resolved_tags) > len(old_tags) and len(resolved_tags) > MAX_MEMORY_TOTAL_TAGS:
            return {
                "outcome": "tags_over_limit",
                "memory_id": int(memory_id),
                "error": (
                    f"tags would total {len(resolved_tags)} (cap {MAX_MEMORY_TOTAL_TAGS}); "
                    "remove_tags first — tags are a retrieval dimension, not an event log "
                    "(one-off state belongs in metadata)"
                ),
                "current_total": len(resolved_tags),
                "cap": MAX_MEMORY_TOTAL_TAGS,
            }
        # 0.16.6 dedup gate: editing content INTO another active row's exact
        # bytes is a duplicate by definition — refuse before any write so the
        # caller's transaction never sees a partial history archive.
        new_sha = content_sha(resolved_content)
        edit_twin = self.active_content_twin_on_conn(
            conn,
            current.get("workspace_canonical") or current.get("workspace"),
            new_sha,
            exclude_id=int(memory_id),
        )
        if edit_twin is not None:
            return {
                "outcome": "duplicate_content",
                "memory_id": int(memory_id),
                "existing_memory_id": int(edit_twin["id"]),
                "existing_subject": edit_twin["subject"],
                "error": (
                    f"new content is byte-identical to active memory #{int(edit_twin['id'])}; "
                    "merge or retire one of the two instead of keeping duplicates"
                ),
            }
        history_cur = conn.execute(
            """
            INSERT INTO memory_history
            (memory_id, content_snapshot, subject_snapshot, tags_snapshot, version, changed_at, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(memory_id),
                old_content,
                old_subject,
                json.dumps(old_tags, ensure_ascii=False),
                old_version,
                utc_now_iso(),
                reason,
            ),
        )
        if history_cur.lastrowid is None:
            raise sqlite3.Error("memory_history insert did not return an id")
        history_id = int(history_cur.lastrowid)
        conn.execute(
            "UPDATE memories SET content=?, subject=?, tags=?, version=?, content_sha=? WHERE id=?",
            (
                resolved_content,
                subject_value,
                json.dumps(resolved_tags, ensure_ascii=False),
                old_version + 1,
                new_sha,
                int(memory_id),
            ),
        )
        # 0.17.0 C5: the unit delete leg is retired; the ROW delete leg
        # replaces it (subquery form — no id-list round-trip). Discovered in
        # review: rows never HAD a delete leg here (P2-2.3 gap) — deleted
        # memories would have leaked live row vectors into KNN windows.
        if self.state.sqlite_vec_available:
            try:
                conn.execute(
                    "DELETE FROM memory_row_vec WHERE id IN "
                    "(SELECT id FROM memory_row WHERE memory_id=?)",
                    (int(memory_id),),
                )
            except sqlite3.OperationalError as exc:
                # Lazy vec tables (embedder never built): nothing was ever
                # indexed, so there is nothing to cascade — the edit itself
                # must not fail over the derived store's absence.
                if "no such table" not in str(exc):
                    raise
        conn.execute(
            "DELETE FROM memory_row WHERE memory_id=?", (int(memory_id),)
        )
        if self.state.fts5_available:
            conn.execute(
                "INSERT INTO memories_fts(memories_fts, rowid, content, tags, subject) VALUES('delete', ?, ?, ?, ?)",
                (int(memory_id), old_content, " ".join(old_tags), old_subject or ""),
            )
            conn.execute(
                "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                (int(memory_id), resolved_content, " ".join(resolved_tags), subject_value or ""),
            )
        updated = self._fetch_memory(conn, int(memory_id))
        return {
            "outcome": "edited",
            "memory_id": int(memory_id),
            "history_id": history_id,
            "new_version": old_version + 1,
            "record": updated,
            "semantic_content_changed": True,
        }

    def edit_memory_intent(
        self,
        memory_id: int,
        *,
        new_content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        patches: list[dict[str, Any]] | None = None,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        reason: str | None = None,
        authorized: bool = False,
        expected_version: int | None = None,
        expected_content_hash: str | None = None,
        conn: sqlite3.Connection | None = None,
        require_active: bool = True,
    ) -> dict[str, Any]:
        if conn is not None:
            return self.edit_memory_intent_on_conn(
                conn,
                memory_id,
                new_content=new_content,
                old_text=old_text,
                new_text=new_text,
                patches=patches,
                new_subject=new_subject,
                new_tags=new_tags,
                add_tags=add_tags,
                remove_tags=remove_tags,
                reason=reason,
                authorized=authorized,
                expected_version=expected_version,
                expected_content_hash=expected_content_hash,
                require_active=require_active,
            )
        if not self._db_available or not self.state.sqlite_writable:
            return {"outcome": "unavailable", "memory_id": int(memory_id)}
        try:
            with self.write_transaction() as txn_conn:
                return self.edit_memory_intent_on_conn(
                    txn_conn,
                    memory_id,
                    new_content=new_content,
                    old_text=old_text,
                    new_text=new_text,
                    patches=patches,
                    new_subject=new_subject,
                    new_tags=new_tags,
                    add_tags=add_tags,
                    remove_tags=remove_tags,
                    reason=reason,
                    authorized=authorized,
                    expected_version=expected_version,
                    expected_content_hash=expected_content_hash,
                    require_active=require_active,
                )
        except sqlite3.Error:
            return {"outcome": "error", "memory_id": int(memory_id)}

    def edit_memory(
        self,
        memory_id: int,
        new_content: str,
        new_subject: str | None = None,
        new_tags: list[str] | None = None,
        reason: str | None = None,
        *,
        conn: sqlite3.Connection | None = None,
        authorized: bool = True,
    ) -> int | None:
        """In-place edit a memory's content, archiving the prior version."""
        result = self.edit_memory_intent(
            memory_id,
            new_content=new_content,
            new_subject=new_subject,
            new_tags=new_tags,
            reason=reason,
            authorized=authorized,
            conn=conn,
            require_active=False,
        )
        if result.get("outcome") == "edited":
            return int(result["history_id"])
        if result.get("outcome") == "invalid" and result.get("error", "").startswith("new_subject is empty"):
            raise ValueError("new_subject must be non-empty when provided")
        return None

    def list_history(self, memory_id: int) -> list[dict[str, Any]]:
        if not self._db_available:
            return []
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_history WHERE memory_id=? ORDER BY version DESC, id DESC",
                (memory_id,),
            ).fetchall()
            return [_row_to_dict(row) for row in rows]

    def cleanup_history(
        self, memory_id: int | None = None, older_than_days: int | None = None,
        *, conn: sqlite3.Connection | None = None,
    ) -> int:
        """Delete historical snapshots from memory_history.

        SAFETY RED LINE: only ever issues DELETE against memory_history.
        """
        if not self._db_available or not self.state.sqlite_writable:
            return 0
        clauses: list[str] = []
        params: list[Any] = []
        if memory_id is not None:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        if older_than_days is not None:
            from datetime import datetime, timedelta, timezone

            cutoff = (datetime.now(timezone.utc) - timedelta(days=int(older_than_days))).replace(microsecond=0).isoformat()
            clauses.append("changed_at < ?")
            params.append(cutoff)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        if conn is not None:
            cur = conn.execute(f"DELETE FROM memory_history {where}", params)
            return int(cur.rowcount)
        with self.write_transaction() as txn_conn:
            cur = txn_conn.execute(f"DELETE FROM memory_history {where}", params)
            return int(cur.rowcount)

    # ------------------------------------------------------------------
    #  Audit
    # ------------------------------------------------------------------
