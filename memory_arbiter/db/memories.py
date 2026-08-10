"""Memory row CRUD, filters, edit/history operations for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple, TYPE_CHECKING

from ..models import MemoryRecord, utc_now_iso
from ..text import (
    canon_entity as _canon_entity,
    canon_scope as _canon_scope,
    coerce_tags as _coerce_tags_db,
    subject_tokens as _subject_tokens,
)

if TYPE_CHECKING:
    from .core import MemoryDB


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("tags", "metadata", "structured_details"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                pass
    return data


class MemoriesStore:
    def __init__(self, db: "MemoryDB"):
        self._db = db

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)

    def insert_memory(
        self, record: MemoryRecord, workspace_canonical: Optional[str] = None
    ) -> Tuple[Optional[int], list[str]]:
        warnings: list[str] = []
        if not record.content:
            raise ValueError("content is required")
        if not record.subject or not str(record.subject).strip():
            raise ValueError("subject is required")
        if not self._db_available or not self.state.sqlite_writable:
            self._append_backup(record)
            warnings.append("SQLite write unavailable; wrote append-only JSONL backup.")
            return None, warnings
        # Double-store: raw workspace stays in `workspace`; resolved canonical
        # (from tools-side alias resolution) lands in `workspace_canonical`.
        # Fall back to the raw workspace so the column is never NULL on new rows.
        canonical = (workspace_canonical or record.workspace or "").strip() or record.workspace
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO memories
                (content, agent_id, workspace, workspace_canonical, tags, source_type, source_ref,
                 event_time, ingest_time, confidence, protection_level, status, subject, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.content,
                    record.agent_id,
                    record.workspace,
                    canonical,
                    json.dumps(record.tags, ensure_ascii=False),
                    record.source_type,
                    record.source_ref,
                    record.event_time,
                    record.ingest_time,
                    record.confidence,
                    record.protection_level,
                    record.status,
                    record.subject,
                    json.dumps(record.metadata, ensure_ascii=False),
                    utc_now_iso(),
                ),
            )
            memory_id = int(cur.lastrowid)
            if self.state.fts5_available:
                conn.execute(
                    "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                    (memory_id, record.content, " ".join(record.tags), record.subject or ""),
                )
            conn.commit()
        return memory_id, warnings

    def _append_backup(self, record: MemoryRecord) -> None:
        self.settings.backup_jsonl.parent.mkdir(parents=True, exist_ok=True)
        payload = record.__dict__.copy()
        payload["backup_written_at"] = utc_now_iso()
        with self.settings.backup_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.state.jsonl_backup_active = True

    @staticmethod
    def _fetch_memory(conn: sqlite3.Connection, memory_id: int) -> Optional[dict[str, Any]]:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def get_memory_on_conn(self, conn: sqlite3.Connection, memory_id: int) -> Optional[dict[str, Any]]:
        """Fetch a memory using the caller's transaction/connection."""
        return self._fetch_memory(conn, int(memory_id))

    def get_memory(self, memory_id: int, *, conn: Optional[sqlite3.Connection] = None) -> Optional[dict[str, Any]]:
        if conn is not None:
            return self.get_memory_on_conn(conn, memory_id)
        if not self._db_available:
            return None
        with self.connection() as conn:
            return self._fetch_memory(conn, memory_id)

    def get_memory_for_workspace(self, memory_id: int, ws_canonical: str) -> Optional[dict[str, Any]]:
        """ACL-specific read-by-id helper; does not change get_memory semantics."""
        if not self._db_available or not str(ws_canonical or "").strip():
            return None
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ? "
                "AND COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?",
                (int(memory_id), str(ws_canonical)),
            ).fetchone()
            return _row_to_dict(row) if row else None

    def list_memories_for_workspace(self, ws_canonical: str, limit: int = 50) -> list[dict[str, Any]]:
        """ACL-specific recent/list helper scoped to one canonical workspace."""
        if not self._db_available or not str(ws_canonical or "").strip():
            return []
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE status != 'deleted' "
                "AND COALESCE(NULLIF(workspace_canonical, ''), workspace) = ? "
                "ORDER BY event_time DESC, ingest_time DESC LIMIT ?",
                (str(ws_canonical), int(limit)),
            ).fetchall()
            return [_row_to_dict(row) for row in rows]

    def update_memory_on_conn(self, conn: sqlite3.Connection, memory_id: int, updates: dict[str, Any]) -> bool:
        """Update allowed memory fields using caller-owned transaction.

        External-transaction mode intentionally does not catch sqlite3.Error:
        callers need an exception to roll back the whole unit of work.
        """
        allowed = {"source_type", "confidence", "protection_level", "status", "metadata"}
        pairs = [(key, value) for key, value in updates.items() if key in allowed]
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
        # Claim judgments depend not only on extracted text but also on
        # trust/protection/status.  Changing any of those must invalidate
        # the old CAS snapshot.  Metadata only participates when its
        # canonical entity/scope changes; bookkeeping such as
        # confirmed_from does not create a second revision bump.
        judgment_semantics_changed = any(
            key in {"source_type", "confidence", "protection_level", "status"}
            and current.get(key) != value
            for key, value in pairs
        )
        metadata_update = next((value for key, value in pairs if key == "metadata"), None)
        if isinstance(metadata_update, dict):
            current_md = current.get("metadata") or {}
            current_md = current_md if isinstance(current_md, dict) else {}
            judgment_semantics_changed = judgment_semantics_changed or (
                _canon_entity(current_md.get("entity")) != _canon_entity(metadata_update.get("entity"))
                or _canon_scope(current_md.get("scope")) != _canon_scope(metadata_update.get("scope"))
            )
        sql = ", ".join(f"{key} = ?" for key, _ in pairs)
        if judgment_semantics_changed:
            sql += ", claim_revision = claim_revision + 1"
        values = [
            json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            for _, v in pairs
        ]
        values.append(int(memory_id))
        conn.execute(f"UPDATE memories SET {sql} WHERE id = ?", values)
        if status_changed and self.state.sqlite_vec_available:
            try:
                conn.execute(
                    "UPDATE memories_vec SET parent_status = ? WHERE id = ?",
                    (str(new_status or "deleted"), int(memory_id)),
                )
                conn.execute(
                    "UPDATE memory_sections_vec SET parent_status = ? WHERE id IN "
                    "(SELECT id FROM memory_sections WHERE memory_id = ?)",
                    (str(new_status or "deleted"), int(memory_id)),
                )
            except sqlite3.Error:
                pass
        return True

    def update_memory(
        self,
        memory_id: int,
        updates: dict[str, Any],
        *,
        conn: Optional[sqlite3.Connection] = None,
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

    def list_memories(self, workspace: Optional[str] = None, subject: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if not self._db_available:
            return []
        clauses = ["status != 'deleted'"]
        params: list[Any] = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        params.append(limit)
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY event_time DESC, ingest_time DESC LIMIT ?",
                params,
            ).fetchall()
            return [_row_to_dict(row) for row in rows]

    @staticmethod
    def _filter_clauses(
        like_status_clause: str,
        tags_filter: Optional[list[str]],
        after_dt: Optional[datetime],
        before_dt: Optional[datetime],
        source_type: Optional[str],
    ) -> Tuple[list[str], list[Any]]:
        """WHERE clause + params shared by count_filtered_memories and recall_by_filters.

        Mirrors search._passes_filters: like_status_clause + per-tag json_each exact
        match (AND semantics) + ingest_time ISO-string bounds + source_type equality.
        workspace is intentionally NOT filtered (v0.7.4 cross-workspace search).
        """
        clauses: list[str] = [like_status_clause]
        params: list[Any] = []
        if tags_filter:
            for tag in tags_filter:
                clauses.append("EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = ?)")
                params.append(tag)
        if after_dt is not None:
            clauses.append("ingest_time >= ?")
            params.append(after_dt.astimezone(timezone.utc).replace(microsecond=0).isoformat())
        if before_dt is not None:
            clauses.append("ingest_time <= ?")
            params.append(before_dt.astimezone(timezone.utc).replace(microsecond=0).isoformat())
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)
        return clauses, params

    def count_filtered_memories(
        self,
        like_status_clause: str,
        tags_filter: Optional[list[str]],
        after_dt: Optional[datetime],
        before_dt: Optional[datetime],
        source_type: Optional[str],
        ws_canonical: Optional[str] = None,
    ) -> int:
        """v0.7.3: COUNT(*) under the same filters used by search's _passes_filters.

        Only called when has_filters=True. Clauses are built by _filter_clauses so the
        SQL count and the SQL recall (recall_by_filters) share one source of truth.
        Cross-workspace (v0.7.4) — workspace is not filtered, EXCEPT under strict
        isolation (v0.9.7) where ``ws_canonical`` scopes the count to one canonical
        workspace so the total matches the paginated recall.
        """
        if not self._db_available:
            return 0
        clauses, params = self._filter_clauses(like_status_clause, tags_filter, after_dt, before_dt, source_type)
        if ws_canonical:
            clauses.append("COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?")
            params.append(ws_canonical)
        sql = f"SELECT COUNT(*) AS c FROM memories WHERE {' AND '.join(clauses)}"
        with self.connection() as conn:
            try:
                row = conn.execute(sql, params).fetchone()
                return int(row["c"]) if row else 0
            except sqlite3.Error:
                return 0

    def recall_by_filters(
        self,
        like_status_clause: str,
        tags_filter: Optional[list[str]],
        after_dt: Optional[datetime],
        before_dt: Optional[datetime],
        source_type: Optional[str],
        limit: int,
        offset: int = 0,
        ws_canonical: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """G6 (v0.8.5): filter-driven recall for empty-query + filters in memory_search.

        SELECT * mirroring count_filtered_memories' WHERE (via _filter_clauses),
        ordered by ingest_time DESC, capped at ``limit`` (pool_cap). Returns
        row_to_dict rows — same shape as _wide_recall pool rows. Enables
        list-by-tag / by-source_type / by-time when query is empty.

        v0.9.4: ``offset`` adds SQL OFFSET for cursor pagination on the
        exact-count filter path (used by ``memory_search_expired``).
        v0.9.7: ``ws_canonical`` hard-scopes to one canonical workspace under
        strict isolation (filters in SQL so pagination and totals stay correct).
        """
        if not self._db_available:
            return []
        clauses, params = self._filter_clauses(like_status_clause, tags_filter, after_dt, before_dt, source_type)
        if ws_canonical:
            clauses.append("COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?")
            params.append(ws_canonical)
        sql = f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY ingest_time DESC LIMIT ? OFFSET ?"
        with self.connection() as conn:
            try:
                rows = conn.execute(sql, params + [int(limit), int(offset)]).fetchall()
            except sqlite3.Error:
                return []
        return [_row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    #  Conflicts
    # ------------------------------------------------------------------

    def update_tags_low_side_effect(
        self,
        memory_id: int,
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """v0.7.6: low-side-effect tag-only update.

        Unlike ``edit_memory``, this does NOT write ``memory_history``,
        does NOT bump ``version``, does NOT touch content/subject/sections/
        split_status/split_revision, and does NOT trigger re-embedding.
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
        if not self._db_available or not self.state.sqlite_writable:
            return {"outcome": "unavailable", "memory_id": memory_id}
        try:
            with self.write_transaction() as conn:
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

                # Protection check inside the transaction (TOCTOU-safe).
                protection = current.get("protection_level")
                source_type = current.get("source_type")
                is_protected = protection == "locked" or source_type == "user_confirmed"
                if is_protected and not authorized:
                    return {
                        "outcome": "forbidden",
                        "memory_id": memory_id,
                        "protection_level": protection,
                        "source_type": source_type,
                    }

                # Compute new tags preserving order + deduping.
                current_set: set[str] = set(old_tags)
                new_tags_list = list(old_tags)
                for t in (remove_tags or []):
                    if t in current_set:
                        current_set.discard(t)
                        new_tags_list = [x for x in new_tags_list if x != t]
                for t in (add_tags or []):
                    if t not in current_set:
                        current_set.add(t)
                        new_tags_list.append(t)

                if new_tags_list == old_tags:
                    # Zero-write no-op (covers idempotent remove of absent tag).
                    return {"outcome": "no_change", "memory_id": memory_id, "tags": old_tags}

                new_tags_json = json.dumps(new_tags_list, ensure_ascii=False)
                from ..claims import resolve_entity
                old_entity, _ = resolve_entity(current)
                new_record = dict(current)
                new_record["tags"] = new_tags_list
                new_entity, _ = resolve_entity(new_record)
                claim_semantics_changed = old_entity != new_entity
                conn.execute(
                    "UPDATE memories SET tags=?, claim_revision=claim_revision+? WHERE id=?",
                    (new_tags_json, 1 if claim_semantics_changed else 0, memory_id),
                )
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
                # write_transaction() commits on normal exit; rolls back on raise.
                return {
                    "outcome": "updated",
                    "memory_id": memory_id,
                    "tags": new_tags_list,
                    "claim_semantics_changed": claim_semantics_changed,
                }
        except sqlite3.Error:
            return {"outcome": "error", "memory_id": memory_id}

    def update_metadata_fields_low_side_effect(
        self,
        memory_id: int,
        set_fields: Optional[dict[str, Any]] = None,
        clear_fields: Optional[list[str]] = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        return self._db.claims.update_metadata_fields_low_side_effect(
            memory_id=memory_id,
            set_fields=set_fields,
            clear_fields=clear_fields,
            authorized=authorized,
        )

    def list_entities(
        self,
        limit: int = 50,
        include_unassigned: bool = True,
    ) -> dict[str, Any]:
        return self._db.claims.list_entities(limit, include_unassigned)

    def find_metadata_overlap_candidates(
        self,
        subject: Optional[str],
        tags: list[str],
        exclude_id: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """v0.7.6: recall active memories that might duplicate/evolve the
        given (subject, tags). Used by write_hints.

        Two recall channels (each capped at *limit*):
          - tag overlap: ``json_each`` match on any of *tags*.
          - subject overlap: LIKE on the first few subject tokens.
        Results are merged/deduped, limited, and returned as
        ``{id, subject, tags, content}`` dicts. Never raises.
        """
        if not self._db_available:
            return []
        candidates: dict[int, dict[str, Any]] = {}
        try:
            with self.connection() as conn:
                # Channel 1: tag overlap.
                if tags:
                    clean_tags = [t for t in tags if isinstance(t, str) and t.strip()]
                    if clean_tags:
                        ph = ",".join("?" * len(clean_tags))
                        ph_placeholders = clean_tags
                        rows = conn.execute(
                            f"SELECT id, subject, tags, content FROM memories "
                            f"WHERE status='active' AND id != ? AND "
                            f"EXISTS (SELECT 1 FROM json_each(tags) "
                            f"WHERE json_each.value IN ({ph}) AND json_each.type='text') "
                            f"LIMIT ?",
                            (exclude_id, *ph_placeholders, limit),
                        ).fetchall()
                        for r in rows:
                            candidates[int(r["id"])] = _row_to_dict(r)
                # Channel 2: subject overlap.
                if subject:
                    tokens = _subject_tokens(subject)
                    like_clauses: list[str] = []
                    like_params: list[Any] = []
                    for tok in tokens[:4]:  # cap at first 4 tokens
                        if len(tok) >= 2:
                            like_clauses.append("subject LIKE ?")
                            like_params.append(f"%{tok}%")
                    if like_clauses:
                        joined = " OR ".join(like_clauses)
                        rows = conn.execute(
                            f"SELECT id, subject, tags, content FROM memories "
                            f"WHERE status='active' AND id != ? AND ({joined}) "
                            f"LIMIT ?",
                            (exclude_id, *like_params, limit),
                        ).fetchall()
                        for r in rows:
                            candidates[int(r["id"])] = _row_to_dict(r)
        except sqlite3.Error:
            return []
        return list(candidates.values())

    def edit_memory_intent_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        *,
        new_content: Optional[str] = None,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        new_subject: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
        reason: Optional[str] = None,
        authorized: bool = False,
        expected_version: Optional[int] = None,
        expected_content_hash: Optional[str] = None,
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
        if new_content is not None and (old_text is not None or new_text is not None):
            return {"outcome": "invalid", "memory_id": int(memory_id), "error": "pass either new_content (full replace) or old_text+new_text (partial), not both"}
        if new_content is not None:
            if not str(new_content).strip():
                return {"outcome": "invalid", "memory_id": int(memory_id), "error": "new_content is empty; refusing to wipe memory content (use memory_supersede to retire it, or pass real content)"}
            resolved_content = str(new_content)
        elif old_text is not None and new_text is not None:
            if str(old_text) not in old_content:
                return {"outcome": "stale_edit", "memory_id": int(memory_id), "reason": "old_text_not_found", "error": "old_text not found in current content"}
            resolved_content = old_content.replace(str(old_text), str(new_text), 1)
        else:
            return {"outcome": "invalid", "memory_id": int(memory_id), "error": "provide new_content for full replace, or old_text+new_text for partial replace, or tags_only=true"}
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
            "UPDATE memories SET content=?, subject=?, tags=?, version=?, "
            "claim_revision=claim_revision+1 WHERE id=?",
            (
                resolved_content,
                subject_value,
                json.dumps(resolved_tags, ensure_ascii=False),
                old_version + 1,
                int(memory_id),
            ),
        )
        self._delete_sections_for_memory(conn, int(memory_id))
        conn.execute(
            "UPDATE memories SET split_status = NULL, "
            "split_revision = split_revision + 1 WHERE id = ?",
            (int(memory_id),),
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
            "claim_semantics_changed": True,
        }

    def edit_memory_intent(
        self,
        memory_id: int,
        *,
        new_content: Optional[str] = None,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        new_subject: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
        reason: Optional[str] = None,
        authorized: bool = False,
        expected_version: Optional[int] = None,
        expected_content_hash: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
        require_active: bool = True,
    ) -> dict[str, Any]:
        if conn is not None:
            return self.edit_memory_intent_on_conn(
                conn,
                memory_id,
                new_content=new_content,
                old_text=old_text,
                new_text=new_text,
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
        new_subject: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
        reason: Optional[str] = None,
        *,
        conn: Optional[sqlite3.Connection] = None,
        authorized: bool = True,
    ) -> Optional[int]:
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

    def cleanup_history(self, memory_id: Optional[int] = None, older_than_days: Optional[int] = None) -> int:
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
        with self.connection() as conn:
            cur = conn.execute(f"DELETE FROM memory_history {where}", params)
            conn.commit()
            return int(cur.rowcount)

    # ------------------------------------------------------------------
    #  Audit
    # ------------------------------------------------------------------
