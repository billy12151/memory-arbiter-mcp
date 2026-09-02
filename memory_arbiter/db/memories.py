"""Memory row CRUD, filters, edit/history operations for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from typing import Any, Iterator, TYPE_CHECKING

from ..config import Settings
from ..degrade import DegradeState

from ..acl import WorkspaceScope, workspace_scope_sql
from ..constants import DEFAULT_WORKSPACE_NAME
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

    @property
    def _db_available(self) -> bool:
        return self._db._db_available

    @property
    def settings(self) -> "Settings":
        return self._db.settings

    @property
    def state(self) -> "DegradeState":
        return self._db.state

    @contextmanager
    def connection(self) -> "Iterator[sqlite3.Connection]":
        with self._db.connection() as conn:
            yield conn

    @contextmanager
    def write_transaction(self) -> "Iterator[sqlite3.Connection]":
        with self._db.write_transaction() as conn:
            yield conn

    def insert_memory(
        self,
        record: MemoryRecord,
        workspace_canonical: str | None = None,
        workspace_embedding: list[float] | None = None,
        *,
        register_workspace_canonical: bool = True,
    ) -> tuple[int | None, list[str]]:
        warnings: list[str] = []
        if not record.content:
            raise ValueError("content is required")
        if not record.subject or not str(record.subject).strip():
            raise ValueError("subject is required")
        if not self._db_available or not self.state.sqlite_writable:
            self._append_backup(record, workspace_canonical)
            warnings.append("SQLite write unavailable; wrote append-only JSONL backup.")
            return None, warnings
        # Double-store: raw workspace stays in `workspace`; resolved canonical
        # (from tools-side alias resolution) lands in `workspace_canonical`.
        # Blank/empty input collapses to DEFAULT_WORKSPACE_NAME so the column
        # is never empty on new rows.
        canonical = (workspace_canonical or record.workspace or "").strip() or DEFAULT_WORKSPACE_NAME
        # Register only the final canonical, atomically with the memory row. The
        # resolver/model runs before this transaction and must never register the
        # raw near-miss workspace (which would leave a phantom canonical).
        with self.write_transaction() as conn:
            if register_workspace_canonical:
                conn.execute(
                    "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, ?)",
                    (canonical, utc_now_iso()),
                )
            memory_id = self.insert_memory_on_conn(conn, record, canonical)
        if register_workspace_canonical:
            warnings.extend(
                self._db.workspaces.publish_workspace_canonical_vector(
                    canonical, workspace_embedding,
                )
            )
        return memory_id, warnings

    def insert_memory_on_conn(
        self, conn: sqlite3.Connection, record: MemoryRecord,
        workspace_canonical: str | None = None,
    ) -> int:
        canonical = (workspace_canonical or record.workspace or "").strip() or DEFAULT_WORKSPACE_NAME
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
        if cur.lastrowid is None:
            raise sqlite3.Error("memory insert did not return an id")
        memory_id = int(cur.lastrowid)
        if self.state.fts5_available:
            conn.execute(
                "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                (memory_id, record.content, " ".join(record.tags), record.subject or ""),
            )
        return memory_id

    def _append_backup(self, record: MemoryRecord, workspace_canonical: str | None = None) -> None:
        from datetime import datetime, timezone
        import os

        self.settings.backup_jsonl.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backup_schema": 1,
            "replay_key": str(uuid.uuid4()),
            "backup_written_at": datetime.now(timezone.utc).isoformat(),
            "workspace_canonical": workspace_canonical or record.workspace,
            "record": record.__dict__.copy(),
        }
        line = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(
            self.settings.backup_jsonl,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - non-POSIX fallback
                fcntl = None  # type: ignore[assignment]
            try:
                written = os.write(fd, line)
                if written != len(line):
                    raise OSError(f"short JSONL backup write: {written} of {len(line)} bytes")
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        self.state.jsonl_backup_active = True

    @staticmethod
    def _fetch_memory(conn: sqlite3.Connection, memory_id: int) -> dict[str, Any] | None:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def get_memory_on_conn(self, conn: sqlite3.Connection, memory_id: int) -> dict[str, Any] | None:
        """Fetch a memory using the caller's transaction/connection."""
        return self._fetch_memory(conn, int(memory_id))

    def get_memory(self, memory_id: int, *, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        if conn is not None:
            return self.get_memory_on_conn(conn, memory_id)
        if not self._db_available:
            return None
        with self.connection() as conn:
            return self._fetch_memory(conn, memory_id)

    def get_memory_for_workspace(
        self, memory_id: int, ws_canonical: str,
        admitted: WorkspaceScope = None,
    ) -> dict[str, Any] | None:
        """ACL-specific read-by-id helper; does not change get_memory semantics.

        ``admitted`` (defaulting to just ``ws_canonical``) is the strict
        vector-admission set. A single element yields the single-name equality
        filter; a larger set widens visibility to the in-radius neighbourhood.
        """
        if not self._db_available or not str(ws_canonical or "").strip():
            return None
        scope_sql, scope_params = workspace_scope_sql(
            "COALESCE(NULLIF(workspace_canonical, ''), workspace)",
            admitted if admitted else ws_canonical,
        )
        if not scope_sql:
            return None
        with self.connection() as conn:
            row = conn.execute(
                f"SELECT * FROM memories WHERE id = ? AND {scope_sql}",
                (int(memory_id), *scope_params),
            ).fetchone()
            return _row_to_dict(row) if row else None

    def list_memories_for_workspace(
        self, ws_canonical: str, limit: int = 50,
        admitted: WorkspaceScope = None,
    ) -> list[dict[str, Any]]:
        """ACL-specific recent/list helper scoped to the admitted canonical set."""
        if not self._db_available or not str(ws_canonical or "").strip():
            return []
        scope_sql, scope_params = workspace_scope_sql(
            "COALESCE(NULLIF(workspace_canonical, ''), workspace)",
            admitted if admitted else ws_canonical,
        )
        if not scope_sql:
            return []
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE status != 'deleted' AND {scope_sql} "
                "ORDER BY event_time DESC, ingest_time DESC LIMIT ?",
                (*scope_params, int(limit)),
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
        snapshot_semantics_changed = any(
            key in {"source_type", "confidence", "protection_level", "status"}
            and current.get(key) != value
            for key, value in pairs
        )
        metadata_update = next((value for key, value in pairs if key == "metadata"), None)
        if isinstance(metadata_update, dict):
            current_md = current.get("metadata") or {}
            current_md = current_md if isinstance(current_md, dict) else {}
            snapshot_semantics_changed = snapshot_semantics_changed or (
                _canon_entity(current_md.get("entity")) != _canon_entity(metadata_update.get("entity"))
                or _canon_scope(current_md.get("scope")) != _canon_scope(metadata_update.get("scope"))
            )
        sql = ", ".join(f"{key} = ?" for key, _ in pairs)
        if snapshot_semantics_changed:
            sql += ", version = version + 1"
        values = [
            json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            for _, v in pairs
        ]
        values.append(int(memory_id))
        conn.execute(f"UPDATE memories SET {sql} WHERE id = ?", values)
        if snapshot_semantics_changed:
            # These updates do not change evidence text. Keep the derived rows
            # pinned to the new authoritative memory version in the same
            # transaction instead of making doctor report false staleness.
            conn.execute(
                "UPDATE memory_evidence SET memory_version=memory_version+1 "
                "WHERE memory_id=?",
                (int(memory_id),),
            )
        if status_changed and self.state.sqlite_vec_available:
            try:
                conn.execute(
                    "UPDATE memory_evidence_vec SET parent_status=? WHERE id IN "
                    "(SELECT id FROM memory_evidence WHERE memory_id=?)",
                    (str(new_status or "deleted"), int(memory_id)),
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

    def list_memories(self, subject: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
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

    def active_subject_tag_rows(
        self, exclude_memory_id: int, workspace_canonical: str | None,
    ) -> list[dict[str, Any]]:
        """Lightweight id/subject/tags rows for the write-time duplicate hint.

        Same-workspace active memories only: the hint must never leak a
        subject the caller could not read, and cross-workspace near-duplicates
        are not this check's business.
        """
        if not self._db_available:
            return []
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id, subject, tags, event_time, ingest_time FROM memories "
                "WHERE status = 'active' AND id != ? "
                "AND COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?",
                (int(exclude_memory_id), workspace_canonical),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                tags = json.loads(row["tags"]) if row["tags"] else []
            except (TypeError, ValueError):
                tags = []
            out.append({
                "id": int(row["id"]),
                "subject": str(row["subject"] or ""),
                "tags": [str(tag) for tag in tags] if isinstance(tags, list) else [],
                "event_time": row["event_time"],
            })
        return out

    @staticmethod
    def _filter_clauses(
        like_status_clause: str,
        tags_filter: list[str] | None,
        after_dt: datetime | None,
        before_dt: datetime | None,
        source_type: str | None,
    ) -> tuple[list[str], list[Any]]:
        """WHERE clause + params shared by count_filtered_memories and recall_by_filters.

        Mirrors search._passes_filters: like_status_clause + per-tag json_each exact
        match (AND semantics) + ingest_time ISO-string bounds + source_type equality.
        workspace is intentionally NOT filtered (v0.7.4 cross-workspace search).
        """
        clauses: list[str] = [like_status_clause]
        params: list[Any] = []
        if tags_filter:
            for tag in tags_filter:
                # CASE guard: json_each raises on malformed JSON, and one
                # legacy/imported row with bad tags would abort the whole
                # aggregate (the callers' sqlite3.Error handlers silently
                # return 0 rows). Bad rows simply match nothing.
                clauses.append(
                    "EXISTS (SELECT 1 FROM json_each("
                    "CASE WHEN json_valid(tags) THEN tags ELSE '[]' END"
                    ") WHERE json_each.value = ?)"
                )
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
        tags_filter: list[str] | None,
        after_dt: datetime | None,
        before_dt: datetime | None,
        source_type: str | None,
        ws_canonical: WorkspaceScope = None,
    ) -> int:
        """v0.7.3: COUNT(*) under the same filters used by search's _passes_filters.

        Only called when has_filters=True. Clauses are built by _filter_clauses so the
        SQL count and the SQL recall (recall_by_filters) share one source of truth.
        Cross-workspace (v0.7.4) — workspace is not filtered, EXCEPT under strict
        isolation where ``ws_canonical`` scopes the count. Strict admission widens
        it from one name to the admitted canonical set, so the total keeps
        matching the paginated recall under vector admission.
        """
        if not self._db_available:
            return 0
        clauses, params = self._filter_clauses(like_status_clause, tags_filter, after_dt, before_dt, source_type)
        scope_sql, scope_params = workspace_scope_sql(
            "COALESCE(NULLIF(workspace_canonical, ''), workspace)", ws_canonical,
        )
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
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
        tags_filter: list[str] | None,
        after_dt: datetime | None,
        before_dt: datetime | None,
        source_type: str | None,
        limit: int,
        offset: int = 0,
        ws_canonical: WorkspaceScope = None,
    ) -> list[dict[str, Any]]:
        """G6 (v0.8.5): filter-driven recall for empty-query + filters in memory_search.

        SELECT * mirroring count_filtered_memories' WHERE (via _filter_clauses),
        ordered by ingest_time DESC, capped at ``limit`` (pool_cap). Returns
        row_to_dict rows — same shape as _wide_recall pool rows. Enables
        list-by-tag / by-source_type / by-time when query is empty.

        v0.9.4: ``offset`` adds SQL OFFSET for cursor pagination on the
        exact-count filter path (used by ``memory_search_expired``).
        v0.9.7/``ws_canonical`` hard-scopes to the admitted canonical
        set in SQL, so pagination and totals stay correct (a Python post-filter
        on an already-paginated page would not).
        """
        if not self._db_available:
            return []
        clauses, params = self._filter_clauses(like_status_clause, tags_filter, after_dt, before_dt, source_type)
        scope_sql, scope_params = workspace_scope_sql(
            "COALESCE(NULLIF(workspace_canonical, ''), workspace)", ws_canonical,
        )
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
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
        conn.execute(
            "UPDATE memories SET tags=? WHERE id=?",
            (json.dumps(new_tags_list, ensure_ascii=False), memory_id),
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
        if metadata == before:
            return {"outcome": "no_change", "memory_id": memory_id, "metadata": metadata}
        conn.execute(
            "UPDATE memories SET metadata=?,version=version+1 WHERE id=?",
            (json.dumps(metadata, ensure_ascii=False), int(memory_id)),
        )
        return {"outcome": "updated", "memory_id": memory_id, "metadata": metadata}

    def list_entities(
        self,
        limit: int = 50,
        include_unassigned: bool = True,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id,metadata FROM memories WHERE status='active' ORDER BY id"
            ).fetchall()
        counts: dict[str, int] = {}
        samples: dict[str, int] = {}
        unassigned: list[int] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata"] or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            entity = _canon_entity(metadata.get("entity")) if isinstance(metadata, dict) else ""
            if entity:
                counts[entity] = counts.get(entity, 0) + 1
                samples.setdefault(entity, int(row["id"]))
            elif include_unassigned:
                unassigned.append(int(row["id"]))
        cap = max(1, min(500, int(limit)))
        return {
            "entities": [
                {"entity": key, "count": counts[key], "sample_memory_id": samples[key]}
                for key in sorted(counts, key=lambda value: (-counts[value], value))[:cap]
            ],
            "distinct_entities": len(counts),
            "assigned_count": sum(counts.values()),
            "total_active": len(rows),
            "unassigned_count": len(rows) - sum(counts.values()),
            "unassigned_ids": unassigned[:cap],
        }

    def find_metadata_overlap_candidates(
        self,
        subject: str | None,
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
                            f"EXISTS (SELECT 1 FROM json_each("
                            f"CASE WHEN json_valid(tags) THEN tags ELSE '[]' END) "
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

    def find_semantic_overlap_candidates(
        self,
        subject: str | None,
        tags: list[str],
        exclude_id: int,
        limit: int = 50,
        canonical_workspace: str | None = None,
        isolation: str = "none",
    ) -> list[dict[str, Any]]:
        """Return a bounded, metadata-only shortlist for semantic classification.

        The SQL uses the same subject/tag overlap channels as write hints, but
        ranks and limits them before rows leave SQLite. Content is intentionally
        excluded; callers fetch it only for the selected pairs.
        """
        if not self._db_available or limit <= 0:
            return []

        # Every query tag is preserved. A single JSON value carries the complete
        # set into SQL, so a distinctive tag near the end cannot disappear behind
        # an arbitrary Python slice and parameter counts stay constant.
        query_tags = list(dict.fromkeys(
            tag.strip().casefold() for tag in tags
            if isinstance(tag, str) and tag.strip()
            and tag.strip().casefold() not in {"todo", "待办"}
            and len(tag.strip()) > 1
        ))
        # Punctuation must delimit ASCII subject words (``parser: behavior``),
        # rather than becoming part of a broad LIKE token.
        clean_subject = "".join(
            char if char.isalnum() or char == "_" else " " for char in (subject or "")
        )
        subject_tokens = list(dict.fromkeys(
            token.casefold() for token in _subject_tokens(clean_subject) if len(token) >= 2
        ))[:4]
        if not query_tags and not subject_tokens:
            return []

        isolation = str(isolation or "none").strip().lower()
        canonical_workspace = str(canonical_workspace or "").strip() or None
        if isolation != "none" and not canonical_workspace:
            return []
        workspace_clause = ""
        workspace_params: list[Any] = []
        if isolation != "none":
            workspace_clause = (
                " AND COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace) = ?"
            )
            workspace_params.append(canonical_workspace)

        # Only this bounded pool is expanded with json_each and scored. The cap is
        # independent of caller input: candidate_limit controls returned rows, not
        # how much of a large workspace may be materialized for expensive scoring.
        pool_limit = min(500, max(64, int(limit) * 8))
        tag_json = json.dumps(query_tags, ensure_ascii=False)
        subject_json = json.dumps(subject_tokens, ensure_ascii=False)

        subject_score = " + ".join(
            "CASE WHEN lower(pool.subject) LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END"
            for _ in subject_tokens
        ) or "0"
        subject_params = [
            "%" + token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            for token in subject_tokens
        ]

        # FTS is the indexed channel. Its query is assembled inside SQLite from
        # the same JSON tag argument (plus bounded subject tokens), avoiding one
        # SQL statement/parameter per tag. A recent metadata channel protects
        # recall when FTS is unavailable or tokenization is unhelpful.
        if self.state.fts5_available:
            indexed_pool = f"""
                indexed_ids AS MATERIALIZED (
                    SELECT m.id, m.subject, m.tags, m.created_at
                    FROM memories_fts
                    JOIN memories m ON m.id=memories_fts.rowid
                    WHERE m.status='active' AND m.id != ?{workspace_clause}
                      AND memories_fts MATCH (
                          SELECT group_concat(term, ' OR ') FROM (
                              SELECT 'tags : "' || replace(tag, '"', '""') || '"' AS term
                              FROM input_tags
                              UNION ALL
                              SELECT 'subject : "' || replace(token, '"', '""') || '"' AS term
                              FROM input_subject
                          )
                      )
                    ORDER BY bm25(memories_fts), m.created_at DESC, m.id DESC
                    LIMIT ?
                ),
            """
            indexed_params: list[Any] = [int(exclude_id), *workspace_params, pool_limit]
        else:
            indexed_pool = "indexed_ids AS MATERIALIZED (SELECT NULL AS id, NULL AS subject, NULL AS tags, NULL AS created_at WHERE 0),"
            indexed_params = []

        recent_where = "m.status='active' AND m.id != ?" + workspace_clause
        sql = f"""
            WITH
            input_tags(tag) AS MATERIALIZED (
                SELECT DISTINCT lower(value) FROM json_each(?) WHERE type='text'
            ),
            input_subject(token) AS MATERIALIZED (
                SELECT DISTINCT lower(value) FROM json_each(?) WHERE type='text'
            ),
            {indexed_pool}
            recent_ids AS MATERIALIZED (
                SELECT m.id, m.subject, m.tags, m.created_at
                FROM memories m
                WHERE {recent_where}
                ORDER BY m.created_at DESC, m.id DESC
                LIMIT ?
            ),
            pool AS MATERIALIZED (
                SELECT id, subject, tags, created_at FROM (
                    SELECT *, 0 AS source_rank FROM indexed_ids
                    UNION ALL
                    SELECT *, 1 AS source_rank FROM recent_ids
                )
                GROUP BY id
                ORDER BY MIN(source_rank), created_at DESC, id DESC
                LIMIT ?
            ),
            tag_hits AS MATERIALIZED (
                SELECT pool.id, input_tags.tag
                FROM pool
                JOIN json_each(CASE WHEN json_valid(pool.tags) THEN pool.tags ELSE '[]' END) stored_tag ON stored_tag.type='text'
                JOIN input_tags ON lower(stored_tag.value)=input_tags.tag
                GROUP BY pool.id, input_tags.tag
            ),
            useful_tags AS MATERIALIZED (
                SELECT input_tags.tag
                FROM input_tags
                LEFT JOIN tag_hits ON tag_hits.tag=input_tags.tag
                GROUP BY input_tags.tag
                HAVING NOT (
                    COUNT(tag_hits.id) + 1 >= 3
                    AND CAST(COUNT(tag_hits.id) + 1 AS REAL) /
                        ((SELECT COUNT(*) FROM pool) + 1) >= 0.5
                )
            ),
            tag_scores AS MATERIALIZED (
                SELECT tag_hits.id, COUNT(*) AS tag_overlap
                FROM tag_hits JOIN useful_tags ON useful_tags.tag=tag_hits.tag
                GROUP BY tag_hits.id
            ),
            scored AS (
                SELECT pool.id, pool.subject, pool.tags, pool.created_at,
                       COALESCE(tag_scores.tag_overlap, 0) AS tag_overlap,
                       ({subject_score}) AS subject_overlap
                FROM pool LEFT JOIN tag_scores ON tag_scores.id=pool.id
            )
            SELECT id, subject, tags, tag_overlap, subject_overlap
            FROM scored
            WHERE tag_overlap > 0 OR subject_overlap > 0
            ORDER BY (tag_overlap > 0 AND subject_overlap > 0) DESC,
                     tag_overlap DESC, subject_overlap DESC,
                     created_at DESC, id DESC
            LIMIT ?
        """
        params = [
            tag_json, subject_json, *indexed_params,
            int(exclude_id), *workspace_params, pool_limit, pool_limit,
            *subject_params, int(limit),
        ]
        try:
            with self.connection() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []
        return [_row_to_dict(row) for row in rows]

    def edit_memory_intent_on_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        *,
        new_content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
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
            "UPDATE memories SET content=?, subject=?, tags=?, version=? WHERE id=?",
            (
                resolved_content,
                subject_value,
                json.dumps(resolved_tags, ensure_ascii=False),
                old_version + 1,
                int(memory_id),
            ),
        )
        evidence_ids = [int(row["id"]) for row in conn.execute(
            "SELECT id FROM memory_evidence WHERE memory_id=?", (int(memory_id),)
        ).fetchall()]
        if evidence_ids and self.state.sqlite_vec_available:
            placeholders = ",".join("?" for _ in evidence_ids)
            conn.execute(f"DELETE FROM memory_evidence_vec WHERE id IN ({placeholders})", evidence_ids)
        conn.execute("DELETE FROM memory_evidence WHERE memory_id=?", (int(memory_id),))
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
