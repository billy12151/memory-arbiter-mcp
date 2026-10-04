"""Memory row CRUD, filters, edit/history operations for MemoryDB (Phase 3 extraction)."""
from __future__ import annotations

import json
import sqlite3

import uuid
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from typing import Any, Iterator, TYPE_CHECKING

from ..config import Settings
from ..degrade import DegradeState

from ..acl import WorkspaceScope, workspace_scope_sql
from ..constants import DEFAULT_WORKSPACE_NAME, MAX_MEMORY_TOTAL_TAGS
from ..models import MemoryRecord, utc_now_iso
from ..text import (
    canon_entity as _canon_entity,
    subject_tokens as _subject_tokens,
)
from ..timeutil import parse_iso8601_utc
from ._mem_edit import _MemEditMixin
from ._mem_helpers import (  # noqa: F401  (split re-export)
    DuplicateActiveContentError as DuplicateActiveContentError,
    content_sha as content_sha,
    _row_to_dict as _row_to_dict,
    _strip_retired_metadata_keys as _strip_retired_metadata_keys,
    _RETIRED_METADATA_KEYS as _RETIRED_METADATA_KEYS,
)
from ._mem_vectors import _MemVectorsMixin

if TYPE_CHECKING:
    from .core import MemoryDB

# Gate-v2 G3 (owner 二次收紧): metadata.entity/scope are RETIRED — the
# provenance hard gate is gone and no metadata JSON in the database may
# carry these keys again, through ANY write path. Storage-level stripping
# at the three serialization points (INSERT, full UPDATE, metadata update)
# is the enforcement; tools-layer warnings are only the user-facing hint.
def _as_utc(value: datetime) -> datetime:
    """Normalise a filter bound to aware UTC (naive treated as UTC), matching
    search._parse_time's contract so direct/embedded callers cannot trip the
    aware/naive comparison (second-round review L2)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def row_passes_filters(
    tags_raw: Any,
    ingest_time_raw: Any,
    source_type_value: Any,
    *,
    tags_filter: list[str] | None,
    after_dt: datetime | None,
    before_dt: datetime | None,
    source_type: str | None,
) -> bool:
    """Single source of truth for the user-provided list filters (B2, 0.15.14).

    Both the COUNT path (count_filtered_memories) and the row paths
    (recall_by_filters, search's post-filter) run this exact predicate, so
    totals and pages can never disagree. It replaces two SQL/Python mirror
    implementations whose drift was measured live (#962 P1#6):
      - sub-second time boundaries: SQL truncated the bound to whole seconds
        (``replace(microsecond=0)``), so a .300s ``after`` bound admitted .100s
        rows that the Python post-filter dropped;
      - numeric tags: SQL ``json_each.value = ?`` never matches a JSON number
        against a string filter (verified: TEXT/REAL storage classes never
        compare equal), while Python's ``str(t)`` set membership does — here
        membership compares string forms: '1' and '1.0' are different tags.

    Only a JSON ARRAY can carry tags: a non-array shape (a bare string would
    iterate character-by-character, an object would expose its keys) matches
    nothing (second-round review L1).
    """
    if tags_filter:
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        except Exception:
            tags = None
        if not isinstance(tags, list):
            return False
        tag_set = {str(tag) for tag in tags}
        if not all(tag in tag_set for tag in tags_filter):
            return False
    if source_type and source_type_value != source_type:
        return False
    if after_dt is not None or before_dt is not None:
        parsed = parse_iso8601_utc(ingest_time_raw)
        if parsed is None:
            # Time filter active but the row has no parseable time — drop
            # conservatively (same as the former post-filter).
            return False
        if after_dt is not None and parsed < _as_utc(after_dt):
            return False
        if before_dt is not None and parsed > _as_utc(before_dt):
            return False
    return True


class MemoriesStore(_MemVectorsMixin, _MemEditMixin):
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
        # 0.16.0 §6⑮: remember persists the whole tag list — cap the total.
        if len(record.tags or []) > MAX_MEMORY_TOTAL_TAGS:
            raise ValueError(
                f"tags total {len(record.tags)} exceeds the cap of {MAX_MEMORY_TOTAL_TAGS}; "
                "tags are a retrieval dimension, not an event log "
                "(one-off state belongs in metadata)"
            )
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
        #
        # A9（0.17.1 修复批）：拒绝注册保护桶的机械等价变体（mema_twin /
        # mematwin / Mema-Twin…）——变体一旦注册，解析器的 1b 折叠会把保护桶
        # 原名折到变体拼写，twin 本体写入即落进攻击者桶（实测可读出 persona）。
        # 精确原名（mema-twin）不受影响。
        if register_workspace_canonical:
            from ..twin_redirect import protected_bucket_variant

            if protected_bucket_variant(canonical):
                raise ValueError(
                    f"workspace {canonical!r} is a protected-bucket spelling variant "
                    "and cannot be registered; use the canonical name"
                )
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

    def active_content_twin_on_conn(
        self, conn: sqlite3.Connection, workspace_canonical: str | None,
        sha: str | None, *, exclude_id: int,
    ) -> Any:
        """The ACTIVE row already holding ``sha`` in this workspace, if any.

        NULL/empty sha or canonical never matches (legacy rows stay outside
        the gate by design). ``exclude_id`` keeps a row from colliding with
        itself on no-op edits."""
        if not sha or not workspace_canonical:
            return None
        return conn.execute(
            "SELECT id, subject, ingest_time FROM memories "
            "WHERE workspace_canonical=? AND content_sha=? AND status='active' "
            "AND id != ? LIMIT 1",
            (workspace_canonical, sha, int(exclude_id)),
        ).fetchone()

    def find_active_content_duplicate(
        self, workspace_canonical: str | None, content: str,
    ) -> dict[str, Any] | None:
        """Post-violation lookup: which ACTIVE row owns this content."""
        if not workspace_canonical:
            return None
        with self.connection() as conn:
            row = self.active_content_twin_on_conn(
                conn, workspace_canonical, content_sha(content), exclude_id=-1,
            )
            return dict(row) if row is not None else None

    def insert_memory_on_conn(
        self, conn: sqlite3.Connection, record: MemoryRecord,
        workspace_canonical: str | None = None,
    ) -> int:
        canonical = (workspace_canonical or record.workspace or "").strip() or DEFAULT_WORKSPACE_NAME
        cur = conn.execute(
                """
                INSERT INTO memories
                (content, agent_id, workspace, workspace_canonical, tags, source_type, source_ref,
                 event_time, ingest_time, confidence, protection_level, status, subject, metadata, created_at,
                 content_sha)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(_strip_retired_metadata_keys(dict(record.metadata or {})), ensure_ascii=False),
                    utc_now_iso(),
                    content_sha(record.content),
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
        # B5（0.17.1 优化批）：记录最后一次降级写入时间（观测字段；不改
        # jsonl_backup_active 的单向闩语义）。
        from ..models import utc_now_iso

        self.state.jsonl_backup_last_used_at = utc_now_iso()

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

    def get_memories_by_ids(
        self, ids: "list[int]", *, conn: sqlite3.Connection | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Batch row fetch for id-driven reads (0.16.12 P1-T4): ONE connection
        (chunked IN) instead of one get_memory connection per id. Visibility
        is NOT applied here — callers run the shared caller predicate.
        ``conn`` (P2-T6): optional caller-owned connection to reuse.
        Errors propagate like get_memory's (no swallow): a mid-call failure
        must surface, not degrade the whole page to not_found."""
        if (conn is None and not self._db_available) or not ids:
            return {}
        unique = sorted({int(i) for i in ids})
        out: dict[int, dict[str, Any]] = {}

        def _fetch(c: sqlite3.Connection) -> None:
            for start in range(0, len(unique), 500):
                chunk = unique[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = c.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    record = _row_to_dict(row)
                    out[int(record["id"])] = record

        if conn is not None:
            _fetch(conn)
            return out
        with self.connection() as owned:
            _fetch(owned)
        return out

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












    @staticmethod
    def _sql_prefilter_clauses(
        tags_filter: list[str] | None,
        after_dt: datetime | None,
        before_dt: datetime | None,
        source_type: str | None,
    ) -> tuple[list[str], list[Any]]:
        """SQL narrowing that can only EXCLUDE rows the Python predicate would
        also exclude (B2 second-round review M3) — the shared predicate stays
        the single source of truth for exactness.

        - source_type: plain equality (exact).
        - tags: ``(json_each.type <> 'text' OR json_each.value = ?)`` — for a
          TEXT tag this is exact; for numbers/bools/null/objects it is a
          deliberate over-approximation (their Python ``str()`` form may equal
          the filter even though SQL cannot compare it), so such rows survive
          to the predicate instead of being wrongly dropped.
        - time: whole-second ±1s padding compared via ``julianday`` (SQLite
          parses ISO strings incl. offsets the same way the Python parser
          does, so no timezone shape is wrongly excluded) — a row outside the
          padded range can never satisfy the exact sub-second comparison; a
          row inside it still runs the exact check. Unparseable-by-SQLite
          values yield NULL julianday and are conservatively kept.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)
        if tags_filter:
            for tag in tags_filter:
                clauses.append(
                    "EXISTS (SELECT 1 FROM json_each("
                    "CASE WHEN json_valid(tags) THEN tags ELSE '[]' END"
                    ") WHERE (json_each.type <> 'text' OR json_each.value = ?))"
                )
                params.append(tag)
        if after_dt is not None:
            lower = _as_utc(after_dt).replace(microsecond=0) - timedelta(seconds=1)
            clauses.append("(julianday(ingest_time) IS NULL OR julianday(ingest_time) >= julianday(?))")
            params.append(lower.replace(tzinfo=None).isoformat()[:19])
        if before_dt is not None:
            upper = _as_utc(before_dt).replace(microsecond=0) + timedelta(seconds=1)
            clauses.append("(julianday(ingest_time) IS NULL OR julianday(ingest_time) <= julianday(?))")
            params.append(upper.replace(tzinfo=None).isoformat()[:19])
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
        """COUNT(*) under the same filters used by search's _passes_filters.

        0.15.14 (B2): SQL narrows (status, workspace scope, and the safe
        pre-filters above), then the shared row_passes_filters decides
        exactly — so the count matches the post-filtered pages on sub-second
        bounds and numeric tags. Cross-workspace (v0.7.4) — workspace is not
        filtered, EXCEPT under strict isolation where ``ws_canonical`` scopes
        the count; strict admission widens it to the admitted canonical set so
        the total keeps matching the paginated recall.
        """
        if not self._db_available:
            return 0
        clauses: list[str] = [like_status_clause]
        params: list[Any] = []
        pre_sql, pre_params = self._sql_prefilter_clauses(
            tags_filter, after_dt, before_dt, source_type,
        )
        clauses.extend(pre_sql)
        params.extend(pre_params)
        scope_sql, scope_params = workspace_scope_sql(
            "COALESCE(NULLIF(workspace_canonical, ''), workspace)", ws_canonical,
        )
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
        sql = f"SELECT tags, ingest_time, source_type FROM memories WHERE {' AND '.join(clauses)}"
        with self.connection() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.Error:
                return 0
        return sum(
            1 for row in rows
            if row_passes_filters(
                row["tags"], row["ingest_time"], row["source_type"],
                tags_filter=tags_filter, after_dt=after_dt, before_dt=before_dt,
                source_type=source_type,
            )
        )

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

        Two-stage filtering (B2, 0.15.14; second-round review M3):
          1. SQL narrowing — status + workspace scope + the EXACT-ONLY
             pre-filters (source_type equality; tags via ``json_each`` over an
             array; whole-second lower/upper time bounds padded by ±1s). These
             can only exclude rows the Python predicate would also exclude
             (numeric tags never match in SQL, so a numeric-tag filter is
             simply not pushed down — never wrongly excluded).
          2. ``row_passes_filters`` (the shared predicate) as the authoritative
             exact check, applied to the narrowed rows while streaming with
             ``fetchmany``. This keeps sub-second boundaries and numeric tags
             exact while avoiding the full-table ``SELECT *`` materialisation.
        Ordered by ingest_time DESC; the ``offset`` window is applied to the
        passing rows (cursor pagination, used by ``memory_search_expired``).
        """
        if not self._db_available:
            return []
        clauses: list[str] = [like_status_clause]
        params: list[Any] = []
        pre_sql, pre_params = self._sql_prefilter_clauses(
            tags_filter, after_dt, before_dt, source_type,
        )
        clauses.extend(pre_sql)
        params.extend(pre_params)
        scope_sql, scope_params = workspace_scope_sql(
            "COALESCE(NULLIF(workspace_canonical, ''), workspace)", ws_canonical,
        )
        if scope_sql:
            clauses.append(scope_sql)
            params.extend(scope_params)
        sql = f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY ingest_time DESC"
        window_end = int(offset) + int(limit)
        passing: list[dict[str, Any]] = []
        with self.connection() as conn:
            try:
                cursor = conn.execute(sql, params)
                while True:
                    batch = cursor.fetchmany(200)
                    if not batch:
                        break
                    for row in batch:
                        if not row_passes_filters(
                            row["tags"], row["ingest_time"], row["source_type"],
                            tags_filter=tags_filter, after_dt=after_dt, before_dt=before_dt,
                            source_type=source_type,
                        ):
                            continue
                        passing.append(_row_to_dict(row))
                    if len(passing) >= window_end:
                        break  # window filled; later rows can never enter the page
            except sqlite3.Error:
                return []
        return passing[int(offset):window_end]

    # ------------------------------------------------------------------
    #  Conflicts
    # ------------------------------------------------------------------





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





