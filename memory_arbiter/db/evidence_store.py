"""Persistence and KNN operations for vNext local-text evidence."""
from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
import struct
from typing import Any, TYPE_CHECKING

from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..acl import WorkspaceScope, scope_names, workspace_scope_sql
from ..evidence import has_indexable_text, INDEXABLE_PREFILTER_SQL
from ..rowseg import RowSegment
from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


def indexable_coverage_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Single eligibility definition shared by coverage, doctor, and the
    vNext migration gate: a memory is eligible when the indexer would
    actually publish units for it. Zero-indexable-text rows (blank /
    whitespace-only legacy artifacts) are counted as non_indexable instead
    of staying a permanent coverage gap."""
    # C5: rows are the coverage oracle (unit tables retired). One SQL for
    # total+covered (subquery discipline); the uncovered walk keeps the
    # two-layer prefilter+has_indexable_text design on purpose.
    try:
        total_row = conn.execute(
            """SELECT
                 (SELECT COUNT(*) FROM memories WHERE status!='deleted') AS total,
                 (SELECT COUNT(DISTINCT m.id) FROM memories m
                  WHERE m.status!='deleted'
                    AND EXISTS(SELECT 1 FROM memory_row r WHERE r.memory_id=m.id)) AS covered"""
        ).fetchone()
    except sqlite3.OperationalError:
        # Pre-additive database (no memory_row yet): nothing indexed, the
        # uncovered walk below degrades to the full prefiltered set.
        total_row = None
    if total_row is not None:
        total = int(total_row["total"])
        covered = int(total_row["covered"])
    else:
        covered = 0
        total = int(
            conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status!='deleted'"
            ).fetchone()[0]
        )
    uncovered = 0
    try:
        pending_rows = conn.execute(
            f"""SELECT m.subject AS subject, m.content AS content FROM memories m
                WHERE m.status!='deleted' AND {INDEXABLE_PREFILTER_SQL}
                  AND NOT EXISTS(SELECT 1 FROM memory_row r WHERE r.memory_id=m.id)"""
        ).fetchall()
    except sqlite3.OperationalError:
        pending_rows = conn.execute(
            f"""SELECT m.subject AS subject, m.content AS content FROM memories m
                WHERE m.status!='deleted' AND {INDEXABLE_PREFILTER_SQL}"""
        ).fetchall()
    for row in pending_rows:
        if has_indexable_text(str(row["subject"] or ""), str(row["content"] or "")):
            uncovered += 1
    return {
        "total_memories": total,
        "eligible_memories": covered + uncovered,
        "indexed_memories": covered,
        "non_indexable_memories": total - covered - uncovered,
    }


class EvidenceStore:
    def __init__(self, db: "MemoryDB") -> None:
        self._db = db

    def text_unit_rows(
        self,
        memory_id: int,
        memory_version: int,
        *,
        span_start: int | None = None,
        span_end: int | None = None,
    ) -> list[dict[str, Any]]:
        """Current-version text units for id-driven reads (0.16.0 batch read).

        Unit-aligned window selection: with a span, units OVERLAPPING
        [span_start, span_end) are returned whole — mema's content atom is the
        evidence unit, so returning complete units removes any half-sentence
        truncation risk by construction (plan §6⑨ four-round final form). No
        vector join: this is a text read, usable while the embedder is down.
        """
        try:
            with self._db.connection() as conn:
                # C4: rows are the content atom (unit tables retired);
                # kind != 'subject' mirrors the old kind='text' intent — the
                # subject row carries no content span.
                if span_start is None or span_end is None:
                    return [dict(row) for row in conn.execute(
                        """SELECT row_index AS unit_index,kind,text,start_offset,end_offset
                           FROM memory_row
                           WHERE memory_id=? AND memory_version=? AND kind != 'subject'
                           ORDER BY row_index""",
                        (int(memory_id), int(memory_version)),
                    ).fetchall()]
                return [dict(row) for row in conn.execute(
                    """SELECT row_index AS unit_index,kind,text,start_offset,end_offset
                       FROM memory_row
                       WHERE memory_id=? AND memory_version=? AND kind != 'subject'
                         AND start_offset < ? AND end_offset > ?
                       ORDER BY row_index""",
                    (int(memory_id), int(memory_version), int(span_end), int(span_start)),
                ).fetchall()]
        except sqlite3.Error:
            return []

    def text_unit_rows_for_ids(
        self, entries: "list[tuple[int, int]]",
    ) -> dict[int, list[dict[str, Any]]]:
        """Batch text-unit fetch for id-driven reads (0.16.12 P1-T4): one
        connection (row-value IN, chunked) for the whole page instead of one
        per id. Span overlap is NOT applied here — callers filter per id in
        Python with the same predicate text_unit_rows uses in SQL. Rows come
        back per memory ordered by unit_index; ids with no current-version
        text units are simply absent from the map."""
        if not entries:
            return {}
        pairs = sorted({(int(mid), int(version)) for mid, version in entries})
        out: dict[int, list[dict[str, Any]]] = {}
        try:
            with self._db.connection() as conn:
                for start in range(0, len(pairs), 200):
                    chunk = pairs[start:start + 200]
                    placeholders = ",".join("(?,?)" for _ in chunk)
                    params = [value for pair in chunk for value in pair]
                    rows = conn.execute(
                        f"""SELECT memory_id, row_index AS unit_index, kind, text, start_offset, end_offset
                            FROM memory_row
                            WHERE (memory_id, memory_version) IN ({placeholders})
                              AND kind != 'subject'
                            ORDER BY memory_id, row_index""",
                        params,
                    ).fetchall()
                    for row in rows:
                        out.setdefault(int(row["memory_id"]), []).append(
                            {key: row[key] for key in
                             ("unit_index", "kind", "text", "start_offset", "end_offset")}
                        )
            return out
        except sqlite3.Error:
            return {}

    def outline_rows(
        self, memory_id: int, memory_version: int, *, limit: int,
    ) -> "dict[str, Any] | None":
        """Current-version outline rows for preview building (0.16.12 P1-T5).

        Returns {"total": N, "rows": [...]} with N the exact count of
        heading/text units (drives the "还有 N 段" marker) and rows the first
        ``limit`` (LIMIT must exceed the caller's max segments by 1 only for
        the caller's own needs — here we return exactly ``limit`` rows).
        None when this version has no heading/text rows published yet — the
        caller falls back to reparsing (parity guaranteed by the P1-T5 audit;
        the None branch covers the post-edit/pre-republish window)."""
        try:
            with self._db.connection() as conn:
                # C4: outline serves from rows (unit tables retired); the
                # subject row is excluded (span-less, and the preview already
                # shows subject as its own field).
                count_row = conn.execute(
                    "SELECT COUNT(*) FROM memory_row "
                    "WHERE memory_id=? AND memory_version=? AND kind != 'subject'",
                    (int(memory_id), int(memory_version)),
                ).fetchone()
                total = int(count_row[0]) if count_row else 0
                if total == 0:
                    return None
                rows = [dict(row) for row in conn.execute(
                    """SELECT row_index AS unit_index, kind, text, start_offset
                       FROM memory_row
                       WHERE memory_id=? AND memory_version=? AND kind != 'subject'
                       ORDER BY row_index LIMIT ?""",
                    (int(memory_id), int(memory_version), int(limit)),
                ).fetchall()]
            return {"total": total, "rows": rows}
        except sqlite3.Error:
            return None

    def outline_rows_for_ids(
        self, entries: "list[tuple[int, int]]",
    ) -> dict[int, list[dict[str, Any]]]:
        """Batch outline fetch (0.16.12 P1-T5): ALL heading/text rows for the
        page's ids in ONE connection (row-value IN, chunked) — keeps P1-T4's
        connection batching intact on preview paths. Per-id lists are ordered
        by unit_index; the caller derives total (=len) and the head slice.
        Ids with no current-version rows are absent from the map."""
        if not entries:
            return {}
        pairs = sorted({(int(mid), int(version)) for mid, version in entries})
        out: dict[int, list[dict[str, Any]]] = {}
        try:
            with self._db.connection() as conn:
                for start in range(0, len(pairs), 200):
                    chunk = pairs[start:start + 200]
                    placeholders = ",".join("(?,?)" for _ in chunk)
                    params = [value for pair in chunk for value in pair]
                    rows = conn.execute(
                        f"""SELECT memory_id, row_index AS unit_index, kind, text, start_offset, end_offset
                            FROM memory_row
                            WHERE (memory_id, memory_version) IN ({placeholders})
                              AND kind != 'subject'
                            ORDER BY memory_id, row_index""",
                        params,
                    ).fetchall()
                    for row in rows:
                        out.setdefault(int(row["memory_id"]), []).append(
                            {key: row[key] for key in
                             ("unit_index", "kind", "text", "start_offset", "end_offset")}
                        )
            return out
        except sqlite3.Error:
            return {}

    def row_spans_for_ids(
        self, entries: "list[tuple[int, int, int, int]]",
    ) -> dict[int, list[dict[str, Any]]]:
        """Range-limited batch row-span fetch (0.17.0 hit_window, plan Step 1).

        Each entry is (memory_id, version, lo_row_index, hi_row_index) — the
        caller derives lo/hi from its evidence hits' row_index ±window, so the
        row count stays bounded by the window, never the document length (F4).
        One connection, OR-of-ranges WHERE, chunked at 200 like the other
        batch fetches; subject rows never come back (they carry no content
        span) and ``text`` is not selected (neighbours only need offsets).

        Returns {memory_id: [{unit_index, kind, start_offset, end_offset}]}
        with each per-memory list ordered by unit_index; memories with no
        in-range current-version rows are absent from the map.
        """
        if not entries:
            return {}
        uniq = sorted({
            (int(mid), int(version), int(lo), int(hi))
            for mid, version, lo, hi in entries
        })
        out: dict[int, list[dict[str, Any]]] = {}
        try:
            with self._db.connection() as conn:
                for start in range(0, len(uniq), 200):
                    chunk = uniq[start:start + 200]
                    where = " OR ".join(
                        "(memory_id=? AND memory_version=? AND row_index BETWEEN ? AND ?)"
                        for _ in chunk
                    )
                    params = [value for entry in chunk for value in entry]
                    rows = conn.execute(
                        f"""SELECT memory_id, row_index AS unit_index, kind,
                                  start_offset, end_offset
                            FROM memory_row
                            WHERE ({where}) AND kind != 'subject'
                            ORDER BY memory_id, row_index""",
                        params,
                    ).fetchall()
                    for row in rows:
                        out.setdefault(int(row["memory_id"]), []).append(
                            {key: row[key] for key in
                             ("unit_index", "kind", "start_offset", "end_offset")}
                        )
            for per_memory in out.values():
                per_memory.sort(key=lambda row: int(row["unit_index"]))
            return out
        except sqlite3.Error:
            return {}

    def row_vectors_for_ids(
        self,
        row_ids: "list[int] | set[int]",
        *,
        conn: "sqlite3.Connection | None" = None,
    ) -> dict[int, list[float]]:
        """Batch row-vector fetch for the true-cosine gates (gate-v2 G2/G4).

        row_knn returns L2 distances only — the cosine band, the exact-hit
        boost and pair_score all need the raw vectors to compute real
        cosines (non-unit rows, |v|≈16.5, make L2-to-cos constants wrong).
        SQL discipline (#1051): one id-IN query per 200-id chunk, no text
        column; missing ids (never published / pre-backfill) are absent
        from the map, callers treat that as "no cosine available"."""
        ids = sorted({int(value) for value in row_ids})
        out: dict[int, list[float]] = {}
        if not ids:
            return out

        def _run(c: sqlite3.Connection) -> None:
            for start in range(0, len(ids), 200):
                chunk = ids[start:start + 200]
                placeholders = ",".join("?" for _ in chunk)
                rows = c.execute(
                    f"SELECT id, embedding FROM memory_row_vec "
                    f"WHERE id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    if row["embedding"] is not None:
                        out[int(row["id"])] = self._blob_to_vector(bytes(row["embedding"]))

        try:
            if conn is not None:
                _run(conn)
            else:
                with self._db.connection() as owned:
                    _run(owned)
        except sqlite3.Error:
            return {}
        return out

    # (0.17.0 C5: the unit-table knn() was retired — row_knn is the
    # only KNN over the evidence channel's vectors.)

    def coverage(self) -> dict[str, int]:
        with self._db.connection() as conn:
            counts = indexable_coverage_counts(conn)
            try:
                units = int(conn.execute("SELECT COUNT(*) FROM memory_row").fetchone()[0])
                vectors = int(conn.execute("SELECT COUNT(*) FROM memory_row_vec").fetchone()[0])
            except sqlite3.OperationalError:
                units = vectors = 0
        return {
            "eligible_memories": counts["eligible_memories"],
            "non_indexable_memories": counts["non_indexable_memories"],
            "indexed_memories": counts["indexed_memories"],
            "units": units,
            "vectors": vectors,
        }


    # (0.17.0 C5: the unit-table publish() was retired with the unit
    # channel — publish_rows is the one true publisher. Test fixtures that
    # used to hand-craft unit vectors now go through rows or the job.)

    def publish_rows(
        self,
        memory_id: int,
        memory_version: int,
        content_hash: str,
        rows: "list[RowSegment]",
        row_embeddings: "list[list[float]]",
    ) -> dict[str, Any]:
        """Rows-only publish for the存量 backfill (P2-2.5): never touches the
        unit tables (publish() would rebuild them empty). Same staleness
        checks and delete+rebuild discipline as the unit publish."""
        if len(rows) != len(row_embeddings) or any(not value for value in row_embeddings):
            return {"outcome": "invalid_row_embeddings", "published": False}
        # 缺陷修复（2026-09-25）：写路径传入的 rows 是值锚定排序（非 row_index
        # 序），而下方 fresh_ids 按 row_index 重读——两个顺序一 zip，subject 与
        # 首个值锚定行的向量互换（实测 row0←sent1vec、row1←subjectvec），每次
        # 新写入的前若干行向量全部错位，冲突检测 KNN 候选因此大面积 below_cos_floor。
        # 插入前按 row_index 排序对齐，行↔向量一一对应。
        aligned = sorted(zip(rows, row_embeddings), key=lambda pair: int(pair[0].row_index))
        rows = [r for r, _v in aligned]
        row_embeddings = [v for _r, v in aligned]
        try:
            with self._db.write_transaction() as conn:
                # C5 write-transaction tightening: ONE memories SELECT carries
                # version/content/status/content_sha (the old body read the
                # same row twice and re-hashed content inside the lock —
                # content_sha is maintained on every content edit, so a
                # non-NULL column compares directly and only legacy NULLs
                # recompute).
                current = conn.execute(
                    "SELECT version, content, status, COALESCE(content_sha,'') AS content_sha "
                    "FROM memories WHERE id=?",
                    (int(memory_id),),
                ).fetchone()
                if current is None or int(current["version"] or 1) != int(memory_version):
                    return {"outcome": "stale_snapshot", "published": False}
                from ..evidence import evidence_content_hash
                stored_sha = str(current["content_sha"] or "")
                effective_hash = (
                    stored_hash if (stored_hash := stored_sha) else
                    evidence_content_hash(str(current["content"] or ""))
                )
                if effective_hash != content_hash:
                    return {"outcome": "stale_snapshot", "published": False}
                parent_status = str(current["status"] or "deleted")
                # Subquery DELETEs — no id-list round-trip through Python.
                conn.execute(
                    "DELETE FROM memory_row_vec WHERE id IN "
                    "(SELECT id FROM memory_row WHERE memory_id=?)",
                    (int(memory_id),),
                )
                conn.execute(
                    "DELETE FROM memory_row WHERE memory_id=?", (int(memory_id),)
                )
                created = utc_now_iso()
                # Batched inserts: one executemany per table. The vec rows
                # align by re-reading the freshly inserted ids ordered by
                # row_index (rows are deleted+reinserted whole, so the pair
                # (memory_id, memory_version, row_index) identifies each).
                conn.executemany(
                    """INSERT INTO memory_row(
                         memory_id,memory_version,content_hash,row_index,kind,text,
                         start_offset,end_offset,created_at
                       ) VALUES (?,?,?,?,?,?,?,?,?)""",
                    [
                        (
                            int(memory_id), int(memory_version), content_hash,
                            int(row.row_index), row.kind, row.text,
                            int(row.start_offset), int(row.end_offset), created,
                        )
                        for row in rows
                    ],
                )
                fresh_ids = [
                    int(r["id"]) for r in conn.execute(
                        "SELECT id FROM memory_row WHERE memory_id=? AND memory_version=? "
                        "ORDER BY row_index",
                        (int(memory_id), int(memory_version)),
                    ).fetchall()
                ]
                if len(fresh_ids) != len(rows):
                    raise sqlite3.Error("row batch insert lost rows")
                conn.executemany(
                    "INSERT INTO memory_row_vec(id,parent_status,embedding) VALUES (?,?,?)",
                    [
                        (row_id, parent_status, json.dumps(embedding))
                        for row_id, embedding in zip(fresh_ids, row_embeddings)
                    ],
                )
            return {"outcome": "published", "published": True, "row_count": len(rows)}
        except sqlite3.Error as exc:
            return {"outcome": "error", "published": False, "error": str(exc)}

    def current_row_vectors(
        self, memory_id: int, memory_version: int, content_hash: str,
    ) -> "list[tuple[RowSegment, list[float]]]":
        """Return the exact current row segments and their published vectors.

        A1 timing bridge: process_conflicts reads this first and falls back to
        an in-job rowseg+embed when the publish has not landed yet — the same
        read-then-recover contract as current_text_vectors.
        """
        if not self._db.state.sqlite_vec_available:
            return []
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    """SELECT r.row_index,r.kind,r.text,r.start_offset,r.end_offset,
                              v.embedding
                       FROM memory_row r
                       JOIN memory_row_vec v ON v.id=r.id
                       WHERE r.memory_id=? AND r.memory_version=? AND r.content_hash=?
                       ORDER BY r.row_index""",
                    (int(memory_id), int(memory_version), str(content_hash)),
                ).fetchall()
            return [
                (
                    RowSegment(
                        kind=str(row["kind"]), text=str(row["text"]),
                        start_offset=int(row["start_offset"]),
                        end_offset=int(row["end_offset"]),
                        row_index=int(row["row_index"]),
                    ),
                    self._blob_to_vector(bytes(row["embedding"])),
                )
                for row in rows
                if row["embedding"] is not None
            ]
        except sqlite3.Error:
            return []

    def row_knn(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
        parent_status_filter: str = "active",
        workspace: "WorkspaceScope" = None,
        exclude_memory_id: int | None = None,
        exclude_workspaces: "list[str] | set[str] | frozenset[str] | None" = None,
        conn: "sqlite3.Connection | None" = None,
        include_subject_rows: bool = True,
        include_memory_ids: "list[int] | set[int] | None" = None,
        subject_rows_only: bool = False,
    ) -> list[dict[str, Any]]:
        """KNN over row vectors (P2-2.4) — the conflict channel's candidate
        source. Identical rowid-IN pre-filter contract as EvidenceStore.knn
        (k applies to the filtered set); candidates are short sentences or
        header-folded table rows, so Qwen always sees clean short text.
        Default k=5 mirrors the write-time unit window (evidence.py).
        Gate-v2 G5: ``include_memory_ids`` restricts the candidate set to the
        screened neighbour list (k applies to the filtered set — spike
        fact); ``subject_rows_only`` turns the query into the title coarse
        screen (one KNN per write, only kind='subject' rows)."""
        if not self._db.state.sqlite_vec_available or not query_embedding:
            return []
        if parent_status_filter == "expired":
            status_sql = "v.parent_status NOT IN ('active','deleted')"
            memory_status_sql = "m.status NOT IN ('active','deleted')"
        elif parent_status_filter == "all":
            status_sql = "v.parent_status != 'deleted'"
            memory_status_sql = "m.status != 'deleted'"
        else:
            status_sql = "v.parent_status='active'"
            memory_status_sql = "m.status='active'"
        requested_k = max(1, int(k))
        workspace_sql, workspace_params = workspace_scope_sql(
            "COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)", workspace,
        )
        from ..acl import workspace_exclusion_sql
        excl_sql, _, excl_params = workspace_exclusion_sql(exclude_workspaces)
        eligible_clauses = [memory_status_sql]
        eligible_params: list[Any] = []
        # Harness-found regression: a peer's subject row is its most-similar
        # hit for a same-topic sentence and POISONED the detection window
        # (k=5) — the opposing body rows never surfaced. Detection excludes
        # subject rows; search/placement/self-recall keep them (default).
        if subject_rows_only:
            eligible_clauses.append("r.kind = 'subject'")
        elif not include_subject_rows:
            eligible_clauses.append("r.kind != 'subject'")
        if include_memory_ids:
            ids = sorted({int(value) for value in include_memory_ids})
            placeholders = ",".join("?" for _ in ids)
            eligible_clauses.append(f"r.memory_id IN ({placeholders})")
            eligible_params.extend(ids)
        if workspace_sql:
            eligible_clauses.append(workspace_sql)
            eligible_params.extend(workspace_params)
        if excl_sql:
            eligible_clauses.append(excl_sql)
            eligible_params.extend(excl_params)
        if exclude_memory_id is not None:
            eligible_clauses.append("r.memory_id != ?")
            eligible_params.append(int(exclude_memory_id))
        filtered = bool(eligible_clauses[1:])
        id_constraint = (
            f" AND v.id IN (SELECT r.id FROM memory_row r "
            f"JOIN memories m ON m.id=r.memory_id WHERE {' AND '.join(eligible_clauses)})"
            if filtered else ""
        )
        sql = f"""SELECT r.*, v.distance AS distance, m.status, m.subject, m.tags,
                     m.workspace, m.workspace_canonical, m.source_type,
                     m.confidence, m.protection_level, m.event_time,
                     m.ingest_time, m.metadata, m.content,
                     m.version AS memory_row_version, m.agent_id,
                     m.source_ref, m.created_at AS memory_created_at
                  FROM memory_row_vec v
                  JOIN memory_row r ON r.id=v.id
                  JOIN memories m ON m.id=r.memory_id
                  WHERE v.embedding MATCH ? AND k=? AND {status_sql}
                    AND {memory_status_sql}{id_constraint}
                  ORDER BY v.distance"""
        params = [json.dumps(query_embedding), requested_k, *eligible_params]
        try:
            if conn is not None:
                rows = conn.execute(sql, params).fetchall()
            else:
                with self._db.connection() as owned:
                    rows = owned.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    @staticmethod
    def _unit_pair_identity(
        anchor_id: int, unit: Any, peer_id: int, hit: dict[str, Any],
    ) -> tuple[frozenset[str], dict[str, Any], str]:
        """Build the record_conflict-compatible identity for one unit pair.

        Single source of truth for the candidate_key hash: the real-candidate
        path and the duplicates_pool path must derive suppression lookups from
        byte-identical keys or already-recorded pairs would re-surface.
        """
        anchor_ref = f"{anchor_id}@{int(unit['memory_version'] or 1)}"
        peer_ref = f"{peer_id}@{int(hit.get('memory_version') or hit.get('memory_row_version') or 1)}"
        member_refs = frozenset([anchor_ref, peer_ref])
        evidence_by_ref = {
            anchor_ref: {
                "member": anchor_ref,
                "unit": int(unit["eid"]),
                "span": [int(unit["start_offset"] or 0), int(unit["end_offset"] or 0)],
                "hash": str(unit["content_hash"] or ""),
            },
            peer_ref: {
                "member": peer_ref,
                "unit": int(hit.get("id") or 0),
                "span": [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)],
                "hash": str(hit.get("content_hash") or ""),
            },
        }
        sorted_refs = sorted(member_refs, key=lambda ref: tuple(int(value) for value in ref.split("@", 1)))
        candidate_key = {
            "detector_version": CONFLICT_DETECTOR_VERSION,
            "members": sorted_refs,
            "evidence": [evidence_by_ref[ref] for ref in sorted_refs],
        }
        candidate_hash = hashlib.sha256(
            json.dumps(
                candidate_key, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return member_refs, candidate_key, candidate_hash

    def scan_rule_candidates(
        self,
        *,
        after_memory_id: int = 0,
        anchor_batch: int = 50,
        neighbor_k: int = 10,
        include_check: bool = False,
        max_distance: float | None = None,
        workspace: "WorkspaceScope" = None,
        similarity_pool_limit: int = 0,
        include_duplicates: bool = False,
        suspected_anomalies: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Enumerate conflict-candidate pairs for an external scan loop.

        Scheduled LLM review cannot load the whole library into a session,
        so the server enumerates the clues: for every active memory's
        current evidence units, KNN neighbours (rank-based, like the
        write-time notice path but with a wider window) pass through the
        deterministic decide_evidence rule. By default only rule-level
        notify routes (numeric/polarity/todo change) are returned —
        similarity-only check pairs are legion in topic-clustered
        libraries and are opt-in via include_check. Each pair carries the
        triggering unit snippets so the agent can triage without reading
        full memories. Pairs with an open conflict, or a version-pinned
        not_a_conflict dismissal, are filtered out.

        include_duplicates additionally exposes same-value near-duplicate
        pairs (ignore/equivalent_value|compatible_evidence) as a bounded
        duplicates_pool for governance merge; recorded pairs are suppressed
        with the same candidate-hash contract.

        C3b (0.15.13): pairing is workspace-grouped. Each anchor's KNN is
        scoped to the anchor's OWN bucket, so cross-bucket pairs are never
        generated (they cannot satisfy record_conflict's single-bucket
        group identity and used to loop weekly without landing).
        suspected_anomalies ({memory_id: suspected_bucket} from the active
        workspace_review notices) additionally sweeps each suspected
        misplaced memory against its SUSPECTED bucket: those hits are
        cross-bucket by construction and surface in a separate
        cross_bucket_references list for the running agent — authoritative
        disposition is the workspace_review notice (move), never
        record_conflict.

        Calibrated on a real 474-memory production copy: absolute vector
        distance has no discrimination there (random same-workspace pairs
        overlap notice pairs), so ranking + rules do the work and
        max_distance stays an optional extra gate.
        """
        from ..semantic_conflict import decide_evidence, is_cross_evolution

        db = self._db
        if not db.state.sqlite_vec_available:
            return {"error": "sqlite_vec_unavailable"}
        workspace_anchor_sql = ""
        anchor_params: list[Any] = []
        workspace_names = scope_names(workspace)
        echo_workspace = workspace_names[0] if workspace_names else None
        if workspace is not None:
            # Strict callers must not anchor on — or leak snippets from —
            # memories outside their admitted workspace set.
            anchor_scope_sql, anchor_scope_params = workspace_scope_sql(
                "COALESCE(NULLIF(workspace_canonical,''),workspace)", workspace,
            )
            if anchor_scope_sql:
                workspace_anchor_sql = f"AND {anchor_scope_sql} "
                anchor_params.extend(anchor_scope_params)
        with db.connection() as conn:
            anchors = [
                int(row["id"]) for row in conn.execute(
                    "SELECT id FROM memories WHERE status='active' AND id > ? "
                    + workspace_anchor_sql
                    + "ORDER BY id LIMIT ?",
                    (int(after_memory_id), *anchor_params, max(1, int(anchor_batch))),
                )
            ]
            if not anchors:
                return {
                    "anchors_scanned": 0, "next_anchor_memory_id": None,
                    "candidates": [], "counts": {"knn_pairs": 0, "rule_pass": 0,
                                                 "filtered_open": 0, "filtered_dismissed": 0,
                                                 "duplicates": 0},
                    "duplicates_pool": [], "duplicates_truncated": False,
                    "cross_bucket_references": [], "anchor_buckets": [],
                }
            # The group schema has no left/right columns. Suppression is tied
            # to the exact candidate snapshot successfully persisted by
            # record_conflict, not merely to a memory pair. That keeps an
            # unrecorded external review repeatable and allows changed member
            # versions/evidence to be reconsidered.
            recorded_candidate_statuses: dict[str, str] = {}
            active_group_members: list[frozenset[str]] = []
            dismissed_group_members: list[frozenset[str]] = []
            for row in conn.execute(
                "SELECT status,candidate_key_hash,member_versions FROM conflicts "
                "WHERE status IN ('open','applying','not_a_conflict')"
            ):
                candidate_hash = str(row["candidate_key_hash"] or "")
                status = str(row["status"])
                if candidate_hash:
                    recorded_candidate_statuses[candidate_hash] = status
                try:
                    members = json.loads(str(row["member_versions"] or "[]"))
                    refs = frozenset(
                        f"{int(member['memory_id'])}@{int(member['version'])}"
                        for member in members
                    )
                except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                    continue
                # C2 (0.15.13): a dismissed not_a_conflict pair now suppresses
                # by MEMORY-PAIR @version, not only by its exact evidence
                # snapshot. The same pair re-enumerated through different unit
                # slices (new hashes) used to resurface forever. Version
                # pinning stays: an edited memory lifts the suppression and
                # the pair is reconsidered. open/applying stay a separate
                # set so a pair with BOTH an open group and a dismissal
                # keeps the open group's precedence.
                if refs and status in {"open", "applying"}:
                    active_group_members.append(refs)
                elif refs and status == "not_a_conflict":
                    dismissed_group_members.append(refs)
            candidates: dict[tuple[int, int], dict[str, Any]] = {}
            # Spec §7.1 wide gate: similarity-only pairs dropped from the
            # default candidate set stay available as a bounded pool for the
            # caller's Qwen union instead of vanishing outright.
            similarity_pool: dict[tuple[int, int], dict[str, Any]] = {}
            # Near-duplicate (ignore/equivalent_value|compatible_evidence)
            # pairs, exposed for governance merge only when include_duplicates
            # is set. Same suppression contract as real candidates: pairs
            # already recorded (not_a_conflict/open/applying) are not
            # re-enumerated — the candidate_hash lookup runs inside the ignore
            # branch, ahead of the historical silent drop.
            duplicates_pool: dict[tuple[int, int], dict[str, Any]] = {}
            duplicates_truncated = False
            duplicates_cap = 2 * max(1, int(anchor_batch))
            pool_limit = max(0, int(similarity_pool_limit))
            knn_pair_count = 0
            stale_anchors = 0
            filtered_open = 0
            filtered_dismissed = 0
            cross_bucket_refs: dict[tuple[int, int], dict[str, Any]] = {}
            anchor_buckets: dict[str, dict[str, int]] = {}
            # C3b: suspected misplaced memories (from ACTIVE workspace_review
            # notices) are ALSO paired against their suspected bucket. Those
            # pairs are cross-bucket, cannot land in record_conflict, and go
            # to a reference list only.
            suspected = {
                int(mid): str(bucket)
                for mid, bucket in (suspected_anomalies or {}).items()
            }
            for anchor_id in anchors:
                anchor_row = conn.execute(
                    "SELECT COALESCE(NULLIF(workspace_canonical,''),workspace) AS workspace "
                    "FROM memories WHERE id=?",
                    (anchor_id,),
                ).fetchone()
                # C3b grouping: pair only within the anchor's own bucket. The
                # strict caller scope (workspace) already bounds the whole
                # page; the anchor bucket narrows pairing further.
                anchor_bucket = (
                    str(anchor_row["workspace"] or "").strip() if anchor_row else ""
                )
                # C5 per-group page accounting: doctor's broken-chain alarm
                # needs to know which bucket the walk was last inside.
                if anchor_bucket:
                    entry = anchor_buckets.setdefault(
                        anchor_bucket, {"count": 0, "last_anchor": 0},
                    )
                    entry["count"] += 1
                    entry["last_anchor"] = max(entry["last_anchor"], anchor_id)
                # C2/C5: rows are the scan source — the job no longer
                # publishes unit vectors. rowseg emits no heading rows and
                # (pre-C3) no subject rows, so the old kind='text' filter's
                # intent (subjects/headings excluded) holds by construction.
                # C3 A+ guard: subject rows never ORIGINATE a scan pair
                # (subject version progression is timeline evolution, not a
                # numeric clue) — the row-channel counterpart of the unit
                # channel's kind='text' anchor filter.
                units = conn.execute(
                    """SELECT r.id AS eid, r.text AS text, v.embedding AS embedding,
                              r.memory_version AS memory_version, r.content_hash AS content_hash,
                              r.start_offset AS start_offset, r.end_offset AS end_offset
                       FROM memory_row r
                       JOIN memory_row_vec v ON v.id=r.id
                       WHERE r.memory_id=? AND r.memory_version=(
                           SELECT version FROM memories WHERE id=?)
                         AND r.kind != 'subject'
                       ORDER BY r.id""",
                    (anchor_id, anchor_id),
                )
                first_unit = units.fetchone()
                if first_unit is None:
                    # Async republish window or a permanently failed publish:
                    # surface it instead of silently skipping forever.
                    stale_anchors += 1
                anchor_content_row = conn.execute(
                    "SELECT content FROM memories WHERE id=?", (anchor_id,),
                ).fetchone()
                anchor_content = str(anchor_content_row["content"]) if anchor_content_row else ""
                peer_content_cache: dict[int, str] = {}

                def peer_content(peer_mid: int) -> str:
                    if peer_mid not in peer_content_cache:
                        row = conn.execute(
                            "SELECT content FROM memories WHERE id=?", (peer_mid,),
                        ).fetchone()
                        peer_content_cache[peer_mid] = str(row["content"]) if row else ""
                    return peer_content_cache[peer_mid]

                def locate_span(content: str, unit_text: str, hint_start: int, hint_end: int) -> dict[str, int] | None:
                    """Validate an exact evidence span and pad it for review.

                    Evidence pipeline v2 guarantees that cleaning the source
                    slice equals the unit text. Do not search for the text:
                    repeated phrases make search ambiguous and can silently
                    choose the wrong occurrence. A failed invariant drops the
                    span and falls back to a full read.
                    """
                    from ..evidence import _clean

                    start, end = int(hint_start), int(hint_end)
                    if not (content and unit_text and 0 <= start < end <= len(content)):
                        return None
                    if _clean(content[start:end]) != unit_text:
                        return None
                    return {
                        "start": max(0, start - 128),
                        "end": min(len(content), end + 128),
                    }

                def _pool_near_duplicate(
                    peer: int, hit: "dict[str, Any]", anchor: int,
                    unit_row: "dict[str, Any]", text_a: str, reason: str,
                ) -> None:
                    """Gate-v2 G4: shared duplicates_pool admission for BOTH
                    near-duplicate sources — the deterministic ignore routes
                    and the at-ceil cosine pairs. Same suppression contract,
                    same free-dict-replace cap accounting."""
                    nonlocal duplicates_truncated
                    pair_key = (min(anchor, peer), max(anchor, peer))
                    member_refs, _key, candidate_hash = self._unit_pair_identity(
                        anchor, unit_row, peer, hit,
                    )
                    recorded = recorded_candidate_statuses.get(candidate_hash)
                    if recorded is not None or any(
                        member_refs <= group_members for group_members in active_group_members
                    ) or any(
                        member_refs <= group_members for group_members in dismissed_group_members
                    ):
                        return
                    hit_text = str(hit.get("text") or "")
                    # Re-hitting an already-pooled pair is a free dict
                    # replace, not pool growth — it must not count against
                    # the cap or flag truncation that never happened.
                    if pair_key in duplicates_pool or len(duplicates_pool) < duplicates_cap:
                        duplicates_pool[pair_key] = {
                            "left_id": pair_key[0], "right_id": pair_key[1],
                            "reason": reason,
                            "distance": float(hit.get("distance") or 0),
                            "candidate_key_hash": candidate_hash,
                            "left_snippet": text_a[:200] if pair_key[0] == anchor else hit_text[:200],
                            "right_snippet": hit_text[:200] if pair_key[1] == peer else text_a[:200],
                            "members": [
                                {
                                    "memory_id": pair_key[0],
                                    "version": int(unit_row["memory_version"] or 1) if pair_key[0] == anchor else int(hit.get("memory_version") or hit.get("memory_row_version") or 1),
                                    "attribute_raw": None, "value_raw": None,
                                    "normalized_attribute": None, "normalized_value": None,
                                    "evidence_quote": text_a if pair_key[0] == anchor else hit_text,
                                    "evidence_span": (
                                        [int(unit_row["start_offset"] or 0), int(unit_row["end_offset"] or 0)]
                                        if pair_key[0] == anchor else
                                        [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)]
                                    ),
                                    "content_hash": str(unit_row["content_hash"] or "") if pair_key[0] == anchor else str(hit.get("content_hash") or ""),
                                    "evidence_unit": int(unit_row["eid"]) if pair_key[0] == anchor else int(hit.get("id") or 0),
                                    "direction": "deterministic", "prompt_version": None,
                                    "detector_version": CONFLICT_DETECTOR_VERSION,
                                },
                                {
                                    "memory_id": pair_key[1],
                                    "version": int(hit.get("memory_version") or hit.get("memory_row_version") or 1) if pair_key[1] == peer else int(unit_row["memory_version"] or 1),
                                    "attribute_raw": None, "value_raw": None,
                                    "normalized_attribute": None, "normalized_value": None,
                                    "evidence_quote": hit_text if pair_key[1] == peer else text_a,
                                    "evidence_span": (
                                        [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)]
                                        if pair_key[1] == peer else
                                        [int(unit_row["start_offset"] or 0), int(unit_row["end_offset"] or 0)]
                                    ),
                                    "content_hash": str(hit.get("content_hash") or "") if pair_key[1] == peer else str(unit_row["content_hash"] or ""),
                                    "evidence_unit": int(hit.get("id") or 0) if pair_key[1] == peer else int(unit_row["eid"]),
                                    "direction": "deterministic", "prompt_version": None,
                                    "detector_version": CONFLICT_DETECTOR_VERSION,
                                },
                            ],
                        }
                    else:
                        duplicates_truncated = True


                # C3b: same-bucket pairing. Under a strict caller scope the
                # anchor bucket must stay inside the admitted set.
                admitted_names = set(workspace_names) if workspace_names else set()
                pairing_scope = anchor_bucket if (
                    anchor_bucket and (workspace is None or anchor_bucket in admitted_names)
                ) else workspace
                suspect_bucket = suspected.get(anchor_id)
                from ..pipeline.gates import candidate_cos_gate
                for unit in (() if first_unit is None else itertools.chain((first_unit,), units)):
                    text = str(unit["text"] or "")
                    if not text:
                        continue
                    unit_vector = self._blob_to_vector(bytes(unit["embedding"]))
                    hits = self.row_knn(
                        unit_vector,
                        k=max(1, int(neighbor_k)) + 1,
                        workspace=pairing_scope,
                        exclude_memory_id=anchor_id,
                        include_subject_rows=False,
                    )
                    # Gate-v2 G4 余弦门 (diagnostic-channel leg): below-floor
                    # pairs are noise; AT/ABOVE-ceil pairs are near-duplicates
                    # — they route into duplicates_pool below instead of being
                    # dropped (that pool IS their governance consumer). The
                    # suspect-bucket sweep stays OUTSIDE the gate: those hits
                    # are cross-bucket references, not conflict candidates.
                    gate_vectors = self.row_vectors_for_ids(
                        [int(hit["id"]) for hit in hits], conn=conn,
                    )
                    _passed, _below, at_ceil_pairs = candidate_cos_gate(unit_vector, hits, gate_vectors)
                    hits = [hit for hit, _cos in _passed]
                    if include_duplicates:
                        for dup_hit, _dup_cos in at_ceil_pairs:
                            _pool_near_duplicate(
                                int(dup_hit["memory_id"]), dup_hit, anchor_id, unit,
                                text, "near_duplicate_cosine",
                            )
                    # C3b: the suspected-bucket sweep for misplaced memories.
                    suspect_hits: list[dict[str, Any]] = []
                    if suspect_bucket and suspect_bucket != anchor_bucket:
                        suspect_hits = self.row_knn(
                            unit_vector,
                            k=max(1, int(neighbor_k)) + 1, include_subject_rows=False,
                            workspace=suspect_bucket,
                            exclude_memory_id=anchor_id,
                        )
                    for hit in itertools.chain(hits, suspect_hits):
                        # C2/C5: rows carry no 'text' kind — the old unit-
                        # channel filter's intent (exclude subject/heading
                        # units) holds by construction (rowseg emits neither).
                        peer_id = int(hit["memory_id"])
                        if peer_id == anchor_id:
                            continue
                        peer_bucket = str(hit.get("workspace_canonical") or hit.get("workspace") or "").strip()
                        if peer_bucket and anchor_bucket and peer_bucket != anchor_bucket:
                            # C3b: cross-bucket hits exist only in the
                            # suspected-bucket sweep (regular pairing is
                            # bucket-scoped). Reference-only: they can never
                            # satisfy record_conflict's single-bucket group
                            # identity. Authority is the workspace_review
                            # notice (move), never a conflict group.
                            pair_key = (min(anchor_id, peer_id), max(anchor_id, peer_id))
                            decision = decide_evidence(text, str(hit.get("text") or ""))
                            ref = cross_bucket_refs.setdefault(pair_key, {
                                "left_id": pair_key[0], "right_id": pair_key[1],
                                "workspace": anchor_bucket,
                                "suspected_workspace": suspect_bucket,
                                "reasons": set(), "distance": float(hit.get("distance") or 0),
                                "left_snippet": text[:200], "right_snippet": str(hit.get("text") or "")[:200],
                                "note": "cross-bucket reference only; disposition via the workspace_review notice (move), not record_conflict",
                            })
                            ref["reasons"].add(decision.reason)
                            ref["distance"] = min(ref["distance"], float(hit.get("distance") or 0))
                            continue
                        knn_pair_count += 1
                        # Every unit pair is judged: an earlier equivalent
                        # match (e.g. identical subjects) must not blacklist
                        # the peer, or a later numeric-change unit on the
                        # same pair would be lost.
                        decision = decide_evidence(text, str(hit.get("text") or ""))
                        # 0.16.4 §1/§0.5: the diagnostic channel routes
                        # through the SAME shared predicate as the scan
                        # pipeline and the write-time KNN loop — evolution-
                        # domain pairs must not surface here either, or the
                        # retroactive void's "re-enqueue and surface the gap"
                        # design would leak them back as notice_ready.
                        if is_cross_evolution(decision):
                            continue
                        if decision.action == "ignore":
                            if include_duplicates and decision.reason in {"equivalent_value", "compatible_evidence"}:
                                _pool_near_duplicate(
                                    peer_id, hit, anchor_id, unit, text, decision.reason,
                                )
                            continue
                        # Numeric deltas remain a deterministic scan baseline
                        # candidate even though they can no longer directly
                        # produce a write-time notice.
                        similarity_only = (
                            decision.action == "check"
                            and decision.reason != "numeric_value_candidate"
                            and not include_check
                        )
                        if similarity_only and pool_limit <= 0:
                            continue
                        if max_distance is not None and float(hit.get("distance") or 0) > float(max_distance):
                            continue
                        pair = (min(anchor_id, peer_id), max(anchor_id, peer_id))
                        distance = float(hit.get("distance") or 0)
                        store = similarity_pool if similarity_only else candidates
                        existing = store.get(pair)
                        hit_text = str(hit.get("text") or "")
                        anchor_span = locate_span(
                            anchor_content, text,
                            int(unit["start_offset"] or 0), int(unit["end_offset"] or 0),
                        )
                        peer_span = locate_span(
                            peer_content(peer_id), hit_text,
                            int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0),
                        )
                        member_refs, candidate_key, candidate_hash = self._unit_pair_identity(
                            anchor_id, unit, peer_id, hit,
                        )
                        recorded_status = recorded_candidate_statuses.get(candidate_hash)
                        if recorded_status is None and any(
                            member_refs <= group_members for group_members in active_group_members
                        ):
                            recorded_status = "open"
                        if recorded_status is None and any(
                            member_refs <= group_members for group_members in dismissed_group_members
                        ):
                            # C2: pair@version dismissal. Counted as dismissed,
                            # never silently folded into filtered_open — the
                            # counter is the convergence observability face.
                            recorded_status = "not_a_conflict"
                        if recorded_status is not None:
                            if recorded_status == "not_a_conflict":
                                filtered_dismissed += 1
                            else:
                                filtered_open += 1
                            continue
                        if existing is None:
                            state = "notice_ready" if decision.action == "notify" else "review_candidate"
                            store[pair] = {
                                "left_id": pair[0], "right_id": pair[1],
                                "state": state, "route": state,
                                "reasons": {decision.reason}, "distance": distance,
                                "candidate_key": candidate_key,
                                "candidate_key_hash": candidate_hash,
                                "members": [
                                    {
                                        "memory_id": pair[0],
                                        "version": int(hit.get("memory_version") or hit.get("memory_row_version") or 1) if pair[0] == peer_id else int(unit["memory_version"] or 1),
                                        "attribute_raw": None, "value_raw": None,
                                        "normalized_attribute": None, "normalized_value": None,
                                        "evidence_quote": text if pair[0] == anchor_id else hit_text,
                                        "evidence_span": (
                                            [int(unit["start_offset"] or 0), int(unit["end_offset"] or 0)]
                                            if pair[0] == anchor_id else
                                            [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)]
                                        ),
                                        "content_hash": str(unit["content_hash"] or "") if pair[0] == anchor_id else str(hit.get("content_hash") or ""),
                                        "evidence_unit": int(unit["eid"]) if pair[0] == anchor_id else int(hit.get("id") or 0),
                                        "direction": "deterministic", "prompt_version": None,
                                        "detector_version": CONFLICT_DETECTOR_VERSION,
                                    },
                                    {
                                        "memory_id": pair[1],
                                        "version": int(hit.get("memory_version") or hit.get("memory_row_version") or 1) if pair[1] == peer_id else int(unit["memory_version"] or 1),
                                        "attribute_raw": None, "value_raw": None,
                                        "normalized_attribute": None, "normalized_value": None,
                                        "evidence_quote": hit_text if pair[1] == peer_id else text,
                                        "evidence_span": (
                                            [int(hit.get("start_offset") or 0), int(hit.get("end_offset") or 0)]
                                            if pair[1] == peer_id else
                                            [int(unit["start_offset"] or 0), int(unit["end_offset"] or 0)]
                                        ),
                                        "content_hash": str(hit.get("content_hash") or "") if pair[1] == peer_id else str(unit["content_hash"] or ""),
                                        "evidence_unit": int(hit.get("id") or 0) if pair[1] == peer_id else int(unit["eid"]),
                                        "direction": "deterministic", "prompt_version": None,
                                        "detector_version": CONFLICT_DETECTOR_VERSION,
                                    },
                                ],
                                "value_groups": [], "slot_key": None, "slot_provenance": None,
                                "left_snippet": text[:200] if pair[0] == anchor_id else hit_text[:200],
                                "right_snippet": hit_text[:200] if pair[0] == anchor_id else text[:200],
                                # Pre-built deep-read calls: reading just the
                                # triggering region (plus context) instead of
                                # the full text keeps triage token cost low.
                                "deep_read": {
                                    "left": {
                                        "memory_id": pair[0],
                                        "span": anchor_span if pair[0] == anchor_id else peer_span,
                                        **({"workspace": echo_workspace} if echo_workspace else {}),
                                    },
                                    "right": {
                                        "memory_id": pair[1],
                                        "span": peer_span if pair[0] == anchor_id else anchor_span,
                                        **({"workspace": echo_workspace} if echo_workspace else {}),
                                    },
                                },
                            }
                        else:
                            # notice_ready outranks review_candidate when
                            # different unit pairs on the same memory pair
                            # disagree. Snippets and spans track the strongest
                            # signal, not the first discovery.
                            if not similarity_only and existing["state"] == "review_candidate" and decision.action == "notify":
                                existing["state"] = "notice_ready"
                                existing["route"] = "notice_ready"
                                existing["left_snippet"] = text[:200] if pair[0] == anchor_id else hit_text[:200]
                                existing["right_snippet"] = hit_text[:200] if pair[0] == anchor_id else text[:200]
                                existing["deep_read"] = {
                                    "left": {
                                        "memory_id": pair[0],
                                        "span": anchor_span if pair[0] == anchor_id else peer_span,
                                        **({"workspace": echo_workspace} if echo_workspace else {}),
                                    },
                                    "right": {
                                        "memory_id": pair[1],
                                        "span": peer_span if pair[0] == anchor_id else anchor_span,
                                        **({"workspace": echo_workspace} if echo_workspace else {}),
                                    },
                                }
                            existing["reasons"].add(decision.reason)
                            existing["distance"] = min(existing["distance"], distance)
            ordered = [candidates[pair] for pair in sorted(candidates)]
            for item in ordered:
                item["reasons"] = sorted(item["reasons"])
            similarity_ordered = sorted(
                similarity_pool.values(), key=lambda item: float(item.get("distance") or 9),
            )[:pool_limit]
            for item in similarity_ordered:
                item["reasons"] = sorted(item["reasons"])
            next_anchor = anchors[-1]
            with db.connection() as conn:
                more = conn.execute(
                    "SELECT 1 FROM memories WHERE status='active' AND id > ? "
                    + workspace_anchor_sql
                    + "LIMIT 1",
                    (next_anchor, *anchor_params),
                ).fetchone()
            duplicates_ordered = [
                duplicates_pool[pair] for pair in sorted(duplicates_pool)
            ]
            cross_refs_ordered = [
                {**cross_bucket_refs[pair], "reasons": sorted(cross_bucket_refs[pair]["reasons"])}
                for pair in sorted(cross_bucket_refs)
            ]
            return {
                "anchors_scanned": len(anchors),
                "next_anchor_memory_id": int(next_anchor) if more else None,
                "candidates": ordered,
                "similarity_pool": similarity_ordered,
                "duplicates_pool": duplicates_ordered,
                "duplicates_truncated": duplicates_truncated,
                "cross_bucket_references": cross_refs_ordered,
                "anchor_buckets": [
                    {"workspace": ws, "anchors_scanned": entry["count"],
                     "last_anchor": entry["last_anchor"]}
                    for ws, entry in sorted(anchor_buckets.items())
                ],
                "counts": {
                    "knn_pairs": knn_pair_count,
                    "rule_pass": len(ordered),
                    "similarity_pool": len(similarity_ordered),
                    "duplicates": len(duplicates_ordered),
                    "filtered_open": filtered_open,
                    "filtered_dismissed": filtered_dismissed,
                    "stale_anchors": stale_anchors,
                },
            }

    def scan_rows(self, memory_id: int, memory_version: int) -> list[dict[str, Any]]:
        """0.17.0 P2-3.1: current-version row segments for the scan side's
        INTERNAL examination. Same dict shape as scan_units — ``unit_index``
        carries row_index on purpose so _examine_internal stays byte-for-byte
        shared with the write side (whose internal rows land with the same
        row indexes; the 0.17.0 detector bump retires the old unit-indexed
        rows). Empty list = no rows published yet (pre-backfill), caller
        falls back to units."""
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    """SELECT r.id AS eid, r.row_index AS unit_index, r.kind AS kind,
                              r.text AS text, r.start_offset AS start_offset,
                              r.end_offset AS end_offset, r.content_hash AS content_hash,
                              v.embedding AS embedding
                       FROM memory_row r
                       LEFT JOIN memory_row_vec v ON v.id=r.id
                       WHERE r.memory_id=? AND r.memory_version=?
                       ORDER BY r.row_index""",
                    (int(memory_id), int(memory_version)),
                ).fetchall()
            decoded = []
            for row in rows:
                item = dict(row)
                blob = item.get("embedding")
                if blob is not None:
                    try:
                        item["embedding"] = self._blob_to_vector(bytes(blob))
                    except (TypeError, ValueError):
                        item["embedding"] = None
                decoded.append(item)
            return decoded
        except sqlite3.Error:
            return []

    def scan_units(self, memory_id: int, memory_version: int) -> list[dict[str, Any]]:
        """0.17.0 C6: the legacy scan-units selector now serves ROW rows
        (same dict shape: eid/text/offsets/content_hash/embedding). Kept for
        the audit surface; the pipeline itself reads scan_rows."""
        return self.scan_rows(memory_id, memory_version)

    @staticmethod
    def _blob_to_vector(blob: bytes) -> list[float]:
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))
