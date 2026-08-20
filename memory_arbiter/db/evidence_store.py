"""Persistence and KNN operations for vNext local-text evidence."""
from __future__ import annotations

import json
import sqlite3
import struct
from typing import Any, TYPE_CHECKING

from ..evidence import EvidenceUnit, has_indexable_text, INDEXABLE_PREFILTER_SQL
from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


def indexable_coverage_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Single eligibility definition shared by coverage, doctor, and the
    vNext migration gate: a memory is eligible when the indexer would
    actually publish units for it. Zero-indexable-text rows (blank /
    whitespace-only legacy artifacts) are counted as non_indexable instead
    of staying a permanent coverage gap."""
    total = int(
        conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status!='deleted'"
        ).fetchone()[0]
    )
    covered = int(
        conn.execute(
            """SELECT COUNT(DISTINCT m.id) FROM memories m
               WHERE m.status!='deleted'
                 AND EXISTS(SELECT 1 FROM memory_evidence e WHERE e.memory_id=m.id)"""
        ).fetchone()[0]
    )
    uncovered = 0
    for row in conn.execute(
        f"""SELECT m.subject AS subject, m.content AS content FROM memories m
            WHERE m.status!='deleted' AND {INDEXABLE_PREFILTER_SQL}
              AND NOT EXISTS(SELECT 1 FROM memory_evidence e WHERE e.memory_id=m.id)"""
    ):
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

    def publish(
        self,
        memory_id: int,
        memory_version: int,
        content_hash: str,
        units: list[EvidenceUnit],
        embeddings: list[list[float]],
    ) -> dict[str, Any]:
        if len(units) != len(embeddings) or any(not value for value in embeddings):
            return {"outcome": "invalid_embeddings", "published": False}
        try:
            with self._db.write_transaction() as conn:
                current = conn.execute(
                    "SELECT version, content FROM memories WHERE id=?", (int(memory_id),)
                ).fetchone()
                if current is None or int(current["version"] or 1) != int(memory_version):
                    return {"outcome": "stale_snapshot", "published": False}
                from ..evidence import evidence_content_hash
                if evidence_content_hash(str(current["content"] or "")) != content_hash:
                    return {"outcome": "stale_snapshot", "published": False}

                old_ids = [
                    int(row["id"]) for row in conn.execute(
                        "SELECT id FROM memory_evidence WHERE memory_id=?", (int(memory_id),)
                    ).fetchall()
                ]
                if old_ids:
                    placeholders = ",".join("?" for _ in old_ids)
                    conn.execute(f"DELETE FROM memory_evidence_vec WHERE id IN ({placeholders})", old_ids)
                conn.execute("DELETE FROM memory_evidence WHERE memory_id=?", (int(memory_id),))

                status_row = conn.execute("SELECT status FROM memories WHERE id=?", (int(memory_id),)).fetchone()
                parent_status = str(status_row["status"] if status_row else "deleted")
                for unit, embedding in zip(units, embeddings):
                    cur = conn.execute(
                        """INSERT INTO memory_evidence(
                             memory_id,memory_version,content_hash,unit_index,kind,text,
                             start_offset,end_offset,created_at
                           ) VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            int(memory_id), int(memory_version), content_hash,
                            int(unit.unit_index), unit.kind, unit.text,
                            int(unit.start_offset), int(unit.end_offset), utc_now_iso(),
                        ),
                    )
                    if cur.lastrowid is None:
                        raise sqlite3.Error("evidence insert did not return an id")
                    conn.execute(
                        "INSERT INTO memory_evidence_vec(id,parent_status,embedding) VALUES (?,?,?)",
                        (int(cur.lastrowid), parent_status, json.dumps(embedding)),
                    )
            return {"outcome": "published", "published": True, "unit_count": len(units)}
        except sqlite3.Error as exc:
            return {"outcome": "error", "published": False, "error": str(exc)}

    def delete_for_memory(self, memory_id: int) -> None:
        with self._db.write_transaction() as conn:
            ids = [int(row["id"]) for row in conn.execute(
                "SELECT id FROM memory_evidence WHERE memory_id=?", (int(memory_id),)
            )]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(f"DELETE FROM memory_evidence_vec WHERE id IN ({placeholders})", ids)
            conn.execute("DELETE FROM memory_evidence WHERE memory_id=?", (int(memory_id),))

    def coverage(self) -> dict[str, int]:
        with self._db.connection() as conn:
            counts = indexable_coverage_counts(conn)
            units = int(conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0])
            try:
                vectors = int(conn.execute("SELECT COUNT(*) FROM memory_evidence_vec").fetchone()[0])
            except sqlite3.Error:
                vectors = 0
        return {
            "eligible_memories": counts["eligible_memories"],
            "non_indexable_memories": counts["non_indexable_memories"],
            "indexed_memories": counts["indexed_memories"],
            "units": units,
            "vectors": vectors,
        }

    def knn(
        self,
        query_embedding: list[float],
        *,
        k: int = 100,
        parent_status_filter: str = "active",
        workspace: str | None = None,
        exclude_memory_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """KNN over evidence vectors, filtered by parent lifecycle state.

        sqlite-vec pushes only the vec0 auxiliary column (``parent_status``)
        into the KNN itself; the workspace and exclude predicates filter
        joined tables *after* the global top-k. To keep per-workspace recall
        from collapsing when other workspaces own the globally nearest rows,
        callers with those filters get a bounded over-fetch factor. A full
        pre-filter would need a workspace partition key on the vec0 table
        (deferred, spec §19).
        """
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
        # parent_status is best-effort (a vec-disabled process can skip its
        # update), so the authoritative memories.status is enforced too.
        clauses = [status_sql, memory_status_sql]
        fetch_k = max(1, int(k))
        if workspace or exclude_memory_id is not None:
            # Bounded over-fetch; the absolute ceiling keeps deep paginated
            # queries (pool_cap grows with offset) from multiplying into a
            # full-index KNN scan.
            fetch_k = min(fetch_k * 4, 2048)
        params: list[Any] = [json.dumps(query_embedding), fetch_k]
        if workspace:
            clauses.append("COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)=?")
            params.append(workspace)
        if exclude_memory_id is not None:
            clauses.append("e.memory_id!=?")
            params.append(int(exclude_memory_id))
        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    f"""SELECT e.*, v.distance AS distance, m.status, m.subject, m.tags,
                               m.workspace, m.workspace_canonical, m.source_type,
                               m.confidence, m.protection_level, m.event_time,
                               m.ingest_time, m.metadata, m.content,
                               m.version AS memory_row_version, m.agent_id,
                               m.source_ref, m.created_at AS memory_created_at
                        FROM memory_evidence_vec v
                        JOIN memory_evidence e ON e.id=v.id
                        JOIN memories m ON m.id=e.memory_id
                        WHERE v.embedding MATCH ? AND k=? AND {' AND '.join(clauses)}
                        ORDER BY v.distance""",
                    params,
                ).fetchall()
            results = [dict(row) for row in rows]
            if workspace or exclude_memory_id is not None:
                results = results[:max(1, int(k))]
            return results
        except sqlite3.Error:
            return []

    def scan_rule_candidates(
        self,
        *,
        after_memory_id: int = 0,
        anchor_batch: int = 50,
        neighbor_k: int = 10,
        include_check: bool = False,
        max_distance: float | None = None,
        workspace: str | None = None,
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

        Calibrated on a real 474-memory production copy: absolute vector
        distance has no discrimination there (random same-workspace pairs
        overlap notice pairs), so ranking + rules do the work and
        max_distance stays an optional extra gate.
        """
        from ..semantic_conflict import decide_evidence

        db = self._db
        if not db.state.sqlite_vec_available:
            return {"error": "sqlite_vec_unavailable"}
        workspace_anchor_sql = ""
        anchor_params: list[Any] = []
        if workspace is not None:
            # Strict callers must not anchor on — or leak snippets from —
            # memories outside their workspace.
            workspace_anchor_sql = (
                "AND COALESCE(NULLIF(workspace_canonical,''),workspace)=? "
            )
            anchor_params.append(workspace)
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
                                                 "filtered_open": 0, "filtered_dismissed": 0},
                }
            open_pairs = {
                (min(int(r["left_id"]), int(r["right_id"])), max(int(r["left_id"]), int(r["right_id"])))
                for r in conn.execute(
                    "SELECT left_id, right_id FROM conflicts WHERE status='open'"
                )
            }
            dismissed_pairs = db.conflicts.dismissed_pairs_snapshot()
            candidates: dict[tuple[int, int], dict[str, Any]] = {}
            knn_pair_count = 0
            stale_anchors = 0
            filtered_open = 0
            filtered_dismissed = 0
            for anchor_id in anchors:
                # Only body-text units participate, matching the write-time
                # notice path: subjects/headings are version-progression
                # heavy and fire numeric_value_changed on their own.
                units = conn.execute(
                    """SELECT e.id AS eid, e.text AS text, v.embedding AS embedding
                       FROM memory_evidence e
                       JOIN memory_evidence_vec v ON v.id=e.id
                       WHERE e.memory_id=? AND e.memory_version=(
                           SELECT version FROM memories WHERE id=?)
                         AND e.kind='text'
                       ORDER BY e.id""",
                    (anchor_id, anchor_id),
                ).fetchall()
                if not units:
                    # Async republish window or a permanently failed publish:
                    # surface it instead of silently skipping forever.
                    stale_anchors += 1
                for unit in units:
                    text = str(unit["text"] or "")
                    if not text:
                        continue
                    hits = self.knn(
                        self._blob_to_vector(bytes(unit["embedding"])),
                        k=max(1, int(neighbor_k)) + 1,
                        workspace=workspace,
                        exclude_memory_id=anchor_id,
                    )
                    for hit in hits:
                        if hit.get("kind") != "text":
                            continue
                        peer_id = int(hit["memory_id"])
                        if peer_id == anchor_id:
                            continue
                        knn_pair_count += 1
                        # Every unit pair is judged: an earlier equivalent
                        # match (e.g. identical subjects) must not blacklist
                        # the peer, or a later numeric-change unit on the
                        # same pair would be lost.
                        decision = decide_evidence(text, str(hit.get("text") or ""))
                        if decision.action == "ignore":
                            continue
                        if decision.action == "check" and not include_check:
                            continue
                        if max_distance is not None and float(hit.get("distance") or 0) > float(max_distance):
                            continue
                        pair = (min(anchor_id, peer_id), max(anchor_id, peer_id))
                        if pair in open_pairs:
                            filtered_open += 1
                            continue
                        if pair in dismissed_pairs:
                            filtered_dismissed += 1
                            continue
                        distance = float(hit.get("distance") or 0)
                        existing = candidates.get(pair)
                        if existing is None:
                            candidates[pair] = {
                                "left_id": pair[0], "right_id": pair[1],
                                "route": decision.action, "reasons": {decision.reason},
                                "distance": distance,
                                "left_snippet": text[:200] if pair[0] == anchor_id else str(hit.get("text") or "")[:200],
                                "right_snippet": str(hit.get("text") or "")[:200] if pair[0] == anchor_id else text[:200],
                            }
                        else:
                            # notify outranks check when different unit pairs
                            # on the same memory pair disagree — and the
                            # snippets must show the strongest signal, not
                            # the first discovery.
                            if existing["route"] == "check" and decision.action == "notify":
                                existing["route"] = "notify"
                                existing["left_snippet"] = text[:200] if pair[0] == anchor_id else str(hit.get("text") or "")[:200]
                                existing["right_snippet"] = str(hit.get("text") or "")[:200] if pair[0] == anchor_id else text[:200]
                            existing["reasons"].add(decision.reason)
                            existing["distance"] = min(existing["distance"], distance)
            ordered = [candidates[pair] for pair in sorted(candidates)]
            for item in ordered:
                item["reasons"] = sorted(item["reasons"])
            next_anchor = anchors[-1]
            with db.connection() as conn:
                more = conn.execute(
                    "SELECT 1 FROM memories WHERE status='active' AND id > ? LIMIT 1",
                    (next_anchor,),
                ).fetchone()
            return {
                "anchors_scanned": len(anchors),
                "next_anchor_memory_id": int(next_anchor) if more else None,
                "candidates": ordered,
                "counts": {
                    "knn_pairs": knn_pair_count,
                    "rule_pass": len(ordered),
                    "filtered_open": filtered_open,
                    "filtered_dismissed": filtered_dismissed,
                    "stale_anchors": stale_anchors,
                },
            }

    @staticmethod
    def _blob_to_vector(blob: bytes) -> list[float]:
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))
