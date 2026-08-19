"""Persistence and KNN operations for vNext local-text evidence."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, TYPE_CHECKING

from ..evidence import EvidenceUnit
from ..models import utc_now_iso

if TYPE_CHECKING:
    from .core import MemoryDB


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
            memories = int(conn.execute("SELECT COUNT(*) FROM memories WHERE status!='deleted'").fetchone()[0])
            indexed = int(conn.execute("SELECT COUNT(DISTINCT memory_id) FROM memory_evidence").fetchone()[0])
            units = int(conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0])
            try:
                vectors = int(conn.execute("SELECT COUNT(*) FROM memory_evidence_vec").fetchone()[0])
            except sqlite3.Error:
                vectors = 0
        return {"eligible_memories": memories, "indexed_memories": indexed, "units": units, "vectors": vectors}

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
