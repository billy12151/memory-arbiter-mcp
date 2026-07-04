from __future__ import annotations

from typing import Any, Optional, Tuple

from .db import MemoryDB, row_to_dict


def _sanitize_fts_query(query: str) -> str:
    """Turn an arbitrary user query into a safe FTS5 MATCH expression.

    FTS5 has its own query grammar where ``. : * " ( ) - + AND OR NOT`` are
    special. A bare query like ``v0.2.1`` raises ``fts5: syntax error near "."``.
    We split on whitespace and wrap each token as a double-quoted phrase
    (with ``"`` escaped as ``""``), joined by ``AND`` so every term must match.
    """
    tokens = [tok for tok in query.split() if tok]
    if not tokens:
        return ""
    quoted = ['"' + tok.replace('"', '""') + '"' for tok in tokens]
    return " AND ".join(quoted)


def search_memories(db: MemoryDB, query: str, workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10) -> Tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if db.conn is None:
        return [], ["SQLite unavailable; search cannot read JSONL backup in MVP."]
    limit = max(1, min(int(limit), 100))
    query = (query or "").strip()
    rows = []
    if db.state.fts5_available and query:
        sql = """
            SELECT m.*, bm25(memories_fts) AS score
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ? AND m.status != 'deleted'
        """
        params: list[Any] = [_sanitize_fts_query(query)]
        if workspace:
            sql += " AND m.workspace = ?"
            params.append(workspace)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        try:
            rows = db.conn.execute(sql, params).fetchall()
        except Exception as exc:
            warnings.append(f"FTS5 query failed: {exc}. Falling back to LIKE search.")
            rows = []
    if not rows:
        like = f"%{query}%"
        clauses = ["status != 'deleted'"]
        params = []
        if query:
            clauses.append("(content LIKE ? OR subject LIKE ? OR tags LIKE ?)")
            params.extend([like, like, like])
        if workspace:
            clauses.append("workspace = ?")
            params.append(workspace)
        for tag in tags or []:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")
        params.append(limit)
        rows = db.conn.execute(
            f"SELECT *, 0 AS score FROM memories WHERE {' AND '.join(clauses)} ORDER BY event_time DESC, ingest_time DESC LIMIT ?",
            params,
        ).fetchall()
        if query and not db.state.fts5_available:
            warnings.append("Using LIKE/keyword search because sqlite-vec and FTS5 are unavailable.")
    if query and not rows:
        clauses = ["status != 'deleted'"]
        params = []
        if workspace:
            clauses.append("workspace = ?")
            params.append(workspace)
        for tag in tags or []:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")
        params.append(limit)
        rows = db.conn.execute(
            f"SELECT *, 0 AS score FROM memories WHERE {' AND '.join(clauses)} ORDER BY ingest_time DESC, event_time DESC LIMIT ?",
            params,
        ).fetchall()
        if rows:
            warnings.append(
                "No direct memory match. Returning recent memories from this workspace; refine keywords, try memory_recent, or compare candidates before reading source files."
            )
    return [row_to_dict(row) for row in rows], warnings
