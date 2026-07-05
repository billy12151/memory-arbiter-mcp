from __future__ import annotations

from typing import Any, Optional, Tuple

from .db import MemoryDB, row_to_dict


import re

_CJK_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]"
)


def _is_cjk_token(token: str) -> bool:
    return bool(_CJK_RE.search(token))


def _split_cjk_token(token: str) -> list[str]:
    """Split a CJK run into overlapping 3-character trigrams (unquoted).

    The FTS5 table uses ``tokenize='trigram'``, which indexes 3-char windows
    and only matches queries that produce at least one trigram. A 2-char
    phrase such as ``"营销"`` matches nothing under trigram indexing, and a
    full-token phrase such as ``"营销交付系统"`` requires every trigram to be
    present contiguously — so an "overspecified" query (extra chars not in the
    document) silently misses.

    Splitting into overlapping 3-grams joined by ``OR`` fixes both: each
    trigram is a bare FTS5 term that matches any document whose trigram set
    contains it, and the OR means shared trigrams still hit even when some
    trigrams of the query are absent from the document. This restores recall
    for Chinese queries without adding a tokenizer dependency.
    """
    cleaned = "".join(c for c in token if _CJK_RE.search(c) or c.isalnum())
    if len(cleaned) < 3:
        # 1-2 char CJK runs cannot form a trigram; emit nothing rather than a
        # 2-char phrase (which is a guaranteed miss under trigram indexing).
        # Caller falls back to LIKE for these via the empty-result path.
        return []
    return [cleaned[i : i + 3] for i in range(len(cleaned) - 2)]


def _quote_phrase(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def _sanitize_fts_query(query: str) -> str:
    """Turn an arbitrary user query into a safe FTS5 MATCH expression.

    FTS5 has its own query grammar where ``. : * " ( ) - + AND OR NOT`` are
    special. A bare query like ``v0.2.1`` raises ``fts5: syntax error near "."``.

    - Non-CJK tokens are wrapped as double-quoted phrases and AND-joined, so
      English/code identifiers keep their precision.
    - CJK tokens are split into overlapping trigrams (unquoted) joined by OR.
      The trigram tokenizer only matches queries that produce ≥3-char tokens,
      and a strict phrase over CJK silently misses when the query is even
      slightly overspecified — OR over shared trigrams restores recall.

    A CJK token shorter than 3 characters cannot form a trigram and is
    dropped from the FTS5 expression; the surrounding AND will then collapse
    and the caller's LIKE fallback handles it.
    """
    tokens = [tok for tok in query.split() if tok]
    if not tokens:
        return ""
    groups: list[str] = []
    for tok in tokens:
        if _is_cjk_token(tok):
            trigrams = _split_cjk_token(tok)
            if trigrams:
                groups.append("(" + " OR ".join(trigrams) + ")")
        else:
            groups.append(_quote_phrase(tok))
    return " AND ".join(groups)


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
        # Rank the recent-memory fallback by trustworthiness, not just recency.
        # A user_confirmed+locked memory is a higher-value hit than an
        # agent_generated+normal one even when the latter is newer — without
        # this, daily agent chatter buries authoritative records.
        rows = db.conn.execute(
            f"""SELECT *, 0 AS score FROM memories
                WHERE {' AND '.join(clauses)}
                ORDER BY
                  CASE protection_level
                    WHEN 'locked' THEN 0
                    WHEN 'protected' THEN 1
                    ELSE 2
                  END,
                  CASE source_type
                    WHEN 'user_confirmed' THEN 0
                    WHEN 'document_extracted' THEN 1
                    ELSE 2
                  END,
                  confidence DESC,
                  ingest_time DESC,
                  event_time DESC
                LIMIT ?""",
            params,
        ).fetchall()
        if rows:
            warnings.append(
                "No direct memory match. Returning recent memories from this workspace; refine keywords, try memory_recent, or compare candidates before reading source files."
            )
    return [row_to_dict(row) for row in rows], warnings
