from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from .anchors import (
    Anchor,
    classify_match_level,
    extract_anchors,
    score_anchor_overlap,
)
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


def _get_ranking_mode() -> str:
    """v0.3.0: read ranking mode from env.

    - bm25   : legacy v0.2.6 ordering (single FTS, bm25 sort)
    - hybrid : wide-recall candidate pool + soft rerank (the new default)

    Unknown values fall back to hybrid. (A third "shadow" mode exists only on
    the dev/shadow-mode branch for local A/B evaluation; it is not part of the
    published package.)
    """
    mode = (os.environ.get("MEMORY_ARBITER_RANKING_MODE") or "hybrid").lower()
    if mode not in ("bm25", "hybrid"):
        mode = "hybrid"
    return mode


# ---- Soft-rerank scoring constants (r4 §7, §8) --------------------------
# These are deliberately conservative initial values. Per r4 risk-5, we only
# tune 1-2 of these based on A/B; the rest stay fixed.
_SUBJECT_SCORE_CAP = 10.0       # r4 §8.2.1: subject score cannot grow unbounded
_TAGS_SCORE_CAP = 7.0           # r4 §8.2.1: tags score cannot grow unbounded
_CONTENT_SCORE_CAP = 3.0        # content is weak signal, capped low
_TRUST_BONUS_USER_CONFIRMED = 0.5   # r4 §7: trust is *small* bonus, not override
_TRUST_BONUS_DOCUMENT_EXTRACTED = 0.3
_TRUST_BONUS_DEFAULT = 0.0
_LONG_CONTENT_PENALTY = 1.5     # r4 §8.4: applied only under 3 conditions
_CONTENT_ONLY_PENALTY = 2.0     # r4 §8.3: subject/tags miss + content hits

# subject/tags match-level weights (after capping)
_SUBJECT_STRONG_WEIGHT = 10.0
_SUBJECT_MEDIUM_WEIGHT = 6.0
_SUBJECT_WEAK_WEIGHT = 2.0
_TAGS_STRONG_WEIGHT = 7.0
_TAGS_MEDIUM_WEIGHT = 4.0
_TAGS_WEAK_WEIGHT = 1.5


def _trust_bonus(record: dict[str, Any]) -> float:
    """Small, capped trust bonus — never enough to override relevance."""
    source = record.get("source_type") or ""
    protection = record.get("protection_level") or ""
    if source == "user_confirmed" or protection == "locked":
        return _TRUST_BONUS_USER_CONFIRMED
    if source == "document_extracted":
        return _TRUST_BONUS_DOCUMENT_EXTRACTED
    return _TRUST_BONUS_DEFAULT


def _score_surface(
    query_anchors: list[Anchor],
    surface_text: str,
    strong_weight: float,
    medium_weight: float,
    weak_weight: float,
    cap: float,
    query_lower: str,
) -> tuple[float, str]:
    """Score a single surface (subject or tags) against the query.

    Returns (score, match_level). Strong = direct contiguous substring hit
    (checked before anchors); otherwise use anchor overlap classification.
    Score is capped per r4 §8.2.1.
    """
    if not surface_text:
        return 0.0, "none"
    surface_lower = surface_text.lower()
    # Strong: query's main phrase is a contiguous substring of the surface.
    # We check the raw query (not anchors) because substring is a stronger
    # signal than anchor overlap.
    if query_lower and query_lower in surface_lower:
        return min(strong_weight, cap), "strong"
    # Fall back to anchor overlap.
    surface_anchors = extract_anchors(surface_text)
    matches = score_anchor_overlap(query_anchors, surface_anchors)
    level = classify_match_level(query_anchors, matches)
    if level == "medium":
        return min(medium_weight, cap), level
    if level == "weak":
        return min(weak_weight, cap), level
    return 0.0, level


def _soft_rerank(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply soft-rerank to a wide-recall candidate pool.

    Adds debug fields (_subject_level, _tag_level, _match_reason, _ranking_notes)
    to each row but does NOT mutate original fields. Returns new list sorted
    by final_score descending.
    """
    if not candidates:
        return []
    query = (query or "").strip()
    query_lower = query.lower()
    query_anchors = extract_anchors(query) if query else []

    scored: list[tuple[float, dict[str, Any]]] = []
    for rec in candidates:
        subject = rec.get("subject") or ""
        tags_raw = rec.get("tags") or "[]"
        # tags field is JSON-encoded list in DB; parse for surface scoring
        try:
            import json as _json
            tags_list = _json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        except Exception:
            tags_list = []
        tags_text = " ".join(str(t) for t in tags_list) if tags_list else ""
        content = rec.get("content") or ""

        # Score each surface (subject > tags > content), all capped.
        subject_score, subject_level = _score_surface(
            query_anchors, subject,
            _SUBJECT_STRONG_WEIGHT, _SUBJECT_MEDIUM_WEIGHT, _SUBJECT_WEAK_WEIGHT,
            _SUBJECT_SCORE_CAP, query_lower,
        )
        tag_score, tag_level = _score_surface(
            query_anchors, tags_text,
            _TAGS_STRONG_WEIGHT, _TAGS_MEDIUM_WEIGHT, _TAGS_WEAK_WEIGHT,
            _TAGS_SCORE_CAP, query_lower,
        )
        # Content: cheap signal — substring check on lowercased text.
        content_hit = bool(query_lower) and query_lower in content.lower()
        # Also count anchor hits in content for a weak content_score signal.
        content_score = 0.0
        if content_hit:
            content_score = _CONTENT_SCORE_CAP
        elif query_anchors and content:
            content_anchors = extract_anchors(content)
            content_matches = score_anchor_overlap(query_anchors, content_anchors)
            cm = content_matches.get("_summary")
            if cm and cm.total_hits >= 2:
                content_score = min(_CONTENT_SCORE_CAP * 0.5, _CONTENT_SCORE_CAP)

        relevance = subject_score + tag_score + content_score

        # content-only penalty (r4 §8.3): if subject/tags didn't even reach
        # weak, and content hit, treat as "incidental mention" — drop score.
        subject_tags_miss = subject_level in ("none",) and tag_level in ("none",)
        if subject_tags_miss and content_score > 0:
            relevance -= _CONTENT_ONLY_PENALTY

        # long-content penalty (r4 §8.4): three conditions must ALL hold:
        # 1. subject/tags no strong or medium hit
        # 2. hits mainly from content
        # 3. content is long
        subject_tags_weak = subject_level in ("none", "weak") and tag_level in ("none", "weak")
        content_long = len(content) > 2000
        if subject_tags_weak and content_long and content_score > 0:
            relevance -= _LONG_CONTENT_PENALTY

        trust = _trust_bonus(rec)
        # Superseded always sinks below active regardless of score (r4 carries
        # this forward from v0.2.6).
        superseded_sink = 1 if rec.get("status") == "superseded" else 0
        final_score = relevance + trust - (superseded_sink * 1000.0)

        # Build debug info (only returned when debug_ranking=True).
        notes: list[str] = []
        match_reason = "subject_or_tag_match"
        if subject_tags_miss and content_score > 0:
            match_reason = "content_only_match"
            notes.append("query terms matched content but not subject/tags")
        if subject_tags_weak and content_long and content_score > 0:
            notes.append("long content penalty applied")
        if superseded_sink:
            notes.append("superseded: sunk below active")

        rec_copy = dict(rec)
        rec_copy["_final_score"] = final_score
        rec_copy["_subject_level"] = subject_level
        rec_copy["_tag_level"] = tag_level
        rec_copy["_match_reason"] = match_reason
        rec_copy["_ranking_notes"] = notes
        rec_copy["_subject_score"] = subject_score
        rec_copy["_tag_score"] = tag_score
        rec_copy["_content_score"] = content_score
        scored.append((final_score, rec_copy))

    # Sort by final_score desc; tiebreak by ingest_time desc for stability.
    scored.sort(key=lambda x: (-x[0], x[1].get("ingest_time", "")), reverse=False)
    # Re-flip the sort: we want highest score first, so use reverse on score.
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored]


def _wide_recall(
    db: MemoryDB,
    query: str,
    workspace: Optional[str],
    tags: Optional[list[str]],
    status_clause_m: str,
    like_status_clause: str,
    pool_cap: int = 50,
    content_like_fallback: bool = True,
) -> list[dict[str, Any]]:
    """v0.3.0 wide recall: merge multiple retrieval channels into a candidate pool.

    Channels (per r4 §6):
      1. FTS top N (main)
      2. FTS OR-query top N (loosened — query tokens OR'd rather than AND'd)
      3. subject/tags LIKE (precise surface recall)
      4. content LIKE — only if pool not yet full, with ≥2 anchor hits, capped

    Returns dedup'd candidate pool (list of dict rows). Each row already has
    its raw fields; soft-rerank will add scoring fields.
    """
    if db.conn is None or not query:
        return []
    pool: dict[int, dict[str, Any]] = {}

    # Channel 1+2: FTS main + OR. _sanitize_fts_query already OR-joins CJK
    # trigrams; for the OR channel we additionally try a loosened query that
    # only requires any single trigram/token to hit.
    if db.state.fts5_available:
        per_channel_cap = max(pool_cap, 30)
        # Main FTS query (AND across token groups).
        fts_main = _sanitize_fts_query(query)
        if fts_main:
            sql = f"""
                SELECT m.*, bm25(memories_fts) AS score
                FROM memories_fts
                JOIN memories m ON memories_fts.rowid = m.id
                WHERE memories_fts MATCH ? AND {status_clause_m}
            """
            params: list[Any] = [fts_main]
            if workspace:
                sql += " AND m.workspace = ?"
                params.append(workspace)
            sql += f" ORDER BY CASE m.status WHEN 'superseded' THEN 1 ELSE 0 END, score LIMIT ?"
            params.append(per_channel_cap)
            try:
                for row in db.conn.execute(sql, params).fetchall():
                    d = row_to_dict(row)
                    pool[d["id"]] = d
            except Exception:
                pass
        # OR channel: only if main didn't fill the pool. This catches the
        # "query was overspecified" case where AND'd trigrams miss.
        if len(pool) < pool_cap:
            fts_or = _sanitize_fts_query_or(query)
            if fts_or and fts_or != fts_main:
                sql = f"""
                    SELECT m.*, bm25(memories_fts) AS score
                    FROM memories_fts
                    JOIN memories m ON memories_fts.rowid = m.id
                    WHERE memories_fts MATCH ? AND {status_clause_m}
                """
                params = [fts_or]
                if workspace:
                    sql += " AND m.workspace = ?"
                    params.append(workspace)
                sql += f" ORDER BY CASE m.status WHEN 'superseded' THEN 1 ELSE 0 END, score LIMIT ?"
                params.append(per_channel_cap)
                try:
                    for row in db.conn.execute(sql, params).fetchall():
                        d = row_to_dict(row)
                        if d["id"] not in pool:
                            pool[d["id"]] = d
                except Exception:
                    pass

    # Channel 3: subject/tags LIKE — precise surface recall.
    if len(pool) < pool_cap:
        like_q = f"%{query}%"
        clauses = [like_status_clause, "(subject LIKE ? OR tags LIKE ?)"]
        params = [like_q, like_q]
        if workspace:
            clauses.append("workspace = ?")
            params.append(workspace)
        for tag in tags or []:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")
        params.append(pool_cap)
        sql = f"""SELECT *, 0 AS score FROM memories
                  WHERE {' AND '.join(clauses)}
                  ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
                           ingest_time DESC LIMIT ?"""
        for row in db.conn.execute(sql, params).fetchall():
            d = row_to_dict(row)
            if d["id"] not in pool:
                pool[d["id"]] = d

    # Channel 4: content LIKE — limited补漏. Requires ≥2 query anchors hit
    # (r4 §6.1) and is capped at 5-10 to avoid noise explosion.
    if content_like_fallback and len(pool) < pool_cap:
        # Only run if query has at least 2 anchors — otherwise the ≥2-anchor
        # gate can never be satisfied and we save the scan.
        q_anchors = extract_anchors(query)
        if len(q_anchors) >= 2:
            like_q = f"%{query}%"
            clauses = [like_status_clause, "content LIKE ?"]
            params = [like_q]
            if workspace:
                clauses.append("workspace = ?")
                params.append(workspace)
            for tag in tags or []:
                clauses.append("tags LIKE ?")
                params.append(f"%{tag}%")
            params.append(10)  # cap at 10 content-LIKE补漏
            sql = f"""SELECT *, 0 AS score FROM memories
                      WHERE {' AND '.join(clauses)}
                      ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
                               ingest_time DESC LIMIT ?"""
            added = 0
            for row in db.conn.execute(sql, params).fetchall():
                d = row_to_dict(row)
                if d["id"] not in pool:
                    # Mark as content_only candidate for soft-rerank awareness.
                    d["_content_only_candidate"] = True
                    pool[d["id"]] = d
                    added += 1
                    if added >= 10 or len(pool) >= pool_cap:
                        break

    return list(pool.values())[:pool_cap]


def _sanitize_fts_query_or(query: str) -> str:
    """Build a loosened FTS5 query that OR's all token groups together.

    Used for the wide-recall OR channel — catches documents that share any
    one trigram/token with the query, even if they don't satisfy the AND.
    """
    tokens = [tok for tok in query.split() if tok]
    if not tokens:
        return ""
    parts: list[str] = []
    for tok in tokens:
        if _is_cjk_token(tok):
            trigrams = _split_cjk_token(tok)
            parts.extend(trigrams)
        else:
            parts.append(_quote_phrase(tok))
    if not parts:
        return ""
    return " OR ".join(parts)


def search_memories(db: MemoryDB, query: str, workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, include_superseded: bool = False, debug_ranking: bool = False) -> Tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if db.conn is None:
        return [], ["SQLite unavailable; search cannot read JSONL backup in MVP."]
    limit = max(1, min(int(limit), 100))
    query = (query or "").strip()
    mode = _get_ranking_mode()
    # Superseded memories are excluded by default: a superseded record is, by
    # definition, no longer authoritative and only pollutes results (release
    # chatter, superseded specs, etc.). Audit/history walkthroughs opt back in
    # via include_superseded=True — and even then the ORDER BY clause below
    # keeps superseded rows below every active row, so they never drown out
    # current truth regardless of bm25 score.
    status_clause = "m.status != 'deleted'" if include_superseded else "(m.status != 'deleted' AND m.status != 'superseded')"
    like_status_clause = "status != 'deleted'" if include_superseded else "(status != 'deleted' AND status != 'superseded')"

    # === bm25 mode: legacy v0.2.6 single-FTS ordering ===
    if mode == "bm25":
        return _search_bm25(db, query, workspace, tags, limit, status_clause, like_status_clause, warnings, debug_ranking)

    # === hybrid mode: wide recall + soft rerank ===
    if not query:
        # Empty query: same as v0.2.6 recent fallback (no reranking needed).
        return _recent_fallback(db, workspace, tags, limit, like_status_clause, warnings)

    pool = _wide_recall(db, query, workspace, tags, status_clause, like_status_clause)
    if not pool:
        # No direct hits: fall back to recent memories (r4 §4.2 safety net).
        return _recent_fallback(db, workspace, tags, limit, like_status_clause, warnings)

    reranked = _soft_rerank(query, pool)
    # Slice to limit.
    reranked = reranked[:limit]

    # hybrid mode: strip debug fields unless explicitly requested.
    if not debug_ranking:
        for r in reranked:
            for k in list(r.keys()):
                if k.startswith("_"):
                    r.pop(k, None)
    return reranked, warnings


def _search_bm25(
    db: MemoryDB,
    query: str,
    workspace: Optional[str],
    tags: Optional[list[str]],
    limit: int,
    status_clause_m: str,
    like_status_clause: str,
    warnings: list[str],
    debug_ranking: bool,
) -> Tuple[list[dict[str, Any]], list[str]]:
    """Legacy v0.2.6 bm25 ordering. Kept for RANKING_MODE=bm25 fallback."""
    rows = []
    if db.state.fts5_available and query:
        sql = f"""
            SELECT m.*, bm25(memories_fts) AS score
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ? AND {status_clause_m}
        """
        params: list[Any] = [_sanitize_fts_query(query)]
        if workspace:
            sql += " AND m.workspace = ?"
            params.append(workspace)
        sql += " ORDER BY CASE m.status WHEN 'superseded' THEN 1 ELSE 0 END, score LIMIT ?"
        params.append(limit)
        try:
            rows = db.conn.execute(sql, params).fetchall()
        except Exception as exc:
            warnings.append(f"FTS5 query failed: {exc}. Falling back to LIKE search.")
            rows = []
    if not rows:
        like = f"%{query}%"
        clauses = [like_status_clause]
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
            f"SELECT *, 0 AS score FROM memories WHERE {' AND '.join(clauses)} ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END, event_time DESC, ingest_time DESC LIMIT ?",
            params,
        ).fetchall()
        if query and not db.state.fts5_available:
            warnings.append("Using LIKE/keyword search because sqlite-vec and FTS5 are unavailable.")
    if query and not rows:
        return _recent_fallback(db, workspace, tags, limit, like_status_clause, warnings)
    out = [row_to_dict(row) for row in rows]
    if not debug_ranking:
        for r in out:
            r.pop("score", None)
    return out, warnings


def _recent_fallback(
    db: MemoryDB,
    workspace: Optional[str],
    tags: Optional[list[str]],
    limit: int,
    like_status_clause: str,
    warnings: list[str],
) -> Tuple[list[dict[str, Any]], list[str]]:
    """Recent-memory fallback when no direct match found (r4 §4.2 safety net)."""
    clauses = [like_status_clause]
    params: list[Any] = []
    if workspace:
        clauses.append("workspace = ?")
        params.append(workspace)
    for tag in tags or []:
        clauses.append("tags LIKE ?")
        params.append(f"%{tag}%")
    params.append(limit)
    rows = db.conn.execute(
        f"""SELECT *, 0 AS score FROM memories
            WHERE {' AND '.join(clauses)}
            ORDER BY
              CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
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
