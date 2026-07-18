from __future__ import annotations

import os
from datetime import datetime, timezone
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
# v0.3.1: floor score for vec0-recalled candidates. These candidates often
# have zero lexical overlap with the query (that's the whole point of
# semantic recall), so without a floor they'd rank last despite being
# semantically relevant. Set just below CONTENT_SCORE_CAP so a vec candidate
# beats content-only noise but never beats a real subject/tags hit.
_VEC_FLOOR_SCORE = 2.5

# v0.6.3: Channel 6 section-vec KNN multiplier. One memory's multiple
# sections can occupy KNN slots; over-fetch by this factor so dedup still
# leaves enough unique memories to fill the pool gap.
_SECTION_KNN_K_MULTIPLIER = 3

# subject/tags match-level weights (after capping)
_SUBJECT_STRONG_WEIGHT = 10.0
_SUBJECT_MEDIUM_WEIGHT = 6.0
_SUBJECT_WEAK_WEIGHT = 2.0
_TAGS_STRONG_WEIGHT = 7.0
_TAGS_MEDIUM_WEIGHT = 4.0
_TAGS_WEAK_WEIGHT = 1.5

# v0.4.1: recency bonus tiers. Capped low so recency only breaks ties between
# equally-relevant records — it must never override a subject/tags hit. The
# smallest subject-medium weight is 6.0, so a 0.30 max bonus is ~5% of that:
# enough to lift "release v0.4.0" above "release v0.2.1" when both cap out at
# the same surface score (the exact failure that buried id=108 under id=27),
# but never enough to promote a content-only match over a subject match.
_RECENCY_BONUS_7D = 0.30
_RECENCY_BONUS_30D = 0.15
_RECENCY_BONUS_90D = 0.05
_RECENCY_BONUS_DEFAULT = 0.0
_RECENCY_THRESHOLDS = (
    (7 * 86400, _RECENCY_BONUS_7D),
    (30 * 86400, _RECENCY_BONUS_30D),
    (90 * 86400, _RECENCY_BONUS_90D),
)


def _trust_bonus(record: dict[str, Any]) -> float:
    """Small, capped trust bonus — never enough to override relevance."""
    source = record.get("source_type") or ""
    protection = record.get("protection_level") or ""
    if source == "user_confirmed" or protection == "locked":
        return _TRUST_BONUS_USER_CONFIRMED
    if source == "document_extracted":
        return _TRUST_BONUS_DOCUMENT_EXTRACTED
    return _TRUST_BONUS_DEFAULT


def _parse_ingest_time(record: dict[str, Any]) -> Optional[datetime]:
    """Parse ingest_time as a timezone-aware UTC datetime, if possible."""
    raw = record.get("ingest_time") or ""
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00")
        ts = datetime.fromisoformat(normalized)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _ingest_sort_key(record: dict[str, Any]) -> float:
    """Chronological sort key for ingest_time; invalid timestamps sort last."""
    ts = _parse_ingest_time(record)
    if ts is None:
        return float("-inf")
    return ts.timestamp()


def _recency_bonus(record: dict[str, Any], now: Optional[datetime] = None) -> float:
    """Tiered recency bonus based on ingest_time, never enough to override relevance.

    Uses ingest_time (when the memory entered the store) rather than event_time
    (when the underlying fact happened). "Find the latest release notes" cares
    about when the record was logged, not when the release shipped.

    Degrades gracefully: unparseable or future timestamps return 0 bonus
    rather than raising — a bad timestamp must never break search.
    """
    ts = _parse_ingest_time(record)
    if ts is None:
        return _RECENCY_BONUS_DEFAULT
    reference = now or datetime.now(timezone.utc)
    age_seconds = (reference - ts).total_seconds()
    if age_seconds < 0:
        # Clock skew or future-dated record; don't penalize, don't reward.
        return _RECENCY_BONUS_DEFAULT
    for threshold, bonus in _RECENCY_THRESHOLDS:
        if age_seconds <= threshold:
            return bonus
    return _RECENCY_BONUS_DEFAULT


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


# ---- v0.7.3: tag-specific scoring (design §2) --------------------------
# _score_surface treats subject and tags the same way — both go through the
# "is the whole query a contiguous substring?" strong check. That's right for
# subject (a natural-language sentence) but wrong for tags (a discrete label
# set that almost never concatenates into the exact query string). The result
# was that tags could only ever reach medium (4.0), never strong (7.0), even
# when every query token was an exact tag — see id=206 / id=210.
#
# _score_tags_surface replaces _score_surface for the tags field only. It
# scores by *semantic token overlap*: split the query on whitespace, normalize
# both sides (strip v-prefix on version-like tokens), and match each query
# token against the tag list. ASCII tokens match by equality (no substring —
# "v0.7" must not match tag "v0.7.0"); pure-CJK tokens match by prefix/suffix
# substring only (middle substrings would let bigram-artifact tags like "版历"
# leak through). See design doc §2.3-§2.6.

def _normalize_token_for_tag_match(token: str) -> str:
    """Normalize a token for tag-level matching.

    Applied to BOTH query tokens and tags (bidirectional — review_1 漏洞 2).
    Strips a leading ``v`` only when it prefixes a version-like token
    (``v0.7.2`` → ``0.7.2``) so ``query="v0.7.2"`` matches ``tag="0.7.2"``.
    Words like ``vue`` are left alone (v not followed by a digit).
    """
    s = (token or "").lower().strip()
    if len(s) > 1 and s[0] == "v" and s[1].isdigit():
        s = s[1:]
    return s


def _cjk_substring_match(tag_norm: str, query_token_norm: str) -> bool:
    """CJK substring match — prefix/suffix only, never middle.

    - prefix: tag ``发版`` matches query token ``发版历史`` (tag is query's prefix)
    - suffix: tag ``历史`` matches query token ``发版历史`` (tag is query's suffix)
    - middle: tag ``版历`` does NOT match query token ``发版历史`` (prevents
      bigram-artifact tags created by anchor slicing from leaking through)

    The ``len >= 2`` gate on both sides also excludes single-char tags, which
    would over-match (design §8 risk 4 / S2: actual risk is under-match of
    single-char tags, accepted).
    """
    if tag_norm == query_token_norm:
        return True
    if len(tag_norm) >= 2 and len(query_token_norm) >= 2:
        return query_token_norm.startswith(tag_norm) or query_token_norm.endswith(tag_norm)
    return False


def _is_pure_cjk_token(token: str) -> bool:
    """A token is "pure CJK" if it contains NO ASCII alphanumeric chars.

    Used by _score_tags_surface to pick the match path per query token:
      - pure CJK  → prefix/suffix substring match (发版 / 发版历史)
      - otherwise → equality match (v0.7.2 / memory / 0.7.2发版 mixed)

    Note this is the OPPOSITE of the existing ``_is_cjk_token`` (which returns
    True if a token contains ANY CJK char, and serves the FTS trigram path).
    A mixed token like ``0.7.2发版`` is _is_cjk_token=True but _is_pure_cjk=False,
    so it correctly takes the equality path (design S1/E2/M3). Do not merge
    these two helpers.
    """
    return not any(c.isascii() and c.isalnum() for c in token)


def _score_tags_surface(
    query: str,
    tags_list: list[str],
    strong_weight: float,
    medium_weight: float,
    weak_weight: float,
    cap: float,
) -> tuple[float, str, dict]:
    """Score tags by semantic token overlap with the query (v0.7.3).

    Algorithm (design §2.3):
      1. Split query on whitespace into semantic tokens.
      2. Normalize each token (_normalize_token_for_tag_match), applied to
         BOTH query tokens and tags.
      3. For each normalized query token, match against the normalized tag set:
         - pure-CJK token → _cjk_substring_match (prefix/suffix only)
         - otherwise      → equality only (ASCII/mixed tokens)
      4. ratio = matched_query_tokens / total_query_tokens.
         - 1.0           → strong (min(strong_weight, cap))
         - 0.5 <= r < 1  → medium
         - 0   < r < 0.5 → weak
         - 0             → none

    Returns (score, level, debug) where debug has keys
    total / matched / ratio for the debug_ranking fields.
    """
    if not tags_list:
        return 0.0, "none", {"total": 0, "matched": 0, "ratio": 0.0}

    query_tokens = [t for t in (query or "").split() if t]
    if not query_tokens:
        return 0.0, "none", {"total": 0, "matched": 0, "ratio": 0.0}

    tags_norm = [_normalize_token_for_tag_match(str(t)) for t in tags_list]
    tags_norm_set = set(tags_norm)

    matched = 0
    for raw_token in query_tokens:
        token_norm = _normalize_token_for_tag_match(raw_token)
        if not token_norm:
            continue
        if _is_pure_cjk_token(token_norm):
            hit = any(_cjk_substring_match(tn, token_norm) for tn in tags_norm_set)
        else:
            hit = token_norm in tags_norm_set
        if hit:
            matched += 1

    total = len(query_tokens)
    ratio = matched / total if total else 0.0
    if ratio >= 1.0:
        level = "strong"
        score = min(strong_weight, cap)
    elif ratio >= 0.5:
        level = "medium"
        score = min(medium_weight, cap)
    elif ratio > 0:
        level = "weak"
        score = min(weak_weight, cap)
    else:
        level = "none"
        score = 0.0
    return score, level, {"total": total, "matched": matched, "ratio": ratio}


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
        tag_score, tag_level, tag_debug = _score_tags_surface(
            query, tags_list,
            _TAGS_STRONG_WEIGHT, _TAGS_MEDIUM_WEIGHT, _TAGS_WEAK_WEIGHT,
            _TAGS_SCORE_CAP,
        ) if tags_list else (0.0, "none", {"total": 0, "matched": 0, "ratio": 0.0})
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
        # v0.6.3: exempt split-active memories — their length is structural
        # (a long doc legitimately split into sections), not "附带提及" noise.
        subject_tags_weak = subject_level in ("none", "weak") and tag_level in ("none", "weak")
        content_long = len(content) > 2000
        is_split_active = rec.get("split_status") == "active"
        if subject_tags_weak and content_long and content_score > 0 and not is_split_active:
            relevance -= _LONG_CONTENT_PENALTY

        # v0.3.1: vec0-recalled candidates. If this candidate came from the
        # semantic channel and lexical relevance is below the floor, raise it
        # to the floor. The floor sits just below content-score cap, so a vec
        # candidate beats content-only noise but loses to any subject/tags hit.
        if rec.get("_vec_candidate") and relevance < _VEC_FLOOR_SCORE:
            relevance = _VEC_FLOOR_SCORE

        trust = _trust_bonus(rec)
        recency = _recency_bonus(rec)
        # Superseded always sinks below active regardless of score (r4 carries
        # this forward from v0.2.6).
        superseded_sink = 1 if rec.get("status") == "superseded" else 0
        final_score = relevance + trust + recency - (superseded_sink * 1000.0)

        # Build debug info (only returned when debug_ranking=True).
        notes: list[str] = []
        match_reason = "subject_or_tag_match"
        if subject_tags_miss and content_score > 0:
            match_reason = "content_only_match"
            notes.append("query terms matched content but not subject/tags")
        if subject_tags_weak and content_long and content_score > 0 and not is_split_active:
            notes.append("long content penalty applied")
        if superseded_sink:
            notes.append("superseded: sunk below active")
        if rec.get("_vec_candidate"):
            if match_reason == "subject_or_tag_match":
                match_reason = "vec_recall"
            notes.append("v0.3.1: semantic recall candidate, floor score applied")
        # v0.6.3: distinguish section-vec (Channel 6) from memory-vec (Channel 5)
        # in debug output. Both set _vec_candidate for the floor; this note
        # disambiguates the recall source.
        if rec.get("_section_vec_candidate"):
            notes.append("section-vec recall candidate (Channel 6)")

        rec_copy = dict(rec)
        rec_copy["_final_score"] = final_score
        rec_copy["_subject_level"] = subject_level
        rec_copy["_tag_level"] = tag_level
        rec_copy["_match_reason"] = match_reason
        rec_copy["_ranking_notes"] = notes
        rec_copy["_subject_score"] = subject_score
        rec_copy["_tag_score"] = tag_score
        rec_copy["_tag_query_tokens"] = tag_debug.get("total", 0)
        rec_copy["_tag_matched_tokens"] = tag_debug.get("matched", 0)
        rec_copy["_tag_match_ratio"] = tag_debug.get("ratio", 0.0)
        rec_copy["_content_score"] = content_score
        rec_copy["_recency_bonus"] = recency
        rec_copy["_trust_bonus"] = trust
        scored.append((final_score, rec_copy))

    # Sort by final_score desc; tiebreak by ingest_time desc (newest first).
    # The previous implementation ran two sorts — first ascending on
    # ingest_time then stable descending on score — which left ties ordered
    # oldest-first (SQLite rowid order). For "find the latest X" queries
    # this buried the newest record, e.g. querying release notes returned
    # v0.2.x ahead of v0.4.0 because every release-summary record hit the
    # same subject/tags cap. One sort, score-desc then time-desc, fixes it.
    scored.sort(key=lambda x: (x[0], _ingest_sort_key(x[1])), reverse=True)
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
    query_embedding: Optional[list[float]] = None,
    content_like_cap: int = 30,
) -> list[dict[str, Any]]:
    """v0.3.0 wide recall: merge multiple retrieval channels into a candidate pool.

    Channels (per r4 §6):
      1. FTS top N (main)
      2. FTS OR-query top N (loosened — query tokens OR'd rather than AND'd)
      3. subject/tags LIKE (precise surface recall)
      4. content LIKE — only if pool not yet full, with ≥2 anchor hits, capped
      5. vec0 KNN — optional (v0.3.1), only when query_embedding provided and
         sqlite-vec available. Catches semantically similar but lexically
         dissimilar memories. Candidates are flagged so soft-rerank can give
         them a floor score (the query text didn't literally match anything).

    Returns dedup'd candidate pool (list of dict rows). Each row already has
    its raw fields; soft-rerank will add scoring fields.
    """
    if not db.db_available or not query:
        return []
    pool: dict[int, dict[str, Any]] = {}
    conn = db._new_connection()

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
                for row in conn.execute(sql, params).fetchall():
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
                    for row in conn.execute(sql, params).fetchall():
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
        for row in conn.execute(sql, params).fetchall():
            d = row_to_dict(row)
            if d["id"] not in pool:
                pool[d["id"]] = d

    # Channel 4: content LIKE — a limited gap-filler. Requires ≥2 query anchors hit
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
            params.append(content_like_cap)  # cap content-LIKE gap-fill (configurable via MEMORY_ARBITER_CONTENT_LIKE_CAP)
            sql = f"""SELECT *, 0 AS score FROM memories
                      WHERE {' AND '.join(clauses)}
                      ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
                               ingest_time DESC LIMIT ?"""
            added = 0
            for row in conn.execute(sql, params).fetchall():
                d = row_to_dict(row)
                if d["id"] not in pool:
                    # Mark as content_only candidate for soft-rerank awareness.
                    d["_content_only_candidate"] = True
                    pool[d["id"]] = d
                    added += 1
                    if added >= content_like_cap or len(pool) >= pool_cap:
                        break
    conn.close()

    # Channel 5 (v0.3.1): vec0 KNN — optional semantic recall. Only runs when
    # the caller supplied a query_embedding AND sqlite-vec is available. This
    # is the only channel that can surface memories whose surface text shares
    # no trigrams/tokens with the query. Candidates are flagged so soft-rerank
    # knows they came from vectors (their lexical score will be 0).
    vec_state = db.get_vec_index_state().get("state")
    if (
        query_embedding
        and db.state.sqlite_vec_available
        and vec_state in {"ready", "unmanaged"}
        and len(pool) < pool_cap
    ):
        knn_rows = db.vec_knn(query_embedding, k=max(pool_cap - len(pool), 10))
        for row in knn_rows:
            # Apply the same workspace filter the other channels use.
            if workspace and row.get("workspace") != workspace:
                continue
            # Respect status filtering — vec0 rows are joined from memories,
            # but we still need to honour the active/superseded gate.
            status = row.get("status")
            if status == "deleted":
                continue
            if status == "superseded" and "superseded" in like_status_clause:
                # like_status_clause filters superseded by default; match that.
                continue
            rid = row.get("id")
            if rid is None or rid in pool:
                continue
            d = dict(row)
            d["_vec_candidate"] = True
            pool[rid] = d

    # Channel 6 (v0.6.3): section-vec KNN — recall memories via their section
    # vectors. Catches the "query semantically matches a late chapter that the
    # memory-level embedding (truncated to ~3600 chars) never saw" case. Same
    # gate as Channel 5; pure gap-filler so existing channels are untouched.
    if (
        query_embedding
        and db.state.sqlite_vec_available
        and vec_state in {"ready", "unmanaged"}
        and len(pool) < pool_cap
    ):
        need = max(pool_cap - len(pool), 10)
        k = need * _SECTION_KNN_K_MULTIPLIER
        sec_rows = db.section_vec_knn(query_embedding, k=k)
        for row in sec_rows:
            # Post-filter: workspace + status (mirror Channel 5's logic).
            if workspace and row.get("workspace") != workspace:
                continue
            status = row.get("status")
            if status == "deleted":
                continue
            if status == "superseded" and "superseded" in like_status_clause:
                continue
            # Only split-active memories have meaningful section vectors.
            if row.get("split_status") != "active":
                continue
            rid = row.get("memory_id")
            if rid is None or rid in pool:
                # Already in pool (recalled by an earlier channel). Do NOT
                # re-add; section-level enhancement is _attach_sections' job.
                continue
            d = {
                "id": rid,
                "workspace": row.get("workspace"),
                "status": row.get("status"),
                "subject": row.get("subject"),
                "tags": row.get("tags"),
                # NOTE: content deliberately omitted (A3). Channel 6 candidates
                # score via vec floor (content_score=0); _attach_sections
                # re-fetches content from current_mem_map for split-active rows.
                "content": "",
                "source_type": row.get("source_type"),
                "confidence": row.get("confidence"),
                "protection_level": row.get("protection_level"),
                "event_time": row.get("event_time"),
                "ingest_time": row.get("ingest_time"),
                "metadata": row.get("metadata"),
                "split_status": row.get("split_status"),
                # Flags for _soft_rerank + debug_ranking.
                "_vec_candidate": True,            # reuse vec floor logic
                "_section_vec_candidate": True,    # debug identity
                "_section_vec_distance": row.get("distance"),
                "_section_vec_section_id": row.get("section_id"),
            }
            pool[rid] = d

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


def search_memories(db: MemoryDB, query: str, workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, include_superseded: bool = False, debug_ranking: bool = False, query_embedding: Optional[list[float]] = None) -> Tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not db.db_available:
        return [], ["SQLite unavailable; search cannot read JSONL backup in MVP."]
    limit = max(1, min(int(limit), 100))
    query = (query or "").strip()
    mode = _get_ranking_mode()
    # v0.3.1: when a query_embedding is supplied but sqlite-vec is not active,
    # warn so the caller knows the semantic channel was silently skipped.
    if query_embedding and not db.state.sqlite_vec_available:
        warnings.append("query_embedding provided but sqlite-vec unavailable; semantic recall skipped.")
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

    pool = _wide_recall(db, query, workspace, tags, status_clause, like_status_clause, query_embedding=query_embedding,
                        pool_cap=getattr(db.settings, "recall_pool_cap", 50),
                        content_like_cap=getattr(db.settings, "content_like_cap", 30))
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
    conn = db._new_connection()
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
            rows = conn.execute(sql, params).fetchall()
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
        rows = conn.execute(
            f"SELECT *, 0 AS score FROM memories WHERE {' AND '.join(clauses)} ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END, event_time DESC, ingest_time DESC LIMIT ?",
            params,
        ).fetchall()
        if query and not db.state.fts5_available:
            warnings.append("Using LIKE/keyword search because sqlite-vec and FTS5 are unavailable.")
    conn.close()
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
    conn = db._new_connection()
    try:
        rows = conn.execute(
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
    finally:
        conn.close()
    if rows:
        warnings.append(
            "No direct memory match. Returning recent memories from this workspace; refine keywords, try memory_recent, or compare candidates before reading source files."
        )
    return [row_to_dict(row) for row in rows], warnings
