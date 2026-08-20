from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Tuple

from .anchors import (
    Anchor,
    classify_match_level,
    extract_anchors,
    score_anchor_overlap,
)
from .db import MemoryDB, row_to_dict

# v0.7.4 (M2): retrieval_mode classifies how the returned rows were produced.
# linked_open_items only triggers on "direct" (a real query hit) — the other
# modes return browse/fallback/empty rows where injecting todos would be noise.
RetrievalMode = Literal[
    "direct",            # FTS/LIKE/evidence channels genuinely matched the query
    "recent_fallback",   # query was non-empty but nothing matched; recent returned
    "recent_browse",     # empty query, no filters — caller is browsing recent
    "empty",             # filters yielded nothing, or pool empty after post-filter
    "unavailable",       # SQLite not available
]

# v0.7.4.1: single source of truth for the recent-fallback warning. The legacy
# bm25 path infers retrieval_mode by sniffing this warning (it has no structured
# mode signal), so the literal must live in ONE place — never inline it.
# Tests match on the prefix substring, so keep the prefix stable.
# Single source: constants.NO_DIRECT_MATCH_PREFIX (Phase 1); re-exported here.
from .constants import NO_DIRECT_MATCH_PREFIX as _NO_DIRECT_MATCH_PREFIX
from .constants import strict_ws


@dataclass
class SearchOutcome:
    """v0.7.4 (M2): structured return for search_memories.

    Replaces the bare (results, warnings, has_more, total_estimate) 4-tuple so
    retrieval_mode can travel with the result without growing into a 5-tuple
    (which would break every tuple-unpacking caller). All callers must use
    attribute access.
    """

    results: list[dict[str, Any]]
    warnings: list[str]
    has_more: bool
    total_estimate: int
    retrieval_mode: RetrievalMode


import re

# Single source: text.CJK_RE_SEARCH (Phase 1). Re-exported here for back-compat.
from .text import CJK_RE_SEARCH as _CJK_RE


def _is_cjk_token(token: str) -> bool:
    return bool(_CJK_RE.search(token))


def _split_cjk_token(token: str) -> list[str]:
    """Split a CJK run into overlapping 3-character trigrams (unquoted).

    Implementation lives in text.split_cjk_token (Phase 1); thin re-export here.
    The FTS5 table uses ``tokenize='trigram'``: OR-joined trigrams restore recall
    for Chinese queries where a strict phrase would silently miss.
    """
    from .text import split_cjk_token
    return split_cjk_token(token)


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
_TAGS_SCORE_CAP = 10.0          # v0.7.3: 从 7.0 提到 10.0（与 subject cap 持平，配套 tag 权重提升）
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
# Reciprocal-rank fusion keeps lexical and evidence channels comparable even
# though BM25 scores and vector distances live on unrelated scales. 60 is the
# conventional RRF damping constant; the multiplier makes fusion meaningful
# beside the existing 0..20 lexical relevance score without overriding a
# strong subject+tag match from a single channel.
_RRF_K = 60.0
_RRF_SCORE_WEIGHT = 300.0

# subject/tags match-level weights (after capping)
_SUBJECT_STRONG_WEIGHT = 10.0
_SUBJECT_MEDIUM_WEIGHT = 6.0
_SUBJECT_WEAK_WEIGHT = 2.0
# v0.7.3: tag 权重从 7.0/4.0/1.5 提到 10.0/6.0/2.0（与 subject 持平）。
# 数据驱动决策（scripts/tune_tag_weights.py，n=2000×5 seed）：tag 是 LLM 主动
# 打的精确分类标签，命中信号比 subject 偶然含字面更可靠（id=210 原始论证）。
# 配合 classify_match_level 的 coverage 0.4→0.6 收紧 subject，让 tag 精确命中
# 的记录（id=206）排到 subject 偶然命中的记录（id=105）之上。详见 id=211。
_TAGS_STRONG_WEIGHT = 10.0
_TAGS_MEDIUM_WEIGHT = 6.0
_TAGS_WEAK_WEIGHT = 2.0

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
    """Parse ingest_time as a timezone-aware UTC datetime, if possible.

    Implementation lives in timeutil.parse_iso8601_utc (Phase 1); thin re-export.
    """
    from .timeutil import parse_iso8601_utc
    return parse_iso8601_utc(record.get("ingest_time"))


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


# v0.9.7: workspace soft-weighting (weak isolation). Same magnitude discipline
# as trust/recency — a small nudge that breaks ties between equally-relevant
# records, never enough to override a subject/tags hit. Same-workspace gets a
# small lift; cross-workspace gets a small penalty. Only applies when the
# caller passes a query workspace AND isolation == "weak".
_WS_BONUS_SAME = 0.30      # ~5% of a subject-medium hit (6.0), like recency max
_WS_PENALTY_CROSS = -0.15  # gentler penalty so cross-ws stays reachable


def _workspace_bonus(record: dict[str, Any], ws_canonical: Optional[str], isolation: str) -> float:
    """Soft workspace nudge for weak isolation. 0 outside weak mode."""
    if isolation != "weak" or not ws_canonical:
        return 0.0
    rec_ws = record.get("workspace_canonical") or record.get("workspace") or ""
    if not rec_ws:
        return 0.0
    return _WS_BONUS_SAME if rec_ws == ws_canonical else _WS_PENALTY_CROSS


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
    """Normalize a token for tag-level matching (query AND tags).

    Implementation lives in text.normalize_token_for_tag_match (Phase 1); thin
    re-export here. Strips a leading ``v`` only when it prefixes a version token.
    """
    from .text import normalize_token_for_tag_match
    return normalize_token_for_tag_match(token)


def _cjk_substring_match(tag_norm: str, query_token_norm: str) -> bool:
    """CJK substring match — prefix/suffix only, never middle.

    Implementation lives in text.cjk_substring_match (Phase 1); thin re-export here.
    """
    from .text import cjk_substring_match
    return cjk_substring_match(tag_norm, query_token_norm)


def _is_pure_cjk_token(token: str) -> bool:
    """True if the token contains NO ASCII alphanumerics (OPPOSITE of _is_cjk_token).

    Implementation lives in text.is_pure_cjk_token (Phase 1); thin re-export here.
    Do NOT merge with _is_cjk_token (any-CJK) — they serve different match paths.
    """
    from .text import is_pure_cjk_token
    return is_pure_cjk_token(token)


def _score_tags_surface(
    query: str,
    tags_list: list[str],
    strong_weight: float,
    medium_weight: float,
    weak_weight: float,
    cap: float,
) -> tuple[float, str, dict[str, Any]]:
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
    total = 0
    for raw_token in query_tokens:
        token_norm = _normalize_token_for_tag_match(raw_token)
        if not token_norm:
            # Skip tokens that normalize to empty (e.g. stray punctuation) so
            # they don't drag down the ratio without a chance to match.
            continue
        total += 1
        if _is_pure_cjk_token(token_norm):
            hit = any(_cjk_substring_match(tn, token_norm) for tn in tags_norm_set)
        else:
            hit = token_norm in tags_norm_set
        if hit:
            matched += 1

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
    ws_canonical: Optional[str] = None,
    isolation: str = "none",
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
        subject_tags_weak = subject_level in ("none", "weak") and tag_level in ("none", "weak")
        content_long = len(content) > 2000
        if subject_tags_weak and content_long and content_score > 0:
            relevance -= _LONG_CONTENT_PENALTY

        # v0.3.1: vec0-recalled candidates. If this candidate came from the
        # semantic channel and lexical relevance is below the floor, raise it
        # to the floor. The floor sits just below content-score cap, so a vec
        # candidate beats content-only noise but loses to any subject/tags hit.
        if rec.get("_vec_candidate") and relevance < _VEC_FLOOR_SCORE:
            relevance = _VEC_FLOOR_SCORE

        trust = _trust_bonus(rec)
        recency = _recency_bonus(rec)
        ws_adjust = _workspace_bonus(rec, ws_canonical, isolation)
        # Superseded always sinks below active regardless of score (r4 carries
        # this forward from v0.2.6).
        superseded_sink = 1 if rec.get("status") == "superseded" else 0
        fusion_score = float(rec.get("_fusion_score") or 0.0)
        final_score = (
            relevance
            + fusion_score * _RRF_SCORE_WEIGHT
            + trust
            + recency
            + ws_adjust
            - (superseded_sink * 1000.0)
        )

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
        if rec.get("_vec_candidate"):
            if match_reason == "subject_or_tag_match":
                match_reason = "vec_recall"
            notes.append("v0.3.1: semantic recall candidate, floor score applied")
        if rec.get("_evidence_vec_candidate"):
            if match_reason == "subject_or_tag_match":
                match_reason = "evidence_vec_recall"
            notes.append("local-text evidence recall candidate (vNext)")

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
        rec_copy["_workspace_bonus"] = ws_adjust
        rec_copy["_fusion_score"] = fusion_score
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
    status_filter: str = "active",  # "active", "expired", "all"
    pool_cap: int = 50,
    content_like_fallback: bool = True,
    query_embedding: Optional[list[float]] = None,
    content_like_cap: int = 30,
    ws_canonical: Optional[str] = None,
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
    workspace_clause_m = ""
    workspace_clause = ""
    workspace_params: list[Any] = []
    if ws_canonical:
        workspace_clause_m = " AND COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace) = ?"
        workspace_clause = " AND COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?"
        workspace_params.append(ws_canonical)
    conn = db._new_connection()
    try:
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
                    WHERE memories_fts MATCH ? AND {status_clause_m}{workspace_clause_m}
                """
                params: list[Any] = [fts_main, *workspace_params]
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
                        WHERE memories_fts MATCH ? AND {status_clause_m}{workspace_clause_m}
                    """
                    params = [fts_or, *workspace_params]
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
            for tag in tags or []:
                clauses.append("tags LIKE ?")
                params.append(f"%{tag}%")
            if ws_canonical:
                clauses.append("COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?")
                params.extend(workspace_params)
            params.append(pool_cap)
            sql = f"""SELECT *, 0 AS score FROM memories
                      WHERE {' AND '.join(clauses)}
                      ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
                               ingest_time DESC LIMIT ?"""
            try:
                for row in conn.execute(sql, params).fetchall():
                    d = row_to_dict(row)
                    if d["id"] not in pool:
                        pool[d["id"]] = d
            except Exception:
                pass

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
                for tag in tags or []:
                    clauses.append("tags LIKE ?")
                    params.append(f"%{tag}%")
                if ws_canonical:
                    clauses.append("COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?")
                    params.extend(workspace_params)
                params.append(content_like_cap)  # cap content-LIKE gap-fill (configurable via MEMORY_ARBITER_CONTENT_LIKE_CAP)
                sql = f"""SELECT *, 0 AS score FROM memories
                          WHERE {' AND '.join(clauses)}
                          ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
                                   ingest_time DESC LIMIT ?"""
                try:
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
                except Exception:
                    pass
    finally:
        conn.close()

    # Freeze lexical rank before adding evidence. Evidence is an independent
    # bounded channel: it must run even when FTS/LIKE already filled the pool.
    lexical_rank = {int(memory_id): rank for rank, memory_id in enumerate(pool, 1)}

    # Local-text evidence KNN aggregates multiple evidence hits into one memory
    # candidate so long documents cannot occupy many result slots.
    if (
        query_embedding
        and db.state.sqlite_vec_available
    ):
        evidence_memory_cap = max(pool_cap, 10)
        evidence_rows = db.evidence_knn(
            query_embedding,
            k=evidence_memory_cap * 8,
            parent_status_filter=status_filter,
            workspace=ws_canonical,
        )
        by_memory: dict[int, dict[str, Any]] = {}
        for row in evidence_rows:
            rid = row.get("memory_id")
            if rid is None:
                continue
            mid = int(rid)
            entry = by_memory.setdefault(mid, {"row": row, "hits": []})
            distance = row.get("distance")
            try:
                score = 1.0 - float(distance) if isinstance(distance, (int, float)) else 0.0
            except (TypeError, ValueError):
                score = 0.0
            entry["hits"].append({
                "evidence_id": row.get("id"),
                "kind": row.get("kind"),
                "text": row.get("text"),
                "start_offset": row.get("start_offset"),
                "end_offset": row.get("end_offset"),
                "distance": distance,
                "score": score,
            })
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for mid, entry in by_memory.items():
            hits = sorted(entry["hits"], key=lambda h: float(h.get("score") or 0.0), reverse=True)
            best = float(hits[0].get("score") or 0.0) if hits else 0.0
            support = 0.0
            seen_kinds: set[str] = set()
            for hit in hits[1:4]:
                kind = str(hit.get("kind") or "")
                if kind in seen_kinds:
                    continue
                seen_kinds.add(kind)
                support += max(0.0, float(hit.get("score") or 0.0) - 0.45)
            ranked.append((best + 0.08 * support, mid, entry["row"]))
        ranked.sort(reverse=True, key=lambda item: item[0])
        evidence_only: list[int] = []
        for evidence_rank, (_score, mid, row) in enumerate(
            ranked[:evidence_memory_cap], 1,
        ):
            lexical_row = pool.get(mid)
            d = dict(lexical_row or row)
            d["id"] = mid
            d["_vec_candidate"] = True
            d["_evidence_vec_candidate"] = True
            d["_evidence_rank"] = evidence_rank
            d["_evidence_hits"] = sorted(
                by_memory[mid]["hits"],
                key=lambda h: float(h.get("score") or 0.0),
                reverse=True,
            )[:3]
            if lexical_row is None:
                evidence_only.append(mid)
            pool[mid] = d
        if evidence_only:
            # Evidence-only candidates must look like every other result row:
            # real memories columns (version, agent_id, source_ref, ...) with
            # evidence details confined to _evidence_hits — not raw KNN join
            # rows carrying evidence fields at the top level.
            try:
                conn = db._new_connection()
                try:
                    placeholders = ",".join("?" for _ in evidence_only)
                    mem_rows = {
                        int(r["id"]): row_to_dict(r)
                        for r in conn.execute(
                            f"SELECT * FROM memories WHERE id IN ({placeholders})",
                            evidence_only,
                        ).fetchall()
                    }
                finally:
                    conn.close()
            except sqlite3.Error:
                mem_rows = {}
            for mid in evidence_only:
                mem_row = mem_rows.get(mid)
                if mem_row is None:
                    pool.pop(mid, None)
                    continue
                preserved: dict[str, Any] = {
                    key: pool[mid][key]
                    for key in (
                        "_vec_candidate", "_evidence_vec_candidate",
                        "_evidence_rank", "_evidence_hits", "id",
                    )
                    if key in pool[mid]
                }
                pool[mid] = {**mem_row, **preserved}

    # Fuse channel ranks, then restore the original bounded pool size. A memory
    # present in both channels naturally receives more support than one present
    # in only one channel. Trust/recency remain later, lightweight adjustments.
    for memory_id, row in pool.items():
        lexical = lexical_rank.get(int(memory_id))
        evidence = row.get("_evidence_rank")
        fusion = 0.0
        if lexical is not None:
            fusion += 1.0 / (_RRF_K + lexical)
            row["_lexical_rank"] = lexical
        if evidence is not None:
            fusion += 1.0 / (_RRF_K + int(evidence))
        row["_fusion_score"] = fusion

    fused = sorted(
        pool.values(),
        key=lambda row: (
            float(row.get("_fusion_score") or 0.0),
            -int(row.get("_lexical_rank") or 10**9),
        ),
        reverse=True,
    )
    if not lexical_rank or not any(row.get("_evidence_rank") for row in fused):
        return fused[:pool_cap]

    # Reserve bounded admission for both channels before the final soft rerank.
    # RRF alone gives a lexical-only and evidence-only candidate at the same
    # rank the same score, so deterministic tie-breaking could still starve one
    # channel when the other is full. The quotas guarantee representation while
    # keeping the candidate count exactly at pool_cap.
    lexical_quota = (pool_cap + 1) // 2
    evidence_quota = pool_cap - lexical_quota
    lexical_candidates = sorted(
        (row for row in fused if row.get("_lexical_rank") is not None),
        key=lambda row: int(row["_lexical_rank"]),
    )
    evidence_candidates = sorted(
        (row for row in fused if row.get("_evidence_rank") is not None),
        key=lambda row: int(row["_evidence_rank"]),
    )
    selected: dict[int, dict[str, Any]] = {}
    for row in lexical_candidates[:lexical_quota]:
        selected[int(row["id"])] = row
    for row in evidence_candidates[:evidence_quota]:
        selected[int(row["id"])] = row
    if len(selected) < pool_cap:
        for row in fused:
            selected.setdefault(int(row["id"]), row)
            if len(selected) >= pool_cap:
                break
    return list(selected.values())


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


def _parse_time(s: Any) -> Optional[datetime]:
    """v0.7.3: parse an ISO 8601 time string for after_time/before_time filtering.

    Implementation lives in timeutil.parse_iso8601 (Phase 1); thin re-export here.
    Naive datetimes are treated as UTC; returns None on falsy/unparseable input.
    """
    from .timeutil import parse_iso8601
    return parse_iso8601(s)


def _sanitize_tags_filter(tags_filter: Optional[list[str]]) -> Optional[list[str]]:
    """v0.7.3: normalize the tags_filter argument (design §3.2).

    Drops non-strings, empty strings, and duplicates (preserving first-seen
    order). An empty result is returned as None so callers treat it as
    "no filter" (same as not passing the argument).
    """
    if tags_filter is None:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for t in tags_filter:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out if out else None


def _passes_filters(
    rec: dict[str, Any],
    tags_filter: Optional[list[str]],
    after_dt: Optional[datetime],
    before_dt: Optional[datetime],
    source_type: Optional[str],
) -> bool:
    """v0.7.3: post-filter a candidate row against user-provided filters.

    Mirrors db.count_filtered_memories so COUNT and post-filter stay in sync
    (design §3.5/§3.7 B2).
      - tags_filter: AND semantics — rec['tags'] (JSON list) must contain every
        listed tag (set intersection, equivalent to SQL json_each AND EXISTS).
      - after_dt / before_dt: compare against ingest_time (parsed aware UTC).
        rec's ingest_time goes through _parse_ingest_time (search.py:162);
        naive treated as UTC.
      - source_type: equality.
    """
    if tags_filter:
        try:
            rec_tags_raw = rec.get("tags") or "[]"
            rec_tags = json.loads(rec_tags_raw) if isinstance(rec_tags_raw, str) else rec_tags_raw
            rec_tags_set = {str(t) for t in rec_tags}
        except Exception:
            rec_tags_set = set()
        if not all(t in rec_tags_set for t in tags_filter):
            return False
    if source_type and rec.get("source_type") != source_type:
        return False
    if after_dt or before_dt:
        rec_dt = _parse_ingest_time(rec)
        if rec_dt is None:
            # Time filter active but rec has no parseable time — drop conservatively.
            return False
        if after_dt and rec_dt < after_dt:
            return False
        if before_dt and rec_dt > before_dt:
            return False
    return True


def search_memories(
    db: MemoryDB,
    query: str,
    workspace: Optional[str] = None,
    tags: Optional[list[str]] = None,
    limit: int = 10,
    status_filter: str = "active",  # "active", "expired", "all" ("superseded" → "expired")
    debug_ranking: bool = False,
    query_embedding: Optional[list[float]] = None,
    # v0.7.3 additions (design §3.1) — all optional, omit == v0.7.2 behaviour
    tags_filter: Optional[list[str]] = None,
    after_time: Optional[str] = None,
    before_time: Optional[str] = None,
    source_type: Optional[str] = None,
    offset: int = 0,
    ws_canonical: Optional[str] = None,
    isolation: str = "none",
) -> SearchOutcome:
    """v0.9.4: returns a SearchOutcome with retrieval_mode.

    has_more/total_estimate give the caller a way to tell exhaustive queries
    ("all release notes") from complete ones. retrieval_mode drives
    linked_open_items triggering (only "direct" is eligible).

    v0.9.4: ``offset`` enables cursor pagination. On the empty-query+filters
    path it maps to SQL OFFSET (exact, backed by count_filtered_memories). On
    the query-recall path it widens the candidate pool to cover the offset
    window — best-effort, since relevance-ranked recall has no exact total and
    the pool cap bounds the reachable depth (deep pages may return empty).
    """
    warnings: list[str] = []
    if not db.db_available:
        return SearchOutcome([], ["SQLite unavailable; search cannot read JSONL backup in MVP."], False, 0, "unavailable")
    limit = max(1, min(int(limit), 100))
    offset = max(0, min(int(offset), 10000))
    query = (query or "").strip()
    mode = _get_ranking_mode()
    # v0.3.1: when a query_embedding is supplied but sqlite-vec is not active,
    # warn so the caller knows the semantic channel was silently skipped.
    if query_embedding and not db.state.sqlite_vec_available:
        warnings.append("query_embedding provided but sqlite-vec unavailable; semantic recall skipped.")
    # Status filter: active (default), expired (non-active non-deleted), or all.
    # "superseded" is accepted as a back-compat alias for "expired" (v0.9.4
    # widened the expired domain from superseded-only to all non-active
    # non-deleted, covering conflicted/pending for audit recall).
    if status_filter == "superseded":
        status_filter = "expired"
    if status_filter == "active":
        status_clause = "m.status = 'active'"
        like_status_clause = "status = 'active'"
    elif status_filter == "expired":
        status_clause = "m.status NOT IN ('active','deleted')"
        like_status_clause = "status NOT IN ('active','deleted')"
    else:  # "all"
        status_clause = "m.status != 'deleted'"
        like_status_clause = "status != 'deleted'"

    # === v0.7.3: parse + sanitize filter params (design §3.5 第一步) ===
    after_dt = _parse_time(after_time) if after_time else None
    if after_time and after_dt is None:
        warnings.append(f"after_time={after_time!r} invalid ISO 8601; ignored")
        after_time = None
    before_dt = _parse_time(before_time) if before_time else None
    if before_time and before_dt is None:
        warnings.append(f"before_time={before_time!r} invalid ISO 8601; ignored")
        before_time = None
    # D4: after > before 矛盾检查（严格 >；== 是单点区间，合法）
    if after_dt and before_dt and after_dt > before_dt:
        warnings.append(
            f"after_time ({after_dt.replace(microsecond=0).isoformat()}) > before_time "
            f"({before_dt.replace(microsecond=0).isoformat()}); interval is empty; both ignored"
        )
        after_dt = None
        before_dt = None
        after_time = None
        before_time = None
    tags_filter = _sanitize_tags_filter(tags_filter)
    has_filters = bool(tags_filter or after_time or before_time or source_type)

    # === bm25 mode: legacy v0.2.6 single-FTS ordering ===
    if mode == "bm25" and has_filters:
        warnings.append(
            "bm25 mode falls back to hybrid ranking when tags_filter/after_time/before_time/source_type are set."
        )
    if mode == "bm25" and not has_filters:
        result, bm_warnings, bm_has_more, bm_total = _search_bm25(
            db, query, workspace, tags, limit, status_clause, like_status_clause, warnings, debug_ranking,
            offset=offset,
            ws_canonical=strict_ws(isolation, ws_canonical),
        )
        # Infer retrieval_mode for the legacy bm25 path: _search_bm25 internally
        # falls back to _recent_fallback when query has no hit (appending its
        # distinctive warning), so sniff the warnings to tell the two apart.
        # Coupling note: the literal lives in _NO_DIRECT_MATCH_PREFIX; matching
        # on the prefix constant (not an inline string) keeps this robust to
        # wording tweaks of the warning's tail.
        bm_mode: RetrievalMode
        if not query:
            bm_mode = "recent_browse"
        elif not result:
            bm_mode = "empty"
        elif any(w.startswith(_NO_DIRECT_MATCH_PREFIX) for w in bm_warnings):
            bm_mode = "recent_fallback"
        else:
            bm_mode = "direct"
        return SearchOutcome(result, bm_warnings, bm_has_more, bm_total, bm_mode)

    # === v0.7.3 F1/C1: search.py empty-query shortcut ===
    # 现状是 `if not query: return _recent_fallback(...)`，会让 query 为空时
    # 无条件走 fallback——即使 has_filters=True 也跳过 post-filter，返回未
    # 过滤的最近记忆。改成 query 空 且 无过滤 才短路；query 空 + 有过滤
    # 继续往下（wide_recall 内部仍会因 not query 返 []，post-filter 后仍空，
    # 最终走第二步的 "query required for filter-aware recall" 精准 warning）。
    if not query and not has_filters:
        fb_ws = strict_ws(isolation, ws_canonical)
        fb_rows, fb_warnings, fb_hm, fb_te = _recent_fallback(
            db, workspace, tags, limit, like_status_clause, warnings, offset=offset, ws_canonical=fb_ws,
        )
        return SearchOutcome(fb_rows, fb_warnings, fb_hm, fb_te, "recent_browse")

    # === G6 (v0.8.5): empty query + filters → filter-driven recall ===
    # query 为空但带了 tags_filter / 时间 / source_type：不再走 wide_recall
    # （它会因 not query 返 []），而是直接按 filter 召回、ingest_time 倒序。
    # 解锁 list-by-tag / by-source_type / by-time。v0.9.4 adds SQL OFFSET for
    # expired audit pagination.
    if not query and has_filters:
        strict_ws_filter = strict_ws(isolation, ws_canonical)
        rows = db.recall_by_filters(
            like_status_clause, tags_filter, after_dt, before_dt, source_type, limit, offset,
            ws_canonical=strict_ws_filter,
        )
        total_estimate = db.count_filtered_memories(
            like_status_clause, tags_filter, after_dt, before_dt, source_type,
            ws_canonical=strict_ws_filter,
        )
        if not rows:
            warning = "offset beyond result set" if total_estimate > 0 and offset >= total_estimate else "no memories match the given filters"
            return SearchOutcome([], warnings + [warning], False, total_estimate, "empty")
        results = rows[:limit]
        has_more = total_estimate > offset + len(results)
        # No _soft_rerank on this branch (empty query) and recall_by_filters returns
        # bare SELECT * rows, so there are no _-prefixed debug fields to strip.
        return SearchOutcome(results, warnings, has_more, total_estimate, "direct")

    # === v0.7.3: pool 组装 + post-filter（design §3.5 第三步） ===
    # v0.9.4: when paginating (offset > 0), widen the recall pool past the
    # requested window by one row so has_more can be inferred without a false
    # negative when the window is exactly full. At offset=0 the original
    # pool_cap is preserved (keeps the pool-saturation / Channel-6 skip
    # semantics intact). Query-recall still has no exact total; the empty-query
    # SQL paths above are the precise pagination paths.
    base_pool_cap = db.settings.recall_pool_cap
    pool_cap = max(base_pool_cap, offset + limit + 1) if offset > 0 else base_pool_cap
    pool = _wide_recall(db, query, workspace, tags, status_clause, like_status_clause,
                        status_filter=status_filter, query_embedding=query_embedding,
                        pool_cap=pool_cap,
                        content_like_cap=db.settings.content_like_cap,
                        ws_canonical=strict_ws(isolation, ws_canonical))

    # v0.9.7: strict isolation — hard-filter the candidate pool to the query's
    # canonical workspace. weak does NOT filter (it only nudges ranking in
    # _soft_rerank); none ignores workspace entirely.
    if isolation == "strict" and ws_canonical:
        pool = [
            r for r in pool
            if (r.get("workspace_canonical") or r.get("workspace")) == ws_canonical
        ]

    if has_filters:
        pool = [r for r in pool if _passes_filters(r, tags_filter, after_dt, before_dt, source_type)]
        if not pool:
            # 有过滤但召回空：返回空结果，不走 fallback（fallback 会返回不
            # 符合过滤条件的记忆，违反语义）。区分两种空因给出精准 warning。
            if not query:
                empty_reason = (
                    "query required for filter-aware recall; tags_filter/after_time/"
                    "before_time/source_type only post-filter query-recalled candidates "
                    "(see §8 risk 9)"
                )
            else:
                empty_reason = "filters too restrictive or no matches; pool was empty after post-filter"
            return SearchOutcome([], warnings + [empty_reason], False, 0, "empty")
    else:
        # 无过滤：保留 v0.7.2 行为，pool 空走 fallback。v0.9.7: strict 下
        # query 未命中本 ws 时不应回退到「最近记忆」——strict 的语义是
        # 「搜不到就是搜不到」，最近兜底会让用户以为 query 命中了。故
        # strict 直接返回空；none/weak 保留全库/同 ws 最近兜底。
        if not pool:
            if isolation == "strict" and ws_canonical:
                return SearchOutcome(
                    [], warnings + ["no same-workspace match; strict isolation does not fall back to recent memories"],
                    False, 0, "empty",
                )
            fb_rows, fb_warnings, fb_hm, fb_te = _recent_fallback(
                db, workspace, tags, limit, like_status_clause, warnings, offset=offset,
            )
            return SearchOutcome(fb_rows, fb_warnings, fb_hm, fb_te, "recent_fallback")

    reranked = _soft_rerank(query, pool, ws_canonical=ws_canonical, isolation=isolation)
    # Slice to the requested page window.
    page = reranked[offset:offset + limit]

    # === v0.7.3: has_more / total_estimate（design §3.6 E1） ===
    # E1: 无过滤场景 total_estimate = len(pool)（query 召回数，反映 query
    # 匹配，不是全库大小）；有过滤场景走 count_filtered_memories（SQL 全表
    # 按过滤计数，受 pool_cap 截断影响的只是 reranked，total_estimate 仍准）。
    if has_filters:
        total_estimate = db.count_filtered_memories(
            like_status_clause, tags_filter, after_dt, before_dt, source_type,
            ws_canonical=strict_ws(isolation, ws_canonical),
        )
    else:
        total_estimate = len(pool)
    # K1: has_more = total > offset + len(page). 修了原公式 len==limit and total>limit
    # 在 pool 召回不足时漏报（reranked<limit 但 total>reranked 应判 True）。
    has_more = total_estimate > offset + len(page)

    # hybrid mode: strip debug fields unless explicitly requested.
    if not debug_ranking:
        for r in page:
            for k in list(r.keys()):
                if k.startswith("_"):
                    r.pop(k, None)
    return SearchOutcome(page, warnings, has_more, total_estimate, "direct")


def _coerce_tags(raw: Any) -> list[str]:
    """v0.7.4: normalise a memory's ``tags`` field into a deduped ``list[str]``.

    Implementation lives in text.coerce_tags (Phase 1 single source); thin re-export
    here so existing imports keep working. Never raises — bad shapes yield [].
    """
    from .text import coerce_tags
    return coerce_tags(raw)


def _linked_open_items_for_search(
    db: MemoryDB,
    results: list[dict[str, Any]],
    warnings: list[str],
    max_items: int = 5,
    ws_canonical: Optional[str] = None,
) -> list[dict[str, Any]]:
    """v0.7.4: attach up to ``max_items`` active todo memories that share
    meaningful tags with the current result set (linked_open_items).

    Pure read-only enhancement — never writes. Never raises: on any DB error
    returns [] and appends a degradation warning to ``warnings``.

    Three-layer short-circuit (design 性能设计):
      L0 (memory): bail without touching DB if results carry no meaningful tag
           (after stripping ``todo`` and single-char tags).
      L1 (DB): EXISTS check for any active memory tagged ``todo``; bail if none.
      L2 (DB): multiple SELECTs on one connection compute ``active_count``,
           per-tag ``df``, todo candidates, apply the M1 stoplist, score, sort,
           truncate. Note: this is a best-effort read, NOT a transactional
           snapshot — the bare SELECTs don't share a read snapshot under WAL,
           so concurrent writes can in principle make the count/df/candidates
           slightly inconsistent. Acceptable for an advisory side-hint; if
           consistency ever matters here, wrap the SELECTs in a read txn.

    Stoplist (M1 — uniform, independent of todo count):
      tag == 'todo' | len(tag) <= 1 | df >= 3 AND df/active_count >= 0.20

    ``json_valid(tags)`` is applied in SQL so malformed-tag rows are silently
    filtered (M4-A) — this does NOT produce a warning. Only a real DB failure
    produces a warning (M4-B).
    """
    if not results:
        return []

    # --- L0: collect meaningful tags from results (strip todo / single-char) ---
    result_id_to_tags: dict[int, set[str]] = {}
    all_meaningful: set[str] = set()
    for rec in results:
        rid = rec.get("id")
        tags = _coerce_tags(rec.get("tags"))
        meaningful = {t for t in tags if t != "todo" and len(t) > 1}
        if rid is not None and meaningful:
            result_id_to_tags[int(rid)] = meaningful
            all_meaningful |= meaningful
    if not all_meaningful or not db.db_available:
        return []

    result_ids = list(result_id_to_tags.keys())

    def _is_stoplisted(tag: str, df: int, active_count: int) -> bool:
        # M1: uniform stoplist — no todo-count branching.
        if tag == "todo":
            return True
        if len(tag) <= 1:
            return True
        if df >= 3 and active_count > 0 and df / active_count >= 0.20:
            return True
        return False

    try:
        conn = db._new_connection()
        workspace_clause = ""
        workspace_params: list[Any] = []
        if ws_canonical:
            workspace_clause = "AND COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace) = ?"
            workspace_params.append(ws_canonical)
        try:
            # --- L1: EXISTS check for active+todo memories ---
            todo_exists = conn.execute(
                "SELECT EXISTS ("
                " SELECT 1 FROM memories m"
                " WHERE m.status='active' " + workspace_clause +
                "   AND EXISTS ("
                "     SELECT 1 FROM json_each("
                "       CASE WHEN json_valid(m.tags) THEN m.tags ELSE '[]' END"
                "     ) WHERE json_each.value='todo' AND json_each.type='text'"
                "   )"
                ") AS e",
                workspace_params,
            ).fetchone()["e"]
            if not todo_exists:
                return []

            # --- L2: active_count ---
            active_count = int(conn.execute(
                "SELECT COUNT(*) AS c FROM memories m WHERE m.status='active' " + workspace_clause,
                workspace_params,
            ).fetchone()["c"])
            if active_count <= 0:
                return []

            # todo candidates: active + tagged 'todo', excluding result IDs.
            ph = ",".join("?" * len(result_ids)) if result_ids else ""
            exclude_clause = f"AND m.id NOT IN ({ph})" if result_ids else ""
            cand_rows = conn.execute(
                f"""
                SELECT m.id, m.subject, m.tags, m.ingest_time
                FROM memories m
                WHERE m.status='active' {exclude_clause} {workspace_clause}
                  AND EXISTS (
                    SELECT 1 FROM json_each(
                      CASE WHEN json_valid(m.tags) THEN m.tags ELSE '[]' END
                    ) WHERE json_each.value='todo' AND json_each.type='text'
                  )
                """,
                list(result_ids) + workspace_params,
            ).fetchall()
            if not cand_rows:
                return []

            # per-tag df across the active set (json_valid guard ⇒ M4-A silence).
            tag_df: dict[str, int] = {}
            df_rows = conn.execute(
                f"""
                SELECT tag.value AS t, COUNT(DISTINCT m.id) AS df
                FROM memories m, json_each(
                  CASE WHEN json_valid(m.tags) THEN m.tags ELSE '[]' END
                ) AS tag
                WHERE m.status='active' {workspace_clause} AND tag.type='text'
                GROUP BY tag.value
                """,
                workspace_params,
            ).fetchall()
            for r in df_rows:
                tag_df[r["t"]] = int(r["df"])

            scored: list[dict[str, Any]] = []
            for row in cand_rows:
                cand_id = int(row["id"])
                cand_tags = _coerce_tags(row["tags"])
                cand_subject = row["subject"] or f"memory #{cand_id}"
                cand_ingest = row["ingest_time"] or ""
                matched_meaningful: set[str] = set()
                matched_result_ids: set[int] = set()
                for tag in cand_tags:
                    if tag not in all_meaningful:
                        continue
                    if _is_stoplisted(tag, tag_df.get(tag, 0), active_count):
                        continue
                    matched_meaningful.add(tag)
                    for rid, rtags in result_id_to_tags.items():
                        if tag in rtags:
                            matched_result_ids.add(rid)
                # score: 2 per matched meaningful tag (≥ 2 ⇒ ≥ 1 overlap).
                score = 2 * len(matched_meaningful)
                if score < 2:
                    continue
                scored.append({
                    "id": cand_id,
                    "subject": cand_subject,
                    "tags": cand_tags,
                    "ingest_time": cand_ingest,
                    "reason": "tag_overlap: " + ", ".join(sorted(matched_meaningful)),
                    "matched_result_ids": sorted(matched_result_ids),
                    "_score": score,
                })
            if not scored:
                return []
            # score DESC → ingest_time DESC → id DESC.
            scored.sort(
                key=lambda x: (x["_score"], x["ingest_time"], x["id"]),
                reverse=True,
            )
            out: list[dict[str, Any]] = []
            for item in scored[:max_items]:
                item.pop("_score", None)
                out.append(item)
            return out
        finally:
            conn.close()
    except sqlite3.Error as exc:
        warnings.append(f"linked_open_items lookup failed: {exc}; returned [].")
        return []
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"linked_open_items lookup failed: {exc}; returned [].")
        return []


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
    offset: int = 0,
    ws_canonical: Optional[str] = None,
) -> Tuple[list[dict[str, Any]], list[str], bool, int]:
    """Legacy v0.2.6 bm25 ordering. Kept for RANKING_MODE=bm25 fallback."""
    rows = []
    workspace_clause_m = ""
    workspace_clause = ""
    workspace_params: list[Any] = []
    if ws_canonical:
        workspace_clause_m = " AND COALESCE(NULLIF(m.workspace_canonical, ''), m.workspace) = ?"
        workspace_clause = "COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?"
        workspace_params.append(ws_canonical)
    conn = db._new_connection()
    if db.state.fts5_available and query:
        sql = f"""
            SELECT m.*, bm25(memories_fts) AS score
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ? AND {status_clause_m}{workspace_clause_m}
        """
        params: list[Any] = [_sanitize_fts_query(query), *workspace_params]
        sql += " ORDER BY CASE m.status WHEN 'superseded' THEN 1 ELSE 0 END, score LIMIT ? OFFSET ?"
        params.extend([limit + 1, offset])
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
        for tag in tags or []:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")
        if ws_canonical:
            clauses.append(workspace_clause)
            params.extend(workspace_params)
        params.extend([limit + 1, offset])
        try:
            rows = conn.execute(
                f"""SELECT *, 0 AS score FROM memories
                    WHERE {' AND '.join(clauses)}
                    ORDER BY CASE status WHEN 'superseded' THEN 1 ELSE 0 END,
                             event_time DESC, ingest_time DESC
                    LIMIT ? OFFSET ?""",
                params,
            ).fetchall()
        except Exception as exc:
            warnings.append(f"LIKE fallback query failed: {exc}.")
            rows = []
        if query and not db.state.fts5_available:
            warnings.append("Using LIKE/keyword search because sqlite-vec and FTS5 are unavailable.")
    conn.close()
    if query and not rows:
        # v0.9.7: strict 下 query 未命中不应回退到最近记忆（会返回本 ws
        # 的无关记忆，误导用户以为 query 命中）。none/weak 保留最近兜底。
        if ws_canonical:
            return [], warnings, False, 0
        fb_rows, fb_warnings, fb_has_more, fb_total = _recent_fallback(
            db, workspace, tags, limit, like_status_clause, warnings, offset=offset, ws_canonical=ws_canonical,
        )
        return fb_rows, fb_warnings, fb_has_more, fb_total
    has_more = len(rows) > limit
    page_rows = rows[:limit]
    total_estimate = offset + len(page_rows) + (1 if has_more else 0)
    out = [row_to_dict(row) for row in page_rows]
    if not debug_ranking:
        for r in out:
            r.pop("score", None)
    return out, warnings, has_more, total_estimate


def _recent_fallback(
    db: MemoryDB,
    workspace: Optional[str],
    tags: Optional[list[str]],
    limit: int,
    like_status_clause: str,
    warnings: list[str],
    offset: int = 0,
    ws_canonical: Optional[str] = None,
) -> Tuple[list[dict[str, Any]], list[str], bool, int]:
    """Recent-memory fallback when no direct match found (r4 §4.2 safety net)."""
    clauses = [like_status_clause]
    params: list[Any] = []
    for tag in tags or []:
        clauses.append("tags LIKE ?")
        params.append(f"%{tag}%")
    # strict isolation: filter to the query's canonical workspace INSIDE the SQL
    # so COUNT and the paginated window agree — a Python post-filter on an
    # already-paginated page reports a wrong total and can under-fill the page.
    # Match canonical with a raw fallback for rows written before the column
    # existed (workspace_canonical NULL → compare against raw workspace).
    if ws_canonical:
        clauses.append("COALESCE(NULLIF(workspace_canonical, ''), workspace) = ?")
        params.append(ws_canonical)
    conn = db._new_connection()
    try:
        count_row = conn.execute(
            f"SELECT COUNT(*) AS c FROM memories WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        total_estimate = int(count_row["c"] or 0) if count_row else 0
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
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()
    finally:
        conn.close()
    if rows:
        warnings.append(
            _NO_DIRECT_MATCH_PREFIX
            + ". Returning recent memories from the shared library; refine keywords, try memory_recent, or compare candidates before reading source files."
        )
    has_more = total_estimate > offset + len(rows)
    return [row_to_dict(row) for row in rows], warnings, has_more, total_estimate
