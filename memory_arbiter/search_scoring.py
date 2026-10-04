"""检索打分层：打分常量/加分函数/surface 打分/软重排/关键词救援（从 search.py 搬出，拆分批 ⑥ 纯移动）。
QUERY_RECALL_SCORE_FLOOR 两读取点留守 search.py（patch 缝不随迁）。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .constants import COS_MIDBAND_CEIL, COS_RECALL_FLOOR
from .workspace_rules import weak_workspace_vector_weight, workspace_vector_distance
from .constants import (
    KEYWORD_QUERY_MAX_TOKENS,
    KEYWORD_RESCUE_BOOST,
    KEYWORD_RESCUE_DF_MAX,
)
from .anchors import Anchor, classify_match_level, extract_anchors, score_anchor_overlap
from .search_text import (
    _is_pure_cjk_token,
    _cjk_substring_match,
    _is_short_cjk_keyword,
    _normalize_token_for_tag_match,
)


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


def _parse_ingest_time(record: dict[str, Any]) -> datetime | None:
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


def _recency_bonus(record: dict[str, Any], now: datetime | None = None) -> float:
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


def _workspace_bonus(
    record: dict[str, Any],
    ws_canonical: str | None,
    isolation: str,
    distance_map: dict[str, float] | None = None,
    min_name_len: int = 3,
) -> float:
    """Soft workspace nudge for weak isolation. 0 outside weak mode.

    With vector weighting enabled, when the caller precomputed a distance_map, the binary
    step becomes a continuous vector weight — full +0.30 inside 0.15, linear
    decay to 0 at 0.30, 0 beyond (a known-far workspace no longer eats the
    -0.15 hard penalty). Every guarded pair (reserved default term, short
    name, generic-only proximity, or a canonical missing from the map) falls
    back to the original binary step, so degradation is exactly v0.9.7.
    """
    if isolation != "weak" or not ws_canonical:
        return 0.0
    rec_ws = record.get("workspace_canonical") or record.get("workspace") or ""
    if not rec_ws:
        return 0.0
    if distance_map is not None:
        distance = workspace_vector_distance(
            ws_canonical, rec_ws, distance_map, min_name_len=min_name_len,
        )
        if distance is not None:
            return weak_workspace_vector_weight(distance)
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

def is_keyword_query(query: str) -> bool:
    """检索线 K1（owner 2026-09-24 拍板 1/8）：关键词模式判定——空格分隔、
    ≥2 个 token、且全部 token 均为 1~4 字纯 CJK 短词（不加新参数，把
    Agent 现有的「向量 唯一键 冲突」式用法升为一等查询形态）。

    「向量 唯一键 冲突」「金营 智能配券 场景」「操作纪律 桥接脚本」→ True；
    「网站备案手续办完了吗」（单长 token）、「deploy pipeline is green」
    （非 CJK）、「发版 之前要跑哪些检查」（短词+长句混合）、「催收 侮辱，」
    （标点 token）→ False。纯函数，无 IO。
    """
    tokens = (query or "").split()
    if not 2 <= len(tokens) <= KEYWORD_QUERY_MAX_TOKENS:
        return False
    return all(_is_short_cjk_keyword(token) for token in tokens)


def _apply_keyword_rescue(query: str, pool: list[dict[str, Any]]) -> None:
    """检索线 K1：关键词模式查询的中间带救济（方案 §3b，in-place）。

    owner 拍板 2 的「短名单 LIKE」形态：向量召回后的池内候选，best 行
    真余弦落在 [COS_RECALL_FLOOR, COS_MIDBAND_CEIL) 中间带、且无词法
    席位（``_lexical_rank`` 为 None 的 evidence-only 行）、content/
    subject 含任一关键词（**整 token 子串，不切词**——owner 2026-09-24
    追加拍板：B07 门实测连写词救不到，结论是「查不到说明查询的关键词
    不对」，不做强行匹配；曾试过 4 字切前2+后2，r2 误召回 2→18，已撤）
    时，融合分加 KEYWORD_RESCUE_BOOST（×300 → final +3.0，置于
    _soft_rerank 之前自然上浮）。content 在池行里现成（SELECT * 进池），
    ``in`` 子串匹配无通配语义；真余弦缺失（向量未发布/拉取失败）不
    救济。纯内存操作，零新增 SQL。
    """
    if not is_keyword_query(query):
        return
    variants: list[str] = []
    for keyword in query.split():
        if keyword not in variants:
            variants.append(keyword)
    # K3 实施标定：区分度闸——关键词在池内合格行（中间带 evidence-only）
    # 命中超过 KEYWORD_RESCUE_DF_MAX 即视为话题词而非探针词，该词整体
    # 不救济（「做法」「预算」类通用词的护栏；七池探针实测分离带见
    # constants.py 注释）。误召回本身可接受（owner 口径：关键看误召回
    # 是否排在相关结果之后），闸只压话题词的整池抬升。
    eligible = [
        row
        for row in pool
        if row.get("_evidence_best_score") is not None
        and COS_RECALL_FLOOR
        <= float(row["_evidence_best_score"]) < COS_MIDBAND_CEIL
        and row.get("_lexical_rank") is None
    ]
    active: list[str] = []
    for form in variants:
        hits = sum(
            1
            for row in eligible
            if form in (row.get("content") or "")
            or form in (row.get("subject") or "")
        )
        if 0 < hits <= KEYWORD_RESCUE_DF_MAX:
            active.append(form)
    if not active:
        return
    for row in eligible:
        content = row.get("content") or ""
        subject = row.get("subject") or ""
        if any(form in content or form in subject for form in active):
            row["_keyword_rescued"] = True
            row["_fusion_score"] = (
                float(row.get("_fusion_score") or 0.0) + KEYWORD_RESCUE_BOOST
            )


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
    ws_canonical: str | None = None,
    isolation: str = "none",
    distance_map: dict[str, float] | None = None,
    ws_min_name_len: int = 3,
) -> list[dict[str, Any]]:
    """Apply soft-rerank to a wide-recall candidate pool.

    Adds debug fields (_subject_level, _tag_level, _match_reason, _ranking_notes)
    to each row but does NOT mutate original fields. Returns new list sorted
    by final_score descending.

    ``distance_map`` is the precomputed {record canonical → cosine
    distance to the query canonical} dict; when present and isolation is weak,
    _workspace_bonus weights on the continuous curve instead of the binary
    step. ``None`` keeps the v0.9.7 binary behaviour.
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
        ws_adjust = _workspace_bonus(
            rec, ws_canonical, isolation,
            distance_map=distance_map, min_name_len=ws_min_name_len,
        )
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
        if rec.get("_keyword_rescued"):
            notes.append(
                "keyword rescue: midband evidence-only row with keyword hit"
            )

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
