"""检索文本处理层：查询清洗/CJK 工具/tag 规范化/后过滤工具（从 search.py 搬出，拆分批 ⑥ 纯移动）。
search.py re-export 保活测试与调用方 import 面。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .text import CJK_RE_SEARCH as _CJK_RE


def _subject_key(value: Any) -> str:
    """Gate-v2 G2 exact-hit normalization: casefold + strip ALL whitespace,
    both sides (方案口径). Not the evidence-text normalize (it folds units
    and aliases) — subject equality is a literal identity test."""
    return "".join(str(value or "").casefold().split())


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
                # v0.15.9 phrase channel: a quoted phrase under the trigram
                # tokenizer is an exact-substring match, so records containing
                # the token verbatim always enter the pool; the OR'd trigrams
                # stay as the recall safety net for slightly-overspecified
                # queries. Strength separation happens in soft-rerank anchors.
                groups.append("(" + _quote_phrase(tok) + " OR " + " OR ".join(trigrams) + ")")
        else:
            groups.append(_quote_phrase(tok))
    return " AND ".join(groups)


# ---- Soft-rerank scoring constants (r4 §7, §8) --------------------------
# These are deliberately conservative initial values. Per r4 risk-5, we only
# tune 1-2 of these based on A/B; the rest stay fixed.

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


def _is_short_cjk_keyword(token: str) -> bool:
    """检索线 K1：1~4 字、逐字符 CJK 的关键词 token（方案 §3a 判据）。

    「纯 CJK」按 CJK_RE_SEARCH 口径（text.py：含假名/谚文，BMP 外
    CJK 扩展不含——R2-P2-4 澄清）；``is_pure_cjk_token`` 只排除 ASCII
    字母数字（标点/emoji 会漏过），仅作快速前置排除复用。
    """
    if not 1 <= len(token) <= 4:
        return False
    if not _is_pure_cjk_token(token):
        return False
    from .text import is_cjk_char
    return all(is_cjk_char(ch) for ch in token)

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


def _parse_time(s: Any) -> datetime | None:
    """v0.7.3: parse an ISO 8601 time string for after_time/before_time filtering.

    Implementation lives in timeutil.parse_iso8601 (Phase 1); thin re-export here.
    Naive datetimes are treated as UTC; returns None on falsy/unparseable input.
    """
    from .timeutil import parse_iso8601
    return parse_iso8601(s)


def _sanitize_tags_filter(tags_filter: list[str] | None) -> list[str] | None:
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
    tags_filter: list[str] | None,
    after_dt: datetime | None,
    before_dt: datetime | None,
    source_type: str | None,
) -> bool:
    """v0.7.3: post-filter a candidate row against user-provided filters.

    0.15.14 (B2): delegates to the single shared predicate
    (db.memories.row_passes_filters) so this post-filter, the COUNT and the
    filter-driven recall all run identical logic — the former SQL mirror
    drifted on sub-second time bounds and numeric tags (#962 P1#6).
    """
    from .db.memories import row_passes_filters

    return row_passes_filters(
        rec.get("tags"), rec.get("ingest_time"), rec.get("source_type"),
        tags_filter=tags_filter, after_dt=after_dt, before_dt=before_dt,
        source_type=source_type,
    )


def _query_non_cjk_dominant(query: str) -> bool:
    """查询语义由 ASCII/拉丁词承载（CJK 占字母比 < 0.5）时为 True。

    分层门槛的豁免闸（owner 2026-09-26「无关召回涨了就修」）：非 CJK 主导
    查询对 CJK 库结构性零字面锚定，其 evidence-only 行值得余弦线豁免；
    CJK 查询对 CJK 库"本该咬中"而没咬中的语义近邻（legal-form 模板文查询
    即此形态）维持复合线——实测主考卷 zh 查询豁免 51 行 0 gold 全噪音，
    en→zh 豁免 15 gold（C01/C02 探针 28 条误召回由此归零）。
    """
    letters = [c for c in query if (c.isascii() and c.isalpha()) or '\u4e00' <= c <= '\u9fff']
    if not letters:
        return False
    cjk = sum(1 for c in letters if '\u4e00' <= c <= '\u9fff')
    return cjk / len(letters) < 0.5
