"""CJK anchor extraction for soft-rerank scoring.

The v0.3.0 search ranking no longer trusts raw trigram overlap on subject/tags
as the only signal. Instead it extracts "anchors" — short fragments that are
closer to human "phrase feel" than single chars, but cheaper and more robust
than a full segmentation. Anchors feed the relevance scorer in search.py.

Design (see ~/Desktop/memory-arbiter-v0.3.0-宽召回软重排方案.md §8.1.2):

  1. Split text into runs by character class
     - CJK run    → sliding bigram
     - ASCII run  → kept whole as a single token (version ids, identifiers)
     - Separators → discarded (they naturally bound runs, no bigrams across them)
  2. Apply two-level stop list
     - STOP_ANCHORS    : dropped entirely (function words, filler)
     - GENERIC_ANCHORS : kept but flagged generic — alone they are weak signal,
                         but combined with a specific anchor they still count
  3. Return a list of Anchor(tokens, is_generic) for downstream scoring

This is intentionally heuristic — not a tokenizer, not a segmenter. It only
needs to be "better than trigram alone, cleaner than single chars".
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Character classes — kept aligned with search.py._CJK_RE.
_CJK_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]"
)
# ASCII run = letters, digits, dots, hyphens, underscores, plus, slash — i.e.
# things that commonly appear in version ids (v0.2.6), identifiers
# (memory-arbiter), file paths, etc. We keep them whole rather than chopping.
_ASCII_RUN_RE = re.compile(r"[A-Za-z0-9._\-+/]+")
# Separators are everything else (whitespace, CJK punctuation, hyphens acting
# as separators, parentheses, slashes when surrounded by spaces, etc.). They
# bound runs but don't become anchors themselves.
_RUN_SPLIT_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]+|[A-Za-z0-9._\-+/]+")


@dataclass
class Anchor:
    """A single extracted anchor.

    - text: the anchor string (CJK bigram, ASCII token, or CJK single char)
    - is_generic: True if the anchor is in GENERIC_ANCHORS — alone it is a
                  weak signal, but combined with a specific anchor it still
                  contributes to "medium match"
    """

    text: str
    is_generic: bool = False


# Two-level stop list. See r4 §8.1.2 for the rationale — STOP is "drop entirely"
# (function words, filler), GENERIC is "keep but weak alone" (system/platform/
# project style words that are everywhere and therefore meaningless alone).

STOP_ANCHORS: frozenset[str] = frozenset({
    # single chars (will rarely appear since we do bigrams, but kept as guard)
    "的", "了", "和", "与", "及", "或", "在", "对", "把", "被", "为", "是",
    "有", "无", "中", "上", "下", "内", "外",
    # bigram-level filler that shows up everywhere
    "一个", "一种", "这个", "那个", "进行", "通过", "相关", "以及", "或者",
    "可以", "需要", "完成", "记录", "总结", "问题", "情况", "内容",
})

GENERIC_ANCHORS: frozenset[str] = frozenset({
    # words that are meaningful but too common to be a strong signal alone
    "系统", "平台", "项目", "方案", "流程", "规范", "配置", "工具", "任务",
    "文档", "业务", "数据", "接口", "功能", "模块", "状态", "规则", "策略",
    "交付", "处理", "管理",
})


def _is_cjk_char(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def _split_runs(text: str) -> list[tuple[str, str]]:
    """Split text into (kind, value) runs.

    kind is "cjk" or "ascii". Separators are dropped (they delimit runs).
    """
    runs: list[tuple[str, str]] = []
    for m in _RUN_SPLIT_RE.finditer(text):
        val = m.group(0)
        if _CJK_RE.match(val[0]):
            runs.append(("cjk", val))
        else:
            runs.append(("ascii", val))
    return runs


def _cjk_bigrams(cjk_run: str) -> list[str]:
    """Sliding window of length 2 over a CJK run.

    Single-char CJK run yields nothing — a 1-char anchor is too noisy (single
    chars hit too widely). If the run is exactly 1 char, return [].
    """
    if len(cjk_run) < 2:
        return []
    return [cjk_run[i : i + 2] for i in range(len(cjk_run) - 1)]


def extract_anchors(text: str, mode: str = "query") -> list[Anchor]:
    """Extract anchors from a text string.

    Phase 1: query and document modes behave identically. The mode parameter
    is reserved for Phase 2 (document-side may apply extra frequency decay).

    Returns anchors in order of appearance, with STOP anchors removed and
    GENERIC anchors flagged. Each CJK bigram and each ASCII run becomes one
    anchor.
    """
    if not text:
        return []
    runs = _split_runs(text)
    raw_anchors: list[str] = []
    for kind, val in runs:
        if kind == "cjk":
            raw_anchors.extend(_cjk_bigrams(val))
            # A 1-char CJK run is too short for a bigram but if it's the whole
            # run (e.g. a standalone character like "金" in "金营/财富"), we
            # still drop it — single-char matching is too noisy.
        else:
            # ASCII run: keep whole. Lowercase for case-insensitive matching
            # downstream (VOP vs vop, FTS5 vs fts5).
            # But skip runs that are purely separators (e.g. a lone "-" or "."
            # that the regex picked up between two CJK runs).
            if not any(c.isalnum() for c in val):
                continue
            raw_anchors.append(val.lower())
    # Apply two-level stop list.
    out: list[Anchor] = []
    for a in raw_anchors:
        if a in STOP_ANCHORS:
            continue
        is_generic = a in GENERIC_ANCHORS
        out.append(Anchor(text=a, is_generic=is_generic))
    return out


def score_anchor_overlap(
    query_anchors: list[Anchor],
    surface_anchors: list[Anchor],
) -> dict[str, "AnchorMatch"]:
    """Score how well a document surface (subject/tags) matches the query.

    Returns a dict mapping each query anchor to an AnchorMatch describing
    whether it hit and how (specific/generic/no). Caller aggregates these
    into a match_level per the r4 §8.1.2 rules.

    Surface anchors are pre-extracted from subject/tags/content. We use
    set membership for matching — anchors are short enough that exact match
    is the right granularity (no fuzzy matching).
    """
    surface_specific = {a.text for a in surface_anchors if not a.is_generic}
    surface_generic = {a.text for a in surface_anchors if a.is_generic}
    surface_all = surface_specific | surface_generic

    matches: dict[str, AnchorMatch] = {}
    specific_hits = 0
    generic_hits = 0
    for qa in query_anchors:
        if qa.text in surface_specific:
            matches[qa.text] = AnchorMatch(hit=True, kind="specific")
            specific_hits += 1
        elif qa.text in surface_generic:
            matches[qa.text] = AnchorMatch(hit=True, kind="generic")
            generic_hits += 1
        elif qa.text in surface_all:
            # Surface had the anchor but neither side flagged it generic —
            # treat as specific (e.g. the anchor is rare enough to be specific).
            matches[qa.text] = AnchorMatch(hit=True, kind="specific")
            specific_hits += 1
        else:
            matches[qa.text] = AnchorMatch(hit=False, kind="none")
    matches["_summary"] = AnchorMatch(
        hit=True,
        kind="summary",
        specific_hits=specific_hits,
        generic_hits=generic_hits,
        total_hits=specific_hits + generic_hits,
    )
    return matches


@dataclass
class AnchorMatch:
    hit: bool
    kind: str  # "specific" | "generic" | "none" | "summary"
    specific_hits: int = 0
    generic_hits: int = 0
    total_hits: int = 0


def classify_match_level(
    query_anchors: list[Anchor],
    matches: dict[str, AnchorMatch],
) -> str:
    """Classify the overlap into one of: none | weak | medium | strong.

    Rules (r4 §8.1.2, v0.7.3 修订):

      strong: query's main contiguous phrase is a substring of the surface
              (handled by the caller via direct substring check, not anchors)
      medium: (specific_hits >= 1 AND total_hits >= 2)
              OR (specific_coverage >= 0.6)
      weak:   some anchors hit but neither medium condition holds
      none:   no specific anchors hit (only generic, or nothing)

    v0.7.3 修订（id=210/id=211 dogfooding 数据驱动）：specific_coverage 阈值从
    0.4 提到 0.6。原 0.4 让"subject 只命中 query 一半 anchor"（coverage 0.5）
    也拿 medium(6.0)，结果是 subject 偶然含一个 query 词的记录（如 id=105 的
    "[已完成] README ... (v0.4.0 发版)"）和真正讲该主题的记录同分，挤掉了 tag
    精确双命中但 subject 不含的记录（id=206）。0.6 阈值让 1/2 命中降到 weak
    (2.0)，2/2 才 medium——subject 的"过度奖励"被收紧。

    合成数据实验（scripts/tune_tag_weights.py，n=2000×5 seed）证明这个阈值
    是临界点：0.5 无效（A>B=0.5），0.6 让 A>B=1.000，0.7+ 无额外收益。
    """
    summary = matches.get("_summary")
    if summary is None:
        return "none"
    specific_hits = summary.specific_hits
    total_hits = summary.total_hits

    # specific_coverage: of the query's specific anchors, how many hit.
    # Denominator is the count of specific anchors in the query (excluding
    # generic). Conservative: noise bigrams stay in the denominator.
    query_specific_count = sum(1 for a in query_anchors if not a.is_generic)
    specific_coverage = (
        specific_hits / query_specific_count if query_specific_count else 0.0
    )

    if specific_hits == 0 and total_hits == 0:
        return "none"
    if specific_hits >= 1 and total_hits >= 2:
        return "medium"
    if specific_coverage >= 0.6:
        return "medium"
    if total_hits >= 1:
        return "weak"
    return "none"
