"""Gate-v2 funnel layers (§2b contract): one function per layer, both
orchestrations (write job / scan task) import the SAME code — copying logic
into a second implementation is forbidden. Behaviours that differ between
orchestrations are编排 differences (which layers a caller invokes), never
forked predicates:
- the sentence prefilter is an OPTIONAL layer: the write path calls it (the
  sync window saves KNN work and Qwen only judges weak pure-prose shapes),
  the scan path skips it (Agent judges prose oppositions better than any
  deterministic gate; low pair_score keeps them at the back of the queue).
- the cosine band gate runs in BOTH orchestrations (write loop, scan slow
  lane, diagnostic channel).
G4 lands the first two layers; G5/G6 append the memory-level screen, pair
collection, adjudication and ranking on the same module.
"""
from __future__ import annotations

import re
from typing import Any, Iterator

from ..constants import (
    SEMANTIC_CANDIDATE_COS_CEIL,
    SEMANTIC_CANDIDATE_COS_FLOOR,
)
from ..semantic_conflict import _SENT_PREFILTER, vector_cosine


def row_prefilter(rows: "list[Any]") -> "Iterator[Any]":
    """③ 句子初筛（编排可选层：写入调用、扫描跳过）— yield rows that carry
    an extractable value, a negation word, a time anchor, or are table rows;
    pure prose rows never originate a KNN query. 0.17.1 (owner 拍板)：claim
    对比通道 B/C/桥退役，claim 覆盖句跳过随之删除——被 claim 覆盖的句子重新
    从通道 A 发起（否则成检测死区），本层不再有任何 claims 耦合。"""
    for row in rows:
        if str(getattr(row, "kind", "") or "") == "table_row":
            yield row
            continue
        if _SENT_PREFILTER.search(str(getattr(row, "text", "") or "")):
            yield row


def candidate_cos_gate(
    own_vec: "list[float] | None",
    hit_rows: "list[dict[str, Any]]",
    hit_vectors: "dict[int, list[float]]",
) -> "tuple[list[tuple[dict[str, Any], float]], list[tuple[dict[str, Any], float]], list[tuple[dict[str, Any], float]]]":
    """余弦门 (both orchestrations): true cosine of the own row against each
    candidate row; pairs INSIDE [SEMANTIC_CANDIDATE_COS_FLOOR, CEIL) pass.
    Returns three (hit, cosine) lists so every caller keeps the band split
    observable: PASSED pairs continue down the funnel; BELOW-floor pairs are
    unrelated noise (保安一号); AT/CEIL pairs are the same text — the write
    loop counts them repeatability_skipped, the diagnostic channel routes
    them into duplicates_pool (that pool IS their governance consumer).
    Hit rows whose vector is missing (pre-backfill) come back nowhere — no
    cosine means no verdict. Non-unit vectors (|v|≈16.5) make L2-based
    conversion impossible, hence the fetched-vector cosine."""
    passed: "list[tuple[dict[str, Any], float]]" = []
    below_floor: "list[tuple[dict[str, Any], float]]" = []
    at_ceil: "list[tuple[dict[str, Any], float]]" = []
    if own_vec is None:
        return passed, below_floor, at_ceil
    for hit in hit_rows:
        vector = hit_vectors.get(int(hit["id"]))
        if not vector:
            continue
        cos = vector_cosine(own_vec, vector)
        # Degenerate-vector guard: byte-identical embeddings carry zero signal — a
        # byte-identical hit vector carries ZERO discrimination — fake test
        # embedders collapse distinct texts onto one constant vector, a real
        # model never does. Treat as "no cosine evidence": pass the pair
        # through (at cos 1.0) and let decide_evidence's duplicate routes
        # settle it, instead of filtering on testimony the embedder cannot
        # actually give.
        if vector == own_vec:
            passed.append((hit, 1.0))
            continue
        if cos < SEMANTIC_CANDIDATE_COS_FLOOR:
            below_floor.append((hit, cos))
        elif cos >= SEMANTIC_CANDIDATE_COS_CEIL:
            at_ceil.append((hit, cos))
        else:
            passed.append((hit, cos))
    return passed, below_floor, at_ceil


# ── ②″ memory-level screen (G5) ─────────────────────────────────────────────

# Release-record shape: a version token (v-prefixed dotted number, bare
# dotted number, or a year) PLUS release wording — "0.16.12 发版闭环",
# "v0.9.2 发版记录", "[已上线 v0.8.5] G6". A year counts as a version token
# (owner 测试预期: 「2024 vs 2025 规划」毙).
_RELEASE_VERSION_TOKEN = re.compile(
    r"(?:v\d+(?:\.\d+)+|\d+\.\d+(?:\.\d+)+|(?:19|20)\d{2})",
    re.IGNORECASE,
)
_VERSION_TOKEN_SPLIT = re.compile(
    r"v\d+(?:\.\d+)*|\d+(?:\.\d+)+|(?:19|20)\d{2}",
    re.IGNORECASE,
)


def _subject_version_primary(subject: str) -> "tuple[int, ...] | None":
    """The subject's release/version identity as a numeric tuple (max wins),
    None when the subject carries no version token at all. Covers the
    lineage marker form (方案 v2), release records (v0.9.2 发版记录) and
    plain year planning (2024 规划)."""
    from ..semantic_conflict import _lineage_primary_version

    lineage = _lineage_primary_version(subject)
    if lineage is not None:
        return lineage
    tokens = _RELEASE_VERSION_TOKEN.findall(subject or "")
    if not tokens:
        return None
    # IGNORECASE makes "V2.5" a match; lstrip must take both cases or
    # int("V2...") raises and poisons the whole job (实施后对抗 review P0).
    return max(
        tuple(int(part) for part in token.casefold().lstrip("v").split("."))
        for token in tokens
    )


def _subject_is_process_record(hit_subject: str, own_subject: str) -> bool:
    """Process-record guard: either subject carrying the process shape
    (review rounds / design→release / re-verify) disqualifies the PAIR —
    whole-text vetoes lose their context once a memory is split into
    sentences (cf-noise-18298/18278/18218 async FPs). Lives here (gates) so
    the memory-level screen calls it once per PEER, not once per hit."""
    from ..semantic_conflict import (
        _PROCESS_DESIGN_RE,
        _PROCESS_RELEASE_RE,
        _PROCESS_REVERIFY_RE,
        _PROCESS_REVIEW_RE,
    )
    for subject in (hit_subject, own_subject):
        if not subject:
            continue
        if _PROCESS_REVIEW_RE.search(subject) or _PROCESS_REVERIFY_RE.search(subject):
            return True
        if _PROCESS_DESIGN_RE.search(subject) and _PROCESS_RELEASE_RE.search(subject):
            return True
    return False


def memory_pair_excluded(
    own_subject: str, own_tags: "list[str] | None",
    peer_subject: str, peer_tags: "list[str] | None",
) -> bool:
    """②″ 记忆级一揽子筛选（方案 G5，标题/tags 可判的全部前置）。

    Three vetoes, one call per peer:
    1+3. 版本对立/发版方案形态：BOTH sides carry a version identity, the
         primary versions differ, and the version-stripped stems match —
         same topic at a different generation is timeline evolution, never a
         competing claim. Different stems are different topics (cf-res-9
         stays); same primary version is same-generation text (cf-res-30
         stays). Single-sided version shapes stay in (cf-res-10 stays).
    2. 过程记录：either side's subject carries the process shape
         (review rounds / design→release / re-verify) — moved from the
         sentence loop where it re-judged the same peer per hit.
    配方对立 (#50) hits none of these: the release-shaped subject has no
    version token on both sides with matching stems, and the opposing
    sentences are compared at the sentence layer.
    """
    own_s = str(own_subject or "")
    peer_s = str(peer_subject or "")
    own_primary = _subject_version_primary(own_s)
    peer_primary = _subject_version_primary(peer_s)
    if own_primary is not None and peer_primary is not None and own_primary != peer_primary:
        own_stem = "".join(_VERSION_TOKEN_SPLIT.sub("", own_s).casefold().split())
        peer_stem = "".join(_VERSION_TOKEN_SPLIT.sub("", peer_s).casefold().split())
        if own_stem == peer_stem:
            return True
    if _subject_is_process_record(peer_s, own_s):
        return True
    return False


def compute_pair_score(
    decision: Any, pair_cos: float, unit_text: str, hit_text: str,
) -> float:
    """⑦ pair_score（G6 重写，纯函数）：预算消费顺序，永不改变判定。

    score = 0.40*band(clamp((cos-FLOOR)/(CEIL-FLOOR))) + 0.25*numeric_route
          + 0.20*values_differ(normalized unequal) + 0.15*negation(单侧)
    Negation opposition = the G4 negation vocab hitting EXACTLY ONE side.
    Value-equal pairs are settled at adjudication — the bonus is only for
    opposing evidence; text pairs of undecided equality score 0. Weights are
    initial values — recalibrated at G7 (五轮基线)."""
    from ..constants import (
        PAIR_SCORE_W_CONFLICT_BAND,
        PAIR_SCORE_W_NEGATION,
        PAIR_SCORE_W_NUMERIC_ROUTE,
        PAIR_SCORE_W_VALUES_DIFFER,
        SEMANTIC_CANDIDATE_COS_CEIL,
        SEMANTIC_CANDIDATE_COS_FLOOR,
    )
    from ..semantic_conflict import _NEGATION_WORDS, normalize_value

    band = max(0.0, min(
        1.0,
        (float(pair_cos) - SEMANTIC_CANDIDATE_COS_FLOOR)
        / (SEMANTIC_CANDIDATE_COS_CEIL - SEMANTIC_CANDIDATE_COS_FLOOR),
    ))
    score = PAIR_SCORE_W_CONFLICT_BAND * band
    if str(getattr(decision, "reason", "") or "") == "numeric_value_candidate":
        score += PAIR_SCORE_W_NUMERIC_ROUTE
    left_value = getattr(decision, "left_value", None)
    right_value = getattr(decision, "right_value", None)
    if left_value and right_value:
        try:
            if normalize_value(left_value) != normalize_value(right_value):
                score += PAIR_SCORE_W_VALUES_DIFFER
        except Exception:
            score += PAIR_SCORE_W_VALUES_DIFFER
    pattern = re.compile(_NEGATION_WORDS, re.IGNORECASE)
    if bool(pattern.search(unit_text or "")) != bool(pattern.search(hit_text or "")):
        score += PAIR_SCORE_W_NEGATION
    return score
