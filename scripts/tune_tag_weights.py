"""Tune tag-scoring weights against synthetic ground truth (v0.7.3).

Goal
----
id=210 says tags are *more precise* than subject, yet the legacy weights
(``_TAGS_STRONG_WEIGHT = 7.0``) sit below subject (``10.0``). The result is
that a memory whose tags precisely contain both query tokens (e.g. id=206)
scores *lower* than a memory whose subject merely contains one of them
(e.g. id=105), even after change 1 fixed the tag algorithm.

This script builds a synthetic corpus where ground truth relevance is
*known by construction*, then sweeps tag-weight settings and reports
which setting best matches the ground-truth ordering. We don't claim the
synthetic distribution mirrors production — we only claim it covers the
hard cases (tag-precise-but-subject-absent vs subject-incidental) that
motivated the change. Relative comparison between weight settings on the
same corpus is the point.

Metrics
-------
- pairwise accuracy: over all (A, B) pairs where ground-truth says A > B,
  fraction where the model ranks A above B. The hardest, most interpretable.
- nDCG@10: ranking quality of the top 10 per query.
- MRR: mean reciprocal rank of the single most-relevant result.
- recall@3 / recall@10: is the top relevant item inside the top-k?

Corpus design (4 memory archetypes, all with identical trust/recency/content
so final_score differences come only from subject+tag scoring):
- A) TAG_PRECISE (the id=206 hero): subject avoids the query words; tags
     contain BOTH query tokens. Ground truth: HIGH relevance. Hard case:
     only tag-strong can lift it.
- B) SUBJ INCIDENTAL (the id=105 villain): subject contains one query token
     incidentally as a suffix; tags contain that same one token. Ground
     truth: MEDIUM. Today's bug: subject-medium (6.0) + tag-medium (4.0)
     = 10.0 beats A's tag-strong (7.0).
- C) BOTH (control, "easy high"): subject contains both, tags contain both.
     Ground truth: HIGH. Should always rank top regardless of weights.
- D) NOISE: subject/tags share no query token. Ground truth: NONE. Should
     never rank above A/B/C.

Each query has one of each archetype (4 candidates, +more noise to fill out
the pool). Ground truth ordering: C ≈ A > B > D. The pairwise generator
emits (C>A, C>B, C>D, A>B, A>D, B>D) per query. The interesting pair is
A>B: today's weights get it WRONG (B's 10.0 > A's 7.0), better weights
should flip it.

Usage
-----
    python scripts/tune_tag_weights.py [--n 300] [--seed 7]

Outputs a table: one row per weight setting, columns = metrics. The
"decisive" metric is pairwise accuracy on A>B pairs specifically.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make the package importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_arbiter.search import (
    _score_surface,
    _score_tags_surface,
    extract_anchors,
    score_anchor_overlap,
    _SUBJECT_STRONG_WEIGHT,
    _SUBJECT_MEDIUM_WEIGHT,
    _SUBJECT_WEAK_WEIGHT,
    _SUBJECT_SCORE_CAP,
    _TAGS_SCORE_CAP,
    _CONTENT_SCORE_CAP,
    _CONTENT_ONLY_PENALTY,
    _LONG_CONTENT_PENALTY,
    _trust_bonus,
    _recency_bonus,
)
from memory_arbiter.anchors import Anchor, AnchorMatch, score_anchor_overlap


# ---- parameterized classify_match_level ---------------------------------
# We re-implement classify_match_level here so we can sweep its thresholds
# without monkey-patching the source. Mirrors anchors.py:201 logic exactly,
# except medium_total_hits and medium_coverage are parameters.
#
# Original (anchors.py):
#   medium if (specific_hits >= 1 AND total_hits >= 2) OR specific_coverage >= 0.4
#   weak   if total_hits >= 1
#
# Why: query "v0.7.2 发版" has specific_count=2. A subject that hits only
# "发版" gets specific_coverage=0.5 >= 0.4 → medium. But that subject only
# matches HALF the query — medium (6.0) over-rewards incidental surface hits
# (the id=105 "[已完成] README ... (v0.4.0 发版)" case). Tightening either
# threshold drops those to weak, which is the C fix under test.

def classify_match_level_param(
    query_anchors: list[Anchor],
    matches: dict[str, AnchorMatch],
    *,
    medium_total_hits: int = 2,
    medium_coverage: float = 0.4,
) -> str:
    summary = matches.get("_summary")
    if summary is None:
        return "none"
    specific_hits = summary.specific_hits
    total_hits = summary.total_hits
    query_specific_count = sum(1 for a in query_anchors if not a.is_generic)
    specific_coverage = (
        specific_hits / query_specific_count if query_specific_count else 0.0
    )
    if specific_hits == 0 and total_hits == 0:
        return "none"
    if specific_hits >= 1 and total_hits >= medium_total_hits:
        return "medium"
    if specific_coverage >= medium_coverage:
        return "medium"
    if total_hits >= 1:
        return "weak"
    return "none"


def _score_surface_param(
    query_anchors: list[Anchor],
    surface_text: str,
    strong_weight: float,
    medium_weight: float,
    weak_weight: float,
    cap: float,
    query_lower: str,
    *,
    medium_total_hits: int = 2,
    medium_coverage: float = 0.4,
) -> tuple[float, str]:
    """Drop-in for _score_surface with parameterized subject thresholds."""
    if not surface_text:
        return 0.0, "none"
    surface_lower = surface_text.lower()
    if query_lower and query_lower in surface_lower:
        return min(strong_weight, cap), "strong"
    surface_anchors = extract_anchors(surface_text)
    matches = score_anchor_overlap(query_anchors, surface_anchors)
    level = classify_match_level_param(
        query_anchors, matches,
        medium_total_hits=medium_total_hits,
        medium_coverage=medium_coverage,
    )
    if level == "medium":
        return min(medium_weight, cap), level
    if level == "weak":
        return min(weak_weight, cap), level
    return 0.0, level


# ---- synthetic vocabulary ------------------------------------------------
# Two disjoint pools so we can build queries with known token overlap.
# CJK words and ASCII version-like identifiers; mixed queries use one of each.

CJK_WORDS = [
    "发版", "决策", "偏好", "部署", "迁移", "修复", "审计", "定位",
    "回归", "诊断", "调优", "重构", "拆分", "合并", "回滚", "监控",
    "告警", "限流", "缓存", "索引", "锁", "事务", "幂等", "补偿",
    "灰度", "熔断", "降级", "路由", "鉴权", "授权", "加密", "签名",
]

VERSIONS = [
    "v0.7.2", "v0.7.3", "v1.2.0", "v2.0.1", "v0.5.4", "v1.0.0",
    "v3.4.5", "v0.8.1", "v2.3.0", "v1.5.2", "v0.6.0", "v4.1.0",
]

# Filler words for subject/content (semantically neutral, never in queries)
SUBJ_FILLERS = [
    "实现记录", "设计说明", "操作手册", "复盘", "总结", "笔记",
    "概述", "规格", "草案", "变更", "更新", "调整",
]


@dataclass
class Memory:
    """A synthetic memory record. Only the fields used by scoring are set."""
    subject: str
    tags: list[str]
    content: str = ""
    source_type: str = "agent_generated"
    ingest_time: str = "2026-07-01T00:00:00+00:00"  # uniform → zero recency delta
    status: str = "active"
    protection_level: str = "normal"
    confidence: float = 0.5
    split_status: Optional[str] = None  # for long_content_penalty exemption test
    # ground-truth bucket: HIGH / MEDIUM / NONE
    gt: str = "NONE"
    # archetype label for diagnosis
    archetype: str = "NOISE"


@dataclass
class QueryCase:
    query: str
    candidates: list[Memory]  # includes the 4 archetypes + noise
    # ordered ground-truth: which gt buckets are considered "relevant" for metrics
    relevant_buckets: tuple[str, ...] = ("HIGH",)


# ---- corpus generation ---------------------------------------------------

def _rand_version(rng: random.Random) -> str:
    return rng.choice(VERSIONS)


def _rand_cjk(rng: random.Random) -> str:
    return rng.choice(CJK_WORDS)


def _make_query(rng: random.Random) -> tuple[str, list[str]]:
    """Return (query, tokens). Query has 2-3 tokens (mix of version + cjk).

    2-token: 'v0.7.2 发版' (specific_count=2)
    3-token: 'v0.7.2 发版 决策' (specific_count=3) — harder, lets us
             distinguish coverage 0.67 (2/3) from 0.33 (1/3).
    """
    v = _rand_version(rng)
    w1 = _rand_cjk(rng)
    if rng.random() < 0.5:
        # 2-token query
        return f"{v} {w1}", [v, w1]
    # 3-token query
    w2 = _rand_cjk(rng)
    while w2 == w1:
        w2 = _rand_cjk(rng)
    return f"{v} {w1} {w2}", [v, w1, w2]


def _build_candidates_for_query(rng: random.Random, query: str, tokens: list[str],
                                 noise_count: int = 4) -> list[Memory]:
    """Build archetypes + noise. tokens = query tokens (2 or 3).

    For 2-token queries, archetypes as before (A=tag-both, B=subj-1, C=subj-both).
    For 3-token queries we add finer-grained archetypes that exercise the
    coverage threshold's gray zone:
      - SUBJ_2_OF_3: subject hits 2/3 (coverage 0.67) — does T2=0.6 keep it medium?
      - SUBJ_1_OF_3: subject hits 1/3 (coverage 0.33) — should drop to weak.
    """
    cands: list[Memory] = []
    n_tok = len(tokens)
    # pick a cjk token and the version token from the query for archetype construction
    ver_tok = next((t for t in tokens if t[0] == "v"), tokens[0])
    cjk_toks = [t for t in tokens if t[0] != "v"]

    def _safe_filler(*forbidden: str) -> str:
        f = rng.choice(SUBJ_FILLERS)
        while any(fb in f for fb in forbidden):
            f = rng.choice(SUBJ_FILLERS)
        return f

    def _other_cjk(*forbidden: str) -> str:
        c = _rand_cjk(rng)
        while c in forbidden:
            c = _rand_cjk(rng)
        return c

    # A) TAG_PRECISE — subject avoids ALL query tokens; tags contain ALL of them.
    subj_a_parts = ["memory-arbiter", _safe_filler(*tokens)]
    subj_a = " ".join(subj_a_parts)
    while any(t in subj_a for t in tokens):
        subj_a = f"memory-arbiter {_safe_filler(*tokens)}"
    cands.append(Memory(
        subject=subj_a,
        tags=list(tokens) + ["memory-arbiter", _other_cjk(*tokens)],
        archetype="TAG_PRECISE",
        gt="HIGH",
    ))

    # B) SUBJ_INCIDENTAL — subject contains exactly ONE cjk query token; tags
    #    contain that same token. (id=105 shape — subject suffix mention)
    incident_cjk = cjk_toks[0]
    subj_b = f"{_safe_filler(*tokens)} {incident_cjk}"
    cands.append(Memory(
        subject=subj_b,
        tags=[incident_cjk, _other_cjk(incident_cjk), _safe_filler()],
        archetype="SUBJ_INCIDENTAL",
        gt="MEDIUM",
    ))

    # C) BOTH — subject + tags both contain ALL query tokens. Easy HIGH.
    subj_c = " ".join(tokens) + f" {_safe_filler(*tokens)}"
    cands.append(Memory(
        subject=subj_c,
        tags=list(tokens) + [_other_cjk(*tokens)],
        archetype="BOTH",
        gt="HIGH",
    ))

    # For 3-token queries: two more subject-overlap archetypes that exercise
    # the coverage threshold's gray zone.
    if n_tok == 3:
        # SUBJ_2_OF_3 — subject contains 2 of 3 tokens (coverage 0.67).
        # Ground truth MEDIUM: it matches more than half the query, so it's
        # a real partial match — medium is the right call. T2=0.6 should keep
        # medium; T2=0.7 would wrongly drop it to weak.
        two_tokens = tokens[:2]
        subj_2 = " ".join(two_tokens) + f" {_safe_filler(*tokens)}"
        # ensure the third token is NOT in subj_2
        while tokens[2] in subj_2:
            subj_2 = " ".join(two_tokens) + f" {_safe_filler(*tokens)}"
        cands.append(Memory(
            subject=subj_2,
            tags=[two_tokens[0], _other_cjk(*tokens), _safe_filler()],
            archetype="SUBJ_2_OF_3",
            gt="MEDIUM",
        ))
        # SUBJ_1_OF_3 — subject contains 1 of 3 (coverage 0.33). Ground truth
        # WEAK-ish; we mark MEDIUM-LOW by bucketing as MEDIUM but expect it to
        # lose to SUBJ_2_OF_3 (which has higher coverage). Actually for cleaner
        # ground truth: 1/3 is clearly less relevant than 2/3, so we make
        # SUBJ_1_OF_3 gt=MEDIUM but with the pairwise constraint that
        # SUBJ_2_OF_3 > SUBJ_1_OF_3.
        one_token = tokens[0:1]
        subj_1 = " ".join(one_token) + f" {_safe_filler(*tokens)} {_safe_filler(*tokens)}"
        for t in tokens[1:]:
            while t in subj_1:
                subj_1 = " ".join(one_token) + f" {_safe_filler(*tokens)} {_safe_filler(*tokens)}"
        cands.append(Memory(
            subject=subj_1,
            tags=[tokens[0], _other_cjk(*tokens), _safe_filler()],
            archetype="SUBJ_1_OF_3",
            gt="MEDIUM",  # but lower than SUBJ_2_OF_3 in the implicit gt ordering
        ))

    # D) NOISE — no overlap
    nc1, nc2 = _other_cjk(*tokens), _other_cjk(*tokens)
    while nc2 == nc1:
        nc2 = _other_cjk(*tokens)
    cands.append(Memory(
        subject=f"{nc1} {nc2} {_safe_filler(*tokens)}",
        tags=[nc1, nc2, _safe_filler()],
        archetype="NOISE",
        gt="NONE",
    ))

    # E) LONG_CONTENT_HIT — subject/tags miss everything, but content contains
    #    the query as substring AND content is long (>2000 chars). Ground truth
    #    NONE (it's an incidental mention in a long doc). This archetype is the
    #    canary for long_content_penalty: if C fix accidentally drops subject
    #    from medium→weak on some other candidate, that candidate could newly
    #    trigger the penalty. We want E to stay ranked below MEDIUM/HIGH.
    long_content = " ".join([_safe_filler(*tokens)] * 200)  # >2000 chars
    long_content = long_content + " " + query  # query as substring at the end
    cands.append(Memory(
        subject=f"{_other_cjk(*tokens)} {_safe_filler(*tokens)}",
        tags=[_other_cjk(*tokens), _safe_filler()],
        content=long_content,
        archetype="LONG_CONTENT_HIT",
        gt="NONE",
    ))

    # F) SUBJ_HALF_WITH_CONTENT — subject hits half the query (like B), tags
    #    hit one token, AND content contains the query. Ground truth MEDIUM.
    #    This tests whether C fix + content scoring keeps the partial-subject
    #    + content case ranked correctly (it should beat pure NOISE and
    #    LONG_CONTENT_HIT but lose to TAG_PRECISE/BOTH).
    subj_f = " ".join(tokens[:1]) + f" {_safe_filler(*tokens)}"
    for t in tokens[1:]:
        while t in subj_f:
            subj_f = " ".join(tokens[:1]) + f" {_safe_filler(*tokens)}"
    cands.append(Memory(
        subject=subj_f,
        tags=[tokens[0], _other_cjk(*tokens), _safe_filler()],
        content=f"{_safe_filler(*tokens)} {query} {_safe_filler(*tokens)}",
        archetype="SUBJ_HALF_WITH_CONTENT",
        gt="MEDIUM",
    ))

    # Extra noise
    for _ in range(noise_count):
        m1, m2 = _other_cjk(*tokens), _other_cjk(*tokens)
        while m2 == m1:
            m2 = _other_cjk(*tokens)
        cands.append(Memory(
            subject=f"{m1} {m2} {_safe_filler(*tokens)}",
            tags=[m1, m2, _safe_filler()],
            archetype="NOISE",
            gt="NONE",
        ))

    rng.shuffle(cands)
    return cands


def generate_corpus(n: int, seed: int) -> list[QueryCase]:
    rng = random.Random(seed)
    cases: list[QueryCase] = []
    for _ in range(n):
        query, tokens = _make_query(rng)
        cands = _build_candidates_for_query(rng, query, tokens)
        cases.append(QueryCase(query=query, candidates=cands))
    return cases


# ---- scoring (mirrors _soft_rerank's subject+tag piece, parameterized) ---

@dataclass
class TagWeights:
    strong: float
    medium: float
    weak: float
    cap: float = 7.0  # default cap matches legacy; we also test cap=stride

    def label(self) -> str:
        return f"tag(S={self.strong}/M={self.medium}/W={self.weak}/cap={self.cap})"


def _score_one(mem: Memory, query: str, tw: TagWeights,
               *, subj_medium_total_hits: int = 2, subj_medium_coverage: float = 0.4) -> tuple[float, dict]:
    """Full _soft_rerank replica for ONE memory, parameterized on subject thresholds.

    Mirrors search.py _soft_rerank line-by-line (subject + tag + content +
    penalties + vec floor + trust + recency), so synthetic-data experiments
    catch regressions in any of those paths (not just subject+tag in isolation).
    """
    query_lower = query.lower()
    query_anchors = extract_anchors(query) if query else []
    # subject uses parameterized _score_surface_param so we can sweep thresholds
    subj_score, subj_level = _score_surface_param(
        query_anchors, mem.subject,
        _SUBJECT_STRONG_WEIGHT, _SUBJECT_MEDIUM_WEIGHT, _SUBJECT_WEAK_WEIGHT,
        _SUBJECT_SCORE_CAP, query_lower,
        medium_total_hits=subj_medium_total_hits,
        medium_coverage=subj_medium_coverage,
    )
    tag_score, tag_level, _ = _score_tags_surface(
        query, mem.tags,
        tw.strong, tw.medium, tw.weak, tw.cap,
    )
    # content: substring check + anchor fallback (mirrors _soft_rerank)
    content = mem.content or ""
    content_hit = bool(query_lower) and query_lower in content.lower()
    content_score = 0.0
    if content_hit:
        content_score = _CONTENT_SCORE_CAP
    elif query_anchors and content:
        content_anchors = extract_anchors(content)
        content_matches = score_anchor_overlap(query_anchors, content_anchors)
        cm = content_matches.get("_summary")
        if cm and cm.total_hits >= 2:
            content_score = min(_CONTENT_SCORE_CAP * 0.5, _CONTENT_SCORE_CAP)

    relevance = subj_score + tag_score + content_score

    # content_only_penalty: subject+tags both none, content hit
    subject_tags_miss = subj_level == "none" and tag_level == "none"
    if subject_tags_miss and content_score > 0:
        relevance -= _CONTENT_ONLY_PENALTY
    # long_content_penalty: subject+tags weak-or-none, content long, not split-active
    subject_tags_weak = subj_level in ("none", "weak") and tag_level in ("none", "weak")
    content_long = len(content) > 2000
    is_split_active = mem.split_status == "active"
    if subject_tags_weak and content_long and content_score > 0 and not is_split_active:
        relevance -= _LONG_CONTENT_PENALTY

    # vec floor not applicable (synthetic memories aren't vec candidates)
    trust = _trust_bonus({"source_type": mem.source_type, "protection_level": mem.protection_level})
    recency = _recency_bonus({"ingest_time": mem.ingest_time})
    superseded_sink = 1 if mem.status == "superseded" else 0
    final_score = relevance + trust + recency - (superseded_sink * 1000.0)
    return final_score, {
        "subject_score": subj_score, "subject_level": subj_level,
        "tag_score": tag_score, "tag_level": tag_level,
        "content_score": content_score,
        "relevance": relevance, "final": final_score,
    }


def _rank_candidates(cands: list[Memory], query: str, tw: TagWeights,
                     *, subj_medium_total_hits: int = 2, subj_medium_coverage: float = 0.4) -> list[tuple[Memory, float]]:
    """Return candidates sorted by relevance desc (tie-break stable on archetype)."""
    scored = [(c, _score_one(c, query, tw,
                             subj_medium_total_hits=subj_medium_total_hits,
                             subj_medium_coverage=subj_medium_coverage)[0]) for c in cands]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---- metrics -------------------------------------------------------------

def _archetype_rank(archetype: str) -> int:
    """Finer-grained ground-truth ordering by archetype (for pairwise).

    Rationale:
      NOISE / LONG_CONTENT_HIT (incidental mention, gt=NONE)
        < SUBJ_1_OF_3 (1/3 query coverage, weak partial)
        < SUBJ_INCIDENTAL (1/2 coverage, medium partial — id=105 shape)
        < SUBJ_HALF_WITH_CONTENT (1/2 subj + content hit, slightly stronger)
        < SUBJ_2_OF_3 (2/3 coverage, stronger partial)
        < TAG_PRECISE ≈ BOTH (full relevance).

    TAG_PRECISE and BOTH share the top rank — the A>B metric (TAG_PRECISE vs
    SUBJ_INCIDENTAL) is the interesting one; TAG_PRECISE vs BOTH is a toss-up
    we don't constrain.
    """
    return {
        "NOISE": 0,
        "LONG_CONTENT_HIT": 0,
        "SUBJ_1_OF_3": 1,
        "SUBJ_INCIDENTAL": 2,
        "SUBJ_HALF_WITH_CONTENT": 3,
        "SUBJ_2_OF_3": 4,
        "TAG_PRECISE": 5,
        "BOTH": 5,
    }.get(archetype, 0)


def _gt_relevance_for_dcg(gt: str) -> int:
    """Coarser mapping for nDCG/MRR/recall (only 3 levels: none/med/high)."""
    if gt == "NONE":
        return 0
    if gt == "HIGH":
        return 2
    return 1  # MEDIUM / SUBJ_1_OF_3 / SUBJ_2_OF_3 all count as "relevant"


def pairwise_accuracy(cases: list[QueryCase], tw: TagWeights,
                      *, subj_medium_total_hits: int = 2, subj_medium_coverage: float = 0.4) -> dict:
    """For each (A,B) where gt_rank(A) > gt_rank(B), check if model ranks A > B.

    Returns overall accuracy plus per-archetype-pair breakdown so we can see
    which specific comparison each weight setting gets right/wrong.
    """
    total = 0
    correct = 0
    pair_stats: dict[str, list[int]] = {}  # key "A>B" → [correct, total]
    for case in cases:
        ranked = _rank_candidates(case.candidates, case.query, tw,
                                  subj_medium_total_hits=subj_medium_total_hits,
                                  subj_medium_coverage=subj_medium_coverage)
        # build id→position AND id→score (for tie detection)
        pos = {id(c): i for i, (c, _) in enumerate(ranked)}
        score_of = {id(c): s for c, s in ranked}
        for a in case.candidates:
            for b in case.candidates:
                if a is b:
                    continue
                ra, rb = _archetype_rank(a.archetype), _archetype_rank(b.archetype)
                if ra <= rb:
                    continue
                # ground truth says a should rank above b
                total += 1
                key = f"{a.archetype}>{b.archetype}"
                pair_stats.setdefault(key, [0, 0])
                pair_stats[key][1] += 1
                sa, sb = score_of[id(a)], score_of[id(b)]
                if pos[id(a)] < pos[id(b)]:
                    # a clearly ahead
                    correct += 1
                    pair_stats[key][0] += 1
                elif abs(sa - sb) < 1e-9:
                    # tie — both orderings are defensible; count as half credit
                    correct += 0.5
                    pair_stats[key][0] += 0.5
    acc = correct / total if total else 0.0
    return {"accuracy": acc, "total_pairs": total, "pairs": pair_stats}


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_and_mrr(cases: list[QueryCase], tw: TagWeights, k: int = 10,
                 *, subj_medium_total_hits: int = 2, subj_medium_coverage: float = 0.4) -> dict:
    """nDCG@k, MRR, recall@k.

    Graded relevance: HIGH=2, MEDIUM=1, NONE=0. Per query we compute DCG over
    the model's ranking, IDCG over the ideal ranking, and average.
    """
    ndcgs: list[float] = []
    mrrs: list[float] = []
    recall_at_3: list[float] = []
    recall_at_10: list[float] = []
    for case in cases:
        ranked = _rank_candidates(case.candidates, case.query, tw,
                                  subj_medium_total_hits=subj_medium_total_hits,
                                  subj_medium_coverage=subj_medium_coverage)
        gains = [_gt_relevance_for_dcg(c.gt) for c, _ in ranked]
        # nDCG@k
        topk = gains[:k]
        dcg = _dcg(topk)
        ideal = sorted(gains, reverse=True)[:k]
        idcg = _dcg(ideal)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        # MRR — first HIGH or MEDIUM position
        mrr = 0.0
        for i, g in enumerate(gains, start=1):
            if g > 0:
                mrr = 1.0 / i
                break
        mrrs.append(mrr)
        # recall@k — any relevant in top-k
        def recall_at(n):
            top = gains[:n]
            return 1.0 if any(g > 0 for g in top) else 0.0
        recall_at_3.append(recall_at(3))
        recall_at_10.append(recall_at(10))
    return {
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
        "recall@3": sum(recall_at_3) / len(recall_at_3) if recall_at_3 else 0.0,
        "recall@10": sum(recall_at_10) / len(recall_at_10) if recall_at_10 else 0.0,
    }


# ---- sweep ---------------------------------------------------------------

def _print_row(name: str, pa: dict, nm: dict) -> None:
    pairs = pa["pairs"]
    def pair_acc(key):
        c, t = pairs.get(key, (0, 0))
        return f"{c/t:.3f}" if t else "  n/a"
    print(
        f"{name:<36} "
        f"{pa['accuracy']:>9.4f} "
        f"{pair_acc('TAG_PRECISE>SUBJ_INCIDENTAL'):>6} "      # A>B
        f"{pair_acc('BOTH>SUBJ_INCIDENTAL'):>6} "              # C>B
        f"{pair_acc('SUBJ_INCIDENTAL>NOISE'):>6} "             # B>D
        f"{pair_acc('SUBJ_2_OF_3>SUBJ_INCIDENTAL'):>6} "       # 2>B
        f"{pair_acc('TAG_PRECISE>LONG_CONTENT_HIT'):>6} "      # A>E
        f"{pair_acc('SUBJ_INCIDENTAL>LONG_CONTENT_HIT'):>7} "  # B>E
        f"{nm['ndcg@10']:>8.4f} "
        f"{nm['mrr']:>6.4f} "
        f"{nm['recall@3']:>6.4f} "
        f"{nm['recall@10']:>6.4f}"
    )


def _print_header() -> None:
    header = (
        f"{'setting':<36} {'pairwise':>9} {'A>B':>6} {'C>B':>6} {'B>D':>6} "
        f"{'2>B':>6} {'A>E':>6} {'B>E':>7} {'ndcg@10':>8} {'mrr':>6} {'r@3':>6} {'r@10':>6}"
    )
    print(header)
    print("-" * len(header))


def _print_legend() -> None:
    print()
    print("解读（所有 pair 都应 = 1.000）：")
    print("  pairwise = 所有 gt 明确的 pair 里，模型排对的比例（越高越好）")
    print("  A>B = TAG_PRECISE 赢 SUBJ_INCIDENTAL（id=206 vs id=105，核心 bug）")
    print("  C>B = BOTH 赢 SUBJ_INCIDENTAL（控制组）")
    print("  B>D = SUBJ_INCIDENTAL 赢 NOISE（subject 降到 weak 后不应沉到噪音下）")
    print("  2>B = SUBJ_2_OF_3 赢 SUBJ_INCIDENTAL（2/3 命中应强于 1/2）")
    print("  A>E = TAG_PRECISE 赢 LONG_CONTENT_HIT（content-only 噪音不应反超 tag strong）")
    print("  B>E = SUBJ_INCIDENTAL 赢 LONG_CONTENT_HIT（C 修复后不应让 B 被 long_content 反超）")


def sweep_subject_thresholds(cases: list[QueryCase]) -> None:
    """Sweep subject medium thresholds with tag weights held at legacy (7/4/1.5).

    The C fix under test: tighten classify_match_level's medium conditions so
    that matching only HALF the query's specific anchors (1/2 → coverage 0.5)
    no longer earns subject medium (6.0).
    """
    n = len(cases)
    tw_legacy = TagWeights(7.0, 4.0, 1.5, cap=7.0)
    print(f"\n=== Subject threshold sweep (n={n}, tag weights = legacy 7/4/1.5) ===")
    print("    medium 条件 = (specific_hits≥1 AND total_hits≥T1) OR specific_coverage≥T2")
    print("    T1 = total_hits 阈值, T2 = coverage 阈值 (原: T1=2, T2=0.4)")
    _print_header()
    # current baseline
    name = "baseline (T1=2,T2=0.4)"
    pa = pairwise_accuracy(cases, tw_legacy, subj_medium_total_hits=2, subj_medium_coverage=0.4)
    nm = ndcg_and_mrr(cases, tw_legacy, subj_medium_total_hits=2, subj_medium_coverage=0.4)
    _print_row(name, pa, nm)
    # tighten coverage: 0.4 → 0.6 / 0.7 / 0.8 (1/2=0.5 drops below 0.6)
    for cov in [0.5, 0.6, 0.7, 0.8, 1.0]:
        name = f"T2={cov:.1f} (T1=2)"
        pa = pairwise_accuracy(cases, tw_legacy, subj_medium_total_hits=2, subj_medium_coverage=cov)
        nm = ndcg_and_mrr(cases, tw_legacy, subj_medium_total_hits=2, subj_medium_coverage=cov)
        _print_row(name, pa, nm)
    # also try raising T1 (total_hits threshold)
    for t1 in [3, 4]:
        name = f"T1={t1} (T2=0.4)"
        pa = pairwise_accuracy(cases, tw_legacy, subj_medium_total_hits=t1, subj_medium_coverage=0.4)
        nm = ndcg_and_mrr(cases, tw_legacy, subj_medium_total_hits=t1, subj_medium_coverage=0.4)
        _print_row(name, pa, nm)
    _print_legend()


def sweep_tag_weights_under_c_fix(cases: list[QueryCase], subj_cov: float, subj_t1: int) -> None:
    """After fixing C (subject coverage threshold), sweep tag weights to see
    if tag still needs adjustment in the fair environment.

    subj_cov/subj_t1 = the C setting chosen from sweep_subject_thresholds.
    """
    n = len(cases)
    print(f"\n=== Tag weight sweep under C fix (n={n}, subject T1={subj_t1}, T2={subj_cov}) ===")
    settings: list[tuple[str, TagWeights]] = [
        ("legacy tag (7/4/1.5)",     TagWeights(7.0, 4.0, 1.5, cap=7.0)),
        ("parity tag (10/6/2)",      TagWeights(10.0, 6.0, 2.0, cap=10.0)),
        ("tag slightly high (10.5/6/2)",  TagWeights(10.5, 6.0, 2.0, cap=10.5)),
        ("tag S-high M-parity (11/6/2)",  TagWeights(11.0, 6.0, 2.0, cap=11.0)),
        ("tag high (11/7/3)",        TagWeights(11.0, 7.0, 3.0, cap=11.0)),
    ]
    _print_header()
    for name, tw in settings:
        pa = pairwise_accuracy(cases, tw, subj_medium_total_hits=subj_t1, subj_medium_coverage=subj_cov)
        nm = ndcg_and_mrr(cases, tw, subj_medium_total_hits=subj_t1, subj_medium_coverage=subj_cov)
        _print_row(name, pa, nm)
    _print_legend()


def sweep_a_plan_vs_alternatives(cases: list[QueryCase]) -> None:
    """Head-to-head: baseline / C-only / tag-only / A plan (C + tag parity).

    The decisive comparison: does the A plan (C fix + tag parity) win on every
    pair metric without losing ground anywhere?
    """
    n = len(cases)
    print(f"\n=== A 方案 vs 备选（n={n}，含 content/long_content 干扰）===")
    print("  baseline    = 不改（subject T2=0.4, tag 7/4/1.5）")
    print("  C-only      = subject T2=0.6, tag 不动")
    print("  tag-only    = subject 不动, tag 10/6/2")
    print("  A plan      = subject T2=0.6 + tag 10/6/2  ← 推荐")
    _print_header()
    configs = [
        ("baseline (T2=0.4, tag 7/4/1.5)",  0.4, TagWeights(7.0, 4.0, 1.5, cap=7.0)),
        ("C-only (T2=0.6, tag 7/4/1.5)",    0.6, TagWeights(7.0, 4.0, 1.5, cap=7.0)),
        ("tag-only (T2=0.4, tag 10/6/2)",   0.4, TagWeights(10.0, 6.0, 2.0, cap=10.0)),
        ("A plan (T2=0.6, tag 10/6/2)",     0.6, TagWeights(10.0, 6.0, 2.0, cap=10.0)),
    ]
    for name, cov, tw in configs:
        pa = pairwise_accuracy(cases, tw, subj_medium_coverage=cov)
        nm = ndcg_and_mrr(cases, tw, subj_medium_coverage=cov)
        _print_row(name, pa, nm)
    _print_legend()



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="查询数（默认 300）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=["subject", "tag", "both", "a_plan"], default="a_plan",
                    help="subject=只扫 subject 阈值（tag 固定 legacy）；tag=修了 C 后扫 tag 权重；both=先 subject 再 tag")
    ap.add_argument("--subj-cov", type=float, default=0.6,
                    help="tag/both 模式下用的 subject coverage 阈值（默认 0.6）")
    ap.add_argument("--subj-t1", type=int, default=2,
                    help="tag/both 模式下用的 subject total_hits 阈值（默认 2）")
    args = ap.parse_args()

    cases = generate_corpus(args.n, args.seed)
    if args.mode == "a_plan":
        sweep_a_plan_vs_alternatives(cases)
    elif args.mode in ("subject", "both"):
        sweep_subject_thresholds(cases)
    if args.mode in ("tag", "both"):
        sweep_tag_weights_under_c_fix(cases, args.subj_cov, args.subj_t1)


if __name__ == "__main__":
    main()
