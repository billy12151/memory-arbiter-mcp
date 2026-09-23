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

from typing import Any, Iterator

from ..constants import (
    SEMANTIC_CANDIDATE_COS_CEIL,
    SEMANTIC_CANDIDATE_COS_FLOOR,
)
from ..semantic_conflict import _SENT_PREFILTER, vector_cosine


def claim_value_spans(content: str, claims: "list[dict[str, Any]]") -> "list[tuple[int, int]]":
    """Locate each claim's value inside the memory body (gate-v2 G4, owner
    拍板: claims 覆盖句跳过). The value contract is a verbatim contiguous
    slice of the content, so a literal find pins its offset; POSITION-based
    matching (not value equality) prevents a same-value-elsewhere sentence
    from being wrongly treated as covered. Values that cannot be located
    (legacy/edited rows) simply contribute no span."""
    spans: "list[tuple[int, int]]" = []
    if not content:
        return spans
    for claim in claims or []:
        value = str(claim.get("value") or "")
        if not value:
            continue
        start = content.find(value)
        if start >= 0:
            spans.append((start, start + len(value)))
    return spans


def row_prefilter(
    rows: "list[Any]", claim_spans: "tuple[tuple[int, int], ...] | list[tuple[int, int]]" = (),
) -> "Iterator[Any]":
    """③ 句子初筛（编排可选层：写入调用、扫描跳过）— yield rows that carry
    an extractable value, a negation word, a time anchor, or are table rows;
    pure prose rows never originate a KNN query. A row OVERLAPPED by any
    claim value span is also dropped (claims 覆盖句跳过): its content is
    already represented in channel B as a structured claim, and keeping it
    would double-report the same statement through channel A. Rows with a
    value but NO covering claim stay (未被覆盖不跳)."""
    for row in rows:
        if str(getattr(row, "kind", "") or "") == "table_row":
            yield row
            continue
        start = int(getattr(row, "start_offset", 0))
        end = int(getattr(row, "end_offset", 0))
        if any(span_start < end and start < span_end for span_start, span_end in claim_spans):
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
        # Degenerate-vector guard (same doctrine as _attr_cos_or_none): a
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
