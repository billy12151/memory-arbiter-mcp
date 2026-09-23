"""Capped recall (gate-v2 拍板 5): denominator min(|relevant|, k).

The B03 shape: a query with MORE relevant targets than k gets punished by
the classic micro-average even when the top-k is full — capped recall reads
1.0 there while classic reads k/R. Both stay reported side by side.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

import score  # noqa: E402


def _recall_with_labels(labels: list[dict], queries: list[dict]) -> dict:
    original = score._load_jsonl

    def _stub(path: Path) -> list[dict]:
        if path.name == "labels.jsonl" and "recall" in str(path):
            return labels
        return original(path)

    score._load_jsonl = _stub
    try:
        return score.score_recall({"queries": queries}) or {}
    finally:
        score._load_jsonl = original


def test_capped_denominator_min_relevant_k() -> None:
    # B03 shape: 6 relevant targets, top-5 completely full → classic 5/6,
    # capped 5/5. A second query with 2 relevant both in top-5 scores the
    # SAME in both regimes (min(R,k)=R when R<=k) — the mixed micro-average
    # checks the summation, not just one row.
    labels = [
        {"qid": "B03", "fixture_key": f"t-{i}", "label": "relevant"}
        for i in range(6)
    ] + [
        {"qid": "A01", "fixture_key": "t-a", "label": "relevant"},
        {"qid": "A01", "fixture_key": "t-b", "label": "relevant"},
    ]
    queries = [
        {"qid": "B03", "kind": "paraphrase", "hits": [
            {"fixture_key": f"t-{i}"} for i in range(5)
        ] + [{"fixture_key": "noise"}]},
        {"qid": "A01", "kind": "paraphrase", "hits": [
            {"fixture_key": "t-a"}, {"fixture_key": "t-b"},
        ]},
    ]
    result = _recall_with_labels(labels, queries)
    assert result["recall_at_5"] == {"hits": 7, "total": 8, "rate": 0.875}
    assert result["recall_at_5_capped"] == {"hits": 7, "total": 7, "rate": 1.0}
    # per-query rows carry both regimes
    by_qid = {row["qid"]: row for row in result["per_query"]}
    assert by_qid["B03"]["recall@5"] == round(5 / 6, 4)
    assert by_qid["B03"]["recall@5_capped"] == 1.0
    assert by_qid["A01"]["recall@5"] == 1.0
    assert by_qid["A01"]["recall@5_capped"] == 1.0


def test_capped_at_10_matches_classic_when_no_r_gt_10() -> None:
    # Corpus invariant (方案 G1: R@10 不变): with no query holding more
    # than 10 relevant targets, capped@10 equals classic@10 exactly.
    labels = [
        {"qid": "Q1", "fixture_key": "t-a", "label": "relevant"},
        {"qid": "Q1", "fixture_key": "t-b", "label": "relevant"},
        {"qid": "Q1", "fixture_key": "t-c", "label": "relevant"},
    ]
    queries = [{"qid": "Q1", "kind": "paraphrase", "hits": [
        {"fixture_key": "t-a"}, {"fixture_key": "t-b"},
    ]}]
    result = _recall_with_labels(labels, queries)
    assert result["recall_at_10"] == result["recall_at_10_capped"]
    assert result["recall_at_10"]["rate"] == round(2 / 3, 4)
