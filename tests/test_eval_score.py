"""eval/score.py unit coverage — P0-1 c5 (plan §2.7).

Synthetic raw payloads (no models): metric arithmetic, count+rate pairing,
shape join from the pair corpus, and gate relative-drop semantics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

import score  # noqa: E402


def test_recall_metrics_arithmetic() -> None:
    raw = {
        "queries": [
            {"qid": "A01", "kind": "paraphrase", "hits": [
                {"fixture_key": "t-a"}, {"fixture_key": "t-b"}, {"fixture_key": "t-x"}]},
            {"qid": "C01", "kind": "legal", "hits": [{"fixture_key": "t-a"}]},
        ],
        "self_recall": [
            {"fixture_key": "t-a", "in_top10": True},
            {"fixture_key": "t-b", "in_top10": False},
        ],
    }
    # 把 A01 的 relevant 定为 t-a 与 t-b（labels 覆盖断言不在此测——直接注入映射）
    original = score._load_jsonl

    def _stub(path: Path) -> list[dict]:
        if path.name == "labels.jsonl" and "recall" in str(path):
            return [
                {"qid": "A01", "fixture_key": "t-a", "label": "relevant"},
                {"qid": "A01", "fixture_key": "t-b", "label": "relevant"},
                {"qid": "B01", "fixture_key": "t-a", "label": "borderline"},
            ]
        return original(path)

    score._load_jsonl = _stub
    try:
        result = score.score_recall(raw)
    finally:
        score._load_jsonl = original
    assert result["recall_at_5"] == {"hits": 2, "total": 2, "rate": 1.0}
    assert result["mrr"]["value"] == 1.0
    # C01（legal）误召回 t-a（relevant）1 条，t-x 无标注=无关不计
    assert result["irrelevant_false_pulls"]["count"] == 1
    assert result["self_recall_top10"] == {"count": 1, "total": 2, "rate": 0.5}


def test_similarity_count_rate_pairing() -> None:
    raw = {"similarity": {"anchor_ids": {}, "cases": [
        {"label": "true_near_dup", "fired": True, "hit_anchor": True},
        {"label": "true_near_dup", "fired": False, "hit_anchor": False},
        {"label": "clearly_different", "fired": False, "hit_anchor": False},
    ]}}
    result = score.score_similarity(raw)
    assert result["hint_total"] == {"count": 1, "total": 3, "rate": round(1 / 3, 4)}
    near = result["by_label"]["true_near_dup"]["fired"]
    assert near == {"count": 1, "total": 2, "rate": 0.5}


def test_conflict_shape_join_and_outcomes() -> None:
    raw = {"conflict": [
        {"pair_id": "cf-oppo-01", "label": "true_conflict", "skipped_member_replay": False,
         "sync": True, "async": False, "notice_missing": False},
        {"pair_id": "cf-oppo-02", "label": "true_conflict", "skipped_member_replay": False,
         "sync": False, "async": False, "notice_missing": True},
        {"pair_id": "cf-res-9-759-767", "label": "true_conflict", "skipped_member_replay": False,
         "sync": False, "async": True, "notice_missing": False},
        {"pair_id": "cf-coexist-377-382", "label": "coexist", "skipped_member_replay": False,
         "sync": False, "async": False, "notice_missing": True},
        {"pair_id": "cf-any", "label": "noise", "skipped_member_replay": True,
         "sync": None, "async": None, "notice_missing": None},
    ]}
    result = score.score_conflict(raw)
    assert result["overall"]["recall"] == {"count": 2, "total": 3, "rate": round(2 / 3, 4)}
    assert result["overall"]["skipped_member_replay"] == 1
    shapes = result["by_shape"]
    assert shapes["write_opposition"]["n"] == 2
    assert shapes["write_opposition"]["sync"]["count"] == 1
    assert shapes["scan_evolution"]["async"]["count"] == 1
    assert shapes["governed_negative"]["n"] == 1


def test_gate_relative_drop_semantics() -> None:
    baseline = {"recall": {"recall_at_10": {"rate": 0.95}}, "similarity": {"hint_total": {"rate": 0.33}}}
    current_ok = {"recall": {"recall_at_10": {"rate": 0.90}}, "similarity": {"hint_total": {"rate": 0.33}}}
    current_bad = {"recall": {"recall_at_10": {"rate": 0.50}}, "similarity": {"hint_total": {"rate": 0.33}}}
    assert score.gate(current_ok, baseline)["gate"] == "PASSED"
    failed = score.gate(current_bad, baseline)
    assert failed["gate"] == "FAILED"
    assert failed["failures"][0]["metric"] == "recall.recall_at_10.rate"
    assert failed["failures"][0]["relative_drop"] == round((0.95 - 0.50) / 0.95, 4)
