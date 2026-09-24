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


def test_compute_perf_stats() -> None:
    raw = {
        "replay_perf": [
            {"fixture_key": "t-a", "elapsed_ms": 100.0, "duplicate_replay": False},
            {"fixture_key": "t-b", "elapsed_ms": 300.0, "duplicate_replay": False},
            {"fixture_key": "t-c", "elapsed_ms": 5.0, "duplicate_replay": True},
        ],
        "queries": [
            {"qid": "A01", "elapsed_ms": 50.0},
            {"qid": "A02", "elapsed_ms": 80.0},
        ],
        "conflict": [
            {"pair_id": "p1", "label": "true_conflict", "skipped_member_replay": False,
             "sync": True, "async": False, "notice_missing": False, "right_write_ms": 4200.0,
             "units": 49, "notice_count": 1,
             "_receipt": {"status": "completed", "notices_created": 1, "reasons_seen": [],
                          "pairs_examined": 3}},
            {"pair_id": "p2", "label": "coexist", "skipped_member_replay": False,
             "sync": False, "async": False, "notice_missing": True, "right_write_ms": 4600.0,
             "units": 70, "notice_count": 0,
             "_receipt": {"status": "completed", "notices_created": 0,
                          "reasons_seen": ["pairs_examined_capped"], "pairs_examined": 10}},
            {"pair_id": "p3", "label": "noise", "skipped_member_replay": True,
             "sync": None, "async": None, "notice_missing": None},
        ],
    }
    perf = score.compute_perf(raw)
    assert perf["write_ms"]["n"] == 3 and perf["write_ms"]["max_ms"] == 300.0
    assert perf["write_ms_fresh"]["n"] == 2  # 幂等重放单列
    assert perf["find_ms"]["p50_ms"] == 50.0  # nearest-rank：两样本取低位
    assert perf["conflict_right_write_ms"]["n"] == 2  # skipped 行不进耗时统计
    window = perf["conflict_window"]
    assert window["n"] == 2 and window["completed"] == 2
    assert window["avg_pairs_examined"] == 6.5
    assert window["pairs_examined_capped_rows"] == 1
    assert window["units_capped_rows"] == 1  # units=70 ≥ 64 预算
    assert window["avg_units"] == 59.5
    assert window["avg_notice_count"] == 0.5


def test_perf_keys_are_excluded_from_gate() -> None:
    # perf 是耗时信息位：基线侧带 perf、当前侧数值大幅波动，门仍 PASSED
    baseline = {
        "perf": {"write_ms": {"n": 3, "p50_ms": 5.0, "p95_ms": 5.0, "max_ms": 5.0}},
        "recall": {"recall_at_10": {"rate": 0.9}},
    }
    current = {
        "perf": {"write_ms": {"n": 3, "p50_ms": 9999.0, "p95_ms": 9999.0, "max_ms": 9999.0}},
        "recall": {"recall_at_10": {"rate": 0.9}},
    }
    flat = score._flatten(current)
    assert not any(key.startswith("perf.") for key in flat)
    assert score.gate(current, baseline)["gate"] == "PASSED"
    # render 只喂 perf/env（其余套件段缺失时跳过渲染）
    markdown = score.render_markdown({"env": {}, "perf": current["perf"]}, None)
    assert "性能" in markdown and "9999" in markdown


def test_conflict_comprehensive_union_any_channel() -> None:
    """H1/D4：综合召回 = 两语料合并 any-channel 口径（Σidentified(true)/Σtrue）。

    复现 r1 推算锚（review R2-7）：句料库 7/43、claims 语料 11/15 →
    recall 18/58=0.3103、precision 18/24=0.75。"""
    raw = {
        "conflict": [
            {"pair_id": f"s{i}", "label": "true_conflict", "skipped_member_replay": False,
             "sync": i < 7, "async": False, "notice_missing": i >= 7}
            for i in range(43)
        ] + [
            {"pair_id": "c1", "label": "coexist", "skipped_member_replay": False,
             "sync": True, "async": False, "notice_missing": False},  # 共存 FP 进精确分母
        ] + [
            # 句料库 identified_all=13 的形态锚（r1：7 真 + 6 非真 → 18/24）
            {"pair_id": f"n{i}", "label": "noise", "skipped_member_replay": False,
             "sync": True, "async": False, "notice_missing": False}
            for i in range(5)
        ] + [
            {"pair_id": "c2", "label": "true_conflict", "skipped_member_replay": True,
             "sync": None, "async": None, "notice_missing": None},  # replay 不计
        ],
        "conflict_claims": [
            {"pair_id": f"b{i}", "channel": "B", "label": "true_conflict",
             "skipped_member_replay": False, "sync": True, "async": False, "notice_missing": False}
            for i in range(5)
        ] + [
            {"pair_id": f"c{i}", "channel": "C", "label": "true_conflict",
             "skipped_member_replay": False, "sync": i < 6, "async": False, "notice_missing": i >= 6}
            for i in range(10)
        ] + [
            {"pair_id": "n1", "channel": "C", "label": "coexist",
             "skipped_member_replay": False, "sync": False, "async": False, "notice_missing": True},
            {"pair_id": "n2", "channel": "B", "label": "true_conflict",
             "skipped_member_replay": True, "sync": None, "async": None, "notice_missing": None},
        ],
    }
    comp = score.score_conflict_comprehensive(raw)
    assert comp["recall"] == {"count": 18, "total": 58, "rate": 0.3103}
    assert comp["precision"] == {"count": 18, "total": 24, "rate": 0.75}
    assert comp["sources"]["conflict"] == {"true_total": 43, "true_identified": 7}
    assert comp["sources"]["conflict_claims"] == {"true_total": 15, "true_identified": 11}
    # score_all 接线 + 门方向（recall/precision 均为 higher-is-better）
    scored = score.score_all(raw)
    assert scored["conflict_comprehensive"] == comp
    gate = score.gate(scored, scored, 0.1)
    assert gate["gate"] == "PASSED"
    worse = json.loads(json.dumps(scored))
    worse["conflict_comprehensive"]["recall"]["rate"] = 0.2
    assert score.gate(worse, scored, 0.1)["gate"] == "FAILED"


def test_conflict_attribution_reads_qwen_budget_and_direct_verdicts() -> None:
    """H1：分通道归因直读 Qwen 回执新键（review R1-6——pairs_examined 全局
    口径之后，A qwen 归因不再用差额推算）；旧 raw 无此二键按 0 计。"""
    raw = {
        "conflict": [
            {"pair_id": "s1", "label": "true_conflict", "skipped_member_replay": False,
             "sync": True, "async": False, "notice_missing": False,
             "_receipt": {"qwen_budget": {"internal": 1, "a_cross": 2},
                          "direct_verdicts": 3}},
            {"pair_id": "s2", "label": "true_conflict", "skipped_member_replay": False,
             "sync": False, "async": True, "notice_missing": False,
             "_receipt": {"pairs_examined": 9}},  # 旧 raw：无新键
        ],
        "conflict_claims": [
            {"pair_id": "b1", "channel": "B", "label": "true_conflict",
             "skipped_member_replay": False, "sync": True, "async": False, "notice_missing": False},
            {"pair_id": "c1", "channel": "C", "label": "true_conflict",
             "skipped_member_replay": False, "sync": True, "async": False, "notice_missing": False,
             "_receipt": {"qwen_budget": {"channel_c": 2}}},
        ],
    }
    attr = score.score_conflict_attribution(raw)
    assert attr == {
        "a_channel_identified": 2,
        "a_direct_verdicts_total": 3,
        "a_qwen_internal_total": 1,
        "a_qwen_cross_total": 2,
        "channel_b_identified": 1,
        "channel_c_identified": 1,
        "channel_c_qwen_total": 2,
    }


def test_gate_excludes_diagnostic_blocks() -> None:
    """H1 修复批（mema #1066 对抗 review）：conflict_attribution 整块与
    comprehensive sources 分母明细不进相对门——归因计数下降（过滤变好 →
    Qwen 派发变少）是 D1 的预期效果，不是回归；sources 是 recall 的分解
    诊断量。recall.rate 真跌仍拦（对照）。"""
    baseline = {
        "conflict_attribution": {
            "a_channel_identified": 7.0, "a_direct_verdicts_total": 3.0,
            "a_qwen_internal_total": 2.0, "a_qwen_cross_total": 6.0,
            "channel_b_identified": 5.0, "channel_c_identified": 6.0,
            "channel_c_qwen_total": 9.0,
        },
        "conflict_comprehensive": {
            "recall": {"count": 18, "total": 58, "rate": 0.3103},
            "precision": {"count": 18, "total": 24, "rate": 0.75},
            "sources": {
                "conflict": {"true_total": 43, "true_identified": 7},
                "conflict_claims": {"true_total": 15, "true_identified": 11},
            },
        },
    }
    current = json.loads(json.dumps(baseline))
    current["conflict_attribution"] = {k: 0.0 for k in baseline["conflict_attribution"]}
    current["conflict_comprehensive"]["sources"] = {
        "conflict": {"true_total": 0, "true_identified": 0},
        "conflict_claims": {"true_total": 0, "true_identified": 0},
    }
    # 归因/明细全部归零：门 PASSED（诊断块不受门）。
    assert score.gate(current, baseline, 0.1)["gate"] == "PASSED"
    # recall.rate 真跌超阈值：门 FAILED，且失败项只有行为键。
    current["conflict_comprehensive"]["recall"]["rate"] = 0.2
    result = score.gate(current, baseline, 0.1)
    assert result["gate"] == "FAILED"
    assert all(
        f["metric"].startswith("conflict_comprehensive.recall")
        for f in result["failures"]
    )


def test_gate_excludes_env_metadata() -> None:
    """提速批（mema #1066）：env 是环境元数据——同步窗等运行配置正当可调
    （--conflict-sync-wait-ms），不进行为相对门；行为键真跌仍拦。"""
    baseline = {
        "env": {"conflict_sync_wait_ms": 3000.0, "targets": 98.0},
        "recall": {"recall_at_10": {"rate": 0.9}},
    }
    current = {
        "env": {"conflict_sync_wait_ms": 100.0, "targets": 98.0},
        "recall": {"recall_at_10": {"rate": 0.9}},
    }
    assert score.gate(current, baseline, 0.1)["gate"] == "PASSED"
    current["recall"]["recall_at_10"]["rate"] = 0.5
    assert score.gate(current, baseline, 0.1)["gate"] == "FAILED"
