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
                          "reasons_seen": ["pairs_examined_capped", "rows_capped"],
                          "pairs_examined": 10}},
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
    # 0.17.0 review R2 r2s-12: units 模式退役（C5），units_capped_rows /
    # units_budget 死指标删除，rows_capped_rows 顶替口径
    assert "units_capped_rows" not in window and "units_budget" not in window
    assert window["rows_capped_rows"] == 1
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


def test_gate_negative_label_firing_is_lower_is_better() -> None:
    """owner 2026-09-25 拍板（E3 实证 FP 改善 7→6 被旧基线误判 FAILED）：
    governed_negative 的 firing 类指标=负样本误报，lower-is-better；miss
    （负样本上不报=正确）保持 higher-is-better 不受牵连。"""
    baseline = {"conflict": {"by_shape": {"governed_negative": {
        "identified": {"rate": 0.2258}, "miss": {"rate": 0.7742}}}}}
    improved = {"conflict": {"by_shape": {"governed_negative": {
        "identified": {"rate": 0.1935}, "miss": {"rate": 0.8065}}}}}
    # 误报降、miss 升（都是改善）：PASSED
    assert score.gate(improved, baseline, 0.1)["gate"] == "PASSED"
    # 误报涨超阈值（真回归）：FAILED
    worse = {"conflict": {"by_shape": {"governed_negative": {
        "identified": {"rate": 0.30}, "miss": {"rate": 0.70}}}}}
    result = score.gate(worse, baseline, 0.1)
    assert result["gate"] == "FAILED"
    assert all(
        "identified" in f["metric"] for f in result["failures"]
    ), result


def test_gate_negative_bucket_miss_is_higher_is_better() -> None:
    """R2 P1：负样本桶的 miss 上升=改善（负样本上不报=正确）——'.miss.'
    全局子串曾把它扫进 lower-is-better，真改善 >10% 会假 FAILED（E3 同类
    事故二次形态）。miss 升幅超阈必须 PASSED。"""
    baseline = {"conflict": {"by_shape": {"governed_negative": {
        "identified": {"rate": 0.2258}, "miss": {"rate": 0.7742}}}}}
    improved = {"conflict": {"by_shape": {"governed_negative": {
        "identified": {"rate": 0.1935}, "miss": {"rate": 0.95}}}}}
    assert score.gate(improved, baseline, 0.1)["gate"] == "PASSED"
    # 真桶（write_opposition）的 miss 上升仍是回归，不受牵连
    true_bucket = {"conflict": {"by_shape": {
        "governed_negative": {"identified": {"rate": 0.2258}, "miss": {"rate": 0.7742}},
        "write_opposition": {"identified": {"rate": 0.8}, "miss": {"rate": 0.15}},
    }}}
    true_worse = {"conflict": {"by_shape": {
        "governed_negative": {"identified": {"rate": 0.2258}, "miss": {"rate": 0.7742}},
        "write_opposition": {"identified": {"rate": 0.8}, "miss": {"rate": 0.30}},
    }}}
    result = score.gate(true_worse, true_bucket, 0.1)
    assert result["gate"] == "FAILED"
    assert all(".miss." in f["metric"] for f in result["failures"]), result


def test_gate_negative_sync_firing_stays_gated() -> None:
    """R2/cand3（owner R8 非对称收益口径）：负样本桶的 SYNC firing 直接出现
    在写响应、侵入性高一档——保持受门；写时正桶 sync（write_opposition）
    仍按 cand2 豁免（3 秒窗划分产物），负样本 async firing=窗口外 advisory
    同样豁免。"""
    baseline = {"conflict": {"by_shape": {
        "governed_negative": {
            "sync": {"rate": 0.1}, "async": {"rate": 0.1}, "identified": {"rate": 0.1},
        },
        "write_opposition": {"sync": {"rate": 0.5}, "async": {"rate": 0.3}},
    }}}
    # 负样本 sync 0.1→0.2（+100%，负样本误报直接进写响应）：FAILED
    # （sync 基线为 0 时相对上升门按既有语义跳过——无从计算）
    sync_worse = {"conflict": {"by_shape": {
        "governed_negative": {
            "sync": {"rate": 0.2}, "async": {"rate": 0.1}, "identified": {"rate": 0.1},
        },
        "write_opposition": {"sync": {"rate": 0.5}, "async": {"rate": 0.3}},
    }}}
    result = score.gate(sync_worse, baseline, 0.1)
    assert result["gate"] == "FAILED"
    assert result["failures"][0]["metric"] == (
        "conflict.by_shape.governed_negative.sync.rate"
    ), result["failures"]
    # 负样本 async 上升（窗口外 advisory）与正桶 sync 上升：均豁免
    split_worse = {"conflict": {"by_shape": {
        "governed_negative": {
            "sync": {"rate": 0.1}, "async": {"rate": 0.3}, "identified": {"rate": 0.1},
        },
        "write_opposition": {"sync": {"rate": 0.7}, "async": {"rate": 0.3}},
    }}}
    assert score.gate(split_worse, baseline, 0.1)["gate"] == "PASSED"


def test_gate_noisy_bucket_not_negative_mixed_corpus() -> None:
    """R2 复评：noisy 从负样本桶收口移除——pairs_noisy 26 对中
    true_conflict 15 + coexist 2 + noise 9（58% 真对），整桶按负样本定方向
    两头都错：miss 上升（真对漏检变多）被当改善放行、sync firing 上升
    （多为真对正当检出）被当假阳性误杀（r2-fix-final 的 noisy.sync
    3→2 窗沿抖动曾触发假 FAILED）。noisy 回归普通桶口径。"""
    baseline = {"conflict": {"by_shape": {"noisy": {
        "identified": {"rate": 0.1154}, "miss": {"rate": 0.8462},
        "sync": {"rate": 0.1154}, "async": {"rate": 0.0385},
    }}}}
    # miss 0.8462→0.95（真对漏检变多，>10%）：必须 FAILED，不再当改善放行
    miss_worse = {"conflict": {"by_shape": {"noisy": {
        "identified": {"rate": 0.1154}, "miss": {"rate": 0.95},
        "sync": {"rate": 0.1154}, "async": {"rate": 0.0385},
    }}}}
    result = score.gate(miss_worse, baseline, 0.1)
    assert result["gate"] == "FAILED"
    assert all(".miss." in f["metric"] for f in result["failures"]), result
    # sync 0.1154→0.2（真对正当检出增多）：cand2 豁免，不再按负样本 sync 受门
    sync_up = {"conflict": {"by_shape": {"noisy": {
        "identified": {"rate": 0.1154}, "miss": {"rate": 0.8462},
        "sync": {"rate": 0.2}, "async": {"rate": 0.0385},
    }}}}
    assert score.gate(sync_up, baseline, 0.1)["gate"] == "PASSED"


def test_keyword_bucket_reuses_main_loop_core_and_gains_capped() -> None:
    """struct 修复：keyword 分桶累加器是 score_recall 主循环的手抄副本且
    已漂移（缺 capped 变体）——收敛到共享实现后，分桶（含 band 子桶）自动
    携带主循环全套指标（classic+capped+MRR），原有题级双口径与字段名不变。"""
    raw = {
        "queries": [
            # K01：7 个 relevant target 只召回 3——classic 分母 7、capped 分母 5
            {"qid": "K01", "kind": "keyword", "hits": [
                {"fixture_key": f"t-{i}"} for i in range(3)]},
            {"qid": "K02", "kind": "keyword", "expected_band": "midband",
             "hits": [{"fixture_key": "t-0"}]},
            {"qid": "A01", "kind": "paraphrase", "hits": [{"fixture_key": "t-a"}]},
        ],
    }
    original = score._load_jsonl

    def _stub(path: Path) -> list[dict]:
        if path.name == "labels.jsonl" and "recall" in str(path):
            return (
                [
                    {"qid": "K01", "fixture_key": f"t-{i}", "label": "relevant"}
                    for i in range(7)
                ]
                + [{"qid": "K02", "fixture_key": "t-0", "label": "relevant"}]
                + [{"qid": "A01", "fixture_key": "t-a", "label": "relevant"}]
            )
        return original(path)

    score._load_jsonl = _stub
    try:
        result = score.score_recall(raw)
    finally:
        score._load_jsonl = original

    kb = result["keyword_bucket"]
    # 原有字段/口径不变：题级双口径 + classic target 级微平均 + MRR
    assert kb["queries"] == 2
    assert kb["query_top5_hit_rate"] == 1.0
    assert kb["query_top10_hit_rate"] == 1.0
    assert kb["recall_at_5"] == {"hits": 4, "total": 8, "rate": 0.5}
    assert kb["recall_at_10"] == {"hits": 4, "total": 8, "rate": 0.5}
    assert kb["mrr"] == {"value": 1.0, "queries_with_target": 2}
    # 新增：capped 变体（K01 分母 min(7,5)=5 + K02 min(1,5)=1；@10 分母 7+1）
    assert kb["recall_at_5_capped"] == {
        "hits": 4, "total": 6, "rate": round(4 / 6, 4),
    }
    assert kb["recall_at_10_capped"] == {"hits": 4, "total": 8, "rate": 0.5}
    # band 子桶同样继承 capped
    assert kb["by_band"]["midband"]["recall_at_5_capped"] == {
        "hits": 1, "total": 1, "rate": 1.0,
    }
    # 主循环自身口径不受重构影响（capped 分母 5+1+1）
    assert result["recall_at_5_capped"] == {
        "hits": 5, "total": 7, "rate": round(5 / 7, 4),
    }


def test_gate_corpus_mismatch_single_helper_all_three_suites() -> None:
    """struct 修复：gate() 三段复制粘贴的语料版本早退收敛为
    _corpus_version_mismatches 单次调用——三个套件（recall/conflict/
    similarity）不匹配都必须在其余门逻辑之前以原消息逐字、单失败负载拦下；
    多套件同炸按 recall→conflict→similarity 优先级只报首项；similarity 侧
    缺键（旧基线）仍跳过。"""
    baseline = {
        "corpus_version": "rc-1",
        "env": {
            "conflict_corpus_version": "cc-1",
            "similarity_corpus_version": "sc-1",
        },
        "recall": {"recall_at_10": {"rate": 0.9}},
    }
    ok = json.loads(json.dumps(baseline))
    assert score.gate(ok, baseline)["gate"] == "PASSED"
    assert score._corpus_version_mismatches(ok, baseline) == []

    cases = [
        ("corpus_version", "rc-1",
         "recall 考卷语料版本不一致，拒绝跨语料对比；重建基线后重试"),
        ("env.conflict_corpus_version", "cc-1",
         "conflict 对集语料版本不一致，拒绝跨语料对比；重建基线后重试"),
        ("env.similarity_corpus_version", "sc-1",
         "similarity 套件语料版本不一致，拒绝跨语料对比；重建基线后重试"),
    ]
    for metric, base_value, note in cases:
        current = json.loads(json.dumps(baseline))
        if "." in metric:
            scope, key = metric.split(".")
            current[scope][key] = "bumped"
        else:
            current[metric] = "bumped"
        result = score.gate(current, baseline)
        assert result["gate"] == "FAILED", metric
        assert len(result["failures"]) == 1, metric
        assert result["failures"][0] == {
            "metric": metric,
            "direction": "corpus_mismatch",
            "baseline": base_value,
            "current": "bumped",
            "note": note,
        }, metric
        # helper 直读与 gate 首项一致（文案/负载同源）
        assert score._corpus_version_mismatches(current, baseline) == (
            result["failures"]
        )

    # 优先级：三套件同时 bump → helper 按序产出，gate 只取首项
    all_bumped = json.loads(json.dumps(baseline))
    all_bumped["corpus_version"] = "rc-2"
    all_bumped["env"]["conflict_corpus_version"] = "cc-2"
    all_bumped["env"]["similarity_corpus_version"] = "sc-2"
    helper_out = score._corpus_version_mismatches(all_bumped, baseline)
    assert [m["metric"] for m in helper_out] == [
        "corpus_version",
        "env.conflict_corpus_version",
        "env.similarity_corpus_version",
    ]
    assert score.gate(all_bumped, baseline)["failures"] == [helper_out[0]]

    # 兼容：similarity 侧任一缺键（旧基线）跳过校验，不误拦
    old_baseline = {"env": {"conflict_corpus_version": "cc-1"}}
    current = {"env": {"conflict_corpus_version": "cc-1"}}
    assert score._corpus_version_mismatches(current, old_baseline) == []
    assert score.gate(current, old_baseline)["gate"] == "PASSED"


def test_perf_markdown_query_label_derives_from_data() -> None:
    """struct 修复：perf 段「recall N query」标签此前硬编码 34，语料扩到
    47 题后过期——题数必须从数据（recall.per_query 行数）推导。"""
    raw = {
        "queries": [
            {"qid": f"Q{i:02d}", "kind": "paraphrase", "hits": [], "elapsed_ms": 10.0}
            for i in range(47)
        ],
    }
    original = score._load_jsonl

    def _stub(path: Path) -> list[dict]:
        if path.name == "labels.jsonl" and "recall" in str(path):
            return []
        return original(path)

    score._load_jsonl = _stub
    try:
        scored = score.score_all(raw)
    finally:
        score._load_jsonl = original
    markdown = score.render_markdown(scored, None)
    assert "recall 47 query" in markdown
    assert "recall 34 query" not in markdown
    # perf-only 渲染（无 recall 块）：回退 find_ms 样本数，不回归旧硬编码
    perf_only = score.render_markdown(
        {"env": {}, "perf": scored["perf"]}, None
    )
    assert "recall 2 query" not in perf_only and "recall 34" not in perf_only
    assert "查询" in perf_only
