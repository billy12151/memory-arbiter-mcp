"""Gate-v2 G4: sentence prefilter (write-only optional layer) and the
candidate cosine band — one shared implementation in pipeline/gates,
orchestrated differently by the write job and the scan."""
from __future__ import annotations

from collections import namedtuple

import pytest

from memory_arbiter.pipeline.gates import (
    candidate_cos_gate,
    row_prefilter,
)
from memory_arbiter.semantic_conflict import _SENT_PREFILTER

_S = namedtuple("_S", "text kind start_offset end_offset unit_index")


def test_prefilter_passes_value_negation_time_and_table_rows() -> None:
    rows = [
        _S("连接池上限为 100。", "sentence", 0, 10, 1),
        _S("该功能不包含缓存模块", "sentence", 10, 20, 2),
        _S("星期一下雨了", "sentence", 20, 27, 3),
        _S("部署平台\t区域", "table_row", 27, 35, 4),
    ]
    passed = list(row_prefilter(rows))
    assert [r.unit_index for r in passed] == [1, 2, 3, 4]
    assert all(_SENT_PREFILTER.search(r.text) for r in rows[:3])


def test_prefilter_drops_pure_prose_rows() -> None:
    rows = [
        _S("这是一段纯粹的叙述性描述文字。", "sentence", 0, 15, 1),
        _S("部署平台是 Vercel。", "sentence", 15, 24, 2),
    ]
    # No claims: only the prose row is dropped. The "X是Y" text-value form is
    # a KNOWN accepted write-path gap (方案 §5) — the scan path judges it.
    assert [r.unit_index for r in row_prefilter(rows)] == []
    # 0.17.1 owner 拍板：claim 覆盖句跳过随 claim 对比通道退役删除——
    # row_prefilter 不再有任何 claims 耦合（无 claim_spans 参数），claim
    # 覆盖句重新从通道 A 发起（否则成检测死区）。
    rows2 = [
        _S("超时阈值为 500ms。", "sentence", 0, 14, 1),
        _S("部署平台是 Vercel。", "sentence", 14, 23, 2),
    ]
    passed = list(row_prefilter(rows2))
    # 纯散文行仍被初筛丢（初筛本身不变）
    assert [r.unit_index for r in passed] == [1]


def test_cosine_gate_band_split() -> None:
    # own is deliberately NOT byte-equal to any hit vector (the degenerate
    # guard would pass byte-equal pairs unconditionally).
    own = [0.99, 0.141]
    hits = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    # cos ≈0.99 (ceil) / ≈0.75 (band, 新带 [0.70,0.98)) / ≈0.14 (floor) / missing vector
    vecs = {
        1: [1.0, 0.0],
        2: [0.65, 0.7599],
        3: [0.0, 1.0],
    }
    passed, below, at_ceil = candidate_cos_gate(own, hits, vecs)
    assert [h["id"] for h, _c in passed] == [2]
    assert [h["id"] for h, _c in below] == [3]
    assert [h["id"] for h, _c in at_ceil] == [1]
    assert [h["id"] for h, _c in candidate_cos_gate(own, [hits[3]], {})[0]] == []


def test_cosine_gate_degenerate_vector_guard() -> None:
    """Byte-identical vectors carry zero discrimination (fake embedders);
    the pair passes at 1.0 and decide_evidence's duplicate routes settle it —
    never filtered on testimony the embedder cannot give (P2-3.3 doctrine)."""
    own = [0.0, 1.0]
    hits = [{"id": 1}]
    passed, below, at_ceil = candidate_cos_gate(own, hits, {1: [0.0, 1.0]})
    assert [h["id"] for h, _c in passed] == [1]
    assert not below and not at_ceil


def test_write_loop_skips_prose_row_without_knn(tmp_path, monkeypatch) -> None:
    """写入编排：纯叙述句不发起 KNN（mock row_knn 断言未调用）且回执计
    prefiltered_rows。"""
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    result = tools.memory_write(content="这是一段纯粹叙述性质的内容描述。", subject="prose", tags=[])
    memory_id = result["data"]["id"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    calls = []
    monkeypatch.setattr(tools.db, "row_knn", lambda embedding, **kw: calls.append(kw) or [])
    receipt = tools._process_semantic_conflict_job(memory_id, tv._job_snapshot(tools, memory_id))
    # The subject coarse screen (k=50, subject_rows_only) is ALLOWED; the
    # sentence KNN (k=16) must never fire for a prose-only memory.
    assert all(kw.get("k") != 16 for kw in calls), "prose row must not originate a sentence KNN"
    assert receipt["candidate_gates"]["prefiltered_rows"] == 1
    assert receipt["rows_examined"] == 0


def test_write_loop_claim_covered_row_originates_normally(tmp_path, monkeypatch) -> None:
    """0.17.1 owner 拍板：claim 对比通道退役，覆盖句跳过删除——claim 覆盖行
    重新从通道 A 发起 KNN（否则成检测死区）。rows_covered_by_claims 键随之
    消失（零值不出现惯例的延伸：键本身退役）。"""
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    peer = tools.memory_write(content="数据库是 MySQL。", subject="a", tags=[])["data"]
    new = tools.memory_write(content="数据库是 PostgreSQL。", subject="b", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "数据库是 MySQL。",
             "start_offset": 0, "end_offset": 12, "distance": 0.2}]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: tv._hits_with_metadata(tools, list(hits)))
    tv._pass_cos_gate(monkeypatch)
    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))
    gates = result.get("candidate_gates") or {}
    assert "rows_covered_by_claims" not in gates, "键已随覆盖句跳过退役"



def json_dumps(obj) -> str:
    import json
    return json.dumps(obj)


def test_scan_orchestration_never_calls_row_prefilter(tmp_path, monkeypatch) -> None:
    """扫描编排跳过初筛（编排可选层）：慢道对纯叙述句仍发起 KNN——层代码
    一份，编排差异只体现在是否调用。"""
    import tests.test_scan_pipeline as tsp

    tools = tsp.make_tools(tmp_path)
    a = tsp._write(tools, "叙述甲", "一段完全没有判定标记的纯叙述内容甲")
    tsp._write(tools, "叙述乙", "一段完全没有判定标记的纯叙述内容乙")
    assert tools.wait_semantic_worker_drained(timeout=10)

    calls = []
    orig = tools.db.row_knn
    monkeypatch.setattr(
        tools.db, "row_knn",
        lambda embedding, **kw: calls.append(1) or orig(embedding, **kw),
    )
    from memory_arbiter.scan_pipeline import ScanPipeline
    pipeline = ScanPipeline(tools)
    pipeline._process_memory(
        a, suppression=pipeline._load_suppression(), neighbor_k=10,
    )
    assert calls, "scan orchestration must NOT skip prose rows (no prefilter)"


# ── G5 memory-level screen ──────────────────────────────────────────────────

def test_memory_pair_excluded_version_evolution() -> None:
    from memory_arbiter.pipeline.gates import memory_pair_excluded
    # Owner 测试预期: same-topic year planning at different generations 毙.
    assert memory_pair_excluded("2024 规划", [], "2025 规划", [])
    assert memory_pair_excluded("0.16.12 发版闭环", [], "0.16.11 发版闭环", [])
    # cf-res-9 (对抗 review P0): different stems are DIFFERENT topics —
    # never excluded even with both lineage markers and different versions.
    assert not memory_pair_excluded("memory_arbiter 重构实施方案 v2", [], "memory_arbiter 修复实施方案 v3", [])
    # Same primary version = same-generation text (cf-res-30) stays.
    assert not memory_pair_excluded("决策：v0.9.7 启用 workspace 隔离", [], "memory-arbiter v0.9.7 对抗性 review", [])
    # Single-sided version shape stays (cf-res-10).
    assert not memory_pair_excluded("mema-core 高 ROI 功能候选分析", [], "mema-core Tier 1 功能方案（0.15.2 合并）", [])


def test_memory_pair_excluded_process_record() -> None:
    from memory_arbiter.pipeline.gates import memory_pair_excluded
    # Either side carrying the process shape vetoes the pair (the vocab is
    # phrase-based: "对抗性 review 结论" carries the review-findings shape).
    assert memory_pair_excluded("adversarial review findings round 3", [], "复盘与下一步", [])
    # ...but #50-style recipe opposition (release-adjacent subject WITHOUT
    # version tokens on both sides with matching stems) is untouched.
    assert not memory_pair_excluded("twine 上传配置", [], "twine dist/* 会 409", [])


def test_knn_restricted_to_clean_neighbor_list(tmp_path, monkeypatch) -> None:
    """句子 KNN 只在邻居名单∖排除名单内做（include_memory_ids rowid-IN）。"""
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    own = tools.memory_write(content="网关超时阈值为 500ms。", subject="网关方案 v2", tags=[])["data"]
    peer_same = tools.memory_write(content="队列长度上限为 100 条。", subject="网关方案 v3", tags=[])["data"]
    peer_diff = tools.memory_write(content="利率上限为一年期 LPR 的四倍。", subject="借贷政策", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    captured: list[dict] = []

    def fake_knn(embedding, **kw):
        captured.append(dict(kw))
        if kw.get("subject_rows_only"):
            return [
                {"memory_id": int(peer_same["id"]), "subject": "网关方案 v3", "tags": []},
                {"memory_id": int(peer_diff["id"]), "subject": "借贷政策", "tags": []},
            ]
        return []

    monkeypatch.setattr(tools.db, "row_knn", fake_knn)
    receipt = tools._process_semantic_conflict_job(own["id"], tv._job_snapshot(tools, own["id"]))
    # peer_same: 网关方案 v2 vs v3 — same stem, different primary → EXCLUDED.
    # peer_diff: different stem → stays in the clean list.
    screen_calls = [kw for kw in captured if kw.get("subject_rows_only")]
    assert screen_calls, "coarse screen must run exactly one subject KNN"
    sentence_calls = [kw for kw in captured if not kw.get("subject_rows_only")]
    for kw in sentence_calls:
        allowed = kw.get("include_memory_ids")
        assert allowed is not None
        assert int(peer_same["id"]) not in allowed
        assert int(peer_diff["id"]) in allowed
    assert receipt["candidate_gates"]["memory_pairs_excluded"] == 1


# ── G6 three-case dispatch + ranking + bridge ───────────────────────────────


def test_pair_score_orders_high_band_first(tmp_path, monkeypatch) -> None:
    """带内两对都被 Qwen 消费（预算顺序的纯函数钉在 compute_pair_score 单测）。"""
    import tests.test_vnext_evidence as tv
    from memory_arbiter.constants import SEMANTIC_MAX_EXAMINED_PAIRS

    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    # Two peers, both numeric-value candidates; the HIGH-band peer has the
    # larger cosine (its gate cos 0.95 vs 0.61).
    # multi-value shapes: ", " in left_value keeps the deterministic direct
    # verdict silent so the pairs actually reach the Qwen budget ordering.
    peer_hi = tools.memory_write(content="队列长度上限为 100 条、超时 20ms。", subject="hi", tags=[])["data"]
    peer_lo = tools.memory_write(content="队列长度上限为 200 条、超时 30ms。", subject="lo", tags=[])["data"]
    new = tools.memory_write(content="队列长度上限为 300 条、超时 40ms。", subject="new", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    consumed: list[int] = []

    class Backend:
        calls = 0

        @staticmethod
        def judge_pairs(pairs):
            # 0.17.1: consumption order rides the batch interface; verdicts
            # are no_conflict so nothing lands — the pin is the ORDER.
            from memory_arbiter.semantic_judge import PairVerdict
            Backend.calls += len(pairs)
            for text_a, text_b in pairs:
                if "200 条" in text_b:
                    consumed.append(int(peer_lo["id"]))
                elif "100 条" in text_b:
                    consumed.append(int(peer_hi["id"]))
            return [PairVerdict("no_conflict",
                                {"conflict": 0.0, "no_conflict": 1.0, "possible_conflict": 0.0},
                                None, "test") for _ in pairs]

    def fake_knn(embedding, **kw):
        if kw.get("subject_rows_only"):
            # coarse screen: both neighbours are in the clean list
            return [
                {"memory_id": int(peer_lo["id"]), "subject": "lo", "tags": []},
                {"memory_id": int(peer_hi["id"]), "subject": "hi", "tags": []},
            ]
        # both peers at identical distance: order must come from the score
        return [
            {"memory_id": int(peer_lo["id"]), "id": 1, "kind": "text",
             "text": "队列长度上限为 200 条、超时 30ms。", "start_offset": 0, "end_offset": 20,
             "distance": 0.5, "subject": "lo", "tags": []},
            {"memory_id": int(peer_hi["id"]), "id": 2, "kind": "text",
             "text": "队列长度上限为 100 条、超时 20ms。", "start_offset": 0, "end_offset": 20,
             "distance": 0.5, "subject": "hi", "tags": []},
        ]

    def fake_vectors(ids, conn=None):
        # own sentence row is [0.8, 0.2] (|v|≈0.825): id 2 (hi) at cos≈0.94 →
        # band saturated to 1.0; id 1 (lo) at cos≈0.785 → band≈0.925.
        return {1: [0.61, 0.79], 2: [0.6, 0.4]}

    monkeypatch.setattr(tools.db, "row_knn", fake_knn)
    monkeypatch.setattr(tools.db.evidence, "row_vectors_for_ids", fake_vectors)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: Backend())
    tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))
    assert sorted(consumed) == sorted([int(peer_hi["id"]), int(peer_lo["id"])])
    # Pure-function ordering pin: identical signals, hi band 1.0 > lo band.
    from types import SimpleNamespace
    from memory_arbiter.pipeline.gates import compute_pair_score
    decision = SimpleNamespace(reason="numeric_value_candidate",
                               left_value="300", right_value="100")
    hi = compute_pair_score(decision, 0.94, "a 500ms", "b 500ms")
    lo = compute_pair_score(decision, 0.785, "a 500ms", "b 500ms")
    assert hi > lo



def test_compute_pair_score_formula() -> None:
    """G6 排序公式纯函数钉：band 主导、numeric/values_differ/negation 各自
    加分、值相等不加分（裁决层已毙）、文本值对判不了等不加、平手链由调用方
    排序键处理。"""
    from types import SimpleNamespace
    from memory_arbiter.pipeline.gates import compute_pair_score

    numeric = SimpleNamespace(reason="numeric_value_candidate",
                              left_value="300", right_value="100")
    # cos 0.94 → band (0.94-0.60)/0.38 ≈ 0.895 → 0.40*0.895+0.25+0.20 ≈ 0.808
    assert compute_pair_score(numeric, 0.94, "网关超时 500ms", "网关超时 300ms") == pytest.approx(0.808, abs=1e-2)
    # cos 0.60 → band 0 → 0.45；cos 0.70 → band (0.10)/0.38 ≈ 0.263 → 0.555；
    # cos 0.74 → band (0.14)/0.38 ≈ 0.368 → 0.597（band 主导）
    assert compute_pair_score(numeric, 0.60, "", "") == pytest.approx(0.45)
    assert compute_pair_score(numeric, 0.70, "", "") == pytest.approx(0.555, abs=1e-2)
    assert compute_pair_score(numeric, 0.74, "", "") == pytest.approx(0.597, abs=1e-2)
    # negation: 单侧命中加 0.15，双侧命中（同一形态）不加
    polarity = SimpleNamespace(reason="polarity_changed", left_value=None, right_value=None)
    assert compute_pair_score(polarity, 0.60, "包含缓存", "不包含缓存") == pytest.approx(0.15)
    assert compute_pair_score(polarity, 0.60, "不包含缓存", "不含缓存") == pytest.approx(0.0)
    # 值相等（换算后）：裁决层已毙的形态，排序不加分
    equal = SimpleNamespace(reason="numeric_value_candidate",
                            left_value="500ms", right_value="0.5s")
    # band(0.94) ≈ 0.895 → 0.40*0.895 + 0.25（numeric，值相等不加分）≈ 0.608
    assert compute_pair_score(equal, 0.94, "x", "y") == pytest.approx(0.608, abs=1e-2)


# ── G6b channel C (claims×sentences) ────────────────────────────────────────


