"""Gate-v2 G4: sentence prefilter (write-only optional layer), claims
coverage skip, and the candidate cosine band — one shared implementation in
pipeline/gates, orchestrated differently by the write job and the scan."""
from __future__ import annotations

from collections import namedtuple

import pytest

from memory_arbiter.pipeline.gates import (
    candidate_cos_gate,
    claim_value_spans,
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


def test_prefilter_drops_pure_prose_and_claim_covered_rows() -> None:
    rows = [
        _S("这是一段纯粹的叙述性描述文字。", "sentence", 0, 15, 1),
        _S("部署平台是 Vercel。", "sentence", 15, 24, 2),
    ]
    # No claims: only the prose row is dropped. The "X是Y" text-value form is
    # a KNOWN accepted write-path gap (方案 §5) — the scan path judges it.
    assert [r.unit_index for r in row_prefilter(rows)] == []
    # Claim value covering row 2 exactly: the covered row drops with its own
    # counter even though it would have passed the vocab.
    rows2 = [
        _S("超时阈值为 500ms。", "sentence", 0, 14, 1),
        _S("部署平台是 Vercel。", "sentence", 14, 23, 2),
    ]
    passed = list(row_prefilter(rows2, [(14, 23)]))
    assert [r.unit_index for r in passed] == [1]


def test_claim_value_spans_position_based() -> None:
    content = "前置内容。超时阈值为 500ms。后续同值 500ms 出现。"
    spans = claim_value_spans(content, [{"value": "超时阈值为 500ms"}])
    assert spans == [(5, 16)]  # find() pins the FIRST literal occurrence
    # Unlocatable values (legacy/edited rows) contribute no span.
    assert claim_value_spans(content, [{"value": "不存在的片段"}]) == []


def test_cosine_gate_band_split() -> None:
    # own is deliberately NOT byte-equal to any hit vector (the degenerate
    # guard would pass byte-equal pairs unconditionally).
    own = [0.99, 0.141]
    hits = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    # cos ≈0.99 (ceil) / ≈0.64 (band) / ≈0.14 (floor) / missing vector
    vecs = {
        1: [1.0, 0.0],
        2: [0.5268, 0.85],
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


def test_write_loop_skips_claim_covered_row(tmp_path, monkeypatch) -> None:
    """claims 覆盖句跳过（owner 拍板）：claim value 定位 offset 落在句子
    span 内 → 该句不从通道 A 发起，回执计 rows_covered_by_claims。"""
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    content = "网关超时阈值为 500ms。\n第二段是纯叙述的背景描述内容。"
    result = tools.memory_write(
        content=content, subject="timeout", tags=[],
        claims=[{"attr": "超时阈值", "value": "网关超时阈值为 500ms。"}],
    )
    memory_id = result["data"]["id"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    calls = []
    monkeypatch.setattr(tools.db, "row_knn", lambda embedding, **kw: calls.append(kw) or [])
    receipt = tools._process_semantic_conflict_job(memory_id, tv._job_snapshot(tools, memory_id))
    assert all(kw.get("k") != 16 for kw in calls), "covered row must not originate a sentence KNN"
    gates = receipt["candidate_gates"]
    assert gates.get("rows_covered_by_claims") == 1
    assert "rows_covered" in json_dumps(gates) or gates.get("rows_covered_by_claims") == 1


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
