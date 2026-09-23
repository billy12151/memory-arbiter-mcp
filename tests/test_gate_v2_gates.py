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


# ── G6 three-case dispatch + ranking + bridge ───────────────────────────────

def test_qwen_dispatch_three_cases() -> None:
    from types import SimpleNamespace
    from memory_arbiter.pipeline.gates import dispatch_hint_text, qwen_dispatch
    assert qwen_dispatch(SimpleNamespace(left_value="500ms", right_value=None)) == "extract_value"
    assert qwen_dispatch(SimpleNamespace(left_value="5秒", right_value="3秒")) == "align_attr"
    assert qwen_dispatch(SimpleNamespace(left_value=None, right_value=None)) == "align_value"
    # Each case has its own task line; same protocol, different instruction.
    assert dispatch_hint_text("extract_value") != dispatch_hint_text("align_attr") != dispatch_hint_text("align_value")
    # The extract_value hint NAMES the attribute (单边桥契约).
    hint = f"{dispatch_hint_text('extract_value')} 需抽取的属性名：上传方式"
    assert "上传方式" in hint


def test_pair_score_orders_high_band_first(tmp_path, monkeypatch) -> None:
    """cos 高（冲突带满分）的对先于 cos 低的对消费 Qwen 预算；排序不改变判定。"""
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
        def classify_pair(left, right, *, deadline_monotonic=None, retry_allowed=True):
            Backend.calls += 1
            consumed.append(int(right.get("memory_id") or 0))
            from memory_arbiter.semantic_conflict import ModelSignal
            return ModelSignal(False, "unknown_field", None, "", None, None)

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
    assert Backend.calls >= 1
    if len(consumed) >= 1:
        assert consumed[0] == int(peer_hi["id"]), "high-band peer must consume budget first"


def test_claim_bridge_extracts_and_reports(tmp_path, monkeypatch) -> None:
    """单边桥：own claim 的属性在 peer claims 无同名属性 → attr 向量定向捞
    peer 句子 → Qwen 情形 a 抽值（prompt 含 attr 名）→ 抽出异值落 notice。"""
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    # claims grounding contract: value MUST be a verbatim content slice.
    peer = tools.memory_write(
        content="twine 上传 dist/* 路径会 409。", subject="twine", tags=[],
        claims=[{"attr": "上传路径", "value": "dist/* 路径会 409"}],
    )["data"]
    new = tools.memory_write(
        content="主仓库就是内部源，不对外提供服务。", subject="internal", tags=[],
        claims=[{"attr": "上传方式", "value": "主仓库就是内部源"}],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    captured_prompts: list[str] = []

    class BridgeBackend:
        @staticmethod
        def classify_pair(left, right, **kw):
            captured_prompts.append(f"{left.get('dispatch_hint')}|{right.get('quote')}")
            from memory_arbiter.semantic_conflict import ModelSignal
            return ModelSignal(
                True, "attribute_value_extraction", None, "",
                # value_b must be a literal slice of the peer quote (grounding).
                {"attribute_a": "上传方式", "value_a": "主仓库就是内部源",
                 "attribute_b": "上传方式", "value_b": "dist/* 路径会 409"},
                None,
            )

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: BridgeBackend())
    # The fake embedder's constant vectors make EVERY attr pair cos=1.0 —
    # raise tau above 1.0 so the bridge (no same-attr peer claim) triggers.
    monkeypatch.setattr("memory_arbiter.constants.CLAIM_ATTR_TAU", 1.1)
    monkeypatch.setattr(
        tools.db, "row_knn",
        lambda embedding, **kw: [{
            "memory_id": int(peer["id"]), "id": 1, "kind": "text",
            "text": "twine 上传 dist/* 路径会 409。", "start_offset": 0, "end_offset": 20,
            "distance": 0.1, "memory_row_version": 1,
        }],
    )
    result = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))
    claims = result.get("claims_channel") or {}
    assert captured_prompts, "bridge must consult Qwen with the attr-named prompt"
    assert "上传方式" in captured_prompts[0], "case-a prompt must name the attribute"
    assert claims.get("notices", 0) >= 1
    notices = tools.db.list_semantic_notices(status="open")
    assert any(n.get("payload", {}).get("claim_bridge") for n in notices)
