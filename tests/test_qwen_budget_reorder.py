"""0.17.0 Q1 (mema #1066): Qwen 预算重排行为测试（方案 §3.4 清单）。

钉死的契约：
- 执行顺序 确定性相 → B → internal Qwen → C → A-cross 派发相（RecordingBackend 序列）；
- 共享 job 全局池（_JobQwenBudget）：internal 保护帽 ≤3、C 可扣穿不可被拦、
  A-cross 余量派发、耗尽 continue（direct 直出照常落地、未派发对进 backlog、
  a_cross_dispatch_skipped 出键）；
- 跨通道去重方向翻转：B∪C surfaced peers 并集进 A-cross skip 集合；
- 回执形状：零活动写入与旧形状逐位一致（无 qwen_budget/direct_verdicts 键）；
- E10①：确定性相截断时 internal keepers 先落地，派发相整体跳过。
"""
from __future__ import annotations

from typing import Any

import pytest

import memory_arbiter.pipeline.gates as _gates
from memory_arbiter.pipeline.evidence import _JobQwenBudget
import tests.test_vnext_evidence as tv


# ── §3.4-3 预算算术（纯对象层：internal 1 + C 2 → A-cross 余量 7）────────────

def test_job_qwen_budget_pool_arithmetic() -> None:
    pool = _JobQwenBudget(total=10, internal_cap=3)
    assert pool.receipt_block() is None  # 零值不出现
    assert pool.spend_internal() is True
    assert pool.spend_internal() is True
    assert pool.internal_used == 2
    pool.spend_a_cross()
    pool.spend_a_cross()
    assert pool.remaining == 6  # 10 − 2 − 2
    # A-cross 余量派发
    for _ in range(6):
        assert pool.spend_a_cross() is True
    assert pool.spend_a_cross() is False  # 余量 0 → skip 标记
    assert pool.a_cross_dispatch_skipped is True
    assert pool.pairs_examined == 10
    assert pool.receipt_block() == {
        "internal": 2, "a_cross": 8, "a_cross_dispatch_skipped": True,
    }
    # internal 保护帽独立于池：帽内但池尽 → 拒
    pool2 = _JobQwenBudget(total=1, internal_cap=3)
    assert pool2.spend_internal() is True
    assert pool2.spend_internal() is False
    # 文档算术锚（方案 §3.4-3）：internal 1 + C 2 → A-cross 派发上限 7
    pool3 = _JobQwenBudget(total=10, internal_cap=3)
    assert pool3.spend_internal() is True
    pool3.spend_a_cross()
    pool3.spend_a_cross()
    assert pool3.remaining == 7
    for _ in range(7):
        assert pool3.spend_a_cross() is True
    assert pool3.spend_a_cross() is False and pool3.a_cross_dispatch_skipped


# ── 共享场景构造 ────────────────────────────────────────────────────────────

_OWN_ROWS = ("连接池上限为 99，队列长度为 99。", "连接池上限为 100，队列长度为 100。")
_OWN_CLAIM_RENDER = "连接池上限=99"
_OWN_CLAIM = {"attr": "连接池上限", "value": "99"}  # grounding：value 须为正文子串


def _subject_hit(peer: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": int(peer["id"]), "id": 900, "kind": "subject",
        "text": "subject", "subject": "budget-peer", "tags": "[]",
        "distance": 0.5, "memory_row_version": 1,
    }


def _peer_hit(peer: dict[str, Any], row_id: int, text: str, distance: float) -> dict[str, Any]:
    return {
        "memory_id": int(peer["id"]), "id": row_id, "kind": "text", "text": text,
        "start_offset": 0, "end_offset": len(text), "distance": distance,
        "memory_row_version": 1,
    }


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch, tools: Any,
    *, peer1: dict[str, Any], a_hits: list[dict[str, Any]],
    c_hits: list[dict[str, Any]], c_vectors: dict[int, list[float]],
) -> None:
    """row_knn 三分支：G5 标题屏（subject_rows_only）→ 名单；带 conn 的句子
    KNN → A 候选；无 conn 的 attr KNN → C 候选。"""
    def stub_row_knn(embedding: Any, **kw: Any) -> list[dict[str, Any]]:
        if kw.get("subject_rows_only"):
            return [_subject_hit(peer1)]
        if kw.get("conn") is not None:
            return list(a_hits)
        return list(c_hits)

    monkeypatch.setattr(tools.db, "row_knn", stub_row_knn)
    monkeypatch.setattr(
        _gates, "candidate_cos_gate",
        lambda own, hits, vecs: ([(h, 0.85) for h in hits], [], []),
    )
    monkeypatch.setattr(
        tools.db.evidence, "row_vectors_for_ids",
        lambda ids, conn=None: {int(i): c_vectors[int(i)] for i in ids if int(i) in c_vectors},
    )


class _Recorder:
    """0.17.1 判定记录器：B/C 退役后只剩 internal（同记忆句对）与 A-cross
    （裸句对）两类输入。返回 conflict 判定。"""

    calls: list[tuple[str, str]] = []  # (kind, right_text)
    own_content = ""
    c_surfaces = False  # 已无意义，保留兼容旧签名

    @classmethod
    def _verdict(cls, kind: str):
        from memory_arbiter.semantic_judge import PairVerdict
        return PairVerdict("conflict",
                           {"conflict": 0.9, "no_conflict": 0.05, "possible_conflict": 0.05},
                           None, "test")

    @classmethod
    def judge_pair(cls, text_a: str, text_b: str):
        kind = "internal" if text_a in set(_OWN_ROWS) else "A"
        cls.calls.append((kind, text_b))
        return cls._verdict(kind)

    @classmethod
    def judge_pairs(cls, pairs):
        out = []
        for text_a, text_b in pairs:
            kind = "internal" if text_a in set(_OWN_ROWS) else "A"
            cls.calls.append((kind, text_b))
            out.append(cls._verdict(kind))
        return out

    @classmethod
    def reset(cls, own_content: str) -> None:
        cls.calls = []
        cls.own_content = own_content


def _write_scene(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, dict[str, Any]]:
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    peer1 = tools.memory_write(
        content="连接池上限为 300，队列长度为 4。", subject="budget-peer", tags=[],
    )["data"]
    content = "\n".join(_OWN_ROWS)
    new = tools.memory_write(
        content=content, subject="budget-own", tags=[],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset(content)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)
    return tools, new, peer1


# ── §3.4-2 顺序：internal → C → A-cross（RecordingBackend 序列）───────────────

def test_receipt_shape_null_run_has_no_new_keys(tmp_path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    new = tools.memory_write(
        content="这是一段纯粹叙述性质的内容描述。", subject="null-run", tags=[],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    monkeypatch.setattr(tools.db, "row_knn", lambda embedding, **kw: [])

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))
    receipt.pop("elapsed_ms", None)

    assert "judge_budget" not in receipt
    assert "direct_verdicts" not in receipt
    # 0.17.1：claims_channel/claims_channel_c 键随通道退役消失
    assert receipt == {
        "status": "completed", "outcome": "checked_no_notice", "notices_created": 0,
        "candidate_gates": {"prefiltered_rows": 1},
        "rows_mode": True, "rows_examined": 0,
    }


# ── §3.4-7 E10①：确定性相截断 → internal keepers 先落地、派发相整体跳过 ──────

def test_internal_keepers_land_when_deterministic_phase_truncates(tmp_path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    new = tools.memory_write(
        content="连接池上限为 99。\n连接池上限为 100。", subject="e10", tags=[],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset("连接池上限为 99。\n连接池上限为 100。")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)
    # 公平墙已过 → 收集循环立即截断（notice_budget_exhausted）。
    monkeypatch.setattr(
        tools._semantic_worker, "pending_job_deadline",
        lambda timeout: 1.0,  # monotonic() 远大于此
    )

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert receipt["status"] == "incomplete"
    assert receipt["reason"] == "notice_budget_exhausted"
    assert receipt["internal_conflicts"] == 1  # E10①：keepers 先于截断落地
    assert receipt["pairs_examined"] == 0
    assert not _Recorder.calls, "truncation terminal skips internal Qwen AND dispatch"
    assert "judge_budget" not in receipt  # C 未派发（allowed 缺失早退）→ 池零活动


# ── 对抗 review 修复批（mema #1066 第二轮）────────────────────────────────────

_NO_SURFACING_C = {"attribute_a": "连接池上限", "value_a": "99",
                   "attribute_b": "连接池上限", "value_b": "99"}


class _NoSurfacingRecorder(_Recorder):
    """C 同值抽取 → unresolved 不浮出（版本守卫/backlog 归因不被 skip 集合遮蔽）。"""

    @classmethod
    def classify_pair(cls, left, right, **kw):
        if "dispatch_hint" in left and left.get("quote") == cls.own_content[:1000]:
            cls.calls.append(("C", str(right.get("quote") or "")))
            return ModelSignal(True, "attribute_value_extraction", None, "",
                               dict(_NO_SURFACING_C), None)
        return super().classify_pair(left, right, **kw)


