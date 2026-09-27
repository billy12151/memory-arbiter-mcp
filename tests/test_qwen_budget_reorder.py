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
from memory_arbiter.semantic_conflict import ModelSignal
import tests.test_vnext_evidence as tv


# ── §3.4-3 预算算术（纯对象层：internal 1 + C 2 → A-cross 余量 7）────────────

def test_job_qwen_budget_pool_arithmetic() -> None:
    pool = _JobQwenBudget(total=10, internal_cap=3)
    assert pool.receipt_block() is None  # 零值不出现
    assert pool.spend_internal() is True
    assert pool.spend_internal() is True
    assert pool.internal_used == 2
    pool.spend_channel_c()
    pool.spend_channel_c()
    assert pool.remaining == 6  # 10 − 2 − 2
    # A-cross 余量派发
    for _ in range(6):
        assert pool.spend_a_cross() is True
    assert pool.spend_a_cross() is False  # 余量 0 → skip 标记
    assert pool.a_cross_dispatch_skipped is True
    assert pool.pairs_examined == 10
    assert pool.receipt_block() == {
        "internal": 2, "channel_c": 2, "a_cross": 6, "a_cross_dispatch_skipped": True,
    }
    # internal 保护帽独立于池：帽内但池尽 → 拒
    pool2 = _JobQwenBudget(total=1, internal_cap=3)
    assert pool2.spend_internal() is True
    assert pool2.spend_internal() is False
    # 文档算术锚（方案 §3.4-3）：internal 1 + C 2 → A-cross 派发上限 7
    pool3 = _JobQwenBudget(total=10, internal_cap=3)
    assert pool3.spend_internal() is True
    pool3.spend_channel_c()
    pool3.spend_channel_c()
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
    """0.17.1 判定记录器：按输入形态判别通道——internal=own 两行、C=左
    `attr=value` 渲染句、A-cross=裸句对。返回 conflict 判定。"""

    calls: list[tuple[str, str]] = []  # (kind, right_text)
    own_content = ""
    c_surfaces = False  # True = C 通道异值浮出（旧 extract→diff→notice）

    @classmethod
    def _verdict(cls, kind: str):
        from memory_arbiter.semantic_judge import PairVerdict
        if kind == "C" and not cls.c_surfaces:
            # 同值不浮出（对齐旧 extract→same-value→unresolved 语义）
            return PairVerdict("no_conflict",
                               {"conflict": 0.0, "no_conflict": 1.0, "possible_conflict": 0.0},
                               None, "test")
        return PairVerdict("conflict",
                           {"conflict": 0.9, "no_conflict": 0.05, "possible_conflict": 0.05},
                           None, "test")

    @classmethod
    def judge_pair(cls, text_a: str, text_b: str):
        kind = cls._kind(text_a, text_b)
        cls.calls.append((kind, text_b))
        return cls._verdict(kind)

    @classmethod
    def judge_pairs(cls, pairs):
        out = []
        for text_a, text_b in pairs:
            kind = cls._kind(text_a, text_b)
            cls.calls.append((kind, text_b))
            out.append(cls._verdict(kind))
        return out

    @classmethod
    def _kind(cls, text_a: str, text_b: str) -> str:
        if text_a.startswith("连接池上限="):
            return "C"
        own = set(_OWN_ROWS) | {_OWN_CLAIM_RENDER}
        if text_a in own and text_b in own:
            return "internal"
        return "A"

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
        content=content, subject="budget-own", tags=[], claims=[dict(_OWN_CLAIM)],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset(content)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)
    return tools, new, peer1


# ── §3.4-2 顺序：internal → C → A-cross（RecordingBackend 序列）───────────────

def test_dispatch_order_internal_then_c_then_a_cross(tmp_path, monkeypatch) -> None:
    tools, new, peer1 = _write_scene(tmp_path, monkeypatch)
    # C 派发但同值不浮出（否则去重翻转会按设计跳过 A-cross——那由专门
    # 用例钉），三相位各自派发可见。
    class _NoSurfacingRecorder(_Recorder):
        @classmethod
        def classify_pair(cls, left, right, **kw):
            if "dispatch_hint" in left and left.get("quote") == cls.own_content[:1000]:
                cls.calls.append(("C", str(right.get("quote") or "")))
                return ModelSignal(True, "attribute_value_extraction", None, "",
                                   {"attribute_a": "连接池上限", "value_a": "99",
                                    "attribute_b": "连接池上限", "value_b": "99"}, None)
            return super().classify_pair(left, right, **kw)

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _NoSurfacingRecorder)
    a_hits = [_peer_hit(peer1, 101, "连接池上限为 300，队列长度为 4。", 0.1)]
    c_hits = [_peer_hit(peer1, 201, "连接池上限为 100。", 0.2),
              _peer_hit(peer1, 202, "队列长度为 4。", 0.3)]
    _install_stubs(monkeypatch, tools, peer1=peer1, a_hits=a_hits, c_hits=c_hits,
                   c_vectors={201: [0.4, 0.9], 202: [0.4, 0.9]})

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    kinds = [k for k, _q in _NoSurfacingRecorder.calls]
    assert kinds[0] == "internal", f"internal Qwen must ride first, got {kinds}"
    assert kinds.count("C") == 2 and kinds.count("A") == 1, kinds
    assert kinds.index("A") > max(i for i, k in enumerate(kinds) if k == "C"), (
        "channel C must dispatch before the A-cross loop"
    )
    assert receipt["judge_budget"] == {"internal": 1, "channel_c": 2, "a_cross": 1}
    assert receipt["pairs_examined"] == 4


# ── §3.4-1 饱和：C 扣穿池 → A-cross 派发 0、direct 直出照常、backlog 落账 ─────

def test_saturation_c_overdraw_skips_dispatch_but_direct_lands(tmp_path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    peer1 = tools.memory_write(
        content="连接池上限为 300，队列长度为 4。", subject="budget-peer", tags=[],
    )["data"]
    peer2 = tools.memory_write(
        content="备份窗口为凌晨 2 点。", subject="backup-peer", tags=[],
    )["data"]
    content = "\n".join((*_OWN_ROWS, "备份窗口为凌晨 3 点。"))
    new = tools.memory_write(
        content=content, subject="budget-own", tags=[], claims=[dict(_OWN_CLAIM)],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset(content)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)
    # 12 个过带 C 命中：1 claim × 12 → C 派发 12 扣穿 10 池。
    c_hits = [_peer_hit(peer1, 300 + i, f"连接池片段 {i}。", 0.2 + i * 0.01) for i in range(12)]
    a_hits = [
        _peer_hit(peer1, 101, "连接池上限为 100，队列长度为 4。", 0.1),
        _peer_hit(peer2, 102, "备份窗口为凌晨 2 点。", 0.2),
    ]
    _install_stubs(monkeypatch, tools, peer1=peer1, a_hits=a_hits, c_hits=c_hits,
                   c_vectors={300 + i: [0.4, 0.9] for i in range(12)})

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    # C 扣穿：internal 1 + C 12 > 10 → A-cross Qwen 派发 0。
    assert not any(k == "A" for k, _q in _Recorder.calls), (
        "saturated pool must skip every A-cross Qwen dispatch"
    )
    assert receipt["judge_budget"]["channel_c"] == 12
    assert receipt["judge_budget"].get("a_cross", 0) == 0
    assert receipt["judge_budget"]["a_cross_dispatch_skipped"] is True
    # direct 直出照常落地（peer2 单键骨架 → 零 Qwen notice）。
    assert receipt.get("direct_verdicts", 0) >= 1
    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == int(new["id"])]
    # 0.17.1：direct 路径保留真属性（对抗 review 修复），占位符退役
    direct_notices = [n for n in notices
                      if n.get("payload", {}).get("reason") == "deterministic_same_key_value_diff"]
    assert direct_notices, "direct verdict must land even under saturation"
    # 未派发 Qwen 对进 backlog（饱和不终止确定性检查）。
    assert receipt.get("backlogged", 0) >= 1
    assert receipt["status"] == "completed" and receipt.get("truncated") is True


# ── §3.4-4 去重翻转：C 已浮出 peer → A-cross 跳过派发 ─────────────────────────

def test_c_surfaced_peer_skips_a_cross_dispatch(tmp_path, monkeypatch) -> None:
    _Recorder.c_surfaces = True
    tools, new, peer1 = _write_scene(tmp_path, monkeypatch)
    a_hits = [_peer_hit(peer1, 101, "连接池上限为 300，队列长度为 4。", 0.1)]
    c_hits = [_peer_hit(peer1, 201, "连接池上限为 100。", 0.2),
              _peer_hit(peer1, 202, "队列长度为 4。", 0.3)]
    _install_stubs(monkeypatch, tools, peer1=peer1, a_hits=a_hits, c_hits=c_hits,
                   c_vectors={201: [0.4, 0.9], 202: [0.4, 0.9]})

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    # C 浮出 peer（异值 notice 已建）→ A-cross 对该 peer 零派发、零 semantic_evidence notice。
    assert not any(k == "A" for k, _q in _Recorder.calls)
    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == int(new["id"])]
    c_notices = [n for n in notices if n.get("payload", {}).get("channel_c")]
    a_notices = [n for n in notices if n.get("notice_type") == "semantic_evidence"]
    assert len(c_notices) == 1 and not a_notices, "single report — C wins the peer"
    # skip 集合不耗预算：无 skipped 标记、无 a_cross 扣减。
    assert "a_cross" not in receipt["judge_budget"]
    assert "a_cross_dispatch_skipped" not in receipt["judge_budget"]
    assert receipt["judge_budget"] == {"internal": 1, "channel_c": 2}


# ── §3.4-4b 去重并集：B surfaced peer 同样进 A-cross skip 集合 ────────────────

def test_b_surfaced_peer_also_skips_a_cross_dispatch(tmp_path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    # peer_b 带同名异值 claims → 通道 B 浮出；其句子同时是 A 候选。
    peer_b = tools.memory_write(
        content="连接池上限为 300，队列长度为 4。", subject="budget-peer", tags=[],
        claims=[{"attr": "连接池上限", "value": "300"}],
    )["data"]
    content = "\n".join(_OWN_ROWS)
    new = tools.memory_write(
        content=content, subject="budget-own", tags=[], claims=[dict(_OWN_CLAIM)],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset(content)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)
    # C 不浮出（同值抽取 → unresolved），B 浮出 peer_b。
    class SameValueRecorder(_Recorder):
        @classmethod
        def classify_pair(cls, left, right, **kw):
            if "dispatch_hint" in left and left.get("quote") == cls.own_content[:1000]:
                cls.calls.append(("C", str(right.get("quote") or "")))
                return ModelSignal(True, "attribute_value_extraction", None, "",
                                   {"attribute_a": "连接池上限", "value_a": "99",
                                    "attribute_b": "连接池上限", "value_b": "99"}, None)
            return super().classify_pair(left, right, **kw)

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: SameValueRecorder)
    a_hits = [_peer_hit(peer_b, 101, "连接池上限为 300，队列长度为 4。", 0.1)]
    c_hits = [_peer_hit(peer_b, 201, "连接池上限为 100。", 0.2)]
    _install_stubs(monkeypatch, tools, peer1=peer_b, a_hits=a_hits, c_hits=c_hits,
                   c_vectors={201: [0.4, 0.9]})

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    claims_notices = [n for n in tools.db.list_semantic_notices()
                      if n["memory_id"] == int(new["id"]) and n.get("payload", {}).get("claims_channel")]
    a_notices = [n for n in tools.db.list_semantic_notices()
                 if n["memory_id"] == int(new["id"]) and n.get("notice_type") == "semantic_evidence"]
    assert len(claims_notices) == 1 and not a_notices, "B-surfaced peer must skip A-cross too"
    assert not any(k == "A" for k, _q in SameValueRecorder.calls)


# ── §3.4-5 形状：零活动写入回执与旧形状逐位一致（无新键）──────────────────────

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
    assert receipt == {
        "status": "completed", "outcome": "checked_no_notice", "notices_created": 0,
        "candidate_gates": {"prefiltered_rows": 1},
        "rows_mode": True, "rows_examined": 0,
        "claims_channel": {"channel_b_checked": 0, "channel_b_notices": 0},
        "claims_channel_c": {
            "channel_c": True, "channel_c_checked": 0, "channel_c_notices": 0,
        },
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


def test_stale_hit_peer_settled_not_backlogged(tmp_path, monkeypatch) -> None:
    """P2-1：派发相读到 peer 已编辑（fresh version > KNN 行版本）——证据/版本
    错位的 notice 不得落库；对按 settled 处理：不派发、不进 backlog、不扣池。"""
    tools, new, peer1 = _write_scene(tmp_path, monkeypatch)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _NoSurfacingRecorder)
    a_hits = [_peer_hit(peer1, 101, "连接池上限为 300，队列长度为 4。", 0.1)]
    c_hits = [_peer_hit(peer1, 201, "连接池上限为 100。", 0.2)]
    _install_stubs(monkeypatch, tools, peer1=peer1, a_hits=a_hits, c_hits=c_hits,
                   c_vectors={201: [0.4, 0.9]})
    original_by_ids = tools.db.get_memories_by_ids

    def bumped_by_ids(ids, **kw):
        rows = original_by_ids(ids, **kw)
        for row in rows.values():
            row["version"] = int(row.get("version") or 1) + 1  # 两相之间被编辑
        return rows

    monkeypatch.setattr(tools.db, "get_memories_by_ids", bumped_by_ids)

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert not any(k == "A" for k, _q in _NoSurfacingRecorder.calls), (
        "stale-hit peer must not reach Qwen dispatch"
    )
    a_notices = [n for n in tools.db.list_semantic_notices()
                 if n["memory_id"] == int(new["id"])
                 and n.get("notice_type") == "semantic_evidence"]
    assert not a_notices, "evidence/version mismatch notice must not land"
    assert "backlogged" not in receipt, "stale pair is settled — never backlogged"
    # C 相先于派发相跑、不受版本守卫影响（C 的 notice 以 hit 行版本为锚，
    # 证据-版本天然一致），故账本 = internal 1 + C 1；A-cross 零花费。
    assert receipt["judge_budget"] == {"internal": 1, "channel_c": 1}


def test_channel_c_deadline_stopped_key(tmp_path, monkeypatch) -> None:
    """P3：通道 C 撞公平墙停走必须 loud（channel_c_deadline_stopped 条件键）；
    墙在派发前拦住 → 零扣池、零 classify。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    peer1 = tools.memory_write(
        content="连接池上限为 300，队列长度为 4。", subject="c-wall-peer", tags=[],
    )["data"]
    new = tools.memory_write(
        content="连接池上限为 99。", subject="c-wall-own", tags=[], claims=[dict(_OWN_CLAIM)],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset("连接池上限为 99。")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)
    c_hits = [_peer_hit(peer1, 201, "连接池上限为 100。", 0.2)]
    monkeypatch.setattr(tools.db, "row_knn", lambda embedding, **kw: list(c_hits))
    spends: list[int] = []

    result = tools._evidence.check_claim_sentence_conflicts(
        int(new["id"]), tv._job_snapshot(tools, new["id"]),
        allowed_memory_ids=[int(peer1["id"])],
        notices_used=0, budget_sink=lambda: spends.append(1),
        deadline_fn=lambda: 1.0,  # 公平墙早已过去
    )

    assert result["channel_c_checked"] == 0 and result["channel_c_notices"] == 0
    assert result["channel_c_deadline_stopped"] is True
    assert spends == [] and not _Recorder.calls


def test_c_surfaced_and_saturated_combo(tmp_path, monkeypatch) -> None:
    _Recorder.c_surfaces = True
    """P3 弱钉补齐：C 浮出 peer2（进 skip 集合、settled 不进 backlog——连
    direct 可直出的对也让位）与池被 C 扣穿（peer1 无法派发、进 backlog）
    同时发生——两机制正交且互不遮蔽。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    peer1 = tools.memory_write(
        content="连接池上限为 300，队列长度为 4。", subject="budget-peer", tags=[],
    )["data"]
    peer2 = tools.memory_write(
        content="备份窗口为凌晨 2 点。", subject="backup-peer", tags=[],
    )["data"]
    content = "\n".join((*_OWN_ROWS, "备份窗口为凌晨 3 点。"))
    new = tools.memory_write(
        content=content, subject="budget-own", tags=[], claims=[dict(_OWN_CLAIM)],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset(content)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)
    # C 的 12 个命中全部指向 peer2：前两个带可 grounding 引文（值 100 在
    # 引文内 → 浮出 peer2），其余 10 个纯片段（unresolved）→ 派发 12 扣穿池。
    c_hits = [_peer_hit(peer2, 300, "连接池上限为 100。", 0.2),
              _peer_hit(peer2, 301, "连接池上限为 100。", 0.21)]
    c_hits += [_peer_hit(peer2, 302 + i, f"连接池片段 {i}。", 0.22 + i * 0.01) for i in range(10)]
    a_hits = [
        _peer_hit(peer1, 101, "连接池上限为 100，队列长度为 4。", 0.1),
        _peer_hit(peer2, 102, "备份窗口为凌晨 2 点。", 0.2),
    ]
    _install_stubs(monkeypatch, tools, peer1=peer1, a_hits=a_hits, c_hits=c_hits,
                   c_vectors={h["id"]: [0.4, 0.9] for h in c_hits})

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == int(new["id"])]
    c_notices = [n for n in notices if n.get("payload", {}).get("channel_c")]
    assert c_notices, "grounded C hit must surface peer2 (skip set)"
    assert not any(k == "A" for k, _q in _Recorder.calls)
    assert receipt["judge_budget"]["channel_c"] == 12
    # peer2 被 C 浮出 → skip 让位（哪怕它是 direct 可直出的对）；
    # peer1 非 direct、非 skip → 池尽被拦 → 唯一 backlog 对。
    assert receipt["judge_budget"]["a_cross_dispatch_skipped"] is True
    assert receipt.get("backlogged") == 1
