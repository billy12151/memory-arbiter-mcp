"""0.17.1 重标合一（owner 拍板 2026-10-03，方案 §2.2/§2.3）行为测试。

钉死的契约：
- 单一总池（_JobJudgeBudget）：internal 帽撤销、spend(source) 单入口、
  receipt 分账 internal/a_cross、a_cross_dispatch_skipped 键退役；
- 合一相位单批：internal keepers 与 A-cross 对在一批内判完（judge_pairs
  调用次数=ceil(N/8)），internal 对在提交列表前部（E10① land-first =
  批量内排序）；
- 池耗尽：internal 尾对静默消失（现状帽 break 的池化等价）、cross 记
  pairs_examined_capped 进 backlog；
- judge_fn 异常逐片免疫（R2 P1）：后端抛异常 → error verdict 按源分账
  （internal unannotated / cross 回 backlog），不再 worker_error；
- 回执形状：零活动写入与旧形状逐位一致；
- E10①：确定性相截断时 internal keepers 先落地，判定相整体跳过。
"""
from __future__ import annotations

from typing import Any

import pytest

import memory_arbiter.pipeline.evidence as ev
import memory_arbiter.pipeline.gates as _gates
from memory_arbiter.pipeline.evidence import _JobJudgeBudget
import tests.test_vnext_evidence as tv


# ── 池算术（纯对象层）────────────────────────────────────────────────────────

def test_judge_budget_pool_arithmetic() -> None:
    pool = _JobJudgeBudget(total=10)
    assert pool.receipt_block() is None  # 零值不出现
    assert pool.spend("internal") is True
    assert pool.spend("internal") is True
    assert pool.internal_used == 2
    pool.spend("a_cross")
    pool.spend("a_cross")
    assert pool.remaining == 6
    for _ in range(6):
        assert pool.spend("a_cross") is True
    # 池耗尽：统一 False，无 skip 标记键（a_cross_dispatch_skipped 退役）
    assert pool.spend("a_cross") is False
    assert pool.spend("internal") is False
    assert pool.pairs_examined == 10
    assert pool.receipt_block() == {"internal": 2, "a_cross": 8}


def test_judge_budget_single_pool_half_share() -> None:
    # 半池守恒（harness 实测回归修复）：internal 份额 ⌈total/2⌉——行密集
    # 语料的 internal keeper 不再吃光全池饿死跨记忆道；cross 保底 ⌊total/2⌋。
    pool = _JobJudgeBudget(total=3)
    assert pool.internal_share == 2
    assert pool.spend("internal") is True
    assert pool.spend("internal") is True
    assert pool.spend("internal") is False  # 超出半池份额 → 静默消失
    assert pool.spend("a_cross") is True    # cross 保底至少 1 席
    assert pool.pairs_examined == 3


# ── 共享场景构造 ────────────────────────────────────────────────────────────

_OWN_ROWS = ("连接池上限为 99，队列长度为 99。", "连接池上限为 100，队列长度为 100。")


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
    """合一判定记录器：internal（同记忆句对）与 A-cross（裸句对）共用一个
    judge_pairs 入口，按 text_a 归源。返回 conflict 判定。"""

    calls: list[tuple[str, str]] = []  # (kind, right_text)
    chunks: list[int] = []  # 每次 judge_pairs 收到的对数

    @classmethod
    def _verdict(cls):
        from memory_arbiter.semantic_judge import PairVerdict
        return PairVerdict("conflict",
                           {"conflict": 0.9, "no_conflict": 0.05, "possible_conflict": 0.05},
                           None, "test")

    @classmethod
    def judge_pairs(cls, pairs):
        cls.chunks.append(len(pairs))
        out = []
        for text_a, text_b in pairs:
            # judge 输入是行窗口（subject+对立行+邻行），不是裸行文本；both
            # text_a 都是 own 行窗口，区分在 text_b：internal 的 text_b 是
            # own 邻行窗口（含「队列长度为 99/100」），cross 的是 peer 行
            # （含「300」）。
            kind = "internal" if ("队列长度为 99" in text_b or "队列长度为 100" in text_b) else "A"
            cls.calls.append((kind, text_b))
            out.append(cls._verdict())
        return out

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.chunks = []


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
    _Recorder.reset()
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)
    return tools, new, peer1


# ── 回执形状：零活动写入无新键 ─────────────────────────────────────────────

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
    # 0.17.1 重标合一：qwen_budget 兼容 echo 退役
    assert "qwen_budget" not in receipt
    assert receipt == {
        "status": "completed", "outcome": "checked_no_notice", "notices_created": 0,
        "candidate_gates": {"prefiltered_rows": 1},
        "rows_mode": True, "rows_examined": 0,
    }


# ── E10①：确定性相截断 → internal keepers 先落地、判定相整体跳过 ─────────────

def test_internal_keepers_land_when_deterministic_phase_truncates(tmp_path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    new = tools.memory_write(
        content="连接池上限为 99。\n连接池上限为 100。", subject="e10", tags=[],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset()
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
    assert not _Recorder.calls, "truncation terminal skips the judge phase entirely"
    assert "judge_budget" not in receipt  # 判定相未运行 → 池零活动


# ── 合一相位：单批 + internal-first + 分账 ──────────────────────────────────

def test_merged_phase_single_batch_internal_first(tmp_path, monkeypatch) -> None:
    tools, new, peer1 = _write_scene(tmp_path, monkeypatch)
    peer_text = "连接池上限为 300，队列长度为 4。"
    _install_stubs(
        monkeypatch, tools, peer1=peer1,
        a_hits=[_peer_hit(peer1, 901, peer_text, 0.1)],
        c_hits=[], c_vectors={},
    )

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    # 单批：一次 _process 内全部判定发生在 drain 的切片里（N=2 → 1 片）；
    # internal 对在提交列表前部（E10① 批量内排序）。
    assert _Recorder.chunks == [2]
    assert _Recorder.calls[0][0] == "internal"
    assert _Recorder.calls[1][0] == "A"
    assert receipt["status"] == "completed"
    assert receipt["internal_conflicts"] == 1
    assert receipt["judge_budget"] == {"internal": 1, "a_cross": 1}
    assert receipt["pairs_examined"] == 2
    assert "qwen_budget" not in receipt
    assert receipt.get("deterministic_filter", {}).get("internal_qwen_confirmed") == 1


def test_pool_exhaustion_internal_tail_vanishes_cross_backlogs(tmp_path, monkeypatch) -> None:
    tools, new, peer1 = _write_scene(tmp_path, monkeypatch)
    peer_text = "连接池上限为 300，队列长度为 4。"
    _install_stubs(
        monkeypatch, tools, peer1=peer1,
        a_hits=[_peer_hit(peer1, 901, peer_text, 0.1)],
        c_hits=[], c_vectors={},
    )
    # 总池压到 1：internal keeper 吃掉唯一席位（判成），cross 对池耗尽 →
    # pairs_examined_capped + backlog。
    monkeypatch.setattr(ev, "_JobJudgeBudget", lambda total=500: _JobJudgeBudget(total=1))

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert [kind for kind, _ in _Recorder.calls] == ["internal"]  # internal 尾对无、cross 未派发
    assert receipt["internal_conflicts"] == 1
    assert receipt["judge_budget"] == {"internal": 1}
    assert receipt["pairs_examined"] == 1
    assert receipt["status"] == "incomplete"
    assert receipt["reason"] == "pairs_examined_capped"
    assert receipt["backlogged"] == 1


def test_judge_fn_exception_is_chunk_immune(tmp_path, monkeypatch) -> None:
    """R2 P1：后端异常 → error verdict 按源分账（cross 回 backlog +
    judge_backend_error 降级），不再 worker_error 掀翻整相。"""
    tools, new, peer1 = _write_scene(tmp_path, monkeypatch)
    peer_text = "连接池上限为 300，队列长度为 4。"
    _install_stubs(
        monkeypatch, tools, peer1=peer1,
        a_hits=[_peer_hit(peer1, 901, peer_text, 0.1)],
        c_hits=[], c_vectors={},
    )

    class _Boom:
        @staticmethod
        def judge_pairs(pairs):
            raise RuntimeError("boom")

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Boom())

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert receipt["status"] == "incomplete"
    assert receipt["reason"] == "judge_backend_error"
    assert receipt["backlogged"] == 1  # error 对回 backlog，不丢
    assert "judge_budget" in receipt  # 池已消费，分账可见


def test_pathological_internal_storm_bounded_by_pool(tmp_path, monkeypatch) -> None:
    """方案 §5.11 病态探针（R2 P1 防线）：非表格「同句型不同数值」多行记忆
    绕过 internal_noise_pair（其只拦表格单元/meta 行），O(n²) keeper 涌入
    ——合并相位下判定+落地必须被总池封顶（帽撤销后唯一的有界保证），
    job 正常完成不挂死。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    rows = "\n".join(f"重试第 {i} 次超时阈值为 {100 + i} 毫秒。" for i in range(40))
    new = tools.memory_write(content=rows, subject="storm", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset()
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)
    monkeypatch.setattr(ev, "_JobJudgeBudget", lambda total=500: _JobJudgeBudget(total=20))

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    # 判定入批对 ≤ 池；internal 落地 ≤ 池；池耗尽尾对静默消失（不落地）。
    assert len(_Recorder.calls) <= 20
    assert receipt.get("internal_conflicts", 0) <= 20
    assert receipt["status"] in ("completed", "incomplete")


def test_drain_stop_probe_yields_mid_batch(tmp_path, monkeypatch) -> None:
    """方案 §5.6 / R1 P2-3：drain 片间让路——越墙停发余片，已发片 verdicts
    照常落地，余片 error verdict 归 judge_budget_exhausted（与收集侧同一
    面墙同口径）回 backlog。1 internal + 9 cross = 10 对 → 2 片（8+2），
    第 2 片前探针触发。"""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    peers = [
        tools.memory_write(content=f"连接池上限为 300，队列长度为 {4 + i}。", subject="yield", tags=[])["data"]
        for i in range(9)
    ]
    own_content = "连接池上限为 99，队列长度为 99。\n连接池上限为 100，队列长度为 100。"
    new = tools.memory_write(content=own_content, subject="yield-own", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _Recorder.reset()
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _Recorder)

    def _subject_hit(peer: dict[str, Any]) -> dict[str, Any]:
        return {"memory_id": int(peer["id"]), "id": 900, "kind": "subject", "text": "subject",
                "subject": "yield", "tags": "[]", "distance": 0.5, "memory_row_version": 1}

    hits = [
        {"memory_id": peer["id"], "id": index + 1, "kind": "text",
         "text": f"连接池上限为 300，队列长度为 {4 + index}。", "start_offset": 0,
         "end_offset": len(f"连接池上限为 300，队列长度为 {4 + index}。"),
         "distance": 0.1 + index * 0.001, "memory_row_version": 1}
        for index, peer in enumerate(peers)
    ]

    def stub_row_knn(embedding: Any, **kw: Any) -> list[dict[str, Any]]:
        if kw.get("subject_rows_only"):
            return [_subject_hit(peers[0])]
        if kw.get("conn") is not None:
            return list(hits)
        return []

    monkeypatch.setattr(tools.db, "row_knn", stub_row_knn)
    monkeypatch.setattr(_gates, "candidate_cos_gate",
                        lambda own, hs, vecs: ([(h, 0.85) for h in hs], [], []))
    monkeypatch.setattr(tools.db.evidence, "row_vectors_for_ids", lambda ids, conn=None: {})
    # 收集期探针（每候选一次）返回 None；drain 开跑（chunk 1 已发出）后
    # 片间探针立即返回已过的墙 → 停发第 2 片。
    def _deadline(timeout):
        return None if not _Recorder.chunks else 1.0

    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", _deadline)

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert _Recorder.chunks == [8]  # 第 2 片未发出
    assert receipt["internal_conflicts"] == 1
    assert receipt["backlogged"] == 2  # 停发余片 error verdict 回 backlog
    # 来源守恒：收集期扣池的每个 cross 对要么落地 notice、要么回 backlog。
    # 场景绑定断言（本场景无直出/dedup/抑制/clear 排除）——禁止照抄到
    # 直出或混合场景（直出不耗池却产 notice，会误红）。
    assert receipt["notices_created"] == 7
    assert receipt["judge_budget"] == {"internal": 1, "a_cross": 9}  # 收集期 10 对全扣池
    assert receipt["notices_created"] + receipt["backlogged"] == receipt["judge_budget"]["a_cross"]
    # 有 notice 落地时 finalize 契约：completed + truncated + reasons_seen
    # 承载让路原因（R1 P2-1 修复口径：停发归 judge_budget_exhausted）
    assert receipt["truncated"] is True
    assert "judge_budget_exhausted" in receipt["reasons_seen"]


def test_internal_storm_cannot_starve_cross(tmp_path, monkeypatch) -> None:
    """半池守恒回归（harness 实测：无上限 internal-first 在行密集语料上
    吃光全池 → 跨记忆 notice 归零、backlog 顶帽驱逐）。钉：internal 超份额
    消失，cross 候选必拿到保底席位并落 notice。"""
    tools, new, peer1 = _write_scene(tmp_path, monkeypatch)
    peer_text = "连接池上限为 300，队列长度为 4。"
    _install_stubs(
        monkeypatch, tools, peer1=peer1,
        a_hits=[_peer_hit(peer1, 901, peer_text, 0.1)],
        c_hits=[], c_vectors={},
    )
    # 总池 2：internal 份额 ⌈2/2⌉=1，cross 保底 1。own 双行数值句 = 1 个
    # internal keeper + 1 个 cross 对——两侧都能判。
    monkeypatch.setattr(ev, "_JobJudgeBudget", lambda total=500: _JobJudgeBudget(total=2))

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    kinds = [kind for kind, _ in _Recorder.calls]
    assert kinds.count("internal") == 1   # 半池份额内
    assert kinds.count("A") == 1          # cross 保底席位未被饿死
    assert receipt["internal_conflicts"] == 1
    assert receipt["judge_budget"] == {"internal": 1, "a_cross": 1}
    assert receipt["notices_created"] == 1
