"""0.17.1 修复批 A4：扫描完成门与毒记忆推进（回卷设计）.

背景（对抗性复核实证）：
- 非尾位毒记忆：空页 → complete=True 且不复位 → _complete_round 清掉
  conflict_scan_required 门，而毒记忆从未被扫描（完整性宣称断裂）；
- 编辑过的旧记忆（id<last_id）：此前靠"complete=True → 下一 kick 重开
  round"自愈——即谎报完成是自愈机制。v1 方案只改返回值会造成真死锁
  （round 永不重开、门永不清），故改为**本轮内显式回卷**；
- 尾位毒记忆：last_id 不推进 → 单 kick 同 id 热重试（实测 253 次）。

钉死契约：
- 编辑旧 id 后下一 kick 回卷补扫、complete=True、pending 清零；
- 毒记忆在（非尾位）：kick complete=False、门仍 true、watermark 仍 NULL；
- 毒记忆修好后：下一 kick complete=True、门清；
- 尾位毒记忆：单 kick 内该 id 调用 ≤2 次；
- poison_failures 达上界 → 回执 poison_skipped；
- 现有 complete 断言不回归（无异常路径）。
"""
from __future__ import annotations

import json
from pathlib import Path

from memory_arbiter.constants import SCAN_POISON_MAX_FAILURES
from memory_arbiter.tools import MemoryTools

import tests.test_scan_pipeline as tsp  # noqa: E402


def make_tools(tmp_path: Path) -> MemoryTools:
    return tsp.make_tools(tmp_path)


def _write(tools: MemoryTools, subject: str, content: str, workspace: str = "ws") -> int:
    return tsp._write(tools, subject, content, workspace)


def _state(tools: MemoryTools) -> dict:
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT value FROM migration_state WHERE key='scan_pipeline_state'"
        ).fetchone()
    return json.loads(row["value"]) if row else {}


def _gate(tools: MemoryTools) -> str | None:
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT value FROM migration_state WHERE key='conflict_scan_required'"
        ).fetchone()
    return row["value"] if row else None


def _arm_gate(tools: MemoryTools) -> None:
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO migration_state(key,value) VALUES('conflict_scan_required','true') "
            "ON CONFLICT(key) DO UPDATE SET value='true'"
        )


def _watermarks(tools: MemoryTools) -> list[tuple[int, int | None]]:
    with tools.db.connection() as conn:
        return [
            (int(r["id"]), r["scan_watermark"])
            for r in conn.execute("SELECT id,scan_watermark FROM memories ORDER BY id")
        ]


def test_edited_old_memory_is_wrapped_and_scanned(tmp_path: Path) -> None:
    """核心钉（A4 死锁回归）：编辑 id<last_id 的记忆，下一 kick 回卷补扫。"""
    tools = make_tools(tmp_path)
    ids = [_write(tools, f"s{i}", f"内容 {i} 债务") for i in range(6)]
    _arm_gate(tools)
    r1 = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 3, "time_budget_s": 5})
    assert (r1.get("data") or {}).get("complete") is False
    # 编辑旧记忆（version 2 > watermark 1）→ 重新 pending 但 id < last_id
    tools.memory("update", {"memory_id": ids[0], "new_content": "内容 0 已修订", "reason": "t"})
    r2 = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 100, "time_budget_s": 5})
    d2 = r2.get("data") or {}
    assert d2.get("complete") is True, "回卷后应扫完并完成"
    assert d2.get("pending_memories") == 0
    assert _gate(tools) == "false", "覆盖完整后门才可清"
    wm = dict(_watermarks(tools))
    assert wm[ids[0]] == 2, "被编辑的旧记忆必须真正被重扫（watermark 追到 v2）"


def test_non_tail_poison_keeps_gate_armed(tmp_path: Path) -> None:
    """核心钉：非尾位毒记忆 → complete=False、门仍 true、watermark 仍 NULL。"""
    tools = make_tools(tmp_path)
    ids = [_write(tools, f"s{i}", f"内容 {i} 债务") for i in range(3)]
    _arm_gate(tools)
    sp = tools._scan_pipeline
    orig = sp._process_memory

    def boom(memory_id: int, **kw):
        if memory_id == ids[1]:
            raise RuntimeError("poison")
        return orig(memory_id, **kw)

    sp._process_memory = boom
    r = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 100, "time_budget_s": 5})
    d = r.get("data") or {}
    assert d.get("complete") is False, "有失败条时不得宣称完成（回卷已尽力补扫）"
    assert d.get("pending_memories") == 1
    assert _gate(tools) == "true", "覆盖不完整 → conflict_scan_required 门不得清"
    wm = dict(_watermarks(tools))
    assert wm[ids[1]] is None, "毒记忆水位不得推进"


def test_poison_recovery_completes_round(tmp_path: Path) -> None:
    """毒记忆修好后：下一 kick complete=True、门清。"""
    tools = make_tools(tmp_path)
    ids = [_write(tools, f"s{i}", f"内容 {i} 债务") for i in range(3)]
    _arm_gate(tools)
    sp = tools._scan_pipeline
    orig = sp._process_memory
    calls: dict[int, int] = {}

    def boom(memory_id: int, **kw):
        calls[memory_id] = calls.get(memory_id, 0) + 1
        if memory_id == ids[1]:
            raise RuntimeError("poison")
        return orig(memory_id, **kw)

    sp._process_memory = boom
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 100, "time_budget_s": 5})
    sp._process_memory = orig  # 修好
    r2 = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 100, "time_budget_s": 5})
    d2 = r2.get("data") or {}
    assert d2.get("complete") is True
    assert _gate(tools) == "false"
    assert dict(_watermarks(tools))[ids[1]] == 1


def test_tail_poison_no_hot_retry(tmp_path: Path) -> None:
    """尾位毒记忆：单 kick 内该 id 调用 ≤2 次（此前 253 次热重试）。"""
    tools = make_tools(tmp_path)
    ids = [_write(tools, f"s{i}", f"内容 {i} 债务") for i in range(3)]
    sp = tools._scan_pipeline
    orig = sp._process_memory
    calls: dict[int, int] = {}

    def boom(memory_id: int, **kw):
        calls[memory_id] = calls.get(memory_id, 0) + 1
        if memory_id == ids[-1]:  # 尾位
            raise RuntimeError("poison")
        return orig(memory_id, **kw)

    sp._process_memory = boom
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 100, "time_budget_s": 2})
    assert calls.get(ids[-1], 0) <= 2, f"尾位毒不得热重试（实测 calls={calls}）"


def test_poison_failures_reach_receipt(tmp_path: Path) -> None:
    """达上界的毒记忆在回执可见（poison_skipped），且计数落轮状态。"""
    tools = make_tools(tmp_path)
    ids = [_write(tools, f"s{i}", f"内容 {i} 债务") for i in range(3)]
    sp = tools._scan_pipeline
    orig = sp._process_memory

    def boom(memory_id: int, **kw):
        if memory_id == ids[1]:
            raise RuntimeError("poison")
        return orig(memory_id, **kw)

    sp._process_memory = boom
    for _ in range(SCAN_POISON_MAX_FAILURES + 1):
        r = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 100, "time_budget_s": 3})
        if (r.get("data") or {}).get("poison_skipped"):
            break
    d = r.get("data") or {}
    assert ids[1] in (d.get("poison_skipped") or []), f"达上界后必须可见: {d.get('poison_skipped')}"
    state = _state(tools)
    assert str(ids[1]) in (state.get("poison_failures") or {})


def test_recovered_kick_does_not_report_poison_after_threshold(tmp_path: Path) -> None:
    """2026-10-05 审查修正：poison_skipped 只报「本 kick 仍失败」的条目。

    修复前按累计计数筛选：毒 6 轮达上界 → 修好 → 恢复 kick 水位已推进、
    complete=True，回执却仍带 poison_skipped（与完成自相矛盾，agent 会按
    「仍有未覆盖记忆」处理一个已扫完的轮）。
    """
    tools = make_tools(tmp_path)
    ids = [_write(tools, f"s{i}", f"内容 {i} 债务") for i in range(3)]
    sp = tools._scan_pipeline
    orig = sp._process_memory

    def boom(memory_id: int, **kw):
        if memory_id == ids[1]:
            raise RuntimeError("poison")
        return orig(memory_id, **kw)

    sp._process_memory = boom
    last = None
    for _ in range(SCAN_POISON_MAX_FAILURES + 1):
        last = tools.memory_repair(
            "scan_pipeline", {"action": "kick", "max_memories": 100, "time_budget_s": 3},
        )
    # 前置确认：仍在失败时可见（口径不回退）
    assert ids[1] in ((last.get("data") or {}).get("poison_skipped") or [])
    sp._process_memory = orig  # 修好
    r2 = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 100, "time_budget_s": 5})
    d2 = r2.get("data") or {}
    assert d2.get("complete") is True
    assert "poison_skipped" not in d2, (
        f"恢复 kick（本 kick 无失败）不得报 poison: {d2.get('poison_skipped')}"
    )
    # 计数仍落轮状态（历史口径），只是不再进回执
    assert str(ids[1]) in (_state(tools).get("poison_failures") or {})


def test_complete_paths_do_not_regress(tmp_path: Path) -> None:
    """无异常路径：complete 语义不回归（回卷不引入多余轮次）。"""
    tools = make_tools(tmp_path)
    for i in range(3):
        _write(tools, f"s{i}", f"内容 {i} 债务")
    r = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 100, "time_budget_s": 5})
    d = r.get("data") or {}
    assert d.get("complete") is True
    assert d.get("pending_memories") == 0
    assert "poison_skipped" not in d
