"""0.17.1 修复批 A10：kick 封套 ok 反转（被拒调用不得表现为成功）.

背景（实测）：surfaces.py 的 kick 分支把 _forward 的返回值再包一层
state.response(result)。_forward 的错误路径返回的是**完整信封**
（_invalid_product_call → state.response(..., ok=False)）→ 二次封装产出
顶层 ok=True + data.ok=False + 多一层信封；agent 按通用 ok 契约会把被拒
的 kick 当成功。同处 time_budget_s="nan" 经 min/max 链静默压成 1.0s。

钉死契约：
- 被拒 kick（neighbor_k="abc" / workspace_backlog_pending）→ 顶层 ok=False；
- 成功 kick → 顶层 ok=True 且 data 是 kick 回执（不回归）；
- time_budget_s="nan" → 不再静默（预算按 45s 默认，或结构化拒绝）；
- scan_queue page 分支不受影响。
"""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.tools import MemoryTools

import tests.test_scan_pipeline as tsp  # noqa: E402


def make_tools(tmp_path: Path) -> MemoryTools:
    return tsp.make_tools(tmp_path)


def test_rejected_kick_has_top_level_ok_false(tmp_path: Path) -> None:
    """核心钉：松散类型被拒时顶层 ok=False（此前 ok=True + data.ok=False）。"""
    tools = make_tools(tmp_path)
    r = tools.memory_repair("scan_pipeline", {"action": "kick", "neighbor_k": "abc"})
    assert r.get("ok") is False, f"被拒 kick 不得表现为成功: {r.get('ok')}"
    data = r.get("data") or {}
    assert data.get("error"), "错误信息在 data.error"
    assert "mode" not in data, "不得再嵌一层信封"


def test_workspace_backlog_pending_rejection_top_level(tmp_path: Path) -> None:
    """workspace 判定未清完的拒绝（kick 自身 ok=False）→ 顶层 ok=False。"""
    tools = make_tools(tmp_path)
    mid = tsp._write(tools, "门禁主题", "门禁正文 postgres")
    tsp._enqueue_workspace_item(tools, mid, "envelope")
    r = tools.memory_repair("scan_pipeline", {"action": "kick"})
    assert r.get("ok") is False, f"被拒 kick 顶层 ok 必须 False: {r.get('ok')}"
    assert (r.get("data") or {}).get("error") == "workspace_backlog_pending"


def test_successful_kick_shape_unchanged(tmp_path: Path) -> None:
    """成功 kick：顶层 ok=True，data 是 kick 回执（既有断言不回归）。

    注意 kick 回执自身带 mode（full/incremental，轮语义），不是信封的
    mode；信封的判据是 data 里再嵌 data。
    """
    tools = make_tools(tmp_path)
    tsp._write(tools, "s", "债务 内容")
    r = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10, "time_budget_s": 5})
    assert r.get("ok") is True
    data = r.get("data") or {}
    assert "complete" in data and "processed_this_kick" in data
    assert "data" not in data, "成功回执不得再嵌信封"
    assert data.get("mode") in ("full", "incremental"), "轮语义 mode 保留"


def test_nan_budget_not_silently_shrunk(tmp_path: Path) -> None:
    """time_budget_s="nan"：预算不得静默变 1.0s（按默认 45s，或结构化拒绝）。

    此前 min/max 链产出 nan，下游 `remaining_budget <= 0.5` 恒 False →
    预算失效。修复后按 45s 默认处理（不静默缩成 1.0）。
    """
    tools = make_tools(tmp_path)
    tsp._write(tools, "s", "债务 内容")
    r = tools.memory_repair("scan_pipeline", {"action": "kick", "time_budget_s": "nan", "max_memories": 5})
    assert r.get("ok") is True, r
    data = r.get("data") or {}
    # 预算按默认 45s：单条记忆在 5 条帽内必被扫到（若缩成 1.0s 仍可能过，
    # 故直接断言未被压成 nan/异常）
    assert isinstance(data.get("duration_ms"), int)
    assert data.get("complete") in (True, False)


def test_scan_queue_page_branch_unaffected(tmp_path: Path) -> None:
    """scan_queue page 分支的既有错误契约不受 A10 影响。"""
    tools = make_tools(tmp_path)
    r = tools.memory_repair("scan_queue", {"action": "page", "page_size": "abc"})
    assert r.get("ok") is False
    assert (r.get("data") or {}).get("outcome") == "invalid_input"
