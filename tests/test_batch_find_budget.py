"""0.17.1 修复批 A6：batch_find 响应字节预算（死代码修复 + hits 洞）.

背景（对抗性复核实证）：预算门读 entry["memory"]，而 find 页条目是扁平
形状（content 在顶层）→ total_bytes 恒 0、门永不触发（实测 360KB 放行）；
hits 页的 hit_spans[].text 也不在预算内（实测未升级全文的 hits 页
509KB 放行，hint 还写 "no contents were returned"）。

钉死契约：
- full 页超 80KB → 整页降级（content 剥离、content_chars 保留全文长度）；
- hits 未升级全文页（无 content、hit_spans 巨大）→ 同门拦截；
- hits 已升级全文页降级后 hit_spans 保留 + 显式标记（绝不静默截断）；
- 预算计入 content + hit_spans 文本；preview 页不进此门；
- total_bytes 数值 = 实际字节和。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import BATCH_READ_FULL_BUDGET_BYTES
from memory_arbiter.tools import MemoryTools

pytest.importorskip("sqlite_vec")

import tests.test_vnext_evidence as tv  # noqa: E402


def make_tools(tmp_path: Path) -> MemoryTools:
    return tv.make_tools(tmp_path)


def _write(tools: MemoryTools, subject: str, content: str, workspace: str = "w", tags=None) -> int:
    res = tools.memory_write(
        content=content, subject=subject, workspace=workspace,
        tags=tags or ["big"], source_type="agent_generated",
    )
    assert res.get("ok"), res
    return int(res["data"]["id"])


def test_batch_find_full_page_over_budget_drops_content(tmp_path: Path) -> None:
    """单条 360KB full 页：此前 over_budget=None 全放行，现整页降级。"""
    tools = make_tools(tmp_path)
    big = "长内容测试。" * 20000  # 360KB utf-8
    _write(tools, "大内容记忆", big)
    res = tools.memory("batch_find", {
        "queries": [{"query": "大内容记忆"}], "content_mode": "full", "tags_filter": ["big"],
    })
    data = res["data"]
    assert data.get("over_budget") is True
    assert data.get("budget_bytes") == BATCH_READ_FULL_BUDGET_BYTES
    assert data.get("total_bytes", 0) > BATCH_READ_FULL_BUDGET_BYTES
    item = data["results"][0]
    assert "content" not in item, "降级必须剥离 content"
    assert item.get("content_chars", 0) >= 120000, "content_chars 保留全文长度（不重算）"


def test_batch_find_budget_counts_hit_spans(tmp_path: Path) -> None:
    """hits 未升级全文页（无 content、hit_spans 巨大）：此前 509KB 放行。

    构造：多段长句 + 高 hit_window，使 hit_spans 文本超 80KB 而 content
    因覆盖率不足 50% 被剥离。
    """
    tools = make_tools(tmp_path)
    # 200 段，每段 ~420 字符；命中若干段 + hit_window 拉宽 -> hit_spans 很大
    content = "\n\n".join(
        f"债务转移 相关句子 {i} " + "填" * 400 for i in range(200)
    )
    _write(tools, "big", content)
    assert tools.wait_semantic_worker_drained(timeout=120)
    res = tools.memory("batch_find", {
        "queries": [{"query": "债务转移"}], "content_mode": "hits",
        "hit_window": 3, "tags_filter": ["big"],
    })
    data = res["data"]
    item = data["results"][0]
    spans_bytes = sum(
        len(str(s.get("text") or "").encode()) for s in (item.get("hit_spans") or [])
        if isinstance(s, dict)
    )
    content_bytes = len(str(item.get("content") or "").encode())
    assert spans_bytes + content_bytes > BATCH_READ_FULL_BUDGET_BYTES, "用例前提：页内容超预算"
    assert data.get("over_budget") is True, "hits 页的 hit_spans 文本必须计入预算"
    assert data.get("total_bytes", 0) >= spans_bytes


def test_batch_find_hits_keeps_spans_with_marker_after_slim(tmp_path: Path) -> None:
    """hits 页降级后：hit_spans 保留（窗口坐标）+ 显式标记，不静默截断。"""
    tools = make_tools(tmp_path)
    content = "\n\n".join(
        f"债务转移 相关句子 {i} " + "填" * 400 for i in range(200)
    )
    _write(tools, "big", content)
    assert tools.wait_semantic_worker_drained(timeout=120)
    res = tools.memory("batch_find", {
        "queries": [{"query": "债务转移"}], "content_mode": "hits",
        "hit_window": 3, "tags_filter": ["big"],
    })
    item = res["data"]["results"][0]
    assert item.get("hit_spans"), "hit_spans 不得被静默剥离"
    if "content" not in item:
        assert item.get("hit_spans_truncated_by_budget") is True
    hint = str(res["data"].get("hint") or "")
    assert "budget" in hint


def test_batch_find_preview_page_skips_budget(tmp_path: Path) -> None:
    """preview 页无内容，不进预算门（不出现 over_budget 键）。"""
    tools = make_tools(tmp_path)
    _write(tools, "大内容记忆", "长内容测试。" * 20000)
    res = tools.memory("batch_find", {
        "queries": [{"query": "大内容记忆"}], "tags_filter": ["big"],
    })
    data = res["data"]
    assert data.get("over_budget") is None
    assert "content" not in data["results"][0]


def test_batch_find_total_bytes_equals_wire_content(tmp_path: Path) -> None:
    """total_bytes = 降级前的实际字节和（与 batch_read 同口径：触发预算的量）。"""
    tools = make_tools(tmp_path)
    a = "甲" * 60000   # ~180KB utf-8
    b = "乙" * 60000
    _write(tools, "甲记忆", a)
    _write(tools, "乙记忆", b)
    res = tools.memory("batch_find", {
        "queries": [{"query": "记忆"}], "content_mode": "full",
        "limit_per_query": 5, "tags_filter": ["big"],
    })
    data = res["data"]
    assert data.get("over_budget") is True
    expected = len(a.encode("utf-8")) + len(b.encode("utf-8"))
    assert data.get("total_bytes", 0) == expected
    for item in data["results"]:
        assert "content" not in item
        assert item.get("content_chars", 0) in (60000,)
