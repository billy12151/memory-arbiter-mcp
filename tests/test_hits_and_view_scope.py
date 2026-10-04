"""0.17.1 修复批 A7 + A8.

A7（P2）：read 邻句 hit_spans.text 与 span 回读不一致。命中行已改
content[s:e]（bb8b215），邻句仍 str(row["text"])（rowseg 折叠文本）——
同一响应内两种口径，违反 README「read span returns exactly that text」。
实测：matched=False text='名称:乙 金额:200' / readback='| 乙 | 200 |'。

A8（P2）：workspaces 视图 LIMIT 先于 ACL。operations 的 SQL LIMIT 在
surfaces 事后按 admitted 过滤之前 —— strict limit=1 时自有桶被更晚更新的
外来桶挤出窗口（实测 []），count 被重算谎报。

钉死契约：
- A7：read hits 邻句 text == content[s:e]（表格行 + 跨行句两形态）；
- A7：find 侧与 read 侧口径三方一致；
- A8：strict + 外来桶更晚更新 + limit=1 → 仍见自有桶（核心钉）；
- A8：count == len(workspaces)（不再谎报）；
- A8：none/weak 全量不变；admitted 空 → denied 优先。
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools

pytest.importorskip("sqlite_vec")

import tests.test_vnext_evidence as tv  # noqa: E402


def make_tools(tmp_path: Path, **kw) -> MemoryTools:
    tools = tv.make_tools(tmp_path)
    for key, value in kw.items():
        setattr(tools.settings, key, value)
    return tools


def _write(tools: MemoryTools, subject: str, content: str, workspace: str = "w") -> int:
    res = tools.memory_write(content=content, subject=subject, workspace=workspace, tags=["t"])
    assert res.get("ok"), res
    return int(res["data"]["id"])


# ── A7 ──────────────────────────────────────────────────────────────────────

def test_read_neighbour_text_matches_span_readback_table(tmp_path: Path) -> None:
    """核心钉：表格行邻句 text == content[s:e]（此前为折叠文本）。"""
    tools = make_tools(tmp_path)
    content = "会议纪要如下。\n\n| 名称 | 金额 |\n|---|---|\n| 甲 | 100 |\n| 乙 | 200 |\n\n结语在此。"
    mid = _write(tools, "s", content)
    assert tools.wait_semantic_worker_drained(timeout=120)
    out = tools.memory("read", {"memory_id": mid, "content_mode": "hits",
                                "hit_window": 2, "span": {"start": 31, "end": 42}})
    spans = ((out.get("data") or {}).get("memory") or {}).get("hit_spans") or []
    assert spans, out
    mismatched = [
        s for s in spans if s["text"] != content[s["start_offset"]:s["end_offset"]]
    ]
    assert not mismatched, f"邻句/命中行 text 必须等于 span 回读: {mismatched}"


def test_read_neighbour_text_matches_span_readback_wrapped(tmp_path: Path) -> None:
    """跨行句邻句同钉（换行折叠差异）。"""
    tools = make_tools(tmp_path)
    content = "第一句跨\n换行到此结束。第二句普通。第三句普通。"
    mid = _write(tools, "s", content)
    assert tools.wait_semantic_worker_drained(timeout=120)
    out = tools.memory("read", {"memory_id": mid, "content_mode": "hits",
                                "hit_window": 1, "span": {"start": 0, "end": 12}})
    spans = ((out.get("data") or {}).get("memory") or {}).get("hit_spans") or []
    for s in spans:
        assert s["text"] == content[s["start_offset"]:s["end_offset"]], s


def test_find_and_read_hit_text_share_one_dialect(tmp_path: Path) -> None:
    """三方一致：find 命中 / read 命中 / read 邻句 都用原文切片。"""
    tools = make_tools(tmp_path)
    content = "| 名称 | 金额 |\n|---|---|\n| 甲 | 100 |\n| 乙 | 200 |\n普通句子结束。"
    mid = _write(tools, "表", content)
    assert tools.wait_semantic_worker_drained(timeout=120)
    f = tools.memory("find", {"query": "名称 金额 甲", "content_mode": "hits",
                              "hit_window": 1, "workspace": "w"})
    for item in (f.get("data") or {}).get("results") or []:
        for s in item.get("hit_spans") or []:
            assert s["text"] == content[s["start_offset"]:s["end_offset"]], ("find", s)
    r = tools.memory("read", {"memory_id": mid, "content_mode": "hits", "hit_window": 1})
    for s in ((r.get("data") or {}).get("memory") or {}).get("hit_spans") or []:
        assert s["text"] == content[s["start_offset"]:s["end_offset"]], ("read", s)


# ── A8 ──────────────────────────────────────────────────────────────────────

def _seed_bucket(tools: MemoryTools, name: str, *, at: str) -> None:
    """直接 SQL 播种一个桶 + 一条 active 行（FakeEmbedder 会把非关键词
    workspace 折叠成同一向量，走 memory_write 无法造出多个独立桶）。"""
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES(?,?)",
            (name, at),
        )
        conn.execute(
            "INSERT INTO memories(workspace, workspace_canonical, subject, content, tags, "
            "source_type, status, version, created_at, agent_id, event_time, "
            "ingest_time, confidence, protection_level) "
            "VALUES(?,?,?,?,?,'agent_generated','active',1,?,'t',?,?,0.5,'normal')",
            (name, name, f"s-{name}", f"c-{name}", "[]", at, at, at),
        )


def test_strict_view_limit_one_sees_own_bucket(tmp_path: Path) -> None:
    """核心钉：strict + 外来桶更晚更新 + limit=1 → 仍返回自有桶（此前 []）。"""
    tools = make_tools(tmp_path, isolation="strict", workspace="Alpha")
    _seed_bucket(tools, "Alpha", at="2026-01-01T00:00:00+00:00")
    # 外来桶更新更晚（排序在前），旧实现会被 LIMIT 挤出窗口
    for i in range(3):
        _seed_bucket(tools, f"Other{i}", at=f"2026-06-0{i + 1}T00:00:00+00:00")
    for lim in (1, 2):
        res = tools.memory_review("workspaces", {"workspace": "Alpha", "limit": lim})
        data = res.get("data") or {}
        names = [b["canonical"] for b in data.get("workspaces") or []]
        assert "Alpha" in names, f"limit={lim} 必须见自有桶（此前被 LIMIT 挤掉）: {names}"


def test_view_count_equals_returned_rows(tmp_path: Path) -> None:
    """count == len(workspaces)（不再重算谎报）。"""
    tools = make_tools(tmp_path)
    _seed_bucket(tools, "Alpha", at="2026-01-01T00:00:00+00:00")
    _seed_bucket(tools, "Beta", at="2026-02-01T00:00:00+00:00")
    res = tools.memory_review("workspaces", {"limit": 50})
    data = res.get("data") or {}
    assert data.get("count") == len(data.get("workspaces") or []) == 2


def test_none_mode_view_unchanged(tmp_path: Path) -> None:
    """none 模式：全量（既有断言不回归）。"""
    tools = make_tools(tmp_path)
    _seed_bucket(tools, "Alpha", at="2026-01-01T00:00:00+00:00")
    _seed_bucket(tools, "Beta", at="2026-02-01T00:00:00+00:00")
    res = tools.memory_review("workspaces", {"limit": 50})
    names = {b["canonical"] for b in (res.get("data") or {}).get("workspaces") or []}
    assert {"Alpha", "Beta"} <= names


def test_strict_without_canonical_denied(tmp_path: Path) -> None:
    """strict 无 caller canonical → denied 优先（不因 admitted 空集变成空列表）。"""
    tools = make_tools(tmp_path, isolation="strict", workspace="")
    res = tools.memory_review("workspaces", {})
    assert res.get("ok") is False
