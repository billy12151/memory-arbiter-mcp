"""0.15.9 batch_find tests (mema 923 §6): boundary validation, merge/dedup
semantics, per-query stats, surface dispatch, size aggregation."""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    return MemoryTools(Settings(db_path=tmp_path / "m.db", backup_jsonl=tmp_path / "b.jsonl"))


def _write(tools: MemoryTools, subject: str, content: str, workspace: str = "w", tags: list[str] | None = None) -> int:
    res = tools.memory_write(
        content=content, subject=subject, workspace=workspace,
        source_type="agent_generated", tags=tags or [],
    )
    assert res.get("ok"), res
    return int(res["data"]["id"])


def _batch(tools: MemoryTools, **payload):
    return tools.memory(action="batch_find", data=payload)


# ── boundary / fail-fast ────────────────────────────────────────────────────

def test_batch_find_requires_nonempty_queries(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    res = _batch(tools, queries=[])
    assert not res.get("ok")
    assert res["data"]["error"]


def test_batch_find_rejects_more_than_eight_queries(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    res = _batch(tools, queries=[{"query": f"查询词组{i}号"} for i in range(9)])
    assert not res.get("ok")
    assert "at most 8" in str(res["data"])


def test_batch_find_rejects_duplicate_default_ids_and_duplicate_queries(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    res = _batch(tools, queries=[{"query": "催收 话术"}, {"query": "催收 话术"}])
    assert not res.get("ok")
    assert "duplicate query" in str(res["data"])
    res2 = _batch(tools, queries=[{"id": "a", "query": "催收 话术"}, {"id": "a", "query": "债务转移"}])
    assert not res2.get("ok")
    assert "duplicate query id" in str(res2["data"])


def test_batch_find_rejects_empty_query_and_bad_limit(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    res = _batch(tools, queries=[{"query": "   "}])
    assert not res.get("ok")
    res2 = _batch(tools, queries=[{"query": "催收"}], limit_per_query=21)
    assert not res2.get("ok")
    assert "limit_per_query" in str(res2["data"])
    res3 = _batch(tools, queries=[{"query": "催收", "extra": 1}])
    assert not res3.get("ok")
    assert "unknown item field" in str(res3["data"])


def test_batch_find_rejects_oversized_payload(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    huge = "债务转移" * 4000  # 16k chars → 48KB+ per item, well over 64KB total
    res = _batch(tools, queries=[{"query": huge}, {"query": huge + "x"}])
    assert not res.get("ok")
    assert res["data"].get("error") == "resource_limit_exceeded"


# ── merge / dedup semantics ─────────────────────────────────────────────────

def test_batch_find_dedup_merges_and_reports_matched_query_ids(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    shared = _write(tools, "催收与债务转移条文", "催收与债务转移的完整条文内容")
    _write(tools, "催收专属规则", "催收行为规范，其他内容无关")
    _write(tools, "债务转移专属笔记", "债务转移相关笔记一条")
    res = _batch(tools, queries=[
        {"id": "c", "query": "催收 条文"},
        {"id": "d", "query": "债务转移 条文"},
    ], limit_per_query=3)
    assert res.get("ok"), res
    data = res["data"]
    by_id = {int(r["id"]): r for r in data["results"]}
    assert shared in by_id
    assert by_id[shared]["matched_query_ids"] == ["c", "d"]
    assert by_id[shared]["best_query_id"] in {"c", "d"}
    for row in data["results"]:
        assert row["matched_query_ids"] and row["best_query_id"] in row["matched_query_ids"]
        assert "content" not in row  # index-page preview default
        assert "outline" in row and "content_chars" in row
    assert data["deduplicated"] is True
    # The relevance floor legitimately drops the weak half-token subject hits
    # (催收专属规则 matches 催收 only); counts are per-query page sizes.
    assert data["per_query"] == [
        {"id": "c", "count": 1, "has_more": False, "retrieval_mode": "direct"},
        {"id": "d", "count": 2, "has_more": False, "retrieval_mode": "direct"},
    ]
    # per_query has no dead error field (fail-fast contract)
    assert "error" not in data["per_query"][0]


def test_batch_find_no_dedup_keeps_one_item_per_query(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    shared = _write(tools, "催收与债务转移条文", "催收与债务转移的完整条文内容")
    res = _batch(tools, queries=[
        {"id": "c", "query": "催收 条文"},
        {"id": "d", "query": "债务转移 条文"},
    ], limit_per_query=3, deduplicate=False)
    data = res["data"]
    shared_rows = [r for r in data["results"] if int(r["id"]) == shared]
    assert len(shared_rows) == 2
    assert {tuple(r["matched_query_ids"]) for r in shared_rows} == {("c",), ("d",)}
    # query submission order first, in-query order second
    qids = [r["best_query_id"] for r in data["results"]]
    assert qids.index("c") < qids.index("d")
    assert data["deduplicated"] is False


def test_batch_find_truncates_per_query_before_merge(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for i in range(5):
        _write(tools, f"催收规则第{i}版", f"催收行为规范第 {i} 版正文")
    res = _batch(tools, queries=[{"query": "催收 规则"}], limit_per_query=2)
    data = res["data"]
    assert len(data["results"]) == 2  # per-query slice happens BEFORE merge
    assert data["per_query"][0]["count"] == 2


def test_batch_find_unmatched_query_reports_empty_not_recent(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "无关主题记忆", "关于别的主题的一段内容")
    res = _batch(tools, queries=[
        {"id": "hit", "query": "无关主题 记忆"},
        {"id": "miss", "query": "南极科考补给计划"},
    ])
    data = res["data"]
    modes = {p["id"]: p["retrieval_mode"] for p in data["per_query"]}
    assert modes["miss"] == "empty"
    miss_stats = next(p for p in data["per_query"] if p["id"] == "miss")
    assert miss_stats["count"] == 0
    subjects = [r["subject"] for r in data["results"]]
    assert subjects  # the hit query's results still arrive
    # nothing from the missed query leaks in as recency stuffing
    assert len(data["results"]) == data["per_query"][0]["count"]


def test_batch_find_shared_workspace_filter_applies_to_all(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "催收规则甲", "催收行为规范正文", workspace="projA")
    _write(tools, "催收规则乙", "催收行为规范正文", workspace="projB")
    res = _batch(tools, queries=[
        {"query": "催收 规则"}, {"query": "催收 规范"},
    ], workspace="projA")
    assert all(r["workspace"] == "projA" for r in res["data"]["results"])


def test_batch_find_default_id_is_query_text(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "催收与债务转移条文", "催收与债务转移的完整条文内容")
    res = _batch(tools, queries=[{"query": "催收 条文"}])
    assert res["data"]["per_query"][0]["id"] == "催收 条文"
    assert res["data"]["results"][0]["matched_query_ids"] == ["催收 条文"]


def test_batch_find_size_block_and_help_surface(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "催收规则", "催收行为规范正文")
    res = _batch(tools, queries=[{"query": "催收 规则"}])
    size = res["data"]["size"]
    assert size["returned_count"] >= 1
    assert "batch_find merged page" in size["display_hint"]
    help_doc = tools.memory(action="help")
    assert "batch_find" in help_doc["data"]["actions"]
    assert "batch_find" in help_doc["data"]["examples"]


@pytest.mark.parametrize("bad", [None, "x", 3.5, [{}]])
def test_batch_find_rejects_non_list_queries(tmp_path: Path, bad) -> None:
    tools = make_tools(tmp_path)
    res = _batch(tools, queries=bad)
    assert not res.get("ok")
