"""FIRST_RUN_DEMO 规程引用钉子 — P0-2（D7-B，owner 2026-09-16 拍板）.

规程文档引用的产品接口改名/消失时，这里先红，逼着先改规程再动接口。
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "memory_arbiter" / "FIRST_RUN_DEMO.zh-CN.md"


def test_demo_doc_exists_with_core_sections() -> None:
    text = DOC.read_text(encoding="utf-8")
    for section in ("同意门", "能力与好处", "兜底扫描授权", "清理", "准确率汇报"):
        assert section in text, section


def test_referenced_tool_surface_exists() -> None:
    text = DOC.read_text(encoding="utf-8")
    # tags_filter 仍是 find 的合法参数
    from memory_arbiter.config import Settings
    import tempfile

    from memory_arbiter.db import MemoryDB
    from memory_arbiter.tools import MemoryTools

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            db_path=Path(tmp) / "t.sqlite3", backup_jsonl=Path(tmp) / "b.jsonl",
            client="t", agent_id="t",
        )
        tools = MemoryTools(settings, MemoryDB(settings))
        result = tools.memory("find", {"query": "x", "tags_filter": ["mema-demo"]})
        assert result["ok"] is True
    # 规程里的两个扫描任务名与 SCHEDULED_TASKS_SPEC 对齐
    from memory_arbiter.scan_tasks import SCHEDULED_TASKS_SPEC

    task_names = {task["name"] for task in SCHEDULED_TASKS_SPEC["tasks"]}
    assert {"conflict_scan", "governance_reminder"} <= task_names
    assert "conflict_scan" in text and "governance_reminder" in text


def test_demo_tag_and_purpose_conventions_pinned() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "mema-demo" in text
    assert "mema_feature_demo" in text
    assert re.search(r"run_id", text)
    # 冲突对 metadata 前提必须写明（写时检测配对硬约束）
    assert "entity" in text and "scope" in text
    # 诚实口径：不得宣称演示命中率=全库准确率
    assert "不代表" in text


def test_referenced_readback_surfaces_exist() -> None:
    """§2/§3①/§3③ 实操引用的面：subject 必填、按 ID read、notice list。"""
    import tempfile

    from memory_arbiter.config import Settings
    from memory_arbiter.db import MemoryDB
    from memory_arbiter.tools import MemoryTools

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            db_path=Path(tmp) / "t.sqlite3", backup_jsonl=Path(tmp) / "b.jsonl",
            client="t", agent_id="t",
        )
        tools = MemoryTools(settings, MemoryDB(settings))
        # §2 统一规范：subject 缺失被 remember 拒绝
        rejected = tools.memory("remember", {"content": "x", "workspace": "default"})
        assert rejected["data"]["field"] == "subject"
        written = tools.memory(
            "remember", {"content": "正文", "subject": "标题", "workspace": "default"},
        )
        # §3① 按 ID read 回全文
        readback = tools.memory("read", {"memory_id": written["data"]["id"]})
        assert (readback.get("data") or {}).get("memory", {}).get("content") == "正文"
        # §3③ 冲突 notice 的主动查询面
        listing = tools.memory_repair("notice", {"action": "list"})
        assert listing.get("ok") is True
        assert isinstance((listing.get("data") or {}).get("notices"), list)
