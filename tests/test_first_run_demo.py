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


def test_prompt_once_marker_pinned() -> None:
    """§0 全局一次原则（owner 2026-09-16）：标记查询门/拒绝写标记/完成写标记
    三面必须留在规程里；README 安装节同样先查标记。标记面在产品侧可用：
    写一条该标签记忆后 find(tags_filter) 必须命中（跨 Agent 共享状态载体）。"""
    import tempfile

    from memory_arbiter.config import Settings
    from memory_arbiter.db import MemoryDB
    from memory_arbiter.tools import MemoryTools

    text = DOC.read_text(encoding="utf-8")
    assert "mema-first-run-demo-status" in text
    assert "全局最多提示一次" in text
    assert "已拒绝" in text and "已完成" in text
    readme = (REPO / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "mema-first-run-demo-status" in readme

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            db_path=Path(tmp) / "t.sqlite3", backup_jsonl=Path(tmp) / "b.jsonl",
            client="t", agent_id="t",
        )
        tools = MemoryTools(settings, MemoryDB(settings))
        written = tools.memory(
            "remember",
            {"content": "首次功能演示已完成（钉子测试）。",
             "subject": "mema 首次演示状态：已完成", "tags": ["mema-first-run-demo-status"]},
        )
        assert written["ok"] is True
        found = tools.memory("find", {"query": "", "tags_filter": ["mema-first-run-demo-status"]})
        assert found["ok"] is True
        results = (found.get("data") or {}).get("results") or []
        assert any(item.get("id") == written["data"]["id"] for item in results)


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
