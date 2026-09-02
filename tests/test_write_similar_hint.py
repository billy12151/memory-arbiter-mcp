"""Write-time duplicate hint: subject/tags similarity over same-workspace
active memories (owner spec 2026-09-02). Deterministic, model-free, info
notice with an agent-first triage instruction; deliberate series entries
must stay quiet.
"""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    return MemoryTools(settings, MemoryDB(settings))


def _write(tools: MemoryTools, subject: str, tags: list[str], workspace: str = "w", content: str = "body"):
    return tools.memory_write(content=content, subject=subject, tags=tags, workspace=workspace)


def _similar_notices(result: dict) -> list[dict]:
    return [n for n in result.get("notices") or [] if n.get("type") == "similar_active_memory"]


def test_exact_duplicate_subject_and_tags_fires_hint(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = _write(tools, "部署方案：金营项目上线流程", ["deployment", "金营"])
    assert _similar_notices(first) == []
    second = _write(tools, "部署方案：金营项目上线流程", ["deployment", "金营"], content="rewrite of the same fact")
    hints = _similar_notices(second)
    assert len(hints) == 1
    match = hints[0]["matches"][0]
    assert match["memory_id"] == first["data"]["id"]
    assert match["subject_similarity"] >= 0.95
    assert match["tag_jaccard"] == 1.0
    assert "Triage silently" in hints[0]["agent_instruction"]
    assert second["ok"] is True


def test_series_entries_stay_quiet(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "mema-core Tier 1 功能方案：治理化 merge / 定时任务引导", ["mema-core", "roadmap", "plan"])
    series = _write(
        tools,
        "mema-core Tier 2 功能候选：访问信号排序 / 遗忘归档生命周期",
        ["mema-core", "roadmap", "plan"],
    )
    assert _similar_notices(series) == [], "deliberate series entries must not trigger the hint"


def test_tag_overlap_alone_does_not_fire(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "数据库选型记录", ["mema", "infra"])
    other = _write(tools, "完全不同主题的会议纪要", ["mema", "infra"])
    assert _similar_notices(other) == []


def test_same_subject_disjoint_tags_does_not_fire(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "季度目标", ["team-a"])
    other = _write(tools, "季度目标", ["team-b"])
    assert _similar_notices(other) == []


def test_no_tags_on_either_side_subject_decides(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = _write(tools, "API token 轮换流程", [])
    dup = _write(tools, "API token 轮换流程", [])
    hints = _similar_notices(dup)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["data"]["id"]


def test_cross_workspace_and_superseded_stay_quiet(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = _write(tools, "同名部署方案", ["ops"], workspace="w")
    # Different workspace: no leakage.
    cross = _write(tools, "同名部署方案", ["ops"], workspace="other")
    assert _similar_notices(cross) == []
    # Superseded originals no longer count as active duplicates.
    tools.memory_supersede(memory_id=first["data"]["id"], reason="gone", authorized=True)
    after = _write(tools, "同名部署方案", ["ops"])
    assert _similar_notices(after) == []
