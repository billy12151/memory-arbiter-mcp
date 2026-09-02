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


def test_digit_run_series_entries_stay_quiet(tmp_path: Path) -> None:
    """Version-number series (subjects identical modulo digit runs) are not
    duplicates: "项目 0.15.1 发版清单" vs "项目 0.15.2 发版清单" must not fire
    even with identical tags (raw ratio measured 0.9524 >= 0.95)."""
    tools = make_tools(tmp_path)
    _write(tools, "项目 0.15.1 发版清单", ["mema-core", "release"])
    series = _write(tools, "项目 0.15.2 发版清单", ["mema-core", "release"])
    assert _similar_notices(series) == [], "digit-run series entries must not trigger the hint"


def test_exact_digit_subject_duplicate_still_fires(tmp_path: Path) -> None:
    """The series suppression only applies when the subjects differ: writing
    the exact same version-numbered subject twice is a true duplicate."""
    tools = make_tools(tmp_path)
    first = _write(tools, "项目 0.15.1 发版清单", ["mema-core", "release"])
    dup = _write(tools, "项目 0.15.1 发版清单", ["mema-core", "release"], content="same checklist again")
    hints = _similar_notices(dup)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["data"]["id"]


def test_tag_jaccard_folds_internal_whitespace(tmp_path: Path) -> None:
    """Tag normalization folds internal whitespace like subjects do:
    "金营 项目" and "金营  项目" are the same tag (Jaccard 1.0)."""
    tools = make_tools(tmp_path)
    first = _write(tools, "上线检查清单", ["金营 项目"])
    dup = _write(tools, "上线检查清单", ["金营  项目"])
    hints = _similar_notices(dup)
    assert len(hints) == 1
    match = hints[0]["matches"][0]
    assert match["memory_id"] == first["data"]["id"]
    assert match["tag_jaccard"] == 1.0


def _make_strict_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "strict.sqlite3",
        backup_jsonl=tmp_path / "strict.jsonl",
        workspace="projA",
        isolation="strict",
    )
    return MemoryTools(settings, MemoryDB(settings))


def _confirm_pending(tools: MemoryTools, memory_id: int) -> None:
    record = tools.db.get_memory(memory_id)
    if record["status"] != "pending":
        return
    confirmed = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": memory_id,
        "canonical": record["workspace_canonical"] or record["workspace"],
        "authorized": True,
    })
    assert confirmed["ok"] is True, confirmed


def test_strict_confirm_pending_workspace_adds_similar_active_notice(tmp_path: Path) -> None:
    """Strict-mode pending memories are activated by confirm/activate; the
    write-time duplicate hint must still fire once a second active duplicate
    exists in the same workspace."""
    tools = _make_strict_tools(tmp_path)
    # Both writes target the same *unconfirmed* workspace, so both are pending.
    first = tools.memory_write(
        content="body one", subject="proj onboarding checklist", tags=["ops"],
        workspace="projA",
    )
    second = tools.memory_write(
        content="body two", subject="proj onboarding checklist", tags=["ops"],
        workspace="projA",
    )
    assert first["data"]["record"]["status"] == "pending"
    assert second["data"]["record"]["status"] == "pending"

    first_confirmed = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": first["data"]["id"],
        "canonical": "projA",
        "authorized": True,
    })
    assert first_confirmed["ok"] is True
    assert _similar_notices(first_confirmed) == [], "first active memory has no peers"

    second_confirmed = tools.memory_govern("confirm_pending_workspace", {
        "memory_id": second["data"]["id"],
        "canonical": "projA",
        "authorized": True,
    })
    assert second_confirmed["ok"] is True
    hints = _similar_notices(second_confirmed)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["data"]["id"]


def test_strict_memory_activate_adds_similar_active_notice(tmp_path: Path) -> None:
    tools = _make_strict_tools(tmp_path)
    first = tools.memory_write(
        content="body one", subject="release checklist", tags=["release"],
        workspace="projA",
    )
    assert first["data"]["record"]["status"] == "pending"
    _confirm_pending(tools, first["data"]["id"])

    # Workspace is now confirmed; an explicit pending write can be activated
    # via memory_activate, and must still receive the duplicate hint.
    second = tools.memory_write(
        content="body two", subject="release checklist", tags=["release"],
        workspace="projA", status="pending",
    )
    assert second["data"]["record"]["status"] == "pending"

    activated_second = tools.memory_activate(second["data"]["id"], authorized=True)
    assert activated_second["ok"] is True, activated_second
    hints = _similar_notices(activated_second)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["data"]["id"]
