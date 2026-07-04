from __future__ import annotations

from pathlib import Path

from memory_arbiter.arbitration import compare_memories
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import SourceType
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=False,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_write_and_search(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="Project API token policy lives in README security section.",
        tags=["policy", "security"],
        source_type="document_extracted",
        event_time="2026-01-01T00:00:00Z",
        subject="api-token-policy",
    )
    assert written["ok"] is True
    assert written["data"]["id"] is not None

    found = tools.memory_search(query="token policy", workspace="repo-a")
    assert found["ok"] is True
    assert found["data"]["count"] >= 1
    assert found["data"]["results"][0]["subject"] == "api-token-policy"


def test_chinese_search_matches_contiguous_fragment(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="京东科技金融营销系统梳理，覆盖金营、营销链路、活动费用申请和活动配置。",
        tags=["营销系统", "金营", "系统梳理"],
        source_type="document_extracted",
        subject="京东科技金融-营销系统梳理",
    )

    found = tools.memory_search(query="营销系统", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] >= 1
    assert found["data"]["results"][0]["subject"] == "京东科技金融-营销系统梳理"


def test_search_returns_recent_memories_when_no_direct_match(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="来源：营销系统梳理.xlsx。营销链路包含需求提报、费用申请、设计图提报、活动配置等环节。",
        tags=["营销系统", "金营", "PRD参考"],
        source_type="document_extracted",
        subject="京东科技金融-营销系统梳理",
        event_time="2026-07-04T09:00:30Z",
    )

    found = tools.memory_search(query="营销运营全流程 系统", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] == 1
    assert found["data"]["results"][0]["subject"] == "京东科技金融-营销系统梳理"
    assert any("No direct memory match" in warning for warning in found["warnings"])


def test_memory_recent_lists_recent_workspace_memories(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="Old memory", subject="old", event_time="2026-01-01T00:00:00Z")
    tools.memory_write(content="New memory", subject="new", event_time="2026-02-01T00:00:00Z")
    tools.memory_write(content="Other workspace", subject="other", workspace="repo-b", event_time="2026-03-01T00:00:00Z")

    recent = tools.memory_recent(workspace="repo-a", limit=10)

    assert recent["ok"] is True
    assert [record["subject"] for record in recent["data"]["results"]] == ["new", "old"]


def test_arbitration_prefers_event_time(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    old = tools.memory_write(content="Use port 3000", subject="dev-port", event_time="2026-01-01T00:00:00Z")
    new = tools.memory_write(content="Use port 5173", subject="dev-port", event_time="2026-02-01T00:00:00Z")
    result = tools.memory_arbitrate(old["data"]["id"], new["data"]["id"], mark_conflict=True)

    assert result["ok"] is True
    assert result["data"]["comparison"]["winner_id"] == new["data"]["id"]
    assert result["data"]["conflict_id"] is not None


def test_user_confirmed_is_protected(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    confirmed = tools.memory_write(
        content="User says production branch is main",
        subject="prod-branch",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    generated = tools.memory_write(
        content="Agent guessed production branch is release",
        subject="prod-branch",
        source_type="agent_generated",
        event_time="2026-03-01T00:00:00Z",
    )
    result = tools.memory_arbitrate(confirmed["data"]["id"], generated["data"]["id"], apply=True)

    assert result["data"]["comparison"]["winner_id"] == confirmed["data"]["id"]
    assert "automatic overwrite is forbidden" in result["data"]["comparison"]["reasons"][0]


def test_confirm_promotes_record(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(content="OpenClaw memory plugin is enabled for agent alpha", source_type="pending")
    confirmed = tools.memory_confirm(written["data"]["id"], source_ref="user-chat")

    assert confirmed["ok"] is True
    assert confirmed["data"]["record"]["source_type"] == SourceType.USER_CONFIRMED.value
    assert confirmed["data"]["record"]["protection_level"] == "locked"


def test_degraded_status_mentions_missing_vec(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    status = tools.memory_status()

    assert status["ok"] is True
    assert status["degraded"] is True
    assert any("sqlite-vec disabled" in warning for warning in status["warnings"])


def test_compare_manual_review_when_both_protected() -> None:
    left = {
        "id": 1,
        "source_type": "user_confirmed",
        "protection_level": "locked",
        "event_time": "2026-01-01T00:00:00Z",
        "ingest_time": "2026-01-02T00:00:00Z",
    }
    right = {
        "id": 2,
        "source_type": "user_confirmed",
        "protection_level": "locked",
        "event_time": "2026-02-01T00:00:00Z",
        "ingest_time": "2026-02-02T00:00:00Z",
    }
    result = compare_memories(left, right)

    assert result["manual_review"] is True
    assert result["winner_id"] is None
