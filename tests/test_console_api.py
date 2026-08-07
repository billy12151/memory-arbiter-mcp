from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.console_api import ConsoleAPI
from memory_arbiter.tools import MemoryTools


def _api(tmp_path: Path) -> ConsoleAPI:
    settings = Settings(
        db_path=tmp_path / "console.sqlite3",
        backup_jsonl=tmp_path / "console.jsonl",
        client="pytest",
        agent_id="console-test",
        workspace="console-ws",
    )
    return ConsoleAPI(MemoryTools(settings))


def test_overview_returns_counts_and_brand(tmp_path: Path) -> None:
    api = _api(tmp_path)
    api.tools.memory_write(
        content="confirmed fact",
        subject="Confirmed",
        tags=["console"],
        source_type="user_confirmed",
        workspace="console-ws",
        agent_id="test",
    )
    overview = api.overview()
    assert overview["brand"]["en"] == "mema"
    assert overview["brand"]["zh"] == "迷码"
    assert overview["counts"]["total"] == 1
    assert overview["counts"]["active"] == 1
    assert overview["support"]["repo_url"] == "https://github.com/billy12151/memory-arbiter-mcp"
    assert overview["support"]["new_issue_url"].endswith("/issues/new")


def test_conflict_detail_returns_left_and_right(tmp_path: Path) -> None:
    api = _api(tmp_path)
    left = api.tools.memory_write(
        content="old scope",
        subject="Old",
        tags=["console"],
        source_type="agent_generated",
        workspace="console-ws",
        agent_id="test",
    )["data"]["id"]
    right = api.tools.memory_write(
        content="new scope",
        subject="New",
        tags=["console"],
        source_type="user_confirmed",
        workspace="console-ws",
        agent_id="test",
    )["data"]["id"]
    conflict = api.tools.memory_record_conflict(
        left_id=left,
        right_id=right,
        reason="scope differs",
        conflict_type="evolution",
        conflict_point="Console MVP scope changed",
        suggested_winner=right,
        confidence_hint="high",
        source="llm_informed",
    )["data"]
    detail = api.conflict_detail(conflict["conflict_id"])
    assert detail["conflict"]["conflict_point"] == "Console MVP scope changed"
    assert detail["left"]["memory"]["id"] == left
    assert detail["right"]["memory"]["id"] == right
    assert detail["winner_side"] == "right"


def test_conflict_detail_exposes_resolution_guidance(tmp_path: Path) -> None:
    api = _api(tmp_path)
    left = api.tools.memory_write(
        content="old full rule", subject="Old", workspace="console-ws",
    )["data"]["id"]
    right = api.tools.memory_write(
        content="new full rule", subject="New", workspace="console-ws",
    )["data"]["id"]
    conflict = api.tools.memory_record_conflict(
        left_id=left, right_id=right, reason="full replacement",
        conflict_type="evolution", suggested_winner=right,
    )["data"]
    with api.tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE conflicts SET resolution_kind='full_replacement', "
            "conflict_scope='whole_memory' WHERE id=?",
            (conflict["conflict_id"],),
        )

    detail = api.conflict_detail(conflict["conflict_id"])

    assert detail["conflict"]["resolution_kind"] == "full_replacement"
    assert detail["conflict"]["conflict_scope"] == "whole_memory"
    assert detail["conflict"]["recommended_resolution_action"] == "supersede_old_memory"
    assert detail["conflict"]["supersede_candidate"] is True


def test_memories_expired_invalid_offset_defaults_to_zero(tmp_path: Path) -> None:
    api = _api(tmp_path)
    result = api.memories(status="expired", offset="abc")
    assert result["status"] == "expired"
    assert result["items"] == []


def test_memories_preserves_tool_error_for_strict_isolation(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "console.sqlite3",
        backup_jsonl=tmp_path / "console.jsonl",
        client="pytest",
        agent_id="console-test",
        workspace="console-ws",
        isolation="strict",
    )
    api = ConsoleAPI(MemoryTools(settings))
    result = api.memories(status="active")
    assert "error" in result
    assert "workspace" in result["error"]


def test_settings_view_handles_missing_config_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.delenv("MEMORY_ARBITER_CONFIG", raising=False)
    api = _api(tmp_path)
    view = api.settings_view()
    assert view["config_file"]["path"] is None
    assert view["config_file"]["exists"] is False


def test_memories_rejects_unknown_status(tmp_path: Path) -> None:
    api = _api(tmp_path)
    result = api.memories(status="deleted")
    assert result["error"] == "status must be active or expired"
    assert result["_http_status"] == 400


def test_memories_rejects_offset_for_active(tmp_path: Path) -> None:
    api = _api(tmp_path)
    result = api.memories(status="active", offset=30)
    assert "error" in result
    assert result["_http_status"] == 400


def test_settings_view_exposes_isolation(tmp_path: Path) -> None:
    api = _api(tmp_path)
    view = api.settings_view()
    isolation = next(
        item for group in view["groups"] for item in group["items"] if item["path"] == "isolation"
    )
    assert isolation["current"] == "none"
    assert isolation["label_zh"] == "工作区隔离等级"


def test_settings_view_is_read_only_and_bilingual(tmp_path: Path) -> None:
    api = _api(tmp_path)
    view = api.settings_view()
    assert view["read_only"] is True
    items = [item for group in view["groups"] for item in group["items"]]
    assert items
    assert all(item["editable"] is False for item in items)
    assert all(item["label_en"] and item["label_zh"] for item in items)
