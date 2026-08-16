from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools
from memory_arbiter.validation import MAX_CONTENT_BYTES


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    return MemoryTools(settings, MemoryDB(settings))


def test_remember_requires_content_and_subject(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for data, field in (({"subject": "s"}, "content"), ({"content": "x"}, "subject")):
        result = tools.memory("remember", data)
        assert result["ok"] is False
        assert result["data"]["field"] == field


def test_content_limit_is_utf8_bytes_and_inclusive(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    accepted = tools.memory("remember", {"content": "a" * MAX_CONTENT_BYTES, "subject": "limit"})
    assert accepted["ok"] is True
    rejected = tools.memory("remember", {"content": "a" * (MAX_CONTENT_BYTES + 1), "subject": "limit"})
    assert rejected["ok"] is False
    assert rejected["data"]["error"] == "resource_limit_exceeded"


def test_unknown_field_warns_but_sensitive_typo_rejects(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    warned = tools.memory("remember", {"content": "x", "subject": "s", "harmless_extra": 1})
    assert warned["ok"] is True
    assert "unknown field ignored: harmless_extra" in warned["warnings"]
    rejected = tools.memory("remember", {"content": "x", "subject": "s", "workspcae": "secret"})
    assert rejected["ok"] is False
    assert rejected["data"]["did_you_mean"] == "workspace"


def test_unknown_field_name_cannot_remove_authorization(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory("remember", {"content": "x", "subject": "s"})
    memory_id = written["data"]["id"]
    result = tools.memory_govern(
        "retire",
        {
            "memory_id": memory_id,
            "reason": "authorized test",
            "authorized": True,
            "noise: authorized": "ignored",
        },
    )
    assert result["ok"] is True


def test_product_ids_must_be_positive(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for value in (0, -1, True):
        result = tools.memory("read", {"memory_id": value})
        assert result["ok"] is False
        assert result["data"]["field"] == "memory_id"


def test_non_finite_confidence_and_bad_time_rejected(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    assert tools.memory("remember", {"content": "x", "subject": "s", "confidence": "NaN"})["ok"] is False
    result = tools.memory("remember", {"content": "x", "subject": "s", "event_time": "yesterday"})
    assert result["ok"] is False
    assert result["data"]["field"] == "event_time"


def test_numeric_resource_limits_reject_extremes(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    assert tools.memory("find", {"query": "x", "limit": 101})["ok"] is False
    assert tools.memory_review("expired", {"query": "x", "offset": 10_001})["ok"] is False
    assert tools.memory_repair("rebuild_claims", {"memory_ids": [1, -2]})["ok"] is False


def test_status_unknown_field_is_warned_and_removed(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory("status", {"unused": "value"})
    assert result["ok"] is True
    assert "unknown field ignored: unused" in result["warnings"]
