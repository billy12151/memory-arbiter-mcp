"""Unified recall size metering (v0.15.6): find/read/expired/history share
one global config key ``include_size`` and one size-block shape
(``returned_chars``/``returned_count``/``tokens_estimate``). ``display_hint``
stays find-only; the old find-only per-call ``include_size`` parameter is
accepted but ignored with a warning.
"""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.config_registry import CONFIG_DESCRIPTORS, grouped_descriptors
from memory_arbiter.db import MemoryDB
from memory_arbiter.tokens import estimate_tokens, meter_payloads
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path, *, include_size: bool = True) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "m.sqlite3",
        backup_jsonl=tmp_path / "b.jsonl",
        include_size=include_size,
    )
    return MemoryTools(settings, MemoryDB(settings))


def _write_memory(tools: MemoryTools, content: str, *, subject: str = "s") -> int:
    return tools.memory_write(content=content, subject=subject, tags=[])["data"]["id"]


# ── helper ────────────────────────────────────────────────────────────────


def test_meter_payloads_matches_manual_dumps() -> None:
    items = [{"a": "x"}, {"b": "中文内容", "n": 3}]
    import json

    dumps = [
        json.dumps(p, ensure_ascii=False, sort_keys=True, default=str)
        for p in items
    ]
    expected = {
        "returned_chars": sum(len(p) for p in dumps),
        "returned_count": 2,
        "tokens_estimate": sum(estimate_tokens(p) for p in dumps),
    }
    assert meter_payloads(items) == expected


def test_meter_payloads_empty_list() -> None:
    assert meter_payloads([]) == {
        "returned_chars": 0,
        "returned_count": 0,
        "tokens_estimate": 0,
    }


# ── read (precise recall) ─────────────────────────────────────────────────


def test_read_full_carries_size(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = _write_memory(tools, "全文内容足够长用于计量" * 20)
    result = tools.memory_get(memory_id=memory_id)
    assert result["ok"] is True
    size = result["data"]["size"]
    # Meters the record as actually returned, with the shared estimator.
    assert size["returned_count"] == 1
    assert {k: v for k, v in size.items() if k != "display_hint"} == meter_payloads([result["data"]["memory"]])
    assert size["tokens_estimate"] > 0
    # The hint is an instruction carrying the number (v0.15.6 follow-up:
    # agents should surface recall cost when citing a record).
    hint = size["display_hint"]
    assert "report this recall cost" in hint
    assert str(size["tokens_estimate"]) in hint
    assert "1 record" in hint


def test_read_span_meters_window_not_full_text(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    body = "第一段足够长的内容不会被合并掉。" * 8
    memory_id = _write_memory(tools, body)
    full = tools.memory_get(memory_id=memory_id)
    window = tools.memory_get(memory_id=memory_id, span={"start": 0, "end": 30})
    assert window["ok"] is True
    window_size = window["data"]["size"]
    assert {k: v for k, v in window_size.items() if k != "display_hint"} == meter_payloads([window["data"]["memory"]])
    assert window_size["tokens_estimate"] < full["data"]["size"]["tokens_estimate"]
    assert "span" in window_size["display_hint"]
    assert "chars" in window_size["display_hint"]


def test_read_error_paths_have_no_size(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = _write_memory(tools, "short")
    missing = tools.memory_get(memory_id=memory_id + 999)
    assert missing["ok"] is False
    assert "size" not in missing["data"]
    past_end = tools.memory_get(memory_id=memory_id, span={"start": 100, "end": 200})
    assert past_end["ok"] is False
    assert "size" not in past_end["data"]


# ── expired audit recall ──────────────────────────────────────────────────


def test_expired_carries_size(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = _write_memory(tools, "archived decision content with details")
    tools.memory_supersede(memory_id=memory_id, reason="retired", authorized=True)
    result = tools.memory_search_expired(query="archived decision", limit=10)
    assert result["ok"] is True
    assert result["data"]["count"] >= 1
    size = result["data"]["size"]
    assert size["returned_count"] == result["data"]["count"]
    assert {k: v for k, v in size.items() if k != "display_hint"} == meter_payloads(result["data"]["results"])
    assert "report this recall cost" in size["display_hint"]
    assert str(size["tokens_estimate"]) in size["display_hint"]


# ── history (version chain) ───────────────────────────────────────────────


def test_history_carries_size(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = _write_memory(tools, "rev one content")
    tools.memory_edit(memory_id=memory_id, new_content="rev two content", reason="second")
    result = tools.memory_history(memory_id=memory_id)
    assert result["ok"] is True
    assert result["data"]["count"] == 1
    size = result["data"]["size"]
    assert size["returned_count"] == 1
    assert {k: v for k, v in size.items() if k != "display_hint"} == meter_payloads(result["data"]["history"])
    assert "report this recall cost" in size["display_hint"]
    assert "version snapshot" in size["display_hint"]


# ── one global switch ─────────────────────────────────────────────────────


def test_global_off_hides_size_on_all_surfaces(tmp_path: Path) -> None:
    tools = make_tools(tmp_path, include_size=False)
    # One record that stays active (for find/read) and one that goes through
    # edit -> supersede (for history/expired).
    active_id = _write_memory(tools, "active probe stays for find")
    memory_id = _write_memory(tools, "metering off probe content")
    tools.memory_edit(memory_id=memory_id, new_content="metering off rev two", reason="edit")
    tools.memory_supersede(memory_id=memory_id, reason="retired", authorized=True)

    found = tools.memory_search(query="active probe", limit=10)
    assert found["ok"] is True
    assert found["data"]["results"]
    assert "size" not in found["data"]

    read = tools.memory_get(memory_id=active_id)
    assert "size" not in read["data"]

    expired = tools.memory_search_expired(query="metering", limit=10)
    assert "size" not in expired["data"]

    history = tools.memory_history(memory_id=memory_id)
    assert "size" not in history["data"]


def test_find_per_call_include_size_warns_and_is_ignored(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)  # config default: on
    _write_memory(tools, "deprecated per-call flag probe")
    for flag in (True, False):
        result = tools.memory_search(query="deprecated", limit=10, include_size=flag)
        assert result["ok"] is True
        assert any("global config key" in w for w in result["warnings"]), flag
        # Global config wins: block stays present (on) regardless of the flag.
        assert "size" in result["data"], flag


def test_config_file_parses_include_size(tmp_path: Path, monkeypatch) -> None:
    import json

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"include_size": False}), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg))
    assert Settings.from_env().include_size is False

    cfg.write_text(json.dumps({"include_size": True}), encoding="utf-8")
    assert Settings.from_env().include_size is True


def test_settings_default_include_size_on(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MEMORY_ARBITER_CONFIG", raising=False)
    cfg_absent = tmp_path / "absent.json"
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_absent))
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    assert settings.include_size is True


def test_config_registry_lists_include_size() -> None:
    descriptor = next(d for d in CONFIG_DESCRIPTORS if d["path"] == "include_size")
    assert descriptor["default"] is True
    groups = {g["key"]: g["items"] for g in grouped_descriptors()}
    assert descriptor in groups["reporting"]


def test_empty_pages_silence_the_display_hint(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # A query that matches nothing: expired/history with no rows carry the
    # numbers but no instruction (nothing was recalled, no cost to report).
    expired = tools.memory_search_expired(query="definitely-no-such-token-xyz", limit=5)
    assert expired["data"]["count"] == 0
    assert expired["data"]["size"]["display_hint"] is None

    memory_id = _write_memory(tools, "no edits yet")
    history = tools.memory_history(memory_id=memory_id)
    assert history["data"]["count"] == 0
    assert history["data"]["size"]["display_hint"] is None
