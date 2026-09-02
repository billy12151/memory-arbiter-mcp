"""v0.15.2 PR3 find enhancements: size metering block, tokens estimator, and
the admitted-scope unresolved_conflict_count.
"""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tokens import TOKEN_ESTIMATE_BASIS, estimate_tokens
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    return MemoryTools(settings, MemoryDB(settings))


def test_estimate_tokens_bucket_logic() -> None:
    assert estimate_tokens("") == 0
    # Pure CJK: 0.77 per char.
    assert estimate_tokens("配置只认配置文件" * 10) == round(0.77 * 80)
    # CJK punctuation bucket.
    assert estimate_tokens("，。：；" * 5) == round(0.85 * 20)
    # Digits: 1.15 per char.
    assert estimate_tokens("0123456789") == round(1.15 * 10)
    # Newlines/spaces.
    assert estimate_tokens("\n" * 10) == 10
    assert estimate_tokens(" " * 100) == 15
    # English words: 1.15 per word + spaces.
    assert estimate_tokens("alpha beta") == round(1.15 * 2 + 0.15)
    # Markdown chars at 0.9, ASCII punctuation at 0.6.
    assert estimate_tokens("##**") == round(0.9 * 4)
    assert estimate_tokens("....") == round(0.6 * 4)
    assert TOKEN_ESTIMATE_BASIS.startswith("heuristic_v1")


def test_find_size_block_default_on_and_opt_out(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="alpha deployment note with details", subject="s", tags=[])
    result = tools.memory_search(query="deployment", limit=10)
    assert result["ok"] is True
    size = result["data"]["size"]
    assert size["returned_count"] == 1
    assert size["returned_chars"] == len("alpha deployment note with details")
    assert size["tokens_estimate"] == estimate_tokens("alpha deployment note with details")
    assert size["estimate_basis"] == TOKEN_ESTIMATE_BASIS
    assert size["matched_beyond_limit_count"] == 0
    assert result["data"]["unresolved_conflict_count"] == 0

    off = tools.memory_search(query="deployment", limit=10, include_size=False)
    assert "size" not in off["data"]


def test_find_size_counts_beyond_limit(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for index in range(3):
        tools.memory_write(content=f"deployment note number {index}", subject=f"s{index}", tags=[])
    result = tools.memory_search(query="deployment", limit=1)
    assert result["ok"] is True
    size = result["data"]["size"]
    assert size["returned_count"] == 1
    assert size["matched_beyond_limit_count"] >= 1


def test_find_unresolved_conflict_count_admitted_scope(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="database is mysql", subject="s", tags=[])["data"]["id"]
    right = tools.memory_write(content="database is sqlite", subject="s2", tags=[])["data"]["id"]

    def member(memory_id: int, value: str):
        quote = f"database is {value}"
        return {
            "memory_id": memory_id, "version": 1, "attribute_raw": "database",
            "value_raw": value, "normalized_attribute": "database",
            "normalized_value": value.casefold(), "evidence_quote": quote,
            "evidence_span": [0, len(quote)], "content_hash": (str(memory_id) * 64)[:64],
            "direction": "a_to_b", "prompt_version": "p1", "detector_version": "d1",
        }

    created = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "p", "attribute": "db", "scope": "g"},
        "members": [member(left, "mysql"), member(right, "sqlite")],
        "value_groups": [
            {"normalized_value": "mysql", "display_value": "mysql", "members": [f"{left}@1"]},
            {"normalized_value": "sqlite", "display_value": "sqlite", "members": [f"{right}@1"]},
        ],
        "status": "open", "detector_version": "d1", "prompt_version": "p1",
        "source": "scan", "reason": "diff", "authorized": True,
    })
    assert created["ok"] is True, created["data"]
    result = tools.memory_search(query="database", limit=10)
    assert result["data"]["unresolved_conflict_count"] == 1
    # The conflict_group signal (with next_executable_call) still attaches.
    signals = [r.get("conflict_signal") for r in result["data"]["results"] if r.get("conflict_signal")]
    assert signals, "conflict signal must still attach"
    assert all(sig.get("next_executable_call") for sig in signals)
