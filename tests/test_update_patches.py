"""update patches mode (v0.15.12 C2): sequential atomic multi-point edits.

One call = one version bump + one history row + one evidence republish + one
post-commit check; any miss rejects the whole batch with zero side effects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "p.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
    )
    return MemoryTools(settings, MemoryDB(settings))


def _write(tools: MemoryTools, content: str) -> dict[str, Any]:
    return tools.memory_write(content=content, subject="svc", tags=[])["data"]


def _history_count(tools: MemoryTools, memory_id: int) -> int:
    return len(tools.db.list_history(memory_id))


def test_two_patches_one_call_one_version(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    record = _write(tools, "db is MySQL 5.7 in region us-east-1, budget 100.")
    memory_id = int(record["id"])
    before_history = _history_count(tools, memory_id)

    result = tools.memory_edit(
        memory_id,
        patches=[
            {"old_text": "MySQL 5.7", "new_text": "MySQL 8.0"},
            {"old_text": "us-east-1", "new_text": "us-west-2"},
        ],
        reason="two spotted corrections",
    )
    assert result["ok"] is True, result
    data = result["data"]
    assert data["edited"] is True
    assert data["new_version"] == 2
    assert _history_count(tools, memory_id) == before_history + 1
    updated = tools.db.get_memory(memory_id)
    assert updated["content"] == "db is MySQL 8.0 in region us-west-2, budget 100."


def test_second_patch_miss_rejects_atomically(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    record = _write(tools, "alpha value is 1. beta value is 2.")
    memory_id = int(record["id"])
    before = tools.db.get_memory(memory_id)
    before_history = _history_count(tools, memory_id)

    result = tools.memory_edit(
        memory_id,
        patches=[
            {"old_text": "alpha value is 1", "new_text": "alpha value is 10"},
            {"old_text": "gamma value is 3", "new_text": "gamma value is 30"},
        ],
    )
    assert result["ok"] is False
    assert result["data"]["outcome"] == "stale_edit"
    assert result["data"]["reason"] == "old_text_not_found"
    assert result["data"]["patch_index"] == 1
    # Atomicity: nothing moved — content, version, history, and the
    # first patch's would-be effect all absent.
    after = tools.db.get_memory(memory_id)
    assert after["content"] == before["content"]
    assert after["version"] == before["version"] == 1
    assert _history_count(tools, memory_id) == before_history


def test_chained_patches_match_on_earlier_result(tmp_path: Path) -> None:
    """patch2 may hit text that patch1's new_text just produced."""
    tools = make_tools(tmp_path)
    record = _write(tools, "release v1 is current.")
    memory_id = int(record["id"])

    result = tools.memory_edit(
        memory_id,
        patches=[
            {"old_text": "v1", "new_text": "v2"},
            {"old_text": "release v2 is current", "new_text": "release v2 is staged"},
        ],
    )
    assert result["ok"] is True, result
    assert tools.db.get_memory(memory_id)["content"] == "release v2 is staged."


def test_first_occurrence_only_replacement(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    record = _write(tools, "flag A on. flag A duplicated.")
    memory_id = int(record["id"])

    result = tools.memory_edit(memory_id, patches=[{"old_text": "flag A", "new_text": "flag B"}])
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["content"] == "flag B on. flag A duplicated."


def test_patch_limits_and_mutual_exclusion(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    record = _write(tools, "x")
    memory_id = int(record["id"])

    # 9 patches rejected at the validation boundary.
    nine = [{"old_text": "x", "new_text": "y"}] * 9
    result = tools.memory("update", {"memory_id": memory_id, "patches": nine})
    assert result["ok"] is False
    assert result["data"]["error"] == "invalid_input"
    assert "patches" in result["data"]["field"]

    # empty list rejected
    result = tools.memory("update", {"memory_id": memory_id, "patches": []})
    assert result["ok"] is False

    # extra key rejected (strict shape)
    result = tools.memory(
        "update", {"memory_id": memory_id, "patches": [{"old_text": "x", "new_text": "y", "note": "hi"}]},
    )
    assert result["ok"] is False
    assert "old_text" in result["data"].get("reason", "") or "keys" in result["data"].get("reason", "")

    # mutual exclusion with new_content and with old_text/new_text
    result = tools.memory(
        "update",
        {"memory_id": memory_id, "patches": [{"old_text": "x", "new_text": "y"}], "new_content": "z"},
    )
    assert result["ok"] is False
    assert "exactly one content mode" in result["data"]["error"]
    result = tools.memory(
        "update",
        {"memory_id": memory_id, "patches": [{"old_text": "x", "new_text": "y"}], "old_text": "x", "new_text": "z"},
    )
    assert result["ok"] is False
    assert "exactly one content mode" in result["data"]["error"]


def test_empty_new_text_deletes_fragment(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    record = _write(tools, "keep this. drop this sentence.")
    memory_id = int(record["id"])

    result = tools.memory_edit(memory_id, patches=[{"old_text": " drop this sentence.", "new_text": ""}])
    assert result["ok"] is True
    assert tools.db.get_memory(memory_id)["content"] == "keep this."


def test_patches_with_new_subject_and_add_tags(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    record = _write(tools, "price is 10 dollars.")
    memory_id = int(record["id"])

    result = tools.memory_edit(
        memory_id,
        patches=[{"old_text": "10 dollars", "new_text": "12 dollars"}],
        new_subject="price",
        add_tags=["finance"],
    )
    assert result["ok"] is True, result
    updated = tools.db.get_memory(memory_id)
    assert updated["content"] == "price is 12 dollars."
    assert updated["subject"] == "price"
    assert "finance" in updated["tags"]
    assert updated["version"] == 2


def test_patches_respect_cas_and_locked_gate(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    record = _write(tools, "stable text.")
    memory_id = int(record["id"])

    # CAS: expected_version checked against the pre-edit row.
    result = tools.memory_edit(
        memory_id,
        patches=[{"old_text": "stable", "new_text": "changed"}],
        expected_version=99,
    )
    assert result["ok"] is False
    assert result["data"]["outcome"] == "stale_edit"
    assert result["data"]["reason"] == "version_mismatch"

    # Locked memory still requires authorized.
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET protection_level='locked' WHERE id=?", (memory_id,))
    result = tools.memory_edit(memory_id, patches=[{"old_text": "stable", "new_text": "changed"}])
    assert result["ok"] is False
    assert "authorized" in result["data"]["error"].lower()
    result = tools.memory_edit(
        memory_id, patches=[{"old_text": "stable", "new_text": "changed"}], authorized=True,
    )
    assert result["ok"] is True


def test_db_layer_rejects_malformed_patches_defensively(tmp_path: Path) -> None:
    """A pipeline caller bypassing validation must not crash the edit loop."""
    tools = make_tools(tmp_path)
    record = _write(tools, "content")
    memory_id = int(record["id"])

    result = tools.db.edit_memory_intent(memory_id, patches=[{"old_text": "content"}])  # type: ignore[list-item]
    assert result["outcome"] == "invalid"

    result = tools.db.edit_memory_intent(memory_id, patches=[])  # type: ignore[list-item]
    assert result["outcome"] == "invalid"


def test_e2e_post_commit_fires_once_for_patches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """make_tools-level e2e: one patches call = one post-commit semantic check."""
    tools = make_tools(tmp_path)
    record = _write(tools, "value is 1.")
    memory_id = int(record["id"])

    fired = {"post_commit": 0, "refresh": 0}
    original_post = tools._operations._post_commit
    original_refresh = tools._tools_ref if False else None  # noqa: F841 — clarity only

    def counting_post(*args: Any, **kwargs: Any) -> Any:
        fired["post_commit"] += 1
        return original_post(*args, **kwargs)

    monkeypatch.setattr(tools._operations, "_post_commit", counting_post)

    result = tools.memory_edit(
        memory_id,
        patches=[
            {"old_text": "value is 1", "new_text": "value is 2"},
            {"old_text": ".", "new_text": "!"},
        ],
    )
    assert result["ok"] is True
    assert fired["post_commit"] == 1


# ── adversarial-review round 2 (R1/R2): silent-intent-eating defenses ──────────


def test_tags_only_cannot_be_combined_with_patches(tmp_path: Path) -> None:
    """R1: tags_only + content edits must be rejected loudly, not half-run.

    The tags-only fast path never touches content; before this guard the
    patches half of the call was silently dropped (content unchanged, tags
    applied) and the caller's edit intent vanished.
    """
    tools = make_tools(tmp_path)
    record = _write(tools, "hello world")
    memory_id = int(record["id"])

    result = tools.memory_edit(memory_id, tags_only=True, patches=[{"old_text": "hello", "new_text": "hi"}], add_tags=["t"])
    assert result["ok"] is False
    assert "tags_only" in result["data"]["error"]
    updated = tools.db.get_memory(memory_id)
    assert updated["content"] == "hello world"
    assert updated["tags"] == [], "the whole call must be rejected, not half-applied"

    # …and the same for the pre-existing content modes the guard now covers.
    result = tools.memory_edit(memory_id, tags_only=True, new_content="other")
    assert result["ok"] is False


def test_patches_cannot_wipe_content_empty(tmp_path: Path) -> None:
    """R2: a deletion-only batch must hit the same wipe guard as new_content."""
    tools = make_tools(tmp_path)
    record = _write(tools, "ab")
    memory_id = int(record["id"])

    result = tools.memory_edit(memory_id, patches=[{"old_text": "ab", "new_text": ""}])
    assert result["ok"] is False
    assert result["data"]["outcome"] == "invalid"
    assert "wipe" in result["data"]["error"]
    assert tools.db.get_memory(memory_id)["content"] == "ab"
