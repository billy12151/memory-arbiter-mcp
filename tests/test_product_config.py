from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from memory_arbiter.arbitration import compare_memories
from memory_arbiter.config import Settings, parse_bool
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import SourceType
from memory_arbiter.tools import MemoryTools


class _MockManagedEmbedder:
    """Minimal mock for ManagedEmbedder — wraps a plain encode function.

    Mirrors the production Never-raises contract: if _encode raises, the
    exception is caught, last_encode_error is set, and an empty EmbedResult
    is returned so callers must check er.embedding.
    """

    def __init__(self, encode_fn):
        self._encode = encode_fn
        self.embedding_space_id = "mock_space_id"
        self.last_encode_error = None

    def embed_text(self, prefix="", body="", max_body_chars=None):
        # Mirror the production separator so the prefix's trailing token and the
        # body's leading token are not merged (e.g. "alpha"+"alpha x" → "alphaalpha").
        sep = "\n" if prefix and body else ""
        text = (prefix + sep + body).strip()
        try:
            emb = self._encode(text)
        except Exception as exc:
            self.last_encode_error = str(exc)
            return EmbedResult(embedding=[], truncated=True, original_tokens=0, used_tokens=0)
        return EmbedResult(embedding=emb, truncated=False, original_tokens=0, used_tokens=0)


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


def make_vec_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    settings = Settings(
        db_path=tmp_path / "memory-vec.sqlite3",
        backup_jsonl=tmp_path / "backup-vec.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=2,
        split_threshold=1,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def clear_config_env(monkeypatch) -> None:
    for key in (
        "MEMORY_ARBITER_CONFIG",
        "MEMORY_ARBITER_DB_PATH",
        "MEMORY_ARBITER_BACKUP_JSONL",
        "MEMORY_ARBITER_POLICY",
        "MEMORY_ARBITER_CLIENT",
        "MEMORY_ARBITER_AGENT_ID",
        "MEMORY_ARBITER_WORKSPACE",
        "MEMORY_ARBITER_ENABLE_SQLITE_VEC",
        "MEMORY_ARBITER_VEC_DIM",
        "MEMORY_ARBITER_RECALL_POOL_CAP",
        "MEMORY_ARBITER_CONTENT_LIKE_CAP",
        "MEMORY_ARBITER_EMBEDDING_PROVIDER",
        "MEMORY_ARBITER_EMBEDDING_MODEL_PATH",
        "MEMORY_ARBITER_EMBEDDING_AUTO_QUERY",
        "MEMORY_ARBITER_EMBEDDING_AUTO_WRITE",
        "MEMORY_ARBITER_GGUF",
        "MEMORY_ARBITER_TOOL_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)



def test_server_memory_edit_preserves_tags_when_new_tags_omitted(tmp_path: Path, monkeypatch) -> None:
    """Regression: the MCP wrapper must pass new_tags=None through.

    Passing [] erases existing tags on a content-only edit, even though the
    MemoryTools layer correctly preserves tags when new_tags is omitted.
    """

    class FakeFastMCP:
        def __init__(self, _name: str) -> None:
            self.tools = {}

        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = func
                return func

            return decorator

    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = FakeFastMCP
    fake_server = types.ModuleType("mcp.server")
    fake_mcp = types.ModuleType("mcp")
    fake_server.fastmcp = fake_fastmcp
    fake_mcp.server = fake_server
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)
    update_cfg = tmp_path / "update-off.json"
    update_cfg.write_text(json.dumps({"update_check": {"enabled": False}}), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(update_cfg))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "server.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "server.backup.jsonl"))
    monkeypatch.setenv("MEMORY_ARBITER_WORKSPACE", "repo-a")
    monkeypatch.setenv("MEMORY_ARBITER_AGENT_ID", "agent-a")

    from memory_arbiter.server import build_server

    app = build_server()
    assert set(app.tools) == {"memory", "memory_review", "memory_govern", "memory_repair"}
    written = app.tools["memory"](
        action="remember",
        data={
            "content": "draft content",
            "subject": "server-wrapper",
            "tags": ["keep-me"],
            "source_type": "agent_generated",
            "event_time": "2026-01-01T00:00:00Z",
        },
    )
    memory_id = written["data"]["id"]

    edited = app.tools["memory"](
        action="update",
        data={"memory_id": memory_id, "new_content": "edited content"},
    )

    assert edited["ok"] is True
    assert edited["data"]["record"]["content"] == "edited content"
    assert edited["data"]["record"]["tags"] == ["keep-me"]


def test_server_legacy_full_profile_exposes_low_level_tools(tmp_path: Path, monkeypatch) -> None:
    class FakeFastMCP:
        def __init__(self, _name: str) -> None:
            self.tools = {}

        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = func
                return func

            return decorator

    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = FakeFastMCP
    fake_server = types.ModuleType("mcp.server")
    fake_mcp = types.ModuleType("mcp")
    fake_server.fastmcp = fake_fastmcp
    fake_mcp.server = fake_server
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)
    monkeypatch.setenv("MEMORY_ARBITER_TOOL_PROFILE", "legacy_full")
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "server.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "server.backup.jsonl"))

    from memory_arbiter.server import build_server

    app = build_server()

    assert "memory" in app.tools
    assert "memory_write" in app.tools
    assert "memory_supersede" in app.tools



def test_product_memory_review_and_govern_wrappers(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory(
        action="remember",
        data={"content": "old whole fact", "subject": "old", "tags": ["govern"]},
    )
    replacement = tools.memory(
        action="remember",
        data={"content": "new whole fact", "subject": "new", "tags": ["govern"]},
    )
    old_id = written["data"]["id"]
    new_id = replacement["data"]["id"]

    found = tools.memory(action="find", data={"query": "whole fact"})
    assert found["ok"] is True
    assert found["data"]["count"] >= 1

    read = tools.memory(action="read", data={"memory_id": old_id})
    assert read["data"]["memory"]["id"] == old_id

    review = tools.memory_review(view="history", data={"memory_id": old_id})
    assert review["ok"] is True

    denied = tools.memory_govern(
        action="retire",
        data={
            "memory_id": old_id,
            "superseded_by": new_id,
            "reason": "user explicitly retired this whole memory",
        },
    )
    assert denied["ok"] is False
    assert denied["data"]["action_required"] == "ask_user_for_authorization"
    assert denied["data"]["governance_action"] == "retire"

    retired = tools.memory_govern(
        action="retire",
        data={
            "memory_id": old_id,
            "superseded_by": new_id,
            "reason": "user explicitly retired this whole memory",
            "authorized": True,
        },
    )
    assert retired["data"]["superseded"] is True


def test_product_wrappers_validate_aliases_and_bad_inputs(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory(action="remember", data={"content": "old", "subject": "old"})
    memory_id = written["data"]["id"]
    tools.memory(action="update", data={"id": memory_id, "new_content": "new"})

    history_by_alias = tools.memory_review(view="history", data={"id": memory_id})
    assert history_by_alias["ok"] is True
    assert history_by_alias["data"]["count"] == 1

    bad_history = tools.memory_review(view="history", data={"id": "abc"})
    assert bad_history["ok"] is False
    assert "memory_id" in bad_history["data"]["error"]

    for label, result in {
        "retire": tools.memory_govern(action="retire", data={"id": "abc", "reason": "x", "authorized": True}),
        "confirm": tools.memory_govern(action="confirm", data={"id": "abc", "authorized": True}),
        "resolve": tools.memory_govern(action="resolve_conflict", data={"id": "abc", "authorized": True}),
        "split": tools.memory_repair(task="split", data={"id": "abc"}),
        "activate": tools.memory_repair(task="activate_pending", data={"id": "abc", "authorized": True}),
        "cleanup": tools.memory_repair(task="cleanup_history", data={"id": "abc", "authorized": True}),
    }.items():
        assert result["ok"] is False, label
        assert "integer" in result["data"]["error"]

    judge_missing = tools.memory(action="judge", data={"id": 1})
    assert judge_missing["ok"] is False
    assert "missing" in judge_missing["data"]["error"]
    assert "expected_left_version" in judge_missing["data"]["help"]["missing_fields"]


def test_product_repair_cleanup_history_id_alias_is_not_full_cleanup(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = tools.memory(action="remember", data={"content": "a", "subject": "a"})["data"]["id"]
    second = tools.memory(action="remember", data={"content": "b", "subject": "b"})["data"]["id"]
    tools.memory(action="update", data={"id": first, "new_content": "a2"})
    tools.memory(action="update", data={"id": second, "new_content": "b2"})
    assert tools.memory_review(view="history", data={"id": first})["data"]["count"] == 1
    assert tools.memory_review(view="history", data={"id": second})["data"]["count"] == 1

    cleaned = tools.memory_repair(
        task="cleanup_history",
        data={"id": first, "authorized": True},
    )

    assert cleaned["ok"] is True
    assert cleaned["data"]["scope"] == "memory"
    assert cleaned["data"]["memory_id"] == first
    assert tools.memory_review(view="history", data={"id": first})["data"]["count"] == 0
    assert tools.memory_review(view="history", data={"id": second})["data"]["count"] == 1

    tools = make_tools(tmp_path)
    help_result = tools.memory_repair(task="help", data={"task": "rebuild_claims"})
    assert help_result["ok"] is True
    assert "rebuild_claims" in help_result["data"]["tasks"]

    dry = tools.memory_repair(task="rebuild_claims", data={"dry_run": True})
    assert dry["ok"] is True
    assert dry["data"]["dry_run"] is True


def test_string_false_authorized_fails_closed_across_product_surfaces(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = tools.memory(action="remember", data={
        "content": "protected fact", "subject": "protected",
        "source_type": "user_confirmed", "protection_level": "locked",
    })["data"]["id"]

    for false_value in ("false", "0", "no", "off", "maybe", "", None):
        edited = tools.memory(action="update", data={
            "memory_id": memory_id,
            "new_content": "tampered",
            "authorized": false_value,
        })
        assert edited["ok"] is False, false_value

        retired = tools.memory_govern(action="retire", data={
            "memory_id": memory_id,
            "reason": "not actually authorized",
            "authorized": false_value,
        })
        assert retired["ok"] is False, false_value

    record = tools.memory(action="read", data={"memory_id": memory_id})["data"]["memory"]
    assert record["content"] == "protected fact"
    assert record["status"] == "active"


def test_string_true_authorized_remains_compatible(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = tools.memory(action="remember", data={
        "content": "protected fact", "subject": "protected",
        "source_type": "user_confirmed", "protection_level": "locked",
    })["data"]["id"]

    edited = tools.memory(action="update", data={
        "memory_id": memory_id,
        "new_content": "authorized correction",
        "authorized": "true",
    })

    assert edited["ok"] is True
    assert edited["data"]["record"]["content"] == "authorized correction"


def test_all_governance_actions_require_explicit_user_authorization(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    valid_payloads = {
        "retire": {"memory_id": 1, "reason": "retire whole memory"},
        "resolve_conflict": {"conflict_id": 1},
        "confirm": {"memory_id": 1},
        "correct_judgment": {
            "conflict_id": 1, "verdict": "evolution", "recommended_use": "merge",
            "suggested_winner": None, "reason": "correction", "expected_judgment_id": 1,
            "expected_left_version": 1, "expected_right_version": 1,
            "expected_left_claim_revision": 1, "expected_right_claim_revision": 1,
            "authorized": False,
        },
        "accept_workspace_alias": {"alias": "alias", "canonical": "canonical"},
        "reject_workspace_alias": {"alias": "alias", "canonical": "canonical"},
        "rename_workspace_canonical": {"old": "old", "new": "new"},
        "migrate_workspace": {"from": "old", "to": "new"},
        "confirm_pending_workspace": {"memory_id": 1, "canonical": "canonical"},
    }

    for action, payload in valid_payloads.items():
        for false_value in (None, False, "false", "0", "no", "maybe"):
            call_payload = dict(payload)
            if false_value is not None:
                call_payload["authorized"] = false_value
            result = tools.memory_govern(action=action, data=call_payload)
            assert result["ok"] is False, (action, false_value)
            assert result["data"]["action_required"] == "ask_user_for_authorization"
            assert result["data"]["governance_action"] == action
            assert result["data"]["impact"]



def test_product_forwards_return_clean_error_when_id_missing(tmp_path: Path) -> None:
    """v0.11.1: product forwards must not raise TypeError on a missing id.

    Before the guard, ``memory(action='read', data={})`` forwarded an empty
    payload to ``memory_get(memory_id: int)`` and raised. Every forward whose
    target has a required positional id should return ``ok=False`` with help,
    matching the rest of the bad-input contract.
    """
    tools = make_tools(tmp_path)

    def assert_clean_error(label: str, result: dict) -> None:
        assert result["ok"] is False, f"{label} should be ok=False, got {result}"
        assert "help" in result["data"], f"{label} must attach help"
        assert "error" in result["data"], f"{label} must attach an error"

    assert_clean_error("memory.read", tools.memory(action="read", data={}))
    assert_clean_error("memory.update", tools.memory(action="update", data={"new_content": "x"}))
    assert_clean_error("govern.retire", tools.memory_govern(action="retire", data={}))
    assert_clean_error("govern.confirm", tools.memory_govern(action="confirm", data={}))
    assert_clean_error("govern.correct_judgment", tools.memory_govern(action="correct_judgment", data={}))
    assert_clean_error("govern.resolve_conflict", tools.memory_govern(action="resolve_conflict", data={}))
    assert_clean_error("repair.split", tools.memory_repair(task="split", data={}))
    assert_clean_error("repair.set_entity", tools.memory_repair(task="set_entity", data={}))
    assert_clean_error("repair.activate_pending", tools.memory_repair(task="activate_pending", data={}))

    # cleanup_history legitimately allows a missing memory_id (full cleanup) —
    # it must still reach its own authorized gate, not crash.
    cleanup = tools.memory_repair(task="cleanup_history", data={})
    assert cleanup["ok"] is False
    assert "authorized" in cleanup["data"]["error"]


def test_product_judge_help_exposes_enum_constraints(tmp_path: Path) -> None:
    """v0.11.1: judge/correct_judgment help must document enums + cross-field rules.

    Without these an agent iterates against invalid_* outcomes because the
    allowed verdict/recommended_use/resolution_kind values are otherwise
    undiscoverable from the help surface.
    """
    tools = make_tools(tmp_path)

    mem_help = tools.memory(action="help")["data"]
    assert "judge_constraints" in mem_help
    jc = mem_help["judge_constraints"]
    # Single source of truth: these mirror ConflictJudgmentStore + submit_conflict_judgment.
    assert jc["verdict"] == ["contradiction", "evolution", "compatible", "uncertain"]
    assert set(jc["recommended_use"]) == {"left", "right", "contextual", "merge", "ask_user", "none"}
    assert "partial_update" in jc["resolution_kind"]
    assert "field" in jc["conflict_scope"]
    assert any("partial_update|merge" in rule for rule in jc["rules"])

    # memory_govern help must also carry the same constraint block (correct_judgment).
    gov_help = tools.memory_govern(action="help")["data"]
    assert "judge_constraints" in gov_help

    # The missing-fields error path must also attach the constraints.
    missing = tools.memory(action="judge", data={"id": 1})
    assert missing["ok"] is False
    assert "judge_constraints" in missing["data"]["help"]
    assert "expected_left_version" in missing["data"]["help"]["missing_fields"]

def test_product_help_exposes_agent_onboarding_topic(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    help_doc = tools.memory(action="help", data={"topic": "agent_onboarding"})["data"]
    assert help_doc["topic"] == "agent_onboarding"
    assert help_doc["notice"] == "agent-onboarding:v1"
    assert help_doc["guide_file"] == "memory_arbiter/AGENT_ONBOARDING.md"
    assert "Compact rule to save" in help_doc["content"]
    assert "memory(action=\"find\")" in help_doc["content"]



def test_product_forwards_handle_bad_secondary_int_args(tmp_path: Path) -> None:
    """v0.11.1: secondary int args must not crash the product surface.

    Before the ``_forward`` wrapper, ``int()`` inside the low-level method
    raised straight through on loosely-typed JSON (e.g. ``"limit": "abc"``).
    These forwards now return ``ok=False`` with help. Numeric strings
    (``"5"``), which MCP clients commonly send, still coerce and succeed.
    """
    tools = make_tools(tmp_path)
    new_id = tools.memory(action="remember", data={"content": "new", "subject": "s"})["data"]["id"]

    def assert_clean(label: str, result: dict) -> None:
        assert result["ok"] is False, f"{label} unexpectedly succeeded: {result}"
        assert "help" in result["data"], f"{label} must attach help"

    # retire: non-int superseded_by (primary memory_id coerced, secondary not)
    old = tools.memory(action="remember", data={"content": "old", "subject": "s"})["data"]["id"]
    assert_clean(
        "govern.retire superseded_by=xyz",
        tools.memory_govern(action="retire", data={
            "memory_id": old, "superseded_by": "xyz", "reason": "r", "authorized": True,
        }),
    )
    # numeric-string superseded_by still works
    ok = tools.memory_govern(action="retire", data={
        "memory_id": old, "superseded_by": str(new_id), "reason": "r", "authorized": True,
    })
    assert ok["ok"] is True and ok["data"]["superseded"] is True

    assert_clean("memory.find limit=abc", tools.memory(action="find", data={"query": "x", "limit": "abc"}))
    # numeric-string limit still coerces and succeeds
    found = tools.memory(action="find", data={"query": "x", "limit": "5"})
    assert found["ok"] is True

    assert_clean("review.conflicts limit=abc", tools.memory_review(view="conflicts", data={"limit": "abc"}))
    assert_clean("review.expired limit=abc", tools.memory_review(view="expired", data={"query": "x", "limit": "abc"}))
    assert_clean("repair.cleanup_history older_than_days=abc",
                 tools.memory_repair(task="cleanup_history", data={"older_than_days": "abc", "authorized": True}))


def test_product_judge_and_correct_judgment_field_ordering(tmp_path: Path) -> None:
    """v0.11.1: a non-integer conflict_id must not mask other missing required fields.

    correct_judgment now runs the required-fields check BEFORE coercing
    conflict_id, so an agent learns about every missing field at once rather
    than one per round-trip. judge coerces conflict_id explicitly so a
    non-numeric id gets a clear error instead of an obscure invalid_input.
    """
    tools = make_tools(tmp_path)

    # judge with a non-numeric conflict_id alias and nothing else: still reports
    # the *missing* required fields (conflict_id is present, just non-numeric),
    # so the agent fills the rest first.
    r = tools.memory(action="judge", data={"id": "abc"})
    assert r["ok"] is False
    assert "missing_fields" in r["data"]["help"]
    assert "verdict" in r["data"]["help"]["missing_fields"]

    # judge with ALL required fields present but a non-numeric conflict_id:
    # clean integer error, not a deep invalid_input.
    r = tools.memory(action="judge", data={
        "conflict_id": "abc", "expected_left_version": 1, "expected_right_version": 1,
        "expected_left_claim_revision": 0, "expected_right_claim_revision": 0,
        "verdict": "evolution", "recommended_use": "merge", "suggested_winner": None,
        "confidence_hint": "medium", "reason": "x", "affects_current_output": True,
        "usage_context": "answer",
    })
    assert r["ok"] is False
    assert "conflict_id" in r["data"]["error"]

    # correct_judgment: non-int conflict_id + other missing fields -> reports MISSING
    # fields (9 of them), not the integer error.
    r = tools.memory_govern(action="correct_judgment", data={"conflict_id": "abc", "verdict": "x"})
    assert r["ok"] is False
    assert "missing_fields" in r["data"]["help"]
    assert len(r["data"]["help"]["missing_fields"]) >= 5

    # correct_judgment: all required present, conflict_id non-int -> integer error
    r = tools.memory_govern(action="correct_judgment", data={
        "conflict_id": "abc", "verdict": "evolution", "recommended_use": "merge",
        "suggested_winner": None, "reason": "fix", "expected_judgment_id": 1,
        "expected_left_version": 1, "expected_right_version": 1,
        "expected_left_claim_revision": 0, "expected_right_claim_revision": 0,
        "authorized": True,
    })
    assert r["ok"] is False
    assert "conflict_id" in r["data"]["error"]



def test_product_non_dict_data_returns_error(tmp_path: Path) -> None:
    """v0.11.1: non-dict ``data`` must return a clean error, not silently coerce to {}.

    Before the guard, ``_payload_dict`` turned a string/list/int ``data`` into
    ``{}``, so ``memory(action="remember", data="x")`` silently wrote an empty
    record. Now it returns ``ok=False`` with help.
    """
    tools = make_tools(tmp_path)

    def assert_clean(label: str, result: dict) -> None:
        assert result["ok"] is False, f"{label} unexpectedly succeeded: {result}"
        assert "help" in result["data"], f"{label} must attach help"
        assert "JSON object" in result["data"]["error"]

    assert_clean("memory.remember data=str", tools.memory(action="remember", data="notadict"))
    assert_clean("memory.remember data=list", tools.memory(action="remember", data=["x"]))
    assert_clean("memory.find data=int", tools.memory(action="find", data=123))
    assert_clean("govern.retire data=str", tools.memory_govern(action="retire", data="bad"))
    assert_clean("repair.rebuild_claims data=str", tools.memory_repair(task="rebuild_claims", data="bad"))
    assert_clean("review.conflicts data=int", tools.memory_review(view="conflicts", data=123))

    # data=None (explicit) is the default and must still work.
    assert tools.memory(action="status", data=None)["ok"] is True
    assert tools.memory(action="find", data=None)["ok"] is True


def test_memory_write_tags_non_list_does_not_split_chars(tmp_path: Path) -> None:
    """v0.11.1: tags as a string must not be split into characters.

    ``MemoryRecord.from_input`` previously did ``list(payload.get("tags") or [])``,
    so ``tags="todo"`` became ``['t','o','d','o']`` — silent data corruption
    in the tag index. Non-list tags now coerce to ``[]``.
    """
    tools = make_tools(tmp_path)

    # string → [] (not character split)
    r = tools.memory(action="remember", data={"content": "a", "subject": "s", "tags": "todo"})
    assert r["ok"] is True
    assert r["data"]["record"]["tags"] == []

    # int → []
    r = tools.memory(action="remember", data={"content": "b", "subject": "s", "tags": 123})
    assert r["ok"] is True
    assert r["data"]["record"]["tags"] == []

    # None → []
    r = tools.memory(action="remember", data={"content": "c", "subject": "s"})
    assert r["ok"] is True
    assert r["data"]["record"]["tags"] == []

    # list preserved
    r = tools.memory(action="remember", data={"content": "d", "subject": "s", "tags": ["todo", "project"]})
    assert r["ok"] is True
    assert r["data"]["record"]["tags"] == ["todo", "project"]

    # tuple also accepted (JSON arrays deserialize to lists, but be lenient)
    r = tools.memory(action="remember", data={"content": "e", "subject": "s", "tags": ("a", "b")})
    assert r["ok"] is True
    assert r["data"]["record"]["tags"] == ["a", "b"]



def test_tool_profile_env_and_validation(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    monkeypatch.setenv("MEMORY_ARBITER_TOOL_PROFILE", "legacy_full")
    settings = Settings.from_env()
    assert settings.tool_profile == "legacy_full"

    monkeypatch.setenv("MEMORY_ARBITER_TOOL_PROFILE", "invalid")
    settings = Settings.from_env()
    assert settings.tool_profile == "product"
    assert any("tool_profile" in warning for warning in settings.config_warnings)



def test_config_file_overrides_env(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    cfg_path = tmp_path / "config.json"
    cfg_db = tmp_path / "from-config.sqlite3"
    env_db = tmp_path / "from-env.sqlite3"
    cfg_path.write_text(
        json.dumps(
            {
                "db_path": str(cfg_db),
                "backup_jsonl": str(tmp_path / "from-config.jsonl"),
                "client": "from-config",
                "vec": {"enabled": True, "dim": 512},
                "embedding": {
                    "provider": "gguf",
                    "model_path": str(tmp_path / "model.gguf"),
                    "auto_query": False,
                    "auto_write": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(env_db))
    monkeypatch.setenv("MEMORY_ARBITER_VEC_DIM", "999")

    settings = Settings.from_env()

    assert settings.db_path == cfg_db
    assert settings.client == "from-config"
    assert settings.enable_sqlite_vec is True
    assert settings.vec_dim == 512
    assert settings.embedding_provider == "gguf"
    assert settings.embedding_model_path == tmp_path / "model.gguf"
    assert settings.embedding_auto_query is False
    assert settings.update_check_enabled is True


def test_update_check_config_switch(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"update_check": {"enabled": False}}), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert settings.update_check_enabled is False


def test_update_check_bad_config_defaults_enabled(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"update_check": {"enabled": "maybe"}}), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert settings.update_check_enabled is True
    assert any("update_check.enabled" in warning for warning in settings.config_warnings)


def test_env_fallback_when_config_absent(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "env.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_ENABLE_SQLITE_VEC", "true")
    monkeypatch.setenv("MEMORY_ARBITER_VEC_DIM", "1024")
    monkeypatch.setenv("MEMORY_ARBITER_GGUF", str(tmp_path / "legacy.gguf"))

    settings = Settings.from_env()

    assert settings.db_path == tmp_path / "env.sqlite3"
    assert settings.enable_sqlite_vec is True
    assert settings.vec_dim == 1024
    assert settings.embedding_provider == "gguf"
    assert settings.embedding_model_path == tmp_path / "legacy.gguf"


def test_config_file_parse_error_graceful(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    bad_cfg = tmp_path / "bad.json"
    bad_cfg.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(bad_cfg))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "env.sqlite3"))

    settings = Settings.from_env()

    assert settings.db_path == tmp_path / "env.sqlite3"
    assert any("JSON parse failed" in warning for warning in settings.config_warnings)


def test_config_env_path_not_exist_fallback_xdg(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    home = tmp_path / "home"
    xdg_cfg = home / ".config" / "memory-arbiter" / "config.json"
    xdg_cfg.parent.mkdir(parents=True)
    xdg_cfg.write_text(json.dumps({"db_path": str(tmp_path / "xdg.sqlite3")}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(tmp_path / "missing.json"))

    settings = Settings.from_env()

    assert settings.db_path == tmp_path / "xdg.sqlite3"
    assert any("does not exist" in warning for warning in settings.config_warnings)


def test_bad_field_value_degrades_with_warning(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"vec": {"enabled": "maybe", "dim": "abc"}, "embedding": {"auto_write": "??"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert settings.enable_sqlite_vec is False
    assert settings.vec_dim == 768
    assert settings.embedding_auto_write is True
    assert any("vec.enabled" in warning for warning in settings.config_warnings)
    assert any("vec.dim" in warning for warning in settings.config_warnings)
    assert any("embedding.auto_write" in warning for warning in settings.config_warnings)


def test_parse_bool_false_string_is_false() -> None:
    assert parse_bool("false", default=True) is False
    assert parse_bool("0", default=True) is False
    assert parse_bool("no", default=True) is False


def test_embedding_model_path_without_provider_defaults_to_gguf(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"embedding": {"model_path": str(tmp_path / "model.gguf")}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert settings.embedding_provider == "gguf"
    assert settings.embedding_model_path == tmp_path / "model.gguf"





def test_server_build_runtime_exposes_tools_for_shutdown(tmp_path: Path, monkeypatch) -> None:
    class FakeFastMCP:
        def __init__(self, _name: str) -> None:
            self.tools = {}

        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = func
                return func

            return decorator

    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = FakeFastMCP
    fake_server = types.ModuleType("mcp.server")
    fake_mcp = types.ModuleType("mcp")
    fake_server.fastmcp = fake_fastmcp
    fake_mcp.server = fake_server
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "server.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "server.backup.jsonl"))
    monkeypatch.setenv("MEMORY_ARBITER_UPDATE_CHECK_ENABLED", "false")

    from memory_arbiter.server import build_runtime

    bundle = build_runtime()
    assert bundle.app.tools["memory"] is not None
    assert bundle.tools.shutdown(timeout=1)["ok"] is True
    assert bundle.tools.shutdown(timeout=1) == {"ok": True, "already_shutdown": True}
