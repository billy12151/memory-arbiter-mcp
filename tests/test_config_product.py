# ── from test_product_config.py ──

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
from memory_arbiter.models import SourceType
from memory_arbiter.tools import MemoryTools
from functools import partial

from tests._helpers import make_tools as _base_make_tools


def _sc(result):
    """Unwrap the 0.16.0 single-copy FastMCP response (CallToolResult)."""
    # 0.16.5 single-copy envelope: the compact JSON rides content[0].text.
    if hasattr(result, "content"):
        return json.loads(result.content[0].text)
    return result



@pytest.fixture(autouse=True)
def _relevance_floor_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0.15.9: the tests in this file exercise scoping/preview/metering with
    # content-only fixtures; the query-recall relevance floor is orthogonal to
    # what they pin (its behavior lives in tests/test_recall_quality_0159.py).
    # Disable it here so weak fixtures stay recallable.
    monkeypatch.setattr("memory_arbiter.search.QUERY_RECALL_SCORE_FLOOR", -1.0)


make_tools = partial(_base_make_tools, client="codex", agent_id="agent-a", workspace="repo-a")


def test_server_memory_edit_preserves_tags_when_new_tags_omitted(tmp_path: Path, monkeypatch) -> None:
    """Regression: the MCP wrapper must pass new_tags=None through.

    Passing [] erases existing tags on a content-only edit, even though the
    MemoryTools layer correctly preserves tags when new_tags is omitted.
    """

    class FakeFastMCP:
        def __init__(self, _name: str, **_kwargs) -> None:
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
    monkeypatch.setenv("MEMORY_ARBITER_CLIENT", "client-a")

    from memory_arbiter.server import build_runtime

    bundle = build_runtime()
    app = bundle.app
    assert set(app.tools) == {"memory", "memory_review", "memory_govern", "memory_repair"}
    written = _sc(app.tools["memory"](
        action="remember",
        data={"workspace": "default", 
            "content": "draft content",
            "subject": "server-wrapper",
            "tags": ["keep-me"],
            "source_type": "agent_generated",
            "event_time": "2026-01-01T00:00:00Z",
        },
    ))
    memory_id = written["data"]["id"]

    edited = _sc(app.tools["memory"](
        action="update",
        data={"memory_id": memory_id, "new_content": "edited content"},
    ))

    assert edited["ok"] is True
    assert edited["data"]["record"]["content"] == "edited content"
    assert edited["data"]["record"]["tags"] == ["keep-me"]
    bundle.tools.shutdown(timeout=1)


def test_server_always_exposes_only_product_tools(tmp_path: Path, monkeypatch) -> None:
    class FakeFastMCP:
        def __init__(self, _name: str, **_kwargs) -> None:
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
    server_config = tmp_path / "server-config.json"
    server_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(server_config))
    monkeypatch.setenv("MEMORY_ARBITER_TOOL_PROFILE", "legacy_full")
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "server.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "server.backup.jsonl"))
    monkeypatch.setenv("MEMORY_ARBITER_CLIENT", "test-client")
    monkeypatch.setenv("MEMORY_ARBITER_AGENT_ID", "test-agent")

    from memory_arbiter.server import build_runtime

    bundle = build_runtime()
    app = bundle.app

    assert set(app.tools) == {"memory", "memory_review", "memory_govern", "memory_repair"}
    bundle.tools.shutdown(timeout=1)



def test_product_memory_review_and_govern_wrappers(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory(
        action="remember",
        data={"workspace": "default", 
            "workspace": "default","content": "old whole fact", "subject": "old", "tags": ["govern"]},
    )
    replacement = tools.memory(
        action="remember",
        data={"workspace": "default", 
            "workspace": "default","content": "new whole fact", "subject": "new", "tags": ["govern"]},
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
    written = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "old", "subject": "old"})
    memory_id = written["data"]["id"]
    tools.memory(action="update", data={"id": memory_id, "new_content": "new"})

    history_by_alias = tools.memory_review(view="history", data={"id": memory_id})
    assert history_by_alias["ok"] is True
    assert history_by_alias["data"]["count"] == 1

    bad_history = tools.memory_review(view="history", data={"id": "abc"})
    assert bad_history["ok"] is False
    assert bad_history["data"].get("field") == "memory_id" or "memory_id" in bad_history["data"]["error"]

    for label, result in {
        "retire": tools.memory_govern(action="retire", data={"id": "abc", "reason": "x", "authorized": True}),
        "confirm": tools.memory_govern(action="confirm", data={"id": "abc", "authorized": True}),
        "resolve": tools.memory_govern(action="resolve_conflict", data={"id": "abc", "authorized": True}),
        "split": tools.memory_repair(task="split", data={"id": "abc"}),
        "activate": tools.memory_repair(task="activate_pending", data={"id": "abc", "authorized": True}),
        "cleanup": tools.memory_repair(task="cleanup_history", data={"id": "abc", "authorized": True}),
    }.items():
        assert result["ok"] is False, label
        assert "integer" in result["data"].get("reason", result["data"]["error"])

    judge_missing = tools.memory(action="judge", data={"id": 1})
    assert judge_missing["ok"] is False
    assert "missing" in judge_missing["data"]["error"]
    assert "expected_revision" in judge_missing["data"]["help"]["missing_fields"]


def test_product_repair_cleanup_history_id_alias_is_not_full_cleanup(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "a", "subject": "a"})["data"]["id"]
    second = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "b", "subject": "b"})["data"]["id"]
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
    help_result = tools.memory_repair(task="help", data={"task": "rebuild_evidence"})
    assert help_result["ok"] is True
    assert "rebuild_evidence" in help_result["data"]["tasks"]

    dry = tools.memory_repair(task="rebuild_evidence", data={"dry_run": True})
    assert dry["ok"] is True
    assert dry["data"]["dry_run"] is True


def test_string_false_authorized_fails_closed_across_product_surfaces(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default",
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
    memory_id = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default",
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
        "resolve_conflict": {"conflict_id": 1, "expected_revision": 2},
        "confirm": {"memory_id": 1},
        "apply_conflict_action": {
            "conflict_id": 1, "expected_revision": 2, "memory_id": 1,
            "action": "update_current_claim", "content": "corrected fact",
        },
        "rename_workspace_canonical": {"old": "old", "new": "new", "workspace": "ws"},
        "migrate_workspace": {"from": "old", "to": "new", "workspace": "ws"},
        "confirm_pending_workspace": {"memory_id": 1, "canonical": "canonical", "workspace": "ws"},
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
    assert_clean_error("govern.apply_conflict_action", tools.memory_govern(action="apply_conflict_action", data={}))
    assert_clean_error("govern.resolve_conflict", tools.memory_govern(action="resolve_conflict", data={}))
    assert_clean_error("repair.split", tools.memory_repair(task="split", data={}))
    assert_clean_error("repair.set_entity", tools.memory_repair(task="set_entity", data={}))
    assert_clean_error("repair.activate_pending", tools.memory_repair(task="activate_pending", data={}))

    # cleanup_history legitimately allows a missing memory_id (full cleanup) —
    # it must still reach its own authorized gate, not crash.
    cleanup = tools.memory_repair(task="cleanup_history", data={})
    assert cleanup["ok"] is False
    assert "authorized" in cleanup["data"]["error"]


def test_product_judge_help_exposes_group_decision_constraints(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    mem_help = tools.memory(action="help")["data"]
    constraints = mem_help["judge_constraints"]
    assert constraints["decided_by"] == ["user", "agent"]
    assert set(constraints["apply_actions"]) == {
        "update_current_claim", "append_superseded_context", "preserve_historical_record",
        "use_as_resolution", "needs_authorization",
    }
    assert any("expected_revision" in rule for rule in constraints["rules"])
    assert any("apply_conflict_action" in rule for rule in constraints["rules"])

    gov_help = tools.memory_govern(action="help")["data"]
    assert gov_help["judge_constraints"] == constraints
    assert gov_help["actions"] == [
        "retire", "merge_memories", "apply_conflict_action", "replan_conflict", "resolve_conflict", "confirm",
        "rename_workspace_canonical", "migrate_workspace", "move_memories_workspace", "rollback_auto_move",
        "separate_workspace_alias", "confirm_pending_workspace", "confirm_workspaces", "help",
    ]
    assert "accept_workspace_alias" not in gov_help["examples"]
    assert "reject_workspace_alias" not in gov_help["accepted_fields"]

    missing = tools.memory(action="judge", data={"id": 1})
    assert missing["ok"] is False
    assert missing["data"]["help"]["judge_constraints"] == constraints
    assert "expected_revision" in missing["data"]["help"]["missing_fields"]

def test_product_help_exposes_agent_onboarding_topic(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    help_doc = tools.memory(action="help", data={"topic": "agent_onboarding"})["data"]
    assert help_doc["topic"] == "agent_onboarding"
    assert help_doc["notice"] == "agent-onboarding:v2"
    assert help_doc["guide_file"] == "memory_arbiter/AGENT_ONBOARDING.md"
    content = help_doc["content"]
    assert "Memory Arbiter Agent Rule" in content
    assert "memory(action" not in content  # compact prose, not a duplicate API manual
    assert "memory_govern" in content
    assert "strict project scope" in content
    assert "authorized=true" in content
    assert len(content.encode("utf-8")) < 3000


def test_product_help_exposes_agent_decision_paths(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    memory_help = tools.memory(action="help")["data"]
    assert memory_help["judge_required_fields"] == [
        "conflict_id", "expected_revision", "chosen_value", "decided_by",
        "ref", "reason", "apply_plan", "resolution_memory_id",
    ]
    paths = memory_help["action_required_paths"]
    assert "not a formal conflict" in paths["read_semantic_notice"]
    assert "freshness.fresh=true" in paths["read_semantic_notice"]
    assert "authorized=true" in paths["confirm_new_workspace"]
    assert "Authorization is mandatory" in paths["ask_user_for_authorization"]
    assert {"apply_conflict_action", "preview_backup_replay", "inspect_backup_replay_manually"} <= paths.keys()
    assert "add_tags" in memory_help["accepted_fields"]["update"]
    assert memory_help["value_reference"]["source_type"] == [
        "user_confirmed", "document_extracted", "agent_generated", "unknown", "pending",
    ]
    assert "tags_only" in memory_help["value_reference"]["update_modes"]

    govern_help = tools.memory_govern(action="help")["data"]
    assert "user_confirmed" in govern_help["confirm_actions"]["confirm"]
    assert "strict isolation" in govern_help["confirm_actions"]["confirm_pending_workspace"]
    assert "authorized" in govern_help["accepted_fields"]["confirm_pending_workspace"]

    repair_help = tools.memory_repair(task="help")["data"]
    assert repair_help["semantic_control_actions"] == [
        "status", "pause", "resume", "enable", "unload", "disable",
    ]
    assert repair_help["action_required_paths"] == paths
    assert "action" in repair_help["accepted_fields"]["semantic_control"]


def test_help_record_conflict_example_is_valid_and_executable(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="database is MySQL", subject="database", workspace="repo-a")["data"]
    right = tools.memory_write(content="database is SQLite", subject="database", workspace="repo-a")["data"]
    example = tools.memory_repair(task="help")["data"]["examples"]["record_conflict"]
    payload = json.loads(json.dumps(example["data"]))
    assert len(payload["members"]) == len(payload["value_groups"]) == 2
    actual_rows = [tools.db.get_memory(left["id"]), tools.db.get_memory(right["id"])]
    for member, actual in zip(payload["members"], actual_rows):
        member["memory_id"] = actual["id"]
        member["version"] = actual["version"]
        member["content_hash"] = hashlib.sha256(actual["content"].encode()).hexdigest()
    payload["value_groups"][0]["members"] = [f"{actual_rows[0]['id']}@{actual_rows[0]['version']}"]
    payload["value_groups"][1]["members"] = [f"{actual_rows[1]['id']}@{actual_rows[1]['version']}"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", member["content_hash"]) for member in payload["members"])
    result = tools.memory_repair(task=example["task"], data=payload)
    assert result["ok"] is True
    assert result["data"]["outcome"] == "inserted"


def test_emitted_action_required_literals_have_help_paths(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    root = Path(__file__).parents[1] / "memory_arbiter"
    emitted: set[str] = set()
    pattern = re.compile(r'["\']action_required["\']\s*:\s*["\']([^"\']+)["\']')
    for source in root.rglob("*.py"):
        emitted.update(pattern.findall(source.read_text(encoding="utf-8")))
    paths = set(tools.memory(action="help")["data"]["action_required_paths"])
    assert emitted <= paths, sorted(emitted - paths)
    assert "authorized=true" in tools.memory(action="help")["data"]["action_required_paths"]["review_workspace_registry"]


def test_product_help_exposes_fields_for_read_only_surface(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    review_help = tools.memory_review(view="help")["data"]
    assert "conflict_id" in review_help["accepted_fields"]["conflict_detail"]
    assert "memory_id" in review_help["accepted_fields"]["history"]
    assert "ask_user" in review_help["action_required_paths"]


def test_semantic_control_invalid_action_is_a_failed_call_with_help(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    result = tools.memory_repair(task="semantic_control", data={"action": "bogus"})
    assert result["ok"] is False
    assert result["data"]["outcome"] == "invalid_action"
    assert result["data"]["valid_actions"] == [
        "status", "pause", "resume", "enable", "unload", "disable",
    ]
    assert "semantic_control" in result["data"]["help"]["accepted_fields"]



def test_product_forwards_handle_bad_secondary_int_args(tmp_path: Path) -> None:
    """v0.11.1: secondary int args must not crash the product surface.

    Before the ``_forward`` wrapper, ``int()`` inside the low-level method
    raised straight through on loosely-typed JSON (e.g. ``"limit": "abc"``).
    These forwards now return ``ok=False`` with help. Numeric strings
    (``"5"``), which MCP clients commonly send, still coerce and succeed.
    """
    tools = make_tools(tmp_path)
    new_id = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "new", "subject": "s"})["data"]["id"]

    def assert_clean(label: str, result: dict) -> None:
        assert result["ok"] is False, f"{label} unexpectedly succeeded: {result}"
        assert "error" in result["data"], f"{label} must attach an error"

    # retire: non-int superseded_by (primary memory_id coerced, secondary not)
    old = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "old", "subject": "s"})["data"]["id"]
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


def test_product_judge_and_apply_conflict_action_field_ordering(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    r = tools.memory(action="judge", data={"id": "abc"})
    assert r["ok"] is False
    assert "missing_fields" in r["data"]["help"]
    assert "chosen_value" in r["data"]["help"]["missing_fields"]

    r = tools.memory(action="judge", data={
        "conflict_id": "abc", "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "agent", "ref": None, "reason": "reviewed",
        "apply_plan": [], "resolution_memory_id": None,
    })
    assert r["ok"] is False
    assert "conflict_id" in r["data"]["error"]

    r = tools.memory_govern(action="apply_conflict_action", data={
        "conflict_id": "abc", "expected_revision": 2, "memory_id": 1,
        "action": "update_current_claim", "content": "database is sqlite",
        "authorized": True,
    })
    assert r["ok"] is False
    assert r["data"]["field"] == "conflict_id"



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
    assert_clean("repair.claims_backfill data=str", tools.memory_repair(task="claims_backfill", data="bad"))
    assert_clean("review.conflicts data=int", tools.memory_review(view="conflicts", data=123))

    # data=None (explicit) is the default and must still work.
    assert tools.memory(action="status", data=None)["ok"] is True
    assert tools.memory(action="find", data=None)["ok"] is True


def test_product_memory_write_rejects_non_list_tags(tmp_path: Path) -> None:
    """The product schema rejects wrong tag types instead of coercing them."""
    tools = make_tools(tmp_path)

    r = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "a", "subject": "s", "tags": "todo"})
    assert r["ok"] is False
    assert r["data"]["field"] == "tags"

    r = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "b", "subject": "s", "tags": 123})
    assert r["ok"] is False
    assert r["data"]["field"] == "tags"

    # None → []
    r = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "c", "subject": "s"})
    assert r["ok"] is True
    assert r["data"]["record"]["tags"] == []

    # list preserved
    r = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "d", "subject": "s", "tags": ["todo", "project"]})
    assert r["ok"] is True
    assert r["data"]["record"]["tags"] == ["todo", "project"]

    # Product JSON accepts arrays only; Python tuples are rejected at the boundary.
    r = tools.memory(action="remember", data={"workspace": "default", 
        "workspace": "default","content": "e", "subject": "s", "tags": ("a", "b")})
    assert r["ok"] is False
    assert r["data"]["field"] == "tags"



def test_tool_profile_env_is_no_longer_read(tmp_path: Path, monkeypatch, clear_config_env) -> None:
    # 0.15.0: tool_profile was dead configuration (the product surface is the
    # only surface); a stale env export must warn instead of silently nothing.
    monkeypatch.setenv("MEMORY_ARBITER_TOOL_PROFILE", "legacy_full")
    settings = Settings.from_env()
    assert not hasattr(settings, "tool_profile")
    assert any(
        "MEMORY_ARBITER_TOOL_PROFILE" in warning and "no longer read" in warning
        for warning in settings.config_warnings
    )



def test_config_file_overrides_env(tmp_path: Path, monkeypatch, clear_config_env) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_db = tmp_path / "from-config.sqlite3"
    env_db = tmp_path / "from-env.sqlite3"
    # vec.enabled / vec.dim / embedding.provider were removed in 0.15.0 —
    # present in the file they must warn and be ignored; model_path IS the
    # embedding intent now.
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
    assert not hasattr(settings, "vec_dim")
    assert not hasattr(settings, "embedding_provider")
    assert settings.embedding_model_path == tmp_path / "model.gguf"
    assert settings.embedding_auto_query is False
    assert settings.update_check_enabled is True
    joined = "\n".join(settings.config_warnings)
    assert "vec.enabled" in joined and "no longer configurable" in joined
    assert "vec.dim" in joined
    assert "embedding.provider" in joined
    assert "MEMORY_ARBITER_VEC_DIM" in joined and "no longer read" in joined


def test_update_check_config_switch(tmp_path: Path, monkeypatch, clear_config_env) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"update_check": {"enabled": False}}), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert settings.update_check_enabled is False


def test_update_check_bad_config_defaults_enabled(tmp_path: Path, monkeypatch, clear_config_env) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"update_check": {"enabled": "maybe"}}), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert settings.update_check_enabled is True
    assert any("update_check.enabled" in warning for warning in settings.config_warnings)


def test_notice_sync_wait_ms_parse_int_edges(tmp_path: Path, monkeypatch, clear_config_env) -> None:
    """Adversarial round 2: semantic_conflict.notice_sync_wait_ms must never
    crash startup or silently mis-parse. bools are rejected (bool is an int
    subclass — True would become a 1 ms wait), Infinity (1e999) overflows
    int() and must fall back to the default instead of raising
    OverflowError, out-of-range ints clamp with a warning, and garbage
    strings fall back to the default.

    (clear_config_env used to be re-invoked inside the case loop; the fixture
    now clears once up front — equivalent here because cfg_path is loop-
    invariant, each iteration rewrites the same file, and setenv overwrites
    the same keys.)"""
    from memory_arbiter.constants import NOTICE_SYNC_WAIT_MS

    cases = [
        # (raw JSON literal, expected value, expects warning)
        ("true", NOTICE_SYNC_WAIT_MS, True),
        ("false", NOTICE_SYNC_WAIT_MS, True),
        ("1e999", NOTICE_SYNC_WAIT_MS, True),  # Infinity: pre-fix OverflowError crash
        ('"abc"', NOTICE_SYNC_WAIT_MS, True),
        ("-100", 0, True),
        ("99999", 5000, True),
        ("0", 0, False),
        ('"100"', 100, False),
        ("1500.7", 1500, False),
    ]
    for raw, expected, expect_warning in cases:
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            '{"semantic_conflict": {"notice_sync_wait_ms": %s}}' % raw,
            encoding="utf-8",
        )
        monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

        settings = Settings.from_env()

        assert settings.semantic_conflict_notice_sync_wait_ms == expected, raw
        warned = any(
            "semantic_conflict.notice_sync_wait_ms" in warning
            for warning in settings.config_warnings
        )
        assert warned is expect_warning, (raw, settings.config_warnings)


def test_env_fallback_when_config_absent(tmp_path: Path, monkeypatch, clear_config_env) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "env.sqlite3"))
    # 0.15.0: only launch-context env remains. The former vec/GGUF env knobs
    # are no longer read — each stale export warns and is ignored.
    monkeypatch.setenv("MEMORY_ARBITER_ENABLE_SQLITE_VEC", "true")
    monkeypatch.setenv("MEMORY_ARBITER_VEC_DIM", "1024")
    monkeypatch.setenv("MEMORY_ARBITER_GGUF", str(tmp_path / "legacy.gguf"))

    settings = Settings.from_env()

    assert settings.db_path == tmp_path / "env.sqlite3"
    assert settings.embedding_model_path is None
    warnings_text = "\n".join(settings.config_warnings)
    for name in (
        "MEMORY_ARBITER_ENABLE_SQLITE_VEC",
        "MEMORY_ARBITER_VEC_DIM",
        "MEMORY_ARBITER_GGUF",
    ):
        assert f"{name} is no longer read" in warnings_text


def test_workspace_recall_cutoff_env_is_no_longer_read(
    tmp_path: Path, monkeypatch,
    clear_config_env,
) -> None:
    # 0.15.0 froze workspace_recall_cutoff at its former default; the env knob
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORY_ARBITER_WORKSPACE_RECALL_CUTOFF", "nan")
    settings = Settings.from_env()
    assert not hasattr(settings, "workspace_recall_cutoff")
    assert any(
        "MEMORY_ARBITER_WORKSPACE_RECALL_CUTOFF" in warning and "no longer read" in warning
        for warning in settings.config_warnings
    )


def test_config_file_parse_error_graceful(tmp_path: Path, monkeypatch, clear_config_env) -> None:
    bad_cfg = tmp_path / "bad.json"
    bad_cfg.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(bad_cfg))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "env.sqlite3"))

    settings = Settings.from_env()

    assert settings.db_path == tmp_path / "env.sqlite3"
    assert any("JSON parse failed" in warning for warning in settings.config_warnings)


def test_config_env_path_not_exist_fallback_xdg(tmp_path: Path, monkeypatch, clear_config_env) -> None:
    home = tmp_path / "home"
    xdg_cfg = home / ".config" / "memory-arbiter" / "config.json"
    xdg_cfg.parent.mkdir(parents=True)
    xdg_cfg.write_text(json.dumps({"db_path": str(tmp_path / "xdg.sqlite3")}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(tmp_path / "missing.json"))

    settings = Settings.from_env()

    assert settings.db_path == tmp_path / "xdg.sqlite3"
    assert any("does not exist" in warning for warning in settings.config_warnings)


def test_bad_field_value_degrades_with_warning(tmp_path: Path, monkeypatch, clear_config_env) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"vec": {"enabled": "maybe", "dim": "abc"}, "embedding": {"auto_write": "??"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert settings.embedding_auto_write is True
    # vec.* keys were removed in 0.15.0: they warn as no-longer-configurable
    # instead of being parsed and degrading.
    assert any(
        "vec.enabled" in warning and "no longer configurable" in warning
        for warning in settings.config_warnings
    )
    assert any(
        "vec.dim" in warning and "no longer configurable" in warning
        for warning in settings.config_warnings
    )
    # A still-live key with a bad value keeps the degrade-with-warning path.
    assert any("embedding.auto_write" in warning for warning in settings.config_warnings)


def test_parse_bool_false_string_is_false() -> None:
    assert parse_bool("false", default=True) is False
    assert parse_bool("0", default=True) is False
    assert parse_bool("no", default=True) is False


def test_embedding_model_path_without_provider_enables_embedding(
    tmp_path: Path, monkeypatch, clear_config_env,
) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({
            "embedding": {
                "provider": "gguf",  # removed key: ignored with a warning
                "model_path": str(tmp_path / "model.gguf"),
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert not hasattr(settings, "embedding_provider")
    assert settings.embedding_model_path == tmp_path / "model.gguf"
    assert any(
        "embedding.provider" in warning and "no longer configurable" in warning
        for warning in settings.config_warnings
    )





def test_server_build_runtime_exposes_tools_for_shutdown(tmp_path: Path, monkeypatch) -> None:
    class FakeFastMCP:
        def __init__(self, _name: str, **_kwargs) -> None:
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
    server_config = tmp_path / "shutdown-config.json"
    server_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(server_config))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "server.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "server.backup.jsonl"))
    server_config.write_text(
        json.dumps({"update_check": {"enabled": False}}), encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CLIENT", "test-client")
    monkeypatch.setenv("MEMORY_ARBITER_AGENT_ID", "test-agent")

    from memory_arbiter.server import build_runtime

    bundle = build_runtime()
    assert bundle.app.tools["memory"] is not None
    assert bundle.tools.shutdown(timeout=1)["ok"] is True
    assert bundle.tools.shutdown(timeout=1) == {"ok": True, "already_shutdown": True}


def test_real_server_product_wrappers_preserve_non_object_data(tmp_path: Path, monkeypatch) -> None:
    class FakeFastMCP:
        def __init__(self, _name: str, **_kwargs) -> None:
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
    wrapper_config = tmp_path / "wrapper-config.json"
    wrapper_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(wrapper_config))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "wrapper.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "wrapper.jsonl"))
    wrapper_config.write_text(
        json.dumps({"update_check": {"enabled": False}}), encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CLIENT", "test-client")
    monkeypatch.setenv("MEMORY_ARBITER_AGENT_ID", "test-agent")

    from memory_arbiter.server import build_runtime

    bundle = build_runtime()
    calls = [
        ("memory", {"action": "remember", "data": []}),
        ("memory_review", {"view": "audit", "data": "bad"}),
        ("memory_govern", {"action": "retire", "data": 1}),
        ("memory_repair", {"task": "notice", "data": False}),
    ]
    for name, kwargs in calls:
        result = _sc(bundle.app.tools[name](**kwargs))
        assert result["ok"] is False
        assert "data must be a JSON object" in result["data"]["error"]
    bundle.tools.shutdown(timeout=1)


def test_server_delegates_generation_gate_to_memorydb(
    tmp_path: Path, monkeypatch,
) -> None:
    import sqlite3
    from memory_arbiter import server

    legacy = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(legacy) as conn:
        conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("CREATE TABLE memories_vec(id INTEGER PRIMARY KEY)")
    settings = Settings(
        db_path=legacy,
        backup_jsonl=tmp_path / "backup.jsonl",
        client="test-client",
        agent_id="test-agent",
    )
    monkeypatch.setattr(server.Settings, "from_env", classmethod(lambda cls: settings))
    with pytest.raises(RuntimeError, match="mema doctor --json"):
        server.build_runtime()
    with sqlite3.connect(legacy) as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='memory_row'"
        ).fetchone() is None


def test_memorydb_library_entry_rejects_legacy_database(tmp_path: Path) -> None:
    import sqlite3

    legacy = tmp_path / "legacy-library.sqlite3"
    with sqlite3.connect(legacy) as conn:
        conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("CREATE TABLE memories_vec(id INTEGER PRIMARY KEY)")
    settings = Settings(
        db_path=legacy,
        backup_jsonl=tmp_path / "backup.jsonl",
    )
    before = legacy.read_bytes()
    with pytest.raises(RuntimeError, match="mema doctor --json"):
        MemoryDB(settings)
    assert legacy.read_bytes() == before
