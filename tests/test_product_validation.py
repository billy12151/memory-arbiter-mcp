import math
from pathlib import Path

from memory_arbiter.config import AgentPolicy, Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools
from memory_arbiter.validation import (
    MAX_CONTENT_BYTES,
    MAX_REPLACEMENT_TEXT_CHARS,
    MAX_TEXT_FIELD_CHARS,
    PRODUCT_FIELD_REGISTRY,
    validate_product_payload,
)


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    return MemoryTools(settings, MemoryDB(settings))


def test_remember_requires_content_and_subject(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for data, field in (({"subject": "s"}, "content"), ({"content": "x"}, "subject")):
        result = tools.memory("remember", data)
        assert result["ok"] is False
        assert result["data"]["field"] == field


def test_remember_rejects_lifecycle_status_values(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for status in ("superseded", "conflicted", "deleted", "bogus"):
        result = tools.memory("remember", {"content": "x", "subject": "s", "status": status})
        assert result["ok"] is False, status
        assert result["data"]["error"] == "invalid_input"
        assert result["data"]["field"] == "status"
    allowed = tools.memory("remember", {"content": "x", "subject": "s", "status": "active"})
    assert allowed["ok"] is True
    assert allowed["data"]["record"]["status"] == "active"
    pending = tools.memory("remember", {"content": "x", "subject": "s", "status": "pending"})
    assert pending["ok"] is True
    assert pending["data"]["record"]["status"] == "pending"


def test_memory_write_guard_rejects_lifecycle_status_without_surface_validation(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_write(content="x", subject="s", status="deleted")
    assert result["ok"] is False
    assert result["data"]["error"] == "invalid_input"
    assert result["data"]["field"] == "status"
    assert tools.db.get_memory(result["data"].get("id") or 0) is None


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


def test_agent_policy_precedence_deny_allow_client_default() -> None:
    policy = AgentPolicy(
        client_defaults={"codex": False, "claude": True},
        default_enabled=False,
        allow_agents=["allowed", "both"],
        deny_agents=["denied", "both"],
    )
    assert policy.enabled_for("claude", "both") is False
    assert policy.enabled_for("codex", "allowed") is True
    assert policy.enabled_for("codex", "ordinary") is False
    assert policy.enabled_for("unknown", "ordinary") is False


def test_controlled_numeric_string_coercion_matches_consumed_value() -> None:
    payload = {"limit": "5", "offset": "2"}
    result = validate_product_payload("memory", "find", payload, vec_dim=2)
    assert result.error is None
    assert payload == {"limit": 5, "offset": 2}


def test_integer_id_and_cas_fields_reject_floats_and_non_finite_values() -> None:
    cases = [
        ("memory", "read", "memory_id"),
        ("memory", "find", "limit"),
        ("memory", "update", "expected_version"),
        ("memory", "judge", "expected_left_version"),
    ]
    for surface, operation, field in cases:
        for value in (1.0, 1.5, math.nan, math.inf, -math.inf):
            payload = {field: value}
            result = validate_product_payload(surface, operation, payload, vec_dim=2)
            assert result.error is not None, (surface, operation, field, value)
            assert result.error["field"] == field

    payload = {"memory_id": " 12 ", "expected_version": "+3"}
    result = validate_product_payload("memory", "update", payload, vec_dim=2)
    assert result.error is None
    assert payload == {"memory_id": 12, "expected_version": 3}

    payload = {"memory_ids": ["1", 2]}
    result = validate_product_payload("memory_repair", "rebuild_evidence", payload, vec_dim=2)
    assert result.error is None
    assert payload["memory_ids"] == [1, 2]
    for value in (1.0, math.nan, math.inf):
        result = validate_product_payload(
            "memory_repair", "rebuild_evidence", {"memory_ids": [value]}, vec_dim=2,
        )
        assert result.error is not None


def test_cas_pins_and_semantic_timeout_bounds(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    judge_base = {
        "conflict_id": 1, "expected_left_version": 1, "expected_right_version": 1,
        "verdict": "uncertain", "recommended_use": "ask_user", "suggested_winner": None,
        "confidence_hint": "low", "reason": "test", "affects_current_output": False,
        "usage_context": "unknown",
    }
    for field in ("expected_left_version", "expected_right_version"):
        bad = dict(judge_base, **{field: 0})
        result = tools.memory("judge", bad)
        assert result["ok"] is False
        assert result["data"]["field"] == field
    for timeout in (-0.1, 601, math.inf, "NaN"):
        result = tools.memory_repair("semantic_control", {"action": "status", "timeout": timeout})
        assert result["ok"] is False
        assert result["data"]["field"] == "timeout"


def test_textual_resource_boundaries(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    memory_id = tools.memory("remember", {"content": "abc", "subject": "s"})["data"]["id"]
    cases = [
        ("memory", "update", {"memory_id": memory_id, "old_text": "x" * (MAX_REPLACEMENT_TEXT_CHARS + 1), "new_text": "y"}, "old_text"),
        ("memory_govern", "accept_workspace_alias", {"alias": "x" * (MAX_TEXT_FIELD_CHARS + 1), "canonical": "c", "authorized": True}, "alias"),
        ("memory_repair", "set_entity", {"memory_id": memory_id, "entity": "x" * (MAX_TEXT_FIELD_CHARS + 1)}, "entity"),
    ]
    dispatch = {"memory": tools.memory, "memory_govern": tools.memory_govern, "memory_repair": tools.memory_repair}
    for surface, operation, payload, field in cases:
        result = dispatch[surface](operation, payload)
        assert result["ok"] is False
        assert result["data"]["field"] == field


def test_notice_authorized_is_not_registered_and_notice_remains_unauthorized(tmp_path: Path) -> None:
    assert "authorized" not in PRODUCT_FIELD_REGISTRY[("memory_repair", "notice")]
    tools = make_tools(tmp_path)
    result = tools.memory_repair("notice", {"action": "list", "authorized": True})
    assert result["ok"] is True
    assert "unknown field ignored: authorized" in result["warnings"]


def test_product_field_registry_covers_all_declared_surface_operations() -> None:
    expected = {
        "memory": {"help", "status", "remember", "find", "read", "update", "judge"},
        "memory_review": {"overview", "doctor", "audit", "conflicts", "conflict_detail", "judgments", "history", "expired", "entities", "help"},
        "memory_govern": {"retire", "resolve_conflict", "confirm", "correct_judgment", "accept_workspace_alias", "reject_workspace_alias", "rename_workspace_canonical", "migrate_workspace", "confirm_pending_workspace", "help"},
        "memory_repair": {"rebuild_evidence", "cleanup_history", "set_entity", "activate_pending", "semantic_control", "notice", "replay_backup", "help"},
    }
    actual = {
        surface: {operation for registered_surface, operation in PRODUCT_FIELD_REGISTRY if registered_surface == surface}
        for surface in expected
    }
    assert actual == expected


def test_workspace_vector_publish_failure_does_not_fail_memory_write(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.isolation = "weak"
    tools.db.state.sqlite_vec_available = True

    class Embedder:
        embedding_space_id = "test"

        def embed_text(self, **_kwargs):
            return type("Result", (), {"embedding": [0.1] * tools.settings.vec_dim})()

    monkeypatch.setattr(tools, "_ensure_embedder", lambda: (Embedder(), []))
    monkeypatch.setattr(tools, "_embedding_configured", lambda: True)
    original_connection = tools.db.connection

    class FailingConnection:
        def __init__(self, conn):
            self._conn = conn

        def __enter__(self):
            self._entered = self._conn.__enter__()
            return self

        def __exit__(self, *args):
            return self._conn.__exit__(*args)

        def execute(self, sql, params=()):
            if "INSERT OR REPLACE INTO workspace_canonicals_vec" in sql:
                import sqlite3
                raise sqlite3.OperationalError("injected workspace vector failure")
            return self._entered.execute(sql, params)

        def commit(self):
            return self._entered.commit()

    monkeypatch.setattr(tools.db.workspaces, "connection", lambda: FailingConnection(original_connection()))
    result = tools.memory("remember", {"content": "fact", "subject": "s", "workspace": "new-project"})
    assert result["ok"] is True
    assert result["data"]["id"] is not None
