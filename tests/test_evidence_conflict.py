from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.semantic_conflict import (
    decide_evidence,
    model_signal_from_text,
    notice_dedupe_key,
)
from memory_arbiter.tools import MemoryTools


def _tools(tmp_path: Path, **overrides) -> MemoryTools:
    values = {
        "db_path": tmp_path / "memory.db",
        "backup_jsonl": tmp_path / "backup.jsonl",
    }
    values.update(overrides)
    return MemoryTools(Settings(**values))


def test_deterministic_routes_are_explainable() -> None:
    notify = decide_evidence("PostgreSQL port is 5432", "PostgreSQL port is 3306")
    assert notify.action == "notify"
    assert notify.reason == "numeric_value_changed"
    assert decide_evidence("pgsql", "PostgreSQL").action == "ignore"
    assert decide_evidence("database connection pool", "database connection policy").action == "check"


def test_qwen_protocol_accepts_candidate_and_rejects_invalid_schema() -> None:
    accepted = model_signal_from_text('{"label":"conflict","same_fact_slot":true,"confidence":0.9}')
    assert accepted.candidate is True
    rejected = model_signal_from_text('{"label":"conflict","same_fact_slot":false,"confidence":0.9}')
    assert rejected.candidate is False
    assert rejected.error == "invalid_schema"
    noisy = model_signal_from_text('{"label":"not_conflict","same_fact_slot":true,"confidence":1.0}')
    assert noisy.candidate is False


def test_notice_dedupe_is_symmetric_and_version_pinned() -> None:
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_evidence") == notice_dedupe_key(
        2, 1, 4, 3, "semantic_evidence"
    )
    assert notice_dedupe_key(1, 2, 3, 4, "semantic_evidence") != notice_dedupe_key(
        1, 2, 4, 4, "semantic_evidence"
    )


def test_check_degrades_to_no_notice_without_qwen(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    assert tools._ensure_semantic_backend() is None
    decision = decide_evidence("database connection pool", "database connection policy")
    assert decision.action == "check"


def test_notice_freshness_uses_only_memory_versions(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    left = tools.memory_write(content="left", subject="s", workspace="w")["data"]["id"]
    right = tools.memory_write(content="right", subject="s", workspace="w")["data"]["id"]
    created = tools.db.record_semantic_notice(
        memory_id=left, peer_id=right, severity="normal",
        notice_type="semantic_evidence", title="candidate", message="check",
        payload={}, left_version=1, right_version=1,
        dedupe_key=notice_dedupe_key(left, right, 1, 1, "semantic_evidence"),
    )
    notice = tools.db.read_semantic_notice(created["notice_id"])
    assert notice["freshness"]["fresh"] is True
    tools.memory_edit(left, new_content="changed", reason="new")
    notice = tools.db.read_semantic_notice(created["notice_id"])
    assert notice["freshness"]["fresh"] is False


def test_qwen_uncertain_label_fails_closed() -> None:
    signal = model_signal_from_text('{"label":"uncertain","same_fact_slot":true,"confidence":0.9}')
    assert signal.candidate is False
    assert signal.candidate_type == "uncertain"
    assert signal.error is None
