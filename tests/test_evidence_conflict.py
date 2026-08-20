from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.semantic_conflict import (
    AttributeValueExtraction,
    coexistence_veto,
    decide_evidence,
    evaluate_pair_extractions,
    extraction_from_text,
    model_signal_from_text,
    notice_dedupe_key,
    value_is_grounded,
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
    numeric = decide_evidence("PostgreSQL port is 5432", "PostgreSQL port is 3306")
    assert numeric.action == "check"
    assert numeric.reason == "numeric_value_candidate"
    assert decide_evidence("pgsql", "PostgreSQL").action == "ignore"
    assert decide_evidence("database connection pool", "database connection policy").action == "check"


def test_qwen_protocol_is_strict_bounded_four_field_extraction() -> None:
    raw = '{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型","value_b":"SQLite"}'
    accepted = model_signal_from_text(raw)
    assert accepted.candidate is True
    extraction, error = extraction_from_text(raw)
    assert error is None and extraction is not None
    for invalid in (
        '{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型"}',
        '{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型","value_b":"SQLite","conflict":true}',
        '{"attribute_a":"数据库选型","value_a":3,"attribute_b":"数据库选型","value_b":"SQLite"}',
    ):
        parsed, parse_error = extraction_from_text(invalid)
        assert parsed is None and parse_error


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


def test_bidirectional_mapping_grounding_and_notice_gate() -> None:
    forward = AttributeValueExtraction("数据库引擎", "MySQL", "数据库选型", "SQLite")
    reverse = AttributeValueExtraction("数据库选型", "SQLite", "数据库引擎", "MySQL")
    result = evaluate_pair_extractions(
        forward, reverse,
        {"quote": "生产数据库使用 MySQL。"},
        {"quote": "生产数据库使用 SQLite。"},
        require_bidirectional=True,
    )
    assert result.state == "notice_ready"
    wrong_side = AttributeValueExtraction("数据库选型", "MySQL", "数据库引擎", "SQLite")
    rejected = evaluate_pair_extractions(
        forward, wrong_side,
        {"quote": "生产数据库使用 MySQL。"}, {"quote": "生产数据库使用 SQLite。"},
        require_bidirectional=True,
    )
    assert rejected.state == "review_candidate"
    assert rejected.reason == "bidirectional_mapping_mismatch"


def test_grounding_is_mechanical_and_coexistence_reasons_are_stable() -> None:
    assert value_is_grounded("5s", "接口超时为 5 秒。")
    assert value_is_grounded("PostgreSQL", "数据库采用 pgsql。")
    assert not value_is_grounded("关系数据库", "数据库采用 PostgreSQL。")
    assert coexistence_veto(
        {"quote": "测试环境数据库使用 MySQL"},
        {"quote": "生产环境数据库使用 SQLite"},
    ) == "coexist_environment_mismatch"
    assert coexistence_veto(
        {"quote": "v1 API timeout 5s"}, {"quote": "v2 API timeout 10s"},
    ) == "coexist_version_mismatch"


def test_single_direction_scan_survives_but_notice_fails_closed() -> None:
    extraction = AttributeValueExtraction("端口", "5432", "端口", "3306")
    scan = evaluate_pair_extractions(
        extraction, None, {"quote": "端口 5432"}, {"quote": "端口 3306"},
        require_bidirectional=False,
    )
    notice = evaluate_pair_extractions(
        extraction, None, {"quote": "端口 5432"}, {"quote": "端口 3306"},
        require_bidirectional=True,
    )
    assert scan.state == "review_candidate" and scan.reason == "single_direction_only"
    assert notice.state == "review_candidate" and notice.reason == "bidirectional_extraction_required"
