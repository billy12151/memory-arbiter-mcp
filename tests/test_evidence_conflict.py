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


# ── 2026-08-21 review round: semantic-layer fixes ───────────────────────────

def test_unknown_sentinel_is_extraction_failure() -> None:
    """The protocol-legal '__unknown__' marker never becomes a usable field."""
    raw = '{"attribute_a":"数据库","value_a":"__unknown__","attribute_b":"数据库","value_b":"SQLite"}'
    extraction, error = extraction_from_text(raw)
    assert extraction is None
    assert error == "unknown_field"
    signal = model_signal_from_text(raw)
    # Distinguished from a protocol violation so diagnostics separate model
    # output from technical failure.
    assert signal.candidate_type == "unknown_field"
    assert signal.candidate is False


def test_top_level_array_output_is_rejected() -> None:
    raw = '[{"attribute_a":"db","value_a":"MySQL","attribute_b":"db","value_b":"SQLite"}]'
    extraction, error = extraction_from_text(raw)
    assert extraction is None
    assert error is not None and error.startswith("invalid_schema")


def test_bare_agent_marker_does_not_trigger_evolution_veto() -> None:
    # "由" as a passive/agent marker with no replacement wording must not veto.
    assert coexistence_veto(
        {"quote": "新网关由运维分配"}, {"quote": "端口是 8080"},
    ) is None
    # Real replacement wording still vetoes.
    assert coexistence_veto(
        {"quote": "旧网关已迁移到新集群"}, {"quote": "当前使用新集群"},
    ) == "coexist_explicit_evolution"


def test_unit_spelling_variants_normalize_equal_at_post_gate() -> None:
    # 8GB vs 8G is a restated duplicate, not a conflict, once units compact.
    result = evaluate_pair_extractions(
        AttributeValueExtraction("内存", "8GB", "内存", "8G"),
        AttributeValueExtraction("内存", "8G", "内存", "8GB"),
        {"quote": "内存 8GB"}, {"quote": "内存 8G"},
        require_bidirectional=True,
    )
    assert result.state == "review_candidate"
    assert result.reason == "not_same_attribute_different_value"


# ── 2026-08-21 review round 2: normalization edge cases ─────────────────────

def test_decimal_point_preserved_in_normalization() -> None:
    from memory_arbiter.semantic_conflict import normalize_value
    assert normalize_value("1.5s") != normalize_value("15s")
    # Integer/fractional pair stays distinct.
    assert normalize_value("1.5") != normalize_value("15")


def test_unknown_sentinel_rejection_is_case_insensitive() -> None:
    raw = '{"attribute_a":"db","value_a":"__UNKNOWN__","attribute_b":"db","value_b":"SQLite"}'
    extraction, error = extraction_from_text(raw)
    assert extraction is None
    assert error == "unknown_field"


def test_prose_prefixed_array_is_rejected() -> None:
    raw = 'Result: [ {"attribute_a":"db","value_a":"MySQL","attribute_b":"db","value_b":"SQLite"} ]'
    extraction, error = extraction_from_text(raw)
    assert extraction is None
    assert error is not None and error.startswith("invalid_schema")


def test_evolution_veto_covers_bian_wei_family() -> None:
    assert coexistence_veto(
        {"quote": "网关由 A 变为 B"}, {"quote": "当前网关是 B"},
    ) == "coexist_explicit_evolution"
    assert coexistence_veto(
        {"quote": "旧配置调整为新值"}, {"quote": "现在使用新值"},
    ) == "coexist_explicit_evolution"
