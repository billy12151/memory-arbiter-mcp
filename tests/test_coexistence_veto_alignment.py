"""B-C1: coexistence_veto dimension markers must align with extracted attributes.

Behaviour change (2026-08-28): when extractions are supplied, a dimension
marker pair only vetoes when each marker appears in the corresponding side's
extracted attribute text. Markers that occur only in the quote body no
longer suppress the notice (the old quote-substring matching dropped real
conflicts as false negatives).
"""

import pytest

from memory_arbiter.semantic_conflict import (
    AttributeValueExtraction,
    coexistence_veto,
    evaluate_pair_extractions,
)


def _extraction(
    left_attribute: str, right_attribute: str,
) -> tuple[AttributeValueExtraction, AttributeValueExtraction]:
    forward = AttributeValueExtraction(left_attribute, "MySQL", right_attribute, "SQLite")
    reverse = AttributeValueExtraction(right_attribute, "SQLite", left_attribute, "MySQL")
    return forward, reverse


# (expected code, left attribute, right attribute, left quote, right quote)
_ALIGNED_VETO_CASES = [
    (
        "coexist_environment_mismatch",
        "测试环境响应时间", "生产环境响应时间",
        "测试环境响应时间为 5s", "生产环境响应时间为 10s",
    ),
    (
        "coexist_region_mismatch",
        "中国区接口超时", "海外区接口超时",
        "中国区接口超时 5s", "海外区接口超时 10s",
    ),
    (
        "coexist_version_mismatch",
        "v1接口超时", "v2接口超时",
        "v1 API timeout 5s", "v2 API timeout 10s",
    ),
    (
        "coexist_object_mismatch",
        "移动端首页加载耗时", "管理后台首页加载耗时",
        "移动端首页加载耗时 5s", "管理后台首页加载耗时 10s",
    ),
    (
        "coexist_observation_time_mismatch",
        "昨天统计的错误数", "今天统计的错误数",
        "昨天统计错误数 5 条", "今天统计错误数 10 条",
    ),
    (
        "coexist_historical_current",
        "历史记录中的配额", "当前配额",
        "历史记录中的配额是 5", "当前配额是 10",
    ),
    (
        "coexist_metric_mismatch",
        "平均响应时间", "峰值响应时间",
        "平均响应时间为 5s", "峰值响应时间为 10s",
    ),
]

# (left quote, right quote, plain attribute shared by both sides)
_QUOTE_ONLY_CASES = [
    ("测试环境响应时间为 5s", "生产环境响应时间为 10s", "响应时间"),
    ("中国区接口超时 5s", "海外区接口超时 10s", "接口超时"),
    ("v1 API timeout 5s", "v2 API timeout 10s", "api超时"),
    ("移动端首页加载耗时 5s", "管理后台首页加载耗时 10s", "首页加载耗时"),
    ("昨天统计错误数 5 条", "今天统计错误数 10 条", "错误数"),
    ("历史记录中的配额是 5", "当前配额是 10", "配额"),
    ("平均响应时间为 5s", "峰值响应时间为 10s", "响应时间"),
]


@pytest.mark.parametrize(
    "code,left_attribute,right_attribute,left_quote,right_quote", _ALIGNED_VETO_CASES,
)
def test_marker_in_attribute_still_vetoes(
    code: str, left_attribute: str, right_attribute: str,
    left_quote: str, right_quote: str,
) -> None:
    forward, reverse = _extraction(left_attribute, right_attribute)
    left, right = {"quote": left_quote}, {"quote": right_quote}
    assert coexistence_veto(left, right, forward, reverse) == code
    # The pair is unordered: swapping sides must resolve to the same code.
    assert coexistence_veto(right, left, reverse, forward) == code


@pytest.mark.parametrize("left_quote,right_quote,attribute", _QUOTE_ONLY_CASES)
def test_marker_only_in_quote_does_not_veto(
    left_quote: str, right_quote: str, attribute: str,
) -> None:
    forward, reverse = _extraction(attribute, attribute)
    assert coexistence_veto(
        {"quote": left_quote}, {"quote": right_quote}, forward, reverse,
    ) is None


def test_real_conflict_with_metric_words_in_quote_still_notices() -> None:
    # "平均"/"峰值" appear in the prose but not in the extracted attribute,
    # so the pair is a genuine same-attribute conflict, not a dimension split.
    forward = AttributeValueExtraction("响应时间", "5s", "响应时间", "10s")
    reverse = AttributeValueExtraction("响应时间", "10s", "响应时间", "5s")
    result = evaluate_pair_extractions(
        forward, reverse,
        {"quote": "平均响应时间为 5s。"},
        {"quote": "峰值响应时间为 10s。"},
        require_bidirectional=True,
    )
    assert result.state == "notice_ready"
    assert result.reason == "same_attribute_different_grounded_value"


def test_gate_still_vetoes_when_attribute_carries_the_markers() -> None:
    # Identical normalized attributes that literally carry the dimension
    # markers keep the veto reachable through the full notice gate.
    attribute = "移动端与管理后台响应时间"
    forward = AttributeValueExtraction(attribute, "5s", attribute, "10s")
    reverse = AttributeValueExtraction(attribute, "10s", attribute, "5s")
    result = evaluate_pair_extractions(
        forward, reverse,
        {"quote": "移动端与管理后台响应时间为 5s。"},
        {"quote": "移动端与管理后台响应时间为 10s。"},
        require_bidirectional=True,
    )
    assert result.state == "review_candidate"
    assert result.reason == "coexist_object_mismatch"


def test_without_extractions_legacy_quote_matching_is_kept() -> None:
    assert coexistence_veto(
        {"quote": "测试环境数据库使用 MySQL"},
        {"quote": "生产环境数据库使用 SQLite"},
    ) == "coexist_environment_mismatch"


def test_evolution_veto_is_not_affected_by_alignment() -> None:
    forward, reverse = _extraction("网关", "网关")
    assert coexistence_veto(
        {"quote": "旧网关已迁移到新集群"}, {"quote": "当前使用新集群"},
        forward, reverse,
    ) == "coexist_explicit_evolution"
