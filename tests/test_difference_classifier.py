"""0.16.2 difference classifier unit tests (plan §1.4): numeric cosine line,
similarity-route small symmetric difference, entity layer, garbage label,
and the real-library boundary pairs recorded during calibration."""
from __future__ import annotations

from memory_arbiter.difference_classifier import (
    classify_pair,
    has_small_symmetric_difference,
    has_value_opposition,
    is_garbage,
    name_cosine,
)

NUM = "numeric_value_candidate"
SIM = "semantic_similarity_only"


def test_numeric_same_sentence_shape_kept() -> None:
    qa = "生产库使用 MySQL 5.7，主从延迟阈值为 300ms"
    qb = "生产库使用 MySQL 8.0，主从延迟阈值为 500ms"
    assert classify_pair(qa, qb, route=NUM) == "keep"


def test_numeric_cross_sentence_coincidence_cleared() -> None:
    qa = "部署完成后第一阶段的压测报告已经归档，共 500ms 采样窗口"
    qb = "会议纪要：第二季度 OKR 评审通过，预算 5.7 万元"
    assert name_cosine(qa, qb) < 0.5
    assert classify_pair(qa, qb, route=NUM) == "clear"


def test_similarity_value_difference_shape_kept() -> None:
    qa = "项目的数据库是 MySQL，主库在华东区"
    qb = "项目的数据库是 PostgreSQL，主库在华东区"
    assert has_small_symmetric_difference(qa, qb)
    assert classify_pair(qa, qb, route=SIM) == "keep"


def test_similarity_duplicates_cleared() -> None:
    qa = "0.16.0 乙包 12 个 commit 全部合入 main，两轮 review 全修后发版"
    qb = "0.16.0 乙包十二个 commit 已合入主干，两轮 review 全部修复并发版"
    assert classify_pair(qa, qb, route=SIM) == "clear"


def test_similarity_boundary_pairs_from_calibration() -> None:
    # Kept in calibration: emoji/status prefix variants of one claim.
    assert classify_pair(
        "🟢 **代码 + 测试 + 文档全部完成，未 commit / 未发版**——等用户确认后再 commit/tag/push/PyPI",
        "🟡 **代码 + 测试全部完成，未 commit / 未发版**——等用户确认后再 commit/tag/push/PyPI",
        route=SIM,
    ) == "keep"
    # Same-version chapter-header fragments kept (the calibration keep set).
    qa = "> 来源：金营项目-完整知识库.md（一、二章），文档更新 2026-06-08 V1.1"
    qb = "> 来源：金营项目-完整知识库.md（三章），文档更新 2026-06-08 V1.1"
    assert classify_pair(qa, qb, route=SIM) == "keep"
    # A VERSION DIFFERENCE on top ("V1.1" vs "V1.2") pushes unique tokens to
    # 3 — cleared under the document-literal (2,2) rule. This is the
    # recorded §1.4.3 deviation from the planner's looser 241-pair estimate:
    # 19 such version/date fragments are cleared rather than kept.
    qc = "> 来源：金营项目-完整知识库.md（三章），文档更新 2026-06-08 V1.2"
    assert classify_pair(qa, qc, route=SIM) == "clear"


def test_entity_layer_clears_different_subjects() -> None:
    qa = "发布时间是下周五"
    qb = "发布时间是下周五"
    assert classify_pair(qa, qb, route=SIM, entity_a="项目A", entity_b="项目B") == "clear"
    # Same entity or one-sided/absent entity never clears by itself.
    assert classify_pair(qa, qb, route=SIM, entity_a="项目A", entity_b="项目A") == "keep"
    assert classify_pair(qa, qb, route=SIM, entity_a="项目A", entity_b=None) == "keep"


def test_garbage_is_a_label_not_a_verdict() -> None:
    assert is_garbage("---")
    assert is_garbage("2026-09-13")
    assert is_garbage("12:30")
    assert is_garbage("短句")
    assert not is_garbage("生产库使用 MySQL 5.7 主从延迟 300ms")
    # A garbage-looking quote pair with a real numeric shape still keeps —
    # the label only feeds the counters.
    assert classify_pair("版本 1.1", "版本 1.2", route=NUM) == "keep"


def test_missing_quotes_clear() -> None:
    assert classify_pair(None, "任何内容", route=SIM) == "clear"
    assert classify_pair("任何内容", None, route=NUM) == "clear"
    assert classify_pair("", "", route=SIM) == "clear"


# ── value-opposition keep (2026-09-16, eval cf-oppo-01/06/07/11) ─────────
# The four harness pairs the (2,2) symdiff budget machine-cleared although
# each is a hard same-attribute opposition.


def test_value_opposition_latin_identifier_kept() -> None:
    # cf-oppo-01: MySQL vs PostgreSQL — unique tokens 8/8, symdiff hopeless.
    qa = "金营平台生产库使用 MySQL 双主架构，主从延迟容忍 500ms。"
    qb = "金营平台生产库使用 PostgreSQL 单主架构，只读副本两个。"
    assert has_value_opposition(qa, qb)
    assert classify_pair(qa, qb, route=SIM) == "keep"


def test_value_opposition_embedded_digit_token_kept() -> None:
    # cf-oppo-07: P6/P7 vs P5/P6 — unique 3/3, one token over the budget.
    qa = "算法工程师岗位定级 P6 到 P7，base 北京。"
    qb = "算法工程师岗位定级 P5 到 P6，base 上海。"
    assert classify_pair(qa, qb, route=SIM) == "keep"


def test_value_opposition_chinese_numeral_kept() -> None:
    # cf-oppo-11: LPR 四倍 vs 固定 24% — 四 canonicalises to 4.
    qa = "民间借贷受保护利率上限为一年期 LPR 的四倍。"
    qb = "民间借贷受保护利率上限为固定年利率 24%。"
    assert classify_pair(qa, qb, route=SIM) == "keep"


def test_value_opposition_one_sided_kept_under_strict_anchor() -> None:
    # cf-oppo-06: only one side carries a value (ISO9001) — kept only
    # because the shared-token anchor is strong (common >= 8).
    qa = "包装材料供应商必须持有 ISO9001 认证方可准入。"
    qb = "包装材料供应商无需任何认证，报价合格即可准入。"
    assert classify_pair(qa, qb, route=SIM) == "keep"


def test_value_opposition_one_sided_weak_anchor_cleared() -> None:
    # Same one-sided shape but barely any shared skeleton: noise, not an
    # opposition — the strict anchor must clear it.
    qa = "订单系统超时时间设置为 30s，超出自动重试。"
    qb = "前端页面改版完成，新增深色模式开关。"
    assert not has_value_opposition(qa, qb)
    assert classify_pair(qa, qb, route=SIM) == "clear"


def test_value_opposition_dotted_version_still_cleared() -> None:
    # The recorded §1.4.3 calibration pair: chapter difference + a VERSION
    # difference ("V1.1" vs "V1.2") pushes unique tokens to 3 — cleared
    # under the (2,2) rule, and dotted versions are NOT values, so the new
    # value-opposition shape must not resurrect it either.
    qa = "> 来源：金营项目-完整知识库.md（一、二章），文档更新 2026-06-08 V1.1"
    qc = "> 来源：金营项目-完整知识库.md（三章），文档更新 2026-06-08 V1.2"
    assert not has_value_opposition(qa, qc)
    assert classify_pair(qa, qc, route=SIM) == "clear"


def test_value_opposition_chinese_arabic_equality_stays_duplicate() -> None:
    # "12 个" vs "十二个": the Chinese numeral canonicalises to the same
    # Arabic token and subtracts away — a duplicate, never an opposition.
    qa = "0.16.0 乙包 12 个 commit 全部合入 main，两轮 review 全修后发版"
    qb = "0.16.0 乙包十二个 commit 已合入主干，两轮 review 全部修复并发版"
    assert not has_value_opposition(qa, qb)
    assert classify_pair(qa, qb, route=SIM) == "clear"


def test_value_opposition_year_only_difference_cleared() -> None:
    # Bare years are not values: a date-only difference on top of ordinary
    # wording divergence stays cleared (the year must not count as a value
    # even though the (2,2) symdiff already fails here).
    qa = "年度复盘会议定于 2025 年 12 月在上海总部召开，全员线下参加。"
    qb = "年度复盘会议定于 2024 年 12 月在北京分部召开，核心成员线上参加。"
    assert not has_value_opposition(qa, qb)
    assert classify_pair(qa, qb, route=SIM) == "clear"
