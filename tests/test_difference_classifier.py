"""0.16.2 difference classifier unit tests (plan §1.4): numeric cosine line,
similarity-route small symmetric difference, entity layer, garbage label,
and the real-library boundary pairs recorded during calibration."""
from __future__ import annotations

from memory_arbiter.difference_classifier import (
    classify_pair,
    has_small_symmetric_difference,
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
