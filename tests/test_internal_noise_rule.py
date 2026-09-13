"""0.16.3 internal structural-noise tests (live-library calibrated): table
slices and note-meta lines are never same-memory contradictions. The
internal numeric cosine question is ALSO pinned here — low-cos numeric pairs
inside one document are often TRUE conflicts (row #34943: 0.29 vs 0.25
人力), so the genuine-shape gate stays and no cosine gate transfers from
the cross-memory calibration."""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.difference_classifier import (
    classify_pair,
    internal_noise_pair,
    name_cosine,
)

from test_scan_pipeline import make_tools, _write


TABLE = "| 需求状态 | 运营状态 | 入口A |\n|---|---|---|\n| 已完成 | 已完成 | ✅ |"
META = "> 更新：2026-07-10 | 来源：用户手动确认 ✅ | 对应文档：PRD_v2.md"


def test_rule_predicates() -> None:
    assert internal_noise_pair(TABLE, "该功能已上线") is True
    assert internal_noise_pair("普通句子", TABLE) is True
    assert internal_noise_pair(META, "普通句子") is True
    assert internal_noise_pair("重试次数为 3 次", "重试次数为 5 次") is False
    # A long quote block that is real content, not meta — must survive.
    assert internal_noise_pair(
        "> 引用用户原话：我认为方案甲更好，理由是……（后续 200 字实质内容）",
        "> 引用用户原话：我认为方案乙更好，理由是……",
    ) is False


def test_internal_numeric_low_cos_is_kept() -> None:
    """The anti-regression pin: the live true conflict #34943 (口径矛盾,
    cos 0.44) must stay a keeper — the cross-memorory cosine gate must never
    be applied to internal numeric pairs."""
    a = "正式统计口径以后以该 Excel 的《基础能力&风险识别（1期）》sheet 为准，不再沿用此前“32 个已上线需求/0.29 人力”口径。"
    b = "- 量化人效收益按 Excel《基础能力&风险识别（1期）》sheet 的“实际收益”列汇总，汇报统一用约 0.25 人力。"
    assert name_cosine(a, b) < 0.5
    # The genuine-shape gate (scan/write path) is what admits it; the noise
    # pair rule must NOT match it.
    assert internal_noise_pair(a, b) is False


def test_write_time_internal_skips_noise_shapes(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # Table + meta-line in one memory: rule-contradiction-looking but noise.
    mid = _write(
        tools,
        "状态矩阵",
        "## 矩阵甲\n" + TABLE + "\n## 矩阵乙\n| 需求状态 | 运营状态 |\n|---|---|\n| 待受理 | 待受理 |",
    )
    assert tools.wait_evidence_worker_drained(timeout=10)
    pending = [r for r in tools.db.internal_conflicts.list_pending() if r["memory_id"] == mid]
    assert pending == [], "table slices must not produce internal rows"


def test_write_time_internal_keeps_real_numeric(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # The LIVE true-conflict sentences (row #34943): numeric route through
    # the genuine-shape gate, low cosine — must still land.
    mid = _write(
        tools,
        "口径真矛盾",
        "## 口径甲\n正式统计口径以后以该 Excel 的《基础能力&风险识别（1期）》sheet 为准，"
        "不再沿用此前“32 个已上线需求/0.29 人力”口径。\n## 口径乙\n量化人效收益按 Excel"
        "《基础能力&风险识别（1期）》sheet 的“实际收益”列汇总，汇报统一用约 0.25 人力。",
    )
    assert tools.wait_evidence_worker_drained(timeout=10)
    pending = [r for r in tools.db.internal_conflicts.list_pending() if r["memory_id"] == mid]
    assert pending, "a genuine same-metric two-value pair must land (no cosine gate on internal)"


def test_stock_dismissal_migration(tmp_path: Path) -> None:
    from memory_arbiter.db import additive

    tools = make_tools(tmp_path)
    now = "2026-09-13T00:00:00+00:00"
    real = _write(tools, "真矛盾", "## 甲\n重试次数为 3 次。\n## 乙\n重试次数为 5 次。")
    assert tools.wait_evidence_worker_drained(timeout=10)
    with tools.db.write_transaction() as conn:
        # burned at boot; reset to simulate the upgrade boot with stock rows
        conn.execute("DELETE FROM migration_state WHERE key='internal_noise_rule_v1'")
        conn.execute(
            """INSERT INTO internal_conflicts(memory_id,memory_version,status,unit_a,unit_b,
                 quote_a,quote_b,span_a,span_b,reason,detector_version,created_at,updated_at)
               VALUES(?,1,'pending',90,91,?,?, '[0,1]','[0,1]','polarity_changed','d1', ?, ?)""",
            (real, TABLE, "已上线 ✅", now, now),
        )
        conn.execute(
            """INSERT INTO internal_conflicts(memory_id,memory_version,status,unit_a,unit_b,
                 quote_a,quote_b,span_a,span_b,reason,detector_version,created_at,updated_at)
               VALUES(?,1,'pending',92,93,?,?, '[0,1]','[0,1]','todo_resolved','d1', ?, ?)""",
            (real, META, "正文待办", now, now),
        )
    with tools.db.connection() as conn:
        applied = additive.ensure_additive_structures(conn)
    assert any("internal_noise_dismissal" in item for item in applied), applied
    pending_ids = {
        r["unit_a"]
        for r in tools.db.internal_conflicts.list_pending(limit=50)
        if r["memory_id"] == real
    }
    assert 90 not in pending_ids and 92 not in pending_ids, "noise stock must be dismissed"
    # the genuine numeric pair from the real write survives
    assert any(ua < 90 for ua in pending_ids), "real pairs are untouched"
    with tools.db.connection() as conn:
        applied2 = additive.ensure_additive_structures(conn)
    assert not any("internal_noise_dismissal" in item for item in applied2), "idempotent"
