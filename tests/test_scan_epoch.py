"""0.16.0 commit 8 tests: detector-epoch semantics — five-class arming via
the boot-time watermark clear, ordinary changes never re-arm, doctor carries
the reason, and the first-call side-channel notice is one-shot self-closing
(E9 four-layer notification)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter import db_generation
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools

from memory_arbiter.db_generation import CONFLICT_DETECTOR_VERSION as _CDV

from test_scan_pipeline import make_tools, _write


def test_detector_bump_clears_watermarks(monkeypatch) -> None:
    tools = make_tools(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    mid = _write(tools, "纪元主题", "纪元正文内容")
    tools.db.mark_scanned(mid, 1)
    assert tools.db.pending_scan_memory_ids() == []
    # Simulate a NEW detector binary: bump the running constant, then boot
    # another MemoryTools on the same library.
    monkeypatch.setattr(db_generation, "CONFLICT_DETECTOR_VERSION", "attribute-value-v3")
    settings = tools.settings
    db2 = MemoryDB(settings)
    tools2 = MemoryTools(settings=settings, db=db2)
    assert tools2.db.pending_scan_memory_ids() == [mid], "detector bump 必须清水位线布防全量"
    arm = tools2.db.meta.scan_epoch_arm()
    assert arm["from"] == _CDV
    assert arm["to"] == "attribute-value-v3"
    assert "cleared 1 watermarks" in arm["reason"]


def test_same_detector_boot_does_not_rearm() -> None:
    tools = make_tools(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    mid = _write(tools, "稳定主题", "稳定正文内容")
    tools.db.mark_scanned(mid, 1)
    # Boot again with the SAME detector identity: no re-arm, watermarks stand.
    settings = tools.settings
    db2 = MemoryDB(settings)
    tools2 = MemoryTools(settings=settings, db=db2)
    assert tools2.db.pending_scan_memory_ids() == []
    arm1 = tools.db.meta.scan_epoch_arm()
    arm2 = tools2.db.meta.scan_epoch_arm()
    # First boot records the arm (from=none → v2); the second boot with the
    # SAME detector must not re-arm it (same timestamp, watermarks intact).
    assert arm1["to"] == _CDV
    assert arm2["at"] == arm1["at"]


def test_doctor_reports_epoch_reason() -> None:
    tools = make_tools(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    report = tools.memory_doctor_overview(deep=False)
    payload = report.get("data") or report
    findings = {f["check_id"]: f for f in payload["findings"]}
    finding = findings.get("conflicts.scan_epoch")
    assert finding is not None
    assert _CDV in finding["detail"]
    assert finding["evidence"]["to"] == _CDV


def test_side_channel_notice_one_shot_and_self_closing() -> None:
    tools = make_tools(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    _write(tools, "侧信道主题", "侧信道正文内容")
    notice = tools._detect_full_scan_notice()
    assert notice is not None
    assert notice["type"] == "full_scan_required"
    assert "scan_pipeline" in notice["message"]
    # Self-closing: complete the round under the armed detector → gone.
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 50})
    assert tools._detect_full_scan_notice() is None


def test_epoch_arm_expires_old_epoch_queue_rows() -> None:
    tools = make_tools(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    a = _write(tools, "过期甲", "数据库是 MySQL")
    b = _write(tools, "过期乙", "数据库是 PostgreSQL")
    tools.wait_evidence_worker_drained(timeout=15)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    with tools.db.connection() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE status='pending'"
        ).fetchone()[0]
    assert pending >= 1
    # Re-boot with a bumped detector: the old epoch's queue rows are DELETED
    # outright (owner standing rule — the full round re-judges the library,
    # so every old-semantics row is residue; deletion also releases the
    # candidate identities for the fresh re-enqueue).
    import tempfile as tf
    from memory_arbiter.db_generation import CONFLICT_DETECTOR_VERSION as _
    monkey = pytest.MonkeyPatch()
    monkey.setattr(db_generation, "CONFLICT_DETECTOR_VERSION", "attribute-value-v3")
    try:
        db2 = MemoryDB(tools.settings)
        tools2 = MemoryTools(settings=tools.settings, db=db2)
        with tools2.db.connection() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM scan_queue WHERE kind='conflict'").fetchone()[0]
        assert remaining == 0, "old-epoch queue rows must be deleted, not carried over"
    finally:
        monkey.undo()


def test_detector_bump_does_not_resurrect_dismissed_pairs(monkeypatch, tmp_path) -> None:
    """Agent-judged pairs must survive a detector upgrade without re-judging.

    The candidate hash embeds detector_version, so after a bump the exact-hash
    suppression lookup MISSES — the only remaining line is the member-refs
    backstop (memory_id@version, detector-agnostic). The epoch re-boot also
    deletes the old-epoch queue rows, which retires the UNIQUE(idempotency)
    defence on purpose: if suppression is broken the fresh enqueue succeeds
    and this test goes red for the right reason.
    """
    from memory_arbiter import scan_pipeline as _sp
    from memory_arbiter.db import evidence_store as _es

    tools = make_tools(tmp_path)
    a = _write(tools, "换代甲", "该功能包含缓存模块")
    b = _write(tools, "换代乙", "该功能不包含缓存模块")
    assert tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT candidate_key_hash FROM scan_queue "
            "WHERE kind='conflict' AND status='pending'"
        ).fetchone()
    assert row is not None, "polarity notify 对必须入队"
    dismissed_hash = row["candidate_key_hash"]

    # The agent dismisses through the REAL submit channel: the disposition
    # lands not_a_conflict in the conflicts table (audit + suppression source).
    submit = tools.memory_repair("scan_queue", {"action": "submit", "decisions": [
        {"candidate_key_hash": dismissed_hash, "status": "dismissed",
         "reason": "epoch 防复活回归：噪音对"},
    ]})
    assert submit["ok"], submit

    # Detector upgrade: bump every from-import binding, then re-boot — the
    # epoch arm clears watermarks (full re-scan) and deletes old queue rows.
    bumped = "attribute-value-v4-test"
    monkeypatch.setattr(db_generation, "CONFLICT_DETECTOR_VERSION", bumped)
    monkeypatch.setattr(_sp, "CONFLICT_DETECTOR_VERSION", bumped)
    monkeypatch.setattr(_es, "CONFLICT_DETECTOR_VERSION", bumped)
    db2 = MemoryDB(tools.settings)
    tools2 = MemoryTools(settings=tools.settings, db=db2)
    with tools2.db.connection() as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE kind='conflict'").fetchone()[0]
        suppressed = conn.execute(
            "SELECT candidate_key_hash FROM conflicts WHERE status='not_a_conflict'"
        ).fetchone()
    assert remaining == 0, "换代 boot 清旧 epoch 队列行（UNIQUE 幂等防线退场）"
    assert suppressed is not None and suppressed["candidate_key_hash"] == dismissed_hash

    result = tools2.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert result["ok"], result
    with tools2.db.connection() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE status='pending'").fetchone()[0]
    assert pending == 0, (
        "detector 换代使精确 hash 匹配失效后，已判 not_a_conflict 的对必须被"
        "成员 refs 兜底挡住——Agent 判过的对不跨代重判"
    )
