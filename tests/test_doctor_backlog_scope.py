"""0.16.2/0.16.3 doctor-口径 tests: conflicts.scan_queue_backlog must count
internal contradictions (their own table) alongside the scan_queue rows —
scan_queue page's queue_backlog merges both, and a doctor that counts only
one table under-reports the agent workload (live case: 954 real items read
as 666)."""
from __future__ import annotations

from pathlib import Path

from memory_arbiter.doctor import run_all_checks
from memory_arbiter.models import utc_now_iso

from test_scan_pipeline import make_tools


def _backlog_finding(tools):
    with tools.db.connection() as conn:
        report = run_all_checks(conn, tools.settings)
    return next(
        (f for f in report.findings if f.check_id == "conflicts.scan_queue_backlog"),
        None,
    )


def test_doctor_backlog_merges_internal_contradictions(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = tools.memory_write(
        content="## 说明\n本条无内部矛盾，仅用于满足外键。",
        subject="internal", tags=[],
    )["data"]["id"]
    now = utc_now_iso()
    # One pending scan_queue row + two pending internal contradictions: the
    # agent workload is 3, not 1.
    with tools.db.write_transaction() as conn:
        conn.execute(
            """INSERT INTO scan_queue(kind,workspace_canonical,status,candidate_key_hash,
                 member_versions,evidence,reason,severity,source,detail,created_at,updated_at)
               VALUES('workspace','ws','pending',?, '[]','[]','vector vote 4/10',
                      'normal','scan_pipeline', NULL, ?, ?)""",
            ("a" * 64, now, now),
        )
        for i in (1, 2):
            conn.execute(
                """INSERT INTO internal_conflicts(memory_id,memory_version,status,
                     unit_a,unit_b,quote_a,quote_b,span_a,span_b,reason,detector_version,
                     created_at,updated_at)
                   VALUES(?,1,'pending',?,?, '甲句','乙句','[0,1]','[0,1]','polarity_changed',
                          'd1', ?, ?)""",
                (mid, i, i + 10, now, now),
            )
    finding = _backlog_finding(tools)
    assert finding is not None
    assert finding.evidence["backlog"] == 3, finding.evidence
    assert finding.evidence["internal_pending"] == 2, finding.evidence


def test_doctor_backlog_empty_library_passes(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    finding = _backlog_finding(tools)
    assert finding is not None and finding.status == "pass"
    assert finding.evidence["backlog"] == 0
    assert finding.evidence["internal_pending"] == 0
