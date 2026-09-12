"""0.16.0 commit 7 tests: write-time internal examination runs FIRST (E10①),
lands in internal_conflicts (never the conflicts table), and survives a
truncated cross-memory loop."""
from __future__ import annotations

from pathlib import Path

from test_scan_pipeline import make_tools, _write


def test_write_time_internal_conflict_detected_first(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # One memory contradicting itself AND sitting next to a cross-memory peer:
    # the internal pair must land even if the cross loop is truncated.
    mid = _write(tools, "内部矛盾", "## 配置甲\n重试次数为 3 次。\n## 配置乙\n重试次数为 5 次。")
    peer = _write(tools, "邻居", "## 配置丙\n重试次数为 7 次。")
    assert tools.wait_evidence_worker_drained(timeout=15)
    pending = tools.db.internal_conflicts.list_pending()
    assert pending, "写时内部矛盾必须落 internal_conflicts"
    assert all(row["memory_id"] == mid for row in pending)
    # Never in the conflicts table (pair invariants untouched)
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT member_versions FROM conflicts WHERE status IN ('open','applying','candidate')"
        ).fetchall()
    for row in rows:
        member_ids = {int(m["memory_id"]) for m in __import__("json").loads(row["member_versions"])}
        assert member_ids != {mid}, "同记忆内部冲突不得进 conflicts 表"


def test_internal_conflict_expires_on_version_lift(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "版本内部", "## 配置甲\n超时 30 秒。\n## 配置乙\n超时 60 秒。")
    assert tools.wait_evidence_worker_drained(timeout=15)
    assert tools.db.internal_conflicts.list_pending(), "当前版本 pending"
    tools.memory("update", {"memory_id": mid, "new_content": "## 配置甲\n超时 30 秒。", "reason": "修正矛盾"})
    assert tools.wait_evidence_worker_drained(timeout=15)
    pending_now = tools.db.internal_conflicts.list_pending()
    assert not any(row["memory_id"] == mid for row in pending_now), "版本抬升后旧内部矛盾不再 pending"
