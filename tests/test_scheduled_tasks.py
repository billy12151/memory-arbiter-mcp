"""Scheduled-task guidance (v0.15.2 PR2): scan_log.jsonl write-back, the
three-tier guidance notice, its self-closing contract, and doctor's
never-run/stale findings.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools
from memory_arbiter.update_monitor import UpdateMonitor


def make_tools(tmp_path: Path, *, with_monitor: bool = True) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "m.sqlite3", backup_jsonl=tmp_path / "b.jsonl")
    tools = MemoryTools(settings, MemoryDB(settings))
    if with_monitor:
        tools.start_update_monitor(UpdateMonitor(
            enabled=False, state_path=tmp_path / "notice_state.json",
        ))
    return tools


def _notice_types(tools: MemoryTools) -> list[str]:
    result = tools.memory("status", {})
    assert result["ok"] is True
    return [notice.get("type") for notice in result.get("notices") or []]


def test_scan_log_written_on_full_boundary_with_audit_fields(tmp_path: Path) -> None:
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    # Ensure both model fields are present so the audit entry records them.
    tools.settings.semantic_conflict_model_path = tools.settings.embedding_model_path
    tools.start_update_monitor(UpdateMonitor(
        enabled=False, state_path=tmp_path / "notice_state.json",
    ))
    tools.memory_write(content="alpha deployment note", subject="alpha note", tags=[], workspace="w")
    tools.memory_write(content="beta deployment note", subject="beta note", tags=[], workspace="w")
    assert tools.wait_evidence_worker_drained(timeout=5)

    result = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10})
    assert result["ok"] is True, result
    assert result["data"].get("next_anchor_memory_id") is None, "two memories fit in batch 50"

    path = tools.db.scan_log_path
    assert path.exists()
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert lines, "successful full-scan boundary must append to scan_log.jsonl"
    entry = lines[-1]
    assert entry["status"] == "completed"
    assert isinstance(entry["duration_sec"], float) and entry["duration_sec"] >= 0
    assert entry["client"] is None and entry["agent_id"] is None  # stdio has no identity
    assert entry["embedding_model"] == str(tools.settings.embedding_model_path)
    assert entry["semantic_model"] == str(tools.settings.semantic_conflict_model_path)
    for dropped in (
        "anchors_scanned", "candidates", "knn_pairs", "rule_pass",
        "duplicates_truncated", "next_anchor_memory_id", "workspace",
    ):
        assert dropped not in entry, f"{dropped!r} is a per-page metric and must be dropped"

    # Failing scans (vec unavailable) must not append activity evidence.
    tools.db.state.sqlite_vec_available = False
    failed = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10})
    assert failed["ok"] is False
    after = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(after) == len(lines)


def test_scan_log_full_boundary_writes_and_partial_page_does_not(tmp_path: Path) -> None:
    import tests.test_vnext_evidence as tv
    from memory_arbiter.doctor import _last_completed_scan

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    tools.settings.semantic_conflict_model_path = tools.settings.embedding_model_path
    tools.memory_write(content="first deployment note", subject="first note", tags=[], workspace="w")
    tools.memory_write(content="second deployment note", subject="second note", tags=[], workspace="w")
    assert tools.wait_evidence_worker_drained(timeout=5)

    path = tools.db.scan_log_path

    # First page with batch=1 stops in the middle of the library.
    first = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 1, "k": 10})
    assert first["ok"] is True, first
    next_anchor = first["data"].get("next_anchor_memory_id")
    assert next_anchor is not None, "two memories require a second page"
    assert not path.exists() or not path.read_text().strip(), "partial page must not write scan_log"

    # Second page reaches the full boundary.
    final = tools.memory_repair("scan_candidates", {"anchor_memory_id": int(next_anchor), "batch": 1, "k": 10})
    assert final["ok"] is True, final
    assert final["data"].get("next_anchor_memory_id") is None

    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1, "only the completed full boundary is logged"
    entry = lines[-1]
    assert entry["status"] == "completed"
    assert set(entry.keys()) == {
        "scan_time", "duration_sec", "status", "client", "agent_id",
        "embedding_model", "semantic_model",
    }

    # Both read paths return the newest completed line.
    assert tools.db.audit.scan_log_last_completed()["scan_time"] == entry["scan_time"]
    assert _last_completed_scan(tools.settings)["scan_time"] == entry["scan_time"]


def test_notice_fires_never_run_then_self_closes_on_scan(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="some memory", subject="s", tags=[], workspace="w")

    types = _notice_types(tools)
    assert "scan_never_run" in types

    # Suppression window: not re-delivered on the very next response.
    assert "scan_never_run" not in _notice_types(tools)

    # Simulate the scheduled task having run: a completed scan_log line is the
    # closing evidence. Reset the suppression state so detection re-runs.
    entry = {
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "duration_sec": 0.1, "status": "completed", "duplicates_truncated": False,
        "anchors_scanned": 1, "candidates": 0, "knn_pairs": 0, "rule_pass": 0,
        "next_anchor_memory_id": None, "client": None, "agent_id": None,
    }
    tools.db.scan_log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    tools._update_monitor.write_state_key(
        tools._scheduled_task_notice_state_key(), {"type": None, "last_at": 0, "checked_at": 0},
    )
    assert "scan_never_run" not in _notice_types(tools)


def test_notice_scan_required_tier(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="some memory", subject="s", tags=[], workspace="w")
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO migration_state(key,value) VALUES('conflict_scan_required','true') "
            "ON CONFLICT(key) DO UPDATE SET value='true'"
        )
    tools._update_monitor.write_state_key(
        tools._scheduled_task_notice_state_key(), {"type": None, "last_at": 0, "checked_at": 0},
    )
    types = _notice_types(tools)
    assert "scan_required" in types
    # Delivery records the tier; the next response is inside the suppress window.
    state = tools._update_monitor.read_state_key(tools._scheduled_task_notice_state_key())
    assert state["type"] == "scan_required"


def test_notice_stale_tier(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="some memory", subject="s", tags=[], workspace="w")
    old = datetime.now(timezone.utc) - timedelta(days=15)
    entry = {
        "scan_time": old.isoformat(), "duration_sec": 0.1, "status": "completed",
        "duplicates_truncated": False, "anchors_scanned": 1, "candidates": 0, "knn_pairs": 0,
        "rule_pass": 0, "next_anchor_memory_id": None, "client": None, "agent_id": None,
    }
    tools.db.scan_log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    types = _notice_types(tools)
    assert "scan_stale" in types


def test_doctor_reports_never_run_and_stale(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="some memory", subject="s", tags=[], workspace="w")

    def finding_map() -> dict[str, dict]:
        report = tools.memory_doctor_overview(deep=False)
        data = report.get("data") or report
        return {f.get("check_id"): f for f in data.get("findings", [])}

    # Fresh library, never scanned: must NOT report the old fake-green text.
    findings = finding_map()
    required = findings["conflicts.scan_required"]
    assert required["severity"] != "info" or required["status"] != "pass" or "never" in required["detail"].lower() or "not set up" in required["detail"]
    assert "complete" not in required["detail"].lower() or "not set up" in required["detail"]

    # Recent activity silences never-run; old activity trips scan_stale.
    recent = {
        "scan_time": datetime.now(timezone.utc).isoformat(), "status": "completed",
    }
    tools.db.scan_log_path.write_text(json.dumps(recent) + "\n", encoding="utf-8")
    findings = finding_map()
    assert findings["conflicts.scan_stale"]["status"] == "pass"
    assert "complete" in findings["conflicts.scan_required"]["detail"].lower()

    old = {
        "scan_time": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
        "status": "completed",
    }
    tools.db.scan_log_path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    findings = finding_map()
    assert findings["conflicts.scan_stale"]["status"] == "warn"


def test_notice_suppression_key_is_per_library(tmp_path: Path) -> None:
    """Round-2 M1: two libraries share the user-home notice state file; the
    suppression key must be namespaced per db_path so a healthy library
    cannot silence another library's guidance."""
    tools = make_tools(tmp_path)
    tools.memory_write(content="some memory", subject="s", tags=[], workspace="w")
    assert "scan_never_run" in _notice_types(tools)
    key = tools._scheduled_task_notice_state_key()
    assert key.startswith("scheduled_task_notice:") and key != "scheduled_task_notice"
    assert str(tmp_path) not in key  # hashed, not a raw path leak

    other_dir = tmp_path / "other-library"
    other_dir.mkdir()
    other = make_tools(other_dir)
    other.memory_write(content="other memory", subject="s", tags=[], workspace="w")
    assert other._scheduled_task_notice_state_key() != key
    assert "scan_never_run" in _notice_types(other)


def test_clean_check_does_not_advance_suppression_window(tmp_path: Path) -> None:
    """Round-2 M2: only a delivered notice advances last_at; a healthy check
    refreshes just the 1h negative cache so a library that goes stale is
    re-detected within the hour instead of after the full 7-day window."""
    tools = make_tools(tmp_path)
    tools.memory_write(content="some memory", subject="s", tags=[], workspace="w")
    key = tools._scheduled_task_notice_state_key()

    # Healthy evidence: no notice; last_at must stay 0.
    entry = {
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "duration_sec": 0.1, "status": "completed", "duplicates_truncated": False,
        "anchors_scanned": 1, "candidates": 0, "knn_pairs": 0, "rule_pass": 0,
        "next_anchor_memory_id": None, "client": None, "agent_id": None,
    }
    tools.db.scan_log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert "scan_never_run" not in _notice_types(tools)
    state = tools._update_monitor.read_state_key(key)
    assert state["type"] is None
    assert float(state["last_at"]) == 0.0, "clean check must not open the suppression window"

    # Goes stale; detection must be possible right after the negative cache
    # expires (not gated by a 7-day last_at that a clean check would have set).
    old = {
        "scan_time": (datetime.now(timezone.utc) - timedelta(days=15)).isoformat(),
        "duration_sec": 0.1, "status": "completed", "duplicates_truncated": False,
        "anchors_scanned": 1, "candidates": 0, "knn_pairs": 0, "rule_pass": 0,
        "next_anchor_memory_id": None, "client": None, "agent_id": None,
    }
    tools.db.scan_log_path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    tools._update_monitor.write_state_key(key, {
        "type": None, "last_at": 0, "checked_at": 0,
    })
    assert "scan_stale" in _notice_types(tools)


def test_scheduled_tasks_help_topic_self_serve(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory("help", {"topic": "scheduled_tasks"})
    assert result["ok"] is True
    data = result["data"]
    assert data["topic"] == "scheduled_tasks"
    names = [task["name"] for task in data["setup"]["tasks"]]
    assert names == ["conflict_scan", "governance_reminder"]
    cadences = {task["name"]: task["cadence"] for task in data["setup"]["tasks"]}
    assert cadences == {"conflict_scan": "hourly", "governance_reminder": "daily"}
