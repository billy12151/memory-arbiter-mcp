"""Tests for scripts/backfill_subjects.py (T5).

Runs the backfill script's main() against a tmp DB with hand-inserted
empty-subject rows, verifying dry-run (plan only) and --apply (subject lands,
version bumps, memory_history records the reason, FTS picks up the new subject).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools

# Load the script as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import backfill_subjects  # noqa: E402


def _make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "backfill.sqlite3",
        backup_jsonl=tmp_path / "backfill.jsonl",
        client="pytest",
        agent_id="backfill-test",
        workspace="default",
        enable_sqlite_vec=False,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _insert_empty_subject_row(tools: MemoryTools, content: str) -> int:
    """Direct INSERT bypassing the now-required subject validation, mirroring
    the historical rows the script exists to fix."""
    db = tools.db
    with db.connection() as conn:
        cur = conn.execute(
            "INSERT INTO memories (content, agent_id, workspace, tags, source_type, "
            "event_time, ingest_time, confidence, protection_level, status, subject, "
            "metadata, version, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (content, "agent-a", "default", "[]", "agent_generated",
             "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", 0.5, "normal",
             "active", None, "{}", 1, "2026-01-01T00:00:00Z"),
        )
        mid = int(cur.lastrowid)
        if db.state.fts5_available:
            conn.execute(
                "INSERT INTO memories_fts(rowid, content, tags, subject) VALUES (?, ?, ?, ?)",
                (mid, content, "", ""),
            )
        conn.commit()
    return mid


def test_backfill_dry_run_and_apply(tmp_path: Path, monkeypatch) -> None:
    tools = _make_tools(tmp_path)
    mid1 = _insert_empty_subject_row(tools, "workspace 归一治理规则")
    mid2 = _insert_empty_subject_row(tools, "JingleAI 自检记录")

    # Patch the script's SUBJECT_MAP + _load_tools for this tmp DB
    monkeypatch.setattr(backfill_subjects, "SUBJECT_MAP", {
        mid1: "workspace语义归一治理",
        mid2: "JingleAI-mema-write自检",
    })
    monkeypatch.setattr(backfill_subjects, "_load_tools", lambda: tools)

    # Dry-run: prints plan, does not apply, returns 0
    monkeypatch.setattr(sys, "argv", ["backfill_subjects.py"])
    rc_dry = backfill_subjects.main()
    assert rc_dry == 0
    # subjects still empty after dry-run
    assert tools.db.get_memory(mid1)["subject"] is None

    # Apply: subjects land, version bumps, history records the reason
    monkeypatch.setattr(sys, "argv", ["backfill_subjects.py", "--apply"])
    rc_apply = backfill_subjects.main()
    assert rc_apply == 0

    m1 = tools.db.get_memory(mid1)
    assert m1["subject"] == "workspace语义归一治理"
    assert m1["version"] == 2  # bumped from 1

    m2 = tools.db.get_memory(mid2)
    assert m2["subject"] == "JingleAI-mema-write自检"
    assert m2["version"] == 2

    # memory_history has the backfill reason
    with tools.db.connection() as conn:
        hist = conn.execute(
            "SELECT reason FROM memory_history WHERE memory_id=? ORDER BY id DESC LIMIT 1",
            (mid1,),
        ).fetchone()
    assert hist is not None
    assert "backfill empty subject" in hist["reason"]

    # No empty-subject rows remain
    assert backfill_subjects._empty_subject_ids(tools) == []


def test_backfill_unmapped_rows_hard_fail(tmp_path: Path, monkeypatch) -> None:
    """If the DB has an empty-subject row with no mapping, --apply must exit 2
    (not silently skip)."""
    tools = _make_tools(tmp_path)
    _insert_empty_subject_row(tools, "unmapped content")

    monkeypatch.setattr(backfill_subjects, "SUBJECT_MAP", {})  # no mapping
    monkeypatch.setattr(backfill_subjects, "_load_tools", lambda: tools)
    monkeypatch.setattr(sys, "argv", ["backfill_subjects.py", "--apply"])
    rc = backfill_subjects.main()
    assert rc == 2  # hard fail, unmapped


def test_backfill_plan_content_hash_mismatch_hard_fails(tmp_path: Path, monkeypatch) -> None:
    """A built-in plan entry must match the expected content hash before apply.
    Same integer id in a different DB is not enough."""
    tools = _make_tools(tmp_path)
    mid = _insert_empty_subject_row(tools, "actual content")

    monkeypatch.setattr(backfill_subjects, "SUBJECT_MAP", {mid: "safe subject"})
    monkeypatch.setattr(backfill_subjects, "BACKFILL_PLAN", {
        mid: {
            "subject": "safe subject",
            "workspace": "default",
            "content_hash": "0" * 64,
        }
    })
    monkeypatch.setattr(backfill_subjects, "_load_tools", lambda: tools)
    monkeypatch.setattr(sys, "argv", ["backfill_subjects.py", "--apply"])

    rc = backfill_subjects.main()

    assert rc == 2
    assert tools.db.get_memory(mid)["subject"] is None


def test_backfill_skips_non_active_rows(tmp_path: Path, monkeypatch) -> None:
    """B3: only status='active' rows are candidates. A superseded empty-subject
    row must NOT appear in the plan (memory_edit would reject it anyway)."""
    tools = _make_tools(tmp_path)
    mid_active = _insert_empty_subject_row(tools, "active empty")
    mid_super = _insert_empty_subject_row(tools, "superseded empty")
    # manually mark mid_super as superseded
    with tools.db.connection() as conn:
        conn.execute("UPDATE memories SET status='superseded' WHERE id=?", (mid_super,))
        conn.commit()

    monkeypatch.setattr(backfill_subjects, "SUBJECT_MAP", {mid_active: "test-subject"})
    monkeypatch.setattr(backfill_subjects, "_load_tools", lambda: tools)
    monkeypatch.setattr(sys, "argv", ["backfill_subjects.py", "--apply"])
    rc = backfill_subjects.main()
    assert rc == 0
    # active row backfilled
    assert tools.db.get_memory(mid_active)["subject"] == "test-subject"
    # superseded row untouched (still empty subject)
    assert tools.db.get_memory(mid_super)["subject"] is None
