"""C4/C5 (0.15.13): triage counters in doctor/console + scan page progress kv.

C4: not_a_conflict cumulative / this-week / latest-time counters ride the
doctor conflicts.backlog evidence (the C2 convergence observability face)
and the console overview counts.

C5: every productive GLOBAL scan page upserts the scan_page_progress kv
(per-workspace tallies + global cursor); doctor reports a broken chain when
the record is incomplete, older than an hour, and no completion line
follows it. scan_log.jsonl keeps its completed-lines-only semantics (#825).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter.tools import MemoryTools


@pytest.fixture()
def vec_tools(tmp_path: Path):
    import tests.test_vnext_evidence as tv

    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    yield tools


def _write(tools: MemoryTools, content: str, workspace: str = "w") -> dict:
    return tools.memory_write(
        content=content, subject="c45", tags=[], workspace=workspace,
    )["data"]


def _dismiss_pair(tools: MemoryTools, a: dict, b: dict) -> None:
    assert tools.wait_evidence_worker_drained(timeout=5)
    scan = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True,
    })
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(
        (c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair),
        None,
    )
    assert clue is not None
    recorded = tools.memory_repair("record_conflict", {
        "slot_key": None,
        "members": clue["members"],
        "value_groups": [{
            "normalized_value": "None", "display_value": "no conflict",
            "members": [f"{m['memory_id']}@{m['version']}" for m in clue["members"]],
        }],
        "status": "not_a_conflict",
        "detector_version": clue["members"][0]["detector_version"],
        "prompt_version": None,
        "source": "scan", "reason": "reviewed",
        "workspace": "w",
        "authorized": True,
    })
    assert recorded["ok"] is True, recorded["data"]


def test_doctor_backlog_carries_triage_counters(vec_tools: MemoryTools) -> None:
    from memory_arbiter.doctor import run_all_checks

    tools = vec_tools
    _dismiss_pair(tools, _write(tools, "上限 10。"), _write(tools, "上限 99。"))

    with tools.db.connection() as conn:
        report = run_all_checks(conn, tools.settings)
    backlog = next(f for f in report.findings if f.check_id == "conflicts.backlog")
    assert backlog.evidence["not_a_conflict_total"] == 1
    assert backlog.evidence["not_a_conflict_this_week"] == 1
    assert backlog.evidence["latest_triage_at"], "latest triage time must surface"
    assert "dismissed 1 total" in backlog.detail


def test_console_overview_counts_dismissed(vec_tools: MemoryTools) -> None:
    from memory_arbiter.console_api import ConsoleAPI

    tools = vec_tools
    _dismiss_pair(tools, _write(tools, "上限 10。"), _write(tools, "上限 99。"))
    api = ConsoleAPI(tools)
    overview = api.overview()
    assert overview["counts"]["dismissed_conflicts"] == 1


# ── C5: page progress kv ──


def test_page_progress_records_per_group_and_completes(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    for i in range(3):
        _write(tools, f"apisvc 记录 {i}。", workspace="apisvc")
    for i in range(2):
        _write(tools, f"dbpgsql 记录 {i}。", workspace="dbpgsql")
    assert tools.wait_evidence_worker_drained(timeout=5)

    page = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 2, "k": 10,
    })
    state = tools.db.scan_page_progress_state()
    assert state is not None
    assert state["complete"] is False
    assert state["after"] == 0
    groups = state["groups"]
    assert set(groups) == {"apisvc"}, "page of 2 anchors covers apisvc only"
    assert groups["apisvc"]["anchors_scanned"] == 2
    assert groups["apisvc"]["last_anchor"] == 2

    # Continue to the end of the library.
    while True:
        data = page["data"]
        if data["next_anchor_memory_id"] is None:
            break
        page = tools.memory_repair("scan_candidates", {
            "anchor_memory_id": data["next_anchor_memory_id"], "batch": 10, "k": 10,
        })
    state = tools.db.scan_page_progress_state()
    assert state is not None
    assert state["complete"] is True
    assert set(state["groups"]) == {"apisvc", "dbpgsql"}
    assert state["groups"]["dbpgsql"]["anchors_scanned"] == 2


def test_page_progress_new_round_resets(vec_tools: MemoryTools) -> None:
    tools = vec_tools
    _write(tools, "one 记录。", workspace="apisvc")
    _write(tools, "two 记录。", workspace="dbpgsql")
    assert tools.wait_evidence_worker_drained(timeout=5)
    tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 10, "k": 10})
    state = tools.db.scan_page_progress_state()
    assert state is not None and state["complete"] is True
    # A new round (after=0) resets the tallies.
    tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 1, "k": 10})
    state = tools.db.scan_page_progress_state()
    assert state is not None
    assert state["complete"] is False
    assert len(state["groups"]) == 1


def test_doctor_reports_broken_chain_after_one_hour(vec_tools: MemoryTools) -> None:
    from datetime import datetime, timedelta, timezone

    from memory_arbiter.doctor import run_all_checks

    tools = vec_tools
    for i in range(3):
        _write(tools, f"apisvc 记录 {i}。", workspace="apisvc")
    assert tools.wait_evidence_worker_drained(timeout=5)
    tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 2, "k": 10})

    # Fresh incomplete progress: no alarm yet.
    with tools.db.connection() as conn:
        report = run_all_checks(conn, tools.settings)
    assert not any(f.check_id == "conflicts.scan_chain" for f in report.findings)

    # Age the record past an hour.
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    with tools.db.connection() as conn:
        state = json.loads(conn.execute(
            "SELECT value FROM migration_state WHERE key='scan_page_progress'"
        ).fetchone()[0])
        state["at"] = old
        for group in state["groups"].values():
            group["at"] = old
        conn.execute(
            "INSERT INTO migration_state(key,value,updated_at) VALUES('scan_page_progress',?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(state, ensure_ascii=False),),
        )
        conn.commit()
        report = run_all_checks(conn, tools.settings)
    chain = next(f for f in report.findings if f.check_id == "conflicts.scan_chain")
    assert chain.severity.value == "warning"
    assert f"anchor {state['next_anchor']}" in chain.detail
    assert chain.evidence["groups"]["apisvc"]["anchors_scanned"] == 2


def test_doctor_no_chain_alarm_when_complete(vec_tools: MemoryTools) -> None:
    from datetime import datetime, timedelta, timezone

    from memory_arbiter.doctor import run_all_checks

    tools = vec_tools
    _write(tools, "apisvc 记录。", workspace="apisvc")
    assert tools.wait_evidence_worker_drained(timeout=5)
    tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 10, "k": 10})
    state = tools.db.scan_page_progress_state()
    assert state is not None and state["complete"] is True
    # Even an old COMPLETE record is no broken chain (scan_stale owns that).
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    with tools.db.connection() as conn:
        state["at"] = old
        conn.execute(
            "INSERT INTO migration_state(key,value,updated_at) VALUES('scan_page_progress',?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(state, ensure_ascii=False),),
        )
        conn.commit()
        report = run_all_checks(conn, tools.settings)
    assert not any(f.check_id == "conflicts.scan_chain" for f in report.findings)
