from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory_arbiter.degrade import DegradeState
from memory_arbiter.update_monitor import UpdateMonitor, compare_versions


FIXED_NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


def _write_state(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_compare_versions_handles_project_versions() -> None:
    assert compare_versions("0.9.5", "0.9.6") < 0
    assert compare_versions("0.10.0", "0.9.9") > 0
    assert compare_versions("0.9.5", "0.9.5") == 0


def test_update_notice_uses_cache_and_suppresses_for_seven_days(tmp_path: Path) -> None:
    state_path = tmp_path / "update_state.json"
    _write_state(
        state_path,
        {
            "installed_version_seen": "0.9.5",
            "latest_version": "0.9.6",
            "latest_checked_at": FIXED_NOW.isoformat(),
        },
    )
    monitor = UpdateMonitor(
        enabled=True,
        state_path=state_path,
        current_version="0.9.5",
        now_func=lambda: FIXED_NOW,
    )

    notices = monitor.consume_notices()
    assert [n["type"] for n in notices] == ["update_available"]

    assert monitor.consume_notices() == []

    later = FIXED_NOW + timedelta(days=8)
    monitor_later = UpdateMonitor(
        enabled=True,
        state_path=state_path,
        current_version="0.9.5",
        now_func=lambda: later,
    )
    notices = monitor_later.consume_notices()
    assert [n["type"] for n in notices] == ["update_available"]


def test_update_notice_stops_after_current_version_catches_up(tmp_path: Path) -> None:
    state_path = tmp_path / "update_state.json"
    _write_state(
        state_path,
        {
            "installed_version_seen": "0.9.5",
            "latest_version": "0.9.6",
            "latest_checked_at": FIXED_NOW.isoformat(),
            "last_update_notified_version": "0.9.6",
            "last_update_notified_at": FIXED_NOW.isoformat(),
            "last_doctor_run_version": "0.9.6",
            "last_doctor_run_at": FIXED_NOW.isoformat(),
        },
    )
    monitor = UpdateMonitor(
        enabled=True,
        state_path=state_path,
        current_version="0.9.6",
        now_func=lambda: FIXED_NOW,
    )

    assert monitor.consume_notices() == []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["installed_version_seen"] == "0.9.6"
    assert state["last_update_notified_version"] is None


def test_post_upgrade_doctor_notice_suppressed_until_doctor_runs(tmp_path: Path) -> None:
    state_path = tmp_path / "update_state.json"
    _write_state(state_path, {"installed_version_seen": "0.9.5"})
    monitor = UpdateMonitor(
        enabled=True,
        state_path=state_path,
        current_version="0.9.6",
        now_func=lambda: FIXED_NOW,
    )

    notices = monitor.consume_notices()
    assert [n["type"] for n in notices] == ["post_upgrade_doctor_recommended"]
    assert monitor.consume_notices() == []

    later = FIXED_NOW + timedelta(days=8)
    monitor_later = UpdateMonitor(
        enabled=True,
        state_path=state_path,
        current_version="0.9.6",
        now_func=lambda: later,
    )
    assert [n["type"] for n in monitor_later.consume_notices()] == ["post_upgrade_doctor_recommended"]

    monitor_later.record_doctor_run()
    assert monitor_later.consume_notices() == []


def test_background_check_is_one_shot_and_exits(tmp_path: Path) -> None:
    state_path = tmp_path / "update_state.json"

    def fetcher(_url: str, _timeout: float) -> str:
        return json.dumps({"info": {"version": "0.9.6"}})

    monitor = UpdateMonitor(
        enabled=True,
        state_path=state_path,
        current_version="0.9.5",
        fetcher=fetcher,
        now_func=lambda: FIXED_NOW,
    )
    monitor.maybe_start_check_if_due()
    thread = monitor._check_thread
    assert thread is not None
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert monitor._check_thread is None
    assert monitor.update_status()["latest_version"] == "0.9.6"


def test_background_check_preserves_concurrent_doctor_run_state(tmp_path: Path) -> None:
    state_path = tmp_path / "update_state.json"
    _write_state(state_path, {"installed_version_seen": "0.9.6"})

    def fetcher(_url: str, _timeout: float) -> str:
        other = UpdateMonitor(
            enabled=True,
            state_path=state_path,
            current_version="0.9.6",
            now_func=lambda: FIXED_NOW,
        )
        other.record_doctor_run()
        return json.dumps({"info": {"version": "0.9.7"}})

    monitor = UpdateMonitor(
        enabled=True,
        state_path=state_path,
        current_version="0.9.6",
        fetcher=fetcher,
        now_func=lambda: FIXED_NOW,
    )
    monitor.maybe_start_check_if_due()
    assert monitor._check_thread is not None
    monitor._check_thread.join(timeout=2)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["latest_version"] == "0.9.7"
    assert state["last_doctor_run_version"] == "0.9.6"


def test_init_version_observation_preserves_existing_doctor_run_state(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "update_state.json"
    _write_state(
        state_path,
        {
            "installed_version_seen": "0.9.5",
            "last_doctor_run_version": "0.9.6",
            "last_doctor_run_at": FIXED_NOW.isoformat(),
        },
    )
    original_load = UpdateMonitor._load_state
    calls = {"count": 0}

    def load_state(self):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"installed_version_seen": "0.9.5"}
        return original_load(self)

    monkeypatch.setattr(UpdateMonitor, "_load_state", load_state)
    UpdateMonitor(
        enabled=True,
        state_path=state_path,
        current_version="0.9.6",
        now_func=lambda: FIXED_NOW,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["installed_version_seen"] == "0.9.6"
    assert state["last_doctor_run_version"] == "0.9.6"
    assert state["last_doctor_run_at"] == FIXED_NOW.isoformat()


def test_unwritable_state_path_does_not_raise(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-dir"
    parent_file.write_text("occupied", encoding="utf-8")
    state_path = parent_file / "update_state.json"

    monitor = UpdateMonitor(enabled=True, state_path=state_path, current_version="0.9.5")
    monitor.record_doctor_run()

    status = monitor.update_status()
    assert status["current_version"] == "0.9.5"


def test_response_adds_notices_only_for_success() -> None:
    state = DegradeState(mode="sqlite_vec")
    state.notice_provider = lambda: [{"type": "update_available"}]

    ok_resp = state.response({"x": 1})
    assert ok_resp["notices"] == [{"type": "update_available"}]

    error_resp = state.response({"error": "bad"}, ok=False)
    assert "notices" not in error_resp
