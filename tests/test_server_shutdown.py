from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from memory_arbiter import server


def test_terminate_after_shutdown_redelivers_signal(monkeypatch) -> None:
    events: list[object] = []

    monkeypatch.setattr(server.signal, "signal", lambda signum, action: events.append((signum, action)))
    monkeypatch.setattr(server.os, "getpid", lambda: 4321)

    def fake_kill(pid: int, signum: int) -> None:
        events.append(("kill", pid, signum))
        raise RuntimeError("test stop")

    monkeypatch.setattr(server.os, "kill", fake_kill)

    with pytest.raises(RuntimeError, match="test stop"):
        server._terminate_after_shutdown(signal.SIGTERM, lambda: events.append("shutdown"))

    assert events == [
        (signal.SIGTERM, signal.SIG_IGN),
        "shutdown",
        (signal.SIGTERM, signal.SIG_DFL),
        ("kill", 4321, signal.SIGTERM),
    ]


def test_stdio_server_sigterm_does_not_abort_during_finalization(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MEMORY_ARBITER_CONFIG": str(tmp_path / "missing-config.json"),
            "MEMORY_ARBITER_DB_PATH": str(tmp_path / "server.sqlite3"),
            "MEMORY_ARBITER_BACKUP_JSONL": str(tmp_path / "server.backup.jsonl"),
            "MEMORY_ARBITER_UPDATE_CHECK_ENABLED": "false",
            "MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED": "false",
            "MEMORY_ARBITER_ENABLE_SQLITE_VEC": "false",
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "memory_arbiter.server"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (tmp_path / "server.sqlite3").exists():
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        assert proc.poll() is None, proc.stderr.read() if proc.stderr else "server exited"

        proc.terminate()
        _stdout, stderr = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)

    assert proc.returncode == -signal.SIGTERM
    assert "Fatal Python error" not in stderr
    assert "_enter_buffered_busy" not in stderr
