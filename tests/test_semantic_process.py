from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from memory_arbiter.semantic_conflict import (
    IsolatedGGUFSemanticBackend,
    ModelSignal,
    WorkspaceCandidateSignal,
)


def _responsive_child(conn, config):
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "shutdown":
                return
            conn.send({
                "ok": True,
                "result": ModelSignal(True, "replacement", 0.9, "{}", {"pid": os.getpid()}),
            })
    except (EOFError, OSError):
        return


def _blocking_child(conn, config):
    try:
        request = conn.recv()
        if request.get("command") == "load":
            conn.send({"ok": True, "result": {"loaded": True}})
            conn.recv()
        time.sleep(10)
    except (EOFError, OSError):
        return


def _serial_probe_child(conn, config):
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "load":
                conn.send({"ok": True, "result": {"loaded": True}})
                continue
            time.sleep(0.08)
            conn.send({"ok": True, "result": ModelSignal(True, "replacement", 0.9, "{}", {"pid": os.getpid()})})
    except (EOFError, OSError):
        return


def _workspace_probe_child(conn, config):
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "load":
                conn.send({"ok": True, "result": {"loaded": True}})
                continue
            if request.get("command") == "classify_pair":
                time.sleep(10)
            conn.send({
                "ok": True,
                "result": WorkspaceCandidateSignal("canonical", "alias", 0.9, "same"),
            })
    except (EOFError, OSError):
        return


def _first_generation_blocks_child(conn, config):
    marker = Path(config["model_path"]).with_suffix(".started")
    first_generation = not marker.exists()
    marker.touch(exist_ok=True)
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "load":
                conn.send({"ok": True, "result": {"loaded": True}})
                continue
            if first_generation:
                time.sleep(10)
            conn.send({"ok": True, "result": ModelSignal(True, "replacement", 0.9, "{}", {"pid": os.getpid()})})
    except (EOFError, OSError):
        return


def _backend(tmp_path: Path, target, timeout=500) -> IsolatedGGUFSemanticBackend:
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    return IsolatedGGUFSemanticBackend(
        model, hard_timeout_ms=timeout, process_target=target,
    )


def test_process_backend_reuses_one_resident_child(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _responsive_child)
    first = backend.classify_pair({}, {})
    first_pid = first.parsed["pid"]
    second = backend.classify_pair({}, {})
    assert second.parsed["pid"] == first_pid
    assert backend.status()["max_concurrency"] == 1
    assert backend.unload(timeout=1)["ok"] is True


def test_process_backend_hard_timeout_terminates_child(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _blocking_child, timeout=100)
    result = backend.classify_pair({}, {})
    assert result.candidate is False
    assert "hard timeout" in (result.error or "")
    status = backend.status()
    assert status["model_state"] == "unloaded"
    assert status["timed_out_jobs"] == 1
    assert status["child_restarts"] == 1


def test_process_backend_concurrent_callers_are_serialized(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _serial_probe_child, timeout=1000)
    results = []
    started = time.monotonic()
    threads = [threading.Thread(target=lambda: results.append(backend.classify_pair({}, {}))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    elapsed = time.monotonic() - started
    assert len(results) == 2
    assert elapsed >= 0.14
    assert backend.status()["max_concurrency"] == 1
    backend.force_terminate()
    assert backend.status()["child_pid"] is None


def test_disable_closes_admission_before_unload_timeout(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _workspace_probe_child, timeout=5_000)
    started = threading.Event()

    def run_inference():
        started.set()
        backend.classify_pair({}, {})

    thread = threading.Thread(target=run_inference)
    thread.start()
    assert started.wait(1)
    deadline = time.monotonic() + 2
    while backend.status()["inflight"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)

    disabled = backend.unload(timeout=0.01, disable=True)
    assert disabled["timeout"] is True
    assert backend.status()["disabled"] is True

    suggest_started = time.monotonic()
    suggestion = backend.suggest_workspace_candidate("raw", {}, ["canonical"])
    assert time.monotonic() - suggest_started < 0.5
    assert suggestion.candidate is None
    assert suggestion.error == "semantic backend disabled"
    backend.force_terminate()
    thread.join(2)
    assert not thread.is_alive()


def test_process_backend_recovers_with_new_generation_after_timeout(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _first_generation_blocks_child, timeout=100)
    first = backend.classify_pair({}, {})
    assert first.candidate is False
    second = backend.classify_pair({}, {})
    assert second.candidate is True
    status = backend.status()
    assert status["generation"] == 2
    assert status["timed_out_jobs"] == 1
    backend.force_terminate()
