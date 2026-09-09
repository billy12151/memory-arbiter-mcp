"""Tests for the 0.15.11 install-execution work (P0a/P0b/P1).

- setup_cli --install: resumable downloader, mirror fallback, size gate,
  config write-back with the qwen model path (all network/pip mocked).
- Degraded-capability banner: persistent per-response warning, gated on
  config_file_loaded, honouring the deliberate semantic opt-out.
- First-call onboarding health card.

Does NOT touch the network, does NOT install real packages.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from memory_arbiter import setup_cli
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    for key in list(os.environ.keys()):
        if key.startswith("MEMORY_ARBITER_"):
            monkeypatch.delenv(key, raising=False)
    return fake_home


# ── downloader ─────────────────────────────────────────────────────────────

_BODY = bytes(range(256)) * 64  # 16 KiB of deterministic content


def _fake_urlopen(calls: list[tuple[str, str | None]], *, honor_range: bool = True, fail: bool = False):
    def fake(request, timeout: int = 0):
        calls.append((request.full_url, request.headers.get("Range")))
        if fail:
            raise OSError("connection refused")
        range_header = request.headers.get("Range")
        status = 200
        data = _BODY
        if honor_range and range_header:
            start = int(range_header.split("=", 1)[1].rstrip("-"))
            data = _BODY[start:]
            status = 206
        response = io.BytesIO(data)
        response.status = status  # type: ignore[attr-defined]
        return response
    return fake


def test_download_resumes_partial_file(monkeypatch, tmp_path):
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(setup_cli.urllib.request, "urlopen", _fake_urlopen(calls))
    part = tmp_path / "model.gguf.part"
    part.write_bytes(_BODY[:100])
    assert setup_cli._download_with_resume("https://example/m.gguf", part, log=lambda _msg: None)
    assert calls == [("https://example/m.gguf", "bytes=100-")]
    assert part.read_bytes() == _BODY


def test_download_restarts_when_server_ignores_range(monkeypatch, tmp_path):
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        setup_cli.urllib.request, "urlopen", _fake_urlopen(calls, honor_range=False),
    )
    part = tmp_path / "model.gguf.part"
    part.write_bytes(b"garbage-partial")
    assert setup_cli._download_with_resume("https://example/m.gguf", part, log=lambda _msg: None)
    assert part.read_bytes() == _BODY  # not garbage + body


def test_download_failure_keeps_part_for_resume(monkeypatch, tmp_path):
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(setup_cli.urllib.request, "urlopen", _fake_urlopen(calls, fail=True))
    part = tmp_path / "model.gguf.part"
    part.write_bytes(_BODY[:50])
    assert not setup_cli._download_with_resume("https://example/m.gguf", part, log=lambda _msg: None)
    assert part.read_bytes() == _BODY[:50]  # partial kept for the next resume


def test_install_model_skips_existing_good_file(monkeypatch, tmp_path):
    dest = tmp_path / "model.gguf"
    dest.write_bytes(_BODY)
    monkeypatch.setattr(
        setup_cli.urllib.request, "urlopen",
        lambda request, timeout=0: (_ for _ in ()).throw(AssertionError("no network expected")),
    )
    assert setup_cli._install_model(
        "m", dest, ["https://a", "https://b"], expected_bytes=len(_BODY), log=lambda _msg: None,
    )


def test_install_model_falls_back_to_second_mirror(monkeypatch, tmp_path):
    dest = tmp_path / "model.gguf"
    calls: list[str] = []

    def urlopen(request, timeout: int = 0):
        calls.append(request.full_url)
        if "first" in request.full_url:
            raise OSError("mirror down")
        response = io.BytesIO(_BODY)
        response.status = 200  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(setup_cli.urllib.request, "urlopen", urlopen)
    assert setup_cli._install_model(
        "m", dest, ["https://first/m", "https://second/m"],
        expected_bytes=len(_BODY), log=lambda _msg: None,
    )
    assert calls == ["https://first/m", "https://second/m"]
    assert dest.read_bytes() == _BODY
    assert not list(tmp_path.glob("*.part"))  # winning part renamed, stale cleaned


def test_install_model_never_resumes_across_mirrors(monkeypatch, tmp_path):
    """A truncated first mirror must not have its bytes continued by the
    second: different content under the same size would pass the size gate as
    a corrupt hybrid (observed live with HF→ModelScope)."""
    dest = tmp_path / "model.gguf"
    ranges_seen: list[str | None] = []

    def urlopen(request, timeout: int = 0):
        ranges_seen.append(request.headers.get("Range"))
        body = _BODY[: len(_BODY) // 4] if "first" in request.full_url else _BODY
        response = io.BytesIO(body)
        response.status = 200  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(setup_cli.urllib.request, "urlopen", urlopen)
    assert setup_cli._install_model(
        "m", dest, ["https://first/m", "https://second/m"],
        expected_bytes=len(_BODY), log=lambda _msg: None,
    )
    assert ranges_seen == [None, None]  # second mirror started from zero
    assert dest.read_bytes() == _BODY


def test_install_model_resumes_same_url_on_rerun(monkeypatch, tmp_path):
    dest = tmp_path / "model.gguf"
    truncated = len(_BODY) // 4

    def urlopen_truncated(request, timeout: int = 0):
        response = io.BytesIO(_BODY[:truncated])
        response.status = 200  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(setup_cli.urllib.request, "urlopen", urlopen_truncated)
    # First run: connection died at a quarter → size gate rejects, .part stays.
    assert not setup_cli._install_model(
        "m", dest, ["https://a/m"], expected_bytes=len(_BODY), log=lambda _msg: None,
    )
    part = setup_cli._part_path_for(dest, "https://a/m")
    assert part.read_bytes() == _BODY[:truncated]

    # Second run resumes that same URL from the kept partial.
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(setup_cli.urllib.request, "urlopen", _fake_urlopen(calls))
    assert setup_cli._install_model(
        "m", dest, ["https://a/m"], expected_bytes=len(_BODY), log=lambda _msg: None,
    )
    assert calls == [("https://a/m", f"bytes={truncated}-")]
    assert dest.read_bytes() == _BODY


def test_install_model_rejects_wrong_size(monkeypatch, tmp_path):
    dest = tmp_path / "model.gguf"
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(setup_cli.urllib.request, "urlopen", _fake_urlopen(calls))
    assert not setup_cli._install_model(
        "m", dest, ["https://a/m", "https://b/m"],
        expected_bytes=len(_BODY) * 4,  # way outside the ±20% tolerance
        log=lambda _msg: None,
    )
    assert not dest.exists()
    assert len(calls) == 2  # both mirrors tried


# ── --install end-to-end (pip + downloads mocked) ───────────────────────────

def test_install_mode_writes_qwen_path_into_config(isolated_env, monkeypatch):
    installed: list[str] = []
    monkeypatch.setattr(setup_cli, "_check_sqlite_vec", lambda: True)
    monkeypatch.setattr(setup_cli, "_check_llama_cpp", lambda: True)
    def fake_install(label, dest, urls, *, expected_bytes, log=print):
        installed.append(label)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-model")
        return True

    monkeypatch.setattr(setup_cli, "_install_model", fake_install)
    rc = setup_cli.run_cli(["--install"])
    config_path = isolated_env / ".config" / "memory-arbiter" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["semantic_conflict"]["model_path"].endswith(setup_cli.QWEN_MODEL_FILENAME)
    assert installed == ["embedding 模型", "Qwen 语义模型"]
    assert rc == 0  # mocked checks all pass → ready


def test_guidance_mode_keeps_qwen_informational(isolated_env, monkeypatch, capsys):
    monkeypatch.setattr(setup_cli, "_check_sqlite_vec", lambda: True)
    monkeypatch.setattr(setup_cli, "_check_llama_cpp", lambda: True)
    # No embedding model on disk → exit 1, but a missing qwen alone must not
    # change readiness in guidance mode.
    rc = setup_cli.run_cli([])
    out = capsys.readouterr().out
    assert "Qwen 语义模型: 未找到" in out
    assert "属可选能力" in out
    config = json.loads(
        (isolated_env / ".config" / "memory-arbiter" / "config.json").read_text(encoding="utf-8"),
    )
    assert config["semantic_conflict"]["model_path"] is None
    assert rc == 1  # embedding model missing; qwen did not add to the failure


def test_pyenv_guidance_for_unsupported_python(monkeypatch, capsys):
    class _FakeVersion:
        major, minor, micro = 3, 13, 1

    monkeypatch.setattr(setup_cli.sys, "version_info", _FakeVersion())
    lines, _all_ok = setup_cli._render_check_step(
        {
            "sqlite_vec": True, "llama_cpp": False,
            "model_exists": True, "model_size_ok": True, "model_size_bytes": 1,
            "qwen_exists": True, "qwen_size_bytes": 1,
            "config_load_ok": True, "config_load_error": "", "config_warnings": [],
        },
        Path("m.gguf"), Path("q.gguf"), use_color=False,
    )
    text = "\n".join(lines)
    assert "pyenv install 3.12" in text
    assert "pyenv local 3.12" in text


# ── degraded-capability banner + health card ────────────────────────────────

def _make_tools(tmp_path: Path, **settings_kwargs) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    settings = Settings(
        db_path=tmp_path / "b.sqlite3",
        backup_jsonl=tmp_path / "b.jsonl",
        **settings_kwargs,
    )
    db = MemoryDB(settings)
    return MemoryTools(settings=settings, db=db)


def test_banner_fires_for_real_config_with_missing_models(tmp_path):
    tools = _make_tools(tmp_path, config_file_loaded=True)
    text = "\n".join(tools.db.state.warnings)
    assert "降级模式" in text
    assert "向量召回未启用" in text
    assert "冲突检测未启用" in text
    assert "mema setup --install" in text
    # The banner persists on every response envelope.
    response = tools.db.state.response({})
    assert response["degraded"] is True
    assert any("降级模式" in warning for warning in response["warnings"])


def test_banner_silent_for_directly_constructed_settings(tmp_path):
    tools = _make_tools(tmp_path)  # config_file_loaded defaults False
    assert not any("降级模式" in warning for warning in tools.db.state.warnings)


def test_banner_honours_deliberate_semantic_opt_out(tmp_path):
    tools = _make_tools(
        tmp_path,
        config_file_loaded=True,
        embedding_model_path=tmp_path / "m.gguf",  # missing file → embedding line stays
        semantic_conflict_enabled=False,
        semantic_conflict_model_path=tmp_path / "q.gguf",  # configured → deliberate opt-out
    )
    text = "\n".join(tools.db.state.warnings)
    assert "向量召回未启用" in text
    assert "冲突检测未启用" not in text


def test_banner_quiet_when_install_is_full(tmp_path):
    embedding = tmp_path / "m.gguf"
    embedding.write_bytes(b"fake")
    qwen = tmp_path / "q.gguf"
    qwen.write_bytes(b"fake")
    tools = _make_tools(
        tmp_path,
        config_file_loaded=True,
        embedding_model_path=embedding,
        semantic_conflict_enabled=True,
        semantic_conflict_model_path=qwen,
    )
    assert not any("降级模式" in warning for warning in tools.db.state.warnings)


def test_setup_health_states(tmp_path):
    tools = _make_tools(tmp_path, config_file_loaded=True)
    health = tools._setup_health()
    assert health["embedding_model"] == "missing"
    assert health["semantic_model"] == "missing"
    assert "hint" in health and "setup --install" in health["hint"]

    qwen = tmp_path / "q.gguf"
    qwen.write_bytes(b"fake")
    opted_out = _make_tools(
        tmp_path, semantic_conflict_enabled=False, semantic_conflict_model_path=qwen,
    )
    assert opted_out._setup_health()["semantic_model"] == "disabled"


def test_onboarding_notice_carries_health_card(tmp_path):
    from memory_arbiter.update_monitor import UpdateMonitor

    tools = _make_tools(tmp_path, config_file_loaded=True)
    tools.start_update_monitor(UpdateMonitor(
        enabled=False, state_path=tmp_path / "update_state.json",
    ))
    notices = [n for n in tools._consume_notices() if n.get("type") == "agent_onboarding"]
    assert len(notices) == 1
    health = notices[0]["health"]
    assert health["embedding_model"] == "missing"
    assert "hint" in health
    # Second call: onboarding already delivered → no duplicate health card.
    assert [n for n in tools._consume_notices() if n.get("type") == "agent_onboarding"] == []
