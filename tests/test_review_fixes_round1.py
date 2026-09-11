# ── Adversarial-review round (0.15.12 baseline) regression tests ──
"""P0/P1 fixes from the v0.15.11 two-round review, re-verified against 0.15.12.

Grouped by the review item they lock in:
  - config root-type validation (P0-1): a non-object config.json must
    degrade to env fallback with a warning, never crash startup.
  - update-check fetcher byte cap (P0-2): an oversized response must
    raise a structured error the monitor records, not feed a daemon
    thread an unbounded body.
"""

from __future__ import annotations

import json
from pathlib import Path

from memory_arbiter.config import Settings, load_config_file
from memory_arbiter.update_monitor import MAX_FETCH_BYTES, _default_fetcher

from tests.test_config import clear_config_env


class TestConfigRootTypeValidation:
    def test_array_root_degrades_to_env_with_warning(self, tmp_path: Path, monkeypatch) -> None:
        clear_config_env(monkeypatch)
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
        monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg))
        monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "env.sqlite3"))

        settings = Settings.from_env()

        assert settings.db_path == tmp_path / "env.sqlite3"
        assert any(
            "root must be a JSON object" in warning and "list" in warning
            for warning in settings.config_warnings
        )

    def test_string_root_degrades_to_env_with_warning(self, tmp_path: Path, monkeypatch) -> None:
        clear_config_env(monkeypatch)
        cfg = tmp_path / "config.json"
        cfg.write_text('"oops"', encoding="utf-8")
        monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg))
        monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "env.sqlite3"))

        settings = Settings.from_env()

        assert settings.db_path == tmp_path / "env.sqlite3"
        assert any(
            "root must be a JSON object" in warning and "str" in warning
            for warning in settings.config_warnings
        )

    def test_empty_object_root_still_ok(self, tmp_path: Path) -> None:
        warnings: list[str] = []
        cfg = tmp_path / "config.json"
        cfg.write_text("{}", encoding="utf-8")

        data = load_config_file(cfg, warnings)

        assert data == {}
        assert warnings == []

    def test_valid_object_root_untouched(self, tmp_path: Path) -> None:
        warnings: list[str] = []
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"db_path": "/tmp/x.sqlite3"}), encoding="utf-8")

        data = load_config_file(cfg, warnings)

        assert data == {"db_path": "/tmp/x.sqlite3"}
        assert warnings == []


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, limit: int = -1) -> bytes:
        if limit < 0:
            return self._body
        return self._body[:limit]

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class TestFetcherByteCap:
    def test_oversized_body_raises_value_error(self, monkeypatch) -> None:
        big = b"x" * (MAX_FETCH_BYTES + 1)

        def fake_urlopen(req, timeout):  # noqa: ANN001
            return _FakeResponse(big)

        monkeypatch.setattr("memory_arbiter.update_monitor.urllib.request.urlopen", fake_urlopen)
        try:
            _default_fetcher("https://pypi.org/pypi/memory-arbiter-mcp/json", 1.0)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "exceeded" in str(exc)

    def test_body_at_cap_passes_through(self, monkeypatch) -> None:
        body = b'{"info": {"version": "0.15.12"}}'

        def fake_urlopen(req, timeout):  # noqa: ANN001
            return _FakeResponse(body)

        monkeypatch.setattr("memory_arbiter.update_monitor.urllib.request.urlopen", fake_urlopen)
        assert _default_fetcher("https://pypi.org/pypi/memory-arbiter-mcp/json", 1.0) == body.decode("utf-8")
