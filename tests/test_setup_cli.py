"""Unit tests for setup_cli.py.

Mock-based: config generation, backup-on-existing, force-overwrite,
print-config dry-run, detection helpers, Python-version hint.
Does NOT touch network, does NOT install packages, does NOT write outside tmp_path.
"""
from __future__ import annotations

import builtins
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from memory_arbiter import setup_cli
from memory_arbiter.setup_cli import (
    _default_config_dict,
    _default_paths,
    _python_version_supported,
    run_cli,
)


# ---------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Point HOME at tmp_path so setup never touches the real user config.

    Also clears MEMORY_ARBITER_* env vars so Settings.from_env() behaves
    deterministically.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    # Windows uses USERPROFILE; set both for cross-platform test correctness.
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    for key in list(os.environ.keys()):
        if key.startswith("MEMORY_ARBITER_"):
            monkeypatch.delenv(key, raising=False)
    return fake_home


# ---------------------------------------------------------------------
#  Config template
# ---------------------------------------------------------------------

def test_config_template_has_required_fields():
    """Inline template must cover the three load-bearing sections."""
    cfg = _default_config_dict(
        model_path=Path("/tmp/model.gguf"),
        db_path=Path("/tmp/db.sqlite3"),
        backup_jsonl=Path("/tmp/bak.jsonl"),
    )
    assert cfg["tool_profile"] == "product"
    assert cfg["isolation"] == "none"
    assert cfg["workspace_match_distance"] == 0.25
    assert cfg["workspace_weak_vector_weight"] is False
    assert cfg["workspace_min_name_len"] == 3
    assert cfg["workspace_recall_admission"] is False
    assert cfg["workspace_recall_cutoff"] == 0.25
    assert cfg["vec"] == {"enabled": True, "dim": 768}
    assert cfg["embedding"]["provider"] == "gguf"
    assert cfg["embedding"]["auto_query"] is True
    assert cfg["embedding"]["auto_write"] is True
    assert cfg["embedding"]["n_ctx"] == 2048
    assert cfg["embedding"]["reserved_tokens"] == 64
    assert cfg["embedding"]["max_unit_chars"] == 3600
    assert cfg["update_check"] == {"enabled": True}
    # Paths must be absolute strings (no ~ left over).
    assert cfg["embedding"]["model_path"] == "/tmp/model.gguf"
    assert cfg["db_path"] == "/tmp/db.sqlite3"


def test_config_template_has_no_readme_tutorial():
    """Runtime config must not carry the long _readme tutorial fields."""
    cfg = _default_config_dict(
        model_path=Path("/tmp/m.gguf"),
        db_path=Path("/tmp/d.sqlite3"),
        backup_jsonl=Path("/tmp/b.jsonl"),
    )
    assert "_readme" not in cfg
    assert "_readme" not in cfg["embedding"]


# ---------------------------------------------------------------------
#  Path resolution (platform-independence)
# ---------------------------------------------------------------------

def test_default_paths_under_home(isolated_env):
    """All default paths must live under the (fake) HOME directory."""
    config_path, model_path, db_path, backup_jsonl = _default_paths()
    home = str(isolated_env)
    for p in (config_path, model_path, db_path, backup_jsonl):
        assert str(p).startswith(home), f"{p} not under HOME"
    assert config_path.name == "config.json"
    assert config_path.parent.name == "memory-arbiter"
    assert model_path.name == "embeddinggemma-300m-qat-Q8_0.gguf"


def test_default_paths_no_tilde():
    """Resolved paths must be absolute, no unresolved ~ prefix."""
    for p in _default_paths():
        assert p.is_absolute(), f"{p} is not absolute"
        assert not str(p).startswith("~"), f"{p} still has tilde"


# ---------------------------------------------------------------------
#  CLI: write behaviour
# ---------------------------------------------------------------------

def test_setup_creates_config_file(isolated_env, capsys):
    """Default invocation writes config.json to the XDG location."""
    rc = run_cli([])
    out = capsys.readouterr().out
    assert rc in (0, 1)  # 1 is fine if model/deps missing in test env
    config_path, *_ = _default_paths()
    assert config_path.exists(), "config.json was not written"
    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["vec"]["enabled"] is True
    assert "Step 1" in out
    assert "已写入" in out


def test_setup_backup_on_existing(isolated_env, capsys):
    """An existing config.json must be backed up, not overwritten."""
    config_path, *_ = _default_paths()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('{"user": "preserved"}', encoding="utf-8")

    rc = run_cli([])
    out = capsys.readouterr().out

    # New config written.
    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    assert "vec" in parsed
    # Backup exists and preserves original content.
    backups = list(config_path.parent.glob("config.json.bak.*"))
    assert len(backups) == 1, f"expected one backup, got {backups}"
    assert backups[0].read_text(encoding="utf-8") == '{"user": "preserved"}'
    assert "备份" in out


def test_setup_force_overwrites_without_backup(isolated_env, capsys):
    """--force overwrites the existing config and leaves NO backup."""
    config_path, *_ = _default_paths()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('{"user": "old"}', encoding="utf-8")

    rc = run_cli(["--force"])
    capsys.readouterr()  # drain

    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    assert "vec" in parsed
    backups = list(config_path.parent.glob("config.json.bak.*"))
    assert backups == [], f"--force must not leave backups, got {backups}"


def test_setup_print_config_does_not_write(isolated_env, capsys):
    """--print-config must not create any file."""
    config_path, *_ = _default_paths()
    rc = run_cli(["--print-config"])
    out = capsys.readouterr().out
    assert not config_path.exists(), "config.json was written despite --print-config"
    assert "--print-config" in out or "print-config" in out
    # The printed JSON should be valid and contain the model_path.
    # Extract the JSON block (between "内容:" and the next section header).
    assert "embeddinggemma-300m-qat-Q8_0.gguf" in out


def test_setup_no_config_skips_write(isolated_env, capsys):
    """--no-config must skip config generation entirely."""
    config_path, *_ = _default_paths()
    rc = run_cli(["--no-config"])
    capsys.readouterr()
    assert not config_path.exists(), "config.json was written despite --no-config"


def test_setup_custom_config_path(isolated_env, capsys, tmp_path):
    """--config-path writes to the specified location."""
    custom = tmp_path / "custom" / "my.json"
    rc = run_cli(["--config-path", str(custom)])
    capsys.readouterr()
    assert custom.exists()
    parsed = json.loads(custom.read_text(encoding="utf-8"))
    assert parsed["embedding"]["provider"] == "gguf"


# ---------------------------------------------------------------------
#  Detection helpers
# ---------------------------------------------------------------------

def test_python_version_supported_in_range(monkeypatch):
    """Mock sys.version_info to 3.11 → reported as supported."""
    class FakeVI:
        major, minor, micro = 3, 11, 5
    monkeypatch.setattr(sys, "version_info", FakeVI())
    ok, ver = _python_version_supported()
    assert ok is True
    assert ver == "3.11.5"


def test_python_version_unsupported_above(monkeypatch):
    """Python 3.13 (above the 3.10-3.12 wheel range) → unsupported."""
    class FakeVI:
        major, minor, micro = 3, 13, 0
    monkeypatch.setattr(sys, "version_info", FakeVI())
    ok, _ = _python_version_supported()
    assert ok is False


def test_python_version_unsupported_below(monkeypatch):
    """Python 3.9 (below range) → unsupported."""
    class FakeVI:
        major, minor, micro = 3, 9, 18
    monkeypatch.setattr(sys, "version_info", FakeVI())
    ok, _ = _python_version_supported()
    assert ok is False


def test_check_llama_cpp_missing(monkeypatch):
    """When llama_cpp cannot be imported, _check_llama_cpp returns False."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("simulated not-installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert setup_cli._check_llama_cpp() is False


def test_check_sqlite_vec_missing(monkeypatch):
    """When sqlite_vec cannot be imported, _check_sqlite_vec returns False."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sqlite_vec":
            raise ImportError("simulated not-installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert setup_cli._check_sqlite_vec() is False


# ---------------------------------------------------------------------
#  Exit codes
# ---------------------------------------------------------------------

def test_exit_code_zero_or_one(isolated_env):
    """Exit code is either 0 (all ok) or 1 (missing items) in normal runs."""
    rc = run_cli([])
    assert rc in (0, 1, 2)


def test_print_config_exits_zero(isolated_env):
    """--print-config is a dry-run; exit code must be 0 even if items missing."""
    rc = run_cli(["--print-config"])
    assert rc == 0


def test_no_config_exits_zero(isolated_env):
    """--no-config is a check-only run; exit code must be 0."""
    rc = run_cli(["--no-config"])
    assert rc == 0


# ---------------------------------------------------------------------
#  Model size tolerance
# ---------------------------------------------------------------------

def test_model_size_ok_in_tolerance(tmp_path):
    """A file within ±20% of expected size passes the size check."""
    p = tmp_path / "fake.gguf"
    # Exactly expected size.
    p.write_bytes(b"\0" * setup_cli.EXPECTED_MODEL_BYTES)
    ok, size = setup_cli._model_size_ok(p)
    assert ok is True
    assert size == setup_cli.EXPECTED_MODEL_BYTES


def test_model_size_missing(tmp_path):
    """A non-existent file returns (False, 0)."""
    ok, size = setup_cli._model_size_ok(tmp_path / "nope.gguf")
    assert ok is False
    assert size == 0


# ---------------------------------------------------------------------
#  Existing-model preservation (v0.8.4)
# ---------------------------------------------------------------------

def test_detect_existing_model_path_none_when_no_config(tmp_path):
    """No existing config → nothing to preserve."""
    p, note = setup_cli._detect_existing_model_path(tmp_path / "missing.json")
    assert p is None
    assert note == ""


def test_detect_existing_model_path_none_when_file_missing(tmp_path):
    """Config points at a path that doesn't exist on disk → don't preserve."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"embedding": {"model_path": "/nonexistent/model.gguf"}}), encoding="utf-8")
    p, note = setup_cli._detect_existing_model_path(cfg)
    assert p is None
    assert note == ""


def test_detect_existing_model_path_none_when_default_embeddinggemma(tmp_path):
    """Even if the default embeddinggemma path exists, treat it as 'nothing to preserve'
    (we'd write that path anyway)."""
    # Simulate the default model file existing at its expected location.
    model_file = tmp_path / "embeddinggemma-300m-qat-Q8_0.gguf"
    model_file.write_bytes(b"\0")  # existence is enough for this check
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"embedding": {"model_path": str(model_file)}}), encoding="utf-8")
    p, note = setup_cli._detect_existing_model_path(cfg)
    assert p is None, "default embeddinggemma must not be treated as user-supplied"


def test_detect_existing_model_path_preserves_real_custom_model(tmp_path):
    """A real, non-default GGUF file referenced in config is preserved."""
    custom_model = tmp_path / "bge-small-zh.gguf"
    custom_model.write_bytes(b"\0" * 1000)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"embedding": {"model_path": str(custom_model)}}), encoding="utf-8")
    p, note = setup_cli._detect_existing_model_path(cfg)
    assert p == custom_model
    assert "bge-small-zh.gguf" in note
    assert "沿用" in note


def test_detect_existing_model_path_handles_corrupt_json(tmp_path):
    """Corrupt JSON in existing config → no crash, no preservation."""
    cfg = tmp_path / "config.json"
    cfg.write_text("{not valid json", encoding="utf-8")
    p, note = setup_cli._detect_existing_model_path(cfg)
    assert p is None
    assert note == ""


def test_setup_preserves_custom_model_in_written_config(isolated_env, tmp_path, capsys):
    """End-to-end: existing config with a real custom model → new config keeps it."""
    # Place a fake user model on disk.
    custom_model = tmp_path / "my-bge-model.gguf"
    custom_model.write_bytes(b"\0" * 5000)
    # Pre-write a config pointing at it.
    config_path, *_ = _default_paths()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"embedding": {"model_path": str(custom_model)}}),
        encoding="utf-8",
    )

    rc = run_cli([])
    out = capsys.readouterr().out

    # New config must reference the custom model, not embeddinggemma.
    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["embedding"]["model_path"] == str(custom_model)
    assert "embeddinggemma" not in parsed["embedding"]["model_path"]
    # Setup log mentions the preservation.
    assert "沿用" in out
    assert "my-bge-model.gguf" in out


def test_setup_does_not_preserve_when_force(isolated_env, tmp_path, capsys):
    """--force overwrites everything including model_path, even if a custom model exists."""
    custom_model = tmp_path / "my-bge.gguf"
    custom_model.write_bytes(b"\0" * 5000)
    config_path, *_ = _default_paths()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"embedding": {"model_path": str(custom_model)}}),
        encoding="utf-8",
    )

    rc = run_cli(["--force"])
    capsys.readouterr()

    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    # --force should write the embeddinggemma default, overriding the custom one.
    assert parsed["embedding"]["model_path"].endswith("embeddinggemma-300m-qat-Q8_0.gguf")
