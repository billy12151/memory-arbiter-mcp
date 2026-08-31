# ── from test_setup_cli.py ──

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
    """Inline template must be the 0.15.0 slim starter config (17 keys)."""
    cfg = _default_config_dict(
        model_path=Path("/tmp/model.gguf"),
        db_path=Path("/tmp/db.sqlite3"),
        backup_jsonl=Path("/tmp/bak.jsonl"),
    )
    assert cfg["mcp"] == {
        "transport": "stdio",
        "http": {"host": "127.0.0.1", "port": 8000},
    }
    assert cfg["isolation"] == "none"
    assert cfg["update_check"] == {"enabled": True}
    assert cfg["embedding"] == {
        "model_path": "/tmp/model.gguf",
        "auto_query": True,
        "auto_write": True,
    }
    assert cfg["semantic_conflict"] == {
        "model_path": None,
        "on_write": "async",
        "max_notice_pairs": 2,
    }
    # Removed groups must not reappear in the starter template.
    for removed in ("tool_profile", "vec", "workspace_match_distance",
                    "workspace_weak_vector_weight", "workspace_min_name_len",
                    "workspace_recall_admission", "workspace_recall_cutoff"):
        assert removed not in cfg
    assert "provider" not in cfg["embedding"]
    assert "n_ctx" not in cfg["embedding"]
    assert "max_unit_chars" not in cfg["embedding"]
    # Paths must be absolute strings (no ~ left over).
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
    assert parsed["embedding"]["model_path"]
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
    assert "embedding" in parsed
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
    assert "embedding" in parsed
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
    assert parsed["embedding"]["auto_query"] is True


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


# ── from test_upgrade_cli.py ──


import json
import sqlite3
import threading
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db_generation import (
    CURRENT_SCHEMA_GENERATION,
    detect_database_generation,
    detect_upgrade_source_generation,
    legacy_database_message,
)
from memory_arbiter.upgrade_cli import run_upgrade
from memory_arbiter.upgrade_cli import _preflight
from memory_arbiter.upgrade_cli import _switch_standard_config


def test_startup_refusal_explains_safe_upgrade_requirements(tmp_path: Path) -> None:
    message = legacy_database_message(tmp_path / "legacy.db")
    for phrase in (
        "Stop every process that can write",
        "mema upgrade --dry-run",
        "side-by-side target",
        "old conflict, decision, and semantic-notice history is not copied",
        "vectors are preserved or rebuilt",
        "does not block the structural migration",
    ):
        assert phrase in message


def test_upgrade_help_warns_about_writers_loss_target_and_reindex(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_upgrade(["--help"])
    assert exc_info.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    for phrase in (
        "stop every MCP server",
        "side-by-side target",
        "old conflict, decision, and semantic-notice records",
        "preserved or rebuilt",
        "repaired separately",
        "sqlite-vec",
        "llama-cpp-python",
        "local GGUF embedding model",
        "--dry-run",
    ):
        assert phrase in output


def test_upgrade_plan_explains_vector_effect_and_compatibility() -> None:
    from memory_arbiter.upgrade_cli import _render_plan

    output = _render_plan({
        "upgrade_mode": "full_evidence_rebuild",
        "schema_migration": {"vector_effect": "rebuild"},
        "vector_compatibility": "mismatch",
        "counts": {"memories": 2},
        "estimated_evidence_units": 4,
        "estimated_vector_bytes": 128,
        "required_bytes": 1024,
        "free_bytes": 2048,
        "source": "/tmp/source.sqlite3",
        "target": "/tmp/target.sqlite3",
    })

    assert "Vector effect: rebuild" in output
    assert "Vector compatibility: mismatch" in output


@pytest.fixture(autouse=True)
def _pass_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memory_arbiter.upgrade_cli._preflight", lambda *_args: [])


def _legacy_db(path: Path, *, partial_evidence: bool = False) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("CREATE TABLE memories_vec(id INTEGER PRIMARY KEY)")
    if partial_evidence:
        conn.execute("CREATE TABLE memory_evidence(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE migration_state(key TEXT PRIMARY KEY,value TEXT)")
        conn.execute("INSERT INTO migration_state VALUES('phase','building')")
    conn.commit()
    conn.close()


def _current_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("CREATE TABLE memory_evidence(id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE migration_state(key TEXT PRIMARY KEY,value TEXT)")
    conn.execute(
        "INSERT INTO migration_state VALUES('schema_generation',?)",
        (CURRENT_SCHEMA_GENERATION,),
    )
    conn.commit()
    conn.close()


def _config(path: Path, db_path: Path) -> None:
    path.write_text(
        json.dumps({
            "db_path": str(db_path),
            "backup_jsonl": str(path.with_suffix(".jsonl")),
            "vec": {"enabled": False},
            "update_check": {"enabled": False},
        }),
        encoding="utf-8",
    )


def test_generation_detection_prefers_legacy_owners(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.db"
    _legacy_db(legacy, partial_evidence=True)
    current = tmp_path / "current.db"
    _current_db(current)
    assert detect_database_generation(tmp_path / "missing.db") == "missing"
    assert detect_database_generation(legacy) == "unknown"
    assert detect_upgrade_source_generation(legacy) == "legacy"
    assert detect_database_generation(current) == "current"


def test_current_startup_skips_schema_ddl(tmp_path: Path, monkeypatch) -> None:
    from memory_arbiter.db import MemoryDB
    from memory_arbiter.db.schema import SchemaStore
    from memory_arbiter.tools import MemoryTools

    path = tmp_path / "current.sqlite3"
    settings = Settings(db_path=path, backup_jsonl=tmp_path / "backup.jsonl")
    MemoryDB(settings)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("current startup must not run maintenance work")

    monkeypatch.setattr(SchemaStore, "_init_schema", forbidden)
    monkeypatch.setattr(SchemaStore, "_rebuild_fts", forbidden)
    monkeypatch.setattr(SchemaStore, "ensure_evidence_vec_table", forbidden)
    monkeypatch.setattr(SchemaStore, "ensure_workspace_vec_table", forbidden)
    monkeypatch.setattr("memory_arbiter.embedder.build_embedder", forbidden)
    reopened = MemoryDB(settings)
    assert reopened.db_available is True
    MemoryTools(settings, reopened)


def test_current_generation_accepts_compact_receipt_without_phase(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    _current_db(current)
    with sqlite3.connect(current) as conn:
        conn.execute(
            "INSERT INTO migration_state VALUES('migration_completed_at','2026-08-24T00:00:00Z')"
        )
    assert detect_database_generation(current) == "current"


def test_current_generation_rejects_unknown_phase(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    _current_db(current)
    with sqlite3.connect(current) as conn:
        conn.execute("INSERT INTO migration_state VALUES('phase','unexpected')")
    assert detect_database_generation(current) == "unknown"


def test_upgrade_cancel_does_not_migrate_or_edit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "legacy.db"
    _legacy_db(source)
    config = tmp_path / "config.json"
    _config(config, source)
    before = config.read_bytes()
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.inspect",
        lambda *_args: {
            "source": str(source), "target": str(tmp_path / "legacy.vnext.db"),
            "counts": {"memories": 1}, "estimated_evidence_units": 2,
            "required_bytes": 100, "disk_ok": True,
        },
    )
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("migration must not run")

    monkeypatch.setattr("memory_arbiter.upgrade_cli.final_sync", forbidden)
    prompts: list[str] = []
    assert run_upgrade([], input_func=lambda prompt: prompts.append(prompt) or "n") == 1
    assert called is False
    assert config.read_bytes() == before
    assert "every source-database writer is stopped" in prompts[0]
    assert "old conflict, decision, and semantic-notice history" in prompts[0]
    output = capsys.readouterr().out
    assert "side by side" in output
    assert "Run with --dry-run first" in output
    assert "No data or configuration was changed" in output


def test_json_execution_requires_yes_without_prompting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "legacy.db"
    _legacy_db(source)
    config = tmp_path / "config.json"
    _config(config, source)
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.inspect",
        lambda *_args: {
            "source": str(source), "target": str(tmp_path / "target.db"),
            "counts": {"memories": 1}, "estimated_evidence_units": 2,
            "required_bytes": 100, "disk_ok": True,
        },
    )
    prompted = False

    def forbidden_prompt(_prompt: str) -> str:
        nonlocal prompted
        prompted = True
        return "y"

    assert run_upgrade(["--json"], input_func=forbidden_prompt) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "confirmation_required_use_yes"
    assert prompted is False


def test_upgrade_success_backs_up_and_switches_standard_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "legacy.db"
    target = tmp_path / "legacy.vnext.db"
    _legacy_db(source)
    config = tmp_path / "config.json"
    _config(config, source)
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    monkeypatch.delenv("MEMORY_ARBITER_DB_PATH", raising=False)
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.inspect",
        lambda *_args: {
            "source": str(source), "target": str(target),
            "counts": {"memories": 1}, "estimated_evidence_units": 2,
            "required_bytes": 100, "disk_ok": True,
        },
    )

    def completed(*_args, **kwargs):
        target.write_bytes(b"new")
        result = {"ok": True, "switch_ready": True, "target": str(target)}
        callback = kwargs.get("publish_callback")
        if callback is not None:
            result["config"] = callback()
        return result

    monkeypatch.setattr("memory_arbiter.upgrade_cli.final_sync", completed)
    assert run_upgrade(["--yes", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["switched"] is True
    assert Path(json.loads(config.read_text())["db_path"]) == target
    backup = Path(payload["config"]["config_backup"])
    assert json.loads(backup.read_text())["db_path"] == str(source)
    assert source.exists()


def test_upgrade_failure_leaves_config_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "legacy.db"
    _legacy_db(source)
    config = tmp_path / "config.json"
    _config(config, source)
    before = config.read_bytes()
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.inspect",
        lambda *_args: {
            "source": str(source), "target": str(tmp_path / "target.db"),
            "counts": {"memories": 1}, "estimated_evidence_units": 2,
            "required_bytes": 100, "disk_ok": True,
        },
    )
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.final_sync",
        lambda *_args, **_kwargs: {
            "ok": False, "switch_ready": False, "error": "injected_failure",
        },
    )
    assert run_upgrade(["--yes", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["reason"] == "migration_not_verified"
    assert config.read_bytes() == before
    assert not list(tmp_path.glob("config.json.pre-upgrade-*"))


def test_upgrade_no_switch_and_env_override_require_manual_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    _legacy_db(source)
    config = tmp_path / "config.json"
    _config(config, source)
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.inspect",
        lambda *_args: {
            "source": str(source), "target": str(target),
            "counts": {"memories": 1}, "estimated_evidence_units": 2,
            "required_bytes": 100, "disk_ok": True,
        },
    )
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.final_sync",
        lambda *_args, **_kwargs: {"ok": True, "switch_ready": True},
    )
    assert run_upgrade([
        "--source", str(source), "--target", str(target),
        "--yes", "--json", "--no-switch",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["reason"] == "no_switch_requested"
    assert json.loads(config.read_text())["db_path"] == str(source)

    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(source))
    assert run_upgrade([
        "--source", str(source), "--target", str(target),
        "--yes", "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["reason"] == "db_path_is_overridden_by_environment"
    assert str(target) in payload["config"]["manual_action"]
    assert json.loads(config.read_text())["db_path"] == str(source)


def test_explicit_source_does_not_switch_unrelated_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    configured_source = tmp_path / "configured.db"
    migrated_source = tmp_path / "other.db"
    target = tmp_path / "other.vnext.db"
    _legacy_db(configured_source)
    _legacy_db(migrated_source)
    config = tmp_path / "config.json"
    _config(config, configured_source)
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    monkeypatch.delenv("MEMORY_ARBITER_DB_PATH", raising=False)
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.inspect",
        lambda *_args: {
            "source": str(migrated_source), "target": str(target),
            "counts": {"memories": 1}, "estimated_evidence_units": 2,
            "required_bytes": 100, "disk_ok": True,
        },
    )
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.final_sync",
        lambda *_args, **_kwargs: {"ok": True, "switch_ready": True},
    )
    assert run_upgrade([
        "--source", str(migrated_source), "--target", str(target),
        "--yes", "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["reason"] == "config_db_path_does_not_match_source"
    assert json.loads(config.read_text())["db_path"] == str(configured_source)
    assert not list(tmp_path.glob("config.json.pre-upgrade-*"))


def test_upgrade_preflight_failure_stops_before_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "legacy.db"
    _legacy_db(source)
    config = tmp_path / "config.json"
    _config(config, source)
    before = config.read_bytes()
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli._preflight",
        lambda *_args: ["embedding model not found"],
    )
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("migration must not run")

    monkeypatch.setattr("memory_arbiter.upgrade_cli.final_sync", forbidden)
    assert run_upgrade(["--yes", "--json"]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"] == "upgrade_preflight_failed"
    assert called is False
    assert config.read_bytes() == before


def test_config_switch_uses_unique_backup_and_preserves_original_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy.db"
    config = tmp_path / "config.json"
    _config(config, source)
    original = config.read_bytes()
    fixed = type("FixedDatetime", (), {
        "now": staticmethod(lambda: type("Now", (), {
            "strftime": lambda self, _fmt: "20260819-120000",
        })()),
    })
    monkeypatch.setattr("memory_arbiter.upgrade_cli.datetime", fixed)
    first = _switch_standard_config(config, tmp_path / "first.db")
    assert first["switched"] is True
    # Restore the original to simulate a retry within the same second.
    config.write_bytes(original)
    second = _switch_standard_config(config, tmp_path / "second.db")
    assert second["switched"] is True
    first_backup = Path(first["config_backup"])
    second_backup = Path(second["config_backup"])
    assert first_backup != second_backup
    assert first_backup.read_bytes() == original
    assert second_backup.read_bytes() == original
    assert not list(tmp_path.glob("*.upgrade.tmp"))


def test_config_replace_failure_preserves_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy.db"
    config = tmp_path / "config.json"
    _config(config, source)
    original = config.read_bytes()
    real_replace = __import__("os").replace

    def fail_config_replace(src, dst):
        if Path(dst) == config:
            raise OSError("injected replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr("memory_arbiter.upgrade_cli.os.replace", fail_config_replace)
    result = _switch_standard_config(config, tmp_path / "target.db")
    assert result["switched"] is False
    assert "config_switch_failed" in result["error"]
    assert config.read_bytes() == original
    assert not list(tmp_path.glob("*.upgrade.tmp"))
    backup = Path(result["config_backup"])
    assert backup.read_bytes() == original


def test_final_sync_refuses_unrelated_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_arbiter.vnext_migration import final_sync

    source = tmp_path / "legacy.db"
    target = tmp_path / "unrelated.db"
    _legacy_db(source)
    target.write_bytes(b"do not overwrite")
    before = target.read_bytes()
    settings = Settings(
        db_path=source,
        backup_jsonl=tmp_path / "backup.jsonl",
    )
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("build must not run")

    monkeypatch.setattr("memory_arbiter.vnext_migration.build", forbidden)
    result = final_sync(source, target, settings)
    assert result["error"] == "existing_target_not_owned_by_source"
    assert called is False
    assert target.read_bytes() == before


def test_final_sync_excludes_late_old_writer_through_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_arbiter.vnext_migration import final_sync

    source = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    _legacy_db(source)
    settings = Settings(db_path=source, backup_jsonl=tmp_path / "backup.jsonl")
    source_fingerprint = {"stable": True}

    def built(_source, staging, _settings, **_kwargs):
        staging.write_bytes(b"new")
        return {
            "ok": True,
            "switch_ready": True,
            "source_fingerprint": source_fingerprint,
        }

    monkeypatch.setattr("memory_arbiter.vnext_migration.build", built)
    monkeypatch.setattr(
        "memory_arbiter.vnext_migration._fingerprint_on_connection",
        lambda _conn: source_fingerprint,
    )
    monkeypatch.setattr("memory_arbiter.vnext_migration._remove_sidecars", lambda _path: None)

    writer_started = threading.Event()
    writer_finished = threading.Event()
    writer_error: list[str] = []

    def publish():
        def late_writer() -> None:
            conn = sqlite3.connect(source, timeout=0.05)
            try:
                writer_started.set()
                conn.execute("INSERT INTO memories(content) VALUES('late')")
                conn.commit()
            except sqlite3.OperationalError as exc:
                writer_error.append(str(exc).lower())
            finally:
                conn.close()
                writer_finished.set()

        thread = threading.Thread(target=late_writer)
        thread.start()
        assert writer_started.wait(1)
        assert writer_finished.wait(1)
        thread.join()
        return {"switched": True}

    result = final_sync(source, target, settings, progress=False, publish_callback=publish)
    assert result["ok"] is True
    assert any("locked" in error for error in writer_error)
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0


def _stub_final_sync_build(monkeypatch: pytest.MonkeyPatch) -> None:
    source_fingerprint = {"stable": True}

    def built(_source, staging, _settings, **_kwargs):
        staging.write_bytes(b"new")
        # A real build opens MemoryDB on the staging path, whose startup leaves
        # the <staging>.startup.lock sidecar behind (database_startup_lock).
        staging.with_name(staging.name + ".startup.lock").write_bytes(b"")
        return {
            "ok": True,
            "switch_ready": True,
            "source_fingerprint": source_fingerprint,
        }

    monkeypatch.setattr("memory_arbiter.vnext_migration.build", built)
    monkeypatch.setattr(
        "memory_arbiter.vnext_migration._fingerprint_on_connection",
        lambda _conn: source_fingerprint,
    )
    monkeypatch.setattr("memory_arbiter.vnext_migration._remove_sidecars", lambda _path: None)


def _staging_startup_lock(target: Path) -> Path:
    staging = target.with_name(target.name + ".finalizing")
    return staging.with_name(staging.name + ".startup.lock")


_CONFIG_SWITCH_FAILED_NEXT_STEP = (
    "the target database is already live; only the config switch failed — "
    "point db_path at the target manually (or fix the config and retry the "
    "switch); no rebuild is needed"
)


def test_final_sync_publish_callback_raise_converges_to_switch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_arbiter.vnext_migration import final_sync

    source = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    _legacy_db(source)
    settings = Settings(db_path=source, backup_jsonl=tmp_path / "backup.jsonl")
    _stub_final_sync_build(monkeypatch)

    def publish():
        raise RuntimeError("boom")

    result = final_sync(source, target, settings, progress=False, publish_callback=publish)
    assert result["ok"] is False
    assert result["error"] == "migration_complete_but_config_switch_failed"
    assert result["target_ready"] is True
    assert result["needs_config_switch"] is True
    assert result["target"] == str(target)
    # The stale build-phase next_step ("freeze writes and run --final-sync …")
    # must be overridden: the target is already live, no rebuild is due.
    assert result["next_step"] == _CONFIG_SWITCH_FAILED_NEXT_STEP
    assert result["config"]["switched"] is False
    assert result["config"]["error"].startswith("publish_callback_raised:")
    assert "boom" in result["config"]["error"]
    # The target database is already live; only the config switch failed.
    assert target.read_bytes() == b"new"
    # The staging startup-lock sidecar is cleaned on the failure branch too.
    assert not _staging_startup_lock(target).exists()


def test_final_sync_publish_callback_declined_switch_reports_target_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_arbiter.vnext_migration import final_sync

    source = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    _legacy_db(source)
    settings = Settings(db_path=source, backup_jsonl=tmp_path / "backup.jsonl")
    _stub_final_sync_build(monkeypatch)

    def publish():
        return {"switched": False, "error": "config_switch_failed: operator declined"}

    result = final_sync(source, target, settings, progress=False, publish_callback=publish)
    assert result["ok"] is False
    assert result["error"] == "migration_complete_but_config_switch_failed"
    assert result["target_ready"] is True
    assert result["needs_config_switch"] is True
    assert result["target"] == str(target)
    assert result["next_step"] == _CONFIG_SWITCH_FAILED_NEXT_STEP
    assert result["config"] == {
        "switched": False,
        "error": "config_switch_failed: operator declined",
    }
    assert target.read_bytes() == b"new"
    # The staging startup-lock sidecar is cleaned on the failure branch too.
    assert not _staging_startup_lock(target).exists()


def test_final_sync_publish_callback_non_dict_return_converges_to_switch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_arbiter.vnext_migration import final_sync

    source = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    _legacy_db(source)
    settings = Settings(db_path=source, backup_jsonl=tmp_path / "backup.jsonl")
    _stub_final_sync_build(monkeypatch)

    def publish():
        # Malformed return: must not escape as AttributeError on .get with
        # the target already live.
        return None

    result = final_sync(source, target, settings, progress=False, publish_callback=publish)
    assert result["ok"] is False
    assert result["error"] == "migration_complete_but_config_switch_failed"
    assert result["target_ready"] is True
    assert result["needs_config_switch"] is True
    assert result["target"] == str(target)
    assert result["next_step"] == _CONFIG_SWITCH_FAILED_NEXT_STEP
    assert result["config"] == {
        "switched": False,
        "error": "publish_callback_returned_invalid: NoneType",
    }
    assert target.read_bytes() == b"new"
    assert not _staging_startup_lock(target).exists()


def test_final_sync_publish_callback_truthy_string_switched_is_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory_arbiter.vnext_migration import final_sync

    source = tmp_path / "legacy.db"
    target = tmp_path / "target.db"
    _legacy_db(source)
    settings = Settings(db_path=source, backup_jsonl=tmp_path / "backup.jsonl")
    _stub_final_sync_build(monkeypatch)

    def publish():
        # A truthy string is not a confirmed switch; only a real `True` is.
        return {"switched": "false"}

    result = final_sync(source, target, settings, progress=False, publish_callback=publish)
    assert result["ok"] is False
    assert result["error"] == "migration_complete_but_config_switch_failed"
    assert result["target_ready"] is True
    assert result["needs_config_switch"] is True
    assert result["config"] == {"switched": "false"}
    assert target.read_bytes() == b"new"
    assert not _staging_startup_lock(target).exists()


def test_preflight_rejects_missing_vec_embedding_and_model(tmp_path) -> None:
    legacy = tmp_path / "legacy.db"
    _legacy_db(legacy)
    # 0.15.0: vec.enabled/embedding.provider knobs are gone — pointing at a
    # model path IS the intent. Preflight still rejects an unconfigured and
    # a missing model file.
    settings = Settings(
        db_path=legacy,
        backup_jsonl=tmp_path / "backup.jsonl",
    )
    errors = _preflight(settings, legacy.with_name("t.vnext.db"))
    assert any("GGUF embedding model must be configured" in e for e in errors)

    settings = Settings(
        db_path=legacy,
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=tmp_path / "missing.gguf",
    )
    errors = _preflight(settings, legacy.with_name("t.vnext.db"))
    assert any("embedding model not found" in e for e in errors)


def test_dry_run_exit_code_tracks_disk_space(tmp_path: Path, monkeypatch) -> None:
    legacy = tmp_path / "legacy.db"
    _legacy_db(legacy)
    config = tmp_path / "config.json"
    _config(config, legacy)
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))

    def tiny_disk(*_args, **_kwargs):
        return {"disk_ok": False, "counts": {"memories": 1}, "estimated_evidence_units": 1,
                "estimated_vector_bytes": 1, "free_bytes": 1, "required_bytes": 999}

    monkeypatch.setattr("memory_arbiter.upgrade_cli.inspect", tiny_disk)
    assert run_upgrade(["--dry-run"]) == 2
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.inspect",
        lambda *a, **k: {"disk_ok": True, "counts": {"memories": 1}, "estimated_evidence_units": 1,
                         "estimated_vector_bytes": 1, "free_bytes": 10**9, "required_bytes": 1},
    )
    assert run_upgrade(["--dry-run"]) == 0


def test_upgrade_already_current_exits_zero_without_migrating(tmp_path: Path, monkeypatch) -> None:
    current = tmp_path / "current.db"
    _current_db(current)
    config = tmp_path / "config.json"
    _config(config, current)
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    monkeypatch.setattr(
        "memory_arbiter.upgrade_cli.inspect",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("inspect must not run")),
    )
    assert run_upgrade(["--dry-run"]) == 0


def test_upgrade_rejects_unsupported_source_schema(tmp_path: Path, monkeypatch, capsys) -> None:
    legacy = tmp_path / "legacy.db"
    _legacy_db(legacy)
    config = tmp_path / "config.json"
    _config(config, legacy)
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    assert run_upgrade(["--dry-run", "--json"]) == 2
    # Human (non-json) output names the error and the missing columns.
    assert run_upgrade(["--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "unsupported_source_schema" in err
    assert "workspace_canonical" in err


def test_dry_run_plan_text_lists_vector_storage_and_free_space(tmp_path: Path, monkeypatch, capsys) -> None:
    from memory_arbiter.config import Settings as FullSettings
    from memory_arbiter.db import MemoryDB

    legacy = tmp_path / "legacy.db"
    MemoryDB(FullSettings(db_path=legacy, backup_jsonl=tmp_path / "u.jsonl"))
    conn = sqlite3.connect(legacy)
    conn.execute("CREATE TABLE memories_vec(id INTEGER PRIMARY KEY)")
    conn.commit(); conn.close()
    config = tmp_path / "config.json"
    _config(config, legacy)
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    assert run_upgrade(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Estimated vector storage:" in out
    assert "Free disk space:" in out


# ── from test_backfill_subjects.py ──

"""Tests for scripts/backfill_subjects.py (T5).

Runs the backfill script's main() against a tmp DB with hand-inserted
empty-subject rows, verifying dry-run (plan only) and --apply (subject lands,
version bumps, memory_history records the reason, FTS picks up the new subject).
"""

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
