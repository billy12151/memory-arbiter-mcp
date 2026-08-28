from __future__ import annotations

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
    monkeypatch.setattr(SchemaStore, "_ensure_evidence_vec_table", forbidden)
    monkeypatch.setattr(SchemaStore, "_ensure_workspace_vec_table", forbidden)
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
    settings = Settings(
        db_path=legacy,
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=False,
    )
    errors = _preflight(settings, legacy.with_name("t.vnext.db"))
    assert any("vec.enabled must be true" in e for e in errors)

    settings = Settings(
        db_path=legacy,
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        embedding_provider="none",
    )
    errors = _preflight(settings, legacy.with_name("t.vnext.db"))
    assert any("GGUF embedding model must be configured" in e for e in errors)

    settings = Settings(
        db_path=legacy,
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
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
