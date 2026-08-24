"""Minimal interactive upgrade wrapper around side-by-side final sync."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import Settings, _find_config_file
from .db_generation import detect_upgrade_source_generation
from .vnext_migration import final_sync, inspect


InputFunc = Callable[[str], str]


def _default_target(source: Path) -> Path:
    return source.with_name(f"{source.stem}.vnext{source.suffix}")


def _format_bytes(value: int) -> str:
    amount = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def _render_plan(plan: dict[str, Any]) -> str:
    conflict_only = plan.get("upgrade_mode") == "conflict_only"
    mode_lines = (
        [
            "The source evidence index is ready in the configured embedding space.",
            "Existing memory_evidence and vector tables can be cloned unchanged; no",
            "model loading or embedding recomputation is required for this upgrade.",
        ]
        if conflict_only else
        [
            "The database structure and evidence index must be rebuilt side by side.",
            "Reindex prerequisites: sqlite-vec, llama-cpp-python, and a configured local",
            "GGUF embedding model. Run with --dry-run first to verify them and disk space.",
        ]
    )
    return "\n".join([
        "Memory Arbiter upgrade",
        "",
        "This upgrade significantly improves long-document recall and conflict discovery.",
        *mode_lines,
        "STOP all MCP servers, consoles, workers, and other processes that can write",
        "to the source database for the entire migration. Otherwise the upgrade refuses",
        "to publish the target. The source database remains unchanged for rollback.",
        "WARNING: old conflict, decision, and semantic-notice records are permanently",
        "excluded from the new database; they remain only in the old database.",
        "Memories, memory history, workspace governance, and evidence are preserved.",
        "",
        f"Memories: {int((plan.get('counts') or {}).get('memories') or 0)}",
        f"Estimated evidence units: {int(plan.get('estimated_evidence_units') or 0)}",
        f"Estimated vector storage: {_format_bytes(int(plan.get('estimated_vector_bytes') or 0))}",
        f"Estimated additional space: {_format_bytes(int(plan.get('required_bytes') or 0))}",
        f"Free disk space: {_format_bytes(int(plan.get('free_bytes') or 0))}",
        f"Vector effect: {(plan.get('schema_migration') or {}).get('vector_effect', 'unknown')}",
        f"Vector compatibility: {plan.get('vector_compatibility', 'unknown')}",
        f"Source database: {plan.get('source')}",
        f"New database: {plan.get('target')}",
    ])


def _preflight(settings: Settings, target: Path) -> list[str]:
    errors: list[str] = []
    if not settings.enable_sqlite_vec:
        errors.append("vec.enabled must be true")
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        errors.append("sqlite-vec is not installed; install memory-arbiter-mcp[vec]")
    if settings.embedding_provider != "gguf" or settings.embedding_model_path is None:
        errors.append("a local GGUF embedding model must be configured")
    elif not settings.embedding_model_path.expanduser().is_file():
        errors.append(f"embedding model not found: {settings.embedding_model_path}")
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        errors.append(
            "llama-cpp-python is not installed; install "
            "memory-arbiter-mcp[semantic-local]"
        )
    if not target.parent.exists():
        errors.append(f"target directory does not exist: {target.parent}")
    elif not os.access(target.parent, os.W_OK):
        errors.append(f"target directory is not writable: {target.parent}")
    return errors


def _selected_config_path() -> Path | None:
    warnings: list[str] = []
    return _find_config_file(warnings)


def _read_config(config_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config root must be a JSON object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"config_read_failed: {exc}"
    return raw, None


def _configured_db_path(config_path: Path) -> Path | None:
    raw, _error = _read_config(config_path)
    value = raw.get("db_path") if raw is not None else None
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _switch_standard_config(config_path: Path, target: Path) -> dict[str, Any]:
    raw, error = _read_config(config_path)
    if raw is None:
        return {"switched": False, "error": error}

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_path.with_name(f"{config_path.name}.pre-upgrade-{timestamp}")
    suffix = 1
    while backup.exists():
        backup = config_path.with_name(
            f"{config_path.name}.pre-upgrade-{timestamp}-{suffix}"
        )
        suffix += 1
    temp = config_path.with_name(f"{config_path.name}.upgrade.tmp")
    try:
        shutil.copy2(config_path, backup)
        raw["db_path"] = str(target)
        temp.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(temp, config_path.stat().st_mode & 0o777)
        os.replace(temp, config_path)
        return {
            "switched": True,
            "config_path": str(config_path),
            "config_backup": str(backup),
            "db_path": str(target),
        }
    except OSError as exc:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        return {
            "switched": False,
            "error": f"config_switch_failed: {exc}",
            "config_path": str(config_path),
            "config_backup": str(backup) if backup.exists() else None,
        }


def run_upgrade(
    argv: Sequence[str] | None = None,
    *,
    input_func: InputFunc = input,
) -> int:
    parser = argparse.ArgumentParser(
        prog="mema upgrade",
        description=(
            "Rebuild a legacy database and evidence index into a separate current "
            "database, verify it, then optionally switch the standard config."
        ),
        epilog=(
            "Before execution, stop every MCP server, console, worker, or other writer "
            "using the source. The source is retained, but old conflict, decision, and "
            "semantic-notice records are not copied to the target. Schema migration "
            "declares whether vectors are preserved or rebuilt; a preserved incompatible "
            "space is disabled and repaired separately. A vector-rebuilding migration "
            "requires sqlite-vec, llama-cpp-python, and a configured local GGUF embedding "
            "model. Run "
            "--dry-run first."
        ),
    )
    parser.add_argument("--source", type=Path, help="Legacy source database (default: configured db_path).")
    parser.add_argument("--target", type=Path, help="Separate side-by-side target database (default: <source>.vnext<suffix>).")
    parser.add_argument("--dry-run", action="store_true", help="Check source, reindex prerequisites, target, and disk space; write nothing and do not switch config.")
    parser.add_argument("--yes", action="store_true", help="Confirm non-interactively that all writers are stopped and old conflict/decision/notice history may be omitted.")
    parser.add_argument("--no-switch", action="store_true", help="Build and verify the side-by-side target but leave config pointing at the source.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    args = parser.parse_args(list(argv or []))

    settings = Settings.from_env()
    source = (args.source or settings.db_path).expanduser().resolve()
    target = (args.target or _default_target(source)).expanduser().resolve()
    generation = detect_upgrade_source_generation(source)

    if generation == "current":
        result = {
            "ok": True,
            "already_current": True,
            "source": str(source),
            "message": "The configured database already uses local-text evidence storage.",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["message"])
        return 0
    if generation != "legacy":
        result = {
            "ok": False,
            "error": f"database_generation_{generation}",
            "source": str(source),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else f"Upgrade cannot continue: {result['error']}", file=sys.stderr)
        return 2
    if source == target:
        result = {"ok": False, "error": "target_must_differ_from_source"}
        print(json.dumps(result, indent=2) if args.json else result["error"], file=sys.stderr)
        return 2

    plan = inspect(source, target, settings)
    preflight_errors = (
        [] if plan.get("upgrade_mode") == "conflict_only"
        else _preflight(settings, target)
    )
    if preflight_errors:
        result = {
            "ok": False,
            "error": "upgrade_preflight_failed",
            "details": preflight_errors,
            "source": str(source),
            "target": str(target),
        }
        print(
            json.dumps(result, ensure_ascii=False, indent=2)
            if args.json else
            "Upgrade preflight failed:\n- " + "\n- ".join(preflight_errors),
            file=sys.stderr,
        )
        return 2

    if plan.get("ok") is False:
        error = plan.get("error", "inspect_failed")
        result = {"ok": False, "error": error, "plan": plan}
        print(
            json.dumps(result, ensure_ascii=False, indent=2) if args.json
            else f"Upgrade cannot continue: {error} (missing columns: {', '.join(plan.get('missing_columns') or [])})",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        result = {"ok": bool(plan.get("disk_ok")), "dry_run": True, "plan": plan}
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else _render_plan(plan))
        return 0 if result["ok"] else 2
    if not plan.get("disk_ok"):
        result = {"ok": False, "error": "insufficient_disk_space", "plan": plan}
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else _render_plan(plan), file=sys.stderr)
        return 2

    if args.json and not args.yes:
        result = {
            "ok": False,
            "error": "confirmation_required_use_yes",
            "plan": plan,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    if not args.json:
        print(_render_plan(plan))
    if not args.yes:
        answer = input_func(
            "\nConfirm every source-database writer is stopped and accept that old "
            "conflict, decision, and semantic-notice history will not exist in the "
            "side-by-side target? [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            result = {"ok": False, "cancelled": True, "source": str(source), "target": str(target)}
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("Upgrade cancelled. No data or configuration was changed.")
            return 1

    config_result: dict[str, Any]
    publish_callback: Callable[[], dict[str, Any]] | None = None
    if args.no_switch:
        config_result = {"switched": False, "reason": "no_switch_requested"}
    elif os.getenv("MEMORY_ARBITER_DB_PATH"):
        config_result = {
            "switched": False,
            "reason": "db_path_is_overridden_by_environment",
            "manual_action": f"Set MEMORY_ARBITER_DB_PATH={target}",
        }
    else:
        config_path = _selected_config_path()
        if config_path is None:
            config_result = {
                "switched": False,
                "reason": "no_json_config_found",
                "manual_action": f"Set db_path to {target}",
            }
        else:
            configured_source = _configured_db_path(config_path)
            if configured_source != source:
                config_result = {
                    "switched": False,
                    "reason": "config_db_path_does_not_match_source",
                    "config_path": str(config_path),
                    "configured_db_path": (
                        str(configured_source) if configured_source is not None else None
                    ),
                    "manual_action": f"Set db_path to {target}",
                }
            else:
                publish_callback = lambda: _switch_standard_config(config_path, target)
                config_result = {"switched": False, "reason": "migration_not_verified"}

    result = final_sync(
        source,
        target,
        settings,
        progress=not args.json,
        publish_callback=publish_callback,
    )
    returned_config = result.get("config")
    if isinstance(returned_config, dict):
        config_result = returned_config
    elif not result.get("ok") or not result.get("switch_ready"):
        config_result = {"switched": False, "reason": "migration_not_verified"}

    result["config"] = config_result
    result["old_database_kept"] = str(source)
    if result.get("ok") and not args.no_switch and not config_result.get("switched") and config_result.get("error"):
        result["ok"] = False
        result["error"] = "migration_complete_but_config_switch_failed"
    if result.get("ok"):
        result["next_step"] = "Restart the MCP client and run `mema doctor --json`."

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 2


def run_cli(argv: Sequence[str] | None = None) -> int:
    return run_upgrade(argv)
