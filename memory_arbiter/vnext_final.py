"""vnext final_sync（从 vnext_migration.py 搬出，拆分批 ⑦ 纯移动）。"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Callable

from .config import Settings


# 0.17.0 C6: memory_evidence is no longer preserved (unit channel retired);
# memory_row carries the derived rows instead and rebuilds like units did.
PRESERVED_TABLES = (
    "memories", "memory_history", "memory_row",
    "workspace_canonicals", "workspace_aliases", "backup_replay_log",
)
FULL_REBUILD_COPY_TABLES = tuple(
    table for table in PRESERVED_TABLES if table != "memory_row"
)
DESTRUCTIVELY_REBUILT_TABLES = (
    "conflicts", "conflict_judgments", "semantic_notices", "workspace_alias_events",
)


def final_sync(
    source: Path,
    target: Path,
    settings: Settings,
    *,
    progress: bool = True,
    publish_callback: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # 拆分批 ⑦：留守模块成员经函数内 import 引用（断环；build/_remove_sidecars 等留 vnext_migration）
    # 拆分批 ⑦：全部经 vnext_migration 取（re-export 面=测试 patch 面，call 时读模块属性）
    from .vnext_migration import (
        _fingerprint_on_connection,
        _target_owned_by_source,
        _remove_sidecars,
        build,
    )

    """Build verified staging and atomically replace the side-by-side target."""
    if target.exists() and not _target_owned_by_source(target, source):
        return {
            "ok": False,
            "error": "existing_target_not_owned_by_source",
            "source": str(source),
            "target": str(target),
        }
    staging = target.with_name(target.name + ".finalizing")
    if staging.exists():
        if not _target_owned_by_source(staging, source):
            return {
                "ok": False,
                "error": "existing_staging_not_owned_by_source",
                "source": str(source),
                "staging": str(staging),
            }
        staging.unlink()
    _remove_sidecars(staging)
    # Fail fast on an active source writer (B-D4): without this probe a
    # wedged old writer is only caught by the post-build fingerprint gate
    # below, after a full staging rebuild. The short busy_timeout keeps the
    # probe cheap; BEGIN EXCLUSIVE doubles as the write probe.
    if source.exists():
        probe = sqlite3.connect(source, timeout=1)
        try:
            probe.execute("PRAGMA busy_timeout=1000")
            probe.execute("BEGIN EXCLUSIVE")
            probe.execute("ROLLBACK")
        except sqlite3.OperationalError:
            return {
                "ok": False,
                "error": "source_has_active_writer",
                "source": str(source),
                "target": str(target),
                "next_step": (
                    "stop all writers on the source database and rerun the "
                    "final sync; no staging rebuild has started"
                ),
            }
        finally:
            probe.close()
    result = build(source, staging, settings, progress=progress)
    if not result.get("ok"):
        result["staging"] = str(staging)
        return result
    if not result.get("switch_ready"):
        result.update({
            "ok": False,
            "error": "staging_not_switch_ready",
            "staging": str(staging),
        })
        return result
    # Hold an EXCLUSIVE transaction on the source from final verification
    # through target publication and, when supplied by the upgrade wrapper,
    # the config switch. This deterministically excludes an already-open old
    # writer from landing a commit in the verification-to-switch gap.
    source_lock = sqlite3.connect(source, timeout=5)
    source_lock.row_factory = sqlite3.Row
    try:
        source_lock.execute("PRAGMA busy_timeout=5000")
        source_lock.execute("BEGIN EXCLUSIVE")
        locked_fingerprint = _fingerprint_on_connection(source_lock)
        if locked_fingerprint != result.get("source_fingerprint"):
            source_lock.execute("ROLLBACK")
            result.update({
                "ok": False,
                "error": "source_changed_during_final_sync",
                "staging": str(staging),
                "next_step": (
                    "stop all writers and rerun the final sync; the fully built "
                    "staging database is kept at the staging path and will be "
                    "rebuilt on the next run (it roughly doubles disk usage "
                    "until then)"
                ),
            })
            return result
        _remove_sidecars(target)
        os.replace(staging, target)
        _remove_sidecars(staging)
        # The staging build's startup-lock sidecar outlives the rename; the
        # switched-in database does not need one. Remove it here — staging is
        # already renamed away — so the config-switch failure branch below
        # cannot leave the orphan lock file behind either.
        staging_lock = staging.with_name(staging.name + ".startup.lock")
        if staging_lock.exists():
            staging_lock.unlink()
        if publish_callback is not None:
            try:
                publish_result = publish_callback()
            except Exception as exc:
                # A raising callback used to escape via the BaseException path
                # with no structured result; converge it into the same failure
                # branch as a declined switch so operators get the target path.
                publish_result = {"switched": False, "error": f"publish_callback_raised: {exc}"}
            if not isinstance(publish_result, dict):
                # A non-dict return (e.g. None) must not escape as an
                # AttributeError on .get below with the target already live;
                # converge it into the same structured failure branch.
                publish_result = {
                    "switched": False,
                    "error": f"publish_callback_returned_invalid: {type(publish_result).__name__}",
                }
            result["config"] = publish_result
            # Strict identity check: only a real boolean True confirms the
            # switch. A truthy string like "false" must take the failure
            # branch instead of committing a switch that never happened.
            if publish_result.get("switched") is not True:
                source_lock.execute("ROLLBACK")
                result.update({
                    "ok": False,
                    "error": "migration_complete_but_config_switch_failed",
                    # The target database is already live (os.replace above);
                    # only the config switch failed. Tell operators exactly
                    # where the new database is and that a switch is still due.
                    "target_ready": True,
                    "needs_config_switch": True,
                    "target": str(target),
                    # build() left "freeze writes and run --final-sync before
                    # switching db_path" in the result, which would send the
                    # operator through another hours-long rebuild even though
                    # only the config switch remains. Override it.
                    "next_step": (
                        "the target database is already live; only the config "
                        "switch failed — point db_path at the target manually "
                        "(or fix the config and retry the switch); no rebuild "
                        "is needed"
                    ),
                })
                return result
        source_lock.execute("COMMIT")
    except BaseException:
        if source_lock.in_transaction:
            source_lock.execute("ROLLBACK")
        raise
    finally:
        source_lock.close()
    result.update({
        "target": str(target),
        "final_sync": True,
        "next_step": (
            "verify the target in normal use, then switch db_path; "
            "keep the source for rollback"
        ),
    })
    return result
