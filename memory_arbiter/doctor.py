"""memory-arbiter doctor: read-only health diagnostics (design doc v1.4).

Three-layer architecture (design doc §5):
  1. ``check_xxx(conn, settings, deep, runtime_state, ...)`` — single check,
     depends only on a read-only sqlite connection + Settings (+ optional
     runtime state / embedder probe). Never on ``MemoryDB``.
  2. ``run_all_checks(conn, settings, deep, runtime_state, embedder_probe)``
     — orchestration: runs all 25 findings on one ro connection (consistent
     snapshot), with per-check try/except isolation (§9 constraint 4).
  3. Platform entries ``doctor_overview_mcp`` (used by tools.py) and the CLI
     entry (doctor_cli.py) — each owns connection acquisition + the global
     ``except Exception`` fallback (§11.1).

All checks are read-only: SELECT + reading Settings/DegradeState fields.
Never calls any authorized write tool, never changes schema.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from .config import Settings
from .degrade import DegradeState
from .models import utc_now_iso
from . import __version__


# =====================================================================
#  Data model (design doc §6)
# =====================================================================

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


@dataclass
class Finding:
    check_id: str
    dimension: str          # "config" | "vector" | "split" | "consistency" | "capacity"
    severity: Severity
    status: str             # "pass" | "fail" | "warn" | "n/a" | "error"
    title: str
    detail: str
    evidence: dict = field(default_factory=dict)
    fix_hint: str = ""
    doc_link: str = ""


@dataclass
class OverviewReport:
    snapshot_ts: str
    overall: Severity
    findings: list[Finding]
    summary: dict


# =====================================================================
#  Helpers
# =====================================================================

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _scalar(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
    """Run a single-value SELECT and return the first column (or None)."""
    row = conn.execute(sql, tuple(params)).fetchone()
    if row is None:
        return None
    return row[0]


def _max_severity(findings: list[Finding]) -> Severity:
    if not findings:
        return Severity.INFO
    return max((f.severity for f in findings), key=lambda s: _SEVERITY_RANK[s])


def _na(check_id: str, dimension: str, reason: str) -> Finding:
    return Finding(
        check_id=check_id, dimension=dimension, severity=Severity.INFO,
        status="n/a", title=f"{check_id}: 不适用",
        detail=reason, evidence={},
    )


def _read_scan_log_last_completed(path) -> Optional[dict]:
    """Read the last ``status=completed`` line from ``scan_log.jsonl``.

    Tolerant: missing file -> None; malformed lines skipped; no completed
    line -> None. Pure diagnostic, never raises.
    """
    import json as _json
    try:
        if not path.exists():
            return None
        last_completed: Optional[dict] = None
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(rec, dict) and rec.get("status") == "completed":
                    last_completed = rec
        return last_completed
    except OSError:
        return None


def _days_since_iso(iso_ts: str) -> Optional[int]:
    """Whole days between an ISO-8601 timestamp and now. None on parse failure."""
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - dt).days)
    except (ValueError, TypeError):
        return None


def _read_attention_log_counts(path, since_days: int = 7) -> dict:
    """Read ``attention_log.jsonl`` and count events within the last
    ``since_days`` days (v0.8.8). Best-effort: missing/corrupt -> empty.

    Counts along two independent axes because write and search fire with
    very different frequency characteristics (write = event, low-freq;
    search = query, the same conflict pair re-fires on every retrieval),
    so a single ``total`` would let high-freq search drown write's signal.
    """
    by_source: dict[str, int] = {}
    by_trigger: dict[str, int] = {}
    by_source_trigger: dict[str, dict[str, int]] = {}
    total = 0
    p = Path(path)
    if not p.exists():
        return {"total": 0, "window_days": since_days, "by_source": {},
                "by_trigger": {}, "by_source_trigger": {}}
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(rec, dict):
                    continue
                ts = rec.get("ts")
                days = _days_since_iso(ts) if isinstance(ts, str) else None
                if days is None or days > since_days:
                    continue
                src = str(rec.get("source", "unknown"))
                trig = str(rec.get("trigger", "unknown"))
                by_source[src] = by_source.get(src, 0) + 1
                by_trigger[trig] = by_trigger.get(trig, 0) + 1
                sub = by_source_trigger.setdefault(src, {})
                sub[trig] = sub.get(trig, 0) + 1
                total += 1
    except OSError:
        pass
    return {"total": total, "window_days": since_days, "by_source": by_source,
            "by_trigger": by_trigger, "by_source_trigger": by_source_trigger}


# Check implementations live in doctor_checks/all_checks.py.  Re-export the
# private names here so existing tests/monkeypatches such as
# ``monkeypatch.setattr(memory_arbiter.doctor, "_check_db_size", ...)`` keep
# affecting run_all_checks (R4/R9).
from .doctor_checks.all_checks import (  # noqa: E402
    _check_attention_volume,
    _check_structured_claims,
    _shallow_gguf_probe,
    _embedder_shallow_probe,
    _check_vector_chain,
    _check_semantic_chain,
    _check_config_warnings,
    _check_db_writable,
    _check_degradation_mode,
    _check_split_capability,
    _check_split_backlog,
    _check_split_failed,
    _check_split_legacy_declined,
    _check_split_legacy_unknown_status,
    _check_split_index_integrity,
    _check_vec_index_state,
    _vec_state_from_meta,
    _check_orphan_sections,
    _check_orphan_vectors,
    _check_vec_parent_status_sync,
    _check_section_vec_coverage,
    _check_history_version_chain,
    _open_conflicts_by_source,
    _check_conflicts_open,
    _check_superseded_ratio,
    _check_history_bloat,
    _check_db_size,
    _check_workspace_alias_health,
)

def run_all_checks(
    conn: sqlite3.Connection,
    settings: Settings,
    deep: bool = False,
    runtime_state: Optional[DegradeState] = None,
    embedder_probe: Optional[Callable[[], tuple[Any, list[str]]]] = None,
    inflight_ids: Optional[set[int]] = None,
) -> OverviewReport:
    """Run all findings on one ro connection (consistent snapshot).

    Per-check try/except isolation (§9 constraint 4): a single check raising
    does not abort the others — it degrades to one ``status="error"`` finding.
    The global connection-level fallback lives in the platform entries (§11.1).
    """
    findings: list[Finding] = []

    def _run(check_id: str, fn: Callable[[], Optional[Finding]], dimension: str) -> None:
        try:
            f = fn()
            if f is not None:
                findings.append(f)
        except Exception as exc:
            findings.append(Finding(
                check_id=check_id, dimension=dimension, severity=Severity.WARNING,
                status="error", title=f"{check_id}: 检查异常（已隔离）",
                detail=f"该 check 抛异常并被隔离，不影响其余检查：{type(exc).__name__}: {exc}",
                evidence={"error_class": type(exc).__name__, "error": str(exc)},
            ))

    # --- config (3) ---
    _run("config.warnings", lambda: _check_config_warnings(settings), "config")
    _run("config.db_writable",
         lambda: _check_db_writable(settings, runtime_state), "config")
    vec_table_exists = _table_exists(conn, "memories_vec")
    fts_table_exists = _table_exists(conn, "memories_fts")
    _run("config.degradation_mode",
         lambda: _check_degradation_mode(runtime_state, vec_table_exists, fts_table_exists),
         "config")

    # --- vector chain (5) ---
    vec_state: dict = {}
    try:
        if _table_exists(conn, "_vec_index_meta"):
            vec_state = {str(r["key"]): str(r["value"])
                         for r in conn.execute("SELECT key, value FROM _vec_index_meta")}
    except sqlite3.Error:
        vec_state = {}

    def _vec_chain() -> Finding:
        # This returns multiple findings; append directly and return None.
        findings.extend(_check_vector_chain(
            conn, settings, deep, runtime_state, embedder_probe, vec_state,
            vec_table_exists=vec_table_exists))
        return None
    _run("vec.chain", _vec_chain, "vector")

    # Did link3 pass? (needed for vec_index_state gating)
    link3_passed = any(
        f.check_id == "vec.link3.extension_loaded" and f.status == "pass" for f in findings
    )

    # --- semantic conflict / Qwen chain (4) ---
    def _semantic_chain() -> Finding:
        findings.extend(_check_semantic_chain(settings))
        return None
    _run("semantic.chain", _semantic_chain, "semantic")

    # --- split (6) — v0.8 capability/backlog/failed/legacy/integrity ---
    _run("split.capability", lambda: _check_split_capability(conn, settings), "split")
    _run("split.long_unsplit_backlog", lambda: _check_split_backlog(conn, settings, inflight_ids), "split")
    _run("split.failed_count", lambda: _check_split_failed(conn, settings), "split")
    _run("split.legacy_declined", lambda: _check_split_legacy_declined(conn, settings), "split")
    _run("split.legacy_unknown_status", lambda: _check_split_legacy_unknown_status(conn, settings), "split")
    _run("split.index_integrity", lambda: _check_split_index_integrity(conn, settings), "split")

    # --- consistency (5) ---
    _run("consistency.vec_index_state",
         lambda: _check_vec_index_state(conn, link3_passed), "consistency")
    _run("consistency.orphan_sections", lambda: _check_orphan_sections(conn), "consistency")
    _run("consistency.orphan_vectors", lambda: _check_orphan_vectors(conn), "consistency")
    _run("consistency.vec_parent_status_sync",
         lambda: _check_vec_parent_status_sync(conn), "consistency")
    _run("consistency.section_vec_coverage", lambda: _check_section_vec_coverage(conn), "consistency")
    _run("consistency.history_version_chain",
         lambda: _check_history_version_chain(conn), "consistency")
    _run("consistency.structured_claims", lambda: _check_structured_claims(conn), "consistency")

    # --- capacity (5) ---
    _run("capacity.conflicts_open",
         lambda: _check_conflicts_open(conn, settings, runtime_state), "capacity")
    _run("capacity.superseded_ratio", lambda: _check_superseded_ratio(conn), "capacity")
    _run("capacity.history_bloat", lambda: _check_history_bloat(conn), "capacity")
    _run("capacity.db_size", lambda: _check_db_size(conn), "capacity")
    _run("capacity.attention_volume", lambda: _check_attention_volume(settings), "capacity")
    _run("capacity.workspace_alias_health",
         lambda: _check_workspace_alias_health(conn, settings), "capacity")

    overall = _max_severity(findings)
    vec_pass_count = sum(1 for f in findings if f.dimension == "vector" and f.status == "pass")
    total_memories = 0
    try:
        total_memories = _scalar(conn, "SELECT count(*) FROM memories") or 0
    except sqlite3.Error:
        pass
    # vec_effective requires BOTH (a) all 5 chain links pass (capability ready:
    # model configured, vec.enabled, extension loaded, model usable, auto on)
    # AND (b) the memories_vec table actually exists (data ready: the DB has
    # been initialized for vector recall). A DB can have the env configured
    # but never have built the vec table (e.g. config added after the DB was
    # created) — in that case semantic recall is NOT actually working, so
    # vec_effective must be False even though every link passes.
    vec_effective = vec_pass_count == 5 and vec_table_exists
    # `mode` must be grounded in the actual DB state, not just the MCP
    # process's startup-time probe (runtime_state.mode). The runtime mode is
    # set once at MemoryDB init and goes stale if the vec table is later
    # dropped or the DB is swapped — so when runtime says sqlite_vec but no
    # vec table exists, downgrade to what the tables actually support. This
    # keeps `mode` consistent with `vec_effective` (no vec table → both agree
    # semantic recall is off).
    if runtime_state is not None:
        mode = runtime_state.mode
        if mode == "sqlite_vec" and not vec_table_exists:
            mode = "fts5" if fts_table_exists else "like"
    else:
        mode = "sqlite_vec" if vec_table_exists else ("fts5" if fts_table_exists else "like")
    summary = {
        "mode": mode,
        "total_memories": total_memories,
        "vec_effective": vec_effective,
        # v0.8: split capability is derived from vec readiness, not a toggle.
        "split_capability_available": vec_effective,
    }
    return OverviewReport(
        snapshot_ts=utc_now_iso(), overall=overall, findings=findings, summary=summary,
    )


# =====================================================================
#  Platform entries (design doc §5 layer 3, §11.1)
# =====================================================================

def build_unopenable_report(settings: Settings, exc: Exception) -> OverviewReport:
    """Minimal critical report when the DB cannot be opened (§11.1).

    Distinguishes two failure modes so the fix_hint is actionable:
      - file does NOT exist at the resolved path → most likely a path/config
        mismatch (doctor resolved the wrong DB, e.g. no config.json and cwd
        has no DB); point the user at --db / config.json.
      - file exists but won't open → corruption / locked / read-only FS;
        point at recovery.
    """
    db_path = settings.db_path
    file_exists = os.path.exists(str(db_path))
    if not file_exists:
        title = "找不到数据库文件，doctor 降级为最小报告"
        detail = (f"解析到的 db_path 不存在：{db_path}（{type(exc).__name__}: {exc}）。"
                  "25 项 check 均未执行。最常见原因：未配置 ~/.config/memory-arbiter/config.json，"
                  "doctor 默认找当前目录下的 memory_arbiter.sqlite3，而你的库在别处。")
        fix_hint = ("用 --db 指定库路径，或确认 ~/.config/memory-arbiter/config.json 的 db_path "
                    "指向你的库（常见位置：~/.local/share/memory-arbiter/memory.sqlite3）。")
    else:
        title = "数据库无法打开，doctor 降级为最小报告"
        detail = (f"连接失败：{exc}。25 项 check 均未执行。"
                  "多数 jsonl_backup 是只读文件系统（文件可读仅不可写），mode=ro 能正常打开 → "
                  "若本应能打开却失败，通常是文件损坏/丢失/locked。")
        fix_hint = "检查 DB 文件权限、是否被独占锁定；可从 backup_jsonl 恢复。"
    return OverviewReport(
        snapshot_ts=utc_now_iso(),
        overall=Severity.CRITICAL,
        summary={"mode": "unopenable", "db_path": str(db_path), "file_exists": file_exists},
        findings=[Finding(
            check_id="db.unopenable", dimension="config", severity=Severity.CRITICAL,
            status="fail", title=title, detail=detail,
            evidence={"error": str(exc), "error_class": type(exc).__name__,
                      "db_path": str(db_path), "file_exists": file_exists},
            fix_hint=fix_hint,
        )],
    )


def doctor_overview_mcp(
    db: Any,
    settings: Settings,
    deep: bool = False,
    embedder_probe: Optional[Callable[[], tuple[Any, list[str]]]] = None,
    runtime_state: Optional[DegradeState] = None,
    inflight_ids: Optional[set[int]] = None,
) -> OverviewReport:
    """MCP platform entry: uses MemoryDB.diagnostic_connection() + global fallback."""
    try:
        with db.diagnostic_connection() as conn:
            return run_all_checks(
                conn, settings, deep,
                runtime_state=runtime_state if runtime_state is not None else getattr(db, "state", None),
                embedder_probe=embedder_probe,
                inflight_ids=inflight_ids,
            )
    except Exception as exc:
        return build_unopenable_report(settings, exc)


@contextmanager
def open_ro_connection(db_path: Path):
    """CLI ro connection context manager: mode=ro + load sqlite-vec if possible (§11.1).

    ``@contextmanager`` turns this generator into a proper context manager
    (a bare ``yield`` in a plain function would not support ``with``).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        import sqlite_vec  # type: ignore
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        pass  # vec not loadable; vec-related checks will return n/a
    try:
        yield conn
    finally:
        conn.close()


def doctor_overview_cli(db_path: Path, settings: Settings, deep: bool = False) -> OverviewReport:
    """CLI platform entry: own ro connection + global fallback (no MemoryDB)."""
    try:
        with open_ro_connection(db_path) as conn:
            return run_all_checks(conn, settings, deep, runtime_state=None,
                                  embedder_probe=None)
    except Exception as exc:
        return build_unopenable_report(settings, exc)


def _fix_metadata_for(finding: Finding) -> dict[str, Any]:
    base = {
        "fix_kind": "none",
        "fix_tool": None,
        "requires_authorized": False,
        "risk": "none",
    }
    if finding.status == "pass" and not finding.fix_hint:
        return base
    mapping: dict[str, dict[str, Any]] = {
        "consistency.vec_parent_status_sync": {
            "fix_kind": "mcp_tool",
            "fix_tool": "memory_resync_vec_parent_status",
            "requires_authorized": False,
            "risk": "low",
        },
        "consistency.orphan_vectors": {
            "fix_kind": "mcp_tool",
            "fix_tool": "memory_cleanup_inactive_vectors",
            "requires_authorized": True,
            "risk": "cleanup",
        },
        "consistency.structured_claims": {
            "fix_kind": "mcp_tool",
            "fix_tool": "memory_rebuild_claims",
            "requires_authorized": False,
            "risk": "medium",
        },
        "capacity.history_bloat": {
            "fix_kind": "mcp_tool",
            "fix_tool": "memory_cleanup_history",
            "requires_authorized": False,
            "risk": "cleanup",
        },
        "consistency.vec_index_state": {
            "fix_kind": "mcp_tool",
            "fix_tool": "memory_rebuild_embeddings",
            "requires_authorized": False,
            "risk": "medium",
        },
        "split.long_unsplit_backlog": {
            "fix_kind": "agent_assisted",
            "fix_tool": "memory_split",
            "requires_authorized": False,
            "risk": "medium",
        },
        "split.failed_count": {
            "fix_kind": "agent_assisted",
            "fix_tool": "memory_split",
            "requires_authorized": False,
            "risk": "medium",
        },
    }
    if finding.check_id in mapping:
        return mapping[finding.check_id]
    if finding.check_id.startswith("config."):
        return {"fix_kind": "manual_config", "fix_tool": None, "requires_authorized": False, "risk": "manual"}
    if finding.check_id.startswith("semantic."):
        return {"fix_kind": "manual_config", "fix_tool": None, "requires_authorized": False, "risk": "manual"}
    if finding.check_id == "capacity.workspace_alias_health":
        return {"fix_kind": "mcp_tool", "fix_tool": "memory_govern",
                "requires_authorized": True, "risk": "low"}
    if finding.check_id == "vec.link3.extension_loaded":
        return {"fix_kind": "dependency_install", "fix_tool": None, "requires_authorized": False, "risk": "manual"}
    if finding.check_id == "vec.link4.model_usable":
        return {"fix_kind": "model_download", "fix_tool": None, "requires_authorized": False, "risk": "manual"}
    if finding.fix_hint:
        return {"fix_kind": "manual_or_none", "fix_tool": None, "requires_authorized": False, "risk": "manual"}
    return base


def report_to_dict(report: OverviewReport) -> dict[str, Any]:
    """Convert OverviewReport to a plain dict for state.response() envelope.

    ``state.response()`` does no serialization (degrade.py:24), so we must
    produce a JSON-friendly dict ourselves — matches the tools.py convention
    of plain dict literals (zero asdict() in existing code).
    """
    return {
        "arbiter_version": __version__,
        "snapshot_ts": report.snapshot_ts,
        "overall": report.overall.value,
        "summary": report.summary,
        "findings": [
            {
                "check_id": f.check_id,
                "dimension": f.dimension,
                "severity": f.severity.value,
                "status": f.status,
                "title": f.title,
                "detail": f.detail,
                "evidence": f.evidence,
                "fix_hint": f.fix_hint,
                "doc_link": f.doc_link,
                **_fix_metadata_for(f),
            }
            for f in report.findings
        ],
    }
