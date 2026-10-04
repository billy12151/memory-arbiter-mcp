"""Read-only health checks for the local-text evidence architecture."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import Settings
from .constants import (
    EMBEDDING_MAX_SECTION_CHARS,
    EMBEDDING_N_CTX,
    EMBEDDING_RESERVED_TOKENS,
)
from .db_generation import detect_upgrade_source_generation
from .degrade import DegradeState
from .models import utc_now_iso


from .doctor_checks.types import (  # noqa: F401
    load_confirmed_workspaces as load_confirmed_workspaces,
    WORKSPACE_REVIEW_SIDECAR as WORKSPACE_REVIEW_SIDECAR,
    Severity as Severity,
    Finding as Finding,
    OverviewReport as OverviewReport,
    _DoctorCtx as _DoctorCtx,
    _finding as _finding,
    _safe_count as _safe_count,
    _workspace_review_finding as _workspace_review_finding,
    _last_completed_scan as _last_completed_scan,
)
from .doctor_checks.shallow import _Check as _Check, _c_config_writable, _c_evidence_coverage, _c_row_vector_coverage, _c_evidence_freshness, _c_evidence_orphans, _c_conflicts_backlog, _c_semantic_judge_model, _c_conflicts_scan_required, _c_conflicts_scan_epoch, _c_conflicts_scan_stale, _c_conflicts_scan_chain, _c_conflicts_spec_drift, _c_conflicts_applying, _c_notices_backlog, _c_conflicts_scan_queue_backlog, _c_normalize_autonomy, _c_tags_over_limit, _c_capacity_attention_volume, _c_recall_blacklist, _c_workspace_review, _c_config_warnings, _c_scan_poison, _c_additive_incomplete
from .doctor_checks.deep import _DeepCheck as _DeepCheck, _DeepCtx as _DeepCtx, _ProbeOutcome as _ProbeOutcome, _d_database_quick_check, _d_database_schema_generation, _d_vector_space, _d_vector_table_dimension, _d_vector_evidence_rows, _d_vector_workspace_rows, _d_vector_dimension_probe, _d_vector_device, _d_evidence_unit_budget
from .doctor_checks.deep import _DeepCheck as _DeepCheck, _DeepCtx as _DeepCtx, _ProbeOutcome as _ProbeOutcome



_CHECKS: tuple[_Check, ...] = (
    _c_config_writable,
    _c_evidence_coverage,
    _c_evidence_freshness,
    _c_evidence_orphans,
    _c_conflicts_backlog,
    _c_conflicts_scan_required,
    _c_conflicts_scan_epoch,
    _c_conflicts_scan_stale,
    _c_conflicts_scan_chain,
    _c_conflicts_spec_drift,
    _c_conflicts_applying,
    _c_notices_backlog,
    _c_conflicts_scan_queue_backlog,
    _c_normalize_autonomy,
    _c_tags_over_limit,
    _c_capacity_attention_volume,
    _c_recall_blacklist,
    _c_workspace_review,
    _c_semantic_judge_model,
    _c_config_warnings,
    _c_row_vector_coverage,
    # B3（0.17.1 修复批）：追加在末尾——干净库不出现（None），不影响
    # console_static 的 findings.slice(0, 6) 与既有 golden 顺序。
    _c_additive_incomplete,
    _c_scan_poison,
)


_DEEP_CHECKS: tuple[_DeepCheck, ...] = (
    _d_database_quick_check,
    _d_database_schema_generation,
    _d_vector_space,
    _d_vector_table_dimension,
    _d_vector_evidence_rows,
    _d_vector_workspace_rows,
    _d_vector_dimension_probe,
    _d_vector_device,
    _d_evidence_unit_budget,
)


def run_all_checks(conn: sqlite3.Connection, settings: Settings, deep: bool = False, runtime_state: DegradeState | None = None, embedder_probe: Callable[[], tuple[Any, list[str]]] | None = None) -> OverviewReport:
    ctx = _DoctorCtx.collect(conn, settings, deep, runtime_state, embedder_probe)
    findings: list[Finding] = []
    # No blanket try/except around the loop: a check raising unexpectedly must
    # keep crashing rather than degrade into one silently missing finding. The
    # per-check guards that already exist stay where they are. A missing table
    # still raises out of collect, which doctor_overview_* turns into a
    # database.open critical report -- unchanged behaviour.
    for check in _CHECKS:
        produced = check(ctx)
        if produced is not None:
            findings.append(produced)
    if deep:
        deep_ctx = _DeepCtx.collect(ctx)
        for deep_check in _DEEP_CHECKS:
            produced = deep_check(deep_ctx)
            if produced is not None:
                findings.append(produced)
    overall = max((f.severity for f in findings), key=lambda s: {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}[s])
    return OverviewReport(utc_now_iso(), overall, findings, {"mode": runtime_state.mode if runtime_state else "sqlite", "total_memories": ctx.total, "evidence_indexed": ctx.indexed, "evidence_units": ctx.units})


def report_to_dict(report: OverviewReport) -> dict[str, Any]:
    data = asdict(report); data["overall"] = report.overall.value
    for item in data["findings"]: item["severity"] = item["severity"].value if isinstance(item["severity"], Severity) else item["severity"]
    return data


def doctor_overview_mcp(db: Any, settings: Settings, deep: bool = False, **kwargs: Any) -> OverviewReport:
    try:
        with db.diagnostic_connection() as conn:
            return run_all_checks(conn, settings, deep, kwargs.get("runtime_state"), kwargs.get("embedder_probe"))
    except Exception as exc:
        return OverviewReport(utc_now_iso(), Severity.CRITICAL, [Finding("database.open", "database", Severity.CRITICAL, "error", "database.open", str(exc))], {"mode": "unavailable", "total_memories": 0})


@contextmanager
def open_ro_connection(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True); conn.row_factory = sqlite3.Row
    try: yield conn
    finally: conn.close()


def doctor_overview_cli(settings: Settings, deep: bool = False) -> OverviewReport:
    generation = detect_upgrade_source_generation(settings.db_path)
    if generation == "legacy":
        return OverviewReport(
            utc_now_iso(),
            Severity.CRITICAL,
            [Finding(
                "database.upgrade_required",
                "database",
                Severity.CRITICAL,
                "error",
                "database.upgrade_required",
                "legacy database generation; current code will not open or modify it",
                {"generation": generation, "path": str(settings.db_path)},
                "Stop all writers, then run `mema upgrade --dry-run` before `mema upgrade`.",
            )],
            {"mode": "upgrade_required", "total_memories": 0},
        )

    def _cli_embedder_probe() -> tuple[Any, list[str]]:
        # The CLI ambulance path has no MemoryTools; build the embedder
        # directly from settings so --deep actually probes the model its
        # help text promises (read-only; no DB access needed).
        if settings.embedding_model_path is None:
            return None, []
        try:
            from .embedder import build_embedder
            return build_embedder(
                str(settings.embedding_model_path),
                n_ctx=EMBEDDING_N_CTX,
                reserved_tokens=EMBEDDING_RESERVED_TOKENS,
                max_section_chars=EMBEDDING_MAX_SECTION_CHARS,
            )
        except Exception:
            return None, []

    try:
        with open_ro_connection(settings.db_path) as conn:
            if settings.embedding_model_path is not None:
                try:
                    import sqlite_vec

                    conn.enable_load_extension(True)
                    sqlite_vec.load(conn)
                    conn.enable_load_extension(False)
                except (ImportError, sqlite3.Error):
                    pass
            return run_all_checks(conn, settings, deep, embedder_probe=_cli_embedder_probe if deep else None)
    except Exception as exc:
        return OverviewReport(utc_now_iso(), Severity.CRITICAL, [Finding("database.open", "database", Severity.CRITICAL, "error", "database.open", str(exc))], {"mode": "unavailable", "total_memories": 0})
