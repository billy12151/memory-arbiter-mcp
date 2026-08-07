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


def _check_attention_volume(settings: Settings) -> Finding:
    """v0.8.8: report ``attention_required`` flag volume (last 7 days), split
    by trigger (write/search) AND source.

    Pure diagnostic (always INFO). The write/search split matters: write is
    event-driven (one fire per new duplicate, low-freq), search is query-driven
    (the same conflict re-fires on every retrieval that hits it). A single
    total would let high-freq search mask whether write is producing real new
    duplicates — so both axes are surfaced and the human judges what's noisy.
    """
    path = settings.db_path.parent / "attention_log.jsonl"
    rep = _read_attention_log_counts(path, since_days=7)
    total = rep["total"]
    window = rep["window_days"]
    by_src = rep["by_source"]
    by_trig = rep["by_trigger"]
    if total == 0:
        return Finding(
            check_id="capacity.attention_volume", dimension="capacity",
            severity=Severity.INFO, status="pass",
            title=f"近 {window} 天无 attention 响铃",
            detail="attention_log.jsonl 无记录（或文件不存在）",
            evidence={"total": 0, "window_days": window, "by_source": {},
                      "by_trigger": {}, "by_source_trigger": {}},
        )
    w = by_trig.get("write", 0)
    s = by_trig.get("search", 0)
    src_parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_src.items(), key=lambda kv: -kv[1]))
    return Finding(
        check_id="capacity.attention_volume", dimension="capacity",
        severity=Severity.INFO, status="pass",
        title=f"近 {window} 天 attention 响铃 {total} 次（write {w} / search {s}；{src_parts}）",
        detail=("write=事件性(每次写入,低频), search=查询性(同一冲突被反复检索会重复响)。"
                "判断刷屏看 search 占比 + by_source_trigger 的来源分布。"),
        evidence={"total": total, "window_days": window, "by_source": by_src,
                  "by_trigger": by_trig, "by_source_trigger": rep["by_source_trigger"]},
    )


def _check_structured_claims(conn: sqlite3.Connection) -> Finding:
    """v0.9 claim-index consistency + pending judgment observability."""
    if not _table_exists(conn, "memory_claims") or not _table_exists(conn, "conflict_judgments"):
        return _na("consistency.structured_claims", "consistency", "v0.9 claim tables do not exist")
    stale_index = int(_scalar(
        conn,
        "SELECT COUNT(*) FROM memories WHERE status='active' "
        "AND (claims_indexed_revision IS NULL OR claims_indexed_revision<>claim_revision)",
    ) or 0)
    unreconciled = int(_scalar(
        conn,
        "SELECT COUNT(*) FROM memories WHERE status='active' "
        "AND claims_indexed_revision=claim_revision "
        "AND (claims_reconciled_revision IS NULL "
        "OR claims_reconciled_revision<>claim_revision)",
    ) or 0)
    stale = stale_index + unreconciled
    total_claims = int(_scalar(conn, "SELECT COUNT(*) FROM memory_claims") or 0)
    indexed_memories = int(_scalar(
        conn,
        "SELECT COUNT(*) FROM memories WHERE status='active' AND claims_indexed_revision=claim_revision",
    ) or 0)
    reconciled_memories = int(_scalar(
        conn,
        "SELECT COUNT(*) FROM memories WHERE status='active' "
        "AND claims_indexed_revision=claim_revision "
        "AND claims_reconciled_revision=claim_revision",
    ) or 0)
    ambiguous = int(_scalar(
        conn, "SELECT COALESCE(SUM(claim_ambiguous_count),0) FROM memories WHERE status='active'"
    ) or 0)
    missing_entity = int(_scalar(
        conn,
        "SELECT COUNT(*) FROM memories WHERE status='active' "
        "AND COALESCE(TRIM(CASE WHEN json_valid(metadata) "
        "THEN json_extract(metadata,'$.entity') ELSE NULL END),'')=''",
    ) or 0)
    pending_rows = conn.execute(
        "SELECT judgment_status, COUNT(*) AS c, MIN(created_at) AS oldest "
        "FROM conflicts WHERE status='open' AND structured_detected_at IS NOT NULL "
        "GROUP BY judgment_status"
    ).fetchall()
    pending = {
        str(row["judgment_status"] or "pending_llm"): {
            "count": int(row["c"]), "oldest": row["oldest"],
        }
        for row in pending_rows
    }
    by_rule = {
        str(row["extractor_rule"] or "unknown"): int(row["c"])
        for row in conn.execute(
            "SELECT extractor_rule, COUNT(*) AS c FROM memory_claims GROUP BY extractor_rule"
        ).fetchall()
    }
    by_attribute = {
        str(row["attribute"]): int(row["c"])
        for row in conn.execute(
            "SELECT attribute, COUNT(*) AS c FROM memory_claims "
            "GROUP BY attribute ORDER BY c DESC, attribute LIMIT 20"
        ).fetchall()
    }
    outcome_rows = conn.execute(
        "SELECT c.status, COALESCE(c.judgment_status,'pending_llm') AS judgment_status, "
        "COALESCE(j.verdict,'unjudged') AS verdict, COUNT(*) AS c "
        "FROM conflicts c LEFT JOIN conflict_judgments j ON j.id=c.active_judgment_id "
        "WHERE c.structured_detected_at IS NOT NULL "
        "GROUP BY c.status, COALESCE(c.judgment_status,'pending_llm'), "
        "COALESCE(j.verdict,'unjudged')"
    ).fetchall()
    outcomes = [
        {
            "status": row["status"], "judgment_status": row["judgment_status"],
            "verdict": row["verdict"], "count": int(row["c"]),
        }
        for row in outcome_rows
    ]
    rule_outcomes: dict[str, dict[str, int]] = {}
    attribute_outcomes: dict[str, dict[str, int]] = {}
    telemetry_rows = conn.execute(
        "SELECT c.status, COALESCE(c.judgment_status,'pending_llm') AS judgment_status, "
        "j.verdict AS verdict, c.structured_details "
        "FROM conflicts c LEFT JOIN conflict_judgments j ON j.id=c.active_judgment_id "
        "WHERE c.structured_detected_at IS NOT NULL AND c.structured_details IS NOT NULL"
    ).fetchall()
    for row in telemetry_rows:
        outcome = (
            "dismiss" if row["status"] == "not_a_conflict"
            else str(row["verdict"] or row["judgment_status"] or row["status"])
        )
        try:
            details = json.loads(row["structured_details"])
        except (TypeError, json.JSONDecodeError):
            continue
        for detail in details if isinstance(details, list) else []:
            if not isinstance(detail, dict):
                continue
            rule = str(detail.get("extractor_rule") or "unknown")
            attribute = str(detail.get("attribute") or "unknown")
            rule_outcomes.setdefault(rule, {})[outcome] = (
                rule_outcomes.setdefault(rule, {}).get(outcome, 0) + 1
            )
            attribute_outcomes.setdefault(attribute, {})[outcome] = (
                attribute_outcomes.setdefault(attribute, {}).get(outcome, 0) + 1
            )

    def numeric_stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "avg": None, "p50": None, "p95": None, "max": None}
        ordered = sorted(values)

        def nearest_rank(percent: int) -> float:
            index = max(0, min(len(ordered) - 1, (len(ordered) * percent + 99) // 100 - 1))
            return round(float(ordered[index]), 3)

        return {
            "count": len(ordered),
            "avg": round(sum(ordered) / len(ordered), 3),
            "p50": nearest_rank(50),
            "p95": nearest_rank(95),
            "max": round(float(ordered[-1]), 3),
        }

    latency_rows = conn.execute(
        "SELECT structured_enrich_ms, structured_candidate_count FROM memories "
        "WHERE status='active' AND claims_reconciled_revision=claim_revision "
        "AND structured_enrich_ms IS NOT NULL"
    ).fetchall()
    latency_stats = numeric_stats([
        float(row["structured_enrich_ms"]) for row in latency_rows
    ])
    candidate_stats = numeric_stats([
        float(row["structured_candidate_count"] or 0) for row in latency_rows
    ])

    channel_rows = conn.execute(
        "SELECT left_id, right_id, MIN(structured_detected_at) AS structured_at, "
        "MIN(scan_detected_at) AS scan_at FROM conflicts "
        "WHERE structured_detected_at IS NOT NULL OR scan_detected_at IS NOT NULL "
        "GROUP BY left_id, right_id"
    ).fetchall()
    detection_channels = {"structured_only": 0, "scan_only": 0, "both": 0}
    structured_lead_ms: list[float] = []
    for row in channel_rows:
        structured_at = row["structured_at"]
        scan_at = row["scan_at"]
        if structured_at and scan_at:
            detection_channels["both"] += 1
            try:
                structured_dt = datetime.fromisoformat(str(structured_at).replace("Z", "+00:00"))
                scan_dt = datetime.fromisoformat(str(scan_at).replace("Z", "+00:00"))
                lead_ms = (scan_dt - structured_dt).total_seconds() * 1000
                if lead_ms >= 0:
                    structured_lead_ms.append(lead_ms)
            except ValueError:
                pass
        elif structured_at:
            detection_channels["structured_only"] += 1
        else:
            detection_channels["scan_only"] += 1
    lead_stats = numeric_stats(structured_lead_ms)

    severity = Severity.WARNING if stale else Severity.INFO
    status = "warn" if stale else "pass"
    return Finding(
        check_id="consistency.structured_claims", dimension="consistency",
        severity=severity, status=status,
        title=(
            f"structured claims: {total_claims} claims / {reconciled_memories} reconciled memories"
            + (f"，{stale} stale" if stale else "，index current")
        ),
        detail=(
            "stale includes extraction/index drift and a completed claim publish whose conflict "
            "reconciliation did not finish. memory_rebuild_claims repairs either state. Detection "
            "channels and latency make the real-time path's incremental value measurable."
        ),
        evidence={
            "claims": total_claims, "indexed_memories": indexed_memories,
            "reconciled_memories": reconciled_memories,
            "stale_memories": stale, "stale_index_memories": stale_index,
            "unreconciled_memories": unreconciled,
            "structured_latency_ms": latency_stats,
            "candidate_peer_count": candidate_stats,
            "detection_channels": detection_channels,
            "structured_lead_ms": lead_stats,
            "ambiguous_keys": ambiguous,
            "metadata_entity_missing": missing_entity, "pending": pending,
            "by_extractor_rule": by_rule, "by_attribute_top20": by_attribute,
            "structured_outcomes": outcomes,
            "outcomes_by_rule": rule_outcomes,
            "outcomes_by_attribute": attribute_outcomes,
        },
        fix_hint=("Run memory_rebuild_claims(dry_run=true), then execute bounded batches."
                  if stale else ""),
    )


def _embedder_shallow_probe(settings: Settings) -> tuple[Optional[Any], list[str]]:
    """Shallow model-usability probe without loading the GGUF (design doc §7).

    Returns (probe_result, warnings). ``probe_result`` is a lightweight dict
    describing availability, or None if not usable. Used by both the link4
    shallow path and as the CLI default.
    """
    warnings: list[str] = []
    model_path = settings.embedding_model_path
    if not model_path:
        return None, []
    try:
        from llama_cpp import Llama  # noqa: F401
    except ImportError:
        warnings.append("llama-cpp-python not installed")
        return None, warnings
    if not os.path.exists(str(model_path)):
        warnings.append(f"GGUF model not found: {model_path}")
        return None, warnings
    return {"model_path": str(model_path), "shallow": True}, warnings


# =====================================================================
#  Vector enablement chain (design doc §7 + §9.B) — 5 links
# =====================================================================

def _check_vector_chain(
    conn: sqlite3.Connection,
    settings: Settings,
    deep: bool,
    runtime_state: Optional[DegradeState],
    embedder_probe: Optional[Callable[[], tuple[Any, list[str]]]],
    vec_state: dict,
    vec_table_exists: bool = False,
) -> list[Finding]:
    """Chain short-circuit: walk links 1→5; first break classifies, rest n/a."""
    dim = "vector"
    findings: list[Finding] = []
    vec_table_exists_at_link3 = vec_table_exists

    # Link 1: configured
    configured = (
        settings.embedding_provider == "gguf"
        and settings.embedding_model_path is not None
    )
    if not configured:
        findings.append(Finding(
            check_id="vec.link1.configured", dimension=dim, severity=Severity.INFO,
            status="fail",
            title="未配置 embedding 模型（语义召回未启用，属正常可选）",
            detail="embedding.provider 非 gguf 或 embedding.model_path 未设置。"
                   "当前为纯 FTS/关键词模式。",
            evidence={"embedding_provider": settings.embedding_provider,
                      "model_path": str(settings.embedding_model_path) if settings.embedding_model_path else None},
            fix_hint='config.json 加 "embedding.provider":"gguf" 和 "embedding.model_path"（绝对路径）',
        ))
        findings += [_na(f"vec.link{i}.{'configured' if i==1 else 'enabled_flag' if i==2 else 'extension_loaded' if i==3 else 'model_usable' if i==4 else 'auto_flags'}",
                          dim, "前序链环未通过，本环不适用") for i in range(2, 6)]
        return findings
    findings.append(Finding(
        check_id="vec.link1.configured", dimension=dim, severity=Severity.INFO,
        status="pass", title="已配置 GGUF embedding 模型",
        detail=f"provider=gguf, model_path={settings.embedding_model_path}",
        evidence={"model_path": str(settings.embedding_model_path)},
    ))

    # Link 2: vec.enabled flag
    if not settings.enable_sqlite_vec:
        findings.append(Finding(
            check_id="vec.link2.enabled_flag", dimension=dim, severity=Severity.WARNING,
            status="fail",
            title="已配置模型但 vec.enabled=false，语义召回实际未生效",
            detail="embedding.model_path 已设，但 vec.enabled=false，导致 _ensure_embedder "
                   "跳过自动向量化（tools.py:56-59）。这是最常见的'以为开了其实没开'病态。",
            evidence={"embedding_configured": True, "enable_sqlite_vec": False,
                      "model_path": str(settings.embedding_model_path)},
            fix_hint='config.json 加 "vec":{"enabled":true} 后重启 MCP',
        ))
        findings += [_na(f"vec.link{i}.{'extension_loaded' if i==3 else 'model_usable' if i==4 else 'auto_flags'}",
                          dim, "前序链环未通过，本环不适用") for i in range(3, 6)]
        return findings
    findings.append(Finding(
        check_id="vec.link2.enabled_flag", dimension=dim, severity=Severity.INFO,
        status="pass", title="vec.enabled=true",
        detail="sqlite-vec 扩展开关已打开",
        evidence={"enable_sqlite_vec": True},
    ))

    # Link 3: extension loaded. CLI can re-derive by loading on the diag conn.
    ext_available: Optional[bool]
    if runtime_state is not None:
        ext_available = runtime_state.sqlite_vec_available
        source = "MCP runtime state"
    else:
        # CLI: open_ro_connection already attempted sqlite_vec.load; verify the
        # extension is actually active on this conn by probing a vec scalar fn
        # (re-loading would be redundant; the probe is both accurate and cheaper).
        try:
            conn.execute("SELECT vec_version()")
            ext_available = True
            source = "CLI re-derived (vec_version probe)"
        except sqlite3.Error:
            ext_available = False
            source = "CLI re-derived (vec_version probe failed)"
    if not ext_available:
        # Distinguish "installed but not enabled" vs "not installed".
        try:
            import sqlite_vec  # type: ignore  # noqa: F401
            installed = True
        except ImportError:
            installed = False
        findings.append(Finding(
            check_id="vec.link3.extension_loaded", dimension=dim, severity=Severity.WARNING,
            status="fail",
            title="sqlite-vec 扩展未加载",
            detail=f"vec.enabled=true 但扩展未加载（来源：{source}）。",
            evidence={"installed": installed, "source": source},
            fix_hint=("已装但未加载→检查 sqlite-vec 安装；未装→"
                      "pip install 'memory-arbiter-mcp[vec]'") ,
        ))
        findings += [_na(f"vec.link{i}.{'model_usable' if i==4 else 'auto_flags'}",
                          dim, "前序链环未通过，本环不适用") for i in range(4, 6)]
        return findings
    findings.append(Finding(
        check_id="vec.link3.extension_loaded", dimension=dim, severity=Severity.INFO,
        status="pass", title="sqlite-vec 扩展已加载",
        detail=f"来源：{source}" + (
            "；注意：扩展可加载，但 memories_vec 表尚未创建（此库未启用过向量召回，"
            "写入新记忆或重启 MCP 触发初始化后才会建表）" if not vec_table_exists_at_link3 else ""),
        evidence={"sqlite_vec_available": True, "source": source,
                  "vec_table_exists": vec_table_exists_at_link3},
    ))

    # Link 4: model usable. MCP reuses the already-loaded embedder via probe;
    # CLI / no-probe does shallow (or deep if requested).
    model_usable = False
    model_detail = ""
    model_warnings: list[str] = []
    probe_source = "shallow"
    if embedder_probe is not None:
        # MCP path: probe is tools._ensure_embedder (idempotent cache).
        embedder, model_warnings = embedder_probe()
        if deep:
            probe_source = "MCP deep (probe returned loaded embedder)"
        else:
            probe_source = "MCP probe (idempotent cache)"
        if embedder is not None:
            # If deep, verify dimension via a real probe on the embedder.
            if deep:
                try:
                    er = embedder.embed_text(prefix="", body="dimension probe")
                    model_usable = bool(er.embedding) and len(er.embedding) == settings.vec_dim
                    if not model_usable:
                        model_detail = f"deep 探针维度不匹配或空 embedding"
                except Exception as exc:
                    model_usable = False
                    model_detail = f"deep 探针失败：{exc}"
            else:
                model_usable = True
        else:
            model_detail = "embedder_probe 返回 None：" + "; ".join(model_warnings) if model_warnings else "embedder_probe 返回 None"
    else:
        probe_result, model_warnings = _embedder_shallow_probe(settings)
        if probe_result is not None:
            model_usable = True
        else:
            model_detail = "shallow 探针失败：" + "; ".join(model_warnings) if model_warnings else "shallow 探针失败"
        probe_source = "CLI shallow" if not deep else "CLI deep"
        if deep:
            # CLI deep: actually build the embedder.
            try:
                from .embedder import build_embedder
                embedder, model_warnings = build_embedder(
                    str(settings.embedding_model_path),
                    settings.vec_dim,
                    n_ctx=getattr(settings, "embedding_n_ctx", 2048),
                    reserved_tokens=getattr(settings, "embedding_reserved_tokens", 64),
                    max_section_chars=getattr(settings, "max_section_chars", 3600),
                )
                model_usable = embedder is not None
                if not model_usable:
                    model_detail = "deep build_embedder 返回 None：" + "; ".join(model_warnings)
            except Exception as exc:
                model_usable = False
                model_detail = f"deep build_embedder 异常：{exc}"
    if not model_usable:
        findings.append(Finding(
            check_id="vec.link4.model_usable", dimension=dim, severity=Severity.CRITICAL,
            status="fail",
            title="GGUF 模型不可用（路径错 / 维度不匹配 / 加载失败）",
            detail=(model_detail or "模型不可用") + f"（探针来源：{probe_source}）",
            evidence={"model_path": str(settings.embedding_model_path),
                      "vec_dim": settings.vec_dim, "warnings": model_warnings,
                      "probe_source": probe_source},
            fix_hint="模型文件不存在：检查路径；或维度 N ≠ vec.dim M：改 config.json vec.dim 或换模型",
        ))
        findings.append(_na("vec.link5.auto_flags", dim, "前序链环未通过，本环不适用"))
        return findings
    findings.append(Finding(
        check_id="vec.link4.model_usable", dimension=dim, severity=Severity.INFO,
        status="pass", title="GGUF 模型可用",
        detail=f"探针来源：{probe_source}；warnings={model_warnings or '无'}",
        evidence={"model_path": str(settings.embedding_model_path), "probe_source": probe_source},
    ))

    # Link 5: auto flags
    aq, aw = settings.embedding_auto_query, settings.embedding_auto_write
    if not (aq and aw):
        findings.append(Finding(
            check_id="vec.link5.auto_flags", dimension=dim, severity=Severity.WARNING,
            status="fail",
            title="已配置但关闭了 auto_query/auto_write",
            detail=f"embedding.auto_query={aq}, auto_write={aw}。模型可用但自动向量化被关闭。",
            evidence={"auto_query": aq, "auto_write": aw},
            fix_hint='config.json 设 "embedding.auto_query":true, "auto_write":true',
        ))
    else:
        findings.append(Finding(
            check_id="vec.link5.auto_flags", dimension=dim, severity=Severity.INFO,
            status="pass", title="auto_query / auto_write 均已开启",
            detail=f"auto_query={aq}, auto_write={aw}",
            evidence={"auto_query": aq, "auto_write": aw},
        ))
    return findings


# =====================================================================
#  Config checks (design doc §9.A) — 3 items
# =====================================================================

def _check_config_warnings(settings: Settings) -> Finding:
    warns = settings.config_warnings or []
    has_severe = any(k in (w.lower()) for w in warns for k in ("invalid", "parse failed", "does not exist"))
    sev = Severity.WARNING if (warns and has_severe) else Severity.INFO
    status = "warn" if warns else "pass"
    title = "配置解析无告警" if not warns else f"配置解析有 {len(warns)} 条告警"
    return Finding(
        check_id="config.warnings", dimension="config", severity=sev, status=status,
        title=title,
        detail="config.py 在解析 config.json / env 时收集的告警（越界/格式错误/文件错误）。"
               + ("无告警。" if not warns else "含 invalid/parse/does-not-exist 关键词→warning。"),
        evidence={"count": len(warns), "items": warns[:20]},
        fix_hint="" if not warns else "按告警条目修正 config.json / 环境变量",
    )


def _check_db_writable(settings: Settings, runtime_state: Optional[DegradeState]) -> Finding:
    if runtime_state is not None:
        writable = runtime_state.sqlite_writable
        source = "MCP runtime state"
    else:
        writable = os.access(str(settings.db_path), os.W_OK)
        source = "CLI 推断 (os.access, 非 MCP 运行时状态；如需精确值请在对话中调 MCP doctor)"
    if writable:
        return Finding(
            check_id="config.db_writable", dimension="config", severity=Severity.INFO,
            status="pass", title="SQLite 可写（写探针通过）",
            detail=f"来源：{source}",
            evidence={"sqlite_writable": True, "source": source},
        )
    return Finding(
        check_id="config.db_writable", dimension="config", severity=Severity.CRITICAL,
        status="fail", title="SQLite 不可写（mode=jsonl_backup，正在丢数据）",
        detail=f"写探针失败，写入将只进 JSONL 备份不落库。来源：{source}",
        evidence={"sqlite_writable": False, "source": source},
        fix_hint="检查 DB 文件权限/磁盘空间；恢复可写后重启服务",
    )


def _check_degradation_mode(runtime_state: Optional[DegradeState],
                            vec_table_exists: bool, fts_table_exists: bool) -> Finding:
    if runtime_state is not None:
        # Ground the runtime mode in actual DB state: runtime_state.mode is set
        # once at MemoryDB init and can go stale if the vec table is later
        # dropped or the DB swapped. If runtime says sqlite_vec but no vec table
        # exists, downgrade to what the tables actually support.
        mode = runtime_state.mode
        if mode == "sqlite_vec" and not vec_table_exists:
            mode = "fts5" if fts_table_exists else "like"
        ev = {"mode": mode, "runtime_mode": runtime_state.mode,
              "sqlite_vec_available": runtime_state.sqlite_vec_available,
              "fts5_available": runtime_state.fts5_available,
              "sqlite_writable": runtime_state.sqlite_writable,
              "vec_table_exists": vec_table_exists}
        source = "MCP runtime state (grounded by table existence)"
    else:
        # CLI static inference from table existence.
        if vec_table_exists:
            mode = "sqlite_vec"
        elif fts_table_exists:
            mode = "fts5"
        else:
            mode = "like"
        ev = {"mode": mode, "vec_table_exists": vec_table_exists,
              "fts_table_exists": fts_table_exists, "source": "CLI 静态推断"}
        source = "CLI 推断 (非 MCP 运行时状态；如需精确值请在对话中调 MCP doctor)"
    sev_map = {"sqlite_vec": Severity.INFO, "fts5": Severity.INFO,
               "like": Severity.WARNING, "jsonl_backup": Severity.CRITICAL}
    sev = sev_map.get(mode, Severity.WARNING)
    status = "pass" if sev == Severity.INFO else ("warn" if sev == Severity.WARNING else "fail")
    detail = f"降级模式={mode}（来源：{source}）"
    fix = ""
    if mode == "jsonl_backup":
        fix = "新写入未落库，需尽快恢复 DB 可写后重启服务"
        detail += "；jsonl_backup = 静默丢数据态"
    elif mode == "like":
        fix = "FTS5 不可用，关键词召回退化为 LIKE；检查 SQLite 是否支持 FTS5"
    return Finding(
        check_id="config.degradation_mode", dimension="config", severity=sev, status=status,
        title=f"运行模式={mode}", detail=detail, evidence=ev, fix_hint=fix,
    )


# =====================================================================
#  Split checks (v0.8 design doc §6.5) — 6 items
#  Replaces the v0.7 split.enabled toggle check. Capability is now bound
#  to vec readiness; backlog/failed/legacy/integrity surface repair work.
# =====================================================================

def _vec_state_from_meta(conn: sqlite3.Connection) -> str:
    if not _table_exists(conn, "_vec_index_meta"):
        return "unmanaged"
    row = conn.execute(
        "SELECT value FROM _vec_index_meta WHERE key='state'"
    ).fetchone()
    return str(row["value"]) if row else "unmanaged"


def _check_split_capability(conn: sqlite3.Connection, settings: Settings) -> Finding:
    """§6.5: whether the server can split (vec + embedder available)."""
    state = _vec_state_from_meta(conn)
    embedder_configured = (
        settings.embedding_provider == "gguf"
        and settings.embedding_model_path is not None
    )
    if state == "ready":
        available, reason = True, "vec_ready"
    elif embedder_configured:
        available, reason = False, "vec_not_ready"
    else:
        available, reason = False, "embedder_unavailable"
    title = "分段能力可用（vec ready）" if available else f"分段能力不可用（{reason}）"
    return Finding(
        check_id="split.capability", dimension="split",
        severity=Severity.INFO if available else Severity.INFO,
        status="pass" if available else "n/a",
        title=title,
        detail=f"vec_state={state}, embedder_configured={embedder_configured}",
        evidence={"available": available, "reason": reason, "vec_state": state},
        fix_hint="" if available else (
            "配置 embedding.model_path + vec.enabled=true 并完成向量迁移"
        ),
    )


def _check_split_backlog(
    conn: sqlite3.Connection,
    settings: Settings,
    inflight_ids: Optional[set[int]] = None,
) -> Finding:
    """§6.5: long active memories with split_status IS NULL (awaiting Agent
    continuation). Only meaningful when vec is ready.

    ``inflight_ids`` are memory_ids currently being processed by the
    background SplitReindexWorker — their split_status is NULL because the
    async publish hasn't landed yet, not because they need Agent action.
    They are excluded from the backlog so the doctor does not misreport a
    normal async window as a backlog requiring intervention.
    """
    if _vec_state_from_meta(conn) != "ready":
        return _na("split.long_unsplit_backlog", "split",
                   "vec 未 ready，分段能力不可用，backlog 检查不适用")
    if not _table_exists(conn, "memories"):
        return _na("split.long_unsplit_backlog", "split", "memories 表不存在")
    threshold = getattr(settings, "split_threshold", 4000)
    rows = conn.execute(
        "SELECT id, length(content) AS clen FROM memories "
        "WHERE status='active' AND split_status IS NULL "
        "AND length(content) >= ? ORDER BY id",
        (threshold,),
    ).fetchall()
    inflight = inflight_ids or set()
    inflight_excluded = [int(r["id"]) for r in rows if int(r["id"]) in inflight]
    backlog_rows = [r for r in rows if int(r["id"]) not in inflight]
    backlog_count = len(backlog_rows)
    sample = [int(r["id"]) for r in backlog_rows[:20]]
    inflight_count = len(inflight_excluded)

    if backlog_count == 0:
        if inflight_count == 0:
            return Finding(
                check_id="split.long_unsplit_backlog", dimension="split",
                severity=Severity.INFO, status="pass",
                title="无未分段长文 backlog",
                detail=f"无 active 且 content≥{threshold} 且 split_status IS NULL 的记录",
                evidence={"backlog_count": 0, "split_threshold": threshold},
            )
        return Finding(
            check_id="split.long_unsplit_backlog", dimension="split",
            severity=Severity.INFO, status="pass",
            title=f"{inflight_count} 条长文正在后台分段中（不计入 backlog）",
            detail="这些记录已进入 SplitReindexWorker 队列，分段 embed 正在后台进行，"
                   "完成后 split_status 会异步变 active，无需 Agent 介入。",
            evidence={"backlog_count": 0, "inflight_count": inflight_count,
                      "inflight_memory_ids": inflight_excluded,
                      "split_threshold": threshold},
        )
    sev = Severity.WARNING if backlog_count > 10 else Severity.INFO
    return Finding(
        check_id="split.long_unsplit_backlog", dimension="split",
        severity=sev, status="warn" if sev == Severity.WARNING else "pass",
        title=f"{backlog_count} 条长文待分段（split_status=NULL）",
        detail="这些记录原文已保存但尚未发布 sections；Agent 收到 split_request 后应续接",
        evidence={"backlog_count": backlog_count,
                  "inflight_excluded": inflight_count,
                  "split_threshold": threshold,
                  "sample_memory_ids": sample},
        fix_hint="对 sample_memory_ids 逐条调 memory_split(memory_id) 续接分段",
    )


def _check_split_failed(conn: sqlite3.Connection, settings: Settings) -> Finding:
    """§6.5: memories whose real publish failed (split_status='failed')."""
    if not _table_exists(conn, "memories"):
        return _na("split.failed_count", "split", "memories 表不存在")
    rows = conn.execute(
        "SELECT id, metadata FROM memories WHERE status='active' AND split_status='failed'"
    ).fetchall()
    count = len(rows)
    if count == 0:
        return Finding(
            check_id="split.failed_count", dimension="split",
            severity=Severity.INFO, status="pass", title="无分段失败记录",
            detail="无 split_status='failed' 的 active 记录",
            evidence={"failed_count": 0},
        )
    recent: list[dict] = []
    for r in rows[:10]:
        try:
            meta = json.loads(r["metadata"] or "{}") if r["metadata"] else {}
            err = (meta.get("_split") or {}).get("last_split_error") or {}
        except Exception:
            err = {}
        recent.append({"memory_id": int(r["id"]), "last_split_error": err})
    return Finding(
        check_id="split.failed_count", dimension="split",
        severity=Severity.WARNING, status="warn",
        title=f"{count} 条分段失败记录",
        detail="真实发布失败（schema/anchor/offset/embedding）；原文仍可读",
        evidence={"failed_count": count, "recent": recent},
        fix_hint="对 memory_id 调 memory_split(memory_id) 重试；修正 anchor 后重新 publish",
    )


def _check_split_legacy_declined(conn: sqlite3.Connection, settings: Settings) -> Finding:
    """§6.5: historical declined records (v0.6/v0.7人工拒绝)."""
    if not _table_exists(conn, "memories"):
        return _na("split.legacy_declined", "split", "memories 表不存在")
    count = _scalar(conn,
        "SELECT count(*) FROM memories WHERE status='active' AND split_status='declined'") or 0
    if count == 0:
        return Finding(
            check_id="split.legacy_declined", dimension="split",
            severity=Severity.INFO, status="pass", title="无历史 declined 记录",
            detail="无 split_status='declined' 的记录",
            evidence={"legacy_declined_count": 0},
        )
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM memories WHERE status='active' AND split_status='declined' LIMIT 20"
    )]
    return Finding(
        check_id="split.legacy_declined", dimension="split",
        severity=Severity.INFO, status="warn",
        title=f"{count} 条历史 declined 记录",
        detail="v0.6/v0.7 人工拒绝分段；新流程不再产生 declined，仅兼容读取",
        evidence={"legacy_declined_count": count, "sample_memory_ids": ids},
        fix_hint="如需分段：memory_split(memory_id) 重新 publish",
    )


def _check_split_legacy_unknown_status(conn: sqlite3.Connection, settings: Settings) -> Finding:
    """§5.2: non-v0.8 statuses (pending/fallback_active) — surfaced read-only."""
    if not _table_exists(conn, "memories"):
        return _na("split.legacy_unknown_status", "split", "memories 表不存在")
    rows = conn.execute(
        "SELECT split_status, count(*) AS c FROM memories "
        "WHERE split_status IS NOT NULL "
        "AND split_status NOT IN ('active','failed','declined') "
        "GROUP BY split_status"
    ).fetchall()
    if not rows:
        return Finding(
            check_id="split.legacy_unknown_status", dimension="split",
            severity=Severity.INFO, status="pass", title="无非 v0.8 分段状态",
            detail="无 pending/fallback_active 或其他未知 split_status",
            evidence={"unknown_status_count": 0},
        )
    by_status = {r["split_status"]: r["c"] for r in rows}
    total = sum(by_status.values())
    return Finding(
        check_id="split.legacy_unknown_status", dimension="split",
        severity=Severity.WARNING, status="warn",
        title=f"{total} 条记录含非 v0.8 分段状态",
        detail=f"未知 split_status 分布：{by_status}。新代码不写入这些状态；仅暴露以便修复",
        evidence={"unknown_status_count": total, "by_status": by_status},
        fix_hint="这些状态无 v0.8 执行语义；逐条 memory_split 重新分段或人工核验",
    )


def _check_split_index_integrity(conn: sqlite3.Connection, settings: Settings) -> Finding:
    """§6.5: active records whose derived section index is inconsistent."""
    if not _table_exists(conn, "memory_sections"):
        return _na("split.index_integrity", "split", "memory_sections 表不存在")
    issues: list[dict] = []
    # active but no sections / fewer than 2 sections
    for r in conn.execute(
        "SELECT m.id, (SELECT count(*) FROM memory_sections s WHERE s.memory_id=m.id) AS sc "
        "FROM memories m WHERE m.status='active' AND m.split_status='active'"
    ).fetchall():
        mid, sc = int(r["id"]), int(r["sc"])
        if sc == 0:
            issues.append({"memory_id": mid, "problem": "active_but_no_sections"})
        elif sc < 2:
            issues.append({"memory_id": mid, "problem": "active_but_fewer_than_2_sections"})
    # missing section vectors for active memories
    if _table_exists(conn, "memory_sections_vec"):
        for r in conn.execute(
            "SELECT s.memory_id, count(*) AS missing FROM memory_sections s "
            "WHERE NOT EXISTS (SELECT 1 FROM memory_sections_vec v WHERE v.id=s.id) "
            "AND s.memory_id IN (SELECT id FROM memories WHERE split_status='active' AND status='active') "
            "GROUP BY s.memory_id LIMIT 20"
        ).fetchall():
            issues.append({"memory_id": int(r["memory_id"]),
                           "problem": "missing_section_vec", "missing": int(r["missing"])})
    # offset continuity spot-check: detect overlaps/gaps within a memory
    bad_offset = conn.execute(
        "SELECT s.memory_id, s.section_index, s.start_offset, s.end_offset "
        "FROM memory_sections s JOIN memories m ON m.id=s.memory_id "
        "WHERE m.split_status='active' AND m.status='active' "
        "AND s.end_offset <= s.start_offset LIMIT 20"
    ).fetchall()
    for r in bad_offset:
        issues.append({"memory_id": int(r["memory_id"]),
                       "problem": "non_positive_section_length",
                       "section_index": int(r["section_index"])})
    if not issues:
        return Finding(
            check_id="split.index_integrity", dimension="split",
            severity=Severity.INFO, status="pass", title="分段派生索引一致",
            detail="所有 active 记忆的 section 数量、向量覆盖、offset 均正常",
            evidence={"issue_count": 0},
        )
    return Finding(
        check_id="split.index_integrity", dimension="split",
        severity=Severity.WARNING, status="warn",
        title=f"{len(issues)} 处分段派生索引不一致",
        detail="active 无 sections / section<2 / 缺向量 / offset 异常",
        evidence={"issue_count": len(issues), "issues": issues[:20]},
        fix_hint="对受影响 memory 调 memory_split(split_decision='rebuild') 重建索引",
    )


# =====================================================================
#  Consistency checks (design doc §9.C) — 5 items
# =====================================================================

def _check_vec_index_state(conn: sqlite3.Connection, vec_chain_passed_link3: bool) -> Finding:
    if not vec_chain_passed_link3:
        return _na("consistency.vec_index_state", "consistency",
                   "向量链未通到 link3，向量索引状态不适用")
    if not _table_exists(conn, "_vec_index_meta"):
        return _na("consistency.vec_index_state", "consistency", "_vec_index_meta 表不存在")
    rows = conn.execute("SELECT key, value FROM _vec_index_meta").fetchall()
    meta = {str(r["key"]): str(r["value"]) for r in rows}
    state = meta.get("state", "unmanaged")
    active = meta.get("active_space_id")
    target = meta.get("target_space_id")
    last_error = meta.get("last_error")
    if last_error:
        return Finding(
            check_id="consistency.vec_index_state", dimension="consistency",
            severity=Severity.CRITICAL, status="fail",
            title="向量索引迁移失败（last_error 非空）",
            detail=f"state={state}, last_error={last_error}",
            evidence=meta, fix_hint="排查迁移错误；必要时 memory_rebuild_embeddings 重建",
        )
    if state == "mismatch" or (active and target and active != target):
        return Finding(
            check_id="consistency.vec_index_state", dimension="consistency",
            severity=Severity.WARNING, status="warn",
            title="向量空间 ID 漂移（迁移中：active≠target）",
            detail=f"state={state}, active={active}, target={target}",
            evidence=meta, fix_hint="调用 memory_rebuild_embeddings 完成向量迁移",
        )
    if state == "unmanaged":
        return Finding(
            check_id="consistency.vec_index_state", dimension="consistency",
            severity=Severity.WARNING, status="warn",
            title="向量索引未托管（unmanaged）但链路应已生效",
            detail="state=unmanaged 与配置链全通矛盾，可能状态未初始化",
            evidence=meta, fix_hint="重启 MCP 触发 _init_vec_state 重新对账",
        )
    return Finding(
        check_id="consistency.vec_index_state", dimension="consistency",
        severity=Severity.INFO, status="pass", title=f"向量索引状态正常（{state}）",
        detail=f"state={state}, active_space_id={active}",
        evidence=meta,
    )


def _check_orphan_sections(conn: sqlite3.Connection) -> Finding:
    if not _table_exists(conn, "memory_sections"):
        return _na("consistency.orphan_sections", "consistency", "memory_sections 表不存在")
    # Physical orphans (memory row gone)
    phys = _scalar(conn,
        "SELECT count(*) FROM memory_sections ms "
        "LEFT JOIN memories m ON ms.memory_id=m.id WHERE m.id IS NULL") or 0
    # Pointing to superseded/deleted
    stale = _scalar(conn,
        "SELECT count(*) FROM memory_sections ms JOIN memories m ON ms.memory_id=m.id "
        "WHERE m.status IN ('superseded','deleted')") or 0
    if phys == 0:
        detail = (
            "所有分段均有父记忆；无历史保留分段"
            if stale == 0
            else (
                f"{stale} 条分段属于 superseded/deleted 父记忆并作为审计历史保留；"
                "默认向量召回先按父状态过滤再计算距离，不会挤占 active 名额"
            )
        )
        return Finding(
            check_id="consistency.orphan_sections", dimension="consistency",
            severity=Severity.INFO, status="pass", title="无物理孤儿分段",
            detail=detail,
            evidence={"physical_orphans": 0, "retained_inactive": stale,
                      "stale_status": stale},
        )
    ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT ms.memory_id FROM memory_sections ms "
        "LEFT JOIN memories m ON ms.memory_id=m.id "
        "WHERE m.id IS NULL LIMIT 20")]
    return Finding(
        check_id="consistency.orphan_sections", dimension="consistency",
        severity=Severity.WARNING, status="warn",
        title=f"存在物理孤儿分段（{phys} 条父记忆不存在）",
        detail=(f"物理孤儿={phys}；另有 {stale} 条 inactive 历史分段被正常保留。"
                "物理孤儿按父状态缺失处理，不参与召回。"),
        evidence={"physical_orphans": phys, "retained_inactive": stale,
                  "stale_status": stale, "memory_ids": ids},
        fix_hint="人工核验受影响 section；必要时清理物理孤儿行",
    )


def _check_orphan_vectors(conn: sqlite3.Connection) -> Finding:
    if not _table_exists(conn, "memories_vec"):
        return _na("consistency.orphan_vectors", "consistency", "memories_vec 表不存在（向量未启用）")
    orphan_mem = _scalar(conn,
        "SELECT count(*) FROM memories_vec v "
        "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = v.id)") or 0
    orphan_sec = 0
    if _table_exists(conn, "memory_sections_vec"):
        orphan_sec = _scalar(conn,
            "SELECT count(*) FROM memory_sections_vec v "
            "WHERE NOT EXISTS (SELECT 1 FROM memory_sections s WHERE s.id = v.id)") or 0
    total = int(orphan_mem) + int(orphan_sec)
    evidence = {
        "orphan_memory_vectors": int(orphan_mem),
        "orphan_section_vectors": int(orphan_sec),
        "orphan_vectors": total,
    }
    if total == 0:
        return Finding(
            check_id="consistency.orphan_vectors", dimension="consistency",
            severity=Severity.INFO, status="pass", title="无孤儿向量",
            detail="所有 memory/section 向量均指向存在的父行",
            evidence=evidence,
        )
    mem_ids = [r[0] for r in conn.execute(
        "SELECT v.id FROM memories_vec v "
        "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = v.id) LIMIT 20")]
    sec_ids: list[int] = []
    if _table_exists(conn, "memory_sections_vec"):
        sec_ids = [r[0] for r in conn.execute(
            "SELECT v.id FROM memory_sections_vec v "
            "WHERE NOT EXISTS (SELECT 1 FROM memory_sections s WHERE s.id = v.id) LIMIT 20")]
    evidence.update({"memory_vector_ids": mem_ids, "section_vector_ids": sec_ids})
    return Finding(
        check_id="consistency.orphan_vectors", dimension="consistency",
        severity=Severity.WARNING, status="warn",
        title=f"存在孤儿向量（{total} 条指向已删除父行）",
        detail=(f"memory_vec 级 {orphan_mem} 条、section_vec 级 {orphan_sec} 条向量指向已物理消失的父行"
                "（外部改库/写入中断/迁移失败残留）。cleanup 只会清理这些 orphan，不会删除 superseded 向量。"),
        evidence=evidence,
        fix_hint="先备份；运行 memory_cleanup_inactive_vectors(dry_run=true) 预览，确认后运行 dry_run=false, authorized=true 清理 orphan vector rows。",
    )


def _check_vec_parent_status_sync(conn: sqlite3.Connection) -> Finding:
    """Report vec.parent_status vs memories.status mismatches (v0.9.4).

    The vec tables carry a ``parent_status`` TEXT column that should match
    the parent memory's ``status`` at write time (active/superseded/deleted).
    When it drifts — e.g. from direct DB edits, migration bugs, or failed
    transactions — the KNN metadata predicate ``AND v.parent_status = 'active'``
    may silently exclude valid rows or resurrect invalid ones.

    This check counts mismatches and reports them as a warn-level finding.
    """
    if not _table_exists(conn, "memories_vec"):
        return _na("consistency.vec_parent_status_sync", "consistency",
                   "memories_vec 表不存在（向量未启用）")
    mem_mismatch = _scalar(conn,
        "SELECT COUNT(*) FROM memories_vec v "
        "JOIN memories m ON m.id=v.id "
        "WHERE v.parent_status IS NULL "
        "OR v.parent_status != COALESCE(m.status, 'deleted')") or 0
    sec_mismatch = 0
    if _table_exists(conn, "memory_sections_vec"):
        sec_mismatch = _scalar(conn,
            "SELECT COUNT(*) FROM memory_sections_vec v "
            "JOIN memory_sections s ON s.id=v.id "
            "JOIN memories m ON m.id=s.memory_id "
            "WHERE v.parent_status IS NULL "
            "OR v.parent_status != COALESCE(m.status, 'deleted')") or 0
    evidence = {"memory_vec_parent_status_mismatches": mem_mismatch,
                "section_vec_parent_status_mismatches": sec_mismatch}
    if mem_mismatch == 0 and sec_mismatch == 0:
        return Finding(
            check_id="consistency.vec_parent_status_sync", dimension="consistency",
            severity=Severity.INFO, status="pass",
            title="vec.parent_status 与 memories.status 一致",
            detail="全部向量的 parent_status 与父记忆 status 同步；KNN metadata 谓词快路径可用",
            evidence=evidence,
        )
    return Finding(
        check_id="consistency.vec_parent_status_sync", dimension="consistency",
        severity=Severity.WARNING, status="warn",
        title=f"{mem_mismatch + sec_mismatch} 条向量的 parent_status 与父记忆不一致",
        detail=(f"memory_vec 级 {mem_mismatch} 条、section_vec 级 {sec_mismatch} 条 "
                f"parent_status 与 memories.status 不匹配。KNN metadata 谓词依赖该列，"
                f"漂移可能导致 active/expired 查询漏召或误召回；请先 resync 再依赖向量召回。"),
        evidence=evidence,
        fix_hint="运行 memory_resync_vec_parent_status（先 dry_run=true 预览，再 "
                 "dry_run=false 修复；resync 为非破坏性 UPDATE，无需 authorized）"
                 "或 memory_rebuild_embeddings 重建全部向量",
    )


def _check_section_vec_coverage(conn: sqlite3.Connection) -> Finding:
    if not _table_exists(conn, "memory_sections") or not _table_exists(conn, "memory_sections_vec"):
        return _na("consistency.section_vec_coverage", "consistency",
                   "memory_sections / memory_sections_vec 表不存在")
    # Only active parents need section vectors. Inactive parents may legitimately
    # retain section rows and/or vectors as audit history; their absence must not
    # trigger a false "missing coverage" warning.
    missing = _scalar(conn,
        "SELECT count(*) FROM memory_sections ms "
        "JOIN memories m ON m.id = ms.memory_id "
        "WHERE m.status = 'active' "
        "AND NOT EXISTS (SELECT 1 FROM memory_sections_vec v WHERE v.id = ms.id)") or 0
    if missing == 0:
        return Finding(
            check_id="consistency.section_vec_coverage", dimension="consistency",
            severity=Severity.INFO, status="pass", title="所有 active 分段均有向量覆盖",
            detail="active 记忆无缺失的 section 向量",
            evidence={"missing_section_vec": 0},
        )
    return Finding(
        check_id="consistency.section_vec_coverage", dimension="consistency",
        severity=Severity.WARNING, status="warn",
        title=f"{missing} 个 active section 缺向量（段落级语义召回失效）",
        detail="这些 active 记忆的 section 退化为整条召回",
        evidence={"missing_section_vec": missing},
        fix_hint="对所属记忆调用 memory_rebuild_embeddings(memory_ids=[...]) 修复",
    )


def _check_history_version_chain(conn: sqlite3.Connection) -> Finding:
    if not _table_exists(conn, "memory_history"):
        return _na("consistency.history_version_chain", "consistency",
                   "memory_history 表不存在")
    broken = conn.execute(
        "SELECT m.id, m.version AS live, "
        "(SELECT max(version) FROM memory_history WHERE memory_id=m.id) AS hist_max "
        "FROM memories m "
        "WHERE m.version > 1 "
        "AND (SELECT max(version) FROM memory_history WHERE memory_id=m.id) IS NOT NULL "
        "AND (SELECT max(version) FROM memory_history WHERE memory_id=m.id) + 1 != m.version"
    ).fetchall()
    if not broken:
        return Finding(
            check_id="consistency.history_version_chain", dimension="consistency",
            severity=Severity.INFO, status="pass", title="版本链连续",
            detail="所有编辑过的记忆 live 版本与历史链连续（max(hist)+1==live）",
            evidence={"broken_count": 0},
        )
    ids = [{"memory_id": r["id"], "live": r["live"], "hist_max": r["hist_max"]} for r in broken[:20]]
    return Finding(
        check_id="consistency.history_version_chain", dimension="consistency",
        severity=Severity.WARNING, status="warn",
        title=f"版本链断链（{len(broken)} 条记忆）",
        detail="live 版本与历史链不连续，属低频异常（写入中断 / 外部改库）",
        evidence={"broken_count": len(broken), "items": ids},
        fix_hint="记录但不自动改；必要时从 memory_history 手动恢复",
    )


# =====================================================================
#  Capacity checks (design doc §9.D) — 4 items
# =====================================================================

def _open_conflicts_by_source(conn: sqlite3.Connection) -> dict:
    """v0.8.8: open conflict counts grouped by source — lets the operator see
    how many open rows are verified (llm_informed) vs advisory write-hints
    (metadata_write_hint), so write落表 doesn't silently inflate the count."""
    try:
        rows = conn.execute(
            "SELECT COALESCE(source, 'unknown') AS src, count(*) AS c "
            "FROM conflicts WHERE status='open' GROUP BY src"
        ).fetchall()
        return {r["src"]: r["c"] for r in rows}
    except sqlite3.Error:
        return {}


def _check_conflicts_open(
    conn: sqlite3.Connection,
    settings: Settings,
    runtime_state: Optional[DegradeState] = None,
) -> Finding:
    """Report open conflict rows from the ``conflicts`` table.

    Historically this check also warned when ``scan_log.jsonl`` showed that the
    vector conflict-candidate scan had never run or was stale. That scan path
    has been removed (the legacy KNN candidate scanner is deprecated), so a
    missing/stale scan_log is no longer a defect and is reported at most as
    INFO context, never as a WARNING. The active conflict-candidate sources are
    scheduled LLM review and the optional write-time ``semantic_conflict`` path.
    """
    if not _table_exists(conn, "conflicts"):
        return _na("capacity.conflicts_open", "capacity", "conflicts 表不存在")

    row = conn.execute(
        "SELECT count(*) AS c, min(created_at) AS oldest "
        "FROM conflicts WHERE status='open'"
    ).fetchone()
    count = row["c"] if row else 0
    oldest = row["oldest"] if row else None
    vec_ok = (runtime_state.sqlite_vec_available if runtime_state else False)

    # scan_log is legacy diagnostic only now (no writer remains). Surface it
    # as context, never as a WARNING pointing at a deprecated feature.
    scan_log_path = settings.db_path.parent / "scan_log.jsonl"
    last_scan = _read_scan_log_last_completed(scan_log_path)
    scan_note = None
    if last_scan:
        scan_time = last_scan.get("scan_time")
        days_ago = _days_since_iso(scan_time) if scan_time else None
        if days_ago is not None:
            scan_note = f"legacy scan_log last={scan_time} ({days_ago}d ago, deprecated path)"

    detail_parts = [f"open_count={count}, oldest={oldest}, vec_available={vec_ok}"]
    if scan_note:
        detail_parts.append(scan_note)
    detail = ", ".join(detail_parts)

    if count == 0:
        return Finding(
            check_id="capacity.conflicts_open", dimension="capacity",
            severity=Severity.INFO, status="pass",
            title="无 open 冲突",
            detail=detail,
            evidence={"open_count": 0, "vec_available": vec_ok,
                      "last_scan_time": (last_scan or {}).get("scan_time")},
        )
    sev = Severity.WARNING if count > 20 else Severity.INFO
    return Finding(
        check_id="capacity.conflicts_open", dimension="capacity", severity=sev,
        status="warn" if sev == Severity.WARNING else "pass",
        title=f"{count} 条 open 冲突未仲裁" + (f"，最老 {oldest}" if oldest else ""),
        detail=detail,
        evidence={"open_count": count, "oldest": oldest, "vec_available": vec_ok,
                  "last_scan_time": (last_scan or {}).get("scan_time"),
                  "by_source": _open_conflicts_by_source(conn)},
        fix_hint="建议 memory_list_conflicts 处理" if count > 20 else "",
    )


def _check_superseded_ratio(conn: sqlite3.Connection) -> Finding:
    rows = conn.execute("SELECT status, count(*) AS c FROM memories GROUP BY status").fetchall()
    counts = {r["status"]: r["c"] for r in rows}
    total = sum(counts.values()) or 1
    superseded = counts.get("superseded", 0) + counts.get("deleted", 0)
    ratio = superseded / total
    sev = Severity.INFO
    return Finding(
        check_id="capacity.superseded_ratio", dimension="capacity", severity=sev,
        status="pass",
        title=f"superseded/deleted 占比 {ratio:.0%}" + ("（偏高，库可瘦身）" if ratio > 0.5 else ""),
        detail="supersede 为逻辑删除、不清向量/FTS 行，占比高意味索引同步膨胀（非错误）",
        evidence={"status_counts": counts, "ratio": round(ratio, 3)},
        fix_hint="可清理废弃记忆以瘦身" if ratio > 0.5 else "",
    )


def _check_history_bloat(conn: sqlite3.Connection) -> Finding:
    if not _table_exists(conn, "memory_history"):
        return _na("capacity.history_bloat", "capacity", "memory_history 表不存在")
    h = _scalar(conn, "SELECT count(*) FROM memory_history") or 0
    m = _scalar(conn, "SELECT count(*) FROM memories") or 0
    ratio = (h / m) if m else 0
    sev = Severity.INFO
    return Finding(
        check_id="capacity.history_bloat", dimension="capacity", severity=sev,
        status="pass",
        title=f"历史快照 {h} 条 / 活跃 {m} 条（{ratio:.1f} 倍）" + ("（偏多）" if ratio > 5 else ""),
        detail="history/memories 比值反映编辑频率",
        evidence={"history": h, "memories": m, "ratio": round(ratio, 2)},
        fix_hint="可用 memory_cleanup_history(older_than_days=30) 瘦身" if ratio > 5 else "",
    )


def _check_db_size(conn: sqlite3.Connection) -> Finding:
    try:
        page_count = _scalar(conn, "PRAGMA page_count") or 0
        page_size = _scalar(conn, "PRAGMA page_size") or 0
        journal = _scalar(conn, "PRAGMA journal_mode")
        size_mb = (page_count * page_size) / (1024 * 1024)
    except sqlite3.Error as exc:
        return Finding(
            check_id="capacity.db_size", dimension="capacity", severity=Severity.INFO,
            status="n/a", title="DB 容量读取失败",
            detail=f"PRAGMA 失败：{exc}", evidence={},
        )
    return Finding(
        check_id="capacity.db_size", dimension="capacity", severity=Severity.INFO,
        status="pass", title=f"DB 容量 {size_mb:.1f} MB（journal={journal}）",
        detail=f"page_count={page_count}, page_size={page_size}",
        evidence={"size_mb": round(size_mb, 2), "journal_mode": journal},
    )


# =====================================================================
#  Orchestration (design doc §5 layer 2, §9 constraint 4)
# =====================================================================

def run_all_checks(
    conn: sqlite3.Connection,
    settings: Settings,
    deep: bool = False,
    runtime_state: Optional[DegradeState] = None,
    embedder_probe: Optional[Callable[[], tuple[Any, list[str]]]] = None,
    inflight_ids: Optional[set[int]] = None,
) -> OverviewReport:
    """Run all 25 findings on one ro connection (consistent snapshot).

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
