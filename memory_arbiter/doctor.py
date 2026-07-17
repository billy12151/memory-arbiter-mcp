"""memory-arbiter doctor: read-only health diagnostics (design doc v1.4).

Three-layer architecture (design doc §5):
  1. ``check_xxx(conn, settings, deep, runtime_state, ...)`` — single check,
     depends only on a read-only sqlite connection + Settings (+ optional
     runtime state / embedder probe). Never on ``MemoryDB``.
  2. ``run_all_checks(conn, settings, deep, runtime_state, embedder_probe)``
     — orchestration: runs all 18 checks on one ro connection (consistent
     snapshot), with per-check try/except isolation (§9 constraint 4).
  3. Platform entries ``doctor_overview_mcp`` (used by tools.py) and the CLI
     entry (doctor_cli.py) — each owns connection acquisition + the global
     ``except Exception`` fallback (§11.1).

All checks are read-only: SELECT + reading Settings/DegradeState fields.
Never calls any authorized write tool, never changes schema.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from .config import Settings
from .degrade import DegradeState
from .models import utc_now_iso


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
#  Split check (design doc §8 + §9) — 1 item
# =====================================================================

def _check_split(conn: sqlite3.Connection, settings: Settings) -> Finding:
    enabled = getattr(settings, "split_enabled", False)
    if not enabled:
        return Finding(
            check_id="split.enabled", dimension="split", severity=Severity.INFO,
            status="pass", title="分段未启用（正常可选）",
            detail="split.enabled=false。长文整条存单向量，召回粒度粗。",
            evidence={"split_enabled": False},
            fix_hint='如需段落级召回：config.json 加 "split":{"enabled":true}',
        )
    if not _table_exists(conn, "memory_sections"):
        return Finding(
            check_id="split.enabled", dimension="split", severity=Severity.INFO,
            status="n/a", title="分段已启用但 memory_sections 表不存在",
            detail="可能为全新库尚未初始化",
            evidence={"split_enabled": True, "table_exists": False},
        )
    count = _scalar(conn, "SELECT count(*) FROM memory_sections") or 0
    sev = Severity.INFO
    status = "pass"
    detail = f"已启用，{count} 条分段记录"
    if count == 0:
        detail = "已启用但尚无分段记录，正常（尚未遇到长文）"
    return Finding(
        check_id="split.enabled", dimension="split", severity=sev, status=status,
        title=f"分段已启用（{count} 条记录）" if count else "分段已启用（暂无记录）",
        detail=detail, evidence={"split_enabled": True, "section_count": count},
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
    orphan = phys + stale
    if orphan == 0:
        return Finding(
            check_id="consistency.orphan_sections", dimension="consistency",
            severity=Severity.INFO, status="pass", title="无孤儿分段",
            detail="所有分段均指向 active 记忆",
            evidence={"physical_orphans": 0, "stale_status": 0},
        )
    ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT ms.memory_id FROM memory_sections ms "
        "LEFT JOIN memories m ON ms.memory_id=m.id "
        "WHERE m.id IS NULL OR m.status IN ('superseded','deleted') LIMIT 20")]
    return Finding(
        check_id="consistency.orphan_sections", dimension="consistency",
        severity=Severity.WARNING, status="warn",
        title=f"存在孤儿分段（{orphan} 条指向已删/已废弃记忆）",
        detail=f"物理孤儿={phys}，指向 superseded/deleted={stale}。召回会命中幽灵。",
        evidence={"physical_orphans": phys, "stale_status": stale, "memory_ids": ids},
        fix_hint="人工核验受影响 memory_id；必要时重建分段",
    )


def _check_orphan_vectors(conn: sqlite3.Connection) -> Finding:
    if not _table_exists(conn, "memories_vec"):
        return _na("consistency.orphan_vectors", "consistency", "memories_vec 表不存在（向量未启用）")
    orphan = _scalar(conn,
        "SELECT count(*) FROM memories_vec v "
        "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = v.id)") or 0
    if orphan == 0:
        return Finding(
            check_id="consistency.orphan_vectors", dimension="consistency",
            severity=Severity.INFO, status="pass", title="无孤儿向量",
            detail="所有向量均指向存在的 memory 行",
            evidence={"orphan_vectors": 0},
        )
    ids = [r[0] for r in conn.execute(
        "SELECT v.id FROM memories_vec v "
        "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = v.id) LIMIT 20")]
    return Finding(
        check_id="consistency.orphan_vectors", dimension="consistency",
        severity=Severity.WARNING, status="warn",
        title=f"存在孤儿向量（{orphan} 条指向已删除 memory 行）",
        detail=(f"{orphan} 条向量指向已物理消失的 memory（外部改库/写入中断/迁移失败残留）。"
                "⚠️ 当前无对外工具可清理（可观测但当前不可修，等 v2）。"),
        evidence={"orphan_vectors": orphan, "vector_ids": ids},
        fix_hint="无对外工具可清理；如需手动处理：DELETE FROM memories_vec "
                 "WHERE id NOT IN (SELECT id FROM memories)（请先备份）。已列入 v2 cleanup 候选。",
    )


def _check_section_vec_coverage(conn: sqlite3.Connection) -> Finding:
    if not _table_exists(conn, "memory_sections") or not _table_exists(conn, "memory_sections_vec"):
        return _na("consistency.section_vec_coverage", "consistency",
                   "memory_sections / memory_sections_vec 表不存在")
    missing = _scalar(conn,
        "SELECT count(*) FROM memory_sections ms "
        "WHERE NOT EXISTS (SELECT 1 FROM memory_sections_vec v WHERE v.id = ms.id)") or 0
    if missing == 0:
        return Finding(
            check_id="consistency.section_vec_coverage", dimension="consistency",
            severity=Severity.INFO, status="pass", title="所有分段均有向量覆盖",
            detail="无缺失的 section 向量",
            evidence={"missing_section_vec": 0},
        )
    return Finding(
        check_id="consistency.section_vec_coverage", dimension="consistency",
        severity=Severity.WARNING, status="warn",
        title=f"{missing} 个 section 缺向量（段落级语义召回失效）",
        detail="这些 section 退化为整条召回",
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

def _check_conflicts_open(conn: sqlite3.Connection) -> Finding:
    if not _table_exists(conn, "conflicts"):
        return _na("capacity.conflicts_open", "capacity", "conflicts 表不存在")
    row = conn.execute(
        "SELECT count(*) AS c, min(created_at) AS oldest FROM conflicts WHERE status='open'"
    ).fetchone()
    count = row["c"] if row else 0
    oldest = row["oldest"] if row else None
    if count == 0:
        return Finding(
            check_id="capacity.conflicts_open", dimension="capacity", severity=Severity.INFO,
            status="pass", title="无 open 冲突",
            detail="所有冲突已仲裁",
            evidence={"open_count": 0},
        )
    sev = Severity.WARNING if count > 20 else Severity.INFO
    return Finding(
        check_id="capacity.conflicts_open", dimension="capacity", severity=sev,
        status="warn" if sev == Severity.WARNING else "pass",
        title=f"{count} 条 open 冲突未仲裁" + (f"，最老 {oldest}" if oldest else ""),
        detail=f"open_count={count}, oldest={oldest}",
        evidence={"open_count": count, "oldest": oldest},
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
) -> OverviewReport:
    """Run all 18 checks on one ro connection (consistent snapshot).

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

    # --- split (1) ---
    _run("split.enabled", lambda: _check_split(conn, settings), "split")

    # --- consistency (5) ---
    _run("consistency.vec_index_state",
         lambda: _check_vec_index_state(conn, link3_passed), "consistency")
    _run("consistency.orphan_sections", lambda: _check_orphan_sections(conn), "consistency")
    _run("consistency.orphan_vectors", lambda: _check_orphan_vectors(conn), "consistency")
    _run("consistency.section_vec_coverage", lambda: _check_section_vec_coverage(conn), "consistency")
    _run("consistency.history_version_chain",
         lambda: _check_history_version_chain(conn), "consistency")

    # --- capacity (4) ---
    _run("capacity.conflicts_open", lambda: _check_conflicts_open(conn), "capacity")
    _run("capacity.superseded_ratio", lambda: _check_superseded_ratio(conn), "capacity")
    _run("capacity.history_bloat", lambda: _check_history_bloat(conn), "capacity")
    _run("capacity.db_size", lambda: _check_db_size(conn), "capacity")

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
        "split_enabled": getattr(settings, "split_enabled", False),
    }
    return OverviewReport(
        snapshot_ts=utc_now_iso(), overall=overall, findings=findings, summary=summary,
    )


# =====================================================================
#  Platform entries (design doc §5 layer 3, §11.1)
# =====================================================================

def build_unopenable_report(settings: Settings, exc: Exception) -> OverviewReport:
    """Minimal critical report when the DB cannot be opened (§11.1)."""
    return OverviewReport(
        snapshot_ts=utc_now_iso(),
        overall=Severity.CRITICAL,
        summary={"mode": "unopenable", "db_path": str(settings.db_path)},
        findings=[Finding(
            check_id="db.unopenable", dimension="config", severity=Severity.CRITICAL,
            status="fail", title="数据库无法打开，doctor 降级为最小报告",
            detail=(f"连接失败：{exc}。18 项 check 均未执行。"
                    "多数 jsonl_backup 是只读文件系统（文件可读仅不可写），mode=ro 能正常打开 → "
                    "若本应能打开却失败，通常是文件损坏/丢失/locked。"),
            evidence={"error": str(exc), "error_class": type(exc).__name__},
            fix_hint="检查 DB 文件是否存在、权限、是否被独占锁定；可从 backup_jsonl 恢复。",
        )],
    )


def doctor_overview_mcp(
    db: Any,
    settings: Settings,
    deep: bool = False,
    embedder_probe: Optional[Callable[[], tuple[Any, list[str]]]] = None,
    runtime_state: Optional[DegradeState] = None,
) -> OverviewReport:
    """MCP platform entry: uses MemoryDB.diagnostic_connection() + global fallback."""
    try:
        with db.diagnostic_connection() as conn:
            return run_all_checks(
                conn, settings, deep,
                runtime_state=runtime_state if runtime_state is not None else getattr(db, "state", None),
                embedder_probe=embedder_probe,
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


def report_to_dict(report: OverviewReport) -> dict[str, Any]:
    """Convert OverviewReport to a plain dict for state.response() envelope.

    ``state.response()`` does no serialization (degrade.py:24), so we must
    produce a JSON-friendly dict ourselves — matches the tools.py convention
    of plain dict literals (zero asdict() in existing code).
    """
    return {
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
            }
            for f in report.findings
        ],
    }
