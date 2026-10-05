"""doctor 深检查 9 项（从 doctor.py 搬出，拆分批 ⑦ 纯移动）。_DEEP_CHECKS 顺序元组留守 doctor.py。"""
from __future__ import annotations
import functools
from dataclasses import dataclass
from typing import Any, Callable

from ..db.meta import active_dim_on_connection, vec_table_dimension
from .types import Finding, _DoctorCtx, _finding, _safe_count


@dataclass
class _ProbeOutcome:
    """How resolving the embedder went -- and nothing about what it produced.

    Deliberately excludes the embed_text result. When the probe hands back a
    handle but embedding fails, vector.device and evidence.unit_budget are
    still reported: the GPU-degradation signal is exactly what an operator
    needs in that situation. Folding the embed outcome in here would make
    ``has_embedder`` false and drop both findings silently.
    """

    kind: str  # "no_resolver" | "raised" | "none" | "ok"
    embedder: Any = None
    error: BaseException | None = None

    @property
    def has_embedder(self) -> bool:
        return self.kind == "ok"


@dataclass  # never slots=True: cached_property needs __dict__
class _DeepCtx:
    """Deep-mode state. Only genuinely read-only values are collected up front.

    ``vec_meta`` and ``active_dim`` are pure reads, so pulling them ahead of
    the quick_check and migration_state reads they now precede changes nothing.
    ``probe`` is not pure and therefore must not be: on the MCP path it is
    MemoryTools._ensure_embedder, whose first success creates the vec0 tables
    and writes _vec_index_meta. Resolving it during collect would change what
    vector.space, vector.table_dimension, vector.evidence_rows and
    vector.workspace_rows see, so it stays lazy and the first reader is
    _d_vector_dimension_probe -- asserted in tests/test_golden_doctor.py.
    """

    base: _DoctorCtx
    vec_meta: dict[str, str]
    active_dim: int | None

    @classmethod
    def collect(cls, base: _DoctorCtx) -> "_DeepCtx":
        vec_meta = {
            str(row[0]): str(row[1])
            for row in base.conn.execute("SELECT key,value FROM _vec_index_meta")
        }
        # The embedding dimension is a per-library fact (meta key, else the
        # vec0 tables' own CREATE SQL) — there is no configured vec.dim.
        return cls(base=base, vec_meta=vec_meta, active_dim=active_dim_on_connection(base.conn))

    @functools.cached_property
    def probe(self) -> _ProbeOutcome:
        if self.base.embedder_probe is None:
            return _ProbeOutcome("no_resolver")
        try:
            embedder, _probe_warnings = self.base.embedder_probe()
        except Exception as exc:
            return _ProbeOutcome("raised", error=exc)
        if embedder is None:
            return _ProbeOutcome("none")
        return _ProbeOutcome("ok", embedder=embedder)


_DeepCheck = Callable[[_DeepCtx], "Finding | None"]


def _d_database_quick_check(ctx: _DeepCtx) -> Finding:
    quick_check = str(ctx.base.conn.execute("PRAGMA quick_check").fetchone()[0])
    return _finding(
        "database.quick_check", quick_check == "ok", quick_check,
        critical=quick_check != "ok",
    )


def _d_database_schema_generation(ctx: _DeepCtx) -> Finding:
    migration = {
        str(row[0]): str(row[1])
        for row in ctx.base.conn.execute(
            "SELECT key,value FROM migration_state "
            "WHERE key IN ('schema_generation','phase','migration_completed_at')"
        )
    }
    generation_ok = (
        migration.get("schema_generation") == "workspace_state_v1"
        and migration.get("phase") not in {"building", "backfill", "resuming", "failed"}
    )
    return _finding(
        "database.schema_generation", generation_ok,
        f"generation={migration.get('schema_generation') or 'missing'}, "
        f"phase={migration.get('phase') or 'complete'}",
        critical=not generation_ok,
        evidence=migration,
    )


def _d_vector_space(ctx: _DeepCtx) -> Finding:
    active_space = ctx.vec_meta.get("active_space_id")
    configured_space: str | None = None
    try:
        from ..vnext_migration import _configured_embedding_space_id
        configured_space = _configured_embedding_space_id(ctx.base.settings, ctx.active_dim)
    except (OSError, ValueError):
        configured_space = None
    space_ok = (
        configured_space is None
        or (ctx.vec_meta.get("state") == "ready" and active_space == configured_space)
    )
    state = ctx.vec_meta.get("state") or "unmanaged"
    detail = (
        f"state={state}, "
        f"active={active_space or 'none'}, configured={configured_space or 'none'}"
    )
    # 0.17.0 强制提示（owner 指令）：向量空间不匹配 = 升级后必须重建，
    # 提示带具体动作且完成前（state 翻 ready）持续出现。
    if state in {"mismatch", "failed"}:
        detail += (
            " — 升级待办：向量索引需重建，请执行 "
            "memory_repair(task='rebuild_evidence')（约 6 分钟/2 万向量）；"
            "完成前向量检索与冲突扫描降级为词法通道。"
        )
    return _finding(
        "vector.space", space_ok,
        detail,
        evidence={
            "state": state,
            "active_space_id": active_space,
            "configured_space_id": configured_space,
        },
    )


def _d_vector_table_dimension(ctx: _DeepCtx) -> Finding:
    evidence_dim = vec_table_dimension(ctx.base.conn, "memory_row_vec")
    workspace_dim = vec_table_dimension(ctx.base.conn, "workspace_canonicals_vec")
    # Lazy table creation means absent tables are only a problem once the
    # index should be live; existing tables must agree with the active dim.
    dimensions_ok = (
        ctx.base.settings.embedding_model_path is None
        or (evidence_dim is None and workspace_dim is None)
        or (evidence_dim is not None and evidence_dim == workspace_dim == ctx.active_dim)
    )
    return _finding(
        "vector.table_dimension", dimensions_ok,
        f"evidence={evidence_dim}, workspace={workspace_dim}, active={ctx.active_dim}",
        evidence={"evidence": evidence_dim, "workspace": workspace_dim,
                  "active": ctx.active_dim},
    )


def _d_vector_evidence_rows(ctx: _DeepCtx) -> Finding:
    conn = ctx.base.conn
    evidence_vectors = _safe_count(conn, "memory_row_vec")
    orphan_vectors = _safe_count(
        conn, "memory_row_vec v LEFT JOIN memory_row e ON e.id=v.id",
        "e.id IS NULL",
    )
    missing_vectors = _safe_count(
        conn, "memory_row e LEFT JOIN memory_row_vec v ON v.id=e.id",
        "v.id IS NULL",
    )
    # Absent tables are expected before the first embedder build (lazy
    # creation); they count against the index only once a dim is active.
    evidence_rows_ok = (
        ctx.base.settings.embedding_model_path is None
        or (evidence_vectors is None and ctx.active_dim is None)
        or (
            evidence_vectors is not None
            and orphan_vectors == 0
            and missing_vectors == 0
        )
    )
    return _finding(
        "vector.evidence_rows",
        evidence_rows_ok,
        f"{ctx.base.units} evidence rows, {evidence_vectors} vectors, "
        f"{orphan_vectors} orphan vectors, {missing_vectors} missing vectors",
        evidence={"evidence": ctx.base.units, "vectors": evidence_vectors,
                  "orphan_vectors": orphan_vectors, "missing_vectors": missing_vectors},
    )


def _d_vector_workspace_rows(ctx: _DeepCtx) -> Finding:
    canonical_count = int(ctx.base.conn.execute(
        "SELECT COUNT(*) FROM workspace_canonicals WHERE lower(trim(name)) "
        "NOT IN ('','default','none','null','unknown') AND trim(name) NOT IN ('默认','未知')"
    ).fetchone()[0])
    canonical_vectors = _safe_count(ctx.base.conn, "workspace_canonicals_vec")
    workspace_rows_ok = (
        ctx.base.settings.embedding_model_path is None
        or (canonical_vectors is None and ctx.active_dim is None)
        or (canonical_vectors is not None and canonical_vectors == canonical_count)
    )
    return _finding(
        "vector.workspace_rows",
        workspace_rows_ok,
        f"{canonical_count} non-default canonicals, {canonical_vectors} vectors",
        evidence={"canonicals": canonical_count, "vectors": canonical_vectors},
    )


def _d_vector_dimension_probe(ctx: _DeepCtx) -> Finding:
    # --deep / memory_review(deep=true): actually run the embedder and
    # compare the live dimension against the library's active dim
    # (seconds-level cost). First reader of ctx.probe -- see _DeepCtx.
    embedding_configured = ctx.base.settings.embedding_model_path is not None
    probe = ctx.probe
    if probe.kind == "no_resolver":
        return _finding("vector.dimension_probe", False, "deep probe requested but no embedder resolver was provided")
    if probe.kind == "raised":
        return _finding("vector.dimension_probe", False, f"embedder probe failed: {probe.error}")
    if probe.kind == "none":
        if embedding_configured:
            return _finding("vector.dimension_probe", False, "embedding configured but the embedder is currently unavailable")
        # Asking for a probe without embedding configured is a
        # no-op, not a health problem — must not force exit 1.
        return _finding("vector.dimension_probe", True, "deep probe skipped: embedding not configured")
    try:
        er = probe.embedder.embed_text(prefix="", body="dimension probe")
        dim = len(er.embedding or [])
    except Exception as exc:
        return _finding("vector.dimension_probe", False, f"embedding failed: {exc}")
    if ctx.active_dim is None:
        # No stored dim yet (fresh library before its first
        # embed): nothing to compare against, report the
        # model's own dim.
        return _finding(
            "vector.dimension_probe", True,
            f"embedding dim {dim} (no stored active dim yet)",
        )
    return _finding(
        "vector.dimension_probe", dim == ctx.active_dim,
        f"embedding dim {dim} vs active dim {ctx.active_dim}",
    )


def _d_vector_device(ctx: _DeepCtx) -> Finding | None:
    # Device visibility: a runtime GPU→CPU self-heal keeps the
    # embedder alive but should not go unnoticed — surface it
    # as a warning so an operator restarts to re-probe the GPU.
    # Emitted whenever a handle exists, including when embedding itself
    # failed: that is precisely when an operator needs to see the device.
    if not ctx.probe.has_embedder:
        return None
    embedder = ctx.probe.embedder
    degraded = bool(getattr(embedder, "device_degraded", False))
    gpu_backed = bool(getattr(embedder, "gpu_backed", False))
    degraded_at = getattr(embedder, "device_degraded_at", None)
    if degraded and degraded_at is None:
        # Latch closed but the CPU rebuild itself failed: the
        # embedder is still pointed at the broken GPU instance
        # and every embed returns the sentinel — that is DOWN,
        # not merely slow.
        device_detail = (
            "GPU failed and the CPU fallback rebuild also failed — "
            "embedding unavailable until restart"
        )
    elif degraded:
        device_detail = (
            f"GPU failed at {degraded_at}; degraded to CPU inference — "
            "restart to re-probe"
        )
    elif gpu_backed:
        device_detail = "embedding on GPU"
    else:
        device_detail = "embedding on CPU"
    return _finding("vector.device", not degraded, device_detail)


def _d_evidence_unit_budget(ctx: _DeepCtx) -> Finding | None:
    # Budget tripwire: units beyond the token budget lose their
    # tail at embed time. The splitter caps text units at 400
    # chars (≈≤402 tokens, under the 512 default), so any hit
    # means an uncapped subject unit or a splitter change.
    # tokenize_locked serialises against embed_text on the
    # shared live instance; the whole scan is guarded because
    # run_all_checks must never raise.
    if not ctx.probe.has_embedder:
        return None
    embedder = ctx.probe.embedder
    tokenize_locked = getattr(embedder, "tokenize_locked", None)
    budget = embedder.token_budget() if hasattr(embedder, "token_budget") else None
    if tokenize_locked is None or budget is None:
        return _finding(
            "evidence.unit_budget", True,
            "skipped: embedder does not expose tokenize_locked/token_budget",
        )
    # Char prefilter scaled to the worst tokenizer density
    # (byte-fallback expansion ≈4 tokens/char): below
    # budget/4 characters an input cannot exceed budget.
    prefilter = max(1, int(budget) // 4)
    try:
        over_budget = 0
        checked_units = 0
        worst_tokens = 0
        for row in ctx.base.conn.execute(
            "SELECT text FROM memory_row WHERE length(text) > ?",
            (prefilter,),
        ).fetchall():
            checked_units += 1
            tokens = len(tokenize_locked(str(row[0])))
            worst_tokens = max(worst_tokens, tokens)
            if tokens > budget:
                over_budget += 1
        return _finding(
            "evidence.unit_budget", over_budget == 0,
            f"{over_budget} of {checked_units} oversized units exceed "
            f"the {budget}-token embed budget (worst {worst_tokens} tokens); "
            "tails beyond the budget are not indexed",
            evidence={
                "over_budget": over_budget,
                "checked_units": checked_units,
                "worst_tokens": worst_tokens,
                "budget": budget,
            },
        )
    except Exception as exc:
        return _finding(
            "evidence.unit_budget", True,
            f"skipped: unit budget scan failed: {exc}",
        )
