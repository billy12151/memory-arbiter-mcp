"""doctor 浅检查 23 项（从 doctor.py 搬出，拆分批 ⑦ 纯移动）。_CHECKS 顺序元组留守 doctor.py。"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ..semantic_judge import device_default_batch as _device_default_batch
from ..constants import (
    SCAN_CHAIN_STALE_HOURS,
    SCAN_TASK_STALE_DAYS,
)
from ..timeutil import parse_iso8601_utc
from .types import Finding, _DoctorCtx, _finding, _workspace_review_finding


_Check = Callable[[_DoctorCtx], "Finding | None"]


def _c_config_writable(ctx: _DoctorCtx) -> Finding:
    return _finding("config.writable", ctx.runtime_state is None or ctx.runtime_state.sqlite_writable, "SQLite writable", critical=True)


def _c_evidence_coverage(ctx: _DoctorCtx) -> Finding:
    return _finding("evidence.coverage", ctx.indexed == ctx.eligible or not ctx.settings.embedding_auto_write, f"{ctx.indexed}/{ctx.eligible} memories indexed", evidence={"indexed": ctx.indexed, "eligible": ctx.eligible, "non_indexable": ctx.counts["non_indexable_memories"], "units": ctx.units})



def _c_row_vector_coverage(ctx: _DoctorCtx) -> Finding:
    # 0.17.0 P2-2.5: row-level conflict vectors follow the evidence-indexed
    # set. Informational by design (plan §4): the gap right after upgrade is
    # the boot backfill's pending queue — a snapshot cannot judge whether it
    # is shrinking, so the numbers go to detail and never to overall status.
    try:
        eligible = int(ctx.conn.execute(
            "SELECT COUNT(DISTINCT e.memory_id) FROM memory_row e"
        ).fetchone()[0])
        # 0.17.0 review R2：covered 原与 eligible 同表同谓词（恒等死仪表，
        # 恒报 100%，回填缺口永不可见）——covered 以 memory_row_vec 侧存在
        # 对应向量为口径（publish 单事务原子写两表，行在⇔向量在）。
        covered = int(ctx.conn.execute(
            "SELECT COUNT(DISTINCT e.memory_id) FROM memory_row e"
            " WHERE EXISTS(SELECT 1 FROM memory_row_vec v WHERE v.id = e.id)"
        ).fetchone()[0])
    except sqlite3.Error:
        return _finding(
            "rows.coverage", True, "row vector coverage unknown (table unavailable)",
            evidence={},
        )
    return _finding(
        "rows.coverage", True,
        f"{covered}/{eligible} evidence-indexed memories have row vectors",
        evidence={"row_covered": covered, "row_eligible": eligible},
    )


def _c_evidence_freshness(ctx: _DoctorCtx) -> Finding:
    return _finding("evidence.freshness", ctx.evidence_stale == 0, f"{ctx.evidence_stale} stale evidence rows", evidence={"stale": ctx.evidence_stale})


def _c_evidence_orphans(ctx: _DoctorCtx) -> Finding:
    return _finding("evidence.orphans", ctx.orphan == 0, f"{ctx.orphan} orphan evidence rows", evidence={"orphan": ctx.orphan})


def _c_conflicts_backlog(ctx: _DoctorCtx) -> Finding:
    unresolved_conflicts = ctx.open_conflicts + len(ctx.applying_rows)
    # C4 (0.15.13): triage counters double as the C2 pair@version
    # suppression's convergence observability — sustained weekly growth means
    # memories keep churning versions or suppression is failing.
    dismissed_total = int(ctx.conn.execute(
        "SELECT COUNT(*) FROM conflicts WHERE status='not_a_conflict'"
    ).fetchone()[0])
    week_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    dismissed_week = int(ctx.conn.execute(
        "SELECT COUNT(*) FROM conflicts WHERE status='not_a_conflict' "
        "AND created_at >= ?", (week_cutoff,),
    ).fetchone()[0])
    dismissed_latest = ctx.conn.execute(
        "SELECT refreshed_at FROM conflicts WHERE status='not_a_conflict' "
        "ORDER BY refreshed_at DESC LIMIT 1"
    ).fetchone()
    dismissed_latest_at = str(dismissed_latest[0]) if dismissed_latest else None
    return _finding(
        "conflicts.backlog", unresolved_conflicts < 100,
        f"{unresolved_conflicts} unresolved conflicts ({ctx.open_conflicts} open, {len(ctx.applying_rows)} applying); "
        f"triage dismissed {dismissed_total} total, {dismissed_week} this week",
        evidence={
            "open": ctx.open_conflicts, "applying": len(ctx.applying_rows),
            "not_a_conflict_total": dismissed_total,
            "not_a_conflict_this_week": dismissed_week,
            "latest_triage_at": dismissed_latest_at,
        },
    )


def _c_semantic_judge_model(ctx: _DoctorCtx) -> Finding | None:
    """0.17.1: the judge is the mDeBERTa checkpoint. Checks, in order:
    enabled-without-ckpt (info — the silent-misconfig breaker, P2 #20),
    configured-but-missing checkpoint file, the tokenizer/config model_dir
    directory, dependency availability (torch / transformers — the
    ``mdeberta`` extra), and the configured summary. Crash-breaker /
    last-error visibility needs the live backend instance, which doctor has
    no reference to (ctx carries conn+settings only). An unset ckpt with
    semantic_conflict.enabled=false is an opt-out, not this check's business
    (the degraded banner covers it)."""
    ckpt = ctx.settings.semantic_conflict_mdeberta_ckpt
    if ckpt is None:
        if bool(getattr(ctx.settings, "semantic_conflict_enabled", False)):
            # P2 #20: enabled without a checkpoint used to pass silently —
            # the user believes arbitration is on while nothing runs.
            return _finding(
                "semantic.judge_model", True,
                "semantic_conflict.enabled=true but mdeberta_ckpt is not "
                "configured — write-time conflict arbitration will not run; "
                "set semantic_conflict.mdeberta_ckpt (or disable "
                "semantic_conflict.enabled)",
                evidence={"enabled": True, "ckpt": None},
            )
        return None
    if not ckpt.exists():
        return _finding(
            "semantic.judge_model", False,
            f"mdeberta checkpoint not found: {ckpt} — write-time conflict "
            "arbitration is disabled; download mdeberta-v4m_dual_v1.pt and set "
            "semantic_conflict.mdeberta_ckpt",
            evidence={"ckpt": str(ckpt)},
        )
    model_dir = ctx.settings.semantic_conflict_mdeberta_model_dir
    if model_dir is not None and not Path(model_dir).is_dir():
        # P2 #20: a configured model_dir that is not a directory means the
        # tokenizer/config load inside the judge will fail at first use —
        # surface it now instead of at write time.
        return _finding(
            "semantic.judge_model", False,
            f"mdeberta model_dir is not a directory: {model_dir} — judge load "
            "will fail; point semantic_conflict.mdeberta_model_dir at the "
            "tokenizer/config directory",
            evidence={"ckpt": str(ckpt), "model_dir": str(model_dir)},
        )
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return _finding(
            "semantic.judge_model", False,
            f"mdeberta judge dependencies missing ({exc}) — install the extra: "
            "pip install memory-arbiter-mcp[mdeberta]",
            evidence={"ckpt": str(ckpt)},
        )
    # Crash-breaker / last-error visibility needs the live backend instance,
    # which doctor has no reference to (ctx carries conn+settings only) —
    # that surfacing branch would need a backend-ref wired through the
    # doctor entry; until then the configured-check below is the whole check.
    return _finding(
        "semantic.judge_model", True,
        f"mdeberta judge configured: {ckpt.name}",
        evidence={
            "ckpt": str(ckpt),
            "model_dir": str(model_dir or ""),
            "notice_min_prob": ctx.settings.semantic_conflict_mdeberta_notice_min_prob,
            "batch_size": (
                int(ctx.settings.semantic_conflict_mdeberta_batch)
                or _device_default_batch()
            ),
        },
    )


def _c_conflicts_scan_required(ctx: _DoctorCtx) -> Finding:
    return _finding(
        "conflicts.scan_required", not ctx.conflict_scan_required,
        "complete a full matching-detector conflict scan" if ctx.conflict_scan_required
        else (
            "conflict rebuild scan complete"
            if (ctx.last_scan is not None or ctx.has_scan_progress)
            else "no completed conflict scan on record — scheduled scan tasks are not set up; see memory_repair help (topic: scheduled_tasks)"
        ),
    )


def _c_conflicts_scan_epoch(ctx: _DoctorCtx) -> Finding | None:
    # 0.16.0 §6⑪: detector-epoch arm visibility — report the identity change
    # WITH its reason (E9 ②) so an upcoming full round is explained, not
    # mysterious. A stale arm (persisted 'to' != running detector) means the
    # arm ran under an older binary; the boot path re-arms anyway.
    epoch_row = ctx.conn.execute(
        "SELECT value FROM migration_state WHERE key='scan_epoch_armed'"
    ).fetchone() if ctx.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='migration_state'"
    ).fetchone() else None
    if epoch_row is None:
        return None
    try:
        epoch_arm = json.loads(str(epoch_row[0]))
    except (TypeError, ValueError):
        epoch_arm = None
    if not isinstance(epoch_arm, dict):
        return None
    from ..db_generation import CONFLICT_DETECTOR_VERSION

    armed_to = str(epoch_arm.get("to") or "")
    fresh = armed_to == CONFLICT_DETECTOR_VERSION
    return _finding(
        "conflicts.scan_epoch", fresh,
        (
            f"detector epoch armed: {epoch_arm.get('from') or 'none'} → {armed_to} "
            f"at {epoch_arm.get('at')} ({epoch_arm.get('reason')}); "
            "first full scan round pending"
        )
        if fresh
        else f"epoch arm stale (persisted to={armed_to}, running={CONFLICT_DETECTOR_VERSION})",
        evidence=epoch_arm,
    )


def _c_conflicts_scan_stale(ctx: _DoctorCtx) -> Finding | None:
    if ctx.last_scan is None:
        return None
    scanned_at = parse_iso8601_utc(ctx.last_scan.get("scan_time"))
    if scanned_at is None:
        return None
    # Same comparison basis as the guidance notice (exact seconds vs
    # the 14-day threshold, not floored .days), so tier and finding
    # cannot disagree inside day 14.
    age = datetime.now(timezone.utc) - scanned_at
    stale = age > timedelta(days=SCAN_TASK_STALE_DAYS)
    age_days = max(0, age.days)
    return _finding(
        "conflicts.scan_stale", not stale,
        f"last completed conflict scan {age_days} day(s) ago; "
        f"activity beyond {SCAN_TASK_STALE_DAYS} days means the scheduled task is not running",
        evidence={"last_scan_time": ctx.last_scan.get("scan_time"), "age_days": age_days},
    )


def _c_conflicts_scan_chain(ctx: _DoctorCtx) -> Finding | None:
    # C5 (0.15.13): broken-chain alarm over the routine scan's page-progress
    # kv. A whole chain normally completes in 15-20 minutes; an incomplete
    # record older than an hour with no completion line after it means the
    # round was interrupted mid-walk. scan_log.jsonl keeps its
    # completed-lines-only audit semantics (#825) — pacing lives in the kv.
    page_progress_row = ctx.conn.execute(
        "SELECT value FROM migration_state WHERE key='scan_page_progress'"
    ).fetchone()
    if page_progress_row is None:
        return None
    try:
        page_progress = json.loads(str(page_progress_row[0]))
    except (TypeError, ValueError):
        page_progress = None
    if not isinstance(page_progress, dict) or page_progress.get("complete"):
        return None
    progress_at = parse_iso8601_utc(str(page_progress.get("at") or ""))
    if progress_at is None:
        return None
    progress_age = datetime.now(timezone.utc) - progress_at
    completed_after = bool(
        ctx.last_scan is not None
        and str(ctx.last_scan.get("scan_time") or "") > str(page_progress.get("at") or "")
    )
    if progress_age <= timedelta(hours=SCAN_CHAIN_STALE_HOURS) or completed_after:
        return None
    return _finding(
        "conflicts.scan_chain", False,
        f"conflict scan interrupted at anchor {page_progress.get('next_anchor')} "
        f"({str(page_progress.get('at') or '')}); a full chain normally completes "
        f"in 15-20 minutes — resume paging from that anchor",
        evidence={
            "after": page_progress.get("after"),
            "next_anchor": page_progress.get("next_anchor"),
            "at": page_progress.get("at"),
            "client": page_progress.get("client"),
            "groups": page_progress.get("groups"),
        },
    )


def _c_conflicts_spec_drift(ctx: _DoctorCtx) -> Finding | None:
    # 0.16.0 §6⑩: scheduled-task spec drift. Any completed scan activity
    # proves A task exists; the spec stamp says WHICH contract it runs. Scan
    # activity without a current-version stamp = a v1 page-driven task that
    # should be rebuilt (the pipeline never got a kick).
    spec_stamp_row = ctx.conn.execute(
        "SELECT value FROM migration_state WHERE key='scheduled_tasks_spec_confirmed'"
    ).fetchone()
    if not ((ctx.last_scan is not None or ctx.has_scan_progress) and not ctx.conflict_scan_required):
        return None
    from ..scan_tasks import SCHEDULED_TASKS_SPEC_VERSION

    try:
        spec_stamp = int(str(spec_stamp_row[0])) if spec_stamp_row is not None else None
    except (TypeError, ValueError):
        spec_stamp = None
    if spec_stamp == SCHEDULED_TASKS_SPEC_VERSION:
        return None
    return _finding(
        "conflicts.spec_drift", False,
        (
            f"scheduled task appears to follow spec v{spec_stamp or 1} while the "
            f"server serves v{SCHEDULED_TASKS_SPEC_VERSION}: rebuild the "
            "conflict_scan task from memory(action='help', "
            "data={'topic': 'scheduled_tasks'}) — v2 tasks kick "
            "memory_repair(task='scan_pipeline') and clear the scan_queue "
            "judgment queue instead of paging scan_candidates"
        ),
        evidence={"served_spec_version": SCHEDULED_TASKS_SPEC_VERSION,
                  "task_spec_version": spec_stamp},
    )


def _c_conflicts_applying(ctx: _DoctorCtx) -> Finding:
    # Applying is a transient execution state: a healthy apply completes in
    # minutes, so any group still applying at doctor time is either mid-flight
    # or wedged. Flag every one with id/idle-days evidence (replaces the
    # removed stale-applying list surfacing); agent steers replan or resolve.
    now = datetime.now(timezone.utc)
    applying_groups: list[dict[str, Any]] = []
    for row in ctx.applying_rows[:10]:
        refreshed = str(row[1] or "")
        idle_days: int | None = None
        try:
            refreshed_at = datetime.fromisoformat(refreshed)
            if refreshed_at.tzinfo is None:
                refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
            idle_days = max(0, (now - refreshed_at).days)
        except ValueError:
            pass
        applying_groups.append({"id": int(row[0]), "refreshed_at": refreshed, "idle_days": idle_days})
    return _finding(
        "conflicts.applying", not ctx.applying_rows,
        f"{len(ctx.applying_rows)} applying conflict group(s) awaiting completion",
        evidence={"groups": applying_groups},
    )


def _c_notices_backlog(ctx: _DoctorCtx) -> Finding:
    return _finding("notices.backlog", ctx.open_notices < 100, f"{ctx.open_notices} open notices")


def _c_conflicts_scan_queue_backlog(ctx: _DoctorCtx) -> Finding:
    # 0.16.0 §6㉑⑦: the scan judgment queue is agent-facing only — the user
    # surface is this backlog COUNT plus a hint, never the items themselves
    # (suspected items are mostly noise). Console overview reads the same
    # numbers via doctor.
    try:
        queue_rows = ctx.conn.execute(
            "SELECT status, COUNT(*) AS c FROM scan_queue GROUP BY status"
        ).fetchall()
        queue_counts = {str(row["status"]): int(row["c"]) for row in queue_rows}
    except sqlite3.Error:
        queue_counts = {}
    queue_backlog = queue_counts.get("pending", 0)
    # 0.16.2: internal contradictions live in their OWN table but count
    # toward the same agent workload — scan_queue page's queue_backlog merges
    # both, doctor must too or it under-reports (a live 954-item workload
    # read as 666).
    try:
        internal_pending = int(ctx.conn.execute(
            "SELECT COUNT(*) FROM internal_conflicts WHERE status='pending'"
        ).fetchone()[0])
    except sqlite3.Error:
        internal_pending = 0
    queue_backlog += internal_pending
    # 0.17.0 P2-4.3: write-time conflict backlog rides the same visible
    # metric (500-cap bounded; eviction/overflow shows in the evidence keys).
    try:
        cb_rows = ctx.conn.execute(
            "SELECT status, COUNT(*) AS c FROM conflict_backlog GROUP BY status"
        ).fetchall()
        cb_counts = {str(row["status"]): int(row["c"]) for row in cb_rows}
    except sqlite3.Error:
        cb_counts = {}
    cb_pending = cb_counts.get("pending", 0)
    queue_backlog += cb_pending
    return _finding(
        "conflicts.scan_queue_backlog", queue_backlog < 100,
        (
            f"{queue_backlog} suspected item(s) awaiting agent judgment "
            f"({queue_counts.get('pending', 0)} pending, "
            f"{internal_pending} internal contradictions); "
            "when convenient, let the agent read the queue "
            "(memory_repair task='scan_queue', action='page')"
        ),
        evidence={**queue_counts, "backlog": queue_backlog, "internal_pending": internal_pending, "conflict_backlog_pending": cb_pending},
    )


def _c_normalize_autonomy(ctx: _DoctorCtx) -> Finding:
    # 0.16.0 §6⑫: autonomous normalization audit board — applied auto-moves,
    # rollbacks, and (never-moved) protected-bucket hints, so the owner sees
    # what the pipeline did on its own and what needs a human instead.
    try:
        applied = int(ctx.conn.execute(
            "SELECT COUNT(*) FROM normalize_audit WHERE status='applied'"
        ).fetchone()[0])
        rolled_back = int(ctx.conn.execute(
            "SELECT COUNT(*) FROM normalize_audit WHERE status='rolled_back'"
        ).fetchone()[0])
        # 0.16.3 default-fallback landings are audited per move; a growing
        # count is the owner's signal that agents are leaning on the escape
        # hatch instead of finding real buckets. 0.16.4: the count covers
        # BOTH entrances — memory_govern (status manual_move) and the
        # judgment-queue channel (status applied with the same gate flag),
        # which is the channel the spec steers agents to.
        default_fallback = int(ctx.conn.execute(
            "SELECT COUNT(*) FROM normalize_audit "
            "WHERE json_extract(gate, '$.default_fallback') = 1 "
            "AND status IN ('manual_move', 'applied')"
        ).fetchone()[0])
    except sqlite3.Error:
        applied = rolled_back = default_fallback = 0
    fallback_note = (
        f", {default_fallback} parked in default (fallback — re-home via "
        "memory_govern(action='move_memories_workspace'))"
        if default_fallback else ""
    )
    return _finding(
        "normalize.autonomy", True,
        f"{applied} autonomous move(s) applied, {rolled_back} rolled back{fallback_note}; "
        "rollback via memory_govern(action='rollback_auto_move', data={audit_id, authorized=true})",
        evidence={
            "applied": applied, "rolled_back": rolled_back,
            "default_fallback": default_fallback,
        },
    )


def _c_tags_over_limit(ctx: _DoctorCtx) -> Finding:
    # 0.16.0 §6⑮: tag-discipline backlog — pre-0.16.0 rows over the total cap
    # are NOT retro-truncated (reads unaffected); doctor lists them for a
    # manual cleanup pass (remove_tags) instead.
    #
    # B4（0.17.1 优化批）：SQL 侧先按 json_array_length 预筛（此前全表取
    # tags != '[]' 行 + 逐行 json.loads，30k 行实测 ~0.26s）。语义等价：
    # 坏 JSON 现实现本就 continue 跳过（json_valid 排除之）；json_array_length
    # 对坏 JSON 直接抛错（OR 不短路），故必须 AND 形式，不能用
    # `NOT json_valid(...) OR ...` 的备选（实测 malformed JSON 抛错）。
    from ..constants import MAX_MEMORY_TOTAL_TAGS as _TAG_CAP

    try:
        over: list[dict[str, int]] = []
        for row in ctx.conn.execute(
            "SELECT id, tags FROM memories WHERE status!='deleted' "
            "AND json_valid(tags) AND json_array_length(tags) > ?",
            (_TAG_CAP,),
        ).fetchall():
            try:
                parsed = json.loads(str(row["tags"]))
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, list) and len(parsed) > _TAG_CAP:
                over.append({"id": int(row["id"]), "count": len(parsed)})
    except sqlite3.Error:
        over = []
    return _finding(
        "tags.over_limit", not over,
        (
            f"{len(over)} memory(ies) exceed the {_TAG_CAP}-tag cap: "
            + ", ".join(f"#{item['id']}({item['count']})" for item in over[:10])
            + (" …" if len(over) > 10 else "")
            + " — tags are a retrieval dimension, not an event log; trim with update remove_tags"
            if over else f"all memories within the {_TAG_CAP}-tag cap"
        ),
        evidence={"over_limit": over, "cap": _TAG_CAP},
    )


def _c_capacity_attention_volume(ctx: _DoctorCtx) -> Finding:
    # v0.8.8 observability (restored): searches that ring conflict signals
    # append to attention_log.jsonl; surfacing the volume keeps advisory
    # flooding visible instead of growing an unread log forever.
    attention_lines = 0
    attention_recent = 0
    attention_path = Path(ctx.settings.db_path).parent / "attention_log.jsonl"
    if attention_path.exists():
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            with open(attention_path, encoding="utf-8") as fh:
                for line in fh:
                    attention_lines += 1
                    try:
                        if str(json.loads(line).get("ts", "")) >= cutoff:
                            attention_recent += 1
                    except json.JSONDecodeError:
                        continue
        except OSError:
            attention_lines = 0
    return _finding(
        "capacity.attention_volume", attention_recent < 200,
        f"{attention_recent} attention events in 7 days ({attention_lines} total)",
        evidence={"recent_7d": attention_recent, "total_lines": attention_lines},
    )


def _c_recall_blacklist(ctx: _DoctorCtx) -> Finding:
    # v0.15.5: recall blacklist visibility — default prefill vs customized
    # file, entry count, and any parse warnings (bad lines degrade the file).
    try:
        from ..recall_blacklist import blacklist_path, load_blacklist
        _bl_path = blacklist_path(ctx.settings.db_path)
        _bl_names, _bl_warnings = load_blacklist(_bl_path)
        _bl_source = "file" if _bl_path.exists() else "default"
        return _finding(
            "recall.blacklist", not _bl_warnings,
            f"unscoped find excludes {len(_bl_names)} workspace(s) via recall blacklist "
            f"({_bl_source}: {', '.join(sorted(_bl_names)) or 'none'})",
            evidence={"source": _bl_source, "workspaces": sorted(_bl_names),
                      "warnings": list(_bl_warnings)},
        )
    except Exception as exc:  # degrade loudly: a silently missing check hides regressions
        return _finding("recall.blacklist", False, f"recall blacklist check failed: {exc}")


def _c_workspace_review(ctx: _DoctorCtx) -> Finding:
    # full-registry workspace confirmation (read-only; the snapshot
    # is only ever written by the authorized confirm_workspaces action).
    return _workspace_review_finding(ctx.conn, ctx.settings)


def _c_config_warnings(ctx: _DoctorCtx) -> Finding | None:
    if not ctx.settings.config_warnings:
        return None
    return _finding("config.warnings", False, "; ".join(ctx.settings.config_warnings))


def _c_scan_poison(ctx: _DoctorCtx) -> Finding | None:
    """A4（0.17.1 修复批）：达上界的毒记忆可见（R3 指出方案 §A4-b/T5 未实现）。

    失败条不推进水位（覆盖不完整），此前只有 kick 回执的 ``poison_skipped``
    可见——没有主动巡检面。读轮状态 ``scan_pipeline_state.poison_failures``
    （跨 kick 累计，R3/R4 建议随轮状态而非 migration_state per-id 键）。
    无失败时返回 None（不新增 finding）。
    """
    import json as _json

    try:
        row = ctx.conn.execute(
            "SELECT value FROM migration_state WHERE key='scan_pipeline_state'"
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None or not row[0]:
        return None
    try:
        state = _json.loads(str(row[0]))
    except (TypeError, ValueError):
        return None
    failures = state.get("poison_failures") if isinstance(state, dict) else None
    if not isinstance(failures, dict) or not failures:
        return None
    from ..constants import SCAN_POISON_MAX_FAILURES as threshold
    skipped = sorted(
        int(mid) for mid, count in failures.items()
        if int(count) >= threshold
    )
    if not skipped:
        return None
    return _finding(
        "scan.poison", False,
        f"{len(skipped)} memory(ies) failed scanning {threshold}+ times "
        f"(watermark not advanced, coverage incomplete): "
        + ", ".join(f"#{mid}" for mid in skipped[:10])
        + (" …" if len(skipped) > 10 else "")
        + " — fix the data or move/retire the row; the scan round stays incomplete until then",
        evidence={"poison_skipped": skipped, "failures": failures},
    )


def _c_additive_incomplete(ctx: _DoctorCtx) -> Finding | None:
    """B3（0.17.1 修复批）：additive 的 vec 表 deferred 可见。

    此前 vec 虚表 DROP 被 deferred（模块未加载）只落一条启动 warning，
    findings 与 console 都看不到；而该状态的后果是"退役通道卡住 + 每次
    启动重试"（若影子表被误清则永久跳过，见 B2-1）。读 B2-3 的
    ``vec_drop_deferred`` 字典键；干净库返回 None（不新增 finding）。
    """
    import json as _json

    try:
        row = ctx.conn.execute(
            "SELECT value FROM migration_state WHERE key='vec_drop_deferred'"
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None or not row[0]:
        return None
    try:
        deferred = _json.loads(str(row[0]))
    except (TypeError, ValueError):
        return None
    if not isinstance(deferred, dict) or not deferred:
        return None
    detail = "; ".join(
        f"{table} (deferred since {stamp})" for table, stamp in sorted(deferred.items())
    )
    return _finding(
        "additive.deferred_drops", False,
        "vec0 tables could not be dropped yet (sqlite-vec module not loaded at "
        f"some boot); they will be cleared once the module is available: {detail}",
        evidence={"deferred": deferred},
    )


# Order is observable twice over: console_static renders findings.slice(0, 6),
# and a check that reads state a previous one could have changed depends on
# staying where it is. Reordering this tuple changes product output.
