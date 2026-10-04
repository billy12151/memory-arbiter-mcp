"""evidence 确定性相 mixin（从 evidence.py 搬出，拆分批 ⑤ 纯移动）。

conflicts_deterministic_phase 及其收集器族（applying slots/internal pairs/
neighbour screen/order candidates）。SEMANTIC_MAX_ROWS/SEMANTIC_MIN_PAIR_BUDGET_MS
读取点随迁本模块（patch 缝路径变更见方案 §6-2）。
"""
from __future__ import annotations

import json
import hashlib
import sqlite3
import time
from typing import Any, TYPE_CHECKING

from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..constants import (
    EMBED_PREFIX_STS,
    SEMANTIC_CROSS_KNN_WINDOW,
    SEMANTIC_MAX_ROWS,
    SEMANTIC_MIN_PAIR_BUDGET_MS,
)
from ..difference_classifier import classify_pair
from ..models import TrustedApplyingContext
from ..semantic_conflict import (
    SemanticBackend,
    _values_all_equivalent,
    decide_evidence,
    is_cross_evolution,
    normalize_value,
)
from ..embedder import ManagedEmbedder
from ._evidence_helpers import _filter_exempted_segments, _job_fair_deadline
from ._evidence_judge import _JobJudgeBudget

if TYPE_CHECKING:
    from ..config import Settings
    from ..db import MemoryDB
    from ..tools import MemoryTools
    from ..workers import SemanticConflictWorker


class _EvidencePhases:
    if TYPE_CHECKING:
        _tools: "MemoryTools"
        db: "MemoryDB"
        settings: "Settings"

        # 主类留守成员的 mypy strict 声明
        @property
        def _semantic_worker(self) -> "SemanticConflictWorker": ...
        def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]": ...
        def _ensure_semantic_backend(self) -> "SemanticBackend | None": ...
        @staticmethod
        def _streamed_pairs(*args: Any, **kwargs: Any) -> Any: ...

    def conflicts_deterministic_phase(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """0.17.0 Q1 相分裂 (owner plan §3.2) — phase 1 of 3: indexing publish,
        G5 clean list, candidate collection/ordering, internal keeper
        collection, truncation early-exit. ZERO Qwen. Returns the job context
        dict; ``ctx["terminal"]`` set means no further phase may run
        (skipped / stale / truncated — R1-2: the wrapper still rides B and C
        on a truncation terminal, matching the pre-split behavior)."""
        ctx = self._new_conflict_ctx(memory_id, snapshot)
        record = self.db.get_memory(int(memory_id))
        if record and record.get("status") == "pending":
            # A pending (workspace-activation) memory is not an incomplete
            # check: the conflict job is simply skipped until activation,
            # matching the "skipped" semantics used by index_memory above.
            ctx["terminal"] = {"status": "skipped", "reason": "pending_workspace_activation", "notices_created": 0}
            return ctx
        if not record or record.get("status") != "active":
            ctx["terminal"] = {"status": "incomplete", "reason": "memory_not_active", "notices_created": 0}
            return ctx
        if int(record.get("version") or 1) != int(snapshot.get("version") or 1):
            ctx["terminal"] = {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
            return ctx
        content = str(record.get("content") or "")
        # 0.16.12 P2-T2: the row's content_sha column IS sha256(content) (set
        # at insert, recomputed on every content edit) — compare against it
        # instead of re-hashing the content again (legacy NULL falls back).
        row_sha = str(record.get("content_sha") or "")
        if not row_sha:
            row_sha = hashlib.sha256(content.encode()).hexdigest()
        if row_sha != snapshot.get("content_hash"):
            ctx["terminal"] = {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
            return ctx
        # 0.16.12 P2-T6: the COLLECTION phase runs on ONE read-only connection
        # under an explicit read transaction — per-unit evidence_knn and the
        # peer probes all reuse it instead of opening one connection each.
        # Q1 相分裂 (review R1-5): the internal/dispatch phases open their own
        # snapshots — B/C notices now land BETWEEN deterministic collection
        # and dispatch, so dispatch must see the world as of dispatch time.
        phase_started = time.monotonic()
        with self.db.connection() as job_conn:
            job_conn.execute("BEGIN")
            try:
                self._conflicts_deterministic_collect(ctx, record, job_conn, content, row_sha)
            finally:
                try:
                    job_conn.rollback()
                except sqlite3.Error:
                    pass
        ctx["phase_ms"].append((time.monotonic() - phase_started) * 1000)
        return ctx

    def _new_conflict_ctx(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """0.17.0 Q1: the job context threading the three conflict phases.
        The Qwen budget pool lives here — owned by this job, passed
        explicitly, never on the EvidencePipeline instance (review R2-3)."""
        return {
            "memory_id": int(memory_id),
            "snapshot": snapshot,
            "record": None,          # set by the deterministic phase
            "content": "",
            "content_hash": "",
            "workspace": None,
            "internal_version": 1,
            "embedder": None,
            "publish_done_at": [],   # C2 anchor for the fairness deadline
            "budget": _JobJudgeBudget(),
            "min_budget": SEMANTIC_MIN_PAIR_BUDGET_MS / 1000.0,
            "applying_slots": set(),
            "applying_pairs": set(),
            "degradation_reasons": set(),
            "reasons_seen": [],
            "reached_pair": set(),
            "ordered": [],
            "internal_judge_pairs": [],
            "allowed_memory_ids": None,
            "units_examined": 0,
            "rows_mode": False,
            "no_difference_filtered": 0,
            "memory_pairs_excluded": 0,
            "prefiltered_rows": 0,
            "below_cos_floor": 0,
            "repeatability_skipped": 0,
            "internal_found": 0,
            # 历史命名（Qwen 时代），回执/eval 链消费方钉住故不改名——现役判定引擎是 mDeBERTa
            "internal_qwen_confirmed": 0,
            "surfaced": 0,
            "surfaced_peer_ids": set(),
            "dropped_unlocalizable": 0,
            # 0.17.1 §3.3 outcome counters (judge engine): clear/below-
            # threshold are OBSERVABILITY keys, never degradation.
            "model_clear": 0,
            "model_conflict_below_threshold": 0,
            "model_possible_count": 0,
            "model_notices_capped": 0,
            "backlogged": 0,
            "sweep_evicted": 0,
            "incomplete_reason": None,
            "direct_verdicts": 0,
            "terminal": None,
            "truncated": False,
            "phase_ms": [],
        }

    def _record_job_degradation(
        self, ctx: dict[str, Any], reason: str, sample: "str | None" = None,
    ) -> None:
        """Behaviour change (v3 hardening): each degradation reason is counted
        at most once per task — the pair loops can hit the same technical
        failure for many pairs, and counting every hit made
        _check_degradation_count grow with pair count rather than with
        distinct failure modes."""
        reasons: set[str] = ctx["degradation_reasons"]
        if reason in reasons:
            return
        reasons.add(reason)
        ctx["reasons_seen"].append(reason)
        self._tools._record_check_degradation(reason, sample)

    def _enqueue_backlog_entries(
        self, ctx: dict[str, Any],
        entries: "list[tuple[int, tuple[dict[str, Any], Any, Any, float]]]",
    ) -> tuple[int, int]:
        """0.17.0 P2-4.2: truncation leftovers land in conflict_backlog
        instead of vanishing. Identity = detector version + both
        members@version + row anchors (review A7: a detector bump or a
        member edit invalidates the frozen pair)."""
        enqueued = 0
        evicted_total = 0
        record = ctx["record"]
        memory_id = int(ctx["memory_id"])
        left_version = int(record.get("version") or 1)
        for peer_id, (hit, seg_view, decision, pair_cos) in entries:
            # Gate-v2 G6: the backlog priority uses the SAME score as the
            # live Qwen budget — a stale formula would starve high-band
            # pairs after a truncation.
            from .gates import compute_pair_score

            score = compute_pair_score(
                decision, pair_cos, seg_view.text, str(hit.get("text") or ""),
            )
            right_version = int(hit.get("version") or hit.get("memory_row_version") or 1)
            key_hash = hashlib.sha256(
                "|".join((
                    CONFLICT_DETECTOR_VERSION,
                    f"{memory_id}@{left_version}",
                    f"{peer_id}@{right_version}",
                    f"{seg_view.start_offset}-{seg_view.end_offset}",
                    f"{hit.get('start_offset')}-{hit.get('end_offset')}",
                )).encode("utf-8"),
            ).hexdigest()
            outcome = self.db.conflict_backlog.enqueue(
                candidate_key_hash=key_hash,
                left_memory_id=memory_id, left_version=left_version,
                right_memory_id=int(peer_id), right_version=right_version,
                left_text=str(seg_view.text), right_text=str(hit.get("text") or ""),
                pair_score=score,
            )
            if outcome.get("outcome") in {"queued", "duplicate"}:
                enqueued += 1
            evicted_total += int(outcome.get("evicted") or 0)
        return enqueued, evicted_total

    def _collect_applying_slots(self, ctx: dict[str, Any], snapshot: dict[str, Any]) -> None:
        """Deterministic phase step 1 (r2s-02 split): slot-scoped suppression
        keys for conflict groups currently under application.

        Spec §5/§15.3: while a conflict group is applying, versions produced
        by its apply plan must not re-notify THE SAME conflict. Suppression is
        therefore slot-scoped and applied only after the gate resolves the
        candidate's slot_key, so a genuinely different conflict between the
        same two memories is still examined and surfaced. Validation is
        server-side against the live conflict rows; the trusted context only
        names which row to revalidate."""
        memory_id = int(ctx["memory_id"])
        applying_slots: set[str] = ctx["applying_slots"]
        applying_groups: list[dict[str, Any]] = []
        trusted = TrustedApplyingContext.from_dict(snapshot.get("trusted_applying_context"))
        if trusted is not None:
            live = self.db.get_conflict(trusted.conflict_id)
            if live is not None and live.get("status") == "applying":
                plan = (live.get("apply_summary") or {}).get("plan") or []
                plan_ids = {int(item.get("memory_id") or 0) for item in plan}
                trusted_memory = trusted.memory_id
                trusted_revision = trusted.revision
                trusted_action = trusted.action
                # Spec §15.3 preconditions: applying status, exact revision,
                # target in the plan, and the exact action from that plan step.
                revision_ok = (
                    trusted_revision is not None
                    and int(trusted_revision) == int(live.get("revision") or 0)
                )
                step = next(
                    (item for item in plan if int(item.get("memory_id") or 0) == trusted_memory),
                    None,
                )
                action_ok = (
                    bool(trusted_action)
                    and step is not None
                    and str(step.get("action") or "") == trusted_action
                )
                target_ok = trusted_memory in plan_ids
                if revision_ok and action_ok and target_ok:
                    applying_groups.append(live)
        applying_groups.extend(
            group for group in self.db.list_open_conflicts_for_memory_ids(
                [int(memory_id)], include_applying=True,
            ) if group.get("status") == "applying"
        )
        for group in applying_groups:
            if group.get("slot_key"):
                applying_slots.add(json.dumps(
                    group["slot_key"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                ))
            # 0.17.1 owner A-3: pair-identity suppression set — the judged
            # path has no extracted attribute, so the same PAIR under
            # application suppresses by (memory, peer) identity.
            applying_pairs: set[tuple[int, int]] = ctx.setdefault("applying_pairs", set())
            ids = sorted(
                int(m.get("memory_id") or 0)
                for m in (group.get("member_versions") or [])
                if m.get("memory_id") is not None
            )
            if len(ids) >= 2:
                applying_pairs.add((ids[0], ids[-1]))

    def _collect_internal_pairs(
        self, ctx: dict[str, Any], job_conn: "sqlite3.Connection",
        views_with_vectors: "list[tuple[Any, Any]]",
    ) -> bool:
        """B2（owner 2026-10-03 拍板：内部自查一律走向量+漏斗门）：numpy 内存
        top-k 邻居配对，替换 O(n²) 双循环。

        - 批路径：已发布行向量（current_row_vectors）在收集点前就位，原地调用；
        - streaming 首写：流 drain + publish 后、truncation 判断前调用
          （landed 原生 RowSegment 经 getattr 双访问归一，消费方 .unit_index 不破；
          晚于 truncation 会拔掉 E10①「internal findings land despite cross
          truncation」的落地）；
        - numpy ImportError → 返回 False（调用方记 internal_skipped_no_numpy，
          fail-open 可见；无 numpy 的部署本就无判定引擎）。
        漏斗门（与 cross 同序，直接常量比较——内存匹配已有全对余弦，不复用
        candidate_cos_gate 的 dict 形态）：cos floor 出局 → decide_evidence 值
        提取同值 skip → at-ceiling 且值不同放行（同键异值恰是自查目标形态，
        值门比表面余弦准——与 cross 的 at_ceil→重复语义在此一处刻意分歧）→
        internal_pair_admission（噪音/exists 探针，历史行由前序 job 落库快照
        内可见）照旧 → 入池（internal-first/半池守恒/落地全部不变）。
        已知后果（如实）：rows cap 截断的行无向量不进内部配对（原 n² 对 cap 外
        行同样丢弃，无净损失）。"""
        memory_id = int(ctx["memory_id"])
        internal_version = int(ctx["internal_version"])
        from ..scan_pipeline import internal_pair_admission
        from ..constants import (
            SEMANTIC_CANDIDATE_COS_CEIL,
            SEMANTIC_CANDIDATE_COS_FLOOR,
            SEMANTIC_INTERNAL_MAX_ROWS,
            SEMANTIC_INTERNAL_SELF_KNN_K,
        )

        try:
            import numpy as _np
        except ImportError:
            return False

        internal_judge_pairs: list[tuple[Any, Any, Any]] = ctx["internal_judge_pairs"]
        originators = [
            (view, vec) for view, vec in views_with_vectors
            if str(view.kind) != "subject" and vec
        ][:SEMANTIC_INTERNAL_MAX_ROWS]
        if len(originators) < 2:
            return True
        matrix = _np.array([list(map(float, vec)) for _v, vec in originators], dtype=_np.float32)
        norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = matrix / norms
        sims = unit @ unit.T
        k = min(int(SEMANTIC_INTERNAL_SELF_KNN_K), len(originators) - 1)
        seen_pairs: set[frozenset[int]] = set()
        for i in range(len(originators)):
            # top-k 邻居（排除自身：先置 -1 再取前 k+1）
            row = sims[i].copy()
            row[i] = -1.0  # 自身在 -row 中成最大值，被 argpartition 前段天然排除
            top = _np.argpartition(-row, k)[:k]
            for j in top:
                j = int(j)
                if j == i:
                    continue
                pair_key = frozenset({i, j})
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                seg_a, seg_b = originators[i][0], originators[j][0]
                cos = float(sims[i, j])
                if cos < SEMANTIC_CANDIDATE_COS_FLOOR:
                    ctx["below_cos_floor"] = ctx.get("below_cos_floor", 0) + 1
                    continue
                internal_decision = decide_evidence(seg_a.text, seg_b.text)
                left_v = getattr(internal_decision, "left_value", None)
                right_v = getattr(internal_decision, "right_value", None)
                # 同值判定双通道：值对齐（近同文本各抽值）或 reason 判等
                # （完全同文本不抽值——decide_evidence 直接 ignore/equivalent_value）
                same_value = (
                    str(getattr(internal_decision, "reason", "") or "") == "equivalent_value"
                    or (
                        left_v is not None and right_v is not None
                        and _values_all_equivalent(str(left_v), str(right_v))
                    )
                )
                if cos >= SEMANTIC_CANDIDATE_COS_CEIL:
                    # at-ceiling 分歧点：值不同放行（同键异值=自查目标），值同=重复 skip
                    if same_value:
                        ctx["no_difference_filtered"] = ctx.get("no_difference_filtered", 0) + 1
                        continue
                elif same_value:
                    ctx["no_difference_filtered"] = ctx.get("no_difference_filtered", 0) + 1
                    continue
                admitted = internal_pair_admission(
                    seg_a.text, seg_b.text,
                    (seg_a.start_offset, seg_a.end_offset),
                    (seg_b.start_offset, seg_b.end_offset),
                    internal_decision,
                    exists_probe=lambda a=seg_a, b=seg_b: self.db.internal_conflicts.exists_on_conn(
                        job_conn, int(memory_id), internal_version,
                        a.unit_index, b.unit_index,
                    ),
                )
                if not admitted:
                    continue
                internal_judge_pairs.append((seg_a, seg_b, internal_decision))
        return True

    def _collect_neighbour_screen(
        self, ctx: dict[str, Any], record: dict[str, Any], paired: "list[Any]",
        embedder: "Any", job_conn: "sqlite3.Connection",
    ) -> "list[int] | None":
        """Deterministic phase step 3 (r2s-02 split): Gate-v2 G5 ②″ 记忆级
        一揽子筛选. ONE subject-row coarse KNN builds the neighbour list;
        memory_pair_excluded vets each neighbour on subject/tags alone; the
        sentence KNN then runs ONLY inside the clean list (rowid-IN
        restriction — window slots are not burned on unrelated or
        already-excluded memories). 宽不罚——窄才漏. Returns the effective
        clean list (ctx["allowed_memory_ids"] stays the shared contract —
        the A-cross dispatch loop restricts the cross KNN to this list)."""
        memory_id = int(ctx["memory_id"])
        workspace = ctx["workspace"]
        from .gates import memory_pair_excluded as _pair_excluded
        from ..constants import SEMANTIC_NEIGHBOR_SCREEN
        subject_vec = next(
            (embedding for seg_view, embedding in paired if seg_view.kind == "subject"),
            None,
        ) if paired else None
        if subject_vec is None and embedder is not None:
            # First-write streaming path: paired vectors are all None until
            # publish — embed the subject inline (one embed, milliseconds) so
            # the screen runs on the MAIN write path too, not just
            # re-detections (实施后对抗 review P0：粗筛+通道 C 首写从不执行).
            subject_embed = embedder.embed_text(
                prefix=EMBED_PREFIX_STS, body=str(record.get("subject") or ""),
            )
            subject_vec = subject_embed.embedding or None
        allowed_memory_ids: "list[int] | None" = ctx["allowed_memory_ids"]
        if subject_vec is not None:
            neighbours = self.db.row_knn(
                subject_vec, k=SEMANTIC_NEIGHBOR_SCREEN, workspace=workspace,
                exclude_memory_id=memory_id, conn=job_conn,
                include_subject_rows=True, subject_rows_only=True,
            )
            excluded_ids: set[int] = set()
            own_tags = record.get("tags") or []
            for neighbour in neighbours:
                peer_id_n = int(neighbour["memory_id"])
                if peer_id_n in excluded_ids:
                    continue
                tags_raw_n = neighbour.get("tags")
                peer_tags_n = (
                    json.loads(tags_raw_n) if isinstance(tags_raw_n, str) and tags_raw_n
                    else (tags_raw_n if isinstance(tags_raw_n, list) else [])
                )
                if _pair_excluded(
                    str(record.get("subject") or ""), own_tags,
                    str(neighbour.get("subject") or ""), peer_tags_n,
                ):
                    excluded_ids.add(peer_id_n)
            ctx["allowed_memory_ids"] = [
                int(n["memory_id"]) for n in neighbours
                if int(n["memory_id"]) not in excluded_ids
            ]
            allowed_memory_ids = ctx["allowed_memory_ids"]
            ctx["memory_pairs_excluded"] = len(excluded_ids)
            if not allowed_memory_ids:
                ctx["allowed_memory_ids"] = []  # everything screened out: no KNN at all
                allowed_memory_ids = []
        return allowed_memory_ids

    def _conflicts_deterministic_collect(
        self, ctx: dict[str, Any], record: dict[str, Any],
        job_conn: "sqlite3.Connection", content: str, row_sha: str,
    ) -> None:
        """Q1 相分裂 phase 1 body: indexing publish, G5 screen, candidate
        collection/ordering, internal keeper collection, truncation early-
        exit (E10①). Zero Qwen — the backend is fetched by the later phases.
        Terminal outcomes (embedder/vec-state/truncation) land in
        ctx["terminal"] and skip every later phase."""
        memory_id = int(ctx["memory_id"])
        snapshot = ctx["snapshot"]
        embedder, _ = self._ensure_active_embedder()
        if embedder is None:
            ctx["terminal"] = {"status": "incomplete", "reason": "embedder_unavailable", "notices_created": 0}
            return
        if self.db.get_vec_index_state().get("state") in {"mismatch", "failed"}:
            ctx["terminal"] = {
                "status": "incomplete",
                "reason": "embedding_space_rebuild_required",
                "notices_created": 0,
            }
            return
        ctx["record"] = record
        ctx["content"] = content
        ctx["content_hash"] = row_sha
        ctx["workspace"] = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        ctx["internal_version"] = int(record.get("version") or 1)
        ctx["embedder"] = embedder
        self._collect_applying_slots(ctx, snapshot)
        applying_slots: set[str] = ctx["applying_slots"]

        def backlog_deadline() -> "float | None":
            # C2: detection-phase deadline = max(fairness wall, this job's
            # own budget counted from publish completion) — the shared
            # implementation lives in _job_fair_deadline.
            return _job_fair_deadline(self._semantic_worker, ctx["publish_done_at"])

        max_rows = max(1, SEMANTIC_MAX_ROWS)
        workspace = ctx["workspace"]
        by_peer: dict[int, tuple[dict[str, Any], Any, Any, float]] = {}

        # P2-T2: same digest as the stale check above — the maintained
        # content_sha column (or its recompute fallback), never a fresh hash.
        content_hash = row_sha
        # C2: publish_done_at anchors this job's own detection budget AFTER
        # the index phase (embedding is index work, not conflict budget).
        # Q1 相分裂: the list lives in ctx so the later phases share one
        # anchor (append is in-place — the local alias stays a live view of
        # ctx state).
        publish_done_at: list[float] = ctx["publish_done_at"]
        # 0.17.0 C2: the index duty lives in the job. Read the published row
        # vectors; when they are missing, recover in-job with
        # segment+batch-embed+PUBLISH (invariant: publish precedes any Qwen
        # call — a Qwen stall must never cost the search vectors). The old
        # two-queue chain (evidence worker → semantic forward) is gone.
        row_vectors = self.db.evidence.current_row_vectors(
            int(memory_id), int(record.get("version") or 1), content_hash,
        )
        # C7 streaming: on the first-write path the job does NOT embed inline.
        # The cross-memory loop below consumes a lazy (segment, embedding)
        # stream produced one batch ahead on a single worker thread, so
        # KNN+gates overlap the GPU work; publish_rows lands AFTER collection
        # (still ahead of every Qwen call — invariant unchanged) and the
        # space-rebuild heal moved with it.
        pending_segments: list[Any] = []
        if not row_vectors and embedder is not None:
            from ..rowseg import segment_rows
            pending_segments = list(
                segment_rows(str(record.get("subject") or ""), content)
            )
        elif row_vectors:
            # Already-published rows: the collection phase is over the index,
            # anchor the detection budget now (the streaming path anchors
            # after its post-collection publish instead).
            publish_done_at.append(time.monotonic())
        # C5 (unit retirement): rows are the only candidate source. No rows
        # recoverable in-job (degraded embedder) → the memory stays pending
        # for the backfill; the detection phase sees an empty segment set
        # rather than falling back to units.
        rows_mode = bool(row_vectors) or bool(pending_segments)
        ctx["rows_mode"] = rows_mode
        # Normalized segment view: rows carry row_index, units carry
        # unit_index — the view exposes .unit_index for BOTH so every
        # downstream consumer (internal create, envelopes, member evidence)
        # stays unchanged (P2-3.1 keeps every gate a pure text-pair function;
        # only the input granularity changed).
        from collections import namedtuple
        # C7: kind rides the view so consumers can skip the subject row
        # (index participant, never a pair originator — C3 A+ guard).
        _SegView = namedtuple("_SegView", "text start_offset end_offset unit_index kind")
        # B3 表格段豁免（owner 2026-10-03 拍板，方案 B3）：单一过滤点盖两条
        # 路径——streaming 滤 pending_segments（不嵌入不发布不建 memory_row）；
        # batch 滤读回行（新规则前入库的老表不再参与检测）。公共 helper 与
        # index-only 路径共用（_giant_table_indexes）。
        _exempted_total = 0
        if pending_segments:
            pending_segments, _n = _filter_exempted_segments(pending_segments)
            _exempted_total += _n
        elif row_vectors:
            _kept_segs, _n = _filter_exempted_segments([seg for seg, _e in row_vectors])
            if _n:
                _kept_idx = {
                    int(getattr(k, "row_index", getattr(k, "unit_index", 0)) or 0)
                    for k in _kept_segs
                }
                row_vectors = [
                    (seg, emb) for seg, emb in row_vectors
                    if int(getattr(seg, "row_index", getattr(seg, "unit_index", 0)) or 0) in _kept_idx
                ]
                _exempted_total += _n
        if _exempted_total:
            ctx["table_rows_exempted"] = ctx.get("table_rows_exempted", 0) + _exempted_total
        paired: list[tuple[Any, Any]] = list(row_vectors) or [
            (seg, None) for seg in pending_segments
        ]
        # P2-3.1 值锚定行优先：rows carrying an extractable value lead the cap
        # order (12th round: value features are the conflict predictor; topic
        # similarity is not). Deterministic tiebreak by segment order. C7:
        # ranking precedes batching, so a deadline/cap hit stops later
        # batches and always cuts the lowest-value tail.
        from ..semantic_conflict import _VALUE_RE
        paired.sort(
            key=lambda pair: (
                0 if _VALUE_RE.search(pair[0].text) else 1,
                int(getattr(pair[0], "row_index", 0)),
            )
        )
        seg_views = [
            _SegView(
                seg.text, int(seg.start_offset), int(seg.end_offset),
                int(seg.row_index), str(getattr(seg, "kind", "sentence")),
            )
            for seg, _embedding in paired
        ]
        seg_embeddings = [embedding for _seg, embedding in paired]
        max_segments = max_rows
        segments_capped_reason = "rows_capped"
        # B2：批路径向量在手，就地收集；streaming 延后到流 drain+publish 后
        # （见 truncation 判断前的 landed 收集点）——job_conn 快照对同相位
        # 发布必然不可见，DB KNN 不可用，内存向量是唯一正确来源。
        if not pending_segments:
            ctx["_internal_knn_ok"] = self._collect_internal_pairs(
                ctx, job_conn, list(zip(seg_views, seg_embeddings)),
            )
        # 0.16.2 write-time pre-gates (owner, data-driven): two deterministic
        # filters run in the KNN collection loop, BEFORE the per-peer dedup —
        # a cleared representative would otherwise burn a peer slot that a
        # kept hit of the same peer could have taken (live-library
        # simulation: 86 slots recoverable).
        # Gate 0 (0.16.4 §1, evolution domain): cross-memory notify shapes
        # die above, before all of the following — timeline phenomena are
        # not conflicts.
        # Gate 1 (provenance) is RETIRED in gate-v2 G3 (owner 拍板 1): the
        # real library left entity/scope empty on both sides of true
        # conflicts (#50), so the gate made everyone mutually invisible.
        # Claims attribute alignment carries the same-subject signal now.
        # Gate 2 (difference classifier): no extractable value difference
        # means the pair can never satisfy Qwen's same-attribute-different-
        # value gate. (Counter lives in ctx — the finalize phase reads it.)
        # Spec §15.5: a bounded check that ran out of budget must not later
        # claim checked_no_notice. The two truncation causes report
        # distinctly (2026-09-10 #957/#959 diagnosis: the shared string cost
        # an extra investigation round): the per-memory row cap is
        # rows_capped; the fair job deadline stays notice_budget_exhausted.
        # The cap is checked first so a state where both hold attributes to
        # the more specific cause.
        truncation_reason: str | None = None
        # 0.17.0 P2-3.1: the cross loop walks the normalized segments —
        # row_knn in rows mode (candidates are clean short sentences or
        # header-folded table rows). Rows carry no 'text'-only kind filter
        # (table rows are first-class candidates).
        landed: list[tuple[Any, list[float]]] = []
        streaming = bool(pending_segments)
        ranked_pending: list[Any] = []
        if streaming:
            # C7: iterate the lazy stream (ranking already applied to
            # pending_segments via `paired`). The stream is ALWAYS drained:
            # the producer thread must exit (no leak) and publish needs every
            # embedding — the detection cap/deadline below stop DETECTION,
            # never the collection of vectors (P1/P2 fixes, adversarial
            # review: partial publishes and dead threads are both gone).
            ranked_pending = [seg for seg, _none in paired]
            pair_iter = self._streamed_pairs(
                ranked_pending, embedder, max_segments,
            )
        else:
            pair_iter = iter(zip(seg_views, seg_embeddings))
        # Gate-v2 G4: the sentence prefilter is an OPTIONAL layer — the
        # write path runs it (and the claims-coverage skip, owner 拍板), the
        # scan path never sees this code (gates.row_prefilter is one shared
        # implementation; the scan编排 simply does not call it). Filtered /
        # covered rows do NOT count against rows_examined — their counters
        # are their own receipt keys.
        from .gates import candidate_cos_gate, row_prefilter
        # 0.17.1（评审 P2）：claim-coverage 跳过删后 current_claims 无消费方，
        # 白开一次表查询——整行删除（claims 数据层已连表 DROP，检测线零读取）。
        # KEYED BY ROW INDEX, never object identity: the streaming path
        # yields the raw segments while seg_views are _SegView copies — the
        # same row under two Python objects. row_index is the stable key
        # across both.
        admissible_row_idx = {
            int(getattr(view, "unit_index", getattr(view, "row_index", 0)))
            for view in row_prefilter(seg_views)
        }
        allowed_memory_ids = self._collect_neighbour_screen(
            ctx, record, paired, embedder, job_conn,
        )
        try:
            for seg_view, embedding in pair_iter:
                if streaming:
                    if embedding is None:
                        # Producer signalled a failed item at this position:
                        # the landed prefix is aligned and complete; the rest
                        # waits for the backfill (no partial-with-holes
                        # publish).
                        break
                    landed.append((seg_view, embedding))
                if seg_view.kind == "subject":
                    continue  # C3 A+ guard: indexed, never a pair originator
                if int(getattr(seg_view, "unit_index", getattr(seg_view, "row_index", 0))) not in admissible_row_idx:
                    # Gate-v2 G4: the row failed the sentence prefilter — it
                    # never originates a KNN query and never spends budget.
                    # 0.17.1: claim 覆盖句跳过随 claim 对比通道退役（owner 拍板：
                    # 覆盖句重新从通道 A 发起，否则成检测死区）。
                    ctx["prefiltered_rows"] += 1
                    continue
                if ctx["units_examined"] >= max_segments:
                    truncation_reason = truncation_reason or segments_capped_reason
                    continue  # detection capped; keep draining for publish
                active_deadline = backlog_deadline()
                if active_deadline is not None and time.monotonic() >= active_deadline:
                    truncation_reason = truncation_reason or "notice_budget_exhausted"
                    continue  # budget gone; keep draining for publish
                ctx["units_examined"] += 1
                if allowed_memory_ids is not None and not allowed_memory_ids:
                    continue  # whole neighbourhood screened out
                knn_hits = self.db.row_knn(
                    embedding, k=SEMANTIC_CROSS_KNN_WINDOW, workspace=workspace,
                    exclude_memory_id=memory_id, conn=job_conn,
                    include_subject_rows=False,  # subject rows poison the window
                    include_memory_ids=allowed_memory_ids,
                )
                # Gate-v2 G4 余弦门: true cosine band on fetched vectors —
                # below floor is noise (保安一号), at/above ceil is a
                # duplicate that belongs to the similarity channel, never a
                # conflict report; the band split stays observable.
                hit_vectors = self.db.evidence.row_vectors_for_ids(
                    [int(hit["id"]) for hit in knn_hits], conn=job_conn,
                )
                gated, below_floor_pairs, at_ceil_pairs = candidate_cos_gate(
                    embedding, knn_hits, hit_vectors,
                )
                ctx["below_cos_floor"] += len(below_floor_pairs)
                ctx["repeatability_skipped"] += len(at_ceil_pairs)
                gated_cos = {id(hit): cos for hit, cos in gated}
                for hit in knn_hits:
                    if id(hit) not in gated_cos:
                        continue
                    decision = decide_evidence(seg_view.text, str(hit.get("text") or ""))
                    if decision.action == "ignore":
                        continue
                    # Gate-v2 G5: the whole-memory process-record veto moved
                    # into memory_pair_excluded (once per peer at the screen,
                    # not once per hit here).
                    # 0.16.4 §1: cross-memory evolution domain — the earliest
                    # kill. It happens BEFORE the provenance gate, so a notify
                    # shape never consumes provenance/classifier work, a peer
                    # slot, a sort position, or Qwen budget. Same predicate as
                    # the scan side (§0.5 single implementation).
                    if is_cross_evolution(decision):
                        continue
                    peer_id = int(hit["memory_id"])
                    if classify_pair(
                        seg_view.text, str(hit.get("text") or ""), route=str(decision.reason or ""),
                    ) == "clear":
                        ctx["no_difference_filtered"] += 1
                        continue
                    existing = by_peer.get(peer_id)
                    closer = existing is not None and float(hit.get("distance") or 9) < float(existing[0].get("distance") or 9)
                    # 0.16.4 §1: only check shapes reach here now, so the
                    # notify-priority protection lost its subject — the closer
                    # neighbour of the same peer wins outright. pair_cos rides
                    # the representative for the gate-v2 G6 ranking (band
                    # membership), computed already by the cosine gate.
                    if existing is None or closer:
                        by_peer[peer_id] = (hit, seg_view, decision, gated_cos[id(hit)])
        except Exception:
            # A mid-collection failure must not strand the producer thread
            # (P2 leak fix): closing the generator wakes its queue wait; the
            # daemon thread's remaining put is unbounded-safe (queue depth 1
            # drains via GC'd consumer... belt: producer puts are followed by
            # a final DONE put that may block — the daemon flag keeps the
            # process free to exit regardless).
            try:
                close = getattr(pair_iter, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            raise
        if streaming:
            # C7 (P1 fix, adversarial review): publish ONLY the complete,
            # position-aligned set. A short stream (phase timeout / failed
            # embed) publishes NOTHING — current_row_vectors stays empty, the
            # next job or backfill re-embeds whole; a partial-with-holes or
            # misaligned publish would poison the row store until the next
            # version bump.
            publishable = (
                landed if len(landed) == len(ranked_pending) else []
            )
            if publishable:
                published = self.db.evidence.publish_rows(
                    int(memory_id), int(record.get("version") or 1), content_hash,
                    [seg for seg, _embedding in publishable],
                    [embedding for _seg, embedding in publishable],
                )
                if published.get("published"):
                    vec_state = self.db.get_vec_index_state()
                    if (
                        vec_state.get("state") == "mismatch"
                        and vec_state.get("target_space_id") == embedder.embedding_space_id
                    ):
                        self.db.maybe_complete_space_rebuild(embedder.embedding_space_id)
                # stale_snapshot et al: in-memory vectors served this run; the
                # next job lands the new version's rows.
            elif ranked_pending:
                truncation_reason = truncation_reason or "embed_phase_incomplete"
            publish_done_at.append(time.monotonic())
            # B2 streaming 插点：流 drain 完成后、truncation 判断前——内部
            # keeper 收集必须在 truncation 消费（E10① unannotated 落地）之前。
            # landed 是原生 RowSegment（无 .unit_index）——getattr 双访问归一
            # 为 _SegView 形态（:1305 同款先例），两个消费方读 .unit_index 不破。
            from collections import namedtuple as _nt
            _LandedView = _nt("_LandedView", "text start_offset end_offset unit_index kind")
            _landed_views = [
                (_LandedView(
                    seg.text, int(seg.start_offset), int(seg.end_offset),
                    int(getattr(seg, "unit_index", getattr(seg, "row_index", 0))),
                    str(getattr(seg, "kind", "sentence")),
                ), embedding)
                for seg, embedding in landed
            ]
            ctx["_internal_knn_ok"] = self._collect_internal_pairs(
                ctx, job_conn, _landed_views,
            )
        if truncation_reason:
            # E10① order guarantee: internal findings land BEFORE the cross
            # loop and survive its truncation. The internal Qwen pass has not
            # run yet at this point (it needs the backend fetched by its own
            # phase), so the collected keepers land unannotated here —
            # fail-open, never lost. (Adversarial self-review: without this,
            # a row cap or budget exhaustion mid-collection silently dropped
            # every internal keep pair of this run.)
            for unit_a, unit_b, internal_decision in ctx["internal_judge_pairs"]:
                if self.db.internal_conflicts.create(
                    memory_id=int(memory_id), memory_version=int(ctx["internal_version"]),
                    unit_a=unit_a.unit_index, unit_b=unit_b.unit_index,
                    quote_a=unit_a.text, quote_b=unit_b.text,
                    span_a=[unit_a.start_offset, unit_a.end_offset],
                    span_b=[unit_b.start_offset, unit_b.end_offset],
                    reason=str(internal_decision.reason or ""),
                    detector_version=CONFLICT_DETECTOR_VERSION,
                ):
                    ctx["internal_found"] += 1
            self._record_job_degradation(ctx, truncation_reason)
            early_result: dict[str, Any] = {
                "status": "incomplete", "reason": truncation_reason,
                "notices_created": 0, "reasons_seen": ctx["reasons_seen"],
                # Internal-truncation exits before the cross-memory loop even
                # starts, so no pairs were examined yet (counter not yet live).
                "pairs_examined": 0,
            }
            if ctx["internal_found"]:
                early_result["internal_conflicts"] = ctx["internal_found"]
            # 0.17.0 P2-4.2: collected-but-unexamined candidates go to the
            # backlog (owner design #8) — the receipt says how many.
            early_backlogged, early_evicted = self._enqueue_backlog_entries(ctx, list(by_peer.items()))
            if early_backlogged:
                early_result["backlogged"] = early_backlogged
            if early_evicted:
                early_result["backlog_evicted"] = early_evicted
            # Q1 R1-2: truncation freezes the whole dispatch side (no internal
            # Qwen, no A-cross) — the wrapper still rides B/C on this terminal,
            # matching the pre-split behavior exactly.
            ctx["terminal"] = early_result
            ctx["truncated"] = True
            return

        self._collect_order_candidates(ctx, by_peer)

    def _collect_order_candidates(
        self, ctx: dict[str, Any], by_peer: dict[int, "tuple[dict[str, Any], Any, Any, float]"],
    ) -> None:
        """Deterministic phase step 4 (r2s-02 split): candidate ordering.

        (Q1 相分裂: the backend fetch moved into the internal/dispatch
        phases — the deterministic phase stays Qwen-free.)
        C4 soft ordering (⑦ 定案): rank same-level pairs by subject+tags
        overlap before distance — the Qwen budget should spend on pairs the
        owner's signals (subject/tag) already flag as related. Zero-overlap
        pairs are only ordered later, never excluded. (0.16.4 §1: the
        notify-first key lost its subject — only check shapes remain.)
        0.17.0 P2-3.4: pair_score orders the Qwen budget — value features
        lead (routed numeric + both-sides-extractable), C4 subject/tags
        overlap is the base, row distance the tiebreak. Order-only: a
        single pair's verdict never changes (owner-approved boundary).
        Weights initial; P2-3.2 recalibrates on the noisy corpus.
        0.15.14 (A5): the former surfaced>=max_notice_pairs early stop is
        gone — notices are recorded per-pair inside the loop (write-on-
        discovery), so an early stop only saved Qwen time, which the
        examined-pairs cap now bounds deterministically.
        Q1 (owner D1): SEMANTIC_MAX_EXAMINED_PAIRS is the job-global judge
        pool living in ctx["budget"] (internal + A-cross); the
        per-phase caps live in _JobJudgeBudget."""
        memory_id = int(ctx["memory_id"])
        from ..semantic_conflict import vector_cosine

        hint_vectors = self.db.memories.subject_tags_vectors(
            [memory_id, *[peer_id for peer_id in by_peer]],
        )
        own_vector = hint_vectors.get(int(memory_id))
        if own_vector is None:
            overlap_rank: dict[int, float] = {peer_id: 0.0 for peer_id in by_peer}
        else:
            overlap_rank = {
                peer_id: vector_cosine(own_vector, hint_vectors.get(peer_id))
                for peer_id in by_peer
            }

        def _pair_score(
            peer_id: int, triple: "tuple[dict[str, Any], Any, Any, float]",
        ) -> float:
            from .gates import compute_pair_score

            _hit, _seg, decision, pair_cos = triple
            return compute_pair_score(decision, pair_cos, _seg.text, str(_hit.get("text") or ""))

        def _value_gap(triple: "tuple[dict[str, Any], Any, Any, float]") -> float:
            _hit, _seg, decision, _pair_cos = triple
            if not (decision.left_value and decision.right_value):
                return 0.0
            try:
                left_num = float(normalize_value(decision.left_value).rstrip("ms条次%") or 0)
                right_num = float(normalize_value(decision.right_value).rstrip("ms条次%") or 0)
                return abs(left_num - right_num)
            except (TypeError, ValueError):
                return 0.0

        ctx["ordered"] = sorted(
            by_peer.items(),
            key=lambda item: (
                -_pair_score(item[0], item[1]),
                -_value_gap(item[1]),
                -float(overlap_rank.get(item[0]) or 0.0),
                float(item[1][0].get("distance") or 9),
            ),
        )
