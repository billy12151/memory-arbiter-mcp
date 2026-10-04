"""Local-text evidence indexing and conflict candidate processing."""
from __future__ import annotations

import threading
import time
from typing import Any, TYPE_CHECKING, Iterator

from ..constants import (
    EMBED_PREFIX_STS,
    SEMANTIC_JUDGE_CONTEXT_BEFORE,
    SEMANTIC_JUDGE_CONTEXT_AFTER,
)
from ..evidence import evidence_content_hash
from ..embedder import ManagedEmbedder
from ..semantic_conflict import (
    _values_all_equivalent,
    SemanticBackend,
    decide_evidence,
    direct_value_verdict,
    is_cross_evolution,
    notice_dedupe_key,
    normalize_value,
)
from ._evidence_helpers import (  # noqa: F401
    _TECHNICAL_REASONS as _TECHNICAL_REASONS,
    _retired_gate_slot_key as _retired_gate_slot_key,
    _conflict_notice_payload as _conflict_notice_payload,
    _job_fair_deadline as _job_fair_deadline,
    _giant_table_indexes as _giant_table_indexes,
    _filter_exempted_segments as _filter_exempted_segments,
    filter_exempted_scan_rows as filter_exempted_scan_rows,
)
from ._evidence_judge import (  # noqa: F401
    _JudgeBatch as _JudgeBatch,
    _judge_outcome as _judge_outcome,
    _judge_pair_compat as _judge_pair_compat,
    _pair_diff_anchor as _pair_diff_anchor,
    _JudgePairView as _JudgePairView,
    _JobJudgeBudget as _JobJudgeBudget,
)
from ._evidence_phases import _EvidencePhases
from ._evidence_land import _EvidenceLand

if TYPE_CHECKING:
    from ..tools import MemoryTools
    from ..workers import SemanticConflictWorker

# Technical failures degrade the check route and keep the job incomplete.
# 0.17.1: qwen_* keys renamed judge_*; qwen_unverified is GONE (grounding
# belonged to the slot-extraction paradigm); qwen_budget_exhausted keeps its
# semantics under the judge_ prefix.
class EvidencePipeline(_EvidencePhases, _EvidenceLand):
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools
        self.db = tools.db
        self.settings = tools.settings


    @property
    def _semantic_worker(self) -> "SemanticConflictWorker":
        return self._tools._semantic_worker

    def _ensure_active_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_active_embedder()

    def _ensure_embedder(self) -> "tuple[ManagedEmbedder | None, list[str]]":
        return self._tools._ensure_embedder()

    def _ensure_semantic_backend(self) -> "SemanticBackend | None":
        return self._tools._ensure_semantic_backend()


    def drain_conflict_backlog(self, limit: int = 2) -> int:
        """0.17.0 P2-4.2: idle-worker consumption of the conflict backlog.

        Bounded per call (``limit`` entries); new writes always win because
        the caller only invokes this when the job queue is empty and rechecks
        between entries. A stored extraction replays through the deterministic
        gate without the judge; a judge-less replay with no extraction lands
        nothing (the entry stays pending for a backend-bearing pass — never
        silently completed)."""
        processed = 0
        skipped: list[int] = []  # unprocessable this pass (no backend) — rotate past, never freeze
        judge_pool: list[dict[str, Any]] = []  # A-5 批量判定池（本轮收齐 pass2 一次判）
        pool_ids: list[int] = []  # 已入池条目必须进 take_next 排除表——条目 pass2
        # 前不 complete 且 processed 不增，不排除则 while 每轮重取同一头部
        # 条目无限循环（0.17.1 review P0）。
        backend_holder: dict[str, Any] = {"backend": None}
        while processed < limit:
            entry = self.db.conflict_backlog.take_next(exclude_ids=skipped + pool_ids)
            if entry is None:
                break
            left_id = int(entry["left_memory_id"])
            right_id = int(entry["right_memory_id"])
            # Version drift re-check at consumption time (refresh_stale is the
            # bulk sweep; this is the per-entry guard).
            rows = self.db.get_memories_by_ids([left_id, right_id])
            left = rows.get(left_id)
            right = rows.get(right_id)
            if (
                not left or not right
                or str(left.get("status")) != "active" or str(right.get("status")) != "active"
                or int(left.get("version") or 1) != int(entry["left_version"])
                or int(right.get("version") or 1) != int(entry["right_version"])
            ):
                self.db.conflict_backlog.refresh_stale()
                continue
            extraction = entry.get("extraction") if isinstance(entry.get("extraction"), dict) else None
            left_text = str(entry["left_text"] or "")
            right_text = str(entry["right_text"] or "")
            decision = decide_evidence(left_text, right_text)
            if decision.action == "ignore" or is_cross_evolution(decision):
                self.db.conflict_backlog.complete(int(entry["id"]))
                processed += 1
                continue
            if extraction is None:
                backend = self._ensure_semantic_backend()
                backend_holder["backend"] = backend
                if backend is None:
                    # P2-4 livelock fix: skip-and-rotate instead of breaking —
                    # the entry stays pending for a backend-bearing pass while
                    # lower-scored entries (if any carry stored extraction)
                    # still drain.
                    skipped.append(int(entry["id"]))
                    # R2-W1：该分支零进展（processed 不增），不设上限时
                    # while processed<limit 只能靠 take_next 扫完全部 pending
                    # 才停（每次调用新开 sqlite 连接，积压 500 ⇒ 空闲 tick 每
                    # 5s 一轮上千次连接空转）。超界即停本轮；条目未 complete
                    # 仍 pending，后端出现后照常重试——契约不变。
                    if len(skipped) > 2 * limit:
                        break
                    continue
                embedder, _warnings = self._ensure_embedder()
                direct = direct_value_verdict(left_text, right_text, decision, embedder=embedder)
                if direct is not None:
                    self._record_backlog_notice(
                        left, right, left_text, right_text, decision,
                        str(direct[0]), str(direct[1]), str(direct[2]),
                        reason="deterministic_same_key_value_diff",
                    )
                    self.db.conflict_backlog.complete(int(entry["id"]))
                    processed += 1
                    continue
                # 0.17.1 (owner A-5)：本 drain 轮收集判定池，pass2 一次批前向
                # （先收后判，与 A-cross 同构）。判定映射：notice/possible →
                # notice；clear → 出队；error/technical → 留队（不 complete，
                # 与 backend 不可用分支同契约；owner 2026-09-28 翻转 0.17.0
                # 的技术失败出队丢弃）。
                entry["left"] = left
                entry["right"] = right
                entry["decision"] = decision
                judge_pool.append(entry)
                pool_ids.append(int(entry["id"]))
                continue
            # Stored extraction replays through the deterministic gate (never
            # re-spends Qwen — owner design #8).
            attr = str(extraction.get("attribute") or "")
            value_a = str(extraction.get("value_a") or "")
            value_b = str(extraction.get("value_b") or "")
            if attr and value_a and value_b and normalize_value(value_a) != normalize_value(value_b):
                self._record_backlog_notice(
                    left, right, left_text, right_text, decision,
                    attr, value_a, value_b, reason="backlog_stored_extraction",
                )
            self.db.conflict_backlog.complete(int(entry["id"]))
            processed += 1
        # ── pass 2: one batched judge call for the collected pool ───────────
        if judge_pool and backend_holder["backend"] is not None:
            from ..semantic_judge import PairVerdict as _PV, locate_row, row_window
            # 0.17.1 §2.2（评审 P1-1 修订版）：drain 判定输入上下文化——
            # 版本复核已通过的 fresh content 走 locate_row 定位窗口。注意：
            # backlog 存的是 rowseg 折叠文本（表格行/跨行句），不是裸
            # content 的逐字子串，locate 对这两类结构性失配 → 退化裸行
            # （无 subject）；逐字单行文本才真正上下文化。与 A-cross（直用
            # 存储 span）不同口径，根治需 backlog 落 span 列（待拍板）。
            def _judge_text(mem: "dict[str, Any]", bare: str) -> str:
                content = str(mem.get("content") or "")
                span = locate_row(content, bare)
                if span is None:
                    return bare
                return row_window(content, span[0], span[1],
                                  subject=str(mem.get("subject") or ""),
                                  before=SEMANTIC_JUDGE_CONTEXT_BEFORE,
                                  after=SEMANTIC_JUDGE_CONTEXT_AFTER)
            pairs = [(_judge_text(e["left"], e["left_text"]),
                      _judge_text(e["right"], e["right_text"])) for e in judge_pool]
            verdicts = backend_holder["backend"].judge_pairs(pairs)
            if len(verdicts) != len(pairs):
                verdicts = [_PV("no_conflict", {}, None, "mdeberta:unavailable",
                                error="judge verdict count mismatch")] * len(pairs)
            min_prob = float(self.settings.semantic_conflict_mdeberta_notice_min_prob)
            for entry, verdict in zip(judge_pool, verdicts):
                outcome = _judge_outcome(verdict, min_prob)
                if outcome == "error":
                    # 留队重试（owner A-5）：下个 backend-bearing pass 再来
                    # （留队靠「不调 complete」实现——entry 不出队即重试）
                    continue
                left, right = entry["left"], entry["right"]
                left_text, right_text = entry["left_text"], entry["right_text"]
                decision = entry["decision"]
                # 量纲口径与 A-cross 同一（评审 P3④）：decision 在手比
                # decision 值对，无值不触发守卫（notice 照常落）。
                _dv = (getattr(decision, "left_value", None),
                       getattr(decision, "right_value", None))
                if outcome in ("notice", "possible") and not (
                    _dv[0] and _dv[1] and _values_all_equivalent(str(_dv[0]), str(_dv[1]))
                ):
                    self._record_backlog_notice(
                        left, right, left_text, right_text, decision,
                        # 值=行文本（§3.4 快照原文形态，D1 双通道放行）；
                        # [:400] 与 A-cross 组装同构——超 _MAX_FIELD_CHARS
                        # 的长行会让 escalate 在 intake 重新堵死。
                        # slot 方言统一（疑似#4，owner 2026-10-04 拍板）：
                        # 与 A-cross 同用 12-hex pair anchor（可复现/定长/
                        # 不泄内容；展示层已有 hex→左引文头的转换）
                        _pair_diff_anchor(left_text, right_text),
                        left_text[:400], right_text[:400],
                        reason=f"backlog_judged:{verdict.label}",
                        severity="normal" if outcome == "notice" else "info",
                        model_signal=_JudgePairView.signal(verdict),
                    )
                self.db.conflict_backlog.complete(int(entry["id"]))
                processed += 1
        return processed

    def _record_backlog_notice(
        self, left: dict[str, Any], right: dict[str, Any],
        left_text: str, right_text: str, decision: Any,
        attribute: str, value_a: str, value_b: str, *, reason: str,
        severity: str = "normal", model_signal: "dict[str, Any] | None" = None,
    ) -> None:
        """Record one notice for a backlog pair through the standard channel
        (dedupe/suppression identical to the write path)."""
        left_id = int(left.get("id") or 0)
        right_id = int(right.get("id") or 0)
        left_version = int(left.get("version") or 1)
        right_version = int(right.get("version") or 1)
        # Gate-v2 G3: the old soft-invisible return (metadata entity/scope
        # missing → drop) is retired WITH the provenance gate — keeping it
        # after the storage strip would have silenced EVERY backlog notice
        # forever (entity/scope are gone from all metadata). Slot identity
        # now rides workspace + subject (plan slot_key 连锁).
        slot_key = _retired_gate_slot_key(
            left.get("workspace_canonical") or left.get("workspace"),
            attribute, str(left.get("subject") or ""),
        )
        extra: dict[str, Any] = {
            "anchors": decision.anchors,
            "backlog": True,
        }
        if model_signal is not None:
            extra["model_signal"] = model_signal
        self.db.record_semantic_notice(
            memory_id=left_id, peer_id=right_id, severity=severity,
            notice_type="semantic_evidence",
            title=f"Possible memory change with #{right_id}",
            message=str(decision.reason or "backlog"),
            payload=_conflict_notice_payload(
                reason=reason,
                attribute="backlog",
                slot_key=slot_key,
                left_id=left_id, left_version=left_version,
                left_value_norm=value_a, left_display=value_a,
                left_quote=left_text,
                right_id=right_id, right_version=right_version,
                right_value_norm=value_b, right_display=value_b,
                right_quote=right_text,
                left_content=str(left.get("content") or ""),
                right_content=str(right.get("content") or ""),
                extra=extra,
            ),
            dedupe_key=notice_dedupe_key(
                left_id, right_id, left_version, right_version, "semantic_evidence",
            ),
            left_version=left_version, right_version=right_version,
            source="semantic_evidence",
        )

    def index_memory(self, memory_id: int, record: dict[str, Any] | None = None) -> dict[str, Any]:
        current = record or self.db.get_memory(int(memory_id))
        embedder, warnings = self._ensure_embedder()
        if current is None:
            return {"status": "skipped", "reason": "memory_not_found"}
        if embedder is None:
            return {"status": "skipped", "reason": "embedder_unavailable", "warnings": warnings}
        vec_state = self.db.get_vec_index_state()
        if vec_state.get("state") in {"mismatch", "failed"} and not (
            vec_state.get("state") == "mismatch"
            and vec_state.get("target_space_id") == embedder.embedding_space_id
            and vec_state.get("space_rebuild_active") is True
        ):
            return {
                "status": "skipped",
                "reason": "embedding_space_rebuild_required",
                "warnings": warnings,
            }
        # C5 (unit retirement): index_memory embeds ROWS ONLY (subject row
        # first, sentences/table rows, C3 fallback — see rowseg), batched via
        # embed_texts; the unit tables are no longer written. Remaining
        # callers: boot backfill / repair — the write path indexes inside the
        # semantic job (C2).
        from ..rowseg import segment_rows
        row_segments = segment_rows(
            str(current.get("subject") or ""), str(current.get("content") or ""),
        )
        # A3（0.17.1 修复批）：B3 超长表格段豁免——本路径是升级
        # （vnext_migration 每记忆调用）与 repair 的入口，此前未接线：
        # 存量巨表行会在这里复活（实测 120 行表 → 119 行 + 一次 scan 落地
        # 7021 条 internal_conflicts）。
        row_segments, exempted_count = _filter_exempted_segments(row_segments)
        if not row_segments:
            # 全豁免形态（空 subject + 全表 >100 行）：不 publish——publish_rows
            # ([],[]) 会清空既有行并把记忆留在 missing_row_vector_rows 选集里，
            # 造成每次启动重 embed 的死循环。保留现状、回执可见。
            return {
                "status": "skipped",
                "reason": "all_rows_exempted",
                "table_rows_exempted": exempted_count,
                "warnings": warnings,
            }
        row_embeddings: list[list[float]] = []
        ok = True
        for embed_result in embedder.embed_texts([seg.text for seg in row_segments], prefix=EMBED_PREFIX_STS):
            if not embed_result.embedding:
                ok = False
                break
            row_embeddings.append([float(x) for x in embed_result.embedding])
        if not ok:
            return {"status": "failed", "reason": "empty_embedding"}
        # 0.16.12 P2-T2: prefer the row's maintained content_sha column (set
        # at insert and every content edit) — one hash per write instead of
        # re-hashing here; NULL only on exotic legacy rows, hence the fallback.
        row_sha = str(current.get("content_sha") or "") or evidence_content_hash(
            str(current.get("content") or "")
        )
        published = self.db.evidence.publish_rows(
            int(memory_id), int(current.get("version") or 1), row_sha,
            row_segments, row_embeddings,
        )
        if published.get("published"):
            # Self-heal the embedding-space mismatch: once a rebuild has
            # republished every non-deleted memory in the target space, the
            # vec channel flips back to ready (spec §19 defers fancier
            # space-migration tooling; this unblocks the common recovery).
            vec_state = self.db.get_vec_index_state()
            if (
                vec_state.get("state") == "mismatch"
                and vec_state.get("target_space_id") == embedder.embedding_space_id
            ):
                self.db.maybe_complete_space_rebuild(embedder.embedding_space_id)
        result = {
            "status": "indexed" if published.get("published") else "failed",
            **published,
        }
        if exempted_count:
            result["table_rows_exempted"] = exempted_count
        return result

    @staticmethod
    def _streamed_pairs(
        ranked_segments: list[Any], embedder: Any, max_segments: int,
    ) -> "Iterator[tuple[Any, list[float]]]":
        """C7: lazy (segment, embedding) stream, one batch ahead.

        A daemon producer thread embeds SEMANTIC_STREAM_BATCH_ROWS at a time
        into a depth-1 queue; the main thread's KNN+gates consume the
        previous batch while the GPU works on the next (the llama call holds
        the embedder lock; KNN never touches it, so the overlap is real).
        Ranking happened BEFORE this call, so the cap/phase-timeout stops
        here always cut the lowest-value tail. Embed failures degrade to
        skipped segments (the memory stays pending for the backfill).
        """
        import queue as _queue
        from ..constants import (
            SEMANTIC_EMBED_PHASE_TIMEOUT_MS, SEMANTIC_STREAM_BATCH_ROWS,
        )

        out: "_queue.Queue[Any]" = _queue.Queue(maxsize=1)
        DONE = object()

        def produce() -> None:
            # P1-1 fix (adversarial review): the EMBED set is uncapped —
            # publish_rows needs every row; the detection cap lives in the
            # consumer. Only the phase wall-clock stops submissions here.
            # P1-2 fix: a failed item keeps its POSITION (yielded as None and
            # the stream ends) so the consumer's landed prefix can never
            # misalign segments with vectors.
            phase_started = time.monotonic()
            try:
                stop = False
                for start in range(0, len(ranked_segments), SEMANTIC_STREAM_BATCH_ROWS):
                    if stop or time.monotonic() - phase_started > SEMANTIC_EMBED_PHASE_TIMEOUT_MS / 1000.0:
                        break  # phase cap: stop submitting, tail is lowest-value
                    batch = ranked_segments[start:start + SEMANTIC_STREAM_BATCH_ROWS]
                    results = embedder.embed_texts([seg.text for seg in batch], prefix=EMBED_PREFIX_STS)
                    pairs: list[tuple[Any, Any]] = []
                    for seg, result in zip(batch, results):
                        if not result.embedding:
                            # A failed item ends the stream AT its position;
                            # publishing the prefix stays aligned, the rest
                            # waits for the backfill.
                            out.put(pairs)
                            out.put([(seg, None)])
                            stop = True
                            break
                        pairs.append((seg, [float(x) for x in result.embedding]))
                    if stop:
                        break
                    out.put(pairs)
            except Exception:
                pass  # degraded embedder: short stream, memory stays pending
            finally:
                out.put(DONE)

        threading.Thread(
            target=produce, name="mema-stream-embed", daemon=True,
        ).start()
        while True:
            item = out.get()
            if item is DONE:
                return
            yield from item

    def index_rows_in_job(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """C2 (0.17.0 worker merge): the job's index-only form.

        Segment + batch-embed + publish rows — the indexing duty that used to
        live on the local-text index worker. Reached for index_only snapshots
        (conflict-apply edits §15.3, replay postprocess) and for every job
        when semantic_conflict_on_write="off". Idempotent: current rows for
        the version win, nothing re-embeds.
        """
        from ..constants import SEMANTIC_EMBED_PHASE_TIMEOUT_MS

        record = self.db.get_memory(int(memory_id))
        if record and record.get("status") == "pending":
            return {"status": "skipped", "reason": "pending_workspace_activation",
                    "index_only": True, "notices_created": 0}
        # 0.17.0 修复（owner 2026-09-25 拍板，C6「非 deleted 保全」口径）：重建
        # 选集含全部非 deleted 状态（active/retired/superseded/expired），而本路径
        # 此前对非 active 一律 memory_not_active 拒绝 → 非活跃记忆的 pending 永远
        # 清不空 → 空间翻转永不触发 → 向量通道锁死（真库实测 106 superseded +
        # 117 retired 卡死翻转）。放行全部非 deleted；检测仍只对 active（上方
        # pending-activation skip 不变）。
        status_now = str((record or {}).get("status") or "")
        if not record or status_now == "deleted":
            return {"status": "incomplete", "reason": "memory_not_active",
                    "index_only": True, "notices_created": 0}
        version = int(record.get("version") or 1)
        if version != int(snapshot.get("version") or 1):
            return {"status": "incomplete", "reason": "stale_snapshot",
                    "index_only": True, "notices_created": 0}
        content = str(record.get("content") or "")
        row_sha = str(record.get("content_sha") or "") or evidence_content_hash(content)
        # _ensure_embedder (NOT the space-gated _ensure_active_embedder): a
        # mismatch rebuild drives this path precisely to WRITE into the new
        # space — the gate would return None and deadlock the flip.
        embedder, _ = self._ensure_embedder()
        if embedder is None:
            return {"status": "incomplete", "reason": "embedder_unavailable",
                    "index_only": True, "notices_created": 0}
        vec_state = self.db.get_vec_index_state()
        if vec_state.get("state") == "failed" or (
            vec_state.get("state") == "mismatch"
            and vec_state.get("target_space_id") != embedder.embedding_space_id
        ):
            return {"status": "incomplete", "reason": "embedding_space_rebuild_required",
                    "index_only": True, "notices_created": 0}
        if vec_state.get("state") in {"mismatch", "failed"}:
            # C2: a mismatch rebuild drives EVERY index_only job through this
            # path — existing rows live in the OLD space, so "already current"
            # would strand the flip. Republish unconditionally; the heal at
            # the tail settles ready once the whole index is in the target
            # space. (The mismatch guard for ordinary detection jobs lives in
            # _conflicts_deterministic_collect.)
            pass
        elif self.db.evidence.current_row_vectors(int(memory_id), version, row_sha):
            return {"status": "indexed", "reason": "already_current",
                    "index_only": True, "notices_created": 0}
        from ..rowseg import segment_rows
        phase_started = time.monotonic()
        segments = segment_rows(str(record.get("subject") or ""), content)
        # B3：index-only 路径同样豁免超长表格段（与检测相共用 helper）——
        # on_write=off / replay postprocess / conflict-apply edits 不得成为
        # 巨表嵌入的后门。
        segments, _exempted = _filter_exempted_segments(segments)
        if not segments:
            # 全豁免形态（空 subject + 全表 >100 行）：不 publish——与
            # index_memory 的 A3 守卫同款。publish_rows([],[]) 会把记忆留在
            # missing_row_vector_rows 选集（NOT EXISTS 谓词永真）且绕过
            # already_current 早退，此后的每个 index_only job 都重复一次
            # 空 publish 事务；回执照实报 skipped、豁免可见。
            return {"status": "skipped", "reason": "all_rows_exempted",
                    "index_only": True, "notices_created": 0,
                    "table_rows_exempted": int(_exempted)}
        results = embedder.embed_texts([segment.text for segment in segments], prefix=EMBED_PREFIX_STS)
        if time.monotonic() - phase_started > SEMANTIC_EMBED_PHASE_TIMEOUT_MS / 1000.0:
            # The llama call itself cannot be interrupted mid-flight; the cap
            # marks the job incomplete (retry) and keeps the stall observable
            # instead of silently treating a wedged embedder as success.
            return {"status": "incomplete", "reason": "embed_phase_timeout",
                    "index_only": True, "notices_created": 0}
        vectors: list[list[float]] = []
        for embed_result in results:
            if not embed_result.embedding:
                return {"status": "incomplete", "reason": "empty_embedding",
                        "index_only": True, "notices_created": 0}
            vectors.append([float(x) for x in embed_result.embedding])
        published = self.db.evidence.publish_rows(
            int(memory_id), version, row_sha, segments, vectors,
        )
        if published.get("published"):
            # C2: the space-rebuild self-heal moved with the indexing duty —
            # once every non-deleted memory has rows in the target space, the
            # vec channel flips back to ready (spec §19).
            vec_state = self.db.get_vec_index_state()
            if (
                vec_state.get("state") == "mismatch"
                and vec_state.get("target_space_id") == embedder.embedding_space_id
            ):
                self.db.maybe_complete_space_rebuild(embedder.embedding_space_id)
            receipt = {"status": "indexed", "index_only": True,
                       "row_count": int(published.get("row_count") or len(segments)),
                       "notices_created": 0}
            if _exempted:
                # B3：豁免可见不静默（index-only 路径与检测相同口径）
                receipt["table_rows_exempted"] = int(_exempted)
            return receipt
        return {"status": "incomplete", "reason": f"publish_{published.get('outcome')}",
                "index_only": True, "notices_created": 0}

