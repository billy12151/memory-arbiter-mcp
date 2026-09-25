"""Local-text evidence indexing and conflict candidate processing."""
from __future__ import annotations

import hashlib
import sqlite3
import json
import threading
import time
from typing import Any, Callable, TYPE_CHECKING, Iterator

from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..constants import (
    EMBED_PREFIX_STS,
    SEMANTIC_JOB_TIMEOUT_MS,
    SEMANTIC_MAX_EXAMINED_PAIRS,
    SEMANTIC_INTERNAL_QWEN_MAX_PAIRS,
    SEMANTIC_CROSS_KNN_WINDOW,
    SEMANTIC_MAX_ROWS,
    SEMANTIC_MIN_PAIR_BUDGET_MS,
)
from ..difference_classifier import classify_pair
from .gates import dispatch_hint_text, qwen_dispatch
from ..evidence import evidence_content_hash
from ..models import TrustedApplyingContext
from ..embedder import ManagedEmbedder
from ..semantic_conflict import (
    PAIR_PROMPT_VERSION,
    PairGateResult,
    SemanticBackend,
    decide_evidence,
    direct_value_verdict,
    evaluate_single_direction_extraction,
    is_cross_evolution,
    notice_dedupe_key,
    normalize_value,
    signal_extraction,
)
from ..text import canon_entity, canon_scope

if TYPE_CHECKING:
    from ..tools import MemoryTools
    from ..workers import SemanticConflictWorker

# Technical failures degrade the check route and keep the job incomplete.
_TECHNICAL_REASONS = {
    "qwen_timeout", "qwen_unavailable", "qwen_backend_error",
    "qwen_invalid_output", "qwen_budget_exhausted", "notice_budget_exhausted",
    "rows_capped", "pairs_examined_capped",
    "notice_write_failed",
}


_NEGATION_COMPILED = None


def _negation_opposition(left_text: str, right_text: str) -> bool:
    """G6: the G4 negation vocab hitting EXACTLY ONE side — an independent
    opposition signal, never mixed with the value-differ signal."""
    global _NEGATION_COMPILED
    import re as _re
    from ..semantic_conflict import _NEGATION_WORDS
    if _NEGATION_COMPILED is None:
        _NEGATION_COMPILED = _re.compile(_NEGATION_WORDS, _re.IGNORECASE)
    left_hit = bool(_NEGATION_COMPILED.search(left_text or ""))
    right_hit = bool(_NEGATION_COMPILED.search(right_text or ""))
    return left_hit != right_hit


def _values_differ_norm(decision: Any) -> bool:
    """G6 owner 修正: value-equal pairs are settled at adjudication — the
    ordering bonus is only for opposing evidence; text pairs whose equality
    cannot be decided score 0 (Qwen case c)."""
    left_value = getattr(decision, "left_value", None)
    right_value = getattr(decision, "right_value", None)
    if not left_value or not right_value:
        return False
    try:
        return normalize_value(left_value) != normalize_value(right_value)
    except Exception:
        return True


def _retired_gate_slot_key(workspace: Any, attribute: Any, subject: Any) -> dict[str, str]:
    """Gate-v2 G3 slot identity without metadata provenance (owner 拍板):
    {"entity": workspace 名, "attribute": 抽取属性, "scope": subject 前 32 字}.
    Identity key only — historical groups keep their old slot_keys untouched,
    new groups speak the new dialect. canon_* stay for storage-side parity
    (both sides of the B-C4 comparison canonicalise identically)."""
    return {
        "entity": canon_entity(str(workspace or "")),
        "attribute": str(attribute or ""),
        "scope": canon_scope(str(subject or "")[:32]),
    }


def _conflict_notice_payload(
    *,
    reason: str,
    attribute: str,
    slot_key: "dict[str, str] | None" = None,
    left_id: int = 0, left_version: int = 1,
    left_value_norm: str = "", left_display: str = "", left_quote: Any = "",
    right_id: int = 0, right_version: int = 1,
    right_value_norm: str = "", right_display: str = "", right_quote: Any = "",
    left_content: str = "", right_content: str = "",
    attr_cos: "float | None" = None,
    left_evidence_extra: "dict[str, Any] | None" = None,
    right_evidence_extra: "dict[str, Any] | None" = None,
    left_member_extra: "dict[str, Any] | None" = None,
    right_member_extra: "dict[str, Any] | None" = None,
    extra: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    """Shared payload assembler for all notice sites (0.17.0 review R2
    r2s-01): the five per-site copies had already drifted — content
    fingerprints existed ONLY on the A-cross leg, so four channels'
    notices had no basis for staleness invalidation. Owner 拍板（2026-09-26
    补齐）：left/right content hashes ride EVERY payload now (same
    evidence_content_hash 口径 as the A-cross leg). A hash is emitted only
    when the site had that side's content in hand — absence means unknown
    (never a hash of the empty string)."""
    left_evidence: dict[str, Any] = {"text": left_quote}
    if left_evidence_extra:
        left_evidence.update(left_evidence_extra)
    right_evidence: dict[str, Any] = {"text": right_quote}
    if right_evidence_extra:
        right_evidence.update(right_evidence_extra)
    payload: dict[str, Any] = {
        "route": "notice_ready",
        "reason": reason,
        "slot_key": slot_key,
        "slot_provenance": {
            "entity": "workspace", "scope": "subject", "attribute": attribute,
        },
        "member_versions": [
            {"memory_id": left_id, "version": left_version,
             "value": left_value_norm,
             "evidence": {"quote": left_quote, **(left_member_extra or {})}},
            {"memory_id": right_id, "version": right_version,
             "value": right_value_norm,
             "evidence": {"quote": right_quote, **(right_member_extra or {})}},
        ],
        "value_groups": [
            {"normalized_value": left_value_norm, "display_value": left_display,
             "members": [f"{left_id}@{left_version}"]},
            {"normalized_value": right_value_norm, "display_value": right_display,
             "members": [f"{right_id}@{right_version}"]},
        ],
        "candidate_key": {
            "detector_version": CONFLICT_DETECTOR_VERSION,
            "members": sorted([f"{left_id}@{left_version}", f"{right_id}@{right_version}"]),
            "evidence": [],
        },
        "left_evidence": left_evidence,
        "right_evidence": right_evidence,
    }
    if left_content:
        payload["left_content_hash"] = evidence_content_hash(left_content)
    if right_content:
        payload["right_content_hash"] = evidence_content_hash(right_content)
    if attr_cos is not None:
        payload["attr_cos"] = round(float(attr_cos), 4)
    if extra:
        payload.update(extra)
    return payload


def _attr_cos_or_none(
    embedder: "Any", forward: "Any",
) -> "float | None":
    """P2-3.3 attr-vector cosine for the single-direction gate: None when the
    attributes already match STRICTLY (no embed spent) or no embedder — the
    gate then falls back to strict-equality-only (legacy/scan semantics)."""
    from ..semantic_conflict import normalize_attribute, vector_cosine
    if forward is None or embedder is None:
        return None
    attr_a = getattr(forward, "attribute_a", None)
    attr_b = getattr(forward, "attribute_b", None)
    if not attr_a or not attr_b:
        return None
    if normalize_attribute(str(attr_a)) == normalize_attribute(str(attr_b)):
        return None
    ea = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=str(attr_a))
    eb = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=str(attr_b))
    if not ea.embedding or not eb.embedding:
        return None
    # Degenerate-vector guard: byte-identical embeddings carry ZERO
    # discrimination between the two attribute strings (fake embedders
    # collapse distinct strings onto one vector; a real model never does).
    # Reporting cos=1.0 here would wave mismatched attributes through on
    # testimony the embedder cannot actually give — fall back to strict.
    if list(ea.embedding) == list(eb.embedding):
        return None
    return vector_cosine(list(ea.embedding), list(eb.embedding))


def _conflict_envelope(memory: dict[str, Any], quote: str) -> dict[str, Any]:
    """0.17.0 Q1 (相分裂): pair envelopes were a job closure, now a pure
    function shared by the internal and cross dispatch phases. Gate-v2 G3:
    metadata.entity/scope retired — the prompt keeps the (now always empty)
    metadata slot so the protocol shape is stable."""
    return {
        "quote": quote[:1000], "subject": str(memory.get("subject") or "")[:200],
        "tags": list(memory.get("tags") or [])[:20],
        "workspace_canonical": memory.get("workspace_canonical") or memory.get("workspace"),
        "memory_id": int(memory.get("id") or 0), "version": int(memory.get("version") or 1),
        "event_time": memory.get("event_time"),
        "metadata": {},
    }


def _job_fair_deadline(semantic_worker: "SemanticConflictWorker", publish_done_at: "list[float]") -> "float | None":
    """0.17.0 Q1 (相分裂): the fairness deadline was a job closure, now a
    module function so the internal/cross dispatch phases AND the wrapper-
    orchestrated channel C can share one wall-clock semantics. Detection-
    phase deadline = max(fairness wall, this job's own budget counted from
    publish completion); identical logic to the former closure."""
    value = semantic_worker.pending_job_deadline(
        SEMANTIC_JOB_TIMEOUT_MS / 1000.0,
    )
    if value is None:
        # Idle queue: the old contract stands — no wall, no cap; an
        # in-flight Qwen pair runs to completion.
        return None
    wall = float(value)
    if publish_done_at:
        own = publish_done_at[0] + SEMANTIC_JOB_TIMEOUT_MS / 1000.0
        if wall <= time.monotonic():
            # The fairness wall has ALREADY blown (the oldest queued
            # job waited past its budget) — truncation wins over the
            # own-anchor extension; never let one slow job park the
            # whole queue behind max(wall, own).
            return wall
        # Busy queue, wall still ahead: the embed phase must not eat
        # the detection budget, so count this job's budget from
        # publish completion — but never SHORTEN the wall other
        # queued jobs already rely on.
        return max(wall, own)
    return wall


class _JobQwenBudget:
    """0.17.0 Q1 (owner D1): SEMANTIC_MAX_EXAMINED_PAIRS upgraded from
    "channel A's internal pair cap" to the write job's GLOBAL Qwen pair
    budget — internal (protection cap) → channel C (draws freely, can
    overdraw the pool but is never blocked by it) → A-cross (residual
    max(0, total − internal − C); exhaustion turns the cross loop's
    break into a continue — deterministic verdicts still land).

    The pool lives in this object, owned by the orchestrator (wrapper or
    standalone process_conflicts) and passed explicitly into the phases —
    never on the EvidencePipeline instance (review R2-3: no cross-job
    mutable state on the shared object). Channel B's bridge keeps its own
    CLAIMS_BRIDGE_MAX_PER_WRITE cap and does NOT draw on this pool (owner
    plan §3.1 scope: internal + C + A-cross)."""

    def __init__(
        self,
        total: int = SEMANTIC_MAX_EXAMINED_PAIRS,
        internal_cap: int = SEMANTIC_INTERNAL_QWEN_MAX_PAIRS,
    ) -> None:
        self.total = max(1, int(total))
        self.internal_cap = max(0, int(internal_cap))
        self.internal_used = 0
        self.channel_c_used = 0
        self.a_cross_used = 0
        self.a_cross_dispatch_skipped = False

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.internal_used - self.channel_c_used - self.a_cross_used)

    def spend_internal(self) -> bool:
        """Internal Qwen dispatch: protection-capped AND pool-bounded."""
        if self.internal_used >= self.internal_cap or self.remaining <= 0:
            return False
        self.internal_used += 1
        return True

    def spend_channel_c(self) -> None:
        """Channel C draws the pool down but is never blocked by it (D3)."""
        self.channel_c_used += 1

    def spend_a_cross(self) -> bool:
        """A-cross Qwen dispatch: residual pool only; exhaustion records the
        dispatch skip (receipt key) and the caller continues, never breaks."""
        if self.remaining <= 0:
            self.a_cross_dispatch_skipped = True
            return False
        self.a_cross_used += 1
        return True

    def receipt_block(self) -> "dict[str, Any] | None":
        """§3.3 conditional receipt block — absent entirely when nothing was
        deducted and nothing was skipped (zero values never appear)."""
        block: dict[str, Any] = {}
        if self.internal_used:
            block["internal"] = self.internal_used
        if self.channel_c_used:
            block["channel_c"] = self.channel_c_used
        if self.a_cross_used:
            block["a_cross"] = self.a_cross_used
        if self.a_cross_dispatch_skipped:
            block["a_cross_dispatch_skipped"] = True
        return block or None

    @property
    def pairs_examined(self) -> int:
        """Job-global pairs_examined (§3.3): internal + C + A-cross."""
        return self.internal_used + self.channel_c_used + self.a_cross_used


def _coexistence_by_attr(claims: "list[dict[str, Any]]") -> dict[str, list[str]]:
    """A4 coexistence map derived locally from claims rows in hand (0.17.0
    review R2 CL-1): groups distinct value_norms by attr_norm in one pass —
    replaces per-attr coexisting_values() re-queries (up to 20 fresh
    connections + ~400 row re-reads per write on the sync path)."""
    grouped: dict[str, list[str]] = {}
    for row in claims:
        attr = str(row["attr_norm"])
        values = grouped.setdefault(attr, [])
        value = str(row["value_norm"])
        if value not in values:
            values.append(value)
    return grouped


class EvidencePipeline:
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

    def _own_claim_vectors(self, memory_id: int, version: int) -> dict[int, list[float]]:
        """Claim-id → vector for one memory's current-version claims (B and C
        shared the identical prefetch block; r2s consolidation)."""
        rows: dict[int, list[float]] = {}
        with self.db.connection() as conn:
            for row in conn.execute(
                """SELECT c.id AS cid, v.embedding FROM memory_claims c
                   LEFT JOIN memory_claim_vec v ON v.id=c.id
                   WHERE c.memory_id=? AND c.memory_version=?""",
                (int(memory_id), int(version)),
            ).fetchall():
                if row["embedding"] is not None:
                    rows[int(row["cid"])] = self.db.evidence._blob_to_vector(
                        bytes(row["embedding"])
                    )
        return rows

    def _claims_exact_lane(
        self, memory_id: int, version: int, record: dict[str, Any],
        own_claims: "list[dict[str, Any]]", skip: "set[int]",
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """0.17.0 review R2（owner 拍板接线）：attr_norm 精确键通道。

        与 KNN 通道（check_claims_conflicts 主体）共享同一套后续闸（值对立
        + A4 共存否决 + pair-closure 去重 + 通知帽），但不依赖向量：不受
        KNN 的 k=10 窗口限制、不需要 sqlite-vec——无 vec 环境是唯一还能
        产出 claims 冲突 notice 的通道。仍有自己的候选上限
        (CLAIMS_EXACT_CANDIDATE_LIMIT)，触顶记 state["exact_capped"]
        （可观测，不静默）。产出的 attr 记入 fired_attrs（A3：KNN 侧同
        attr 自动跳过）；surfaced peers 并入跨通道 skip 集合。读/写失败
        上抛（wrapper 的 claims_channel_error loud 路径），与 KNN lane
        一致——不在这里吞。

        ``state``（跨调用共享的可变计数）：notices / capped /
        versional_vetoed / fired_attrs / surfaced / exact_capped /
        own_coexistence / peer_coexistence。返回 {"checked": 精确候选检查数}。"""
        own_coexistence: dict[str, list[str]] = state["own_coexistence"]
        peer_coexistence: dict[int, dict[str, list[str]]] = state["peer_coexistence"]
        fired_attrs: set[str] = state["fired_attrs"]
        surfaced: list[int] = state["surfaced"]
        checked = 0
        if not own_claims:
            return {"checked": 0}
        from ..constants import (
            CLAIMS_EXACT_CANDIDATE_LIMIT, CLAIMS_MAX_NOTICES_PER_WRITE,
        )
        from ..semantic_conflict import attr_is_versional

        with self.db.connection() as conn:
            for claim in own_claims:
                # D1 (owner 2026-09-23): version-like attrs are expected
                # timeline evolution — skip before any channel work and
                # count. Judged on the RAW attr: attr_norm strips spaces,
                # which defeats the vocabulary's word boundaries
                # (releasenotes).
                if attr_is_versional(str(claim.get("attr") or claim["attr_norm"])):
                    state["versional_vetoed"] = int(state["versional_vetoed"]) + 1
                    continue
                candidates = self.db.claims.attr_conflict_candidates(
                    attr_norm=str(claim["attr_norm"]),
                    exclude_memory_id=int(memory_id),
                    limit=CLAIMS_EXACT_CANDIDATE_LIMIT,
                    conn=conn,
                )
                if len(candidates) >= CLAIMS_EXACT_CANDIDATE_LIMIT:
                    state["exact_capped"] = int(state["exact_capped"]) + 1
                for hit in candidates:
                    checked += 1
                    peer_id = int(hit["memory_id"])
                    if peer_id in skip:
                        continue  # evidence channel already surfaced this pair
                    if str(claim["attr_norm"]) in fired_attrs:
                        continue  # A3: same attr already reported this write
                    # D1 hit-level fallback (same as the KNN lane): the
                    # peer's RAW attr can carry version semantics even
                    # when attr_norm collides with a non-versional own
                    # attr (attr_norm strips spaces: "release notes" →
                    # "releasenotes", defeating the vocabulary's word
                    # boundaries) — exempt without counting as fired.
                    if attr_is_versional(str(hit["attr"] or hit["attr_norm"])):
                        state["versional_vetoed"] = (
                            int(state["versional_vetoed"]) + 1
                        )
                        continue
                    if str(hit["value_norm"]) == str(claim["value_norm"]):
                        continue
                    # A4: either side declaring multiple values for the
                    # attr is self-coexistence, not opposition.
                    if len(own_coexistence.get(str(claim["attr_norm"]), ())) > 1:
                        continue
                    peer_attrs = peer_coexistence.get(peer_id)
                    if peer_attrs is None:
                        peer_attrs = _coexistence_by_attr(
                            self.db.claims.current_claims(peer_id),
                        )
                        peer_coexistence[peer_id] = peer_attrs
                    if len(peer_attrs.get(str(hit["attr_norm"]), ())) > 1:
                        continue
                    # Cross-channel dedup (appendix C 7): a pair the
                    # evidence channel already settled this version never
                    # double-fires.
                    if self.db.semantic_notices.is_semantic_pair_closed_on_conn(
                        conn, int(memory_id), peer_id, version,
                        int(hit["memory_version"] or 1),
                    ):
                        continue
                    if int(state["notices"]) >= CLAIMS_MAX_NOTICES_PER_WRITE:
                        state["capped"] = int(state["capped"]) + 1
                        continue
                    slot_key = _retired_gate_slot_key(
                        record.get("workspace_canonical") or record.get("workspace"),
                        str(hit["attr_norm"]), str(record.get("subject") or ""),
                    )
                    peer_record = self.db.get_memory(peer_id) or {}
                    outcome = self.db.record_semantic_notice(
                        memory_id=int(memory_id), peer_id=peer_id,
                        severity="normal", notice_type="claim_conflict",
                        title=f"Claim conflict with #{peer_id}",
                        message=(
                            f"claims attr {claim['attr']} value differs"
                            " (exact attr match)"
                        ),
                        payload=_conflict_notice_payload(
                            reason="claim_attr_exact_gate",
                            attribute="claims_channel_exact",
                            slot_key=slot_key,
                            left_id=int(memory_id), left_version=version,
                            left_value_norm=str(claim["value_norm"]),
                            left_display=str(claim["value"]),
                            left_quote=str(claim["value"]),
                            right_id=peer_id,
                            right_version=int(hit["memory_version"] or 1),
                            right_value_norm=str(hit["value_norm"]),
                            right_display=str(hit["value"]),
                            right_quote=str(hit["value"]),
                            left_content=str(record.get("content") or ""),
                            right_content=str(peer_record.get("content") or ""),
                            attr_cos=1.0,
                            extra={
                                "source": "claim_conflict",
                                "claims_channel": True,
                                "claims_exact_lane": True,
                            },
                        ),
                        dedupe_key=notice_dedupe_key(
                            int(memory_id), peer_id, version,
                            int(hit["memory_version"] or 1), "claim_conflict",
                        ),
                        left_version=version,
                        right_version=int(hit["memory_version"] or 1),
                        source="claim_conflict",
                    )
                    if outcome.get("outcome") == "created":
                        state["notices"] = int(state["notices"]) + 1
                        fired_attrs.add(str(claim["attr_norm"]))
                        surfaced.append(peer_id)
        return {"checked": checked}

    def check_claims_conflicts(
        self, memory_id: int, snapshot: dict[str, Any],
        skip_peers: "set[int] | None" = None,
    ) -> dict[str, Any]:
        """0.17.0 P2-5.3: the zero-Qwen claims channel (owner decision #6).

        Every claim of the freshly-written memory searches the claim-vector
        KNN (same workspace, active, current version); attr_cos ≥
        CLAIM_ATTR_TAU + value_norm difference + coexistence veto (A4) +
        evidence-channel pair-closure dedup → notice. Gate-v2 G3: the soft
        metadata-provenance gate is retired — claims attr alignment IS the
        identity signal now. Bounded by CLAIMS_MAX_NOTICES_PER_WRITE
        (review A3); overflow is counted, never silent."""
        from ..constants import (
            CLAIMS_BRIDGE_MAX_PER_WRITE,
            CLAIMS_MAX_NOTICES_PER_WRITE,
            CLAIM_ATTR_TAU,
        )
        from ..semantic_conflict import attr_is_versional, vector_cosine

        record = snapshot if snapshot.get("content") is not None else (
            self.db.get_memory(int(memory_id)) or {}
        )
        version = int(record.get("version") or 1)
        workspace = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        own_claims = self.db.claims.current_claims(int(memory_id))
        if not own_claims:
            return {"channel_b_checked": 0, "channel_b_notices": 0}
        own_coexistence: dict[str, list[str]] = _coexistence_by_attr(own_claims)
        peer_coexistence: dict[int, dict[str, list[str]]] = {}
        # 0.17.0 review R2（owner 拍板接线）：精确键通道——attr_norm 完全
        # 相等的同属性对立不需要向量、不受 KNN 的 k=10 窗口限制、也不依赖
        # sqlite-vec，在 KNN 之前跑（候选上限 CLAIMS_EXACT_CANDIDATE_LIMIT，
        # 触顶 channel_b_exact_capped 可观测）。产出的 attr 记入 fired_attrs
        # （KNN 侧同 attr 自动跳过，A3）；无向量环境也可单独产出确定性 notice。
        skip = skip_peers or set()
        surfaced: list[int] = []  # Q1 R1-3: peers this channel surfaced — the wrapper unions them into the A-cross skip set (dedup direction flip)
        notices = 0
        capped_count = 0
        checked = 0
        exact_capped = 0
        versional_vetoed = 0  # D1: evolution exemption must stay observable
        fired_attrs: set[str] = set()  # A3: one notice per attr per write
        exact_state: dict[str, Any] = {
            "notices": notices, "capped": capped_count,
            "versional_vetoed": versional_vetoed,
            "exact_capped": exact_capped,
            "fired_attrs": fired_attrs, "surfaced": surfaced,
            "own_coexistence": own_coexistence,
            "peer_coexistence": peer_coexistence,
        }
        exact_lane = self._claims_exact_lane(
            memory_id, version, record, own_claims, skip, exact_state,
        )
        notices = int(exact_state["notices"])
        capped_count = int(exact_state["capped"])
        versional_vetoed = int(exact_state["versional_vetoed"])
        exact_capped = int(exact_state["exact_capped"])
        own_coexistence = exact_state["own_coexistence"]
        peer_coexistence = exact_state["peer_coexistence"]
        if not self.db.state.sqlite_vec_available:
            # 精确键通道不依赖向量——无 vec 环境仍产出确定性 notice 后返回。
            vec_free: dict[str, Any] = {
                "channel_b_checked": int(exact_lane["checked"]),
                "channel_b_notices": notices,
                "channel_b_exact_checked": int(exact_lane["checked"]),
            }
            if capped_count:
                vec_free["channel_b_capped"] = capped_count
            if exact_capped:
                vec_free["channel_b_exact_capped"] = exact_capped
            if versional_vetoed:
                vec_free["channel_b_versional_vetoed"] = versional_vetoed
            if surfaced:
                # Internal cross-channel key: consumed by the job wrapper for
                # the A-cross skip set (R1-3), never a receipt field
                # (underscore: wrapper MUST pop it — r2s-08 one-convention rule).
                vec_free["_surfaced_peers"] = sorted(set(surfaced))
            return vec_free
        own_rows = self._own_claim_vectors(int(memory_id), version)
        if not own_rows:
            # KNN leg can't run (claim vectors still absent), but the exact
            # lane ran first and may already have created notices — carry its
            # real result, same shape as the vec-free branch, plus a reason
            # marking that the KNN leg did not run.
            pending: dict[str, Any] = {
                "channel_b_checked": int(exact_lane["checked"]),
                "channel_b_notices": notices,
                "channel_b_exact_checked": int(exact_lane["checked"]),
                "reason": "vectors_pending",
            }
            if capped_count:
                pending["channel_b_capped"] = capped_count
            if exact_capped:
                pending["channel_b_exact_capped"] = exact_capped
            if versional_vetoed:
                pending["channel_b_versional_vetoed"] = versional_vetoed
            if surfaced:
                pending["_surfaced_peers"] = sorted(set(surfaced))
            return pending

        from ..acl import workspace_scope_sql
        workspace_sql, workspace_params = workspace_scope_sql(
            "COALESCE(NULLIF(m.workspace_canonical,''),m.workspace)", workspace,
        )
        eligible = "m.status='active' AND c.memory_version = m.version AND c.memory_id != ?"
        eligible_params: list[Any] = [int(memory_id)]
        if workspace_sql:
            eligible += f" AND {workspace_sql}"
            eligible_params.extend(workspace_params)
        id_constraint = f"c.id IN (SELECT c.id FROM memory_claims c JOIN memories m ON m.id=c.memory_id WHERE {eligible})"

        unresolved_bridges = 0
        bridge_budget = [CLAIMS_BRIDGE_MAX_PER_WRITE]
        with self.db.connection() as conn:
            for claim in own_claims:
                own_vector = own_rows.get(int(claim["id"]))
                if not own_vector:
                    continue
                # D1 (owner 2026-09-23): version-like attrs are expected
                # timeline evolution — skip before any KNN work. The COUNT
                # happens exactly once, in the exact lane (same predicate ran
                # there already); counting here too would double every entry.
                if attr_is_versional(str(claim.get("attr") or claim["attr_norm"])):
                    continue
                checked += 1
                hits = conn.execute(
                    f"""SELECT c.*, v.distance AS distance,
                               m.subject, m.metadata, m.content,
                               m.workspace, m.workspace_canonical
                        FROM memory_claim_vec v
                        JOIN memory_claims c ON c.id=v.id
                        JOIN memories m ON m.id=c.memory_id
                        WHERE v.embedding MATCH ? AND k=10 AND {id_constraint}
                        ORDER BY v.distance""",
                    [json.dumps(own_vector), *eligible_params],
                ).fetchall()
                if str(claim["attr_norm"]) in fired_attrs:
                    continue  # A3: same attr already reported this write
                bridge_candidate_rows: list[dict[str, Any]] = []
                attr_matched = False
                for hit in hits:
                    # Gate-v2 G6: every claim-KNN neighbour is a bridge
                    # candidate peer (their SENTENCE rows are the attr's
                    # semantic neighbourhood) — collected BEFORE the tau
                    # continue, which only gates the same-attr comparison.
                    peer_subject_raw = hit["subject"] if "subject" in hit.keys() else ""
                    bridge_candidate_rows.append({
                        "peer_id": int(hit["memory_id"]),
                        "peer_version": int(hit["memory_version"] or 1),
                        "peer_subject": str(peer_subject_raw or ""),
                    })
                    hit_vector = None
                    vec_row = conn.execute(
                        "SELECT embedding FROM memory_claim_vec WHERE id=?", (int(hit["id"]),)
                    ).fetchone()
                    if vec_row is not None and vec_row["embedding"] is not None:
                        hit_vector = self.db.evidence._blob_to_vector(bytes(vec_row["embedding"]))
                    attr_cos = vector_cosine(own_vector, hit_vector)
                    same_exact = str(hit["attr_norm"]) == str(claim["attr_norm"])
                    if not same_exact and (attr_cos is None or attr_cos < CLAIM_ATTR_TAU):
                        continue
                    attr_matched = True
                    # D1 hit-level fallback: the own attr is not versional
                    # (own-level skip above already took those), but a
                    # τ-similar peer attr can still be version semantics
                    # (发布说明 ≈ release notes) — same exemption applies.
                    if attr_is_versional(str(hit["attr"] or hit["attr_norm"])):
                        versional_vetoed += 1
                        continue
                    if str(hit["value_norm"]) == str(claim["value_norm"]):
                        continue
                    peer_id = int(hit["memory_id"])
                    if peer_id in skip:
                        continue  # evidence channel already surfaced this pair
                    # A4 coexistence: either side declaring multiple values for
                    # the attr is a self-coexistence, not an opposing claim.
                    # Prefetched per (memory, attr) — per-hit connection churn
                    # was ~800 opens/write (adversarial review P2-7).
                    if len(own_coexistence.get(str(claim["attr_norm"]), ())) > 1:
                        continue
                    peer_attrs = peer_coexistence.get(peer_id)
                    if peer_attrs is None:
                        peer_attrs = _coexistence_by_attr(
                            self.db.claims.current_claims(peer_id),
                        )
                        peer_coexistence[peer_id] = peer_attrs
                    if len(peer_attrs.get(str(hit["attr_norm"]), ())) > 1:
                        continue
                    # Cross-channel dedup (appendix C 7): a pair the evidence
                    # channel already settled this version never double-fires.
                    if self.db.semantic_notices.is_semantic_pair_closed_on_conn(
                        conn, int(memory_id), peer_id, version, int(hit["memory_version"] or 1),
                    ):
                        continue
                    if notices >= CLAIMS_MAX_NOTICES_PER_WRITE:
                        capped_count += 1
                        continue  # count the overflow, keep scanning cheaper
                    slot_key = _retired_gate_slot_key(
                        record.get("workspace_canonical") or record.get("workspace"),
                        str(hit["attr_norm"]), str(record.get("subject") or ""),
                    )
                    outcome = self.db.record_semantic_notice(
                        memory_id=int(memory_id), peer_id=peer_id, severity="normal",
                        notice_type="claim_conflict",
                        title=f"Claim conflict with #{peer_id}",
                        message=f"claims attr {claim['attr']} value differs",
                        payload=_conflict_notice_payload(
                            reason="claim_attr_vector_gate",
                            attribute="claims_channel",
                            slot_key=slot_key,
                            left_id=int(memory_id), left_version=version,
                            left_value_norm=str(claim["value_norm"]),
                            left_display=str(claim["value"]),
                            left_quote=str(claim["value"]),
                            right_id=peer_id,
                            right_version=int(hit["memory_version"] or 1),
                            right_value_norm=str(hit["value_norm"]),
                            right_display=str(hit["value"]),
                            right_quote=str(hit["value"]),
                            left_content=str(record.get("content") or ""),
                            right_content=str(hit["content"] or ""),
                            attr_cos=float(attr_cos or 1.0),
                            extra={
                                "source": "claim_conflict",
                                "claims_channel": True,
                            },
                        ),
                        dedupe_key=notice_dedupe_key(
                            int(memory_id), peer_id, version, int(hit["memory_version"] or 1),
                            "claim_conflict",
                        ),
                        left_version=version, right_version=int(hit["memory_version"] or 1),
                        source="claim_conflict",
                    )
                    if outcome.get("outcome") == "created":
                        notices += 1
                        fired_attrs.add(str(claim["attr_norm"]))
                        surfaced.append(peer_id)
                # Gate-v2 G6 单边桥: no peer claim carries this attr — aim the
                # attr vector at the peer's SENTENCE rows and let Qwen pull
                # the value (case a, prompt names the attr). Unresolvable
                # bridges are counted, never silent (claim_bridge_unresolved).
                if (
                    not attr_matched
                    and str(claim["attr_norm"]) not in fired_attrs
                    and bridge_candidate_rows
                    and notices < CLAIMS_MAX_NOTICES_PER_WRITE
                    and bridge_budget[0] > 0
                ):
                    bridge_budget[0] -= 1
                    bridge_outcome, bridge_peer = self._run_claim_bridge(
                        conn, int(memory_id), version, record, claim,
                        own_vector, bridge_candidate_rows, skip,
                    )
                    if bridge_outcome == "created":
                        notices += 1
                        fired_attrs.add(str(claim["attr_norm"]))
                        if bridge_peer is not None:
                            surfaced.append(bridge_peer)
                    elif bridge_outcome == "unresolved":
                        unresolved_bridges += 1
        result: dict[str, Any] = {
            "channel_b_checked": checked, "channel_b_notices": notices,
        }
        if int(exact_lane["checked"]):
            result["channel_b_exact_checked"] = int(exact_lane["checked"])
        if capped_count:
            result["channel_b_capped"] = capped_count
        if exact_capped:
            result["channel_b_exact_capped"] = exact_capped
        if versional_vetoed:
            result["channel_b_versional_vetoed"] = versional_vetoed
        if unresolved_bridges:
            result["channel_b_unresolved"] = unresolved_bridges
        if surfaced:
            # Internal cross-channel key: consumed by the job wrapper for the
            # A-cross skip set (R1-3), never a receipt field (underscore:
            # wrapper MUST pop it — r2s-08 one-convention rule).
            result["_surfaced_peers"] = sorted(set(surfaced))
        return result

    def _run_claim_bridge(
        self, conn: "sqlite3.Connection", memory_id: int, version: int,
        record: dict[str, Any], claim: dict[str, Any], attr_vector: list[float],
        candidate_rows: "list[dict[str, Any]]", skip: "set[int]",
    ) -> "tuple[str, int | None]":
        """Gate-v2 G6 单边桥: own claim (attr, value) has no same-attr peer
        claim — case a of the three-case dispatch. The attr vector aims at
        the peer's sentence rows (the channel-C KNN shape); the TOP row's
        text goes to Qwen with a prompt naming the attr; an extracted value
        that differs from the own claim lands the notice. Returns
        (created / unresolved / skipped, surfaced peer id or None).

        Q1: the bridge's Qwen rides channel B's own CLAIMS_BRIDGE_MAX_PER_WRITE
        cap — it does NOT draw on the job-global pool (owner plan §3.1 scope:
        internal + C + A-cross)."""
        from ..constants import SEMANTIC_CROSS_KNN_WINDOW
        from .gates import dispatch_hint_text

        backend = self._ensure_semantic_backend()
        if backend is None:
            return ("skipped", None)
        row_text: str | None = None
        row_peer: tuple[int, int] | None = None
        for candidate in candidate_rows:
            peer_id = int(candidate["peer_id"])
            hits = self.db.row_knn(
                attr_vector, k=SEMANTIC_CROSS_KNN_WINDOW,
                workspace=(
                    record.get("workspace_canonical") or record.get("workspace")
                    if self.settings.isolation == "strict" else None
                ),
                exclude_memory_id=memory_id, conn=conn,
                include_subject_rows=False,
                include_memory_ids=[peer_id],
            )
            if hits:
                row_text = str(hits[0].get("text") or "")
                row_peer = (peer_id, int(hits[0].get("memory_row_version") or candidate["peer_version"]))
                break
        if not row_text or row_peer is None:
            return ("unresolved", None)
        peer_id, peer_version = row_peer
        if peer_id in skip:
            return ("skipped", None)
        left_env: dict[str, Any] = {
            # FULL own body as the quote: the claim value is a slice of it,
            # and grounding rejects a value equal to the whole quote (anti
            # copy-the-sentence rule) — Qwen must extract a compact slot.
            "quote": str(record.get("content") or "")[:1000],
            "subject": str(record.get("subject") or "")[:200],
            "tags": [], "memory_id": int(memory_id), "version": version,
            "metadata": {},
            # case a: name the attr — Qwen extracts THAT attribute's value.
            # (_pair_text renders the hint from the LEFT env.)
            "dispatch_hint": (
                f"{dispatch_hint_text('extract_value')} 需抽取的属性名：{claim['attr']}"
            ),
        }
        peer_record = self.db.get_memory(peer_id) or {}
        right_env: dict[str, Any] = {
            "quote": row_text[:1000], "subject": str(peer_record.get("subject") or "")[:200],
            "tags": list(peer_record.get("tags") or [])[:20],
            "workspace_canonical": peer_record.get("workspace_canonical") or peer_record.get("workspace"),
            "memory_id": peer_id, "version": peer_version,
            "metadata": {},
        }
        forward = self._conflict_classify(backend, left_env, right_env, retry_allowed=False)
        gate = evaluate_single_direction_extraction(
            signal_extraction(forward), left_env, right_env,
        )
        if gate.state != "notice_ready":
            return ("unresolved", None)
        extracted_b = str(gate.value_b or "")
        if not extracted_b or normalize_value(extracted_b) == normalize_value(str(claim["value_norm"])):
            return ("unresolved", None)  # extracted the SAME value: no conflict
        slot_key = _retired_gate_slot_key(
            record.get("workspace_canonical") or record.get("workspace"),
            str(claim["attr_norm"]), str(record.get("subject") or ""),
        )
        outcome = self.db.record_semantic_notice(
            memory_id=memory_id, peer_id=peer_id, severity="normal",
            notice_type="claim_conflict",
            title=f"Claim conflict with #{peer_id}",
            message=f"claim bridge attr {claim['attr']} value differs",
            payload=_conflict_notice_payload(
                reason="claim_bridge_extract_value",
                attribute="claim_bridge",
                slot_key=slot_key,
                left_id=int(memory_id), left_version=version,
                left_value_norm=str(claim["value_norm"]),
                left_display=str(claim["value"]),
                left_quote=str(claim["value"]),
                right_id=peer_id, right_version=peer_version,
                right_value_norm=extracted_b,
                right_display=extracted_b,
                right_quote=row_text,
                left_content=str(record.get("content") or ""),
                right_content=str(peer_record.get("content") or ""),
                attr_cos=1.0,
                extra={
                    "source": "claim_conflict",
                    "claims_channel": True,
                    "claim_bridge": True,
                },
            ),
            dedupe_key=notice_dedupe_key(
                memory_id, peer_id, version, peer_version, "claim_conflict",
            ),
            left_version=version, right_version=peer_version,
            source="claim_conflict",
        )
        if outcome.get("outcome") == "created":
            return ("created", peer_id)
        return ("skipped", None)

    def check_claim_sentence_conflicts(
        self, memory_id: int, snapshot: dict[str, Any],
        allowed_memory_ids: "list[int] | None" = None,
        notices_used: int = 0,
        budget_sink: "Callable[[], None] | None" = None,
        deadline_fn: "Callable[[], float | None] | None" = None,
    ) -> dict[str, Any]:
        """Gate-v2 G6b 通道 C: claims×sentences across the clean neighbour
        list (owner 2026-09-23). Each own claim's ATTR vector queries the
        sentence rows inside the G5 clean list (claims never depend on the
        peer filling claims); pairs passing the cosine band go to Qwen as
        case a (the prompt names the attr). Versional attrs are exempted
        (D1, own counter). Cross-channel dedup does NOT ride an input skip
        set here (0.17.0 review R2 r2s-12: every production/test caller
        passed None since the R1-3 direction flip) — pairs the other
        channels already settled are excluded via the wrapper's A-cross
        skip set union instead; the shared per-write notice cap still
        applies.

        Q1 (owner D1): channel C now runs AHEAD of the A-cross loop and
        draws the job-global Qwen pool down via ``budget_sink`` (one call per
        ACTUAL dispatch — the pool is never a cap on C, D3). ``deadline_fn``
        (review R1-1) stops further dispatches once the fairness wall is
        blown — C is unbounded in PAIRS but must not eat the queue's clock.
        ``_surfaced_peers`` in the result is an internal cross-channel key
        for the wrapper's A-cross skip set, never a receipt field."""
        from ..constants import (
            CLAIMS_MAX_NOTICES_PER_WRITE,
            SEMANTIC_CANDIDATE_COS_CEIL,
            SEMANTIC_CHANNEL_C_COS_FLOOR,
            SEMANTIC_CROSS_KNN_WINDOW,
        )
        from ..semantic_conflict import attr_is_versional, vector_cosine
        from ..rowseg import row_context_text
        from .gates import dispatch_hint_text

        record = snapshot if snapshot.get("content") is not None else (
            self.db.get_memory(int(memory_id)) or {}
        )
        version = int(record.get("version") or 1)
        if not self.db.state.sqlite_vec_available:
            return {"channel_c": True, "reason": "vec_unavailable"}
        own_claims = self.db.claims.current_claims(int(memory_id))
        if not own_claims or allowed_memory_ids is None or not allowed_memory_ids:
            return {"channel_c": True, "channel_c_checked": 0, "channel_c_notices": 0}
        workspace = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        # own claim vectors (attr embeddings published on the write path)
        own_vectors = self._own_claim_vectors(int(memory_id), version)
        notices = 0
        capped = 0
        versional = 0
        unresolved = 0
        checked = 0
        surfaced: list[int] = []  # Q1 R1-3: wrapper unions into the A-cross skip set
        _unres_reasons: dict[str, int] = {}
        backend = self._ensure_semantic_backend()
        embedder, _warnings = self._ensure_active_embedder()
        peer_content_cache: dict[int, dict[str, Any]] = {}
        min_budget = SEMANTIC_MIN_PAIR_BUDGET_MS / 1000.0
        deadline_stopped = False

        def peer_row(peer_id: int) -> dict[str, Any]:
            if peer_id not in peer_content_cache:
                peer_content_cache[peer_id] = self.db.get_memory(peer_id) or {}
            return peer_content_cache[peer_id]

        for claim in own_claims:
            if deadline_stopped:
                break  # the fairness wall is monotonic — no later pair can dispatch
            attr_vector = own_vectors.get(int(claim["id"]))
            if not attr_vector:
                continue
            if attr_is_versional(str(claim.get("attr") or claim["attr_norm"])):
                versional += 1
                continue  # D1: expected timeline evolution
            hits = self.db.row_knn(
                attr_vector, k=SEMANTIC_CROSS_KNN_WINDOW, workspace=workspace,
                exclude_memory_id=int(memory_id), include_subject_rows=False,
                include_memory_ids=allowed_memory_ids,
                include_content=True,  # claims channel: row content feeds the
                # peer-side context (row-version-pinned, preferred over the
                # peer's current content) and the notice content fingerprint
            )
            hit_vectors = self.db.evidence.row_vectors_for_ids(
                [int(h["id"]) for h in hits],
            )
            for hit in hits:
                # P2-6: the cap is SHARED with the claims channel — the
                # wrapper hands in how many notices that channel already
                # spent this write, so the write total stays ≤ 5.
                if notices + notices_used >= CLAIMS_MAX_NOTICES_PER_WRITE:
                    capped += 1
                    break
                # R1-1: the wall-clock fairness gate A lives by — C may
                # overdraw the PAIR pool (D3) but must not eat the queue's
                # clock. Same margin as the A-cross loop (min_budget × 2).
                if deadline_fn is not None:
                    active_deadline = deadline_fn()
                    if (
                        active_deadline is not None
                        and active_deadline - time.monotonic() < min_budget * 2
                    ):
                        deadline_stopped = True
                        break
                peer_id = int(hit["memory_id"])
                # cosine band on the TRUE attr-vector×sentence cosine
                # (vectors prefetched per claim above — per-hit single-row
                # fetches were N round-trips per KNN window)
                vector = hit_vectors.get(int(hit["id"]))
                if not vector:
                    continue
                cos = vector_cosine(attr_vector, vector)
                if not (SEMANTIC_CHANNEL_C_COS_FLOOR <= cos < SEMANTIC_CANDIDATE_COS_CEIL):
                    continue
                checked += 1
                peer = peer_row(peer_id)
                if str(peer.get("status") or "") != "active":
                    continue
                left_env: dict[str, Any] = {
                    "quote": str(record.get("content") or "")[:1000],
                    "subject": str(record.get("subject") or "")[:200],
                    "tags": list(record.get("tags") or [])[:20],
                    "workspace_canonical": record.get("workspace_canonical") or record.get("workspace"),
                    "memory_id": int(memory_id), "version": version,
                    "event_time": record.get("event_time"), "metadata": {},
                    "dispatch_hint": (
                        f"{dispatch_hint_text('extract_value')} 需抽取的属性名：{claim['attr']}"
                    ),
                    # case a: own value is KNOWN (structured claim) — hint it
                    # so the 0.6B only extracts the peer side's compact value.
                    "rule_value": str(claim["value"]),
                }
                right_env: dict[str, Any] = {
                    "quote": str(hit.get("text") or "")[:1000],
                    "subject": str(peer.get("subject") or "")[:200],
                    "tags": list(peer.get("tags") or [])[:20],
                    "workspace_canonical": peer.get("workspace_canonical") or peer.get("workspace"),
                    "memory_id": peer_id,
                    "version": int(hit.get("memory_row_version") or 1),
                    "event_time": peer.get("event_time"), "metadata": {},
                }
                # D3（owner 2026-09-25 翻案）：C 的 peer 侧同样接行上下文——
                # 属性已由 dispatch_hint 点名，上下文供 peer 行的值语境恢复；
                # left 是结构化 claim（无句子）不加。FP=0 是硬线，验收盯防。
                # content 优先取 hit 自带（与偏移严格同版；对抗 review P3）
                peer_context = row_context_text(
                    str(hit.get("content") or peer.get("content") or ""),
                    int(hit.get("start_offset") or 0),
                    int(hit.get("end_offset") or 0),
                )
                if peer_context:
                    right_env["context"] = peer_context
                if backend is None:
                    unresolved += 1
                    continue
                forward = self._conflict_classify(backend, left_env, right_env)
                # Q1 R1-4: the pool is charged per ACTUAL dispatch —
                # claims_checked above also counts band-passers that never
                # dispatched (inactive peer, backend None); those must not
                # erode the A-cross residual.
                if budget_sink is not None:
                    budget_sink()
                gate = evaluate_single_direction_extraction(
                    signal_extraction(forward), left_env, right_env,
                    attr_cos=_attr_cos_or_none(embedder, signal_extraction(forward)),
                )
                if gate.state != "notice_ready":
                    unresolved += 1
                    _unres_reasons[gate.reason] = _unres_reasons.get(gate.reason, 0) + 1
                    continue
                extracted = str(gate.value_b or "")
                if not extracted or normalize_value(extracted) == normalize_value(
                    str(claim["value_norm"]),
                ):
                    unresolved += 1
                    continue
                slot_key = _retired_gate_slot_key(
                    record.get("workspace_canonical") or record.get("workspace"),
                    str(claim["attr_norm"]), str(record.get("subject") or ""),
                )
                peer_version = int(hit.get("memory_row_version") or 1)
                outcome = self.db.record_semantic_notice(
                    memory_id=int(memory_id), peer_id=peer_id, severity="normal",
                    notice_type="claim_conflict",
                    title=f"Claim conflict with #{peer_id}",
                    message=f"channel-C claim vs sentence attr {claim['attr']} differs",
                    payload=_conflict_notice_payload(
                        reason="claim_channel_c_attr_sentence",
                        attribute="claims_channel_c",
                        slot_key=slot_key,
                        left_id=int(memory_id), left_version=version,
                        left_value_norm=str(claim["value_norm"]),
                        left_display=str(claim["value"]),
                        left_quote=str(claim["value"]),
                        right_id=peer_id, right_version=peer_version,
                        right_value_norm=normalize_value(extracted),
                        right_display=extracted,
                        right_quote=str(hit.get("text") or ""),
                        left_content=str(record.get("content") or ""),
                        right_content=str(peer.get("content") or ""),
                        attr_cos=float(cos),
                        extra={
                            "source": "claim_conflict",
                            "claims_channel": True,
                            "channel_c": True,
                        },
                    ),
                    dedupe_key=notice_dedupe_key(
                        int(memory_id), peer_id, version, peer_version, "claim_conflict",
                    ),
                    left_version=version, right_version=peer_version,
                    source="claim_conflict",
                )
                if outcome.get("outcome") == "created":
                    notices += 1
                    surfaced.append(peer_id)
        result: dict[str, Any] = {
            "channel_c": True,
            "channel_c_checked": checked, "channel_c_notices": notices,
        }
        if capped:
            result["channel_c_capped"] = capped
        if versional:
            result["channel_c_versional_vetoed"] = versional
        if unresolved:
            result["channel_c_unresolved"] = unresolved
            result["channel_c_unresolved_reasons"] = _unres_reasons
        if surfaced:
            # Internal cross-channel key: consumed by the job wrapper for the
            # A-cross skip set (R1-3), never a receipt field (underscore:
            # wrapper MUST pop it — r2s-08 one-convention rule).
            result["_surfaced_peers"] = sorted(set(surfaced))
        if deadline_stopped:
            # 对抗 review 修复（R1-1 收尾）：C 撞公平墙停走必须 loud——
            # 「干到一半被墙砍」与「自然跑完」在回执上可区分（§3.3 惯例：
            # 停走条件键，未撞墙不出现）。
            result["channel_c_deadline_stopped"] = True
        return result

    def drain_conflict_backlog(self, limit: int = 2) -> int:
        """0.17.0 P2-4.2: idle-worker consumption of the conflict backlog.

        Bounded per call (``limit`` entries); new writes always win because
        the caller only invokes this when the job queue is empty and rechecks
        between entries. A stored extraction replays through the deterministic
        gate without Qwen; a Qwen-less replay with no extraction lands
        nothing (the entry stays pending for a backend-bearing pass — never
        silently completed)."""
        processed = 0
        skipped: list[int] = []  # unprocessable this pass (no backend) — rotate past, never freeze
        while processed < limit:
            entry = self.db.conflict_backlog.take_next(exclude_ids=skipped)
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
                left_env = _conflict_envelope(left, left_text)
                right_env = _conflict_envelope(right, right_text)
                forward = self._conflict_classify(backend, left_env, right_env)
                gate = evaluate_single_direction_extraction(
                    signal_extraction(forward), left_env, right_env,
                    attr_cos=_attr_cos_or_none(embedder, signal_extraction(forward)),
                )
                if gate.state == "notice_ready":
                    self._record_backlog_notice(
                        left, right, left_text, right_text, decision,
                        str(gate.attribute), str(gate.value_a), str(gate.value_b),
                        reason=str(gate.reason),
                    )
                self.db.conflict_backlog.complete(int(entry["id"]))
                processed += 1
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
        return processed

    def _record_backlog_notice(
        self, left: dict[str, Any], right: dict[str, Any],
        left_text: str, right_text: str, decision: Any,
        attribute: str, value_a: str, value_b: str, *, reason: str,
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
        self.db.record_semantic_notice(
            memory_id=left_id, peer_id=right_id, severity="normal",
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
                extra={
                    "prompt_version": PAIR_PROMPT_VERSION,
                    "anchors": decision.anchors,
                    "backlog": True,
                },
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
        return {
            "status": "indexed" if published.get("published") else "failed",
            **published,
        }

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
            # space. (The mismatch guard for ordinary detection jobs stays in
            # process_conflicts.)
            pass
        elif self.db.evidence.current_row_vectors(int(memory_id), version, row_sha):
            return {"status": "indexed", "reason": "already_current",
                    "index_only": True, "notices_created": 0}
        from ..rowseg import segment_rows
        phase_started = time.monotonic()
        segments = segment_rows(str(record.get("subject") or ""), content)
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
            return {"status": "indexed", "index_only": True,
                    "row_count": int(published.get("row_count") or len(segments)),
                    "notices_created": 0}
        return {"status": "incomplete", "reason": f"publish_{published.get('outcome')}",
                "index_only": True, "notices_created": 0}

    def process_conflicts(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Standalone full-A entry (tests, direct callers): deterministic →
        internal Qwen → A-cross dispatch, in the owner-D1 order with the
        job-global pool. The write job does NOT use this — the wrapper in
        tools.py calls the phase methods directly so channels B and C ride
        between them (确定性相 → B → internal → C → 派发相)."""
        ctx = self.conflicts_deterministic_phase(memory_id, snapshot)
        terminal = ctx.get("terminal")
        if terminal is None:
            self.conflicts_internal_qwen_phase(ctx)
            self.conflicts_dispatch_phase(ctx, skip_peers=set())
            result = self.conflicts_finalize_receipt(ctx)
        else:
            result = terminal
        return self.conflicts_receipt_tail(ctx, result)

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
            "budget": _JobQwenBudget(),
            "min_budget": SEMANTIC_MIN_PAIR_BUDGET_MS / 1000.0,
            "applying_slots": set(),
            "degradation_reasons": set(),
            "reasons_seen": [],
            "reached_pair": set(),
            "ordered": [],
            "internal_qwen_pairs": [],
            "allowed_memory_ids": None,
            "units_examined": 0,
            "rows_mode": False,
            "no_difference_filtered": 0,
            "memory_pairs_excluded": 0,
            "prefiltered_rows": 0,
            "rows_covered_by_claims": 0,
            "below_cos_floor": 0,
            "repeatability_skipped": 0,
            "internal_found": 0,
            "internal_qwen_confirmed": 0,
            "internal_qwen_vetoed": 0,
            "surfaced": 0,
            "surfaced_peer_ids": set(),
            "dropped_unlocalizable": 0,
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

    def _collect_internal_pairs(
        self, ctx: dict[str, Any], job_conn: "sqlite3.Connection",
        seg_views: "list[Any]",
    ) -> None:
        """Deterministic phase step 2 (r2s-02 split): O(n²) same-memory
        keeper collection, feeding the internal Qwen phase (E10①).

        0.16.0 E10① (§6⑳): same-memory internal examination comes FIRST —
        units are in hand (no KNN), the rule is deterministic, and the
        finding lands in the dedicated internal_conflicts structure (the
        conflicts table's pair invariants reject a single memory@version
        twice). It consumes no Qwen budget; the cross-memory loop is
        untouched in shape, and a truncated cross loop still reports the
        internal findings already landed this run.
        0.16.2 (owner, unified flow): the internal check uses the SAME
        filter-plus-slot-extraction logic as the cross-memory route.
        0.16.4 §0.5/§2: the whole filter sequence is ONE shared gate —
        internal_pair_admission (scan_pipeline) — called identically by
        the scan side. Here: EVERY admitted shape (check AND notify)
        collects for the Qwen final review — a real in-memory
        self-contradiction has a recognition duty, and Qwen's verdict is
        the triple ready→pending+attribution / definitive negative→
        dismissed veto / technical failure→pending unannotated (fail-open).
        0.17.0 P2-3.1: internal (same-memory) pairs are row segments in
        rows mode (row_index lands in internal_conflicts' unit_a/unit_b)."""
        memory_id = int(ctx["memory_id"])
        internal_version = int(ctx["internal_version"])
        from ..scan_pipeline import internal_pair_admission

        # append-only through the alias — the list object lives in ctx and
        # feeds the internal Qwen phase (E10① keepers).
        internal_qwen_pairs: list[tuple[Any, Any, Any]] = ctx["internal_qwen_pairs"]
        # C3 A+ guard: subject rows never originate pairs (internal or cross).
        # Adversarial-review follow-up: an INDEPENDENT cap (not the cross
        # loop's) — O(n²) construction with row granularity needs its own
        # ceiling, while E10①'s guarantee (internal keepers land despite
        # cross truncation) forbids sharing the cross cap.
        from ..constants import SEMANTIC_INTERNAL_MAX_ROWS
        originator_views = [
            v for v in seg_views if v.kind != "subject"
        ][:SEMANTIC_INTERNAL_MAX_ROWS]
        for i in range(len(originator_views)):
            for j in range(i + 1, len(originator_views)):
                seg_a, seg_b = originator_views[i], originator_views[j]
                internal_decision = decide_evidence(seg_a.text, seg_b.text)
                admitted = internal_pair_admission(
                    seg_a.text, seg_b.text,
                    (seg_a.start_offset, seg_a.end_offset),
                    (seg_b.start_offset, seg_b.end_offset),
                    internal_decision,
                    exists_probe=lambda: self.db.internal_conflicts.exists_on_conn(
                        job_conn, int(memory_id), internal_version,
                        seg_a.unit_index, seg_b.unit_index,
                    ),
                )
                if not admitted:
                    continue
                internal_qwen_pairs.append((seg_a, seg_b, internal_decision))

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
        clean list (ctx["allowed_memory_ids"] stays the shared contract for
        the wrapper-orchestrated channel C)."""
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
        # Q1 相分裂: the list lives in ctx so the later phases and the
        # wrapper-orchestrated channel C share one anchor (append is
        # in-place — the local alias stays a live view of ctx state).
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
        self._collect_internal_pairs(ctx, job_conn, seg_views)
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
        from .gates import candidate_cos_gate, claim_value_spans, row_prefilter
        own_claims = self.db.claims.current_claims(int(memory_id))
        own_claim_spans = claim_value_spans(content, own_claims)
        # KEYED BY ROW INDEX, never object identity: the streaming path
        # yields the raw segments while seg_views are _SegView copies — the
        # same row under two Python objects. row_index is the stable key
        # across both.
        admissible_row_idx = {
            int(getattr(view, "unit_index", getattr(view, "row_index", 0)))
            for view in row_prefilter(seg_views, own_claim_spans)
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
                    # Gate-v2 G4: the row failed the sentence prefilter OR is
                    # already represented by an own claim (覆盖句跳过) — it
                    # never originates a KNN query and never spends budget.
                    if any(
                        span_start < seg_view.end_offset and seg_view.start_offset < span_end
                        for span_start, span_end in own_claim_spans
                    ):
                        ctx["rows_covered_by_claims"] += 1
                    else:
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
        if truncation_reason:
            # E10① order guarantee: internal findings land BEFORE the cross
            # loop and survive its truncation. The internal Qwen pass has not
            # run yet at this point (it needs the backend fetched by its own
            # phase), so the collected keepers land unannotated here —
            # fail-open, never lost. (Adversarial self-review: without this,
            # a row cap or budget exhaustion mid-collection silently dropped
            # every internal keep pair of this run.)
            for unit_a, unit_b, internal_decision in ctx["internal_qwen_pairs"]:
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
        Q1 (owner D1): SEMANTIC_MAX_EXAMINED_PAIRS is the job-global Qwen
        pool living in ctx["budget"] (internal + channel C + A-cross); the
        per-phase caps live in _JobQwenBudget."""
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

    def _conflict_classify(
        self, backend: "SemanticBackend", left_env: dict[str, Any], right_env: dict[str, Any],
        retry_allowed: "bool | None" = None,
    ) -> Any:
        """0.16.2 unified-flow classify closure, hoisted to a method for the
        Q1 phase split. Once a pair starts, only the inference hard timeout
        may stop it. The job budget is a fairness gate between pairs; the
        retry gate (A6) is the same fairness idea one level down: with
        another job queued, a protocol-invalid output fails fast instead of
        doubling its own latency. ``retry_allowed`` overrides the default
        queue-derived gate explicitly (the claim bridge passes False — its
        write-time budget cannot absorb a retry)."""
        try:
            return backend.classify_pair(
                left_env, right_env, deadline_monotonic=None,
                retry_allowed=(
                    not self._semantic_worker.has_pending_jobs()
                    if retry_allowed is None else retry_allowed
                ),
            )
        except TypeError:
            # Test/legacy backends implementing the original two-arg protocol.
            return backend.classify_pair(left_env, right_env)

    def conflicts_job_deadline(self, ctx: dict[str, Any]) -> "float | None":
        """Q1 R1-1: the fairness wall for wrapper-orchestrated channel C —
        the SAME deadline semantics the internal/dispatch phases live by
        (C may overdraw the pair pool per D3, but must not eat the queue's
        clock)."""
        return _job_fair_deadline(self._semantic_worker, ctx["publish_done_at"])

    def conflicts_internal_qwen_phase(self, ctx: dict[str, Any]) -> None:
        """Q1 相分裂 phase 2 (owner plan §3.1): internal (same-memory) keeper
        Qwen review — E10① lands before the cross loop; protection-capped at
        SEMANTIC_INTERNAL_QWEN_MAX_PAIRS and drawing the job-global pool
        (D1/D7: internal keeps its priority ahead of channel C)."""
        phase_started = time.monotonic()
        memory_id = int(ctx["memory_id"])
        record = ctx["record"]
        internal_version = int(ctx["internal_version"])
        budget: _JobQwenBudget = ctx["budget"]
        min_budget: float = ctx["min_budget"]
        backend = self._ensure_semantic_backend()
        # 0.16.2 unified flow: internal check-keepers go through the SAME Qwen
        # slot extraction as cross-memory pairs, BEFORE the cross loop (E10①
        # order guarantee: internal findings land first and survive a
        # truncated cross loop). Shared job-global pool and deadline —
        # internal keepers are naturally few (filter-calibrated). Verdicts:
        #   notice_ready  → land pending, reason annotated with the extracted
        #                    attribute/values
        #   definitive negative (not the same attribute with different
        #   values, unknown_field) → land DISMISSED — the veto must outlive
        #   the write or the scan-side re-examination would resurrect the pair
        #   (internal_conflicts.exists() blocks any-status-non-stale rows)
        #   technical failure / no backend → land pending unannotated
        #   (fail-open: an advisory rule signal must not be lost to an
        #   unavailable model)
        for unit_a, unit_b, internal_decision in ctx["internal_qwen_pairs"]:
            if budget.internal_used >= budget.internal_cap:
                # Harness regression fix: row granularity multiplied internal
                # keepers and they starved the cross pairs out of the shared
                # Qwen budget. E10① keeps its land-first guarantee — within
                # this smaller, value-ranked budget.
                break
            reason_text = str(internal_decision.reason or "")
            if backend is not None:
                active_deadline = _job_fair_deadline(self._semantic_worker, ctx["publish_done_at"])
                budget_ok = not (
                    active_deadline is not None
                    and active_deadline - time.monotonic() < min_budget * 2
                ) and budget.spend_internal()
                if budget_ok:
                    env_a = _conflict_envelope(record, unit_a.text)
                    env_b = _conflict_envelope(record, unit_b.text)
                    if internal_decision.left_value and internal_decision.right_value:
                        env_a["rule_value"] = internal_decision.left_value
                        env_b["rule_value"] = internal_decision.right_value
                    # Single-direction judging (owner 2026-09-17): all three
                    # paths share the one-forward-extraction gate — the
                    # bidirectional mirror was falsified on the eval line.
                    forward = self._conflict_classify(backend, env_a, env_b)
                    gate = evaluate_single_direction_extraction(
                        signal_extraction(forward), env_a, env_b,
                        # internal path keeps STRICT attribute equality
                        # (docstring contract; P2-3.3 targets the peer path).
                    )
                    if gate.state == "notice_ready":
                        reason_text = (
                            f"{reason_text} | qwen:{gate.attribute}="
                            f"{gate.value_a}|{gate.value_b}"
                        )
                        ctx["internal_qwen_confirmed"] += 1
                    else:
                        technical = (
                            (forward.error and "timeout" in str(forward.error).lower())
                            or forward.candidate_type == "backend_unavailable"
                            or forward.candidate_type == "backend_error"
                            or forward.candidate_type in {"invalid_json", "invalid_schema"}
                        )
                        if not technical and gate.reason != "qwen_unverified":
                            ctx["internal_qwen_vetoed"] += 1
                            self.db.internal_conflicts.create(
                                memory_id=int(memory_id), memory_version=internal_version,
                                unit_a=unit_a.unit_index, unit_b=unit_b.unit_index,
                                quote_a=unit_a.text, quote_b=unit_b.text,
                                span_a=[unit_a.start_offset, unit_a.end_offset],
                                span_b=[unit_b.start_offset, unit_b.end_offset],
                                reason=reason_text, detector_version=CONFLICT_DETECTOR_VERSION,
                                status="dismissed",
                                decided_reason=f"qwen veto: {gate.reason}",
                            )
                            continue
            if self.db.internal_conflicts.create(
                memory_id=int(memory_id), memory_version=internal_version,
                unit_a=unit_a.unit_index, unit_b=unit_b.unit_index,
                quote_a=unit_a.text, quote_b=unit_b.text,
                span_a=[unit_a.start_offset, unit_a.end_offset],
                span_b=[unit_b.start_offset, unit_b.end_offset],
                reason=reason_text, detector_version=CONFLICT_DETECTOR_VERSION,
            ):
                ctx["internal_found"] += 1
        ctx["phase_ms"].append((time.monotonic() - phase_started) * 1000)

    def conflicts_dispatch_phase(
        self, ctx: dict[str, Any], skip_peers: "set[int] | None",
    ) -> None:
        """Q1 相分裂 phase 3 (owner plan §3.1): the A-cross ordered-pair loop.
        Qwen dispatch is gated by the job-global pool's RESIDUAL
        (max(0, total − internal − channel_c)); exhaustion turns the old
        break into a continue — deterministic direct verdicts still land and
        undispatched pairs fall to the backlog sweep (D2). ``skip_peers`` is
        the B∪C surfaced-peer union (review R1-3 — the dedup direction flip:
        C now runs ahead of A-cross)."""
        phase_started = time.monotonic()
        ordered = ctx["ordered"]
        backend = self._ensure_semantic_backend()
        # R1-5: the dispatch probes run on their OWN read snapshot — channels
        # B/C landed notices between the deterministic collection and this
        # phase, so the collection snapshot (job_conn) no longer describes
        # the world the closed-pair probes must see. The explicit skip set
        # (R1-3) replaces whatever the probes could have inferred about B/C.
        with self.db.connection() as dispatch_conn:
            dispatch_conn.execute("BEGIN")
            try:
                self.conflicts_dispatch_loop(ctx, dispatch_conn, backend, skip_peers)
            finally:
                try:
                    dispatch_conn.rollback()
                except sqlite3.Error:
                    pass
        # 0.17.0 P2-4.2: budget/cap leftovers land in the backlog — bounded,
        # visible, never silently dropped (owner design #8). Stale/duplicate
        # keys report as enqueued here; eviction counts ride the store.
        leftover_entries = [item for item in ordered if item[0] not in ctx["reached_pair"]]
        ctx["sweep_evicted"] = 0
        if leftover_entries:
            ctx["backlogged"], ctx["sweep_evicted"] = self._enqueue_backlog_entries(
                ctx, leftover_entries,
            )
        ctx["phase_ms"].append((time.monotonic() - phase_started) * 1000)

    def conflicts_dispatch_loop(
        self, ctx: dict[str, Any], dispatch_conn: "sqlite3.Connection",
        backend: "SemanticBackend | None", skip_peers: "set[int] | None",
    ) -> None:
        """The ordered-pair loop body (kept as its own method so the dispatch
        phase's read transaction wraps every probe)."""
        memory_id = int(ctx["memory_id"])
        record = ctx["record"]
        content = str(ctx["content"])
        workspace = ctx["workspace"]
        embedder = ctx["embedder"]
        budget: _JobQwenBudget = ctx["budget"]
        min_budget: float = ctx["min_budget"]
        reached_pair: set[int] = ctx["reached_pair"]
        applying_slots: set[str] = ctx["applying_slots"]
        surfaced_peer_ids: set[int] = ctx["surfaced_peer_ids"]
        # P2-T6: batch-prefetch every candidate peer in ONE id-IN query
        # instead of one get_memory connection per pair.
        peer_rows = self.db.get_memories_by_ids(
            [int(pid) for pid, _triple in ctx["ordered"]], conn=dispatch_conn,
        )
        for peer_id, (hit, unit, decision, _pair_cos) in ctx["ordered"]:
            peer = peer_rows.get(int(peer_id))
            if not peer or peer.get("status") != "active":
                reached_pair.add(peer_id)  # settled (inactive) — not backlog
                continue
            if skip_peers and peer_id in skip_peers:
                # Q1 R1-3 dedup direction flip: channel B/C already surfaced
                # this peer on this write — A-cross stands down (cross-channel
                # single report; strongest-evidence-first primitives like the
                # closed-pair check are unchanged).
                reached_pair.add(peer_id)  # settled (surfaced by B/C) — not backlog
                continue
            record_row: dict[str, Any] = record or {}
            peer_row: dict[str, Any] = peer or {}
            left_version = int(record.get("version") or 1)
            right_version = int(peer.get("version") or 1)
            if self.db.semantic_notices.is_semantic_pair_closed_on_conn(
                dispatch_conn, memory_id, peer_id, left_version, right_version,
            ):
                reached_pair.add(peer_id)  # settled (closed) — not backlog
                continue
            # 对抗 review 修复（Q1 相分裂 R1-5 的后果）：hit 证据来自确定性相
            # 快照，peer 行来自派发相新快照——两相之间 peer 被编辑时
            # memory_row_version（KNN 行自带）与 fresh version 不再一致，
            # 证据/版本错位的 notice 不可落库。视为 settled（本 job 跳过、
            # 不进 backlog——冻结对身份已过期，下次写会重收集）。
            if int(hit.get("memory_row_version") or 1) != right_version:
                reached_pair.add(peer_id)  # settled (stale hit) — not backlog
                continue
            # NOT reached yet: budget/cap skips below leave the pair
            # unmarked so the post-loop backlog sweep picks it up.
            # Deterministic direct path (2026-09-16, owner-approved): same
            # value-stripped key + canonical value difference IS the
            # same-attribute-different-value shape — land the notice without
            # spending Qwen, whose budget is reserved for pairs only
            # judgment can settle. Runs BEFORE the backend/budget checks:
            # a direct pair consumes no Qwen budget and works even while the
            # backend is unavailable.
            direct = direct_value_verdict(
                unit.text, str(hit.get("text") or ""), decision, embedder=embedder,
            )
            if direct is not None:
                reached_pair.add(peer_id)  # deterministic verdict — settled
                # Q1 §3.3: deterministic 直出 counter — the stage-2
                # comprehensive-recall channel attribution reads it.
                ctx["direct_verdicts"] += 1
                gate = PairGateResult(
                    "notice_ready", "deterministic_same_key_value_diff",
                    direct[0], direct[1], direct[2], True,
                )
                qwen = {
                    "status": "bypassed", "reason": gate.reason,
                    "forward_type": "deterministic", "reverse_type": "deterministic",
                }
                forward_signal = None
            else:
                if backend is None:
                    self._record_job_degradation(ctx, "qwen_unavailable")
                    ctx["incomplete_reason"] = ctx["incomplete_reason"] or "qwen_unavailable"
                    continue
                active_deadline = _job_fair_deadline(self._semantic_worker, ctx["publish_done_at"])
                if active_deadline is not None and active_deadline - time.monotonic() < min_budget * 2:
                    self._record_job_degradation(ctx, "qwen_budget_exhausted")
                    ctx["incomplete_reason"] = ctx["incomplete_reason"] or "qwen_budget_exhausted"
                    continue
                # Q1 (owner D1/D2): the job-global pool's RESIDUAL gates Qwen
                # dispatch; exhaustion no longer breaks the loop — it
                # CONTINUES: direct verdicts still land and undispatched
                # pairs stay unsettled for the backlog sweep (饱和不终止确
                # 定性检查). First skip reason wins (same first-trip-wins
                # semantics the old break had).
                if not budget.spend_a_cross():
                    self._record_job_degradation(ctx, "pairs_examined_capped")
                    ctx["incomplete_reason"] = ctx["incomplete_reason"] or "pairs_examined_capped"
                    continue
                reached_pair.add(peer_id)  # Qwen examined — settled
                left_env = _conflict_envelope(record_row, unit.text)
                right_env = _conflict_envelope(peer_row, str(hit.get("text") or ""))
                # 行上下文 envelope（owner 2026-09-25 方案）：属性名可从
                # 标题/邻行恢复，值必须取自主行——grounding 只读 quote，
                # 机制上挡住从上下文捞值。空 context 不设键、prompt 不渲染。
                from ..rowseg import row_context_text

                own_context = row_context_text(content, unit.start_offset, unit.end_offset)
                if own_context:
                    left_env["context"] = own_context
                peer_context = row_context_text(
                    str(peer_row.get("content") or ""),
                    int(hit.get("start_offset") or 0),
                    int(hit.get("end_offset") or 0),
                )
                if peer_context:
                    right_env["context"] = peer_context
                # pair-v7: hand Qwen the rule layer's extracted value difference
                # (numeric check route only) as a locating hint — see _pair_text.
                if decision.left_value and decision.right_value:
                    left_env["rule_value"] = decision.left_value
                    right_env["rule_value"] = decision.right_value
                # Gate-v2 G6 three-case dispatch (pair-v9): same output
                # protocol, only the task instruction differs by shape.
                dispatch_case = qwen_dispatch(decision)
                left_env["dispatch_hint"] = dispatch_hint_text(dispatch_case)
                started = time.monotonic()
                # Single-direction gate (owner 2026-09-17): the reverse
                # extraction was the side-attribution hedge the 0.5B needed;
                # Qwen3-0.6B doesn't commit that error and the bidirectional
                # cross-mapping kept killing real conflicts at reverse
                # attribute drift (eval: 9/12 bidirectional vs 11/12 single,
                # hard false positives 0/43 on the product chain). One clean
                # extraction + grounding + veto lands the notice; scan and
                # internal-conflict paths keep the bidirectional gate.
                forward_signal = self._conflict_classify(backend, left_env, right_env)
                reverse_signal = None
                self._tools._record_pair_sample(
                    pair_ms=int((time.monotonic() - started) * 1000),
                    forward=forward_signal,
                    reverse=None,
                )
                gate = evaluate_single_direction_extraction(
                    signal_extraction(forward_signal), left_env, right_env,
                    attr_cos=_attr_cos_or_none(embedder, signal_extraction(forward_signal)),
                )
                qwen = {
                    "status": gate.state, "reason": gate.reason,
                    "forward_type": forward_signal.candidate_type,
                    "reverse_type": "single_direction",
                }
            if gate.state != "notice_ready":
                signals = tuple(
                    signal for signal in (forward_signal, reverse_signal) if signal is not None
                )
                if any(signal.error and "timeout" in str(signal.error).lower() for signal in signals):
                    reason = "qwen_timeout"
                elif any(signal.candidate_type == "backend_unavailable" for signal in signals):
                    reason = "qwen_unavailable"
                elif any(signal.candidate_type == "backend_error" for signal in signals):
                    reason = "qwen_backend_error"
                elif any(signal.candidate_type in {"invalid_json", "invalid_schema"} for signal in signals):
                    reason = "qwen_invalid_output"
                elif any(signal.candidate_type == "unknown_field" for signal in signals):
                    # The model explicitly reported an unextractable field: a
                    # completed negative decision (fail-closed for notice), not
                    # a technical failure (spec §8 diagnostics distinction).
                    ctx["dropped_unlocalizable"] += 1
                    continue
                else:
                    reason = gate.reason
                if reason == "qwen_unverified":
                    # Grounding failed: uncertain — fail-closed for notices and
                    # the pair remains a scan review candidate. Owner ruling
                    # #9: unlocalizable candidates are DROPPED, counted loud.
                    self._record_job_degradation(ctx, reason)
                    ctx["dropped_unlocalizable"] += 1
                    ctx["incomplete_reason"] = reason
                    continue
                if reason in _TECHNICAL_REASONS:
                    sample = None
                    if reason == "qwen_invalid_output":
                        # Preserve the offending raw output so the failure mode
                        # (truncation / prose / malformed) is visible in status.
                        sample = next(
                            (
                                str(signal.raw) for signal in signals
                                if signal.candidate_type in {"invalid_json", "invalid_schema"} and signal.raw
                            ),
                            None,
                        )
                    self._record_job_degradation(ctx, reason, sample)
                    ctx["incomplete_reason"] = reason
                    continue
                # Definitive strict-gate negatives (not_same_attribute_different_value,
                # coexist_*, direction_invalid, bidirectional_*): the pair was
                # examined and decided. The check stays complete and no
                # degradation counter fires (spec §9/§15.5/§8).
                continue
            # Gate-v2 G3: slot identity rides workspace + subject (plan
            # slot_key 连锁) — the metadata entity/scope source is retired,
            # and the old `if not entity or not scope: continue` drop died
            # with it (keeping it would have discarded every notice).
            slot_key = _retired_gate_slot_key(
                workspace or record_row.get("workspace_canonical") or record_row.get("workspace"),
                gate.attribute, str(record_row.get("subject") or ""),
            )
            slot_json = json.dumps(slot_key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            raw_slot_json = json.dumps(
                {"entity": slot_key["entity"], "attribute": gate.attribute, "scope": slot_key["scope"]},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            if slot_json in applying_slots or raw_slot_json in applying_slots:
                # Suppression matches either form: conflict groups stored before
                # storage-side canonicalisation may still carry the raw
                # (unnormalised) slot_key. A third-party fact landing on a slot
                # currently under application: scan review only (spec §15.3),
                # no new notice.
                continue
            # Model output may omit keys (or parsed may not be a dict at all):
            # fall back to the gate's normalised value instead of raising
            # KeyError (mirrors the scan-path defence in tools.py).
            forward_parsed = (
                forward_signal.parsed
                if forward_signal is not None and isinstance(forward_signal.parsed, dict)
                else {}
            )
            outcome = self.db.record_semantic_notice(
                memory_id=memory_id, peer_id=peer_id,
                # 0.16.4 §1: cross-memory notify is excluded at collection,
                # so the write-time notice severity is uniformly normal.
                severity="normal",
                notice_type="semantic_evidence",
                title=f"Possible memory change with #{peer_id}", message=decision.reason,
                payload=_conflict_notice_payload(
                    reason=str(gate.reason),
                    attribute=(
                        "deterministic_skeleton" if direct is not None
                        else "single_direction_extraction"
                    ),
                    slot_key=slot_key,
                    left_id=int(memory_id), left_version=left_version,
                    left_value_norm=str(gate.value_a or ""),
                    left_display=str(forward_parsed.get("value_a") or gate.value_a),
                    left_quote=unit.text,
                    left_member_extra={"start": unit.start_offset, "end": unit.end_offset},
                    left_evidence_extra={
                        "start_offset": unit.start_offset, "end_offset": unit.end_offset,
                    },
                    right_id=int(peer_id), right_version=right_version,
                    right_value_norm=str(gate.value_b or ""),
                    right_display=str(forward_parsed.get("value_b") or gate.value_b),
                    right_quote=hit.get("text"),
                    right_member_extra={
                        "start": hit.get("start_offset"), "end": hit.get("end_offset"),
                    },
                    right_evidence_extra={
                        "start_offset": hit.get("start_offset"),
                        "end_offset": hit.get("end_offset"),
                    },
                    left_content=str(content or ""),
                    right_content=str(peer.get("content") or ""),
                    extra={
                        "prompt_version": PAIR_PROMPT_VERSION,
                        "anchors": decision.anchors,
                        "qwen_signal": qwen,
                    },
                ),
                dedupe_key=notice_dedupe_key(
                    memory_id, peer_id, left_version, right_version, "semantic_evidence",
                ),
                left_version=left_version, right_version=right_version,
                source="semantic_evidence",
            )
            # surfaced counts notices actually created (deduped pairs were
            # already surfaced) — since A5 it is a result summary, not a gate.
            if outcome.get("outcome") == "created":
                ctx["surfaced"] += 1
                surfaced_peer_ids.add(int(peer_id))
            elif outcome.get("outcome") not in {"deduped"}:
                # Second-round review: a ready pair whose notice could not be
                # persisted (workspace_mismatch / invalid_snapshot /
                # unavailable / error) must not vanish silently — without this
                # the run could report checked_no_notice while a real conflict
                # was found and lost.
                self._record_job_degradation(ctx, "notice_write_failed")
                ctx["incomplete_reason"] = ctx["incomplete_reason"] or "notice_write_failed"
        # 0.17.0 P2-4.2: budget/cap leftovers land in the backlog — bounded,
        # visible, never silently dropped (owner design #8). Stale/duplicate
        # keys report as enqueued here; eviction counts ride the store.
    def conflicts_finalize_receipt(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Q1 相分裂: merge the three phases' ctx state into the ONE job
        receipt (shape-compatible with the pre-split contract — additive
        keys only, and the zero case emits nothing new)."""
        if ctx["surfaced"]:
            result: dict[str, Any] = {
                "status": "completed", "outcome": "notices_created", "notices_created": ctx["surfaced"],
                # 0.17.0: peers the evidence channel surfaced THIS run — the
                # wrapper pops this internal key; channels B/C ride BEFORE the
                # dispatch phase now and feed its skip set directly (Q1).
                # Underscore = internal convention (r2s-08).
                "_surfaced_peers": sorted(ctx["surfaced_peer_ids"]),
            }
            if ctx["internal_found"]:
                result["internal_conflicts"] = ctx["internal_found"]
            if ctx["incomplete_reason"]:
                # Notices went out, but later pairs hit a truncation/degradation
                # — surface it instead of a bare completed (second-round
                # review): the caller would otherwise read a bounded, partial
                # check as a full one.
                result["truncated"] = True
        elif ctx["incomplete_reason"]:
            result = {"status": "incomplete", "reason": ctx["incomplete_reason"], "notices_created": 0}
        else:
            result = {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0}
        if ctx["internal_found"] and "internal_conflicts" not in result:
            # Internal findings survive a truncated cross-memory loop: they
            # were landed BEFORE the loop ran (E10① order guarantee).
            result["internal_conflicts"] = ctx["internal_found"]
        # 0.16.2 write-time pre-gate visibility (conditional — the unfiltered
        # zero case keeps the exact-shape response contract unchanged):
        # what the deterministic filters killed this run, and what the
        # unified internal Qwen flow confirmed/vetoed.
        filter_summary: dict[str, int] = {}
        if ctx["no_difference_filtered"]:
            filter_summary["no_difference_skipped"] = ctx["no_difference_filtered"]
        # Gate-v2 G4 observability: the prefilter/coverage/band split (the
        # unfiltered zero case keeps the exact-shape response contract).
        gate_rows: dict[str, int] = {}
        if ctx["memory_pairs_excluded"]:
            gate_rows["memory_pairs_excluded"] = ctx["memory_pairs_excluded"]
        if ctx["prefiltered_rows"]:
            gate_rows["prefiltered_rows"] = ctx["prefiltered_rows"]
        if ctx["rows_covered_by_claims"]:
            gate_rows["rows_covered_by_claims"] = ctx["rows_covered_by_claims"]
        if ctx["below_cos_floor"]:
            gate_rows["below_cos_floor"] = ctx["below_cos_floor"]
        if ctx["repeatability_skipped"]:
            gate_rows["repeatability_skipped"] = ctx["repeatability_skipped"]
        if gate_rows:
            result["candidate_gates"] = gate_rows
        # INTERNAL key (popped by the job wrapper): the G5 clean neighbour
        # list, shared by channel C (claims×sentences) — never in receipts.
        if ctx["allowed_memory_ids"] is not None:
            result["_allowed_memory_ids"] = ctx["allowed_memory_ids"]
        if ctx["internal_qwen_confirmed"]:
            filter_summary["internal_qwen_confirmed"] = ctx["internal_qwen_confirmed"]
        if ctx["internal_qwen_vetoed"]:
            filter_summary["internal_qwen_vetoed"] = ctx["internal_qwen_vetoed"]
        if ctx["dropped_unlocalizable"]:
            # 0.17.0 P2-3.5 (owner ruling #9): dropped unlocalizable pairs are
            # never silent — the counter rides every completed receipt.
            filter_summary["dropped_unlocalizable"] = ctx["dropped_unlocalizable"]
        if filter_summary:
            result["deterministic_filter"] = filter_summary
        if ctx["backlogged"]:
            # 0.17.0 P2-4.2: truncation leftovers went to the conflict
            # backlog instead of vanishing.
            result["backlogged"] = ctx["backlogged"]
        if ctx["sweep_evicted"]:
            # P2-4.3: cap evictions are visible, never silent.
            result["backlog_evicted"] = ctx["sweep_evicted"]
        if ctx["rows_mode"]:
            result["rows_mode"] = True
            result["rows_examined"] = int(ctx["units_examined"])
        if ctx["reasons_seen"]:
            # Degradations may also occur on pairs before a later pair surfaces
            # a notice, so the list is attached to completed outcomes too.
            result["reasons_seen"] = ctx["reasons_seen"]
        if ctx["direct_verdicts"]:
            result["direct_verdicts"] = int(ctx["direct_verdicts"])
        return result

    def conflicts_receipt_tail(self, ctx: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """r2s-08: ONE place stamps the receipt tail — qwen_budget /
        pairs_examined / elapsed_ms. The finalize path, the wrapper's
        truncation-terminal branch, and process_conflicts all ride it (the
        tail was previously stamped three ways and had already drifted: the
        terminal branch forgot elapsed, finalize stamped pairs_examined: 0
        unconditionally, breaking the §3.3 zero-values-never-appear rule)."""
        if ctx["phase_ms"]:
            result["elapsed_ms"] = round(sum(ctx["phase_ms"]), 1)
        if ctx["budget"].pairs_examined:
            result["pairs_examined"] = int(ctx["budget"].pairs_examined)
        qwen_budget = ctx["budget"].receipt_block()
        if qwen_budget is not None:
            # Q1 §3.3 additive observability — absent entirely when nothing
            # was deducted and nothing was skipped.
            result["qwen_budget"] = qwen_budget
        return result

