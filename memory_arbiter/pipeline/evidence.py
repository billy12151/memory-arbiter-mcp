"""Local-text evidence indexing and conflict candidate processing."""
from __future__ import annotations

import hashlib
import sqlite3
import json
import threading
import time
from typing import Any, TYPE_CHECKING, Iterator

from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..constants import (
    SEMANTIC_JOB_TIMEOUT_MS,
    SEMANTIC_MAX_EVIDENCE_UNITS,
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
    "evidence_units_capped", "rows_capped", "pairs_examined_capped",
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
    ea = embedder.embed_text(prefix="", body=str(attr_a))
    eb = embedder.embed_text(prefix="", body=str(attr_b))
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
            return {"claims_checked": 0, "notices": 0}
        if not self.db.state.sqlite_vec_available:
            return {"claims_checked": len(own_claims), "notices": 0, "reason": "vec_unavailable"}
        # Own claim vectors (published on the write path).
        own_rows: dict[int, list[float]] = {}
        with self.db.connection() as conn:
            for row in conn.execute(
                """SELECT c.id AS cid, v.embedding FROM memory_claims c
                   LEFT JOIN memory_claim_vec v ON v.id=c.id
                   WHERE c.memory_id=? AND c.memory_version=?""",
                (int(memory_id), version),
            ).fetchall():
                if row["embedding"] is not None:
                    own_rows[int(row["cid"])] = self.db.evidence._blob_to_vector(bytes(row["embedding"]))
        if not own_rows:
            return {"claims_checked": len(own_claims), "notices": 0, "reason": "vectors_pending"}

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

        notices = 0
        capped_count = 0
        checked = 0
        versional_vetoed = 0  # D1: evolution exemption must stay observable
        unresolved_bridges = 0
        bridge_budget = [CLAIMS_BRIDGE_MAX_PER_WRITE]
        fired_attrs: set[str] = set()  # A3: one notice per attr per write
        skip = skip_peers or set()
        own_coexistence: dict[str, list[str]] = {
            str(claim_row["attr_norm"]): [
                str(v) for v in self.db.claims.coexisting_values(
                    int(memory_id), str(claim_row["attr_norm"]),
                )
            ]
            for claim_row in own_claims
        }
        peer_coexistence: dict[int, dict[str, list[str]]] = {}
        with self.db.connection() as conn:
            for claim in own_claims:
                own_vector = own_rows.get(int(claim["id"]))
                if not own_vector:
                    continue
                # D1 (owner 2026-09-23): version-like attrs are expected
                # timeline evolution — skip before any KNN work and count.
                # Judged on the RAW attr: attr_norm strips spaces, which
                # defeats the vocabulary's word boundaries (releasenotes).
                if attr_is_versional(str(claim.get("attr") or claim["attr_norm"])):
                    versional_vetoed += 1
                    continue
                checked += 1
                hits = conn.execute(
                    f"""SELECT c.*, v.distance AS distance,
                               m.subject, m.metadata, m.workspace, m.workspace_canonical
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
                        peer_attrs = {
                            str(row["attr_norm"]): [
                                str(v) for v in self.db.claims.coexisting_values(
                                    peer_id, str(row["attr_norm"]),
                                )
                            ]
                            for row in self.db.claims.current_claims(peer_id)
                        }
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
                        payload={
                            "route": "notice_ready",
                            "reason": "claim_attr_vector_gate",
                            "source": "claim_conflict",
                            "slot_key": slot_key,
                            "slot_provenance": {
                                "entity": "workspace", "scope": "subject",
                                "attribute": "claims_channel",
                            },
                            "attr_cos": round(float(attr_cos or 1.0), 4),
                            "member_versions": [
                                {"memory_id": int(memory_id), "version": version,
                                 "value": str(claim["value_norm"]),
                                 "evidence": {"quote": str(claim["value"])}},
                                {"memory_id": peer_id, "version": int(hit["memory_version"] or 1),
                                 "value": str(hit["value_norm"]),
                                 "evidence": {"quote": str(hit["value"])}},
                            ],
                            "value_groups": [
                                {"normalized_value": str(claim["value_norm"]),
                                 "display_value": str(claim["value"]),
                                 "members": [f"{memory_id}@{version}"]},
                                {"normalized_value": str(hit["value_norm"]),
                                 "display_value": str(hit["value"]),
                                 "members": [f"{peer_id}@{int(hit['memory_version'] or 1)}"]},
                            ],
                            "candidate_key": {
                                "detector_version": CONFLICT_DETECTOR_VERSION,
                                "members": sorted([
                                    f"{memory_id}@{version}",
                                    f"{peer_id}@{int(hit['memory_version'] or 1)}",
                                ]),
                                "evidence": [],
                            },
                            "left_evidence": {"text": str(claim["value"])},
                            "right_evidence": {"text": str(hit["value"])},
                            "claims_channel": True,
                        },
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
                    bridge_outcome = self._run_claim_bridge(
                        conn, int(memory_id), version, record, claim,
                        own_vector, bridge_candidate_rows, skip,
                    )
                    if bridge_outcome == "created":
                        notices += 1
                        fired_attrs.add(str(claim["attr_norm"]))
                    elif bridge_outcome == "unresolved":
                        unresolved_bridges += 1
        result: dict[str, Any] = {"claims_checked": checked, "notices": notices}
        if capped_count:
            result["claims_notices_capped"] = capped_count
        if versional_vetoed:
            result["versional_vetoed"] = versional_vetoed
        if unresolved_bridges:
            result["claim_bridge_unresolved"] = unresolved_bridges
        return result

    def _run_claim_bridge(
        self, conn: "sqlite3.Connection", memory_id: int, version: int,
        record: dict[str, Any], claim: dict[str, Any], attr_vector: list[float],
        candidate_rows: "list[dict[str, Any]]", skip: "set[int]",
    ) -> str:
        """Gate-v2 G6 单边桥: own claim (attr, value) has no same-attr peer
        claim — case a of the three-case dispatch. The attr vector aims at
        the peer's sentence rows (the channel-C KNN shape); the TOP row's
        text goes to Qwen with a prompt naming the attr; an extracted value
        that differs from the own claim lands the notice. Returns
        created / unresolved / skipped."""
        from ..constants import SEMANTIC_CROSS_KNN_WINDOW
        from .gates import dispatch_hint_text

        backend = self._ensure_semantic_backend()
        if backend is None:
            return "skipped"
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
            return "unresolved"
        peer_id, peer_version = row_peer
        if peer_id in skip:
            return "skipped"
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
        try:
            forward = backend.classify_pair(
                left_env, right_env, deadline_monotonic=None, retry_allowed=False,
            )
        except TypeError:
            forward = backend.classify_pair(left_env, right_env)
        gate = evaluate_single_direction_extraction(
            signal_extraction(forward), left_env, right_env,
        )
        if gate.state != "notice_ready":
            return "unresolved"
        extracted_b = str(gate.value_b or "")
        if not extracted_b or normalize_value(extracted_b) == normalize_value(str(claim["value_norm"])):
            return "unresolved"  # extracted the SAME value: no conflict
        slot_key = _retired_gate_slot_key(
            record.get("workspace_canonical") or record.get("workspace"),
            str(claim["attr_norm"]), str(record.get("subject") or ""),
        )
        outcome = self.db.record_semantic_notice(
            memory_id=memory_id, peer_id=peer_id, severity="normal",
            notice_type="claim_conflict",
            title=f"Claim conflict with #{peer_id}",
            message=f"claim bridge attr {claim['attr']} value differs",
            payload={
                "route": "notice_ready",
                "reason": "claim_bridge_extract_value",
                "source": "claim_conflict",
                "slot_key": slot_key,
                "slot_provenance": {
                    "entity": "workspace", "scope": "subject",
                    "attribute": "claim_bridge",
                },
                "attr_cos": 1.0,
                "member_versions": [
                    {"memory_id": memory_id, "version": version,
                     "value": str(claim["value_norm"]),
                     "evidence": {"quote": str(claim["value"])}},
                    {"memory_id": peer_id, "version": peer_version,
                     "value": extracted_b,
                     "evidence": {"quote": row_text}},
                ],
                "value_groups": [
                    {"normalized_value": str(claim["value_norm"]),
                     "display_value": str(claim["value"]),
                     "members": [f"{memory_id}@{version}"]},
                    {"normalized_value": extracted_b,
                     "display_value": extracted_b,
                     "members": [f"{peer_id}@{peer_version}"]},
                ],
                "candidate_key": {
                    "detector_version": CONFLICT_DETECTOR_VERSION,
                    "members": sorted([f"{memory_id}@{version}", f"{peer_id}@{peer_version}"]),
                    "evidence": [],
                },
                "left_evidence": {"text": str(claim["value"])},
                "right_evidence": {"text": row_text},
                "claims_channel": True,
                "claim_bridge": True,
            },
            dedupe_key=notice_dedupe_key(
                memory_id, peer_id, version, peer_version, "claim_conflict",
            ),
            left_version=version, right_version=peer_version,
            source="claim_conflict",
        )
        return "created" if outcome.get("outcome") == "created" else "skipped"

    def check_claim_sentence_conflicts(
        self, memory_id: int, snapshot: dict[str, Any],
        skip_peers: "set[int] | None" = None,
        allowed_memory_ids: "list[int] | None" = None,
    ) -> dict[str, Any]:
        """Gate-v2 G6b 通道 C: claims×sentences across the clean neighbour
        list (owner 2026-09-23). Each own claim's ATTR vector queries the
        sentence rows inside the G5 clean list (claims never depend on the
        peer filling claims); pairs passing the cosine band go to Qwen as
        case a (the prompt names the attr). Versional attrs are exempted
        (D1, own counter). Cross-channel dedup rides skip_peers (channel A's
        surfaced peers) and the shared per-write notice cap."""
        from ..constants import (
            CLAIMS_MAX_NOTICES_PER_WRITE,
            SEMANTIC_CANDIDATE_COS_CEIL,
            SEMANTIC_CANDIDATE_COS_FLOOR,
            SEMANTIC_CROSS_KNN_WINDOW,
        )
        from ..semantic_conflict import attr_is_versional, vector_cosine
        from .gates import dispatch_hint_text

        record = snapshot if snapshot.get("content") is not None else (
            self.db.get_memory(int(memory_id)) or {}
        )
        version = int(record.get("version") or 1)
        if not self.db.state.sqlite_vec_available:
            return {"channel_c": True, "reason": "vec_unavailable"}
        own_claims = self.db.claims.current_claims(int(memory_id))
        if not own_claims or allowed_memory_ids is None or not allowed_memory_ids:
            return {"channel_c": True, "claims_checked": 0, "notices": 0}
        workspace = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        skip = skip_peers or set()
        # own claim vectors (attr embeddings published on the write path)
        own_vectors: dict[int, list[float]] = {}
        with self.db.connection() as conn:
            for row in conn.execute(
                """SELECT c.id AS cid, v.embedding FROM memory_claims c
                   LEFT JOIN memory_claim_vec v ON v.id=c.id
                   WHERE c.memory_id=? AND c.memory_version=?""",
                (int(memory_id), version),
            ).fetchall():
                if row["embedding"] is not None:
                    own_vectors[int(row["cid"])] = self.db.evidence._blob_to_vector(
                        bytes(row["embedding"])
                    )
        notices = 0
        capped = 0
        versional = 0
        unresolved = 0
        checked = 0
        backend = self._ensure_semantic_backend()
        embedder, _warnings = self._ensure_active_embedder()
        peer_content_cache: dict[int, dict[str, Any]] = {}

        def peer_row(peer_id: int) -> dict[str, Any]:
            if peer_id not in peer_content_cache:
                peer_content_cache[peer_id] = self.db.get_memory(peer_id) or {}
            return peer_content_cache[peer_id]

        for claim in own_claims:
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
            )
            for hit in hits:
                if notices >= CLAIMS_MAX_NOTICES_PER_WRITE:
                    capped += 1
                    break
                peer_id = int(hit["memory_id"])
                if peer_id in skip:
                    continue  # channel A already surfaced this peer
                # cosine band on the TRUE attr-vector×sentence cosine
                hit_vectors = self.db.evidence.row_vectors_for_ids(
                    [int(hit["id"])],
                )
                vector = hit_vectors.get(int(hit["id"]))
                if not vector:
                    continue
                cos = vector_cosine(attr_vector, vector)
                if not (SEMANTIC_CANDIDATE_COS_FLOOR <= cos < SEMANTIC_CANDIDATE_COS_CEIL):
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
                if backend is None:
                    unresolved += 1
                    continue
                try:
                    forward = backend.classify_pair(
                        left_env, right_env, deadline_monotonic=None, retry_allowed=False,
                    )
                except TypeError:
                    forward = backend.classify_pair(left_env, right_env)
                gate = evaluate_single_direction_extraction(
                    signal_extraction(forward), left_env, right_env,
                    attr_cos=_attr_cos_or_none(embedder, signal_extraction(forward)),
                )
                if gate.state != "notice_ready":
                    unresolved += 1
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
                    payload={
                        "route": "notice_ready",
                        "reason": "claim_channel_c_attr_sentence",
                        "source": "claim_conflict",
                        "slot_key": slot_key,
                        "slot_provenance": {
                            "entity": "workspace", "scope": "subject",
                            "attribute": "claims_channel_c",
                        },
                        "attr_cos": round(float(cos), 4),
                        "member_versions": [
                            {"memory_id": int(memory_id), "version": version,
                             "value": str(claim["value_norm"]),
                             "evidence": {"quote": str(claim["value"])}},
                            {"memory_id": peer_id, "version": peer_version,
                             "value": extracted,
                             "evidence": {"quote": str(hit.get("text") or "")}},
                        ],
                        "value_groups": [
                            {"normalized_value": str(claim["value_norm"]),
                             "display_value": str(claim["value"]),
                             "members": [f"{memory_id}@{version}"]},
                            {"normalized_value": normalize_value(extracted),
                             "display_value": extracted,
                             "members": [f"{peer_id}@{peer_version}"]},
                        ],
                        "candidate_key": {
                            "detector_version": CONFLICT_DETECTOR_VERSION,
                            "members": sorted([
                                f"{memory_id}@{version}", f"{peer_id}@{peer_version}",
                            ]),
                            "evidence": [],
                        },
                        "left_evidence": {"text": str(claim["value"])},
                        "right_evidence": {"text": str(hit.get("text") or "")},
                        "claims_channel": True,
                        "channel_c": True,
                    },
                    dedupe_key=notice_dedupe_key(
                        int(memory_id), peer_id, version, peer_version, "claim_conflict",
                    ),
                    left_version=version, right_version=peer_version,
                    source="claim_conflict",
                )
                if outcome.get("outcome") == "created":
                    notices += 1
        result: dict[str, Any] = {
            "channel_c": True, "claims_checked": checked, "notices": notices,
        }
        if capped:
            result["channel_c_capped"] = capped
        if versional:
            result["channel_c_versional_vetoed"] = versional
        if unresolved:
            result["channel_c_unresolved"] = unresolved
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
                def _env(memory: dict[str, Any], quote: str) -> dict[str, Any]:
                    return {
                        "quote": quote[:1000], "subject": str(memory.get("subject") or "")[:200],
                        "tags": list(memory.get("tags") or [])[:20],
                        "workspace_canonical": memory.get("workspace_canonical") or memory.get("workspace"),
                        "memory_id": int(memory.get("id") or 0),
                        "version": int(memory.get("version") or 1),
                        "event_time": memory.get("event_time"),
                        "metadata": {},
                    }
                try:
                    forward = backend.classify_pair(
                        _env(left, left_text), _env(right, right_text),
                        retry_allowed=not self._semantic_worker.has_pending_jobs(),
                    )
                except TypeError:
                    forward = backend.classify_pair(_env(left, left_text), _env(right, right_text))
                gate = evaluate_single_direction_extraction(
                    signal_extraction(forward), _env(left, left_text), _env(right, right_text),
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
            payload={
                "route": "notice_ready", "reason": reason,
                "prompt_version": PAIR_PROMPT_VERSION,
                "anchors": decision.anchors,
                "slot_key": slot_key,
                "slot_provenance": {"entity": "workspace", "scope": "subject", "attribute": "backlog"},
                "member_versions": [
                    {"memory_id": left_id, "version": left_version, "value": value_a,
                     "evidence": {"quote": left_text}},
                    {"memory_id": right_id, "version": right_version, "value": value_b,
                     "evidence": {"quote": right_text}},
                ],
                "value_groups": [
                    {"normalized_value": value_a, "display_value": value_a, "members": [f"{left_id}@{left_version}"]},
                    {"normalized_value": value_b, "display_value": value_b, "members": [f"{right_id}@{right_version}"]},
                ],
                "candidate_key": {
                    "detector_version": CONFLICT_DETECTOR_VERSION,
                    "members": sorted([f"{left_id}@{left_version}", f"{right_id}@{right_version}"]),
                    "evidence": [],
                },
                "left_evidence": {"text": left_text},
                "right_evidence": {"text": right_text},
                "backlog": True,
            },
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
        for embed_result in embedder.embed_texts([seg.text for seg in row_segments]):
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
                    results = embedder.embed_texts([seg.text for seg in batch])
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
        if not record or record.get("status") != "active":
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
        results = embedder.embed_texts([segment.text for segment in segments])
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
        """Run the bounded notice gate and report completion to sync callers."""
        record = self.db.get_memory(int(memory_id))
        if record and record.get("status") == "pending":
            # A pending (workspace-activation) memory is not an incomplete
            # check: the conflict job is simply skipped until activation,
            # matching the "skipped" semantics used by index_memory above.
            return {"status": "skipped", "reason": "pending_workspace_activation", "notices_created": 0}
        if not record or record.get("status") != "active":
            return {"status": "incomplete", "reason": "memory_not_active", "notices_created": 0}
        if int(record.get("version") or 1) != int(snapshot.get("version") or 1):
            return {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
        content = str(record.get("content") or "")
        # 0.16.12 P2-T2: the row's content_sha column IS sha256(content) (set
        # at insert, recomputed on every content edit) — compare against it
        # instead of re-hashing the content again (legacy NULL falls back).
        row_sha = str(record.get("content_sha") or "")
        if not row_sha:
            row_sha = hashlib.sha256(content.encode()).hexdigest()
        if row_sha != snapshot.get("content_hash"):
            return {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
        # 0.16.12 P2-T6: the whole job body runs on ONE read-only connection
        # under an explicit read transaction — per-unit evidence_knn, per-pair
        # exists/closed probes and the peer prefetch all reuse it instead of
        # opening one connection each. The BEGIN gives the job a single WAL
        # snapshot (the old open-per-read behaviour saw a different
        # point-in-time on every read); the job's own notice writes keep their
        # own per-notice write transactions and never touch this connection.
        with self.db.connection() as job_conn:
            job_conn.execute("BEGIN")
            job_started = time.monotonic()
            try:
                result = self._process_conflicts_job(
                    memory_id, snapshot, record, job_conn, content, row_sha,
                )
            finally:
                try:
                    job_conn.rollback()
                except sqlite3.Error:
                    pass
        # 0.16.12 eval contract: ACTUAL job execution time (embedding +
        # KNN candidate collection + Qwen pairs + notice writes), excluding
        # the caller's notice_sync_wait window — additive receipt key so the
        # harness can report processing cost without the wait-window noise.
        result["elapsed_ms"] = round((time.monotonic() - job_started) * 1000, 1)
        return result

    def _process_conflicts_job(
        self, memory_id: int, snapshot: dict[str, Any],
        record: dict[str, Any], job_conn: "sqlite3.Connection",
        content: str, row_sha: str,
    ) -> dict[str, Any]:
        embedder, _ = self._ensure_active_embedder()
        if embedder is None:
            return {"status": "incomplete", "reason": "embedder_unavailable", "notices_created": 0}
        if self.db.get_vec_index_state().get("state") in {"mismatch", "failed"}:
            return {
                "status": "incomplete",
                "reason": "embedding_space_rebuild_required",
                "notices_created": 0,
            }
        # Spec §5/§15.3: while a conflict group is applying, versions produced
        # by its apply plan must not re-notify THE SAME conflict. Suppression is
        # therefore slot-scoped and applied only after the gate resolves the
        # candidate's slot_key, so a genuinely different conflict between the
        # same two memories is still examined and surfaced. Validation is
        # server-side against the live conflict rows; the trusted context only
        # names which row to revalidate.
        applying_slots: set[str] = set()
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
        min_budget = SEMANTIC_MIN_PAIR_BUDGET_MS / 1000.0
        # Behaviour change (v3 hardening): each degradation reason is counted
        # at most once per task. The pair loop can hit the same technical
        # failure for many pairs, and counting every hit made
        # _check_degradation_count grow with pair count rather than with
        # distinct failure modes. reasons_seen keeps the per-task list that
        # _check_degradation_reason (last write wins) would otherwise lose.
        degradation_reasons: set[str] = set()
        reasons_seen: list[str] = []

        def record_degradation(reason: str, sample: str | None = None) -> None:
            if reason in degradation_reasons:
                return
            degradation_reasons.add(reason)
            reasons_seen.append(reason)
            self._tools._record_check_degradation(reason, sample)

        def backlog_deadline() -> float | None:
            # C2: detection-phase deadline = max(fairness wall, this job's
            # own budget counted from publish completion). The fairness wall
            # (oldest queued job's enqueue time + timeout, from the worker's
            # pending snapshot) is unchanged; publish_done_at carries the
            # execution-time anchor the pending snapshot cannot know.
            value = self._semantic_worker.pending_job_deadline(
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

        max_units = max(1, SEMANTIC_MAX_EVIDENCE_UNITS)
        max_rows = max(1, SEMANTIC_MAX_ROWS)
        workspace = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        by_peer: dict[int, tuple[dict[str, Any], Any, Any, float]] = {}
        reached_pair: set[int] = set()

        def _enqueue_backlog(
            entries: "list[tuple[int, tuple[dict[str, Any], Any, Any, float]]]",
        ) -> tuple[int, int]:
            """0.17.0 P2-4.2: truncation leftovers land in conflict_backlog
            instead of vanishing. Identity = detector version + both
            members@version + row anchors (review A7: a detector bump or a
            member edit invalidates the frozen pair)."""
            enqueued = 0
            evicted_total = 0
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
                    left_memory_id=int(memory_id), left_version=left_version,
                    right_memory_id=int(peer_id), right_version=right_version,
                    left_text=str(seg_view.text), right_text=str(hit.get("text") or ""),
                    pair_score=score,
                )
                if outcome.get("outcome") in {"queued", "duplicate"}:
                    enqueued += 1
                evicted_total += int(outcome.get("evicted") or 0)
            return enqueued, evicted_total

        # P2-T2: same digest as the stale check above — the maintained
        # content_sha column (or its recompute fallback), never a fresh hash.
        content_hash = row_sha
        # C2: publish_done_at anchors this job's own detection budget AFTER
        # the index phase (embedding is index work, not conflict budget).
        publish_done_at: list[float] = []
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
        max_segments = max_rows if rows_mode else max_units
        segments_capped_reason = "rows_capped" if rows_mode else "evidence_units_capped"
        # 0.16.0 E10① (§6⑳): same-memory internal examination comes FIRST —
        # units are in hand (no KNN), the rule is deterministic, and the
        # finding lands in the dedicated internal_conflicts structure (the
        # conflicts table's pair invariants reject a single memory@version
        # twice). It consumes no Qwen budget; the cross-memory loop below is
        # untouched in shape, and a truncated cross loop still reports the
        # internal findings already landed this run.
        # 0.16.2 (owner, unified flow): the internal check uses the SAME
        # filter-plus-slot-extraction logic as the cross-memory route. A
        # contradiction a reader would flag in one document must not become
        # invisible just because it was written inside ONE memory.
        # 0.16.4 §0.5/§2: the whole filter sequence is ONE shared gate —
        # internal_pair_admission (scan_pipeline) — called identically by
        # the scan side; the callers differ only in what an admitted pair
        # means. Here: EVERY admitted shape (check AND notify) collects for
        # the Qwen final review — a real in-memory self-contradiction has a
        # recognition duty, and Qwen's verdict is the triple
        # ready→pending+attribution / definitive negative→dismissed veto /
        # technical failure→pending unannotated (fail-open).
        internal_found = 0
        internal_version = int(record.get("version") or 1)
        # 0.17.0 P2-3.1: internal (same-memory) pairs are row segments in
        # rows mode (row_index lands in internal_conflicts' unit_a/unit_b —
        # the 0.17.0 detector bump separates the index semantics cleanly).
        from ..scan_pipeline import internal_pair_admission

        internal_qwen_pairs: list[tuple[Any, Any, Any]] = []
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
        units_examined = 0
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
        # value gate.
        no_difference_filtered = 0
        # Spec §15.5: a bounded check that ran out of budget must not later
        # claim checked_no_notice. The two truncation causes report
        # distinctly (2026-09-10 #957/#959 diagnosis: the shared string cost
        # an extra investigation round): the per-memory evidence-unit cap is
        # evidence_units_capped; the fair job deadline stays
        # notice_budget_exhausted. The cap is checked first so a state where
        # both hold attributes to the more specific cause.
        truncation_reason: str | None = None
        # 0.17.0 P2-3.1: the cross loop walks the normalized segments —
        # row_knn in rows mode (candidates are clean short sentences or
        # header-folded table rows), evidence_knn otherwise. Rows carry no
        # 'text'-only kind filter (table rows are first-class candidates).
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
        prefiltered_rows = 0
        rows_covered_by_claims = 0
        # Gate-v2 G5 ②″ 记忆级一揽子筛选: ONE subject-row coarse KNN builds
        # the neighbour list; memory_pair_excluded vets each neighbour on
        # subject/tags alone; the sentence KNN then runs ONLY inside the
        # clean list (rowid-IN restriction — window slots are not burned on
        # unrelated or already-excluded memories). 宽不罚——窄才漏.
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
                prefix="", body=str(record.get("subject") or ""),
            )
            subject_vec = subject_embed.embedding or None
        allowed_memory_ids: list[int] | None = None
        memory_pairs_excluded = 0
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
            allowed_memory_ids = [
                int(n["memory_id"]) for n in neighbours
                if int(n["memory_id"]) not in excluded_ids
            ]
            memory_pairs_excluded = len(excluded_ids)
            if not allowed_memory_ids:
                allowed_memory_ids = []  # everything screened out: no KNN at all
        below_cos_floor = 0
        repeatability_skipped = 0
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
                        rows_covered_by_claims += 1
                    else:
                        prefiltered_rows += 1
                    continue
                if units_examined >= max_segments:
                    truncation_reason = truncation_reason or segments_capped_reason
                    continue  # detection capped; keep draining for publish
                active_deadline = backlog_deadline()
                if active_deadline is not None and time.monotonic() >= active_deadline:
                    truncation_reason = truncation_reason or "notice_budget_exhausted"
                    continue  # budget gone; keep draining for publish
                units_examined += 1
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
                below_cos_floor += len(below_floor_pairs)
                repeatability_skipped += len(at_ceil_pairs)
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
                        no_difference_filtered += 1
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
            # run yet at this point (it needs the backend fetched below), so
            # the collected keepers land unannotated here — fail-open, never
            # lost. (Adversarial self-review: without this, an evidence-unit
            # cap or budget exhaustion mid-collection silently dropped every
            # internal keep pair of this run.)
            for unit_a, unit_b, internal_decision in internal_qwen_pairs:
                if self.db.internal_conflicts.create(
                    memory_id=int(memory_id), memory_version=internal_version,
                    unit_a=unit_a.unit_index, unit_b=unit_b.unit_index,
                    quote_a=unit_a.text, quote_b=unit_b.text,
                    span_a=[unit_a.start_offset, unit_a.end_offset],
                    span_b=[unit_b.start_offset, unit_b.end_offset],
                    reason=str(internal_decision.reason or ""),
                    detector_version=CONFLICT_DETECTOR_VERSION,
                ):
                    internal_found += 1
            record_degradation(truncation_reason)
            early_result: dict[str, Any] = {
                "status": "incomplete", "reason": truncation_reason,
                "notices_created": 0, "reasons_seen": reasons_seen,
                # Internal-truncation exits before the cross-memory loop even
                # starts, so no pairs were examined yet (counter not yet live).
                "pairs_examined": 0,
            }
            if internal_found:
                early_result["internal_conflicts"] = internal_found
            # 0.17.0 P2-4.2: collected-but-unexamined candidates go to the
            # backlog (owner design #8) — the receipt says how many.
            early_backlogged, early_evicted = _enqueue_backlog(list(by_peer.items()))
            if early_backlogged:
                early_result["backlogged"] = early_backlogged
            if early_evicted:
                early_result["backlog_evicted"] = early_evicted
            return early_result

        backend = self._ensure_semantic_backend()
        # C4 soft ordering (⑦ 定案): rank same-level pairs by subject+tags
        # overlap before distance — the Qwen budget should spend on pairs the
        # owner's signals (subject/tag) already flag as related. Zero-overlap
        # pairs are only ordered later, never excluded. (0.16.4 §1: the
        # notify-first key lost its subject — only check shapes remain.)
        from ..semantic_conflict import vector_cosine

        hint_vectors = self.db.memories.subject_tags_vectors(
            [memory_id, *[peer_id for peer_id in by_peer]],
        )
        own_vector = hint_vectors.get(int(memory_id))
        if own_vector is None:
            overlap_rank = {peer_id: 0.0 for peer_id in by_peer}
        else:
            overlap_rank = {
                peer_id: vector_cosine(own_vector, hint_vectors.get(peer_id))
                for peer_id in by_peer
            }
        # 0.17.0 P2-3.4: pair_score orders the Qwen budget — value features
        # lead (routed numeric + both-sides-extractable), C4 subject/tags
        # overlap is the base, row distance the tiebreak. Order-only: a
        # single pair's verdict never changes (owner-approved boundary).
        # Weights initial; P2-3.2 recalibrates on the noisy corpus.
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

        ordered = sorted(
            by_peer.items(),
            key=lambda item: (
                -_pair_score(item[0], item[1]),
                -_value_gap(item[1]),
                -float(overlap_rank.get(item[0]) or 0.0),
                float(item[1][0].get("distance") or 9),
            ),
        )
        # 0.15.14 (A5): the former surfaced>=max_notice_pairs early stop is
        # gone — notices are recorded per-pair inside the loop (write-on-
        # discovery), so an early stop only saved Qwen time, which the
        # examined-pairs cap now bounds deterministically. The check examines
        # up to SEMANTIC_MAX_EXAMINED_PAIRS pairs (fair deadline first, cap
        # second) and surfaces every notice it finds; the notice count is
        # therefore bounded by the pairs cap.
        max_examined_pairs = max(1, SEMANTIC_MAX_EXAMINED_PAIRS)
        pairs_examined = 0
        surfaced = 0
        surfaced_peer_ids: set[int] = set()
        dropped_unlocalizable = 0
        backlogged = 0
        incomplete_reason: str | None = None

        def envelope(memory: dict[str, Any], quote: str) -> dict[str, Any]:
            # Gate-v2 G3: metadata.entity/scope retired — the prompt keeps the
            # (now always empty) metadata slot so the protocol shape is stable.
            return {
                "quote": quote[:1000], "subject": str(memory.get("subject") or "")[:200],
                "tags": list(memory.get("tags") or [])[:20],
                "workspace_canonical": memory.get("workspace_canonical") or memory.get("workspace"),
                "memory_id": int(memory.get("id") or 0), "version": int(memory.get("version") or 1),
                "event_time": memory.get("event_time"),
                "metadata": {},
            }

        def classify(left_env: dict[str, Any], right_env: dict[str, Any]) -> Any:
            if backend is None:
                # The per-pair guard below never lets a None reach the call;
                # keep the narrowed local so the closure type-checks.
                raise RuntimeError("semantic backend unavailable mid-pair")
            try:
                # Once a pair starts, only the inference hard timeout may stop
                # it. The job budget is a fairness gate between pairs; the
                # retry gate (A6) is the same fairness idea one level down:
                # with another job queued, a protocol-invalid output fails
                # fast instead of doubling its own latency.
                return backend.classify_pair(
                    left_env, right_env, deadline_monotonic=None,
                    retry_allowed=not self._semantic_worker.has_pending_jobs(),
                )
            except TypeError:
                # Test/legacy backends implementing the original two-arg protocol.
                return backend.classify_pair(left_env, right_env)

        # 0.16.2 unified flow: internal check-keepers go through the SAME Qwen
        # slot extraction as cross-memory pairs, BEFORE the cross loop (E10①
        # order guarantee: internal findings land first and survive a
        # truncated cross loop). Shared pairs_examined budget and deadline —
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
        internal_qwen_confirmed = 0
        internal_qwen_vetoed = 0
        internal_qwen_budget = max(0, SEMANTIC_INTERNAL_QWEN_MAX_PAIRS)
        for unit_a, unit_b, internal_decision in internal_qwen_pairs:
            if internal_qwen_budget <= 0:
                # Harness regression fix: row granularity multiplied internal
                # keepers and they starved the cross pairs out of the shared
                # Qwen budget. E10① keeps its land-first guarantee — within
                # this smaller, value-ranked budget.
                break
            reason_text = str(internal_decision.reason or "")
            if backend is not None:
                active_deadline = backlog_deadline()
                budget_ok = not (
                    active_deadline is not None
                    and active_deadline - time.monotonic() < min_budget * 2
                ) and pairs_examined < max_examined_pairs
                if budget_ok:
                    pairs_examined += 1
                    internal_qwen_budget -= 1
                    env_a = envelope(record, unit_a.text)
                    env_b = envelope(record, unit_b.text)
                    if internal_decision.left_value and internal_decision.right_value:
                        env_a["rule_value"] = internal_decision.left_value
                        env_b["rule_value"] = internal_decision.right_value
                    # Single-direction judging (owner 2026-09-17): all three
                    # paths share the one-forward-extraction gate — the
                    # bidirectional mirror was falsified on the eval line.
                    forward = classify(env_a, env_b)
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
                        internal_qwen_confirmed += 1
                    else:
                        technical = (
                            (forward.error and "timeout" in str(forward.error).lower())
                            or forward.candidate_type == "backend_unavailable"
                            or forward.candidate_type == "backend_error"
                            or forward.candidate_type in {"invalid_json", "invalid_schema"}
                        )
                        if not technical and gate.reason != "qwen_unverified":
                            internal_qwen_vetoed += 1
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
                internal_found += 1

        # P2-T6: batch-prefetch every candidate peer in ONE id-IN query
        # instead of one get_memory connection per pair.
        peer_rows = self.db.get_memories_by_ids(
            [int(pid) for pid, _triple in ordered], conn=job_conn,
        )
        for peer_id, (hit, unit, decision, _pair_cos) in ordered:
            peer = peer_rows.get(int(peer_id))
            if not peer or peer.get("status") != "active":
                reached_pair.add(peer_id)  # settled (inactive) — not backlog
                continue
            record_row: dict[str, Any] = record or {}
            peer_row: dict[str, Any] = peer or {}
            left_version = int(record.get("version") or 1)
            right_version = int(peer.get("version") or 1)
            if self.db.semantic_notices.is_semantic_pair_closed_on_conn(
                job_conn, memory_id, peer_id, left_version, right_version,
            ):
                reached_pair.add(peer_id)  # settled (closed) — not backlog
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
            if direct is not None:
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
                    record_degradation("qwen_unavailable")
                    incomplete_reason = "qwen_unavailable"
                    continue
                active_deadline = backlog_deadline()
                if active_deadline is not None and active_deadline - time.monotonic() < min_budget * 2:
                    record_degradation("qwen_budget_exhausted")
                    incomplete_reason = "qwen_budget_exhausted"
                    continue
                # A5 deterministic cap: only pairs that actually reach Qwen count;
                # skipped (closed/inactive) pairs never consume the budget.
                if pairs_examined >= max_examined_pairs:
                    record_degradation("pairs_examined_capped")
                    incomplete_reason = "pairs_examined_capped"
                    break
                pairs_examined += 1
                reached_pair.add(peer_id)  # Qwen examined — settled
                left_env = envelope(record_row, unit.text)
                right_env = envelope(peer_row, str(hit.get("text") or ""))
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
                forward_signal = classify(left_env, right_env)
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
                    dropped_unlocalizable += 1
                    continue
                else:
                    reason = gate.reason
                if reason == "qwen_unverified":
                    # Grounding failed: uncertain — fail-closed for notices and
                    # the pair remains a scan review candidate. Owner ruling
                    # #9: unlocalizable candidates are DROPPED, counted loud.
                    record_degradation(reason)
                    dropped_unlocalizable += 1
                    incomplete_reason = reason
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
                    record_degradation(reason, sample)
                    incomplete_reason = reason
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
            member_versions = [
                {"memory_id": memory_id, "version": left_version, "value": gate.value_a,
                 "evidence": {"quote": unit.text, "start": unit.start_offset, "end": unit.end_offset}},
                {"memory_id": peer_id, "version": right_version, "value": gate.value_b,
                 "evidence": {"quote": hit.get("text"), "start": hit.get("start_offset"), "end": hit.get("end_offset")}},
            ]
            # Model output may omit keys (or parsed may not be a dict at all):
            # fall back to the gate's normalised value instead of raising
            # KeyError (mirrors the scan-path defence in tools.py).
            forward_parsed = (
                forward_signal.parsed
                if forward_signal is not None and isinstance(forward_signal.parsed, dict)
                else {}
            )
            value_groups = [
                {"normalized_value": gate.value_a, "display_value": forward_parsed.get("value_a") or gate.value_a,
                 "members": [f"{memory_id}@{left_version}"]},
                {"normalized_value": gate.value_b, "display_value": forward_parsed.get("value_b") or gate.value_b,
                 "members": [f"{peer_id}@{right_version}"]},
            ]
            outcome = self.db.record_semantic_notice(
                memory_id=memory_id, peer_id=peer_id,
                # 0.16.4 §1: cross-memory notify is excluded at collection,
                # so the write-time notice severity is uniformly normal.
                severity="normal",
                notice_type="semantic_evidence",
                title=f"Possible memory change with #{peer_id}", message=decision.reason,
                payload={
                    "route": "notice_ready", "reason": gate.reason,
                    "prompt_version": PAIR_PROMPT_VERSION,
                    "anchors": decision.anchors,
                    "slot_key": slot_key,
                    "slot_provenance": {
                        "entity": "workspace", "scope": "subject",
                        "attribute": (
                            "deterministic_skeleton" if direct is not None
                            else "single_direction_extraction"
                        ),
                    },
                    "member_versions": member_versions,
                    "value_groups": value_groups,
                    "candidate_key": {
                        "detector_version": CONFLICT_DETECTOR_VERSION,
                        "members": sorted([f"{memory_id}@{left_version}", f"{peer_id}@{right_version}"]),
                        "evidence": [],
                    },
                    "left_evidence": {
                        "text": unit.text, "start_offset": unit.start_offset,
                        "end_offset": unit.end_offset,
                    },
                    "right_evidence": {
                        "text": hit.get("text"), "start_offset": hit.get("start_offset"),
                        "end_offset": hit.get("end_offset"),
                    },
                    "left_content_hash": evidence_content_hash(content),
                    "right_content_hash": evidence_content_hash(str(peer.get("content") or "")),
                    "qwen_signal": qwen,
                },
                dedupe_key=notice_dedupe_key(
                    memory_id, peer_id, left_version, right_version, "semantic_evidence",
                ),
                left_version=left_version, right_version=right_version,
                source="semantic_evidence",
            )
            # surfaced counts notices actually created (deduped pairs were
            # already surfaced) — since A5 it is a result summary, not a gate.
            if outcome.get("outcome") == "created":
                surfaced += 1
                surfaced_peer_ids.add(int(peer_id))
            elif outcome.get("outcome") not in {"deduped"}:
                # Second-round review: a ready pair whose notice could not be
                # persisted (workspace_mismatch / invalid_snapshot /
                # unavailable / error) must not vanish silently — without this
                # the run could report checked_no_notice while a real conflict
                # was found and lost.
                record_degradation("notice_write_failed")
                incomplete_reason = "notice_write_failed"
        # 0.17.0 P2-4.2: budget/cap leftovers land in the backlog — bounded,
        # visible, never silently dropped (owner design #8). Stale/duplicate
        # keys report as enqueued here; eviction counts ride the store.
        leftover_entries = [item for item in ordered if item[0] not in reached_pair]
        sweep_evicted = 0
        if leftover_entries:
            backlogged, sweep_evicted = _enqueue_backlog(leftover_entries)
        if surfaced:
            result: dict[str, Any] = {
                "status": "completed", "outcome": "notices_created", "notices_created": surfaced,
                # 0.17.0: peers the evidence channel surfaced THIS run — the
                # claims channel skips them (cross-channel single-report,
                # plan appendix C-7 ruling: strongest evidence wins).
                "surfaced_peers": sorted(surfaced_peer_ids),
            }
            if internal_found:
                result["internal_conflicts"] = internal_found
            if incomplete_reason:
                # Notices went out, but later pairs hit a truncation/degradation
                # — surface it instead of a bare completed (second-round
                # review): the caller would otherwise read a bounded, partial
                # check as a full one.
                result["truncated"] = True
        elif incomplete_reason:
            result = {"status": "incomplete", "reason": incomplete_reason, "notices_created": 0}
        else:
            result = {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0}
        if internal_found and "internal_conflicts" not in result:
            # Internal findings survive a truncated cross-memory loop: they
            # were landed BEFORE the loop ran (E10① order guarantee).
            result["internal_conflicts"] = internal_found
        # 0.16.2 write-time pre-gate visibility (conditional — the unfiltered
        # zero case keeps the exact-shape response contract unchanged):
        # what the deterministic filters killed this run, and what the
        # unified internal Qwen flow confirmed/vetoed.
        filter_summary: dict[str, int] = {}
        if no_difference_filtered:
            filter_summary["no_difference_skipped"] = no_difference_filtered
        # Gate-v2 G4 observability: the prefilter/coverage/band split (the
        # unfiltered zero case keeps the exact-shape response contract).
        gate_rows: dict[str, int] = {}
        if memory_pairs_excluded:
            gate_rows["memory_pairs_excluded"] = memory_pairs_excluded
        if prefiltered_rows:
            gate_rows["prefiltered_rows"] = prefiltered_rows
        if rows_covered_by_claims:
            gate_rows["rows_covered_by_claims"] = rows_covered_by_claims
        if below_cos_floor:
            gate_rows["below_cos_floor"] = below_cos_floor
        if repeatability_skipped:
            gate_rows["repeatability_skipped"] = repeatability_skipped
        if gate_rows:
            result["candidate_gates"] = gate_rows
        # INTERNAL key (popped by the job wrapper): the G5 clean neighbour
        # list, shared by channel C (claims×sentences) — never in receipts.
        if allowed_memory_ids is not None:
            result["_allowed_memory_ids"] = allowed_memory_ids
        if internal_qwen_confirmed:
            filter_summary["internal_qwen_confirmed"] = internal_qwen_confirmed
        if internal_qwen_vetoed:
            filter_summary["internal_qwen_vetoed"] = internal_qwen_vetoed
        if dropped_unlocalizable:
            # 0.17.0 P2-3.5 (owner ruling #9): dropped unlocalizable pairs are
            # never silent — the counter rides every completed receipt.
            filter_summary["dropped_unlocalizable"] = dropped_unlocalizable
        if filter_summary:
            result["deterministic_filter"] = filter_summary
        if backlogged:
            # 0.17.0 P2-4.2: truncation leftovers went to the conflict
            # backlog instead of vanishing.
            result["backlogged"] = backlogged
        if sweep_evicted:
            # P2-4.3: cap evictions are visible, never silent.
            result["backlog_evicted"] = sweep_evicted
        if rows_mode:
            result["rows_mode"] = True
            result["rows_examined"] = int(units_examined)
        if reasons_seen:
            # Degradations may also occur on pairs before a later pair surfaces
            # a notice, so the list is attached to completed outcomes too.
            result["reasons_seen"] = reasons_seen
        # 0.16.12 perf baseline: the examination budget actually consumed this
        # run (additive receipt key; also visible on completed write receipts).
        result["pairs_examined"] = int(pairs_examined)
        return result
