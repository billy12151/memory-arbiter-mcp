"""Local-text evidence indexing and conflict candidate processing."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, TYPE_CHECKING

from ..db_generation import CONFLICT_DETECTOR_VERSION
from ..evidence import evidence_content_hash, local_text_units
from ..semantic_conflict import (
    PAIR_PROMPT_VERSION,
    decide_evidence,
    evaluate_pair_extractions,
    notice_dedupe_key,
    signal_extraction,
)

if TYPE_CHECKING:
    from ..tools import MemoryTools

# Technical failures degrade the check route and keep the job incomplete.
_TECHNICAL_REASONS = {
    "qwen_timeout", "qwen_unavailable", "qwen_backend_error",
    "qwen_invalid_output", "qwen_budget_exhausted", "notice_budget_exhausted",
}


class EvidencePipeline:
    def __init__(self, tools: "MemoryTools") -> None:
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tools, name)

    def index_memory(self, memory_id: int, record: dict[str, Any] | None = None) -> dict[str, Any]:
        current = record or self.db.get_memory(int(memory_id))
        embedder, warnings = self._ensure_embedder()
        if current is None:
            return {"status": "skipped", "reason": "memory_not_found"}
        if embedder is None:
            return {"status": "skipped", "reason": "embedder_unavailable", "warnings": warnings}
        units = local_text_units(
            str(current.get("subject") or ""), str(current.get("content") or ""),
        )
        embeddings: list[list[float]] = []
        for unit in units:
            result = embedder.embed_text(prefix="", body=unit.text)
            if not result.embedding:
                return {"status": "failed", "reason": "empty_embedding"}
            embeddings.append(list(result.embedding))
        published = self.db.evidence.publish(
            int(memory_id), int(current.get("version") or 1),
            evidence_content_hash(str(current.get("content") or "")), units, embeddings,
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

    def process_conflicts(self, memory_id: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Run the bounded notice gate and report completion to sync callers."""
        record = self.db.get_memory(int(memory_id))
        if not record or record.get("status") != "active":
            return {"status": "incomplete", "reason": "memory_not_active", "notices_created": 0}
        if int(record.get("version") or 1) != int(snapshot.get("version") or 1):
            return {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
        content = str(record.get("content") or "")
        if hashlib.sha256(content.encode()).hexdigest() != snapshot.get("content_hash"):
            return {"status": "incomplete", "reason": "stale_snapshot", "notices_created": 0}
        embedder, _ = self._ensure_embedder()
        if embedder is None:
            return {"status": "incomplete", "reason": "embedder_unavailable", "notices_created": 0}
        # Spec §5/§15.3: while a conflict group is applying, versions produced
        # by its apply plan must not re-notify the same conflict. Validation is
        # server-side against the live conflict rows (status=applying, plan
        # membership); the trusted context only names which row to revalidate.
        applying_pairs: set[frozenset[int]] = set()
        applying_slots: set[str] = set()
        applying_groups: list[dict[str, Any]] = []
        trusted = snapshot.get("trusted_applying_context")
        if isinstance(trusted, dict) and trusted.get("conflict_id") is not None:
            live = self.db.get_conflict(int(trusted["conflict_id"]))
            if live is not None and live.get("status") == "applying":
                plan_ids = {
                    int(item.get("memory_id") or 0)
                    for item in (live.get("apply_summary") or {}).get("plan") or []
                }
                trusted_memory = int(trusted.get("memory_id") or 0)
                if trusted_memory in plan_ids or (
                    live.get("resolution_memory_id") is not None
                    and trusted_memory == int(live["resolution_memory_id"])
                ):
                    applying_groups.append(live)
        applying_groups.extend(
            group for group in self.db.list_open_conflicts_for_memory_ids(
                [int(memory_id)], include_applying=True,
            ) if group.get("status") == "applying"
        )
        for group in applying_groups:
            ids = {int(member["memory_id"]) for member in group.get("member_versions") or []}
            if group.get("resolution_memory_id") is not None:
                ids.add(int(group["resolution_memory_id"]))
            for left in ids:
                for right in ids:
                    if left != right:
                        applying_pairs.add(frozenset((left, right)))
            if group.get("slot_key"):
                applying_slots.add(json.dumps(
                    group["slot_key"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                ))
        deadline = time.monotonic() + self.settings.semantic_conflict_job_timeout_ms / 1000.0
        min_budget = self.settings.semantic_conflict_min_pair_budget_ms / 1000.0
        max_units = max(1, int(getattr(
            self.settings, "semantic_conflict_max_evidence_units", 24,
        )))
        workspace = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        by_peer: dict[int, tuple[dict[str, Any], Any, Any]] = {}
        content_hash = evidence_content_hash(content)
        unit_vectors = self.db.evidence.current_text_vectors(
            int(memory_id), int(record.get("version") or 1), content_hash,
        )
        if not unit_vectors:
            # Recovery fallback for an incomplete/legacy evidence publish. The
            # normal write path has just published these vectors, so avoid a
            # second GGUF embedding pass in the common synchronous-notice path.
            unit_vectors = []
            for unit in local_text_units(str(record.get("subject") or ""), content):
                if unit.kind != "text":
                    continue
                embedded = embedder.embed_text(prefix="", body=unit.text)
                if embedded.embedding:
                    unit_vectors.append((unit, list(embedded.embedding)))
        units_examined = 0
        gathering_truncated = False
        for unit, embedding in unit_vectors:
            if units_examined >= max_units or time.monotonic() >= deadline:
                # Spec §15.5: a bounded check that ran out of budget must not
                # later claim checked_no_notice.
                gathering_truncated = True
                break
            units_examined += 1
            for hit in self.db.evidence_knn(
                embedding, k=5, workspace=workspace,
                exclude_memory_id=memory_id,
            ):
                if hit.get("kind") != "text":
                    continue
                decision = decide_evidence(unit.text, str(hit.get("text") or ""))
                if decision.action == "ignore":
                    continue
                peer_id = int(hit["memory_id"])
                existing = by_peer.get(peer_id)
                priority = 2 if decision.action == "notify" else 1
                existing_priority = 2 if existing and existing[2].action == "notify" else 1
                closer = existing is not None and float(hit.get("distance") or 9) < float(existing[0].get("distance") or 9)
                # A deterministic notify must never be demoted to a weaker
                # check decision by a merely closer neighbour.
                if existing is None or priority > existing_priority or (priority == existing_priority and closer):
                    by_peer[peer_id] = (hit, unit, decision)
        if gathering_truncated:
            self._tools._record_check_degradation("notice_budget_exhausted")
            return {"status": "incomplete", "reason": "notice_budget_exhausted", "notices_created": 0}

        backend = self._ensure_semantic_backend()
        ordered = sorted(
            by_peer.items(),
            key=lambda item: (
                item[1][2].action != "notify",
                float(item[1][0].get("distance") or 9),
            ),
        )
        # Spec §6.4: at most 1-2 memory pairs may become Agent notices per
        # write (hard cap 3 for debug/high-recall runs).
        max_pairs = int(getattr(
            self.settings, "semantic_conflict_max_notice_pairs", 2,
        ))
        surfaced = 0
        incomplete_reason: str | None = None

        def envelope(memory: dict[str, Any], quote: str) -> dict[str, Any]:
            metadata_value = memory.get("metadata")
            metadata = metadata_value if isinstance(metadata_value, dict) else {}
            return {
                "quote": quote[:1000], "subject": str(memory.get("subject") or "")[:200],
                "tags": list(memory.get("tags") or [])[:20],
                "workspace_canonical": memory.get("workspace_canonical") or memory.get("workspace"),
                "memory_id": int(memory.get("id") or 0), "version": int(memory.get("version") or 1),
                "event_time": memory.get("event_time"),
                "metadata": {key: metadata.get(key) for key in ("entity", "scope") if metadata.get(key)},
            }

        def classify(left_env: dict[str, Any], right_env: dict[str, Any]) -> Any:
            try:
                return backend.classify_pair(left_env, right_env, deadline_monotonic=deadline)
            except TypeError:
                # Test/legacy backends implementing the original two-arg protocol.
                return backend.classify_pair(left_env, right_env)

        for peer_id, (hit, unit, decision) in ordered:
            if surfaced >= max(1, min(3, max_pairs)):
                incomplete_reason = "notice_budget_exhausted"
                break
            peer = self.db.get_memory(peer_id)
            if not peer or peer.get("status") != "active":
                continue
            left_version = int(record.get("version") or 1)
            right_version = int(peer.get("version") or 1)
            if self.db.is_semantic_pair_closed(memory_id, peer_id, left_version, right_version):
                continue
            if frozenset((memory_id, peer_id)) in applying_pairs:
                # Same applying group (member or resolution): the re-entry rule
                # routes this pair to scan review, never to a fresh notice.
                continue
            if backend is None:
                self._tools._record_check_degradation("qwen_unavailable")
                incomplete_reason = "qwen_unavailable"
                continue
            if deadline - time.monotonic() < min_budget * 2:
                self._tools._record_check_degradation("qwen_budget_exhausted")
                incomplete_reason = "qwen_budget_exhausted"
                continue
            left_env = envelope(record, unit.text)
            right_env = envelope(peer, str(hit.get("text") or ""))
            started = time.monotonic()
            forward_signal = classify(left_env, right_env)
            reverse_signal = classify(right_env, left_env)
            self._tools._last_pair_duration_ms = int((time.monotonic() - started) * 1000)
            gate = evaluate_pair_extractions(
                signal_extraction(forward_signal), signal_extraction(reverse_signal), left_env, right_env,
                require_bidirectional=True,
            )
            qwen = {
                "status": gate.state, "reason": gate.reason,
                "forward_type": forward_signal.candidate_type,
                "reverse_type": reverse_signal.candidate_type,
            }
            if gate.state != "notice_ready":
                signals = (forward_signal, reverse_signal)
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
                    continue
                else:
                    reason = gate.reason
                if reason == "qwen_unverified":
                    # Grounding failed: uncertain — fail-closed for notices and
                    # the pair remains a scan review candidate.
                    self._tools._record_check_degradation(reason)
                    incomplete_reason = reason
                    continue
                if reason in _TECHNICAL_REASONS:
                    self._tools._record_check_degradation(reason)
                    incomplete_reason = reason
                    continue
                # Definitive strict-gate negatives (not_same_attribute_different_value,
                # coexist_*, direction_invalid, bidirectional_*): the pair was
                # examined and decided. The check stays complete and no
                # degradation counter fires (spec §9/§15.5/§8).
                continue
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            peer_metadata = peer.get("metadata") if isinstance(peer.get("metadata"), dict) else {}
            entity = metadata.get("entity") if metadata.get("entity") == peer_metadata.get("entity") else None
            scope = metadata.get("scope") if metadata.get("scope") == peer_metadata.get("scope") else None
            if not entity or not scope:
                incomplete_reason = "slot_provenance_insufficient"
                continue
            slot_key = {"entity": entity, "attribute": gate.attribute, "scope": scope}
            if json.dumps(slot_key, ensure_ascii=False, sort_keys=True, separators=(",", ":")) in applying_slots:
                # A third-party fact landing on a slot currently under
                # application: scan review only (spec §15.3), no new notice.
                continue
            member_versions = [
                {"memory_id": memory_id, "version": left_version, "value": gate.value_a,
                 "evidence": {"quote": unit.text, "start": unit.start_offset, "end": unit.end_offset}},
                {"memory_id": peer_id, "version": right_version, "value": gate.value_b,
                 "evidence": {"quote": hit.get("text"), "start": hit.get("start_offset"), "end": hit.get("end_offset")}},
            ]
            value_groups = [
                {"normalized_value": gate.value_a, "display_value": forward_signal.parsed["value_a"],
                 "members": [f"{memory_id}@{left_version}"]},
                {"normalized_value": gate.value_b, "display_value": forward_signal.parsed["value_b"],
                 "members": [f"{peer_id}@{right_version}"]},
            ]
            outcome = self.db.record_semantic_notice(
                memory_id=memory_id, peer_id=peer_id,
                severity="high" if decision.action == "notify" else "normal",
                notice_type="semantic_evidence",
                title=f"Possible memory change with #{peer_id}", message=decision.reason,
                payload={
                    "route": "notice_ready", "reason": gate.reason,
                    "prompt_version": PAIR_PROMPT_VERSION,
                    "anchors": decision.anchors,
                    "slot_key": slot_key,
                    "slot_provenance": {"entity": "metadata", "scope": "metadata", "attribute": "bidirectional_extraction"},
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
            # Only a notice actually created consumes the per-write budget:
            # deduped pairs were already surfaced, and error/unavailable
            # outcomes must not starve a fresh deterministic notify pair.
            if outcome.get("outcome") == "created":
                surfaced += 1
        if surfaced:
            return {"status": "completed", "outcome": "notices_created", "notices_created": surfaced}
        if incomplete_reason:
            return {"status": "incomplete", "reason": incomplete_reason, "notices_created": 0}
        return {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0}
