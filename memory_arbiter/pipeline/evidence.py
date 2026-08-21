"""Local-text evidence indexing and conflict candidate processing."""
from __future__ import annotations

import hashlib
import time
from typing import Any, TYPE_CHECKING

from ..evidence import evidence_content_hash, local_text_units
from ..semantic_conflict import (
    AttributeValueExtraction,
    PAIR_PROMPT_VERSION,
    decide_evidence,
    evaluate_pair_extractions,
    notice_dedupe_key,
)

if TYPE_CHECKING:
    from ..tools import MemoryTools


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
        for unit, embedding in unit_vectors:
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

        backend = self._ensure_semantic_backend()
        deadline = time.monotonic() + self.settings.semantic_conflict_job_timeout_ms / 1000.0
        min_budget = self.settings.semantic_conflict_min_pair_budget_ms / 1000.0
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

        def extraction(signal: Any) -> AttributeValueExtraction | None:
            parsed = signal.parsed if isinstance(signal.parsed, dict) else None
            if not parsed:
                return None
            try:
                return AttributeValueExtraction(**{key: parsed[key] for key in (
                    "attribute_a", "value_a", "attribute_b", "value_b",
                )})
            except (KeyError, TypeError):
                return None

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
            forward_signal = backend.classify_pair(left_env, right_env)
            reverse_signal = backend.classify_pair(right_env, left_env)
            self._tools._last_pair_duration_ms = int((time.monotonic() - started) * 1000)
            gate = evaluate_pair_extractions(
                extraction(forward_signal), extraction(reverse_signal), left_env, right_env,
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
                else:
                    reason = gate.reason
                self._tools._record_check_degradation(reason)
                incomplete_reason = reason
                continue
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            peer_metadata = peer.get("metadata") if isinstance(peer.get("metadata"), dict) else {}
            entity = metadata.get("entity") if metadata.get("entity") == peer_metadata.get("entity") else None
            scope = metadata.get("scope") if metadata.get("scope") == peer_metadata.get("scope") else None
            if not entity or not scope:
                incomplete_reason = "slot_provenance_insufficient"
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
                    "slot_key": {"entity": entity, "attribute": gate.attribute, "scope": scope},
                    "slot_provenance": {"entity": "metadata", "scope": "metadata", "attribute": "bidirectional_extraction"},
                    "member_versions": member_versions,
                    "value_groups": value_groups,
                    "candidate_key": {
                        "detector_version": "attribute-value-v1",
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
