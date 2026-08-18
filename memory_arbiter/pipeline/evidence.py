"""Local-text evidence indexing and conflict candidate processing."""
from __future__ import annotations

import hashlib
import time
from typing import Any, TYPE_CHECKING

from ..evidence import evidence_content_hash, local_text_units
from ..semantic_conflict import decide_evidence, notice_dedupe_key

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
        return {
            "status": "indexed" if published.get("published") else "failed",
            **published,
        }

    def process_conflicts(self, memory_id: int, snapshot: dict[str, Any]) -> None:
        record = self.db.get_memory(int(memory_id))
        if not record or record.get("status") != "active":
            return
        if int(record.get("version") or 1) != int(snapshot.get("version") or 1):
            return
        content = str(record.get("content") or "")
        if hashlib.sha256(content.encode()).hexdigest() != snapshot.get("content_hash"):
            return
        embedder, _ = self._ensure_embedder()
        if embedder is None:
            return
        workspace = (
            record.get("workspace_canonical") or record.get("workspace")
            if self.settings.isolation == "strict" else None
        )
        by_peer: dict[int, tuple[dict[str, Any], Any, Any]] = {}
        for unit in local_text_units(str(record.get("subject") or ""), content):
            if unit.kind != "text":
                continue
            embedded = embedder.embed_text(prefix="", body=unit.text)
            if not embedded.embedding:
                continue
            for hit in self.db.evidence_knn(
                embedded.embedding, k=5, workspace=workspace,
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
                if existing is None or priority > existing_priority or closer:
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
        for peer_id, (hit, unit, decision) in ordered:
            peer = self.db.get_memory(peer_id)
            if not peer or peer.get("status") != "active":
                continue
            left_version = int(record.get("version") or 1)
            right_version = int(peer.get("version") or 1)
            if self.db.is_semantic_pair_closed(memory_id, peer_id, left_version, right_version):
                continue
            qwen: dict[str, Any] = {"status": "not_required"}
            if decision.action == "check":
                if backend is None or deadline - time.monotonic() < min_budget:
                    continue
                started = time.monotonic()
                signal = backend.classify_pair(
                    {"content": unit.text}, {"content": str(hit.get("text") or "")},
                )
                self._tools._last_pair_duration_ms = int((time.monotonic() - started) * 1000)
                qwen = {
                    "status": "candidate" if signal.candidate else "rejected",
                    "type": signal.candidate_type,
                    "confidence": signal.confidence,
                }
                if not signal.candidate:
                    continue
            self.db.record_semantic_notice(
                memory_id=memory_id, peer_id=peer_id,
                severity="high" if decision.action == "notify" else "normal",
                notice_type="semantic_evidence",
                title=f"Possible memory change with #{peer_id}", message=decision.reason,
                payload={
                    "route": decision.action, "reason": decision.reason,
                    "anchors": decision.anchors,
                    "left_evidence": {
                        "text": unit.text, "start_offset": unit.start_offset,
                        "end_offset": unit.end_offset,
                    },
                    "right_evidence": {
                        "text": hit.get("text"), "start_offset": hit.get("start_offset"),
                        "end_offset": hit.get("end_offset"),
                    },
                    "qwen_signal": qwen,
                },
                dedupe_key=notice_dedupe_key(
                    memory_id, peer_id, left_version, right_version, "semantic_evidence",
                ),
                left_version=left_version, right_version=right_version,
                source="semantic_evidence",
            )
