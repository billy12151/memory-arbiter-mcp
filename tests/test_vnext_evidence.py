from __future__ import annotations

from pathlib import Path
import sqlite3
import contextlib
import uuid
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db_generation import CONFLICT_DETECTOR_VERSION
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.evidence import evidence_content_hash, local_text_units
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.semantic_conflict import (
    decide_evidence,
    normalize_value,
)
from memory_arbiter.tools import MemoryTools
from memory_arbiter.vnext_migration import build, inspect


class FakeEmbedder:
    embedding_space_id = "fake-vnext-space"
    # ManagedEmbedder.dim since 0.15.0: the model's own output dimension is
    # the library fact source (there is no configured vec.dim any more).
    dim = 2
    last_encode_error = None

    @classmethod
    def embed_texts(cls, texts: list[str], prefix: str = "") -> list["EmbedResult"]:
        # C1: index-path batch API — the fake has no batch closure, so it
        # delegates per item (the same route ManagedEmbedder takes when
        # encode_batch is None).
        return [cls.embed_text(prefix=prefix, body=text) for text in texts]

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars=None) -> EmbedResult:
        text = f"{prefix}\n{body}".casefold()
        if "postgres" in text or "pgsql" in text:
            vector = [1.0, 0.0]
        elif "超时" in text:
            vector = [0.8, 0.2]
        else:
            vector = [0.0, 1.0]
        return EmbedResult(vector, False, len(text), len(text))


class CountingEmbedder(FakeEmbedder):
    calls = 0

    @classmethod
    def embed_text(cls, prefix: str, body: str, max_body_chars=None) -> EmbedResult:
        cls.calls += 1
        return super().embed_text(prefix, body, max_body_chars)


def make_tools(tmp_path: Path, *, semantic_enabled: bool = False) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "vnext.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
        embedding_auto_write=True,
        embedding_auto_query=True,
        semantic_conflict_enabled=semantic_enabled,
        semantic_conflict_model_path=model if semantic_enabled else None,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = FakeEmbedder()
    tools._embedder_loaded = True
    # Mirrors what the first successful embedder build does in production
    # (tools._ensure_embedder): lazy vec0-table creation at the model's dim,
    # then the space state flips with the same dim recorded as active_dim.
    assert db.ensure_vec_tables(FakeEmbedder.dim) == []
    tools.db.init_vec_index_state("fake-vnext-space", True, active_dim=FakeEmbedder.dim)
    return tools



def _pass_cos_gate(monkeypatch, cos: float = 0.85):
    """Gate-v2 G4: pass every hand-built hit through the cosine band
    (embedder-agnostic — patches the gates module, which the detection loop
    imports at call time)."""
    import memory_arbiter.pipeline.gates as _gates
    monkeypatch.setattr(
        _gates, "candidate_cos_gate",
        lambda own, hits, vecs: ([(h, cos) for h in hits], [], []),
    )

def _hits_with_metadata(tools, hits):
    """0.16.2 write-time provenance gate reads hit['metadata']; real knn rows
    always carry it (from the peer memory row). Hand-built test hits borrow
    their peer's metadata so the fixture keeps the old send-to-Qwen shape."""
    import json as _json
    for hit in hits:
        if "metadata" in hit:
            continue
        record = tools.db.get_memory(int(hit["memory_id"]))
        meta = (record or {}).get("metadata")
        if isinstance(meta, (dict, list)):
            meta = _json.dumps(meta, ensure_ascii=False)
        hit["metadata"] = meta
    return hits


def _job_snapshot(tools: MemoryTools, memory_id: int) -> dict:
    record = tools.db.get_memory(memory_id)
    return {
        "memory_id": int(memory_id),
        "version": record["version"],
        "content_hash": evidence_content_hash(record["content"]),
    }


class ModelSignal:
    """0.17.1 test stand-in for the retired extraction signal dataclass —
    the compat shim reads only .candidate."""

    def __init__(self, candidate, candidate_type, confidence, raw, parsed, error):
        self.candidate = candidate
        self.candidate_type = candidate_type
        self.confidence = confidence
        self.raw = raw
        self.parsed = parsed
        self.error = error


def _strict_pair_backend():
    class Backend:
        @staticmethod
        def classify_pair(left, right):
            import re
            left_value = re.search(r"\d+", left["quote"]).group(0)
            right_value = re.search(r"\d+", right["quote"]).group(0)
            parsed = {
                "attribute_a": "连接池上限", "value_a": left_value,
                "attribute_b": "连接池上限", "value_b": right_value,
            }
            return ModelSignal(True, "attribute_value", None, "", parsed, None)
    return Backend()

def _conflict_member(memory_id: int, value: str, quote: str, *, version: int = 1) -> dict[str, Any]:
    return ConflictMember(
        memory_id=memory_id,
        version=version,
        attribute_raw="接口超时",
        value_raw=value,
        normalized_attribute="接口超时",
        normalized_value=normalize_value(value),
        evidence_quote=quote,
        evidence_span=(0, len(quote)),
        content_hash=evidence_content_hash(quote),
        direction="a_to_b",
        prompt_version="pair-v1",
        detector_version=CONFLICT_DETECTOR_VERSION,
    ).to_dict()


def _structured_conflict_payload(left_id: int, right_id: int, *, left_version: int = 1, right_version: int = 1) -> dict[str, Any]:
    left_quote, right_quote = "接口超时为 5 秒。", "接口超时为 30 秒。"
    return {
        "slot_key": {"entity": "checkout-api", "attribute": "接口超时", "scope": "production"},
        "members": [
            _conflict_member(left_id, "5秒", left_quote, version=left_version),
            _conflict_member(right_id, "30秒", right_quote, version=right_version),
        ],
        "value_groups": [
            ConflictValueGroup(normalize_value("5 秒"), "5 秒", (f"{left_id}@{left_version}",)).to_dict(),
            ConflictValueGroup(normalize_value("30 秒"), "30 秒", (f"{right_id}@{right_version}",)).to_dict(),
        ],
        "detector_version": CONFLICT_DETECTOR_VERSION,
        "prompt_version": "pair-v1",
        "source": "scheduled_scan",
        "reason": "同一生产接口超时槽位存在不同值",
        "status": "open",
    }


def _seed_notice(
    tools: MemoryTools,
    left_id: int,
    right_id: int,
    *,
    left_version: int = 1,
    right_version: int = 1,
    complete_slot: bool = False,
) -> int:
    payload: dict[str, Any] = {
        "route": "notice_ready",
        "reason": "bidirectional_attribute_value_difference",
        "left_evidence": {"text": "接口超时为 5 秒。", "start_offset": 0, "end_offset": 10},
        "right_evidence": {"text": "接口超时为 30 秒。", "start_offset": 0, "end_offset": 11},
        "candidate_key": {
            "detector_version": CONFLICT_DETECTOR_VERSION,
            "members": sorted([f"{left_id}@{left_version}", f"{right_id}@{right_version}"]),
        },
        "task_id": f"semantic:{left_id}@{left_version}",
    }
    if complete_slot:
        structured = _structured_conflict_payload(
            left_id, right_id, left_version=left_version, right_version=right_version,
        )
        payload.update({
            "slot_key": structured["slot_key"],
            "slot_provenance": {"entity": "metadata", "scope": "metadata", "attribute": "bidirectional_extraction"},
            "member_versions": structured["members"],
            "value_groups": structured["value_groups"],
        })
    created = tools.db.record_semantic_notice(
        memory_id=left_id,
        peer_id=right_id,
        severity="high",
        notice_type="semantic_evidence",
        title=f"Possible memory change with #{right_id}",
        message="bidirectional_attribute_value_difference",
        payload=payload,
        dedupe_key=f"notice:{left_id}@{left_version}:{right_id}@{right_version}:{complete_slot}",
        left_version=left_version,
        right_version=right_version,
    )
    return int(created["notice_id"])

# Domains split to test_vnext_evidence_{publish,worker,space_rebuild,structured,scan_spans,job_budget}.py (commit 12)
