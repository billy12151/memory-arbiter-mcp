"""C4 soft ordering (v0.15.12): Qwen pair budget ranked by subject+tag overlap.

⑦ 定案 red lines pinned here:
  * ordering only — zero-overlap pairs are still fully checked under a large
    budget (exclusion would be a hard filter, which the owner rejected);
  * deterministic notify pairs stay ahead of check pairs regardless of score;
  * a missing subject_tags_vec row scores 0, ranks last, never errors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.evidence import evidence_content_hash
from memory_arbiter.tools import MemoryTools

_META = {"entity": "soft-order", "scope": "production"}


class _VecEmbedder:
    """Deterministic stand-in that publishes real 2-dim vectors."""

    embedding_space_id = "soft-order-space"
    dim = 2
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars: int | None = None) -> EmbedResult:
        text = f"{prefix}\n{body}".casefold()
        if "deploy" in text:
            vector = [1.0, 0.0]
        elif "invoice" in text or "billing" in text or "manual" in text:
            vector = [0.0, 1.0]
        else:
            vector = [0.7071067811865476, 0.7071067811865476]
        return EmbedResult(vector, False, len(text), len(text))


def make_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "so.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = _VecEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(_VecEmbedder.dim) == []
    tools.db.init_vec_index_state(_VecEmbedder.embedding_space_id, True, active_dim=_VecEmbedder.dim)
    return tools


class _RecordingBackend:
    """Classify nothing as notice; records pair evaluation order (once per pair)."""

    name = "recording"

    def __init__(self) -> None:
        self.order: list[tuple[int, int]] = []

    def classify_pair(
        self, left: dict[str, Any], right: dict[str, Any], *, deadline_monotonic: float | None = None,
    ) -> Any:
        from memory_arbiter.semantic_conflict import ModelSignal
        # forward+reverse both run per pair; dedupe by unordered id pair so
        # `order` lists each evaluated pair exactly once, in evaluation order.
        pair = {int(left["memory_id"]), int(right["memory_id"])}
        if not self.order or self.order[-1] != pair:
            self.order.append(pair)
        parsed = {
            "attribute_a": "export_format", "value_a": "json",
            "attribute_b": "export_format", "value_b": "csv",
        }
        return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)


def _write(tools: MemoryTools, content: str, subject: str, tags: list[str]) -> dict[str, Any]:
    return tools.memory_write(content=content, subject=subject, tags=tags, metadata=dict(_META))["data"]


def _snapshot(tools: MemoryTools, memory_id: int) -> dict[str, Any]:
    record = tools.db.get_memory(memory_id)
    return {
        "memory_id": int(memory_id),
        "version": record["version"],
        "content_hash": evidence_content_hash(record["content"]),
    }


def _publish_hint_vectors(tools: MemoryTools) -> None:
    """Backfill subject_tags_vec rows the async write path may not have run."""
    from memory_arbiter.evidence import evidence_content_hash  # noqa: F401
    for record in tools.db.list_memories_for_backfill() if hasattr(tools.db, "list_memories_for_backfill") else _active_records(tools):
        if record["status"] != "active":
            continue
        result = tools._embedder.embed_text(
            prefix="", body=f"{record['subject']} {' '.join(record['tags'] or [])}",
        )
        tools.db.upsert_subject_tags_vector(int(record["id"]), list(result.embedding))


def _active_records(tools: MemoryTools) -> list[dict[str, Any]]:
    with tools.db.connection() as conn:
        rows = conn.execute("SELECT * FROM memories ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def test_write_path_orders_check_level_by_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same action level: high-overlap peer evaluated before low-overlap."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    far = _write(tools, "invoice process is manual", "billing", ["billing"])
    near = _write(tools, "deploy pipeline is green", "deploy", ["deploy"])
    new = _write(tools, "deploy pipeline is blue", "deploy2", ["deploy"])
    assert tools.wait_evidence_worker_drained(timeout=5)
    _publish_hint_vectors(tools)

    # new shares subject/tags vector space with `near` (deploy) and is
    # orthogonal to `far` (invoice).
    backend = _RecordingBackend()
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)

    hits_by_peer = {
        int(near["id"]): {"memory_id": int(near["id"]), "id": 1, "kind": "text", "text": "deploy pipeline is green",
                         "start_offset": 0, "end_offset": 23, "distance": 0.9, "metadata": dict(_META)},
        int(far["id"]): {"memory_id": int(far["id"]), "id": 2, "kind": "text", "text": "invoice process is manual",
                         "start_offset": 0, "end_offset": 23, "distance": 0.1, "metadata": dict(_META)},
    }

    def fake_knn(embedding: Any, k: Any = 5, workspace: Any = None, exclude_memory_id: Any = None) -> list[dict[str, Any]]:
        return [
            {"memory_id": pid, "id": hit["id"], "kind": "text", "text": hit["text"],
             "start_offset": hit["start_offset"], "end_offset": hit["end_offset"],
             "distance": hit["distance"], "metadata": hit.get("metadata")}
            for pid, hit in hits_by_peer.items()
        ]

    monkeypatch.setattr(tools.db, "evidence_knn", fake_knn)

    result = tools._process_semantic_conflict_job(int(new["id"]), _snapshot(tools, int(new["id"])))

    # The far (0.1) but zero-overlap pair must NOT come first any more: the
    # near (0.9) same-topic pair is evaluated first under equal check level.
    first_pair = backend.order[0]
    assert int(near["id"]) in first_pair
    assert int(far["id"]) not in first_pair
    assert result["status"] in {"completed", "incomplete"}


def test_write_path_notify_level_not_demoted_by_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A deterministic notify pair outranks a check pair even with zero overlap."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    check_peer = _write(tools, "deploy pipeline is green", "deploy", ["deploy"])  # same topic
    notify_peer = _write(tools, "invoice process is manual", "billing", ["billing"])  # zero overlap
    new = _write(tools, "deploy pipeline is blue", "deploy2", ["deploy"])
    assert tools.wait_evidence_worker_drained(timeout=5)
    _publish_hint_vectors(tools)

    backend = _RecordingBackend()
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)

    from memory_arbiter import pipeline

    real_decide = pipeline.evidence.decide_evidence

    def decide(text: str, hit_text: str) -> Any:
        if "invoice" in text or "invoice" in hit_text:
            class _D:
                action = "notify"
                anchors: list[str] = []
                reason = "date contradiction"
            return _D()
        return real_decide(text, hit_text)

    monkeypatch.setattr(pipeline.evidence, "decide_evidence", decide)

    def fake_knn(embedding: Any, k: Any = 5, workspace: Any = None, exclude_memory_id: Any = None) -> list[dict[str, Any]]:
        return [
            {"memory_id": int(check_peer["id"]), "id": 1, "kind": "text", "text": "deploy pipeline is green",
             "start_offset": 0, "end_offset": 23, "distance": 0.1, "metadata": dict(_META)},
            {"memory_id": int(notify_peer["id"]), "id": 2, "kind": "text", "text": "invoice process is manual",
             "start_offset": 0, "end_offset": 23, "distance": 0.9, "metadata": dict(_META)},
        ]

    monkeypatch.setattr(tools.db, "evidence_knn", fake_knn)

    tools._process_semantic_conflict_job(int(new["id"]), _snapshot(tools, int(new["id"])))

    # 0.16.4 §1: cross-memory notify is the evolution domain — it dies at
    # collection, never reaching the sort or the Qwen loop. The check pair
    # (subject-overlap) is the only evaluated peer.
    assert backend.order, "the check pair must still be evaluated"
    assert all(
        int(notify_peer["id"]) not in order for order in backend.order
    ), "evolution-domain peer must not reach the Qwen loop"


def test_missing_hint_vector_scores_zero_not_error(tmp_path: Path) -> None:
    from memory_arbiter.semantic_conflict import vector_cosine

    assert vector_cosine(None, [1.0, 0.0]) == 0.0
    assert vector_cosine([1.0, 0.0], None) == 0.0
    assert vector_cosine([1.0], [1.0, 0.0]) == 0.0
    assert vector_cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert vector_cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert vector_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_subject_tags_vectors_batch_read(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    one = _write(tools, "deploy note", "deploy", ["deploy"])
    two = _write(tools, "invoice note", "billing", ["billing"])
    assert tools.wait_evidence_worker_drained(timeout=5)
    _publish_hint_vectors(tools)

    found = tools.db.memories.subject_tags_vectors([int(one["id"]), int(two["id"]), 999999])
    assert int(one["id"]) in found and int(two["id"]) in found
    assert 999999 not in found
    assert len(found[int(one["id"])]) == 2
    # deploy-family vector is [1,0]; invoice-family is [0,1]
    assert found[int(one["id"])][0] > 0.9
    assert found[int(two["id"])][1] > 0.9


