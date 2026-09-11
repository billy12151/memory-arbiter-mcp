"""A1 (0.15.14): examined-pair timing ring + status/doctor aggregation.

The ring replaces the single last_pair_duration_ms scalar so one real run
can distinguish queue competition (long pair_ms, modest tokens) from long
decodes and retries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import SEMANTIC_PAIR_LONG_DECODE_TOKENS, SEMANTIC_PAIR_RING_SIZE
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.semantic_conflict import ModelSignal
from memory_arbiter.tools import MemoryTools

_VALID = (
    '{"attribute_a":"数据库选择","value_a":"MySQL",'
    '"attribute_b":"数据库选择","value_b":"SQLite"}'
)
_TRUNCATED = '{"attribute_a":"x","value_a":"'


class _UsageLLM:
    """Replays canned raw outputs and reports growing usage per call."""

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raw = self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]
        return {
            "choices": [{"message": {"content": raw}}],
            "usage": {"prompt_tokens": 700 + self.calls, "completion_tokens": 40 * self.calls},
        }


class FakeEmbedder:
    embedding_space_id = "fake-ring-space"
    dim = 2
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars: int | None = None) -> EmbedResult:
        text = f"{prefix}\n{body}".casefold()
        vector = [1.0, 0.0] if "json" in text else [0.0, 1.0]
        return EmbedResult(vector, False, len(text), len(text))


def make_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "ring.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = FakeEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(FakeEmbedder.dim) == []
    tools.db.init_vec_index_state(FakeEmbedder.embedding_space_id, True, active_dim=FakeEmbedder.dim)
    return tools


def _extraction_signal(generated_tokens: int, retried: bool = False) -> ModelSignal:
    return ModelSignal(
        True, "attribute_value_extraction", None, _VALID, {}, None,
        prompt_tokens=100, generated_tokens=generated_tokens, retried=retried,
    )


def test_classify_pair_reports_usage_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage accumulates across the feedback retry and rides the signal out."""
    from memory_arbiter.semantic_conflict import LocalGGUFSemanticBackend

    backend = LocalGGUFSemanticBackend(Path("unused.gguf"))
    llm = _UsageLLM([_TRUNCATED, _VALID])
    monkeypatch.setattr(backend, "_build_llm", lambda: llm)
    signal = backend.classify_pair({"quote": "A"}, {"quote": "B"})
    assert signal.candidate_type == "attribute_value_extraction"
    assert signal.retried is True
    assert signal.prompt_tokens == (700 + 1) + (700 + 2)
    assert signal.generated_tokens == 40 + 80


def test_classify_pair_usage_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from memory_arbiter.semantic_conflict import LocalGGUFSemanticBackend

    backend = LocalGGUFSemanticBackend(Path("unused.gguf"))
    llm = _UsageLLM([_VALID])
    monkeypatch.setattr(backend, "_build_llm", lambda: llm)
    signal = backend.classify_pair({"quote": "A"}, {"quote": "B"})
    assert signal.candidate_type == "attribute_value_extraction"
    assert signal.retried is False
    assert signal.prompt_tokens == 701
    assert signal.generated_tokens == 40


def test_ring_records_and_aggregates(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools._record_pair_sample(pair_ms=1000, forward=_extraction_signal(120), reverse=_extraction_signal(130))
    tools._record_pair_sample(
        pair_ms=3000, forward=_extraction_signal(280, retried=True), reverse=_extraction_signal(320),
    )
    tools._record_pair_sample(pair_ms=500, forward=_extraction_signal(60), reverse=_extraction_signal(60))
    summary = tools._pair_timing_summary()
    assert summary["samples"] == 3
    assert summary["mean_pair_ms"] == round((1000 + 3000 + 500) / 3)
    assert summary["p95_pair_ms"] == 3000  # sorted [500, 1000, 3000]
    assert summary["retried_ratio"] == round(1 / 3, 3)
    # generated sums per sample: 250, 600, 120 — only the 600 one is long decode
    assert summary["long_decode_ratio"] == round(1 / 3, 3)
    assert summary["mean_generated_tokens"] == round((250 + 600 + 120) / 3)


def test_ring_long_decode_threshold_boundary(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    just_under = SEMANTIC_PAIR_LONG_DECODE_TOKENS - 1
    tools._record_pair_sample(
        pair_ms=10, forward=_extraction_signal(just_under), reverse=_extraction_signal(0),
    )
    assert tools._pair_timing_summary()["long_decode_ratio"] == 0.0
    tools._record_pair_sample(
        pair_ms=10, forward=_extraction_signal(just_under), reverse=_extraction_signal(1),
    )
    assert tools._pair_timing_summary()["long_decode_ratio"] == 0.5


def test_ring_bounded_and_handles_missing_usage(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    plain = ModelSignal(True, "attribute_value_extraction", None, _VALID, {}, None)
    for index in range(SEMANTIC_PAIR_RING_SIZE + 10):
        tools._record_pair_sample(pair_ms=index, forward=plain, reverse=plain)
    assert len(tools._pair_samples) == SEMANTIC_PAIR_RING_SIZE
    summary = tools._pair_timing_summary()
    assert summary["samples"] == SEMANTIC_PAIR_RING_SIZE
    assert summary["mean_generated_tokens"] is None  # no backend usage reported


def test_empty_ring_summary(tmp_path: Path) -> None:
    assert make_tools(tmp_path)._pair_timing_summary() == {"samples": 0}


def test_status_and_doctor_expose_pair_timing(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools._record_pair_sample(
        pair_ms=1234, forward=_extraction_signal(90), reverse=_extraction_signal(60),
    )
    status = tools._semantic_status()
    assert status["last_pair_duration_ms"] == 1234
    assert status["pair_timing"]["samples"] == 1

    doctor = tools._operations.memory_doctor_overview()
    assert doctor["ok"] is True
    assert doctor["data"]["semantic_pair_timing"]["samples"] == 1
    assert doctor["data"]["semantic_pair_timing"]["mean_pair_ms"] == 1234
