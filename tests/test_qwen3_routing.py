"""Qwen3 decode-family routing (0.16.8): metadata probe, parameter split,
and the behavioural self-heal — all against a fake Llama, no GGUF load.

Owner compatibility constraint (2026-09-17): the Qwen2.5 path keeps its
legacy decode params byte-for-byte (stop=["\n\n"], no user-turn prefix);
only qwen3-family models drop the stop (their <think>\n\n shell dies at the
second token under it) and carry /no_think.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from memory_arbiter.semantic_conflict import LocalGGUFSemanticBackend

MODEL = Path("/nonexistent/qwen3.gguf")
ENV_L = {"quote": "生产数据库用 MySQL。", "subject": "t", "metadata": {"entity": "e", "scope": "s"}}
ENV_R = {"quote": "生产数据库用 PostgreSQL。", "subject": "t", "metadata": {"entity": "e", "scope": "s"}}
GOOD_JSON = (
    '{"attribute_a":"数据库选型","value_a":"MySQL",'
    '"attribute_b":"数据库选型","value_b":"PostgreSQL"}'
)


class _FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.metadata: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []
        self.outputs: list[str] = [GOOD_JSON]

    def create_chat_completion(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        content = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        return {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20}}


def _backend(fake: _FakeLlama) -> LocalGGUFSemanticBackend:
    backend = LocalGGUFSemanticBackend(MODEL)
    backend._llm = fake  # bypass load/unload state machinery
    backend._loaded_at = 1.0
    return backend


def test_architecture_probe_routes_qwen3(monkeypatch) -> None:
    # _build_llm constructs the Llama itself, so each probe round injects a
    # subclass carrying the metadata variant under test.
    for arch, want in (("qwen3", True), ("qwen2", False), ("", False)):
        architecture = arch

        class _Probe(_FakeLlama):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.metadata = {"general.architecture": architecture} if architecture else {}

        monkeypatch.setattr("llama_cpp.Llama", _Probe)
        backend = LocalGGUFSemanticBackend(Path(__file__))
        backend._build_llm()
        assert backend._qwen3_style is want, architecture or "<missing>"


def test_qwen3_decode_params_drop_stop_and_add_nothink() -> None:
    fake = _FakeLlama()
    backend = _backend(fake)
    backend._qwen3_style = True
    signal = backend.classify_pair(ENV_L, ENV_R, retry_allowed=False)
    assert signal.parsed is not None
    call = fake.calls[0]
    assert call["kwargs"]["stop"] is None
    assert call["messages"][1]["content"].startswith("/no_think\n")


def test_legacy_decode_params_unchanged() -> None:
    fake = _FakeLlama()
    backend = _backend(fake)
    backend._qwen3_style = False
    signal = backend.classify_pair(ENV_L, ENV_R, retry_allowed=False)
    assert signal.parsed is not None
    call = fake.calls[0]
    assert call["kwargs"]["stop"] == ["\n\n"]
    assert not call["messages"][1]["content"].startswith("/no_think")


def test_behavioural_self_heal_after_two_think_strikes() -> None:
    fake = _FakeLlama()
    fake.outputs = ["<think>", "<think>", GOOD_JSON, GOOD_JSON]
    backend = _backend(fake)
    backend._qwen3_style = False
    first = backend.classify_pair(ENV_L, ENV_R, retry_allowed=False)
    assert first.parsed is None  # strike one: legacy params truncated
    assert backend._qwen3_style is False
    second = backend.classify_pair(ENV_L, ENV_R, retry_allowed=False)
    assert second.parsed is not None  # strike two: self-healed and rerun
    assert backend._qwen3_style is True
    assert backend._family_autodetected == 1
    # The rerun (third call) used the corrected params.
    assert fake.calls[2]["kwargs"].get("stop") is None
    assert fake.calls[2]["messages"][1]["content"].startswith("/no_think\n")
    # A single think echo does NOT flip the family (anti false-trigger).
    fake2 = _FakeLlama()
    fake2.outputs = ["<think>", GOOD_JSON]
    backend2 = _backend(fake2)
    backend2._qwen3_style = False
    backend2.classify_pair(ENV_L, ENV_R, retry_allowed=False)
    assert backend2._qwen3_style is False


def test_stats_expose_family() -> None:
    fake = _FakeLlama()
    backend = _backend(fake)
    backend._qwen3_style = True
    stats = backend.pair_retry_stats()
    assert stats["model_family"] == "qwen3"
    assert stats["family_autodetected"] == 0
