"""0.15.14 A-pack gates: A5 examined-pairs cap + removed key, A6 retry queue
gate, A3 gpu-layer config plumbing, A2 L3 truncation edges."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import SEMANTIC_MAX_EXAMINED_PAIRS
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.evidence import evidence_content_hash
from memory_arbiter.models import MemoryStatus
from memory_arbiter.semantic_conflict import (
    IsolatedGGUFSemanticBackend,
    LocalGGUFSemanticBackend,
    ModelSignal,
    _l3_truncated_signal,
)
from memory_arbiter.tools import MemoryTools

_META = {"entity": "perf-gates", "scope": "production"}


class FakeEmbedder:
    embedding_space_id = "fake-perf-space"
    dim = 2
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars: int | None = None) -> EmbedResult:
        text = f"{prefix}\n{body}".casefold()
        vector = [1.0, 0.0] if "取值" in text else [0.0, 1.0]
        return EmbedResult(vector, False, len(text), len(text))


def make_tools(tmp_path: Path, *, semantic_enabled: bool = False) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "perf.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
        semantic_conflict_enabled=semantic_enabled,
        semantic_conflict_model_path=model if semantic_enabled else None,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = FakeEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(FakeEmbedder.dim) == []
    tools.db.init_vec_index_state(FakeEmbedder.embedding_space_id, True, active_dim=FakeEmbedder.dim)
    return tools


def _write_check_scene(tools: MemoryTools, peers: int) -> dict[str, Any]:
    """One new memory whose knn surface returns `peers` candidate pairs.

    Multi-value contents: the 2026-09-16 deterministic direct path bypasses
    Qwen for single-value same-skeleton pairs, which would leave these
    Qwen-budget gates with nothing to measure."""
    tools.settings.semantic_conflict_on_write = "off"
    peer_rows = [
        tools.memory_write(
            content=f"连接池上限为 {index + 10}，队列长度为 {index + 3}。",
            subject=f"v{index}", tags=[], metadata=dict(_META),
        )["data"]
        for index in range(peers)
    ]
    new = tools.memory_write(
        content="连接池上限为 99，队列长度为 99。", subject="new", tags=[], metadata=dict(_META),
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    hits = [
        {
            "memory_id": peer["id"], "id": index, "kind": "text",
            "text": f"连接池上限为 {index + 10}，队列长度为 {index + 3}。", "start_offset": 0, "end_offset": 12,
            "distance": 0.10 + index * 0.01,
            # 0.16.2 provenance gate reads hit['metadata'] like real knn rows.
            "metadata": dict(_META),
        }
        for index, peer in enumerate(peer_rows)
    ]
    tools.db.__dict__.setdefault("_stub_hits", None)
    return {"new": new, "hits": hits}


def _run_job(monkeypatch: pytest.MonkeyPatch, tools: MemoryTools, scene: dict[str, Any], backend: Any) -> dict[str, Any]:
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(scene["hits"]))
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
    record = tools.db.get_memory(int(scene["new"]["id"]))
    return tools._process_semantic_conflict_job(
        int(scene["new"]["id"]),
        {"memory_id": int(scene["new"]["id"]), "version": record["version"],
         "content_hash": evidence_content_hash(record["content"])},
    )


class _ValueBackend:
    """Extracts 取值=<trailing number> from each quote — grounded, differing."""

    calls = 0

    @staticmethod
    def _value(env: dict[str, Any]) -> str:
        return str(env["quote"]).split("为 ")[-1].rstrip("。")

    @classmethod
    def classify_pair(cls, left: dict[str, Any], right: dict[str, Any], *, deadline_monotonic: float | None = None, retry_allowed: bool = True) -> ModelSignal:
        cls.calls += 1
        parsed = {
            "attribute_a": "取值", "value_a": cls._value(left),
            "attribute_b": "取值", "value_b": cls._value(right),
        }
        return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)


def test_pairs_cap_reports_incomplete_when_no_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A5: a definitive-negative backend still counts examined pairs; beyond
    the cap the job reports incomplete/pairs_examined_capped (visible as a
    technical degradation), with exactly cap pairs sent to Qwen."""
    tools = make_tools(tmp_path)
    scene = _write_check_scene(tools, peers=SEMANTIC_MAX_EXAMINED_PAIRS + 3)
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)

    class NegativeBackend(_ValueBackend):
        @classmethod
        def classify_pair(cls, left, right, **kwargs: Any) -> ModelSignal:
            cls.calls += 1
            parsed = {
                "attribute_a": "属性甲", "value_a": cls._value(left),
                "attribute_b": "属性乙", "value_b": cls._value(right),
            }
            return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)

    result = _run_job(monkeypatch, tools, scene, NegativeBackend)
    assert result["status"] == "incomplete"
    assert result["reason"] == "pairs_examined_capped"
    assert result["notices_created"] == 0
    assert NegativeBackend.calls == SEMANTIC_MAX_EXAMINED_PAIRS * 2  # fwd+rev
    degradation = tools._check_degradation_status()
    assert degradation["last_reason"] == "pairs_examined_capped"
    # A1 wiring: one ring sample per examined pair, cap-bounded.
    assert tools._pair_timing_summary()["samples"] == SEMANTIC_MAX_EXAMINED_PAIRS


def test_pairs_cap_bounds_notices_but_keeps_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A5: with every pair notice-worthy, the first cap pairs surface notices
    (no count-based early stop anymore) and the cap rides along in
    reasons_seen on an otherwise completed result."""
    tools = make_tools(tmp_path)
    scene = _write_check_scene(tools, peers=SEMANTIC_MAX_EXAMINED_PAIRS + 3)
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)

    result = _run_job(monkeypatch, tools, scene, _ValueBackend)

    assert result["status"] == "completed"
    assert result["outcome"] == "notices_created"
    assert result["notices_created"] == SEMANTIC_MAX_EXAMINED_PAIRS
    assert result["reasons_seen"] == ["pairs_examined_capped"]
    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == int(scene["new"]["id"])]
    assert len(notices) == SEMANTIC_MAX_EXAMINED_PAIRS


def test_under_cap_memory_completes_without_pairs_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = make_tools(tmp_path)
    scene = _write_check_scene(tools, peers=3)
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)
    result = _run_job(monkeypatch, tools, scene, _ValueBackend)
    assert result["status"] == "completed"
    assert result["notices_created"] == 3
    assert "reasons_seen" not in result or "pairs_examined_capped" not in result.get("reasons_seen", [])


# ---------------------------------------------------------------------------
# A6: retry queue gate
# ---------------------------------------------------------------------------

_VALID = (
    '{"attribute_a":"数据库选择","value_a":"MySQL",'
    '"attribute_b":"数据库选择","value_b":"SQLite"}'
)
_TRUNCATED = '{"attribute_a":"x","value_a":"'


class _ScriptedLLM:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        raw = self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]
        return {"choices": [{"message": {"content": raw}}]}


def test_retry_allowed_false_runs_single_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = LocalGGUFSemanticBackend(Path("unused.gguf"))
    llm = _ScriptedLLM([_TRUNCATED, _VALID])
    monkeypatch.setattr(backend, "_build_llm", lambda: llm)
    signal = backend.classify_pair({"quote": "A"}, {"quote": "B"}, retry_allowed=False)
    assert signal.candidate_type == "invalid_json"  # treated as final, no retry
    assert len(llm.calls) == 1
    assert backend._pair_retried == 0


def test_retry_allowed_default_keeps_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = LocalGGUFSemanticBackend(Path("unused.gguf"))
    llm = _ScriptedLLM([_TRUNCATED, _VALID])
    monkeypatch.setattr(backend, "_build_llm", lambda: llm)
    signal = backend.classify_pair({"quote": "A"}, {"quote": "B"})
    assert signal.candidate_type == "attribute_value_extraction"
    assert len(llm.calls) == 2
    assert backend._pair_retried == 1


def _retry_flag_child(conn: Any, config: dict[str, Any]) -> None:
    try:
        while True:
            request = conn.recv()
            if request.get("command") == "load":
                conn.send({"ok": True, "result": {"loaded": True}})
                continue
            conn.send({
                "ok": True,
                "result": ModelSignal(
                    True, "flagged", 0.9, "{}",
                    {"retry_allowed": request.get("retry_allowed")},
                ),
            })
    except (EOFError, OSError):
        return


def test_scheduler_queue_gate_reaches_child(tmp_path: Path) -> None:
    """The caller half of the A6 gate reaches the child request payload; the
    scheduler-busy half is a pure in-process merge over the admission queues."""
    backend = IsolatedGGUFSemanticBackend(
        tmp_path / "fake.gguf", process_target=_retry_flag_child,
    )
    try:
        idle = backend.classify_pair({}, {})
        assert idle.parsed == {"retry_allowed": True}

        caller_gate = backend.classify_pair({}, {}, retry_allowed=False)
        assert caller_gate.parsed == {"retry_allowed": False}

        # Scheduler half (unit): a waiting admission token forces one-attempt.
        assert backend._effective_retry_allowed(True) is True
        with backend._schedule_cond:
            backend._schedule_queues["workspace"].append((object(), None))
        assert backend._effective_retry_allowed(True) is False
        assert backend._effective_retry_allowed(False) is False
    finally:
        backend.unload()


# ---------------------------------------------------------------------------
# A3: gpu-layer config plumbing
# ---------------------------------------------------------------------------

def _settings_from_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> Settings:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(body), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(config))
    return Settings.from_env()


def test_gpu_layers_config_default_and_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings_from_config_file(tmp_path, monkeypatch, {})
    assert settings.semantic_conflict_gpu_layers == -1  # default: full offload

    settings = _settings_from_config_file(tmp_path, monkeypatch, {
        "semantic_conflict": {"n_gpu_layers": 0},
    })
    assert settings.semantic_conflict_gpu_layers == 0  # explicit CPU

    settings = _settings_from_config_file(tmp_path, monkeypatch, {
        "semantic_conflict": {"n_gpu_layers": 12},
    })
    assert settings.semantic_conflict_gpu_layers == 12

    settings = _settings_from_config_file(tmp_path, monkeypatch, {
        "semantic_conflict": {"n_gpu_layers": 100000},
    })
    assert settings.semantic_conflict_gpu_layers == 999
    assert any("above maximum 999" in w for w in settings.config_warnings)

    settings = _settings_from_config_file(tmp_path, monkeypatch, {
        "semantic_conflict": {"n_gpu_layers": "lots"},
    })
    assert settings.semantic_conflict_gpu_layers == -1
    assert any("n_gpu_layers" in w and "invalid" in w for w in settings.config_warnings)


def test_removed_notice_pairs_key_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings_from_config_file(tmp_path, monkeypatch, {
        "semantic_conflict": {"max_notice_pairs": 3},
    })
    assert not hasattr(settings, "semantic_conflict_max_notice_pairs")
    assert any("max_notice_pairs is no longer read" in w for w in settings.config_warnings)


def test_gpu_layers_flows_to_backends(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    tools = make_tools(tmp_path, semantic_enabled=True)
    assert tools._ensure_semantic_backend() is not None
    assert isinstance(tools._semantic_backend, IsolatedGGUFSemanticBackend)
    assert tools._semantic_backend.n_gpu_layers == tools.settings.semantic_conflict_gpu_layers == -1
    assert tools._semantic_backend.status()["n_gpu_layers"] == -1
    local = LocalGGUFSemanticBackend(model, n_gpu_layers=5)
    assert local.status()["n_gpu_layers"] == 5
    assert local.n_gpu_layers == 5


# ---------------------------------------------------------------------------
# A2: L3 truncation edges
# ---------------------------------------------------------------------------

def test_l3_truncates_only_over_limit_values() -> None:
    """Second-round M1: the cut lands on the LAST clause/word boundary inside
    the cap — never mid-run."""
    over = "第一段取值说明，" + "值" * 80
    raw = json.dumps(
        {"attribute_a": "属性", "value_a": "短值", "attribute_b": "属性", "value_b": over},
        ensure_ascii=False,
    )
    signal = _l3_truncated_signal(raw)
    assert signal is not None
    assert signal.candidate_type == "attribute_value_extraction"
    assert signal.parsed is not None
    assert signal.parsed["value_b"] == "第一段取值说明"
    assert signal.parsed["value_a"] == "短值"


def test_l3_refuses_mid_run_cut_without_boundary() -> None:
    """No clause/word boundary inside the first 64 chars -> retry path, so no
    beheaded fragment is ever manufactured (second-round M1)."""
    run = "值" * 80
    raw = json.dumps(
        {"attribute_a": "属性", "value_a": "短", "attribute_b": "属性", "value_b": run},
        ensure_ascii=False,
    )
    assert _l3_truncated_signal(raw) is None


def test_l3_refuses_equal_values_after_cut() -> None:
    """Second-round M2: when cutting collapses both sides to the same value,
    L3 refuses — the pair retries instead of reading as a clean negative."""
    shared = "共同前缀" * 20  # >64 chars, boundary-free, identical on both sides
    raw = json.dumps(
        {"attribute_a": "属性", "value_a": shared, "attribute_b": "属性", "value_b": shared},
        ensure_ascii=False,
    )
    assert _l3_truncated_signal(raw) is None


def test_l3_cut_lands_on_last_boundary_inside_cap() -> None:
    from memory_arbiter.semantic_conflict import _l3_cut_value

    # clause punctuation inside the window: cut just before it
    assert _l3_cut_value("甲" * 50 + "，" + "乙" * 40) == "甲" * 50
    # clause punctuation wins over a LATER whitespace inside the window
    assert _l3_cut_value("甲" * 50 + "，" + "乙" * 10 + " " + "丙" * 20) == "甲" * 50
    # whitespace fallback when no clause char is in the window
    assert _l3_cut_value("甲" * 50 + " " + "乙" * 30) == "甲" * 50
    # no boundary at all inside the window -> retry path
    assert _l3_cut_value("甲" * 70) is None
    # a window whose only boundary is beyond the cap does not count
    assert _l3_cut_value("甲" * 70 + " " + "乙" * 10) is None
    # trailing boundary chars are stripped from the cut head
    assert _l3_cut_value("甲" * 50 + "，、：") == "甲" * 50


def test_l3_rejects_non_qualifying_shapes() -> None:
    four = {"attribute_a": "属性", "value_a": "a", "attribute_b": "属性", "value_b": "b"}
    # No over-limit field -> nothing to truncate.
    assert _l3_truncated_signal(json.dumps(four, ensure_ascii=False)) is None
    # Extra field -> retry path owns it.
    assert _l3_truncated_signal(json.dumps({**four, "confidence": 0.9}, ensure_ascii=False)) is None
    # Over-long attribute -> retry path (a malformed question is not a value).
    long_attr = json.dumps({**four, "attribute_a": "问" * 90}, ensure_ascii=False)
    assert _l3_truncated_signal(long_attr) is None
    # Newline inside the over-long value -> retry path.
    newline_val = json.dumps({**four, "value_a": "行\n" * 40}, ensure_ascii=False)
    assert _l3_truncated_signal(newline_val) is None
    # Missing json entirely.
    assert _l3_truncated_signal("no json here") is None


def test_l3_keeps_strict_protocol_rejections() -> None:
    """First-round review M1: L3 relaxes the LENGTH caps only. Array-wrapped
    raws (spec §15.4) and duplicated fields must stay on the retry path even
    when the surviving value is over the cap (plain json.loads would accept
    both, silently widening the protocol)."""
    over = "值" * 70
    inner = json.dumps(
        {"attribute_a": "属性", "value_a": "短", "attribute_b": "属性", "value_b": over},
        ensure_ascii=False,
    )
    assert _l3_truncated_signal(f"[{inner}]") is None
    duplicated = (
        '{"attribute_a":"属性","value_a":"短","attribute_b":"属性",'
        f'"value_b":"ok","value_b":"{over}"}}'
    )
    assert _l3_truncated_signal(duplicated) is None


def test_classify_pair_error_keeps_accumulated_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """First-round review L3: a raised retry attempt must not reset the pair's
    token accounting — the error signal carries what the first attempt used."""
    from memory_arbiter.semantic_conflict import LocalGGUFSemanticBackend

    backend = LocalGGUFSemanticBackend(Path("unused.gguf"))

    class _ExplodingSecondCall:
        def __init__(self) -> None:
            self.calls = 0

        def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "choices": [{"message": {"content": _TRUNCATED}}],
                    "usage": {"prompt_tokens": 700, "completion_tokens": 10},
                }
            raise RuntimeError("context window blown")

    llm = _ExplodingSecondCall()
    monkeypatch.setattr(backend, "_build_llm", lambda: llm)
    signal = backend.classify_pair({"quote": "A"}, {"quote": "B"})
    assert signal.candidate_type == "backend_error"
    assert signal.prompt_tokens == 700
    assert signal.generated_tokens == 10
    assert signal.retried is True
    assert llm.calls == 2


def test_example_config_uses_only_live_keys() -> None:
    """B3/round-1 M2: the reference example and the setup starter template
    must carry the live key set — no removed max_notice_pairs (which would
    trip the bespoke removed-key warning on every startup), and
    n_gpu_layers present."""
    example = json.loads(
        (Path(__file__).parents[1] / "examples" / "memory-arbiter.config.example.json")
        .read_text(encoding="utf-8")
    )
    sem = example["semantic_conflict"]
    assert "max_notice_pairs" not in sem
    assert sem["n_gpu_layers"] == -1
    assert "policy_path" not in example

    from memory_arbiter.setup_cli import _default_config_dict
    template = _default_config_dict(Path("/tmp/m.gguf"), Path("/tmp/db"), Path("/tmp/bk"))
    assert "max_notice_pairs" not in template["semantic_conflict"]
    assert "policy_path" not in template


# ---------------------------------------------------------------------------
# Second-round adversarial review fixes
# ---------------------------------------------------------------------------

def test_retry_is_targeted_text_without_echo_or_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-3 (measured 2026-09-11): the retry is grammar-free text feedback
    that names the specific violation; the failed output is NOT echoed (echo
    keeps the 0.5B locked in its copy state — 0/9 echo variants vs 5/5
    no-echo) and no response_format is sent on any attempt."""
    backend = LocalGGUFSemanticBackend(Path("unused.gguf"))
    llm = _ScriptedLLM([_TRUNCATED, _VALID])
    backend._llm = llm
    signal = backend.classify_pair({"quote": "A"}, {"quote": "B"})
    assert signal.candidate_type == "attribute_value_extraction"
    assert "response_format" not in llm.calls[0]
    retry = llm.calls[1]
    assert "response_format" not in retry
    assert all(m.get("role") != "assistant" for m in retry["messages"])
    assert retry["messages"][-1]["role"] == "user"

    # gated single-attempt path: same shape, one call
    llm2 = _ScriptedLLM([_TRUNCATED, _VALID])
    backend._llm = llm2
    backend.classify_pair({"quote": "A"}, {"quote": "B"}, retry_allowed=False)
    assert len(llm2.calls) == 1 and "response_format" not in llm2.calls[0]


def test_feedback_names_extra_field_keys() -> None:
    """Round-3: the 'extra field' family gets the actual offending key names
    (the 0.5B writes __unknown__ / event_time as field names)."""
    from memory_arbiter.semantic_conflict import _pair_retry_feedback

    raw = ('{"attribute_a":"a","value_a":"b","attribute_b":"c","value_b":"d",'
           '"__unknown__":"6 个开放问题","event_time":"2026-08-27"}')
    feedback = _pair_retry_feedback("schema", "invalid_schema", raw)
    assert "“__unknown__”" in feedback
    assert "“event_time”" in feedback
    # Generic shape (unparseable raw) falls back to the four-field reminder.
    generic = _pair_retry_feedback("schema", "invalid_schema", "not json")
    assert "四个字符串字段" in generic
    # over_limit branch still names the field
    assert "value_a" in _pair_retry_feedback("over_limit", "invalid_value_a", "")


def test_gpu_load_failure_falls_back_to_cpu_once(tmp_path: Path) -> None:
    """Round-2 L3: with offload now the default, a failing GPU init must fall
    back to CPU (once, remembered, visible) instead of failing every pair."""
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    backend = LocalGGUFSemanticBackend(model, n_gpu_layers=-1)
    calls: list[dict[str, Any]] = []

    class _FakeLlama:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            if kwargs.get("n_gpu_layers", 0) != 0:
                raise RuntimeError("Metal init failed")

    import sys
    import types
    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama  # type: ignore[attr-defined]
    monkeypatched = sys.modules.get("llama_cpp")
    sys.modules["llama_cpp"] = fake_module
    try:
        llm = backend._build_llm()
        assert isinstance(llm, _FakeLlama)
        assert calls[0]["n_gpu_layers"] == -1
        assert calls[1]["n_gpu_layers"] == 0
        assert backend._gpu_fallback is True
        assert backend.status()["gpu_fallback"] is True
        assert backend.pair_retry_stats()["gpu_fallback"] is True
    finally:
        if monkeypatched is not None:
            sys.modules["llama_cpp"] = monkeypatched
        else:
            del sys.modules["llama_cpp"]


def test_gpu_fallback_absent_when_cpu_already(tmp_path: Path) -> None:
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    backend = LocalGGUFSemanticBackend(model, n_gpu_layers=0)
    class _Boom:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("no model")
    import sys, types
    fake = types.ModuleType("llama_cpp")
    fake.Llama = _Boom  # type: ignore[attr-defined]
    prev = sys.modules.get("llama_cpp")
    sys.modules["llama_cpp"] = fake
    try:
        with pytest.raises(RuntimeError):
            backend._build_llm()
        assert backend._gpu_fallback is False
    finally:
        if prev is not None:
            sys.modules["llama_cpp"] = prev
        else:
            del sys.modules["llama_cpp"]


def test_notice_write_failure_is_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-2: a ready pair whose notice cannot be persisted must degrade
    visibly (notice_write_failed) instead of reporting checked_no_notice."""
    tools = make_tools(tmp_path)
    scene = _write_check_scene(tools, peers=1)
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)
    monkeypatch.setattr(
        tools.db, "record_semantic_notice",
        lambda **kwargs: {"outcome": "unavailable"},
    )
    result = _run_job(monkeypatch, tools, scene, _ValueBackend)
    assert result["status"] == "incomplete"
    assert result["reason"] == "notice_write_failed"
    assert result["notices_created"] == 0
    assert tools._check_degradation_status()["last_reason"] == "notice_write_failed"


def test_notice_dedupe_still_not_a_degradation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = make_tools(tmp_path)
    scene = _write_check_scene(tools, peers=1)
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)
    monkeypatch.setattr(
        tools.db, "record_semantic_notice",
        lambda **kwargs: {"outcome": "deduped"},
    )
    result = _run_job(monkeypatch, tools, scene, _ValueBackend)
    assert result["outcome"] == "checked_no_notice"
    assert "reasons_seen" not in result


def test_b2_sql_prefilter_is_a_safe_superset(tmp_path: Path) -> None:
    """Round-2 M3: the SQL prefilter must never drop a row the shared
    predicate would have kept (numeric tags, sub-second bounds)."""
    from memory_arbiter.db.memories import MemoriesStore, row_passes_filters
    from datetime import datetime, timezone

    tools = make_tools(tmp_path)
    rows = [
        # (tags json, ingest_time, source_type)
        ('[1.0]', "2026-09-11T10:00:00.700000+00:00", "agent_generated"),
        ("[1]", "2026-09-11T10:00:00+00:00", "agent_generated"),
        ('["1.0"]', "2026-09-11T10:00:01+00:00", "user_confirmed"),
        ('"abc"', "2026-09-11T10:00:02+00:00", "agent_generated"),
        ("{bad", "2026-09-11T10:00:03+00:00", "agent_generated"),
    ]
    for i, (tags, ts, st) in enumerate(rows):
        with tools.db.write_transaction() as conn:
            conn.execute(
                "INSERT INTO memories (subject, content, agent_id, tags, event_time, ingest_time, "
                "source_type, status, workspace, workspace_canonical, version, protection_level, created_at) "
                "VALUES (?,?,?,?,?,?,?,'active','ws','ws',1,'normal',?)",
                (f"s{i}", "c", "tester", tags, ts, ts, st, ts),
            )
    after = datetime(2026, 9, 11, 10, 0, 0, 300000, tzinfo=timezone.utc)
    cases = [
        {"tags_filter": ["1.0"], "after_dt": None, "before_dt": None, "source_type": None},
        {"tags_filter": None, "after_dt": after, "before_dt": None, "source_type": None},
        {"tags_filter": ["agent_generated"], "after_dt": None, "before_dt": None, "source_type": "agent_generated"},
    ]
    for case in cases:
        with tools.db.connection() as conn:
            all_rows = conn.execute("SELECT tags, ingest_time, source_type FROM memories").fetchall()
        expected = sum(
            1 for r in all_rows
            if row_passes_filters(r["tags"], r["ingest_time"], r["source_type"], **case)
        )
        count = tools.db.count_filtered_memories("status='active'", **case)
        page = tools.db.recall_by_filters("status='active'", **case, limit=50, offset=0)
        assert count == expected, case
        assert len(page) == expected, case


def test_feedback_names_the_real_cause_for_long_fields() -> None:
    """Round-3 fix: the over_limit feedback reads the RAW to tell 'too long'
    from 'empty' — pre-0.15.14 wording always said 'empty' (grammar-era
    maxLength kept over-long values from reaching a retry), which made the
    retry invent values out of thin air (live: {"value_a":"无"})."""
    from memory_arbiter.semantic_conflict import _pair_retry_feedback

    long_raw = json.dumps({
        "attribute_a": "代码评审", "value_a": "亮点" * 60,
        "attribute_b": "代码评审", "value_b": "问题",
    }, ensure_ascii=False)
    feedback = _pair_retry_feedback("over_limit", "invalid_value_a", long_raw)
    assert "超过 64 字（" in feedback
    assert "为空" not in feedback

    empty_raw = json.dumps({
        "attribute_a": "代码评审", "value_a": "  ",
        "attribute_b": "代码评审", "value_b": "问题",
    }, ensure_ascii=False)
    feedback = _pair_retry_feedback("over_limit", "invalid_value_a", empty_raw)
    assert "为空" in feedback
    assert "超过 64 字（" not in feedback  # the too-long diagnosis must not appear

    attr_raw = json.dumps({
        "attribute_a": "问" * 90, "value_a": "短",
        "attribute_b": "属性", "value_b": "短",
    }, ensure_ascii=False)
    feedback = _pair_retry_feedback("over_limit", "invalid_attribute_a", attr_raw)
    assert "超过 80 字" in feedback

    # Unparseable raw: the shape fallback, not a fabricated cause.
    feedback = _pair_retry_feedback("over_limit", "invalid_value_a", "not json")
    assert "不合协议" in feedback
