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
    """One new memory whose knn surface returns `peers` candidate pairs."""
    tools.settings.semantic_conflict_on_write = "off"
    peer_rows = [
        tools.memory_write(
            content=f"取值为 {index + 10}。", subject=f"v{index}", tags=[], metadata=dict(_META),
        )["data"]
        for index in range(peers)
    ]
    new = tools.memory_write(
        content="取值为 99。", subject="new", tags=[], metadata=dict(_META),
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    hits = [
        {
            "memory_id": peer["id"], "id": index, "kind": "text",
            "text": f"取值为 {index + 10}。", "start_offset": 0, "end_offset": 9,
            "distance": 0.10 + index * 0.01,
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
    over = "值" * 80
    raw = json.dumps(
        {"attribute_a": "属性", "value_a": "短值", "attribute_b": "属性", "value_b": over},
        ensure_ascii=False,
    )
    signal = _l3_truncated_signal(raw)
    assert signal is not None
    assert signal.candidate_type == "attribute_value_extraction"
    assert signal.parsed is not None
    assert len(signal.parsed["value_b"]) == 64
    assert signal.parsed["value_a"] == "短值"


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
