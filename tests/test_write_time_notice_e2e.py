"""Write-time semantic-notice end-to-end acceptance tests (0.15.8).

Release gate: every change must pass these before shipping. The scenarios
replay the 2026-09-08 live-host verification that exposed the three
silent-drop root causes (see mema memory id=919):
  1. digit-free contradiction -> notice (notice #11 replay)
  2. shared-date contradiction -> notice (duplicate_guard fix acceptance;
     pre-0.15.8 this pair died at the deterministic gate, ignore/equivalent_value)
  3. true duplicates stay guarded (no Qwen call, no notice)
  4. sync-window three states (completed / async / wait=0 never blocks)
  5. pair prompt input caps quotes at 400 chars

Real-model variants live at the bottom under @pytest.mark.slow. Since 0.15.9.1 they
run by default (the 0.15.8 release process relied on a manual "pytest -m slow" step
that was missed): machines without the GGUF model pytest.skip cleanly, so the local
full suite always exercises them and CI stays green via skips, never misses them.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.evidence import evidence_content_hash
from memory_arbiter.semantic_conflict import ModelSignal
from memory_arbiter.tools import MemoryTools

_META = {"entity": "e2e-notice", "scope": "export-format"}


class FakeEmbedder:
    """Deterministic 2-dim stand-in (same pattern as test_vnext_evidence)."""

    embedding_space_id = "fake-e2e-space"
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
        db_path=tmp_path / "e2e.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
        embedding_auto_write=True,
        embedding_auto_query=True,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = FakeEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(FakeEmbedder.dim) == []
    tools.db.init_vec_index_state(FakeEmbedder.embedding_space_id, True, active_dim=FakeEmbedder.dim)
    return tools


def _job_snapshot(tools: MemoryTools, memory_id: int) -> dict[str, Any]:
    record = tools.db.get_memory(memory_id)
    return {
        "memory_id": int(memory_id),
        "version": record["version"],
        "content_hash": evidence_content_hash(record["content"]),
    }


class _FormatBackend:
    """Deterministic backend: extracts export_format=json|csv from the quote."""

    calls = 0

    @staticmethod
    def _value(env: dict[str, Any]) -> str:
        text = str(env["quote"]).casefold()
        if "json" in text:
            return "json"
        if "csv" in text:
            return "csv"
        return "__unknown__"

    @classmethod
    def classify_pair(cls, left: dict[str, Any], right: dict[str, Any], *, deadline_monotonic: float | None = None) -> ModelSignal:
        cls.calls += 1
        parsed = {
            "attribute_a": "export_format", "value_a": cls._value(left),
            "attribute_b": "export_format", "value_b": cls._value(right),
        }
        return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)


def _write_pair(tools: MemoryTools, left: str, right: str) -> tuple[int, int]:
    peer = tools.memory_write(content=left, subject="peer", tags=[], metadata=dict(_META))["data"]
    new = tools.memory_write(content=right, subject="new", tags=[], metadata=dict(_META))["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    return int(peer["id"]), int(new["id"])


def _stub_knn_peer(monkeypatch: pytest.MonkeyPatch, tools: MemoryTools, peer_id: int, text: str) -> None:
    hits = [{
        "memory_id": peer_id, "id": 1, "kind": "text", "text": text,
        "start_offset": 0, "end_offset": len(text), "distance": 0.1,
    }]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))


def _run_check(monkeypatch: pytest.MonkeyPatch, tools: MemoryTools, peer_id: int, new_id: int, peer_text: str, backend: Any = None) -> dict[str, Any]:
    _stub_knn_peer(monkeypatch, tools, peer_id, peer_text)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend or _FormatBackend())
    return tools._process_semantic_conflict_job(new_id, _job_snapshot(tools, new_id))


def test_digit_free_contradiction_produces_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 1 (notice #11 replay): json vs csv, no shared digits -> notice."""
    tools = make_tools(tmp_path)
    peer_id, new_id = _write_pair(
        tools, "conflict-bench 的 export-format 取值为 csv。", "conflict-bench 的 export-format 取值为 json。",
    )
    result = _run_check(
        monkeypatch, tools, peer_id, new_id, "conflict-bench 的 export-format 取值为 csv。",
    )
    assert result["status"] == "completed", result
    assert result["outcome"] == "notices_created", result
    assert result["notices_created"] == 1


def test_shared_date_contradiction_produces_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 2 (R1 acceptance): a shared date must not read as duplicate.

    Pre-0.15.8 the equal numeric sets ({2026,09,08}) + cosine rider killed this
    pair at decide_evidence (ignore/equivalent_value) before Qwen ran.
    """
    tools = make_tools(tmp_path)
    peer_id, new_id = _write_pair(
        tools,
        "2026-09-08 复核：bench-export 的取值为 csv，当日已确认。",
        "2026-09-08 复核：bench-export 的取值为 json，当日已确认。",
    )
    from memory_arbiter.semantic_conflict import decide_evidence
    gate = decide_evidence(
        "2026-09-08 复核：bench-export 的取值为 csv，当日已确认。",
        "2026-09-08 复核：bench-export 的取值为 json，当日已确认。",
    )
    assert gate.action == "check", gate  # no longer ignore/equivalent_value
    result = _run_check(
        monkeypatch, tools, peer_id, new_id, "2026-09-08 复核：bench-export 的取值为 csv，当日已确认。",
    )
    assert result["outcome"] == "notices_created", result
    assert result["notices_created"] == 1


def test_true_duplicates_stay_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 3: equal numeric sets AND equal value-stripped skeleton -> duplicate.

    Two flavours: identical numbers with identical prose (port 6789) and the
    same sentence modulo whitespace/punctuation. Neither may reach Qwen.
    """
    tools = make_tools(tmp_path)
    peer = tools.memory_write(content="clash 的代理端口是 6789。", subject="p1", tags=[], metadata=dict(_META))["data"]
    new = tools.memory_write(content="clash 的代理端口是 6789", subject="n1", tags=[], metadata=dict(_META))["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    _FormatBackend.calls = 0
    result = _run_check(monkeypatch, tools, int(peer["id"]), int(new["id"]), "clash 的代理端口是 6789。")
    assert result["outcome"] == "checked_no_notice", result
    assert _FormatBackend.calls == 0  # the duplicate never reaches Qwen

    # Same numeric value restated with identical prose shape (same guard path).
    peer2 = tools.memory_write(content="扫描超时配置为 5000ms，已冻结。", subject="p2", tags=[], metadata=dict(_META))["data"]
    new2 = tools.memory_write(content="扫描超时配置为 5000ms 已冻结", subject="n2", tags=[], metadata=dict(_META))["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    _FormatBackend.calls = 0
    result2 = _run_check(monkeypatch, tools, int(peer2["id"]), int(new2["id"]), "扫描超时配置为 5000ms，已冻结。")
    assert result2["outcome"] == "checked_no_notice", result2
    assert _FormatBackend.calls == 0


def _isolated_write(tools: MemoryTools, content: str, subject: str) -> dict[str, Any]:
    """Write with the real check route off so memory_write's own post-commit
    wait cannot run a backend-less job that pollutes the task result; scenario
    tests flip the route back on around their explicit post-commit call."""
    tools.settings.semantic_conflict_on_write = "off"
    data = tools.memory_write(content=content, subject=subject, tags=[], metadata=dict(_META))["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    tools.settings.semantic_conflict_on_write = "async"
    return data


def test_sync_window_completed_rides_along(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 4a: fast backend inside the wait -> completed in the write path."""
    tools = make_tools(tmp_path)
    record = _isolated_write(tools, "syncbench 的取值为 json。", "s1")
    peer = _isolated_write(tools, "syncbench 的取值为 csv。", "s2")
    tools.settings.semantic_conflict_notice_sync_wait_ms = 3000
    _stub_knn_peer(monkeypatch, tools, int(record["id"]), "syncbench 的取值为 json。")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _FormatBackend())
    _index, check = tools._enqueue_content_postcommit(int(peer["id"]))
    assert check["status"] == "completed", check
    assert check["outcome"] == "notices_created", check


def test_sync_window_timeout_returns_async_and_survives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 4b: slow backend past a short wait -> async, result not lost."""
    tools = make_tools(tmp_path)
    record = _isolated_write(tools, "slowbench 的取值为 json。", "s3a")
    peer = _isolated_write(tools, "slowbench 的取值为 csv。", "s3b")
    tools.settings.semantic_conflict_notice_sync_wait_ms = 50

    class _SlowBackend(_FormatBackend):
        @classmethod
        def classify_pair(cls, left, right, *, deadline_monotonic=None):
            time.sleep(0.4)
            return super().classify_pair(left, right, deadline_monotonic=deadline_monotonic)

    _stub_knn_peer(monkeypatch, tools, int(record["id"]), "slowbench 的取值为 json。")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _SlowBackend())
    started = time.monotonic()
    _index, check = tools._enqueue_content_postcommit(int(peer["id"]))
    elapsed = time.monotonic() - started
    assert check["status"] == "async", check
    assert elapsed < 1.0  # did not block for the backend's 2x0.4s inside a 50ms window + margin
    # The job is not lost: wait beyond the backend's sleep and it completes.
    completed = tools._semantic_worker.wait_task(str(check["task_id"]), 5.0)
    assert completed is not None and completed.get("status") == "completed", completed


def test_sync_window_zero_never_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 4c: notice_sync_wait_ms=0 -> immediate async (batch ingestion)."""
    tools = make_tools(tmp_path)
    peer = _isolated_write(tools, "zerobench 的取值为 csv。", "s4")
    tools.settings.semantic_conflict_notice_sync_wait_ms = 0

    class _NeverBackend(_FormatBackend):
        @classmethod
        def classify_pair(cls, left, right, *, deadline_monotonic=None):
            time.sleep(5)
            return super().classify_pair(left, right, deadline_monotonic=deadline_monotonic)

    _stub_knn_peer(monkeypatch, tools, int(peer["id"]), "zerobench 的取值为 csv。")
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _NeverBackend())
    started = time.monotonic()
    _index, check = tools._enqueue_content_postcommit(int(peer["id"]))
    elapsed = time.monotonic() - started
    assert check["status"] == "async", check
    assert elapsed < 0.5, elapsed  # returned immediately, never waited on the backend


def test_pair_prompt_caps_quotes_at_400_chars() -> None:
    """Scenario 5: model input quotes are capped at the segmenter's unit cap."""
    from memory_arbiter.semantic_conflict import LocalGGUFSemanticBackend

    long_quote = "甲" * 600
    text = LocalGGUFSemanticBackend._pair_text({"quote": long_quote}, {"quote": long_quote})
    assert "甲" * 400 in text
    assert "甲" * 401 not in text


# ── slow: real Qwen2.5-0.5B model (pytest -m slow; excluded by default) ──────

_SLOW_MODEL = Path(
    "~/.local/share/memory-arbiter/models/semantic-conflict/"
    "Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf"
).expanduser()


def _real_backend() -> "Any":
    if not _SLOW_MODEL.exists():
        pytest.skip(f"real model not installed at {_SLOW_MODEL}")
    from memory_arbiter.constants import SEMANTIC_N_CTX
    from memory_arbiter.semantic_conflict import LocalGGUFSemanticBackend

    backend = LocalGGUFSemanticBackend(_SLOW_MODEL, n_ctx=SEMANTIC_N_CTX, n_threads=4, n_batch=128)
    backend.load()
    return backend


def _real_gate(backend: Any, left: str, right: str):
    from memory_arbiter.semantic_conflict import evaluate_pair_extractions, signal_extraction

    def env(text: str) -> dict[str, Any]:
        return {"quote": text[:400], "subject": "slow", "tags": ["slow"],
                "workspace_canonical": "memory-arbiter-mcp", "memory_id": 1, "version": 1,
                "event_time": None, "metadata": {"entity": "slow", "scope": "slow"}}

    forward = backend.classify_pair(env(left), env(right), deadline_monotonic=None)
    reverse = backend.classify_pair(env(right), env(left), deadline_monotonic=None)
    gate = evaluate_pair_extractions(
        signal_extraction(forward), signal_extraction(reverse), env(left), env(right),
        require_bidirectional=True,
    )
    return gate, forward, reverse


@pytest.mark.slow
def test_slow_date_pair_end_to_end_notice_ready() -> None:
    """Dates differing only in value must extract and reach notice_ready."""
    backend = _real_backend()
    gate, _f, _r = _real_gate(
        backend,
        "新功能 scheduled-launch 的上线日期为 2026-09-01。",
        "新功能 scheduled-launch 的上线日期为 2026-09-10。",
    )
    assert gate.state == "notice_ready", gate
    assert gate.attribute and gate.value_a != gate.value_b


@pytest.mark.slow
def test_slow_long_quote_pair_outputs_complete_json() -> None:
    """The 2026-09-08 live-host invalid_output sample: long prose pair must
    produce a complete, extractable JSON (no truncation) under v5+2048+400."""
    backend = _real_backend()
    left = (
        "经复核确认：memory-arbiter 的写入路径已不含任何 Qwen 语义冲突检测，写时语义冲突检测已整体移除。"
        "当前冲突检测仅在定时 LLM scan 中执行，写入时不同步执行语义检查。"
    )
    right = (
        "核心链路为 0.5B 粗召回 + pair 文本证据 gate；pair text medium gate 与 pair text strong gate "
        "做成可选配置，默认使用 medium，整个检测在写入提交后异步执行，输出为 advisory notice。"
    )
    gate, forward, reverse = _real_gate(backend, left, right)
    # Both directions must be complete extractions (invalid_json/invalid_schema
    # is the regression this test guards against); the gate verdict itself may
    # legitimately be a strict negative for hard pairs.
    assert forward.candidate_type == "attribute_value_extraction", forward.candidate_type
    assert reverse.candidate_type == "attribute_value_extraction", reverse.candidate_type
    assert forward.parsed and reverse.parsed


@pytest.mark.slow
def test_slow_long_values_keep_difference() -> None:
    """45–58 char historical value shapes must keep their distinction under
    the v5 fragment-selection wording (compression wording flattened them)."""
    backend = _real_backend()
    gate, _f, _r = _real_gate(
        backend,
        "Tier1 功能方案状态：Tier1 已确认功能方案（810 v2 重排版：merge / 定时任务引导 / find 增强，0.16 规划）。",
        "Tier1 功能方案状态：Tier1 仍为高 ROI 候选分析（809：三项待排期，merge 可复用冲突生命周期）。",
    )
    assert gate.value_a != gate.value_b, gate


@pytest.mark.slow
def test_slow_short_value_regression() -> None:
    """json vs csv short values: the stable baseline."""
    backend = _real_backend()
    gate, _f, _r = _real_gate(
        backend,
        "conflict-bench 的 export-format 取值为 json。",
        "conflict-bench 的 export-format 取值为 csv。",
    )
    assert gate.state == "notice_ready", gate
