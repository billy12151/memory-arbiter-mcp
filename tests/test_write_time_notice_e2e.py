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
  6. live-degradation replays: the two 2026-09-08/09 qwen_invalid_output
     samples (mema id=925 / id=926 evidence) must extract cleanly under the
     current prompt+retry protocol

Real-model variants live at the bottom under @pytest.mark.slow. Since 0.15.9.1 they
run by default (the 0.15.8 release process relied on a manual "pytest -m slow" step
that was missed): machines without the GGUF model pytest.skip cleanly, so the local
full suite always exercises them and CI stays green via skips, never misses them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.evidence import evidence_content_hash
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
        # Gate-v2 G4: cos≈0.68 between the two directions — inside the
        # candidate cosine band, so csv/json template pairs survive the gate
        # (orthogonal fakes would be filtered as below-floor noise).
        vector = [0.9308, 0.3653] if "json" in text else [0.3653, 0.9308]
        return EmbedResult(vector, False, len(text), len(text))


    @classmethod
    def embed_texts(cls, texts, prefix: str = ""):
        return [cls.embed_text(prefix="", body=t) for t in texts]

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
    assert tools.wait_semantic_worker_drained(timeout=5)
    return int(peer["id"]), int(new["id"])


def _stub_knn_peer(monkeypatch: pytest.MonkeyPatch, tools: MemoryTools, peer_id: int, text: str) -> None:
    # 0.16.2 write-time provenance gate reads hit['metadata'] exactly like
    # the real knn row does — the hand-built hit borrows the peer's own.
    record = tools.db.get_memory(peer_id)
    meta = (record or {}).get("metadata")
    if isinstance(meta, (dict, list)):
        meta = json.dumps(meta, ensure_ascii=False)
    hits = [{
        "memory_id": peer_id, "id": 1, "kind": "text", "text": text,
        "start_offset": 0, "end_offset": len(text), "distance": 0.1,
        "metadata": meta,
    }]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: list(hits))
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: list(hits))  # 0.17.0 P2-3 行级候选同注入


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
    # A1: an examined pair leaves a ring sample on the tools instance.
    assert tools._pair_timing_summary()["samples"] == 1


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
    # A1: an examined pair leaves a ring sample on the tools instance.
    assert tools._pair_timing_summary()["samples"] == 1


def test_true_duplicates_stay_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 3: equal numeric sets AND equal value-stripped skeleton -> duplicate.

    Two flavours: identical numbers with identical prose (port 6789) and the
    same sentence modulo whitespace/punctuation. Neither may reach Qwen.
    """
    tools = make_tools(tmp_path)
    peer = tools.memory_write(content="clash 的代理端口是 6789。", subject="p1", tags=[], metadata=dict(_META))["data"]
    new = tools.memory_write(content="clash 的代理端口是 6789", subject="n1", tags=[], metadata=dict(_META))["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _FormatBackend.calls = 0
    result = _run_check(monkeypatch, tools, int(peer["id"]), int(new["id"]), "clash 的代理端口是 6789。")
    assert result["outcome"] == "checked_no_notice", result
    assert _FormatBackend.calls == 0  # the duplicate never reaches Qwen

    # Same numeric value restated with identical prose shape (same guard path).
    peer2 = tools.memory_write(content="扫描超时配置为 5000ms，已冻结。", subject="p2", tags=[], metadata=dict(_META))["data"]
    new2 = tools.memory_write(content="扫描超时配置为 5000ms 已冻结", subject="n2", tags=[], metadata=dict(_META))["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
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
    assert tools.wait_semantic_worker_drained(timeout=5)
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
        def judge_pairs(cls, pairs):
            time.sleep(0.4)  # overflow the 50ms sync window
            from memory_arbiter.semantic_judge import PairVerdict
            return [PairVerdict("conflict",
                                {"conflict": 0.9, "no_conflict": 0.05, "possible_conflict": 0.05},
                                None, "test") for _ in pairs]

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



# ── 0.15.12 C1: the two gathering-truncation causes report distinctly ─────────


def _write_many_units(tools: MemoryTools, paragraphs: int) -> int:
    """Write one memory whose content segments into many text units."""
    content = "\n\n".join(
        f"部署记录 {index}：网关超时阈值 {index}00ms。" for index in range(paragraphs)
    )
    written = tools.memory_write(content=content, subject="deployment log", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    return int(written["id"])


def test_over_cap_memory_reports_rows_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """>SEMANTIC_MAX_ROWS row candidates (P2-3.1) -> incomplete/rows_capped."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)
    memory_id = _write_many_units(tools, 8)

    # 0.17.0 P2-3.1：行级模式帽=SEMANTIC_MAX_ROWS、原因=rows_capped
    monkeypatch.setattr("memory_arbiter.pipeline.evidence.SEMANTIC_MAX_ROWS", 3)

    result = tools._process_semantic_conflict_job(memory_id, _job_snapshot(tools, memory_id))

    assert result["status"] == "incomplete"
    assert result["reason"] == "rows_capped"
    assert result["notices_created"] == 0
    assert result["reasons_seen"] == ["rows_capped"]
    status = tools._check_degradation_status()
    assert status["last_reason"] == "rows_capped"
    assert "rows_capped" in status["note"]


def test_job_deadline_keeps_notice_budget_exhausted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fair-deadline truncation (units under the cap) keeps the original reason string."""
    import time as _time

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    monkeypatch.setattr(
        tools._semantic_worker, "pending_job_deadline", lambda timeout: _time.monotonic() - 1.0,
    )
    # Gate-v2 G4: the row must pass the sentence prefilter to enter the
    # examination loop — only then can the (already-past) deadline truncate
    # it and surface notice_budget_exhausted.
    written = tools.memory_write(content="两条记录而已：超时阈值 500ms。", subject="small", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    memory_id = int(written["id"])

    result = tools._process_semantic_conflict_job(memory_id, _job_snapshot(tools, memory_id))

    assert result["status"] == "incomplete"
    assert result["reason"] == "notice_budget_exhausted"
    assert result["notices_created"] == 0


def test_units_cap_attributed_first_when_both_causes_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both the unit cap and a past deadline -> the more specific cap wins.

    The deadline is pushed past only after the 24th unit has been examined, so
    the 25th loop iteration sees both causes; the cap check runs first.
    """
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    memory_id = _write_many_units(tools, 8)

    # 0.17.0 P2-3.1：rows 模式帽语义（_ROWS 行、第 _ROWS 次 KNN 后推死线）
    from memory_arbiter.constants import SEMANTIC_MAX_ROWS as _ROWS
    clock = {"now": 100.0}
    fairness_deadline = 100.5
    knn_calls = {"n": 0}

    def fake_knn(*a: Any, **k: Any) -> list[dict[str, Any]]:
        # Gate-v2 G5: the subject coarse screen (subject_rows_only) is a
        # separate KNN before the loop — count only sentence KNNs. The
        # screen returns a placeholder neighbour so the clean list is
        # non-empty and the sentence loop actually runs.
        if k.get("subject_rows_only"):
            return [{"memory_id": 987654, "subject": "neighbour", "tags": []}]
        knn_calls["n"] += 1
        if knn_calls["n"] >= 3:
            clock["now"] = fairness_deadline + 1.0
        return []

    monkeypatch.setattr(tools.db, "row_knn", fake_knn)
    monkeypatch.setattr("memory_arbiter.pipeline.evidence.SEMANTIC_MAX_ROWS", 3)
    monkeypatch.setattr("memory_arbiter.pipeline.evidence.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        tools._semantic_worker, "pending_job_deadline", lambda timeout: fairness_deadline,
    )

    result = tools._process_semantic_conflict_job(memory_id, _job_snapshot(tools, memory_id))

    assert knn_calls["n"] == 3
    assert result["reason"] == "rows_capped"


def test_technical_reasons_registry_rows_cap() -> None:
    from memory_arbiter.pipeline.evidence import _TECHNICAL_REASONS

    # 0.17.0 review R2 r2s-12: the units mode is retired (C5) — the cap
    # reason rides the rows-mode registry only.
    assert "evidence_units_capped" not in _TECHNICAL_REASONS
    assert "rows_capped" in _TECHNICAL_REASONS
    assert "notice_budget_exhausted" in _TECHNICAL_REASONS


# ── slow: real Qwen2.5-0.5B model (pytest -m slow; excluded by default) ──────

_SLOW_MODEL = Path(
    "~/.local/share/memory-arbiter/models/semantic-conflict/"
    "Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf"
).expanduser()


@pytest.fixture(scope="module")
def real_backend() -> "Any":
    """Load the GGUF model ONCE per module and share it across the slow tests.

    The classifier is stateless (production shares one resident instance
    across all traffic); per-test cold loads were a 0.15.8-era convenience
    that cost ~4x model loads per suite run. Skips cleanly on machines
    without the model file, so CI never fails on it.

    Teardown contract (owner rule, 2026-09-11): real-model tests must
    release explicitly — unload, drop the reference, gc. Finalizing at
    interpreter exit crashes in ggml_metal_device_free: pytest reports all
    green while the process exits 134, poisoning the release gate's and
    CI's exit-code checks.
    """
    if not _SLOW_MODEL.exists():
        pytest.skip(f"real model not installed at {_SLOW_MODEL}")
    import gc

    from memory_arbiter.constants import SEMANTIC_N_CTX

    backend = LocalGGUFSemanticBackend(_SLOW_MODEL, n_ctx=SEMANTIC_N_CTX, n_threads=4, n_batch=128)
    backend.load()
    yield backend
    try:
        backend.unload()
    except Exception:
        pass
    del backend
    gc.collect()


def _real_gate(backend: Any, left: str, right: str):

    def env(text: str) -> dict[str, Any]:
        return {"quote": text[:400], "subject": "slow", "tags": ["slow"],
                "workspace_canonical": "memory-arbiter-mcp", "memory_id": 1, "version": 1,
                "event_time": None, "metadata": {"entity": "slow", "scope": "slow"}}

    forward = backend.classify_pair(env(left), env(right), deadline_monotonic=None)
    reverse = backend.classify_pair(env(right), env(left), deadline_monotonic=None)
    gate = evaluate_single_direction_extraction(signal_extraction(forward), env(left), env(right)
    )
    return gate, forward, reverse



def _assert_no_invalid_output(forward: "Any", reverse: "Any") -> None:
    """The regression these replays guard: qwen_invalid_output (invalid_json /
    invalid_schema) in either direction. unknown_field stays acceptable — it is
    a protocol-legal negative, not a technical failure."""
    for signal in (forward, reverse):
        assert signal.candidate_type not in {"invalid_json", "invalid_schema"}, (
            f"{signal.candidate_type}: {signal.error} raw={signal.raw[:200]}"
        )


def _faithful_env(
    quote: str, *, subject: str, tags: list[str], workspace: str,
    memory_id: int, version: int, event_time: "str | None",
) -> dict[str, Any]:
    """Envelope matching the live pair call: real subjects/tags/workspace from
    the degrading memories. Teeth depend on it — with a minimal placeholder
    envelope even the pre-fix code extracts these pairs cleanly (verified
    2026-09-09 by replaying pair-v5 + no-retry against both fixtures)."""
    return {"quote": quote[:400], "subject": subject, "tags": tags,
            "workspace_canonical": workspace, "memory_id": memory_id, "version": version,
            "event_time": event_time, "metadata": {}}

