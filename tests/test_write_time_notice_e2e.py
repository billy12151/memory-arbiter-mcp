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


# ── 0.15.12 C1: the two gathering-truncation causes report distinctly ─────────


def _write_many_units(tools: MemoryTools, paragraphs: int) -> int:
    """Write one memory whose content segments into many text units."""
    content = "\n\n".join(
        f"第{index}条部署记录涉及网关配置与索引参数。" for index in range(paragraphs)
    )
    written = tools.memory_write(content=content, subject="deployment log", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    return int(written["id"])


def test_over_cap_memory_reports_evidence_units_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """>24 text units -> incomplete/evidence_units_capped (was notice_budget_exhausted)."""
    from memory_arbiter.constants import SEMANTIC_MAX_EVIDENCE_UNITS

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)
    memory_id = _write_many_units(tools, SEMANTIC_MAX_EVIDENCE_UNITS + 10)

    result = tools._process_semantic_conflict_job(memory_id, _job_snapshot(tools, memory_id))

    assert result["status"] == "incomplete"
    assert result["reason"] == "evidence_units_capped"
    assert result["notices_created"] == 0
    assert result["reasons_seen"] == ["evidence_units_capped"]
    status = tools._check_degradation_status()
    assert status["last_reason"] == "evidence_units_capped"
    assert "evidence_units_capped" in status["note"]


def test_job_deadline_keeps_notice_budget_exhausted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fair-deadline truncation (<24 units) keeps the original reason string."""
    import time as _time

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    monkeypatch.setattr(
        tools._semantic_worker, "pending_job_deadline", lambda timeout: _time.monotonic() - 1.0,
    )
    written = tools.memory_write(content="两条记录而已。", subject="small", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
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
    memory_id = _write_many_units(tools, 40)

    clock = {"now": 100.0}
    fairness_deadline = 100.5
    knn_calls = {"n": 0}

    def fake_knn(*a: Any, **k: Any) -> list[dict[str, Any]]:
        knn_calls["n"] += 1
        if knn_calls["n"] >= 24:
            clock["now"] = fairness_deadline + 1.0
        return []

    monkeypatch.setattr(tools.db, "evidence_knn", fake_knn)
    monkeypatch.setattr("memory_arbiter.pipeline.evidence.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        tools._semantic_worker, "pending_job_deadline", lambda timeout: fairness_deadline,
    )

    result = tools._process_semantic_conflict_job(memory_id, _job_snapshot(tools, memory_id))

    assert knn_calls["n"] == 24
    assert result["reason"] == "evidence_units_capped"


def test_technical_reasons_registry_includes_evidence_units_capped() -> None:
    from memory_arbiter.pipeline.evidence import _TECHNICAL_REASONS

    assert "evidence_units_capped" in _TECHNICAL_REASONS
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
    """
    if not _SLOW_MODEL.exists():
        pytest.skip(f"real model not installed at {_SLOW_MODEL}")
    from memory_arbiter.constants import SEMANTIC_N_CTX
    from memory_arbiter.semantic_conflict import LocalGGUFSemanticBackend

    backend = LocalGGUFSemanticBackend(_SLOW_MODEL, n_ctx=SEMANTIC_N_CTX, n_threads=4, n_batch=128)
    backend.load()
    yield backend
    try:
        backend.unload()
    except Exception:
        pass


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
def test_slow_date_pair_end_to_end_notice_ready(real_backend: "Any") -> None:
    """Dates differing only in value must extract and reach notice_ready."""
    backend = real_backend
    gate, _f, _r = _real_gate(
        backend,
        "新功能 scheduled-launch 的上线日期为 2026-09-01。",
        "新功能 scheduled-launch 的上线日期为 2026-09-10。",
    )
    assert gate.state == "notice_ready", gate
    assert gate.attribute and gate.value_a != gate.value_b


@pytest.mark.slow
def test_slow_long_quote_pair_outputs_complete_json(real_backend: "Any") -> None:
    """The 2026-09-08 live-host invalid_output sample: long prose pair must
    produce a complete, extractable JSON (no truncation) under v5+2048+400."""
    backend = real_backend
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
def test_slow_long_values_keep_difference(real_backend: "Any") -> None:
    """45–58 char historical value shapes must keep their distinction under
    the v5 fragment-selection wording (compression wording flattened them)."""
    backend = real_backend
    gate, _f, _r = _real_gate(
        backend,
        "Tier1 功能方案状态：Tier1 已确认功能方案（810 v2 重排版：merge / 定时任务引导 / find 增强，0.16 规划）。",
        "Tier1 功能方案状态：Tier1 仍为高 ROI 候选分析（809：三项待排期，merge 可复用冲突生命周期）。",
    )
    assert gate.value_a != gate.value_b, gate


@pytest.mark.slow
def test_slow_short_value_regression(real_backend: "Any") -> None:
    """json vs csv short values: the stable baseline."""
    backend = real_backend
    gate, _f, _r = _real_gate(
        backend,
        "conflict-bench 的 export-format 取值为 json。",
        "conflict-bench 的 export-format 取值为 csv。",
    )
    assert gate.state == "notice_ready", gate


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


@pytest.mark.slow
def test_slow_live_degradation_replay_over_limit_value(real_backend: "Any") -> None:
    """Replay of the 2026-09-08 17:31 degradation (mema id=925 evidence).

    Pre-fix (pair-v5, no retry) the reverse direction of this exact fixture
    answers invalid_schema/invalid_value_b — the 0.5B copies the 76-char
    "合并覆盖≥50%…（owner 拍板：…）" clause into value_b, reproducing the live
    qwen_invalid_output sample almost verbatim. Both directions must now
    produce protocol-valid extractions (short values or legal unknown_field).
    """
    backend = real_backend
    left = _faithful_env(
        "透出前过滤 kind=subject（坐标(0,0)）+重叠区间合并（overlap=60 防重复计数）；"
        "永不截断：合并覆盖≥50%全文献条目升级全文+hit_spans 转标注"
        "（owner 拍板：服务端不替 Agent 挑重要命中，法规 RAG 漏但书是系统性偏差）",
        subject="0.15.10 方案定稿待审：content_mode 三态（preview/hits/full）+ hit_spans 命中透出，include_content 删除（breaking），多轮否决清单存档",
        tags=["mema", "mema-core", "0.15.10", "content-mode", "hit-spans", "find", "batch-find",
              "implementation-plan", "owner-pending-review"],
        workspace="memory-arbiter-mcp", memory_id=925, version=5,
        event_time="2026-09-09T00:00:00+00:00",
    )
    right = _faithful_env(
        "透出前过滤：kind=subject（坐标(0,0)）+重叠区间合并（overlap=60 防重复计数），"
        "其余命中一律不透出，不做升级全文。",
        subject="透出前过滤", tags=["hit-spans"],
        workspace="memory-arbiter-mcp", memory_id=903, version=1,
        event_time="2026-09-08T00:00:00+00:00",
    )
    forward = backend.classify_pair(left, right, deadline_monotonic=None)
    reverse = backend.classify_pair(right, left, deadline_monotonic=None)
    _assert_no_invalid_output(forward, reverse)
    # Surface (do not assert) retry activity: today the grammar cap alone
    # passes these fixtures; if a future change lets copying through again,
    # this readout shows whether the retry backstop is absorbing it.
    print(f"pair retry stats: {backend.pair_retry_stats()}")


@pytest.mark.slow
def test_slow_live_degradation_replay_truncated_json(real_backend: "Any") -> None:
    """Replay of the 2026-09-09 04:48 degradation (mema id=926 evidence).

    Pre-fix (pair-v5, no retry) the forward direction of this exact fixture
    answers invalid_schema/invalid_value_a — the 0.5B copies the 200+ char
    review clause into value_a (in the live incident it kept going until the
    384-token budget killed the JSON mid-key). Both directions must now
    complete a protocol-valid extraction.
    """
    backend = real_backend
    left = _faithful_env(
        "代码评审综合 2.7/5：亮点=测试隔离专业、SQL 参数化、优雅降级、CAS 冲突仲裁、loopback 安全；"
        "硬伤 P0=4个 tool 全是 (action:str, data:dict) 反模式（server.py:347-427，LLM 必须先调 help，"
        "无 Literal 校验）、P1=90+处 except Exception 静默吞异常、P2=operations.py 2969行上帝文件、"
        "P3=help 文本 50 行术语、P4=5步安装含 200MB GGUF、P5=无 optional-dependencies。",
        subject="memarbiter/mema 项目战略评估结论",
        tags=["memarbiter", "mema", "mema-twin", "项目战略", "竞品分析", "代码评审", "止损决策"],
        workspace="default", memory_id=926, version=1,
        event_time="2026-09-09T04:48:12+00:00",
    )
    right = _faithful_env(
        "memory_arbiter 两轮 review 汇总：core 历史待办 + 新一轮 adversarial 发现（6 个开放问题）。"
        "建议修复前用 core/main 最新代码重新确认问题仍存在。",
        subject="memory_arbiter 两轮 review 汇总", tags=["code-review", "memory-arbiter"],
        workspace="memory-arbiter-mcp", memory_id=758, version=10,
        event_time="2026-08-27T14:32:04+00:00",
    )
    forward = backend.classify_pair(left, right, deadline_monotonic=None)
    reverse = backend.classify_pair(right, left, deadline_monotonic=None)
    _assert_no_invalid_output(forward, reverse)
    # Surface (do not assert) retry activity: today the grammar cap alone
    # passes these fixtures; if a future change lets copying through again,
    # this readout shows whether the retry backstop is absorbing it.
    print(f"pair retry stats: {backend.pair_retry_stats()}")
