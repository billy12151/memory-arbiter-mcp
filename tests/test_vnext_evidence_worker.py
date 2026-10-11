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

from tests.test_vnext_evidence import (
    CountingEmbedder, FakeEmbedder, ModelSignal,
    _conflict_member, _hits_with_metadata, _job_snapshot, _pass_cos_gate,
    _seed_notice, _strict_pair_backend, _structured_conflict_payload,
    make_tools,
)  # noqa: F401  # facade 基建（无状态），域文件统一带包 import

# ── test_vnext_evidence worker/notice domain (commit 12 split) ──



def test_notice_pairs_not_capped_by_count(tmp_path: Path, monkeypatch) -> None:
    """0.15.14 (A5): the surfaced>=max_notice_pairs early stop is gone — every
    examined pair surfaces its notice (count bounded only by the examined-
    pairs cap, which these 4 pairs stay under)."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    metadata = {"entity": "checkout-api", "scope": "production"}
    peers = [
        tools.memory_write(
            content=f"连接池上限为 {i + 10}。", subject=f"pool{i}", tags=[], metadata=metadata,
        )["data"]
        for i in range(4)
    ]
    new = tools.memory_write(
        content="连接池上限为 99。", subject="poolx", tags=[], metadata=metadata,
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    hits = [
        {"memory_id": peer["id"], "id": i, "kind": "text", "text": f"连接池上限为 {i + 10}。",
         "start_offset": 0, "end_offset": 11, "distance": 0.10 + i * 0.05}
        for i, peer in enumerate(peers)
    ]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _strict_pair_backend)

    first = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    first.pop("elapsed_ms", None)
    for _row_key in ("rows_mode", "rows_examined", "claims_channel", "claims_channel_c",
                     "judge_budget", "direct_verdicts"):  # Q1 additive receipt keys
        first.pop(_row_key, None)
    assert first == {"status": "completed", "outcome": "notices_created", "notices_created": 4}
    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]
    assert len(notices) == 4
    assert all(n["payload"]["route"] == "notice_ready" for n in notices)
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conflicts WHERE notice_type IS NOT NULL").fetchone()[0] == 4

    second = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    second.pop("elapsed_ms", None)
    for _row_key in ("rows_mode", "rows_examined", "claims_channel", "claims_channel_c",
                     "judge_budget", "direct_verdicts"):  # Q1 additive receipt keys
        second.pop(_row_key, None)
    assert second == {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0}
    assert len([n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]) == 4


def test_unified_notice_dedupe_does_not_starve_fresh_pair(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    metadata = {"entity": "checkout-api", "scope": "production"}
    peers = [
        tools.memory_write(
            content=f"连接池上限为 {i + 10}。", subject=f"pool{i}", tags=[], metadata=metadata,
        )["data"]
        for i in range(3)
    ]
    new = tools.memory_write(
        content="连接池上限为 99。", subject="poolx", tags=[], metadata=metadata,
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    hits = [
        {"memory_id": peer["id"], "id": i, "kind": "text", "text": f"连接池上限为 {i + 10}。",
         "start_offset": 0, "end_offset": 11, "distance": 0.10 + i * 0.05}
        for i, peer in enumerate(peers)
    ]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _strict_pair_backend)

    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]
    assert sorted(n["peer_id"] for n in notices) == [peer["id"] for peer in peers]
    assert len({n["dedupe_key"] for n in notices}) == 3


def test_search_surfaces_vector_lag(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_search(query="连接池")
    assert "vector_lag" in result["data"]
    assert result["data"]["vector_lag"]["pending_evidence_index"] >= 0


def test_check_degradation_is_visible_in_semantic_status(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, semantic_enabled=False)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "checkout-api", "scope": "production"}
    peer = tools.memory_write(content="database connection policy 8 and 16", subject="pool", tags=[], metadata=meta)["data"]
    new = tools.memory_write(content="database connection pool size 3 and 4", subject="pool2", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    assert tools._ensure_semantic_backend() is None
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "database connection policy 8 and 16", "start_offset": 0, "end_offset": 26, "distance": 0.2}]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)
    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    assert tools.db.list_semantic_notices(status="open") == []
    degradation = tools._semantic_status()["check_degradation"]
    assert degradation["last_reason"] == "judge_unavailable"
    assert degradation["count"] >= 1


def test_short_paragraphs_merge_instead_of_drop() -> None:
    from memory_arbiter.evidence import local_text_units as ltu
    units = ltu("s", "这是一个足够长的第一段内容。\n短。\n这是另一个足够长的段落内容。")
    merged_texts = [u.text for u in units if u.kind == "text"]
    assert any("短" in t for t in merged_texts)
    lone = ltu("s", "仅此。")
    assert any(u.kind == "text" and u.text for u in lone)


def test_knn_enforces_memory_status_despite_stale_parent(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    # Simulate a vec-disabled process superseding the memory: parent_status
    # stays 'active' while memories.status is authoritative.
    tools.db.state.sqlite_vec_available = False
    assert tools.db.update_memory(written["id"], {"status": "superseded"})
    tools.db.state.sqlite_vec_available = True
    hits = tools.db.row_knn([0.8, 0.2], k=5, parent_status_filter="active")
    assert all(h["memory_id"] != written["id"] for h in hits)


def test_knn_workspace_overfetch_restores_recall(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # Insert B first so the later A units win sqlite-vec's equal-distance tie
    # order and own the initial global top-k window.
    peer = tools.memory_write(
        content="postgres 端口为 5432。", subject="pg port", tags=[], workspace="wsB",
    )["data"]
    bulk = "\n\n".join(f"postgres 配置项 {i}：连接池参数说明 {i}。" for i in range(8))
    bulk_record = tools.memory_write(
        content=bulk, subject="pg bulk", tags=[], workspace="wsA",
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    # FakeEmbedder intentionally maps all workspace names to one vector, so
    # normal workspace resolution treats wsB as an alias of wsA. This test is
    # about KNN filtering rather than canonicalization; restore the intended
    # independent canonical fixture explicitly.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET workspace_canonical=workspace WHERE id IN (?,?)",
            (peer["id"], bulk_record["id"]),
        )
    hits = tools.db.row_knn([1.0, 0.0], k=5, workspace="wsB", exclude_memory_id=999999)
    assert any(h["memory_id"] == peer["id"] for h in hits)


def test_evidence_only_candidates_carry_memory_row_shape(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="postgresql 连接池上限为 20。", subject="pool limit", tags=["db"], workspace="w")
    # A memory with zero lexical overlap so it can only be recalled via evidence.
    tools.memory_write(content="pgsql 的 pool 上限数值是二十，注意与连接数区分。", subject="pool", tags=["db"], workspace="w")
    assert tools.wait_semantic_worker_drained(timeout=2)
    result = tools.memory_search(query="连接池 上限")
    for row in result["data"]["results"]:
        assert isinstance(row.get("version"), int)
        assert "kind" not in row and "start_offset" not in row and "content_hash" not in row
        assert "_evidence_hits" in row or row.get("_match_reason") != "evidence_vec_recall"


def test_semantic_worker_sync_wait_observes_same_task_completion() -> None:
    from types import SimpleNamespace
    from memory_arbiter.workers import SemanticConflictWorker

    calls: list[tuple[int, str]] = []

    class Tools:
        settings = SimpleNamespace(
            semantic_conflict_on_write="async",
            semantic_conflict_preload=False,
            semantic_conflict_queue_max_size=10,
        )

        @staticmethod
        def _process_semantic_conflict_job(memory_id, snapshot):
            calls.append((memory_id, snapshot["task_id"]))
            return {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0}

    worker = SemanticConflictWorker(Tools())
    task_id = "semantic:42@3"
    queued = worker.enqueue(42, {"version": 3, "task_id": task_id, "dedupe_key": task_id})
    assert queued["task_id"] == task_id and queued["dedupe_key"] == task_id
    completed = worker.wait_task(task_id, 2)
    assert completed == {
        "status": "completed", "outcome": "checked_no_notice", "notices_created": 0,
        "task_id": task_id, "dedupe_key": task_id,
    }
    assert calls == [(42, task_id)]
    assert worker.enqueue(42, {"version": 3, "task_id": task_id})["status"] == "completed"
    assert calls == [(42, task_id)]


def test_semantic_worker_coalescing_completes_displaced_task() -> None:
    from types import SimpleNamespace
    from memory_arbiter.workers import SemanticConflictWorker

    class Tools:
        settings = SimpleNamespace(
            semantic_conflict_on_write="async", semantic_conflict_preload=False,
            semantic_conflict_queue_max_size=10,
        )

    worker = SemanticConflictWorker(Tools())
    worker._ensure_thread = lambda: None
    first = "semantic:42@1"
    second = "semantic:42@2"
    worker.enqueue(42, {"version": 1, "task_id": first})
    worker.enqueue(42, {"version": 2, "task_id": second})
    assert worker.wait_task(first, 0) == {
        "status": "incomplete", "reason": "coalesced_by_newer_snapshot", "notices_created": 0,
        "task_id": first, "dedupe_key": first,
    }
    assert second in worker._expected
    worker.shutdown(discard_pending=True)


def test_evidence_index_error_completes_exact_reserved_task(tmp_path: Path, monkeypatch) -> None:
    """C2 语义：索引失败发生在语义 job 内（单队列），worker 把精确的
    task 标 incomplete——写入侧 sync 窗收到同一 task 的失败回执。"""
    tools = make_tools(tmp_path)

    def failing_job(memory_id, snapshot):
        # Simulates the job's index phase failing (C2: indexing lives in the
        # semantic job now — the failure surface moved with it). Q1 相分裂:
        # the wrapper enters through the deterministic phase, so the failure
        # stamps a terminal receipt on the job context.
        ctx = tools._evidence._new_conflict_ctx(memory_id, snapshot)
        ctx["terminal"] = {"status": "incomplete", "reason": "index_synthetic_failure",
                           "notices_created": 0}
        return ctx

    monkeypatch.setattr(
        tools._evidence, "conflicts_deterministic_phase", failing_job,
    )
    result = tools.memory_write(content="index me", subject="index failure", tags=[])
    check = result["data"]["semantic_conflict_check"]
    task_id = result["data"]["evidence_index"]["semantic_task_id"]
    assert check["status"] == "incomplete"
    assert "synthetic_failure" in str(check.get("reason"))
    assert check["task_id"] == task_id and check["dedupe_key"] == task_id
    assert "timeout_continuing_async" not in str(result)


def test_queue_full_drop_completes_exact_reserved_task_end_to_end(
    tmp_path: Path, monkeypatch,
) -> None:
    tools = make_tools(tmp_path)
    # queue_max_size froze into a constant (0.15.0); shrink it through the
    # module attribute the worker reads at enqueue time.
    monkeypatch.setattr("memory_arbiter.workers.SEMANTIC_QUEUE_MAX_SIZE", 1)
    tools._semantic_worker._ensure_thread = lambda: None
    tools._semantic_worker.enqueue(111, {"version": 1, "task_id": "semantic:111@1"})
    task_id = "semantic:222@1"
    tools._semantic_worker.reserve(task_id)
    outcome = tools._semantic_worker.enqueue(222, {"version": 1, "task_id": task_id})
    assert outcome == {
        "status": "incomplete", "reason": "queue_full", "notices_created": 0,
        "task_id": task_id, "dedupe_key": task_id,
    }
    assert tools._semantic_worker.wait_task(task_id, 0) == outcome
    status = tools._semantic_status()["worker"]
    assert status["dropped_queue_full"] == 1
    tools._semantic_worker.shutdown(discard_pending=True)


def test_paused_disabled_and_shutdown_enqueue_complete_exact_task(tmp_path: Path) -> None:
    for state in ("paused", "runtime_disabled", "shutdown"):
        state_path = tmp_path / state
        state_path.mkdir()
        tools = make_tools(state_path)
        worker = tools._semantic_worker
        worker._ensure_thread = lambda: None
        if state == "paused":
            worker.pause()
        elif state == "runtime_disabled":
            worker.disable_runtime()
        else:
            worker.shutdown()
        task_id = f"semantic:7@{state}"
        worker.reserve(task_id)
        outcome = worker.enqueue(7, {"version": 1, "task_id": task_id})
        assert outcome == {
            "status": "incomplete", "reason": state, "notices_created": 0,
            "task_id": task_id, "dedupe_key": task_id,
        }
        assert worker.wait_task(task_id, 0) == outcome


def test_judge_timeout_and_backend_error_map_to_check_degradation(tmp_path: Path, monkeypatch) -> None:

    tools = make_tools(tmp_path, semantic_enabled=False)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "checkout-api", "scope": "production"}
    # Multi-value corpus: passes the gate-v2 prefilter AND keeps the
    # deterministic direct verdict silent (", " in left_value) so the pair
    # reaches the failing-Qwen mocks this test exists to pin.
    peer = tools.memory_write(content="database connection policy 8 and 16", subject="pool", tags=[], metadata=meta)["data"]
    new = tools.memory_write(content="database connection pool size 3 and 4", subject="pool2", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "database connection policy 8 and 16", "start_offset": 0, "end_offset": 26, "distance": 0.2}]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)

    cases = [
        ("mdeberta inference hard timeout after 30ms", "judge_timeout"),
        ("mdeberta child exited", "judge_backend_error"),
        ("mdeberta judge disabled", "judge_unavailable"),
    ]
    for error, expected in cases:
        class ErrBackend:
            @staticmethod
            def judge_pair(text_a, text_b):
                from memory_arbiter.semantic_judge import PairVerdict
                return PairVerdict("no_conflict", {}, None, "mdeberta:unavailable", error=error)

        monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: ErrBackend())
        tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
        degradation = tools._semantic_status()["check_degradation"]
        assert degradation["last_reason"] == expected
    assert tools.db.list_semantic_notices(status="open") == []
    # 0.17.1: the judge produces no raw output — the invalid_output sample
    # assertion retired with the decode-retry protocol.

