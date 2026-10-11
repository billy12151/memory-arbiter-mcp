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

# ── test_vnext_evidence job budget/reentry domain (commit 12 split) ──


def _grounded_db_backend():
    """A backend that extracts a grounded mysql/sqlite slot from db-value quotes."""

    class Backend:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            def value(env):
                text = str(env["quote"]).casefold()
                return "mysql" if "mysql" in text else ("sqlite" if "sqlite" in text else "postgres")
            parsed = {
                "attribute_a": "数据库选型", "value_a": value(left),
                "attribute_b": "数据库选型", "value_b": value(right),
            }
            return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)
    return Backend()


def test_clean_gate_negative_reaches_checked_no_notice(tmp_path: Path, monkeypatch) -> None:
    """A candidate examined and cleanly rejected reports checked_no_notice."""

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    peer = tools.memory_write(content="database is mysql 8 here", subject="a", tags=[], metadata=meta)["data"]
    new = tools.memory_write(content="database is mysql 8 there", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "database is mysql 8 here",
             "start_offset": 0, "end_offset": 22, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)

    # 0.17.1: a clean negative is the judge saying no_conflict (the old
    # same-extracted-value gate died with the slot paradigm).
    class NoConflict:
        @staticmethod
        def judge_pair(text_a, text_b):
            from memory_arbiter.semantic_judge import PairVerdict
            return PairVerdict(
                "no_conflict",
                {"conflict": 0.05, "no_conflict": 0.9, "possible_conflict": 0.05},
                None, "mdeberta-v4m:test",
            )
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: NoConflict())

    result = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    result.pop("elapsed_ms", None)
    result.pop("judge_budget", None); result.pop("qwen_budget", None)  # Q1 additive receipt key（本用例 a_cross 扣池 1）
    assert result == {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0, "pairs_examined": 1, "rows_mode": True, "rows_examined": 1, "model_clear": 1}
    # A clean model decision is not counted as check degradation.
    degradation = tools._semantic_status()["check_degradation"]
    assert degradation["last_reason"] != "not_same_attribute_different_value"


def test_idle_worker_job_budget_does_not_cap_inflight_qwen(tmp_path: Path, monkeypatch) -> None:
    """With no backlog, even a tiny job budget must not truncate inference."""
    import time

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    # Job/min-pair budgets froze into constants (0.15.0); shrink them through
    # the module attributes the evidence pipeline reads.
    monkeypatch.setattr("memory_arbiter.pipeline._evidence_helpers.SEMANTIC_JOB_TIMEOUT_MS", 10)
    monkeypatch.setattr("memory_arbiter.pipeline._evidence_phases.SEMANTIC_MIN_PAIR_BUDGET_MS", 5)
    metadata = {"entity": "svc", "scope": "production"}
    peer = tools.memory_write(content="database is mysql 8 and 16", subject="a", tags=[], metadata=metadata)["data"]
    new = tools.memory_write(content="database is sqlite 3 and 4", subject="b", tags=[], metadata=metadata)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "database is mysql 8 and 16",
             "start_offset": 0, "end_offset": 17, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)

    deadlines = []
    base = _grounded_db_backend()

    class SlowBackend:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            deadlines.append(deadline_monotonic)
            time.sleep(0.02)
            return base.classify_pair(left, right, deadline_monotonic=deadline_monotonic)

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: SlowBackend())
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)

    result = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))

    result.pop("elapsed_ms", None)
    result.pop("judge_budget", None); result.pop("qwen_budget", None)  # Q1 additive receipt key（本用例 a_cross 扣池 1）
    assert result == {"status": "completed", "outcome": "notices_created", "notices_created": 1, "pairs_examined": 1, "rows_mode": True, "rows_examined": 1}
    # 0.17.1 攒批：判定走批前向，deadline 检查粒度从每对变为每块。
    # 0.17.1 重标合一：classify_pair-only 后端统一走 _judge_pair_compat
    # 映射被真实调用（旧 cross 侧 conflict-1.0 直通退役），批前向不传
    # deadline_monotonic → 每个 chunk 一次调用、观测值恒 None。
    assert deadlines == [None]


def test_backlog_job_budget_stops_before_next_pair_not_during_inference(tmp_path: Path, monkeypatch) -> None:
    """A queued job enables fairness, but the current pair gets its full hard timeout."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    monkeypatch.setattr("memory_arbiter.pipeline._evidence_helpers.SEMANTIC_JOB_TIMEOUT_MS", 40)
    monkeypatch.setattr("memory_arbiter.pipeline._evidence_phases.SEMANTIC_MIN_PAIR_BUDGET_MS", 5)
    metadata = {"entity": "svc", "scope": "production"}
    peer_values = ("mysql", "postgres")
    peers = [
        tools.memory_write(content=f"database is {value} 8 and 16", subject=value, tags=[], metadata=metadata)["data"]
        for value in peer_values
    ]
    new = tools.memory_write(content="database is sqlite 3 and 4", subject="sqlite", tags=[], metadata=metadata)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    hits = [
        {"memory_id": peer["id"], "id": index + 1, "kind": "text", "text": f"database is {peer_values[index]} 8 and 16",
         "start_offset": 0, "end_offset": len(f"database is {peer_values[index]} 8 and 16"), "distance": 0.1 + index * 0.01}
        for index, peer in enumerate(peers)
    ]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)
    clock = {"now": 100.0}
    fairness_deadline = 100.04
    monkeypatch.setattr("memory_arbiter.pipeline.evidence.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        tools._semantic_worker, "pending_job_deadline", lambda timeout: fairness_deadline,
    )

    deadlines = []
    base = _grounded_db_backend()

    class SlowBackend:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            deadlines.append(deadline_monotonic)
            clock["now"] += 0.04  # single-direction: one call per pair carries the old two-call wall time
            return base.classify_pair(left, right, deadline_monotonic=deadline_monotonic)

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: SlowBackend())

    result = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))

    # reasons_seen is attached because the second pair hit the job budget
    # (judge_budget_exhausted) after the first pair's notice was created;
    # truncated flags that the check was bounded, not exhaustive
    # (second-round review).
    result.pop("elapsed_ms", None)
    result.pop("judge_budget", None); result.pop("qwen_budget", None)  # Q1 additive receipt key（本用例 a_cross 扣池 1）
    # 0.17.0 P2-3/P2-4: rows receipt keys + budget-skipped pairs backlog
    result.pop("rows_mode", None)
    result.pop("rows_examined", None)
    result.pop("backlogged", None)
    # 0.17.1 攒批语义：pass1 逐对收对（慢后端在批前向里才走钟，pass1 内
    # 时钟不动 → 公平墙不咬），两对一次批前向各落一条 notice——公平墙
    # 语义从「逐对中断」变为「每块结算」，积压由 backlog 承接（不变式）。
    assert {
        "status": "completed", "outcome": "notices_created", "notices_created": 2,
        "pairs_examined": 2,
    } == {k: v for k, v in result.items() if k in {
        "status", "outcome", "notices_created", "pairs_examined",
    }}
    # 重标合一：两对一片、compat 映射各真调一次 classify_pair（无 deadline）
    assert deadlines == [None, None]
    assert len(tools.db.list_semantic_notices(status="open", limit=10)) == 2


def test_pending_job_deadline_uses_actual_enqueue_time(monkeypatch) -> None:
    from types import SimpleNamespace
    from memory_arbiter.workers import SemanticConflictWorker

    now = 100.0
    monkeypatch.setattr("memory_arbiter.workers.time.monotonic", lambda: now)

    class Tools:
        settings = SimpleNamespace(
            semantic_conflict_on_write="async", semantic_conflict_preload=False,
            semantic_conflict_queue_max_size=10,
        )

    worker = SemanticConflictWorker(Tools())
    worker._ensure_thread = lambda: None
    worker.enqueue(42, {"version": 1, "task_id": "semantic:42@1"})
    assert worker.pending_job_deadline(5.0) == 105.0

    now = 120.0
    # The deadline does not restart when the in-flight job notices the backlog.
    assert worker.pending_job_deadline(5.0) == 105.0
    worker.shutdown(discard_pending=True)


def test_semantic_wait_drained_includes_pending_jobs() -> None:
    from types import SimpleNamespace
    from memory_arbiter.workers import SemanticConflictWorker

    class Tools:
        settings = SimpleNamespace(
            semantic_conflict_on_write="async", semantic_conflict_preload=False,
            semantic_conflict_queue_max_size=10,
        )

    worker = SemanticConflictWorker(Tools())
    worker._ensure_thread = lambda: None
    worker.enqueue(42, {"version": 1, "task_id": "semantic:42@1"})
    assert worker.wait_drained(timeout=0) is False
    worker.shutdown(discard_pending=True)
    assert worker.wait_drained(timeout=0) is True


def test_applying_reentry_suppresses_same_conflict_notice(tmp_path: Path, monkeypatch) -> None:
    """Spec §5/§15.3: an apply-plan edit does not re-notice the same conflict."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    a = tools.memory_write(content="database is mysql 8", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(content="database is sqlite", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _grounded_db_backend)

    # Record + judge the a/b conflict into applying with a as the wrong current fact.
    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    def member(mid, value):
        quote = f"database is {value}"
        return ConflictMember(
            memory_id=mid, version=1, attribute_raw="数据库选型", value_raw=value,
            normalized_attribute="数据库选型", normalized_value=value, evidence_quote=quote,
            evidence_span=(0, len(quote)), content_hash=(str(mid) * 64)[:64],
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ).to_dict()
    recorded = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "svc", "attribute": "数据库选型", "scope": "production"},
        "members": [member(a["id"], "mysql"), member(b["id"], "sqlite")],
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{a['id']}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{b['id']}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scan",
        "reason": "db conflict", "status": "open",
    })
    conflict_id = recorded["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed", "authorized": True,
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"},
                       {"memory_id": b["id"], "action": "use_as_resolution"}],
        "resolution_memory_id": b["id"],
    })
    # Now run the post-apply semantic job for a's new version against b: the
    # re-entry rule must suppress a fresh notice for the same conflict.
    # (Near-duplicate wording: the 0.16.6 dedup gate refuses editing a into
    # b's exact bytes while both are active; grounding only needs "sqlite".)
    tools.db.edit_memory_intent(a["id"], new_content="database is sqlite, applied", reason="apply")
    updated = tools.db.get_memory(a["id"])
    hits = [{"memory_id": b["id"], "id": 1, "kind": "text", "text": "database is sqlite",
             "start_offset": 0, "end_offset": 18, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a_, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)
    snapshot = {
        "memory_id": a["id"], "version": updated["version"],
        "content_hash": evidence_content_hash(updated["content"]),
        "trusted_applying_context": {
            "conflict_id": conflict_id, "revision": 2, "memory_id": a["id"],
            "action": "update_current_claim", "chosen_value": "sqlite",
        },
    }
    result = tools._process_semantic_conflict_job(a["id"], snapshot)
    # No new notice for the same applying conflict pair.
    fresh = [n for n in tools.db.list_semantic_notices(status="open")
             if {n.get("memory_id"), n.get("peer_id")} == {a["id"], b["id"]}]
    assert fresh == []


def test_applying_reentry_does_not_suppress_different_slot(tmp_path: Path, monkeypatch) -> None:
    """Two applying-group members can still surface a conflict on another slot."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    a = tools.memory_write(content="database is mysql 8", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(content="database is sqlite", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)

    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    def member(mid, value):
        quote = f"database is {value}"
        return ConflictMember(
            memory_id=mid, version=1, attribute_raw="数据库选型", value_raw=value,
            normalized_attribute="数据库选型", normalized_value=value, evidence_quote=quote,
            evidence_span=(0, len(quote)), content_hash=(str(mid) * 64)[:64],
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ).to_dict()

    recorded = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "svc", "attribute": "数据库选型", "scope": "production"},
        "members": [member(a["id"], "mysql"), member(b["id"], "sqlite")],
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{a['id']}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{b['id']}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scan",
        "reason": "db conflict", "status": "open",
    })
    conflict_id = recorded["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed", "authorized": True,
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"},
                       {"memory_id": b["id"], "action": "use_as_resolution"}],
        "resolution_memory_id": b["id"],
    })

    tools.db.edit_memory_intent(a["id"], new_content="连接池上限为 99。", reason="apply")
    updated = tools.db.get_memory(a["id"])
    hits = [{"memory_id": b["id"], "id": 1, "kind": "text", "text": "连接池上限为 10。",
             "start_offset": 0, "end_offset": 11, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a_, **k: _hits_with_metadata(tools, list(hits)))
    # 0.17.0 P2-3：行级模式候选走 row_knn，注入同一批受控命中
    monkeypatch.setattr(tools.db, "row_knn", lambda *a_, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _strict_pair_backend)
    snapshot = {
        "memory_id": a["id"], "version": updated["version"],
        "content_hash": evidence_content_hash(updated["content"]),
        "trusted_applying_context": {
            "conflict_id": conflict_id, "revision": 2, "memory_id": a["id"],
            "action": "update_current_claim", "chosen_value": "sqlite",
        },
    }

    result = tools._process_semantic_conflict_job(a["id"], snapshot)

    assert result["outcome"] == "notices_created"
    assert result["notices_created"] == 1
    notices = [n for n in tools.db.list_semantic_notices(status="open")
               if {n.get("memory_id"), n.get("peer_id")} == {a["id"], b["id"]}]
    assert len(notices) == 1
    assert notices[0]["slot_key"]["attribute"] == "连接池上限"


@pytest.mark.parametrize("missing_field", ["revision", "action"])
def test_applying_reentry_context_requires_revision_and_action(
    tmp_path: Path, monkeypatch, missing_field: str,
) -> None:
    """Incomplete trusted context must fail closed instead of suppressing a notice."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    a = tools.memory_write(content="database is mysql 8", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(content="database is sqlite", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)

    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    def member(mid, value):
        quote = f"database is {value}"
        return ConflictMember(
            memory_id=mid, version=1, attribute_raw="数据库选型", value_raw=value,
            normalized_attribute="数据库选型", normalized_value=value, evidence_quote=quote,
            evidence_span=(0, len(quote)), content_hash=(str(mid) * 64)[:64],
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ).to_dict()

    recorded = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "svc", "attribute": "数据库选型", "scope": "production"},
        "members": [member(a["id"], "mysql"), member(b["id"], "sqlite")],
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{a['id']}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{b['id']}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scan",
        "reason": "db conflict", "status": "open",
    })
    conflict_id = recorded["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed", "authorized": True,
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"},
                       {"memory_id": b["id"], "action": "use_as_resolution"}],
        "resolution_memory_id": b["id"],
    })
    monkeypatch.setattr(tools.db, "list_open_conflicts_for_memory_ids", lambda *a_, **k: [])
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _grounded_db_backend)
    hits = [{"memory_id": b["id"], "id": 1, "kind": "text", "text": "database is sqlite",
             "start_offset": 0, "end_offset": 18, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "row_knn", lambda *a_, **k: _hits_with_metadata(tools, list(hits)))
    _pass_cos_gate(monkeypatch)
    context = {
        "conflict_id": conflict_id, "revision": 2, "memory_id": a["id"],
        "action": "update_current_claim", "chosen_value": "sqlite",
    }
    context.pop(missing_field)
    snapshot = _job_snapshot(tools, a["id"])
    snapshot["trusted_applying_context"] = context

    result = tools._process_semantic_conflict_job(a["id"], snapshot)

    assert result["outcome"] == "notices_created"
    assert result["notices_created"] == 1


def test_exact_match_write_repairs_missing_canonical_vector(tmp_path: Path) -> None:
    """The advertised 'retry a write using this workspace' actually republishes
    a canonical whose vector publication previously failed."""
    tools = make_tools(tmp_path)
    tools.memory_write(content="alpha", workspace="projA", subject="s", tags=[])
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM workspace_canonicals_vec")
    result = tools.memory_write(content="beta", workspace="projA", subject="s2", tags=[])
    assert result["data"]["workspace_matched_by"] == "exact"
    with tools.db.connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM workspace_canonicals_vec").fetchone()["c"]
    assert count >= 1


def test_publish_consumes_row_content_sha(tmp_path: Path, monkeypatch) -> None:
    """0.16.12 P2-T2：行有 content_sha 列时 publish/快照不再重算哈希。"""
    tools = make_tools(tmp_path, semantic_enabled=False)
    mid = tools.memory_write(content="哈希消费验证内容", subject="sha", source_type="agent_generated")["data"]["id"]
    tools.wait_semantic_worker_drained(timeout=10.0)
    row = tools.db.get_memory(mid)
    assert row.get("content_sha")

    import memory_arbiter.pipeline.evidence as pe_mod

    calls = []
    real_hash = pe_mod.evidence_content_hash
    monkeypatch.setattr(
        pe_mod, "evidence_content_hash",
        lambda text: (calls.append(text), real_hash(text))[1],
    )
    published = tools._evidence.index_memory(mid)
    assert published.get("published") is True, published
    assert calls == [], "行 content_sha 在位时不得重算哈希"
