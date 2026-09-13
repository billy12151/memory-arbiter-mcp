"""0.16.2 write-time pre-gate tests (owner, unified flow): provenance gate +
difference classifier BEFORE the per-peer dedup in the KNN collection loop,
and the internal-contradiction Qwen slot extraction (ready → annotated,
definitive negative → persistent dismissal, technical failure → fail-open).

Live-library calibration behind the design (2026-09-13 simulation over
51,437 KNN top-5 hits): ignore 41.3% (existing), classifier-clear 56.6%,
provenance blocks ~95% of representatives; clear representatives wasted 86
peer slots that a notify/keep hit of the same peer could have taken.
"""
from __future__ import annotations

import json
from pathlib import Path

from memory_arbiter.evidence import evidence_content_hash

from test_vnext_evidence import make_tools, _strict_pair_backend


META = {"entity": "checkout-api", "scope": "production"}


def _snapshot(tools, memory_id: int) -> dict:
    record = tools.db.get_memory(memory_id)
    return {
        "memory_id": int(memory_id),
        "version": record["version"],
        "content_hash": evidence_content_hash(record["content"]),
    }


def _hits(tools, peers, texts=None, distances=None):
    """Hand-built knn hits borrowing each peer's real metadata (the
    provenance gate reads hit['metadata'] exactly like the real knn row)."""
    hits = []
    for i, peer in enumerate(peers):
        record = tools.db.get_memory(peer["id"])
        meta = record.get("metadata")
        if isinstance(meta, (dict, list)):
            meta = json.dumps(meta, ensure_ascii=False)
        hits.append({
            "memory_id": peer["id"], "id": i + 1, "kind": "text",
            "text": (texts[i] if texts else record["content"]),
            "start_offset": 0, "end_offset": len(record["content"]),
            "distance": (distances[i] if distances else 0.1 + i * 0.05),
            "metadata": meta,
        })
    return hits


class _CountingBackend:
    """Wraps a real gate-shaped backend and counts classify_pair calls."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def classify_pair(self, left, right, *args, **kwargs):
        self.calls += 1
        try:
            return self._inner.classify_pair(left, right, *args, **kwargs)
        except TypeError:
            return self._inner.classify_pair(left, right)


# ── gate 1: provenance ──────────────────────────────────────────────────────

def test_provenance_gate_skips_qwen_and_reports(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    # Same-value pair on a slot the backend WOULD confirm — but the peer has
    # no entity/scope metadata, so no notice can ever land for it.
    peer = tools.memory_write(content="连接池上限为 10。", subject="pool", tags=[])["data"]
    new = tools.memory_write(content="连接池上限为 99。", subject="poolx", tags=[], metadata=META)["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    backend = _CountingBackend(_strict_pair_backend())
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: _hits(tools, [peer]))
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
    result = tools._process_semantic_conflict_job(new["id"], _snapshot(tools, new["id"]))
    assert result["status"] == "completed"
    assert result["notices_created"] == 0
    assert result["deterministic_filter"]["provenance_skipped"] >= 1
    assert backend.calls == 0, "a provenance-dead pair must never reach Qwen"
    assert tools.db.list_semantic_notices(status="open") == []


def test_provenance_gate_entity_mismatch_blocks(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    other_meta = {"entity": "billing-api", "scope": "production"}
    peer = tools.memory_write(content="连接池上限为 10。", subject="pool", tags=[], metadata=other_meta)["data"]
    new = tools.memory_write(content="连接池上限为 99。", subject="poolx", tags=[], metadata=META)["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    backend = _CountingBackend(_strict_pair_backend())
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: _hits(tools, [peer]))
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
    result = tools._process_semantic_conflict_job(new["id"], _snapshot(tools, new["id"]))
    assert backend.calls == 0
    assert result["deterministic_filter"]["provenance_skipped"] >= 1


# ── gate 2: difference classifier ───────────────────────────────────────────

def test_no_difference_check_pair_skipped_before_qwen(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    peer = tools.memory_write(
        content="压测报告已归档，采样窗口五百毫秒", subject="bench", tags=[], metadata=META,
    )["data"]
    new = tools.memory_write(
        content="压测报告完成归档，五百毫秒的采样窗口", subject="bench2", tags=[], metadata=META,
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    backend = _CountingBackend(_strict_pair_backend())
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: _hits(tools, [peer]))
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
    result = tools._process_semantic_conflict_job(new["id"], _snapshot(tools, new["id"]))
    assert result["deterministic_filter"]["no_difference_skipped"] >= 1
    assert backend.calls == 0


def test_keep_shape_check_pair_still_reaches_qwen(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    peer = tools.memory_write(content="连接池上限为 10。", subject="pool", tags=[], metadata=META)["data"]
    new = tools.memory_write(content="连接池上限为 99。", subject="poolx", tags=[], metadata=META)["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    backend = _CountingBackend(_strict_pair_backend())
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: _hits(tools, [peer]))
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
    result = tools._process_semantic_conflict_job(new["id"], _snapshot(tools, new["id"]))
    assert result["outcome"] == "notices_created"
    assert backend.calls == 2, "bidirectional extraction still runs for keepers"


def test_notify_pair_passes_both_gates(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    # Polarity shape WITH a number so the strict test backend can extract
    # values (notify routing is decided by 包含/不包含, the number only
    # feeds the gate envelope).
    peer = tools.memory_write(
        content="该功能不包含 100 条缓存模块", subject="cache", tags=[], metadata=META,
    )["data"]
    new = tools.memory_write(
        content="该功能包含 100 条缓存模块", subject="cache2", tags=[], metadata=META,
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    backend = _CountingBackend(_strict_pair_backend())
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: _hits(tools, [peer]))
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
    result = tools._process_semantic_conflict_job(new["id"], _snapshot(tools, new["id"]))
    assert backend.calls == 2, "notify pairs are never classifier-filtered (pass provenance only)"


def test_clear_hit_does_not_burn_peer_slot(tmp_path: Path, monkeypatch) -> None:
    """Slot-recovery regression (the 86-slot calibration case): a CLOSER
    clear-shaped hit and a FARTHER keep-shaped hit of the SAME peer — the
    gate must run BEFORE the per-peer dedup so the keeper represents the
    peer instead of the closer duplicate."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    peer = tools.memory_write(content="连接池上限为 10。", subject="pool", tags=[], metadata=META)["data"]
    new = tools.memory_write(content="连接池上限为 99。", subject="poolx", tags=[], metadata=META)["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    clear_hit = _hits(tools, [peer], texts=["压测报告已归档，采样窗口五百毫秒"], distances=[0.05])[0]
    keep_hit = _hits(tools, [peer], texts=["连接池上限为 10。"], distances=[0.30])[0]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: [clear_hit, keep_hit])
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _strict_pair_backend)
    result = tools._process_semantic_conflict_job(new["id"], _snapshot(tools, new["id"]))
    assert result["outcome"] == "notices_created", "the keeper hit must represent the peer"


# ── unified internal flow ───────────────────────────────────────────────────


def test_internal_keep_shape_qwen_confirmed_annotates_reason(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    mid = tools.memory_write(
        content="## 配置甲\n连接池上限为 10。\n## 配置乙\n连接池上限为 99。",
        subject="internal", tags=[],
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: [])
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _strict_pair_backend)
    result = tools._process_semantic_conflict_job(mid["id"], _snapshot(tools, mid["id"]))
    pending = tools.db.internal_conflicts.list_pending()
    assert pending, "the internal keeper must land"
    row = next(r for r in pending if r["memory_id"] == mid["id"])
    assert "qwen:连接池上限=10|99" in row["reason"], row["reason"]
    assert result["deterministic_filter"]["internal_qwen_confirmed"] >= 1


def test_internal_qwen_veto_persists_and_scan_cannot_resurrect(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    # Rule-keeper shape (3 vs 5) but the backend reports the SAME value for
    # both sides — a definitive semantic negative.
    mid = tools.memory_write(
        content="## 配置甲\n重试次数为 3 次。\n## 配置乙\n重试次数为 5 次。",
        subject="internal-veto", tags=[],
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: [])

    class SameValue:
        @staticmethod
        def classify_pair(left, right, *args, **kwargs):
            from memory_arbiter.semantic_conflict import ModelSignal
            return ModelSignal(
                True, "attribute_value_extraction", None, "",
                {"attribute_a": "重试次数", "value_a": "3",
                 "attribute_b": "重试次数", "value_b": "3"},
                None,
            )

    monkeypatch.setattr(tools, "_ensure_semantic_backend", SameValue)
    result = tools._process_semantic_conflict_job(mid["id"], _snapshot(tools, mid["id"]))
    assert result["deterministic_filter"]["internal_qwen_vetoed"] >= 1
    assert tools.db.internal_conflicts.list_pending() == [], "vetoed pairs never enter the pending queue"
    with tools.db.connection() as conn:
        dismissed = conn.execute(
            "SELECT COUNT(*) FROM internal_conflicts WHERE memory_id=? AND status='dismissed'",
            (mid["id"],),
        ).fetchone()[0]
    assert dismissed >= 1, "the veto must persist as a decided row"
    # Scan-side re-examination must not resurrect the vetoed pair.
    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 50})
    assert kick["ok"], kick
    assert not any(
        r for r in tools.db.internal_conflicts.list_pending() if r["memory_id"] == mid["id"]
    ), "scan re-examination must respect the write-time veto"


def test_internal_technical_failure_fails_open(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    mid = tools.memory_write(
        content="## 配置甲\n连接池上限为 10。\n## 配置乙\n连接池上限为 99。",
        subject="internal-open", tags=[],
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: [])

    class Broken:
        @staticmethod
        def classify_pair(left, right):
            from memory_arbiter.semantic_conflict import ModelSignal
            return ModelSignal(False, "backend_error", None, "boom", None, "boom")

    monkeypatch.setattr(tools, "_ensure_semantic_backend", Broken)
    tools._process_semantic_conflict_job(mid["id"], _snapshot(tools, mid["id"]))
    pending = [r for r in tools.db.internal_conflicts.list_pending() if r["memory_id"] == mid["id"]]
    assert pending, "a technical failure must fail OPEN — the rule signal still lands"
    assert "qwen:" not in pending[0]["reason"]


def test_internal_no_backend_lands_unannotated(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    mid = tools.memory_write(
        content="## 配置甲\n连接池上限为 10。\n## 配置乙\n连接池上限为 99。",
        subject="internal-nobackend", tags=[],
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: [])
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: None)
    tools._process_semantic_conflict_job(mid["id"], _snapshot(tools, mid["id"]))
    pending = [r for r in tools.db.internal_conflicts.list_pending() if r["memory_id"] == mid["id"]]
    assert pending, "no backend → land unannotated (0.16.0 behavior preserved)"
    assert "qwen:" not in pending[0]["reason"]


def test_internal_no_difference_shape_never_lands(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    # Same topic, enough shared tokens to route as a check pair, but the
    # unique-token spread (5 per side) exceeds the (2,2) window — a rewrite
    # of one claim, not two values of one claim.
    mid = tools.memory_write(
        content="## 段落甲\n压测报告已归档到本地目录。\n## 段落乙\n压测报告完成归档并上传到远端存储。",
        subject="internal-dup", tags=[],
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: [])
    backend = _CountingBackend(_strict_pair_backend())
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
    tools._process_semantic_conflict_job(mid["id"], _snapshot(tools, mid["id"]))
    assert not any(
        r for r in tools.db.internal_conflicts.list_pending() if r["memory_id"] == mid["id"]
    ), "a same-claim rewrite inside one memory is duplication, not contradiction"
    assert backend.calls == 0


def test_internal_keepers_survive_collection_truncation(tmp_path: Path, monkeypatch) -> None:
    """Adversarial regression (self-review): an evidence-unit cap or budget
    exhaustion hit DURING the KNN collection loop must not drop the already
    collected internal keepers — E10① says internal findings land first and
    survive cross-loop truncation; the fail-open landing closes the gap."""
    from memory_arbiter.constants import SEMANTIC_MAX_EVIDENCE_UNITS

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    mid = tools.memory_write(
        content="## 配置甲\n连接池上限为 10。\n## 配置乙\n连接池上限为 99。",
        subject="internal-trunc", tags=[],
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    # Force the unit cap to trip on the FIRST unit: the internal keepers were
    # collected before collection even started, but the Qwen pass (and the
    # old landing point) sits after the truncation return.
    monkeypatch.setattr(
        "memory_arbiter.pipeline.evidence.SEMANTIC_MAX_EVIDENCE_UNITS", 0,
    )
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _strict_pair_backend)
    result = tools._process_semantic_conflict_job(mid["id"], _snapshot(tools, mid["id"]))
    assert result["status"] == "incomplete"
    assert result["reason"] == "evidence_units_capped"
    assert result["internal_conflicts"] >= 1, "internal keepers must land despite truncation"
    pending = [r for r in tools.db.internal_conflicts.list_pending() if r["memory_id"] == mid["id"]]
    assert pending, "the internal keeper row exists"
    assert "qwen:" not in pending[0]["reason"], "truncated runs land unannotated (no backend pass)"
