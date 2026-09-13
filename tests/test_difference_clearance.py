"""0.16.2 scan-route tests (plan §1.4/§1.5/§1.8): top-3 rank gate for
machine-decidable routes, numeric-suppression exemption (owner ⑩), and the
boot difference-clearance stock migration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter.db import additive
from memory_arbiter.db_generation import CONFLICT_DETECTOR_VERSION

from test_scan_pipeline import make_tools, _write


def _conflict_member_sets(tools) -> list[set[int]]:
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT member_versions FROM scan_queue WHERE kind='conflict'"
        ).fetchall()
    return [
        {int(m["memory_id"]) for m in json.loads(row["member_versions"])}
        for row in rows
    ]


def test_top3_rank_gate_skips_deep_check_pairs(tmp_path: Path) -> None:
    """check-route pairs beyond rank 3 are not generated at all; notify
    pairs queue from any rank (real-conflict recall has no threshold)."""
    tools = make_tools(tmp_path)
    a = _write(tools, "主题甲", "该功能包含缓存模块")
    notify_peer = _write(tools, "主题乙", "该功能不包含缓存模块")
    check_peer = _write(tools, "数值甲", "重试次数为 3 次")
    _write(tools, "数值乙", "重试次数为 5 次")
    assert tools.wait_evidence_worker_drained(timeout=10)

    pipeline = tools._scan_pipeline

    def _fake_knn(embedding, *, k, workspace=None, exclude_memory_id=None, **_):
        # Both interesting peers sit at rank 5 (index 4) — beyond top-3.
        hits = []
        filler = 0
        for idx in range(4):
            hits.append({
                "kind": "text", "memory_id": 10 ** 6 + filler,
                "memory_version": 1, "text": f"填充邻居句子内容{idx}",
                "start_offset": 0, "end_offset": 10, "id": idx,
                "content_hash": f"filler{idx}", "workspace": "ws",
            })
            filler += 1
        hits.append({
            "kind": "text", "memory_id": notify_peer, "memory_version": 1,
            "text": "该功能不包含缓存模块", "start_offset": 0, "end_offset": 10,
            "id": 90, "content_hash": "notify-hash", "workspace": "ws",
        })
        hits.append({
            "kind": "text", "memory_id": check_peer, "memory_version": 1,
            "text": "重试次数为 5 次", "start_offset": 0, "end_offset": 10,
            "id": 91, "content_hash": "check-hash", "workspace": "ws",
        })
        return hits

    original_knn = pipeline.db.evidence.knn
    pipeline.db.evidence.knn = _fake_knn
    try:
        outcome = pipeline._process_memory(
            a, suppression=pipeline._load_suppression(),
            neighbor_k=10, auto_reject_remaining=5000,
        )
    finally:
        pipeline.db.evidence.knn = original_knn
    assert outcome["machine_cleared"] == 0, "deep check pairs are SKIPPED, not counted as cleared"
    sets = _conflict_member_sets(tools)
    assert {a, notify_peer} in sets, "notify pairs queue from any rank"
    assert {a, check_peer} not in sets, "check pairs beyond rank 3 must not generate"


def test_notify_pairs_survive_numeric_autoreject_suppression(tmp_path: Path) -> None:
    """Owner ⑩: scan_numeric_autoreject rows no longer suppress notify pairs
    (91/121 real notify pairs were refs-subset shadowed by them)."""
    tools = make_tools(tmp_path)
    a = _write(tools, "解蔽甲", "该功能包含缓存模块")
    b = _write(tools, "解蔽乙", "该功能不包含缓存模块")
    assert tools.wait_evidence_worker_drained(timeout=10)
    version_a = int(tools.db.get_memory(a)["version"] or 1)
    version_b = int(tools.db.get_memory(b)["version"] or 1)
    # Machine numeric-reject audit row covering the same pair@version refs.
    members = [
        {
            "memory_id": a, "version": version_a,
            "attribute_raw": None, "value_raw": None,
            "normalized_attribute": None, "normalized_value": None,
            "evidence_quote": "该功能包含缓存模块", "evidence_span": [0, 10],
            "content_hash": "x" * 64, "evidence_unit": 1,
            "direction": "deterministic", "prompt_version": None,
            "detector_version": CONFLICT_DETECTOR_VERSION,
        },
        {
            "memory_id": b, "version": version_b,
            "attribute_raw": None, "value_raw": None,
            "normalized_attribute": None, "normalized_value": None,
            "evidence_quote": "该功能不包含缓存模块", "evidence_span": [0, 11],
            "content_hash": "y" * 64, "evidence_unit": 1,
            "direction": "deterministic", "prompt_version": None,
            "detector_version": CONFLICT_DETECTOR_VERSION,
        },
    ]
    outcome = tools.db.record_conflict_group(
        workspace_canonical="ws", slot_key=None,
        members=members,
        value_groups=[], status="not_a_conflict",
        detector_version=CONFLICT_DETECTOR_VERSION,
        source="scan_numeric_autoreject", detection_reason="historical machine row",
    )
    assert outcome.get("outcome") in {"inserted", "deduped"}, outcome

    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert kick["ok"], kick
    sets = _conflict_member_sets(tools)
    assert {a, b} in sets, "the notify pair must NOT be silenced by the machine row"


def test_boot_clearance_migration_keeps_notify_and_value_shapes(tmp_path: Path) -> None:
    """The one-shot stock clearance uses the SAME classifier: notify rows and
    extractable-difference shapes survive; duplicates are voided with counts
    recorded, idempotent on second boot."""
    tools = make_tools(tmp_path)
    now = "2026-09-13T00:00:00+00:00"

    def _insert(hash_seed, severity, reason, quotes):
        evidence = json.dumps([
            {"evidence_quote": q} for q in quotes
        ], ensure_ascii=False)
        with tools.db.write_transaction() as conn:
            conn.execute(
                """INSERT INTO scan_queue(kind,workspace_canonical,status,candidate_key_hash,
                     member_versions,evidence,reason,severity,source,detail,created_at,updated_at)
                   VALUES('conflict','ws','pending',?, '[]', ?, ?, ?, 'scan_pipeline', NULL, ?, ?)""",
                (hash_seed, evidence, reason, severity, now, now),
            )

    _insert("b" * 64, "high", "polarity_changed", ["该功能包含缓存模块", "该功能不包含缓存模块"])
    _insert("c" * 64, "normal", "semantic_similarity_only",
            ["0.16.0 乙包十二个 commit 已合入主干并完成两轮复查", "0.16.0 乙包 12 个 commit 全部合入 main 并完成两轮复查"])
    _insert("d" * 64, "normal", "numeric_value_candidate",
            ["生产库使用 MySQL 5.7，延迟阈值 300ms", "生产库使用 MySQL 8.0，延迟阈值 500ms"])
    _insert("e" * 64, "normal", "numeric_value_candidate",
            ["部署压测报告已归档，共 500ms 采样窗口", "第二季度 OKR 评审通过，预算 5.7 万元"])

    # Burned at tools boot (empty queue); reset to simulate the upgrade boot.
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM migration_state WHERE key='scan_queue_difference_clearance_v1'")
    with tools.db.connection() as conn:
        applied = additive.ensure_additive_structures(conn)
    assert any("difference_clearance" in item for item in applied), applied

    with tools.db.connection() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT candidate_key_hash,status,decided_reason FROM scan_queue WHERE kind='conflict'"
        ).fetchall()]
    by_hash = {r["candidate_key_hash"]: r["status"] for r in rows}
    cleared = [r for r in rows if (r["decided_reason"] or "").startswith("difference clearance")]
    assert by_hash.get("b" * 64) == "pending", "notify must survive"
    assert by_hash.get("d" * 64) == "pending", "same-sentence numeric shape must survive"
    assert len(cleared) == 2, "duplicate paraphrase and cross-sentence numeric must clear"
    assert all(r["status"] == "voided" for r in cleared)
    with tools.db.connection() as conn:
        summary = conn.execute(
            "SELECT value FROM migration_state WHERE key='scan_queue_difference_clearance_v1'"
        ).fetchone()[0]
    assert "cleared=2" in summary and "kept=2" in summary, summary
    # Idempotent: second boot is a no-op.
    with tools.db.connection() as conn:
        applied2 = additive.ensure_additive_structures(conn)
    assert not any("difference_clearance" in item for item in applied2)
