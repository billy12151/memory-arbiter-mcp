"""Slow e2e: judge the WHOLE queue in one batched pass, then prove the
dispositions survive re-scans — same detector (watermark reset) AND a
detector bump (epoch re-arm, where the candidate hash misses and the
member-refs backstop is the only defence). The edit escape hatch is
asserted last so suppression cannot silently over-reach.

Cleanly skipped when no real embedding model is configured on the host.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter import db_generation, scan_pipeline
from memory_arbiter.db import evidence_store
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools

from test_scan_pipeline_e2e import _build, _teardown, _write, _page, _kick

pytestmark = pytest.mark.slow

MAX_PAGES = 20  # loop guard: a broken cursor must fail loud, not hang


def _pending_conflicts(tools: MemoryTools) -> int:
    with tools.db.connection() as conn:
        return int(conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE status='pending'"
        ).fetchone()[0])


def _drain(tools: MemoryTools) -> int:
    """Batch-judge every pending item (conflict groups group-level, internal
    rows one by one) until the queue reports backlog zero. Returns pages."""
    pages = 0
    while pages < MAX_PAGES:
        page = _page(tools, page_size=30)
        backlog = int(page.get("queue_backlog") or 0)
        if not page["items"] and backlog == 0:
            return pages
        decisions: list[dict] = []
        for item in page["items"]:
            if item["kind"] == "internal":
                decisions.append({
                    "kind": "internal", "internal_id": item["internal_id"],
                    "status": "dismissed", "reason": "e2e internal 驳回",
                })
            elif item.get("group_token") and item.get("pair_hashes"):
                decisions.append({
                    "group_token": item["group_token"], "status": "dismissed",
                    "reason": "e2e 组级驳回",
                    "pair_hashes": item["pair_hashes"],
                })
            else:
                for pair in item["pairs"]:
                    decisions.append({
                        "candidate_key_hash": pair["candidate_key_hash"],
                        "status": "dismissed", "reason": "e2e 单对驳回",
                    })
        if decisions:
            submit = tools.memory_repair("scan_queue", {
                "action": "submit", "decisions": decisions,
            })
            assert submit["ok"], submit
        pages += 1
    raise AssertionError("queue never drained — pagination loop suspected")


def test_judge_all_then_no_resurrection_e2e(tmp_path: Path) -> None:
    tools = _build(tmp_path)
    try:
        _run(tools)
    finally:
        _teardown(tools)


def _run(tools: MemoryTools) -> None:
    # ── seed: polarity pair + numeric pair + self-contradiction + fillers ─
    p1 = _write(tools, "演进甲", "该功能包含缓存模块。")
    p2 = _write(tools, "演进乙", "该功能不包含缓存模块。")
    _write(tools, "数值甲", "生产环境数据库端口设置为 5432。")
    _write(tools, "数值乙", "生产环境数据库端口设置为 5433。")
    _write(tools, "自相矛盾", "## 配置甲\n超时时间为 30 秒。\n## 配置乙\n超时时间为 60 秒。")
    for i in range(4):
        _write(tools, f"填充主题{i}", f"园区通行证流程第{i}条说明。")
    assert tools.wait_evidence_worker_drained(timeout=120)
    assert tools.wait_semantic_worker_drained(timeout=120)

    # ── 1) full scan, then judge EVERYTHING in batched passes ────────────
    kick = _kick(tools, max_memories=500, time_budget_s=180.0)
    assert kick["complete"] is True, kick
    assert _pending_conflicts(tools) > 0, "种子对必须入队"
    _drain(tools)
    page = _page(tools)
    assert page["items"] == [] and int(page.get("queue_backlog") or 0) == 0, "判完后队列必须清零"
    assert tools.db.internal_conflicts.list_pending() == [], "internal 也必须判完"

    # ── 2) no-resurrection, same detector: full watermark reset + kick ───
    tools.db.clear_all_scan_watermarks()
    kick = _kick(tools, max_memories=500, time_budget_s=180.0)
    assert kick["complete"] is True
    assert _pending_conflicts(tools) == 0, "同代重扫不得复活已判对"
    assert tools.db.internal_conflicts.list_pending() == [], "同代重扫不得复活已判 internal"

    # ── 3) no-resurrection, detector bump: hash misses, refs must hold ───
    bumped = "difference-classifier-e2e-next"
    monkey = pytest.MonkeyPatch()
    monkey.setattr(db_generation, "CONFLICT_DETECTOR_VERSION", bumped)
    monkey.setattr(scan_pipeline, "CONFLICT_DETECTOR_VERSION", bumped)
    monkey.setattr(evidence_store, "CONFLICT_DETECTOR_VERSION", bumped)
    try:
        db2 = MemoryDB(tools.settings)
        tools2 = MemoryTools(settings=tools.settings, db=db2)
        try:
            arm = tools2.db.meta.scan_epoch_arm()
            assert arm and arm["to"] == bumped, "换代 boot 必须布防"
            with tools2.db.connection() as conn:
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM scan_queue WHERE kind='conflict'").fetchone()[0]
            assert remaining == 0, "换代 boot 清旧 epoch 队列行"
            kick = _kick(tools2, max_memories=500, time_budget_s=180.0)
            assert kick["complete"] is True
            assert _pending_conflicts(tools2) == 0, (
                "换代后精确 hash 匹配失效，成员 refs 兜底必须挡住已判对"
                "——Agent 判过的不跨代重判"
            )
            assert tools2.db.internal_conflicts.list_pending() == [], \
                "internal exists 不含 detector，同须零复活"
            with tools2.db.connection() as conn:
                nac = conn.execute(
                    "SELECT COUNT(*) FROM conflicts WHERE status='not_a_conflict'"
                ).fetchone()[0]
            assert nac > 0, "防复活必须来自落表的 not_a_conflict（而非清库巧合）"
        finally:
            # Real-model teardown rule: the bumped instance loaded its own
            # embedder — unload it before the original tools takes over again.
            _teardown(tools2)
    finally:
        monkey.undo()

    # ── 4) escape hatch: an edit lifts suppression for THAT member only ──
    tools.memory("update", {
        "memory_id": p2, "new_content": "该功能不包含缓存模块，改为包含 Redis 缓存。",
        "reason": "e2e edit lift",
    })
    assert tools.wait_evidence_worker_drained(timeout=120)
    assert tools.wait_semantic_worker_drained(timeout=120)
    kick = _kick(tools, max_memories=500, time_budget_s=180.0)
    assert kick["complete"] is True
    with tools.db.connection() as conn:
        rows = [json.loads(r[0]) for r in conn.execute(
            "SELECT member_versions FROM scan_queue WHERE kind='conflict' "
            "AND status='pending'").fetchall()]
    edited_members = [m for members in rows for m in members if m["memory_id"] == p2]
    assert edited_members, "编辑抬版本后该成员的新身份必须重新入队"
    assert all(int(m["version"]) >= 2 for m in edited_members), \
        "重入队必须是新版本身份——旧版本对的抑制不被编辑解除"
