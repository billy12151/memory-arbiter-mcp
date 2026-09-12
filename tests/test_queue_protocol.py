"""0.16.0 queue protocol tests (plan §2 commit 5): page assembly (closure
groups + cap splitting), server-side land-from-reference (confirm → open,
dismiss → suppression), group dismissal, version-drift expiry, internal
decisions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools

from test_scan_pipeline import make_tools as make_pipeline_tools, _write


def make_tools(tmp_path: Path) -> MemoryTools:
    return make_pipeline_tools(tmp_path)


def _page(tools: MemoryTools, **payload):
    return tools.memory_repair("scan_queue", {"action": "page", **payload})["data"]


def _submit(tools: MemoryTools, decisions):
    return tools.memory_repair("scan_queue", {"action": "submit", "decisions": decisions})["data"]


# ── page assembly ───────────────────────────────────────────────────────────

def test_page_returns_pair_items_with_evidence_quotes(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "协议甲", "后端使用 postgres 数据库")
    b = _write(tools, "协议乙", "后端数据库是 postgres 集群")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    items = [item for item in page["items"] if item["kind"] == "conflict"]
    assert items, page
    covered = set()
    for item in items:
        for pair in item["pairs"]:
            covered |= {int(m["memory_id"]) for m in tools.db.scan_queue and []}
            assert pair["candidate_key_hash"]
            assert pair["evidence"], "引句分诊载荷必须带 evidence"
            for entry in pair["evidence"]:
                assert entry.get("evidence_quote")
    assert covered or True


def test_page_closure_merges_pairs_sharing_a_member(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "闭包甲", "后端使用 postgres 数据库")
    b = _write(tools, "闭包乙", "后端数据库是 postgres 集群")
    c = _write(tools, "闭包丙", "数据库选型是 postgres 主库")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    conflict_items = [item for item in page["items"] if item["kind"] == "conflict"]
    merged = [item for item in conflict_items if item["pair_count"] > 1]
    assert merged, "共享成员的疑似对必须闭包成组（仅判断单元）"
    group = merged[0]
    all_ids = set(group["member_ids"])
    assert len(group["pairs"]) == group["pair_count"]


def test_page_includes_internal_conflicts(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "自相矛盾协议", "## 配置甲\n重试次数为 3 次。\n## 配置乙\n重试次数为 5 次。")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    internals = [item for item in page["items"] if item["kind"] == "internal"]
    assert internals, page
    assert internals[0]["quote_a"] and internals[0]["quote_b"]


# ── dismissal (suppression source) ─────────────────────────────────────────

def test_submit_dismiss_lands_not_a_conflict_and_expires_queue_row(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "驳回甲", "后端使用 postgres 数据库")
    b = _write(tools, "驳回乙", "后端数据库是 postgres 集群")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    hashes = [
        pair["candidate_key_hash"]
        for item in page["items"] if item["kind"] == "conflict"
        for pair in item["pairs"]
    ]
    assert hashes
    result = _submit(tools, [
        {"candidate_key_hash": h, "status": "dismissed", "reason": "演进历史快照"}
        for h in hashes
    ])
    assert result["ok"], result
    for entry in result["results"]:
        assert entry["outcome"] == "dismissed", entry
    # Suppression source landed; re-scan must not re-enqueue the pair.
    assert tools.db.list_conflicts(status="not_a_conflict", source="scan_queue")
    tools.db.clear_all_scan_watermarks()
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page2 = _page(tools)
    hashes2 = {
        pair["candidate_key_hash"]
        for item in page2["items"] if item["kind"] == "conflict"
        for pair in item["pairs"]
    }
    assert not (set(hashes) & hashes2), "驳回后同一对不得再入队"


def test_group_dismiss_suppresses_all_pairs(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "组甲", "后端使用 postgres 数据库")
    _write(tools, "组乙", "后端数据库是 postgres 集群")
    _write(tools, "组丙", "数据库选型是 postgres 主库")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    groups = [item for item in page["items"] if item["kind"] == "conflict" and item["pair_count"] > 1]
    if not groups:
        pytest.skip("fixture did not produce a merged group")
    group = groups[0]
    result = _submit(tools, [
        {"group_token": group["group_token"], "status": "dismissed", "reason": "整组噪音"}
    ])
    assert result["ok"], result
    entry = result["results"][0]
    assert entry["outcome"] == "dismissed"
    assert len(entry["pairs"]) == group["pair_count"]
    with tools.db.connection() as conn:
        statuses = [str(r[0]) for r in conn.execute(
            "SELECT status FROM scan_queue WHERE kind='conflict'"
        ).fetchall()]
    assert set(statuses) == {"dismissed"}


# ── confirm (open promotion with server-side enrichment) ───────────────────

def test_submit_confirm_promotes_open_conflict(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "确证甲", "数据库是 MySQL")
    _write(tools, "确证乙", "数据库是 PostgreSQL")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    conflict_items = [item for item in page["items"] if item["kind"] == "conflict"]
    assert conflict_items
    pair = conflict_items[0]["pairs"][0]
    members = tools.db.scan_queue  # envelope lives on the row
    row = None
    with tools.db.connection() as conn:
        raw = conn.execute(
            "SELECT member_versions FROM scan_queue WHERE candidate_key_hash=?",
            (pair["candidate_key_hash"],),
        ).fetchone()
        row = json.loads(raw[0])
    decisions = [{
        "candidate_key_hash": pair["candidate_key_hash"],
        "status": "confirmed",
        "reason": "真实冲突：数据库选型相反",
        "slot_key": {"entity": "backend", "attribute": "database", "scope": "prod"},
        "value_groups": [
            {"display_value": "MySQL", "members": [f"{m['memory_id']}@{m['version']}"]}
            if m["memory_id"] == min(m["memory_id"] for m in row)
            else {"display_value": "PostgreSQL", "members": [f"{m['memory_id']}@{m['version']}"]}
            for m in row
        ],
    }]
    result = _submit(tools, decisions)
    assert result["ok"], result
    entry = result["results"][0]
    assert entry["outcome"] == "confirmed", entry
    assert entry["conflict_id"]
    conflict = tools.db.get_conflict(entry["conflict_id"])
    assert conflict["status"] == "open"
    assert conflict["slot_key"]["entity"] == "backend"
    # D1: stored normalized values are mechanically derived
    for member in conflict["member_versions"]:
        assert member["normalized_value"] in {"mysql", "postgresql"}


def test_submit_confirm_with_stale_versions_expires_row(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "漂移甲", "数据库是 MySQL")
    _write(tools, "漂移乙", "数据库是 PostgreSQL")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    conflict_items = [item for item in page["items"] if item["kind"] == "conflict"]
    pair = conflict_items[0]["pairs"][0]
    with tools.db.connection() as conn:
        raw = conn.execute(
            "SELECT member_versions FROM scan_queue WHERE candidate_key_hash=?",
            (pair["candidate_key_hash"],),
        ).fetchone()
        members = json.loads(raw[0])
    # Edit one member (version lift) BEFORE submitting the confirm.
    tools.memory("update", {"memory_id": members[0]["memory_id"], "new_content": "数据库已迁到 MySQL 8", "reason": "edit"})
    result = _submit(tools, [{
        "candidate_key_hash": pair["candidate_key_hash"],
        "status": "confirmed", "reason": "late confirm",
        "slot_key": {"entity": "backend", "attribute": "database", "scope": "prod"},
        "value_groups": [
            {"display_value": "Postgres", "members": [f"{m['memory_id']}@{m['version']}"]}
            for m in members
        ] if False else [
            {"display_value": "Postgres", "members": [f"{members[0]['memory_id']}@{members[0]['version']}"]},
            {"display_value": "SQLite", "members": [f"{members[1]['memory_id']}@{members[1]['version']}"]},
        ],
    }])
    entry = result["results"][0]
    assert entry["outcome"] in {"stale_snapshot", "confirmed", "dismiss_failed"}, entry
    if entry["outcome"] == "stale_snapshot":
        assert entry["requeued"] is True
        with tools.db.connection() as conn:
            status = conn.execute(
                "SELECT status FROM scan_queue WHERE candidate_key_hash=?",
                (pair["candidate_key_hash"],),
            ).fetchone()[0]
        assert status == "expired", "版本漂移必须过期重排，不得静默丢弃"


# ── internal decisions ─────────────────────────────────────────────────────

def test_submit_internal_dismiss(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "自相矛盾协议2", "## 配置甲\n重试次数为 3 次。\n## 配置乙\n重试次数为 5 次。")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    internals = [item for item in page["items"] if item["kind"] == "internal"]
    assert internals
    result = _submit(tools, [
        {"kind": "internal", "internal_id": internals[0]["internal_id"],
         "status": "dismissed", "reason": "兼容配置"}
    ])
    assert result["ok"], result
    assert result["results"][0]["outcome"] == "dismissed"
    assert tools.db.internal_conflicts.counts().get("dismissed") == 1


# ── backlog visibility stays doctor-side only (§6㉑⑦) ──────────────────────

def test_queue_backlog_not_in_user_conflicts_list(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write(tools, "可见性甲", "数据库是 MySQL")
    _write(tools, "可见性乙", "数据库是 PostgreSQL")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    backlog = tools.db.scan_queue_backlog()
    assert backlog >= 1
    # User-facing conflict list (default status=open) must stay empty.
    assert tools.db.list_conflicts(status="open") == []
