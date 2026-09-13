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
    # 0.16.4: numeric shape (same-sentence two-values) — the only reliable
    # cross-memory fixture (similarity-only pairs machine-clear; polarity
    # pairs are the excluded evolution domain).
    a = _write(tools, "协议甲", "重试次数为 3 次")
    b = _write(tools, "协议乙", "重试次数为 5 次")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    items = [item for item in page["items"] if item["kind"] == "conflict"]
    assert items, page
    for item in items:
        for pair in item["pairs"]:
            assert pair["candidate_key_hash"]
            assert pair["evidence"], "引句分诊载荷必须带 evidence"
            for entry in pair["evidence"]:
                assert entry.get("evidence_quote")


def test_page_closure_merges_pairs_sharing_a_member(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # 0.16.4: three same-sentence numeric values → three pairwise suspects
    # sharing members → closure into one group.
    a = _write(tools, "闭包甲", "重试次数为 3 次")
    b = _write(tools, "闭包乙", "重试次数为 5 次")
    c = _write(tools, "闭包丙", "重试次数为 7 次")
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
    # 0.16.4 §3: one aggregated item per MEMORY (kind internal_memory).
    internals = [item for item in page["items"] if item["kind"] == "internal_memory"]
    assert internals, page
    item = internals[0]
    assert item["pair_count"] >= 1
    assert item["pairs"], "聚合页必须带 pairs 预览"
    assert item["pairs"][0]["quote_a"] and item["pairs"][0]["quote_b"]
    assert item["pairs"][0]["internal_id"]
    assert item["reasons_summary"], "聚合页必须带 reason 分布"


# ── dismissal (suppression source) ─────────────────────────────────────────

def test_submit_dismiss_lands_not_a_conflict_and_expires_queue_row(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # 0.16.4: numeric shape — polarity fixtures no longer queue (evolution
    # domain); queue-source dismissal suppression is unaffected by the ⑩
    # exclusion, which filters only scan_numeric_autoreject rows.
    a = _write(tools, "驳回甲", "重试次数为 3 次")
    b = _write(tools, "驳回乙", "重试次数为 5 次")
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
    assert entry["outcome"] == "stale_snapshot", entry
    assert entry["requeued"] is True
    with tools.db.connection() as conn:
        status = conn.execute(
            "SELECT status FROM scan_queue WHERE candidate_key_hash=?",
            (pair["candidate_key_hash"],),
        ).fetchone()[0]
    assert status == "expired", "版本漂移必须过期重排，不得静默丢弃"


# ── internal decisions ─────────────────────────────────────────────────────

def test_submit_internal_dismiss(tmp_path: Path) -> None:
    """Per-row channel (downward compatible): the aggregated page carries
    internal_id per pair, the per-row submit stays operational."""
    tools = make_tools(tmp_path)
    _write(tools, "自相矛盾协议2", "## 配置甲\n重试次数为 3 次。\n## 配置乙\n重试次数为 5 次。")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    internals = [item for item in page["items"] if item["kind"] == "internal_memory"]
    assert internals
    result = _submit(tools, [
        {"kind": "internal", "internal_id": internals[0]["pairs"][0]["internal_id"],
         "status": "dismissed", "reason": "兼容配置"}
    ])
    assert result["ok"], result
    assert result["results"][0]["outcome"] == "dismissed"
    assert tools.db.internal_conflicts.counts().get("dismissed") == 1


# ── 0.16.4 §3: memory-level aggregation + batched dispositions ─────────────

_FIVE_VALUES = (
    "## 配置一\n超时时间为 30 秒。\n## 配置二\n超时时间为 40 秒。\n"
    "## 配置三\n超时时间为 50 秒。\n## 配置四\n超时时间为 60 秒。\n"
    "## 配置五\n超时时间为 70 秒。"
)


def _internal_page_item(tools) -> dict:
    page = _page(tools)
    internals = [item for item in page["items"] if item["kind"] == "internal_memory"]
    assert internals, page
    return internals[0]


def test_internal_memory_page_caps_pairs_preview(tmp_path: Path) -> None:
    """Five value sections → 10 pairs; the page shows a preview of 8 with
    the FULL pair_count and the reason distribution."""
    tools = make_tools(tmp_path)
    _write(tools, "五值矛盾", _FIVE_VALUES)
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    item = _internal_page_item(tools)
    assert item["pair_count"] >= 10, item["pair_count"]
    assert len(item["pairs"]) == 8, len(item["pairs"])
    assert sum(item["reasons_summary"].values()) == item["pair_count"]


def test_internal_memory_dismiss_clears_whole_memory_no_resurrect(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "五值清空", _FIVE_VALUES)
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    item = _internal_page_item(tools)
    assert item["pair_count"] > 8  # guard applies; expanded=true passes below
    result = _submit(tools, [{
        "kind": "internal_memory", "memory_id": mid,
        "status": "dismissed", "reason": "枚举值对照非矛盾", "expanded": True,
    }])
    assert result["ok"], result
    entry = result["results"][0]
    assert entry["outcome"] == "dismissed", entry
    assert entry["updated"] == entry["pair_count"] > 0
    assert tools.db.internal_conflicts.list_pending() == [], "整记忆清空"
    with tools.db.connection() as conn:
        dismissed = conn.execute(
            "SELECT COUNT(*) FROM internal_conflicts WHERE memory_id=? AND status='dismissed'",
            (mid,),
        ).fetchone()[0]
    assert dismissed == entry["pair_count"]
    # exists() blocks the scan re-examination (no resurrection).
    tools.db.clear_all_scan_watermarks()
    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert kick["ok"], kick
    assert not any(r for r in tools.db.internal_conflicts.list_pending() if r["memory_id"] == mid)


def test_internal_memory_expanded_guard(tmp_path: Path) -> None:
    """pair_count beyond the preview cap: a memory-level dismissal without
    expanded=true is rejected with the read-first instruction; the guard is
    waived for resolved and for within-cap dismissals."""
    tools = make_tools(tmp_path)
    mid = _write(tools, "五值守卫", _FIVE_VALUES)
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    item = _internal_page_item(tools)
    assert item["pair_count"] > 8
    refused = _submit(tools, [{
        "kind": "internal_memory", "memory_id": mid,
        "status": "dismissed", "reason": "盲判尝试",
    }])
    assert refused["ok"] is False, refused
    assert refused["results"][0]["outcome"] == "invalid_input"
    assert "expanded=true" in refused["results"][0]["error"]
    assert tools.db.internal_conflicts.list_pending(), "拒后行不得被翻"
    # 0.16.4 review P2: loosely-typed booleans cannot waive the guard — a
    # "false" STRING is truthy under bool() but must not count as expanded.
    string_false = _submit(tools, [{
        "kind": "internal_memory", "memory_id": mid,
        "status": "dismissed", "reason": "字符串假值", "expanded": "false",
    }])
    assert string_false["ok"] is False, string_false
    assert string_false["results"][0]["outcome"] == "invalid_input"
    string_true = _submit(tools, [{
        "kind": "internal_memory", "memory_id": mid,
        "status": "dismissed", "reason": "字符串真值", "expanded": "true",
    }])
    assert string_true["ok"], string_true
    # resolved carries no guard (misuse is auditable, not silent).
    ok = _submit(tools, [{
        "kind": "internal_memory", "memory_id": mid,
        "status": "resolved", "reason": "确认真矛盾",
    }])
    assert ok["ok"], ok


def test_internal_memory_version_guard_leaves_drifted_rows(tmp_path: Path) -> None:
    """A version lift between page and submit (raw lift — no write-time
    re-examination): drifted pending rows stay for expire_stale; the
    memory-level UPDATE only touches current-version rows."""
    tools = make_tools(tmp_path)
    mid = _write(tools, "五值漂移", _FIVE_VALUES)
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert _internal_page_item(tools)
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET version=version+1 WHERE id=?", (mid,))
    result = _submit(tools, [{
        "kind": "internal_memory", "memory_id": mid,
        "status": "dismissed", "reason": "迟到的判定",
    }])
    entry = result["results"][0]
    assert entry["outcome"] == "stale_snapshot", entry
    with tools.db.connection() as conn:
        drifted = conn.execute(
            "SELECT COUNT(*) FROM internal_conflicts WHERE memory_id=? AND status='pending'",
            (mid,),
        ).fetchone()[0]
    assert drifted > 0, "漂移行留给 expire_stale，不混入 decided 口径"


def test_internal_memory_not_found_and_mixed_batch(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "混合甲", "重试次数为 3 次")
    b = _write(tools, "混合乙", "重试次数为 5 次")
    mid = _write(tools, "混合矛盾", "## 配置甲\n重试次数为 3 次。\n## 配置乙\n重试次数为 5 次。")
    tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    conflict_item = next(i for i in page["items"] if i["kind"] == "conflict")
    result = _submit(tools, [
        {"kind": "internal_memory", "memory_id": mid, "status": "dismissed", "reason": "批量"},
        {"candidate_key_hash": conflict_item["pairs"][0]["candidate_key_hash"],
         "status": "dismissed", "reason": "混合批"},
        {"kind": "internal_memory", "memory_id": 999999, "status": "dismissed", "reason": "无此记忆"},
    ])
    assert result["ok"], result
    outcomes = [r["outcome"] for r in result["results"]]
    assert outcomes[0] == "dismissed"
    assert outcomes[1] == "dismissed"
    assert outcomes[2] == "not_found"


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
