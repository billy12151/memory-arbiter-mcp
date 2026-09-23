"""0.16.0 queue protocol tests (plan §2 commit 5): page assembly (closure
groups + cap splitting), server-side land-from-reference (confirm → open,
dismiss → suppression), group dismissal, version-drift expiry, internal
decisions."""
from __future__ import annotations

import hashlib
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    item = _internal_page_item(tools)
    assert item["pair_count"] >= 10, item["pair_count"]
    assert len(item["pairs"]) == 8, len(item["pairs"])
    assert sum(item["reasons_summary"].values()) == item["pair_count"]


def test_internal_memory_dismiss_clears_whole_memory_no_resurrect(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "五值清空", _FIVE_VALUES)
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
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
    tools.wait_semantic_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    backlog = tools.db.scan_queue_backlog()
    assert backlog >= 1
    # User-facing conflict list (default status=open) must stay empty.
    assert tools.db.list_conflicts(status="open") == []


# ── 0.16.4 live-judgment review: page byte budget + no-strand merge ─────────

def test_page_caps_group_pair_hashes_preview(tmp_path: Path) -> None:
    """The hash list is a byte-budget PREVIEW; pair_count stays full."""
    from memory_arbiter.queue_protocol import GROUP_HASHES_CAP

    tools = make_tools(tmp_path)
    for i in range(7):
        _write(tools, f"帽甲{i}", f"重试次数为 {3 + i} 次")
    tools.wait_semantic_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 20})
    page = _page(tools)
    groups = [i for i in page["items"] if i["kind"] == "conflict" and i["pair_count"] > GROUP_HASHES_CAP]
    assert groups, "7 成员组必须闭包成 >5 对的组"
    item = groups[0]
    assert len(item["pair_hashes"]) == GROUP_HASHES_CAP
    assert item["pair_count"] > GROUP_HASHES_CAP
    assert "preview" in page["instruction"]


def test_partial_hashes_plus_token_cannot_strand_group(tmp_path: Path) -> None:
    """Caller holds only the capped preview hashes; token re-assembly merges
    the rest of the group — zero stranding."""
    tools = make_tools(tmp_path)
    for i in range(6):
        _write(tools, f"合甲{i}", f"超时时间为 {10 + i} 秒")
    tools.wait_semantic_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 20})
    page = _page(tools)
    group = next(i for i in page["items"] if i["kind"] == "conflict" and i["pair_count"] > 1)
    result = _submit(tools, [{
        "group_token": group["group_token"], "status": "dismissed",
        "reason": "仅带预览 hash 的组级驳回", "pair_hashes": group["pair_hashes"][:1],
    }])
    assert result["ok"], result
    entry = result["results"][0]
    assert entry["outcome"] == "dismissed"
    assert len(entry["pairs"]) == group["pair_count"], "必须清完整组（合并补齐）"
    with tools.db.connection() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE status='pending' AND kind='conflict'"
        ).fetchone()[0]
    assert pending == 0, "组残余不得搁浅"


def test_token_only_deep_group_full_retrieval(tmp_path: Path, monkeypatch) -> None:
    """Depth beyond one assembly window: paged full retrieval still finds the
    complete closure for a token-only disposition."""
    import memory_arbiter.queue_protocol as qp

    tools = make_tools(tmp_path)
    for i in range(5):
        _write(tools, f"深甲{i}", f"刷新次数为 {20 + i} 次")
    tools.wait_semantic_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 20})
    page = _page(tools)
    group = next(i for i in page["items"] if i["kind"] == "conflict" and i["pair_count"] > 1)
    # Shrink the fetch window so the group's rows exceed one window: the
    # paged full retrieval must still assemble the complete closure.
    monkeypatch.setattr(qp, "ASSEMBLY_WINDOW", 1)
    result = _submit(tools, [{
        "group_token": group["group_token"], "status": "dismissed", "reason": "深组 token-only",
    }])
    assert result["ok"], result
    entry = result["results"][0]
    assert entry["outcome"] == "dismissed"
    assert len(entry["pairs"]) == group["pair_count"], "窗口外仍须清完整组"
    with tools.db.connection() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE status='pending' AND kind='conflict'"
        ).fetchone()[0]
    assert pending == 0


def test_submit_backlog_counts_internal_rows(tmp_path: Path) -> None:
    """submit()'s queue_backlog matches page()'s semantics (internal rows
    included) — clearing internal rows must move the number."""
    tools = make_tools(tmp_path)
    mid = _write(tools, "口径矛盾", "## 配置甲\n重试次数为 3 次。\n## 配置乙\n重试次数为 5 次。")
    tools.wait_semantic_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    page = _page(tools)
    page_backlog = int(page.get("queue_backlog") or 0)
    assert page_backlog >= 1
    internal_rows = tools.db.internal_conflicts.pending_pair_count(mid)
    assert internal_rows >= 1
    result = _submit(tools, [{
        "kind": "internal_memory", "memory_id": mid, "status": "dismissed", "reason": "口径",
    }])
    assert result["ok"], result
    # 与 page() 同口径：internal 清掉的行必须反映在 submit 返回的 backlog 里
    assert int(result["queue_backlog"]) == page_backlog - internal_rows, (
        result["queue_backlog"], page_backlog, internal_rows
    )


# ── 0.17.1 workspace dismiss 持久化（门 B 的落点） ────────────────────────────

def _workspace_clan(tools: MemoryTools) -> int:
    """9 postgres in dbpgsql + 1 postgres in ws (the suspect). Returns the ws id.

    内容必须逐条不同——content_sha 去重会把同内容写入折叠成既有 id。"""
    for i in range(9):
        _write(tools, f"族甲{i}", f"后端使用 postgres 主库，兄弟 {i}", workspace="dbpgsql")
    mid = _write(tools, "错位主题", "后端使用 postgres 主库，独苗", workspace="ws")
    assert tools.wait_semantic_worker_drained(timeout=10)
    return mid


def _enqueue_workspace_candidate(
    tools: MemoryTools, mid: int, *, own: str = "ws", suspected: str = "dbpgsql", tag: str,
) -> None:
    """Pending workspace row with the FULL detail envelope — 门 B/清场都解析
    (current_workspace, suspected_workspace)，detail={} 的构造器会让断言假绿。"""
    outcome = tools.db.scan_queue.enqueue(
        kind="workspace",
        workspace_canonical=own,
        candidate_key_hash=hashlib.sha256(f"ws-dismiss:{tag}".encode("utf-8")).hexdigest(),
        member_versions=[{"memory_id": mid, "version": 1}],
        evidence=[],
        reason="vector vote 9/9 -> 'dbpgsql'",
        severity="normal",
        source="test",
        detail={"current_workspace": own, "suspected_workspace": suspected},
    )
    assert outcome.get("outcome") == "queued", outcome


def _workspace_row_count(tools: MemoryTools) -> int:
    with tools.db.connection() as conn:
        return int(conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE kind='workspace'"
        ).fetchone()[0])


def test_dismiss_records_durable_row(tmp_path: Path) -> None:
    """Agent 显式 dismiss → workspace_dismissals 落 (mid,version,suspected) +
    reason（同事务于队列状态翻转）；hint/confirmed 分支不落。"""
    tools = make_tools(tmp_path)
    mid = _workspace_clan(tools)
    _enqueue_workspace_candidate(tools, mid, tag="durable")

    result = _submit(tools, [
        {"kind": "workspace", "memory_id": mid, "status": "dismissed", "reason": "闲聊留在 default"},
    ])
    assert result["ok"], result
    assert result["results"][0]["outcome"] == "dismissed"
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT memory_id, version, suspected_workspace, reason FROM workspace_dismissals"
        ).fetchall()
        queue_status = conn.execute(
            "SELECT status FROM scan_queue WHERE kind='workspace'"
        ).fetchall()
    assert [(int(r[0]), int(r[1]), str(r[2])) for r in rows] == [(mid, 1, "dbpgsql")]
    assert str(rows[0]["reason"]) == "闲聊留在 default"
    assert [str(r[0]) for r in queue_status] == ["dismissed"]


def test_dismiss_survives_queue_wipe(tmp_path: Path) -> None:
    """启动 purge / 检测器换代整表 DELETE 清空 scan_queue 后，同身份不再
    入队（门 B 幸存——队列是工作台，workspace_dismissals 才是决策记录）。"""
    tools = make_tools(tmp_path)
    mid = _workspace_clan(tools)

    # 基线：无 dismissal 时该身份确实会入队（防测试真空）。
    tools.db.clear_all_scan_watermarks()
    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 20})
    assert kick["ok"], kick
    assert _workspace_row_count(tools) == 1, "无 dismissal 时基线必须入队"

    result = _submit(tools, [
        {"kind": "workspace", "memory_id": mid, "status": "dismissed", "reason": "一代噪音"},
    ])
    assert result["results"][0]["outcome"] == "dismissed"
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM scan_queue")
    tools.db.clear_all_scan_watermarks()

    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 20})
    assert kick["ok"], kick
    assert _workspace_row_count(tools) == 0, "队列清空后同身份不得复发"


def test_version_bump_reopens(tmp_path: Path) -> None:
    """门 B 版本钉死：编辑 bump version 后同桶豁免失效，提议重新入队
    （与 conflict 类 stale_snapshot「版本漂移重开」语义一致）。"""
    tools = make_tools(tmp_path)
    mid = _workspace_clan(tools)
    _enqueue_workspace_candidate(tools, mid, tag="reopen")
    result = _submit(tools, [
        {"kind": "workspace", "memory_id": mid, "status": "dismissed", "reason": "先压一轮"},
    ])
    assert result["results"][0]["outcome"] == "dismissed"

    edited = tools.memory_edit(mid, new_content="后端使用 postgres 主库，已修订口径")
    assert edited.get("ok"), edited
    assert tools.wait_semantic_worker_drained(timeout=10)
    assert int(tools.db.get_memory(mid)["version"]) == 2
    tools.db.clear_all_scan_watermarks()

    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 20})
    assert kick["ok"], kick
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT status FROM scan_queue WHERE kind='workspace' AND status='pending'"
        ).fetchall()
    assert len(rows) == 1, "version+1 后同桶提议必须重新入队"


def test_hint_branches_leave_no_durable_row(tmp_path: Path) -> None:
    """R1 补测（R2-3 语义钉）：protected / multi_family hint 分支只翻队列
    状态，绝不落 workspace_dismissals——hint 语义是「请用户处置」，一旦落
    持久表就变成永久豁免、治理入口静默关闭。"""
    tools = make_tools(tmp_path)
    # protected 分支：own ∈ PROTECTED_WORKSPACES，confirmed 提交在 protected
    # 检查（current/target 任一受保护）即降级 hint，走不到投票门。
    protected_mid = _write(tools, "受保护桶条目", "mema-twin 专属内容，独苗", workspace="mema-twin")
    _enqueue_workspace_candidate(tools, protected_mid, own="mema-twin",
                                 suspected="dbpgsql", tag="hint-protected")
    result = _submit(tools, [
        {"kind": "workspace", "memory_id": protected_mid, "status": "confirmed",
         "target_workspace": "dbpgsql", "conf": 0.9, "reason": "投票门不会执行"},
    ])
    assert result["results"][0]["outcome"] == "protected_bucket_hint"

    # multi_family 分支（E7-4）：subject/tags 提及 ≥2 个已注册项目族。
    # 注：FakeEmbedder 下未注册桶名会被写时向量归并进既有桶（见 cap 测试
    # 注释），两个族与跨族记忆的桶都必须经 move_memories_workspace 直写
    # 才能得到确定 canonical。
    _write(tools, "族甲", "dbpgsql 后端栈内容，兄弟 0", workspace="dbpgsql")
    fw_id = _write(tools, "族乙", "前端栈内容，兄弟 0", workspace="dbpgsql")
    moved = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [fw_id], "new_workspace": "frontweb",
        "reason": "seed second family", "authorized": True,
    })
    assert moved["ok"], moved
    multi_mid = _write(tools, "跨族条目", "dbpgsql 与 frontweb 的跨项目内容，独苗",
                       workspace="dbpgsql", tags=["dbpgsql", "frontweb"])
    _enqueue_workspace_candidate(tools, multi_mid, own="dbpgsql", suspected="frontweb",
                                 tag="hint-multi")
    result = _submit(tools, [
        {"kind": "workspace", "memory_id": multi_mid, "status": "confirmed",
         "target_workspace": "frontweb", "conf": 0.9, "reason": "跨族提示"},
    ])
    assert result["results"][0]["outcome"] == "multi_family_hint"

    with tools.db.connection() as conn:
        durable = int(conn.execute("SELECT COUNT(*) FROM workspace_dismissals").fetchone()[0])
        queue_status = conn.execute(
            "SELECT status FROM scan_queue WHERE kind='workspace'"
        ).fetchall()
    assert durable == 0, "hint 分支不得落持久豁免"
    assert queue_status, "hint 分支仍须了结 pending 行"
    assert all(str(r[0]) == "dismissed" for r in queue_status)


def test_dismiss_pins_current_version_after_edit(tmp_path: Path) -> None:
    """R2 补测（旧钉复发窗口实锤）：edit 先于 dismiss 时 durable 身份必须钉
    当前版本——行钉 v1/记忆 v2 时按行钉落表，门 B 按当前版本比较立即不匹配，
    agent 刚处置的噪音下轮原样复发（重开应由 dismiss 之后的编辑触发）。"""
    tools = make_tools(tmp_path)
    mid = _workspace_clan(tools)
    _enqueue_workspace_candidate(tools, mid, tag="stale-pin")
    edited = tools.memory_edit(mid, new_content="后端使用 postgres 主库，已修订口径")
    assert edited.get("ok"), edited
    assert tools.wait_semantic_worker_drained(timeout=10)
    assert int(tools.db.get_memory(mid)["version"]) == 2

    result = _submit(tools, [
        {"kind": "workspace", "memory_id": mid, "status": "dismissed", "reason": "编辑后噪音"},
    ])
    assert result["results"][0]["outcome"] == "dismissed"
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT memory_id, version, suspected_workspace FROM workspace_dismissals"
        ).fetchall()
    assert [(int(r[0]), int(r[1]), str(r[2])) for r in rows] == [(mid, 2, "dbpgsql")]

    # 整表 DELETE 模拟启动 purge/换代后重扫：钉当前版本 → 同身份不复发。
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM scan_queue")
    tools.db.clear_all_scan_watermarks()
    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 20})
    assert kick["ok"], kick
    assert _workspace_row_count(tools) == 0, "edit-先于-dismiss 的处置不得一轮后复发"


def test_duplicate_dismiss_idempotent(tmp_path: Path) -> None:
    """§5「重复 dismiss | INSERT OR IGNORE 幂等」补测：同身份两个 pending 行
    （fail-open 窗口/存量行，不同 candidate_key_hash）一次 dismiss——两行都
    了结，durable 表只落一行（变异实锤 plain INSERT 会假绿 + 第二行永 pending）。"""
    tools = make_tools(tmp_path)
    mid = _workspace_clan(tools)
    _enqueue_workspace_candidate(tools, mid, tag="dup-a")
    outcome = tools.db.scan_queue.enqueue(
        kind="workspace",
        workspace_canonical="ws",
        candidate_key_hash=hashlib.sha256(b"ws-dismiss:dup-b").hexdigest(),
        member_versions=[{"memory_id": mid, "version": 1}],
        evidence=[], reason="vector vote 9/9 -> 'dbpgsql'", severity="normal",
        source="test",
        detail={"current_workspace": "ws", "suspected_workspace": "dbpgsql"},
    )
    assert outcome.get("outcome") == "queued", outcome

    result = _submit(tools, [
        {"kind": "workspace", "memory_id": mid, "status": "dismissed", "reason": "重复处置"},
    ])
    assert result["results"][0]["outcome"] == "dismissed"
    with tools.db.connection() as conn:
        durable = int(conn.execute("SELECT COUNT(*) FROM workspace_dismissals").fetchone()[0])
        statuses = conn.execute(
            "SELECT status FROM scan_queue WHERE kind='workspace'"
        ).fetchall()
    assert durable == 1
    assert len(statuses) == 2 and all(str(r[0]) == "dismissed" for r in statuses)


def test_dismiss_falls_back_when_dismissals_table_missing(tmp_path: Path) -> None:
    """R2 F1（P1）回归：additive 建表被跳过（只读/损坏库 boot 只降级 warning，
    4a07cdc 实锤过同类事故）时，durable INSERT 不再连坐队列翻转——except 回退
    旧纯 UPDATE 路径保底翻转（行不卡 §九）；durable 豁免丢失至多再 dismiss
    一次补回。"""
    tools = make_tools(tmp_path)
    mid = _workspace_clan(tools)
    _enqueue_workspace_candidate(tools, mid, tag="no-table")
    with tools.db.write_transaction() as conn:
        conn.execute("DROP TABLE workspace_dismissals")

    result = _submit(tools, [
        {"kind": "workspace", "memory_id": mid, "status": "dismissed", "reason": "表缺失保底"},
    ])
    assert result["results"][0]["outcome"] == "dismissed"
    with tools.db.connection() as conn:
        statuses = conn.execute(
            "SELECT status FROM scan_queue WHERE kind='workspace'"
        ).fetchall()
    assert [str(r[0]) for r in statuses] == ["dismissed"], "队列翻转必须落地（旧语义保底）"


def test_load_workspace_dismissals_bad_row_fail_open(tmp_path: Path) -> None:
    """R1 修复①的回归钉：dismissal 表内坏行（SQLite 灵活类型允许 TEXT 存入
    INTEGER 列）不得让索引构建抛 TypeError/ValueError 直上无 try 兜底的
    weekly 主循环——fail-open 空索引，扫描照跑（回到现状噪音方向）。"""
    tools = make_tools(tmp_path)
    mid = _workspace_clan(tools)
    _enqueue_workspace_candidate(tools, mid, tag="bad-row")
    result = _submit(tools, [
        {"kind": "workspace", "memory_id": mid, "status": "dismissed", "reason": "先落一行好的"},
    ])
    assert result["results"][0]["outcome"] == "dismissed"
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO workspace_dismissals(memory_id, version, suspected_workspace, decided_at)"
            " VALUES (999, 'not-an-int', 'dbpgsql', '2026-09-23T00:00:00Z')"
        )
    assert tools.db.scan_queue.load_workspace_dismissals() == {}
    data = tools.memory_repair("scan_workspace_anomalies", {})["data"]
    assert data["status"] == "ok", "坏行不得崩掉整轮 weekly 扫描"


def test_confirmed_move_leaves_no_durable_row(tmp_path: Path) -> None:
    """R2 变异 Mut-D 补测：confirmed 搬桶成功（:661 默认 durable_record=False，
    :706 守卫）不得 mint 持久豁免——守卫被删会让「已搬走的旧身份」永久豁免。"""
    tools = make_tools(tmp_path)
    mid = _workspace_clan(tools)
    _enqueue_workspace_candidate(tools, mid, tag="confirmed-move")
    result = _submit(tools, [
        {"kind": "workspace", "memory_id": mid, "status": "confirmed",
         "target_workspace": "dbpgsql", "conf": 0.9, "reason": "投票门 9/9"},
    ])
    entry = result["results"][0]
    assert entry["outcome"] == "moved", entry
    with tools.db.connection() as conn:
        durable = int(conn.execute("SELECT COUNT(*) FROM workspace_dismissals").fetchone()[0])
        queue_status = conn.execute(
            "SELECT status FROM scan_queue WHERE kind='workspace'"
        ).fetchall()
    assert durable == 0, "confirmed 搬桶不得落持久豁免"
    # 搬桶使行钉（旧桶/旧版本）失效，队列行被作废——重点是 durable 不落表。
    assert all(str(r[0]) != "pending" for r in queue_status)
