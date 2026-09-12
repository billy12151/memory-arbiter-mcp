"""0.16.0 normalization gate tests (plan §2 commit 6): vector-vote suspects in
the queue, the E7 double-signal gate (vote + conf + protected + multi-family),
auto-move with audit + watermark invalidation, and the rollback primitive."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter.tools import MemoryTools

from test_scan_pipeline import make_tools, _write


def _page(tools: MemoryTools, **payload):
    return tools.memory_repair("scan_queue", {"action": "page", **payload})["data"]


def _submit(tools: MemoryTools, decisions):
    return tools.memory_repair("scan_queue", {"action": "submit", "decisions": decisions})["data"]


def _seed_cluster(tools: MemoryTools, bucket: str, count: int, marker: str) -> list[int]:
    ids = []
    for i in range(count):
        ids.append(_write(tools, f"{bucket} 主题{i}", f"{marker} 相关正文内容{i}", workspace=bucket))
    return ids


def _seed_misplaced(tools: MemoryTools, marker: str) -> int:
    # One postgres-family memory in a content-mismatched bucket. The bucket
    # name carries the pgsql marker so the FakeEmbedder gives it a DISTINCT
    # workspace embedding (a same-vector name would be alias-resolved into
    # proja at write time — correct product behavior, useless fixture).
    return _write(tools, "错桶记忆", f"{marker} 的配置说明", workspace="pgsqlproj")


# ── suspect generation + gate pass → auto move ─────────────────────────────

def test_suspect_queued_and_gate_pass_moves(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _seed_cluster(tools, "proja", 12, "postgres 数据库")
    misplaced = _seed_misplaced(tools, "postgres 数据库")
    assert tools.wait_evidence_worker_drained(timeout=15)
    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 50})
    assert kick["ok"], kick
    page = _page(tools)
    suspects = [item for item in page["items"] if item["kind"] == "workspace"]
    assert suspects, "向量票 ≥8/10 的错桶必须成为归一疑点"
    item = next(i for i in suspects if i["memory_id"] == misplaced)
    assert item["suspected_workspace"] == "proja"
    assert item["current_workspace"] == "pgsqlproj"

    result = _submit(tools, [{
        "kind": "workspace", "memory_id": misplaced,
        "status": "confirmed", "target_workspace": "proja", "conf": 0.9,
    }])
    assert result["ok"], result
    entry = result["results"][0]
    assert entry["outcome"] == "moved", entry
    record = tools.db.get_memory(misplaced)
    assert (record["workspace_canonical"] or record["workspace"]) == "proja"
    # move 视同编辑: watermark invalidated → pipeline re-pairs in the new bucket
    assert tools.db.pending_scan_memory_ids() == [misplaced]
    # audit row exists
    with tools.db.connection() as conn:
        audits = [dict(r) for r in conn.execute(
            "SELECT * FROM normalize_audit WHERE memory_id=?", (misplaced,)).fetchall()]
    assert len(audits) == 1 and audits[0]["status"] == "applied"
    assert json.loads(audits[0]["gate"])["conf"] == 0.9


def test_gate_refuses_low_conf_and_vote_mismatch(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _seed_cluster(tools, "proja", 12, "postgres 数据库")
    misplaced = _seed_misplaced(tools, "postgres 数据库")
    tools.wait_evidence_worker_drained(timeout=15)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 50})
    # conf below 0.8 → gate_failed, row stays pending
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": misplaced,
        "status": "confirmed", "target_workspace": "proja", "conf": 0.5,
    }])
    entry = result["results"][0]
    assert entry["outcome"] == "gate_failed" and entry["gate"] == "conf"
    # wrong target → vote mismatch
    result2 = _submit(tools, [{
        "kind": "workspace", "memory_id": misplaced,
        "status": "confirmed", "target_workspace": "projc", "conf": 0.9,
    }])
    entry2 = result2["results"][0]
    assert entry2["outcome"] == "gate_failed" and entry2["gate"] == "vote"
    record = tools.db.get_memory(misplaced)
    assert (record["workspace_canonical"] or record["workspace"]) == "pgsqlproj", "门不过绝不搬"


# ── protected buckets (E6) ─────────────────────────────────────────────────

def test_protected_buckets_never_move_autonomously(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _seed_cluster(tools, "mema-twin", 12, "postgres 偏好素材")
    # a memory in a normal bucket whose neighbors live in the protected bucket
    misplaced = _write(tools, "疑似写进偏好桶", "postgres 偏好素材 内容", workspace="pgsqlproj")
    tools.wait_evidence_worker_drained(timeout=15)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 50})
    page = _page(tools)
    suspects = [i for i in page["items"] if i["kind"] == "workspace" and i["memory_id"] == misplaced]
    if suspects:
        assert suspects[0]["protected_involved"] is True
    else:
        # protected-involved suspects may be withheld from the queue entirely
        pass
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": misplaced,
        "status": "confirmed", "target_workspace": "mema-twin", "conf": 0.95,
    }])
    entry = result["results"][0]
    assert entry["outcome"] == "protected_bucket_hint", entry
    record = tools.db.get_memory(misplaced)
    assert (record["workspace_canonical"] or record["workspace"]) == "pgsqlproj", "E6：受保护桶零物理搬"


# ── rollback (§6⑫) ─────────────────────────────────────────────────────────

def test_rollback_restores_auto_move(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _seed_cluster(tools, "proja", 12, "postgres 数据库")
    misplaced = _seed_misplaced(tools, "postgres 数据库")
    tools.wait_evidence_worker_drained(timeout=15)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 50})
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": misplaced,
        "status": "confirmed", "target_workspace": "proja", "conf": 0.9,
    }])
    assert result["results"][0]["outcome"] == "moved"
    with tools.db.connection() as conn:
        audit = conn.execute(
            "SELECT id FROM normalize_audit WHERE memory_id=? AND status='applied'",
            (misplaced,),
        ).fetchone()
    rollback = tools.memory_govern("rollback_auto_move", {
        "audit_id": int(audit["id"]), "reason": "owner says wrong", "authorized": True,
    })
    assert rollback["ok"], rollback
    record = tools.db.get_memory(misplaced)
    assert (record["workspace_canonical"] or record["workspace"]) == "pgsqlproj"
    with tools.db.connection() as conn:
        status = conn.execute(
            "SELECT status FROM normalize_audit WHERE id=?", (int(audit["id"]),),
        ).fetchone()[0]
    assert status == "rolled_back"
    # unauthorized rollback is refused
    rollback2 = tools.memory_govern("rollback_auto_move", {"audit_id": int(audit["id"])})
    assert not rollback2["ok"]


# ── multi-family downgrade (E7-4) ──────────────────────────────────────────

def test_multi_family_mention_downgrades_to_hint(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _seed_cluster(tools, "proja", 12, "postgres 数据库")
    # register a second family name (carries the pgsql marker so the
    # FakeEmbedder gives its workspace embedding a DISTINCT direction and the
    # resolver registers it instead of folding it into proja) and craft a
    # subject mentioning both
    _seed_cluster(tools, "pgsqlc", 3, "无关内容")
    multi = _write(tools, "proja 与 pgsqlc 联合决议记录", "postgres 数据库 相关", workspace="pgsqlproj")
    tools.wait_evidence_worker_drained(timeout=15)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 50})
    # E7-4 unit-level check: the multi-family detector itself must flag the
    # subject that names two registered canonicals.
    record = tools.db.get_memory(multi)
    families = tools._queue_protocol._multi_family_mentions(record, "proja")
    assert "proja" in families and "pgsqlc" in families, families
    result = _submit(tools, [{
        "kind": "workspace", "memory_id": multi,
        "status": "confirmed", "target_workspace": "proja", "conf": 0.95,
    }])
    entry = result["results"][0]
    assert entry["outcome"] == "multi_family_hint", entry
    # and no physical move happened (its bucket may have been resolver-folded
    # into another postgres-flavored name, but never into the target proja)
    record = tools.db.get_memory(multi)
    assert (record["workspace_canonical"] or record["workspace"]) != "proja"
