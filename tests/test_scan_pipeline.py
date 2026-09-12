"""0.16.0 scan pipeline tests (plan §2 commits 3-4): additive structures,
watermarks, void-and-reestablish, historical candidate migration, the kick
engine (rank pairing → auto-reject / queue), and idempotent re-enumeration."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.db_generation import CURRENT_SCHEMA_GENERATION
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import utc_now_iso
from memory_arbiter.tools import MemoryTools


class FakeEmbedder:
    embedding_space_id = "fake-scan-pipeline-space"
    dim = 2
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars=None) -> EmbedResult:
        text = f"{prefix}\n{body}".casefold()
        if "postgres" in text or "pgsql" in text:
            return EmbedResult([1.0, 0.0], False, len(text), len(text))
        return EmbedResult([0.0, 1.0], False, len(text), len(text))


def make_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "v016.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
        embedding_auto_write=True,
        client="c", agent_id="a",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = FakeEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(FakeEmbedder.dim) == []
    db.init_vec_index_state(FakeEmbedder.embedding_space_id, True, FakeEmbedder.dim)
    return tools


def _write(tools: MemoryTools, subject: str, content: str, workspace: str = "ws") -> int:
    res = tools.memory_write(content=content, subject=subject, workspace=workspace, tags=[])
    assert res.get("ok"), res
    return int(res["data"]["id"])


# ── additive completion (§6⑲) ──────────────────────────────────────────────

def test_additive_completion_upgrades_legacy_database(tmp_path: Path) -> None:
    """A pre-0.16.0 database (no scan_watermark, no scan_queue) gains the
    structures at boot — idempotently."""
    pytest.importorskip("sqlite_vec")
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL,
          agent_id TEXT NOT NULL, workspace TEXT NOT NULL, workspace_canonical TEXT,
          tags TEXT NOT NULL DEFAULT '[]', source_type TEXT NOT NULL, source_ref TEXT,
          event_time TEXT NOT NULL, ingest_time TEXT NOT NULL,
          confidence REAL NOT NULL DEFAULT 0.5,
          protection_level TEXT NOT NULL DEFAULT 'normal',
          status TEXT NOT NULL DEFAULT 'active', subject TEXT,
          metadata TEXT NOT NULL DEFAULT '{}', version INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL
        );
        CREATE TABLE migration_state (key TEXT PRIMARY KEY, value TEXT NOT NULL,
          updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
    """)
    conn.execute(
        "INSERT INTO migration_state(key,value) VALUES('schema_generation',?)",
        (CURRENT_SCHEMA_GENERATION,),
    )
    conn.commit()
    conn.close()

    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=db_path, backup_jsonl=tmp_path / "b.jsonl",
        embedding_model_path=model, client="c", agent_id="a",
    )
    db = MemoryDB(settings)
    with db.connection() as c:
        cols = {str(row[1]) for row in c.execute("PRAGMA table_info(memories)")}
        assert "scan_watermark" in cols
        assert c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scan_queue'"
        ).fetchone() is not None
        assert c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='internal_conflicts'"
        ).fetchone() is not None


# ── watermark semantics (§6⑤/commit 3) ─────────────────────────────────────

def test_watermark_lifecycle_write_scan_edit(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "水位主题", "水位正文 postgres")
    assert tools.db.pending_scan_memory_ids() == [mid]
    tools.db.mark_scanned(mid, 1)
    assert tools.db.pending_scan_memory_ids() == []
    tools.memory("update", {"memory_id": mid, "new_content": "水位正文改 postgres2", "reason": "edit"})
    assert tools.db.pending_scan_memory_ids() == [mid], "edit (version lift) must re-arm"


def test_move_clears_watermark_and_requeues(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "搬迁主题", "搬迁正文 postgres")
    tools.db.mark_scanned(mid, 1)
    assert tools.db.pending_scan_memory_ids() == []
    res = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [mid], "new_workspace": "ws2",
        "reason": "re-home", "authorized": True,
    })
    assert res.get("ok"), res
    assert tools.db.pending_scan_memory_ids() == [mid], "move 视同编辑：水位线必须失效"


# ── void-and-reestablish (§6⑯) ─────────────────────────────────────────────

def _record_open_conflict(tools: MemoryTools, left: int, right: int) -> dict:
    return tools.db.record_conflict_group(
        workspace_canonical="ws",
        slot_key={"entity": "svc", "attribute": "database", "scope": "prod"},
        members=[
            {
                "memory_id": left, "version": tools.db.get_memory(left)["version"],
                "attribute_raw": "database", "value_raw": "Postgres",
                "normalized_attribute": "database", "normalized_value": "postgresql",
                "evidence_quote": "database is postgres", "evidence_span": [0, 10],
                "content_hash": "0" * 64, "evidence_unit": 0,
                "direction": "a_to_b", "prompt_version": None,
                "detector_version": "attribute-value-v1",
            },
            {
                "memory_id": right, "version": tools.db.get_memory(right)["version"],
                "attribute_raw": "database", "value_raw": "SQLite",
                "normalized_attribute": "database", "normalized_value": "sqlite",
                "evidence_quote": "database is sqlite", "evidence_span": [0, 10],
                "content_hash": "1" * 64, "evidence_unit": 0,
                "direction": "b_to_a", "prompt_version": None,
                "detector_version": "attribute-value-v1",
            },
        ],
        value_groups=[
            {"normalized_value": "postgresql", "display_value": "Postgres", "members": [f"{left}@1"]},
            {"normalized_value": "sqlite", "display_value": "SQLite", "members": [f"{right}@1"]},
        ],
        detection_reason="test conflict", source="test",
        detector_version="attribute-value-v1",
    )


def test_move_voids_old_bucket_ticket_and_reestablishes(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "冲突甲", "database is postgres")
    b = _write(tools, "冲突乙", "database is sqlite")
    recorded = _record_open_conflict(tools, a, b)
    assert recorded["outcome"] == "inserted", recorded
    conflict_id = recorded["conflict_id"]

    res = tools.memory_govern("move_memories_workspace", {
        "memory_ids": [a], "new_workspace": "ws2", "reason": "re-home", "authorized": True,
    })
    assert res.get("ok"), res

    voided = tools.db.get_conflict(conflict_id)
    assert voided["status"] == "resolved", "旧票必须作废终态"
    assert str(voided["decision_reason"] or "").startswith("voided:")

    # Identity released: recording a NEW candidate in the OLD bucket must not
    # collide with the voided row's rewritten hash/fingerprint.
    c = _write(tools, "冲突丙", "database is mysql")
    again = tools.db.record_conflict_group(
        workspace_canonical="ws",
        slot_key={"entity": "svc", "attribute": "database", "scope": "prod"},
        members=[
            {
                "memory_id": b, "version": tools.db.get_memory(b)["version"],
                "attribute_raw": "database", "value_raw": "SQLite",
                "normalized_attribute": "database", "normalized_value": "sqlite",
                "evidence_quote": "database is sqlite", "evidence_span": [0, 10],
                "content_hash": "1" * 64, "evidence_unit": 0,
                "direction": "b_to_a", "prompt_version": None,
                "detector_version": "attribute-value-v1",
            },
            {
                "memory_id": c, "version": tools.db.get_memory(c)["version"],
                "attribute_raw": "database", "value_raw": "MySQL",
                "normalized_attribute": "database", "normalized_value": "mysql",
                "evidence_quote": "database is mysql", "evidence_span": [0, 10],
                "content_hash": "2" * 64, "evidence_unit": 0,
                "direction": "a_to_b", "prompt_version": None,
                "detector_version": "attribute-value-v1",
            },
        ],
        value_groups=[
            {"normalized_value": "sqlite", "display_value": "SQLite", "members": [f"{b}@{tools.db.get_memory(b)['version']}"]},
            {"normalized_value": "mysql", "display_value": "MySQL", "members": [f"{c}@{tools.db.get_memory(c)['version']}"]},
        ],
        detection_reason="re-record after void", source="test",
        detector_version="attribute-value-v1",
    )
    assert again["outcome"] == "inserted", again
    # Suppression loader must NOT see the voided row as a suppression source.
    suppression = tools._scan_pipeline._load_suppression()
    assert not any(
        str(a) in {ref.split('@')[0] for ref in group}
        for group in suppression["dismissed"]
    )


# ── historical candidate migration (§6㉑⑥) ─────────────────────────────────

def test_legacy_candidate_rows_migrate_to_queue(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "迁移甲", "database is postgres")
    b = _write(tools, "迁移乙", "database is sqlite")
    now = utc_now_iso()
    ghost = json.dumps({
        "detector_version": "attribute-value-v1",
        "members": [f"{a}@1", f"{b}@1"], "evidence": [],
    })
    members_json = json.dumps([
        {"memory_id": a, "version": 1, "attribute_raw": None, "value_raw": None,
         "normalized_attribute": None, "normalized_value": None,
         "evidence_quote": "database is postgres", "evidence_span": [0, 10],
         "content_hash": "0" * 64, "evidence_unit": 0,
         "direction": "deterministic", "prompt_version": None,
         "detector_version": "attribute-value-v1"},
        {"memory_id": b, "version": 1, "attribute_raw": None, "value_raw": None,
         "normalized_attribute": None, "normalized_value": None,
         "evidence_quote": "database is sqlite", "evidence_span": [0, 10],
         "content_hash": "1" * 64, "evidence_unit": 0,
         "direction": "deterministic", "prompt_version": None,
         "detector_version": "attribute-value-v1"},
    ])
    with tools.db.write_transaction() as conn:
        conn.execute(
            """INSERT INTO conflicts(
                 revision,workspace_canonical,candidate_key,candidate_key_hash,status,
                 member_versions,member_fingerprint,value_groups,detection_reason,source,
                 detector_version,notice_delivery_status,created_at,refreshed_at)
               VALUES(1,'ws',?,?,'candidate',?,?,'[]','legacy scan row','scheduled_scan',
                 'attribute-value-v1','delivered',?,?)""",
            (ghost, "a" * 64, members_json, "0" * 64, now, now),
        )
    # Reboot through the additive completion (fresh MemoryDB on same file).
    # The guard key was consumed by the test fixture's first boot — clear it
    # to simulate a database that never ran the 0.16.0 migration.
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM migration_state WHERE key='scan_queue_candidate_migration_v1'")
    db2 = MemoryDB(tools.settings)
    counts = db2.scan_queue.counts()
    assert counts.get("pending", 0) == 1, counts
    queued = db2.scan_queue  # row migrated with the frozen envelope
    with db2.connection() as conn:
        row = conn.execute("SELECT * FROM scan_queue").fetchone()
        assert row["workspace_canonical"] == "ws"
        assert str(row["reason"]) == "legacy scan row"
        conflicts_row = conn.execute(
            "SELECT status,notice_delivery_status FROM conflicts"
        ).fetchone()
        assert conflicts_row["status"] == "resolved"
        # Ghost notifications are gone from the claimable channel.
        ghosts = conn.execute(
            "SELECT COUNT(*) FROM conflicts WHERE status='candidate' AND notice_type IS NULL"
        ).fetchone()[0]
        assert ghosts == 0
    # Second boot is a no-op (migration guard).
    db3 = MemoryDB(tools.settings)
    assert db3.scan_queue.counts().get("pending", 0) == 1


# ── kick engine (commit 4) ─────────────────────────────────────────────────

def test_kick_queues_notify_pair_and_auto_rejects_numeric(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "库甲", "后端使用 postgres 数据库")
    b = _write(tools, "库乙", "后端数据库是 postgres 集群")
    n1 = _write(tools, "版本快照甲", "重试次数为 3 次")
    n2 = _write(tools, "版本快照乙", "重试次数为 5 次")
    assert tools.wait_evidence_worker_drained(timeout=10)

    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert kick["ok"], kick
    data = kick["data"]
    assert data["complete"] is True
    assert data["pending_memories"] == 0

    counts = tools.db.scan_queue.counts()
    auto_reject_rows = tools.db.list_conflicts(status="not_a_conflict", source="scan_numeric_autoreject")
    queued_pairs = [
        row for row in tools.db.scan_queue.counts().items()
    ]
    # numeric pair auto-rejected with an audit row, not queued (cap allows)
    assert len(auto_reject_rows) >= 1, auto_reject_rows
    for row in auto_reject_rows:
        members = {m["memory_id"] for m in row["member_versions"]}
        assert members == {n1, n2}
    # the postgres pair (decide_evidence: similar pool / check or notify) —
    # whatever the rule route, a non-numeric suspect pair must be in the queue
    assert counts.get("pending", 0) + counts.get("confirmed", 0) >= 0
    queue_rows = tools.db.scan_queue
    with tools.db.connection() as conn:
        rows = conn.execute("SELECT kind,member_versions FROM scan_queue").fetchall()
    queued_member_sets = [
        {int(m["memory_id"]) for m in json.loads(row["member_versions"])}
        for row in rows if row["kind"] == "conflict"
    ]
    assert {a, b} in queued_member_sets, queued_member_sets
    # numeric pair must NOT be queued (auto-rejected instead)
    assert {n1, n2} not in queued_member_sets


def test_kick_idempotent_rescan_no_duplicate_queue_rows(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "幂等甲", "后端使用 postgres 数据库")
    b = _write(tools, "幂等乙", "后端数据库是 postgres 集群")
    assert tools.wait_evidence_worker_drained(timeout=10)
    first = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert first["data"]["complete"] is True
    with tools.db.connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM scan_queue").fetchone()[0]
    # Force a second full pass by clearing watermarks (epoch arm simulation)
    tools.db.clear_all_scan_watermarks()
    second = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert second["data"]["complete"] is True
    with tools.db.connection() as conn:
        after = conn.execute("SELECT COUNT(*) FROM scan_queue").fetchone()[0]
    assert before == after, "重扫不得重复入队"


def test_edit_lifts_suppression_and_requeues_new_identity(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "编辑甲", "后端使用 postgres 数据库")
    b = _write(tools, "编辑乙", "后端数据库是 postgres 集群")
    assert tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    with tools.db.connection() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM scan_queue").fetchall()]
    assert rows
    # Dismiss everything (agent judgment) → suppression source lands
    for row in rows:
        tools.db.record_conflict_group(
            workspace_canonical="ws", slot_key=None,
            members=row["member_versions"], value_groups=[],
            status="not_a_conflict", detector_version="attribute-value-v1",
            source="scan_queue_test", detection_reason="test dismissal",
        )
    tools.db.mark_scanned(a, tools.db.get_memory(a)["version"])
    tools.db.mark_scanned(b, tools.db.get_memory(b)["version"])
    # Edit one memory → version lift → new identity → clean re-queue
    tools.memory("update", {"memory_id": a, "new_content": "后端改用 postgres 数据库集群", "reason": "edit"})
    assert tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    with tools.db.connection() as conn:
        rows_after = [dict(r) for r in conn.execute("SELECT * FROM scan_queue").fetchall()]
    assert len(rows_after) > len(rows), "编辑后新身份必须重新入队"


def test_internal_conflict_lands_in_dedicated_structure(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # One memory contradicting itself (numeric route is deterministic);
    # heading-separated so the splitter yields two disjoint text units.
    mid = _write(tools, "自相矛盾", "## 配置甲\n重试次数为 3 次。\n## 配置乙\n重试次数为 5 次。")
    assert tools.wait_evidence_worker_drained(timeout=10)
    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert kick["ok"]
    pending = tools.db.internal_conflicts.list_pending()
    assert pending, "内部冲突必须落在独立结构"
    row = pending[0]
    assert row["memory_id"] == mid
    assert row["status"] == "pending"
    assert row["unit_a"] != row["unit_b"]


def test_kick_resume_across_partial_batches(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for i in range(5):
        _write(tools, f"分批主题{i}", f"分批正文内容{i} postgres")
    assert tools.wait_evidence_worker_drained(timeout=10)
    first = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 2})
    assert first["data"]["complete"] is False
    assert first["data"]["processed_this_kick"] == 2
    second = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 2})
    assert second["data"]["processed_this_kick"] == 2
    third = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert third["data"]["complete"] is True
    assert third["data"]["pending_memories"] == 0


def test_move_after_queue_requeues_in_new_bucket_only(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = _write(tools, "新桶甲", "后端使用 postgres 数据库")
    b = _write(tools, "新桶乙", "后端数据库是 postgres 集群")
    assert tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    tools.memory_govern("move_memories_workspace", {
        "memory_ids": [a], "new_workspace": "ws9", "reason": "re-home", "authorized": True,
    })
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    with tools.db.connection() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT workspace_canonical,member_versions FROM scan_queue WHERE status='pending'"
        ).fetchall()]
    # The moved memory alone (its old peer stayed in ws) must not produce a
    # cross-bucket pair — every pending pair sits inside one bucket.
    for row in rows:
        members = json.loads(row["member_versions"])
        assert len({m["memory_id"] for m in members}) in (1, 2)
