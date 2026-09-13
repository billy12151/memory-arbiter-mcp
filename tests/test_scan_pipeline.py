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


def _write(tools: MemoryTools, subject: str, content: str, workspace: str = "ws", tags: list | None = None) -> int:
    res = tools.memory_write(content=content, subject=subject, workspace=workspace, tags=list(tags or []))
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

def test_kick_excludes_evolution_and_keeps_numeric(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # 0.16.4 §1: the polarity pair is a cross-memory evolution-domain shape
    # — is_cross_evolution excludes it BEFORE the rank gate (scan) and the
    # provenance gate (write-time); it must never queue. The numeric pair
    # (same-sentence two-values) is the shape the difference classifier
    # keeps for agent judgment (E11③ auto-reject retired).
    a = _write(tools, "演进排除甲", "该功能包含缓存模块")
    b = _write(tools, "演进排除乙", "该功能不包含缓存模块")
    n1 = _write(tools, "版本快照甲", "重试次数为 3 次")
    n2 = _write(tools, "版本快照乙", "重试次数为 5 次")
    assert tools.wait_evidence_worker_drained(timeout=10)

    kick = tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    assert kick["ok"], kick
    data = kick["data"]
    assert data["complete"] is True
    assert data["pending_memories"] == 0

    with tools.db.connection() as conn:
        rows = conn.execute("SELECT kind,member_versions FROM scan_queue").fetchall()
    queued_member_sets = [
        {int(m["memory_id"]) for m in json.loads(row["member_versions"])}
        for row in rows if row["kind"] == "conflict"
    ]
    # evolution-domain pair: excluded on BOTH paths (write-time KNN + scan)
    assert {a, b} not in queued_member_sets, queued_member_sets
    # the same-sentence numeric pair is kept by the classifier and enqueued
    assert {n1, n2} in queued_member_sets, queued_member_sets
    auto_reject_rows = tools.db.list_conflicts(status="not_a_conflict", source="scan_numeric_autoreject")
    assert auto_reject_rows == [], auto_reject_rows


def test_evolution_excluded_write_time_no_notice(tmp_path: Path) -> None:
    """Write-time path of the 0.16.4 exclusion: a polarity pair whose
    entity/scope metadata is aligned (so the provenance gate would NOT kill
    it — hit metadata is joined live from memories) still produces no
    semantic notice and no queue row. The only remaining killer is the
    evolution-domain exclusion: the earliest kill, ahead of provenance."""
    tools = make_tools(tmp_path)
    a = _write(tools, "写时演进甲", "该功能包含缓存模块", tags=["x"])
    b = _write(tools, "写时演进乙", "该功能不包含缓存模块", tags=["x"])
    assert tools.memory_set_entity(a, "网关", "路由")["data"]["updated"]
    assert tools.memory_set_entity(b, "网关", "路由")["data"]["updated"]
    # Version lift re-runs the write-time evidence loop with metadata live.
    tools.memory("update", {"memory_id": b, "new_content": "该功能不包含缓存模块与限流", "reason": "evo"})
    assert tools.wait_evidence_worker_drained(timeout=10)
    with tools.db.connection() as conn:
        notices = conn.execute(
            "SELECT COUNT(*) FROM conflicts WHERE source='semantic_evidence'"
        ).fetchone()[0]
        queued = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE kind='conflict'"
        ).fetchone()[0]
    assert notices == 0, "跨记忆演进对写时不得产生 semantic notice"
    assert queued == 0, "跨记忆演进对不得入队"


def test_evolution_retroactive_migration_idempotent(tmp_path: Path) -> None:
    """The additive one-shot voids pending notify stock exactly once."""
    tools = make_tools(tmp_path)
    a = _write(tools, "迁移甲", "重试次数为 3 次")
    b = _write(tools, "迁移乙", "重试次数为 5 次")
    assert tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT candidate_key_hash FROM scan_queue WHERE kind='conflict' AND status='pending'"
        ).fetchone()
    assert row is not None
    # Forge two pending notify-shaped rows (the stock 0.16.4 clears) and
    # reset the migration guard so this boot replays the "upgraded binary
    # first meets the stocked library" moment (the make_tools boot above
    # already wrote the guard against an empty queue).
    now = "2026-09-13T00:00:00+00:00"
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM migration_state WHERE key='scan_pipeline_evolution_void_v1'")
        for reason in ("polarity_changed", "todo_resolved"):
            conn.execute(
                """INSERT INTO scan_queue(kind,workspace_canonical,status,candidate_key_hash,
                     member_versions,evidence,reason,severity,source,detail,created_at,updated_at)
                   VALUES('conflict','ws','pending',?,'[]','[]',?,'normal','scan_pipeline',NULL,?,?)""",
                (f"{'f' * 60}{reason[:4]}", reason, now, now),
            )
    # Re-boot triggers the additive completion (one-shot, guard-keyed).
    from memory_arbiter.db import MemoryDB

    db2 = MemoryDB(tools.settings)
    with db2.connection() as conn:
        pending_notify = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE status='pending' "
            "AND (reason LIKE '%todo_resolved%' OR reason LIKE '%polarity_changed%')"
        ).fetchone()[0]
        voided = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE status='voided' "
            "AND (reason LIKE '%todo_resolved%' OR reason LIKE '%polarity_changed%')"
        ).fetchone()[0]
        kept = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE status='pending' AND kind='conflict'"
        ).fetchone()[0]
    assert pending_notify == 0, "存量 notify 行必须全部 voided"
    assert voided == 2
    assert kept == 1, "numeric 行不受迁移影响"
    # Idempotent: a third boot keeps the guard (no re-run) and — per the
    # standing boot hygiene — the voided rows are DELETED outright, which is
    # the designed terminal path, not a migration failure.
    db3 = MemoryDB(tools.settings)
    with db3.connection() as conn:
        guard = conn.execute(
            "SELECT value FROM migration_state WHERE key='scan_pipeline_evolution_void_v1'"
        ).fetchone()
        pending_after = conn.execute(
            "SELECT COUNT(*) FROM scan_queue WHERE status='pending' AND kind='conflict'"
        ).fetchone()[0]
    assert guard and guard[0] == "voided=2", "迁移守卫键保留（幂等）"
    assert pending_after == 1, "numeric 行跨 boot 存活"


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
    # 0.16.4: the numeric shape (same-sentence two-values) is the reliable
    # cross-memory fixture — the polarity shape is the excluded evolution
    # domain now.
    a = _write(tools, "编辑甲", "重试次数为 3 次")
    b = _write(tools, "编辑乙", "重试次数为 5 次")
    assert tools.wait_evidence_worker_drained(timeout=10)
    tools.memory_repair("scan_pipeline", {"action": "kick", "max_memories": 10})
    with tools.db.connection() as conn:
        rows = [dict(r) for r in conn.execute("SELECT kind, member_versions FROM scan_queue WHERE kind='conflict'").fetchall()]
    assert rows
    # Dismiss everything (agent judgment) → suppression source lands
    from memory_arbiter.db_generation import CONFLICT_DETECTOR_VERSION

    for row in rows:
        outcome = tools.db.record_conflict_group(
            workspace_canonical="ws", slot_key=None,
            members=json.loads(row["member_versions"]), value_groups=[],
            status="not_a_conflict", detector_version=CONFLICT_DETECTOR_VERSION,
            source="scan_queue_test", detection_reason="test dismissal",
        )
        assert outcome.get("outcome") in {"inserted", "deduped"}, outcome
    tools.db.mark_scanned(a, tools.db.get_memory(a)["version"])
    tools.db.mark_scanned(b, tools.db.get_memory(b)["version"])
    # Edit one memory → version lift → new identity → clean re-queue
    # (keep the pure numeric sentence shape — extra prose shifts the
    # skeleton and the difference classifier clears the pair).
    tools.memory("update", {"memory_id": a, "new_content": "重试次数为 7 次", "reason": "edit"})
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
    # §6⑤ negative assertion: the moved memory must NOT re-pair with its old
    # "ws" peer — every pending pair sits wholly inside its row's bucket.
    current_buckets = {}
    for row in rows:
        members = json.loads(row["member_versions"])
        for m in members:
            mid = int(m["memory_id"])
            if mid not in current_buckets:
                current_buckets[mid] = (
                    tools.db.get_memory(mid).get("workspace_canonical")
                    or tools.db.get_memory(mid).get("workspace")
                )
        member_buckets = {current_buckets[int(m["memory_id"])] for m in members}
        assert member_buckets <= {row["workspace_canonical"]}, (
            f"cross-bucket pair leaked: {members} buckets={member_buckets} "
            f"row_ws={row['workspace_canonical']}"
        )


def test_scan_candidates_diagnostic_excludes_evolution(tmp_path: Path) -> None:
    """0.16.4 review P2: the diagnostic channel (scan_candidates) routes
    through the SAME shared predicate — evolution-domain pairs must not
    surface as notice_ready candidates there either, or the retroactive
    void's released identities would leak back first-class."""
    tools = make_tools(tmp_path)
    a = _write(tools, "诊断演进甲", "该功能包含缓存模块")
    b = _write(tools, "诊断演进乙", "该功能不包含缓存模块")
    n1 = _write(tools, "诊断数值甲", "重试次数为 3 次")
    n2 = _write(tools, "诊断数值乙", "重试次数为 5 次")
    assert tools.wait_evidence_worker_drained(timeout=10)
    result = tools.memory_repair("scan_candidates", {
        "anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True,
    })
    assert result["ok"], result
    candidates = result["data"].get("candidates") or []
    pairs = {
        frozenset((int(c["left_id"]), int(c["right_id"]))) for c in candidates
    }
    assert frozenset((a, b)) not in pairs, "演进域对不得在诊断通道出现"
    assert any(p == frozenset((n1, n2)) for p in pairs), "numeric 对照对应保留"
