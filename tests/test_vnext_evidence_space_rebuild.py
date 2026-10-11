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

# ── test_vnext_evidence space rebuild/resume domain (commit 12 split) ──

def test_failed_migration_target_is_not_current_generation(tmp_path: Path, monkeypatch) -> None:
    from memory_arbiter.db_generation import detect_database_generation
    from memory_arbiter.vnext_migration import build

    tools = make_tools(tmp_path)
    legacy = tmp_path / "legacy.sqlite3"
    src = MemoryDB(Settings(db_path=legacy, backup_jsonl=tmp_path / "unused.jsonl"))
    from memory_arbiter.models import MemoryRecord
    src.insert_memory(
        MemoryRecord.from_input({"content": "内容", "subject": "s", "workspace": "w"}, tools.settings.defaults()),
        "w",
    )

    target = tmp_path / "target.sqlite3"
    monkeypatch.setattr(
        "memory_arbiter.vnext_migration.MemoryTools",
        lambda **kwargs: type("FailingTools", (), {
            "_ensure_embedder": staticmethod(lambda: (FakeEmbedder(), [])),
            "_index_local_text_evidence": staticmethod(
                lambda *a, **k: {"status": "failed", "reason": "test"}
            ),
        })(),
    )
    result = build(legacy, target, tools.settings)
    assert result["ok"] is False
    assert detect_database_generation(target) == "unknown"


def test_final_sync_rejects_source_writes_landing_before_replace(tmp_path: Path, monkeypatch) -> None:
    from memory_arbiter.vnext_migration import final_sync

    from memory_arbiter.models import MemoryRecord

    tools = make_tools(tmp_path)
    legacy = tmp_path / "legacy.sqlite3"
    settings = Settings(
        db_path=legacy, backup_jsonl=tmp_path / "unused.jsonl",
        embedding_model_path=tools.settings.embedding_model_path,
    )
    original_init = MemoryTools.__init__

    def patched_init(self, settings=None, db=None):
        original_init(self, settings=settings, db=db)
        self._embedder = FakeEmbedder()
        self._embedder_loaded = True
        self.db.ensure_vec_tables(FakeEmbedder.dim)
        self.db.init_vec_index_state("fake-vnext-space", True, active_dim=FakeEmbedder.dim)

    monkeypatch.setattr(MemoryTools, "__init__", patched_init)
    src = MemoryDB(settings)
    src.insert_memory(
        MemoryRecord.from_input({"content": "原始内容", "subject": "s", "workspace": "w"}, tools.settings.defaults()),
        "w",
    )
    target = legacy.with_name("legacy.vnext.sqlite3")

    import memory_arbiter.vnext_migration as vm
    original_checkpoint = vm._checkpoint

    def mutating_checkpoint(path):
        # A write lands while the target is being checkpointed.
        with src.write_transaction() as conn:
            conn.execute("UPDATE memories SET content='并发写入' WHERE id=1")
        return original_checkpoint(path)

    monkeypatch.setattr(vm, "_checkpoint", mutating_checkpoint)
    result = final_sync(legacy, target, tools.settings)
    assert result["ok"] is False
    assert result["error"] == "source_changed_during_final_sync"
    assert not target.exists()
    # The next_step promises the built staging DB is retained for the rerun.
    assert Path(result["staging"]).exists()


def test_space_rebuild_flips_mismatch_back_to_ready(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = tools.memory_write(
        content="接口超时为 5 秒。", subject="timeout", tags=[], workspace="project",
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    # Simulate a model swap: the active space no longer matches the live embedder.
    tools.db.init_vec_index_state("fake-vnext-space", True)
    tools.db.init_vec_index_state("new-space-v2", True)
    state = tools.db.get_vec_index_state()
    assert state["state"] == "mismatch" and state["target_space_id"] == "new-space-v2"

    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "new-space-v2"
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM workspace_canonicals_vec")
    rebuild = tools.memory_repair("rebuild_evidence", {"dry_run": False})
    assert rebuild["ok"] is True and rebuild["data"]["queued"] >= 1
    assert rebuild["data"]["workspace_vector_rebuild"]["ok"] is True
    assert tools.wait_semantic_worker_drained(timeout=5)

    state = tools.db.get_vec_index_state()
    assert state["state"] == "ready"
    assert state["active_space_id"] == "new-space-v2"
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspace_canonicals_vec").fetchone()[0] == 1
        assert conn.execute(
            "SELECT 1 FROM _vec_index_meta WHERE key='workspace_rebuild_space_id'"
        ).fetchone() is None
    result = tools.memory_search(query="接口超时")
    assert result["data"]["vector_lag"]["pending_evidence_index"] == 0


def test_lazy_space_check_blocks_ordinary_publish_until_rebuild_starts(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = tools.memory_write(content="old space", subject="old", tags=[])["data"]["id"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    with tools.db.connection() as conn:
        before = int(conn.execute("SELECT COUNT(*) FROM memory_row_vec").fetchone()[0])

    tools.db.init_vec_index_state("new-space", True)
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "new-space"
    blocked = tools._index_local_text_evidence(first, tools.db.get_memory(first))
    assert blocked["status"] == "skipped"
    assert blocked["reason"] == "embedding_space_rebuild_required"
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memory_row_vec").fetchone()[0] == before

    tools.db.mark_space_rebuild_started()
    allowed = tools._index_local_text_evidence(first, tools.db.get_memory(first))
    assert allowed["status"] == "indexed"


def test_explicit_rebuild_recovers_missing_vector_tables(tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    tools = make_tools(tmp_path)
    tools.memory_write(content="repair me", subject="repair", workspace="project")
    assert tools.wait_semantic_worker_drained(timeout=5)
    with tools.db.write_transaction() as conn:
        conn.execute("DROP TABLE memory_row_vec")
        conn.execute("DROP TABLE workspace_canonicals_vec")
    tools.db.state.sqlite_vec_available = False
    tools.db._sqlite_vec_loadable = False

    preview = tools.memory_repair("rebuild_evidence", {"dry_run": True, "batch_size": 10})
    assert preview["ok"] is True
    assert preview["data"]["vector_table_repair"] == {
        "required": True,
        "missing_tables": ["memory_row_vec", "workspace_canonicals_vec"],
        "recreated": False,
        "warnings": [],
    }
    assert preview["data"]["workspace_vectors"] == "rebuild_required"

    result = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert result["ok"] is True, result
    assert result["data"]["vector_table_repair"]["recreated"] is True
    assert tools.wait_semantic_worker_drained(timeout=5)
    assert tools.db.get_vec_index_state()["state"] == "ready"
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memory_row_vec").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM workspace_canonicals_vec").fetchone()[0] == 1


def test_space_rebuild_completion_purges_deleted_evidence_and_requires_vectors(
    tmp_path: Path,
) -> None:
    tools = make_tools(tmp_path)
    active = tools.memory_write(content="active", subject="a", tags=[])["data"]["id"]
    deleted = tools.memory_write(content="deleted", subject="d", tags=[])["data"]["id"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status='deleted' WHERE id=?", (deleted,))
    rebuild = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert rebuild["ok"] is True
    assert tools.wait_semantic_worker_drained(timeout=5)
    with tools.db.write_transaction() as conn:
        active_evidence = conn.execute(
            "SELECT id FROM memory_row WHERE memory_id=? ORDER BY id", (active,),
        ).fetchall()
        deleted_evidence = conn.execute(
            "SELECT id FROM memory_row WHERE memory_id=?", (deleted,),
        ).fetchall()
        missing_vector_id = int(active_evidence[0]["id"])
        conn.execute("DELETE FROM memory_row_vec WHERE id=?", (missing_vector_id,))
        conn.executemany(
            "INSERT INTO _vec_index_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (
                ("state", "mismatch"),
                ("target_space_id", "space-b"),
                ("space_rebuild_evidence_id", "0"),
                ("workspace_rebuild_space_id", "space-b"),
            ),
        )
        conn.execute(
            "INSERT INTO memory_row_vec(id,parent_status,embedding) VALUES(?,?,?)",
            (999999, "active", "[0.0,1.0]"),
        )
    assert tools.db.maybe_complete_space_rebuild("space-b") is False
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO memory_row_vec(id,parent_status,embedding) VALUES(?,?,?)",
            (missing_vector_id, "active", "[0.0,1.0]"),
        )
    assert tools.db.maybe_complete_space_rebuild("space-b") is True
    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_row WHERE memory_id=?", (deleted,)
        ).fetchone()[0] == 0
        assert not deleted_evidence or conn.execute(
            "SELECT COUNT(*) FROM memory_row_vec WHERE id IN ("
            + ",".join("?" for _ in deleted_evidence) + ")",
            tuple(int(row["id"]) for row in deleted_evidence),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_row_vec WHERE id=999999"
        ).fetchone()[0] == 0


def test_mismatch_rebuild_paginates_across_batches_and_flips_only_at_end(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    ids = [
        tools.memory_write(content=f"条目 {i}：连接池为 {i}0。", subject=f"item{i}", tags=[])["data"]["id"]
        for i in range(6)
    ]
    assert tools.wait_semantic_worker_drained(timeout=5)
    tools.db.init_vec_index_state("new-space-v2", True)
    assert tools.db.get_vec_index_state()["state"] == "mismatch"
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "new-space-v2"

    # batch_size 2 over 6 memories: each execute call must select the NEXT
    # pending batch, never re-select the first one forever.
    seen: list[int] = []
    for _round in range(3):
        result = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 2})
        assert result["ok"] is True and result["data"]["queued"] == 2
        assert tools.wait_semantic_worker_drained(timeout=5)
        batch_ids = sorted(item["memory_id"] for item in result["data"]["results"])
        assert not set(batch_ids) & set(seen), "rebuild re-selected an already republished memory"
        seen.extend(batch_ids)
        if len(seen) < len(ids):
            assert tools.db.get_vec_index_state()["state"] == "mismatch"
    assert sorted(seen) == sorted(ids)
    state = tools.db.get_vec_index_state()
    assert state["state"] == "ready" and state["active_space_id"] == "new-space-v2"


def test_space_rebuild_epoch_ignores_same_second_old_rows(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="旧空间内容甲。", subject="a", tags=[])
    tools.memory_write(content="旧空间内容乙。", subject="b", tags=[])
    assert tools.wait_semantic_worker_drained(timeout=5)
    tools.db.init_vec_index_state("new-space-v2", True)
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "new-space-v2"

    # Start the rebuild (marks the evidence-id epoch) but republish only one
    # memory in the same second — old-space rows must NOT count as rebuilt.
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 1})
    assert tools.wait_semantic_worker_drained(timeout=5)
    assert tools.db.get_vec_index_state()["state"] == "mismatch"
    # An ordinary publish of the other memory completes the rebuild.
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 5})
    assert tools.wait_semantic_worker_drained(timeout=5)
    assert tools.db.get_vec_index_state()["state"] == "ready"


def test_rebuild_dry_run_has_no_side_effects_in_mismatch_mode(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="内容。", subject="s", tags=[])
    assert tools.wait_semantic_worker_drained(timeout=5)
    tools.db.init_vec_index_state("new-space-v2", True)
    preview = tools.memory_repair("rebuild_evidence", {"dry_run": True})
    assert preview["data"]["count"] >= 1
    with tools.db.connection() as conn:
        epoch = conn.execute(
            "SELECT value FROM _vec_index_meta WHERE key='space_rebuild_evidence_id'"
        ).fetchone()
        workspace_marker = conn.execute(
            "SELECT value FROM _vec_index_meta WHERE key='workspace_rebuild_space_id'"
        ).fetchone()
    assert epoch is None, "dry-run must not persist the rebuild epoch"
    assert workspace_marker is None, "dry-run must not rebuild workspace vectors"


def test_knn_truncates_to_requested_k_with_filters(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for i in range(4):
        tools.memory_write(content=f"postgres 条目 {i}：连接池说明 {i}。", subject=f"pg{i}", tags=[], workspace="w")
    assert tools.wait_semantic_worker_drained(timeout=5)
    hits = tools.db.row_knn([1.0, 0.0], k=3, workspace="w", exclude_memory_id=999999)
    assert 0 < len(hits) <= 3


def test_migrate_vnext_cli_exit_codes(tmp_path: Path, monkeypatch, capsys) -> None:
    from memory_arbiter.vnext_migration import run_cli

    # Dry-run on an unsupported source schema exits 2 with ok:false.
    bad = tmp_path / "bad.sqlite3"
    conn = sqlite3.connect(bad)
    conn.execute("CREATE TABLE memories(id INTEGER PRIMARY KEY, content TEXT)")
    conn.commit(); conn.close()
    assert run_cli(["--source", str(bad)]) == 2
    assert '"ok": false' in capsys.readouterr().out

    # Build that completes but cannot be checkpointed (switch_ready False)
    # also exits 2 on the execute path.
    good = tmp_path / "good.sqlite3"
    MemoryDB(Settings(db_path=good, backup_jsonl=tmp_path / "u.jsonl"))
    monkeypatch.setattr(
        "memory_arbiter.vnext_migration.build",
        lambda *a, **k: {"ok": True, "switch_ready": False, "target": "t"},
    )
    assert run_cli(["--source", str(good), "--execute"]) == 2
    capsys.readouterr()


def test_retarget_mid_rebuild_resets_epoch_and_republishes_everything(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = tools.memory_write(content="条目一内容。", subject="a", tags=[])["data"]["id"]
    second = tools.memory_write(content="条目二内容。", subject="b", tags=[])["data"]["id"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    # Swap A->B, rebuild one of two memories (partial), then retarget B->C.
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 1})
    assert tools.wait_semantic_worker_drained(timeout=5)
    assert tools.db.get_vec_index_state()["state"] == "mismatch"
    with tools.db.connection() as conn:
        b_phase_max = conn.execute("SELECT COALESCE(MAX(id),0) FROM memory_row").fetchone()[0]

    tools._embedder.embedding_space_id = "space-c"
    tools.db.init_vec_index_state("space-c", True)
    state = tools.db.get_vec_index_state()
    assert state["state"] == "mismatch" and state["target_space_id"] == "space-c"

    # Full C rebuild: BOTH memories must be republished above the old B rows
    # (a stale epoch would leave the first memory's B-space vectors in place
    # while still flipping to ready).
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert tools.wait_semantic_worker_drained(timeout=5)
    state = tools.db.get_vec_index_state()
    assert state["state"] == "ready" and state["active_space_id"] == "space-c"
    with tools.db.connection() as conn:
        for mid in (first, second):
            newest = conn.execute(
                "SELECT COALESCE(MAX(id),0) FROM memory_row WHERE memory_id=?",
                (mid,),
            ).fetchone()[0]
            assert newest > b_phase_max, "memory not republished after retarget"


def test_zero_unit_memory_does_not_block_space_rebuild_flip(tmp_path: Path) -> None:
    from memory_arbiter.models import utc_now_iso

    tools = make_tools(tmp_path)
    tools.memory_write(content="正常内容。", subject="s", tags=[])
    assert tools.wait_semantic_worker_drained(timeout=5)
    # A legacy/imported row with no indexable text (product writes validate
    # non-blank content, so reach around them with raw SQL).
    with tools.db.write_transaction() as conn:
        conn.execute(
            """INSERT INTO memories(content, agent_id, workspace, tags, source_type,
               event_time, ingest_time, status, subject, metadata, version, created_at)
               VALUES (' ', 'agent', 'default', '[]', 'agent_generated', ?, ?, 'active',
                       NULL, '{}', 1, ?)""",
            (utc_now_iso(), utc_now_iso(), utc_now_iso()),
        )

    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    result = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert tools.wait_semantic_worker_drained(timeout=5)
    # The zero-unit memory must not stay pending forever: after the batch the
    # next execute selects nothing new and settles the flip immediately.
    again = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert again["data"]["queued"] == 0
    state = tools.db.get_vec_index_state()
    assert state["state"] == "ready" and state["active_space_id"] == "space-b"


def test_backfill_phase_target_is_not_current_generation(tmp_path: Path) -> None:
    from memory_arbiter.db_generation import detect_database_generation

    path = tmp_path / "staging.sqlite3"
    MemoryDB(Settings(db_path=path, backup_jsonl=tmp_path / "u.jsonl"))
    with contextlib.closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute(
                "INSERT INTO migration_state(key, value) VALUES ('phase', 'backfill')"
            )
    # A build that crashed mid-backfill must refuse to start as current.
    assert detect_database_generation(path) == "unknown"


def test_empty_batch_execute_settles_flip_without_unrelated_write(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="内容。", subject="s", tags=[])
    assert tools.wait_semantic_worker_drained(timeout=5)
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    # Everything already republished above an epoch=0 (constructed state):
    # pending is empty, and the execute path itself must settle the flip.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO _vec_index_meta(key, value) VALUES ('space_rebuild_evidence_id', '0') "
            "ON CONFLICT(key) DO UPDATE SET value='0'"
        )
    result = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert result["data"]["queued"] == 0
    state = tools.db.get_vec_index_state()
    assert state["state"] == "ready" and state["active_space_id"] == "space-b"


def test_unicode_whitespace_memory_does_not_block_space_rebuild_flip(tmp_path: Path) -> None:
    # SQLite TRIM strips only ASCII spaces while the extractor's whitespace
    # semantics are Python's; rows blank only in the Python sense (tab
    # subject, U+3000 content) must be excluded from every pending set
    # instead of staying pending forever and blocking the flip.
    from memory_arbiter.models import utc_now_iso

    tools = make_tools(tmp_path)
    real_id = tools.memory_write(content="正常内容。", subject="s", tags=[])["data"]["id"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    with tools.db.write_transaction() as conn:
        conn.execute(
            """INSERT INTO memories(content, agent_id, workspace, tags, source_type,
               event_time, ingest_time, status, subject, metadata, version, created_at)
               VALUES ('\u3000\u3000', 'agent', 'default', '[]', 'agent_generated', ?, ?, 'active',
                       '\t', '{}', 1, ?)""",
            (utc_now_iso(), utc_now_iso(), utc_now_iso()),
        )

    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    dry = tools.memory_repair("rebuild_evidence", {"dry_run": True, "batch_size": 10})
    assert dry["data"]["memory_ids"] == [real_id]
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert tools.wait_semantic_worker_drained(timeout=5)
    again = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert again["data"]["queued"] == 0
    state = tools.db.get_vec_index_state()
    assert state["state"] == "ready" and state["active_space_id"] == "space-b"
    coverage = tools.db.evidence.coverage()
    assert coverage["eligible_memories"] == 1
    assert coverage["indexed_memories"] == 1
    assert coverage["non_indexable_memories"] == 1


def test_ready_revert_mid_rebuild_purges_foreign_space_rows(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    first = tools.memory_write(content="条目一内容。", subject="a", tags=[])["data"]["id"]
    second = tools.memory_write(content="条目二内容。", subject="b", tags=[])["data"]["id"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    with tools.db.connection() as conn:
        second_rows = conn.execute(
            "SELECT COUNT(*) FROM memory_row WHERE memory_id=?", (second,)
        ).fetchone()[0]
    assert second_rows > 0

    # Partial rebuild into space-b: `first` is republished there (its
    # original rows are dropped by publish's delete-then-insert).
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 1})
    assert tools.wait_semantic_worker_drained(timeout=5)
    with tools.db.connection() as conn:
        epoch = int(conn.execute(
            "SELECT value FROM _vec_index_meta WHERE key='space_rebuild_evidence_id'"
        ).fetchone()["value"])

    # Revert the embedding model to the active space mid-rebuild: the
    # foreign-space rows above the epoch must not survive in a ready channel.
    tools._embedder = FakeEmbedder()
    tools.db.init_vec_index_state("fake-vnext-space", True)
    state = tools.db.get_vec_index_state()
    assert state["state"] == "ready" and state["active_space_id"] == "fake-vnext-space"
    with tools.db.connection() as conn:
        first_rows = conn.execute(
            "SELECT COUNT(*) FROM memory_row WHERE memory_id=?", (first,)
        ).fetchone()[0]
        second_now = conn.execute(
            "SELECT COUNT(*) FROM memory_row WHERE memory_id=?", (second,)
        ).fetchone()[0]
        max_second_id = conn.execute(
            "SELECT COALESCE(MAX(id),0) FROM memory_row WHERE memory_id=?", (second,)
        ).fetchone()[0]
        vec_orphans = conn.execute(
            "SELECT COUNT(*) FROM memory_row_vec v "
            "WHERE NOT EXISTS(SELECT 1 FROM memory_row e WHERE e.id=v.id)"
        ).fetchone()[0]
    assert first_rows == 0  # purged B-space rows; republished on next rebuild
    assert second_now == second_rows and max_second_id <= epoch
    assert vec_orphans == 0

    # The purged memory resurfaces as a stale candidate and re-embeds in the
    # active space.
    dry = tools.memory_repair("rebuild_evidence", {"dry_run": True, "batch_size": 10})
    assert dry["data"]["memory_ids"] == [first]
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert tools.wait_semantic_worker_drained(timeout=5)
    coverage = tools.db.evidence.coverage()
    assert coverage["indexed_memories"] == coverage["eligible_memories"] == 2


def test_rebuild_evidence_response_surfaces_vec_index_state(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="内容。", subject="s", tags=[])
    assert tools.wait_semantic_worker_drained(timeout=5)

    # Empty pending set with a live embedder: the execute settles the flip
    # and the response must say so.
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO _vec_index_meta(key, value) VALUES ('space_rebuild_evidence_id', '0') "
            "ON CONFLICT(key) DO UPDATE SET value='0'"
        )
    result = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert result["data"]["queued"] == 0
    assert result["data"]["vec_index_state"]["state"] == "ready"

    # Same empty pending set without an embedder: the flip cannot settle
    # (the target space is unverifiable) and queued=0 must not read as success.
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-c"
    tools.db.init_vec_index_state("space-c", True)
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO _vec_index_meta(key, value) VALUES ('space_rebuild_evidence_id', '0') "
            "ON CONFLICT(key) DO UPDATE SET value='0'"
        )
    tools._embedder = None
    tools._embedder_loaded = True
    result = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert result["data"]["queued"] == 0
    assert result["data"]["vec_index_state"]["state"] == "mismatch"



def test_migration_resume_recovers_from_failed_build(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("sqlite_vec")
    # Legacy-shaped source database with three memories.
    source = tmp_path / "legacy.sqlite3"
    db = MemoryDB(Settings(db_path=source, backup_jsonl=tmp_path / "b.jsonl"))
    with db.write_transaction() as conn:
        for i in range(3):
            conn.execute(
                """INSERT INTO memories(content, agent_id, workspace, tags, source_type,
                   event_time, ingest_time, status, subject, metadata, version, created_at)
                   VALUES (?, 'agent', 'default', '[]', 'agent_generated', '2026-01-01T00:00:00+00:00',
                           '2026-01-01T00:00:00+00:00', 'active', ?, '{}', 1, '2026-01-01T00:00:00+00:00')""",
                (f"迁移样本 {i} 内容。", f"mig{i}"),
            )
    target = tmp_path / "new.vnext.sqlite3"

    real_index = MemoryTools._index_local_text_evidence
    calls = {"n": 0}

    def flaky(self, memory_id, record=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            return {"status": "failed", "reason": "injected"}
        return real_index(self, memory_id, record, **kwargs)

    settings = Settings(
        db_path=source, backup_jsonl=tmp_path / "b.jsonl",
        embedding_model_path=tmp_path / "fake.gguf",
    )
    (tmp_path / "fake.gguf").write_bytes(b"fake")
    # build() constructs its own MemoryTools with a real embedder build;
    # inject the test FakeEmbedder so indexing works without a real GGUF.
    real_init = MemoryTools.__init__

    def init_with_fake_embedder(self, settings=None, db=None):
        real_init(self, settings=settings, db=db)
        self._embedder = FakeEmbedder()
        self._embedder_loaded = True
        self.db.ensure_vec_tables(FakeEmbedder.dim)
        self.db.init_vec_index_state(
            self._embedder.embedding_space_id, True, active_dim=FakeEmbedder.dim,
        )

    monkeypatch.setattr(MemoryTools, "__init__", init_with_fake_embedder)
    monkeypatch.setattr(MemoryTools, "_index_local_text_evidence", flaky)
    failed_build = build(source, target, settings, resume=False, progress=False)
    assert failed_build["ok"] is False
    with sqlite3.connect(target) as conn:
        phase = conn.execute("SELECT value FROM migration_state WHERE key='phase'").fetchone()[0]
    assert phase == "failed"

    # --resume must be able to open the failed target and finish the build.
    monkeypatch.setattr(MemoryTools, "_index_local_text_evidence", real_index)
    resumed = build(source, target, settings, resume=True, progress=False)
    assert resumed["ok"] is True, resumed
    assert resumed["switch_ready"] is True
    with sqlite3.connect(target) as conn:
        phase = conn.execute("SELECT value FROM migration_state WHERE key='phase'").fetchone()
        completed_at = conn.execute(
            "SELECT value FROM migration_state WHERE key='migration_completed_at'"
        ).fetchone()
    assert phase is None
    assert completed_at is not None


def test_migration_resume_with_changed_space_restarts_derived_index(
    tmp_path: Path, monkeypatch,
) -> None:
    pytest.importorskip("sqlite_vec")
    source = tmp_path / "legacy.sqlite3"
    source_db = MemoryDB(Settings(db_path=source, backup_jsonl=tmp_path / "b.jsonl"))
    for index in range(3):
        source_db.insert_memory(
            MemoryRecord.from_input(
                {"content": f"内容 {index}", "subject": f"s{index}", "workspace": "project"},
                source_db.settings.defaults(),
            ),
            "project",
        )
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=source, backup_jsonl=tmp_path / "b.jsonl",
        embedding_model_path=model,
    )
    target = tmp_path / "target.sqlite3"
    original_init = MemoryTools.__init__
    calls = {"count": 0}

    def init_old(self, settings=None, db=None):
        original_init(self, settings=settings, db=db)
        embedder = FakeEmbedder()
        embedder.embedding_space_id = "space-old"
        self._embedder = embedder
        self._embedder_loaded = True
        self.db.ensure_vec_tables(FakeEmbedder.dim)
        self.db.init_vec_index_state(embedder.embedding_space_id, True, active_dim=FakeEmbedder.dim)

    real_index = MemoryTools._index_local_text_evidence

    def fail_second(self, memory_id, record=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            return {"status": "failed", "reason": "injected"}
        return real_index(self, memory_id, record, **kwargs)

    monkeypatch.setattr(MemoryTools, "__init__", init_old)
    monkeypatch.setattr(MemoryTools, "_index_local_text_evidence", fail_second)
    first = build(source, target, settings, progress=False)
    assert first["ok"] is False

    def init_new(self, settings=None, db=None):
        original_init(self, settings=settings, db=db)
        embedder = FakeEmbedder()
        embedder.embedding_space_id = "space-new"
        self._embedder = embedder
        self._embedder_loaded = True
        self.db.ensure_vec_tables(FakeEmbedder.dim)
        self.db.init_vec_index_state(embedder.embedding_space_id, True, active_dim=FakeEmbedder.dim)

    monkeypatch.setattr(MemoryTools, "__init__", init_new)
    monkeypatch.setattr(MemoryTools, "_index_local_text_evidence", real_index)
    resumed = build(source, target, settings, resume=True, progress=False)

    assert resumed["ok"] is True, resumed
    assert resumed["indexed"] == 3
    with sqlite3.connect(target) as conn:
        state = dict(conn.execute("SELECT key,value FROM _vec_index_meta"))
        migration = dict(conn.execute("SELECT key,value FROM migration_state"))
    assert state["state"] == "ready"
    assert state["active_space_id"] == "space-new"
    assert "evidence_rebuild_space_id" not in migration
    assert "phase" not in migration
    assert migration["migration_completed_at"]


def test_migration_inspect_accepts_missing_parent_dir(tmp_path: Path) -> None:
    source = tmp_path / "legacy.sqlite3"
    db = MemoryDB(Settings(db_path=source, backup_jsonl=tmp_path / "b.jsonl"))
    with db.write_transaction() as conn:
        conn.execute(
            """INSERT INTO memories(content, agent_id, workspace, tags, source_type,
               event_time, ingest_time, status, subject, metadata, version, created_at)
               VALUES ('内容。', 'agent', 'default', '[]', 'agent_generated',
                       '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',
                       'active', 's', '{}', 1, '2026-01-01T00:00:00+00:00')""",
        )
    settings = Settings(
        db_path=source, backup_jsonl=tmp_path / "b.jsonl",
    )
    plan = inspect(source, tmp_path / "nested" / "dir" / "new.vnext.sqlite3", settings)
    assert plan.get("ok") is not False
    assert "disk_ok" in plan


def test_wave9_resuming_phase_refused_and_foreign_targets_clean(tmp_path: Path) -> None:
    from memory_arbiter.db_generation import detect_database_generation

    # M1: a kill -9 inside the resume window must not leave the incomplete
    # target openable as "current".
    source = tmp_path / "legacy.sqlite3"
    db = MemoryDB(Settings(db_path=source, backup_jsonl=tmp_path / "b.jsonl"))
    with db.write_transaction() as conn:
        conn.execute(
            """INSERT INTO memories(content, agent_id, workspace, tags, source_type,
               event_time, ingest_time, status, subject, metadata, version, created_at)
               VALUES ('x。', 'agent', 'default', '[]', 'agent_generated',
                       '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',
                       'active', 's', '{}', 1, '2026-01-01T00:00:00+00:00')""",
        )
    target = tmp_path / "t.vnext.sqlite3"
    settings = Settings(
        db_path=source, backup_jsonl=tmp_path / "b.jsonl",
    )
    built = build(source, target, settings, resume=False, progress=False)
    assert built["ok"] is False or built["ok"] is True  # indexing may skip without embedder
    with sqlite3.connect(target) as conn:
        conn.execute("INSERT INTO migration_state(key, value) VALUES ('phase', 'resuming') "
                     "ON CONFLICT(key) DO UPDATE SET value='resuming'")
    assert detect_database_generation(target) == "unknown"
    with pytest.raises(RuntimeError):
        MemoryDB(Settings(db_path=target, backup_jsonl=tmp_path / "b2.jsonl"))

    # M2: resume against foreign/empty targets returns clean errors.
    empty = tmp_path / "empty.sqlite3"
    empty.write_bytes(b"")
    foreign = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(foreign) as conn:
        conn.execute("CREATE TABLE unrelated(x)")
    for bad_target, expected in ((empty, None), (foreign, "target_not_a_vnext_database")):
        result = build(source, bad_target, settings, resume=True, progress=False)
        if expected is None:
            assert result.get("error") != "target_not_a_vnext_database" or result.get("ok") is False
        else:
            assert result["ok"] is False
            assert expected in result.get("error", "")