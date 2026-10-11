# ── from test_conflict_groups.py ──

from __future__ import annotations

import sqlite3

from pathlib import Path

import pytest
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.semantic_conflict import (
    decide_evidence, normalize_value, notice_dedupe_key,
)
from memory_arbiter.tools import MemoryTools
from memory_arbiter.db_generation import (
    CONFLICT_DETECTOR_VERSION,
    CURRENT_SCHEMA_GENERATION,
    detect_database_generation,
)
from memory_arbiter.evidence import evidence_content_hash
from memory_arbiter.rowseg import segment_rows
from memory_arbiter.vnext_migration import (
    _configured_embedding_space_id,
    build_conflict_only,
    _copy_preserved_tables,
    _fingerprint,
    _mark_conflict_rebuild_ready,
    inspect,
)


def _mark_current_space(db: MemoryDB, settings: Settings) -> str:
    pytest.importorskip("sqlite_vec")
    # Lazy vec-table semantics (0.15.0): the vec0 tables exist only after the
    # first successful embedder build. Hand-injecting vectors below mirrors
    # that build: create the tables at dim 2, then record the same-dim space.
    assert db.ensure_vec_tables(2) == []
    space_id = _configured_embedding_space_id(settings, 2)
    assert space_id is not None
    with db.connection() as conn:
        memories = [dict(row) for row in conn.execute(
            "SELECT id,version,content,subject,status FROM memories WHERE status!='deleted'"
        )]
    for memory in memories:
        # 0.17.0 C5: rows are the evidence channel — publish_rows is the one
        # true publisher (unit tables are never written).
        rows = segment_rows(
            str(memory.get("subject") or ""), str(memory.get("content") or ""),
        )
        embeddings = [[0.0, 1.0] for _row in rows]
        published = db.evidence.publish_rows(
            int(memory["id"]), int(memory.get("version") or 1),
            evidence_content_hash(str(memory.get("content") or "")), rows, embeddings,
        )
        assert published.get("published") is True
    with db.connection() as conn:
        canonicals = [
            str(row["name"])
            for row in conn.execute("SELECT name FROM workspace_canonicals ORDER BY id")
            if str(row["name"] or "").casefold() != "default"
        ]
    for canonical in canonicals:
        assert db.workspaces.publish_workspace_canonical_vector(
            canonical, [0.0, 1.0],
        ) == []
    with db.write_transaction() as conn:
        conn.execute("DELETE FROM _vec_index_meta")
        conn.executemany(
            "INSERT INTO _vec_index_meta(key,value) VALUES(?,?)",
            (("state", "ready"), ("active_space_id", space_id)),
        )
    return space_id

def _settings(path: Path, tmp_path: Path) -> Settings:
    return Settings(db_path=path, backup_jsonl=tmp_path / "backup.jsonl")


def _same_space_settings(path: Path, tmp_path: Path) -> Settings:
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"same-space-model")
    return Settings(
        db_path=path,
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
    )


def _upgrade_memory(defaults: dict[str, object]) -> MemoryRecord:
    return MemoryRecord.from_input(
        {"content": "数据库使用 SQLite。", "subject": "数据库选型", "workspace": "project"},
        defaults,
    )


def test_destructive_copy_preserves_core_data_but_not_conflict_history(tmp_path: Path) -> None:
    source_path = tmp_path / "source.sqlite3"
    target_path = tmp_path / "target.sqlite3"
    source = MemoryDB(_settings(source_path, tmp_path))
    memory_id, _ = source.insert_memory(_upgrade_memory(_settings(source_path, tmp_path).defaults()), "project")
    assert memory_id is not None
    with source.write_transaction() as conn:
        conn.execute(
            "INSERT INTO memory_history(memory_id,content_snapshot,subject_snapshot,tags_snapshot,version,changed_at,reason) VALUES(?,?,?,?,?,?,?)",
            (memory_id, "old", "数据库选型", "[]", 1, "2026-08-20T00:00:00Z", "test"),
        )
        conn.execute(
            "INSERT INTO workspace_aliases(alias_workspace,canonical,status,updated_at) VALUES(?,?,?,?)",
            ("proj", "project", "confirmed", "2026-08-20T00:00:00Z"),
        )
        # Removed tables may exist in a previous-generation source even though
        # current runtime schema intentionally does not create them.
        conn.execute("CREATE TABLE semantic_notices(id INTEGER PRIMARY KEY, message TEXT)")
        conn.execute("INSERT INTO semantic_notices VALUES(1,'legacy notice')")
        conn.execute("CREATE TABLE conflict_judgments(id INTEGER PRIMARY KEY, reason TEXT)")
        conn.execute("INSERT INTO conflict_judgments VALUES(1,'legacy judgment')")
        conn.execute(
            "INSERT INTO backup_replay_log("
            "replay_key,memory_id,payload_hash,replayed_at,postprocess_status,"
            "postprocess_stages,postprocess_error_code) VALUES(?,?,?,?,?,?,?)",
            (
                "receipt-1", memory_id, "payload-hash", "2026-08-20T00:00:00Z",
                "failed", '{"evidence":"complete","semantic":"failed"}', "model_timeout",
            ),
        )

    target = MemoryDB(_settings(target_path, tmp_path))
    _copy_preserved_tables(source_path, target)

    with target.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM memory_history").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM workspace_aliases").fetchone()[0] == 1
        receipt = conn.execute(
            "SELECT replay_key,memory_id,payload_hash,replayed_at,postprocess_status,"
            "postprocess_stages,postprocess_error_code FROM backup_replay_log"
        ).fetchone()
        assert tuple(receipt) == (
            "receipt-1", memory_id, "payload-hash", "2026-08-20T00:00:00Z",
            "failed", '{"evidence":"complete","semantic":"failed"}', "model_timeout",
        )
        assert conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 0
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_notices'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflict_judgments'"
        ).fetchone() is None
    assert _fingerprint(source_path) == _fingerprint(target_path)


def test_current_startup_does_not_run_schema_migrations(tmp_path: Path) -> None:
    path = tmp_path / "legacy-alias.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_new")
        conn.execute(
            "CREATE TABLE workspace_aliases(alias_workspace TEXT NOT NULL UNIQUE, canonical TEXT NOT NULL, relation TEXT NOT NULL, status TEXT NOT NULL, source TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO workspace_aliases VALUES('raw','candidate-a','alias','rejected','user','2026-08-20T00:00:00Z')"
        )
        conn.execute("DROP TABLE workspace_aliases_new")
    reopened = MemoryDB(_settings(path, tmp_path))
    assert reopened.db_available is True
    with reopened.connection() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
    assert columns == [
        "alias_workspace", "canonical", "relation", "status", "source", "updated_at",
    ]


def test_startup_defers_partial_table_health_to_doctor_or_runtime(tmp_path: Path) -> None:
    path = tmp_path / "partial.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_good")
        conn.execute(
            "CREATE TABLE workspace_aliases(alias_workspace TEXT,canonical TEXT,updated_at TEXT)"
        )
        conn.execute("INSERT INTO workspace_aliases VALUES('raw','target','2026-01-01')")
        conn.execute("DROP TABLE workspace_aliases_good")
    reopened = MemoryDB(_settings(path, tmp_path))
    assert reopened.db_available is True
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspace_aliases").fetchone()[0] == 1
        columns = [row[1] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
    assert columns == ["alias_workspace", "canonical", "updated_at"]


def test_startup_defers_invalid_compact_rows_to_doctor_or_runtime(tmp_path: Path) -> None:
    path = tmp_path / "invalid.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_good")
        conn.execute(
            "CREATE TABLE workspace_aliases("
            "alias_workspace TEXT,canonical TEXT,status TEXT,updated_at TEXT,"
            "PRIMARY KEY(alias_workspace,canonical))"
        )
        conn.execute("INSERT INTO workspace_aliases VALUES('raw','target','bogus','2026-01-01')")
        conn.execute("DROP TABLE workspace_aliases_good")
    reopened = MemoryDB(_settings(path, tmp_path))
    assert reopened.db_available is True
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT status FROM workspace_aliases").fetchone()
    assert row[0] == "bogus"


def test_startup_does_not_repair_interrupted_table_migration(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    assert db.record_workspace_decision("raw", "target")[0]
    with db.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_legacy")
    reopened = MemoryDB(_settings(path, tmp_path))
    assert reopened.db_available is True
    assert reopened.resolve_workspace_canonical("raw", None)["canonical"] == "raw"
    with reopened.connection() as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='workspace_aliases_legacy'"
        ).fetchone() is not None


def test_full_build_copy_failure_never_leaves_current_target(
    tmp_path: Path, monkeypatch,
) -> None:
    from memory_arbiter import vnext_migration

    source_path = tmp_path / "source-current.sqlite3"
    target_path = tmp_path / "target-partial.sqlite3"
    MemoryDB(_settings(source_path, tmp_path))

    def fail_copy(*args, **kwargs):
        raise sqlite3.IntegrityError("injected copy failure")

    monkeypatch.setattr(vnext_migration, "_copy_preserved_tables", fail_copy)
    with pytest.raises(sqlite3.IntegrityError):
        vnext_migration.build(
            source_path, target_path, _settings(source_path, tmp_path), progress=False,
        )
    assert detect_database_generation(target_path) != "current"
    with sqlite3.connect(target_path) as conn:
        state = dict(conn.execute("SELECT key,value FROM migration_state"))
    assert state["phase"] == "failed"
    assert state["schema_generation"].endswith(":building")


def test_full_build_checkpoint_failure_marks_target_failed(
    tmp_path: Path, monkeypatch,
) -> None:
    from memory_arbiter import vnext_migration

    source_path = tmp_path / "source-empty.sqlite3"
    target_path = tmp_path / "target-checkpoint.sqlite3"
    MemoryDB(_settings(source_path, tmp_path))
    monkeypatch.setattr(vnext_migration, "_checkpoint", lambda _path: False)
    result = vnext_migration.build(
        source_path, target_path, _settings(source_path, tmp_path), progress=False,
    )
    assert result["ok"] is False
    assert result["switch_ready"] is False
    assert detect_database_generation(target_path) != "current"
    with sqlite3.connect(target_path) as conn:
        state = dict(conn.execute("SELECT key,value FROM migration_state"))
    assert state["phase"] == "failed"


def test_previous_generation_is_refused_until_destructive_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "previous.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' WHERE key='schema_generation'"
        )
    assert CURRENT_SCHEMA_GENERATION != "local_text_evidence_v1"
    assert detect_database_generation(path) == "legacy"


def test_previous_conflict_generation_is_refused_until_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "conflict-v2.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    with db.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='conflict_groups_v2' "
            "WHERE key='schema_generation'"
        )
    assert detect_database_generation(path) == "legacy"


def test_conflict_only_upgrade_compacts_workspace_state_and_drops_events(tmp_path: Path) -> None:
    source_path = tmp_path / "conflict-v2.sqlite3"
    target_path = tmp_path / "workspace-v1.sqlite3"
    migration_settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(migration_settings)
    assert source.record_workspace_decision("raw", "target")[0]
    with source.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_compact")
        conn.execute(
            "CREATE TABLE workspace_aliases("
            "alias_workspace TEXT NOT NULL,canonical TEXT NOT NULL,relation TEXT NOT NULL,"
            "status TEXT NOT NULL,source TEXT NOT NULL,updated_at TEXT NOT NULL,"
            "PRIMARY KEY(alias_workspace,canonical))"
        )
        conn.execute(
            "INSERT INTO workspace_aliases SELECT alias_workspace,canonical,'alias',"
            "status,'user',updated_at FROM workspace_aliases_compact"
        )
        conn.execute("DROP TABLE workspace_aliases_compact")
        conn.execute(
            "CREATE TABLE workspace_alias_events("
            "id INTEGER PRIMARY KEY,alias_workspace TEXT,action TEXT)"
        )
        conn.execute("INSERT INTO workspace_alias_events VALUES(1,'raw','accept')")
        conn.execute(
            "UPDATE migration_state SET value='conflict_groups_v2' "
            "WHERE key='schema_generation'"
        )
    _mark_current_space(source, migration_settings)
    result = build_conflict_only(source_path, target_path, migration_settings)
    assert result["ok"] is True, result
    with sqlite3.connect(target_path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
        events = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='workspace_alias_events'"
        ).fetchone()
    assert columns == ["alias_workspace", "canonical", "status", "updated_at"]
    assert events is None
    assert detect_database_generation(target_path) == "current"


def test_previous_evidence_generation_rebuilds_only_conflict_domain(tmp_path: Path) -> None:
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    memory_id, _ = source.insert_memory(_upgrade_memory(settings.defaults()), "project")
    assert memory_id is not None
    _mark_current_space(source, settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )
    before = _fingerprint(source_path)

    result = build_conflict_only(source_path, target_path, settings)

    assert result["ok"] is True
    assert result["indexed"] == 0
    assert result["evidence_reused"] is True
    assert result["vector_effect"] == "preserve"
    assert result["vec_index_state"]["state"] == "ready"
    assert result["source_fingerprint"] == result["target_fingerprint"] == before
    assert detect_database_generation(target_path) == "current"
    with sqlite3.connect(target_path) as conn:
        state = dict(conn.execute("SELECT key,value FROM migration_state"))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(conflicts)")}
        assert conn.execute("SELECT COUNT(*) FROM memory_row").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 0
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_notices'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflict_judgments'"
        ).fetchone() is None
    assert {"revision", "member_versions", "value_groups", "notice_type"} <= columns
    assert state["schema_generation"] == CURRENT_SCHEMA_GENERATION
    assert state["migration_completed_at"]
    assert "phase" not in state
    assert state["source_path"] == str(source_path)
    assert state["conflict_scan_required"] == "true"


def test_previous_generation_with_old_embedding_space_preserves_then_marks_mismatch(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    source_settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(source_settings)
    memory_id, _ = source.insert_memory(_upgrade_memory(source_settings.defaults()), "project")
    assert memory_id is not None
    _mark_current_space(source, source_settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )
        conn.execute("DELETE FROM _vec_index_meta")
        conn.executemany(
            "INSERT INTO _vec_index_meta(key,value) VALUES(?,?)",
            (("state", "ready"), ("active_space_id", "old-pipeline-space")),
        )
        source_vectors = [
            tuple(row) for row in conn.execute(
                "SELECT id,parent_status,hex(embedding) FROM memory_row_vec ORDER BY id"
            ).fetchall()
        ]
    settings = source_settings

    plan = inspect(source_path, target_path, settings)
    migrated = build_conflict_only(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"
    assert plan["vector_compatibility"] == "mismatch"
    assert migrated["ok"] is True
    assert migrated["vec_index_state"]["state"] == "mismatch"
    with sqlite3.connect(target_path) as conn:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        state = dict(conn.execute("SELECT key,value FROM _vec_index_meta"))
        target_vectors = conn.execute(
            "SELECT id,parent_status,hex(embedding) FROM memory_row_vec ORDER BY id"
        ).fetchall()
    assert state["state"] == "mismatch"
    assert state["active_space_id"] == "old-pipeline-space"
    assert state["target_space_id"] == plan["configured_space_id"]
    assert target_vectors == source_vectors


def test_previous_generation_without_managed_space_still_preserves_vectors(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _settings(source_path, tmp_path)
    source = MemoryDB(settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    plan = inspect(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"
    assert plan["vector_compatibility"] == "unmanaged"


def test_previous_generation_with_incomplete_same_space_index_defers_to_doctor(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    memory_id, _ = source.insert_memory(_upgrade_memory(settings.defaults()), "project")
    assert memory_id is not None
    _mark_current_space(source, settings)
    with source.write_transaction() as conn:
        row_id = conn.execute(
            "SELECT id FROM memory_row WHERE memory_id=? ORDER BY id LIMIT 1",
            (memory_id,),
        ).fetchone()[0]
        conn.execute("DELETE FROM memory_row_vec WHERE id=?", (row_id,))
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    plan = inspect(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"


def test_previous_generation_with_stale_same_space_evidence_defers_to_doctor(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    memory_id, _ = source.insert_memory(_upgrade_memory(settings.defaults()), "project")
    assert memory_id is not None
    _mark_current_space(source, settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET content='changed without republish',version=version+1 WHERE id=?",
            (memory_id,),
        )
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    plan = inspect(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"


def test_conflict_only_validation_failure_marks_phase_failed(tmp_path: Path, monkeypatch) -> None:
    """A conflict-only rebuild whose post-transaction validation fails must not
    leave a phase=ready/current target that detect_database_generation trusts."""
    import memory_arbiter.vnext_migration as vm

    source = tmp_path / "source.sqlite3"
    settings = _same_space_settings(source, tmp_path)
    db = MemoryDB(settings)
    db.insert_memory(_upgrade_memory(settings.defaults()), "project")
    _mark_current_space(db, settings)
    # Move it to the previous generation so build_conflict_only is the path.
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO migration_state(key,value,updated_at) VALUES('schema_generation',?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
            ("local_text_evidence_v1",),
        )

    # Force validation to fail after the rebuild transaction commits.
    monkeypatch.setattr(vm, "_checkpoint", lambda *_a, **_k: False)
    target = tmp_path / "target.sqlite3"
    result = build_conflict_only(source, target, settings)
    assert result["ok"] is False
    # The stranded artifact is marked failed, so it is not classified current.
    assert detect_database_generation(target) != "current"
    with sqlite3.connect(target) as conn:
        conn.row_factory = sqlite3.Row
        phase = conn.execute("SELECT value FROM migration_state WHERE key='phase'").fetchone()
    assert phase is not None and phase["value"] == "failed"


def test_conflict_only_does_not_turn_post_publish_space_drift_into_schema_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    import memory_arbiter.vnext_migration as vm

    source_path = tmp_path / "source.sqlite3"
    target_path = tmp_path / "target.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    _mark_current_space(source, settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    original_checkpoint = vm._checkpoint

    def corrupt_after_checkpoint(path: Path) -> bool:
        checkpointed = original_checkpoint(path)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "UPDATE _vec_index_meta SET value='foreign-space' "
                "WHERE key='active_space_id'"
            )
        return checkpointed

    monkeypatch.setattr(vm, "_checkpoint", corrupt_after_checkpoint)

    result = build_conflict_only(source_path, target_path, settings)

    assert result["ok"] is True
    assert detect_database_generation(target_path) == "current"
    with sqlite3.connect(target_path) as conn:
        phase = conn.execute(
            "SELECT value FROM migration_state WHERE key='phase'"
        ).fetchone()
    assert phase is None


# ── from test_semantic_process.py ──


import os
import threading
import time
from pathlib import Path
from tests._helpers import _db, _member, _memory, _record



def _blocking_child(conn, config):
    try:
        request = conn.recv()
        if request.get("command") == "load":
            conn.send({"ok": True, "result": {"loaded": True}})
            conn.recv()
        time.sleep(10)
    except (EOFError, OSError):
        return


def test_bare_candidate_promoted_when_rerecorded_as_open(tmp_path: Path) -> None:
    # mema 924 #2（Team 93d17fc 移植）：裸 candidate（无 notice、无 slot）撞上后续
    # 同事件 open 记录时必须就地提升，而不是返回 deduped 把事件永久卡死在 candidate。
    db = _db(tmp_path)
    m1, m2 = _memory(db, "database is sqlite"), _memory(db, "database is postgres")
    a, b = _member(m1, "sqlite"), _member(m2, "postgres")
    created = _record(db, [a, b])
    assert created["outcome"] == "inserted", created
    cid = int(created["conflict_id"])
    # 模拟任一生产者把该事件冻结为裸 candidate：status 降 candidate、slot/notice 清空。
    with db.connection() as conn:
        conn.execute(
            "UPDATE conflicts SET status='candidate', slot_key=NULL, slot_key_hash=NULL, notice_type=NULL WHERE id=?",
            (cid,),
        )
        conn.commit()
    with db.connection() as conn:
        row = conn.execute("SELECT status, slot_key, notice_type FROM conflicts WHERE id=?", (cid,)).fetchone()
    assert row["status"] == "candidate" and row["slot_key"] is None and row["notice_type"] is None

    op = _record(db, [a, b])
    assert op["outcome"] == "deduped", op
    assert int(op["conflict_id"]) == cid
    assert int(op["revision"]) == int(created["revision"]) + 1
    with db.connection() as conn:
        row2 = conn.execute("SELECT status, slot_key FROM conflicts WHERE id=?", (cid,)).fetchone()
    assert row2["status"] == "open", f"candidate row was not promoted: {dict(row2)}"
    assert row2["slot_key"] is not None


def test_pre_qwen_vetoes_ops_marker_and_decision_vs_observation() -> None:
    """2026-09-17 (owner): 进 Qwen 前的两个 ignore veto。
    ops_marker 拦运维标记（时间戳尾 slug / test marker 措辞）；
    decision_vs_observation 拦决策措辞×实测措辞的来源不对称
    （owner 原则：证据 vs 裁决不构成对立）。正样本与 owner 裁定为真
    冲突的 conflict-test 基准对必须保活。"""
    from memory_arbiter.semantic_conflict import decide_evidence
    # 目标命中
    assert decide_evidence(
        "JingleAI global bridge self-test marker jingleai-global-bridge-self-test-1783837684.",
        "JingleAI global bridge full-test marker jingleai-global-bridge-fulltest-1783837846.",
    ).reason == "ops_marker"
    assert decide_evidence(
        "用户进一步修正 weak 归一策略：Qwen 高置信的直接静默处理。",
        "2026-08-08 按用户最新原则测试 weak/strict 策略：样本扩展到 54 条。",
    ).reason == "decision_vs_observation"
    # owner 示例：实测结论 vs 用户确认
    assert decide_evidence("用户确认上限 5000 元。", "实测结论上限 500 元。").reason \
        == "decision_vs_observation"
    # 保活：conflict-test 基准对（owner 裁定为真冲突，marker 词面刻意不匹配）
    d = decide_evidence("conflict-test 的 export-format 取值为 json。",
                        "conflict-test 的 export-format 取值为 csv。")
    assert d.action == "check"
    # 保活：数值正样本不受 veto 影响
    assert decide_evidence("单笔退款超过 5000 元需要财务复核。",
                           "单笔退款超过 500 元就需要财务复核。").reason == "numeric_value_candidate"


def test_chinese_duration_folds_to_same_value() -> None:
    """0.17.0 P2-1.2：中文时长词预折——「半秒 vs 500毫秒」曾因归一化失败
    误报为冲突（基线 ny-neg-samevalue 活体证据）；第 1-5 轮 spike 铁证
    D1 型（值都抽对、全死在归一化）。"""
    from memory_arbiter.semantic_conflict import normalize_value
    for left, right in [
        ("半秒", "500毫秒"), ("半秒", "0.5s"), ("半秒", "500ms"),
        ("一刻钟", "15分钟"), ("一刻钟", "900000ms"), ("半小时", "30分钟"),
        ("两秒", "2秒"), ("两秒", "2000ms"), ("十点", "10点"),
        ("三小时", "180分钟"),
    ]:
        assert normalize_value(left) == normalize_value(right), (left, right)
    # 不等仍是冲突：十点 vs 两点
    assert normalize_value("十点") != normalize_value("两点")
    assert normalize_value("半秒") != normalize_value("三秒")
    # 散文不折：折算只作用于整值锚定形态
    from memory_arbiter.semantic_conflict import _fold_chinese_duration
    assert _fold_chinese_duration("大约半秒的延迟") == "大约半秒的延迟"



def test_pre_qwen_veto_process_records() -> None:
    """2026-09-17（owner 原则延伸）：过程记录（review 轮次/设计→发版/
    重启复验）在 Qwen 之前过滤——单向实测三个穿透形态的收口。业务
    对象 id 的值对立（任务 id=123 预算 5000 vs 500）必须保活。"""
    from memory_arbiter.semantic_conflict import _PROCESS_REVERIFY_RE, decide_evidence
    assert decide_evidence(
        "id=609 adversarial review findings after implementation: backend import",
        "Follow-up review for id=609 after attempted fixes: environment confirmed",
    ).reason == "process_record"
    assert decide_evidence(
        "## 设计文档\n最终设计：v0.7.3_design_v3_final.md，审核已经过 8 轮。",
        "# memory-arbiter v0.8.0 发版完成\n分支已合并到 main。",
    ).reason == "process_record"
    assert decide_evidence(
        "JingleAI 使用 OpenClaw venv 验证 sqlite_vec 与 GGUF embedding 链路。",
        "JingleAI 重启后再次验证 memory-arbiter：sqlite_vec、embedding、检索链路正常。",
    ).reason == "process_record"
    # 保活：业务对象 id 的真值对立（same-object-id 规则被刻意砍掉的原因）
    d = decide_evidence("任务 id=123 的预算上限 5000 元。", "任务 id=123 的预算上限 500 元。")
    assert d.reason == "numeric_value_candidate"
    assert decide_evidence("单笔退款超过 5000 元需要财务复核。",
                           "单笔退款超过 500 元就需要财务复核。").reason == "numeric_value_candidate"
    # 0.17.0 P2-1.1：裸「复审」是政策语句不是评审记录——第 11 轮生产误杀
    # （"每半年复审"配置对永远到不了 Qwen）；带语境的评审记录仍 veto。
    assert decide_evidence(
        "网关证书每半年复审一次，由安全组执行。",
        "网关证书每季度复审一次，由安全组执行。",
    ).reason != "process_record"
    assert decide_evidence(
        "方案复审结论：通过，可进入实施。",
        "方案复审记录归档于 id=77。",
    ).reason == "process_record"
    # 同组排查：裸「复验」与 "pre-verify" 不再误杀
    assert decide_evidence(
        "灰度复验通过后全量放开。",
        "灰度复验通过后按计划推进。",
    ).reason != "process_record"
    assert "pre-verify" not in _PROCESS_REVERIFY_RE.pattern  # 边界在位
    assert not _PROCESS_REVERIFY_RE.search("we pre-verify the checksum")


def test_lineage_version_evolution_veto_e1() -> None:
    """0.17.0 P2-6.3（E1）：双侧自报不同谱系版本=取代演进，不按冲突报；
    E2（产品版本差异）与无谱系对不受影响。"""
    from memory_arbiter.semantic_conflict import decide_evidence
    assert decide_evidence(
        "缓存 TTL 设计文档 v1.0：本地缓存 TTL 5 分钟，按业务域配置。",
        "缓存 TTL 设计文档 v2.0 取代 v1.0：TTL 调整为 10 分钟。",
    ).reason == "lineage_version_evolution"
    assert decide_evidence(
        "限流规则第一版：单机 100 QPS。",
        "限流规则第 2 版：单机 200 QPS。",
    ).reason == "lineage_version_evolution"
    # E2：同谱系语境下的产品版本差异仍是候选（不 veto）
    d = decide_evidence(
        "网关设计文档定稿：产品版本要求 1.0，协议走 gRPC。",
        "网关设计文档定稿：产品版本要求 2.0，协议走 REST。",
    )
    assert d.reason != "lineage_version_evolution"
    # 无谱系对：原行为不变
    assert decide_evidence("连接池上限为 10。", "连接池上限为 99。").reason == "numeric_value_candidate"
    # 对抗 review P1-1 反例：标记词+十进制数值是中文配置句式，不是版本号
    assert decide_evidence("熔断方案 1.5 秒超时。", "熔断方案 2.0 秒超时。").reason == "numeric_value_candidate"
    assert decide_evidence("上传规格 0.5mb 上限。", "上传规格 2.5mb 上限。").reason == "numeric_value_candidate"
    # v 前缀整数（无点）也生效
    assert decide_evidence(
        "方案 v1 上线后参数 X 为 5。", "方案 v2 上线后参数 X 为 9。",
    ).reason == "lineage_version_evolution"




def test_judge_and_replan_reject_out_of_range_plan_ids(tmp_path: Path) -> None:
    """R2 对抗轮 pin: apply_plan item ids bypass the top-level id-range
    validation, and the R3 IN-ized member fetch binds them directly — an
    out-of-range id must come back as a structured invalid_plan, not a raw
    OverflowError through the MCP surface (the old per-row order rejected it
    via the empty-matches check before binding)."""
    db = _db(tmp_path)
    left, right = _memory(db, "mysql"), _memory(db, "sqlite")
    created = _record(db, [_member(left, "mysql"), _member(right, "sqlite")])
    huge = 10**30

    judged = db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="sqlite",
        decided_by="user", decided_ref="chat", decision_reason="confirmed",
        apply_plan=[{"memory_id": huge, "action": "preserve_historical_record"}],
    )
    assert judged["outcome"] == "invalid_plan"

    applying = db.judge_conflict(
        created["conflict_id"], expected_revision=1, chosen_value="sqlite",
        decided_by="user", decided_ref="chat", decision_reason="confirmed",
        apply_plan=[{"memory_id": left, "action": "preserve_historical_record"}],
    )
    assert applying["outcome"] == "applying"

    replanned = db.conflicts.replan_conflict(
        created["conflict_id"], expected_revision=applying["revision"],
        apply_plan=[{"memory_id": huge, "action": "preserve_historical_record"}],
    )
    assert replanned["outcome"] == "invalid_plan"