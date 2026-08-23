from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.db_generation import (
    CONFLICT_DETECTOR_VERSION,
    CURRENT_SCHEMA_GENERATION,
    detect_database_generation,
)
from memory_arbiter.evidence import evidence_content_hash, local_text_units
from memory_arbiter.models import MemoryRecord
from memory_arbiter.tools import MemoryTools
from memory_arbiter.vnext_migration import (
    _configured_embedding_space_id,
    build_conflict_only,
    _copy_preserved_tables,
    _fingerprint,
    _mark_conflict_rebuild_ready,
    inspect,
)


def _settings(path: Path, tmp_path: Path) -> Settings:
    return Settings(db_path=path, backup_jsonl=tmp_path / "backup.jsonl")


def _same_space_settings(path: Path, tmp_path: Path) -> Settings:
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"same-space-model")
    return Settings(
        db_path=path,
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        vec_dim=2,
        embedding_provider="gguf",
        embedding_model_path=model,
    )


def _mark_current_space(db: MemoryDB, settings: Settings) -> str:
    pytest.importorskip("sqlite_vec")
    space_id = _configured_embedding_space_id(settings)
    assert space_id is not None
    with db.connection() as conn:
        memories = [dict(row) for row in conn.execute(
            "SELECT id,version,content,subject,status FROM memories WHERE status!='deleted'"
        )]
    for memory in memories:
        units = local_text_units(
            str(memory.get("subject") or ""), str(memory.get("content") or ""),
        )
        embeddings = [[0.0, 1.0] for _unit in units]
        published = db.evidence.publish(
            int(memory["id"]), int(memory.get("version") or 1),
            evidence_content_hash(str(memory.get("content") or "")), units, embeddings,
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


def _memory(defaults: dict[str, object]) -> MemoryRecord:
    return MemoryRecord.from_input(
        {"content": "数据库使用 SQLite。", "subject": "数据库选型", "workspace": "project"},
        defaults,
    )


def test_destructive_copy_preserves_core_data_but_not_conflict_history(tmp_path: Path) -> None:
    source_path = tmp_path / "source.sqlite3"
    target_path = tmp_path / "target.sqlite3"
    source = MemoryDB(_settings(source_path, tmp_path))
    memory_id, _ = source.insert_memory(_memory(_settings(source_path, tmp_path).defaults()), "project")
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


def test_startup_migrates_legacy_alias_unique_key_to_pair_key(tmp_path: Path) -> None:
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
    assert reopened.record_workspace_decision("raw", "candidate-b", status="rejected")[0]
    with reopened.connection() as conn:
        rows = conn.execute(
            "SELECT canonical FROM workspace_aliases WHERE alias_workspace='raw' ORDER BY canonical"
        ).fetchall()
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
        events = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='workspace_alias_events'"
        ).fetchone()
    assert [row["canonical"] for row in rows] == ["candidate-a", "candidate-b"]
    assert columns == ["alias_workspace", "canonical", "status", "updated_at"]
    assert events is None


def test_startup_refuses_partial_workspace_decision_schema_without_data_loss(tmp_path: Path) -> None:
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
    assert reopened.db_available is False
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspace_aliases").fetchone()[0] == 1
        columns = [row[1] for row in conn.execute("PRAGMA table_info(workspace_aliases)")]
    assert columns == ["alias_workspace", "canonical", "updated_at"]


def test_startup_refuses_invalid_compact_workspace_decisions(tmp_path: Path) -> None:
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
    assert reopened.db_available is False
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT status FROM workspace_aliases").fetchone()
    assert row[0] == "bogus"


def test_startup_recovers_interrupted_legacy_workspace_migration(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    assert db.record_workspace_decision("raw", "target")[0]
    with db.write_transaction() as conn:
        conn.execute("ALTER TABLE workspace_aliases RENAME TO workspace_aliases_legacy")
    reopened = MemoryDB(_settings(path, tmp_path))
    assert reopened.db_available is True
    assert reopened.resolve_workspace_canonical("raw", None, register_new=False)["canonical"] == "target"
    with reopened.connection() as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='workspace_aliases_legacy'"
        ).fetchone() is None


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


def test_scan_gate_requires_matching_epoch_detector_boundary_and_live_set(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    settings = _settings(path, tmp_path)
    db = MemoryDB(settings)
    memory_id, _ = db.insert_memory(_memory(settings.defaults()), "project")
    assert memory_id is not None
    state = _mark_conflict_rebuild_ready(db)
    visible = db.conflict_scan_state()
    assert visible["required"] is True
    assert visible["detector_version"] == CONFLICT_DETECTOR_VERSION

    # Completion is not a caller assertion: without persisted pages it fails,
    # even when epoch/detector/boundary are otherwise correct.
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is False
    assert db.complete_conflict_scan(
        epoch="wrong", detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is False
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version="old-detector",
        boundary=visible["boundary"],
    ) is False

    with db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status='superseded' WHERE id=?", (memory_id,))
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is False

    with db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status='active' WHERE id=?", (memory_id,))
    assert db.record_conflict_scan_page(
        epoch=state["conflict_scan_epoch"],
        detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
        after_memory_id=0,
        next_anchor_memory_id=None,
        anchors_scanned=1,
        workspace=None,
    ) is True
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is True
    assert db.conflict_scan_state()["required"] is False
    assert db.complete_conflict_scan(
        epoch=state["conflict_scan_epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=visible["boundary"],
    ) is False


def test_scan_candidates_pages_persist_progress_and_clear_gate(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    settings = _settings(path, tmp_path)
    db = MemoryDB(settings)
    for index in range(3):
        memory_id, _ = db.insert_memory(
            MemoryRecord.from_input(
                {"content": f"配置值为 {index}。", "subject": "配置", "workspace": "project"},
                settings.defaults(),
            ),
            "project",
        )
        assert memory_id is not None
    _mark_conflict_rebuild_ready(db)
    tools = MemoryTools(settings=settings, db=db)
    # The scan path is exercised without requiring sqlite-vec in this gate test.
    tools.db.state.sqlite_vec_available = True
    tools.db.scan_rule_candidates = lambda **kwargs: {
        "anchors_scanned": min(2, 3 - int(kwargs["after_memory_id"])),
        "next_anchor_memory_id": 2 if int(kwargs["after_memory_id"]) == 0 else None,
        "candidates": [],
        "counts": {},
    }

    first = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 2})
    assert first["data"]["conflict_scan_progress"]["complete"] is False
    assert db.conflict_scan_state()["required"] is True
    # Skipping the persisted cursor is rejected and cannot clear the gate.
    skipped = tools.memory_repair("scan_candidates", {"anchor_memory_id": 1, "batch": 2})
    assert skipped["data"]["conflict_scan_progress_rejected"] is True
    assert db.conflict_scan_state()["required"] is True

    final = tools.memory_repair("scan_candidates", {"anchor_memory_id": 2, "batch": 2})
    assert final["data"]["conflict_scan_completed"] is True
    assert db.conflict_scan_state()["required"] is False


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
    memory_id, _ = source.insert_memory(_memory(settings.defaults()), "project")
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
    assert result["target_space_reusable"] is True
    assert result["target_space_reason"] == "embedding_space_match"
    assert result["source_fingerprint"] == result["target_fingerprint"] == before
    assert detect_database_generation(target_path) == "current"
    with sqlite3.connect(target_path) as conn:
        state = dict(conn.execute("SELECT key,value FROM migration_state"))
        columns = {row[1] for row in conn.execute("PRAGMA table_info(conflicts)")}
        assert conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0] == 0
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_notices'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflict_judgments'"
        ).fetchone() is None
    assert {"revision", "member_versions", "value_groups", "notice_type"} <= columns
    assert state["schema_generation"] == CURRENT_SCHEMA_GENERATION
    assert state["conflict_scan_required"] == "true"


def test_previous_generation_with_old_embedding_space_cannot_use_fast_path(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    source_settings = _settings(source_path, tmp_path)
    source = MemoryDB(source_settings)
    memory_id, _ = source.insert_memory(_memory(source_settings.defaults()), "project")
    assert memory_id is not None
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
    settings = _same_space_settings(source_path, tmp_path)

    plan = inspect(source_path, target_path, settings)
    rejected = build_conflict_only(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "full_evidence_rebuild"
    assert plan["evidence_reuse_reason"] == "embedding_space_mismatch"
    assert rejected["ok"] is False
    assert rejected["error"] == "evidence_space_not_reusable"
    assert not target_path.exists()


def test_previous_generation_without_verifiable_space_cannot_use_fast_path(
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

    assert plan["upgrade_mode"] == "full_evidence_rebuild"
    assert plan["evidence_reuse_reason"] == "source_vec_state_missing"


def test_previous_generation_with_incomplete_same_space_index_rebuilds(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    memory_id, _ = source.insert_memory(_memory(settings.defaults()), "project")
    assert memory_id is not None
    _mark_current_space(source, settings)
    with source.write_transaction() as conn:
        evidence_id = conn.execute(
            "SELECT id FROM memory_evidence WHERE memory_id=? ORDER BY id LIMIT 1",
            (memory_id,),
        ).fetchone()[0]
        conn.execute("DELETE FROM memory_evidence_vec WHERE id=?", (evidence_id,))
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    plan = inspect(source_path, target_path, settings)

    assert plan["upgrade_mode"] == "full_evidence_rebuild"
    assert plan["evidence_reuse_reason"] == "source_vectors_incomplete"


def test_previous_generation_with_stale_same_space_evidence_rebuilds(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(settings)
    memory_id, _ = source.insert_memory(_memory(settings.defaults()), "project")
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

    assert plan["upgrade_mode"] == "full_evidence_rebuild"
    assert plan["evidence_reuse_reason"] == "source_evidence_stale"


def test_previous_generation_with_forged_space_but_wrong_vector_dimension_rebuilds(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    target_path = tmp_path / "current.sqlite3"
    source_settings = _same_space_settings(source_path, tmp_path)
    source = MemoryDB(source_settings)
    _mark_current_space(source, source_settings)
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )
    mismatched_settings = Settings(
        db_path=source_path,
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        vec_dim=3,
        embedding_provider="gguf",
        embedding_model_path=source_settings.embedding_model_path,
    )
    forged_space = _configured_embedding_space_id(mismatched_settings)
    assert forged_space is not None
    with source.write_transaction() as conn:
        conn.execute(
            "UPDATE _vec_index_meta SET value=? WHERE key='active_space_id'",
            (forged_space,),
        )

    plan = inspect(source_path, target_path, mismatched_settings)

    assert plan["upgrade_mode"] == "full_evidence_rebuild"
    assert plan["evidence_reuse_reason"] == "source_vector_dimension_mismatch"


def test_generation_switch_marker_is_atomic_with_scan_metadata(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    db = MemoryDB(_settings(path, tmp_path))
    _mark_conflict_rebuild_ready(db)
    with sqlite3.connect(path) as conn:
        state = dict(conn.execute("SELECT key,value FROM migration_state"))
    assert state["schema_generation"] == CURRENT_SCHEMA_GENERATION
    assert state["conflict_scan_required"] == "true"
    assert state["conflict_scan_detector_version"] == CONFLICT_DETECTOR_VERSION
    boundary = json.loads(state["conflict_scan_boundary"])
    assert boundary["active_count"] == 0
    assert boundary["max_memory_id"] == 0
    assert len(boundary["active_set_digest"]) == 64


def test_status_and_doctor_expose_pending_rebuild_scan(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    settings = _settings(path, tmp_path)
    db = MemoryDB(settings)
    _mark_conflict_rebuild_ready(db)
    tools = MemoryTools(settings=settings, db=db)
    status = tools.memory_status()["data"]
    assert status["conflict_scan_required"] is True
    assert status["conflict_scan"]["detector_version"] == CONFLICT_DETECTOR_VERSION
    findings = tools.memory_doctor_overview()["data"]["findings"]
    scan = next(item for item in findings if item["check_id"] == "conflicts.scan_required")
    assert scan["status"] == "warn"


def test_write_between_upgrade_and_scan_rearms_instead_of_wedging(tmp_path: Path) -> None:
    """Spec §15.7/§15.8.24 recovery: a live-boundary drift re-arms the epoch so a
    fresh full scan can still clear conflict_scan_required, instead of wedging."""
    path = tmp_path / "db.sqlite3"
    settings = _settings(path, tmp_path)
    db = MemoryDB(settings)
    first_id, _ = db.insert_memory(_memory(settings.defaults()), "project")
    assert first_id is not None
    original = _mark_conflict_rebuild_ready(db)
    original_boundary = db.conflict_scan_state()["boundary"]

    # A memory write after upgrade drifts the live active-set boundary: pages
    # recorded against the original boundary are now rejected.
    second_id, _ = db.insert_memory(
        MemoryRecord.from_input(
            {"content": "另一个配置。", "subject": "配置", "workspace": "project"},
            settings.defaults(),
        ),
        "project",
    )
    assert second_id is not None
    assert db.record_conflict_scan_page(
        epoch=original["conflict_scan_epoch"],
        detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=original_boundary,
        after_memory_id=0, next_anchor_memory_id=None, anchors_scanned=2,
        workspace=None,
    ) is False
    assert db.conflict_scan_state()["required"] is True

    # Re-arm against the current live set, then a full scan of the new boundary
    # clears the flag.
    assert db.rearm_conflict_scan_if_drifted() is True
    rearmed = db.conflict_scan_state()
    assert rearmed["required"] is True
    assert rearmed["epoch"] != original["conflict_scan_epoch"]
    assert rearmed["progress"] is None
    assert db.record_conflict_scan_page(
        epoch=rearmed["epoch"],
        detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=rearmed["boundary"],
        after_memory_id=0, next_anchor_memory_id=None, anchors_scanned=2,
        workspace=None,
    ) is True
    assert db.complete_conflict_scan(
        epoch=rearmed["epoch"], detector_version=CONFLICT_DETECTOR_VERSION,
        boundary=rearmed["boundary"],
    ) is True
    assert db.conflict_scan_state()["required"] is False
    # No re-arm happens when the boundary is already consistent.
    assert db.rearm_conflict_scan_if_drifted() is False


def test_conflict_only_validation_failure_marks_phase_failed(tmp_path: Path, monkeypatch) -> None:
    """A conflict-only rebuild whose post-transaction validation fails must not
    leave a phase=ready/current target that detect_database_generation trusts."""
    import memory_arbiter.vnext_migration as vm

    source = tmp_path / "source.sqlite3"
    settings = _same_space_settings(source, tmp_path)
    db = MemoryDB(settings)
    db.insert_memory(_memory(settings.defaults()), "project")
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


def test_conflict_only_revalidates_copied_target_space(
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

    assert result["ok"] is False
    assert result["target_space_reusable"] is False
    assert result["target_space_reason"] == "embedding_space_mismatch"
    assert detect_database_generation(target_path) != "current"
    with sqlite3.connect(target_path) as conn:
        phase = conn.execute(
            "SELECT value FROM migration_state WHERE key='phase'"
        ).fetchone()
    assert phase is not None and phase[0] == "failed"
