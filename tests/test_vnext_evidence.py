from __future__ import annotations

from pathlib import Path
import sqlite3
import ast

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.evidence import evidence_content_hash, local_text_units
from memory_arbiter.semantic_conflict import ModelSignal, decide_evidence
from memory_arbiter.tools import MemoryTools
from memory_arbiter.vnext_migration import build, inspect


class FakeEmbedder:
    embedding_space_id = "fake-vnext-space"
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars=None) -> EmbedResult:
        text = f"{prefix}\n{body}".casefold()
        if "postgres" in text or "pgsql" in text:
            vector = [1.0, 0.0]
        elif "超时" in text:
            vector = [0.8, 0.2]
        else:
            vector = [0.0, 1.0]
        return EmbedResult(vector, False, len(text), len(text))


def make_tools(tmp_path: Path, *, semantic_enabled: bool = False) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "vnext.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        vec_dim=2,
        embedding_provider="gguf",
        embedding_model_path=model,
        embedding_auto_write=True,
        embedding_auto_query=True,
        semantic_conflict_enabled=semantic_enabled,
        semantic_conflict_model_path=model if semantic_enabled else None,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = FakeEmbedder()
    tools._embedder_loaded = True
    tools.db.init_vec_index_state("fake-vnext-space", True)
    return tools


def test_local_text_units_are_simple_and_cover_headings() -> None:
    units = local_text_units(
        "Database policy",
        "# Runtime\n生产环境数据库使用 PostgreSQL 16。\n接口超时为 5 秒。",
    )
    assert [unit.kind for unit in units][:2] == ["subject", "heading"]
    assert any(unit.kind == "text" and "PostgreSQL" in unit.text for unit in units)
    assert all(unit.unit_index == index for index, unit in enumerate(units))


def test_local_text_worker_has_single_definition() -> None:
    worker_source = Path(__file__).parents[1] / "memory_arbiter" / "workers.py"
    tree = ast.parse(worker_source.read_text(encoding="utf-8"))
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LocalTextIndexWorker"
    ]
    assert len(definitions) == 1


def test_decide_evidence_has_narrow_notify_check_ignore_contract() -> None:
    assert decide_evidence("数据库使用 pgsql。", "数据库使用 PostgreSQL。").action == "ignore"
    changed = decide_evidence("接口超时为 5 秒。", "接口超时为 30 秒。")
    assert changed.action == "notify"
    assert changed.reason == "numeric_value_changed"
    assert decide_evidence("测试环境数据库使用 MySQL。", "生产环境数据库使用 PostgreSQL。").action == "ignore"
    assert decide_evidence("服务使用数据库。", "服务迁移到新的存储引擎。").action in {"check", "ignore"}


def test_write_publishes_local_text_evidence(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_write(
        content="生产环境数据库使用 PostgreSQL 16。接口超时为 5 秒。",
        subject="database runtime",
        tags=["database"],
    )
    assert result["ok"] is True
    memory_id = result["data"]["id"]
    assert result["data"]["evidence_index"]["status"] == "queued"
    assert tools.wait_evidence_worker_drained(timeout=2)
    coverage = tools.db.evidence.coverage()
    assert coverage["indexed_memories"] == 1
    assert coverage["units"] >= 2
    with tools.db.connection() as conn:
        row = conn.execute("SELECT content_hash FROM memory_evidence WHERE memory_id=? LIMIT 1", (memory_id,)).fetchone()
    assert row["content_hash"] == evidence_content_hash(result["data"]["record"]["content"])


def test_vnext_search_uses_evidence_knn_not_legacy_vectors(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    target = tools.memory_write(
        content="服务数据库选择 PostgreSQL 16。",
        subject="runtime database",
        tags=["db"],
    )["data"]["id"]
    tools.memory_write(content="用户今天想喝咖啡。", subject="lunch", tags=["food"])
    assert tools.wait_evidence_worker_drained(timeout=2)
    result = tools.memory_search(query="pgsql database", limit=5)
    ids = [row["id"] for row in result["data"]["results"]]
    assert target in ids
    hit = next(row for row in result["data"]["results"] if row["id"] == target)
    assert hit.get("_evidence_hits") or hit.get("content")


def test_notify_surfaces_without_qwen_and_check_fails_closed(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, semantic_enabled=False)
    old = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=["api"])["data"]
    new = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=["api"])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    record = tools.db.get_memory(new["id"])
    snapshot = {
        "memory_id": new["id"],
        "version": record["version"],
        "content_hash": evidence_content_hash(record["content"]),
    }
    tools._process_semantic_conflict_job(new["id"], snapshot)
    notices = tools.db.list_semantic_notices()
    assert any(notice["peer_id"] == old["id"] for notice in notices)
    notice = next(notice for notice in notices if notice["peer_id"] == old["id"])
    assert notice["payload"]["route"] == "notify"
    assert notice["payload"]["qwen_signal"]["status"] == "not_required"

    tools.db.update_semantic_notice_status(notice["id"], "dismissed")
    monkeypatch.setattr(
            "memory_arbiter.pipeline.evidence.decide_evidence",
        lambda _a, _b: type("D", (), {
            "action": "check", "reason": "semantic_similarity_only", "anchors": [],
            "left_value": None, "right_value": None,
        })(),
    )
    tools._process_semantic_conflict_job(new["id"], snapshot)
    assert tools.db.list_semantic_notices(status="open") == []


def test_vnext_semantic_job_is_chained_after_evidence_publish(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, semantic_enabled=False)
    calls = []
    original = tools._enqueue_semantic_conflict_check

    def tracked(memory_id, record, *, after_evidence=False):
        calls.append((memory_id, after_evidence, tools.db.evidence.coverage()["indexed_memories"]))
        return original(memory_id, record, after_evidence=after_evidence)

    monkeypatch.setattr(tools, "_enqueue_semantic_conflict_check", tracked)
    result = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])
    assert result["data"]["semantic_conflict_check"] == {
        "status": "deferred", "reason": "waiting_for_evidence_index",
    }
    assert tools.wait_evidence_worker_drained(timeout=2)
    assert any(after and indexed >= 1 for _mid, after, indexed in calls)


def test_evidence_publish_rejects_stale_snapshot(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(content="旧内容。", subject="snapshot", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    memory_id = written["id"]
    current = tools.db.get_memory(memory_id)
    stale_units = local_text_units(current["subject"], current["content"])
    edited = tools.memory_edit(memory_id=memory_id, new_content="新内容。")
    assert edited["ok"] is True
    result = tools.db.evidence.publish(
        memory_id,
        int(current["version"]),
        evidence_content_hash(current["content"]),
        stale_units,
        [[0.0, 1.0] for _ in stale_units],
    )
    assert result["outcome"] == "stale_snapshot"


def test_vnext_status_change_updates_evidence_parent_status(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(content="待废弃事实。", subject="status", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    assert tools.db.update_memory(written["id"], {"status": "superseded"}) is True
    with tools.db.connection() as conn:
        statuses = {
            row["parent_status"] for row in conn.execute(
                "SELECT v.parent_status FROM memory_evidence_vec v "
                "JOIN memory_evidence e ON e.id=v.id WHERE e.memory_id=?",
                (written["id"],),
            )
        }
    assert statuses == {"superseded"}


def test_vnext_weak_isolation_does_not_hard_filter_semantic_candidates(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.isolation = "weak"
    written = tools.memory_write(
        content="接口超时为 5 秒。", subject="timeout", tags=[], workspace="alpha",
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    seen = []

    def evidence_knn(*args, **kwargs):
        seen.append(kwargs.get("workspace"))
        return []

    monkeypatch.setattr(tools.db, "evidence_knn", evidence_knn)
    record = tools.db.get_memory(written["id"])
    tools._process_semantic_conflict_job(written["id"], {
        "version": record["version"],
        "content_hash": evidence_content_hash(record["content"]),
    })
    assert seen and set(seen) == {None}


def test_side_by_side_migration_builds_verified_vnext_database(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("sqlite_vec")
    source_settings = Settings(
        db_path=tmp_path / "source.sqlite3",
        backup_jsonl=tmp_path / "source.jsonl",
    )
    source_tools = MemoryTools(source_settings, MemoryDB(source_settings))
    first = source_tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])
    second = source_tools.memory_write(content="数据库使用 PostgreSQL。", subject="database", tags=[])
    assert first["ok"] and second["ok"]

    target = tmp_path / "target.vnext.sqlite3"
    plan = inspect(source_settings.db_path, target)
    assert plan["counts"]["memories"] == 2
    assert plan["estimated_evidence_units"] >= 4

    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    migration_settings = Settings(
        db_path=source_settings.db_path,
        backup_jsonl=tmp_path / "migration.jsonl",
        enable_sqlite_vec=True,
        vec_dim=2,
        embedding_provider="gguf",
        embedding_model_path=model,
    )

    original_init = MemoryTools.__init__

    def patched_init(self, settings=None, db=None):
        original_init(self, settings=settings, db=db)
        self._embedder = FakeEmbedder()
        self._embedder_loaded = True
        self.db.init_vec_index_state("fake-vnext-space", True)

    monkeypatch.setattr(MemoryTools, "__init__", patched_init)
    result = build(source_settings.db_path, target, migration_settings, progress=False)
    assert result["ok"] is True
    assert result["switch_ready"] is True
    assert result["row_counts_match"] is True
    assert result["coverage"]["indexed_memories"] == 2
    assert target.stat().st_mode & 0o777 == 0o600
    assert not Path(str(target) + "-wal").exists()
    assert not Path(str(target) + "-shm").exists()
    with sqlite3.connect(target) as migrated:
        assert migrated.execute(
            "SELECT 1 FROM sqlite_master WHERE name='memories_vec'"
        ).fetchone() is None
        assert migrated.execute(
            "SELECT 1 FROM sqlite_master WHERE name='memory_sections_vec'"
        ).fetchone() is None

    target_settings = Settings(
        db_path=target,
        backup_jsonl=tmp_path / "target.jsonl",
        enable_sqlite_vec=True,
        vec_dim=2,
        embedding_provider="gguf",
        embedding_model_path=model,
    )
    target_tools = MemoryTools(target_settings, MemoryDB(target_settings))
    target_tools._embedder = FakeEmbedder()
    target_tools._embedder_loaded = True
    target_tools.db.init_vec_index_state("fake-vnext-space", True)
    searched = target_tools.memory_search(query="pgsql database", limit=5)
    assert second["data"]["id"] in [row["id"] for row in searched["data"]["results"]]


def test_update_remember_fields_get_explicit_recovery_hint(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory("update", {"memory_id": 1, "content": "wrong field"})
    assert result["ok"] is False
    assert result["data"]["did_you_mean"] == "new_content"
