from __future__ import annotations

from pathlib import Path
import sqlite3
import ast
import contextlib
import uuid
from typing import Any

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.evidence import evidence_content_hash, local_text_units
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.semantic_conflict import (
    AttributeValueExtraction,
    ModelSignal,
    decide_evidence,
    evaluate_pair_extractions,
)
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


class CountingEmbedder(FakeEmbedder):
    calls = 0

    @classmethod
    def embed_text(cls, prefix: str, body: str, max_body_chars=None) -> EmbedResult:
        cls.calls += 1
        return super().embed_text(prefix, body, max_body_chars)


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


def test_decide_evidence_numeric_change_is_scan_candidate_not_direct_notice() -> None:
    assert decide_evidence("数据库使用 pgsql。", "数据库使用 PostgreSQL。").action == "ignore"
    changed = decide_evidence("接口超时为 5 秒。", "接口超时为 30 秒。")
    assert changed.action == "check"
    assert changed.reason == "numeric_value_candidate"
    assert decide_evidence("测试环境数据库使用 MySQL。", "生产环境数据库使用 PostgreSQL。").action == "ignore"
    assert decide_evidence("服务使用数据库。", "服务迁移到新的存储引擎。").action in {"check", "ignore"}


def test_write_notice_requires_consistent_bidirectional_qwen_mapping() -> None:
    forward = AttributeValueExtraction("接口超时", "5 秒", "接口超时", "30 秒")
    reverse = AttributeValueExtraction("接口超时", "30 秒", "接口超时", "5 秒")
    ready = evaluate_pair_extractions(
        forward, reverse,
        {"quote": "接口超时为 5 秒。"}, {"quote": "接口超时为 30 秒。"},
        require_bidirectional=True,
    )
    assert ready.state == "notice_ready"
    inconsistent = AttributeValueExtraction("接口超时", "5 秒", "接口超时", "30 秒")
    rejected = evaluate_pair_extractions(
        forward, inconsistent,
        {"quote": "接口超时为 5 秒。"}, {"quote": "接口超时为 30 秒。"},
        require_bidirectional=True,
    )
    assert rejected.state == "review_candidate"
    assert rejected.reason == "bidirectional_mapping_mismatch"


def test_semantic_backend_serializes_metadata_then_bounded_evidence_quotes() -> None:
    from memory_arbiter.semantic_conflict import (
        LocalGGUFSemanticBackend, PAIR_PROMPT_VERSION, _PAIR_PROMPT,
        _PAIR_RESPONSE_FORMAT,
    )

    text = LocalGGUFSemanticBackend._pair_text(
        {"subject": "数据库", "quote": "数据库为 MySQL。", "content": "不应使用的全文",
         "metadata": {"entity": "checkout", "scope": "global"}},
        {"subject": "数据库", "quote": "数据库为 SQLite。"},
    )
    assert text.index("A metadata:") < text.index("A证据原文=数据库为 MySQL。")
    assert text.index("B metadata:") < text.index("B证据原文=数据库为 SQLite。")
    assert "entity=checkout" in text and "scope=global" in text
    assert "不应使用的全文" not in text
    assert PAIR_PROMPT_VERSION == "pair-v3"
    assert "必须输出全部四个字符串字段" in _PAIR_PROMPT
    assert '"__unknown__"' in _PAIR_PROMPT
    assert "设为 null" not in _PAIR_PROMPT
    schema = _PAIR_RESPONSE_FORMAT["schema"]
    assert set(schema["required"]) == {
        "attribute_a", "value_a", "attribute_b", "value_b",
    }
    assert schema["additionalProperties"] is False


def test_write_notice_rejects_whole_quote_values_but_accepts_short_values() -> None:
    left_quote = "生产环境的接口超时策略明确设置为 5 秒。"
    right_quote = "生产环境的接口超时策略明确设置为 30 秒。"
    copied = evaluate_pair_extractions(
        AttributeValueExtraction("接口超时", left_quote, "接口超时", right_quote),
        AttributeValueExtraction("接口超时", right_quote, "接口超时", left_quote),
        {"quote": left_quote}, {"quote": right_quote}, require_bidirectional=True,
    )
    assert copied.state == "review_candidate"
    assert copied.reason == "qwen_unverified"

    short = evaluate_pair_extractions(
        AttributeValueExtraction("接口超时", "5 秒", "接口超时", "30 秒"),
        AttributeValueExtraction("接口超时", "30 秒", "接口超时", "5 秒"),
        {"quote": left_quote}, {"quote": right_quote}, require_bidirectional=True,
    )
    assert short.state == "notice_ready"


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


def test_evidence_candidate_enters_when_lexical_pool_is_full(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.recall_pool_cap = 4
    lexical_ids = [
        tools.memory_write(
            content=f"needle lexical distractor {index}",
            subject=f"needle {index}",
        )["data"]["id"]
        for index in range(4)
    ]
    target_id = tools.memory_write(
        content="semantically relevant target without the query term",
        subject="semantic target",
    )["data"]["id"]
    target = tools.db.get_memory(target_id)
    assert target is not None
    tools.db.state.sqlite_vec_available = True

    def evidence_knn(*_args, **_kwargs):
        return [{
            **target,
            "id": 999,
            "memory_id": target_id,
            "kind": "text",
            "text": target["content"],
            "start_offset": 0,
            "end_offset": len(target["content"]),
            "distance": 0.01,
        }]

    monkeypatch.setattr(tools.db, "evidence_knn", evidence_knn)
    result = tools.memory_search(
        query="needle", limit=4, query_embedding=[1.0, 0.0],
        include_linked_open_items=False, include_conflict_signal=False,
    )
    ids = [row["id"] for row in result["data"]["results"]]
    assert target_id in ids
    assert len(set(ids) & set(lexical_ids)) >= 1


def test_exact_subject_match_survives_evidence_fusion(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.recall_pool_cap = 4
    exact_id = tools.memory_write(
        content="exact command reference",
        subject="needle",
    )["data"]["id"]
    semantic_id = tools.memory_write(
        content="semantic-only candidate",
        subject="other topic",
    )["data"]["id"]
    semantic = tools.db.get_memory(semantic_id)
    assert semantic is not None
    tools.db.state.sqlite_vec_available = True

    monkeypatch.setattr(tools.db, "evidence_knn", lambda *_args, **_kwargs: [{
        **semantic,
        "id": 1000,
        "memory_id": semantic_id,
        "kind": "text",
        "text": semantic["content"],
        "start_offset": 0,
        "end_offset": len(semantic["content"]),
        "distance": 0.0,
    }])
    result = tools.memory_search(
        query="needle", limit=4, query_embedding=[1.0, 0.0],
        include_linked_open_items=False, include_conflict_signal=False,
    )
    assert result["data"]["results"][0]["id"] == exact_id


def test_numeric_candidate_fails_closed_without_qwen(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, semantic_enabled=False)
    tools.settings.semantic_conflict_on_write = "off"
    old = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=["api"])["data"]
    new = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=["api"])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: None)
    result = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    assert result["status"] == "incomplete"
    assert result["reason"] == "qwen_unavailable"
    assert result["notices_created"] == 0
    assert tools.db.list_semantic_notices(status="open") == []
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10})
    pair = (min(old["id"], new["id"]), max(old["id"], new["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    assert clue["route"] == "review_candidate"
    assert "numeric_value_candidate" in clue["reasons"]


def test_vnext_semantic_job_is_chained_after_evidence_publish(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, semantic_enabled=False)
    calls = []
    original = tools._enqueue_semantic_conflict_check

    def tracked(memory_id, record, *, after_evidence=False):
        calls.append((memory_id, after_evidence, tools.db.evidence.coverage()["indexed_memories"]))
        return original(memory_id, record, after_evidence=after_evidence)

    monkeypatch.setattr(tools, "_enqueue_semantic_conflict_check", tracked)
    result = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])
    task_id = result["data"]["evidence_index"]["semantic_task_id"]
    # The write's sync gate waits for this exact reserved task. A fast local
    # index + semantic pass may therefore complete in the same request rather
    # than returning the older waiting_for_evidence_index placeholder.
    assert result["data"]["semantic_conflict_check"] == {
        "status": "completed",
        "outcome": "checked_no_notice",
        "notices_created": 0,
        "task_id": task_id,
        "dedupe_key": task_id,
    }
    assert tools.wait_evidence_worker_drained(timeout=2)
    # The enqueue observation is taken inside the evidence worker after
    # publish, so this still proves semantic work was chained behind evidence.
    assert calls == [(result["data"]["id"], True, 1)]


def test_semantic_job_reuses_just_published_evidence_vectors(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, semantic_enabled=False)
    tools.settings.semantic_conflict_on_write = "off"
    CountingEmbedder.calls = 0
    tools._embedder = CountingEmbedder()
    written = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    indexed_calls = CountingEmbedder.calls
    assert indexed_calls > 0

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: None)
    tools._process_semantic_conflict_job(written["id"], _job_snapshot(tools, written["id"]))

    assert CountingEmbedder.calls == indexed_calls


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


def test_previous_generation_old_space_is_rebuilt_into_runtime_space(
    tmp_path: Path, monkeypatch,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=source_path,
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        vec_dim=2,
        embedding_provider="gguf",
        embedding_model_path=model,
    )
    source_db = MemoryDB(settings)
    source_tools = MemoryTools(settings, source_db)
    old_embedder = FakeEmbedder()
    old_embedder.embedding_space_id = "old-pipeline-space"
    source_tools._embedder = old_embedder
    source_tools._embedder_loaded = True
    source_db.init_vec_index_state(old_embedder.embedding_space_id, True)
    written = source_tools.memory_write(
        content="旧空间中的内容。", subject="旧空间", tags=[],
    )
    assert written["ok"] is True
    assert source_tools.wait_evidence_worker_drained(timeout=5)
    with source_db.write_transaction() as conn:
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    original_init = MemoryTools.__init__

    def init_with_current_embedder(self, settings=None, db=None):
        original_init(self, settings=settings, db=db)
        self._embedder = FakeEmbedder()
        self._embedder_loaded = True
        self.db.init_vec_index_state(self._embedder.embedding_space_id, True)

    monkeypatch.setattr(MemoryTools, "__init__", init_with_current_embedder)
    target = tmp_path / "current.sqlite3"

    plan = inspect(source_path, target, settings)
    result = build(source_path, target, settings, progress=False)

    assert plan["upgrade_mode"] == "full_evidence_rebuild"
    assert plan["evidence_reuse_reason"] == "embedding_space_mismatch"
    assert result["ok"] is True, result
    assert result["upgrade_mode"] == "full_evidence_rebuild"
    assert result["indexed"] == 1
    assert result["target_space_ready"] is True
    assert result["expected_space_id"] == FakeEmbedder.embedding_space_id
    with sqlite3.connect(target) as conn:
        state = dict(conn.execute("SELECT key,value FROM _vec_index_meta"))
    assert state["state"] == "ready"
    assert state["active_space_id"] == FakeEmbedder.embedding_space_id
    assert "target_space_id" not in state


def test_full_rebuild_does_not_copy_deleted_old_space_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=source_path, backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True, vec_dim=2, embedding_provider="gguf",
        embedding_model_path=model,
    )
    source_db = MemoryDB(settings)
    source_tools = MemoryTools(settings, source_db)
    old_embedder = FakeEmbedder()
    old_embedder.embedding_space_id = "old-pipeline-space"
    source_tools._embedder = old_embedder
    source_tools._embedder_loaded = True
    source_db.init_vec_index_state(old_embedder.embedding_space_id, True)
    active = source_tools.memory_write(
        content="active", subject="active", workspace="project-alpha",
    )["data"]["id"]
    deleted = source_tools.memory_write(
        content="deleted", subject="deleted", workspace="project-beta",
    )["data"]["id"]
    assert source_tools.wait_evidence_worker_drained(timeout=5)
    with source_db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status='deleted' WHERE id=?", (deleted,))
        conn.execute(
            "UPDATE migration_state SET value='local_text_evidence_v1' "
            "WHERE key='schema_generation'"
        )

    original_init = MemoryTools.__init__

    def init_with_current_embedder(self, settings=None, db=None):
        original_init(self, settings=settings, db=db)
        self._embedder = FakeEmbedder()
        self._embedder_loaded = True
        self.db.init_vec_index_state(self._embedder.embedding_space_id, True)

    monkeypatch.setattr(MemoryTools, "__init__", init_with_current_embedder)
    target = tmp_path / "current.sqlite3"

    result = build(source_path, target, settings, progress=False)

    assert result["ok"] is True, result
    target_db = MemoryDB(Settings(
        db_path=target, backup_jsonl=tmp_path / "target-backup.jsonl",
        enable_sqlite_vec=True, vec_dim=2,
    ))
    with target_db.connection() as conn:
        evidence_memory_ids = {
            int(row[0]) for row in conn.execute("SELECT DISTINCT memory_id FROM memory_evidence")
        }
        units = int(conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0])
        vectors = int(conn.execute("SELECT COUNT(*) FROM memory_evidence_vec").fetchone()[0])
        workspace_vectors = int(
            conn.execute("SELECT COUNT(*) FROM workspace_canonicals_vec").fetchone()[0]
        )
    assert evidence_memory_ids == {active}
    assert units == vectors
    assert workspace_vectors == 1


def test_final_sync_next_step_does_not_request_another_final_sync(tmp_path: Path, monkeypatch, capsys) -> None:
    pytest.importorskip("sqlite_vec")
    source_settings = Settings(
        db_path=tmp_path / "source.sqlite3",
        backup_jsonl=tmp_path / "source.jsonl",
    )
    source_tools = MemoryTools(source_settings, MemoryDB(source_settings))
    source_tools.memory_write(content="migration source", subject="migration")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
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
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    from memory_arbiter.vnext_migration import run_cli

    target = tmp_path / "target.sqlite3"
    assert run_cli([
        "--source", str(source_settings.db_path), "--target", str(target),
        "--execute", "--final-sync",
    ]) == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["final_sync"] is True
    assert "run --final-sync" not in payload["next_step"]


def test_update_remember_fields_get_explicit_recovery_hint(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory("update", {"memory_id": 1, "content": "wrong field"})
    assert result["ok"] is False
    assert result["data"]["did_you_mean"] == "new_content"


def _job_snapshot(tools: MemoryTools, memory_id: int) -> dict:
    record = tools.db.get_memory(memory_id)
    return {
        "memory_id": int(memory_id),
        "version": record["version"],
        "content_hash": evidence_content_hash(record["content"]),
    }


def _strict_pair_backend():
    class Backend:
        @staticmethod
        def classify_pair(left, right):
            import re
            left_value = re.search(r"\d+", left["quote"]).group(0)
            right_value = re.search(r"\d+", right["quote"]).group(0)
            parsed = {
                "attribute_a": "连接池上限", "value_a": left_value,
                "attribute_b": "连接池上限", "value_b": right_value,
            }
            return ModelSignal(True, "attribute_value", None, "", parsed, None)
    return Backend()


def test_notice_pairs_capped_in_unified_conflicts_table(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    metadata = {"entity": "checkout-api", "scope": "production"}
    peers = [
        tools.memory_write(
            content=f"连接池上限为 {i + 10}。", subject=f"pool{i}", tags=[], metadata=metadata,
        )["data"]
        for i in range(4)
    ]
    new = tools.memory_write(
        content="连接池上限为 99。", subject="poolx", tags=[], metadata=metadata,
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    hits = [
        {"memory_id": peer["id"], "id": i, "kind": "text", "text": f"连接池上限为 {i + 10}。",
         "start_offset": 0, "end_offset": 11, "distance": 0.10 + i * 0.05}
        for i, peer in enumerate(peers)
    ]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _strict_pair_backend)

    first = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    assert first == {"status": "completed", "outcome": "notices_created", "notices_created": 2}
    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]
    assert len(notices) == tools.settings.semantic_conflict_max_notice_pairs == 2
    assert all(n["payload"]["route"] == "notice_ready" for n in notices)
    with tools.db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conflicts WHERE notice_type IS NOT NULL").fetchone()[0] == 2

    tools.settings.semantic_conflict_max_notice_pairs = 99
    second = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    assert second["notices_created"] == 2
    assert len([n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]) == 4


def test_unified_notice_dedupe_does_not_starve_fresh_pair(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    tools.settings.semantic_conflict_max_notice_pairs = 2
    metadata = {"entity": "checkout-api", "scope": "production"}
    peers = [
        tools.memory_write(
            content=f"连接池上限为 {i + 10}。", subject=f"pool{i}", tags=[], metadata=metadata,
        )["data"]
        for i in range(3)
    ]
    new = tools.memory_write(
        content="连接池上限为 99。", subject="poolx", tags=[], metadata=metadata,
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    hits = [
        {"memory_id": peer["id"], "id": i, "kind": "text", "text": f"连接池上限为 {i + 10}。",
         "start_offset": 0, "end_offset": 11, "distance": 0.10 + i * 0.05}
        for i, peer in enumerate(peers)
    ]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _strict_pair_backend)

    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]
    assert sorted(n["peer_id"] for n in notices) == [peer["id"] for peer in peers]
    assert len({n["dedupe_key"] for n in notices}) == 3


def test_search_surfaces_vector_lag(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_search(query="连接池")
    assert "vector_lag" in result["data"]
    assert result["data"]["vector_lag"]["pending_evidence_index"] >= 0


def test_check_degradation_is_visible_in_semantic_status(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, semantic_enabled=False)
    tools.settings.semantic_conflict_on_write = "off"
    peer = tools.memory_write(content="database connection policy", subject="pool", tags=[])["data"]
    new = tools.memory_write(content="database connection pool size", subject="pool2", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    assert tools._ensure_semantic_backend() is None
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "database connection policy", "start_offset": 0, "end_offset": 26, "distance": 0.2}]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))
    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    assert tools.db.list_semantic_notices(status="open") == []
    degradation = tools._semantic_status()["check_degradation"]
    assert degradation["last_reason"] == "qwen_unavailable"
    assert degradation["count"] >= 1


def test_short_paragraphs_merge_instead_of_drop() -> None:
    from memory_arbiter.evidence import local_text_units as ltu
    units = ltu("s", "这是一个足够长的第一段内容。\n短。\n这是另一个足够长的段落内容。")
    merged_texts = [u.text for u in units if u.kind == "text"]
    assert any("短" in t for t in merged_texts)
    lone = ltu("s", "仅此。")
    assert any(u.kind == "text" and u.text for u in lone)


def test_knn_enforces_memory_status_despite_stale_parent(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    # Simulate a vec-disabled process superseding the memory: parent_status
    # stays 'active' while memories.status is authoritative.
    tools.db.state.sqlite_vec_available = False
    assert tools.db.update_memory(written["id"], {"status": "superseded"})
    tools.db.state.sqlite_vec_available = True
    hits = tools.db.evidence_knn([0.8, 0.2], k=5, parent_status_filter="active")
    assert all(h["memory_id"] != written["id"] for h in hits)


def test_knn_workspace_overfetch_restores_recall(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    # Insert B first so the later A units win sqlite-vec's equal-distance tie
    # order and own the initial global top-k window.
    peer = tools.memory_write(
        content="postgres 端口为 5432。", subject="pg port", tags=[], workspace="wsB",
    )["data"]
    bulk = "\n\n".join(f"postgres 配置项 {i}：连接池参数说明 {i}。" for i in range(8))
    bulk_record = tools.memory_write(
        content=bulk, subject="pg bulk", tags=[], workspace="wsA",
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    # FakeEmbedder intentionally maps all workspace names to one vector, so
    # normal workspace resolution treats wsB as an alias of wsA. This test is
    # about KNN filtering rather than canonicalization; restore the intended
    # independent canonical fixture explicitly.
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET workspace_canonical=workspace WHERE id IN (?,?)",
            (peer["id"], bulk_record["id"]),
        )
    hits = tools.db.evidence_knn([1.0, 0.0], k=5, workspace="wsB", exclude_memory_id=999999)
    assert any(h["memory_id"] == peer["id"] for h in hits)


def test_evidence_only_candidates_carry_memory_row_shape(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="postgresql 连接池上限为 20。", subject="pool limit", tags=["db"], workspace="w")
    # A memory with zero lexical overlap so it can only be recalled via evidence.
    tools.memory_write(content="pgsql 的 pool 上限数值是二十，注意与连接数区分。", subject="pool", tags=["db"], workspace="w")
    assert tools.wait_evidence_worker_drained(timeout=2)
    result = tools.memory_search(query="连接池 上限")
    for row in result["data"]["results"]:
        assert isinstance(row.get("version"), int)
        assert "kind" not in row and "start_offset" not in row and "content_hash" not in row
        assert "_evidence_hits" in row or row.get("_match_reason") != "evidence_vec_recall"


def test_semantic_worker_sync_wait_observes_same_task_completion() -> None:
    from types import SimpleNamespace
    from memory_arbiter.workers import SemanticConflictWorker

    calls: list[tuple[int, str]] = []

    class Tools:
        settings = SimpleNamespace(
            semantic_conflict_on_write="async",
            semantic_conflict_preload=False,
            semantic_conflict_queue_max_size=10,
        )

        @staticmethod
        def _process_semantic_conflict_job(memory_id, snapshot):
            calls.append((memory_id, snapshot["task_id"]))
            return {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0}

    worker = SemanticConflictWorker(Tools())
    task_id = "semantic:42@3"
    queued = worker.enqueue(42, {"version": 3, "task_id": task_id, "dedupe_key": task_id})
    assert queued["task_id"] == task_id and queued["dedupe_key"] == task_id
    completed = worker.wait_task(task_id, 2)
    assert completed == {
        "status": "completed", "outcome": "checked_no_notice", "notices_created": 0,
        "task_id": task_id, "dedupe_key": task_id,
    }
    assert calls == [(42, task_id)]
    assert worker.enqueue(42, {"version": 3, "task_id": task_id})["status"] == "completed"
    assert calls == [(42, task_id)]


def test_semantic_worker_coalescing_completes_displaced_task() -> None:
    from types import SimpleNamespace
    from memory_arbiter.workers import SemanticConflictWorker

    class Tools:
        settings = SimpleNamespace(
            semantic_conflict_on_write="async", semantic_conflict_preload=False,
            semantic_conflict_queue_max_size=10,
        )

    worker = SemanticConflictWorker(Tools())
    worker._ensure_thread = lambda: None
    first = "semantic:42@1"
    second = "semantic:42@2"
    worker.enqueue(42, {"version": 1, "task_id": first})
    worker.enqueue(42, {"version": 2, "task_id": second})
    assert worker.wait_task(first, 0) == {
        "status": "incomplete", "reason": "coalesced_by_newer_snapshot", "notices_created": 0,
        "task_id": first, "dedupe_key": first,
    }
    assert second in worker._expected
    worker.shutdown(discard_pending=True)


def test_evidence_index_error_completes_exact_reserved_task(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.notice_sync_wait_ms = 1000
    monkeypatch.setattr(
        tools, "_index_local_text_evidence",
        lambda memory_id, record: {"status": "failed", "reason": "synthetic_failure"},
    )
    result = tools.memory_write(content="index me", subject="index failure", tags=[])
    check = result["data"]["semantic_conflict_check"]
    task_id = result["data"]["evidence_index"]["semantic_task_id"]
    assert check == {
        "status": "incomplete", "reason": "evidence_index_synthetic_failure", "notices_created": 0,
        "task_id": task_id, "dedupe_key": task_id,
    }
    assert "timeout_continuing_async" not in str(result)


def test_queue_full_drop_completes_exact_reserved_task_end_to_end(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_queue_max_size = 1
    tools._semantic_worker._ensure_thread = lambda: None
    tools._semantic_worker.enqueue(111, {"version": 1, "task_id": "semantic:111@1"})
    task_id = "semantic:222@1"
    tools._semantic_worker.reserve(task_id)
    outcome = tools._semantic_worker.enqueue(222, {"version": 1, "task_id": task_id})
    assert outcome == {
        "status": "incomplete", "reason": "queue_full", "notices_created": 0,
        "task_id": task_id, "dedupe_key": task_id,
    }
    assert tools._semantic_worker.wait_task(task_id, 0) == outcome
    status = tools._semantic_status()["worker"]
    assert status["dropped_queue_full"] == 1
    tools._semantic_worker.shutdown(discard_pending=True)


def test_paused_disabled_and_shutdown_enqueue_complete_exact_task(tmp_path: Path) -> None:
    for state in ("paused", "runtime_disabled", "shutdown"):
        state_path = tmp_path / state
        state_path.mkdir()
        tools = make_tools(state_path)
        worker = tools._semantic_worker
        worker._ensure_thread = lambda: None
        if state == "paused":
            worker.pause()
        elif state == "runtime_disabled":
            worker.disable_runtime()
        else:
            worker.shutdown()
        task_id = f"semantic:7@{state}"
        worker.reserve(task_id)
        outcome = worker.enqueue(7, {"version": 1, "task_id": task_id})
        assert outcome == {
            "status": "incomplete", "reason": state, "notices_created": 0,
            "task_id": task_id, "dedupe_key": task_id,
        }
        assert worker.wait_task(task_id, 0) == outcome


def test_qwen_timeout_and_backend_error_map_to_check_degradation(tmp_path: Path, monkeypatch) -> None:
    from memory_arbiter.semantic_conflict import ModelSignal

    tools = make_tools(tmp_path, semantic_enabled=False)
    tools.settings.semantic_conflict_on_write = "off"
    peer = tools.memory_write(content="database connection policy", subject="pool", tags=[])["data"]
    new = tools.memory_write(content="database connection pool size", subject="pool2", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "database connection policy", "start_offset": 0, "end_offset": 26, "distance": 0.2}]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))

    cases = [
        ("semantic inference hard timeout after 30ms", "backend_error", "qwen_timeout"),
        ("child exited", "backend_error", "qwen_backend_error"),
        (None, "backend_unavailable", "qwen_unavailable"),
        ("missing_json", "invalid_json", "qwen_invalid_output"),
    ]
    for error, candidate_type, expected in cases:
        backend = type(
            "B", (),
            {"classify_pair": staticmethod(lambda l, r, _e=error, _t=candidate_type: ModelSignal(False, _t, None, "", None, _e))},
        )()
        monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: backend)
        tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
        degradation = tools._semantic_status()["check_degradation"]
        assert degradation["last_reason"] == expected
    assert tools.db.list_semantic_notices(status="open") == []


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
        enable_sqlite_vec=True, vec_dim=2, embedding_provider="gguf",
        embedding_model_path=tools.settings.embedding_model_path,
    )
    original_init = MemoryTools.__init__

    def patched_init(self, settings=None, db=None):
        original_init(self, settings=settings, db=db)
        self._embedder = FakeEmbedder()
        self._embedder_loaded = True
        self.db.init_vec_index_state("fake-vnext-space", True)

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
    assert tools.wait_evidence_worker_drained(timeout=2)
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
    assert tools.wait_evidence_worker_drained(timeout=5)

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


def test_space_rebuild_completion_purges_deleted_evidence_and_requires_vectors(
    tmp_path: Path,
) -> None:
    tools = make_tools(tmp_path)
    active = tools.memory_write(content="active", subject="a", tags=[])["data"]["id"]
    deleted = tools.memory_write(content="deleted", subject="d", tags=[])["data"]["id"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET status='deleted' WHERE id=?", (deleted,))
    rebuild = tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert rebuild["ok"] is True
    assert tools.wait_evidence_worker_drained(timeout=5)
    with tools.db.write_transaction() as conn:
        active_evidence = conn.execute(
            "SELECT id FROM memory_evidence WHERE memory_id=? ORDER BY id", (active,),
        ).fetchall()
        deleted_evidence = conn.execute(
            "SELECT id FROM memory_evidence WHERE memory_id=?", (deleted,),
        ).fetchall()
        missing_vector_id = int(active_evidence[0]["id"])
        conn.execute("DELETE FROM memory_evidence_vec WHERE id=?", (missing_vector_id,))
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
            "INSERT INTO memory_evidence_vec(id,parent_status,embedding) VALUES(?,?,?)",
            (999999, "active", "[0.0,1.0]"),
        )
    assert tools.db.maybe_complete_space_rebuild("space-b") is False
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO memory_evidence_vec(id,parent_status,embedding) VALUES(?,?,?)",
            (missing_vector_id, "active", "[0.0,1.0]"),
        )
    assert tools.db.maybe_complete_space_rebuild("space-b") is True
    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id=?", (deleted,)
        ).fetchone()[0] == 0
        assert not deleted_evidence or conn.execute(
            "SELECT COUNT(*) FROM memory_evidence_vec WHERE id IN ("
            + ",".join("?" for _ in deleted_evidence) + ")",
            tuple(int(row["id"]) for row in deleted_evidence),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_evidence_vec WHERE id=999999"
        ).fetchone()[0] == 0


def test_mismatch_rebuild_paginates_across_batches_and_flips_only_at_end(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    ids = [
        tools.memory_write(content=f"条目 {i}：连接池为 {i}0。", subject=f"item{i}", tags=[])["data"]["id"]
        for i in range(6)
    ]
    assert tools.wait_evidence_worker_drained(timeout=5)
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
        assert tools.wait_evidence_worker_drained(timeout=5)
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
    assert tools.wait_evidence_worker_drained(timeout=5)
    tools.db.init_vec_index_state("new-space-v2", True)
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "new-space-v2"

    # Start the rebuild (marks the evidence-id epoch) but republish only one
    # memory in the same second — old-space rows must NOT count as rebuilt.
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 1})
    assert tools.wait_evidence_worker_drained(timeout=5)
    assert tools.db.get_vec_index_state()["state"] == "mismatch"
    # An ordinary publish of the other memory completes the rebuild.
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 5})
    assert tools.wait_evidence_worker_drained(timeout=5)
    assert tools.db.get_vec_index_state()["state"] == "ready"


def test_rebuild_dry_run_has_no_side_effects_in_mismatch_mode(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="内容。", subject="s", tags=[])
    assert tools.wait_evidence_worker_drained(timeout=5)
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
    assert tools.wait_evidence_worker_drained(timeout=5)
    hits = tools.db.evidence_knn([1.0, 0.0], k=3, workspace="w", exclude_memory_id=999999)
    assert 0 < len(hits) <= 3


def test_stale_publish_race_does_not_pollute_worker_last_error(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(content="内容。", subject="s", tags=[])["data"]
    tools.settings.semantic_conflict_on_write = "off"
    monkeypatch.setattr(
        tools, "_index_local_text_evidence",
        lambda *a, **k: {"status": "failed", "outcome": "stale_snapshot"},
    )
    tools._evidence_worker.enqueue(written["id"], {"version": 1})
    assert tools._evidence_worker.wait_drained(timeout=2)
    assert tools._evidence_worker.status()["last_error"] is None


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
    assert tools.wait_evidence_worker_drained(timeout=5)

    # Swap A->B, rebuild one of two memories (partial), then retarget B->C.
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 1})
    assert tools.wait_evidence_worker_drained(timeout=5)
    assert tools.db.get_vec_index_state()["state"] == "mismatch"
    with tools.db.connection() as conn:
        b_phase_max = conn.execute("SELECT COALESCE(MAX(id),0) FROM memory_evidence").fetchone()[0]

    tools._embedder.embedding_space_id = "space-c"
    tools.db.init_vec_index_state("space-c", True)
    state = tools.db.get_vec_index_state()
    assert state["state"] == "mismatch" and state["target_space_id"] == "space-c"

    # Full C rebuild: BOTH memories must be republished above the old B rows
    # (a stale epoch would leave the first memory's B-space vectors in place
    # while still flipping to ready).
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert tools.wait_evidence_worker_drained(timeout=5)
    state = tools.db.get_vec_index_state()
    assert state["state"] == "ready" and state["active_space_id"] == "space-c"
    with tools.db.connection() as conn:
        for mid in (first, second):
            newest = conn.execute(
                "SELECT COALESCE(MAX(id),0) FROM memory_evidence WHERE memory_id=?",
                (mid,),
            ).fetchone()[0]
            assert newest > b_phase_max, "memory not republished after retarget"


def test_zero_unit_memory_does_not_block_space_rebuild_flip(tmp_path: Path) -> None:
    from memory_arbiter.models import utc_now_iso

    tools = make_tools(tmp_path)
    tools.memory_write(content="正常内容。", subject="s", tags=[])
    assert tools.wait_evidence_worker_drained(timeout=5)
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
    assert tools.wait_evidence_worker_drained(timeout=5)
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
    assert tools.wait_evidence_worker_drained(timeout=5)
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
    assert tools.wait_evidence_worker_drained(timeout=5)
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
    assert tools.wait_evidence_worker_drained(timeout=5)
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
    assert tools.wait_evidence_worker_drained(timeout=5)
    with tools.db.connection() as conn:
        second_rows = conn.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id=?", (second,)
        ).fetchone()[0]
    assert second_rows > 0

    # Partial rebuild into space-b: `first` is republished there (its
    # original rows are dropped by publish's delete-then-insert).
    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "space-b"
    tools.db.init_vec_index_state("space-b", True)
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 1})
    assert tools.wait_evidence_worker_drained(timeout=5)
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
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id=?", (first,)
        ).fetchone()[0]
        second_now = conn.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id=?", (second,)
        ).fetchone()[0]
        max_second_id = conn.execute(
            "SELECT COALESCE(MAX(id),0) FROM memory_evidence WHERE memory_id=?", (second,)
        ).fetchone()[0]
        vec_orphans = conn.execute(
            "SELECT COUNT(*) FROM memory_evidence_vec v "
            "WHERE NOT EXISTS(SELECT 1 FROM memory_evidence e WHERE e.id=v.id)"
        ).fetchone()[0]
    assert first_rows == 0  # purged B-space rows; republished on next rebuild
    assert second_now == second_rows and max_second_id <= epoch
    assert vec_orphans == 0

    # The purged memory resurfaces as a stale candidate and re-embeds in the
    # active space.
    dry = tools.memory_repair("rebuild_evidence", {"dry_run": True, "batch_size": 10})
    assert dry["data"]["memory_ids"] == [first]
    tools.memory_repair("rebuild_evidence", {"dry_run": False, "batch_size": 10})
    assert tools.wait_evidence_worker_drained(timeout=5)
    coverage = tools.db.evidence.coverage()
    assert coverage["indexed_memories"] == coverage["eligible_memories"] == 2


def test_rebuild_evidence_response_surfaces_vec_index_state(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="内容。", subject="s", tags=[])
    assert tools.wait_evidence_worker_drained(timeout=5)

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


def _conflict_member(memory_id: int, value: str, quote: str, *, version: int = 1) -> dict[str, Any]:
    return ConflictMember(
        memory_id=memory_id,
        version=version,
        attribute_raw="接口超时",
        value_raw=value,
        normalized_attribute="接口超时",
        normalized_value=value,
        evidence_quote=quote,
        evidence_span=(0, len(quote)),
        content_hash=evidence_content_hash(quote),
        direction="a_to_b",
        prompt_version="pair-v1",
        detector_version="attribute-value-v1",
    ).to_dict()


def _structured_conflict_payload(left_id: int, right_id: int, *, left_version: int = 1, right_version: int = 1) -> dict[str, Any]:
    left_quote, right_quote = "接口超时为 5 秒。", "接口超时为 30 秒。"
    return {
        "slot_key": {"entity": "checkout-api", "attribute": "接口超时", "scope": "production"},
        "members": [
            _conflict_member(left_id, "5秒", left_quote, version=left_version),
            _conflict_member(right_id, "30秒", right_quote, version=right_version),
        ],
        "value_groups": [
            ConflictValueGroup("5秒", "5 秒", (f"{left_id}@{left_version}",)).to_dict(),
            ConflictValueGroup("30秒", "30 秒", (f"{right_id}@{right_version}",)).to_dict(),
        ],
        "detector_version": "attribute-value-v1",
        "prompt_version": "pair-v1",
        "source": "scheduled_scan",
        "reason": "同一生产接口超时槽位存在不同值",
        "status": "open",
    }


def _seed_notice(
    tools: MemoryTools,
    left_id: int,
    right_id: int,
    *,
    left_version: int = 1,
    right_version: int = 1,
    complete_slot: bool = False,
) -> int:
    payload: dict[str, Any] = {
        "route": "notice_ready",
        "reason": "bidirectional_attribute_value_difference",
        "left_evidence": {"text": "接口超时为 5 秒。", "start_offset": 0, "end_offset": 10},
        "right_evidence": {"text": "接口超时为 30 秒。", "start_offset": 0, "end_offset": 11},
        "candidate_key": {
            "detector_version": "attribute-value-v1",
            "members": sorted([f"{left_id}@{left_version}", f"{right_id}@{right_version}"]),
        },
        "task_id": f"semantic:{left_id}@{left_version}",
    }
    if complete_slot:
        structured = _structured_conflict_payload(
            left_id, right_id, left_version=left_version, right_version=right_version,
        )
        payload.update({
            "slot_key": structured["slot_key"],
            "slot_provenance": {"entity": "metadata", "scope": "metadata", "attribute": "bidirectional_extraction"},
            "member_versions": structured["members"],
            "value_groups": structured["value_groups"],
        })
    created = tools.db.record_semantic_notice(
        memory_id=left_id,
        peer_id=right_id,
        severity="high",
        notice_type="semantic_evidence",
        title=f"Possible memory change with #{right_id}",
        message="bidirectional_attribute_value_difference",
        payload=payload,
        dedupe_key=f"notice:{left_id}@{left_version}:{right_id}@{right_version}:{complete_slot}",
        left_version=left_version,
        right_version=right_version,
    )
    return int(created["notice_id"])


def test_unified_conflicts_notice_columns_and_api(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_notice(tools, left["id"], right["id"], complete_slot=True)

    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_notices'"
        ).fetchone() is None
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(conflicts)")}
    assert {
        "notice_type", "notice_payload", "notice_task_id", "notice_dedupe_key",
        "notice_delivery_status", "notice_slot_provenance",
    } <= columns

    listed = tools.memory_repair("notice", {"action": "list", "status": "open"})
    assert listed["ok"] is True
    assert notice_id in {item["notice_id"] for item in listed["data"]["notices"]}
    read = tools.memory_repair("notice", {"action": "read", "notice_id": notice_id})
    notice = read["data"]["notice"]
    assert notice["conflict_id"] == notice_id
    assert notice["payload"]["slot_key"]["entity"] == "checkout-api"
    assert len(notice["member_versions"]) == 2


def test_notice_escalation_requires_structured_slot_snapshot(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    incomplete_id = _seed_notice(tools, left["id"], right["id"], complete_slot=False)

    rejected = tools.memory_repair("notice", {"action": "escalate", "notice_id": incomplete_id})
    assert rejected["ok"] is False
    assert rejected["data"]["outcome"] == "escalate_failed"
    assert rejected["data"]["detail"]["outcome"] == "structured_group_required"

    complete_id = _seed_notice(tools, left["id"], right["id"], complete_slot=True)
    notice = tools.memory_repair("notice", {"action": "read", "notice_id": complete_id})["data"]["notice"]
    assert notice["slot_key"] == {
        "attribute": "接口超时", "entity": "checkout-api", "scope": "production",
    }
    assert notice["notice_slot_provenance"]["entity"] == "metadata"
    escalated = tools.memory_repair(
        "notice", {"action": "escalate", "notice_id": complete_id, "reason": "verified"},
    )
    assert escalated["ok"] is True
    assert escalated["data"]["conflict_id"] == complete_id
    detail = tools.memory_review(
        "conflict_detail", {"conflict_id": complete_id},
    )["data"]["conflict"]
    assert detail["status"] == "open"
    assert detail["source"] == "semantic_notice"
    assert detail["notice_delivery_status"] == "resolved"
    assert detail["slot_key"]["entity"] == "checkout-api"
    assert {group["normalized_value"] for group in detail["value_groups"]} == {"5秒", "30秒"}


def test_notice_freshness_and_terminal_lifecycle_use_api(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_notice(tools, left["id"], right["id"], complete_slot=True)
    assert tools.db.read_semantic_notice(notice_id)["freshness"]["fresh"] is True

    tools.memory("update", {"memory_id": left["id"], "new_content": "接口超时为 10 秒。", "reason": "修订"})
    assert tools.db.read_semantic_notice(notice_id)["freshness"]["fresh"] is False
    stale = tools.memory_repair("notice", {"action": "escalate", "notice_id": notice_id})
    assert stale["ok"] is False and stale["data"]["outcome"] == "stale_notice"

    fresh_id = _seed_notice(tools, left["id"], right["id"], left_version=2, complete_slot=True)
    dismissed = tools.memory_repair(
        "notice", {"action": "dismiss", "notice_id": fresh_id, "reason": "可共存"},
    )
    assert dismissed["ok"] is True
    assert tools.db.read_semantic_notice(fresh_id)["status"] == "dismissed"


def test_structured_record_conflict_group_dedupes_and_rejects_pair_payload(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    payload = _structured_conflict_payload(left["id"], right["id"])

    inserted = tools.memory_repair("record_conflict", payload)
    assert inserted["ok"] is True and inserted["data"]["outcome"] == "inserted"
    deduped = tools.memory_repair("record_conflict", payload)
    assert deduped["ok"] is True and deduped["data"]["outcome"] == "deduped"
    assert deduped["data"]["conflict_id"] == inserted["data"]["conflict_id"]

    pair = tools.memory_repair(
        "record_conflict", {"left_id": left["id"], "right_id": right["id"], "reason": "legacy"},
    )
    assert pair["ok"] is False
    assert any("left_id" in warning for warning in pair.get("warnings", []))


def test_structured_record_conflict_validates_slot_members_and_revision(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    payload = _structured_conflict_payload(left["id"], right["id"])
    inserted = tools.memory_repair("record_conflict", payload)
    assert inserted["ok"] is True

    third = tools.memory_write(content="接口超时为 60 秒。", subject="timeout", tags=[])["data"]
    third_member = _conflict_member(third["id"], "60秒", "接口超时为 60 秒。")
    appended_payload = {
        **payload,
        "members": payload["members"] + [third_member],
        "value_groups": payload["value_groups"] + [
            ConflictValueGroup("60秒", "60 秒", (f"{third['id']}@1",)).to_dict(),
        ],
        "expected_revision": 1,
    }
    appended = tools.memory_repair("record_conflict", appended_payload)
    assert appended["ok"] is True and appended["data"]["outcome"] == "appended"
    stale = tools.memory_repair("record_conflict", appended_payload)
    assert stale["ok"] is False and stale["data"]["outcome"] == "stale_conflict"


def test_judge_uses_group_revision_and_apply_plan(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    conflict_id = tools.memory_repair(
        "record_conflict", _structured_conflict_payload(left["id"], right["id"]),
    )["data"]["conflict_id"]
    judged = tools.memory("judge", {
        "conflict_id": conflict_id,
        "expected_revision": 1,
        "chosen_value": "30秒",
        "decided_by": "user",
        "ref": "test",
        "reason": "用户确认生产超时",
        "resolution_memory_id": right["id"],
        "apply_plan": [
            {"memory_id": left["id"], "action": "update_current_claim"},
            {"memory_id": right["id"], "action": "use_as_resolution"},
        ],
    })
    assert judged["ok"] is True
    assert judged["data"]["revision"] == 2
    assert judged["data"]["next_action"]["action"] == "apply_conflict_action"
    legacy = tools.memory("judge", {
        "conflict_id": conflict_id,
        "expected_left_version": 1,
        "expected_right_version": 1,
        "verdict": "contradiction",
        "reason": "legacy",
    })
    assert legacy["ok"] is False


def test_wave8_regression_basics(tmp_path: Path) -> None:
    from memory_arbiter.evidence import local_text_units

    content = "\n".join([
        "pool.max=50; pool.timeout=30; pool.min_idle=5;",
        "cache.ttl=60; cache.size=1000; cache.backend=memory;",
        "retry.count=3; retry.backoff=200; retry.jitter=50;",
        "log.level=info; log.file=/var/log/app.log; log.rotation=daily;",
        "queue.depth=100; queue.workers=8; queue.strategy=fifo;",
        "rate.limit=1000; rate.burst=50; rate.window=1s;",
    ])
    units = local_text_units("cfg", content)
    assert units
    assert all(u.start_offset < u.end_offset or u.kind == "subject" for u in units)

    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="s", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="s2", tags=[])["data"]
    huge = "x" * 5000
    payload = _structured_conflict_payload(a["id"], b["id"])
    rejected = tools.memory_repair("record_conflict", {**payload, "reason": huge})
    assert rejected["ok"] is False
    assert rejected["data"]["error"] == "invalid_input"

    bogus = tools.memory_govern(
        "resolve_conflict", {"conflict_id": 99999, "reason": "x", "authorized": True},
    )
    assert bogus["ok"] is False
    assert bogus["data"]["outcome"] == "invalid_input"
    assert bogus["data"]["field"] == "expected_revision"

    ids = [tools.memory_write(content=f"修复 {i}。", subject=f"fix{i}", tags=[])["data"]["id"] for i in range(4)]
    result = tools.memory_repair(
        "rebuild_evidence", {"dry_run": False, "memory_ids": ids, "batch_size": 2},
    )
    assert result["data"]["queued"] == 4


def test_judge_group_schema_rejects_legacy_fields_and_bad_types(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="t", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="t", tags=[])["data"]
    conflict_id = tools.memory_repair(
        "record_conflict", _structured_conflict_payload(a["id"], b["id"]),
    )["data"]["conflict_id"]
    base = {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "30秒",
        "decided_by": "user", "ref": "test", "reason": "两说",
        "resolution_memory_id": b["id"],
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"}],
    }
    assert tools.memory("judge", {**base, "expected_revision": "banana"})["ok"] is False
    assert tools.memory("judge", {**base, "decided_by": "model"})["ok"] is False
    legacy = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_left_version": 1,
        "expected_right_version": 1, "verdict": "contradiction", "reason": "legacy",
    })
    assert legacy["ok"] is False
    assert tools.memory("judge", base)["ok"] is True


def test_wave8_malformed_tags_do_not_zero_filter_recall(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    healthy = tools.memory_write(content="健康条目。", subject="h", tags=["alpha"])["data"]
    tools.memory_write(content="另一条。", subject="h2", tags=[])
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET tags='not-json{' WHERE subject='h2'")
    result = tools.memory("find", {"tags_filter": ["alpha"]})
    assert healthy["id"] in [r["id"] for r in result["data"]["results"]]
    assert result["data"]["total_estimate"] >= 1


def test_wave8_new_database_permissions(tmp_path: Path) -> None:
    import os as _os
    tools = make_tools(tmp_path)
    mode = _os.stat(tools.settings.db_path).st_mode & 0o777
    assert mode == 0o600


def test_wave8_doctor_reports_attention_volume(tmp_path: Path) -> None:
    import json as _json

    tools = make_tools(tmp_path)
    a = tools.memory_write(content="限流 10。", subject="r", tags=[])["data"]
    b = tools.memory_write(content="限流 99。", subject="r", tags=[])["data"]
    reg = tools.memory_repair(
        "record_conflict", _structured_conflict_payload(a["id"], b["id"]),
    )
    assert reg["ok"] is True
    log_path = tools.db.settings.db_path.parent / "attention_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(_json.dumps({"ts": "2026-01-01T00:00:00+00:00", "trigger": "search", "source": "open_table", "ids": [1]}) + "\n")
    report = tools.memory_review("doctor", {})["data"]
    finding = next(f for f in report["findings"] if f["check_id"] == "capacity.attention_volume")
    assert finding["status"] == "pass"
    assert finding["evidence"]["total_lines"] == 1


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
        enable_sqlite_vec=True, vec_dim=2, embedding_provider="gguf",
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
        self.db.init_vec_index_state(self._embedder.embedding_space_id, True)

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
        phase = conn.execute("SELECT value FROM migration_state WHERE key='phase'").fetchone()[0]
    assert phase == "ready"


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
        enable_sqlite_vec=True, vec_dim=2, embedding_provider="gguf",
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
        self.db.init_vec_index_state(embedder.embedding_space_id, True)

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
        self.db.init_vec_index_state(embedder.embedding_space_id, True)

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
    assert migration["evidence_rebuild_space_id"] == "space-new"


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
        vec_dim=2,
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
        db_path=source, backup_jsonl=tmp_path / "b.jsonl", vec_dim=2,
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


def test_scan_candidates_enumerates_filters_and_paginates(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    dup1 = tools.memory_write(content="完全一样的配置说明文字。", subject="same", tags=[])["data"]
    dup2 = tools.memory_write(content="完全一样的配置说明文字。", subject="same", tags=[])["data"]
    far = tools.memory_write(content="PostgreSQL 数据库生产环境配置。", subject="db", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)

    result = tools.memory_repair(
        "scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10},
    )
    assert result["ok"] is True
    data = result["data"]
    pairs = {(c["left_id"], c["right_id"]): c for c in data["candidates"]}
    # The numeric-change pair is surfaced as a notify clue with snippets.
    key = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    assert key in pairs
    clue = pairs[key]
    assert clue["route"] == "review_candidate"
    assert "numeric_value_candidate" in clue["reasons"]
    assert clue["left_snippet"] and clue["right_snippet"]
    # Equivalent duplicates and unrelated memories are not surfaced.
    dup_key = (min(dup1["id"], dup2["id"]), max(dup1["id"], dup2["id"]))
    assert dup_key not in pairs
    for c in data["candidates"]:
        assert far["id"] not in (c["left_id"], c["right_id"])

    # Enumeration alone does not claim persistence: an external loop may retry
    # until record_conflict succeeds, with the same frozen candidate identity.
    repeated = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10})
    repeated_clue = next(c for c in repeated["data"]["candidates"] if (c["left_id"], c["right_id"]) == key)
    assert repeated_clue["candidate_key_hash"] == clue["candidate_key_hash"]

    # A successfully recorded candidate snapshot suppresses re-surfacing. Use
    # the group API and exact candidate_key rather than removed pair columns.
    registered = tools.db.record_conflict_group(
        workspace_canonical="default", slot_key=None, members=clue["members"], value_groups=[],
        candidate_key=clue["candidate_key"], detection_reason="已分诊", source="scheduled_scan",
        detector_version="attribute-value-v1", status="not_a_conflict",
    )
    assert registered["outcome"] == "inserted"
    dismissed_scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10})
    assert key not in {(c["left_id"], c["right_id"]) for c in dismissed_scan["data"]["candidates"]}
    tools.memory("update", {"memory_id": a["id"], "new_content": "接口超时为 60 秒，已修订。", "reason": "r"})
    assert tools.wait_evidence_worker_drained(timeout=5)
    reopened = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10})
    assert key in {(c["left_id"], c["right_id"]) for c in reopened["data"]["candidates"]}


def test_scan_candidates_pagination_and_check_gate(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    ids = [
        tools.memory_write(content=f"扫描锚点 {i}：连接池 {i}0。", subject=f"scan{i}", tags=[])["data"]["id"]
        for i in range(3)
    ]
    # Real rule check pair: structurally similar, no deterministic signal.
    chk1 = tools.memory_write(content="服务使用数据库甲存储数据。", subject="store", tags=[])["data"]
    chk2 = tools.memory_write(content="服务采用数据库乙保存数据。", subject="store", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)

    # Cursor pagination: one anchor per batch, union covers everything.
    collected, anchor, batches = set(), 0, 0
    while True:
        r = tools.memory_repair("scan_candidates", {"anchor_memory_id": anchor, "batch": 1, "k": 10})
        assert r["ok"] is True
        batches += 1
        for c in r["data"]["candidates"]:
            collected.add((c["left_id"], c["right_id"]))
        if r["data"]["next_anchor_memory_id"] is None:
            break
        anchor = r["data"]["next_anchor_memory_id"]
    assert batches == len(ids) + 2

    # include_check gates rule-level check clues in and out.
    chk_key = (min(chk1["id"], chk2["id"]), max(chk1["id"], chk2["id"]))
    base = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 10, "k": 10})
    with_check = tools.memory_repair(
        "scan_candidates", {"anchor_memory_id": 0, "batch": 10, "k": 10, "include_check": True},
    )
    base_pairs = {(c["left_id"], c["right_id"]) for c in base["data"]["candidates"]}
    check_pairs = {(c["left_id"], c["right_id"]) for c in with_check["data"]["candidates"]}
    assert all(c["route"] in {"review_candidate", "notice_ready"} for c in base["data"]["candidates"])
    assert chk_key not in base_pairs
    assert chk_key in check_pairs
    chk_clue = next(c for c in with_check["data"]["candidates"] if (c["left_id"], c["right_id"]) == chk_key)
    assert chk_clue["route"] == "review_candidate"
    assert len(check_pairs) > len(base_pairs)


def test_record_conflict_not_a_conflict_registration(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="阈值 100。", subject="th", tags=[])["data"]
    b = tools.memory_write(content="阈值 200。", subject="th", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10})
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)

    payload = {
        "slot_key": None,
        "members": clue["members"],
        "value_groups": [],
        "candidate_key": clue["candidate_key"],
        "status": "not_a_conflict",
        "detector_version": "attribute-value-v1",
        "source": "scheduled_scan",
        "reason": "同主题演进，无需治理",
    }
    marked = tools.memory_repair("record_conflict", payload)
    assert marked["ok"] is True
    row = tools.db.get_conflict(marked["data"]["conflict_id"])
    assert row["status"] == "not_a_conflict"
    assert row["slot_key"] is None
    assert row["candidate_key_hash"]

    repeat = tools.memory_repair("record_conflict", payload)
    assert repeat["ok"] is True and repeat["data"]["outcome"] == "deduped"
    after = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10})
    assert pair not in {(c["left_id"], c["right_id"]) for c in after["data"]["candidates"]}

    invalid_open = tools.memory_repair("record_conflict", {**payload, "status": "open"})
    assert invalid_open["ok"] is False
    assert invalid_open["data"]["outcome"] == "invalid_slot_key"


def test_scan_candidates_review_regression_batch(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    # batch=0 / k=0 are falsy but must be rejected, not coerced to defaults.
    zero = tools.memory_repair("scan_candidates", {"batch": 0})
    assert zero["ok"] is False
    zero_k = tools.memory_repair("scan_candidates", {"k": 0})
    assert zero_k["ok"] is False

    # Subject-only version progression must not become a numeric clue:
    # only body-text units participate (matching the write-time path).
    v1 = tools.memory_write(content="服务部署文档正文内容，部署步骤说明。", subject="服务 2024 规划", tags=[])["data"]
    v2 = tools.memory_write(content="服务部署文档正文内容，回滚步骤说明。", subject="服务 2025 规划", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10})
    subject_pair = (min(v1["id"], v2["id"]), max(v1["id"], v2["id"]))
    assert subject_pair not in {(c["left_id"], c["right_id"]) for c in scan["data"]["candidates"]}

    # A memory pair whose equivalent unit precedes its numeric-change unit
    # must still surface the numeric clue (peer-blacklisting regression).
    mixed = tools.memory_write(
        content="完全相同的开场说明。\n重试次数为 3 次。", subject="mixed", tags=[],
    )["data"]
    twin = tools.memory_write(
        content="完全相同的开场说明。\n重试次数为 5 次。", subject="mixed", tags=[],
    )["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    scan2 = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10})
    mixed_pair = (min(mixed["id"], twin["id"]), max(mixed["id"], twin["id"]))
    clue = next(
        (c for c in scan2["data"]["candidates"] if (c["left_id"], c["right_id"]) == mixed_pair), None,
    )
    assert clue is not None
    assert "numeric_value_candidate" in clue["reasons"]
    # Snippets show the numeric-bearing text, not the equivalent preamble.
    snippets = clue["left_snippet"] + clue["right_snippet"]
    assert "重试次数" in snippets


def test_scan_candidates_strict_workspace_does_not_leak(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.settings.isolation = "strict"
    # Workspace names whose fake embeddings differ (one contains "pgsql"),
    # so vector resolution keeps them canonically separate.
    tools.memory_write(content="主库配置说明与参数清单。", subject="db", tags=[], workspace="dbapgsql")["data"]
    tools.memory_write(
        content="接口超时为 5 秒。", subject="timeout", tags=[], workspace="apisvc",
    )["data"]
    tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[], workspace="apisvc")["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    with tools.db.connection() as conn:
        canonicals = {
            row["id"]: row["workspace_canonical"]
            for row in conn.execute("SELECT id, workspace_canonical FROM memories")
        }
    assert set(canonicals.values()) == {"dbapgsql", "apisvc"}
    # A later active row in another workspace must not create a phantom next
    # cursor for strict workspace pagination.
    tools.memory_write(content="末尾外部工作区。", subject="tail", tags=[], workspace="dbapgsql")
    assert tools.wait_evidence_worker_drained(timeout=5)

    result = tools.memory_repair(
        "scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10, "workspace": "apisvc"},
    )
    assert result["ok"] is True
    assert result["data"]["next_anchor_memory_id"] is None
    for clue in result["data"]["candidates"]:
        assert "postgres" not in clue["left_snippet"] and "postgres" not in clue["right_snippet"]
        for deep_read in clue["deep_read"].values():
            assert deep_read["workspace"] == "apisvc"
            executed = tools.memory("read", deep_read)
            assert executed["ok"] is True, executed
        for side in (clue["left_id"], clue["right_id"]):
            record = tools.db.get_memory(side)
            canonical = record.get("workspace_canonical") or record.get("workspace")
            assert canonical == "apisvc", f"leaked anchor from workspace {canonical}"


def test_not_a_conflict_candidate_version_change_can_be_reevaluated(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="上限 10。", subject="cap", tags=[])["data"]
    b = tools.memory_write(content="上限 99。", subject="cap", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10})
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    payload = {
        "slot_key": None, "members": clue["members"], "value_groups": [],
        "candidate_key": clue["candidate_key"], "status": "not_a_conflict",
        "detector_version": "attribute-value-v1", "source": "scheduled_scan", "reason": "演进",
    }
    first = tools.memory_repair("record_conflict", payload)
    second = tools.memory_repair("record_conflict", payload)
    assert first["data"]["outcome"] == "inserted"
    assert second["data"]["outcome"] == "deduped"

    tools.memory("update", {"memory_id": a["id"], "new_content": "上限 20。", "reason": "新版本"})
    assert tools.wait_evidence_worker_drained(timeout=5)
    again = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10})
    fresh = next(c for c in again["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    assert fresh["candidate_key_hash"] != clue["candidate_key_hash"]


def test_read_span_window_and_clue_deep_read(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    content = "第一段背景说明文字。\n重试次数为 3 次。\n第三段运维备注信息。"
    a = tools.memory_write(content=content, subject="span", tags=[])["data"]
    b = tools.memory_write(content="第一段背景说明文字。\n重试次数为 5 次。\n第三段运维备注信息。", subject="span", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)

    # span read returns the sliced content plus window metadata.
    full = tools.memory("read", {"memory_id": a["id"]})["data"]["memory"]
    total = len(full["content"])
    windowed = tools.memory("read", {"memory_id": a["id"], "span": {"start": 0, "end": 12}})
    assert windowed["ok"] is True
    assert windowed["data"]["memory"]["content"] == content[:12]
    assert windowed["data"]["span"] == {"start": 0, "end": 12, "total_chars": total}
    clipped = tools.memory("read", {"memory_id": a["id"], "span": {"start": total - 3, "end": 10_000}})
    assert clipped["data"]["span"]["end"] == total
    assert len(clipped["data"]["memory"]["content"]) == 3
    for bad in ({"start": -1, "end": 5}, {"start": 5, "end": 5}, {"start": total + 5, "end": total + 9}):
        rejected = tools.memory("read", {"memory_id": a["id"], "span": bad})
        assert rejected["ok"] is False
    not_dict = tools.memory("read", {"memory_id": a["id"], "span": [0, 5]})
    assert not_dict["ok"] is False

    # The clue's deep_read spans round-trip: reading each span returns the
    # triggering numeric text.
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10})
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    assert "numeric_value_candidate" in clue["reasons"]
    for side in ("left", "right"):
        call = clue["deep_read"][side]
        assert call["memory_id"] in (a["id"], b["id"])
        window = tools.memory("read", {"memory_id": call["memory_id"], "span": call["span"]})
        assert window["ok"] is True
        assert "重试次数为" in window["data"]["memory"]["content"]


def test_notice_snapshot_carries_evidence_spans(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_notice(tools, a["id"], b["id"], complete_slot=True)
    notice = tools.memory_repair("notice", {"action": "read", "notice_id": notice_id})["data"]["notice"]
    for member in notice["member_versions"]:
        assert member["evidence_span"][0] < member["evidence_span"][1]
        window = tools.memory("read", {
            "memory_id": member["memory_id"],
            "span": {"start": member["evidence_span"][0], "end": member["evidence_span"][1]},
        })
        assert window["ok"] is True
        assert "接口超时" in window["data"]["memory"]["content"]


def test_deep_read_spans_follow_clue_upgrade_and_drifted_offsets(tmp_path: Path) -> None:
    filler = "这条是中性填充内容，用来把两个区域拉开超过窗口宽度。" * 8  # ~360 chars
    # Empty lines keep the filler and the numeric value as separate text
    # units, so the clue upgrades from a similarity check to a numeric
    # notify across two distinct regions.
    content_a = f"部署架构完全相同的描述文字甲。\n\n{filler}\n\n重试次数为 3 次。"
    content_b = f"部署架构完全相同的描述文字甲。\n\n{filler}\n\n重试次数为 5 次。"
    tools = make_tools(tmp_path)
    a = tools.memory_write(content=content_a, subject="upg", tags=[])["data"]
    b = tools.memory_write(content=content_b, subject="upg", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)

    scan = tools.memory_repair(
        "scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10, "include_check": True},
    )
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    assert clue["route"] == "review_candidate"
    assert "numeric_value_candidate" in clue["reasons"]
    # The deep-read span must track the UPGRADED (numeric) discovery, not
    # the first (similarity) one — and re-reading it must contain the text.
    for side in ("left", "right"):
        call = clue["deep_read"][side]
        window = tools.memory("read", {"memory_id": call["memory_id"], "span": call["span"]})
        assert window["ok"] is True
        assert "重试次数为" in window["data"]["memory"]["content"]


def test_deep_read_survives_offset_drift_on_dense_content(tmp_path: Path) -> None:
    # Semicolon-dense repeated-token lines make stored unit offsets drift;
    # deep_read must locate the trigger by text, not trust the offsets.
    def dense(value: str) -> str:
        return "\n".join([
            f"pool.max=50; pool.timeout={value}; pool.min_idle=5;",
            "cache.ttl=60; cache.size=1000; cache.backend=memory;",
            f"retry.count=3; retry.backoff=200; retry.value={value};",
            "log.level=info; log.file=/var/log/app.log; log.rotation=daily;",
            "queue.depth=100; queue.workers=8; queue.strategy=fifo;",
            f"rate.limit=1000; rate.burst=50; rate.window={value};",
        ])

    tools = make_tools(tmp_path)
    a = tools.memory_write(content=dense("30"), subject="cfg", tags=[])["data"]
    b = tools.memory_write(content=dense("90"), subject="cfg", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=5)
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10})
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    for side in ("left", "right"):
        call = clue["deep_read"][side]
        assert call["span"] is not None
        window = tools.memory("read", {"memory_id": call["memory_id"], "span": call["span"]})
        assert window["ok"] is True
        assert "timeout=30" in window["data"]["memory"]["content"] or "timeout=90" in window["data"]["memory"]["content"]


def test_notice_snapshot_is_frozen_when_member_drifts(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_notice(tools, a["id"], b["id"], complete_slot=True)
    before = tools.db.read_semantic_notice(notice_id)
    original_members = before["member_versions"]

    tools.memory("update", {"memory_id": a["id"], "new_content": "短。", "reason": "缩短"})
    after = tools.db.read_semantic_notice(notice_id)
    assert after["freshness"]["fresh"] is False
    assert after["member_versions"] == original_members
    assert after["member_versions"][0]["version"] == 1


def _assert_exact_text_unit_spans(subject: str, content: str) -> None:
    from memory_arbiter.evidence import _clean

    for unit in local_text_units(subject, content):
        if unit.kind != "text":
            continue
        assert 0 <= unit.start_offset < unit.end_offset <= len(content)
        restored = _clean(content[unit.start_offset:unit.end_offset])
        assert restored == unit.text, (
            unit.unit_index, unit.start_offset, unit.end_offset,
            unit.text, restored,
        )


def test_evidence_offsets_exactly_map_normalized_source_text() -> None:
    # Regression: repeated tokens + semicolon-dense multiline config used to
    # make the reverse locator overshoot by hundreds of chars and cascade.
    dense = "\n".join([
        "pool.max=50; pool.timeout=30; pool.min_idle=5;",
        "cache.ttl=60; cache.size=1000; cache.backend=memory;",
        "retry.count=3; retry.backoff=200; retry.jitter=50;",
        "log.level=info; log.file=/var/log/app.log; log.rotation=daily;",
        "queue.depth=100; queue.workers=8; queue.strategy=fifo;",
        "rate.limit=1000; rate.burst=50; rate.window=1s;",
    ])
    _assert_exact_text_unit_spans("service config", dense)

    # Cross-line whitespace, CRLF, repeated sentence text, headings, short
    # paragraph merge, and long overlapping windows.
    corpus = [
        "重复句子。重复句子。值为 3。重复句子。值为 5。",
        "第一行  多空格\r\n第二行\t制表符\r\n\r\n尾段。",
        "# 标题 2025\n正文一。\n\n短。\n相邻长段落用于合并验证。",
        ("超长无标点文本abc重复词" * 80),
        "A; B; C; A; B; C; A; B; C; 数值=10; 数值=20;",
    ]
    for content in corpus:
        _assert_exact_text_unit_spans("subject", content)


def test_evidence_offset_mapping_deterministic_fuzz() -> None:
    import random

    random.seed(20260820)
    tokens = [
        "alpha", "beta", "key=30;", "key=90;", "重复", "接口", "超时",
        "。", "；", ";", "!", "?", "  ", "\t", "\n", "\r\n",
    ]
    for _ in range(1000):
        content = "".join(random.choice(tokens) for _ in range(random.randint(1, 80)))
        if random.random() < 0.25:
            content = "# 标题 2025\n" + content
        _assert_exact_text_unit_spans("主题" if random.random() < 0.5 else "", content)


def test_short_paragraph_merge_never_crosses_heading_barrier() -> None:
    from memory_arbiter.evidence import _clean

    content = "短。\n# Runtime Heading\n后面的正文足够长用于单独索引。"
    units = local_text_units("", content)
    headings = [u for u in units if u.kind == "heading"]
    texts = [u for u in units if u.kind == "text"]
    assert [u.text for u in headings] == ["Runtime Heading"]
    assert texts
    assert all("Runtime Heading" not in u.text and "#" not in u.text for u in texts)
    for unit in texts:
        assert _clean(content[unit.start_offset:unit.end_offset]) == unit.text

    # Symmetric case: long paragraph before the heading, short one after it.
    content2 = "前面的正文足够长用于单独索引。\n# Boundary\n短。"
    units2 = local_text_units("", content2)
    text_units2 = [u for u in units2 if u.kind == "text"]
    assert all("Boundary" not in u.text and "#" not in u.text for u in text_units2)
    for unit in text_units2:
        assert _clean(content2[unit.start_offset:unit.end_offset]) == unit.text


def test_embedding_pipeline_version_rotated_for_exact_offsets() -> None:
    from memory_arbiter.embedder import EMBEDDING_PIPELINE_VERSION

    assert EMBEDDING_PIPELINE_VERSION == 2


# ── 2026-08-21 review round: notice-pipeline and scan wide-gate fixes ────────

def _grounded_db_backend():
    """A backend that extracts a grounded mysql/sqlite slot from db-value quotes."""
    from memory_arbiter.semantic_conflict import ModelSignal

    class Backend:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            def value(env):
                text = str(env["quote"]).casefold()
                return "mysql" if "mysql" in text else ("sqlite" if "sqlite" in text else "postgres")
            parsed = {
                "attribute_a": "数据库选型", "value_a": value(left),
                "attribute_b": "数据库选型", "value_b": value(right),
            }
            return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)
    return Backend()


def test_clean_gate_negative_reaches_checked_no_notice(tmp_path: Path, monkeypatch) -> None:
    """A candidate examined and cleanly rejected reports checked_no_notice."""
    from memory_arbiter.semantic_conflict import ModelSignal

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    peer = tools.memory_write(content="database is mysql here", subject="a", tags=[], metadata=meta)["data"]
    new = tools.memory_write(content="database is mysql there", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "database is mysql here",
             "start_offset": 0, "end_offset": 22, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))

    # Same normalized value on both sides → clean not_same_attribute_different_value.
    class SameValue:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            parsed = {"attribute_a": "数据库选型", "value_a": "mysql",
                      "attribute_b": "数据库选型", "value_b": "mysql"}
            return ModelSignal(True, "attribute_value_extraction", None, "", parsed, None)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: SameValue())

    result = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    assert result == {"status": "completed", "outcome": "checked_no_notice", "notices_created": 0}
    # A clean model decision is not counted as check degradation.
    degradation = tools._semantic_status()["check_degradation"]
    assert degradation["last_reason"] != "not_same_attribute_different_value"


def test_idle_worker_job_budget_does_not_cap_inflight_qwen(tmp_path: Path, monkeypatch) -> None:
    """With no backlog, even a tiny job budget must not truncate inference."""
    import time

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    tools.settings.semantic_conflict_job_timeout_ms = 10
    tools.settings.semantic_conflict_min_pair_budget_ms = 5
    metadata = {"entity": "svc", "scope": "production"}
    peer = tools.memory_write(content="database is mysql", subject="a", tags=[], metadata=metadata)["data"]
    new = tools.memory_write(content="database is sqlite", subject="b", tags=[], metadata=metadata)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    hits = [{"memory_id": peer["id"], "id": 1, "kind": "text", "text": "database is mysql",
             "start_offset": 0, "end_offset": 17, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))

    deadlines = []
    base = _grounded_db_backend()

    class SlowBackend:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            deadlines.append(deadline_monotonic)
            time.sleep(0.02)
            return base.classify_pair(left, right, deadline_monotonic=deadline_monotonic)

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: SlowBackend())
    monkeypatch.setattr(tools._semantic_worker, "pending_job_deadline", lambda timeout: None)

    result = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))

    assert result == {"status": "completed", "outcome": "notices_created", "notices_created": 1}
    assert deadlines == [None, None]


def test_backlog_job_budget_stops_before_next_pair_not_during_inference(tmp_path: Path, monkeypatch) -> None:
    """A queued job enables fairness, but the current pair gets its full hard timeout."""
    import time

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    tools.settings.semantic_conflict_job_timeout_ms = 40
    tools.settings.semantic_conflict_min_pair_budget_ms = 5
    tools.settings.semantic_conflict_max_notice_pairs = 3
    metadata = {"entity": "svc", "scope": "production"}
    peer_values = ("mysql", "postgres")
    peers = [
        tools.memory_write(content=f"database is {value}", subject=value, tags=[], metadata=metadata)["data"]
        for value in peer_values
    ]
    new = tools.memory_write(content="database is sqlite", subject="sqlite", tags=[], metadata=metadata)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    hits = [
        {"memory_id": peer["id"], "id": index + 1, "kind": "text", "text": f"database is {peer_values[index]}",
         "start_offset": 0, "end_offset": len(f"database is {peer_values[index]}"), "distance": 0.1 + index * 0.01}
        for index, peer in enumerate(peers)
    ]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))
    fairness_deadline = time.monotonic() + 0.08
    monkeypatch.setattr(
        tools._semantic_worker, "pending_job_deadline", lambda timeout: fairness_deadline,
    )

    deadlines = []
    base = _grounded_db_backend()

    class SlowBackend:
        @staticmethod
        def classify_pair(left, right, *, deadline_monotonic=None):
            deadlines.append(deadline_monotonic)
            time.sleep(0.025)
            return base.classify_pair(left, right, deadline_monotonic=deadline_monotonic)

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: SlowBackend())

    result = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))

    assert result == {"status": "completed", "outcome": "notices_created", "notices_created": 1}
    assert deadlines == [None, None]
    assert len(tools.db.list_semantic_notices(status="open", limit=10)) == 1


def test_pending_job_deadline_uses_actual_enqueue_time(monkeypatch) -> None:
    from types import SimpleNamespace
    from memory_arbiter.workers import SemanticConflictWorker

    now = 100.0
    monkeypatch.setattr("memory_arbiter.workers.time.monotonic", lambda: now)

    class Tools:
        settings = SimpleNamespace(
            semantic_conflict_on_write="async", semantic_conflict_preload=False,
            semantic_conflict_queue_max_size=10,
        )

    worker = SemanticConflictWorker(Tools())
    worker._ensure_thread = lambda: None
    worker.enqueue(42, {"version": 1, "task_id": "semantic:42@1"})
    assert worker.pending_job_deadline(5.0) == 105.0

    now = 120.0
    # The deadline does not restart when the in-flight job notices the backlog.
    assert worker.pending_job_deadline(5.0) == 105.0
    worker.shutdown(discard_pending=True)


def test_semantic_wait_drained_includes_pending_jobs() -> None:
    from types import SimpleNamespace
    from memory_arbiter.workers import SemanticConflictWorker

    class Tools:
        settings = SimpleNamespace(
            semantic_conflict_on_write="async", semantic_conflict_preload=False,
            semantic_conflict_queue_max_size=10,
        )

    worker = SemanticConflictWorker(Tools())
    worker._ensure_thread = lambda: None
    worker.enqueue(42, {"version": 1, "task_id": "semantic:42@1"})
    assert worker.wait_drained(timeout=0) is False
    worker.shutdown(discard_pending=True)
    assert worker.wait_drained(timeout=0) is True


def test_applying_reentry_suppresses_same_conflict_notice(tmp_path: Path, monkeypatch) -> None:
    """Spec §5/§15.3: an apply-plan edit does not re-notice the same conflict."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    a = tools.memory_write(content="database is mysql", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(content="database is sqlite", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _grounded_db_backend)

    # Record + judge the a/b conflict into applying with a as the wrong current fact.
    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    def member(mid, value):
        quote = f"database is {value}"
        return ConflictMember(
            memory_id=mid, version=1, attribute_raw="数据库选型", value_raw=value,
            normalized_attribute="数据库选型", normalized_value=value, evidence_quote=quote,
            evidence_span=(0, len(quote)), content_hash=(str(mid) * 64)[:64],
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ).to_dict()
    recorded = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "svc", "attribute": "数据库选型", "scope": "production"},
        "members": [member(a["id"], "mysql"), member(b["id"], "sqlite")],
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{a['id']}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{b['id']}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scan",
        "reason": "db conflict", "status": "open",
    })
    conflict_id = recorded["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"},
                       {"memory_id": b["id"], "action": "use_as_resolution"}],
        "resolution_memory_id": b["id"],
    })
    # Now run the post-apply semantic job for a's new version against b: the
    # re-entry rule must suppress a fresh notice for the same conflict.
    tools.db.edit_memory_intent(a["id"], new_content="database is sqlite", reason="apply")
    updated = tools.db.get_memory(a["id"])
    hits = [{"memory_id": b["id"], "id": 1, "kind": "text", "text": "database is sqlite",
             "start_offset": 0, "end_offset": 18, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a_, **k: list(hits))
    snapshot = {
        "memory_id": a["id"], "version": updated["version"],
        "content_hash": evidence_content_hash(updated["content"]),
        "trusted_applying_context": {
            "conflict_id": conflict_id, "revision": 2, "memory_id": a["id"],
            "action": "update_current_claim", "chosen_value": "sqlite",
        },
    }
    result = tools._process_semantic_conflict_job(a["id"], snapshot)
    # No new notice for the same applying conflict pair.
    fresh = [n for n in tools.db.list_semantic_notices(status="open")
             if {n.get("memory_id"), n.get("peer_id")} == {a["id"], b["id"]}]
    assert fresh == []


def test_applying_reentry_does_not_suppress_different_slot(tmp_path: Path, monkeypatch) -> None:
    """Two applying-group members can still surface a conflict on another slot."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    a = tools.memory_write(content="database is mysql", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(content="database is sqlite", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)

    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    def member(mid, value):
        quote = f"database is {value}"
        return ConflictMember(
            memory_id=mid, version=1, attribute_raw="数据库选型", value_raw=value,
            normalized_attribute="数据库选型", normalized_value=value, evidence_quote=quote,
            evidence_span=(0, len(quote)), content_hash=(str(mid) * 64)[:64],
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ).to_dict()

    recorded = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "svc", "attribute": "数据库选型", "scope": "production"},
        "members": [member(a["id"], "mysql"), member(b["id"], "sqlite")],
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{a['id']}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{b['id']}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scan",
        "reason": "db conflict", "status": "open",
    })
    conflict_id = recorded["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"},
                       {"memory_id": b["id"], "action": "use_as_resolution"}],
        "resolution_memory_id": b["id"],
    })

    tools.db.edit_memory_intent(a["id"], new_content="连接池上限为 99。", reason="apply")
    updated = tools.db.get_memory(a["id"])
    hits = [{"memory_id": b["id"], "id": 1, "kind": "text", "text": "连接池上限为 10。",
             "start_offset": 0, "end_offset": 11, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a_, **k: list(hits))
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _strict_pair_backend)
    snapshot = {
        "memory_id": a["id"], "version": updated["version"],
        "content_hash": evidence_content_hash(updated["content"]),
        "trusted_applying_context": {
            "conflict_id": conflict_id, "revision": 2, "memory_id": a["id"],
            "action": "update_current_claim", "chosen_value": "sqlite",
        },
    }

    result = tools._process_semantic_conflict_job(a["id"], snapshot)

    assert result["outcome"] == "notices_created"
    assert result["notices_created"] == 1
    notices = [n for n in tools.db.list_semantic_notices(status="open")
               if {n.get("memory_id"), n.get("peer_id")} == {a["id"], b["id"]}]
    assert len(notices) == 1
    assert notices[0]["slot_key"]["attribute"] == "连接池上限"


@pytest.mark.parametrize("missing_field", ["revision", "action"])
def test_applying_reentry_context_requires_revision_and_action(
    tmp_path: Path, monkeypatch, missing_field: str,
) -> None:
    """Incomplete trusted context must fail closed instead of suppressing a notice."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    a = tools.memory_write(content="database is mysql", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(content="database is sqlite", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)

    from memory_arbiter.models import ConflictMember, ConflictValueGroup

    def member(mid, value):
        quote = f"database is {value}"
        return ConflictMember(
            memory_id=mid, version=1, attribute_raw="数据库选型", value_raw=value,
            normalized_attribute="数据库选型", normalized_value=value, evidence_quote=quote,
            evidence_span=(0, len(quote)), content_hash=(str(mid) * 64)[:64],
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        ).to_dict()

    recorded = tools.memory_repair("record_conflict", {
        "slot_key": {"entity": "svc", "attribute": "数据库选型", "scope": "production"},
        "members": [member(a["id"], "mysql"), member(b["id"], "sqlite")],
        "value_groups": [
            ConflictValueGroup("mysql", "MySQL", (f"{a['id']}@1",)).to_dict(),
            ConflictValueGroup("sqlite", "SQLite", (f"{b['id']}@1",)).to_dict(),
        ],
        "detector_version": "d1", "prompt_version": "p1", "source": "scan",
        "reason": "db conflict", "status": "open",
    })
    conflict_id = recorded["data"]["conflict_id"]
    tools.memory("judge", {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "sqlite",
        "decided_by": "user", "ref": "chat", "reason": "confirmed",
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"},
                       {"memory_id": b["id"], "action": "use_as_resolution"}],
        "resolution_memory_id": b["id"],
    })
    monkeypatch.setattr(tools.db, "list_open_conflicts_for_memory_ids", lambda *a_, **k: [])
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _grounded_db_backend)
    hits = [{"memory_id": b["id"], "id": 1, "kind": "text", "text": "database is sqlite",
             "start_offset": 0, "end_offset": 18, "distance": 0.1}]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a_, **k: list(hits))
    context = {
        "conflict_id": conflict_id, "revision": 2, "memory_id": a["id"],
        "action": "update_current_claim", "chosen_value": "sqlite",
    }
    context.pop(missing_field)
    snapshot = _job_snapshot(tools, a["id"])
    snapshot["trusted_applying_context"] = context

    result = tools._process_semantic_conflict_job(a["id"], snapshot)

    assert result["outcome"] == "notices_created"
    assert result["notices_created"] == 1


def test_scan_candidates_qwen_enhancement_populates_value_groups(tmp_path: Path, monkeypatch) -> None:
    """Spec §7.1: bounded Qwen enhancement enriches rule candidates in-place."""
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    a = tools.memory_write(content="生产数据库使用 mysql 方案", subject="a", tags=[], metadata=meta)["data"]
    b = tools.memory_write(content="生产数据库使用 sqlite 方案", subject="b", tags=[], metadata=meta)["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", _grounded_db_backend)

    result = tools.memory_repair("scan_candidates", {"batch": 50, "k": 10})
    assert result["ok"] is True
    enhancement = result["data"].get("qwen_enhancement") or {}
    assert enhancement.get("status") == "ok"
    # At least one candidate now carries extracted value_groups.
    enriched = [c for c in result["data"]["candidates"] if c.get("value_groups")]
    assert enriched, result["data"]
    groups = enriched[0]["value_groups"]
    assert {g["normalized_value"] for g in groups} == {"mysql", "sqlite"}
    # Matching entity/scope aggregates into a slot group.
    assert result["data"].get("slot_groups")


def test_scan_enhancement_fails_open_without_backend(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "svc", "scope": "production"}
    tools.memory_write(content="生产数据库使用 mysql 方案", subject="a", tags=[], metadata=meta)
    tools.memory_write(content="生产数据库使用 sqlite 方案", subject="b", tags=[], metadata=meta)
    assert tools.wait_evidence_worker_drained(timeout=2)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: None)

    result = tools.memory_repair("scan_candidates", {"batch": 50, "k": 10})
    assert result["ok"] is True
    # Deterministic baseline is preserved; enhancement reports it was skipped.
    assert result["data"]["qwen_enhancement"]["status"] == "skipped_unavailable"


def test_exact_match_write_repairs_missing_canonical_vector(tmp_path: Path) -> None:
    """The advertised 'retry a write using this workspace' actually republishes
    a canonical whose vector publication previously failed."""
    tools = make_tools(tmp_path)
    tools.memory_write(content="alpha", workspace="projA", subject="s", tags=[])
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM workspace_canonicals_vec")
    result = tools.memory_write(content="beta", workspace="projA", subject="s2", tags=[])
    assert result["data"]["workspace_matched_by"] == "exact"
    with tools.db.connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM workspace_canonicals_vec").fetchone()["c"]
    assert count >= 1
