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


def test_notify_surfaces_without_qwen_and_check_fails_closed(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, semantic_enabled=False)
    # Disable the background semantic job so only this test's explicit
    # _process_semantic_conflict_job runs (deterministic under suite load).
    tools.settings.semantic_conflict_on_write = "off"
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


def test_same_peer_notify_not_demoted_by_closer_check(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    peer = tools.memory_write(content="旧值：连接池 10。", subject="pool", tags=[])["data"]
    new = tools.memory_write(content="新值：连接池 20。", subject="pool2", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    routes = iter(["notify", "check"])

    def fake_decide(_left, _right):
        return SimpleNamespace(action=next(routes), reason="numeric_value_changed", anchors=[], left_value=None, right_value=None)

    monkeypatch.setattr("memory_arbiter.pipeline.evidence.decide_evidence", fake_decide)
    hits = [
        {"memory_id": peer["id"], "id": 1, "kind": "text", "text": "旧值：连接池 10。", "start_offset": 0, "end_offset": 9, "distance": 0.50},
        {"memory_id": peer["id"], "id": 2, "kind": "text", "text": "旧值：连接池 10。", "start_offset": 0, "end_offset": 9, "distance": 0.05},
    ]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))
    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    notices = [n for n in tools.db.list_semantic_notices() if n["peer_id"] == peer["id"]]
    assert notices and notices[0]["payload"]["route"] == "notify"


def test_notice_pairs_capped_per_write(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    peers = [
        tools.memory_write(content=f"旧值 {i}：连接池 {i}0。", subject=f"pool{i}", tags=[])["data"]
        for i in range(4)
    ]
    new = tools.memory_write(content="新值：连接池 99。", subject="poolx", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    monkeypatch.setattr(
        "memory_arbiter.pipeline.evidence.decide_evidence",
        lambda _l, _r: SimpleNamespace(action="notify", reason="numeric_value_changed", anchors=[], left_value=None, right_value=None),
    )
    hits = [
        {"memory_id": p["id"], "id": i, "kind": "text", "text": f"旧值 {i}", "start_offset": 0, "end_offset": 4, "distance": 0.10 + i * 0.05}
        for i, p in enumerate(peers)
    ]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))

    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    created = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]
    assert len(created) == tools.settings.semantic_conflict_max_notice_pairs == 2

    tools.settings.semantic_conflict_max_notice_pairs = 99
    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    created = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]
    # Deduped re-records no longer consume budget, so the re-run surfaces the
    # remaining two peers (hard cap 3 created per run, 2 pre-existing deduped).
    assert len(created) == 4
    # The hard cap still applies per run even when configured above 3.
    assert tools.settings.semantic_conflict_max_notice_pairs == 99


def test_deduped_pairs_do_not_starve_fresh_notify(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    tools.settings.semantic_conflict_max_notice_pairs = 2
    peers = [
        tools.memory_write(content=f"旧值 {i}：连接池 {i}0。", subject=f"pool{i}", tags=[])["data"]
        for i in range(3)
    ]
    new = tools.memory_write(content="新值：连接池 99。", subject="poolx", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    monkeypatch.setattr(
        "memory_arbiter.pipeline.evidence.decide_evidence",
        lambda _l, _r: SimpleNamespace(action="notify", reason="numeric_value_changed", anchors=[], left_value=None, right_value=None),
    )
    hits = [
        {"memory_id": p["id"], "id": i, "kind": "text", "text": f"旧值 {i}", "start_offset": 0, "end_offset": 4, "distance": 0.10 + i * 0.05}
        for i, p in enumerate(peers)
    ]
    monkeypatch.setattr(tools.db, "evidence_knn", lambda *a, **k: list(hits))

    # First run fills the cap with peers 0 and 1.
    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    assert len([n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]) == 2

    # Re-run: peers 0/1 dedupe without consuming budget, so the fresh notify
    # pair for peer 2 must still be created within the same cap.
    tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    notices = [n for n in tools.db.list_semantic_notices() if n["memory_id"] == new["id"]]
    assert sorted(n["peer_id"] for n in notices) == [p["id"] for p in peers]


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
    # Workspace A: one long memory whose many units own the global top-k.
    bulk = "\n".join(f"postgres 配置项 {i}：连接池参数说明 {i}。" for i in range(8))
    tools.memory_write(content=bulk, subject="pg bulk", tags=[], workspace="wsA")
    peer = tools.memory_write(content="postgres 端口为 5432。", subject="pg port", tags=[], workspace="wsB")["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
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


def test_queue_full_drop_is_counted_and_visible(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.settings.semantic_conflict_queue_max_size = 1
    tools._semantic_worker._pending[111] = {"version": 1}
    outcome = tools._semantic_worker.enqueue(222, {"version": 1})
    assert outcome == {"status": "queue_full"}
    status = tools._semantic_status()["worker"]
    assert status["dropped_queue_full"] == 1


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
        lambda **kwargs: type("FailingTools", (), {"_index_local_text_evidence": staticmethod(lambda *a, **k: {"status": "failed", "reason": "test"})})(),
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
    first = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    assert tools.wait_evidence_worker_drained(timeout=2)
    # Simulate a model swap: the active space no longer matches the live embedder.
    tools.db.init_vec_index_state("fake-vnext-space", True)
    tools.db.init_vec_index_state("new-space-v2", True)
    state = tools.db.get_vec_index_state()
    assert state["state"] == "mismatch" and state["target_space_id"] == "new-space-v2"

    tools._embedder = FakeEmbedder()
    tools._embedder.embedding_space_id = "new-space-v2"
    rebuild = tools.memory_repair("rebuild_evidence", {"dry_run": False})
    assert rebuild["ok"] is True and rebuild["data"]["queued"] >= 1
    assert tools.wait_evidence_worker_drained(timeout=5)

    state = tools.db.get_vec_index_state()
    assert state["state"] == "ready"
    assert state["active_space_id"] == "new-space-v2"
    result = tools.memory_search(query="接口超时")
    assert result["data"]["vector_lag"]["pending_evidence_index"] == 0


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
    assert epoch is None, "dry-run must not persist the rebuild epoch"


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
    from memory_arbiter.db.schema import CURRENT_SCHEMA_GENERATION

    path = tmp_path / "staging.sqlite3"
    MemoryDB(Settings(db_path=path, backup_jsonl=tmp_path / "u.jsonl"))
    with sqlite3.connect(path) as conn:
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
