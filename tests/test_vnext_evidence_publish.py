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

# ── test_vnext_evidence publish/decide/migration-core domain (commit 12 split) ──



def test_local_text_units_are_simple_and_cover_headings() -> None:
    units = local_text_units(
        "Database policy",
        "# Runtime\n生产环境数据库使用 PostgreSQL 16。\n接口超时为 5 秒。",
    )
    assert [unit.kind for unit in units][:2] == ["subject", "heading"]
    assert any(unit.kind == "text" and "PostgreSQL" in unit.text for unit in units)
    assert all(unit.unit_index == index for index, unit in enumerate(units))





def test_decide_evidence_numeric_change_is_scan_candidate_not_direct_notice() -> None:
    assert decide_evidence("数据库使用 pgsql。", "数据库使用 PostgreSQL。").action == "ignore"
    changed = decide_evidence("接口超时为 5 秒。", "接口超时为 30 秒。")
    assert changed.action == "check"
    assert changed.reason == "numeric_value_candidate"
    assert decide_evidence("测试环境数据库使用 MySQL。", "生产环境数据库使用 PostgreSQL。").action == "ignore"
    assert decide_evidence("服务使用数据库。", "服务迁移到新的存储引擎。").action in {"check", "ignore"}


def test_decide_evidence_operator_and_sign_pairs_are_check_not_duplicates() -> None:
    """Round-2 (0.15.8): normalization strips comparators/signs, so
    ">= 100ms" vs "< 100ms" and "+5%" vs "-5%" normalize equal — but they
    contradict, and must surface as check candidates instead of dying as
    ignore/equivalent_value duplicates."""
    comparator = decide_evidence("bench 的超时阈值必须 >= 100ms。", "bench 的超时阈值必须 < 100ms。")
    assert comparator.action == "check"
    assert comparator.reason != "equivalent_value"
    sign = decide_evidence("误差允许 +5%。", "误差允许 -5%。")
    assert sign.action == "check"
    assert sign.reason != "equivalent_value"


def test_decide_evidence_true_duplicate_still_ignored() -> None:
    """The operator guard must not blanket-disable duplicate detection: a
    verbatim repeat (identical operator signature) stays ignore."""
    duplicate = decide_evidence("超时阈值必须 100ms。", "超时阈值必须 100ms。")
    assert duplicate.action == "ignore"
    assert duplicate.reason == "equivalent_value"


def test_decide_evidence_date_hyphens_are_not_signs() -> None:
    """Date hyphens sit between two digits and are NOT operator signs; a
    same-date pair differing only in the deliverable (json vs csv) is a real
    conflict candidate, not a duplicate — the value-stripped skeletons
    differ, so the duplicate guard must not fire."""
    decision = decide_evidence("2026-09-08 交付 json 版本", "2026-09-08 交付 csv 版本")
    assert decision.action == "check"
    assert decision.reason != "equivalent_value"






_GUILOTINE_QUOTE_A = (
    "永不截断：合并覆盖≥50%全文献条目升级全文+hit_spans 转标注"
    "（owner 拍板：服务端不替 Agent 挑重要命中，法规 RAG 漏但书是系统性偏差）"
)

def test_bounded_short_value_rejects_guillotine_fragments() -> None:
    from memory_arbiter.semantic_conflict import _bounded_short_value

    # Exact head of the live sample, cut mid-clause by the decoding cap (ends
    # "不替 Agent 挑重要命中，" — the observed guillotine shape).
    beheaded = (
        "合并覆盖≥50%全文献条目升级全文+hit_spans 转标注"
        "（owner 拍板：服务端不替 Agent 挑重要命中，"
    )
    assert beheaded in _GUILOTINE_QUOTE_A
    assert len(beheaded) <= 64
    assert not _bounded_short_value(beheaded, _GUILOTINE_QUOTE_A)
    # Mid-word cut (every occurrence followed by a word character).
    assert not _bounded_short_value("合并覆盖≥50%全文献条目升级全文+hit_spans 转标注（owner 拍板：服务端不替 Agent 挑重要命", _GUILOTINE_QUOTE_A)


def test_bounded_short_value_accepts_clean_fragments() -> None:
    from memory_arbiter.semantic_conflict import _bounded_short_value

    # Legit short values still pass: bounded by punctuation/space/quote-end.
    assert _bounded_short_value("全部透出", "owner 拍板：命中全部透出。")
    assert _bounded_short_value("MySQL", "生产数据库使用 MySQL。")
    assert _bounded_short_value("5 秒", "生产环境的接口超时策略明确设置为 5 秒。")
    assert _bounded_short_value("高 ROI 候选分析", "Tier1 仍为高 ROI 候选分析（809：三项待排期）。")
    # A fragment ending before a comma stays fine (clause boundary, not a cut).
    assert _bounded_short_value("只透出 kind=subject 的命中", "只透出 kind=subject 的命中，其余折叠。")
    # Non-verbatim-but-grounded values (spacing/units) are unaffected.
    assert _bounded_short_value("8GB", "内存配额为 8 GB。")




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
    assert tools.wait_semantic_worker_drained(timeout=2)
    coverage = tools.db.evidence.coverage()
    assert coverage["indexed_memories"] == 1
    # C2/C3: rows are the store (subject row + sentences); units retired.
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT kind, content_hash FROM memory_row WHERE memory_id=? ORDER BY row_index",
            (memory_id,),
        ).fetchall()
    kinds = [r["kind"] for r in rows]
    assert kinds[0] == "subject" and len(rows) >= 2  # A+ subject row leads
    assert all(
        r["content_hash"] == evidence_content_hash(result["data"]["record"]["content"])
        for r in rows
    )


def test_vnext_search_uses_row_knn_not_legacy_vectors(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    target = tools.memory_write(
        content="服务数据库选择 PostgreSQL 16。",
        subject="runtime database",
        tags=["db"],
    )["data"]["id"]
    tools.memory_write(content="用户今天想喝咖啡。", subject="lunch", tags=["food"])
    assert tools.wait_semantic_worker_drained(timeout=2)
    result = tools.memory_search(query="pgsql database", limit=5, content_mode="full")
    ids = [row["id"] for row in result["data"]["results"]]
    assert target in ids
    hit = next(row for row in result["data"]["results"] if row["id"] == target)
    assert hit.get("_evidence_hits") or hit.get("content")


def test_evidence_candidate_enters_when_lexical_pool_is_full(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    # RECALL_POOL_CAP is a frozen constant since 0.15.0: fill the lexical pool
    # exactly to the cap with content-only "needle" matches, so the semantic
    # target's only path into the pool is the independent evidence channel
    # (which must run even when FTS/LIKE already filled the pool).
    from memory_arbiter.constants import RECALL_POOL_CAP

    lexical_ids = [
        tools.memory_write(
            content=f"needle lexical distractor {index}",
            subject=f"distractor {index}",
        )["data"]["id"]
        for index in range(RECALL_POOL_CAP)
    ]
    target_id = tools.memory_write(
        content="semantically relevant target without the query term",
        subject="semantic target",
    )["data"]["id"]
    target = tools.db.get_memory(target_id)
    assert target is not None
    tools.db.state.sqlite_vec_available = True

    def row_knn(*_args, **_kwargs):
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

    # 0.15.9: this pins pool ADMISSION, not page relevance; disable the floor.
    import memory_arbiter.search as _search_mod
    monkeypatch.setattr(_search_mod, "QUERY_RECALL_SCORE_FLOOR", -1.0)
    monkeypatch.setattr(tools.db, "row_knn", row_knn)
    _pass_cos_gate(monkeypatch)
    result = tools.memory_search(
        query="needle", limit=4, query_embedding=[1.0, 0.0],
        include_linked_open_items=False, include_conflict_signal=False,
    )
    ids = [row["id"] for row in result["data"]["results"]]
    assert target_id in ids
    assert len(set(ids) & set(lexical_ids)) >= 1


def test_exact_subject_match_survives_evidence_fusion(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
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

    monkeypatch.setattr(tools.db, "row_knn", lambda *_args, **_kwargs: [{
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
    """Non-direct-eligible numeric candidates (multi-value, ambiguous
    pairwise) still fail closed without Qwen and stay scan review
    candidates. Direct-eligible pairs landing without a backend are covered
    by test_direct_path_works_without_qwen."""
    tools = make_tools(tmp_path, semantic_enabled=False)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "checkout-api", "scope": "production"}
    old = tools.memory_write(content="接口超时为 5 秒，队列长度为 3。", subject="timeout", tags=["api"], metadata=meta)["data"]
    new = tools.memory_write(content="接口超时为 30 秒，队列长度为 5。", subject="timeout", tags=["api"], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: None)
    result = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    assert result["status"] == "incomplete"
    assert result["reason"] == "judge_unavailable"
    assert result["notices_created"] == 0
    assert tools.db.list_semantic_notices(status="open") == []
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10, "include_quotes": True})
    pair = (min(old["id"], new["id"]), max(old["id"], new["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    assert clue["route"] == "review_candidate"
    assert "numeric_value_candidate" in clue["reasons"]


def test_direct_path_works_without_qwen(tmp_path: Path, monkeypatch) -> None:
    """The deterministic direct path needs no backend: same key + canonical
    value difference lands the notice while Qwen is unavailable (owner
    2026-09-16 — the pair carries its own mechanical proof)."""
    tools = make_tools(tmp_path, semantic_enabled=False)
    tools.settings.semantic_conflict_on_write = "off"
    meta = {"entity": "checkout-api", "scope": "production"}
    tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=["api"], metadata=meta)
    new = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=["api"], metadata=meta)["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: None)
    result = tools._process_semantic_conflict_job(new["id"], _job_snapshot(tools, new["id"]))
    assert result["status"] == "completed"
    assert result["outcome"] == "notices_created"
    notices = tools.db.list_semantic_notices(status="open")
    assert len(notices) == 1
    payload = notices[0]["payload"]
    if isinstance(payload, str):
        import json as _json
        payload = _json.loads(payload)
    assert payload["reason"] == "deterministic_same_key_value_diff"


def test_vnext_semantic_job_is_chained_after_evidence_publish(tmp_path: Path, monkeypatch) -> None:
    """C2 单队列不变式（原「链式转发」语义反转重写）：写入只排一个语义
    job，索引（分段→批嵌→publish_rows）与检测在同一 job 内按序完成——
    publish 先于检测、行在库、evidence worker 无任务。"""
    tools = make_tools(tmp_path, semantic_enabled=False)
    result = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])
    memory_id = result["data"]["id"]
    task_id = result["data"]["evidence_index"]["semantic_task_id"]
    # The write's sync gate waits for this exact task; a fast local index +
    # detect pass completes inside the same request.
    check_receipt = result["data"]["semantic_conflict_check"]
    check_receipt.pop("elapsed_ms", None)
    for _row_key in ("rows_mode", "rows_examined", "claims_channel", "claims_channel_c"):
        check_receipt.pop(_row_key, None)
    assert check_receipt == {
        "status": "completed",
        "outcome": "checked_no_notice",
        "notices_created": 0,
        "task_id": task_id,
        "dedupe_key": task_id,
    }
    assert tools.wait_semantic_worker_drained(timeout=2)
    # 单队列不变式：行已发布（含 A+ subject 行），检测在同一 job 完成，
    # 旧索引工人不再收到任何任务。
    with tools.db.connection() as conn:
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM memory_row WHERE memory_id=? ORDER BY row_index", (memory_id,)
        )]
    assert kinds and kinds[0] == "subject"
    assert tools.db.evidence.coverage()["indexed_memories"] == 1
    assert tools._semantic_worker.status()["queue_depth"] == 0


def test_semantic_job_reuses_just_published_evidence_vectors(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path, semantic_enabled=False)
    tools.settings.semantic_conflict_on_write = "off"
    CountingEmbedder.calls = 0
    tools._embedder = CountingEmbedder()
    written = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    indexed_calls = CountingEmbedder.calls
    assert indexed_calls > 0

    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: None)
    tools._process_semantic_conflict_job(written["id"], _job_snapshot(tools, written["id"]))

    assert CountingEmbedder.calls == indexed_calls


def test_evidence_publish_rejects_stale_snapshot(tmp_path: Path) -> None:
    from memory_arbiter.rowseg import segment_rows

    tools = make_tools(tmp_path)
    written = tools.memory_write(content="旧内容。", subject="snapshot", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    memory_id = written["id"]
    current = tools.db.get_memory(memory_id)
    stale_rows = segment_rows(current["subject"], current["content"])
    edited = tools.memory_edit(memory_id=memory_id, new_content="新内容。")
    assert edited["ok"] is True
    # C5: publish_rows is the publisher; a stale (id, version) snapshot
    # must still reject instead of clobbering the newer version's rows.
    result = tools.db.evidence.publish_rows(
        memory_id,
        int(current["version"]),
        evidence_content_hash(current["content"]),
        stale_rows,
        [[0.0, 1.0] for _ in stale_rows],
    )
    assert result["outcome"] == "stale_snapshot"


def test_vnext_status_change_updates_evidence_parent_status(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(content="待废弃事实。", subject="status", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    assert tools.db.update_memory(written["id"], {"status": "superseded"}) is True
    # C5: the flip lives on the row store (unit tables retired).
    with tools.db.connection() as conn:
        statuses = {
            row["parent_status"] for row in conn.execute(
                "SELECT v.parent_status FROM memory_row_vec v "
                "JOIN memory_row r ON r.id=v.id WHERE r.memory_id=?",
                (written["id"],),
            )
        }
        versions = {
            (int(row["memory_version"]), int(row["version"]))
            for row in conn.execute(
                "SELECT r.memory_version,m.version FROM memory_row r "
                "JOIN memories m ON m.id=r.memory_id WHERE r.memory_id=?",
                (written["id"],),
            )
        }
    assert statuses == {"superseded"}
    assert versions == {(2, 2)}


def test_vnext_weak_isolation_does_not_hard_filter_semantic_candidates(tmp_path: Path, monkeypatch) -> None:
    tools = make_tools(tmp_path)
    tools.settings.isolation = "weak"
    written = tools.memory_write(
        content="接口超时为 5 秒。", subject="timeout", tags=[], workspace="alpha",
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=2)
    seen = []

    def row_knn(*args, **kwargs):
        seen.append(kwargs.get("workspace"))
        return []

    monkeypatch.setattr(tools.db, "row_knn", row_knn)
    # 0.17.0 P2-3：行级模式候选走 row_knn，同样不得被 workspace 硬过滤
    monkeypatch.setattr(tools.db, "row_knn", row_knn)
    _pass_cos_gate(monkeypatch)
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
        embedding_model_path=model,
    )

    original_init = MemoryTools.__init__

    def patched_init(self, settings=None, db=None):
        original_init(self, settings=settings, db=db)
        self._embedder = FakeEmbedder()
        self._embedder_loaded = True
        self.db.ensure_vec_tables(FakeEmbedder.dim)
        self.db.init_vec_index_state("fake-vnext-space", True, active_dim=FakeEmbedder.dim)

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
        embedding_model_path=model,
    )
    target_tools = MemoryTools(target_settings, MemoryDB(target_settings))
    target_tools._embedder = FakeEmbedder()
    target_tools._embedder_loaded = True
    target_tools.db.ensure_vec_tables(FakeEmbedder.dim)
    target_tools.db.init_vec_index_state("fake-vnext-space", True, active_dim=FakeEmbedder.dim)
    searched = target_tools.memory_search(query="pgsql database", limit=5)
    assert second["data"]["id"] in [row["id"] for row in searched["data"]["results"]]


def test_previous_generation_old_space_is_preserved_and_disabled(
    tmp_path: Path, monkeypatch,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=source_path,
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
    )
    source_db = MemoryDB(settings)
    source_tools = MemoryTools(settings, source_db)
    old_embedder = FakeEmbedder()
    old_embedder.embedding_space_id = "old-pipeline-space"
    source_tools._embedder = old_embedder
    source_tools._embedder_loaded = True
    source_db.ensure_vec_tables(FakeEmbedder.dim)
    source_db.init_vec_index_state(
        old_embedder.embedding_space_id, True, active_dim=FakeEmbedder.dim,
    )
    written = source_tools.memory_write(
        content="旧空间中的内容。", subject="旧空间", tags=[],
    )
    assert written["ok"] is True
    assert source_tools.wait_semantic_worker_drained(timeout=5)
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
        self.db.ensure_vec_tables(FakeEmbedder.dim)
        self.db.init_vec_index_state(
            self._embedder.embedding_space_id, True, active_dim=FakeEmbedder.dim,
        )

    monkeypatch.setattr(MemoryTools, "__init__", init_with_current_embedder)
    target = tmp_path / "current.sqlite3"

    plan = inspect(source_path, target, settings)
    result = build(source_path, target, settings, progress=False)

    assert plan["upgrade_mode"] == "conflict_only"
    assert plan["schema_migration"]["vector_effect"] == "preserve"
    assert result["ok"] is True, result
    assert result["upgrade_mode"] == "conflict_only"
    assert result["indexed"] == 0
    with sqlite3.connect(target) as conn:
        state = dict(conn.execute("SELECT key,value FROM _vec_index_meta"))
    assert state["state"] == "mismatch"
    assert state["active_space_id"] == "old-pipeline-space"
    assert state["target_space_id"] == plan["configured_space_id"]


def test_preserve_migration_keeps_derived_rows_for_later_health_repair(
    tmp_path: Path, monkeypatch,
) -> None:
    pytest.importorskip("sqlite_vec")
    source_path = tmp_path / "previous.sqlite3"
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=source_path, backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
    )
    source_db = MemoryDB(settings)
    source_tools = MemoryTools(settings, source_db)
    old_embedder = FakeEmbedder()
    old_embedder.embedding_space_id = "old-pipeline-space"
    source_tools._embedder = old_embedder
    source_tools._embedder_loaded = True
    source_db.ensure_vec_tables(FakeEmbedder.dim)
    source_db.init_vec_index_state(
        old_embedder.embedding_space_id, True, active_dim=FakeEmbedder.dim,
    )
    active = source_tools.memory_write(
        content="active", subject="active", workspace="project-alpha",
    )["data"]["id"]
    deleted = source_tools.memory_write(
        content="deleted", subject="deleted", workspace="project-beta",
    )["data"]["id"]
    assert source_tools.wait_semantic_worker_drained(timeout=5)
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
        self.db.ensure_vec_tables(FakeEmbedder.dim)
        self.db.init_vec_index_state(
            self._embedder.embedding_space_id, True, active_dim=FakeEmbedder.dim,
        )

    monkeypatch.setattr(MemoryTools, "__init__", init_with_current_embedder)
    target = tmp_path / "current.sqlite3"

    result = build(source_path, target, settings, progress=False)

    assert result["ok"] is True, result
    # Keep the embedding model configured so the extension actually loads —
    # without a model path the runtime no longer loads sqlite-vec and the
    # preserved vec0 tables below would be unreadable ("no such module: vec0").
    target_db = MemoryDB(Settings(
        db_path=target, backup_jsonl=tmp_path / "target-backup.jsonl",
        embedding_model_path=model,
    ))
    with target_db.connection() as conn:
        evidence_memory_ids = {
            int(row[0]) for row in conn.execute("SELECT DISTINCT memory_id FROM memory_row")
        }
        units = int(conn.execute("SELECT COUNT(*) FROM memory_row").fetchone()[0])
        vectors = int(conn.execute("SELECT COUNT(*) FROM memory_row_vec").fetchone()[0])
        workspace_vectors = int(
            conn.execute("SELECT COUNT(*) FROM workspace_canonicals_vec").fetchone()[0]
        )
    assert evidence_memory_ids == {active, deleted}
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
        embedding_model_path=model,
    )
    original_init = MemoryTools.__init__

    def patched_init(self, settings=None, db=None):
        original_init(self, settings=settings, db=db)
        self._embedder = FakeEmbedder()
        self._embedder_loaded = True
        self.db.ensure_vec_tables(FakeEmbedder.dim)
        self.db.init_vec_index_state("fake-vnext-space", True, active_dim=FakeEmbedder.dim)

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
