"""0.15.3 write-time duplicate hint: vector recall over subject_tags_vec.

Covers the recall upgrade decided in mema 825 (problem 1): the hint embeds
"subject + sorted tags", publishes into subject_tags_vec, KNN-recalls the
top-k same-workspace active rows, and falls back to the capped scan when no
embedder/index exists. Also covers the vector lifecycle: publish on
write/activation, refresh on edit, delete on leaving active, and the startup
backfill for pre-existing rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.pipeline.write import WritePipeline
from memory_arbiter.tools import MemoryTools


class CharHistogramEmbedder:
    """Deterministic text-similarity fake: char histogram over 32 buckets.

    Unlike the keyword FakeEmbedder in test_vnext_evidence (which maps every
    non-keyword text to the same vector), a char histogram keeps near-identical
    texts near-identical in vector space so sqlite-vec KNN genuinely ranks the
    near-duplicate subject first.
    """

    embedding_space_id = "char-histogram-space"
    dim = 32
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars=None) -> EmbedResult:
        vector = [0.0] * 32
        text = f"{prefix}\n{body}".casefold()
        for ch in text:
            vector[ord(ch) % 32] += 1.0
        return EmbedResult(vector, False, len(text), len(text))


def make_vec_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "vec-hint.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_model_path=model,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = CharHistogramEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(CharHistogramEmbedder.dim) == []
    tools.db.init_vec_index_state(
        CharHistogramEmbedder.embedding_space_id, True, active_dim=CharHistogramEmbedder.dim,
    )
    return tools


def _vec_ids(tools: MemoryTools) -> set[int]:
    with tools.db.connection() as conn:
        return {int(row["id"]) for row in conn.execute("SELECT id FROM subject_tags_vec")}


def _vec_blob(tools: MemoryTools, memory_id: int) -> str | None:
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT embedding FROM subject_tags_vec WHERE id = ?", (int(memory_id),)
        ).fetchone()
        return str(row["embedding"]) if row is not None else None


def _write(tools: MemoryTools, subject: str, tags: list[str], workspace: str = "w") -> dict:
    result = tools.memory_write(
        content=f"body for {subject}", subject=subject, tags=tags, workspace=workspace,
    )
    assert result["ok"] is True, result
    return result["data"]


def _similar_notices(result: dict) -> list[dict]:
    return [
        notice for notice in (result.get("notices") or [])
        if notice.get("type") == "similar_active_memory"
    ]


def test_vec_recall_publishes_vector_and_fires_hint(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    first = _write(tools, "PostgreSQL 升级步骤", ["db", "ops"])
    second = tools.memory_write(
        content="duplicate body", subject="PostgreSQL 升级步骤", tags=["ops", "db"],
        workspace="w",
    )
    assert second["ok"] is True, second
    hints = _similar_notices(second)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["id"]
    # Sorted tags must not change the vector: both writes published one.
    assert _vec_ids(tools) == {int(first["id"]), int(second["data"]["id"])}


def test_vec_recall_scopes_to_same_workspace(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    other = _write(tools, "PostgreSQL 升级步骤", ["db"], workspace="other-ws")
    mine = _write(tools, "PostgreSQL 升级步骤", ["db"], workspace="w")
    third = tools.memory_write(
        content="third", subject="PostgreSQL 升级步骤", tags=["db"], workspace="w",
    )
    hints = _similar_notices(third)
    matched = {match["memory_id"] for match in hints[0]["matches"]} if hints else set()
    assert matched == {mine["id"]}, f"hint leaked cross-workspace row {other['id']}"


def test_vec_knn_prefers_the_near_duplicate_subject(tmp_path: Path) -> None:
    """With more same-workspace actives than the recall window the near-dup
    still has to come back — the histogram embedder ranks it nearest."""
    tools = make_vec_tools(tmp_path)
    for index in range(8):
        _write(tools, f"完全无关的主题条目编号{index}", ["noise"])
    target = _write(tools, "金营项目发版流程说明", ["release"])
    result = tools.memory_write(
        content="near dup", subject="金营项目发版流程说明", tags=["release"], workspace="w",
    )
    hints = _similar_notices(result)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == target["id"]


def test_pending_write_skips_vector_until_activation(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    first = _write(tools, "activation checklist", ["ops"])
    pending = tools.memory_write(
        content="pending body", subject="activation checklist", tags=["ops"],
        workspace="w", status="pending",
    )
    assert pending["data"]["record"]["status"] == "pending"
    assert int(pending["data"]["id"]) not in _vec_ids(tools)

    activated = tools.memory_activate(int(pending["data"]["id"]), authorized=True)
    assert activated["ok"] is True, activated
    assert int(pending["data"]["id"]) in _vec_ids(tools)
    hints = _similar_notices(activated)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["id"]


def test_edit_refreshes_and_supersede_deletes_vector(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    row = _write(tools, "金营架构决策记录", ["arch"])
    memory_id = int(row["id"])
    before = _vec_blob(tools, memory_id)
    assert before is not None

    edited = tools.memory_edit(memory_id=memory_id, tags_only=True, add_tags=["new-tag"])
    assert edited["ok"] is True, edited
    after = _vec_blob(tools, memory_id)
    assert after is not None and after != before, "tags edit must re-embed the vector"

    superseded = tools.memory_supersede(memory_id=memory_id, reason="gone", authorized=True)
    assert superseded["ok"] is True, superseded
    assert memory_id not in _vec_ids(tools), "leaving active must drop the hint vector"


def test_backfill_restores_missing_actives_only(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    keep = _write(tools, "存量记忆一", ["legacy"])
    drop = _write(tools, "存量记忆二", ["legacy"])
    assert tools.wait_evidence_worker_drained(timeout=5)
    # Simulate a library created before 0.15.3: no hint vectors at all.
    with tools.db.write_transaction() as conn:
        conn.execute("DELETE FROM subject_tags_vec")
    assert _vec_ids(tools) == set()

    assert tools.memory_supersede(memory_id=int(drop["id"]), reason="gone", authorized=True)[
        "ok"
    ]
    missing = tools.db.missing_subject_tags_rows()
    assert [row["id"] for row in missing] == [int(keep["id"])]

    restored = tools._backfill_subject_tags_vectors(tools._embedder)
    assert restored == 1
    assert _vec_ids(tools) == {int(keep["id"])}


def test_fallback_scan_limit_binds_row_count(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fallback.sqlite3",
        backup_jsonl=tmp_path / "fallback.jsonl",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    ids = [
        int(_write(tools, f"scan fallback row {index}", ["cap"])["id"])
        for index in range(6)
    ]
    rows = db.active_subject_tag_rows(ids[0], "w", limit=4)
    assert [row["id"] for row in rows] == sorted(ids[1:5])
    # Unbounded call keeps the legacy behavior for direct consumers.
    assert len(db.active_subject_tag_rows(ids[0], "w")) == 5


def test_subject_tags_embed_text_sorts_tags(tmp_path: Path) -> None:
    left = WritePipeline._subject_tags_embed_text(" 主题 ", ["b", "a", " c "])
    right = WritePipeline._subject_tags_embed_text("主题", ["c", "a", "b"])
    assert left == right == "主题\na b c"
    assert WritePipeline._subject_tags_embed_text(None, None) == ""


def test_vec_knn_filters_inactive_and_excludes_self(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    active = _write(tools, "KNN 过滤验证", ["vec"])
    gone = _write(tools, "KNN 过滤验证", ["vec"])
    assert tools.memory_supersede(memory_id=int(gone["id"]), reason="gone", authorized=True)[
        "ok"
    ]
    query = CharHistogramEmbedder.embed_text(
        "", WritePipeline._subject_tags_embed_text("KNN 过滤验证", ["vec"]),
    ).embedding
    rows = tools.db.subject_tags_knn(
        query,
        k=10,
        exclude_memory_id=int(active["id"]),
        workspace_canonical="w",
    )
    assert [row["id"] for row in rows] == [], "superseded rows must not be recalled"


def test_vec_knn_window_grows_past_foreign_workspaces(tmp_path: Path) -> None:
    """vec0's k-window is global: 30 closer rows from OTHER workspaces must
    not starve the scoped recall (the growth ceiling is the unscoped active
    count, evidence-knn style)."""
    tools = make_vec_tools(tmp_path)
    # Foreign rows share most characters with the query text, so their
    # histograms sit closer than the target's own workspace peers.
    for index in range(30):
        _write(tools, f"金营项目发版流程说明补充材料之{index}", ["release"], workspace="crowd-ws")
    target = _write(tools, "金营项目发版流程说明", ["release"], workspace="w")
    excluded = _write(tools, "完全无关的第二条", ["other"], workspace="w")
    query = CharHistogramEmbedder.embed_text(
        "", WritePipeline._subject_tags_embed_text("金营项目发版流程说明", ["release"]),
    ).embedding
    rows = tools.db.subject_tags_knn(
        query, k=5, exclude_memory_id=int(excluded["id"]), workspace_canonical="w",
    )
    assert [row["id"] for row in rows] == [int(target["id"])]


def test_knn_failure_falls_back_to_scan_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = make_vec_tools(tmp_path)
    first = _write(tools, "回退扫描验证", ["fb"])

    def _raise(**kwargs):
        raise RuntimeError("vec exploded")

    monkeypatch.setattr(tools.db, "subject_tags_knn", _raise)
    result = tools.memory_write(
        content="fallback body", subject="回退扫描验证", tags=["fb"], workspace="w",
    )
    hints = _similar_notices(result)
    assert len(hints) == 1
    assert hints[0]["matches"][0]["memory_id"] == first["id"]
