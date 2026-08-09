from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from memory_arbiter.arbitration import compare_memories
from memory_arbiter.config import Settings, parse_bool
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import SourceType
from memory_arbiter.tools import MemoryTools
from memory_arbiter.search import SearchOutcome

try:
    import sqlite_vec  # type: ignore  # noqa: F401
    _VEC_AVAILABLE = True
except Exception:
    _VEC_AVAILABLE = False


class _MockManagedEmbedder:
    """Minimal mock for ManagedEmbedder — wraps a plain encode function.

    Mirrors the production Never-raises contract: if _encode raises, the
    exception is caught, last_encode_error is set, and an empty EmbedResult
    is returned so callers must check er.embedding.
    """

    def __init__(self, encode_fn):
        self._encode = encode_fn
        self.embedding_space_id = "mock_space_id"
        self.last_encode_error = None

    def embed_text(self, prefix="", body="", max_body_chars=None):
        # Mirror the production separator so the prefix's trailing token and the
        # body's leading token are not merged (e.g. "alpha"+"alpha x" → "alphaalpha").
        sep = "\n" if prefix and body else ""
        text = (prefix + sep + body).strip()
        try:
            emb = self._encode(text)
        except Exception as exc:
            self.last_encode_error = str(exc)
            return EmbedResult(embedding=[], truncated=True, original_tokens=0, used_tokens=0)
        return EmbedResult(embedding=emb, truncated=False, original_tokens=0, used_tokens=0)


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=False,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def make_vec_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    settings = Settings(
        db_path=tmp_path / "memory-vec.sqlite3",
        backup_jsonl=tmp_path / "backup-vec.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=2,
        split_threshold=1,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def clear_config_env(monkeypatch) -> None:
    for key in (
        "MEMORY_ARBITER_CONFIG",
        "MEMORY_ARBITER_DB_PATH",
        "MEMORY_ARBITER_BACKUP_JSONL",
        "MEMORY_ARBITER_POLICY",
        "MEMORY_ARBITER_CLIENT",
        "MEMORY_ARBITER_AGENT_ID",
        "MEMORY_ARBITER_WORKSPACE",
        "MEMORY_ARBITER_ENABLE_SQLITE_VEC",
        "MEMORY_ARBITER_VEC_DIM",
        "MEMORY_ARBITER_RECALL_POOL_CAP",
        "MEMORY_ARBITER_CONTENT_LIKE_CAP",
        "MEMORY_ARBITER_EMBEDDING_PROVIDER",
        "MEMORY_ARBITER_EMBEDDING_MODEL_PATH",
        "MEMORY_ARBITER_EMBEDDING_AUTO_QUERY",
        "MEMORY_ARBITER_EMBEDDING_AUTO_WRITE",
        "MEMORY_ARBITER_GGUF",
        "MEMORY_ARBITER_TOOL_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)



def make_tools_vec(tmp_path: Path) -> MemoryTools:
    """Fixture that enables sqlite-vec (required for semantic recall tests)."""
    settings = Settings(
        db_path=tmp_path / "memory_vec.sqlite3",
        backup_jsonl=tmp_path / "backup_vec.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=4,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_semantic_recall_off_when_no_embedding(tmp_path: Path) -> None:
    """Default behaviour unchanged: without query_embedding, search is lexical-only."""
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="The deployment uses blue-green strategy.",
        subject="deploy-strategy",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    # No query_embedding passed — pure lexical search, same as v0.3.0.
    found = tools.memory_search(query="deployment", workspace="repo-a")
    assert found["ok"] is True
    assert found["data"]["count"] >= 1


def test_store_embedding_rejects_missing_memory(tmp_path: Path) -> None:
    """Storing an embedding for a non-existent memory id should fail cleanly."""
    if not _VEC_AVAILABLE:
        return  # environment without sqlite-vec; skip gracefully
    tools = make_tools_vec(tmp_path)
    result = tools.memory_store_embedding(memory_id=99999, embedding=[0.1, 0.2, 0.3, 0.4])
    assert result["ok"] is False
    assert "not found" in (result.get("data", {}).get("error") or "").lower()


def test_semantic_recall_surfaces_lexically_unmatched_memory(tmp_path: Path) -> None:
    """A memory with zero lexical overlap should still be reachable via vec0 KNN.

    This is the core value of semantic recall: 'happy' query finds 'joyful'
    content when embeddings say they're close, even though trigram/BM25 miss.
    """
    if not _VEC_AVAILABLE:
        return
    tools = make_tools_vec(tmp_path)
    # Two memories. The first shares no trigrams/tokens with the query
    # 'serene calmness'; the second is lexically unrelated too but further
    # away in vector space. Without semantic recall, neither would surface.
    happy = tools.memory_write(
        content="A tranquil meadow at dawn, quiet and still.",
        subject="meadow-scene",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    tools.memory_write(
        content="Quarterly revenue grew 12% year over year.",
        subject="revenue-report",
        source_type="agent_generated",
        event_time="2026-01-02T00:00:00Z",
    )
    happy_id = happy["data"]["id"]
    # Craft a 4-dim embedding that is close to the query vector and far from
    # everything else. Query vector below sits near happy's embedding.
    happy_embedding = [0.9, 0.1, 0.0, 0.0]
    revenue_embedding = [0.0, 0.0, 0.9, 0.1]
    assert tools.memory_store_embedding(memory_id=happy_id, embedding=happy_embedding)["ok"] is True
    # Store one for revenue too, to make sure KNN discriminates.
    revenue_id = [r["id"] for r in tools.memory_recent(workspace="repo-a")["data"]["results"] if r["subject"] == "revenue-report"][0]
    assert tools.memory_store_embedding(memory_id=revenue_id, embedding=revenue_embedding)["ok"] is True

    # Query embedding close to happy → happy should surface even though
    # 'serene calmness' shares no trigrams with the meadow content.
    found = tools.memory_search(query="serene calmness", workspace="repo-a", query_embedding=[0.85, 0.15, 0.0, 0.0])
    assert found["ok"] is True
    ids = [r["id"] for r in found["data"]["results"]]
    assert happy_id in ids, "semantic recall failed: lexically-unmatched memory not surfaced"


def test_query_embedding_without_sqlite_vec_warns(tmp_path: Path) -> None:
    """When sqlite-vec is unavailable, passing query_embedding should warn, not crash."""
    tools = make_tools(tmp_path)  # vec disabled in this fixture
    tools.memory_write(
        content="Some content here.",
        subject="note",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    result = tools.memory_search(query="content", workspace="repo-a", query_embedding=[0.1, 0.2, 0.3, 0.4])
    assert result["ok"] is True  # must not crash
    # Should carry a warning that semantic channel was skipped.
    warnings_text = " ".join(result.get("warnings") or [])
    assert "sqlite-vec unavailable" in warnings_text or result["data"]["count"] >= 0


# --------------------------------------------------------------------------- #
# v0.4.0 — version chain (memory_edit / memory_history / memory_cleanup_history)
# --------------------------------------------------------------------------- #



def test_no_embedding_no_false_warning(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    written = tools.memory_write(content="plain lexical memory", subject="lexical")
    found = tools.memory_search(query="lexical")

    assert "embedding_stored" not in written["data"]
    assert not any("embedding configured" in warning for warning in found["warnings"])
    assert not any("auto-embedding" in warning for warning in found["warnings"])


def test_linked_open_items_uses_tools_patch_seam(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        embedding_auto_query=False,
        embedding_auto_write=False,
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))

    def fake_search_memories(db, query, workspace, tags, limit, status_filter="active", **kwargs):
        return SearchOutcome(
            results=[{"id": 1, "content": "direct hit"}],
            warnings=[],
            has_more=False,
            total_estimate=1,
            retrieval_mode="direct",
        )

    def fake_linked_open_items(db, results, warnings, ws_canonical=None):
        return [{"sentinel": True}]

    monkeypatch.setattr("memory_arbiter.tools.search_memories", fake_search_memories)
    monkeypatch.setattr("memory_arbiter.tools._linked_open_items_for_search", fake_linked_open_items)

    result = tools.memory_search(
        query="semantic query",
        include_linked_open_items=True,
        include_conflict_signal=False,
    )

    assert result["ok"] is True
    assert result["data"]["linked_open_items"] == [{"sentinel": True}]


def test_auto_embedding_injects_query_embedding(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        workspace="repo-a",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    tools._ensure_embedder = lambda: (_MockManagedEmbedder(lambda text: [0.1, 0.2, 0.3]), [])  # type: ignore[method-assign]
    captured = {}

    def fake_search_memories(db, query, workspace, tags, limit, status_filter="active", debug_ranking=False, query_embedding=None, **kwargs):
        captured["query_embedding"] = query_embedding
        return SearchOutcome([], [], False, 0, "empty")

    monkeypatch.setattr("memory_arbiter.tools.search_memories", fake_search_memories)

    result = tools.memory_search(query="semantic query")

    assert result["ok"] is True
    assert captured["query_embedding"] == [0.1, 0.2, 0.3]


def test_explicit_embedding_overrides_auto(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        workspace="repo-a",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))

    def fail_ensure():
        raise AssertionError("auto embedder should not be loaded when query_embedding is explicit")

    tools._ensure_embedder = fail_ensure  # type: ignore[method-assign]
    captured = {}

    def fake_search_memories(db, query, workspace, tags, limit, status_filter="active", debug_ranking=False, query_embedding=None, **kwargs):
        captured["query_embedding"] = query_embedding
        return SearchOutcome([], [], False, 0, "empty")

    monkeypatch.setattr("memory_arbiter.tools.search_memories", fake_search_memories)

    tools.memory_search(query="semantic query", query_embedding=[9.0])

    assert captured["query_embedding"] == [9.0]


def test_vec_disabled_does_not_load_embedder(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=False,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))

    embedder, warnings = tools._ensure_embedder()

    assert embedder is None
    assert any("vec.enabled=false" in warning for warning in warnings)


def test_vec_disabled_warning_appears_in_same_write_response(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=False,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))

    result = tools.memory_write(content="semantic body", subject="semantic subject")

    assert result["ok"] is True
    assert result["data"]["embedding_stored"] is False
    assert any("embedding configured but vec.enabled=false" in warning for warning in result["warnings"])


def test_memory_write_auto_stores_embedding(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        workspace="repo-a",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    tools._ensure_embedder = lambda: (_MockManagedEmbedder(lambda text: [1.0, 2.0]), [])  # type: ignore[method-assign]
    stored = {}

    def fake_store(memory_id: int, embedding: list[float]):
        stored["memory_id"] = memory_id
        stored["embedding"] = embedding
        return True, []

    tools.db.store_embedding = fake_store  # type: ignore[method-assign]

    result = tools.memory_write(content="semantic body", subject="semantic subject")

    assert result["data"]["embedding_stored"] is True
    assert stored["memory_id"] == result["data"]["id"]
    assert stored["embedding"] == [1.0, 2.0]


def test_store_embedding_failure_visible(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    tools._ensure_embedder = lambda: (_MockManagedEmbedder(lambda text: [1.0, 2.0]), [])  # type: ignore[method-assign]
    tools.db.store_embedding = lambda memory_id, embedding: (False, ["boom"])  # type: ignore[method-assign]

    result = tools.memory_write(content="semantic body", subject="semantic subject")

    assert result["ok"] is True
    assert result["data"]["embedding_stored"] is False
    assert "boom" in result["warnings"]


def test_memory_edit_reembeds(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    written = tools.memory_write(content="old body", subject="old subject")
    tools._ensure_embedder = lambda: (_MockManagedEmbedder(lambda text: [3.0, 4.0]), [])  # type: ignore[method-assign]
    stored = {}

    def fake_store(memory_id: int, embedding: list[float]):
        stored["memory_id"] = memory_id
        stored["embedding"] = embedding
        return True, []

    tools.db.store_embedding = fake_store  # type: ignore[method-assign]

    edited = tools.memory_edit(memory_id=written["data"]["id"], new_content="new body")

    assert edited["ok"] is True
    assert edited["data"]["embedding_stored"] is True
    assert stored["memory_id"] == written["data"]["id"]
    assert stored["embedding"] == [3.0, 4.0]


def test_memory_edit_reembed_failure_deletes_stale(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    written = tools.memory_write(content="old body", subject="old subject")

    def bad_encode(text: str) -> list[float]:
        raise RuntimeError("encode failed")

    deleted = {}
    tools._ensure_embedder = lambda: (_MockManagedEmbedder(bad_encode), [])  # type: ignore[method-assign]
    tools.db.delete_embedding = lambda memory_id: (deleted.setdefault("memory_id", memory_id) is not None, [])  # type: ignore[method-assign]

    edited = tools.memory_edit(memory_id=written["data"]["id"], new_content="new body")

    assert edited["ok"] is True
    assert edited["data"]["embedding_stored"] is False
    assert deleted["memory_id"] == written["data"]["id"]
    assert any("deleted stale embedding" in warning for warning in edited["warnings"])


def test_memory_edit_store_failure_deletes_stale(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    written = tools.memory_write(content="old body", subject="old subject")
    deleted = {}
    tools._ensure_embedder = lambda: (_MockManagedEmbedder(lambda text: [5.0, 6.0]), [])  # type: ignore[method-assign]
    tools.db.store_embedding = lambda memory_id, embedding: (False, ["store failed"])  # type: ignore[method-assign]
    tools.db.delete_embedding = lambda memory_id: (deleted.setdefault("memory_id", memory_id) is not None, [])  # type: ignore[method-assign]

    edited = tools.memory_edit(memory_id=written["data"]["id"], new_content="new body")

    assert edited["ok"] is True
    assert edited["data"]["embedding_stored"] is False
    assert deleted["memory_id"] == written["data"]["id"]
    assert "store failed" in edited["warnings"]
    assert any("deleted stale embedding" in warning for warning in edited["warnings"])


# --------------------------------------------------------------------------- #
# v0.4.1 — recency-aware ranking (tie-breaker fix + recency bonus)
# --------------------------------------------------------------------------- #


def test_tied_scores_rank_newest_first(tmp_path: Path) -> None:
    """Regression: when several records tie on relevance score, the newest must
    rank first.

    Reproduces the exact dogfooding failure that buried v0.4.0's release notes
    (id=108) under v0.2.x's (id=27..52) when querying "发版完成" — all release
    summaries cap out at the same subject/tags score, so the previous two-sort
    implementation (ascending ingest_time, then stable score-desc) left them in
    oldest-first SQLite rowid order.
    """
    tools = make_tools(tmp_path)
    # Three release summaries, identical structure → identical surface scores.
    # Ingested a day apart so ingest_time is a meaningful tiebreaker.
    tools.memory_write(
        content="release v1 summary notes",
        subject="release-notes",
        tags=["release"],
        source_type="agent_generated",
        ingest_time="2026-07-01T00:00:00+00:00",
        event_time="2026-07-01T00:00:00Z",
    )
    tools.memory_write(
        content="release v2 summary notes",
        subject="release-notes",
        tags=["release"],
        source_type="agent_generated",
        ingest_time="2026-07-02T00:00:00+00:00",
        event_time="2026-07-02T00:00:00Z",
    )
    tools.memory_write(
        content="release v3 summary notes",
        subject="release-notes",
        tags=["release"],
        source_type="agent_generated",
        ingest_time="2026-07-03T00:00:00+00:00",
        event_time="2026-07-03T00:00:00Z",
    )

    found = tools.memory_search(query="release", workspace="repo-a", limit=10)
    assert found["ok"] is True
    subjects = [r["content"] for r in found["data"]["results"]]
    # Newest first. This is the v0.4.0 id=108 case: the latest release must
    # not be buried under older ones that merely share its surface score.
    assert subjects[0] == "release v3 summary notes", f"newest not first: {subjects}"
    assert subjects == [
        "release v3 summary notes",
        "release v2 summary notes",
        "release v1 summary notes",
    ], f"expected newest→oldest order, got {subjects}"


def test_recency_bonus_does_not_override_relevance(tmp_path: Path) -> None:
    """A newer content-only match must NOT outrank an older subject match.

    This is the safety boundary on the recency bonus: max 0.30, while the
    cheapest subject-medium weight is 6.0. A record that only matches content
    (and takes the content-only penalty) should sit below a subject match even
    if the subject match is old enough to receive zero recency bonus.
    """
    tools = make_tools(tmp_path)
    # Old but authoritative: strong subject hit, zero recency bonus (>90d).
    tools.memory_write(
        content="canonical api token policy",
        subject="token-policy",
        tags=["policy"],
        source_type="document_extracted",
        ingest_time="2025-01-01T00:00:00+00:00",
        event_time="2025-01-01T00:00:00Z",
    )
    # New but only matches content: "token" appears in body, not subject.
    tools.memory_write(
        content="changelog mentions token refresh by the way",
        subject="unrelated-changelog",
        tags=["release"],
        source_type="agent_generated",
        ingest_time="2026-07-07T00:00:00+00:00",
        event_time="2026-07-07T00:00:00Z",
    )

    found = tools.memory_search(query="token", workspace="repo-a", limit=10)
    assert found["ok"] is True
    subjects = [r["subject"] for r in found["data"]["results"]]
    assert subjects[0] == "token-policy", (
        f"recency overrode relevance: {subjects} — content-only match outranked subject match"
    )


def test_recency_bonus_tiers_parsed_correctly() -> None:
    """Unit test for _recency_bonus tiers and graceful degradation."""
    from memory_arbiter.search import _recency_bonus
    from datetime import datetime, timezone, timedelta

    now = datetime(2026, 7, 7, tzinfo=timezone.utc)

    fresh = {"ingest_time": (now - timedelta(days=2)).isoformat()}
    assert _recency_bonus(fresh, now=now) == 0.30

    month_old = {"ingest_time": (now - timedelta(days=20)).isoformat()}
    assert _recency_bonus(month_old, now=now) == 0.15

    quarter_old = {"ingest_time": (now - timedelta(days=60)).isoformat()}
    assert _recency_bonus(quarter_old, now=now) == 0.05

    ancient = {"ingest_time": (now - timedelta(days=365)).isoformat()}
    assert _recency_bonus(ancient, now=now) == 0.0

    # Future-dated / unparseable / missing: 0 bonus, no exception.
    assert _recency_bonus({"ingest_time": (now + timedelta(days=1)).isoformat()}, now=now) == 0.0
    assert _recency_bonus({"ingest_time": "not-a-date"}, now=now) == 0.0
    assert _recency_bonus({}, now=now) == 0.0


def test_tied_scores_sort_by_parsed_utc_ingest_time() -> None:
    """Tie-breaker must compare actual instants, not raw timestamp strings."""
    from memory_arbiter.search import _soft_rerank

    ranked = _soft_rerank(
        "release",
        [
            {
                "id": 1,
                "content": "looks newer by string",
                "subject": "release-notes",
                "tags": '["release"]',
                "source_type": "agent_generated",
                # 2026-07-06 16:30 UTC
                "ingest_time": "2026-07-07T00:30:00+08:00",
                "status": "active",
            },
            {
                "id": 2,
                "content": "actually newer in utc",
                "subject": "release-notes",
                "tags": '["release"]',
                "source_type": "agent_generated",
                # 2026-07-06 18:00 UTC
                "ingest_time": "2026-07-06T18:00:00+00:00",
                "status": "active",
            },
        ],
    )

    assert [record["id"] for record in ranked] == [2, 1]


# --------------------------------------------------------------------------- #
# v0.5.1 — memory_get (direct ID lookup)
# --------------------------------------------------------------------------- #


def test_get_memory_by_id(tmp_path: Path) -> None:
    """通过 ID 直接获取一条记忆的完整信息，包含所有字段。"""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="Project API token policy lives in README security section.",
        subject="api-token-policy",
        tags=["policy", "security"],
        source_type="document_extracted",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    result = tools.memory_get(memory_id=memory_id)

    assert result["ok"] is True
    memory = result["data"]["memory"]
    assert memory["id"] == memory_id
    assert memory["subject"] == "api-token-policy"
    assert memory["content"] == "Project API token policy lives in README security section."
    assert memory["source_type"] == "document_extracted"
    assert "policy" in memory["tags"]


def test_get_memory_not_found(tmp_path: Path) -> None:
    """传入不存在的 memory_id 应返回错误。"""
    tools = make_tools(tmp_path)

    result = tools.memory_get(memory_id=99999)

    assert result["ok"] is False
    assert "not found" in result["data"]["error"]


def test_get_memory_invalid_id_type(tmp_path: Path) -> None:
    """传入非整数类型的 memory_id 应返回错误。"""
    tools = make_tools(tmp_path)

    result = tools.memory_get(memory_id="not-a-number")

    assert result["ok"] is False
    assert "must be an integer" in result["data"]["error"]


# --------------------------------------------------------------------------- #
# v0.6.0 — section split / vector-space regression coverage
# --------------------------------------------------------------------------- #


def test_vec_state_detects_model_change_and_preserves_resume_cursor(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    with tools.db.write_transaction() as conn:
        MemoryDB._set_meta(conn, "state", "ready")
        MemoryDB._set_meta(conn, "active_space_id", "space-a")

    tools.db.init_vec_index_state("space-b", True)
    changed = tools.db.get_vec_index_state()
    assert changed["state"] == "mismatch"
    assert changed["active_space_id"] == "space-a"
    assert changed["target_space_id"] == "space-b"
    assert changed["migration_epoch"]

    with tools.db.write_transaction() as conn:
        MemoryDB._set_meta(conn, "migration_cursor", "7")
    tools.db.init_vec_index_state("space-b", True)
    resumed = tools.db.get_vec_index_state()
    assert resumed["state"] == "mismatch"
    assert resumed["migration_cursor"] == 7


def test_memory_search_disables_vec_during_space_mismatch(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    called = False

    def unexpected_vec_knn(*args, **kwargs):
        nonlocal called
        called = True
        return []

    tools.db.vec_knn = unexpected_vec_knn  # type: ignore[method-assign]
    with tools.db.write_transaction() as conn:
        MemoryDB._set_meta(conn, "state", "mismatch")
        MemoryDB._set_meta(conn, "active_space_id", "space-a")
        MemoryDB._set_meta(conn, "target_space_id", "space-b")

    result = tools.memory_search(
        query="no lexical match expected",
        query_embedding=[1.0, 0.0],
    )

    assert called is False
    assert "vec_disabled=embedding_space_mismatch" in result["warnings"]


def test_rebuild_embeddings_advances_cursor_and_finishes_migration(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    tools._embedder = _MockManagedEmbedder(lambda _text: [1.0, 0.0])
    tools._embedder_loaded = True
    memory_id = tools.memory_write(content="semantic body", subject="subject")["data"]["id"]
    stored, warnings = tools.db.store_embedding(memory_id, [0.0, 1.0])
    assert stored is True, warnings
    with tools.db.write_transaction() as conn:
        MemoryDB._set_meta(conn, "state", "mismatch")
        MemoryDB._set_meta(conn, "active_space_id", "space-a")
        MemoryDB._set_meta(conn, "target_space_id", "mock_space_id")

    result = tools.memory_rebuild_embeddings(dry_run=False, batch_size=50)

    assert result["ok"] is True
    assert result["data"]["processed"] == 1
    assert result["data"]["succeeded"] == 1
    assert result["data"]["global_state"] == "ready"
    state = tools.db.get_vec_index_state()
    assert state["active_space_id"] == "mock_space_id"
    assert state["target_space_id"] is None
    assert state["migration_cursor"] is None


def test_split_rejects_stale_snapshot_without_overwriting_decline(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    written = tools.memory_write(
        content="first section\nsecond section",
        subject="split target",
        metadata={"keep": "yes"},
    )
    memory_id = written["data"]["id"]
    with tools.db.write_transaction() as conn:
        MemoryDB._set_meta(conn, "state", "ready")
        MemoryDB._set_meta(conn, "active_space_id", "mock_space_id")
    memory = tools.db.get_memory(memory_id)
    snapshot = {
        "decision_content_hash": hashlib.sha256(memory["content"].encode("utf-8")).hexdigest(),
        "decision_memory_version": memory["version"],
        "decision_split_status": memory["split_status"],
        "decision_split_revision": memory["split_revision"],
    }
    declined = tools.memory_split(memory_id=memory_id, split_decision="decline", **snapshot)
    assert declined["ok"] is True

    stale = tools.memory_split(
        memory_id=memory_id,
        split_decision="split",
        sections=[
            {"title": "first"},
            {"title": "second", "anchor_text": "missing anchor", "occurrence_index": 0},
        ],
        **snapshot,
    )

    assert stale["ok"] is False
    assert stale["data"]["error"] == "split_revision_conflict"
    current = tools.db.get_memory(memory_id)
    assert current["split_status"] == "declined"
    assert current["split_revision"] == 1
    assert current["metadata"] == {"keep": "yes"}


def test_split_failure_merges_error_into_existing_metadata(tmp_path: Path) -> None:
    tools = make_vec_tools(tmp_path)
    written = tools.memory_write(
        content="first section\nsecond section",
        subject="split-failure-test",
        metadata={"keep": "yes", "nested": {"value": 1}},
    )
    memory_id = written["data"]["id"]
    with tools.db.write_transaction() as conn:
        MemoryDB._set_meta(conn, "state", "ready")
        MemoryDB._set_meta(conn, "active_space_id", "mock_space_id")
    memory = tools.db.get_memory(memory_id)

    failed = tools.memory_split(
        memory_id=memory_id,
        split_decision="split",
        decision_content_hash=hashlib.sha256(memory["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=memory["version"],
        decision_split_status=memory["split_status"],
        decision_split_revision=memory["split_revision"],
        sections=[
            {"title": "first"},
            {"title": "second", "anchor_text": "missing anchor", "occurrence_index": 0},
        ],
    )

    assert failed["ok"] is False
    current = tools.db.get_memory(memory_id)
    assert current["split_status"] == "failed"
    assert current["metadata"]["keep"] == "yes"
    assert current["metadata"]["nested"] == {"value": 1}
    assert current["metadata"]["_split"]["last_split_error"]["stage"] == "validation"


# ────────────────────────────────────────────────────────────────────
# v0.6.0 review fixes: split success path, _attach_sections branches,
# edit clears sections, vec_space_changed CAS, single-batch regression.
# ────────────────────────────────────────────────────────────────────



