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
        split_enabled=True,
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
    ):
        monkeypatch.delenv(key, raising=False)


def test_write_and_search(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="Project API token policy lives in README security section.",
        tags=["policy", "security"],
        source_type="document_extracted",
        event_time="2026-01-01T00:00:00Z",
        subject="api-token-policy",
    )
    assert written["ok"] is True
    assert written["data"]["id"] is not None

    found = tools.memory_search(query="token policy", workspace="repo-a")
    assert found["ok"] is True
    assert found["data"]["count"] >= 1
    assert found["data"]["results"][0]["subject"] == "api-token-policy"


def test_chinese_search_matches_contiguous_fragment(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="京东科技金融营销系统梳理，覆盖金营、营销链路、活动费用申请和活动配置。",
        tags=["营销系统", "金营", "系统梳理"],
        source_type="document_extracted",
        subject="京东科技金融-营销系统梳理",
    )

    found = tools.memory_search(query="营销系统", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] >= 1
    assert found["data"]["results"][0]["subject"] == "京东科技金融-营销系统梳理"


def test_chinese_search_overspecified_query_still_matches(tmp_path: Path) -> None:
    """Regression for the original '营销交付系统' bug: a query whose bigrams
    are not all present in the document must still hit FTS5 via OR recall on
    the shared bigrams, instead of falling back to recent memories."""
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="营销交付需求提报：银行走XBP，自持走邮件，财富走JoySpace。活动配置走权益中台。",
        tags=["营销交付", "营销链路"],
        source_type="document_extracted",
        subject="营销交付-需求提报链路",
    )

    # "营销交付系统" has bigrams (付系, 系统) absent from the doc; the shared
    # bigrams (营销, 销交, 交付) must still match via OR.
    found = tools.memory_search(query="营销交付系统", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] >= 1
    assert found["data"]["results"][0]["subject"] == "营销交付-需求提报链路"
    # Must NOT have triggered the recent-memory fallback.
    assert not any("No direct memory match" in w for w in found["warnings"])


def test_search_returns_recent_memories_when_no_direct_match(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="来源：营销系统梳理.xlsx。营销链路包含需求提报、费用申请、设计图提报、活动配置等环节。",
        tags=["营销系统", "金营", "PRD参考"],
        source_type="document_extracted",
        subject="京东科技金融-营销系统梳理",
        event_time="2026-07-04T09:00:30Z",
    )

    found = tools.memory_search(query="营销运营全流程 系统", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] == 1
    assert found["data"]["results"][0]["subject"] == "京东科技金融-营销系统梳理"
    assert any("No direct memory match" in warning for warning in found["warnings"])


def test_recent_fallback_ranks_by_trust_then_recency(tmp_path: Path) -> None:
    """Regression: when a query misses FTS5 and falls back to recent memories,
    a user_confirmed+locked record must outrank a newer agent_generated+normal
    one. Previously pure ``ingest_time DESC`` ordering let daily agent chatter
    bury authoritative records."""
    tools = make_tools(tmp_path)
    # Newer but low-trust: agent chatter from today.
    tools.memory_write(
        content="memory-arbiter v0.2.4 release notes supersede tooling spec",
        subject="release-chatter",
        source_type="agent_generated",
        event_time="2026-07-05T00:00:00Z",
    )
    # Older but authoritative: the actual system overview the user confirmed.
    tools.memory_write(
        content="营销系统梳理：金营平台、权益中台、活动配置全链路系统清单",
        subject="营销系统-权威",
        source_type="user_confirmed",
        confidence=1.0,
        event_time="2026-07-04T00:00:00Z",
    )

    found = tools.memory_search(query="完全不存在的查询词 xyz123", workspace="repo-a")

    assert found["ok"] is True
    assert any("No direct memory match" in w for w in found["warnings"])
    results = found["data"]["results"]
    assert len(results) == 2
    # Authoritative record must rank first despite being older.
    assert results[0]["subject"] == "营销系统-权威"
    assert results[0]["protection_level"] == "locked"
    assert results[1]["subject"] == "release-chatter"


def test_memory_recent_ignores_workspace_filter(tmp_path: Path) -> None:
    # v0.7.4 (M3): workspace is reserved metadata; memory_recent lists the
    # whole shared library regardless of the workspace argument.
    tools = make_tools(tmp_path)
    tools.memory_write(content="Old memory", subject="old", event_time="2026-01-01T00:00:00Z")
    tools.memory_write(content="New memory", subject="new", event_time="2026-02-01T00:00:00Z")
    tools.memory_write(content="Other workspace", subject="other", workspace="repo-b", event_time="2026-03-01T00:00:00Z")

    recent = tools.memory_recent(workspace="repo-a", limit=10)

    assert recent["ok"] is True
    subjects = [record["subject"] for record in recent["data"]["results"]]
    # All three visible — passing workspace="repo-a" does NOT hide repo-b.
    assert "other" in subjects
    assert "new" in subjects
    assert "old" in subjects
    # Ordered newest-first by event_time.
    assert subjects == ["other", "new", "old"]


def test_arbitration_prefers_event_time(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    old = tools.memory_write(content="Use port 3000", subject="dev-port", event_time="2026-01-01T00:00:00Z")
    new = tools.memory_write(content="Use port 5173", subject="dev-port", event_time="2026-02-01T00:00:00Z")
    result = tools.memory_arbitrate(old["data"]["id"], new["data"]["id"], mark_conflict=True)

    assert result["ok"] is True
    assert result["data"]["comparison"]["winner_id"] == new["data"]["id"]
    assert result["data"]["conflict_id"] is not None


def test_user_confirmed_is_protected(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    confirmed = tools.memory_write(
        content="User says production branch is main",
        subject="prod-branch",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    generated = tools.memory_write(
        content="Agent guessed production branch is release",
        subject="prod-branch",
        source_type="agent_generated",
        event_time="2026-03-01T00:00:00Z",
    )
    result = tools.memory_arbitrate(confirmed["data"]["id"], generated["data"]["id"], apply=True)

    assert result["data"]["comparison"]["winner_id"] == confirmed["data"]["id"]
    assert "automatic overwrite is forbidden" in result["data"]["comparison"]["reasons"][0]


def test_confirm_promotes_record(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(content="OpenClaw memory plugin is enabled for agent alpha", source_type="pending")
    confirmed = tools.memory_confirm(written["data"]["id"], source_ref="user-chat")

    assert confirmed["ok"] is True
    assert confirmed["data"]["record"]["source_type"] == SourceType.USER_CONFIRMED.value
    assert confirmed["data"]["record"]["protection_level"] == "locked"


def test_supersede_requires_authorization(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="Old spec",
        subject="spec",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    rejected = tools.memory_supersede(memory_id=memory_id, reason="replaced", authorized=False)
    assert rejected["ok"] is False
    assert rejected["data"]["superseded"] is False
    # Memory must remain active+locked when unauthorized
    still_active = tools.db.get_memory(memory_id)
    assert still_active["status"] == "active"
    assert still_active["protection_level"] == "locked"


def test_supersede_marks_record_and_resolves_conflicts(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    old = tools.memory_write(
        content="Old release spec",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    new = tools.memory_write(
        content="New release spec supersedes the old one",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-02-01T00:00:00Z",
    )
    old_id, new_id = old["data"]["id"], new["data"]["id"]

    # arbitrate hits the user-protected wall and leaves an open conflict
    blocked = tools.memory_arbitrate(old_id, new_id, mark_conflict=True, apply=True)
    assert blocked["data"]["comparison"]["manual_review"] is True
    open_conflicts_before = tools.memory_list_conflicts(status="open")["data"]["count"]
    assert open_conflicts_before >= 1

    result = tools.memory_supersede(
        memory_id=old_id,
        reason="replaced by newer spec",
        superseded_by=new_id,
        authorized=True,
    )

    assert result["ok"] is True
    assert result["data"]["superseded"] is True
    assert result["data"]["memory_id"] == old_id
    assert result["data"]["conflict_id"] is not None
    assert result["data"]["linked_conflicts_resolved"] >= 1

    updated = tools.db.get_memory(old_id)
    assert updated["status"] == "superseded"
    assert updated["protection_level"] == "normal"

    # The open conflict from the blocked arbitrate must now be resolved
    open_conflicts_after = tools.memory_list_conflicts(status="open")["data"]["count"]
    assert open_conflicts_after == 0
    # And an audit row exists for the supersede itself
    resolved = tools.memory_list_conflicts(status="resolved")["data"]["conflicts"]
    assert any("USER-AUTHORIZED SUPERSEDE" in c["reason"] for c in resolved)


def test_supersede_rejects_already_superseded(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="Stale memory",
        subject="stale",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    first = tools.memory_supersede(memory_id=memory_id, reason="stale", authorized=True)
    assert first["ok"] is True

    second = tools.memory_supersede(memory_id=memory_id, reason="stale again", authorized=True)
    assert second["ok"] is False
    assert "already" in second["data"]["error"]


def test_degraded_status_mentions_missing_vec(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    status = tools.memory_status()

    assert status["ok"] is True
    assert status["degraded"] is True
    # Wording varies by whether the package is installed; pin only the
    # invariant — vec is off and the warning says so.
    assert any("disabled" in warning and "sqlite-vec" in warning for warning in status["warnings"])


def test_vec_disabled_but_installed_warns_with_enable_hint(tmp_path: Path) -> None:
    """When the package is loadable but the env switch is off, the warning
    should point at the exact env var to flip — not just say "disabled".

    This is the diagnostic gap that made the last reinstall-overwrite incident
    hard to spot: the user saw a generic "disabled by configuration" and had
    no way to tell the package was actually fine, only the switch was missing.
    Skipped on machines where sqlite-vec isn't installed (the hint would be
    misleading there — the install line covers that path instead).
    """
    pytest.importorskip("sqlite_vec")
    tools = make_tools(tmp_path)  # enable_sqlite_vec=False in this fixture
    status = tools.memory_status()

    assert status["data"]["sqlite_vec_available"] is False
    joined = " ".join(status["warnings"])
    assert "MEMORY_ARBITER_ENABLE_SQLITE_VEC=true" in joined


def test_compare_manual_review_when_both_protected() -> None:
    left = {
        "id": 1,
        "source_type": "user_confirmed",
        "protection_level": "locked",
        "event_time": "2026-01-01T00:00:00Z",
        "ingest_time": "2026-01-02T00:00:00Z",
    }
    right = {
        "id": 2,
        "source_type": "user_confirmed",
        "protection_level": "locked",
        "event_time": "2026-02-01T00:00:00Z",
        "ingest_time": "2026-02-02T00:00:00Z",
    }
    result = compare_memories(left, right)

    assert result["manual_review"] is True
    assert result["winner_id"] is None


def test_audit_summary_aggregates_per_workspace(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="Confirmed port 5173",
        subject="dev-port",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    tools.memory_write(
        content="Agent guessed port 3000",
        subject="dev-port",
        source_type="agent_generated",
        event_time="2026-03-01T00:00:00Z",
    )
    tools.memory_write(
        content="Other workspace memory",
        subject="other",
        workspace="repo-b",
        source_type="document_extracted",
        event_time="2026-02-01T00:00:00Z",
    )
    # Create an open conflict inside repo-a
    ids = [r["id"] for r in tools.memory_recent(workspace="repo-a", limit=10)["data"]["results"]]
    tools.memory_arbitrate(ids[0], ids[1], mark_conflict=True)

    summary = tools.memory_audit_summary()
    data = summary["data"]

    assert summary["ok"] is True
    assert data["total_memories"] == 3
    assert data["total_open_conflicts"] == 1
    repo_a = data["workspaces"]["repo-a"]
    assert repo_a["count"] == 2
    assert repo_a["oldest"] == "2026-01-01T00:00:00+00:00"
    assert repo_a["newest"] == "2026-03-01T00:00:00+00:00"
    assert repo_a["open_conflicts"] == 1
    assert repo_a["by_source_type"] == {"user_confirmed": 1, "agent_generated": 1}
    assert data["workspaces"]["repo-b"]["count"] == 1


def test_audit_summary_empty_when_no_memories(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    summary = tools.memory_audit_summary()
    assert summary["ok"] is True
    assert summary["data"] == {
        "workspaces": {},
        "total_memories": 0,
        "total_open_conflicts": 0,
    }


def test_fts5_search_handles_query_with_dots(tmp_path: Path) -> None:
    """Regression: queries containing '.' (e.g. version numbers, file paths)
    used to raise ``fts5: syntax error near "."`` and silently fall back to
    LIKE. FTS5 must now run without warnings."""
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="Release notes for memory-arbiter v0.2.1, the token optimization release.",
        subject="v0.2.1 release",
        source_type="document_extracted",
    )

    found = tools.memory_search(query="v0.2.1", workspace="repo-a")

    assert found["ok"] is True
    assert found["data"]["count"] == 1
    assert found["data"]["results"][0]["subject"] == "v0.2.1 release"
    assert not any("FTS5 query failed" in w for w in found["warnings"])


def test_fts5_search_handles_special_chars_without_warning(tmp_path: Path) -> None:
    """FTS5 special chars (``: * ( ) -``) in the query must not trigger a
    syntax-error fallback to LIKE."""
    tools = make_tools(tmp_path)
    tools.memory_write(
        content="Config lives at config/db.py with key apiKey:path(0)",
        subject="config-paths",
        source_type="document_extracted",
    )

    for query in ("config/db.py", "apiKey:path(0)", "config * (db)"):
        found = tools.memory_search(query=query, workspace="repo-a")
        assert found["ok"] is True
        assert not any("FTS5 query failed" in w for w in found["warnings"]), (
            f"query {query!r} triggered FTS5 fallback: {found['warnings']}"
        )


def test_sanitize_fts_query_quotes_and_joins_tokens() -> None:
    from memory_arbiter.search import _sanitize_fts_query

    assert _sanitize_fts_query("") == ""
    assert _sanitize_fts_query("v0.2.1") == '"v0.2.1"'
    assert _sanitize_fts_query("v0.2.1 release task") == '"v0.2.1" AND "release" AND "task"'
    # Embedded double-quotes are escaped as "" per FTS5 phrase syntax
    assert _sanitize_fts_query('a"b') == '"a""b"'


def test_sanitize_fts_query_splits_cjk_into_trigram_or_group() -> None:
    """Regression: a CJK token used to be wrapped as a single strict phrase,
    so ``营销交付系统`` missed documents that contained ``营销交付需求提报``
    (one trigram absent). It must now expand to an OR of overlapping trigrams.

    The FTS5 table uses ``tokenize='trigram'``, which only matches queries
    that produce 3-char tokens — so 2-char phrases never hit, and a strict
    CJK phrase silently misses when overspecified. Bare trigrams joined by
    OR restore recall without any new tokenizer dependency.
    """
    from memory_arbiter.search import _sanitize_fts_query

    # Pure CJK: overlapping trigrams (unquoted) joined by OR.
    assert _sanitize_fts_query("营销交付") == "(营销交 OR 销交付)"
    assert _sanitize_fts_query("营销交付系统") == (
        "(营销交 OR 销交付 OR 交付系 OR 付系统)"
    )
    # Single/double CJK chars cannot form a trigram → dropped (LIKE fallback
    # handles them via the empty-FTS-result path).
    assert _sanitize_fts_query("营") == ""
    assert _sanitize_fts_query("营销") == ""
    # Mixed CJK + ASCII: short CJK token dropped, ASCII token preserved.
    assert _sanitize_fts_query("营销 marketing") == '"marketing"'
    # ASCII behavior unchanged when no CJK present.
    assert _sanitize_fts_query("v0.2.1 release") == '"v0.2.1" AND "release"'


def test_search_excludes_superseded_by_default(tmp_path: Path) -> None:
    """v0.2.6: search filters out superseded records unless explicitly opted in.

    A superseded record is, by definition, no longer authoritative. Letting it
    leak into default results pollutes recall — release chatter, superseded
    specs, etc. drown out current truth. The default must hide them.
    """
    tools = make_tools(tmp_path)
    active = tools.memory_write(
        content="release-spec-v2 active authoritative",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-02-01T00:00:00Z",
    )
    stale = tools.memory_write(
        content="release-spec-v1 stale superseded",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    active_id, stale_id = active["data"]["id"], stale["data"]["id"]
    superseded = tools.memory_supersede(memory_id=stale_id, reason="replaced", superseded_by=active_id, authorized=True)
    assert superseded["data"]["superseded"] is True

    found = tools.memory_search(query="release-spec", workspace="repo-a")
    ids = [r["id"] for r in found["data"]["results"]]
    assert active_id in ids
    assert stale_id not in ids, "superseded record leaked into default search results"


def test_search_includes_superseded_when_requested(tmp_path: Path) -> None:
    """v0.2.6: include_superseded=True returns both, with superseded ranked below.

    Audit/history walkthroughs need to see the full supersede chain. The flag
    restores them — but the ORDER BY clause must still sink superseded rows
    below every active row so history audits don't bury current truth.
    """
    tools = make_tools(tmp_path)
    active = tools.memory_write(
        content="release-spec-v2 active authoritative",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-02-01T00:00:00Z",
    )
    stale = tools.memory_write(
        content="release-spec-v1 stale superseded record",
        subject="release-spec",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    active_id, stale_id = active["data"]["id"], stale["data"]["id"]
    tools.memory_supersede(memory_id=stale_id, reason="replaced", superseded_by=active_id, authorized=True)

    found = tools.memory_search(query="release-spec", workspace="repo-a", include_superseded=True)
    ids = [r["id"] for r in found["data"]["results"]]
    assert active_id in ids, "active record missing under include_superseded=True"
    assert stale_id in ids, "superseded record not returned under include_superseded=True"
    # Superseded must rank below active even though both match.
    assert ids.index(stale_id) > ids.index(active_id), "superseded ranked above active"


def test_supersede_rejects_non_active_replacement(tmp_path: Path) -> None:
    """v0.2.6: supersede chain-breakage guard.

    Starting in v0.2.6, search hides superseded records by default. If a
    supersede points at a replacement that is itself deleted/superseded, the
    default would leave the chain pointing at a record search can't see — the
    user loses both views. Reject early with an explicit error.
    """
    tools = make_tools(tmp_path)
    # Build a chain: A (active) ← supersedes — B (active) ← supersedes — C (active)
    a = tools.memory_write(content="A active", subject="s", source_type="user_confirmed", event_time="2026-01-01T00:00:00Z")
    b = tools.memory_write(content="B active", subject="s", source_type="user_confirmed", event_time="2026-02-01T00:00:00Z")
    c = tools.memory_write(content="C active", subject="s", source_type="user_confirmed", event_time="2026-03-01T00:00:00Z")
    a_id, b_id, c_id = a["data"]["id"], b["data"]["id"], c["data"]["id"]
    # B supersedes A — A is now superseded, B still active.
    tools.memory_supersede(memory_id=a_id, reason="B replaces A", superseded_by=b_id, authorized=True)

    # Now try to supersede C by pointing at A (which is itself superseded) — must be rejected.
    rejected = tools.memory_supersede(memory_id=c_id, reason="C replaced by A", superseded_by=a_id, authorized=True)
    assert rejected["ok"] is False
    assert "not active" in rejected["data"]["error"]
    # C must remain active (the supersede was blocked before any state change).
    c_record = tools.db.get_memory(c_id)
    assert c_record["status"] == "active"


def test_superseded_always_ranked_below_active_even_with_higher_score(tmp_path: Path) -> None:
    """v0.2.6: the soft-demote clause is unconditional, not gated on the filter.

    Even when include_superseded=True lets superseded rows back into the result
    set, they must rank below active rows regardless of bm25 score. A
    superseded record that happens to mention the query terms more often than
    the active record (common: release chatter repeats the codename) must not
    bubble up. This is the audit-mode safety net.
    """
    tools = make_tools(tmp_path)
    # Active record: short, mentions query term once → lower bm25 signal.
    active = tools.memory_write(
        content="release release-spec canonical",
        subject="release-spec-active",
        source_type="user_confirmed",
        event_time="2026-02-01T00:00:00Z",
    )
    # Superseded record: long, repeats query term many times → higher bm25 signal.
    stale_blob = "release release release release release release-spec release-spec release-spec"
    stale = tools.memory_write(
        content=stale_blob,
        subject="release-spec-stale",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    active_id, stale_id = active["data"]["id"], stale["data"]["id"]
    tools.memory_supersede(memory_id=stale_id, reason="replaced", superseded_by=active_id, authorized=True)

    found = tools.memory_search(query="release", workspace="repo-a", include_superseded=True)
    ids = [r["id"] for r in found["data"]["results"]]
    assert active_id in ids and stale_id in ids
    assert ids.index(stale_id) > ids.index(active_id), (
        "superseded ranked above active despite the demote clause — ORDER BY is broken"
    )


# ---- v0.3.1: optional semantic recall (sqlite-vec vec0) -----------------
try:
    import sqlite_vec  # type: ignore  # noqa: F401
    _VEC_AVAILABLE = True
except Exception:
    _VEC_AVAILABLE = False


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


def test_edit_full_replacement_stores_history_and_updates_fts(tmp_path: Path) -> None:
    """Full content replace: version bumps, old content archived, FTS re-synced."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="version one content here",
        subject="s1",
        tags=["t"],
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    result = tools.memory_edit(memory_id=memory_id, new_content="version two content here", reason="refresh")
    assert result["ok"] is True
    assert result["data"]["edited"] is True
    assert result["data"]["new_version"] == 2
    assert result["data"]["history_id"] is not None

    updated = tools.db.get_memory(memory_id)
    assert updated["content"] == "version two content here"
    assert updated["version"] == 2

    # History archived the old snapshot at the old version
    history = tools.memory_history(memory_id=memory_id)["data"]["history"]
    assert len(history) == 1
    assert history[0]["content_snapshot"] == "version one content here"
    assert history[0]["version"] == 1
    assert history[0]["reason"] == "refresh"

    # FTS must reflect the new content, not the old. Query the FTS index
    # directly — memory_search falls back to "recent" when a token matches
    # nothing, which would mask a stale-FTS bug. Use a token unique to the old
    # body (kangaroo) vs unique to the new body (platypus).
    written2 = tools.memory_write(
        content="alpha kangaroo draft",
        subject="s2",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    mid2 = written2["data"]["id"]
    tools.memory_edit(memory_id=mid2, new_content="alpha platypus final", reason="swap")

    with tools.db.connection() as fts_conn:
        fts_new = [r["rowid"] for r in fts_conn.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'platypus'"
        ).fetchall()]
        fts_old = [r["rowid"] for r in fts_conn.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'kangaroo'"
        ).fetchall()]
    assert mid2 in fts_new, f"new token not in FTS: {fts_new}"
    assert mid2 not in fts_old, f"old token still in FTS (stale index): {fts_old}"
    # And the live memories row carries the new text (sanity)
    assert tools.db.get_memory(mid2)["content"] == "alpha platypus final"


def test_existing_database_is_migrated_to_version_chain_schema(tmp_path: Path) -> None:
    """Opening a pre-v0.4.0 DB adds version + memory_history idempotently."""
    db_path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          content TEXT NOT NULL,
          agent_id TEXT NOT NULL,
          workspace TEXT NOT NULL,
          tags TEXT NOT NULL DEFAULT '[]',
          source_type TEXT NOT NULL,
          source_ref TEXT,
          event_time TEXT NOT NULL,
          ingest_time TEXT NOT NULL,
          confidence REAL NOT NULL DEFAULT 0.5,
          protection_level TEXT NOT NULL DEFAULT 'normal',
          status TEXT NOT NULL DEFAULT 'active',
          subject TEXT,
          metadata TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL
        );
        INSERT INTO memories
        (content, agent_id, workspace, tags, source_type, event_time, ingest_time, subject, metadata, created_at)
        VALUES ('legacy content', 'agent-a', 'repo-a', '["legacy"]', 'agent_generated',
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'legacy', '{}',
                '2026-01-01T00:00:00Z');
        """
    )
    conn.commit()
    conn.close()

    settings = Settings(
        db_path=db_path,
        backup_jsonl=tmp_path / "backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=False,
    )
    db = MemoryDB(settings)
    with db.connection() as schema_conn:
        cols = {row["name"] for row in schema_conn.execute("PRAGMA table_info(memories)").fetchall()}
        history_table = schema_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_history'"
        ).fetchone()
    record = db.get_memory(1)

    assert "version" in cols
    assert history_table is not None
    assert record["content"] == "legacy content"
    assert record["version"] == 1

    # Idempotency: reopening the same DB should not fail or alter the row.
    db2 = MemoryDB(settings)
    assert db2.get_memory(1)["version"] == 1


def test_edit_partial_replacement(tmp_path: Path) -> None:
    """old_text+new_text does an exact substring substitution, not full replace."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="the api returns status 200 on success",
        subject="s",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    result = tools.memory_edit(memory_id=memory_id, old_text="200", new_text="404", reason="typo")
    assert result["ok"] is True
    assert result["data"]["record"]["content"] == "the api returns status 404 on success"
    assert result["data"]["new_version"] == 2

    # old_text not present -> explicit error, no mutation
    bad = tools.memory_edit(memory_id=memory_id, old_text="nonexistent", new_text="x")
    assert bad["ok"] is False
    assert "old_text not found" in bad["data"]["error"]
    # version unchanged after the failed edit
    assert tools.db.get_memory(memory_id)["version"] == 2


def test_edit_requires_authorization_for_locked(tmp_path: Path) -> None:
    """user_confirmed/locked records require authorized=True to edit."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="confirmed fact",
        subject="s",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    rejected = tools.memory_edit(memory_id=memory_id, new_content="tampered", reason="try", authorized=False)
    assert rejected["ok"] is False
    assert rejected["data"]["edited"] is False
    assert "authorized" in rejected["data"]["error"]
    # content untouched
    assert tools.db.get_memory(memory_id)["content"] == "confirmed fact"

    allowed = tools.memory_edit(memory_id=memory_id, new_content="corrected fact", reason="auth", authorized=True)
    assert allowed["ok"] is True
    assert allowed["data"]["edited"] is True
    assert tools.db.get_memory(memory_id)["content"] == "corrected fact"


def test_edit_normal_memory_no_auth_needed(tmp_path: Path) -> None:
    """agent_generated/normal records can be edited without authorized flag."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="draft note",
        subject="s",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    assert tools.db.get_memory(memory_id)["protection_level"] == "normal"

    result = tools.memory_edit(memory_id=memory_id, new_content="revised note", reason="cleanup")
    assert result["ok"] is True
    assert result["data"]["edited"] is True


def test_edit_rejects_superseded(tmp_path: Path) -> None:
    """Editing a superseded record is refused (idempotency / terminal-state gate)."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="doomed",
        subject="s",
        source_type="user_confirmed",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    tools.memory_supersede(memory_id=memory_id, reason="retired", authorized=True)

    rejected = tools.memory_edit(memory_id=memory_id, new_content="revived", reason="try", authorized=True)
    assert rejected["ok"] is False
    assert "already" in rejected["data"]["error"]
    assert rejected["data"]["edited"] is False


def test_history_returns_version_chain(tmp_path: Path) -> None:
    """Two edits produce two history rows; history is newest-version-first."""
    tools = make_tools(tmp_path)
    written = tools.memory_write(
        content="rev one",
        subject="s",
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]
    tools.memory_edit(memory_id=memory_id, new_content="rev two", reason="second")
    tools.memory_edit(memory_id=memory_id, new_content="rev three", reason="third")

    result = tools.memory_history(memory_id=memory_id)
    assert result["ok"] is True
    assert result["data"]["current_version"] == 3
    assert result["data"]["count"] == 2
    versions = [h["version"] for h in result["data"]["history"]]
    assert versions == [2, 1]  # newest version snapshot first
    assert result["data"]["history"][0]["content_snapshot"] == "rev two"


def test_cleanup_history_full_requires_authorization(tmp_path: Path) -> None:
    """Full history cleanup needs authorized=True; per-memory cleanup does not.
    And under no arguments must the memories table lose zero rows."""
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="a one", subject="a", source_type="agent_generated", event_time="2026-01-01T00:00:00Z")
    b = tools.memory_write(content="b one", subject="b", source_type="agent_generated", event_time="2026-01-01T00:00:00Z")
    a_id, b_id = a["data"]["id"], b["data"]["id"]
    tools.memory_edit(memory_id=a_id, new_content="a two", reason="e")
    tools.memory_edit(memory_id=b_id, new_content="b two", reason="e")
    # 2 history rows now

    # Full cleanup without auth -> rejected, nothing removed
    rejected = tools.memory_cleanup_history()
    assert rejected["ok"] is False
    assert rejected["data"]["cleaned"] == 0
    assert "authorized" in rejected["data"]["error"]
    assert tools.memory_history(memory_id=a_id)["data"]["count"] == 1  # history intact

    # Per-memory cleanup needs no auth
    single = tools.memory_cleanup_history(memory_id=a_id)
    assert single["ok"] is True
    assert single["data"]["cleaned"] == 1
    assert tools.memory_history(memory_id=a_id)["data"]["count"] == 0
    assert tools.memory_history(memory_id=b_id)["data"]["count"] == 1  # b untouched

    negative_age = tools.memory_cleanup_history(older_than_days=-1)
    assert negative_age["ok"] is False
    assert negative_age["data"]["cleaned"] == 0
    assert tools.memory_history(memory_id=b_id)["data"]["count"] == 1

    # Full cleanup WITH auth clears the rest
    full = tools.memory_cleanup_history(authorized=True)
    assert full["ok"] is True
    assert full["data"]["cleaned"] == 1  # b's remaining row

    # SAFETY RED LINE: memories table must be fully intact despite "full cleanup"
    assert tools.db.get_memory(a_id)["content"] == "a two"
    assert tools.db.get_memory(b_id)["content"] == "b two"


def test_server_memory_edit_preserves_tags_when_new_tags_omitted(tmp_path: Path, monkeypatch) -> None:
    """Regression: the MCP wrapper must pass new_tags=None through.

    Passing [] erases existing tags on a content-only edit, even though the
    MemoryTools layer correctly preserves tags when new_tags is omitted.
    """

    class FakeFastMCP:
        def __init__(self, _name: str) -> None:
            self.tools = {}

        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = func
                return func

            return decorator

    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = FakeFastMCP
    fake_server = types.ModuleType("mcp.server")
    fake_mcp = types.ModuleType("mcp")
    fake_server.fastmcp = fake_fastmcp
    fake_mcp.server = fake_server
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "server.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_BACKUP_JSONL", str(tmp_path / "server.backup.jsonl"))
    monkeypatch.setenv("MEMORY_ARBITER_WORKSPACE", "repo-a")
    monkeypatch.setenv("MEMORY_ARBITER_AGENT_ID", "agent-a")

    from memory_arbiter.server import build_server

    app = build_server()
    written = app.tools["memory_write"](
        content="draft content",
        subject="server-wrapper",
        tags=["keep-me"],
        source_type="agent_generated",
        event_time="2026-01-01T00:00:00Z",
    )
    memory_id = written["data"]["id"]

    edited = app.tools["memory_edit"](memory_id=memory_id, new_content="edited content")

    assert edited["ok"] is True
    assert edited["data"]["record"]["content"] == "edited content"
    assert edited["data"]["record"]["tags"] == ["keep-me"]


# --------------------------------------------------------------------------- #
# v0.5.0 — config file + automatic embedding
# --------------------------------------------------------------------------- #


def test_config_file_overrides_env(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    cfg_path = tmp_path / "config.json"
    cfg_db = tmp_path / "from-config.sqlite3"
    env_db = tmp_path / "from-env.sqlite3"
    cfg_path.write_text(
        json.dumps(
            {
                "db_path": str(cfg_db),
                "backup_jsonl": str(tmp_path / "from-config.jsonl"),
                "client": "from-config",
                "vec": {"enabled": True, "dim": 512},
                "embedding": {
                    "provider": "gguf",
                    "model_path": str(tmp_path / "model.gguf"),
                    "auto_query": False,
                    "auto_write": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(env_db))
    monkeypatch.setenv("MEMORY_ARBITER_VEC_DIM", "999")

    settings = Settings.from_env()

    assert settings.db_path == cfg_db
    assert settings.client == "from-config"
    assert settings.enable_sqlite_vec is True
    assert settings.vec_dim == 512
    assert settings.embedding_provider == "gguf"
    assert settings.embedding_model_path == tmp_path / "model.gguf"
    assert settings.embedding_auto_query is False


def test_env_fallback_when_config_absent(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "env.sqlite3"))
    monkeypatch.setenv("MEMORY_ARBITER_ENABLE_SQLITE_VEC", "true")
    monkeypatch.setenv("MEMORY_ARBITER_VEC_DIM", "1024")
    monkeypatch.setenv("MEMORY_ARBITER_GGUF", str(tmp_path / "legacy.gguf"))

    settings = Settings.from_env()

    assert settings.db_path == tmp_path / "env.sqlite3"
    assert settings.enable_sqlite_vec is True
    assert settings.vec_dim == 1024
    assert settings.embedding_provider == "gguf"
    assert settings.embedding_model_path == tmp_path / "legacy.gguf"


def test_config_file_parse_error_graceful(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    bad_cfg = tmp_path / "bad.json"
    bad_cfg.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(bad_cfg))
    monkeypatch.setenv("MEMORY_ARBITER_DB_PATH", str(tmp_path / "env.sqlite3"))

    settings = Settings.from_env()

    assert settings.db_path == tmp_path / "env.sqlite3"
    assert any("JSON parse failed" in warning for warning in settings.config_warnings)


def test_config_env_path_not_exist_fallback_xdg(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    home = tmp_path / "home"
    xdg_cfg = home / ".config" / "memory-arbiter" / "config.json"
    xdg_cfg.parent.mkdir(parents=True)
    xdg_cfg.write_text(json.dumps({"db_path": str(tmp_path / "xdg.sqlite3")}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(tmp_path / "missing.json"))

    settings = Settings.from_env()

    assert settings.db_path == tmp_path / "xdg.sqlite3"
    assert any("does not exist" in warning for warning in settings.config_warnings)


def test_bad_field_value_degrades_with_warning(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"vec": {"enabled": "maybe", "dim": "abc"}, "embedding": {"auto_write": "??"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert settings.enable_sqlite_vec is False
    assert settings.vec_dim == 768
    assert settings.embedding_auto_write is True
    assert any("vec.enabled" in warning for warning in settings.config_warnings)
    assert any("vec.dim" in warning for warning in settings.config_warnings)
    assert any("embedding.auto_write" in warning for warning in settings.config_warnings)


def test_parse_bool_false_string_is_false() -> None:
    assert parse_bool("false", default=True) is False
    assert parse_bool("0", default=True) is False
    assert parse_bool("no", default=True) is False


def test_embedding_model_path_without_provider_defaults_to_gguf(tmp_path: Path, monkeypatch) -> None:
    clear_config_env(monkeypatch)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"embedding": {"model_path": str(tmp_path / "model.gguf")}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg_path))

    settings = Settings.from_env()

    assert settings.embedding_provider == "gguf"
    assert settings.embedding_model_path == tmp_path / "model.gguf"


def test_no_embedding_no_false_warning(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    written = tools.memory_write(content="plain lexical memory", subject="lexical")
    found = tools.memory_search(query="lexical")

    assert "embedding_stored" not in written["data"]
    assert not any("embedding configured" in warning for warning in found["warnings"])
    assert not any("auto-embedding" in warning for warning in found["warnings"])


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

    def fake_search_memories(db, query, workspace, tags, limit, include_superseded=False, debug_ranking=False, query_embedding=None, **kwargs):
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

    def fake_search_memories(db, query, workspace, tags, limit, include_superseded=False, debug_ranking=False, query_embedding=None, **kwargs):
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


def _keyword_embedding(text: str) -> list[float]:
    """Deterministic 2D embedding keyed off the first token in text.

    Maps the first run of word-ish chars to one of a fixed set of orthogonal /
    diametrically-opposed unit vectors so cosine distances are predictable:
    two distinct known tokens are always > 0.7 apart, while identical tokens
    are at distance 0.  Unknown tokens map to a sentinel direction.
    """
    # Ordered fixed directions around the unit circle (90° apart → cosine dist 1.0
    # between neighbours, 2.0 between opposites).  The key's hash picks one index.
    m = re.search(r"[A-Za-z0-9\u4e00-\u9fff]+", text or "")
    key = m.group(0) if m else "x"
    table = {
        "alpha": (1.0, 0.0),
        "beta": (0.0, 1.0),
        "gamma": (-1.0, 0.0),
        "delta": (0.0, -1.0),
    }
    return [table[key][0], table[key][1]] if key in table else [-0.7071, -0.7071]


def _keyword_embedder(space_id: str = "mock_space_id"):
    """A ManagedEmbedder-like mock whose embedding is keyed off text content."""
    return _MockManagedEmbedder(lambda text: _keyword_embedding(text))


def _set_vec_ready(tools: MemoryTools, space_id: str = "mock_space_id") -> None:
    with tools.db.write_transaction() as conn:
        MemoryDB._set_meta(conn, "state", "ready")
        MemoryDB._set_meta(conn, "active_space_id", space_id)


def _publish_two_sections(tools: MemoryTools, memory_id: int, content: str,
                          first_anchor_token: str, second_anchor_token: str) -> dict:
    """Helper: publish a 2-section split whose anchors genuinely exist in content."""
    mem = tools.db.get_memory(memory_id)
    return tools.memory_split(
        memory_id=memory_id,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": first_anchor_token},
            {"title": second_anchor_token, "anchor_text": second_anchor_token, "occurrence_index": 0},
        ],
    )


def test_split_publish_success_then_search_returns_matched_sections(tmp_path: Path) -> None:
    """Happy path: valid anchors → offsets resolve → publish → search returns matched_sections."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    # Content: two distinct anchors so each section embeds to a different vector.
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)

    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True, published
    assert published["data"]["split_active"] is True
    assert published["data"]["section_count"] == 2

    mem = tools.db.get_memory(memory_id)
    assert mem["split_status"] == "active"
    # Sections + section vecs written
    sections = tools.db.get_sections_by_memory(memory_id)
    assert len(sections) == 2
    with tools.db.connection() as conn:
        vec_ids = MemoryDB._get_section_vec_ids(conn, memory_id)
    assert len(vec_ids) == 2

    # Search with a query whose embedding matches one section's keyword.
    result = tools.memory_search(query="beta", query_embedding=_keyword_embedding("beta"))
    assert result["ok"] is True
    hit = next(r for r in result["data"]["results"] if r["id"] == memory_id)
    # 1/2 matched → partial branch → content_scope=matched_sections, full
    # section bodies present in matched_sections.
    assert hit["content_scope"] == "matched_sections"
    assert hit["section_enhancement_applied"] is True
    assert hit.get("matched_sections")
    assert hit["matched_sections"][0]["title"] == "beta"
    # v0.8: matched_sections carry the full section content.
    assert hit["matched_sections"][0].get("content")


def test_split_publish_success_zero_hit_returns_full_memory(tmp_path: Path) -> None:
    """Zero section matches → return the FULL memory (design §6.3), no preview."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)
    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True

    # Query with a token that maps to neither section's vector.
    result = tools.memory_search(query="zzz", query_embedding=_keyword_embedding("zzz"))
    hit = next(r for r in result["data"]["results"] if r["id"] == memory_id)
    # v0.8: zero-match returns the full memory, not a bounded preview.
    assert hit["content_scope"] == "full_memory"
    assert hit["content"] == content
    assert hit.get("content_truncated") is None  # removed in v0.8
    assert "content_omitted" not in hit           # removed in v0.8
    assert hit["section_enhancement_applied"] is True


def test_split_publish_success_fulltext_fallback(tmp_path: Path) -> None:
    """When the matched fraction ≥ section_fulltext_threshold → return full text."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    # Lower fulltext threshold to 0.0 so any match counts as "most matched".
    tools.settings.section_fulltext_threshold = 0.0
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)
    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True

    result = tools.memory_search(query="alpha", query_embedding=_keyword_embedding("alpha"))
    hit = next(r for r in result["data"]["results"] if r["id"] == memory_id)
    assert hit["content_scope"] == "full_memory"
    assert hit["section_enhancement_applied"] is True
    assert hit.get("matched_sections")  # reference list still present
    assert hit.get("content")  # full text returned


def test_edit_clears_sections_and_bumps_revision(tmp_path: Path) -> None:
    """After a successful publish, editing content clears sections + bumps split_revision."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)
    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True
    assert tools.db.get_memory(memory_id)["split_revision"] == 1

    edited = tools.memory_edit(memory_id=memory_id, new_content=content + "\n appended")
    assert edited["ok"] is True

    after = tools.db.get_memory(memory_id)
    assert after["split_status"] is None
    assert after["split_revision"] == 2
    assert tools.db.get_sections_by_memory(memory_id) == []
    with tools.db.connection() as conn:
        assert MemoryDB._get_section_vec_ids(conn, memory_id) == set()


def test_attach_sections_invariant_missing_section_vec(tmp_path: Path) -> None:
    """Manually deleting a section vec → invariant reported + full text returned."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)
    published = _publish_two_sections(tools, memory_id, content, "alpha", "beta")
    assert published["ok"] is True

    # Sabotage: delete one section vector.
    sections = tools.db.get_sections_by_memory(memory_id)
    victim = sections[0]["id"]
    with tools.db.connection() as conn:
        conn.execute("DELETE FROM memory_sections_vec WHERE id = ?", (victim,))
        conn.commit()

    result = tools.memory_search(query="alpha", query_embedding=_keyword_embedding("alpha"))
    hit = next(r for r in result["data"]["results"] if r["id"] == memory_id)
    assert "split_invariant_broken_missing_section_vec" in hit.get("warnings", [])
    assert hit["content_scope"] == "full_memory"
    assert hit["section_enhancement_applied"] is False


def test_split_publish_rejects_when_vec_space_changed(tmp_path: Path) -> None:
    """active_space_id != embedder.embedding_space_id at publish → vec_space_changed, no write."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    # ready but active_space_id is a DIFFERENT space than the embedder.
    _set_vec_ready(tools, space_id="some_other_space")

    mem = tools.db.get_memory(memory_id)
    rejected = tools.memory_split(
        memory_id=memory_id,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": "alpha"},
            {"title": "beta", "anchor_text": "beta", "occurrence_index": 0},
        ],
    )
    assert rejected["ok"] is False
    assert rejected["data"]["error"] == "vec_space_changed"
    # Nothing written.
    assert tools.db.get_sections_by_memory(memory_id) == []
    assert tools.db.get_memory(memory_id)["split_status"] is None


def test_split_single_batch_ignores_legacy_batch_params(tmp_path: Path) -> None:
    """Regression: the dropped prepare_batch_index/llm_batch_chars kwargs are absorbed by **_, not errors."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)

    # Prepare call passing the now-removed params — must not raise.
    prepared = tools.memory_split(
        memory_id=memory_id,
        prepare_batch_index=0,  # legacy, now ignored via **_
        llm_batch_chars=9999,   # legacy, now ignored via **_
    )
    assert prepared["ok"] is True
    assert "content" in prepared["data"]
    # New single-batch response has no batch_count / llm_batch_chars fields.
    assert "batch_count" not in prepared["data"]
    assert "llm_batch_chars" not in prepared["data"]


def test_empty_embedding_not_stored_on_write_and_search(tmp_path: Path) -> None:
    """Never-raises contract: when embed_text returns an empty embedding (encode
    failure), memory_write must not store it and memory_search must not open the
    vec gate.  Does not require sqlite-vec.
    """
    settings = Settings(
        db_path=tmp_path / "memory.sqlite3",
        backup_jsonl=tmp_path / "backup.jsonl",
        enable_sqlite_vec=True,
        embedding_provider="gguf",
        embedding_model_path=tmp_path / "model.gguf",
        embedding_auto_write=True,
        embedding_auto_query=True,
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))

    def bad_encode(_text: str) -> list[float]:
        raise RuntimeError("model crashed")

    failing = _MockManagedEmbedder(bad_encode)
    tools._ensure_embedder = lambda: (failing, [])  # type: ignore[method-assign]

    # ---- memory_write: empty embedding must not be stored ----
    written = tools.memory_write(content="body text", subject="subject")
    assert written["ok"] is True
    assert written["data"].get("embedding_stored") is not True
    assert any("auto-embedding write failed" in w for w in written["warnings"])

    # ---- memory_search: empty query embedding must not be used ----
    result = tools.memory_search(query="anything", query_embedding=None)
    assert result["ok"] is True
    assert any("auto-embedding query failed" in w for w in result["warnings"])


def test_empty_embedding_rejects_split_publish(tmp_path: Path) -> None:
    """Never-raises contract: when section embed_text returns empty, the split
    publish must be rejected and no sections/vecs written.
    """
    pytest.importorskip("sqlite_vec")
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    memory_id = tools.memory_write(content=content, subject="doc")["data"]["id"]
    _set_vec_ready(tools)

    # Prepare with a working embedder (no embedding needed for prepare).
    mem = tools.db.get_memory(memory_id)
    prepared = tools.memory_split(memory_id=memory_id)
    assert prepared["ok"] is True

    # Switch to a failing embedder for the publish step.
    def bad_encode(_text: str) -> list[float]:
        raise RuntimeError("model crashed")

    failing = _MockManagedEmbedder(bad_encode)
    tools._ensure_embedder = lambda: (failing, [])  # type: ignore[method-assign]

    rejected = tools.memory_split(
        memory_id=memory_id,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": "alpha"},
            {"title": "beta", "anchor_text": "beta", "occurrence_index": 0},
        ],
    )
    assert rejected["ok"] is False
    assert "section embedding failed" in rejected["data"]["error"]
    # No sections or section vecs written.
    assert tools.db.get_sections_by_memory(memory_id) == []
    with tools.db.connection() as conn:
        vec_ids = MemoryDB._get_section_vec_ids(conn, memory_id)
    assert len(vec_ids) == 0


# ===========================================================================
# v0.6.1 Channel 6 (section-vec KNN) tests — T1 through T14.
# See docs/v0.6.1_detailed_design_channel6.md §6.2 for the test matrix.
# ===========================================================================


def _make_channel6_tools(tmp_path: Path, pool_cap: int = 50) -> MemoryTools:
    """Vec-enabled tools with split on + a small pool cap (for saturation tests)."""
    pytest.importorskip("sqlite_vec")
    settings = Settings(
        db_path=tmp_path / "ch6.sqlite3",
        backup_jsonl=tmp_path / "ch6-backup.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="repo-a",
        enable_sqlite_vec=True,
        vec_dim=2,
        split_enabled=True,
        split_threshold=1,
        recall_pool_cap=pool_cap,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_v061_t1_channel6_recalls_what_channel5_misses(tmp_path: Path) -> None:
    """T1: Channel 6 can surface a memory that Channel 5 cannot. We verify the
    mechanism directly: section_vec_knn returns the target (via its section vec)
    while vec_knn returns nothing (no memory-level vec stored). Per §6.2 the mock
    embedder can't simulate true dilution, so we test the mechanism, not the
    end-to-end KNN ranking."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # Target memory: NO memory-level vec stored (Channel 5 can't recall it).
    target_content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    target_id = tools.memory_write(content=target_content, subject="target")["data"]["id"]
    published = _publish_two_sections(tools, target_id, target_content, "alpha", "beta")
    assert published["ok"] is True

    # Channel 5 (memory-level vec KNN) finds nothing — no vec stored.
    ch5_rows = tools.db.vec_knn(_keyword_embedding("beta"), k=5)
    assert target_id not in [r["id"] for r in ch5_rows], (
        "Channel 5 should NOT recall a memory with no memory-level vec"
    )
    # Channel 6 (section-level vec KNN) DOES find it — the "beta" section vec matches.
    ch6_rows = tools.db.section_vec_knn(_keyword_embedding("beta"), k=5)
    recalled_ids = {r["memory_id"] for r in ch6_rows}
    assert target_id in recalled_ids, (
        "Channel 6 (section_vec_knn) should recall the target via its section vec"
    )


def test_v061_t2_dedup_one_memory_multiple_sections(tmp_path: Path) -> None:
    """T2: one memory with 3 sections all near the query enters the pool once."""
    tools = _make_channel6_tools(tmp_path, pool_cap=10)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # Three sections, each starting with "alpha" so each section vector = (1,0).
    content = "alpha " + ("x" * 60) + "\nalpha " + ("y" * 60) + "\nalpha " + ("z" * 60)
    mid = tools.memory_write(content=content, subject="dedup")["data"]["id"]
    mem = tools.db.get_memory(mid)
    published = tools.memory_split(
        memory_id=mid,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": "alpha"},
            {"title": "alpha", "anchor_text": "alpha", "occurrence_index": 1},
            {"title": "alpha", "anchor_text": "alpha", "occurrence_index": 2},
        ],
    )
    assert published["ok"] is True

    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    # The memory should appear exactly once in debug ranking (dedup held).
    target_entries = [r for r in result["data"]["results"] if r["id"] == mid]
    assert len(target_entries) == 1


def test_v061_t3_split_active_exempt_from_long_content_penalty(tmp_path: Path) -> None:
    """T3: a split-active long doc recalled by FTS does NOT get long-content penalty."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    long_content = "alpha " + ("x" * 3000) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=long_content, subject=None, tags=None)["data"]["id"]
    _publish_two_sections(tools, mid, long_content, "alpha", "beta")

    # Query "alpha" hits content lexically (FTS) but subject/tags are None (weak).
    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    assert mid in debug_map, "target memory must appear in results for the test to be meaningful"
    notes = debug_map[mid].get("_ranking_notes", [])
    assert "long content penalty applied" not in notes, (
        "split-active long doc should be exempt from long-content penalty"
    )


def test_v061_t4_non_split_still_gets_long_content_penalty(tmp_path: Path) -> None:
    """T4: regression — a non-split long memory still incurs long-content penalty."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    long_content = "alpha " + ("x" * 3000)
    mid = tools.memory_write(content=long_content, subject=None, tags=None)["data"]["id"]
    # NOT split — so the penalty should still apply.

    result = tools.memory_search(query="alpha", debug_ranking=True)
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    assert mid in debug_map
    notes = debug_map[mid].get("_ranking_notes", [])
    assert "long content penalty applied" in notes


def test_v061_t5_channel6_skipped_when_vec_disabled(tmp_path: Path) -> None:
    """T5: with the vec gate closed, no Channel 6 candidates appear."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    # Deliberately do NOT call _set_vec_ready — gate stays closed.
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="gate")["data"]["id"]
    _publish_two_sections(tools, mid, content, "alpha", "beta")

    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    for r in result["data"].get("_debug_ranking", []):
        assert not r.get("_section_vec_candidate"), (
            "Channel 6 must not fire when the vec gate is closed"
        )


def test_v061_t6_pool_cap_not_exceeded(tmp_path: Path) -> None:
    """T6: Channel 6 does not push the pool beyond pool_cap."""
    tools = _make_channel6_tools(tmp_path, pool_cap=4)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    for i in range(6):
        c = f"alpha {i} " + ("x" * 60) + "\nbeta " + ("y" * 60)
        mid = tools.memory_write(content=c, subject=f"cap-{i}")["data"]["id"]
        _publish_two_sections(tools, mid, c, "alpha", "beta")

    result = tools.memory_search(query="alpha", query_embedding=_keyword_embedding("alpha"))
    # The number of unique results never exceeds pool_cap.
    assert len(result["data"]["results"]) <= 4


def test_v061_t7_channel5_candidate_has_split_status(tmp_path: Path) -> None:
    """T7: vec_knn (Channel 5) now returns split_status (§2.1前置 checkpoint)."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)
    content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    mid = tools.memory_write(content=content, subject="t7")["data"]["id"]
    # memory_write doesn't auto-store a memory-level vec under the mock embedder
    # (embedding_provider != "gguf"), so store one explicitly.
    tools.memory_store_embedding(mid, _keyword_embedding("alpha"))
    _publish_two_sections(tools, mid, content, "alpha", "beta")

    rows = tools.db.vec_knn(_keyword_embedding("alpha"), k=5)
    target_row = next((r for r in rows if r["id"] == mid), None)
    assert target_row is not None, "target not in Channel 5 KNN results"
    assert "split_status" in target_row, "vec_knn must return split_status (§2.1)"
    assert target_row["split_status"] == "active"


def test_v080_long_content_zero_match_returns_full_memory(tmp_path: Path) -> None:
    """v0.8 §6.3: a long split-active doc under zero-match returns the FULL
    memory, not a bounded preview. Supersedes the v0.6.1 preview behaviour."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    tools.settings.section_zero_match_preview_chars = 2000  # legacy; ignored in v0.8
    _set_vec_ready(tools)

    # Two compliant sections (each ≤ max_section_chars=3600), total > preview.
    chunk_a = "alpha " + ("q" * 3000)
    chunk_b = "beta " + ("r" * 3000)
    big = chunk_a + "\n" + chunk_b
    mid = tools.memory_write(content=big, subject="bigdoc")["data"]["id"]
    published = _publish_two_sections(tools, mid, big, "alpha", "beta")
    assert published["ok"] is True

    result = tools.memory_search(query="zzz", query_embedding=_keyword_embedding("zzz"))
    hit = next((r for r in result["data"]["results"] if r["id"] == mid), None)
    assert hit is not None, "big doc must appear in zero-match results for the test to be meaningful"
    # v0.8: full memory returned, no truncation.
    assert hit["content_scope"] == "full_memory"
    assert hit["content"] == big
    assert len(hit["content"]) > 2000, "full text must exceed the legacy preview bound"
    assert "content_truncated" not in hit


def test_v061_t9_debug_ranking_channel6_fields(tmp_path: Path) -> None:
    """T9: a Channel 6 candidate carries section-vec debug fields + note. We feed
    a synthetic Channel 6 candidate through _soft_rerank directly — in a real
    end-to-end search the candidate may also be recalled by FTS first, masking
    the Channel 6 flag, so testing the scorer in isolation is more reliable."""
    from memory_arbiter.search import _soft_rerank

    candidate = {
        "id": 999,
        "subject": "t9",
        "tags": "[]",
        "content": "",               # Channel 6 omits content (A3)
        "split_status": "active",
        "status": "active",
        "source_type": "unknown",
        "protection_level": "normal",
        "ingest_time": "2020-01-01T00:00:00Z",
        "_vec_candidate": True,
        "_section_vec_candidate": True,
        "_section_vec_distance": 0.3,
        "_section_vec_section_id": 42,
    }
    reranked = _soft_rerank("anything", [candidate])
    rec = reranked[0]
    assert rec.get("_section_vec_candidate") is True
    assert rec.get("_section_vec_distance") == 0.3
    assert rec.get("_section_vec_section_id") == 42
    notes = rec.get("_ranking_notes", [])
    assert "section-vec recall candidate (Channel 6)" in notes


def test_v061_t10_penalty_baseline_c9(tmp_path: Path) -> None:
    """T10/C9 (blocking): non-split long memory incurs BOTH content_only and
    long-content penalties — the baseline T3/T4 exemption logic regresses against."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    long_content = "alpha " + ("x" * 3000)
    mid = tools.memory_write(content=long_content, subject=None, tags=None)["data"]["id"]
    result = tools.memory_search(query="alpha", debug_ranking=True)
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    assert mid in debug_map
    rec = debug_map[mid]
    notes = rec.get("_ranking_notes", [])
    assert "content_only_match" in str(rec.get("_match_reason", "")) or any(
        "matched content but not subject/tags" in n for n in notes
    )
    assert "long content penalty applied" in notes
    # relevance = 3.0 - 2.0 - 1.5 = -0.5; trust=0 (default source), recency ∈ [0, 0.8]
    assert -0.5 <= rec["_final_score"] < 0.5


def test_v061_t11_pool_saturation_skips_channel6_c2(tmp_path: Path) -> None:
    """T11/C2 (blocking): when Channels 1-5 fill the pool, Channel 6 is skipped."""
    tools = _make_channel6_tools(tmp_path, pool_cap=2)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # Two FTS-recallable memories fill the small pool (cap=2) before Channel 6.
    tools.memory_write(content="alpha match one " + "x" * 60, subject="fill-1")
    tools.memory_write(content="alpha match two " + "y" * 60, subject="fill-2")
    # A split-active memory that only Channel 6 could surface.
    target_content = "gamma " + ("x" * 60) + "\nbeta " + ("y" * 60)
    target_id = tools.memory_write(content=target_content, subject="late")["data"]["id"]
    _publish_two_sections(tools, target_id, target_content, "gamma", "beta")

    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    # Pool was saturated by FTS hits → Channel 6 never ran → no section-vec candidates.
    assert all(
        not r.get("_section_vec_candidate") for r in result["data"]["results"]
    ), "Channel 6 should be skipped when the pool is already full"


def test_v061_t12_content_only_penalty_still_applies_to_split_active(tmp_path: Path) -> None:
    """T12 (A5 regression): content_only_penalty still hits split-active, but
    long-content penalty is exempted. Score uses a range (not exact) per §6.2."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    long_content = "alpha " + ("x" * 3000) + "\nbeta " + ("y" * 60)
    mid = tools.memory_write(content=long_content, subject=None, tags=None)["data"]["id"]
    _publish_two_sections(tools, mid, long_content, "alpha", "beta")

    result = tools.memory_search(
        query="alpha", query_embedding=_keyword_embedding("alpha"), debug_ranking=True
    )
    debug_map = {r["id"]: r for r in result["data"]["results"]}
    assert mid in debug_map, "target memory must appear in results for the test to be meaningful"
    rec = debug_map[mid]
    notes = rec.get("_ranking_notes", [])
    # The core assertion: split-active exempts long-content penalty regardless
    # of which channel recalled the memory.
    assert "long content penalty applied" not in notes, (
        "split-active long doc should be exempt from long-content penalty"
    )
    # content_only_penalty still applies on the FTS path (A5: NOT exempted).
    # Only check this when the memory was NOT recalled by vec (vec floor would
    # mask the content_only signal).
    if not rec.get("_vec_candidate"):
        assert any("matched content but not subject/tags" in n for n in notes) or (
            rec.get("_match_reason") == "content_only_match"
        )
        # relevance = 3.0 - 2.0 = 1.0; with recency the final lands in [1.0, 2.0).
        assert 1.0 <= rec["_final_score"] < 2.0


def test_v061_t13_fulltext_branch_channel6_not_empty_content(tmp_path: Path) -> None:
    """T13 (blocking, second-round Bug regression): a Channel 6-only candidate
    entering the fulltext branch must NOT return empty content. §4.2 归一化 fixes
    this. We unit-test _attach_sections directly with a content='' candidate."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    full_text = "alpha " + ("x" * 60) + "\nbeta " + ("y" * 60)
    mid = tools.memory_write(content=full_text, subject="t13")["data"]["id"]
    _publish_two_sections(tools, mid, full_text, "alpha", "beta")
    # Lower fulltext threshold so ≥1 match counts as "most matched".
    tools.settings.section_fulltext_threshold = 0.0

    # Simulate a Channel 6 candidate: content="" but the memory is split-active.
    # _attach_sections must normalize content from current_mem_map.
    fake_candidate = {
        "id": mid,
        "content": "",          # Channel 6 deliberately omits content (A3)
        "split_status": "active",
        "_vec_candidate": True,
        "_section_vec_candidate": True,
        "subject": "t13",
    }
    normalized = tools._attach_sections(
        [fake_candidate], _keyword_embedding("alpha"), []
    )
    hit = normalized[0]
    # v0.8: fulltext branch returns the full memory for Channel 6 candidates
    # (content normalized from current_mem_map).
    assert hit.get("content_scope") == "full_memory"
    assert hit.get("content"), (
        "fulltext branch must return non-empty content for Channel 6 candidates "
        "(second-round Bug regression)"
    )


def test_v061_t14_partial_branch_does_not_leak_full_content(tmp_path: Path) -> None:
    """T14 (§4.2 interaction): a Channel 6 candidate in the partial branch must
    have content=None (normalization's full text is correctly discarded)."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    full_text = "alpha " + ("x" * 60) + "\nbeta " + ("y" * 60)
    mid = tools.memory_write(content=full_text, subject="t14")["data"]["id"]
    _publish_two_sections(tools, mid, full_text, "alpha", "beta")

    # Query "alpha" → section "alpha" matches (1/2 = partial, below fulltext 0.8).
    fake_candidate = {
        "id": mid,
        "content": "",
        "split_status": "active",
        "_vec_candidate": True,
        "_section_vec_candidate": True,
        "subject": "t14",
    }
    normalized = tools._attach_sections(
        [fake_candidate], _keyword_embedding("alpha"), []
    )
    hit = normalized[0]
    # v0.8: partial branch returns only the matched section's full text (joined),
    # NOT the full memory — the unmatched "beta" section body must not leak.
    assert hit.get("content_scope") == "matched_sections"
    assert hit.get("content")  # non-empty: the matched section's text
    # The matched alpha section's content is present...
    assert any(ms.get("content") for ms in hit.get("matched_sections", []))
    # ...but the unmatched beta section body must NOT appear in the partial content.
    beta_body = full_text.split("\nbeta ", 1)[-1] if "\nbeta " in full_text else "y" * 60
    assert beta_body not in hit["content"]


# ===========================================================================
# v0.6.3 provenance tests — section source attribution (parser vs agent).
# ===========================================================================


def test_v080_provenance_is_explicit_not_inferred(tmp_path: Path) -> None:
    """v0.8 (§6.2): provenance is an explicit caller argument, not inferred
    from whether an anchor happens to equal a Markdown heading. The
    memory_split Agent path is always 'agent' regardless of whether the
    caller reused the document's own heading text as a section title."""
    tools = make_vec_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # A document with real Markdown headings — but the caller goes through
    # memory_split (the Agent continuation path), so provenance='agent'.
    content = (
        "# alpha\n" + ("x" * 60) + "\n\n"
        "## beta\n" + ("y" * 60)
    )
    mid = tools.memory_write(content=content, subject="md-doc")["data"]["id"]
    mem = tools.db.get_memory(mid)
    published = tools.memory_split(
        memory_id=mid,
        split_decision="split",
        decision_content_hash=hashlib.sha256(mem["content"].encode("utf-8")).hexdigest(),
        decision_memory_version=mem["version"],
        decision_split_status=mem["split_status"],
        decision_split_revision=mem["split_revision"],
        sections=[
            {"title": "alpha"},
            {"title": "beta", "anchor_text": "## beta", "occurrence_index": 0},
        ],
    )
    assert published["ok"] is True

    sections = tools.db.get_sections_by_memory(mid)
    assert len(sections) == 2
    # Even though the anchors ARE Markdown headings, the Agent path records
    # provenance='agent' — the old heading-text heuristic is removed.
    assert sections[0]["provenance"] == "agent"
    assert sections[1]["provenance"] == "agent"


def test_v061_r1_channel6_recall_superseded_with_include_superseded(tmp_path: Path) -> None:
    """R1: include_superseded=True lets Channel 6 recall a superseded split-active
    memory. Channel 6's post-filter mirrors Channel 5: superseded rows are only
    skipped when 'superseded' is in like_status_clause (the default). With
    include_superseded=True they pass through."""
    tools = _make_channel6_tools(tmp_path)
    tools._embedder = _keyword_embedder()
    tools._embedder_loaded = True
    _set_vec_ready(tools)

    # A split-active memory (no memory-level vec → only Channel 6 can recall).
    target_content = "alpha " + ("x" * 60) + "\n" + "beta " + ("y" * 60)
    target_id = tools.memory_write(content=target_content, subject="stale")["data"]["id"]
    _publish_two_sections(tools, target_id, target_content, "alpha", "beta")

    # Supersede it (requires authorization).
    replacement = tools.memory_write(content="replacement active", subject="stale")
    tools.memory_supersede(
        memory_id=target_id, reason="replaced",
        superseded_by=replacement["data"]["id"], authorized=True,
    )

    # Default search excludes superseded → target should be absent.
    default_result = tools.memory_search(
        query="beta", query_embedding=_keyword_embedding("beta"), debug_ranking=True
    )
    default_ids = {r["id"] for r in default_result["data"]["results"]}
    assert target_id not in default_ids, "superseded memory leaked into default search"

    # include_superseded=True → Channel 6 should recall it via section vec.
    included_result = tools.memory_search(
        query="beta", query_embedding=_keyword_embedding("beta"),
        include_superseded=True, debug_ranking=True,
    )
    included_map = {r["id"]: r for r in included_result["data"]["results"]}
    assert target_id in included_map, (
        "Channel 6 should recall superseded split-active memory when "
        "include_superseded=True"
    )


# ---- v0.7.3 change 1: tag scoring unit tests (design §2.6 matrix) ------
# These exercise _score_tags_surface / _cjk_substring_match /
# _normalize_token_for_tag_match directly, not the full search pipeline.

from memory_arbiter.search import (
    _score_tags_surface,
    _cjk_substring_match,
    _normalize_token_for_tag_match,
    _is_pure_cjk_token,
    _TAGS_STRONG_WEIGHT,
    _TAGS_MEDIUM_WEIGHT,
    _TAGS_WEAK_WEIGHT,
    _TAGS_SCORE_CAP,
    SearchOutcome,
)


def _tags_score(query: str, tags: list[str]) -> tuple[float, str]:
    """Convenience wrapper returning (score, level)."""
    return _score_tags_surface(
        query, tags,
        _TAGS_STRONG_WEIGHT, _TAGS_MEDIUM_WEIGHT, _TAGS_WEAK_WEIGHT, _TAGS_SCORE_CAP,
    )[:2]


def test_tag_all_tokens_match_strong() -> None:
    # id=206 修复目标：query 两个 token 都是精确 tag → strong
    score, level = _tags_score("v0.7.2 发版", ["v0.7.2", "发版"])
    assert level == "strong"
    assert score == _TAGS_STRONG_WEIGHT


def test_tag_long_cjk_query() -> None:
    # 长 CJK query 单 token，tags 前缀+后缀都命中 → strong（修 v1 CJK bug）
    score, level = _tags_score("发版历史", ["发版", "历史"])
    assert level == "strong"
    assert score == _TAGS_STRONG_WEIGHT


def test_tag_mixed_long_query() -> None:
    score, level = _tags_score("v0.7.2 发版历史", ["v0.7.2", "发版", "历史"])
    assert level == "strong"
    assert score == _TAGS_STRONG_WEIGHT


def test_tag_half_match_medium() -> None:
    score, level = _tags_score("v0.7.2 发版", ["v0.7.2", "技术参考"])
    assert level == "medium"
    assert score == _TAGS_MEDIUM_WEIGHT


def test_tag_one_token_match_weak() -> None:
    # 3 个 query token，只命中 1 个 → ratio 1/3 < 0.5 → weak
    score, level = _tags_score("v0.7.2 发版 历史", ["发版", "其它"])
    assert level == "weak"
    assert score == _TAGS_WEAK_WEIGHT


def test_tag_no_match_none() -> None:
    score, level = _tags_score("v0.7.2 发版", ["doctor", "bug"])
    assert level == "none"
    assert score == 0.0


def test_tag_version_word_boundary() -> None:
    # query v0.7.2 vs tag v0.7.0 → ASCII equality 不命中（防伪召回）
    score, level = _tags_score("v0.7.2", ["v0.7.0"])
    assert level == "none"
    assert score == 0.0


def test_tag_version_normalization_bidirectional() -> None:
    # query 0.7.2 vs tag V0.7.2 → 双向归一化后都成 0.7.2 → 命中
    score, level = _tags_score("0.7.2", ["V0.7.2"])
    assert level == "strong"
    assert score == _TAGS_STRONG_WEIGHT


def test_tag_v_not_stripped_for_words() -> None:
    # query vue 不剥 v（不跟数字）→ 不匹配 tag ue
    score, level = _tags_score("vue", ["ue"])
    assert level == "none"


def test_tag_cjk_prefix_substring() -> None:
    # tag 发版 是 query token 发版历史 的前缀 → 命中
    score, level = _tags_score("发版历史", ["发版"])
    assert level == "strong"


def test_tag_cjk_suffix_substring() -> None:
    # tag 历史 是 query token 发版历史 的后缀 → 命中
    score, level = _tags_score("发版历史", ["历史"])
    assert level == "strong"


def test_tag_cjk_middle_substring_excluded() -> None:
    # tag 版历 是 query token 发版历史 的中间子串 → 不命中（review_2 漏洞 1）
    score, level = _tags_score("发版历史", ["版历"])
    assert level == "none"


def test_tag_ascii_no_substring() -> None:
    # tag memory vs query memory-arbiter → ASCII 不 substring → none
    score, level = _tags_score("memory-arbiter", ["memory"])
    assert level == "none"


def test_tag_empty_tags_list() -> None:
    score, level = _tags_score("v0.7.2 发版", [])
    assert level == "none"
    assert score == 0.0


def test_tag_empty_query() -> None:
    score, level = _tags_score("", ["v0.7.2", "发版"])
    assert level == "none"
    assert score == 0.0


def test_tag_subject_unchanged() -> None:
    # subject 仍走原 _score_surface：整串 substring 命中仍判 strong，不受 tag 改动影响
    from memory_arbiter.search import _score_surface, extract_anchors
    q = "v0.7.2 发版"
    subject = "v0.7.2 发版记录"
    score, level = _score_surface(
        extract_anchors(q), subject,
        10.0, 6.0, 2.0, 10.0, q.lower(),
    )
    assert level == "strong"


def test_tag_mixed_ascii_cjk_no_space_none() -> None:
    # 第五轮 S1 / 第八轮 E2：无空格混合 token 走 equality → 不命中（已知盲区）
    score, level = _tags_score("v0.7.2发版", ["v0.7.2", "发版"])
    assert level == "none"


def test_tag_mixed_with_space_strong() -> None:
    # 对照：有空格的混合 query → strong（推荐写法）
    score, level = _tags_score("v0.7.2 发版", ["v0.7.2", "发版"])
    assert level == "strong"


def test_tag_debug_fields_populated() -> None:
    # debug dict 字段齐全（用于 _soft_rerank 写 _tag_query_tokens 等）
    from memory_arbiter.search import _score_tags_surface
    _, _, debug = _score_tags_surface(
        "v0.7.2 发版", ["v0.7.2", "发版"],
        _TAGS_STRONG_WEIGHT, _TAGS_MEDIUM_WEIGHT, _TAGS_WEAK_WEIGHT, _TAGS_SCORE_CAP,
    )
    assert debug == {"total": 2, "matched": 2, "ratio": 1.0}


# ---- _is_pure_cjk_token / _cjk_substring_match / _normalize direct -------
# 设计 §2.3 E2 明确：_is_pure_cjk_token 不能用 token.isascii() 反向判定，
# 否则混合 token "0.7.2发版"（含 ASCII 数字）会被归入 CJK 类走 substring。
# 这些单元测试钉死判定函数的行为契约。


def test_is_pure_cjk_token_contract() -> None:
    assert _is_pure_cjk_token("发版") is True
    assert _is_pure_cjk_token("发版历史") is True
    assert _is_pure_cjk_token("v0.7.2") is False   # 纯 ASCII
    assert _is_pure_cjk_token("0.7.2发版") is False  # 混合（含 ASCII 数字）—— 关键
    assert _is_pure_cjk_token("memory") is False
    assert _is_pure_cjk_token("") is True           # 无 ASCII alnum → 视为 pure（空 query 已在调用方拦截）


def test_normalize_token_for_tag_match_contract() -> None:
    assert _normalize_token_for_tag_match("v0.7.2") == "0.7.2"
    assert _normalize_token_for_tag_match("0.7.2") == "0.7.2"
    assert _normalize_token_for_tag_match("V0.7.2") == "0.7.2"
    assert _normalize_token_for_tag_match("vue") == "vue"           # v 不跟数字，不剥
    assert _normalize_token_for_tag_match("  Abc  ") == "abc"        # strip + lower
    assert _normalize_token_for_tag_match("发版") == "发版"


def test_cjk_substring_match_contract() -> None:
    assert _cjk_substring_match("发版", "发版历史") is True   # prefix
    assert _cjk_substring_match("历史", "发版历史") is True   # suffix
    assert _cjk_substring_match("发版历史", "发版历史") is True  # equal
    assert _cjk_substring_match("版历", "发版历史") is False  # middle（bigram 伪 tag）
    assert _cjk_substring_match("发", "发版") is False       # 单字 tag 长度门槛
    # _cjk_substring_match 本身是纯字符串 prefix/suffix 判定，ASCII 串按同样的规则：
    assert _cjk_substring_match("xyz", "abcxyz") is True    # suffix 命中（调用方 _is_pure_cjk_token 保证 ASCII 串不进这条路径）
    assert _cjk_substring_match("abc", "abcxyz") is True    # prefix 命中
    assert _cjk_substring_match("bcxy", "abcxyz") is False  # middle 不命中


# ---- v0.7.3 commit 5: subject classify_match_level coverage threshold ----
# 守护 anchors.classify_match_level 的 specific_coverage 阈值（0.4→0.6，
# id=210/id=211 dogfooding 真根因）。这是 commit 5 三大修正之一，但此前
# anchors.py 没有任何测试 import，改阈值不会触发测试失败。这一组测试
# 用直接构造的 Anchor/AnchorMatch 钉死阈值数值，防止手滑改回 0.4/0.5
# 让 id=105 场景（subject 偶然含一字）静默升回 medium(6.0)，重新挤掉
# tag 双命中的 id=206。
#
# 合成数据证据见 scripts/tune_tag_weights.py（n=2000×5 seed）：
#   coverage 0.5 无效（A>B=0.5），0.6 是临界点（A>B=1.000），0.7+ 无额外收益。

from memory_arbiter.anchors import (
    Anchor as _Anchor,
    AnchorMatch as _AnchorMatch,
    classify_match_level as _classify_match_level,
)


def _classify(
    specific_hits: int, generic_hits: int, query_specific_count: int
) -> str:
    """直接构造 matches dict + query_anchors，绕开 extract_anchors 的 bigram
    干扰，精确控制 specific_coverage = specific_hits / query_specific_count。

    query_specific_count = query 里非 generic 的 anchor 数（分母）。
    构造的 query_anchors 全部 is_generic=False，所以 query_specific_count
    等于传入值；total_hits 由 specific+generic 决定（走 summary）。
    """
    query_anchors = [_Anchor(text=f"q{i}", is_generic=False)
                     for i in range(query_specific_count)]
    matches = {
        "_summary": _AnchorMatch(
            hit=True, kind="summary",
            specific_hits=specific_hits,
            generic_hits=generic_hits,
            total_hits=specific_hits + generic_hits,
        ),
    }
    return _classify_match_level(query_anchors, matches)


def test_classify_coverage_half_is_weak_not_medium() -> None:
    # id=210 dogfooding 核心 bug：subject 偶然含 query 一半 anchor
    # （specific=1, query_specific=2 → coverage=0.500）。
    # 0.6 阈值下落 weak(2.0)，旧 0.4 阈值会误升 medium(6.0)。
    # 改回 0.4/0.5 这个断言会失败 —— 这就是回归守门员。
    assert _classify(specific_hits=1, generic_hits=0, query_specific_count=2) == "weak"


def test_classify_coverage_full_is_medium() -> None:
    # 对照：query 两个 specific anchor 全命中（coverage=1.000）→ medium。
    # 这是 id=206 真正讲主题时该拿的 level。
    assert _classify(specific_hits=2, generic_hits=0, query_specific_count=2) == "medium"


def test_classify_coverage_threshold_boundary_0_6() -> None:
    # 阈值数值本身：3/5 = 0.600 刚好 >= 0.6 → medium。
    # 构造必须让第一条 medium 规则（specific>=1 AND total>=2）不触发，
    # 才能真正走到 coverage 判断：这里 specific=3 但 total 也=3 会先命中
    # 第一条规则——所以用 specific=3, total=1 是不可能的（total>=specific）。
    # 改用 1/2=0.5（守 weak）+ 2/2=1.0（守 medium）这对边界，见上下两条。
    # 本条留作"第一规则优先于 coverage"的文档性断言：3/5 走 medium 是因为
    # specific>=1 AND total>=2，不是因为 coverage。
    assert _classify(specific_hits=3, generic_hits=0, query_specific_count=5) == "medium"


def test_classify_coverage_just_below_threshold_is_weak() -> None:
    # 守"低于 0.6 阈值（且不触发第一规则）必须落 weak"。
    # 构造 specific=1, total=1（避开第一规则 specific>=1 AND total>=2），
    # query_specific=3 → coverage=0.333 < 0.6 → weak。
    # 这条钉死 coverage 规则的阈值：若有人改回 0.4 阈值，0.333 仍是 weak
    # （因为 0.333 < 0.4），所以它守的是"阈值不能低于 0.333"；真正守"0.5
    # 边界"的是上面那条 coverage_half 测试。
    assert _classify(specific_hits=1, generic_hits=0, query_specific_count=3) == "weak"


def test_classify_medium_via_specific_plus_total_rule() -> None:
    # medium 的第一条规则：specific_hits>=1 AND total_hits>=2（与 coverage 无关）。
    # 1 specific + 1 generic = total 2，query_specific=3 → coverage 0.333 < 0.6，
    # 但靠 specific+total 规则仍升 medium。
    assert _classify(specific_hits=1, generic_hits=1, query_specific_count=3) == "medium"


def test_classify_only_generic_is_weak() -> None:
    # 只有 generic 命中、specific=0：coverage=0，不满足 medium 两条规则，
    # 但 total>=1 → weak（不是 none）。
    assert _classify(specific_hits=0, generic_hits=2, query_specific_count=3) == "weak"


def test_classify_no_hits_is_none() -> None:
    # 一个都没命中 → none。
    assert _classify(specific_hits=0, generic_hits=0, query_specific_count=3) == "none"


def test_classify_single_specific_anchor_hit_is_medium() -> None:
    # coverage 规则的独占触发区：query 只有 1 个 specific anchor（如裸 query
    # "发版"），它命中时 specific=1 total=1，第一规则（total>=2）不满足，
    # 靠 coverage=1.0>=0.6 升 medium。
    # 这条守"coverage 规则不能被废掉"：把阈值改到 >1.0（如 2.0）会让
    # 单 token query 永远拿不到 medium，subject 命中只剩 weak(2.0)。
    # 上一轮把阈值改 2.0 跑全量 184 全绿，就是漏了这个场景。
    assert _classify(specific_hits=1, generic_hits=0, query_specific_count=1) == "medium"


def test_classify_id105_regression_via_real_pipeline() -> None:
    # 端到端回归：用真实 extract_anchors 复现 id=105 bug 场景。
    # query "v0.7.2 发版" 的两个 specific anchor 中，subject 只命中"发版"
    # （v0.4.0 ≠ v0.7.2）→ coverage=0.500 → 必须落 weak，不能 medium。
    # 这条覆盖"extract_anchors → score_anchor_overlap → classify_match_level"
    # 完整链路，守 subject 路径在真实 bigram 切分下的阈值行为。
    from memory_arbiter.anchors import (
        extract_anchors, score_anchor_overlap, classify_match_level,
    )
    query = "v0.7.2 发版"
    subject_id105 = "[已完成] README v0.4.0 发版"  # 含"发版"但不含 v0.7.2
    qa = extract_anchors(query)
    sa = extract_anchors(subject_id105)
    matches = score_anchor_overlap(qa, sa)
    assert classify_match_level(qa, matches) == "weak"
    # 同时验证 subject 语义全命中时能正确升 medium（id=206 的 subject 路径）
    subject_full = "v0.7.2 发版记录"
    matches_full = score_anchor_overlap(qa, extract_anchors(subject_full))
    assert classify_match_level(qa, matches_full) == "medium"


# ---- v0.7.3 change 2: search enhancement (tags_filter / time /
# source_type / has_more / 4-tuple / shortcut) end-to-end tests ---------
#
# 这些测试走 MemoryTools.memory_search 端到端（真实 sqlite + 真实 search_memories），
# 覆盖设计 §5.2 测试矩阵的核心场景。每次测试用一个全新的 tmp_path 库，写入已知
# 数据，断言行为。

import datetime as _dt


def _write_mem(tools: MemoryTools, *, content: str, subject: str, tags: list[str],
               source_type: str = "agent_generated", ingest_time: Optional[str] = None,
               workspace: str = "ws") -> int:
    """Helper: write one memory via tools.memory_write, return its id."""
    payload = {
        "content": content, "subject": subject, "tags": tags,
        "source_type": source_type, "workspace": workspace,
        "agent_id": "tester",
    }
    if ingest_time is not None:
        payload["ingest_time"] = ingest_time
    res = tools.memory_write(**payload)
    assert res["ok"], f"write failed: {res}"
    return res["data"]["id"]


def test_tags_filter_exact_match(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="v0.7.2 发版记录", subject="发版", tags=["v0.7.2", "发版"])
    _write_mem(tools, content="其他无关", subject="其他", tags=["其它"])
    res = tools.memory_search(query="发版", tags_filter=["发版"])
    ids = [r["id"] for r in res["data"]["results"]]
    assert len(ids) == 1, f"tags_filter should exact-match only the 发版 tag, got {ids}"


def test_tags_filter_no_substring_false_positive(tmp_path: Path) -> None:
    # tags_filter=["v0.7"] 不应该命中 tags=["v0.7.0"]（精确匹配，防 LIKE 误命中）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=["v0.7.0"])
    res = tools.memory_search(query="v0.7.0", tags_filter=["v0.7"])
    assert res["data"]["results"] == [], "tags_filter must not substring-match"


def test_tags_filter_and_semantics(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="both", subject="s", tags=["发版", "v0.7.2"])
    _write_mem(tools, content="only one", subject="s2", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版", "v0.7.2"])
    ids = [r["id"] for r in res["data"]["results"]]
    assert len(ids) == 1, f"AND semantics: only the memory with both tags, got {ids}"


def test_tags_filter_empty_result_no_fallback(tmp_path: Path) -> None:
    # 匹配不到时不走 fallback（fallback 会返回不符合过滤条件的记忆）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="recent1", subject="r1", tags=["other"])
    res = tools.memory_search(query="发版", tags_filter=["发版"])
    assert res["data"]["results"] == []
    # 不应有 fallback warning（fallback 才会附加 "No direct memory match"）
    assert not any("No direct memory match" in w for w in res["warnings"])


def test_tags_filter_empty_list_treated_as_none(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="c", subject="s", tags=["发版"])
    # tags_filter=[] 等同不传 → 不过滤 → 命中（走 fallback 或正常召回）
    res = tools.memory_search(query="不存在的词", tags_filter=[])
    # 空 query 路径才走 fallback；这里 query 非空无匹配 → fallback
    assert "count" in res["data"]


def test_tags_filter_duplicates_deduped(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="c", subject="s", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版", "发版"])
    assert len(res["data"]["results"]) == 1


def test_tags_filter_empty_string_ignored(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="c", subject="s", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版", ""])
    assert len(res["data"]["results"]) == 1


def test_after_time_filter(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="old", subject="old", tags=["t"], ingest_time="2026-01-15T00:00:00+00:00")
    _write_mem(tools, content="new", subject="new", tags=["t"], ingest_time="2026-07-15T00:00:00+00:00")
    res = tools.memory_search(query="t", after_time="2026-06-01")
    subjects = [r["subject"] for r in res["data"]["results"]]
    assert "new" in subjects and "old" not in subjects


def test_after_time_with_timezone(tmp_path: Path) -> None:
    # after_time=2026-06-01T00:00:00+08:00 == 2026-05-31T16:00:00 UTC
    tools = make_tools(tmp_path)
    _write_mem(tools, content="before", subject="before", tags=["t"], ingest_time="2026-05-31T15:00:00+00:00")
    _write_mem(tools, content="after", subject="after", tags=["t"], ingest_time="2026-05-31T17:00:00+00:00")
    res = tools.memory_search(query="t", after_time="2026-06-01T00:00:00+08:00")
    subjects = [r["subject"] for r in res["data"]["results"]]
    assert "after" in subjects and "before" not in subjects


def test_after_time_invalid_format(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=["t"])
    res = tools.memory_search(query="t", after_time="xyz")
    assert any("invalid ISO 8601" in w for w in res["warnings"])
    # 无效 after_time 被忽略 → 正常返回
    assert res["ok"]


def test_before_time_filter(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="old", subject="old", tags=["t"], ingest_time="2026-01-15T00:00:00+00:00")
    _write_mem(tools, content="new", subject="new", tags=["t"], ingest_time="2026-07-15T00:00:00+00:00")
    res = tools.memory_search(query="t", before_time="2026-06-01")
    subjects = [r["subject"] for r in res["data"]["results"]]
    assert "old" in subjects and "new" not in subjects


def test_filters_disable_fallback(tmp_path: Path) -> None:
    # 有过滤但 pool 空 → 返回空 + 精准 warning，不走 recent_fallback
    tools = make_tools(tmp_path)
    _write_mem(tools, content="recent", subject="recent", tags=["other"])
    res = tools.memory_search(query="发版", tags_filter=["发版"])
    assert res["data"]["results"] == []
    assert not any("No direct memory match" in w for w in res["warnings"])


def test_source_type_filter(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="uc", subject="uc", tags=["t"], source_type="user_confirmed")
    _write_mem(tools, content="ag", subject="ag", tags=["t"], source_type="agent_generated")
    res = tools.memory_search(query="t", source_type="user_confirmed")
    subjects = [r["subject"] for r in res["data"]["results"]]
    assert subjects == ["uc"]


def test_has_more_when_more_exist(tmp_path: Path) -> None:
    # 库里 > limit 条匹配 tags_filter → has_more=True
    tools = make_tools(tmp_path)
    for i in range(15):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    assert res["data"]["has_more"] is True
    assert res["data"]["total_estimate"] == 15
    assert len(res["data"]["results"]) == 10


def test_has_more_false_when_exact(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for i in range(10):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    assert res["data"]["has_more"] is False
    assert res["data"]["total_estimate"] == 10


def test_has_more_false_when_fewer(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    for i in range(7):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    assert res["data"]["has_more"] is False
    assert res["data"]["total_estimate"] == 7


def test_has_more_false_when_query_matches_few_no_filter(tmp_path: Path) -> None:
    # E1：无过滤场景，query 只匹配少数，全库更大 → has_more=False（修 count_active 误报）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="alpha beta", subject="alpha beta", tags=[])
    _write_mem(tools, content="alpha beta", subject="alpha beta", tags=[])
    _write_mem(tools, content="alpha beta", subject="alpha beta", tags=[])
    # 写一堆不匹配 query 的记忆
    for i in range(20):
        _write_mem(tools, content=f"noise{i}", subject=f"noise{i}", tags=[])
    res = tools.memory_search(query="alpha beta", limit=10)
    assert res["data"]["has_more"] is False, (
        f"E1: 无过滤场景 total_estimate 应=len(pool)=3，不是全库 23；got has_more={res['data']['has_more']}, total={res['data']['total_estimate']}"
    )
    assert res["data"]["total_estimate"] == 3


def test_no_filters_backward_compat(tmp_path: Path) -> None:
    # 不传任何新参数 → 完全同 v0.7.2（含 fallback 行为）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="hello world", subject="hello", tags=[])
    res = tools.memory_search(query="hello")
    assert res["ok"]
    assert len(res["data"]["results"]) >= 1
    # 返回结构应含新字段 has_more/total_estimate（即使不用过滤）
    assert "has_more" in res["data"]
    assert "total_estimate" in res["data"]


def test_search_returns_search_outcome_at_search_memories_level(tmp_path: Path) -> None:
    # v0.7.4 (M2): search_memories now returns a SearchOutcome dataclass, not a tuple.
    from memory_arbiter.search import search_memories, SearchOutcome
    tools = make_tools(tmp_path)
    _write_mem(tools, content="hello", subject="hello", tags=[])
    result = search_memories(tools.db, "hello")
    assert isinstance(result, SearchOutcome), f"search_memories must return SearchOutcome, got {type(result)}"
    assert isinstance(result.results, list)
    assert isinstance(result.warnings, list)
    assert len(result.results) >= 1
    assert result.retrieval_mode == "direct"
    assert isinstance(result.has_more, bool)
    assert isinstance(result.total_estimate, int)


def test_tools_layer_exposes_has_more(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=[])
    res = tools.memory_search(query="x")
    assert "has_more" in res["data"]
    assert "total_estimate" in res["data"]


def test_invalid_after_time_falls_back_to_none(tmp_path: Path) -> None:
    # after_time 无效 → warning + 视为 None；如果同时有其他 filter，has_filters 仍 True
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=["发版"])
    res = tools.memory_search(query="发版", after_time="not-a-date", tags_filter=["发版"])
    assert any("invalid ISO 8601" in w for w in res["warnings"])
    # tags_filter 仍生效
    assert len(res["data"]["results"]) == 1


def test_after_gt_before_warns(tmp_path: Path) -> None:
    # D4：after > before 矛盾 → warning + 两者都忽略
    tools = make_tools(tmp_path)
    _write_mem(tools, content="x", subject="x", tags=["t"], ingest_time="2026-06-15T00:00:00+00:00")
    res = tools.memory_search(query="t", after_time="2026-07-01", before_time="2026-06-01")
    assert any("after_time" in w and "before_time" in w and "empty" in w for w in res["warnings"]), (
        f"D4: after>before should warn; got {res['warnings']}"
    )


def test_empty_query_with_tags_filter_returns_empty(tmp_path: Path) -> None:
    # K2/C1/D3：空 query + tags_filter → 不走短路，post-filter 后空，精准 warning
    tools = make_tools(tmp_path)
    for i in range(5):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="", tags_filter=["发版"])
    assert res["data"]["results"] == [], "空 query + tags_filter 本版不独立召回（K2/C1）"
    # 不应返回未过滤的 fallback 记忆（C1 短路改造的验证）
    assert not any("No direct memory match" in w for w in res["warnings"])
    # 应有精准 warning（D3）
    assert any("query required" in w or "filters too restrictive" in w for w in res["warnings"])


def test_empty_query_no_filters_goes_fallback(tmp_path: Path) -> None:
    # 短路改造后，空 query + 无过滤仍应走 fallback（保留 v0.7.2 行为）
    tools = make_tools(tmp_path)
    _write_mem(tools, content="r1", subject="r1", tags=[])
    res = tools.memory_search(query="")
    # fallback 路径：返回 recent memories + fallback warning
    assert any("No direct memory match" in w for w in res["warnings"]) or len(res["data"]["results"]) > 0


def test_bm25_mode_warns_on_filter_params(tmp_path: Path, monkeypatch) -> None:
    # D2：bm25 模式 + 过滤参数 → warning 提示过滤被忽略
    tools = make_tools(tmp_path)
    _write_mem(tools, content="hello", subject="hello", tags=["t"])
    monkeypatch.setenv("MEMORY_ARBITER_RANKING_MODE", "bm25")
    try:
        res = tools.memory_search(query="hello", tags_filter=["t"])
        assert any("bm25 mode ignores" in w for w in res["warnings"]), (
            f"D2: bm25 + tags_filter should warn; got {res['warnings']}"
        )
        assert res["data"]["has_more"] is False  # bm25 写死
        assert res["data"]["total_estimate"] == 0
    finally:
        monkeypatch.delenv("MEMORY_ARBITER_RANKING_MODE", raising=False)


def test_passes_filters_unit() -> None:
    # B3：_passes_filters 直接单元测试
    from memory_arbiter.search import _passes_filters
    from datetime import datetime, timezone

    def mk(ingest_time: Optional[str] = "2026-06-15T00:00:00+00:00", tags: list = None, source_type: str = "agent_generated"):
        rec = {"tags": json.dumps(tags or []), "ingest_time": ingest_time, "source_type": source_type}
        return rec

    after = datetime(2026, 6, 1, tzinfo=timezone.utc)
    before = datetime(2026, 7, 1, tzinfo=timezone.utc)

    # tags AND
    assert _passes_filters(mk(tags=["发版", "v0.7.2"]), ["发版", "v0.7.2"], None, None, None) is True
    assert _passes_filters(mk(tags=["发版"]), ["发版", "v0.7.2"], None, None, None) is False

    # time bounds
    assert _passes_filters(mk(ingest_time="2026-06-15T00:00:00+00:00"), None, after, before, None) is True
    assert _passes_filters(mk(ingest_time="2026-05-15T00:00:00+00:00"), None, after, None, None) is False
    assert _passes_filters(mk(ingest_time="2026-08-15T00:00:00+00:00"), None, None, before, None) is False

    # time 无效 → 过滤掉
    assert _passes_filters(mk(ingest_time="not-a-date"), None, after, None, None) is False

    # source_type 等值
    assert _passes_filters(mk(source_type="user_confirmed"), None, None, None, "user_confirmed") is True
    assert _passes_filters(mk(source_type="agent_generated"), None, None, None, "user_confirmed") is False

    # JSON parse 失败 → 空集 → 不命中
    rec_bad = {"tags": "{bad json", "ingest_time": "2026-06-15T00:00:00+00:00", "source_type": "x"}
    assert _passes_filters(rec_bad, ["any"], None, None, None) is False


def test_pool_cap_truncates_results(tmp_path: Path) -> None:
    # B1/T1：库里很多匹配 tags_filter，pool_cap 截断后 reranked ≤ limit，has_more=True，total=全库匹配数
    tools = make_tools(tmp_path)
    for i in range(60):  # 超过 pool_cap=50
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    assert len(res["data"]["results"]) <= 10
    assert res["data"]["has_more"] is True
    # total_estimate 走 count_filtered（SQL 全表），=60
    assert res["data"]["total_estimate"] == 60


def test_count_matches_post_filter(tmp_path: Path) -> None:
    # T2：count_filtered_memories 返回值 == Python post-filter 后、切片前的 pool 长度
    tools = make_tools(tmp_path)
    for i in range(8):
        _write_mem(tools, content=f"c{i}", subject=f"s{i}", tags=["发版"])
    # 用 tags_filter 让 has_filters=True，count_filtered 会算全表匹配数
    res = tools.memory_search(query="发版", tags_filter=["发版"], limit=10)
    # 8 条全匹配，pool 召回 8，reranked=8，total=8
    assert res["data"]["total_estimate"] == 8
    assert len(res["data"]["results"]) == 8
    assert res["data"]["has_more"] is False
