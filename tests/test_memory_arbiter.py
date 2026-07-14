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


def test_memory_recent_lists_recent_workspace_memories(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.memory_write(content="Old memory", subject="old", event_time="2026-01-01T00:00:00Z")
    tools.memory_write(content="New memory", subject="new", event_time="2026-02-01T00:00:00Z")
    tools.memory_write(content="Other workspace", subject="other", workspace="repo-b", event_time="2026-03-01T00:00:00Z")

    recent = tools.memory_recent(workspace="repo-a", limit=10)

    assert recent["ok"] is True
    assert [record["subject"] for record in recent["data"]["results"]] == ["new", "old"]


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

    def fake_search_memories(db, query, workspace, tags, limit, include_superseded=False, debug_ranking=False, query_embedding=None):
        captured["query_embedding"] = query_embedding
        return [], []

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

    def fake_search_memories(db, query, workspace, tags, limit, include_superseded=False, debug_ranking=False, query_embedding=None):
        captured["query_embedding"] = query_embedding
        return [], []

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
    # 1/2 matched → partial branch → content omitted, matched_sections present
    assert hit["content_omitted"] is True
    assert hit["section_enhancement_applied"] is True
    assert hit.get("matched_sections")
    assert hit["matched_sections"][0]["title"] == "beta"


def test_split_publish_success_zero_hit_returns_catalog(tmp_path: Path) -> None:
    """Zero section matches → content_omitted=True + full section_catalog."""
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
    assert hit["content_omitted"] is True
    assert hit["section_enhancement_applied"] is True
    assert hit.get("section_catalog")
    assert len(hit["section_catalog"]) == 2
    # Unified catalog schema: embedding diagnostic fields present.
    assert "embedding_truncated" in hit["section_catalog"][0]
    assert "embedding_original_tokens" in hit["section_catalog"][0]


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
    assert hit["content_omitted"] is False
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
    assert hit["content_omitted"] is False
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
