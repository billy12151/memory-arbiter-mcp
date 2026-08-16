import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from unittest.mock import patch

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools


def _settings(path: Path, **overrides) -> Settings:
    return Settings(db_path=path / "m.sqlite3", backup_jsonl=path / "m.jsonl", **overrides)


def _tools(path: Path, **overrides) -> MemoryTools:
    settings = _settings(path, **overrides)
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _notice(
    tools: MemoryTools, *, severity: str = "normal", key: str = "notice", workspace: str | None = None,
) -> int:
    left = tools.memory_write(content=f"left {key}", subject=f"secret {key}", workspace=workspace)["data"]
    right = tools.memory_write(content=f"right {key}", subject=f"secret {key}", workspace=workspace)["data"]
    left_row = tools.db.get_memory(left["id"])
    right_row = tools.db.get_memory(right["id"])
    assert left_row and right_row
    result = tools.db.record_semantic_notice(
        memory_id=left["id"], peer_id=right["id"], severity=severity,
        notice_type="semantic_pair", title=f"private title {key}",
        message=f"private message {key}", payload={"private": key},
        dedupe_key=key, left_version=left_row["version"],
        right_version=right_row["version"],
        left_claim_revision=left_row["claim_revision"],
        right_claim_revision=right_row["claim_revision"],
    )
    return result["notice_id"]


def _semantic_stubs(response: dict) -> list[dict]:
    return [n for n in response.get("notices", []) if n.get("action_required") == "read_semantic_notice"]


@pytest.mark.parametrize(
    ("surface", "kwargs"),
    [
        ("memory", {"action": "help"}),
        ("memory_review", {"view": "overview"}),
        ("memory_govern", {"action": "help"}),
        ("memory_repair", {"task": "help"}),
    ],
)
def test_each_product_surface_delivers_one_private_stub(tmp_path: Path, surface: str, kwargs: dict) -> None:
    tools = _tools(tmp_path)
    notice_id = _notice(tools, key=surface)

    response = getattr(tools, surface)(**kwargs)

    stubs = _semantic_stubs(response)
    assert len(stubs) == 1
    assert stubs[0]["notice_id"] == notice_id
    assert set(stubs[0]) == {"notice_id", "severity", "type", "action_required", "agent_instruction", "read_call"}
    assert "Do not claim this notice is a confirmed conflict" in stubs[0]["agent_instruction"]
    serialized = repr(stubs[0])
    for secret in ("private title", "private message", "private", "secret", "memory_id", "peer_id", "subject"):
        assert secret not in serialized


def test_overview_nested_responses_do_not_consume_extra_notice(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    first = _notice(tools, severity="high", key="first")
    second = _notice(tools, severity="normal", key="second")

    response = tools.memory_review(view="overview")

    assert [n["notice_id"] for n in _semantic_stubs(response)] == [first]
    open_notices = tools.db.list_semantic_notices()
    delivered = {row["id"]: row["delivered_at"] for row in open_notices}
    assert delivered[first] is not None
    assert delivered[second] is None


def test_failed_product_response_does_not_consume_notice(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    notice_id = _notice(tools, key="failure")

    response = tools.memory(action="unknown")

    assert response["ok"] is False
    assert tools.db.read_semantic_notice(notice_id)["delivered_at"] is None


def test_atomic_claim_is_single_delivery_across_instances(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    notice_id = _notice(tools, key="concurrent")
    other = MemoryDB(_settings(tmp_path))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda db: db.claim_next_semantic_notice(), (tools.db, other)))

    claimed = [result for result in results if result is not None]
    assert len(claimed) == 1
    assert claimed[0]["notice_id"] == notice_id


def test_stale_notice_is_closed_and_next_fresh_notice_is_claimed(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    stale_id = _notice(tools, severity="high", key="stale")
    fresh_id = _notice(tools, severity="normal", key="fresh")
    stale = tools.db.read_semantic_notice(stale_id)
    assert stale
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET version=version+1 WHERE id=?", (stale["memory_id"],))

    claimed = tools.db.claim_next_semantic_notice()

    assert claimed and claimed["notice_id"] == fresh_id
    assert tools.db.read_semantic_notice(stale_id)["status"] == "stale"


def test_notice_read_list_and_terminal_lifecycle(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    notice_id = _notice(tools, key="lifecycle")
    tools.memory(action="help")

    listed = tools.memory_repair(task="notice", data={"action": "list"})["data"]["notices"]
    row = next(item for item in listed if item["id"] == notice_id)
    assert row["status"] == "open"
    assert row["delivered_at"] is not None

    read = tools.memory_repair(task="notice", data={"action": "read", "notice_id": notice_id})
    assert read["ok"] is True
    notice = read["data"]["notice"]
    assert notice["freshness"]["fresh"] is True
    assert notice["left_read_call"] == {
        "tool": "memory", "action": "read",
        "data": {"memory_id": notice["memory_id"], "workspace": "default"},
    }
    assert notice["right_read_call"] == {
        "tool": "memory", "action": "read",
        "data": {"memory_id": notice["peer_id"], "workspace": "default"},
    }
    assert "Only after both reads succeed" in notice["agent_instruction"]
    for call in (notice["left_read_call"], notice["right_read_call"]):
        side = tools.memory(action=call["action"], data=call["data"])
        assert side["ok"] is True

    dismissed = tools.memory_repair(
        task="notice", data={"action": "dismiss", "notice_id": notice_id, "reason": "false positive"},
    )
    assert dismissed["ok"] is True
    assert dismissed["data"]["outcome"] == "updated"
    repeated = tools.memory_repair(task="notice", data={"action": "resolve", "notice_id": notice_id})
    assert repeated["ok"] is False
    assert repeated["data"] == {"outcome": "already_terminal", "status": "dismissed"}
    stored = tools.db.read_semantic_notice(notice_id)
    assert stored["status"] == "dismissed"
    assert stored["resolution_reason"] == "false positive"


def test_strict_workspace_notice_delivery_and_api_require_both_sides_visible(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    alpha = _notice(tools, key="alpha", workspace="alpha")
    beta = _notice(tools, key="beta", workspace="beta")
    mixed = _notice(tools, key="mixed", workspace="alpha")
    strict_settings = _settings(tmp_path, isolation="strict", workspace="alpha")
    tools = MemoryTools(settings=strict_settings, db=MemoryDB(strict_settings))
    mixed_row = tools.db.read_semantic_notice(mixed)
    assert mixed_row
    with tools.db.write_transaction() as conn:
        conn.execute(
            "UPDATE memories SET workspace='beta', workspace_canonical='beta' WHERE id=?",
            (mixed_row["peer_id"],),
        )

    delivered = tools.memory(action="status")
    assert [n["notice_id"] for n in _semantic_stubs(delivered)] == [alpha]
    assert tools.db.read_semantic_notice(beta)["delivered_at"] is None
    assert tools.db.read_semantic_notice(mixed)["delivered_at"] is None

    listed = tools.memory_repair(task="notice", data={"action": "list", "workspace": "alpha"})
    assert {n["id"] for n in listed["data"]["notices"]} == {alpha}
    for notice_id in (beta, mixed):
        assert tools.memory_repair(
            task="notice", data={"action": "read", "notice_id": notice_id, "workspace": "alpha"},
        )["ok"] is False
        assert tools.memory_repair(
            task="notice", data={"action": "dismiss", "notice_id": notice_id, "workspace": "alpha"},
        )["ok"] is False
        assert tools.db.read_semantic_notice(notice_id)["status"] == "open"

    beta_delivery = tools.memory(action="status", data={"workspace": "beta"})
    beta_stub = _semantic_stubs(beta_delivery)[0]
    assert beta_stub["notice_id"] == beta
    assert beta_stub["read_call"]["data"]["workspace"] == "beta"
    read_call = beta_stub["read_call"]
    read_via_stub = tools.memory_repair(
        task=read_call["task"], data=read_call["data"],
    )
    assert read_via_stub["ok"] is True
    assert read_via_stub["data"]["notice"]["id"] == beta

    beta_extra = _notice(tools, key="beta-extra", workspace="beta")

    alpha_status = tools.memory(action="status", data={"workspace": "alpha"})
    beta_overview = tools.memory_review(view="overview", data={"workspace": "beta"})
    alpha_control = tools.memory_repair(
        task="semantic_control", data={"action": "status", "workspace": "alpha"},
    )
    assert alpha_status["data"]["semantic_conflict"]["notices"] == {"open": 1}
    assert beta_overview["data"]["status"]["semantic_conflict"]["notices"] == {"open": 2}
    assert alpha_control["data"]["notices"] == {"open": 1}


def test_claim_stale_scan_is_bounded_and_returns_without_reaching_fresh(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    stale_ids = [_notice(tools, severity="high", key=f"stale-{i}") for i in range(25)]
    fresh_id = _notice(tools, severity="normal", key="after-bound")
    with tools.db.write_transaction() as conn:
        for notice_id in stale_ids:
            row = tools.db.read_semantic_notice(notice_id)
            assert row
            conn.execute("UPDATE memories SET version=version+1 WHERE id=?", (row["memory_id"],))

    assert tools.db.claim_next_semantic_notice() is None
    assert tools.db.read_semantic_notice(fresh_id)["delivered_at"] is None
    claimed = tools.db.claim_next_semantic_notice()
    assert claimed and claimed["notice_id"] == fresh_id


def test_notice_rejects_invalid_action_status_and_public_stale_update(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    notice_id = _notice(tools, key="invalid")

    invalid_action = tools.memory_repair(task="notice", data={"action": ["read"]})
    assert invalid_action["ok"] is False
    invalid_status = tools.memory_repair(task="notice", data={"action": "list", "status": "bogus"})
    assert invalid_status["ok"] is False
    assert tools.db.update_semantic_notice_status(notice_id, "stale")["outcome"] == "invalid_status"
    assert tools.db.read_semantic_notice(notice_id)["status"] == "open"


def test_empty_claim_uses_read_probe_without_writer(tmp_path: Path, monkeypatch) -> None:
    tools = _tools(tmp_path)

    def forbidden_writer():
        raise AssertionError("empty notice queue must not acquire a writer")

    monkeypatch.setattr(tools.db, "write_transaction", forbidden_writer)
    assert tools.db.claim_next_semantic_notice() is None


def test_open_undelivered_claim_plan_uses_partial_priority_index_after_history(
    tmp_path: Path,
) -> None:
    tools = _tools(tmp_path)
    notice_id = _notice(tools, severity="high", key="indexed-open")
    with tools.db.write_transaction() as conn:
        template = conn.execute(
            "SELECT severity, source, memory_id, peer_id, notice_type, title, message, payload, "
            "left_version, right_version, left_claim_revision, right_claim_revision "
            "FROM semantic_notices WHERE id=?",
            (notice_id,),
        ).fetchone()
        assert template is not None
        conn.executemany(
            "INSERT INTO semantic_notices(created_at,status,severity,source,memory_id,peer_id,"
            "notice_type,title,message,payload,left_version,right_version,left_claim_revision,"
            "right_claim_revision,delivered_at) VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    f"2025-01-{(i % 28) + 1:02d}T00:00:00+00:00",
                    template["severity"], template["source"], template["memory_id"],
                    template["peer_id"], template["notice_type"], template["title"],
                    template["message"], template["payload"], template["left_version"],
                    template["right_version"], template["left_claim_revision"],
                    template["right_claim_revision"], "2025-02-01T00:00:00+00:00",
                )
                for i in range(2000)
            ],
        )
    with tools.db.connection() as conn:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT id, severity, notice_type, memory_id, peer_id, "
            "left_version, right_version, left_claim_revision, right_claim_revision "
            "FROM semantic_notices INDEXED BY idx_semantic_notices_open_undelivered_priority "
            "WHERE status='open' AND delivered_at IS NULL "
            "ORDER BY CASE lower(severity) "
            "WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'warning' THEN 2 "
            "WHEN 'normal' THEN 3 WHEN 'info' THEN 4 ELSE 5 END, created_at, id LIMIT ?",
            (25,),
        ).fetchall()
    details = " ".join(str(row["detail"]) for row in plan)
    assert "idx_semantic_notices_open_undelivered_priority" in details
    assert "USE TEMP B-TREE" not in details


def test_claim_priority_then_oldest_order(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    normal = _notice(tools, severity="normal", key="priority-normal")
    older_high = _notice(tools, severity="high", key="priority-high-old")
    newer_high = _notice(tools, severity="HIGH", key="priority-high-new")
    critical = _notice(tools, severity="critical", key="priority-critical")
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE semantic_notices SET created_at='2026-01-01' WHERE id=?", (older_high,))
        conn.execute("UPDATE semantic_notices SET created_at='2026-01-02' WHERE id=?", (newer_high,))

    claimed = [tools.db.claim_next_semantic_notice()["notice_id"] for _ in range(4)]

    assert claimed == [critical, older_high, newer_high, normal]


def test_legacy_notice_schema_migrates_delivery_pins_and_reason(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL, agent_id TEXT NOT NULL,
          workspace TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', source_type TEXT NOT NULL,
          source_ref TEXT, event_time TEXT NOT NULL, ingest_time TEXT NOT NULL,
          confidence REAL NOT NULL DEFAULT 0.5, protection_level TEXT NOT NULL DEFAULT 'normal',
          status TEXT NOT NULL DEFAULT 'active', subject TEXT, metadata TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
          workspace_canonical TEXT, claim_revision INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE semantic_notices (
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'open', severity TEXT NOT NULL, source TEXT NOT NULL,
          memory_id INTEGER NOT NULL, peer_id INTEGER, conflict_id INTEGER,
          notice_type TEXT NOT NULL, title TEXT NOT NULL, message TEXT NOT NULL,
          payload TEXT NOT NULL DEFAULT '{}', dedupe_key TEXT,
          left_version INTEGER, right_version INTEGER,
          left_claim_revision INTEGER, right_claim_revision INTEGER,
          delivered_at TEXT, dismissed_at TEXT, resolved_at TEXT, reason TEXT
        );
        INSERT INTO memories(content,agent_id,workspace,tags,source_type,event_time,ingest_time,
          status,subject,metadata,created_at,workspace_canonical,version,claim_revision)
          VALUES ('left','a','ws','[]','agent_generated','2026-01-01','2026-01-01',
          'active','left','{}','2026-01-01','ws',1,1);
        INSERT INTO memories(content,agent_id,workspace,tags,source_type,event_time,ingest_time,
          status,subject,metadata,created_at,workspace_canonical,version,claim_revision)
          VALUES ('right','a','ws','[]','agent_generated','2026-01-01','2026-01-01',
          'active','right','{}','2026-01-01','ws',1,1);
        INSERT INTO semantic_notices(created_at,status,severity,source,memory_id,peer_id,
          notice_type,title,message,payload,dedupe_key,left_version,right_version,
          left_claim_revision,right_claim_revision,reason)
          VALUES ('2026-01-02','delivered','normal','legacy',1,2,'semantic_pair','t','m','{}',
          'delivered',1,1,1,1,'legacy reason');
        INSERT INTO semantic_notices(created_at,status,severity,source,memory_id,peer_id,
          notice_type,title,message,payload,dedupe_key,reason)
          VALUES ('2026-01-03','open','normal','legacy',1,2,'semantic_pair','t','m','{}',
          'unpinned','cannot verify');
        """
    )
    conn.commit()
    conn.close()

    db = MemoryDB(_settings(tmp_path))
    delivered = db.read_semantic_notice(1)
    unpinned = db.read_semantic_notice(2)
    assert delivered and delivered["status"] == "open"
    assert delivered["delivered_at"] == "2026-01-02"
    assert delivered["resolution_reason"] == "legacy reason"
    assert unpinned and unpinned["status"] == "stale"
    assert unpinned["resolution_reason"] == "cannot verify"
    with db.connection() as migrated:
        index = migrated.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND name='idx_semantic_notices_open_undelivered_priority'"
        ).fetchone()
    assert index is not None
    assert "WHERE status='open' AND delivered_at IS NULL" in str(index["sql"])


def test_confirm_pending_workspace_does_not_embed(tmp_path: Path, monkeypatch) -> None:
    tools = _tools(tmp_path, isolation="strict", workspace="alpha")
    written = tools.memory(
        action="remember", data={"content": "pending", "subject": "pending", "workspace": "new-space"},
    )
    memory_id = written["data"]["id"]
    assert tools.db.get_memory(memory_id)["status"] == "pending"

    def forbidden_embedder():
        raise AssertionError("confirm-pending must not load or invoke an embedder")

    monkeypatch.setattr(tools, "_ensure_embedder", forbidden_embedder)
    confirmed = tools.memory_confirm_pending_workspace(
        memory_id=memory_id, canonical="new-space", authorized=True,
    )
    assert confirmed["ok"] is True
    assert tools.db.get_memory(memory_id)["status"] == "active"


def test_real_fastmcp_delivers_system_and_semantic_together_and_failure_consumes_neither(
    tmp_path: Path,
) -> None:
    from memory_arbiter import server as server_module

    settings = _settings(tmp_path, update_check_enabled=False)
    with patch("memory_arbiter.server.Settings.from_env", return_value=settings):
        bundle = server_module.build_runtime()
    try:
        notice_id = _notice(bundle.tools, key="fastmcp")
        system_notice = {"type": "update_available", "latest_version": "9.9.9"}
        consume_calls = 0

        def consume():
            nonlocal consume_calls
            consume_calls += 1
            return [dict(system_notice)]

        bundle.tools._consume_notices = consume
        _, response = asyncio.run(bundle.app._tool_manager.call_tool(
            "memory", {"action": "help"}, convert_result=True,
        ))
        assert response["notices"][0] == system_notice
        assert _semantic_stubs(response)[0]["notice_id"] == notice_id
        assert consume_calls == 1

        second_id = _notice(bundle.tools, key="fastmcp-failure")
        consume_calls = 0
        _, failed = asyncio.run(bundle.app._tool_manager.call_tool(
            "memory", {"action": "unknown"}, convert_result=True,
        ))
        assert failed["ok"] is False
        assert "notices" not in failed
        assert consume_calls == 0
        assert bundle.tools.db.read_semantic_notice(second_id)["delivered_at"] is None
    finally:
        bundle.tools.shutdown(timeout=1)
