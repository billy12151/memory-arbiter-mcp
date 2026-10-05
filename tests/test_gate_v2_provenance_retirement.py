"""Gate-v2 G3: provenance retirement — metadata.entity/scope are gone.

Three storage serialization points strip the retired keys (INSERT, full
UPDATE, metadata update — the third was missed by the plan and found by
adversarial review), the additive keyed migration purges the stock, and the
four notice drop-points that used to depend on the keys are gone with the
gate. Enforcement lives at storage; tools-layer warnings are only the hint.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.db.additive import _purge_retired_metadata_keys
from memory_arbiter.models import MemoryRecord
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    settings = Settings(db_path=tmp_path / "g3.sqlite3", backup_jsonl=tmp_path / "g3.jsonl")
    return MemoryTools(settings, MemoryDB(settings))


def _raw_metadata(tools: MemoryTools, memory_id: int) -> dict:
    with tools.db.connection() as conn:
        row = conn.execute(
            "SELECT metadata FROM memories WHERE id=?", (int(memory_id),)
    ).fetchone()
    return json.loads(row["metadata"]) if row and row["metadata"] else {}


def test_write_strips_retired_keys_and_warns(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    result = tools.memory_write(
        content="c", subject="s", tags=[],
        metadata={"entity": "e", "scope": "sc", "related_memory_ids": [1, 2]},
    )
    data = result["data"]
    assert any("已废弃" in warning for warning in (result.get("warnings") or []))
    stored = _raw_metadata(tools, data["id"])
    assert "entity" not in stored and "scope" not in stored
    assert stored.get("related_memory_ids") == [1, 2]  # other keys survive


def test_full_update_strips_and_entity_only_update_is_no_change(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid, _ = tools.db.insert_memory(MemoryRecord(
        content="c", agent_id="a", workspace="main", tags=[],
        source_type="agent_generated", subject="s",
    ))
    before = tools.db.get_memory(mid)
    with tools.db.write_transaction() as conn:
        changed = tools.db.memories.update_memory_on_conn(
            conn, int(mid), {"metadata": {"entity": "new", "other": "x"}},
        )
    assert changed is True
    stored = _raw_metadata(tools, mid)
    assert "entity" not in stored and stored.get("other") == "x"
    after = tools.db.get_memory(mid)
    assert int(after["version"]) == int(before["version"])  # no snapshot bump

    # An update touching ONLY the retired keys is a no_change: the stripped
    # value equals the stored one, so nothing rewrites or bumps.
    untouched = tools.db.get_memory(mid)
    with tools.db.write_transaction() as conn:
        tools.db.memories.update_memory_on_conn(
            conn, int(mid), {"metadata": {"entity": "different", "other": "x"}},
        )
    still = tools.db.get_memory(mid)
    assert int(still["version"]) == int(untouched["version"])
    assert _raw_metadata(tools, mid) == {"other": "x"}


def test_metadata_update_on_conn_retired_keys_are_no_change(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid, _ = tools.db.insert_memory(MemoryRecord(
        content="c", agent_id="a", workspace="main", tags=[],
        source_type="agent_generated", subject="s",
    ))
    with tools.db.write_transaction() as conn:
        result = tools.db.memories.update_metadata_fields_low_side_effect_on_conn(
            conn, int(mid), set_fields={"entity": "x"}, authorized=True,
        )
    assert result["outcome"] == "no_change"  # NOT an empty rewrite + version bump loop
    assert int(tools.db.get_memory(mid)["version"]) == 1

    with tools.db.write_transaction() as conn:
        result = tools.db.memories.update_metadata_fields_low_side_effect_on_conn(
            conn, int(mid), clear_fields=["scope"], authorized=True,
        )
    assert result["outcome"] == "no_change"


def test_additive_purge_migration_strips_stock_and_is_idempotent(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid, _ = tools.db.insert_memory(MemoryRecord(
        content="c", agent_id="a", workspace="main", tags=[],
        source_type="agent_generated", subject="s",
    ))
    with tools.db.write_transaction() as conn:
        # Seed the stock shape directly (a 0.16 library): both retired keys
        # plus unrelated keys, AND one malformed-JSON row that must not
        # abort the whole statement.
        conn.execute(
            "UPDATE memories SET metadata=? WHERE id=?",
            (json.dumps({"entity": "e", "scope": "s", "tests": [1], "note": "keep"}), mid),
        )
        conn.execute(
            "INSERT INTO memories(content, agent_id, workspace, workspace_canonical, tags, "
            "source_type, status, subject, metadata, created_at, content_sha, event_time, ingest_time) VALUES "
            "('x','a','main','main','[]','agent_generated','active','bad','{not json',"
            "'2026-01-01T00:00:00Z','deadbeef','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')"
        )

    with tools.db.write_transaction() as conn:
        # Boot already ran the migration against an empty library and wrote
        # its guard — reset the key to exercise the migration body itself.
        conn.execute("DELETE FROM migration_state WHERE key='metadata_entity_scope_purged_v1'")
    with tools.db.write_transaction() as conn:
        applied = _purge_retired_metadata_keys(conn)
    assert applied.startswith("metadata_entity_scope_purged(1)")  # only the valid row
    stored = _raw_metadata(tools, mid)
    assert stored == {"tests": [1], "note": "keep"}

    with tools.db.write_transaction() as conn:
        assert _purge_retired_metadata_keys(conn) == ""  # keyed re-entry: no-op

    with tools.db.connection() as conn:
        bad = conn.execute(
            "SELECT metadata FROM memories WHERE subject='bad'"
        ).fetchone()
    assert bad["metadata"] == "{not json"  # untouched, boot not aborted


def test_replay_and_merge_paths_inherit_the_insert_strip(tmp_path: Path) -> None:
    """replay restore (backup_replay.replay_one) and confirm/merge metadata
    writes all funnel through insert_memory_on_conn / update_memory_on_conn —
    one record through the replay-shaped primitive pins the whole family."""
    tools = make_tools(tmp_path)
    record = MemoryRecord(
        content="replayed", agent_id="a", workspace="main", tags=[],
        source_type="agent_generated", subject="r",
        metadata={"entity": "e", "scope": "s", "keep": True},
    )
    with tools.db.write_transaction() as conn:
        mid = tools.db.insert_memory_on_conn(conn, record)
    stored = _raw_metadata(tools, mid)
    assert "entity" not in stored and "scope" not in stored
    assert stored.get("keep") is True
