"""Shared non-fixture test helpers (structure batch, 2026-10-11).

make_tools: the 2-key Settings form (db_path/backup_jsonl only) is the
dominant variant across the suite — 16 of 42 pre-consolidation defs. The
BASELINE IS DELIBERATE: do NOT switch it to a 5-key codex form, that would
silently give the 2-key files a codex/agent-a write identity. Files whose
defaults drifted pass explicit overrides (or keep a thin def wrapper when
they carry extra positional params, e.g. isolation).

Not collected by pytest (name does not match test_*). Import style across
the suite is `from tests._helpers import ...` (pyproject pythonpath=["."]).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.semantic_conflict import normalize_value
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path, **settings_overrides: Any) -> MemoryTools:
    # db_path/backup_jsonl defaults live in the merge dict (not the call) so
    # callers CAN override the filenames (e.g. claims_retirement legacy
    # fixtures hardcode tmp_path / "memory.sqlite3").
    # client/agent_id/workspace pinned to today's Settings defaults on
    # purpose: 15 consolidated files rely on them, and a future default flip
    # upstream must not silently change the write identity these tests
    # exercise. Override freely per call.
    return MemoryTools(Settings(**{
        "db_path": tmp_path / "m.db",
        "backup_jsonl": tmp_path / "b.jsonl",
        "client": "",
        "agent_id": "",
        "workspace": "default",
        **settings_overrides,
    }))


def record_conflict(tools: MemoryTools, workspace: str, left: int, right: int) -> int:
    members = [
        ConflictMember(
            memory_id=memory_id, version=1, attribute_raw="database", value_raw=value,
            normalized_attribute="database", normalized_value=value,
            evidence_quote=value, evidence_span=(0, len(value)), content_hash=char * 64,
            direction="a_to_b", prompt_version="p1", detector_version="d1",
        )
        for memory_id, value, char in ((left, "mysql", "a"), (right, "sqlite", "b"))
    ]
    outcome = tools.db.record_conflict_group(
        workspace_canonical=workspace,
        slot_key={"entity": "svc", "attribute": "database", "scope": "production"},
        members=members,
        value_groups=[
            ConflictValueGroup("mysql", "MySQL", (f"{left}@1",)),
            ConflictValueGroup("sqlite", "SQLite", (f"{right}@1",)),
        ],
        detection_reason="different", source="scan", detector_version="d1",
    )
    assert outcome["outcome"] == "inserted"
    return int(outcome["conflict_id"])


# ── moved from test_workspace_registry.py (commit 8 split) ──

NOW = "2026-01-01T00:00:00+00:00"
_fact_serial = 0

def register(tools: MemoryTools, *names: str) -> None:
    """Register canonicals directly, bypassing the resolver's mechanical-twin
    fold, to simulate legacy double-registration."""
    with tools.db.write_transaction() as conn:
        for name in names:
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES(?,?)",
                (name, NOW),
            )

def write(tools: MemoryTools, workspace: str, content: str | None = None) -> int:
    # 0.16.6 write gate: byte-identical ACTIVE content can no longer coexist
    # in one workspace, so the default content is uniquified per call. Callers
    # that need an exact value still pass one.
    global _fact_serial
    if content is None:
        content = f"workspace fact #{_fact_serial}"
        _fact_serial += 1
    return int(tools.memory_write(
        content=content, subject="workspace", workspace=workspace,
        source_type="agent_generated",
    )["data"]["id"])


def alias_governance_make_tools(
    tmp_path: Path, isolation: str = "weak", *, vec: bool = False,
) -> MemoryTools:
    # vec=True points at a (fake) GGUF model — the model path IS the intent
    # since 0.15.0 — and mirrors the first successful embedder build by
    # creating the lazy vec0 tables at dim 2.
    model = tmp_path / "fake.gguf"
    settings = Settings(
        db_path=tmp_path / "gov.sqlite3",
        backup_jsonl=tmp_path / "gov.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        embedding_model_path=model if vec else None, isolation=isolation,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    if vec:
        model.write_bytes(b"fake")
        assert db.ensure_vec_tables(2) == []
    return tools

# ── moved from test_conflict_lifecycle.py (commit 9 split) ──

def _db(tmp_path: Path) -> MemoryDB:
    return MemoryDB(Settings(db_path=tmp_path / "memory.db", backup_jsonl=tmp_path / "backup.jsonl"))


def _memory(db: MemoryDB, content: str, workspace: str = "w") -> int:
    memory_id, _ = db.insert_memory(MemoryRecord(content=content, subject="database", agent_id="a", workspace=workspace), workspace)
    assert memory_id is not None
    return memory_id


def _member(memory_id: int, value: str, version: int = 1) -> ConflictMember:
    quote = f"database is {value}"
    return ConflictMember(
        memory_id=memory_id, version=version, attribute_raw="database", value_raw=value,
        normalized_attribute="database", normalized_value=normalize_value(value), evidence_quote=quote,
        evidence_span=(0, len(quote)), content_hash=(str(memory_id) * 64)[:64], direction="a_to_b",
        prompt_version="p1", detector_version="d1",
    )


def _groups(*members: ConflictMember) -> list[ConflictValueGroup]:
    by_value: dict[str, list[str]] = {}
    for member in members:
        by_value.setdefault(member.normalized_value, []).append(f"{member.memory_id}@{member.version}")
    return [ConflictValueGroup(value, value.title(), tuple(refs)) for value, refs in by_value.items()]


def _record(db: MemoryDB, members: list[ConflictMember], *, expected_revision=None, status="open", slot=True):
    return db.record_conflict_group(
        workspace_canonical="w",
        slot_key={"entity": "project", "attribute": "database", "scope": "global"} if slot else None,
        members=members, value_groups=_groups(*members), detection_reason="different database",
        source="scan", detector_version="d1", prompt_version="p1", conflict_point="database",
        status=status, expected_revision=expected_revision,
    )
