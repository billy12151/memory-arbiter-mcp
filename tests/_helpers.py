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
from memory_arbiter.models import ConflictMember, ConflictValueGroup
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path, **settings_overrides: Any) -> MemoryTools:
    # db_path/backup_jsonl defaults live in the merge dict (not the call) so
    # callers CAN override the filenames (e.g. claims_retirement legacy
    # fixtures hardcode tmp_path / "memory.sqlite3").
    return MemoryTools(Settings(**{
        "db_path": tmp_path / "m.db",
        "backup_jsonl": tmp_path / "b.jsonl",
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
