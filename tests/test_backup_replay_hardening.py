"""Boundary hardening for backup replay: a damaged line must land in
invalid_entries, never abort the inspection.

The per-line except is the whole point of the invalid-line channel -- it exists
for exactly the corrupted-file scenarios an operator would replay to recover
from. Found by adversarial review of the 0.16.6 boundary fix: json.loads on a
deep-nesting line raises RecursionError, which none of the previously caught
exception types covered.
"""
from __future__ import annotations

import json
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.db.backup_replay import BackupReplayStore

def _write_backup(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_deeply_nested_line_is_invalid_not_fatal(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "memory.db",
        backup_jsonl=tmp_path / "backup.jsonl",
        client="t",
        agent_id="t",
    )
    # A well-formed line survives alongside the pathological one: the bad line
    # must be quarantined, not allowed to take the good one down with it.
    good = json.dumps({
        "backup_schema": 1,
        "replay_key": "rk-hardening-1",
        "backup_written_at": "2026-09-14T00:00:00+00:00",
        "workspace_canonical": "ws",
        "record": {"content": "body", "subject": "subj"},
    })
    deep = "[" * 200_000 + "]" * 200_000
    _write_backup(settings.backup_jsonl, [good, deep])
    report = BackupReplayStore(MemoryDB(settings)).inspect()
    assert report["invalid"] == 1
    assert report["invalid_entries"][0]["line"] == 2
    assert "maximum recursion depth" in report["invalid_entries"][0]["reason"]
    assert report["entries"], "the well-formed line must still be readable"


def test_replay_postprocess_is_instance_bound(tmp_path: Path) -> None:
    """Adversarial-review pin (0.16.6 release round): a dangling @staticmethod
    over ``_postprocess_replayed_memory`` shifted every argument one slot at
    the only call site — an instance call would have passed the replay key as
    ``self`` and blown up with AttributeError. No test reached that branch, so
    pytest was green while the path was broken. This pin calls the method
    through the product pipeline and asserts the four-tuple contract."""
    from memory_arbiter.tools import MemoryTools

    settings = Settings(
        db_path=tmp_path / "memory.db",
        backup_jsonl=tmp_path / "backup.jsonl",
        client="t",
        agent_id="t",
    )
    tools = MemoryTools(settings, MemoryDB(settings))
    written = tools.memory_write(content="正文内容", subject="标题", tags=[])
    status, stages, error_code, warnings = tools._operations._postprocess_replayed_memory(
        "rk-pin-1", int(written["data"]["id"]),
    )
    assert isinstance(status, str) and isinstance(stages, dict)
    assert error_code is None and warnings == []
