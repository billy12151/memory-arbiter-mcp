"""Extract every monkeypatch target in tests/ whose bound location may move.

Phase 3/4/5 moves private methods from MemoryTools/MemoryDB/doctor into
pipeline/store/doctor_checks modules. A monkeypatch that patches the OLD
binding (e.g. ``tools._publish_sections`` or ``memory_arbiter.tools.search_memories``)
will silently miss the new implementation if the facade does not delegate
through the patched name (R4).

This script greps tests/ for:
  - monkeypatch.setattr(<target>, ...) with any target
  - unittest.mock.patch("dotted.path")
and prints a table of file:line, the patched object/module, and the attribute,
so each can be triaged as (a) keep facade delegation so the old patch point
still reaches the real impl, or (b) explicitly migrate the patch target.

Usage:
    .venv/bin/python scripts/refactor_baseline/monkeypatch_inventory.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TESTS = ROOT / "tests"

SETATTR_RE = re.compile(r"monkeypatch\.setattr\(\s*([^,]+),")
PATCH_RE = re.compile(r"(?:mock\.patch|patch)\(\s*[\"']([^\"']+)[\"']")

# Tokens that suggest the patched object is one of the moving facades or a
# module-level function that Phase 3/4/5 relocates. sys/builtins/env patches
# are irrelevant to the move and are filtered out.
INTERESTING = (
    "tools", "MemoryTools", "db", "MemoryDB", "doctor", "search",
    "memory_arbiter", "_publish_sections", "search_memories", "extract_claims",
    "publish_memory_claims", "find_structured_claim_pairs", "record_conflict",
    "is_pair_dismissed", "_check_", "_load_state", "UpdateMonitor",
)
NOISE = ('"sys', '"builtins', "sys.", "builtins.", "backfill", "version_info", "isatty", "SUBJECT_MAP", "BACKFILL_PLAN")


def is_interesting(text: str) -> bool:
    if any(n in text for n in NOISE):
        return False
    return any(tok in text for tok in INTERESTING)


def main() -> None:
    rows: list[tuple[str, int, str, str]] = []
    for path in sorted(TESTS.glob("test_*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            m = SETATTR_RE.search(stripped)
            if m and is_interesting(stripped):
                rows.append((path.name, lineno, "setattr", m.group(1).strip()))
                continue
            m = PATCH_RE.search(stripped)
            if m and is_interesting(m.group(1)):
                rows.append((path.name, lineno, "patch", m.group(1)))

    print(f"# monkeypatch/mock.patch inventory ({len(rows)} hits)\n")
    current = None
    for name, lineno, kind, target in rows:
        if name != current:
            print(f"\n## {name}")
            current = name
        print(f"  L{lineno:<4} [{kind:7}] {target}")


if __name__ == "__main__":
    main()
