"""Fail if mypy --strict reports any error NOT in the recorded baseline.

The pre-refactor tree already has 175 strict errors (mostly untyped MCP
decorator noise). Phase work must not ADD new type errors. This script runs
mypy, strips the error-code brackets, and diffs against
``mypy_strict_baseline.txt``. Any current error absent from the baseline is a
NEW type error and fails the check. Baseline errors that disappeared are fine
(fixing is allowed); they are reported but do not fail.

Usage:
    .venv/bin/python scripts/refactor_baseline/check_mypy.py
Exit code 0 = no new errors; 1 = new errors found.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE = Path(__file__).resolve().parent / "mypy_strict_baseline.txt"
PY = ROOT / ".venv" / "bin" / "python"


_LINE_RE = re.compile(r"(memory_arbiter/[a-z_0-9/]+\.py):\d+:")

# A file that was MOVED by the refactor keeps the same errors under a new path.
# Normalise the moved path back to its pre-refactor location so a pure move does
# not look like "N new errors". Key: post-move path -> pre-move path.
_PATH_MOVE_ALIASES = {
    "memory_arbiter/db/core.py": "memory_arbiter/db.py",
}


def _strip_lineno(err: str) -> str:
    """Normalise an error to file + message, dropping the line number.

    Move commits shorten/lengthen functions, shifting the line an error is
    reported on without changing the error itself. Comparing by full
    ``file:line: message`` would flag pure line-drift as a NEW error. We only
    care whether the *kind* of error at a *file* is new.
    """
    out = _LINE_RE.sub(r"\1:@:", err)
    for new_path, old_path in _PATH_MOVE_ALIASES.items():
        out = out.replace(new_path, old_path)
    return out


def current_errors() -> list[str]:
    proc = subprocess.run(
        [str(PY), "-m", "mypy", "--strict", "memory_arbiter/"],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    lines = [ln for ln in out.splitlines() if "error:" in ln]
    return sorted(_strip_lineno(re.sub(r" \[[a-z-]+\]$", "", ln)) for ln in lines)


def main() -> int:
    baseline = {_strip_lineno(e) for e in BASELINE.read_text(encoding="utf-8").splitlines()}
    now = current_errors()
    new = [e for e in now if e not in baseline]
    gone = [e for e in baseline if e not in now]
    if gone:
        print(f"OK: {len(gone)} baseline errors fixed (allowed):")
        for e in sorted(gone):
            print(f"  - {e}")
    if new:
        print(f"FAIL: {len(new)} NEW mypy strict errors introduced:")
        for e in new:
            print(f"  + {e}")
        return 1
    print(f"OK: no new mypy strict errors ({len(now)} current, baseline {len(baseline)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
