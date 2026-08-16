"""Fail if mypy --strict reports any error NOT in the recorded baseline.

The current tree has a recorded strict-error baseline (mostly untyped MCP
decorator noise). Future work must not ADD new type errors. This script runs
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
_LINE_RE = re.compile(r"(memory_arbiter/[a-z_0-9/]+\.py):\d+:")
_SUMMARY_RE = re.compile(
    r"^(?:Found (?P<errors>\d+) errors? in \d+ files? \(checked \d+ source files?\)|"
    r"Success: no issues found in \d+ source files?)$",
    re.MULTILINE,
)

_OPTIONAL_IMPORTS = ('module named "llama_cpp"', 'module named "sqlite_vec"')


def _strip_lineno(err: str) -> str:
    """Normalise location and mypy-version-only diagnostic wording drift."""
    out = _LINE_RE.sub(r"\1:@:", err)
    out = re.sub(r" on line \d+", " on line @", out)
    out = out.replace(" | SupportsTrunc", "")
    out = re.sub(
        r'Value of type ".+" is not indexable',
        'Value is not indexable',
        out,
    )
    return out.strip()


def current_errors() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "memory_arbiter/"],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    summary = _SUMMARY_RE.search(out)
    if proc.returncode not in {0, 1} or "No module named mypy" in out or summary is None:
        raise RuntimeError(
            f"mypy execution failed or emitted no parseable summary (exit={proc.returncode}):\n{out.strip()}"
        )
    lines = [
        ln for ln in out.splitlines()
        if "error:" in ln and not (
            "Cannot find implementation or library stub" in ln
            and any(module in ln for module in _OPTIONAL_IMPORTS)
        )
    ]
    reported_errors = int(summary.group("errors") or 0)
    if proc.returncode != (1 if reported_errors else 0) or reported_errors < len(lines):
        raise RuntimeError(
            "mypy return code/summary/error output disagree "
            f"(exit={proc.returncode}, summary={reported_errors}, parsed={len(lines)})"
        )
    return sorted(_strip_lineno(re.sub(r" \[[a-z-]+\]$", "", ln)) for ln in lines)


def main() -> int:
    baseline = {_strip_lineno(e) for e in BASELINE.read_text(encoding="utf-8").splitlines()}
    try:
        now = current_errors()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 2
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
