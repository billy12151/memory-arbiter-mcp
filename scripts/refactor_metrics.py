"""Refactor health metrics: size, complexity, god-object and back-reference load.

``--check`` compares against ``refactor_metrics_baseline.json`` and exits
non-zero when a metric regresses, so a structural refactor cannot quietly
trade one oversized function for an oversized class.

A metric that moves for a legitimate reason (a shim that adds back-references,
a method that changes file) is handled by regenerating the baseline *in the
same commit* and justifying it in the commit message: the gate catches
unintended drift, and intended drift shows up as a reviewable baseline diff.

Usage:
    python scripts/refactor_metrics.py                 # print current metrics
    python scripts/refactor_metrics.py --check         # gate against baseline
    python scripts/refactor_metrics.py --write-baseline
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple


class _Fn(NamedTuple):
    lines: int
    cc: int
    filename: str
    name: str

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "memory_arbiter"
BASELINE = Path(__file__).resolve().parent / "refactor_metrics_baseline.json"

# max = must not exceed, min = must not fall below, eq = must stay equal.
# "eq" marks a ceiling owned by code the current plan deliberately does not
# touch: it must neither regress nor be silently "improved" out of scope.
DIRECTIONS = {
    "validation_max_lines": "max",
    "validation_max_cc": "max",
    "doctor_max_lines": "max",
    "doctor_max_cc": "max",
    "fns_over_150": "max",
    "fns_over_80": "max",
    "max_cc": "max",
    "max_lines": "max",
    "tools_py_lines": "max",
    "memorytools_solid_methods": "max",
    "backref_sites": "max",
    "backref_distinct_members": "max",
}


def complexity(node: ast.AST) -> int:
    return 1 + sum(
        1
        for n in ast.walk(node)
        if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.BoolOp, ast.IfExp))
    )


def scan() -> dict[str, int]:
    fns: list[_Fn] = []
    backrefs: Counter[str] = Counter()
    for path in sorted(PKG.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        for member in re.findall(r"self\._tools\.(\w+)", src):
            backrefs[member] += 1
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = node.end_lineno if node.end_lineno is not None else node.lineno
                fns.append(_Fn(end - node.lineno + 1, complexity(node), path.name, node.name))

    tools_src = (PKG / "tools.py").read_text(encoding="utf-8")
    tools_cls = next(
        n for n in ast.parse(tools_src).body
        if isinstance(n, ast.ClassDef) and n.name == "MemoryTools"
    )
    # "Solid" = a method with a real body rather than a one-line delegation:
    # the number that tells whether MemoryTools is converging on a pure facade.
    solid = sum(
        1
        for m in tools_cls.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        and m.end_lineno is not None
        and m.end_lineno - m.lineno + 1 > 12
    )

    def peak_lines(filename: str) -> int:
        values = [f.lines for f in fns if f.filename == filename]
        return max(values) if values else 0

    def peak_cc(filename: str) -> int:
        values = [f.cc for f in fns if f.filename == filename]
        return max(values) if values else 0

    return {
        "max_lines": max(f.lines for f in fns),
        "max_cc": max(f.cc for f in fns),
        "fns_over_150": sum(1 for f in fns if f.lines > 150),
        "fns_over_80": sum(1 for f in fns if f.lines > 80),
        "tools_py_lines": len(tools_src.splitlines()),
        "memorytools_solid_methods": solid,
        "backref_sites": sum(backrefs.values()),
        "backref_distinct_members": len(backrefs),
        "validation_max_lines": peak_lines("validation.py"),
        "validation_max_cc": peak_cc("validation.py"),
        "doctor_max_lines": peak_lines("doctor.py"),
        "doctor_max_cc": peak_cc("doctor.py"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="gate against the baseline file")
    parser.add_argument("--write-baseline", action="store_true", help="overwrite the baseline file")
    args = parser.parse_args(argv)

    current = scan()
    if args.write_baseline:
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline written: {BASELINE}")
        return 0

    print(json.dumps(current, indent=2, sort_keys=True))
    if not args.check:
        return 0
    if not BASELINE.exists():
        print(f"ERROR: missing baseline {BASELINE}", file=sys.stderr)
        return 2

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    regressions: list[str] = []
    for key, value in sorted(current.items()):
        want = baseline.get(key)
        direction = DIRECTIONS.get(key, "max")
        if want is None:
            regressions.append(f"{key}: missing from baseline")
            continue
        worse = (
            (direction == "max" and value > want)
            or (direction == "min" and value < want)
            or (direction == "eq" and value != want)
        )
        if worse:
            regressions.append(f"{key}: now {value}, baseline {want} (direction {direction})")
    for line in regressions:
        print("REGRESSION " + line, file=sys.stderr)
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
