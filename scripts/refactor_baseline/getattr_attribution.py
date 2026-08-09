"""Attribute getattr(settings, "field", default) call sites against the dataclass.

Phase 2 replaces ``getattr(self.settings, "x", <default>)`` with direct
attribute access. That is only safe when the literal ``<default>`` equals the
dataclass field default — if they differ, the getattr is NOT redundant, it is a
(deliberate or accidental) behavior difference and must be copied verbatim, not
"corrected" (R5).

This script:
  1. Parses memory_arbiter/config.py for @dataclass Settings field defaults.
  2. Greps all package modules for ``getattr(<...settings...>, "name", <literal>)``.
  3. Compares each literal default to the field default and reports
     MATCH / MISMATCH / UNKNOWN (non-literal default we cannot evaluate).

Usage:
    .venv/bin/python scripts/refactor_baseline/getattr_attribution.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PKG = ROOT / "memory_arbiter"
CONFIG = PKG / "config.py"

GETATTR_RE = re.compile(
    r"getattr\(\s*(?:self\.)?(?:db\.)?settings\s*,\s*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']\s*,\s*([^)]+)\)"
)


def settings_field_defaults() -> dict[str, str]:
    """Return {field_name: source-text-of-default} for the Settings dataclass."""
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    defaults: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            for stmt in node.body:
                # ``field: type = default``
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.value is not None:
                        defaults[stmt.target.id] = ast.unparse(stmt.value)
                    else:
                        defaults[stmt.target.id] = "<no default (required)>"
    return defaults


def _eval(lit: str):
    """Best-effort literal evaluation; returns the source text if not a literal."""
    try:
        return ast.literal_eval(lit)
    except (ValueError, SyntaxError):
        return ("<unevaluated>", normalize(lit))


def normalize(lit: str) -> str:
    return re.sub(r"\s+", "", lit.strip())


def main() -> None:
    defaults = settings_field_defaults()
    rows: list[tuple[str, int, str, str, str]] = []
    for path in sorted(PKG.glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = GETATTR_RE.search(line)
            if not m:
                continue
            name, lit = m.group(1), m.group(2).strip()
            field_def = defaults.get(name)
            if field_def is None:
                verdict = "NOT-A-FIELD"
            elif _eval(lit) == _eval(field_def):
                verdict = "MATCH"
            else:
                verdict = "MISMATCH"
            rows.append((path.name, lineno, name, lit, f"{verdict} (field={field_def})"))

    mismatch = [r for r in rows if "MISMATCH" in r[4]]
    notfield = [r for r in rows if "NOT-A-FIELD" in r[4]]
    print(f"# getattr(settings) attribution: {len(rows)} call sites\n")
    print(f"## MISMATCH (literal default != field default) — {len(mismatch)}  [REVIEW: copy verbatim, do NOT 'fix']")
    for r in mismatch:
        print(f"  {r[0]}:{r[1]}  getattr(..., \"{r[2]}\", {r[3]})  -> {r[4]}")
    print(f"\n## NOT-A-FIELD (read but not a Settings field) — {len(notfield)}  [would AttributeError on direct access]")
    for r in notfield:
        print(f"  {r[0]}:{r[1]}  getattr(..., \"{r[2]}\", {r[3]})  -> {r[4]}")
    print(f"\n## MATCH (safe to collapse to attribute access) — {len(rows) - len(mismatch) - len(notfield)}")
    for r in rows:
        if r not in mismatch and r not in notfield:
            print(f"  {r[0]}:{r[1]}  \"{r[2]}\"  (field default {defaults.get(r[2])})")


if __name__ == "__main__":
    main()
