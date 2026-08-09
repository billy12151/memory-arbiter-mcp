"""Replace static ``getattr(<recv>.settings, "field", <default>)`` with direct access.

Phase 2 (v0.12.4). The attribution report (getattr_attribution.md) proved all
static call sites read a real Settings field whose literal default equals the
field default (or, for db.py:881, is behaviourally equivalent). So the getattr
is redundant and can collapse to ``<recv>.settings.<field>``.

Only STATIC call sites are rewritten: the receiver must literally be
``self.settings`` / ``db.settings`` / ``settings``, the field a string literal,
and a default present. Dynamic receivers (console_api's runtime attr name) are
skipped. The replacement preserves the original receiver text verbatim.

Run:  .venv/bin/python scripts/refactor_baseline/collapse_getattr.py [--apply]
Default is dry-run (prints the diff hunks).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Match: getattr( RECV , "field" , default )  — single line only.
# RECV is captured so we can re-emit it verbatim.
GETATTR_RE = re.compile(
    r"getattr\(\s*(?P<recv>self\.settings|db\.settings|settings)\s*,\s*"
    r"[\"'](?P<field>[a-zA-Z_][a-zA-Z0-9_]*)[\"']\s*,"
    r"(?P<default>[^()]*?)\)"
)

TARGETS = [
    "memory_arbiter/tools.py",
    "memory_arbiter/db.py",
    "memory_arbiter/doctor.py",
    "memory_arbiter/search.py",
    "memory_arbiter/console_api.py",
]


def collapse(text: str) -> tuple[str, int]:
    def sub(m: re.Match) -> str:
        recv, field = m.group("recv"), m.group("field")
        return f"{recv}.{field}"
    new, n = GETATTR_RE.subn(sub, text)
    return new, n


def main() -> int:
    apply = "--apply" in sys.argv
    total = 0
    for rel in TARGETS:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        new, n = collapse(text)
        if n == 0:
            continue
        total += n
        print(f"{rel}: {n} replacements")
        if apply:
            path.write_text(new, encoding="utf-8")
        else:
            # show the changed lines for review
            old_lines = text.splitlines()
            new_lines = new.splitlines()
            for i, (o, nn) in enumerate(zip(old_lines, new_lines), 1):
                if o != nn:
                    print(f"  - {o.strip()}")
                    print(f"  + {nn.strip()}")
    print(f"\nTotal: {total} replacements ({'APPLIED' if apply else 'dry-run'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
