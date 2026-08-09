"""Diff two symbol snapshots and fail on any unsafe surface change.

Compares a "before" snapshot (pre-refactor) against an "after" snapshot taken
during Phase 3/4/5. A change is UNSAFE when:
  - a name present before is missing after              (lost symbol — R8)
  - a callable's signature changed                      (contract drift — R6)
  - a callable's __module__ changed to a module NOT in the allow-list
        (delegation moved the impl somewhere unplanned — R6)

__module__ moves ARE expected when a method relocates into a db/ sub-store,
pipeline/ module, workers.py, surfaces.py, search/ submodule, or doctor_checks/
module. Those are whitelisted via --allow-module-move (defaults cover the plan).

Usage:
    .venv/bin/python scripts/refactor_baseline/diff_symbols.py \
        scripts/refactor_baseline/symbols_before.json /tmp/symbols_after.json
Exit 0 = only safe/whitelisted changes; 1 = unsafe change found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# Memory addresses (``at 0x10a1f77e0``) appear in signatures for dataclass
# MISSING sentinels and default function objects. They are per-process and
# meaningless across snapshot runs — normalise before comparing (R6 self-diff
# must be stable).
_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _norm_sig(sig):
    if not isinstance(sig, str):
        return sig
    return _ADDR_RE.sub(" at 0xADDR", sig)

DEFAULT_ALLOWED_MODULE_MOVES = {
    # db.py -> db/ package stores
    "memory_arbiter.db",
    "memory_arbiter.db.core", "memory_arbiter.db.schema", "memory_arbiter.db.vectors",
    "memory_arbiter.db.workspaces", "memory_arbiter.db.memories", "memory_arbiter.db.conflicts",
    "memory_arbiter.db.audit", "memory_arbiter.db.sections_store",
    # tools.py -> workers/surfaces/pipeline
    "memory_arbiter.tools", "memory_arbiter.workers", "memory_arbiter.surfaces",
    "memory_arbiter.pipeline", "memory_arbiter.pipeline.write",
    "memory_arbiter.pipeline.signals", "memory_arbiter.pipeline.sections",
    # search.py -> search/ package
    "memory_arbiter.search", "memory_arbiter.search.recall", "memory_arbiter.search.rerank",
    "memory_arbiter.search.filters", "memory_arbiter.search.fts",
    # doctor.py -> doctor_checks/ package
    "memory_arbiter.doctor", "memory_arbiter.doctor_checks",
    "memory_arbiter.doctor_checks.config_env", "memory_arbiter.doctor_checks.vector",
    "memory_arbiter.doctor_checks.semantic", "memory_arbiter.doctor_checks.split",
    "memory_arbiter.doctor_checks.integrity", "memory_arbiter.doctor_checks.conflicts",
    # shared leaf modules
    "memory_arbiter.text", "memory_arbiter.timeutil", "memory_arbiter.constants",
}

# Symbols the refactor REMOVES on purpose (internal aliases with no external or
# test dependency), each justified in the plan. Keyed label -> reason.
ALLOWED_REMOVALS = {
    # db.py's duplicate `import re as _re` (line ~3789); superseded by
    # text.CJK_RE_SUBJECT re-export. R11-补. No test/external import of `_re`.
    "memory_arbiter.db::_re": "R11-补: duplicate `import re as _re` removed; superseded by text.CJK_RE_SUBJECT",
}


def _index(snapshot: dict) -> dict[str, dict]:
    """Flatten snapshot into {label: descriptor} keyed by module/class + name."""
    flat: dict[str, dict] = {}
    for mod, attrs in snapshot.get("modules", {}).items():
        for name, desc in attrs.items():
            flat[f"{mod}::{name}"] = desc
    for cls, methods in snapshot.get("classes", {}).items():
        for name, desc in methods.items():
            flat[f"{cls}::{name}"] = desc
    return flat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    before = _index(json.load(open(args.before, encoding="utf-8")))
    after = _index(json.load(open(args.after, encoding="utf-8")))

    unsafe: list[str] = []
    moved_ok: list[str] = []
    removed_ok: list[str] = []

    for label, bdesc in before.items():
        adesc = after.get(label)
        if adesc is None:
            if label in ALLOWED_REMOVALS:
                removed_ok.append(f"removed {label}  ({ALLOWED_REMOVALS[label]})")
            else:
                unsafe.append(f"MISSING  {label}  (was {bdesc.get('kind')} in {bdesc.get('module')})")
            continue
        bsig, asig = _norm_sig(bdesc.get("signature")), _norm_sig(adesc.get("signature"))
        if bsig != asig:
            unsafe.append(f"SIG-CHANGE  {label}\n    before: {bsig}\n    after:  {asig}")
        bmod, amod = bdesc.get("module"), adesc.get("module")
        if bmod != amod:
            if amod in DEFAULT_ALLOWED_MODULE_MOVES or bmod == amod:
                moved_ok.append(f"module-move {label}: {bmod} -> {amod}")
            else:
                unsafe.append(f"MODULE-MOVE-UNALLOWED  {label}: {bmod} -> {amod}")

    # New symbols introduced by the refactor are fine (additions don't break
    # back-compat); report but do not fail.
    added = [label for label in after if label not in before]

    if moved_ok:
        print(f"OK module moves ({len(moved_ok)}):")
        for line in moved_ok:
            print(f"  ~ {line}")
    if removed_ok:
        print(f"OK allowed removals ({len(removed_ok)}):")
        for line in removed_ok:
            print(f"  ~ {line}")
    if added:
        print(f"New symbols ({len(added)}, allowed):")
        for label in sorted(added):
            print(f"  + {label}")
    if unsafe:
        print(f"\nFAIL: {len(unsafe)} unsafe surface changes:")
        for line in unsafe:
            print(f"  ✗ {line}")
        return 1
    print(f"\nOK: all {len(before)} pre-refactor symbols preserved with stable signatures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
