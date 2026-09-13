"""Generate the run_all_checks equivalence corpus.

Run once against the *unmodified* implementation; the snapshot it writes is
what ``tests/test_golden_doctor.py`` then holds the refactor to.

    python scripts/gen_golden_doctor.py
    python scripts/gen_golden_doctor.py --dry-run   # generate twice, diff

``--dry-run`` is the convergence check: two independent runs must produce byte
identical output. Anything that still differs is an unmasked volatile field,
and leaving one in makes the gate flaky, which is how gates get relaxed.

What the corpus pins, and why each part matters:

* the **ordered** findings list -- console_static's doctorSummaryCard renders
  ``findings.slice(0, 6)``, and every existing test looks findings up by
  check_id, so nothing else in the suite can catch a reordering
* conditional checks that may emit nothing at all (scan_epoch, scan_stale,
  scan_chain, spec_drift, config.warnings)
* all six embedder probe outcomes. The fifth one is the trap: when the probe
  returns a handle but ``embed_text`` raises, ``vector.device`` and
  ``evidence.unit_budget`` are still emitted, because the try only wraps the
  embed call. Folding the embed result into the probe outcome would silently
  drop the GPU-degradation warning.

Fixtures deliberately never create the sqlite-vec virtual tables, so the
snapshot is identical with and without the extension installed -- CI runs both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from doctor_fixtures import build_snapshots  # noqa: E402

OUT = ROOT / "tests" / "golden" / "doctor.json"


def _dump(snapshots: list[dict[str, Any]]) -> str:
    return json.dumps(snapshots, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="generate twice and diff instead of writing")
    args = parser.parse_args(argv)

    first = _dump(build_snapshots())
    if args.dry_run:
        second = _dump(build_snapshots())
        if first == second:
            print(f"convergence OK: two runs identical ({len(json.loads(first))} snapshots)")
            return 0
        import difflib

        diff = list(difflib.unified_diff(
            first.splitlines(), second.splitlines(), "run1", "run2", lineterm="", n=2,
        ))
        print(f"UNMASKED VOLATILE FIELDS: {len(diff)} diff lines", file=sys.stderr)
        print("\n".join(diff[:80]), file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(first, encoding="utf-8")
    snapshots = json.loads(first)
    print(f"wrote {len(snapshots)} snapshots to {OUT}")
    for snap in snapshots:
        findings = snap["report"]["findings"]
        print(f"  {snap['case_id']:48s} {len(findings):2d} findings  overall={snap['report']['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
