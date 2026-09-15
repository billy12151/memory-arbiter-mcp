#!/usr/bin/env python3
"""Backfill ``content_sha`` on memory records that are missing it.

Context (0.16.6 write dedup gate): every new write computes the content hash
inline, and the one-shot boot migration backfilled everything that existed at
upgrade time. Rows can still lack a hash afterwards in three ways: a writer on
pre-0.16.6 code ran against the upgraded DB (e.g. a second service sharing the
library before its own restart), a backup replay from a pre-gate JSONL
imported rows with NULL sha, or manual sqlite surgery. NULL-sha ACTIVE rows
sit OUTSIDE the dedup gate (the partial unique index only covers non-NULL) —
this script closes that gap. It refuses to fill a hash that would collide with
another ACTIVE row's effective hash (same governance rule as the boot
migration: retire/merge one of the pair first).

Usage:
    python scripts/backfill_content_sha.py            # plan only (dry-run)
    python scripts/backfill_content_sha.py --apply    # fill the hashes
    python scripts/backfill_content_sha.py --all      # include non-active rows too
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.db.memories import content_sha  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the hashes (default: dry-run)")
    parser.add_argument("--all", action="store_true",
                        help="fill non-active rows too (they hold no gate slot; observability only)")
    args = parser.parse_args()

    db = MemoryDB(Settings.from_env())
    status_filter = "" if args.all else " AND status='active'"
    with db.connection() as conn:
        rows = conn.execute(
            f"SELECT id, status, content FROM memories WHERE content_sha IS NULL{status_filter} ORDER BY id"
        ).fetchall()

    # Effective-sha collision pre-check over ALL active rows (stored or
    # about-to-be-filled) — mirrors additive._add_content_sha_dedupe: filling
    # a hash that another ACTIVE row already owns would violate the partial
    # unique index mid-script.
    effective: dict[tuple[str, str], list[int]] = {}
    with db.connection() as conn:
        for row in conn.execute(
            "SELECT id, workspace_canonical, workspace, content_sha, content "
            "FROM memories WHERE status='active'"
        ).fetchall():
            sha = row["content_sha"] if row["content_sha"] is not None else content_sha(row["content"] or "")
            key = (row["workspace_canonical"] or row["workspace"] or "", sha)
            effective.setdefault(key, []).append(int(row["id"]))
    dupes = {key: ids for key, ids in effective.items() if len(ids) > 1}
    if dupes:
        print("ABORT: filling would create ACTIVE duplicate pairs (govern one row of each first):")
        for (ws, sha), ids in list(dupes.items())[:20]:
            print(f"  ws={ws!r} sha={sha[:8]}… ids={ids}")
        return 2

    print(f"rows missing content_sha{' (active only)' if not args.all else ''}: {len(rows)}")
    for row in rows[:10]:
        print(f"  #{row['id']} [{row['status']}] -> {content_sha(row['content'] or '')[:8]}…")
    if len(rows) > 10:
        print(f"  … and {len(rows) - 10} more")
    if not rows:
        return 0
    if not args.apply:
        print("dry-run; pass --apply to fill")
        return 0

    with db.write_transaction() as conn:
        conn.executemany(
            "UPDATE memories SET content_sha=? WHERE id=? AND content_sha IS NULL",
            [(content_sha(row["content"] or ""), int(row["id"])) for row in rows],
        )
    print(f"filled {len(rows)} hash(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
