#!/usr/bin/env python3
"""Purge scan_numeric_autoreject not_a_conflict rows from the conflicts table.

0.16.0/0.16.1 machine numeric auto-reject landed one audit row per rejected
pair (17,835 in the live library, all 2026-09). 0.16.2 E11③ retired the
machine rejection (cleared pairs are counted, never landed) and §1.8 already
excludes these rows from the suppression source — they are pure residue.
This one-shot op removes them. The rows being deleted were never a
suppression source, so scan behaviour is unchanged by construction.

Safety: dry-run by default (reports the count only); --apply requires a
fresh timestamped backup of the database file first and refuses to run if
the backup fails.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

CONDITION = "status='not_a_conflict' AND COALESCE(source,'')='scan_numeric_autoreject'"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", required=True, type=Path,
        help="path to the live sqlite database (e.g. ~/.local/share/memory-arbiter/memory.sqlite3)",
    )
    parser.add_argument("--apply", action="store_true", help="execute the purge (default: dry-run)")
    args = parser.parse_args()

    db_path = args.db.expanduser().resolve()
    if not db_path.exists():
        print(f"database not found: {db_path}", file=sys.stderr)
        return 2

    import os
    os.environ["MEMORY_ARBITER_DB_PATH"] = str(db_path)
    from memory_arbiter.config import Settings
    from memory_arbiter.db import MemoryDB

    settings = Settings.from_env()
    db = MemoryDB(settings)
    with db.connection() as conn:
        count = conn.execute(f"SELECT COUNT(*) FROM conflicts WHERE {CONDITION}").fetchone()[0]
    kept = None
    print(f"target rows (status=not_a_conflict, source=scan_numeric_autoreject): {count}")
    if not args.apply:
        print("dry-run: nothing deleted (pass --apply to execute)")
        return 0

    if count == 0:
        print("nothing to delete")
        return 0

    backup_path = db_path.with_name(
        f"{db_path.stem}.bak-before-autoreject-purge-{time.strftime('%Y%m%d-%H%M%S')}{db_path.suffix}"
    )
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup_path))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    if not backup_path.exists() or backup_path.stat().st_size == 0:
        print("backup failed — refusing to purge", file=sys.stderr)
        return 3
    print(f"backup written: {backup_path} ({backup_path.stat().st_size} bytes)")

    with db.write_transaction() as conn:
        cur = conn.execute(f"DELETE FROM conflicts WHERE {CONDITION}")
        deleted = cur.rowcount
    with db.connection() as conn:
        kept = conn.execute(f"SELECT COUNT(*) FROM conflicts WHERE {CONDITION}").fetchone()[0]
        total_nac = conn.execute(
            "SELECT COUNT(*) FROM conflicts WHERE status='not_a_conflict'").fetchone()[0]
    print(f"deleted={deleted} remaining_target={kept} "
          f"not_a_conflict_total_now={total_nac}")
    return 0 if (deleted == count and kept == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
