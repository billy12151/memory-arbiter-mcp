#!/usr/bin/env python3
"""Backfill empty ``subject`` on historical memory records.

Context: ``subject`` used to be nullable. DB-layer validation now makes it
required on insert (db.insert_memory), and memory_edit refuses to wipe it.
23 historical rows in the default workspace were written with ``subject IS NULL
OR TRIM(subject)=''``. This script gives each a real subject, derived from its
content, by calling the standard ``memory_edit`` path (not a raw SQL UPDATE) so
the change goes through version+1 / memory_history / FTS re-sync / embedding
recompute — i.e. it is a first-class edit, not a silent backdoor.

Usage:
    python scripts/backfill_subjects.py            # show plan (dry-run)
    python scripts/backfill_subjects.py --apply    # execute the backfill
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

# Make the package importable when run from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.tools import MemoryTools  # noqa: E402

# id -> subject, hand-derived from each row's content on 2026-08-09.
# All 23 are in the default workspace.
BACKFILL_PLAN: dict[int, dict[str, str]] = {
    152: {
        "subject": "JingleAI-mema-write自检",
        "workspace": "default",
        "content_hash": "491ff23cbfdb4841d15e8d14a1b7027ed1727974f6cb6c0024acc8c272af3afc",
    },
    600: {
        "subject": "金科营销运营平台接口文档",
        "workspace": "default",
        "content_hash": "40f6316e40d5a8655a83eef031b53d9ca594d0ae31aa0b9c9f1f7a4703594889",
    },
    # 618-637 are the workspace 语义归一治理 cluster:
    618: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "b051a16920553c343a16819f081136e1823ac60588188de6cbe7a1f6ed4c155e"},
    619: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "4abdcdb661bdd380c1ab494e7c5ec2ac24e51aa79ba80adaede99c53dea1ab83"},
    620: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "5edbc16824ad80ddf069f82bbe3bd20bb602afa5079206910ad20cdd62f82b11"},
    621: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "7294c610a61df3e00b5b5e2c44169e0e626a2e715948d3e5935462f2e67e89e9"},
    622: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "37528edeec5035a9614dc9f7768696b6696c50058f198d50d53028d5d9e61062"},
    623: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "9b73021d539655e50160216b852546ce8f7d2f003e4c4cf54a5f82e5fbb9a164"},
    624: {"subject": "workspace归一评测与测试", "workspace": "default", "content_hash": "2436e3ef21155a88c316cb6dad87b57b0639ea2f6a281edbfd6c03b20cf98729"},
    625: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "b0d7027e3ec92ea6e3d66ab4e079d90d5a21d5066488994c59cdfb24f4fc5a05"},
    626: {"subject": "workspace归一评测与测试", "workspace": "default", "content_hash": "c26731f5b9ba78357fb64a6fa3cbca2cc8c56417a105eeb578b06a6e99f29397"},
    627: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "933cc3ad298d477e7512306995fa0f634e10e155df4032b133d1c63868db67f6"},
    628: {"subject": "workspace归一评测与测试", "workspace": "default", "content_hash": "3b782f11a9c40efd75f3b6ca3d45f92548e74412a00276ddc39fa221bd6a3acb"},
    629: {"subject": "workspace归一评测与测试", "workspace": "default", "content_hash": "c8d7acfd3ca8de1c97309d14d81dd777eb0347677cc3c26cc0d430b3503c426e"},
    630: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "6097ada46a26ede20673f8e78f2612c59d1059ec99a134ec64f6ad52cb4743b1"},
    631: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "8181ce843ee3bb50b2d462b5d341e219f66dfd35300cb85feacd626b749c9ce6"},
    632: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "b09223d85b285d4ee927f32fba466dbb2a0ac9e9a2db28fe9e027ff99b514256"},
    633: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "5aeaf4fd227de4c2c3d3a9409f0ef90540f48b12a949db62f21b4114cce1db59"},
    634: {"subject": "workspace归一评测与测试", "workspace": "default", "content_hash": "fec8679bdcacd4bb4176a37e44cb01c22ff80d02dbe66aaca4e5563dd49debaa"},
    635: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "2f389705ab7cee674be5b4397a2cadcc2c8327d6237a109687d9bfb6f7761d6b"},
    636: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "474b5124050cb7205a995a43cfe0be363d6081523799ea880f8b094bb4d2109e"},
    637: {"subject": "workspace语义归一治理", "workspace": "default", "content_hash": "0d9911a645b927fc42ae7a6d7900b54592179fb68479eb399e9c6dbc46d3bdd6"},
    638: {"subject": "版本号单一数据源机制", "workspace": "default", "content_hash": "9e33873f1e5fee3150725478965da9c81e86412e20da3b44d036f809cec1c5c9"},
}

SUBJECT_MAP: dict[int, str] = {mid: plan["subject"] for mid, plan in BACKFILL_PLAN.items()}

REASON = "backfill empty subject (was nullable pre-v0.13)"


def _subject_map() -> dict[int, str]:
    """Compatibility hook for tests that monkeypatch the old SUBJECT_MAP name."""
    legacy = globals().get("SUBJECT_MAP")
    if isinstance(legacy, dict):
        return {int(k): str(v) for k, v in legacy.items()}
    return {mid: plan["subject"] for mid, plan in BACKFILL_PLAN.items()}


def _load_tools() -> MemoryTools:
    settings = Settings.from_env()
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _empty_subject_ids(tools: MemoryTools) -> list[int]:
    db = tools.db
    if not db.db_available:
        raise SystemExit("database not available")
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT id FROM memories WHERE (subject IS NULL OR TRIM(subject)='') "
            "AND status='active' ORDER BY id"
        ).fetchall()
    return [int(r["id"]) for r in rows]


def _content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _validate_row(mid: int, row: dict[str, Any], subject_map: dict[int, str]) -> list[str]:
    """Return safety-gate errors for the built-in production backfill plan.

    Tests monkeypatch the legacy SUBJECT_MAP hook with synthetic ids; in that
    mode no content hashes exist, so validation is intentionally skipped. The
    real script plan, however, must verify workspace + content hash before any
    edit so a different DB with the same integer ids cannot be corrupted.
    """
    if mid not in BACKFILL_PLAN:
        return []
    plan = BACKFILL_PLAN[mid]
    errors: list[str] = []
    if subject_map.get(mid) != plan["subject"]:
        errors.append(f"subject map mismatch: expected {plan['subject']!r}, got {subject_map.get(mid)!r}")
    if str(row.get("workspace") or "") != plan["workspace"]:
        errors.append(f"workspace mismatch: expected {plan['workspace']!r}, got {row.get('workspace')!r}")
    actual_hash = _content_hash(str(row.get("content") or ""))
    if actual_hash != plan["content_hash"]:
        errors.append(f"content hash mismatch: expected {plan['content_hash']}, got {actual_hash}")
    if row.get("status") != "active":
        errors.append(f"status mismatch: expected 'active', got {row.get('status')!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="execute the backfill (default is dry-run: print plan only)",
    )
    args = parser.parse_args()

    tools = _load_tools()
    empty_ids = _empty_subject_ids(tools)
    subject_map = _subject_map()

    # Plan consistency check: every empty row must be in the subject map, and
    # every mapped id must still exist (a mapped id that is no longer empty means
    # the map is stale — surface it instead of silently skipping).
    empty_set = set(empty_ids)
    mapped_set = set(subject_map)
    unmapped = empty_set - mapped_set
    stale = mapped_set - empty_set

    print(f"empty-subject rows in DB : {len(empty_ids)}")
    print(f"rows in SUBJECT_MAP     : {len(subject_map)}")
    print(f"unmapped (DB has, map does not): {sorted(unmapped)}")
    print(f"stale   (map has, DB no longer empty): {sorted(stale)}")

    if unmapped:
        print("\nERROR: DB has empty-subject rows with no subject mapping.")
        print("Add them to SUBJECT_MAP before running --apply.")
        return 2

    plan = sorted(empty_set & mapped_set)
    if not plan:
        print("\nNothing to backfill.")
        return 0

    print(f"\n{'DRY-RUN' if not args.apply else 'APPLY'}: backfilling {len(plan)} rows")
    print(f"{'id':>5}  {'workspace':<16} {'source_type':<16} {'protection':<10} subject")
    print("-" * 88)
    db = tools.db
    with db.connection() as conn:
        rows = {
            int(r["id"]): dict(r)
            for r in conn.execute(
                "SELECT id, workspace, status, source_type, protection_level, content FROM memories WHERE id IN "
                f"({','.join('?' * len(plan))})",
                plan,
            )
        }
    validation_errors: dict[int, list[str]] = {}
    for mid in plan:
        row = rows.get(mid) or {}
        errors = _validate_row(mid, row, subject_map)
        if errors:
            validation_errors[mid] = errors
        ws = str(row.get("workspace") or "?")
        st = str(row.get("source_type") or "?")
        pl = str(row.get("protection_level") or "?")
        print(f"{mid:>5}  {ws:<16} {st:<16} {pl:<10} {subject_map[mid]}")
    if validation_errors:
        print("\nERROR: safety validation failed; refusing to backfill.")
        for mid, errors in validation_errors.items():
            for err in errors:
                print(f"  {mid}: {err}")
        return 2

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to execute.")
        return 0

    # Execute via the standard memory_edit path. user_confirmed / locked rows
    # require authorized=True (mirrors memory_supersede). new_content must equal
    # the current content (edit_memory is a content-edit API; we are only
    # changing subject, so content is passed through unchanged).
    ok, fail = 0, 0
    failed_ids: list[int] = []
    for mid in plan:
        got = tools.memory_get(memory_id=mid, sections="none")
        if not got["ok"]:
            print(f"  [fail] {mid}: memory_get failed: {got['data'].get('error')}")
            fail += 1
            failed_ids.append(mid)
            continue
        current_content = got["data"]["memory"]["content"]
        res = tools.memory_edit(
            memory_id=mid,
            new_content=current_content,
            new_subject=subject_map[mid],
            authorized=True,
            reason=REASON,
        )
        if res["ok"]:
            ok += 1
            print(f"  [ok]   {mid} -> {subject_map[mid]} (v{res['data'].get('new_version')})")
        else:
            fail += 1
            failed_ids.append(mid)
            print(f"  [fail] {mid}: {res['data'].get('error')}")

    print(f"\nDone: {ok} ok, {fail} failed")
    if failed_ids:
        print(f"Failed ids: {failed_ids}")
        return 1

    # Verify
    remaining = _empty_subject_ids(tools)
    print(f"Post-backfill empty-subject count: {len(remaining)}")
    if remaining:
        print(f"  still empty: {remaining}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
