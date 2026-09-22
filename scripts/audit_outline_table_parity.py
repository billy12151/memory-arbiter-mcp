#!/usr/bin/env python3
"""P1-T5 step 1（必做审计）：重解析 outline vs memory_evidence 查表 outline 全量比对.

对真实库副本的全部 active 记忆，逐条比较「local_text_units 重解析（_content_outline
的现行数据源）」与「memory_evidence 表查表（candidate 数据源）」两种 outline 的
(head 截断, offset) 与「还有 N 段」计数。**不一致率 > 0 → 本任务关闭**
（数据归档方案附录 A），维持重解析。

用法：
  python scripts/audit_outline_table_parity.py --db <真实库副本路径>
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.evidence import local_text_units  # noqa: E402

_OUTLINE_MAX_SEGMENTS = 8
_OUTLINE_HEAD_CHARS = 40
_KINDS = ("heading", "text")


def _reparse_outline(subject: str, content: str) -> list[tuple[str, int | None]]:
    units = [u for u in local_text_units(subject, content) if u.kind in _KINDS]
    out = [
        ((u.text.splitlines()[0] if u.text else "")[:_OUTLINE_HEAD_CHARS], u.start_offset)
        for u in units[:_OUTLINE_MAX_SEGMENTS]
    ]
    if len(units) > _OUTLINE_MAX_SEGMENTS:
        out.append((f"…还有 {len(units) - _OUTLINE_MAX_SEGMENTS} 段", None))
    return out


def _table_outline(rows: list[sqlite3.Row]) -> list[tuple[str, int | None]]:
    out = []
    for row in rows[:_OUTLINE_MAX_SEGMENTS]:
        text = str(row["text"] or "")
        out.append(((text.splitlines()[0] if text else "")[:_OUTLINE_HEAD_CHARS],
                    int(row["start_offset"])))
    if len(rows) > _OUTLINE_MAX_SEGMENTS:
        out.append((f"…还有 {len(rows) - _OUTLINE_MAX_SEGMENTS} 段", None))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="真实库副本路径")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    memories = conn.execute(
        "SELECT id, version, subject, content FROM memories WHERE status='active' ORDER BY id",
    ).fetchall()
    total = len(memories)
    compared = mismatched = no_units = stale = 0
    samples: list[str] = []
    for mem in memories:
        rows = conn.execute(
            "SELECT unit_index, kind, text, start_offset FROM memory_evidence "
            "WHERE memory_id=? AND memory_version=? AND kind IN ('heading','text') "
            "ORDER BY unit_index",
            (mem["id"], mem["version"]),
        ).fetchall()
        if not rows:
            has_any = conn.execute(
                "SELECT 1 FROM memory_evidence WHERE memory_id=? LIMIT 1", (mem["id"],),
            ).fetchone()
            if has_any:
                stale += 1  # 有旧行但当前版本无（编辑后未重发布）
            else:
                no_units += 1
            continue
        compared += 1
        left = _reparse_outline(str(mem["subject"] or ""), str(mem["content"] or ""))
        right = _table_outline(rows)
        if left != right:
            mismatched += 1
            if len(samples) < 10:
                diff = next(
                    (f"reparse={a!r} vs table={b!r}" for a, b in zip(left, right) if a != b),
                    f"length {len(left)} vs {len(right)}",
                )
                samples.append(f"  memory {mem['id']}@{mem['version']}: {diff}")
    conn.close()
    print(f"active={total} compared={compared} mismatched={mismatched} "
          f"no_units={no_units} stale_version={stale}")
    for line in samples:
        print(line)
    rate = mismatched / compared if compared else 0.0
    print(f"不一致率（compared 口径）= {rate:.4%}  → {'关闭任务（维持重解析）' if mismatched else '审计通过，可查表化'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
