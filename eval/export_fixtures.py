#!/usr/bin/env python3
"""export_fixtures — P0-1 harness 召回语料导出工具（c1，方案 §2.3）.

从真库只读导出召回评测 fixtures：

  targets.jsonl      34 query 的被标 target 记忆（envelope 快照，content_sha 寻址）
  labels.jsonl       qid × fixture_key × label 映射（源 docs/eval/labels.jsonl，
                     未标注即无关的当年口径由 runner 侧实现，本文件只载非无关标注）
  distractors.jsonl  陪衬记忆（active，排除 twin 桶与 targets，确定性采样）
  manifest.json      导出指纹（源库路径、行数、语料版本、导出时间）

只读：以 immutable=1 打开源库（mode=ro 打不开带 WAL 的活库），不写任何字节。

设计要点：
  - 内容寻址：fixture 主键 = 't-' + content_sha[:12]，不绑真库自增 ID（真库
    演化后 ID 失效、内容不失效；source_id 仅作导出审计保留）；
  - qid 覆盖断言：每个在 labels 中出现的 qid 至少映射 1 个 target，缺失显式
    报错退出而非静默跳过（方案 §2.7-5）；
  - twin 桶排除：workspace_canonical 为 mema-twin 的行不进陪衬（对齐当年
    eval_relevance_floor.py 的 exclude_ws 口径）；
  - 确定性采样：陪衬按 id 升序均匀步长抽取，同源库重跑结果一致。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABELS_SOURCE = REPO / "docs/eval/labels.jsonl"
DEFAULT_SOURCE = Path.home() / ".local/share/memory-arbiter/memory.sqlite3"
CORPUS_VERSION = "recall-v1"
DEFAULT_DISTRACTORS = 200

ENVELOPE_FIELDS = (
    "content", "subject", "tags", "workspace", "workspace_canonical",
    "event_time", "source_type", "metadata",
)


def _open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?immutable=1", uri=True)


def _row_to_envelope(row: sqlite3.Row) -> dict:
    tags = json.loads(row["tags"]) if row["tags"] else []
    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    return {
        "fixture_key": "t-" + str(row["content_sha"])[:12],
        "source_id": int(row["id"]),
        "content_sha": row["content_sha"],
        "content": row["content"],
        "subject": row["subject"],
        "tags": tags,
        "workspace": row["workspace"],
        "workspace_canonical": row["workspace_canonical"],
        "event_time": row["event_time"],
        "source_type": row["source_type"],
        "metadata": metadata,
        "version_at_export": int(row["version"]),
        "status_at_export": row["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=REPO / "eval/fixtures/recall")
    parser.add_argument("--labels", type=Path, default=LABELS_SOURCE)
    parser.add_argument("--distractors", type=int, default=DEFAULT_DISTRACTORS)
    args = parser.parse_args()

    if not args.source.exists():
        print(f"error: source db not found: {args.source}", file=sys.stderr)
        return 2
    if not args.labels.exists():
        print(f"error: labels archive not found: {args.labels}", file=sys.stderr)
        return 2

    labels: list[dict] = [
        json.loads(line)
        for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target_ids = sorted({int(r["mem_id"]) for r in labels})
    labeled_qids = sorted({r["qid"] for r in labels})

    args.out.mkdir(parents=True, exist_ok=True)
    conn = _open_ro(args.source)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(target_ids))
        target_rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", target_ids,
        ).fetchall()
        rows_by_id = {int(r["id"]): r for r in target_rows}
        missing_ids = [i for i in target_ids if i not in rows_by_id]
        if missing_ids:
            print(f"error: labeled mem_id absent from source: {missing_ids}", file=sys.stderr)
            return 2

        targets = [_row_to_envelope(rows_by_id[i]) for i in target_ids]

        pool = conn.execute(
            "SELECT id FROM memories WHERE status='active' "
            "AND (workspace_canonical IS NULL OR workspace_canonical != 'mema-twin') "
            f"AND id NOT IN ({placeholders}) ORDER BY id",
            target_ids,
        ).fetchall()
        pool_ids = [int(r["id"]) for r in pool]
        step = max(1, len(pool_ids) // max(1, args.distractors))
        picked = pool_ids[::step][: args.distractors]
        distractor_rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({','.join('?' * len(picked))})",
            picked,
        ).fetchall()
        distractors_by_id = {int(r["id"]): r for r in distractor_rows}
        distractors = [_row_to_envelope(distractors_by_id[i]) for i in picked]
    finally:
        conn.close()

    # qid 覆盖断言：每个有标注的 qid 至少 1 个 target（§2.7-5）
    key_by_id = {t["source_id"]: t["fixture_key"] for t in targets}
    qid_keys: dict[str, list[str]] = {}
    for row in labels:
        qid_keys.setdefault(row["qid"], []).append(key_by_id[int(row["mem_id"])])
    uncovered = [q for q in labeled_qids if not qid_keys.get(q)]
    if uncovered:
        print(f"error: qids with no target fixture: {uncovered}", file=sys.stderr)
        return 2

    def _dump(path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    _dump(args.out / "targets.jsonl", targets)
    _dump(args.out / "distractors.jsonl", distractors)
    _dump(
        args.out / "labels.jsonl",
        [
            {"qid": r["qid"], "fixture_key": key_by_id[int(r["mem_id"])], "label": r["label"]}
            for r in labels
        ],
    )
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_db": str(args.source),
        "labels_source": str(args.labels),
        "targets": len(targets),
        "distractors": len(distractors),
        "labeled_qids": len(labeled_qids),
        "unlabeled_qids_default_irrelevant": "C01-C08,D01-D04",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"[export] targets={len(targets)} distractors={len(distractors)} "
        f"labeled_qids={len(labeled_qids)} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
