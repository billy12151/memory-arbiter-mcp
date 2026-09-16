#!/usr/bin/env python3
"""export_conflict_pairs — P0-1 c4 冲突对集导出（方案 §2.4，D3 拍板路线）.

从真库治理历史选对（owner 不做全量标注：resolved 组信任 owner 判定，
其余施工侧自标 + 定稿后抽 5~8 条 owner 过目）：

  raw 草稿（本工具产出，供人工复核）：
    eval/fixtures/conflict/pairs.raw.jsonl

  取材三源：
    1. conflicts.status='resolved'   → label=true_conflict（owner 已判），
       成员 memory_id 展开成对；只留同 workspace 对（semantic notice 成组
       硬约束：mixed-workspace 快照会被 workspace_mismatch 拒绝）
    2. conflicts.status='not_a_conflict' → label=noise（owner 驳回=非冲突；
       这些是「扫出来像、判过不是」的最强干扰对），id 均匀采样
    3. coexist 构造候选：active 记忆中同 workspace、subject 词面重叠但
       属性不同的对（label 待人工复核为 coexist / noise）

只读：immutable=1。每对带原文正文（重放需要）与出处（group id + 判定语）。
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path.home() / ".local/share/memory-arbiter/memory.sqlite3"
OUT_DIR = REPO / "eval" / "fixtures" / "conflict"
NOISE_SAMPLE = 25
COEXIST_CANDIDATES = 20
_CJK = re.compile(r"[\u3400-\u9fff]")


def _open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?immutable=1", uri=True)


def _subject_overlap(a: str, b: str) -> float:
    """粗粒度主题重叠（2-gram Jaccard）——coexist 候选初筛用，不参与产品逻辑."""
    def grams(text: str) -> set[str]:
        clean = re.sub(r"\s+", "", text or "")
        return {clean[i:i + 2] for i in range(len(clean) - 1)} if len(clean) > 1 else {clean}

    ga, gb = grams(a), grams(b)
    return len(ga & gb) / len(ga | gb) if ga | gb else 0.0


def _memory_envelope(conn: sqlite3.Connection, memory_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id,content,subject,tags,workspace,"
        "COALESCE(NULLIF(workspace_canonical,''),workspace) AS ws,"
        "event_time,source_type,metadata,version,status FROM memories WHERE id=?",
        (memory_id,),
    ).fetchone()
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        rec = {name: row[name] for name in row.keys()}
    else:
        rec = dict(zip(
            ["id", "content", "subject", "tags", "workspace", "ws",
             "event_time", "source_type", "metadata", "version", "status"], row,
        ))
    return {
        "memory_id": int(rec["id"]),
        "content": rec["content"],
        "subject": rec["subject"],
        "tags": json.loads(rec["tags"]) if rec["tags"] else [],
        "workspace": rec["ws"],
        "event_time": rec["event_time"],
        "source_type": rec["source_type"],
        "metadata": json.loads(rec["metadata"]) if rec["metadata"] else {},
        "version": int(rec["version"]),
        "status": rec["status"],
    }


def _pairs_from_group(conn: sqlite3.Connection, group: sqlite3.Row) -> list[tuple[dict, dict]]:
    members = json.loads(group["member_versions"])
    ids = sorted({int(m["memory_id"]) for m in members})
    envs = [_memory_envelope(conn, mid) for mid in ids]
    envs = [env for env in envs if env]
    pairs = []
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            left, right = envs[i], envs[j]
            if left["workspace"] != right["workspace"]:
                continue  # 成组硬约束
            pairs.append((left, right))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--noise-sample", type=int, default=NOISE_SAMPLE)
    parser.add_argument("--coexist-candidates", type=int, default=COEXIST_CANDIDATES)
    args = parser.parse_args()
    conn = _open_ro(args.source)
    conn.row_factory = sqlite3.Row

    records: list[dict] = []

    # 1) resolved → true_conflict（owner 判定）
    resolved = conn.execute(
        "SELECT id, member_versions, conflict_point, detection_reason "
        "FROM conflicts WHERE status='resolved' ORDER BY id",
    ).fetchall()
    for group in resolved:
        for left, right in _pairs_from_group(conn, group):
            records.append({
                "pair_id": f"cf-res-{group['id']}-{left['memory_id']}-{right['memory_id']}",
                "label_draft": "true_conflict",
                "label_source": "owner_resolved",
                "origin": f"conflicts#{group['id']} resolved",
                "adjudication": (group["detection_reason"] or "")[:200],
                "left": left, "right": right,
            })

    # 2) not_a_conflict → noise（owner 驳回；均匀采样）
    dismissed = conn.execute(
        "SELECT id, member_versions, conflict_point, detection_reason FROM conflicts "
        "WHERE status='not_a_conflict' ORDER BY id",
    ).fetchall()
    if dismissed:
        step = max(1, len(dismissed) // args.noise_sample)
        for group in dismissed[::step][: args.noise_sample]:
            for left, right in _pairs_from_group(conn, group):
                records.append({
                    "pair_id": f"cf-noise-{group['id']}-{left['memory_id']}-{right['memory_id']}",
                    "label_draft": "noise",
                    "label_source": "owner_dismissed",
                    "origin": f"conflicts#{group['id']} not_a_conflict",
                    "adjudication": (group["detection_reason"] or "")[:200],
                    "left": left, "right": right,
                })
                break  # 每组取一对即可（noise 用于考干扰，无需全展开）

    # 3) coexist 候选：同 workspace、主题重叠 0.3~0.75、均 active 的对
    actives = conn.execute(
        "SELECT id, subject, COALESCE(NULLIF(workspace_canonical,''),workspace) AS ws "
        "FROM memories WHERE status='active' AND subject IS NOT NULL AND TRIM(subject)<>'' "
        "AND COALESCE(NULLIF(workspace_canonical,''),workspace) != 'mema-twin'",
    ).fetchall()
    by_ws: dict[str, list[sqlite3.Row]] = {}
    for row in actives:
        by_ws.setdefault(row["ws"], []).append(row)
    picked = 0
    seen_ids: set[int] = set()
    for ws, rows in sorted(by_ws.items()):
        rows = sorted(rows, key=lambda r: int(r["id"]))
        i = 0
        while picked < args.coexist_candidates and i < len(rows) - 1:
            a, b = rows[i], rows[i + 1]
            overlap = _subject_overlap(a["subject"], b["subject"])
            if 0.30 <= overlap <= 0.75 and a["id"] not in seen_ids and b["id"] not in seen_ids:
                left, right = _memory_envelope(conn, int(a["id"])), _memory_envelope(conn, int(b["id"]))
                if left and right:
                    records.append({
                        "pair_id": f"cf-coexist-{int(a['id'])}-{int(b['id'])}",
                        "label_draft": "coexist?",
                        "label_source": "auto_subject_overlap",
                        "origin": f"active 同库主题相邻对 overlap={overlap:.2f}",
                        "adjudication": "",
                        "left": left, "right": right,
                    })
                    seen_ids.update({int(a["id"]), int(b["id"])})
                    picked += 1
            i += 2

    conn.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pairs.raw.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[export] {len(records)} pairs -> {out}")
    print("draft labels:", Counter(r["label_draft"] for r in records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
