#!/usr/bin/env python3
"""probe_keyword_bands — 检索线（关键词模式+余弦档位）标定探针.

对 (query, fixture_key) 对计算 target 记忆的 best-row TRUE cosine（与
search.py G2 完全同口径：auto-embed → row_knn 800 行窗 → best 行向量
vector_cosine）。两个用途：
  1. P1-6 核验：45 对 corpus (query, relevant) 里有没有「已进 top10 但
     best cos < COS_RECALL_FLOOR」的对（准入线唯一可能炸 R@10 的形态）；
  2. K 组出题：候选关键词 query × 目标 target 的 band 归属（midband /
     under_floor / above），定稿写进 queries.json 的 expected_band。

用法：
  .venv/bin/python eval/probe_keyword_bands.py --pairs <pairs.jsonl> --out <out.json>
pairs.jsonl 每行 {"pair_id", "query", "fixture_key", "expect_rank_ok"(可选)}。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

import runner  # noqa: E402

from memory_arbiter.semantic_conflict import vector_cosine  # noqa: E402


def probe_pairs(tools, pairs: list[dict], id_map: dict[str, int]) -> list[dict]:
    warnings: list[str] = []
    by_query: dict[str, list[dict]] = {}
    for pair in pairs:
        by_query.setdefault(pair["query"], []).append(pair)
    out: list[dict] = []
    for query, group in by_query.items():
        qvec = tools._read_pipeline._auto_embed(query, None, warnings)
        if not qvec:
            for pair in group:
                out.append({**pair, "best_cos": None, "note": "embed_failed"})
            continue
        rows = tools.db.row_knn(qvec, k=800, parent_status_filter="active")
        best_row: dict[int, int] = {}
        for row in rows:
            mid = row.get("memory_id")
            if mid is None:
                continue
            mid = int(mid)
            if mid not in best_row:
                best_row[mid] = int(row["id"])
        vecs = tools.db.evidence.row_vectors_for_ids(list(best_row.values()))
        cos_by_mid: dict[int, float | None] = {}
        for mid, rid in best_row.items():
            vec = vecs.get(rid)
            cos_by_mid[mid] = None if not vec else float(vector_cosine(qvec, vec))
        for pair in group:
            mid = id_map.get(pair["fixture_key"])
            if mid is None:
                out.append({**pair, "best_cos": None, "note": "fixture_not_in_library"})
                continue
            cos = cos_by_mid.get(mid)
            note = "" if cos is not None else "no_vector_row"
            out.append({
                **pair,
                "best_cos": None if cos is None else round(cos, 4),
                "note": note,
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--keep-db", type=Path, default=None)
    args = parser.parse_args()

    pairs = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embed_model = runner.default_embed_model()
    if embed_model is None:
        print("error: embedder model required", file=sys.stderr)
        return 2
    recall_dir = runner.FIXTURES / "recall"
    targets = runner._load_jsonl(recall_dir / "targets.jsonl")
    distractors = runner._load_jsonl(recall_dir / "distractors.jsonl")
    with runner.temp_library(embed_model, keep_db=args.keep_db) as tools:
        id_map, _ = runner.replay_fixtures(tools, targets + distractors)
        results = probe_pairs(tools, pairs, id_map)
    payload = {"pairs": results}
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"[probe] -> {args.out}")
    for row in results:
        print(row["pair_id"], row["fixture_key"], row["best_cos"], row.get("note", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
