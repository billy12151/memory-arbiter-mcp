"""v0.3.0 A/B test: bm25 (v0.2.6) vs hybrid (v0.3.0 soft rerank).

Runs the 15 baseline queries from id=41 against the real memory DB under both
RANKING_MODE=bm25 and RANKING_MODE=hybrid, and prints a side-by-side comparison
plus Top-1 / Top-3 / pairwise pass-rate statistics.

This is NOT a pytest test — it's a standalone evaluation script. Run with:
    python tests/test_v0.3.0_ab.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the package importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.search import search_memories


# === Golden queries (from id=41 baseline, expanded for pairwise) ===
# Each query has:
#   - expected_top1: the id that SHOULD be Top-1 (or None if no clear single answer)
#   - expected_top3: ids that should be in Top-3 (at least one)
#   - pairwise_before: list of (better_id, worse_id) — better_id should rank above worse_id
#   - category: which r4 §14 layer this belongs to

GOLDEN_QUERIES = [
    {
        "query": "营销交付系统",
        "category": "业务查业务 / 正文强相关但标题弱相关",
        "expected_top1": 1,
        "expected_top3": [1],
        "pairwise_before": [(1, 37), (1, 39)],  # id=1 业务资料 应排在 id=37/39 工具元前面
    },
    {
        "query": "金营平台 项目背景",
        "category": "业务查业务 / 多词中文查询",
        "expected_top1": 4,  # 金营平台-项目总览与业务背景
        "expected_top3": [4, 5, 6, 11, 12],
        "pairwise_before": [(4, 32), (4, 29), (4, 26), (4, 25), (4, 21)],
    },
    {
        "query": "金营 一期 建设成果",
        "category": "业务查业务 / 多词中文",
        "expected_top1": 5,  # 金营平台-一期建设成果
        "expected_top3": [5],
        "pairwise_before": [(5, 32), (5, 29)],
    },
    {
        "query": "金营 二期 立项",
        "category": "业务查业务 / 多词中文",
        "expected_top1": 6,  # 金营平台-二期立项规划
        "expected_top3": [6],
        "pairwise_before": [(6, 32), (6, 29)],
    },
    {
        "query": "金营 系统架构 接口",
        "category": "业务查业务 / 多词中文",
        "expected_top1": 11,  # 金营平台-系统架构与接口
        "expected_top3": [11],
        "pairwise_before": [(11, 32), (11, 29)],
    },
    {
        "query": "金融带货 VOP",
        "category": "业务查业务 / ASCII+CJK 混合",
        "expected_top1": 13,  # 金融带货-VOP核心知识
        "expected_top3": [13, 14, 15],
        "pairwise_before": [(13, 32), (13, 29)],
    },
    {
        "query": "金融带货 协议 结算",
        "category": "业务查业务 / 多词中文",
        "expected_top1": 15,  # 金融带货-协议结算与VOP方案
        "expected_top3": [15, 14],
        "pairwise_before": [(15, 21)],
    },
    {
        "query": "专利 撰写 指南",
        "category": "业务查业务 / 专利",
        "expected_top1": None,  # 专利类查询，至少应该命中专利相关记忆
        "expected_top3": [],  # 不确定具体 id，看是否至少不掉进 fallback
        "pairwise_before": [],
    },
    {
        "query": "已产出 专利 清单",
        "category": "业务查业务 / 专利",
        "expected_top1": None,
        "expected_top3": [],
        "pairwise_before": [],
    },
    {
        "query": "Codex 委托 规范",
        "category": "规范查规范",
        "expected_top1": None,  # 应命中 architect workspace 的 Codex 委托规范
        "expected_top3": [],
        "pairwise_before": [(21, 39)],  # 规范类应排在工具元前面
    },
    {
        "query": "系统工具 环境 配置",
        "category": "规范查规范 / 系统配置",
        "expected_top1": 3,  # 系统工具与环境配置
        "expected_top3": [3],
        "pairwise_before": [],
    },
    {
        "query": "memory-arbiter v0.2.6 发版",
        "category": "工具查工具",
        "expected_top1": 39,  # v0.2.6 发版完成
        "expected_top3": [39],
        "pairwise_before": [],
    },
    {
        "query": "memory-arbiter CJK trigram FTS5",
        "category": "工具查工具",
        "expected_top1": 37,  # v0.2.5 发版完成 (CJK trigram)
        "expected_top3": [37],
        "pairwise_before": [(37, 1)],  # 工具查工具时，工具元应排在业务前面
    },
    {
        "query": "记忆冲突 处理规范",
        "category": "规范查规范",
        "expected_top1": 24,  # 记忆冲突处理规范
        "expected_top3": [24],
        "pairwise_before": [],
    },
    {
        "query": "汇报策略 沟通要点",
        "category": "业务查业务 / 多词中文",
        "expected_top1": None,
        "expected_top3": [],
        "pairwise_before": [],  # 至少不该掉进工具元 fallback
    },
]


def run_eval(ranking_mode: str) -> dict:
    """Run all golden queries under the given RANKING_MODE, return stats."""
    os.environ["MEMORY_ARBITER_RANKING_MODE"] = ranking_mode
    db_path = "/Users/zhangzhiwei17/.local/share/memory-arbiter/memory.sqlite3"
    settings = Settings(db_path=Path(db_path), backup_jsonl=Path("/tmp/_unused.jsonl"))
    db = MemoryDB(settings)

    results = []
    top1_hits = 0
    top3_hits = 0
    pairwise_pass = 0
    pairwise_total = 0
    fallback_count = 0

    for g in GOLDEN_QUERIES:
        rows, warnings = search_memories(
            db, g["query"], workspace=None, tags=None, limit=10,
            include_superseded=False, debug_ranking=True,
        )
        top_ids = [r["id"] for r in rows[:3]]
        all_ids = [r["id"] for r in rows]
        is_fallback = any("recent memories" in w for w in warnings)

        # Top-1
        top1_ok = False
        if g["expected_top1"] is not None and top_ids and top_ids[0] == g["expected_top1"]:
            top1_ok = True
            top1_hits += 1
        # Top-3
        top3_ok = False
        if g["expected_top3"] and any(tid in top_ids for tid in g["expected_top3"]):
            top3_ok = True
            top3_hits += 1
        elif not g["expected_top3"]:
            top3_ok = True  # no expectation, count as pass
        # Pairwise
        pair_ok = []
        for better, worse in g["pairwise_before"]:
            pairwise_total += 1
            if better in all_ids and worse in all_ids:
                if all_ids.index(better) < all_ids.index(worse):
                    pairwise_pass += 1
                    pair_ok.append(f"{better}>{worse}✓")
                else:
                    pair_ok.append(f"{better}>{worse}✗")
            elif better in all_ids and worse not in all_ids:
                # worse not even in results — counts as pass (better ranks above by default)
                pairwise_pass += 1
                pair_ok.append(f"{better}>{worse}✓(worse absent)")
            elif better not in all_ids:
                pair_ok.append(f"{better}>{worse}?(better absent)")
        if is_fallback:
            fallback_count += 1

        results.append({
            "query": g["query"],
            "category": g["category"],
            "top_ids": top_ids,
            "top1_ok": top1_ok,
            "top3_ok": top3_ok,
            "pairwise": pair_ok,
            "is_fallback": is_fallback,
            "expected_top1": g["expected_top1"],
            "expected_top3": g["expected_top3"],
        })

    n = len(GOLDEN_QUERIES)
    return {
        "mode": ranking_mode,
        "top1_rate": top1_hits / n,
        "top3_rate": top3_hits / n,
        "pairwise_rate": pairwise_pass / pairwise_total if pairwise_total else 0,
        "fallback_rate": fallback_count / n,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "pairwise_pass": pairwise_pass,
        "pairwise_total": pairwise_total,
        "results": results,
    }


def main() -> None:
    print("=" * 80)
    print("memory-arbiter v0.3.0 A/B Evaluation: bm25 (v0.2.6) vs hybrid (v0.3.0)")
    print("=" * 80)
    print()

    bm25_eval = run_eval("bm25")
    hybrid_eval = run_eval("hybrid")

    # Side-by-side summary
    print("=== Aggregate Stats ===")
    print(f"{'Metric':<25} {'bm25 (v0.2.6)':<20} {'hybrid (v0.3.0)':<20} {'Δ':<10}")
    print("-" * 75)
    for metric in ["top1_rate", "top3_rate", "pairwise_rate", "fallback_rate"]:
        b = bm25_eval[metric]
        h = hybrid_eval[metric]
        delta = h - b
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<25} {b:<20.1%} {h:<20.1%} {sign}{delta:.1%}")
    print(f"{'top1_hits':<25} {bm25_eval['top1_hits']}/15{'':<13} {hybrid_eval['top1_hits']}/15")
    print(f"{'pairwise_pass':<25} {bm25_eval['pairwise_pass']}/{bm25_eval['pairwise_total']}{'':<13} {hybrid_eval['pairwise_pass']}/{hybrid_eval['pairwise_total']}")
    print()

    # Per-query detail
    print("=== Per-Query Detail ===")
    print(f"{'Query':<32} {'bm25 top3':<22} {'hybrid top3':<22} {'Δ Top1':<8}")
    print("-" * 90)
    for b, h in zip(bm25_eval["results"], hybrid_eval["results"]):
        q = b["query"][:30]
        bt = str(b["top_ids"])
        ht = str(h["top_ids"])
        d1 = ""
        if b["top1_ok"] != h["top1_ok"]:
            d1 = "↑" if h["top1_ok"] else "↓"
        elif h["top1_ok"]:
            d1 = "="
        print(f"{q:<32} {bt:<22} {ht:<22} {d1:<8}")
    print()

    # Pairwise detail
    print("=== Pairwise Constraint Results ===")
    for b, h in zip(bm25_eval["results"], hybrid_eval["results"]):
        if not b["pairwise"] and not h["pairwise"]:
            continue
        print(f"  {b['query']}")
        print(f"    bm25:   {b['pairwise']}")
        print(f"    hybrid: {h['pairwise']}")
    print()

    # Fallback comparison
    print("=== Fallback Cases (no direct match) ===")
    for b, h in zip(bm25_eval["results"], hybrid_eval["results"]):
        if b["is_fallback"] or h["is_fallback"]:
            print(f"  {b['query']}: bm25_fallback={b['is_fallback']} hybrid_fallback={h['is_fallback']}")


if __name__ == "__main__":
    main()
