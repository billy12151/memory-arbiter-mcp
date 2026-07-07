"""Three-way A/B test: bm25 (v0.2.6) vs hybrid (v0.3.0) vs hybrid+semantic (v0.3.1).

Reuses the 15 golden queries + 18 pairwise constraints from docs/v0_3_0_ab.py
so all three modes are scored against the SAME criteria. Adds a third mode that
supplies query_embedding from the local GGUF embedder.

Run after backfilling embeddings via docs/semantic_example.py.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs"))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.search import search_memories  # noqa: E402
from semantic_example import load_gguf_embedder  # noqa: E402

# The v0.3.0 A/B script is a standalone evaluation script, so load it by path
# and reuse its golden-query set + lexical eval function. This keeps the three
# modes scored against the SAME 15 queries + 18 pairwise constraints.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "v030_ab", REPO / "docs" / "v0_3_0_ab.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GOLDEN_QUERIES = _mod.GOLDEN_QUERIES
run_eval = _mod.run_eval


def run_eval_semantic(encode_fn) -> dict:
    """Run golden queries under hybrid mode + query_embedding (v0.3.1)."""
    os.environ["MEMORY_ARBITER_RANKING_MODE"] = "hybrid"
    db_path = "/Users/zhangzhiwei17/.local/share/memory-arbiter/memory.sqlite3"
    settings = Settings(db_path=Path(db_path), backup_jsonl=Path("/tmp/_unused.jsonl"))
    db = MemoryDB(settings)

    results = []
    top1_hits = 0
    top3_hits = 0
    pairwise_pass = 0
    pairwise_total = 0
    fallback_count = 0
    latencies = []

    for g in GOLDEN_QUERIES:
        qemb = encode_fn(g["query"])
        t0 = time.time()
        rows, warnings = search_memories(
            db, g["query"], workspace=None, tags=None, limit=10,
            include_superseded=False, debug_ranking=True, query_embedding=qemb,
        )
        latencies.append((time.time() - t0) * 1000)
        top_ids = [r["id"] for r in rows[:3]]
        all_ids = [r["id"] for r in rows]
        is_fallback = any("recent memories" in w for w in warnings)

        top1_ok = False
        if g["expected_top1"] is not None and top_ids and top_ids[0] == g["expected_top1"]:
            top1_ok = True
            top1_hits += 1
        top3_ok = False
        if g["expected_top3"] and any(tid in top_ids for tid in g["expected_top3"]):
            top3_ok = True
            top3_hits += 1
        elif not g["expected_top3"]:
            top3_ok = True
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
                pairwise_pass += 1
                pair_ok.append(f"{better}>{worse}✓(worse absent)")
            elif better not in all_ids:
                pair_ok.append(f"{better}>{worse}?(better absent)")
        if is_fallback:
            fallback_count += 1
        results.append({"query": g["query"], "top_ids": top_ids, "pairwise": pair_ok})

    n = len(GOLDEN_QUERIES)
    return {
        "mode": "hybrid+semantic",
        "top1_rate": top1_hits / n,
        "top3_rate": top3_hits / n,
        "pairwise_rate": pairwise_pass / pairwise_total if pairwise_total else 0,
        "fallback_rate": fallback_count / n,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "pairwise_pass": pairwise_pass,
        "pairwise_total": pairwise_total,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "results": results,
    }


def main() -> None:
    print("=" * 80)
    print("Three-way A/B: bm25 (v0.2.6) vs hybrid (v0.3.0) vs hybrid+semantic (v0.3.1)")
    print("=" * 80)
    print()

    # Load embedder for the semantic mode.
    model_path = os.getenv("MEMORY_ARBITER_GGUF", os.path.expanduser(
        "~/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf"))
    print(f"Loading embedder: {model_path}", file=sys.stderr)
    encode_fn, dim = load_gguf_embedder(model_path)
    print(f"Embedder ready (dim={dim}).\n", file=sys.stderr)

    bm25_eval = run_eval("bm25")
    hybrid_eval = run_eval("hybrid")
    sem_eval = run_eval_semantic(encode_fn)

    n = len(GOLDEN_QUERIES)
    print(f"{'Metric':<22}  {'bm25 (v0.2.6)':<18}  {'hybrid (v0.3.0)':<18}  {'hybrid+sem (v0.3.1)':<18}")
    print("-" * 80)
    print(f"{'Top-1 hit rate':<22}  {bm25_eval['top1_hits']}/{n} ({100*bm25_eval['top1_rate']:.1f}%){'':<5}  {hybrid_eval['top1_hits']}/{n} ({100*hybrid_eval['top1_rate']:.1f}%){'':<5}  {sem_eval['top1_hits']}/{n} ({100*sem_eval['top1_rate']:.1f}%)")
    print(f"{'Top-3 hit rate':<22}  {bm25_eval['top3_hits']}/{n} ({100*bm25_eval['top3_rate']:.1f}%){'':<5}  {hybrid_eval['top3_hits']}/{n} ({100*hybrid_eval['top3_rate']:.1f}%){'':<5}  {sem_eval['top3_hits']}/{n} ({100*sem_eval['top3_rate']:.1f}%)")
    pt = sem_eval['pairwise_total']
    print(f"{'Pairwise pass rate':<22}  {bm25_eval['pairwise_pass']}/{pt} ({100*bm25_eval['pairwise_rate']:.1f}%){'':<3}  {hybrid_eval['pairwise_pass']}/{pt} ({100*hybrid_eval['pairwise_rate']:.1f}%){'':<3}  {sem_eval['pairwise_pass']}/{pt} ({100*sem_eval['pairwise_rate']:.1f}%)")
    print(f"{'Fallback rate':<22}  {100*bm25_eval['fallback_rate']:.1f}%{'':<13}  {100*hybrid_eval['fallback_rate']:.1f}%{'':<13}  {100*sem_eval['fallback_rate']:.1f}%")

    print()
    print("Per-query Top-3 (bm25 / hybrid / hybrid+sem):")
    print("-" * 80)
    for i, (b, h, s) in enumerate(zip(bm25_eval["results"], hybrid_eval["results"], sem_eval["results"]), 1):
        print(f"{i:>2}  {b['query']:<28}  {str(b['top_ids']):<16}  {str(h['top_ids']):<16}  {str(s['top_ids']):<16}")


if __name__ == "__main__":
    main()
