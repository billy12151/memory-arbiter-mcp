"""A/B test: lexical recall (v0.3.0) vs lexical + semantic recall (v0.3.1).

Uses the 15 golden queries from the v0.2.6 baseline (memory id=41) to measure
whether semantic recall improves Top-1 / Top-3 hit rate. Also reports per-query
latency for both modes.

Run after backfilling embeddings via docs/semantic_example.py.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.tools import MemoryTools  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_example import load_gguf_embedder  # noqa: E402

# 15 golden queries with their expected-top memory id (from baseline id=41).
# Format: (query, expected_id, notes)
GOLDEN_QUERIES = [
    ("营销交付系统", 1, "业务资料"),
    ("金营平台 项目背景", 1, "业务资料"),
    ("金营 一期 建设成果", 1, "业务资料"),
    ("金营 二期 立项", 1, "业务资料"),
    ("金营 系统架构 接口", 1, "业务资料"),
    ("金融带货 VOP", 1, "业务资料"),
    ("金融带货 协议 结算", 1, "业务资料"),
    ("专利 撰写 指南", 22, "专利规范类"),
    ("已产出 专利 清单", 22, "专利规范类"),
    ("Codex 委托 规范", 2, "codex 规范"),
    ("系统工具 环境 配置", 3, "环境配置"),
    ("memory-arbiter v0.3.0 发版", 44, "发版记录"),
    ("memory-arbiter 宽召回 软重排", 44, "发版记录"),
    ("记忆冲突 处理规范", 24, "冲突规范"),
    ("汇报策略 沟通要点", 21, "汇报规范"),
]


def run_mode(tools, encode_fn, query, use_semantic, limit=5):
    """Run one query, return (top_ids, elapsed_ms)."""
    qemb = encode_fn(query) if use_semantic else None
    t0 = time.time()
    result = tools.memory_search(query=query, query_embedding=qemb, limit=limit)
    elapsed = (time.time() - t0) * 1000
    ids = [r["id"] for r in result.get("data", {}).get("results", [])]
    return ids, elapsed


def main() -> None:
    settings = Settings.from_env()
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))

    model_path = os.getenv("MEMORY_ARBITER_GGUF", os.path.expanduser(
        "~/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf"))
    print(f"Loading embedder: {model_path}", file=sys.stderr)
    encode_fn, dim = load_gguf_embedder(model_path)
    print(f"Embedder ready (dim={dim}).\n", file=sys.stderr)

    lex_top1, lex_top3 = 0, 0
    sem_top1, sem_top3 = 0, 0
    lex_latencies, sem_latencies = [], []
    flipped = []  # queries where semantic changed the result

    print(f"{'#':>2}  {'query':<28}  {'exp':>4}  {'lex-top3':<16}  {'sem-top3':<16}  {'Δ':<8}")
    print("-" * 90)
    for i, (query, expected, note) in enumerate(GOLDEN_QUERIES, 1):
        lex_ids, lex_ms = run_mode(tools, encode_fn, query, use_semantic=False)
        sem_ids, sem_ms = run_mode(tools, encode_fn, query, use_semantic=True)
        lex_latencies.append(lex_ms)
        sem_latencies.append(sem_ms)

        lex_hit1 = lex_ids and lex_ids[0] == expected
        lex_hit3 = expected in lex_ids[:3]
        sem_hit1 = sem_ids and sem_ids[0] == expected
        sem_hit3 = expected in sem_ids[:3]
        lex_top1 += int(lex_hit1)
        lex_top3 += int(lex_hit3)
        sem_top1 += int(sem_hit1)
        sem_top3 += int(sem_hit3)

        if lex_ids != sem_ids:
            flipped.append(i)
        delta = ""
        if sem_hit3 and not lex_hit3:
            delta = "GAINED"
        elif not sem_hit3 and lex_hit3:
            delta = "LOST"

        print(f"{i:>2}  {query:<28}  {expected:>4}  {str(lex_ids[:3]):<16}  {str(sem_ids[:3]):<16}  {delta:<8}")

    n = len(GOLDEN_QUERIES)
    print("-" * 90)
    print(f"\n{'指标':<24}  {'纯字面 (v0.3.0)':<18}  {'+语义 (v0.3.1)':<18}  {'Δ'}")
    print("-" * 70)
    print(f"{'Top-1 命中率':<22}  {lex_top1}/{n} ({100*lex_top1/n:.1f}%)       {sem_top1}/{n} ({100*sem_top1/n:.1f}%)       {(sem_top1-lex_top1)/n*100:+.1f}pp")
    print(f"{'Top-3 命中率':<22}  {lex_top3}/{n} ({100*lex_top3/n:.1f}%)       {sem_top3}/{n} ({100*sem_top3/n:.1f}%)       {(sem_top3-lex_top3)/n*100:+.1f}pp")
    print(f"{'平均查询延迟 (ms)':<20}  {sum(lex_latencies)/n:.1f}              {sum(sem_latencies)/n:.1f}              {(sum(sem_latencies)-sum(lex_latencies))/n:+.1f}")
    print(f"{'结果发生变化的查询':<19}  {len(flipped)} 个: {flipped}")


if __name__ == "__main__":
    main()
