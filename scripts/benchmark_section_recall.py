#!/usr/bin/env python3
"""Benchmark: section-vec recall (Channel 6) vs memory-vec recall (Channel 5).

Measures ACCURACY (primary) + token cost (secondary) on real long documents.

Setup: 8 long memories (>4000 chars) already split into sections. We probe each
document with queries that target content in its LATER sections — the part the
memory-level embedding (truncated to max_section_chars ≈ 3600) never saw.

For each query we compare:
  - memory-vec KNN:  does the target memory appear? at what rank? what distance?
  - section-vec KNN: does the *correct section* appear? at what rank? what distance?

Accuracy = can the channel locate the right paragraph, not just the document.
Token cost = full-text return vs matched-section-metadata + on-demand fetch.

Read-only against the real DB. No writes.

Usage:
  cd /path/to/memory-arbiter-mcp
  python scripts/benchmark_section_recall.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure the package is importable
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Use the real config + DB (read-only access only)

from memory_arbiter.tools import MemoryTools  # noqa: E402

MAX_SECTION_CHARS = 3600  # memory-level embedding truncates content to ~this

# --- Golden queries --------------------------------------------------------
# Each query targets a specific section that sits PAST the max_section_chars
# truncation point (or is a fine-grained detail that a whole-doc embedding
# would dilute). Manually authored from the document content. Each annotated
# with (memory_id, expected_section_title) so we can score precision.
QUERIES = [
    # id=141 12186 chars, 9 sections — 补充5/6/7 在 3600 字之后
    dict(q="section split 强一致发布策略，embedding 失败时怎么处理",
         mid=141, expect="补充5（P0）：强一致发布策略+embedding不可用静默降级"),
    dict(q="首次写入长记忆时用户确认后怎么一次性写入，两阶段协议字段",
         mid=141, expect="补充6（P0）：首次写入确认后一次性写入"),
    dict(q="section 命中后 memory_search 返回结构，matched_sections 字段",
         mid=141, expect="补充7（P1）：section命中返回结构"),
    # id=70 8702 chars — P-05~P-08 + 策略在后半
    dict(q="渐进式优雅降级机制，sqlite-vec 到 LIKE 到 JSONL 多级回退",
         mid=70, expect="P-08 渐进式优雅降级机制"),
    dict(q="专利组合策略，核心专利群和防御性专利怎么划分",
         mid=70, expect="专利组合策略建议"),
    # id=164 8209 chars — 第二轮 review 在后半
    dict(q="第二轮 review 修复发版阻塞，版本号 0.5.4 到 0.6.0 和 space-id 不变量",
         mid=164, expect="第二轮review-fix详情"),
    dict(q="v0.6.0 原始实现记录，PR1 到 PR8 连接工厂和 embedder 重写",
         mid=164, expect="原始实现记录"),
    # id=63 6143 chars — 外观设计/撰写流程在后半
    dict(q="外观设计专利交底书，简要说明和六视图要求",
         mid=63, expect="外观设计专利标准结构"),
    dict(q="专利撰写流程，自我审查清单和 docx 输出格式要求",
         mid=63, expect="撰写流程建议与质量标准"),
    # id=181 6141 chars, no markdown — 第三轮/守门员/provenance 在后半
    dict(q="第三轮评审逐行追踪 content 归一化，设计可进入实施",
         mid=181, expect="第三轮评审：终审通过"),
    dict(q="provenance 动态判定，parser 还是 agent，剥 # 前缀的坑",
         mid=181, expect="provenance动态判定与最终修复"),
    # id=88 5267 chars — seq27-52 表格后半
    dict(q="菁管+ 线上审批流，菁卡运管平台审批流线上化",
         mid=88, expect="seq27-38：菁管+/金采2.0/预审/风险提示"),
    dict(q="二期需求待沟通，业务负责人字段和数字人民币奖品线上变更",
         mid=88, expect="seq39-52：二期需求（待沟通）"),
    # id=79 5185 chars — 接口统计/关联在后半
    dict(q="VOP 接口数量统计，哪个模块接口最多，订单模块多少页",
         mid=79, expect="接口数量统计（92个接口）"),
    dict(q="金营平台对接 VOP 必接接口，核心下单链路和首期优先接入",
         mid=79, expect="与金营平台/金融带货的关联与接入建议"),
    # id=106 4564 chars — FTS修复/测试/dogfooding 在后半
    dict(q="FTS 同步修复，contentless FTS5 不支持 DELETE 用特殊命令",
         mid=106, expect="FTS同步修复（关键工程决策）"),
    dict(q="真实库迁移 dogfooding 验证，拷贝 100 条生产库测试",
         mid=106, expect="测试与真实库dogfooding验证"),
]


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> int:
    tools = MemoryTools()
    embedder, ew = tools._ensure_embedder()
    if embedder is None:
        print("ERROR: embedder not available — cannot run benchmark", file=sys.stderr)
        print("warnings:", ew, file=sys.stderr)
        return 1
    print(f"# Embedding model: {tools.settings.embedding_model_path}")
    print(f"# Dimension: {len(embedder.embed_text('', 'test').embedding)}")
    print(f"# Documents: {len(QUERIES)} golden queries\n")

    # Pre-load section metadata for each memory (title -> section_id, offsets)
    section_info: dict[int, dict] = {}  # mid -> {title: {id, start, end}}
    content_len: dict[int, int] = {}
    for item in QUERIES:
        mid = item["mid"]
        if mid in section_info:
            continue
        # v0.8.0: memory_get(sections="all") replaces the removed memory_split_status +
        # get_sections pair — the title catalog and offsets come back in one read-only call.
        g = tools.memory_get(mid, sections="all")["data"]
        secs = g.get("sections") or []
        content_len[mid] = len((g.get("memory") or {}).get("content", ""))
        section_info[mid] = {
            s["title"]: {"id": s["id"], "start": s["start_offset"], "end": s["end_offset"]}
            for s in secs
        }

    # --- Run queries -------------------------------------------------------
    rows = []
    mem_vec_recall_hits = 0      # memory-vec KNN returned the doc in top-k
    section_locate_correct = 0   # section-vec KNN put the *correct* section first
    section_locate_top3 = 0      # correct section in top-3 sections of that doc
    beyond_truncation = 0        # query targeted content past max_section_chars

    for item in QUERIES:
        q = item["q"]
        mid = item["mid"]
        expect_title = item["expect"]
        er = embedder.embed_text(prefix="", body=q)
        qe = er.embedding

        # 1) memory-vec KNN over ALL memories — does target doc appear?
        mem_knn = tools.db.vec_knn(qe, k=20)
        mem_ranks = {r["id"]: (i, r["distance"]) for i, r in enumerate(mem_knn)}
        mem_hit = mid in mem_ranks
        mem_rank, mem_dist = mem_ranks.get(mid, (None, None))

        # 2) section-vec KNN over ALL sections — restricted to this doc's sections
        sec_match = tools.db.section_vec_distance_match(mid, qe, threshold=2.0)
        # rank correct section among matched
        expect_sec = section_info[mid].get(expect_title)
        expect_sid = expect_sec["id"] if expect_sec else None
        sec_rank = None
        for i, h in enumerate(sec_match):
            if h["section_id"] == expect_sid:
                sec_rank = i
                break
        best_sec_dist = sec_match[0]["distance"] if sec_match else None

        # Did the target content sit beyond the memory-embedding truncation?
        if expect_sec and expect_sec["start"] >= MAX_SECTION_CHARS:
            beyond_truncation += 1

        rows.append({
            "mid": mid,
            "q": q[:42],
            "expect": expect_title[:28],
            "expect_offset": expect_sec["start"] if expect_sec else None,
            "mem_rank": mem_rank,
            "mem_dist": round(mem_dist, 3) if mem_dist is not None else None,
            "sec_rank": sec_rank,
            "sec_dist": round(best_sec_dist, 3) if best_sec_dist else None,
            "sec_match_n": len(sec_match),
        })

        if mem_hit:
            mem_vec_recall_hits += 1
        if sec_rank == 0:
            section_locate_correct += 1
        if sec_rank is not None and sec_rank < 3:
            section_locate_top3 += 1

    # --- Print results -----------------------------------------------------
    print("=" * 110)
    print(f"{'mid':>4} {'q':<44} {'memRank':>7} {'memDist':>7} | "
          f"{'secRank':>7} {'secDist':>7} {'#sec':>4}  expect_section")
    print("-" * 110)
    for r in rows:
        mr = str(r["mem_rank"]) if r["mem_rank"] is not None else "MISS"
        sr = str(r["sec_rank"]) if r["sec_rank"] is not None else "MISS"
        print(f"{r['mid']:>4} {r['q']:<44} {mr:>7} {str(r['mem_dist']):>7} | "
              f"{sr:>7} {str(r['sec_dist']):>7} {r['sec_match_n']:>4}  "
              f"@{r['expect_offset']}>{MAX_SECTION_CHARS if (r['expect_offset'] and r['expect_offset']>=MAX_SECTION_CHARS) else ''}")
    print("=" * 110)

    n = len(QUERIES)
    print(f"\n## Accuracy Summary ({n} queries, targeting fine-grained / late-section details)")
    print(f"  Memory-vec recall (doc in top-20):  {mem_vec_recall_hits}/{n} = {fmt_pct(mem_vec_recall_hits/n)}")
    print(f"  Section-vec locate correct § top-1:  {section_locate_correct}/{n} = {fmt_pct(section_locate_correct/n)}")
    print(f"  Section-vec locate correct § top-3:  {section_locate_top3}/{n} = {fmt_pct(section_locate_top3/n)}")
    print(f"  Queries targeting content past ~{MAX_SECTION_CHARS}-char truncation: {beyond_truncation}/{n}")

    # --- Token cost estimate ----------------------------------------------
    print(f"\n## Token cost (secondary): full-text vs on-demand section fetch")
    total_full = 0
    total_split = 0
    for item in QUERIES:
        mid = item["mid"]
        full = content_len[mid]
        total_full += full
        # split path: partial-match returns content=None + matched metadata.
        # Caller then fetches only the matched section(s). Estimate: 1 section
        # (the correct one) + catalog metadata overhead (~200 chars/section).
        expect_sec = section_info[mid].get(item["expect"])
        sec_size = (expect_sec["end"] - expect_sec["start"]) if expect_sec else 0
        n_sections = len(section_info[mid])
        metadata_overhead = n_sections * 200  # title+summary+catalog per section
        total_split += sec_size + metadata_overhead
    print(f"  Full-text total (no split):         {total_full:,} chars")
    print(f"  Split total (1 section + metadata): {total_split:,} chars")
    print(f"  Reduction:                          {fmt_pct(1 - total_split/total_full)}  "
          f"(avg {total_full//n} -> {total_split//n} chars/query)")

    # JSON for machine consumption
    out = ROOT / "scripts" / "benchmark_section_recall.json"
    out.write_text(json.dumps({
        "config": {
            "model": str(tools.settings.embedding_model_path),
            "max_section_chars": MAX_SECTION_CHARS,
            "n_queries": n,
        },
        "accuracy": {
            "memory_vec_recall": mem_vec_recall_hits / n,
            "section_vec_top1": section_locate_correct / n,
            "section_vec_top3": section_locate_top3 / n,
            "queries_beyond_truncation": beyond_truncation,
        },
        "token_cost": {
            "full_text_chars": total_full,
            "split_chars": total_split,
            "reduction": 1 - total_split / total_full,
        },
        "rows": rows,
    }, ensure_ascii=False, indent=2))
    print(f"\n# JSON written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
