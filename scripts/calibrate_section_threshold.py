#!/usr/bin/env python3
"""Calibrate section_vec_distance_threshold using real corpus + real model.

One-shot offline tool. Reads two long, well-structured memories from the
production DB, splits them at their real Markdown H2 boundaries (the same
logic production uses), embeds each section + a hand-written natural-language
query with the real GGUF model, and measures the cosine-distance distribution
of positive (query ↔ its target section) vs negative pairs.

Read-only: never writes to the DB. Not packaged (not in MANIFEST.in).

Run:
    cd <project root>
    python scripts/calibrate_section_threshold.py
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — resolved relative to the project root (parent of scripts/).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path.home() / ".local" / "share" / "memory-arbiter" / "memory.sqlite3"
MODEL_PATH = Path.home() / ".node-llama-cpp" / "models" / "hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf"
N_CTX = 2048
MAX_SECTION_CHARS = 3600  # matches Settings.max_section_chars default

# The two calibration source memories (long, clean H2 structure).
SOURCE_MEMORY_IDS = [141, 70]

# ---------------------------------------------------------------------------
# Hand-written natural-language queries.
#
# Each query is phrased the way a user would actually ask — NOT using the
# document's own vocabulary — and mapped to the H2 section heading it should
# match. A section is identified by the exact text of its `## ` heading line.
# These drive the POSITIVE pairs (query ↔ its target section).
#
# The queries deliberately paraphrase: e.g. the section "补充 1（P0）：召回融合"
# gets the query "检索的多个通道结果怎么去重合并", which shares almost no
# literal surface tokens with the heading/body — this is the hard case the
# threshold must survive in production.
# ---------------------------------------------------------------------------

# id=141 — "## 补充N" sections (plus 背景 / 与 id140 的关系)
QUERIES_141 = {
    "背景": "这篇文章要解决什么问题、和 id140 是什么关系",
    "补充 1（P0）：召回融合——复用 _wide_recall + _soft_rerank，新增 section 向量通道": "检索的多个通道结果怎么去重合并到一个候选池",
    "补充 2（P0）：FTS→section 映射——用字符偏移落点，不存在 token↔字符单位问题": "全文检索命中的词怎么对应到具体段落，字符偏移怎么算",
    "补充 3（P0）：section 重建事务边界——单 commit，re-embed 失败删旧向量": "重新生成段落向量时事务怎么写，失败了怎么回滚不残留",
    "补充 4（修正）：拆分是 LLM 驱动 + 用户确认的闭环，准确性由 LLM 保证——撤回原\"拆分成功率\"担忧": "长文档拆分成段这件事谁保证准确性，要不要统计成功率",
    "补充 5（P0）：section split 采用强一致发布策略；embedding 不可用时静默降级": "向量生成失败时还能不能用分段检索，要不要强行启用",
    "补充 6（P0）：首次写入采用“确认后一次性写入”，LLM 由当前 Agent 调用": "超长记忆第一次保存时要不要先问用户、写库流程分几步",
    "补充 7（P1）：section 命中返回结构——memory 级 limit 不等于返回整条长文": "搜到一条分段的长记忆时返回什么结构才不浪费 token",
    "与 id140 的关系": "这一条和架构定稿基线之间是什么关系，会不会推翻原决策",
}

# id=70 — "## P-0N" patent points (plus 总体评估 / 专利组合策略建议)
QUERIES_70 = {
    "总体评估": "这个项目一共能提炼出几个专利点，哪些值得重点申请",
    "P-01 宽召回多通道检索与软重排混合排序方法（★ 重点）": "怎么在不依赖机器学习模型的情况下做多路召回和打分排序",
    "P-02 双时间轴冲突仲裁方法与可解释裁决引擎（★ 重点）": "两条记忆信息冲突时按什么规则自动裁决，事件时间和录入时间怎么比",
    "P-03 CJK 自适应 FTS5 查询构建方法（★ 重点）": "中文短词在 trigram 全文检索里查不到怎么办，查询怎么改写",
    "P-04 规则锚点提取与多级匹配分类方法（★ 重点）": "不装分词器怎么从中文里提取关键词并分强弱等级",
    "P-05 记忆替代状态管理与链完整性防护机制": "记忆被新版本替代后怎么管理状态，怎么防止引用链断掉",
    "P-06 信任优先的分层回退排序机制": "搜索一个词什么都没匹配到时，按什么顺序回退展示结果",
    "P-07 多 AI 代理记忆共享系统与 Token 优化方法": "多个 AI 编程工具怎么共用一套记忆，怎么省 token",
    "P-08 渐进式优雅降级机制": "运行环境缺这少那的时候检索怎么逐级降级不崩溃",
    "专利组合策略建议": "这几个专利怎么组合申请优先级怎么排",
}

# Cross-document NEGATIVE queries — topically unrelated to BOTH source docs.
# These ask about things neither document covers, so every (query, section)
# pair here should be a non-match. They span the other workspaces' themes.
NEGATIVE_QUERIES = [
    "京东 VOP 对账周期和结算流程",
    "专利申请的官费和代理费大概多少",
    "金营平台二期的里程碑和干系人有哪些",
    "金融带货的选品策略和转化漏斗",
    "京东科技金融营销系统的整体业务全景",
    "怎么做项目进度复盘和经验沉淀",
]


# ---------------------------------------------------------------------------
# Section splitting — mirrors MemoryTools._detect_markdown_headings +
# offset computation, but splits only at H2 (`##`) so sections are
# semantically substantial (H3 would over-fragment for calibration).
# ---------------------------------------------------------------------------

def split_at_h2(content: str) -> list[dict]:
    """Split content into sections at top-level `## ` headings.

    Returns a list of {title, body, start, end}.  Text before the first H2
    becomes a section titled by its first non-empty line (or "（前言）").
    """
    lines = content.splitlines(keepends=True)
    # Find H2 heading offsets (line starts with exactly "## ", not inside a fence).
    in_fence = False
    fence_marker = None
    h2_boundaries: list[tuple[int, str]] = []  # (char_offset, title)
    pos = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, None
        elif not in_fence:
            m = re.match(r"^(##)\s+(.+?)\s*$", line.rstrip())
            if m:
                h2_boundaries.append((pos, line.rstrip()))
        pos += len(line)

    sections: list[dict] = []
    if not h2_boundaries:
        sections.append({"title": "（全文）", "body": content, "start": 0, "end": len(content)})
        return sections

    # Prelude (before first H2) — keep it as a section so coverage is whole-doc.
    if h2_boundaries[0][0] > 0:
        pre = content[: h2_boundaries[0][0]].strip()
        if pre:
            first_line = pre.split("\n", 1)[0].strip(" #>—") or "（前言）"
            sections.append({"title": first_line, "body": pre, "start": 0, "end": h2_boundaries[0][0]})

    for i, (start, title) in enumerate(h2_boundaries):
        end = h2_boundaries[i + 1][0] if i + 1 < len(h2_boundaries) else len(content)
        body = content[start:end]
        sections.append({"title": title, "body": body, "start": start, "end": end})
    return sections


# ---------------------------------------------------------------------------
# Embedding — load the real GGUF model and embed text the same way
# production embed_text does (prefix + "\n" + body).
# ---------------------------------------------------------------------------

def load_embedder():
    from llama_cpp import Llama  # type: ignore

    print(f"Loading model: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size // (1024*1024)} MiB)...", flush=True)
    llm = Llama(model_path=str(MODEL_PATH), embedding=True, verbose=False, n_ctx=N_CTX)

    def encode(text: str) -> list[float]:
        return [float(x) for x in llm.create_embedding(text)["data"][0]["embedding"]]

    def embed(prefix: str, body: str, max_body_chars: int | None = MAX_SECTION_CHARS) -> list[float]:
        if max_body_chars is not None and len(body) > max_body_chars:
            body = body[:max_body_chars]
        sep = "\n" if prefix and body else ""
        return encode(prefix + sep + body)

    return embed


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def fmt_dist(label: str, vals: list[float]) -> str:
    s = sorted(vals)
    return (f"  {label:32s} n={len(s):3d}  "
            f"min={percentile(s,0):.3f} P25={percentile(s,.25):.3f} "
            f"P50={percentile(s,.5):.3f} P75={percentile(s,.75):.3f} "
            f"P90={percentile(s,.9):.3f} max={percentile(s,1):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: DB not found: {DB_PATH}", file=sys.stderr)
        return 1
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    # 1. Load + split source documents.
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    docs: dict[int, dict] = {}  # memory_id -> {subject, sections: [...]}
    for mid in SOURCE_MEMORY_IDS:
        row = conn.execute("SELECT subject, content FROM memories WHERE id = ?", (mid,)).fetchone()
        if row is None:
            print(f"WARNING: memory id={mid} not found, skipping", file=sys.stderr)
            continue
        secs = split_at_h2(row["content"])
        docs[mid] = {"subject": row["subject"], "sections": secs}
        print(f"memory {mid} ({row['subject'][:40]}): {len(secs)} H2 sections")
    conn.close()
    if not docs:
        print("ERROR: no source documents loaded", file=sys.stderr)
        return 1

    query_map = {141: QUERIES_141, 70: QUERIES_70}
    for mid in docs:
        titles = {s["title"] for s in docs[mid]["sections"]}
        # Normalise titles for matching (strip the leading "## ").
        titles_norm = {t.lstrip("# ").strip(): t for t in titles}
        for q_key in query_map[mid]:
            if q_key not in titles and q_key not in titles_norm:
                print(f"WARNING: query target {q_key!r} (memory {mid}) not found among sections: "
                      f"{sorted(titles_norm)}", file=sys.stderr)

    # 2. Embed all sections + queries.
    embed = load_embedder()

    # section embeddings keyed by (memory_id, title)
    section_emb: dict[tuple[int, str], list[float]] = {}
    for mid, doc in docs.items():
        for sec in doc["sections"]:
            title_clean = sec["title"].lstrip("# ").strip()
            section_emb[(mid, title_clean)] = embed(prefix=title_clean, body=sec["body"])
    print(f"Embedded {len(section_emb)} sections.", flush=True)

    query_emb: dict[str, list[float]] = {}
    for mid, qmap in query_map.items():
        for q_text in qmap.values():
            query_emb[q_text] = embed(prefix="", body=q_text)
    for q in NEGATIVE_QUERIES:
        query_emb[q] = embed(prefix="", body=q)
    print(f"Embedded {len(query_emb)} queries.", flush=True)

    # 3. Build the three distance groups.
    pos_dists: list[float] = []
    pos_samples: list[tuple[str, str, float]] = []
    same_doc_neg_dists: list[float] = []
    cross_doc_neg_dists: list[float] = []
    negative_dists: list[float] = []

    for mid, doc in docs.items():
        titles_clean = [s["title"].lstrip("# ").strip() for s in doc["sections"]]
        for q_target, q_text in query_map[mid].items():
            q_key = q_target.lstrip("# ").strip()
            qe = query_emb[q_text]
            for title in titles_clean:
                se = section_emb[(mid, title)]
                d = cosine_distance(qe, se)
                if title == q_key:
                    pos_dists.append(d)
                    pos_samples.append((q_text[:30], title[:30], d))
                else:
                    same_doc_neg_dists.append(d)

    # Cross-document negatives: each doc's queries vs the OTHER doc's sections.
    mids = list(docs.keys())
    if len(mids) == 2:
        for q_target, q_text in query_map[mids[0]].items():
            qe = query_emb[q_text]
            for s in docs[mids[1]]["sections"]:
                title = s["title"].lstrip("# ").strip()
                cross_doc_neg_dists.append(cosine_distance(qe, section_emb[(mids[1], title)]))
        for q_target, q_text in query_map[mids[1]].items():
            qe = query_emb[q_text]
            for s in docs[mids[0]]["sections"]:
                title = s["title"].lstrip("# ").strip()
                cross_doc_neg_dists.append(cosine_distance(qe, section_emb[(mids[0], title)]))

    # Topically-unrelated negative queries vs ALL sections.
    for q in NEGATIVE_QUERIES:
        qe = query_emb[q]
        for mid, doc in docs.items():
            for s in doc["sections"]:
                title = s["title"].lstrip("# ").strip()
                negative_dists.append(cosine_distance(qe, section_emb[(mid, title)]))

    # 4. Report distributions.
    print("\n" + "=" * 70)
    print("=== Distance distribution (cosine distance, lower = more similar) ===")
    print("=" * 70)
    print(fmt_dist("positive (query↔target)", pos_dists))
    print(fmt_dist("same-doc negative", same_doc_neg_dists))
    print(fmt_dist("cross-doc negative", cross_doc_neg_dists))
    print(fmt_dist("unrelated-topic negative", negative_dists))

    all_neg = same_doc_neg_dists + cross_doc_neg_dists + negative_dists
    print(fmt_dist("ALL negative (combined)", all_neg))

    # 5. Threshold sweep.
    print("\n" + "=" * 70)
    print("=== Threshold sweep (recall = % positives hit; false_hit = % negatives hit) ===")
    print("=" * 70)
    print(f"  {'T':>5s}  {'recall':>8s}  {'false_hit':>10s}  {'margin':>8s}  note")
    pos_s = sorted(pos_dists)
    neg_s = sorted(all_neg)
    recommended: list[float] = []
    for t10 in range(3, 13):  # 0.3 .. 1.2
        t = t10 / 10.0
        recall = sum(1 for d in pos_s if d <= t) / len(pos_s) if pos_s else 0.0
        false_hit = sum(1 for d in neg_s if d <= t) / len(neg_s) if neg_s else 0.0
        margin = false_hit - recall  # negative margin = good separation
        note = ""
        if recall >= 0.9 and false_hit <= 0.1:
            note = "← good zone"
            recommended.append(t)
        elif recall >= 0.8 and false_hit <= 0.2:
            note = "← acceptable"
        print(f"  {t:5.1f}  {recall:8.1%}  {false_hit:10.1%}  {margin:+8.2f}  {note}")

    # 6. Recommendation.
    print("\n" + "=" * 70)
    print("=== Recommendation ===")
    print("=" * 70)
    pos_p90 = percentile(pos_s, 0.9)
    neg_p10 = percentile(neg_s, 0.10)
    neg_p50 = percentile(neg_s, 0.5)
    print(f"  positive P90 = {pos_p90:.3f}  (90% of relevant queries hit at or below this)")
    print(f"  negative P10 = {neg_p10:.3f}  (10% of irrelevant sections already this close)")
    print(f"  negative P50 = {neg_p50:.3f}")
    if pos_p90 < neg_p10:
        midpt = (pos_p90 + neg_p10) / 2
        print(f"\n  ✅ Clean separation: pos-P90 ({pos_p90:.3f}) < neg-P10 ({neg_p10:.3f})")
        print(f"     Recommended threshold = {midpt:.2f} (midpoint of the gap)")
        print(f"     Rationale: catches ≥90% positives while filtering ≥90% negatives.")
    elif pos_p90 < neg_p50:
        midpt = (pos_p90 + neg_p50) / 2
        print(f"\n  ⚠️  Partial overlap: pos-P90 ({pos_p90:.3f}) < neg-P50 ({neg_p50:.3f}) but > neg-P10")
        print(f"     Recommended threshold = {midpt:.2f} (lean slightly tight to control noise)")
        print(f"     Rationale: accepts some false hits to preserve recall — pair with FTS recall.")
    else:
        print(f"\n  ❌ Heavy overlap: pos-P90 ({pos_p90:.3f}) >= neg-P50 ({neg_p50:.3f})")
        print(f"     Section-level semantic discrimination is weak for this model.")
        print(f"     Recommend threshold = {pos_p90:.2f} (keep recall) + rely on FTS/LIKE as primary.")
    if recommended:
        print(f"  Good-zone thresholds from sweep: {recommended}")

    # 7. Positive samples (for sanity-checking the labels).
    print("\n=== Positive samples (query → target section, distance) ===")
    for q, title, d in sorted(pos_samples, key=lambda x: x[2]):
        print(f"  {d:.3f}  {q!s:34s} → {title}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
