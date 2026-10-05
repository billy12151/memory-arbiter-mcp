"""M0 前缀矩阵标定（embedder 前缀方案 §2.1/§4，owner 要求：数据裁决甲/乙/丙）。

预注册裁决规则（方案 §9.1）：
- 主指标 = 句料库 87 有效对的配对分离度 gap（min(true) − max(non-true)）；
- 次指标 = self-recall R@10（98 targets，query=subject）；
- 平手（gap 差 <0.02）以 R@10 高者胜；仍平手取甲（改动面最小）。
- 丙（双向量）不单独测：其配对质量=乙的 sts↔sts 数，检索质量=甲的 doc+search 数。

产物：eval/results/prefix-calibration-m0.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from llama_cpp import Llama

REPO = Path(__file__).resolve().parent.parent
MODEL = "/Users/zhangzhiwei17/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf"

P_SEARCH = "task: search result | query: "
P_DOC = "title: none | text: "
P_STS = "task: sentence similarity | query: "

PROBE_PAIRS = [
    ("para", "会议改到周三", "会议顺延到周三"),
    ("contra", "会议改到周三", "会议改到周四"),
    ("para", "这个项目由张三负责", "该项目由张三牵头"),
    ("contra", "这个项目由张三负责", "这个项目由李四负责"),
    ("para", "本产品支持无线连接", "该产品具备无线连接能力"),
    ("contra", "本产品支持无线连接", "本产品不支持无线连接"),
    ("para", "这家餐厅只提供素食", "该餐厅仅有素食供应"),
    ("contra", "这家餐厅只提供素食", "这家餐厅提供牛排"),
    ("contra", "A 确定性层恒跑（行预滤/cos带/无差异/direct_value_verdict 直出/closed-inactive/backlog 记账）",
               "确定性直判通道 direct_value_verdict + evidence.py 接线，条件全确定性：numeric_value_candidate 路由 + 每侧恰一个值 + coexistence_veto 干净"),
    ("unrel", "会议改到周三", "今天天气不错适合跑步"),
    ("unrel", "本产品支持无线连接", "这个项目由张三负责"),
]


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return d / (na * nb) if na and nb else 0.0


def main() -> None:
    llm = Llama(model_path=MODEL, embedding=True, pooling_type=-1,
                n_ctx=2048, n_gpu_layers=-1, verbose=False)

    def emb(prefix: str, text: str) -> list[float]:
        # 与生产 embed_text 同一拼接（prefix + "\n" + body，对抗 review P1-2）
        return list(llm.create_embedding(prefix + "\n" + text if prefix else text)["data"][0]["embedding"])

    started = time.monotonic()
    out: dict[str, Any] = {"model": Path(MODEL).name, "prefixes": {
        "search": P_SEARCH, "doc": P_DOC, "sts": P_STS}}

    # 1) 探针 11 对：no / doc / sts 三种配对前缀
    probes: dict[str, Any] = {}
    for tag, prefix in (("none", ""), ("doc", P_DOC), ("sts", P_STS)):
        vals = []
        for cat, a, b in PROBE_PAIRS:
            vals.append((cat, cos(emb(prefix, a), emb(prefix, b))))
        graded = [(c, v) for c, v in vals if c in ("para", "contra")]
        cands = sorted({v for _, v in graded})
        best_acc, best_thr = max(
            (sum(1 for c, v in graded if (v >= thr) == (c == "para")), thr)
            for thr in cands
        )
        paras = [v for c, v in vals if c == "para"]
        contras = [v for c, v in vals if c == "contra"]
        probes[tag] = {
            "acc": f"{best_acc}/{len(graded)}", "best_thr": round(best_thr, 4),
            "cos": {f"{c}:{i}": round(v, 4) for i, (c, v) in enumerate(vals)},
            "para_min": round(min(paras), 4) if paras else None,
            "contra_max": round(max(contras), 4) if contras else None,
        }
    out["probe_pairs"] = probes

    # 2) 句料库配对分离度：no / doc / sts（配对两侧同前缀；三文件全量=pairs+large+noisy）
    pair_lines: list[str] = []
    for name in ("pairs.jsonl", "pairs_large.jsonl", "pairs_noisy.jsonl"):
        pair_lines += (REPO / "eval/fixtures/conflict" / name).read_text(encoding="utf-8").splitlines()
    pairs = [json.loads(line) for line in pair_lines if line.strip()]
    pairs = [p for p in pairs if p.get("label")]  # 87 有效口径（无标签剔除）
    corpus: dict[str, Any] = {}

    def band_stats(true_cos: list[float], nontrue_cos: list[float], ceil: float) -> dict[str, Any]:
        """子 CEIL 口径（对抗 review P2 指出的 gap 污染：近重复对 cos≥0.98
        本就该被上沿排除，混进 nontrue 会把 gap 打穿）。"""
        sub_nt = [v for v in nontrue_cos if v < ceil]
        sub_t = [v for v in true_cos if v < ceil]
        return {
            "n_true": len(true_cos), "n_nontrue": len(nontrue_cos),
            "true_min": round(min(true_cos), 4),
            "true_p5": round(sorted(true_cos)[max(0, int(len(true_cos) * 0.05))], 4),
            "true_max": round(max(true_cos), 4),
            "nontrue_max": round(max(nontrue_cos), 4),
            "nontrue_p95": round(sorted(nontrue_cos)[int(len(nontrue_cos) * 0.95) - 1], 4),
            "gap_raw": round(min(true_cos) - max(nontrue_cos), 4),
            "sub_ceil": {
                "ceil": ceil, "n_nontrue_below": len(sub_nt),
                "nontrue_max_below": round(max(sub_nt), 4) if sub_nt else None,
                "true_min_below": round(min(sub_t), 4) if sub_t else None,
                "gap_below_ceil": round(min(sub_t) - max(sub_nt), 4) if sub_t and sub_nt else None,
            },
        }

    for tag, prefix in (("none", ""), ("doc", P_DOC), ("sts", P_STS)):
        true_cos: list[float] = []
        nontrue_cos: list[float] = []
        for p in pairs:
            ca = emb(prefix, p["left"]["content"])
            cb = emb(prefix, p["right"]["content"])
            c = cos(ca, cb)
            (true_cos if p["label"] == "true_conflict" else nontrue_cos).append(c)
        corpus[tag] = band_stats(true_cos, nontrue_cos, 0.98)
        corpus[tag]["raw_true"] = [round(v, 4) for v in sorted(true_cos)]
        corpus[tag]["raw_nontrue"] = [round(v, 4) for v in sorted(nontrue_cos)]
    out["conflict_corpus_pairs"] = corpus

    # 3) self-recall R@10：98 targets，query=subject
    targets = [json.loads(line) for line in
               (REPO / "eval/fixtures/recall/targets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    recall: dict[str, Any] = {}
    for stored_tag, stored_prefix, query_prefix in (
        ("none/none", "", ""),
        ("doc/search", P_DOC, P_SEARCH),
        ("sts/search", P_STS, P_SEARCH),
    ):
        stored = [emb(stored_prefix, t["content"]) for t in targets]
        hit = 0
        for t in targets:
            q = emb(query_prefix, t["subject"])
            ranked = sorted(range(len(stored)), key=lambda i: -cos(q, stored[i]))
            if t["fixture_key"] in {targets[i]["fixture_key"] for i in ranked[:10]}:
                hit += 1
        recall[f"stored={stored_tag}"] = {"self_recall_top10": f"{hit}/{len(targets)}"}
    out["self_recall_proxy"] = recall

    # 4) attr↔row 跨型（claims 语料 C 通道形态）
    claims_pairs = [json.loads(line) for line in
                    (REPO / "eval/fixtures/conflict/pairs_claims.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cross: dict[str, Any] = {}
    for attr_tag, attr_prefix in (("doc", P_DOC), ("sts", P_STS)):
        for row_tag, row_prefix in (("doc", P_DOC), ("sts", P_STS)):
            vals = []
            for p in claims_pairs:
                for claim in (p["left"].get("claims") or []):
                    if p["right"].get("content"):
                        vals.append(cos(emb(attr_prefix, claim["attr"]),
                                        emb(row_prefix, p["right"]["content"])))
            if vals:
                cross[f"attr={attr_tag}/row={row_tag}"] = {
                    "n": len(vals), "min": round(min(vals), 4),
                    "mean": round(sum(vals) / len(vals), 4), "max": round(max(vals), 4),
                }
    # 4b) attr↔attr（CLAIM_ATTR_TAU 直接依据）：claims 语料两侧 claims 的
    # 同 attr_norm（应相似）vs 异 attr（应远离）在 sts 前缀下的余弦分布。
    import re as _re

    def _norm_attr(s: str) -> str:
        return _re.sub(r"\s+", "", s).lower()

    same_attr: list[float] = []
    diff_attr: list[float] = []
    attr_cache: dict[str, list[float]] = {}
    for p in claims_pairs:
        for side_a, side_b in ((p["left"], p["right"]), (p["right"], p["left"])):
            for ca in side_a.get("claims") or []:
                for cb in side_b.get("claims") or []:
                    na, nb = _norm_attr(ca["attr"]), _norm_attr(cb["attr"])
                    if na not in attr_cache:
                        attr_cache[na] = emb(P_STS, ca["attr"])
                    if nb not in attr_cache:
                        attr_cache[nb] = emb(P_STS, cb["attr"])
                    c = cos(attr_cache[na], attr_cache[nb])
                    (same_attr if na == nb else diff_attr).append(c)
    out["attr_cross_attr"] = {
        "sts": {
            "n_same": len(same_attr),
            "same_min": round(min(same_attr), 4) if same_attr else None,
            "same_mean": round(sum(same_attr) / len(same_attr), 4) if same_attr else None,
            "n_diff": len(diff_attr),
            "diff_p95": round(sorted(diff_attr)[int(len(diff_attr) * 0.95) - 1], 4) if diff_attr else None,
            "diff_max": round(max(diff_attr), 4) if diff_attr else None,
        },
    }
    out["attr_cross_row"] = cross

    # 5) 检索 B1/B2：47 题 × 相关 target 的 query↔sts 存储余弦分布
    # （COS_RECALL_FLOOR/MIDBAND_CEIL 的直接依据；query=search 前缀、存储=sts）
    queries = json.load(open(REPO / "eval/fixtures/recall/queries.json"))["queries"]
    labels = [json.loads(line) for line in
              (REPO / "eval/fixtures/recall/labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rel_by_qid: dict[str, set[str]] = {}
    for row in labels:
        if row["label"] == "relevant":
            rel_by_qid.setdefault(row["qid"], set()).add(row["fixture_key"])
    key_to_idx = {t["fixture_key"]: i for i, t in enumerate(targets)}
    stored_sts = [emb(P_STS, t["content"]) for t in targets]
    rel_best: list[float] = []
    for item in queries:
        keys = rel_by_qid.get(item["qid"])
        if not keys:
            continue
        q = emb(P_SEARCH, item["query"])
        best = max(cos(q, stored_sts[key_to_idx[k]]) for k in keys if k in key_to_idx)
        rel_best.append(best)
    rel_sorted = sorted(rel_best)
    out["recall_relevant_best_cos"] = {
        "query_stored": "search/sts", "n_queries": len(rel_best),
        "min": round(min(rel_best), 4),
        "p5": round(rel_sorted[int(len(rel_sorted) * 0.05)], 4),
        "p25": round(rel_sorted[int(len(rel_sorted) * 0.25)], 4),
        "median": round(rel_sorted[len(rel_sorted) // 2], 4),
        "raw": [round(v, 4) for v in rel_sorted],
    }

    # 6) 相似 C1：similarity 语料 anchor↔variant 的 subject↔subject sts 余弦
    # 按 label 分组（true_near_dup 应高、负例应低——WRITE_SIMILAR_SUBJECT_* 依据）
    sim_cases = [json.loads(line) for line in
                 (REPO / "eval/fixtures/similarity/cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    sim_cases += [json.loads(line) for line in
                  (REPO / "eval/fixtures/similarity/cases_noisy.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    sim_by_label: dict[str, list[float]] = {}
    for case in sim_cases:
        c = cos(emb(P_STS, case["anchor"]["subject"]), emb(P_STS, case["variant"]["subject"]))
        sim_by_label.setdefault(case["label"], []).append(c)
    out["similarity_subject_axis"] = {
        label: {"n": len(vs), "min": round(min(vs), 4), "max": round(max(vs), 4),
                "mean": round(sum(vs) / len(vs), 4)}
        for label, vs in sorted(sim_by_label.items())
    }
    out["recall_relevant_best_cos"]["p75"] = round(rel_sorted[int(len(rel_sorted) * 0.75)], 4)

    # 7) E 组：真实库 workspace alias/canonical 距离分布（只读 immutable；
    # WORKSPACE_MATCH_DISTANCE/WORKSPACE_RECALL_CUTOFF/QWEN_CANDIDATE_DISTANCE 依据）
    import sqlite3

    db_path = Path.home() / ".local/share/memory-arbiter/memory.sqlite3"
    if db_path.exists():
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            canon = [r[0] for r in conn.execute(
                "SELECT DISTINCT name FROM workspace_canonicals"
            ).fetchall()]
            pairs_ws = conn.execute(
                "SELECT alias_workspace, canonical FROM workspace_aliases LIMIT 400"
            ).fetchall()
        finally:
            conn.close()
        vecs = {name: emb(P_STS, name) for name in canon}
        dists = []
        for alias, canonical in pairs_ws:
            va = vecs.get(alias) or emb(P_STS, alias)
            vc = vecs.get(canonical)
            if vc is None:
                vc = emb(P_STS, canonical)
                vecs[canonical] = vc
            c = cos(va, vc)
            dists.append(1.0 - c)  # 距离口径=1−cos（与 vec_distance_cosine 对齐）
        if dists:
            dists.sort()
            out["workspace_alias_distance"] = {
                "n": len(dists), "max": round(dists[-1], 4),
                "p95": round(dists[int(len(dists) * 0.95)], 4),
                "mean": round(sum(dists) / len(dists), 4),
            }
    out["elapsed_s"] = round(time.monotonic() - started, 1)

    dest = REPO / "eval/calibration/prefix-calibration-m0.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
