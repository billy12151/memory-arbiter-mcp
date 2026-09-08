#!/usr/bin/env python3
"""eval_relevance_floor — 0.15.9 commit ① 定参评测（方案 docs/mema-0159-recall-quality-and-batch-find-plan-2026-09-08.md §3）.

两相运行：
  dump    对真库跑固定 query 集，产出盲标文件（无分数）+ 特征文件（s/final/group）。
  analyze 合并人工标签，按 候选组（纯语义/纯词法/混合）× 标签 出分布，
          给出分级抬曲线 X/Y 与页面线 F 的建议值及 sweep 表。

只读：仅 SELECT 路径（search_memories）+ embedder 推理，不写库。
盲标方法：dump 阶段的 candidates_for_labeling.jsonl 不含任何分数字段，
标签在 analyze 之前人工（agent 预标，owner 抽检）写入 labels.jsonl。
s 近似：evidence 复合分 = best + 0.08×support（support 仅能用 top3 hits
近似重算，与 ② 将落行的 _evidence_score 有微小差异，报告中注明）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.constants import EMBEDDING_MAX_SECTION_CHARS  # noqa: E402
from memory_arbiter.recall_blacklist import blacklist_path, load_blacklist  # noqa: E402
from memory_arbiter.search import search_memories  # noqa: E402
from memory_arbiter.tools import MemoryTools  # noqa: E402

# ---- 固定 query 集（34 条：改述 12 / 精确 10 / 法规形态 8 / 无关远尾 4）----
QUERIES: list[dict[str, str]] = [
    # A. 改述式：不用 subject 原词，主测语义通道
    {"qid": "A01", "kind": "paraphrase", "query": "检索不到记忆的时候返回什么行为"},
    {"qid": "A02", "kind": "paraphrase", "query": "小模型抽槽最后选了哪个"},
    {"qid": "A03", "kind": "paraphrase", "query": "网站备案手续办完了吗"},
    {"qid": "A04", "kind": "paraphrase", "query": "代理服务稳定性加固做了什么"},
    {"qid": "A05", "kind": "paraphrase", "query": "中文文档被重写丢了怎么恢复的"},
    {"qid": "A06", "kind": "paraphrase", "query": "一键配置功能谁拍板的"},
    {"qid": "A07", "kind": "paraphrase", "query": "向量表插入数据报唯一键冲突"},
    {"qid": "A08", "kind": "paraphrase", "query": "发版之前要跑哪些检查"},
    {"qid": "A09", "kind": "paraphrase", "query": "查询权限边界怎么定义"},
    {"qid": "A10", "kind": "paraphrase", "query": "二期立项里程碑和资源投入"},
    {"qid": "A11", "kind": "paraphrase", "query": "服务部署方式改成什么了"},
    {"qid": "A12", "kind": "paraphrase", "query": "流量压缩验证效果怎么样"},
    # B. 精确查找式：直接用库里术语，主测词法通道
    {"qid": "B01", "kind": "lookup", "query": "memory-arbiter v0.8 接口收敛"},
    {"qid": "B02", "kind": "lookup", "query": "migrate_workspace UNIQUE 冲突"},
    {"qid": "B03", "kind": "lookup", "query": "workspace归一评测"},
    {"qid": "B04", "kind": "lookup", "query": "金营二期 PRD 审批流程"},
    {"qid": "B05", "kind": "lookup", "query": "Qwen2.5-0.5B 抽槽模型"},
    {"qid": "B06", "kind": "lookup", "query": "README 信息存储规则"},
    {"qid": "B07", "kind": "lookup", "query": "操作纪律 桥接脚本"},
    {"qid": "B08", "kind": "lookup", "query": "llama.cpp n_batch 截断"},
    {"qid": "B09", "kind": "lookup", "query": "recall_blacklist mema-twin"},
    {"qid": "B10", "kind": "lookup", "query": "金营 智能配券 场景"},
    # C. 法规形态（下游样例形态；本库预期绝大多数不相关，提供远尾分布）
    {"qid": "C01", "kind": "legal", "query": "催收 辱骂 侮辱"},
    {"qid": "C02", "kind": "legal", "query": "债务转移 债权人同意"},
    {"qid": "C03", "kind": "legal", "query": "民法典 第五百五十一条 债务转移"},
    {"qid": "C04", "kind": "legal", "query": "个人信息保护法 敏感个人信息 单独同意"},
    {"qid": "C05", "kind": "legal", "query": "消费者权益保护法 退一赔三"},
    {"qid": "C06", "kind": "legal", "query": "电子商务法 平台责任 连带责任"},
    {"qid": "C07", "kind": "legal", "query": "民间借贷 利率上限"},
    {"qid": "C08", "kind": "legal", "query": "数据出境 安全评估"},
    # D. 无关远尾（预期全部不相关，标定远端）
    {"qid": "D01", "kind": "far", "query": "年会预算 场地预订"},
    {"qid": "D02", "kind": "far", "query": "旅行计划 日本 樱花"},
    {"qid": "D03", "kind": "far", "query": "猫咪疫苗 猫三联"},
    {"qid": "D04", "kind": "far", "query": "红烧肉 做法 五花肉"},
]

TOP_K = 10


def _group_of(row: dict) -> str:
    lex = row.get("_lexical_rank") is not None
    ev = row.get("_evidence_rank") is not None
    if lex and ev:
        return "mixed"
    if ev:
        return "semantic"
    if lex:
        return "lexical"
    return "fallback_or_browse"


def _evidence_features(row: dict) -> tuple[float | None, float | None]:
    """(best_hit_score, approx_composite) from _evidence_hits top3.

    approx = best + 0.08 * sum(max(0, h - 0.45) for h in hits[1:])  # 近似重算
    """
    hits = row.get("_evidence_hits") or []
    scores = sorted(
        (float(h.get("score") or 0.0) for h in hits if isinstance(h, dict)),
        reverse=True,
    )
    if not scores:
        return None, None
    best = scores[0]
    support = sum(max(0.0, s - 0.45) for s in scores[1:])
    return best, best + 0.08 * support


def phase_dump(out: Path) -> None:
    settings = Settings.from_env()
    tools = MemoryTools(settings)
    warnings: list[str] = []
    exclude_ws = None
    if settings.isolation != "strict":
        exclude_ws, bl_warn = load_blacklist(blacklist_path(settings.db_path))
        warnings.extend(bl_warn)
        if exclude_ws and settings.workspace in exclude_ws:
            exclude_ws = None
    embedder, ensure_warn = tools._ensure_embedder()
    warnings.extend(ensure_warn)
    print(f"[dump] embedder={'ok' if embedder else 'UNAVAILABLE'}; "
          f"exclude_ws={sorted(exclude_ws) if exclude_ws else None}; warn={len(warnings)}")
    label_f = out / "candidates_for_labeling.jsonl"
    feat_f = out / "features.jsonl"
    with label_f.open("w", encoding="utf-8") as lf, feat_f.open("w", encoding="utf-8") as ff:
        for q in QUERIES:
            emb = None
            if embedder is not None:
                er = embedder.embed_text(
                    prefix="", body=q["query"],
                    max_body_chars=max(EMBEDDING_MAX_SECTION_CHARS, 2048),
                )
                emb = er.embedding
            outcome = search_memories(
                tools.db, q["query"], None, None, TOP_K,
                status_filter="active", debug_ranking=True, query_embedding=emb,
                isolation=settings.isolation, exclude_workspaces=exclude_ws,
            )
            for row in outcome.results:
                mid = int(row["id"])
                content = str(row.get("content") or "")
                best, comp = _evidence_features(row)
                # 盲标文件：无任何分数/距离字段
                lf.write(json.dumps({
                    "qid": q["qid"], "kind": q["kind"], "query": q["query"],
                    "mem_id": mid, "workspace": row.get("workspace"),
                    "subject": row.get("subject"),
                    "content_head": content[:180],
                    "tags": row.get("tags"),
                }, ensure_ascii=False) + "\n")
                ff.write(json.dumps({
                    "qid": q["qid"], "mem_id": mid,
                    "retrieval_mode": outcome.retrieval_mode,
                    "group": _group_of(row),
                    "s_best": best, "s_composite": comp,
                    "final_score": row.get("_final_score"),
                    "match_reason": row.get("_match_reason"),
                    "evidence_rank": row.get("_evidence_rank"),
                }, ensure_ascii=False) + "\n")
            print(f"[dump] {q['qid']} {q['kind']:10s} mode={outcome.retrieval_mode} "
                  f"n={len(outcome.results)}")
    print(f"[dump] wrote {label_f} / {feat_f}")


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def phase_analyze(out: Path) -> None:
    feats: dict[tuple[str, int], dict] = {}
    for line in (out / "features.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        feats[(r["qid"], r["mem_id"])] = r
    labels: dict[tuple[str, int], str] = {}
    labels_path = out / "labels.jsonl"
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            labels[(r["qid"], r["mem_id"])] = r["label"]
    rows = []
    for key, f in feats.items():
        label = labels.get(key)
        if label is None:
            label = "irrelevant"  # 缺省：未标注即不相关（labels 只写非无关项）
        rows.append({**f, "label": label})
    n_labeled_pos = sum(1 for r in rows if r["label"] in ("relevant", "borderline"))
    print(f"[analyze] candidates={len(rows)} labeled_relevant_or_borderline={n_labeled_pos}")

    def dist(field: str, group: str | None, label: str | None) -> list[float]:
        vals = sorted(
            float(r[field]) for r in rows
            if r.get(field) is not None
            and (group is None or r["group"] == group)
            and (label is None or r["label"] == label)
        )
        return vals

    lines: list[str] = ["# 分布（min/p10/p25/p50/p75/p90/max）", ""]
    for field, title in (("s_composite", "s 复合分（分级抬自变量）"), ("final_score", "final_score（页面线自变量）")):
        lines.append(f"## {title}")
        for group in ("semantic", "lexical", "mixed"):
            for label in ("relevant", "borderline", "irrelevant"):
                v = dist(field, group, label)
                if not v:
                    continue
                lines.append(
                    f"- {group:8s} × {label:10s} n={len(v):3d} "
                    + " ".join(f"{_pct(v, p):.3f}" for p in (0, 10, 25, 50, 75, 90, 100))
                )
        lines.append("")

    # X/Y 建议：纯语义组，relevant p10 与 irrelevant p90 的分离带
    sem_rel = dist("s_composite", "semantic", "relevant")
    sem_bor = dist("s_composite", "semantic", "borderline")
    sem_irr = dist("s_composite", "semantic", "irrelevant")
    x_cand = _pct(sem_rel, 10) if sem_rel else float("nan")
    y_cand = _pct(sem_irr, 90) if sem_irr else float("nan")
    # F 建议：全组 relevant final p05（recall-safe：只切几乎不可能正确的）
    all_rel = dist("final_score", None, "relevant")
    f_cand = _pct(all_rel, 5) if all_rel else float("nan")
    lines += [
        "# 参数建议（待 owner 确认）",
        f"- X（满抬边界，语义 relevant p10）= {x_cand:.3f}"
        f"（borderline p50 参照 = {_pct(sem_bor, 50) if sem_bor else float('nan'):.3f}）",
        f"- Y（衰减终点，语义 irrelevant p90）= {y_cand:.3f}",
        f"- F（页面线，全组 relevant final p05）= {f_cand:.3f}",
        "",
        "# sweep：页面线 F 对 标注数据 的 切除率/保留率",
        "",
        "| F | 切掉 irrelevant | 切掉 borderline | 误切 relevant | 保留 relevant |",
        "|---|---|---|---|---|",
    ]
    n_irr = len(dist("final_score", None, "irrelevant"))
    n_bor = len(dist("final_score", None, "borderline"))
    n_rel = len(dist("final_score", None, "relevant"))
    base = f_cand
    for f in [base - 1.0, base - 0.5, base, base + 0.5, base + 1.0, base + 2.0]:
        cut_irr = sum(1 for r in rows if r["label"] == "irrelevant" and r.get("final_score") is not None and r["final_score"] < f)
        cut_bor = sum(1 for r in rows if r["label"] == "borderline" and r.get("final_score") is not None and r["final_score"] < f)
        cut_rel = sum(1 for r in rows if r["label"] == "relevant" and r.get("final_score") is not None and r["final_score"] < f)
        lines.append(
            f"| {f:.2f} | {cut_irr}/{n_irr} | {cut_bor}/{n_bor} | {cut_rel}/{n_rel} | "
            f"{n_rel - cut_rel}/{n_rel} |"
        )
    report = out / "analysis.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[analyze] wrote {report}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["dump", "analyze"])
    ap.add_argument("--out", default="/tmp/mema-eval")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "dump":
        phase_dump(out)
    else:
        phase_analyze(out)


if __name__ == "__main__":
    main()
