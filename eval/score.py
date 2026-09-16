#!/usr/bin/env python3
"""scorers — P0-1 c5 指标计算 + 回归门（方案 §2.7）.

从 runner 的 raw 产物计算三套件指标；所有占比指标与绝对条数成对输出
（owner 2026-09-16 全局规则）。gate 对比基线：缺基线只报告不判；
有基线逐指标比相对下降，超 provisional 阈值（默认 10%，待 owner 拿
首份基线报告后定正式门槛——#999 禁止拍脑袋）标 FAILED 并 exit 非零。

指标口径：
  recall  Recall@5/@10 微平均（分母=各 qid 的 relevant target 数）；
          MRR=首个 relevant target 排名倒数均值；borderline 单列不进分母；
          无关误召回=C/D 组 query 返回条目中非无关标注（relevant/borderline）
          的计数——临时库无 C/D 真 target，命中 A/B target 即误召回。
  similarity 相似记忆提示条数/占比（总量）+ 四类分型各带条数/率。
  conflict 按 shape 分层 + 总体：Precision/Recall/三结局（sync/async/miss
          计数与占比）+ 总识别率；skipped_member_replay 不计。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "eval" / "fixtures"
DEFAULT_REL_DROP = 0.10  # provisional：首份基线报告后由 owner 定正式门槛


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pct(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def score_recall(raw: dict) -> dict[str, Any] | None:
    queries = raw.get("queries")
    if queries is None:
        return None
    labels = _load_jsonl(FIXTURES / "recall" / "labels.jsonl")
    relevant_by_qid: dict[str, set[str]] = {}
    borderline_by_qid: dict[str, set[str]] = {}
    for row in labels:
        bucket = relevant_by_qid if row["label"] == "relevant" else borderline_by_qid
        bucket.setdefault(row["qid"], set()).add(row["fixture_key"])
    labeled_keys = {k for qid in relevant_by_qid for k in relevant_by_qid[qid]}
    labeled_keys |= {k for qid in borderline_by_qid for k in borderline_by_qid[qid]}

    hit5 = hit10 = total_rel = 0
    rr_sum = 0.0
    rr_count = 0
    false_pull_count = 0
    false_pull_returned = 0
    per_query: list[dict[str, Any]] = []
    for query in queries:
        qid = query["qid"]
        targets = relevant_by_qid.get(qid, set())
        ranked = [hit["fixture_key"] for hit in query.get("hits") or []]
        total_rel += len(targets)
        hit5 += len(targets & set(ranked[:5]))
        hit10 += len(targets & set(ranked[:10]))
        first_rank = next(
            (i + 1 for i, key in enumerate(ranked) if key in targets), None,
        )
        if first_rank:
            rr_sum += 1.0 / first_rank
            rr_count += 1
        row = {
            "qid": qid, "kind": query["kind"], "relevant_targets": len(targets),
            "recall@5": _pct(len(targets & set(ranked[:5])), len(targets)),
            "recall@10": _pct(len(targets & set(ranked[:10])), len(targets)),
            "first_relevant_rank": first_rank,
        }
        if query["kind"] in {"legal", "far"}:
            false_hits = [k for k in ranked if k in labeled_keys]
            false_pull_count += len(false_hits)
            false_pull_returned += len(ranked)
            row["irrelevant_query_false_pulls"] = len(false_hits)
        per_query.append(row)

    self_recall = raw.get("self_recall")
    self_top10 = None
    if self_recall:
        top10 = sum(1 for row in self_recall if row["in_top10"])
        self_top10 = {"count": top10, "total": len(self_recall), "rate": _pct(top10, len(self_recall))}
    return {
        "recall_at_5": {"hits": hit5, "total": total_rel, "rate": _pct(hit5, total_rel)},
        "recall_at_10": {"hits": hit10, "total": total_rel, "rate": _pct(hit10, total_rel)},
        "mrr": {"value": round(rr_sum / rr_count, 4) if rr_count else None,
                "queries_with_target": rr_count},
        "irrelevant_false_pulls": {"count": false_pull_count, "returned": false_pull_returned,
                                   "rate": _pct(false_pull_count, false_pull_returned)},
        "self_recall_top10": self_top10,
        "per_query": per_query,
    }


def score_similarity(raw: dict) -> dict[str, Any] | None:
    similarity = raw.get("similarity")
    if similarity is None:
        return None
    cases = similarity["cases"]
    by_label: dict[str, dict[str, int]] = {}
    for case in cases:
        bucket = by_label.setdefault(case["label"], {"n": 0, "fired": 0, "hit_anchor": 0})
        bucket["n"] += 1
        bucket["fired"] += int(case["fired"])
        bucket["hit_anchor"] += int(case["hit_anchor"])
    fired_total = sum(int(c["fired"]) for c in cases)
    return {
        "hint_total": {"count": fired_total, "total": len(cases), "rate": _pct(fired_total, len(cases))},
        "by_label": {
            label: {
                "fired": {"count": bucket["fired"], "total": bucket["n"], "rate": _pct(bucket["fired"], bucket["n"])},
                "hit_anchor": {"count": bucket["hit_anchor"], "total": bucket["n"], "rate": _pct(bucket["hit_anchor"], bucket["n"])},
            }
            for label, bucket in sorted(by_label.items())
        },
    }


def score_conflict(raw: dict) -> dict[str, Any] | None:
    conflict = raw.get("conflict")
    if conflict is None:
        return None
    valid = [row for row in conflict if not row["skipped_member_replay"]]
    # runner 采集不带 shape：按 pair_id 从对集 join（对集是 shape 的权威源）
    shape_of = {
        pair["pair_id"]: pair.get("shape") or "governed_negative"
        for pair in _load_jsonl(FIXTURES / "conflict" / "pairs.jsonl")
    }

    def _outcome_row(rows: list[dict]) -> dict[str, Any]:
        sync = sum(1 for r in rows if r["sync"])
        async_ = sum(1 for r in rows if r["async"])
        miss = sum(1 for r in rows if r["notice_missing"])
        return {
            "n": len(rows),
            "sync": {"count": sync, "rate": _pct(sync, len(rows))},
            "async": {"count": async_, "rate": _pct(async_, len(rows))},
            "miss": {"count": miss, "rate": _pct(miss, len(rows))},
            "identified": {"count": sync + async_, "rate": _pct(sync + async_, len(rows))},
        }

    true_rows = [r for r in valid if r["label"] == "true_conflict"]
    coexist_rows = [r for r in valid if r["label"] == "coexist"]
    noise_rows = [r for r in valid if r["label"] == "noise"]
    identified_all = [r for r in valid if r["sync"] or r["async"]]
    true_identified = [r for r in identified_all if r["label"] == "true_conflict"]
    return {
        "overall": {
            "precision": {"count": len(true_identified), "total": len(identified_all),
                          "rate": _pct(len(true_identified), len(identified_all))},
            "recall": {"count": len(true_identified), "total": len(true_rows),
                       "rate": _pct(len(true_identified), len(true_rows))},
            "coexist_false_positive": {"count": sum(1 for r in identified_all if r["label"] == "coexist"),
                                       "total": len(coexist_rows),
                                       "rate": _pct(sum(1 for r in identified_all if r["label"] == "coexist"), len(coexist_rows))},
            "skipped_member_replay": len(conflict) - len(valid),
        },
        "by_label": {
            "true_conflict": _outcome_row(true_rows),
            "coexist": _outcome_row(coexist_rows),
            "noise": _outcome_row(noise_rows),
        },
        "by_shape": {
            shape: _outcome_row([r for r in valid if shape_of.get(r["pair_id"], "governed_negative") == shape])
            for shape in ("scan_evolution", "governed_negative", "write_opposition")
        },
    }


def score_all(raw: dict) -> dict[str, Any]:
    return {
        "mema_version": raw.get("mema_version"),
        "corpus_version": raw.get("corpus_version"),
        "env": raw.get("env"),
        "recall": score_recall(raw),
        "similarity": score_similarity(raw),
        "conflict": score_conflict(raw),
    }


# ---- gate：基线对比（相对下降，provisional 阈值） ---------------------------

def _flatten(metrics: dict, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[prefix + key] = float(value)
    return flat


# Gate 方向语义（0.16.6 发版轮修正）：这些指标越低越好——漏检率/误召回/共存
# 误报下降是改善，原先把任何下降都当回归拦，把冲突修复线的进步判成了 FAILED
# （miss.rate 0.86→0.68 触发 10% 相对下降门，实际是召回翻倍）。语料元数据
# （skip 数、返回条数）不是产品指标，不进门。
_LOWER_IS_BETTER_SUBSTR = (
    ".miss.", "irrelevant_false_pulls", "coexist_false_positive",
)
_SIM_FALSE_LABELS = ("clearly_different", "opposite_semantics", "same_entity_diff_attr")
_GATE_META_KEYS = (".skipped_member_replay", ".returned", ".queries_with_target")


def _lower_is_better(key: str) -> bool:
    if any(s in key for s in _LOWER_IS_BETTER_SUBSTR):
        return True
    return any(f".{label}." in key for label in _SIM_FALSE_LABELS)


def gate(current: dict, baseline: dict, rel_drop: float = DEFAULT_REL_DROP) -> dict[str, Any]:
    cur = _flatten(current)
    base = _flatten(baseline)
    failures: list[dict[str, Any]] = []
    for key, base_value in sorted(base.items()):
        if key not in cur or key.endswith((".count", ".total", ".n", "first_relevant_rank")):
            continue
        if any(key.endswith(meta) for meta in _GATE_META_KEYS):
            continue
        cur_value = cur[key]
        if _lower_is_better(key):
            if base_value <= 0 or cur_value <= base_value:
                continue
            rise = (cur_value - base_value) / base_value
            if rise > rel_drop:
                failures.append({
                    "metric": key, "direction": "lower_is_better",
                    "baseline": base_value, "current": cur_value,
                    "relative_rise": round(rise, 4), "threshold": rel_drop,
                })
            continue
        if base_value <= 0 or cur_value >= base_value:
            continue
        drop = (base_value - cur_value) / base_value
        if drop > rel_drop:
            failures.append({
                "metric": key, "direction": "higher_is_better",
                "baseline": base_value, "current": cur_value,
                "relative_drop": round(drop, 4), "threshold": rel_drop,
            })
    return {"gate": "FAILED" if failures else "PASSED", "rel_drop_threshold": rel_drop,
            "failures": failures}


def render_markdown(scored: dict, gate_result: dict[str, Any] | None) -> str:
    lines: list[str] = [
        "# mema eval harness 报告",
        "",
        f"- mema `{scored.get('mema_version')}` · 语料 `{scored.get('corpus_version')}`",
        f"- embedder `{(scored.get('env') or {}).get('embed_model')}`",
        "",
    ]
    recall = scored.get("recall")
    if recall:
        lines += [
            "## 召回",
            "",
            f"- Recall@5 = **{recall['recall_at_5']['rate']}**（{recall['recall_at_5']['hits']}/{recall['recall_at_5']['total']}）",
            f"- Recall@10 = **{recall['recall_at_10']['rate']}**（{recall['recall_at_10']['hits']}/{recall['recall_at_10']['total']}）",
            f"- MRR = **{recall['mrr']['value']}**（{recall['mrr']['queries_with_target']} query 有 relevant target）",
            f"- 无关误召回 = **{recall['irrelevant_false_pulls']['count']}** 条 / 返回 {recall['irrelevant_false_pulls']['returned']} 条（{recall['irrelevant_false_pulls']['rate']}）",
        ]
        if recall.get("self_recall_top10"):
            sr = recall["self_recall_top10"]
            lines.append(f"- 自召回 top10 = **{sr['count']}/{sr['total']}**（{sr['rate']}）")
        lines.append("")
    sim = scored.get("similarity")
    if sim:
        lines += ["## 相似记忆提示（差异化能力）", ""]
        total = sim["hint_total"]
        lines.append(f"- 总提示 = **{total['count']}/{total['total']}**（{total['rate']}）")
        for label, bucket in sim["by_label"].items():
            fired = bucket["fired"]
            lines.append(f"  - {label}: {fired['count']}/{fired['total']}（{fired['rate']}）")
        lines.append("")
    conflict = scored.get("conflict")
    if conflict:
        lines += ["## 冲突识别（三结局：sync=3 秒窗内 / async=job 补上 / miss=漏检）", ""]
        for shape, row in conflict["by_shape"].items():
            lines.append(
                f"- **{shape}** n={row['n']}：sync {row['sync']['count']}（{row['sync']['rate']}） · "
                f"async {row['async']['count']}（{row['async']['rate']}） · "
                f"miss {row['miss']['count']}（{row['miss']['rate']}）"
            )
        overall = conflict["overall"]
        lines += [
            "",
            f"- Precision = **{overall['precision']['rate']}**（{overall['precision']['count']}/{overall['precision']['total']}）",
            f"- Recall = **{overall['recall']['rate']}**（{overall['recall']['count']}/{overall['recall']['total']}）",
            f"- 共存误报 = **{overall['coexist_false_positive']['count']}/{overall['coexist_false_positive']['total']}**（{overall['coexist_false_positive']['rate']}）",
            f"- 成员重叠跳过 {overall['skipped_member_replay']} 对（不计指标）",
            "",
        ]
    if gate_result:
        lines += [
            "## 回归门（provisional 阈值，待 owner 依首份基线定正式门槛）",
            "",
            f"- {gate_result['gate']}（相对下降阈值 {gate_result['rel_drop_threshold']:.0%}）",
        ]
        for failure in gate_result["failures"]:
            if failure.get("direction") == "lower_is_better":
                lines.append(
                    f"  - {failure['metric']}（越低越好）: {failure['baseline']} → {failure['current']}"
                    f"（升 {failure['relative_rise']:.1%}）"
                )
            else:
                lines.append(
                    f"  - {failure['metric']}: {failure['baseline']} → {failure['current']}"
                    f"（降 {failure['relative_drop']:.1%}）"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="runner 产物 JSON")
    parser.add_argument("--baseline-write", type=Path, default=None, help="写入基线文件")
    parser.add_argument("--baseline", type=Path, default=None, help="对比基线文件")
    parser.add_argument("--rel-drop", type=float, default=DEFAULT_REL_DROP)
    parser.add_argument("--out", type=Path, default=None, help="报告输出（默认 run 同名 .md）")
    args = parser.parse_args()

    raw = json.loads(args.run.read_text(encoding="utf-8"))
    scored = score_all(raw)
    gate_result = None
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        gate_result = gate(scored, baseline, args.rel_drop)
    markdown = render_markdown(scored, gate_result)
    out = args.out or args.run.with_suffix(".md")
    out.write_text(markdown, encoding="utf-8")
    scored_path = args.run.with_name(args.run.stem + "-scored.json")
    payload = {"scored": scored, **({"gate_result": gate_result} if gate_result else {})}
    scored_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.baseline_write:
        args.baseline_write.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_write.write_text(
            json.dumps(scored, ensure_ascii=False, indent=1), encoding="utf-8",
        )
        print(f"[baseline] written -> {args.baseline_write}")
    print(markdown)
    print(f"[scored] -> {scored_path} / {out}")
    if gate_result and gate_result["gate"] == "FAILED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
