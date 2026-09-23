#!/usr/bin/env python3
"""scorers — P0-1 c5 指标计算 + 回归门（方案 §2.7）.

从 runner 的 raw 产物计算三套件指标；所有占比指标与绝对条数成对输出
（owner 2026-09-16 全局规则）。gate 对比基线：缺基线只报告不判；
有基线逐指标比相对下降，超 provisional 阈值（默认 10%，待 owner 拿
首份基线报告后定正式门槛——#999 禁止拍脑袋）标 FAILED 并 exit 非零。

指标口径：
  recall  Recall@5/@10 微平均（分母=各 qid 的 relevant target 数）；
          capped 变体（gate-v2 拍板 5）分母=min(相关数, k)——「该进前 k
          的都进了」，R>k 的 query 不再被 k+1 名罚分；classic 与 capped
          双口径同时输出；
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
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "eval" / "fixtures"
DEFAULT_REL_DROP = 0.10  # provisional：首份基线报告后由 owner 定正式门槛


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    # Capped recall (gate-v2 拍板 5): "everything that belongs in the top-k
    # got in" — the denominator per query is min(|relevant|, k), so a query
    # with 6 relevant targets and a full top-5 scores 1.0 instead of being
    # punished for the k+1th target no ranking could have returned. Micro-
    # averaged: Σhits / Σmin(R_i, k); classic and capped are both reported
    # so old numbers stay traceable.
    capped5_total = capped10_total = 0
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
        hits_at_5 = len(targets & set(ranked[:5]))
        hits_at_10 = len(targets & set(ranked[:10]))
        hit5 += hits_at_5
        hit10 += hits_at_10
        capped5_total += min(len(targets), 5)
        capped10_total += min(len(targets), 10)
        first_rank = next(
            (i + 1 for i, key in enumerate(ranked) if key in targets),
            None,
        )
        if first_rank:
            rr_sum += 1.0 / first_rank
            rr_count += 1
        row = {
            "qid": qid,
            "kind": query["kind"],
            "relevant_targets": len(targets),
            "recall@5": _pct(hits_at_5, len(targets)),
            "recall@10": _pct(hits_at_10, len(targets)),
            "recall@5_capped": _pct(hits_at_5, min(len(targets), 5)),
            "recall@10_capped": _pct(hits_at_10, min(len(targets), 10)),
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
        self_top10 = {
            "count": top10,
            "total": len(self_recall),
            "rate": _pct(top10, len(self_recall)),
        }
    return {
        "recall_at_5": {
            "hits": hit5,
            "total": total_rel,
            "rate": _pct(hit5, total_rel),
        },
        "recall_at_10": {
            "hits": hit10,
            "total": total_rel,
            "rate": _pct(hit10, total_rel),
        },
        "recall_at_5_capped": {
            "hits": hit5,
            "total": capped5_total,
            "rate": _pct(hit5, capped5_total),
        },
        "recall_at_10_capped": {
            "hits": hit10,
            "total": capped10_total,
            "rate": _pct(hit10, capped10_total),
        },
        "mrr": {
            "value": round(rr_sum / rr_count, 4) if rr_count else None,
            "queries_with_target": rr_count,
        },
        "irrelevant_false_pulls": {
            "count": false_pull_count,
            "returned": false_pull_returned,
            "rate": _pct(false_pull_count, false_pull_returned),
        },
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
        bucket = by_label.setdefault(
            case["label"], {"n": 0, "fired": 0, "hit_anchor": 0}
        )
        bucket["n"] += 1
        bucket["fired"] += int(case["fired"])
        bucket["hit_anchor"] += int(case["hit_anchor"])
    fired_total = sum(int(c["fired"]) for c in cases)
    return {
        "hint_total": {
            "count": fired_total,
            "total": len(cases),
            "rate": _pct(fired_total, len(cases)),
        },
        "by_label": {
            label: {
                "fired": {
                    "count": bucket["fired"],
                    "total": bucket["n"],
                    "rate": _pct(bucket["fired"], bucket["n"]),
                },
                "hit_anchor": {
                    "count": bucket["hit_anchor"],
                    "total": bucket["n"],
                    "rate": _pct(bucket["hit_anchor"], bucket["n"]),
                },
            }
            for label, bucket in sorted(by_label.items())
        },
    }


def score_conflict(raw: dict) -> dict[str, Any] | None:
    conflict = raw.get("conflict")
    if conflict is None:
        return None
    valid = [row for row in conflict if not row["skipped_member_replay"]]
    # runner 采集不带 shape：按 pair_id 从对集 join（对集是 shape 的权威源）；
    # 0.16.12 起合并 pairs_large.jsonl（large_unit 中大型用例组）；
    # 0.17.0 P2-0.1 起合并 pairs_noisy.jsonl（noisy 真实噪音对集）
    shape_of = {
        pair["pair_id"]: pair.get("shape") or "governed_negative"
        for pair in (
            _load_jsonl(FIXTURES / "conflict" / "pairs.jsonl")
            + _load_jsonl(FIXTURES / "conflict" / "pairs_large.jsonl")
            + _load_jsonl(FIXTURES / "conflict" / "pairs_noisy.jsonl")
        )
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
            "identified": {
                "count": sync + async_,
                "rate": _pct(sync + async_, len(rows)),
            },
        }

    true_rows = [r for r in valid if r["label"] == "true_conflict"]
    coexist_rows = [r for r in valid if r["label"] == "coexist"]
    noise_rows = [r for r in valid if r["label"] == "noise"]
    identified_all = [r for r in valid if r["sync"] or r["async"]]
    true_identified = [r for r in identified_all if r["label"] == "true_conflict"]
    return {
        "overall": {
            "precision": {
                "count": len(true_identified),
                "total": len(identified_all),
                "rate": _pct(len(true_identified), len(identified_all)),
            },
            "recall": {
                "count": len(true_identified),
                "total": len(true_rows),
                "rate": _pct(len(true_identified), len(true_rows)),
            },
            "coexist_false_positive": {
                "count": sum(1 for r in identified_all if r["label"] == "coexist"),
                "total": len(coexist_rows),
                "rate": _pct(
                    sum(1 for r in identified_all if r["label"] == "coexist"),
                    len(coexist_rows),
                ),
            },
            "skipped_member_replay": len(conflict) - len(valid),
        },
        "by_label": {
            "true_conflict": _outcome_row(true_rows),
            "coexist": _outcome_row(coexist_rows),
            "noise": _outcome_row(noise_rows),
        },
        "by_shape": {
            shape: _outcome_row(
                [
                    r
                    for r in valid
                    if shape_of.get(r["pair_id"], "governed_negative") == shape
                ]
            )
            for shape in (
                "scan_evolution",
                "governed_negative",
                "write_opposition",
                "large_unit",
                "noisy",
            )
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
        "perf": compute_perf(raw),
    }


# ---- perf：性能基线段（P0-T3，informational——不进回归门） -------------------


def _percentile(values: list[float], fraction: float) -> float | None:
    """nearest-rank 百分位（与 tools._pair_timing_summary 同口径）."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[idx], 1)


def _ms_stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_ms": _percentile(values, 0.5),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": round(max(values), 1) if values else None,
    }


def compute_perf(raw: dict) -> dict[str, Any] | None:
    """0.16.12 perf 基线：write/find/冲突窗耗时与预算消耗（纯信息位）.

    耗时指标天然有噪声，绝不进 gate()（_flatten 显式跳过顶层 perf 键）；
    未来独立 perf 门阈值在噪声带数据齐后另定。
    """
    from memory_arbiter.constants import (
        SEMANTIC_MAX_EVIDENCE_UNITS,
        SEMANTIC_MAX_EXAMINED_PAIRS,
    )

    replay = raw.get("replay_perf") or []
    queries = raw.get("queries") or []
    conflict = raw.get("conflict") or []
    if not (replay or queries or conflict):
        return None
    perf: dict[str, Any] = {}
    if replay:
        perf["write_ms"] = _ms_stats(
            [
                float(row["elapsed_ms"])
                for row in replay
                if isinstance(row.get("elapsed_ms"), (int, float))
            ]
        )
        fresh = [
            float(row["elapsed_ms"])
            for row in replay
            if isinstance(row.get("elapsed_ms"), (int, float))
            and not row.get("duplicate_replay")
        ]
        if fresh:
            perf["write_ms_fresh"] = _ms_stats(fresh)
    if queries:
        perf["find_ms"] = _ms_stats(
            [
                float(row["elapsed_ms"])
                for row in queries
                if isinstance(row.get("elapsed_ms"), (int, float))
            ]
        )
    if conflict:
        valid = [row for row in conflict if not row.get("skipped_member_replay")]
        writes = [
            float(row["right_write_ms"])
            for row in valid
            if isinstance(row.get("right_write_ms"), (int, float))
        ]
        if writes:
            perf["conflict_right_write_ms"] = _ms_stats(writes)
        receipts = [row.get("_receipt") or {} for row in valid]
        with_receipt = [r for r in receipts if r.get("status") is not None]
        completed = sum(1 for r in with_receipt if r.get("status") == "completed")
        pairs_values = [
            int(r["pairs_examined"])
            for r in with_receipt
            if isinstance(r.get("pairs_examined"), int)
        ]
        units_values = [
            int(row["units"]) for row in valid if isinstance(row.get("units"), int)
        ]
        notice_counts = [
            int(row["notice_count"])
            for row in valid
            if isinstance(row.get("notice_count"), int)
        ]
        job_ms_values = [
            float(r["elapsed_ms"])
            for r in with_receipt
            if isinstance(r.get("elapsed_ms"), (int, float))
        ]
        internal_values = [
            int(r["internal_conflicts"])
            for r in with_receipt
            if isinstance(r.get("internal_conflicts"), int)
        ]
        qwen_filter_confirmed = sum(
            int(
                ((r.get("deterministic_filter") or {}).get("internal_qwen_confirmed"))
                or 0
            )
            for r in with_receipt
        )
        qwen_filter_vetoed = sum(
            int(
                ((r.get("deterministic_filter") or {}).get("internal_qwen_vetoed")) or 0
            )
            for r in with_receipt
        )
        perf["conflict_window"] = {
            "n": len(valid),
            "receipt_n": len(with_receipt),
            "completed": completed,
            "completed_rate": _pct(completed, len(valid)) if valid else None,
            "avg_pairs_examined": (
                round(sum(pairs_values) / len(pairs_values), 2)
                if pairs_values
                else None
            ),
            # 0.16.12 eval contract: job 实际执行耗时（不含 3 秒同步等待窗）
            "job_ms": _ms_stats(job_ms_values),
            "avg_units": (
                round(sum(units_values) / len(units_values), 1)
                if units_values
                else None
            ),
            "avg_internal_pairs": (
                round(sum(internal_values) / len(internal_values), 2)
                if internal_values
                else None
            ),
            "internal_qwen_confirmed": qwen_filter_confirmed,
            "internal_qwen_vetoed": qwen_filter_vetoed,
            "pairs_examined_capped_rows": sum(
                1
                for r in with_receipt
                if "pairs_examined_capped" in (r.get("reasons_seen") or [])
            ),
            "units_capped_rows": sum(
                1 for u in units_values if u >= SEMANTIC_MAX_EVIDENCE_UNITS
            ),
            "pairs_budget": SEMANTIC_MAX_EXAMINED_PAIRS,
            "units_budget": SEMANTIC_MAX_EVIDENCE_UNITS,
            "avg_notice_count": (
                round(sum(notice_counts) / len(notice_counts), 2)
                if notice_counts
                else None
            ),
        }
    return perf or None


# ---- gate：基线对比（相对下降，provisional 阈值） ---------------------------


def _flatten(metrics: dict, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        # perf 段是耗时/预算信息位：噪声大，绝不进回归门（P0-T3 显式排除）
        if prefix == "" and key == "perf":
            continue
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
    ".miss.",
    "irrelevant_false_pulls",
    "coexist_false_positive",
)
_SIM_FALSE_LABELS = ("clearly_different", "opposite_semantics", "same_entity_diff_attr")
# 0.17.0 校准轮：conflict 的 noise 标签 firing 同为假阳性（半秒 bug 时代曾
# 3/27 sync），gate 方向必须 lower-is-better——首轮对比曾把 3→2 的改善误判 FAILED。
_CONFLICT_FALSE_LABELS = ("noise",)
_GATE_META_KEYS = (".skipped_member_replay", ".returned", ".queries_with_target")
# 0.17.0 cand2：sync/async 单项是 3 秒窗与 job 延迟的划分产物（行级化后 job
# 变慢、更多对跨窗补上≠行为回归）；行为指标=identified/miss/precision/recall。
_GATE_SPLIT_KEYS = (".sync.rate", ".async.rate")
# 0.17.0 cand3（owner R8 非对称收益口径）：noise 的 ASYNC firing=窗口外
# advisory 通知，与 C2 贴线误报同类（attr 门论证已接受）；SYNC firing 直接
# 出现在写响应里、侵入性高一档，保持受门。
_GATE_DOCTRINE_EXEMPT = ("noise.async.rate",)


def _lower_is_better(key: str) -> bool:
    if any(s in key for s in _LOWER_IS_BETTER_SUBSTR):
        return True
    if any(f".{label}." in key for label in _SIM_FALSE_LABELS):
        return True
    return any(f".{label}." in key for label in _CONFLICT_FALSE_LABELS)


def gate(
    current: dict, baseline: dict, rel_drop: float = DEFAULT_REL_DROP
) -> dict[str, Any]:
    # 0.16.12 第二轮对抗 review：跨语料对比会把不同分母的 rate 直接比较，
    # 既不报错也不可解释——两侧 env.conflict_corpus_version 必须都在位且一致。
    cur_corpus = (current.get("env") or {}).get("conflict_corpus_version")
    base_corpus = (baseline.get("env") or {}).get("conflict_corpus_version")
    if cur_corpus != base_corpus:
        return {
            "gate": "FAILED",
            "rel_drop_threshold": rel_drop,
            "failures": [
                {
                    "metric": "env.conflict_corpus_version",
                    "direction": "corpus_mismatch",
                    "baseline": base_corpus,
                    "current": cur_corpus,
                    "note": "conflict 对集语料版本不一致，拒绝跨语料对比；重建基线后重试",
                }
            ],
        }
    # 0.17.0 P2-0.1（review R1-5）：相似套件语料同样可变，版本不一致同样拒绝对比。
    # 兼容：一侧缺 similarity_corpus_version 键（旧基线）时跳过该校验。
    cur_sim_corpus = (current.get("env") or {}).get("similarity_corpus_version")
    base_sim_corpus = (baseline.get("env") or {}).get("similarity_corpus_version")
    if (
        cur_sim_corpus is not None
        and base_sim_corpus is not None
        and cur_sim_corpus != base_sim_corpus
    ):
        return {
            "gate": "FAILED",
            "rel_drop_threshold": rel_drop,
            "failures": [
                {
                    "metric": "env.similarity_corpus_version",
                    "direction": "corpus_mismatch",
                    "baseline": base_sim_corpus,
                    "current": cur_sim_corpus,
                    "note": "similarity 套件语料版本不一致，拒绝跨语料对比；重建基线后重试",
                }
            ],
        }
    cur = _flatten(current)
    base = _flatten(baseline)
    failures: list[dict[str, Any]] = []
    for key, base_value in sorted(base.items()):
        if key not in cur or key.endswith(
            (".count", ".total", ".n", "first_relevant_rank")
        ):
            continue
        if any(key.endswith(meta) for meta in _GATE_META_KEYS):
            continue
        if any(key.endswith(meta) for meta in _GATE_SPLIT_KEYS):
            continue
        if any(key.endswith(meta) for meta in _GATE_DOCTRINE_EXEMPT):
            continue
        cur_value = cur[key]
        if _lower_is_better(key):
            if base_value <= 0 or cur_value <= base_value:
                continue
            rise = (cur_value - base_value) / base_value
            if rise > rel_drop:
                failures.append(
                    {
                        "metric": key,
                        "direction": "lower_is_better",
                        "baseline": base_value,
                        "current": cur_value,
                        "relative_rise": round(rise, 4),
                        "threshold": rel_drop,
                    }
                )
            continue
        if base_value <= 0 or cur_value >= base_value:
            continue
        drop = (base_value - cur_value) / base_value
        if drop > rel_drop:
            failures.append(
                {
                    "metric": key,
                    "direction": "higher_is_better",
                    "baseline": base_value,
                    "current": cur_value,
                    "relative_drop": round(drop, 4),
                    "threshold": rel_drop,
                }
            )
    return {
        "gate": "FAILED" if failures else "PASSED",
        "rel_drop_threshold": rel_drop,
        "failures": failures,
    }


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
            lines.append(
                f"- 自召回 top10 = **{sr['count']}/{sr['total']}**（{sr['rate']}）"
            )
        lines.append("")
    sim = scored.get("similarity")
    if sim:
        lines += ["## 相似记忆提示（差异化能力）", ""]
        total = sim["hint_total"]
        lines.append(
            f"- 总提示 = **{total['count']}/{total['total']}**（{total['rate']}）"
        )
        for label, bucket in sim["by_label"].items():
            fired = bucket["fired"]
            lines.append(
                f"  - {label}: {fired['count']}/{fired['total']}（{fired['rate']}）"
            )
        lines.append("")
    conflict = scored.get("conflict")
    if conflict:
        lines += [
            "## 冲突识别（三结局：sync=3 秒窗内 / async=job 补上 / miss=漏检）",
            "",
        ]
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
    perf = scored.get("perf")
    if perf:
        lines += ["## 性能（informational——不进回归门）", ""]
        for label, key in (
            ("写入（fixture 重放）", "write_ms"),
            ("写入（非幂等重放）", "write_ms_fresh"),
            ("查询（recall 34 query）", "find_ms"),
            ("冲突对右侧写入", "conflict_right_write_ms"),
        ):
            bucket = perf.get(key)
            if bucket:
                lines.append(
                    f"- {label}：p50 **{bucket['p50_ms']}ms** · p95 **{bucket['p95_ms']}ms** · "
                    f"max {bucket['max_ms']}ms（n={bucket['n']}）"
                )
        window = perf.get("conflict_window")
        if window:
            lines.append(
                f"- 冲突窗：完成率 **{window['completed_rate']}**（{window['completed']}/{window['n']}） · "
                f"平均 Qwen 检查对数 {window['avg_pairs_examined']}（预算 {window['pairs_budget']}，"
                f"capped 行 {window['pairs_examined_capped_rows']}） · "
                f"job 实际耗时 p50 **{window['job_ms']['p50_ms']}ms** / p95 {window['job_ms']['p95_ms']}ms · "
                f"平均单元 {window['avg_units']}（预算 {window['units_budget']}，"
                f"capped 行 {window['units_capped_rows']}） · 平均 notice 数 {window['avg_notice_count']}"
            )
        lines.append("")
    if gate_result:
        lines += [
            "## 回归门（provisional 阈值，待 owner 依首份基线定正式门槛）",
            "",
            f"- {gate_result['gate']}（相对下降阈值 {gate_result['rel_drop_threshold']:.0%}）",
        ]
        for failure in gate_result["failures"]:
            if failure.get("direction") == "corpus_mismatch":
                lines.append(
                    f"  - {failure['metric']}: 基线 {failure['baseline']} vs 当前 {failure['current']}"
                    f"（{failure.get('note') or '语料版本不一致'}）"
                )
                continue
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
    parser.add_argument(
        "--baseline-write", type=Path, default=None, help="写入基线文件"
    )
    parser.add_argument("--baseline", type=Path, default=None, help="对比基线文件")
    parser.add_argument("--rel-drop", type=float, default=DEFAULT_REL_DROP)
    parser.add_argument(
        "--out", type=Path, default=None, help="报告输出（默认 run 同名 .md）"
    )
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
    payload = {
        "scored": scored,
        **({"gate_result": gate_result} if gate_result else {}),
    }
    scored_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if args.baseline_write:
        args.baseline_write.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_write.write_text(
            json.dumps(scored, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"[baseline] written -> {args.baseline_write}")
    print(markdown)
    print(f"[scored] -> {scored_path} / {out}")
    if gate_result and gate_result["gate"] == "FAILED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
