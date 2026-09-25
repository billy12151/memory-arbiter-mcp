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


def _query_rank_row(
    query: dict[str, Any], relevant_by_qid: dict[str, set[str]]
) -> dict[str, Any]:
    """每题核心检索量——主循环与 keyword 分桶共用的唯一实现（struct 修复：
    keyword 分桶此前是主循环的手抄副本且已漂移（缺 capped 分母），收敛后
    分桶自动继承主循环后续新增的每题指标）。"""
    targets = relevant_by_qid.get(query["qid"], set())
    ranked = [hit["fixture_key"] for hit in query.get("hits") or []]
    return {
        "qid": query["qid"],
        "kind": query["kind"],
        "ranked": ranked,
        "n_targets": len(targets),
        "hits_at_5": len(targets & set(ranked[:5])),
        "hits_at_10": len(targets & set(ranked[:10])),
        "capped5_denom": min(len(targets), 5),
        "capped10_denom": min(len(targets), 10),
        "first_rank": next(
            (i + 1 for i, key in enumerate(ranked) if key in targets),
            None,
        ),
    }


def _aggregate_rank_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """每题核心量微平均成 recall_at_5/@10（classic + capped）与 MRR。

    Capped recall (gate-v2 拍板 5): "everything that belongs in the top-k
    got in" — the denominator per query is min(|relevant|, k), so a query
    with 6 relevant targets and a full top-5 scores 1.0 instead of being
    punished for the k+1th target no ranking could have returned. Micro-
    averaged: Σhits / Σmin(R_i, k); classic and capped are both reported
    so old numbers stay traceable.
    """
    hit5 = sum(r["hits_at_5"] for r in rows)
    hit10 = sum(r["hits_at_10"] for r in rows)
    total_rel = sum(r["n_targets"] for r in rows)
    capped5_total = sum(r["capped5_denom"] for r in rows)
    capped10_total = sum(r["capped10_denom"] for r in rows)
    rr = [1.0 / r["first_rank"] for r in rows if r["first_rank"]]
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
            "value": round(sum(rr) / len(rr), 4) if rr else None,
            "queries_with_target": len(rr),
        },
    }


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

    # 每题核心量走唯一实现（struct 修复见 _query_rank_row），主循环只补
    # 自己的扩展量：per_query 明细行与无关误召回（C/D 组）。
    rank_rows = [_query_rank_row(query, relevant_by_qid) for query in queries]
    false_pull_count = 0
    false_pull_returned = 0
    per_query: list[dict[str, Any]] = []
    for query, stats in zip(queries, rank_rows):
        row = {
            "qid": stats["qid"],
            "kind": stats["kind"],
            "relevant_targets": stats["n_targets"],
            "recall@5": _pct(stats["hits_at_5"], stats["n_targets"]),
            "recall@10": _pct(stats["hits_at_10"], stats["n_targets"]),
            "recall@5_capped": _pct(stats["hits_at_5"], stats["capped5_denom"]),
            "recall@10_capped": _pct(stats["hits_at_10"], stats["capped10_denom"]),
            "first_relevant_rank": stats["first_rank"],
        }
        if query["kind"] in {"legal", "far"}:
            false_hits = [k for k in stats["ranked"] if k in labeled_keys]
            false_pull_count += len(false_hits)
            false_pull_returned += len(stats["ranked"])
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

    # 检索线 K3：keyword 模式分桶（题级命中率与 target 级微平均双口径——
    # 两口径可一过一炸，必须都出，R2-P1-8）；band 子桶按 queries 的
    # expected_band（midband=救济带专考 / above=≥0.75 不误伤专考；
    # subfloor 配额 owner 2026-09-24 拍板撤销）。
    keyword_bucket = _score_keyword_bucket(queries, relevant_by_qid)
    return {
        **_aggregate_rank_rows(rank_rows),
        "irrelevant_false_pulls": {
            "count": false_pull_count,
            "returned": false_pull_returned,
            "rate": _pct(false_pull_count, false_pull_returned),
        },
        "self_recall_top10": self_top10,
        "keyword_bucket": keyword_bucket,
        "per_query": per_query,
    }


def _score_keyword_bucket(
    queries: list[dict[str, Any]],
    relevant_by_qid: dict[str, set[str]],
) -> dict[str, Any] | None:
    k_queries = [query for query in queries if query.get("kind") == "keyword"]
    if not k_queries:
        return None

    def _accumulate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        # 核心每题量走与主循环同一实现（struct 修复：此前是手抄副本且缺
        # capped 变体——分桶自动继承主循环全套 classic+capped+MRR），
        # 分桶只补自己的题级命中率双口径。
        rank_rows = [_query_rank_row(q, relevant_by_qid) for q in rows]
        q_top5 = sum(1 for r in rank_rows if r["hits_at_5"] > 0)
        q_top10 = sum(1 for r in rank_rows if r["hits_at_10"] > 0)
        return {
            "queries": len(rows),
            "query_top5_hit_rate": _pct(q_top5, len(rows)),
            "query_top10_hit_rate": _pct(q_top10, len(rows)),
            **_aggregate_rank_rows(rank_rows),
        }

    bucket = _accumulate(k_queries)
    by_band: dict[str, Any] = {}
    for band in ("midband", "above"):
        rows = [q for q in k_queries if q.get("expected_band") == band]
        if rows:
            by_band[band] = _accumulate(rows)
    bucket["by_band"] = by_band
    return bucket


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


def score_conflict_claims(raw: dict) -> dict[str, Any] | None:
    """Gate-v2 G7b: per-channel metrics for the claims corpus (B =
    deterministic claims×claims, C = claims×sentence KNN), plus the merged
    totals. Row shape matches the conflict suite; `channel` rides the
    corpus."""
    conflict = raw.get("conflict_claims")
    if conflict is None:
        return None
    valid = [row for row in conflict if not row["skipped_member_replay"]]

    def _bucket(rows: list[dict]) -> dict[str, Any]:
        sync = sum(1 for r in rows if r["sync"])
        async_ = sum(1 for r in rows if r["async"])
        miss = sum(1 for r in rows if r["notice_missing"])
        identified = sync + async_
        true_rows = [r for r in rows if r["label"] == "true_conflict"]
        true_identified = sum(
            1 for r in rows if r["label"] == "true_conflict" and (r["sync"] or r["async"])
        )
        identified_rows = [r for r in rows if r["sync"] or r["async"]]
        return {
            "n": len(rows),
            "sync": {"count": sync, "rate": _pct(sync, len(rows))},
            "async": {"count": async_, "rate": _pct(async_, len(rows))},
            "miss": {"count": miss, "rate": _pct(miss, len(rows))},
            "identified": {"count": identified, "rate": _pct(identified, len(rows))},
            "recall": {
                "count": true_identified,
                "total": len(true_rows),
                "rate": _pct(true_identified, len(true_rows)),
            },
            "precision": {
                "count": true_identified,
                "total": len(identified_rows),
                "rate": _pct(true_identified, len(identified_rows)),
            },
        }

    def _by_label(rows: list[dict]) -> dict[str, Any]:
        labels = sorted({r["label"] for r in rows})
        return {
            label: _bucket([r for r in rows if r["label"] == label])
            for label in labels
        }

    b_rows = [r for r in valid if r.get("channel") == "B"]
    c_rows = [r for r in valid if r.get("channel") == "C"]
    return {
        "channel_b": _bucket(b_rows),
        "channel_b_by_label": _by_label(b_rows),
        "channel_c": _bucket(c_rows),
        "channel_c_by_label": _by_label(c_rows),
        "merged": _bucket(valid),
        "skipped_member_replay": len(conflict) - len(valid),
    }


def score_conflict_comprehensive(raw: dict) -> dict[str, Any] | None:
    """0.17.0 Q1/H1 (owner D4): the headline conflict metric — 综合召回.
    conflict ∪ conflict_claims 两语料合并的 any-channel 口径：
    Σidentified(true_conflict) / Σtrue_conflict；综合精确 = 真 notice 数 /
    总 notice 数（两语料合并，coexist FP 计入分母）。分母明细给出两语料各
    自的分子分母贡献。块形状只含 recall/precision/sources-counts——
    recall/precision.rate 进相对门；sources 计数由 _GATE_META_KEYS 排除
    （H1 修复批：分母明细是诊断量，review R2-6 + 对抗 review）。"""
    conflict = raw.get("conflict")
    claims = raw.get("conflict_claims")
    if conflict is None and claims is None:
        return None

    def _valid(rows: "list[dict] | None") -> list[dict]:
        return [r for r in (rows or []) if not r.get("skipped_member_replay")]

    def _true_total(rows: list[dict]) -> int:
        return sum(1 for r in rows if r["label"] == "true_conflict")

    def _true_identified(rows: list[dict]) -> int:
        return sum(
            1 for r in rows if r["label"] == "true_conflict" and (r["sync"] or r["async"])
        )

    def _identified(rows: list[dict]) -> int:
        return sum(1 for r in rows if r["sync"] or r["async"])

    sent = _valid(conflict)
    clm = _valid(claims)
    true_total = _true_total(sent) + _true_total(clm)
    true_id = _true_identified(sent) + _true_identified(clm)
    id_total = _identified(sent) + _identified(clm)
    return {
        "recall": {
            "count": true_id, "total": true_total, "rate": _pct(true_id, true_total),
        },
        "precision": {
            "count": true_id, "total": id_total, "rate": _pct(true_id, id_total),
        },
        "sources": {
            "conflict": {
                "true_total": _true_total(sent), "true_identified": _true_identified(sent),
            },
            "conflict_claims": {
                "true_total": _true_total(clm), "true_identified": _true_identified(clm),
            },
        },
    }


def score_conflict_attribution(raw: dict) -> dict[str, Any] | None:
    """0.17.0 H1: 分通道归因表（诊断，不进 gate——gate() 在 flatten 前整块
    剔除，基线文件里留作人工对照）。

    Q1 之后 pairs_examined 是 job 全局口径（internal+C+A-cross），A qwen 的
    归因直读回执 qwen_budget 分量（review R1-6）；A direct 读 direct_verdicts。
    旧 raw 无这些键 → 各按 0 计（纯函数对存档 r1-r5 仍可跑）。"""

    def _valid(rows: "list[dict] | None") -> list[dict]:
        return [r for r in (rows or []) if not r.get("skipped_member_replay")]

    sent = _valid(raw.get("conflict"))
    clm = _valid(raw.get("conflict_claims"))

    def _receipt_sum(rows: list[dict], key: str) -> int:
        total = 0
        for row in rows:
            receipt = row.get("_receipt") or {}
            if key == "direct_verdicts":
                total += int(receipt.get("direct_verdicts") or 0)
            else:
                total += int((receipt.get("qwen_budget") or {}).get(key) or 0)
        return total

    return {
        "a_channel_identified": sum(1 for r in sent if r["sync"] or r["async"]),
        "a_direct_verdicts_total": _receipt_sum(sent, "direct_verdicts"),
        "a_qwen_internal_total": _receipt_sum(sent, "internal"),
        "a_qwen_cross_total": _receipt_sum(sent, "a_cross"),
        "channel_b_identified": sum(
            1 for r in clm if r.get("channel") == "B" and (r["sync"] or r["async"])
        ),
        "channel_c_identified": sum(
            1 for r in clm if r.get("channel") == "C" and (r["sync"] or r["async"])
        ),
        "channel_c_qwen_total": _receipt_sum(clm, "channel_c"),
    }


def score_all(raw: dict) -> dict[str, Any]:
    consistency = raw.get("batch_find_consistency")
    return {
        "mema_version": raw.get("mema_version"),
        "corpus_version": raw.get("corpus_version"),
        "env": raw.get("env"),
        "recall": score_recall(raw),
        "batch_find_consistency": (
            {
                "queries": consistency.get("queries"),
                "batches": consistency.get("batches"),
                "mismatch_count": consistency.get("mismatch_count"),
            }
            if consistency
            else None
        ),
        "similarity": score_similarity(raw),
        "conflict": score_conflict(raw),
        "conflict_claims": score_conflict_claims(raw),
        "conflict_comprehensive": score_conflict_comprehensive(raw),
        "conflict_attribution": score_conflict_attribution(raw),
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
    from memory_arbiter.constants import SEMANTIC_MAX_EXAMINED_PAIRS

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
            "rows_capped_rows": sum(
                1
                for r in with_receipt
                if "rows_capped" in (r.get("reasons_seen") or [])
            ),
            "pairs_budget": SEMANTIC_MAX_EXAMINED_PAIRS,
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
# 2026-09-25 owner 拍板（E3 实证：FP 改善 7→6 被旧基线门误判 FAILED）：负样本
# governed_negative 的 firing 类指标（identified/sync）同规——用精确段
# 匹配而非整标签，miss（负样本上不报=正确）保持 higher-is-better 不受牵连。
# 0.17.0 review R2：".miss." 全局子串与上面「miss 保持 higher-is-better」的
# 声明自相矛盾——负样本桶（by_shape governed_negative、by_label noise）的
# miss 上升是改善，却被 _LOWER_IS_BETTER_SUBSTR 扫进 lower-is-better
# （E3 同类事故二次形态：真改善 >10% 会假 FAILED）。_negative_bucket 收口。
# 0.17.0 review R2 复评：noisy 从负样本桶收口移除——pairs_noisy 26 对中
# true_conflict 15 + coexist 2 + noise 9，58% 是真对，整桶按负样本定方向
# 两头都错：miss 上升（真对漏检变多）被当改善放行、firing 上升（多为真对
# 正当检出）被当假阳性误杀。noisy 回归普通桶口径（miss 受门、sync/async
# 走 cand2 豁免）；governed_negative（纯负）与 noise 标签不变。
_CONFLICT_FALSE_LABELS = (
    "noise",
    "governed_negative.identified",
    "governed_negative.sync",
)
_GATE_NEGATIVE_LABELS = ("noise", "governed_negative")


def _negative_bucket(key: str) -> bool:
    return any(f".{label}." in key for label in _GATE_NEGATIVE_LABELS)
_GATE_META_KEYS = (".skipped_member_replay", ".returned", ".queries_with_target",
                   # 检索线 K3（R2-P2-3）：语料元数据不是行为指标——同语料
                   # 内恒定，不进相对门（corpus bump 由 corpus_version 前置校验拦）
                   ".queries", ".batches",
                   # H1 修复批（mema #1066 对抗 review）：comprehensive 的
                   # sources 分母明细是 recall 块的分解诊断量（true_identified
                   # 与 recall.count 同值），单独受门只会制造假回归。
                   ".true_total", ".true_identified")
# 0.17.0 cand2：sync/async 单项是 3 秒窗与 job 延迟的划分产物（行级化后 job
# 变慢、更多对跨窗补上≠行为回归）；行为指标=identified/miss/precision/recall。
_GATE_SPLIT_KEYS = (".sync.rate", ".async.rate")


def _lower_is_better(key: str) -> bool:
    # 0.17.0 review R2：负样本桶的 miss 必须先于 ".miss." 子串规则判断——
    # 负样本上不报=正确，miss 上升是改善（higher-is-better）。
    if ".miss." in key and _negative_bucket(key):
        return False
    if any(s in key for s in _LOWER_IS_BETTER_SUBSTR):
        return True
    if any(f".{label}." in key for label in _SIM_FALSE_LABELS):
        return True
    return any(f".{label}." in key for label in _CONFLICT_FALSE_LABELS)


def _corpus_version_mismatches(
    current: dict, baseline: dict
) -> list[dict[str, Any]]:
    """三套件语料版本前置校验的唯一实现（struct 修复：gate() 里原是三段
    复制粘贴的早退块）。按既有优先级产出不匹配失败项：recall 顶层 →
    conflict → similarity；similarity 侧兼容旧基线（任一侧缺键即跳过）。
    消息文案与失败项负载逐字保持原样。"""
    cur_env = current.get("env") or {}
    base_env = baseline.get("env") or {}
    mismatches: list[dict[str, Any]] = []
    for metric, cur, base, note, skip_if_missing in (
        (
            "corpus_version",
            current.get("corpus_version"),
            baseline.get("corpus_version"),
            "recall 考卷语料版本不一致，拒绝跨语料对比；重建基线后重试",
            False,
        ),
        (
            "env.conflict_corpus_version",
            cur_env.get("conflict_corpus_version"),
            base_env.get("conflict_corpus_version"),
            "conflict 对集语料版本不一致，拒绝跨语料对比；重建基线后重试",
            False,
        ),
        (
            "env.similarity_corpus_version",
            cur_env.get("similarity_corpus_version"),
            base_env.get("similarity_corpus_version"),
            "similarity 套件语料版本不一致，拒绝跨语料对比；重建基线后重试",
            True,
        ),
    ):
        if skip_if_missing and (cur is None or base is None):
            continue
        if cur != base:
            mismatches.append(
                {
                    "metric": metric,
                    "direction": "corpus_mismatch",
                    "baseline": base,
                    "current": cur,
                    "note": note,
                }
            )
    return mismatches


def gate(
    current: dict, baseline: dict, rel_drop: float = DEFAULT_REL_DROP
) -> dict[str, Any]:
    # 检索线 K3（R2-P1-9）：语料版本不一致拒绝跨语料对比——corpus bump 后
    # 基线未重建会被静默当成回归/持平（版本号不进 _flatten，相对门看不见
    # 它）。三套件校验收敛为单一 helper（struct 修复），仍先于其余门逻辑、
    # 首个不匹配即拦（单失败负载与文案同旧三段逐字一致）。
    mismatches = _corpus_version_mismatches(current, baseline)
    if mismatches:
        return {
            "gate": "FAILED",
            "rel_drop_threshold": rel_drop,
            "failures": [mismatches[0]],
        }
    # H1 修复批（mema #1066 对抗 review）：conflict_attribution 是分通道
    # 归因诊断块——派发计数随检测效率正当波动（过滤变好 → Qwen 派发变少
    # = 改善），默认 higher-is-better 方向会把 D1 的预期效果判成回归。
    # 整块不进相对门（基线文件里可以留作人工对照）。
    diagnostic_blocks = ("conflict_attribution",)
    cur = _flatten({k: v for k, v in current.items() if k not in diagnostic_blocks})
    base = _flatten({k: v for k, v in baseline.items() if k not in diagnostic_blocks})
    failures: list[dict[str, Any]] = []
    for key, base_value in sorted(base.items()):
        if key not in cur or key.endswith(
            (".count", ".total", ".n", "first_relevant_rank")
        ):
            continue
        if key.startswith("env."):
            # 提速批（mema #1066）：env 是环境元数据（语料计数/同步窗/模型
            # 路径 sha）——同步窗等运行配置正当可调，绝不进行为相对门。
            continue
        if any(key.endswith(meta) for meta in _GATE_META_KEYS):
            continue
        if any(key.endswith(meta) for meta in _GATE_SPLIT_KEYS):
            # 0.17.0 cand3（owner R8 非对称收益口径）：负样本桶的 SYNC firing
            # 直接出现在写响应、侵入性高一档——保持受门（lower-is-better，
            # 上升才是回归）；其余 sync/async 单项是 3 秒窗与 job 延迟的划分
            # 产物（cand2），跳过。原 _GATE_DOCTRINE_EXEMPT（noise.async.rate）
            # 与 governed_negative.async 死条目一并删除：async 窗口外豁免口径
            # 下二者永不可达。
            if not (key.endswith(".sync.rate") and _negative_bucket(key)):
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
        kb = recall.get("keyword_bucket")
        if kb:
            lines.append(
                f"- 关键词模式（{kb['queries']} 题）题级 top10 命中 = **{kb['query_top10_hit_rate']}** · "
                f"R@10 = {kb['recall_at_10']['rate']}（target 级） · MRR = {kb['mrr']['value']}"
                "（K 组每题单 target，双口径当前恒等）"
            )
            for band, row in (kb.get("by_band") or {}).items():
                lines.append(
                    f"  - {band}（{row['queries']} 题）：题级 top5 命中 **{row['query_top5_hit_rate']}** · "
                    f"题级 top10 命中 {row['query_top10_hit_rate']} · R@5 = {row['recall_at_5']['rate']}"
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
    comprehensive = scored.get("conflict_comprehensive")
    if comprehensive:
        lines += [
            "## 冲突综合召回（D4 headline：conflict ∪ conflict_claims，any-channel）",
            "",
            f"- Recall = **{comprehensive['recall']['rate']}**（{comprehensive['recall']['count']}/{comprehensive['recall']['total']}）",
            f"- Precision = **{comprehensive['precision']['rate']}**（{comprehensive['precision']['count']}/{comprehensive['precision']['total']}）",
        ]
        for source, row in (comprehensive.get("sources") or {}).items():
            lines.append(
                f"  - {source}：true {row['true_identified']}/{row['true_total']}"
            )
        lines.append("")
    attribution = scored.get("conflict_attribution")
    if attribution:
        lines += [
            "## 分通道归因（诊断，不进 gate）",
            "",
            f"- 句料库 identified（A 链路）= **{attribution['a_channel_identified']}**"
            f"（回执级 direct 直出 {attribution['a_direct_verdicts_total']} ·"
            f" Qwen internal {attribution['a_qwen_internal_total']} ·"
            f" A-cross {attribution['a_qwen_cross_total']}）",
            f"- claims 语料 identified：B = **{attribution['channel_b_identified']}** ·"
            f" C = **{attribution['channel_c_identified']}**"
            f"（C Qwen 派发 {attribution['channel_c_qwen_total']}）",
            "",
        ]
    perf = scored.get("perf")
    if perf:
        lines += ["## 性能（informational——不进回归门）", ""]
        # struct 修复：题数标签从数据推导（recall.per_query 行数=实际查询
        # 数；旧硬编码 34 在语料扩到 47 题后已过期）。perf-only 渲染（无
        # recall 块）回退 find_ms 样本数。
        n_find = None
        if recall and recall.get("per_query"):
            n_find = len(recall["per_query"])
        elif (perf.get("find_ms") or {}).get("n"):
            n_find = perf["find_ms"]["n"]
        find_label = f"查询（recall {n_find} query）" if n_find else "查询"
        for label, key in (
            ("写入（fixture 重放）", "write_ms"),
            ("写入（非幂等重放）", "write_ms_fresh"),
            (find_label, "find_ms"),
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
                f"平均单元 {window['avg_units']}（rows_capped 行 {window['rows_capped_rows']}）"
                f" · 平均 notice 数 {window['avg_notice_count']}"
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
