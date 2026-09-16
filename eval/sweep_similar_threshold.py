#!/usr/bin/env python3
"""sweep_similar_threshold — P0-1 调优线①：WRITE_SIMILAR_SUBJECT_REASON 阈值标定（方案 A13-发现1）.

对相似语料 48 例 + 真库同 workspace 对，扫描 subject 相似阈值（tags jaccard
固定 0.8，双门语义不变），输出每档的真近似命中 / 三类误报 / 真库存量噪音。

判定逻辑逐字复用产品实现（WritePipeline 的 _normalized_subject /
_tag_jaccard / _DIGIT_RUN / 长度预检 / 系列抑制）——import 而非抄写，
产品改实现本 sweep 自动跟随。

近似声明：静态 sweep 不含 KNN 候选召回（subject_tags_vec top-k）与
MAX_HINTS=2 截断——双门全过是对「会提示」的近似上界；语料 48 例为单对
场景无截断影响，真库存量数字按对计数读作「提示量级」。

用法：
  python eval/sweep_similar_threshold.py [--low 0.80] [--high 0.96] [--step 0.01]
"""
from __future__ import annotations

import argparse
import difflib
import itertools
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.constants import WRITE_SIMILAR_TAG_JACCARD  # noqa: E402
from memory_arbiter.pipeline.write import WritePipeline  # noqa: E402

DEFAULT_SOURCE = Path.home() / ".local/share/memory-arbiter/memory.sqlite3"
CORPUS = REPO / "eval" / "fixtures" / "similarity" / "cases.jsonl"


def _pair_verdict(anchor_subject: str, anchor_tags: list[str],
                  variant_subject: str, variant_tags: list[str],
                  threshold: float) -> tuple[bool, float, float]:
    """复刻 _similar_active_notice 的双门+预检+系列抑制，返回 (过门, ratio, jaccard)."""
    subject = WritePipeline._normalized_subject(anchor_subject)
    row_subject = WritePipeline._normalized_subject(variant_subject)
    if not subject or not row_subject:
        return False, 0.0, 0.0
    shorter = min(len(subject), len(row_subject))
    if 2.0 * shorter < threshold * (len(subject) + len(row_subject)):
        return False, 0.0, 0.0
    if row_subject != subject and WritePipeline._DIGIT_RUN.sub("#", row_subject) == WritePipeline._DIGIT_RUN.sub("#", subject):
        return False, 0.0, 0.0  # 数字系列抑制
    ratio = difflib.SequenceMatcher(None, subject, row_subject).ratio()
    if ratio < threshold:
        return False, ratio, 0.0
    jaccard = WritePipeline._tag_jaccard(anchor_tags, variant_tags)
    if jaccard < WRITE_SIMILAR_TAG_JACCARD:
        return False, ratio, jaccard
    return True, ratio, jaccard


def sweep_corpus(thresholds: list[float]) -> list[dict]:
    cases = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for threshold in thresholds:
        label_hits: Counter = Counter()
        label_totals: Counter = Counter()
        for case in cases:
            anchor, variant = case["anchor"], case["variant"]
            label_totals[case["label"]] += 1
            passed, _, _ = _pair_verdict(
                anchor["subject"], anchor["tags"], variant["subject"], variant["tags"], threshold,
            )
            label_hits[case["label"]] += int(passed)
        rows.append({
            "threshold": threshold,
            "near": f"{label_hits['true_near_dup']}/{label_totals['true_near_dup']}",
            "false_by_label": {
                label: f"{label_hits[label]}/{label_totals[label]}"
                for label in ("clearly_different", "same_entity_diff_attr", "opposite_semantics")
            },
        })
    return rows


def sweep_production(thresholds: list[float]) -> list[dict]:
    conn = sqlite3.connect(f"file:{DEFAULT_SOURCE}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT subject, tags, COALESCE(NULLIF(workspace_canonical,''),workspace) AS ws "
        "FROM memories WHERE status='active' AND subject IS NOT NULL AND TRIM(subject)<>'' "
        "AND COALESCE(NULLIF(workspace_canonical,''),workspace) != 'mema-twin'",
    ).fetchall()
    conn.close()
    by_ws: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_ws.setdefault(row["ws"], []).append(row)

    # 对的 (ratio, jaccard, suppressed, len_gate_ratio) 一次算完，阈值档只做过滤
    pair_stats: list[tuple[float, float, bool, float]] = []
    for ws_rows in by_ws.values():
        for a, b in itertools.combinations(ws_rows, 2):
            subject = WritePipeline._normalized_subject(a["subject"])
            row_subject = WritePipeline._normalized_subject(b["subject"])
            if not subject or not row_subject:
                continue
            suppressed = (
                row_subject != subject
                and WritePipeline._DIGIT_RUN.sub("#", row_subject) == WritePipeline._DIGIT_RUN.sub("#", subject)
            )
            len_gate = (2.0 * min(len(subject), len(row_subject))) / (len(subject) + len(row_subject))
            ratio = difflib.SequenceMatcher(None, subject, row_subject).ratio()
            try:
                tags_a = json.loads(a["tags"]) if a["tags"] else []
                tags_b = json.loads(b["tags"]) if b["tags"] else []
            except ValueError:
                tags_a, tags_b = [], []
            jaccard = WritePipeline._tag_jaccard(tags_a, tags_b)
            pair_stats.append((ratio, jaccard, suppressed, len_gate))
    total_pairs = len(pair_stats)
    out = []
    for threshold in thresholds:
        passed = sum(
            1 for ratio, jaccard, suppressed, len_gate in pair_stats
            if not suppressed and len_gate >= threshold and ratio >= threshold
            and jaccard >= WRITE_SIMILAR_TAG_JACCARD
        )
        out.append({"threshold": threshold, "active_pairs": passed, "total_pairs": total_pairs})
    return out


def verify_threshold(threshold: float) -> dict:
    """真机验证：运行时 patch 阈值（不动产品文件），跑真实 similarity 套件."""
    import memory_arbiter.pipeline.write as write_module

    original = write_module.WRITE_SIMILAR_SUBJECT_RATIO
    write_module.WRITE_SIMILAR_SUBJECT_RATIO = threshold
    try:
        import runner as runner_module

        embed_model = runner_module.default_embed_model()
        cases = [
            json.loads(line)
            for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        with runner_module.temp_library(embed_model) as tools:
            outcome = runner_module.run_similarity_suite(tools, cases)
    finally:
        write_module.WRITE_SIMILAR_SUBJECT_RATIO = original
    by_label: Counter = Counter()
    totals: Counter = Counter()
    detail = []
    for case in outcome["cases"]:
        totals[case["label"]] += 1
        by_label[case["label"]] += int(case["fired"])
        detail.append(f"{case['case_id']} fired={int(case['fired'])}")
    return {
        "threshold": threshold,
        "fired_by_label": dict(by_label),
        "totals": dict(totals),
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=float, default=0.80)
    parser.add_argument("--high", type=float, default=0.96)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--verify", type=float, default=None,
                        help="真机验证该阈值：运行时 patch 后跑 similarity 套件")
    args = parser.parse_args()

    if args.verify is not None:
        result = verify_threshold(args.verify)
        print(f"[verify@{args.verify}] 真机 similarity 套件命中分布：")
        for label, count in sorted(result["fired_by_label"].items()):
            print(f"  {label:22s} {count}/{result['totals'][label]}")
        return 0

    steps = max(1, round((args.high - args.low) / args.step) + 1)
    thresholds = [round(args.low + i * args.step, 2) for i in range(steps)]
    corpus_rows = sweep_corpus(thresholds)
    prod_rows = sweep_production(thresholds) if DEFAULT_SOURCE.exists() else []

    print(f"subject 阈值 │ 真近似命中 │ 明显不同 │ 同实体异属性 │ 相反语义 │ 真库存量双门全过对")
    print("---|---|---|---|---|---")
    for corpus_row in corpus_rows:
        threshold = corpus_row["threshold"]
        prod = next((p for p in prod_rows if p["threshold"] == threshold), {})
        false = corpus_row["false_by_label"]
        print(
            f"{threshold:.2f} │ {corpus_row['near']} │ {false['clearly_different']} │ "
            f"{false['same_entity_diff_attr']} │ {false['opposite_semantics']} │ "
            f"{prod.get('active_pairs', '-')} / {prod.get('total_pairs', '-')}"
        )
    out_path = REPO / "eval" / "results" / "sweep-similar-threshold.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"corpus": corpus_rows, "production": prod_rows,
                    "tag_jaccard": WRITE_SIMILAR_TAG_JACCARD}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\n[sweep] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
