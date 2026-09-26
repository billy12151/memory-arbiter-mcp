#!/usr/bin/env python3
"""trim_recall_v2 — recall-v2 考卷修剪（owner 2026-09-26 拍板）.

规则（全部显式，可复查）：
  R1 短 target（content<500 字符）从 A/B 组标注中移除——生产库 ≥1k 占 52.9%，
     短条目不是主场景；K 组（关键词考题）引用的 3 条短记忆保护不删
     （关键词模式的本职就是精确命中短标题）。
  R2 真跨语言（zh 查询→en 记忆 / 反之）保留 2 条探针：A11×t-7e35f83b3b64
     （zh→en）与 B02 整题（en→mixed）；超出部分移除（A11×t-80939d044dc9）。
  R3 死题移除：修剪后 relevant=0 的 A/B 题（A01/A05/B09 本就 rel=0；
     A03/A07 的唯一 relevant 是短 target 被删）——对指标零贡献只耗运行。
  R4 target 失去全部标注引用后从 targets.jsonl 移除（self_recall 同步收缩）。
  R5 C/D（legal/far 无关探针）与 K 组原样保留。

corpus_version → recall-v2-kw-trim；旧基线因语料版本不一致被 gate 拒比（预期），
修剪后重跑全套门重录基线。
"""
from __future__ import annotations

import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "recall"
NEW_VERSION = "recall-v2-kw-trim"
SHORT = 500
CROSS_KEEP = {("A11", "t-7e35f83b3b64")}  # zh→en 探针保留 1 条；B02 整题为 en→mixed 探针
CROSS_DROP = {("A11", "t-80939d044dc9")}
PROTECT_K_SHORT = {"t-067ed356d645", "t-968fccbf3d26", "t-fad591d44a64"}


def main() -> int:
    queries = json.loads((D / "queries.json").read_text(encoding="utf-8"))["queries"]
    labels = [json.loads(l) for l in (D / "labels.jsonl").open()] if False else [
        json.loads(l) for l in (D / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    targets = [json.loads(l) for l in (D / "targets.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tmap = {t["fixture_key"]: t for t in targets}
    manifest = json.loads((D / "manifest.json").read_text(encoding="utf-8"))

    gold: dict[str, set[str]] = {}
    for l in labels:
        gold.setdefault(l["qid"], set()).add(l["fixture_key"])

    # R1+R2：标注级移除
    removed_pairs: list[tuple[str, str, str]] = []
    keep_labels: list[dict] = []
    for l in labels:
        qid, key = l["qid"], l["fixture_key"]
        t = tmap.get(key)
        reason = None
        if qid.startswith("K"):
            pass  # R5：K 组原样
        elif t is not None and len(t["content"]) < SHORT:
            reason = "short"
        elif (qid, key) in CROSS_DROP:
            reason = "cross-excess"
        if reason:
            removed_pairs.append((qid, key, reason))
        else:
            keep_labels.append(l)

    # R3：死题判定（修剪后 relevant=0 的 paraphrase/lookup 题）
    qkind = {q["qid"]: q["kind"] for q in queries}
    relevant_count: dict[str, int] = {}
    for l in keep_labels:
        if l["label"] == "relevant":
            relevant_count[l["qid"]] = relevant_count.get(l["qid"], 0) + 1
    removed_queries: list[str] = []
    for q in queries:
        qid, kind = q["qid"], q["kind"]
        if kind in ("paraphrase", "lookup") and qid in gold and relevant_count.get(qid, 0) == 0:
            removed_queries.append(qid)
    keep_labels = [l for l in keep_labels if l["qid"] not in removed_queries]
    keep_queries = [q for q in queries if q["qid"] not in removed_queries]

    # R4：失引用 target 移除
    referenced = {l["fixture_key"] for l in keep_labels}
    k_referenced = {l["fixture_key"] for l in labels
                    if l["qid"].startswith("K") and l["fixture_key"] in referenced | {
                        k for k in gold.get(l["qid"], set())}}
    keep_targets = [t for t in targets if t["fixture_key"] in referenced | k_referenced]
    dropped_targets = [t["fixture_key"] for t in targets if t["fixture_key"] not in referenced]

    # 一致性断言
    assert all(l["fixture_key"] in {t["fixture_key"] for t in keep_targets} for l in keep_labels)
    for q in keep_queries:
        if q["kind"] in ("paraphrase", "lookup"):
            n_rel = sum(1 for l in keep_labels if l["qid"] == q["qid"] and l["label"] == "relevant")
            assert n_rel >= 1, f"{q['qid']} still dead"
    assert len({l["qid"] for l in keep_labels} - {q["qid"] for q in keep_queries}) == 0

    # 写回
    manifest["corpus_version"] = NEW_VERSION
    manifest["targets"] = len(keep_targets)
    manifest["labeled_qids"] = len({l["qid"] for l in keep_labels})
    manifest["trimmed"] = {
        "date": "2026-09-26",
        "rules": "R1 短(<500)移除(K组3条保护); R2 跨语言保留2条探针(A11 zh→en + B02 en→mixed); "
                 "R3 死题(relevant=0)移除; R4 失引用 target 移除",
        "removed_queries": removed_queries,
        "removed_label_pairs": len(removed_pairs),
        "removed_targets": dropped_targets,
        "targets_before_after": [len(targets), len(keep_targets)],
        "queries_before_after": [len(queries), len(keep_queries)],
    }
    manifest_q = {"corpus_version": NEW_VERSION, "queries": keep_queries}
    (D / "queries.json").write_text(
        json.dumps(manifest_q, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (D / "labels.jsonl").write_text(
        "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in keep_labels), encoding="utf-8")
    (D / "targets.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in keep_targets), encoding="utf-8")
    (D / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"removed queries ({len(removed_queries)}): {removed_queries}")
    print(f"removed label pairs: {len(removed_pairs)} (short={sum(1 for _,_,r in removed_pairs if r=='short')}, "
          f"cross-excess={sum(1 for _,_,r in removed_pairs if r=='cross-excess')})")
    print(f"removed orphan targets: {len(dropped_targets)}")
    print(f"targets {len(targets)}->{len(keep_targets)}  queries {len(queries)}->{len(keep_queries)}  "
          f"labels {len(labels)}->{len(keep_labels)}")
    print(f"corpus_version -> {NEW_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
