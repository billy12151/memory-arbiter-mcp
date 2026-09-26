#!/usr/bin/env python3
"""export_recall_len — recall-v3-len 语料导出（长度×语言分层，owner 2026-09-26 拍板方向）.

与 export_fixtures.py（recall-v1/v2）同一只读导出纪律，差异：
  1. targets 按 (语言 × 长度) 分层配额采样，中长/很长加权——生产库 ≥1k 占 52.9%，
     而现有 recall-v2 targets <500 占 28.6%，长度维度欠配；
  2. 语言分桶：zh（CJK 主导）/ mixed（中英混）/ en（ASCII 主导），生产分布
     mixed 80% / zh 15% / en 4.5%，语料按此配额（en 滤掉 self-test 标记垃圾）；
  3. 每个 fixture 行携带 lang 与 len_bucket，供 score.py 语言分桶指标与噪音指标用；
  4. queries.json / labels.jsonl 由 _digest.md 人工出题后由 --assemble 组装
     （paraphrase 白话问句，风格对齐 recall-v2 A 组）。

只读：immutable=1 打开源库；twin 桶排除；内容寻址 fixture_key 与现有口径一致。
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path.home() / ".local/share/memory-arbiter/memory.sqlite3"
OUT_DIR = REPO / "eval" / "fixtures" / "recall-len"
CORPUS_VERSION = "recall-v3-len"

EXCLUDE_WS = {"mema-twin", "mema-twin-dev"}
EN_JUNK_MARKERS = ("self-test marker", "fulltest", "full-test marker")

TARGET_QUOTAS = {
    # (lang, len_bucket) -> quota；中长/很长加权（生产 1k-2k 29%/2k-4k 15%/>4k 9%）
    ("mixed", "<500"): 8, ("mixed", "500-1k"): 12, ("mixed", "1k-2k"): 18,
    ("mixed", "2k-4k"): 14, ("mixed", ">4k"): 8,
    ("zh", "<500"): 4, ("zh", "500-1k"): 5, ("zh", "1k-2k"): 6,
    ("zh", "2k-4k"): 3, ("zh", ">4k"): 2,
    # en 按库存量给满（真实英文记忆本来就少，垃圾标记已滤）
    ("en", "<500"): 0, ("en", "500-1k"): 2, ("en", "1k-2k"): 4,
    ("en", "2k-4k"): 2, ("en", ">4k"): 1,
}
DISTRACTOR_QUOTAS = {  # 近似生产联合分布（非 twin active）
    ("mixed", "<500"): 28, ("mixed", "500-1k"): 40, ("mixed", "1k-2k"): 56,
    ("mixed", "2k-4k"): 30, ("mixed", ">4k"): 20,
    ("zh", "<500"): 10, ("zh", "500-1k"): 9, ("zh", "1k-2k"): 9,
    ("zh", "2k-4k"): 3, ("zh", ">4k"): 2,
    ("en", "<500"): 3, ("en", "500-1k"): 1, ("en", "1k-2k"): 4,
    ("en", "2k-4k"): 2, ("en", ">4k"): 1,
}
LEN_BUCKETS = ("<500", "500-1k", "1k-2k", "2k-4k", ">4k")
MAX_TARGETS_PER_WS = 30


def lang_class(text: str) -> str:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cjk + lat
    if total == 0:
        return "other"
    r = cjk / total
    return "zh" if r >= 0.7 else ("en" if r <= 0.15 else "mixed")


def len_bucket(n: int) -> str:
    if n < 500: return "<500"
    if n < 1000: return "500-1k"
    if n < 2000: return "1k-2k"
    if n < 4000: return "2k-4k"
    return ">4k"


def open_ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def envelope(row: sqlite3.Row) -> dict:
    tags = json.loads(row["tags"]) if row["tags"] else []
    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    return {
        "content": row["content"],
        "subject": row["subject"],
        "tags": tags,
        "workspace": row["workspace_canonical"],
        "workspace_canonical": row["workspace_canonical"],
        "event_time": row["event_time"],
        "source_type": row["source_type"] if row["source_type"] in
        {"agent_generated", "document_extracted", "pending", "unknown", "user_confirmed"}
        else "unknown",
        "metadata": metadata,
    }


def sample_stratum(pool: list[dict], quota: int, taken_subjects: set, rng: random.Random) -> list[dict]:
    """确定性：池按 id 排序后步长采样；同 subject 去重。"""
    if quota <= 0 or not pool:
        return []
    pool = sorted(pool, key=lambda r: r["id"])
    stride = max(1, len(pool) // (quota * 2))
    picked: list[dict] = []
    for start in range(min(quota, len(pool))):
        window = pool[start * stride:(start + 1) * stride]
        for cand in window:
            subj = cand["subject"].strip().lower()
            if subj in taken_subjects:
                continue
            picked.append(cand)
            taken_subjects.add(subj)
            break
        if len(picked) >= quota:
            break
    rng.shuffle(picked) if False else None  # 保持 id 序，纯确定性
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--distractors", type=int, default=204)
    args = ap.parse_args()

    con = open_ro(args.source)
    rows = con.execute(
        "select id, subject, content, tags, workspace_canonical, event_time, "
        "source_type, metadata from memories where status='active'"
    ).fetchall()
    pool = []
    for r in rows:
        if r["workspace_canonical"] in EXCLUDE_WS:
            continue
        d = dict(r)
        d["lang"] = lang_class(d["content"])
        d["bucket"] = len_bucket(len(d["content"]))
        pool.append(d)
    print(f"pool: {len(pool)} (excluded twin buckets)")

    # 纯英文目标垃圾滤除（self-test 标记等）——distractor 保留（它们本来就是噪音角色）
    def is_en_junk(d: dict) -> bool:
        return d["lang"] == "en" and any(m in d["content"] for m in EN_JUNK_MARKERS)

    rng = random.Random(170326)  # v3-len 固定种子，确定性重放
    taken_subjects: set = set()
    targets: list[dict] = []
    per_ws: Counter = Counter()
    for key in sorted(TARGET_QUOTAS, key=lambda k: (k[0], LEN_BUCKETS.index(k[1]))):
        lang, bucket = key
        cand = [d for d in pool if d["lang"] == lang and d["bucket"] == bucket
                and not is_en_junk(d)]
        got = sample_stratum(cand, TARGET_QUOTAS[key], taken_subjects, rng)
        for d in got:
            if per_ws[d["workspace_canonical"]] >= MAX_TARGETS_PER_WS:
                continue
            targets.append(d)
            per_ws[d["workspace_canonical"]] += 1
    taken_shas = {d["id"] for d in targets}

    distractors: list[dict] = []
    d_taken = set(taken_subjects)
    for key in sorted(DISTRACTOR_QUOTAS, key=lambda k: (k[0], LEN_BUCKETS.index(k[1]))):
        lang, bucket = key
        cand = [d for d in pool if d["lang"] == lang and d["bucket"] == bucket
                and d["id"] not in taken_shas]
        got = sample_stratum(cand, DISTRACTOR_QUOTAS[key], d_taken, rng)
        distractors.extend(got)

    def fixture_row(d: dict) -> dict:
        import hashlib
        sha = hashlib.sha256(d["content"].encode("utf-8")).hexdigest()
        row = envelope(d)  # d 即 DB 原始行：tags/metadata 本就是 JSON 字符串字段
        row["fixture_key"] = f"t-{sha[:12]}"
        row["source_id"] = d["id"]
        row["content_sha"] = sha
        row["lang"] = d["lang"]
        row["len_bucket"] = d["bucket"]
        return row

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t_rows = [fixture_row(d) for d in targets]
    d_rows = [fixture_row(d) for d in distractors]
    with (out / "targets.jsonl").open("w", encoding="utf-8") as f:
        for r in t_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "distractors.jsonl").open("w", encoding="utf-8") as f:
        for r in d_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    digest = []
    for i, r in enumerate(t_rows):
        digest.append(
            f"## L{i+1:02d} [{r['fixture_key']}] lang={r['lang']} len={r['len_bucket']} "
            f"ws={r['workspace_canonical']}\n"
            f"subject: {r['subject']}\ntags: {', '.join(r['tags'])}\n"
            f"content_head: {r['content'][:600]}\n"
        )
    (out / "_digest.md").write_text("\n".join(digest), encoding="utf-8")

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(args.source),
        "targets": len(t_rows),
        "distractors": len(d_rows),
        "lang_mix_targets": dict(Counter(r["lang"] for r in t_rows)),
        "len_mix_targets": dict(Counter(r["len_bucket"] for r in t_rows)),
        "design": "owner 2026-09-26：长度分层（中长/很长加权，生产 ≥1k 占 52.9%）"
                  "× 语言分桶（mixed/zh/en 按生产分布 80/15/4.5）；queries/labels 人工出题后组装",
        "lang_def": "CJK/ASCII 字母比：zh≥0.7，en≤0.15，否则 mixed",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"targets={len(t_rows)} distractors={len(d_rows)} -> {out}")
    print(f"lang mix: {manifest['lang_mix_targets']} len mix: {manifest['len_mix_targets']}")
    print(f"digest -> {out / '_digest.md'}（人工出题用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
