#!/usr/bin/env python3
"""runner — P0-1 评测执行器（方案 §2.2/§2.6，c2 起步）.

一条命令完成：创建临时库 → 重放 fixtures → 执行套件 → 采集原始结果 → 销毁临时库。
本文件只负责「执行与采集」；指标计算在 scorers.py（c5）、对比门在 gate.py（c5）。

链路（D2 已拍板）：进程内 MemoryTools——即 MCP 工具的实现层，被测指标（排序/
分数/写时提示/notice 语义）在此层已定型；不走 HTTP，不继承用户 ~/.config
（§2.6 防配置漂移：Settings 直构，唯一外部输入是模型路径与语料文件）。

c2 覆盖：--suite recall（34 query 召回采集 + 98 target 自召回）。
c3/c4 将扩展 similarity / conflict 套件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter import __version__ as MEMA_VERSION  # noqa: E402
from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.tools import MemoryTools  # noqa: E402

FIXTURES = REPO / "eval" / "fixtures"
EVAL_CLIENT = "mema-eval"
EVAL_AGENT = "mema-eval-harness"
# 当年真库存在已废弃的 source_type 枚举值；重放时统一落到 unknown
# （source_type 不参与召回排序，不在被测范围）
VALID_SOURCE_TYPES = {
    "agent_generated", "document_extracted", "pending", "unknown", "user_confirmed",
}


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def default_embed_model() -> Path | None:
    """Convenience default: the local install's configured embedder path.

    Reads exactly one value from the user config (the nested embedding.model_path)
    — never Settings.from_env() — so no other user knob can leak into the harness.
    """
    import json as _json

    cfg = Path.home() / ".config/memory-arbiter/config.json"
    if not cfg.exists():
        return None
    try:
        raw = _json.loads(cfg.read_text(encoding="utf-8"))
        value = (raw.get("embedding") or {}).get("model_path")
        return Path(value).expanduser() if value else None
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_settings(workdir: Path, embed_model: Path | None) -> Settings:
    return Settings(
        db_path=workdir / "harness.sqlite3",
        backup_jsonl=workdir / "backup.jsonl",
        client=EVAL_CLIENT,
        agent_id=EVAL_AGENT,
        workspace="default",
        isolation="none",  # 当年标定口径（eval_relevance_floor.py 同款默认）
        embedding_model_path=embed_model,
        # recall 套件不起 Qwen；conflict 套件（c4）另行构建开启
        semantic_conflict_enabled=False,
    )


@contextmanager
def temp_library(embed_model: Path | None, keep_db: Path | None = None) -> Iterator[MemoryTools]:
    """临时库生命周期：建库 → 起 workers → yield → drain + shutdown → 销毁."""
    import tempfile

    workdir = Path(tempfile.mkdtemp(prefix="mema-eval-"))
    settings = build_settings(workdir, embed_model)
    tools = MemoryTools(settings, MemoryDB(settings))
    tools.start_evidence_worker()
    try:
        yield tools
    finally:
        try:
            tools.wait_evidence_worker_drained(timeout=60.0)
        finally:
            tools.shutdown()
            if keep_db is not None:
                keep_db.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(workdir, keep_db, dirs_exist_ok=True)
            shutil.rmtree(workdir, ignore_errors=True)


def _remember(tools: MemoryTools, envelope: dict) -> tuple[int | None, bool]:
    """写入一条 fixture，返回 (新库 id, 是否 duplicate_replay 幂等返回)."""
    data: dict[str, Any] = {
        "content": envelope["content"],
        "subject": envelope["subject"],
        "tags": envelope.get("tags") or [],
        "workspace": envelope.get("workspace") or "default",
        "event_time": envelope.get("event_time"),
        "source_type": envelope.get("source_type") if envelope.get("source_type") in VALID_SOURCE_TYPES else "unknown",
        "metadata": envelope.get("metadata") or {},
        "agent_id": EVAL_AGENT,
    }
    result = tools.memory("remember", data)
    payload = result.get("data") or {}
    # 0.16.6 防重门：同内容重放幂等返回 ok=True + duplicate_replay（无 record）
    if payload.get("duplicate_replay"):
        replay_of = payload.get("replay_of") or {}
        return int(replay_of["memory_id"]), True
    if result.get("ok"):
        record = payload.get("record") or {}
        return int(record["id"]), False
    raise RuntimeError(f"fixture replay failed: {json.dumps(payload, ensure_ascii=False)[:400]}")


def replay_fixtures(tools: MemoryTools, envelopes: list[dict], progress_every: int = 50) -> dict[str, int]:
    """重放全部 fixtures；fixture_key（content_sha 寻址）→ 新库 id."""
    id_map: dict[str, int] = {}
    sha_to_key = {row["content_sha"]: row["fixture_key"] for row in envelopes}
    duplicates = 0
    started = time.perf_counter()
    for index, envelope in enumerate(envelopes, 1):
        new_id, replayed = _remember(tools, envelope)
        id_map[envelope["fixture_key"]] = new_id
        duplicates += int(replayed)
        if index % progress_every == 0:
            print(f"[replay] {index}/{len(envelopes)} ({time.perf_counter() - started:.0f}s)")
    # 同内容不同 fixture_key 的重放落同一 id：把 sha 级映射补齐
    for envelope in envelopes:
        key = sha_to_key.get(envelope["content_sha"])
        if key:
            id_map.setdefault(key, id_map[envelope["fixture_key"]])
    if duplicates:
        print(f"[replay] {duplicates} duplicate_replay (identical content, idempotent)")
    tools.wait_evidence_worker_drained(timeout=120.0)
    return id_map


def _run_find(tools: MemoryTools, query: str, limit: int = 10) -> dict[str, Any]:
    started = time.perf_counter()
    result = tools.memory("find", {"query": query, "limit": limit})
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    payload = result.get("data") or {}
    return {"elapsed_ms": elapsed_ms, "payload": payload, "ok": bool(result.get("ok"))}


def run_recall_queries(
    tools: MemoryTools, queries: list[dict], id_map: dict[str, int],
) -> list[dict[str, Any]]:
    """34 条固定 query：采集排名/分数/耗时（judge 留给 scorers）."""
    reverse_map = {new_id: key for key, new_id in id_map.items()}
    collected: list[dict[str, Any]] = []
    for query in queries:
        outcome = _run_find(tools, query["query"])
        payload = outcome["payload"]
        hits: list[dict[str, Any]] = []
        for rank, row in enumerate(payload.get("results") or [], 1):
            mid = int(row.get("id") or 0)
            hits.append({
                "rank": rank,
                "fixture_key": reverse_map.get(mid, f"id:{mid}"),
                "subject": row.get("subject"),
                "score": row.get("score") if row.get("score") is not None else row.get("final_score"),
                "workspace": row.get("workspace"),
            })
        collected.append({
            "qid": query["qid"],
            "kind": query["kind"],
            "query": query["query"],
            "ok": outcome["ok"],
            "elapsed_ms": outcome["elapsed_ms"],
            "retrieval_mode": payload.get("mode") or payload.get("retrieval_mode"),
            "hits": hits,
        })
    return collected


def run_self_recall(
    tools: MemoryTools, targets: list[dict], id_map: dict[str, int],
) -> list[dict[str, Any]]:
    """自召回（run_eval.py 模式移植）：query=subject，期望自己进 top-10."""
    reverse_map = {new_id: key for key, new_id in id_map.items()}
    collected: list[dict[str, Any]] = []
    for target in targets:
        subject = str(target["subject"] or "")
        if not subject:
            continue
        payload = _run_find(tools, subject)["payload"]
        rank_hit: int | None = None
        for rank, row in enumerate(payload.get("results") or [], 1):
            if int(row.get("id") or 0) == id_map[target["fixture_key"]]:
                rank_hit = rank
                break
        collected.append({
            "fixture_key": target["fixture_key"],
            "subject": subject,
            "self_rank": rank_hit,
            "in_top10": rank_hit is not None,
        })
    return collected


def run_similarity_suite(tools: MemoryTools, cases: list[dict]) -> dict[str, Any]:
    """近似提示套件（方案 §2.5）：按组写锚→逐变体写入，采集 similar_active_memory.

    判定口径：变体写响应 notices 中 type=similar_active_memory 视为 fired；
    matches 命中本组锚 id=hit_anchor（正确触发）；仅命中他组=hit_other
    （跨组噪音，单独计数，不算误报也不算正确）。opposite_semantics 类
    按机制先验可能触发（subject 一词之差 ratio≈0.96 + tags 相同）——
    该类触发即「被误当重复」的事实计量，不是套件缺陷。
    """
    grouped: dict[str, list[dict]] = {}
    for row in cases:
        grouped.setdefault(row["group"], []).append(row)
    results: list[dict[str, Any]] = []
    anchor_ids: dict[str, int] = {}
    for group, group_cases in grouped.items():
        anchor = group_cases[0]["anchor"]
        anchor_result = tools.memory("remember", {
            "content": anchor["content"], "subject": anchor["subject"],
            "tags": anchor["tags"], "workspace": "eval-sim",
            "source_type": "agent_generated", "agent_id": EVAL_AGENT,
        })
        record = (anchor_result.get("data") or {}).get("record") or {}
        anchor_ids[group] = int(record["id"])
        for case in group_cases:
            variant = case["variant"]
            write = tools.memory("remember", {
                "content": variant["content"], "subject": variant["subject"],
                "tags": variant["tags"], "workspace": "eval-sim",
                "source_type": "agent_generated", "agent_id": EVAL_AGENT,
            })
            notices = [
                notice for notice in (write.get("notices") or [])
                if notice.get("type") == "similar_active_memory"
            ]
            matches = [m for notice in notices for m in (notice.get("matches") or [])]
            anchor_id = anchor_ids[group]
            hit_anchor = any(int(m.get("memory_id") or 0) == anchor_id for m in matches)
            results.append({
                "case_id": case["case_id"],
                "label": case["label"],
                "theme": case["theme"],
                "fired": bool(notices),
                "hit_anchor": hit_anchor,
                "hit_other_only": bool(matches) and not hit_anchor,
                "match_ids": [int(m.get("memory_id") or 0) for m in matches],
                "subject_similarity": [m.get("subject_similarity") for m in matches],
                "tag_jaccard": [m.get("tag_jaccard") for m in matches],
            })
    return {"anchor_ids": anchor_ids, "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="recall", choices=["recall", "similarity", "all"],
                        help="c2=recall，c3=similarity；conflict 套件 c4 扩展")
    parser.add_argument("--embed-model", type=Path, default=None, help="embedder GGUF 路径")
    parser.add_argument("--out", type=Path, default=REPO / "eval" / "results")
    parser.add_argument("--label", default="run", help="产物文件名标签")
    parser.add_argument("--keep-db", type=Path, default=None, help="调试：保留临时库副本到该目录")
    args = parser.parse_args()

    embed_model = args.embed_model or default_embed_model()
    if embed_model is None or not Path(embed_model).exists():
        print("error: embedder model required (--embed-model 或本机 config 配置)", file=sys.stderr)
        return 2

    recall_dir = FIXTURES / "recall"
    queries = json.loads((recall_dir / "queries.json").read_text(encoding="utf-8"))["queries"]
    targets = _load_jsonl(recall_dir / "targets.jsonl")
    distractors = _load_jsonl(recall_dir / "distractors.jsonl")
    manifest = json.loads((recall_dir / "manifest.json").read_text(encoding="utf-8"))

    print(f"[harness] mema={MEMA_VERSION} corpus={manifest['corpus_version']} "
          f"embedder={Path(embed_model).name} suite={args.suite}")
    want_recall = args.suite in {"recall", "all"}
    want_similarity = args.suite in {"similarity", "all"}
    with temp_library(embed_model, keep_db=args.keep_db) as tools:
        recall: list[dict[str, Any]] | None = None
        self_recall: list[dict[str, Any]] | None = None
        if want_recall:
            id_map = replay_fixtures(tools, targets + distractors)
            print(f"[replay] library size={len(set(id_map.values()))}")
            recall = run_recall_queries(tools, queries, id_map)
            self_recall = run_self_recall(tools, targets, id_map)
        similarity: dict[str, Any] | None = None
        if want_similarity:
            sim_cases = _load_jsonl(FIXTURES / "similarity" / "cases.jsonl")
            similarity = run_similarity_suite(tools, sim_cases)

    raw = {
        "suite": args.suite,
        "mema_version": MEMA_VERSION,
        "corpus_version": manifest["corpus_version"],
        "env": {
            "embed_model": str(embed_model),
            "embed_model_sha256": _sha256(Path(embed_model)),
            "isolation": "none",
            "targets": len(targets),
            "distractors": len(distractors),
            "semantic_conflict_enabled": False,
        },
        "queries": recall,
        "self_recall": self_recall,
        "similarity": similarity,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.suite}-{args.label}.json"
    out_path.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    if self_recall:
        top10_self = sum(1 for row in self_recall if row["in_top10"])
        print(f"[recall] self_recall_top10={top10_self}/{len(self_recall)}")
    if similarity:
        fired = sum(1 for row in similarity["cases"] if row["fired"])
        hit_anchor = sum(1 for row in similarity["cases"] if row["hit_anchor"])
        print(f"[similarity] fired={fired}/48 hit_anchor={hit_anchor}/48")
    print(f"[done] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
