#!/usr/bin/env python3
"""诊断「候选/Qwen 层未成对」桶（cf-oppo-04/05/10/12）.

插桩三站：evidence_knn 召回（是否捞到对端）、SemanticBackend.classify_pair
（Qwen 是否被调用、判决为何）、conflicts 表（notice 是否落库），定位断点。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.runner import (  # noqa: E402
    _remember_envelope,
    default_embed_model,
    default_qwen_model,
    temp_library,
)
from memory_arbiter import semantic_conflict as sc  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402

PAIR_IDS = [f"cf-oppo-{i:02d}" for i in range(1, 13)]

# DIAG_FORCE_EN=1：强制英文 prompt 跑中文证据（验证「英文 prompt 对中文输入
# 持平或更好则只维护一套英文」的假设）
import os  # noqa: E402

if os.environ.get("DIAG_FORCE_EN") == "1":
    sc.evidence_is_cjk = lambda left, right: False  # noqa: ARG005

knn_log: list[dict] = []
qwen_log: list[dict] = []

_orig_knn = MemoryDB.evidence_knn


def _logged_knn(self, embedding, **kw):  # noqa: ANN001, ANN202
    hits = _orig_knn(self, embedding, **kw)
    knn_log.append({
        "exclude": kw.get("exclude_memory_id"),
        "hits": [
            (int(h["memory_id"]), round(float(h.get("distance") or 0), 4),
             str(h.get("text") or "")[:50])
            for h in hits
        ],
    })
    return hits


MemoryDB.evidence_knn = _logged_knn

_orig_cp = sc.IsolatedGGUFSemanticBackend.classify_pair


def _logged_cp(self, left, right, **kw):  # noqa: ANN001, ANN202
    sig = _orig_cp(self, left, right, **kw)
    qwen_log.append({
        "pair": (int(left.get("memory_id") or 0), int(right.get("memory_id") or 0)),
        "left_quote": str(left.get("quote") or "")[:60],
        "right_quote": str(right.get("quote") or "")[:60],
        "signal": repr(sig),
        "signal_obj": sig,
    })
    return sig


sc.IsolatedGGUFSemanticBackend.classify_pair = _logged_cp


def main() -> None:
    embed, qwen = default_embed_model(), default_qwen_model()
    print(f"embed={embed}\nqwen={qwen}")
    pairs = [
        json.loads(line)
        for line in (REPO / "eval/fixtures/conflict/pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = [p for p in pairs if p["pair_id"] in PAIR_IDS]
    id_map: dict[str, tuple[int, int]] = {}
    with temp_library(embed, qwen_model=qwen) as tools:
        for idx, p in enumerate(pairs, 1):
            entity = f"eval-pair-{idx:03d}"
            lid, _, _ = _remember_envelope(tools, p["left"], entity)
            rid, _, rres = _remember_envelope(tools, p["right"], entity)
            id_map[p["pair_id"]] = (int(lid), int(rid))
            check = ((rres.get("data") or {}).get("semantic_conflict_check")) or {}
            tid = str(check.get("task_id") or "")
            if tid:
                tools._semantic_worker.wait_task(tid, timeout=180.0)
            summary = {k: v for k, v in check.items() if k != "task_id"}
            print(f"[{p['pair_id']}] left={lid} right={rid} check={json.dumps(summary, ensure_ascii=False)}")
        tools.wait_evidence_worker_drained(timeout=60.0)
        conn = sqlite3.connect(tools.settings.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, status, notice_type, conflict_point, member_versions FROM conflicts ORDER BY id"
        ).fetchall()
        for r in rows:
            print("[conflicts]", json.dumps(dict(r), ensure_ascii=False)[:300])
        conn.close()

    for pid, (lid, rid) in id_map.items():
        print(f"\n===== {pid} left={lid} right={rid}")
        rel = [e for e in knn_log if e["exclude"] == rid]
        got = [h for e in rel for h in e["hits"] if h[0] == lid]
        print(f"  KNN召回: {len(rel)} 次查询, 捞到 left 次数={len(got)}")
        for i, e in enumerate(rel):
            print(f"   query{i} hits={e['hits']}")
        calls = [q for q in qwen_log if {q["pair"][0], q["pair"][1]} == {lid, rid}]
        print(f"  Qwen调用: {len(calls)} 次")
        for q in calls:
            print(f"    {q['pair']} L={q['left_quote']!r} R={q['right_quote']!r}")
            print(f"      -> {q['signal']}")
        # 复算 gate 结局（调用按 forward/reverse 成对出现）
        from memory_arbiter.semantic_conflict import (
            evaluate_pair_extractions, signal_extraction,
        )
        for k in range(0, len(calls) - 1, 2):
            fwd, rev = calls[k], calls[k + 1]
            gate = evaluate_pair_extractions(
                signal_extraction(fwd["signal_obj"]),
                signal_extraction(rev["signal_obj"]),
                {"quote": fwd["left_quote"]}, {"quote": fwd["right_quote"]},
                require_bidirectional=True,
            )
            print(f"    GATE[{k // 2}]: state={gate.state} reason={gate.reason}")


if __name__ == "__main__":
    main()
