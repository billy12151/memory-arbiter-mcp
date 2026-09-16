#!/usr/bin/env python3
"""复现/验证英文证据对的写时冲突检测（另一个 agent 报告：0.5B+中文 prompt
处理英文证据不确认对立，job 完成零 notice）。real file —— 子进程 spawn 需要。"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.runner import default_embed_model, default_qwen_model, temp_library  # noqa: E402
from memory_arbiter import semantic_conflict as sc  # noqa: E402

CALLS: list[str] = []

_orig = sc.IsolatedGGUFSemanticBackend.classify_pair


def _logged(self, left, right, **kw):  # noqa: ANN001, ANN202
    sig = _orig(self, left, right, **kw)
    CALLS.append(repr(sig)[:300])
    return sig


sc.IsolatedGGUFSemanticBackend.classify_pair = _logged

BASE = {"tags": ["eval-en"], "workspace": "eval-en",
        "event_time": "2026-09-16T00:00:00+00:00", "source_type": "agent_generated",
        "metadata": {}}
PAIRS = [
    ("The production database uses MySQL with a dual-primary setup.",
     "The production database uses PostgreSQL with a single primary.",
     "Production database engine selection"),
    ("Refund requests over 5000 CNY require finance review.",
     "Refund requests over 500 CNY require finance review.",
     "Refund review threshold"),
]


def main() -> None:
    with temp_library(default_embed_model(), qwen_model=default_qwen_model()) as tools:
        for idx, (left, right, subject) in enumerate(PAIRS, 1):
            for content in (left, right):
                data = dict(BASE)
                data["content"] = content
                data["subject"] = subject
                data["metadata"] = {"entity": f"en-pair-{idx}", "scope": f"en-pair-{idx}-scope"}
                data["agent_id"] = "mema-eval-harness"
                result = tools.memory("remember", data)
                check = ((result.get("data") or {}).get("semantic_conflict_check")) or {}
                tid = check.get("task_id")
                if tid:
                    tools._semantic_worker.wait_task(str(tid), timeout=180.0)
                if content is right:
                    summary = {k: v for k, v in check.items() if k != "task_id"}
                    print(f"[pair {idx}] {json.dumps(summary, ensure_ascii=False)}")
        tools.wait_evidence_worker_drained(timeout=60.0)
        conn = sqlite3.connect(tools.settings.db_path)
        rows = conn.execute("SELECT id, status, conflict_point FROM conflicts").fetchall()
        print("conflicts rows:", rows)
        conn.close()
    print("\n--- Qwen raw ---")
    for call in CALLS:
        print(call)


if __name__ == "__main__":
    main()
