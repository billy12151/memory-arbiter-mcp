#!/usr/bin/env python3
"""冲突候选向量层探针（无 Qwen）：真冲突对在候选生成阶段的窗口覆盖率与带门阈值扫描。

漏斗语义对齐：left 记忆的每个句子行在池内 KNN top-16，并集为候选记忆集合；
right ∈ 候选集 = 窗口命中（真实测 92% 覆盖口径的同款近似，池=语料 174 记忆）。
对每个配置输出：真对窗口命中率、真对最佳句对余弦分布、各 floor 档的
真对带内通过数（召回上限）与非真对带内通过数（Qwen 成本压力）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from llama_cpp import Llama

sys.path.insert(0, str(Path(".").resolve()))
from memory_arbiter.rowseg import segment_rows

CONFIGS = [
    ("gemma-none", "/Users/zhangzhiwei17/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf", ""),
    ("bge-none", "/Users/zhangzhiwei17/models/bge-m3/bge-m3-Q8_0.gguf", ""),
]
K_WINDOW = 16
FLOORS = (0.50, 0.55, 0.60, 0.65, 0.70)
CEIL = 0.98

lines = []
for name in ("pairs.jsonl", "pairs_large.jsonl", "pairs_noisy.jsonl"):
    lines += open(f"eval/fixtures/conflict/{name}").read().splitlines()
pairs = [json.loads(l) for l in lines if l.strip() and json.loads(l).get("label")]

out = {}
for tag, model, prefix in CONFIGS:
    llm = Llama(model_path=model, embedding=True, pooling_type=-1,
                n_ctx=2048, n_gpu_layers=-1, verbose=False)
    cache: dict[str, list[float]] = {}

    def emb(text: str) -> list[float]:
        if text not in cache:
            full = (prefix + "\n" + text) if prefix else text
            cache[text] = list(llm.create_embedding(full)["data"][0]["embedding"])
        return cache[text]

    # 记忆级句子行池（numpy 矩阵）
    mems: dict[str, np.ndarray] = {}
    mem_texts: dict[str, list[str]] = {}
    for p in pairs:
        for tag_side, side in (("L", p["left"]), ("R", p["right"])):
            key = f'{p["pair_id"]}:{tag_side}'
            if key in mems:
                continue
            rows = [s.text for s in segment_rows("", side["content"]) if s.kind in ("sentence", "table_row")]
            rows = rows or [side["content"]]
            mem_texts[key] = rows
            mems[key] = np.array([emb(t) for t in rows], dtype=np.float32)
    all_keys = list(mems)
    pool = np.vstack([mems[k] for k in all_keys])
    pool_norm = pool / np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
    key_of_row = [k for k in all_keys for _ in range(len(mems[k]))]

    def norm_rows(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    true_hit = n_true = n_nontrue = 0
    nontrue_hit = 0
    true_best: list[float] = []
    band_true = {f: 0 for f in FLOORS}
    band_nontrue = {f: 0 for f in FLOORS}
    for p in pairs:
        left_key, right_key = f'{p["pair_id"]}:L', f'{p["pair_id"]}:R'
        Q = norm_rows(mems[left_key])
        sim = Q @ pool_norm.T  # (left_rows, pool_rows)
        # 排除自身记忆的行
        own_idx = [i for i, k in enumerate(key_of_row) if k == left_key]
        sim[:, own_idx] = -1.0
        # 逐行 top-16 → 候选记忆并集
        cand_keys: set[str] = set()
        for row in sim:
            top = np.argsort(-row)[:K_WINDOW]
            cand_keys.update(key_of_row[i] for i in top)
        hit = right_key in cand_keys
        # right 最佳句对余弦（任意 left 行 × right 行的最大值）
        best = float(sim.max())
        if p["label"] == "true_conflict":
            n_true += 1
            true_hit += hit
            true_best.append(best)
            if best < CEIL:
                for f in FLOORS:
                    if best >= f:
                        band_true[f] += 1
        else:
            n_nontrue += 1
            nontrue_hit += hit
            if best < CEIL:
                for f in FLOORS:
                    if best >= f:
                        band_nontrue[f] += 1
    ts = sorted(true_best)
    out[tag] = {
        "n_true": n_true, "n_nontrue": n_nontrue,
        "true_window16_hit": f"{true_hit}/{n_true}",
        "nontrue_window16": f"{nontrue_hit}/{n_nontrue}",
        "true_best_cos_min": round(ts[0], 4),
        "true_best_cos_p5": round(ts[max(0, int(len(ts) * 0.05))], 4),
        "true_band_pass": {str(f): band_true[f] for f in FLOORS},
        "nontrue_band_pass": {str(f): band_nontrue[f] for f in FLOORS},
    }
    print(tag, json.dumps(out[tag], ensure_ascii=False), flush=True)
    llm.close()

json.dump(out, open("eval/calibration/conflict-vector-stage.json", "w"), ensure_ascii=False, indent=1)
print("saved -> eval/calibration/conflict-vector-stage.json")
