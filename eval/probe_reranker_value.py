#!/usr/bin/env python3
"""bge-reranker-v2-m3 价值评估（owner 2026-09-25 指挥，纯向量层、无 Qwen）。

①检索重排：各配置自嵌 298 文档 → query 向量 top-50 候选 → reranker 重排 →
  R@5/R@10 对比（vector-only 口径，隔离 reranker 净效果；与产品 R@10 的差异
  在于无词法融合/exact boost，仅作配置间相对比较）。
②冲突通道可用性：87 对语料 reranker 分数的真/非真分离度（双向取 max；重点
  write_opposition 值对立形态——冒烟显示值差异矛盾对 0.10，若语料复现则
  reranker 不能进冲突候选链）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import onnxruntime as ort
from llama_cpp import Llama
from tokenizers import Tokenizer

REPO = Path(".")
RDIR = "/Users/zhangzhiwei17/models/bge-reranker-v2-m3"
sess = ort.InferenceSession(RDIR + "/onnx/model_int8.onnx", providers=["CPUExecutionProvider"])
tok = Tokenizer.from_file(RDIR + "/tokenizer.json")
tok.enable_truncation(max_length=512)

CONFIGS = [
    ("gemma-none", "/Users/zhangzhiwei17/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf", ""),
    ("bge-none", "/Users/zhangzhiwei17/models/bge-m3/bge-m3-Q8_0.gguf", ""),
]


def rscore_batch(pairs: list[tuple[str, str]], batch: int = 16) -> list[float]:
    out: list[float] = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i : i + batch]
        encs = [tok.encode(q, d) for q, d in chunk]
        maxlen = max(len(e.ids) for e in encs)
        ids = [e.ids + [1] * (maxlen - len(e.ids)) for e in encs]
        mask = [[1] * len(e.ids) + [0] * (maxlen - len(e.ids)) for e in encs]
        logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        out += [float(x) for x in logits[:, 0]]
    return out


targets = {t["fixture_key"]: t for t in (json.loads(l) for l in open("eval/fixtures/recall/targets.jsonl"))}
distractors = {d["fixture_key"]: d for d in (json.loads(l) for l in open("eval/fixtures/recall/distractors.jsonl"))}
docs = {**{k: f"{t['subject']}\n{t['content']}" for k, t in targets.items()},
        **{k: f"{d['subject']}\n{d['content']}" for k, d in distractors.items()}}
labels = [json.loads(l) for l in open("eval/fixtures/recall/labels.jsonl")]
rel: dict[str, set[str]] = {}
for r in labels:
    if r["label"] == "relevant":
        rel.setdefault(r["qid"], set()).add(r["fixture_key"])
queries = json.load(open("eval/fixtures/recall/queries.json"))["queries"]


def recall_at(rankings: dict[str, list[str]], k: int) -> tuple[float, int]:
    hits = 0
    for qid, keys in rankings.items():
        hits += len(rel.get(qid, set()) & set(keys[:k]))
    total = sum(len(v) for v in rel.values())
    return round(hits / total, 4), hits


for tag, model, prefix in CONFIGS:
    llm = Llama(model_path=model, embedding=True, pooling_type=-1,
                n_ctx=2048, n_gpu_layers=-1, verbose=False)
    cache: dict[str, list[float]] = {}

    def emb(text: str) -> list[float]:
        if text not in cache:
            full = (prefix + "\n" + text) if prefix else text
            cache[text] = list(llm.create_embedding(full)["data"][0]["embedding"])
        return cache[text]

    def cos(a, b):
        d = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return d / (na * nb) if na and nb else 0.0

    doc_vecs = {k: emb(t) for k, t in docs.items()}
    base_rank: dict[str, list[str]] = {}
    rerank_pairs: list[tuple[str, str]] = []
    index: list[tuple[str, str]] = []
    for q in queries:
        qv = emb(q["query"])
        top50 = sorted(doc_vecs, key=lambda k: -cos(qv, doc_vecs[k]))[:50]
        base_rank[q["qid"]] = top50
        for fk in top50:
            rerank_pairs.append((q["query"], docs[fk]))
            index.append((q["qid"], fk))
    b5, bh5 = recall_at(base_rank, 5)
    b10, bh10 = recall_at(base_rank, 10)
    print(f"[{tag}] rerank-scoring {len(rerank_pairs)} pairs ...", flush=True)
    sc = rscore_batch(rerank_pairs)
    scored: dict[str, list[tuple[str, float]]] = {q["qid"]: [] for q in queries}
    for (qid, fk), s in zip(index, sc):
        scored[qid].append((fk, s))
    rerank_rank = {qid: [fk for fk, _s in sorted(v, key=lambda x: -x[1])] for qid, v in scored.items()}
    r5, rh5 = recall_at(rerank_rank, 5)
    r10, rh10 = recall_at(rerank_rank, 10)
    print(f"[{tag}] vector-R@5 {b5}({bh5}) -> rerank {r5}({rh5}) | vector-R@10 {b10}({bh10}) -> rerank {r10}({rh10})")
    llm.close()

# 冲突通道：87 对真/非真分离度
lines = []
for name in ("pairs.jsonl", "pairs_large.jsonl", "pairs_noisy.jsonl"):
    lines += open(f"eval/fixtures/conflict/{name}").read().splitlines()
pairs = [json.loads(l) for l in lines if l.strip() and json.loads(l).get("label")]
conf: dict[str, list[float]] = {"true": [], "nontrue": []}
opp: list[float] = []
for p in pairs:
    s1 = rscore_batch([(p["left"]["content"], p["right"]["content"])])[0]
    s2 = rscore_batch([(p["right"]["content"], p["left"]["content"])])[0]
    v = max(s1, s2)
    conf["true" if p["label"] == "true_conflict" else "nontrue"].append(v)
    if p.get("shape") == "write_opposition":
        opp.append(v)
tmin, tmax = min(conf["true"]), max(conf["true"])
nmax = max(conf["nontrue"])
print("\n== conflict separation (reranker logit, 双向 max) ==")
print(f"true: min {tmin:+.3f} max {tmax:+.3f} | nontrue max {nmax:+.3f} | gap(真下限−非真上限) {tmin - nmax:+.3f}")
print(f"write_opposition 值对立 ({len(opp)} 对): min {min(opp):+.3f} max {max(opp):+.3f} mean {sum(opp)/len(opp):+.3f}")
print(f"sigmoid: true min {1/(1+math.exp(-tmin)):.4f} | nontrue max {1/(1+math.exp(-nmax)):.4f}")
