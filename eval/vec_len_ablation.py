"""Vector-only 3x2 prefix ablation on the recall-v3-len corpus (zh 中长文为主).

Same corpus texts (targets+distractors, product row segmentation), re-embedded
under three doc-side prefixes x two query-side prefixes, pure cosine ranking
over the whole corpus (no FTS/fusion/floor/pool). Isolates the embedding-space
effect on the owner-corpus-shaped data (mixed 80% / zh 15% / en 4.5%,
median ~1.2k chars).
"""
import glob, json, os, sys, time

REPO = "/Users/zhangzhiwei17/BillyProject/memory-arbiter-mcp"
sys.path.insert(0, REPO)

import numpy as np

CORPUS = f"{REPO}/eval/fixtures/recall-len"
targets = [json.loads(l) for l in open(f"{CORPUS}/targets.jsonl") if l.strip()]
distractors = [json.loads(l) for l in open(f"{CORPUS}/distractors.jsonl") if l.strip()]
queries = json.load(open(f"{CORPUS}/queries.json"))["queries"]
labels = [json.loads(l) for l in open(f"{CORPUS}/labels.jsonl") if l.strip()]

from memory_arbiter.rowseg import segment_rows
from memory_arbiter.embedder import build_embedder

gold = {}
for l in labels:
    gold.setdefault(l["qid"], set()).add(l["fixture_key"])

mem_rows = []   # (fixture_key, row_text)
for m in targets + distractors:
    segs = segment_rows(m["subject"], m["content"])
    for s in segs:
        mem_rows.append((m["fixture_key"], s.text))
row_keys = [k for k, _ in mem_rows]
row_texts = [t for _, t in mem_rows]
print(f"memories={len(targets)+len(distractors)} rows={len(mem_rows)} queries={len(queries)}", flush=True)

model = glob.glob("/Users/zhangzhiwei17/.node-llama-cpp/models/hf_ggml-org_embeddinggemma*.gguf")[0]
emb, _ = build_embedder(model_path=model)
assert emb is not None
DIM = emb.dim
QXQ = "task: search result | query: "
OFFICIAL = "title: none | text: "


def embed_all(texts, prefix, tag, chunk=256, *, embedder=None):
    emb2 = embedder or emb
    out = np.empty((len(texts), DIM), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), chunk):
        rs = emb2.embed_texts(texts[i:i + chunk], prefix=prefix)
        for j, r in enumerate(rs):
            assert r.embedding and len(r.embedding) == DIM
            out[i + j] = r.embedding
        done = min(i + chunk, len(texts))
        if done % 4096 < chunk or done == len(texts):
            print(f"  {tag}: {done}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n


CACHE = f"{CORPUS}/../vec_len_cache.npz"
cache = {}
if os.path.exists(CACHE):
    with np.load(CACHE) as z:
        cache = {k: z[k] for k in z.files}
    print(f"cache: {sorted(cache)}", flush=True)


def embed_cached(texts, prefix, tag):
    key = tag.replace("/", "_")
    if key in cache and cache[key].shape == (len(texts), DIM):
        return cache[key]
    v = embed_all(texts, prefix, tag)
    cache[key] = v
    np.savez_compressed(CACHE, **cache)
    return v


DOC = {"bare": "", "qxq": QXQ, "official": OFFICIAL}
QRY = {"bare": "", "qxq": QXQ}
doc_vecs = {n: embed_cached(row_texts, p, f"doc/{n}") for n, p in DOC.items()}
q_texts = [q["query"] for q in queries]
qry_vecs = {n: embed_cached(q_texts, p, f"qry/{n}") for n, p in QRY.items()}

results = {}
for dn in DOC:
    for qn in QRY:
        V = doc_vecs[dn]
        Q = qry_vecs[qn]
        sims = Q @ V.T                       # (nq, nrows)
        per_q = {}
        for qi, q in enumerate(queries):
            # 取该查询的本 top 排名（全库排序）
            order = np.argsort(-sims[qi])
            ranked = []
            seen = set()
            for idx in order:
                k = row_keys[idx]
                if k not in seen:
                    seen.add(k)
                    ranked.append(k)
                if len(ranked) >= 10:
                    break
            g = gold.get(q["qid"], set())
            per_q[q["qid"]] = {
                "lang": q.get("lang"), "ranked": ranked, "gold": g,
                "h5": len(g & set(ranked[:5])), "h10": len(g & set(ranked[:10])),
                "noise": sum(1 for k in ranked if k not in g),
            }
        n = len(per_q)
        r5 = sum(r["h5"] for r in per_q.values()) / n
        r10 = sum(r["h10"] for r in per_q.values()) / n
        noise = sum(r["noise"] for r in per_q.values()) / (n * 10)
        # 语言分桶（查询语言 -> target 语言；语料 target lang 从 targets 文件）
        tlang = {m["fixture_key"]: m["lang"] for m in targets}
        buckets = {}
        for r in per_q.values():
            for gk in r["gold"]:
                b = f"{r['lang']}->{tlang.get(gk, 'unknown')}"
                buckets.setdefault(b, [0, 0])
                buckets[b][0] += r["h5"]; buckets[b][1] += 1
        results[f"doc={dn} qry={qn}"] = {
            "R@5": round(r5, 4), "R@10": round(r10, 4), "noise@10": round(noise, 4),
            "lang": {b: round(h / n_, 3) for b, (h, n_) in sorted(buckets.items())},
        }
        print(f"doc={dn:<8} qry={qn:<5} R@5={r5:.3f} R@10={r10:.3f} noise@10={noise:.3f}", flush=True)

print("\nlang buckets (R@5 by query_lang->target_lang), doc=bare qry=qxq:")
print(json.dumps(results["doc=bare qry=qxq"]["lang"], ensure_ascii=False, indent=1))
print("lang buckets, doc=official qry=qxq:")
print(json.dumps(results["doc=official qry=qxq"]["lang"], ensure_ascii=False, indent=1))
with open(f"{CORPUS}/../vec_len_3x2_summary.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("saved -> vec_len_3x2_summary.json", flush=True)

emb.encode_raw = None; emb.encode_batch = None; emb.tokenize = None
emb._close_instance = None  # closures released -> Metal buffers freed
import gc; gc.collect()
sys.stdout.flush()
os._exit(0)
