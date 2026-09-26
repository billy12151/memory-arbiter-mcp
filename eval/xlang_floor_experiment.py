#!/usr/bin/env python3
"""xlang_floor_experiment — 跨语言×8.25 门槛反事实实验（owner 2026-09-26 提问）.

问题：跨语言失败（中文记忆被英文查询/反之）是不是 8.25 门槛卡的？若能识别
语言不匹配并对这类候选放宽/取消门槛，能救回多少召回、同语言噪音是否上涨？

方法：recall-v3-len 语料（终局形态 bare-doc 空间）一次性入临时库，floor=0
拉深 50 记录每个候选的融合分（_final_score），离线套门槛政策（floor 在
rerank 后过滤、池子与排序不变，故 floor-0 深拉列表 = 完整反事实基础）：
  P0 8.25（现产品）  P1 7.0（天花板报告过渡值）  P2 0（全取消）
  P3 xlang-0（仅查询语言≠候选语言时取消，同语言维持 8.25）—— owner 提案
  P4 xlang-7.0（仅不匹配对放宽到 7.0）
语言判定：查询语言=题目显式标签；候选语言=语料 fixture 行的 lang 字段
（zh/mixed/en，CJK/ASCII 比）。题集=76 条原题 + 29 条新增跨语言题
（en→zh ×20、zh→en ×9）+ 既有 4 条探针（zh→en ×3、en→zh ×1）。
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CORPUS = REPO / "eval" / "fixtures" / "recall-len"

spec = importlib.util.spec_from_file_location("harness_runner", REPO / "eval" / "runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

import memory_arbiter.search as search_mod  # noqa: E402

search_mod.QUERY_RECALL_SCORE_FLOOR = 0.0  # 反事实基础：先不过滤，离线套政策

targets = R._load_jsonl(CORPUS / "targets.jsonl")
distractors = R._load_jsonl(CORPUS / "distractors.jsonl")
labels = R._load_jsonl(CORPUS / "labels.jsonl")
queries_main = json.loads((CORPUS / "queries.json").read_text(encoding="utf-8"))["queries"]

# 新增跨语言题：qid → (query, qlang, target Lxx)
XLANG_EN2ZH = {
    "X01": ("Which system do bank marketing new requirements go through, and which one handles renewals?", "L53"),
    "X02": ("Why doesn't the Jinying platform build one-click configuration for the guobu subsidy campaigns?", "L54"),
    "X03": ("Can a requester withdraw a requirement after submission on the Jinying platform, and is closed a final state?", "L55"),
    "X04": ("What is the user's name, preferred way of address, and job background?", "L56"),
    "X05": ("In the architecture work guidelines, who is engineering implementation delegated to?", "L57"),
    "X06": ("Who are the product POs of the Jinying platform and when is phase two due?", "L58"),
    "X07": ("What is the difference between Xiaojinku Consumption and Xiaojinku Super Group-buy?", "L59"),
    "X08": ("What is the monthly total effort for resource-slot setup and placement in Jinying phase two?", "L60"),
    "X09": ("How are the 132 scenarios of unified demand cutover batched for Jinying?", "L61"),
    "X10": ("What are the three product modes in the financial goods knowledge base, VOP VSP and Jinli?", "L62"),
    "X11": ("Per the patent-writer conflict rules, can entries marked with a check mark be overwritten?", "L63"),
    "X12": ("Which four core questions does the financial goods role confirmation track align with the manager?", "L64"),
    "X13": ("Which integration items remained unfinished in Jinying phase one?", "L65"),
    "X14": ("Which scenario ranks first by volume among the twelve smart voucher horizontal scenarios?", "L66"),
    "X15": ("What cutover principles did the boss require in the Jinying phase two v4 baseline table?", "L67"),
    "X16": ("What is the positioning of JD VOP?", "L68"),
    "X17": ("What does internal control consolidation mean in the financial goods project goals?", "L69"),
    "X18": ("How are responsibilities split between Zhang Zhiwei and Pan Yunjia in the financial goods roadmap?", "L70"),
    "X19": ("Who is the default technical contact in a patent disclosure form?", "L71"),
    "X20": ("What is the API rate limit error code in the VOP operation manual?", "L72"),
}
XLANG_ZH2EN = {
    "X21": ("AgentLane 的 run monitor UI 推了什么改动，MCP 输出为什么改成纯文本？", "L01"),
    "X22": ("memory-arbiter 0.14.1 什么时候发的版，语义时序修复包含什么？", "L02"),
    "X23": ("计划模式对低风险任务继续执行和执行后审查是什么策略？", "L03"),
    "X24": ("JD socrates 本地代理怎么切到 v1 兼容层的？", "L04"),
    "X25": ("JD socrates 代理 47821 端口的路由分流怎么配的？", "L05"),
    "X26": ("Claude Code 过代理报上下文长度 40001 是怎么修的？", "L06"),
    "X27": ("ZCode 代理 40002 扩展思考报错的排查结论是什么？", "L07"),
    "X28": ("更新提醒 7 天抑制不做 ack 的 UX 决策是什么？", "L08"),
}
# 既有探针复用（queries.json 中 kind=paraphrase_crosslang）
CROSS_EXISTING = {q["qid"]: q["lang"] for q in queries_main if q["kind"] == "paraphrase_crosslang"}

target_by_key = {t["fixture_key"]: t for t in targets}
gold_by_qid = defaultdict(set)
for l in labels:
    gold_by_qid[l["qid"]].add(l["fixture_key"])

qid_of_l = {f"L{i+1:02d}": t["fixture_key"] for i, t in enumerate(targets)}


def lang_of_key(key: str) -> str:
    return target_by_key.get(key, {}).get("lang", "unknown")


def main() -> int:
    embed_model = R.default_embed_model()
    with R.temp_library(embed_model, keep_db=None) as tools:
        id_map, _perf = R.replay_fixtures(tools, targets + distractors)
        mid_to_key = {}
        for k, mid in id_map.items():
            mid_to_key.setdefault(int(mid), k)
        print(f"[replay] {len(id_map)} fixtures", flush=True)

        runs = []  # (qid, qlang, target_keys, hits[(key, score)])
        def run_query(qid, qlang, text, target_keys):
            r = tools.memory("find", {"query": text, "limit": 50, "debug_ranking": True})
            payload = r.get("data") or {}
            items = payload.get("results") or []
            hits = []
            for it in items:
                key = mid_to_key.get(int(it.get("id") or 0))
                score = it.get("_final_score")  # floor 消费的融合分（harness debug_ranking 上线字段）
                if key is not None:
                    hits.append((key, float(score or 0.0)))
            runs.append({"qid": qid, "qlang": qlang, "targets": set(target_keys), "hits": hits})

        for q in queries_main:
            if q["qid"] in CROSS_EXISTING:
                tkey = next(iter(gold_by_qid[q["qid"]]), None)
                run_query(q["qid"], q["lang"], q["query"], [tkey] if tkey else [])
            else:
                run_query(q["qid"], q["lang"], q["query"], gold_by_qid.get(q["qid"], []))
        for qid, (text, src) in XLANG_EN2ZH.items():
            tkey = qid_of_l[src]
            run_query(qid, "en", text, [tkey])
            gold_by_qid[qid].add(tkey)
        for qid, (text, src) in XLANG_ZH2EN.items():
            tkey = qid_of_l[src]
            run_query(qid, "zh", text, [tkey])
            gold_by_qid[qid].add(tkey)
        print(f"[queries] {len(runs)}", flush=True)

    # ---- 离线政策评估 ----
    POLICIES = {
        "P0_floor8.25": lambda q, s: 8.25,
        "P1_floor7.0": lambda q, s: 7.0,
        "P2_none": lambda q, s: 0.0,
        "P3_xlang0": lambda q, s: 0.0 if q["mismatch"] else 8.25,
        "P4_xlang7.0": lambda q, s: 7.0 if q["mismatch"] else 8.25,
    }

    def evaluate(policy_fn):
        buckets = defaultdict(lambda: {"n": 0, "h5": 0, "h10": 0, "d5": 0, "d10": 0,
                                       "noise": 0, "returned": 0})
        for r in runs:
            mismatch = any(lang_of_key(k) != r["qlang"] for k in r["targets"])
            thr = policy_fn({"mismatch": mismatch, "qlang": r["qlang"]}, None)
            page = [(k, s) for k, s in r["hits"] if s >= thr][:10]
            keys = [k for k, _ in page]
            gold = r["targets"]
            classes = set()
            for k in gold:
                tl = lang_of_key(k)
                classes.add("same" if tl == r["qlang"] else f"cross_{r['qlang']}->{tl}")
            for cls in classes or {"unknown"}:
                b = buckets[cls]
                b["n"] += 1
                b["h5"] += len(gold & set(keys[:5]))
                b["h10"] += len(gold & set(keys[:10]))
                b["d5"] += min(len(gold), 5)
                b["d10"] += min(len(gold), 10)
                b["noise"] += sum(1 for k in keys if k not in gold)
                b["returned"] += len(keys)
        out = {}
        for cls, b in buckets.items():
            out[cls] = {
                "n": b["n"],
                "R@5": round(b["h5"] / b["d5"], 4) if b["d5"] else None,
                "R@10": round(b["h10"] / b["d10"], 4) if b["d10"] else None,
                "noise_rate": round(b["noise"] / b["returned"], 4) if b["returned"] else None,
            }
        return out

    print("\n=== 门槛政策反事实（recall-v3-len + 33 跨语言题，floor=0 深拉 50 离线套政策） ===", flush=True)
    table = {}
    for name, fn in POLICIES.items():
        table[name] = evaluate(fn)
        print(f"\n[{name}]", flush=True)
        for cls in sorted(table[name]):
            m = table[name][cls]
            print(f"  {cls:<12} n={m['n']:<3} R@5={m['R@5']} R@10={m['R@10']} noise={m['noise_rate']}", flush=True)

    # 失败归因：跨语言题在 P0 下 gold 去哪了
    print("\n=== P0(8.25) 下跨语言 gold 归因 ===", flush=True)
    attr = defaultdict(int)
    gold_scores_cross, gold_scores_same = [], []
    for r in runs:
        mismatch = any(lang_of_key(k) != r["qlang"] for k in r["targets"])
        gold = r["targets"]
        keys50 = [k for k, _ in r["hits"]]
        top10 = keys50[:10]
        score_of = dict(r["hits"])
        for g in gold:
            gs = score_of.get(g)
            if g in top10:
                continue
            if gs is None:
                attr[f"{'cross' if mismatch else 'same'}: pool-miss(gold 不在融合池50)"] += 1
            elif gs < 8.25:
                attr[f"{'cross' if mismatch else 'same'}: floor-kill(池内但 <8.25)"] += 1
                (gold_scores_cross if mismatch else gold_scores_same).append(gs)
            else:
                attr[f"{'cross' if mismatch else 'same'}: rank>10(过门槛但排后面)"] += 1
    for k, v in sorted(attr.items()):
        print(f"  {v:>4}  {k}", flush=True)
    import statistics
    for name, arr in (("cross floor-killed gold scores", gold_scores_cross),
                      ("same floor-killed gold scores", gold_scores_same)):
        if arr:
            print(f"  {name}: n={len(arr)} min={min(arr):.2f} med={statistics.median(arr):.2f} max={max(arr):.2f}", flush=True)
        else:
            print(f"  {name}: n=0", flush=True)

    out_path = REPO / "eval" / "results" / "xlang-floor-policies.json"
    out_path.write_text(json.dumps({"policies": table, "attribution": dict(attr)},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
