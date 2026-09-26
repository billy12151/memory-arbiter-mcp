#!/usr/bin/env python3
"""assemble_recall_len — 由人工出题组装 recall-v3-len 的 queries.json + labels.jsonl.

题目为 paraphrase 白话问句（风格对齐 recall-v2 A 组），每条锚定对应 target 的
区分性细节。L73-L76 为跨语言探针（zh 查询→en target ×3、en 查询→zh target ×1），
对应 owner「跨语言召回真实场景极少，列指标但轻量覆盖」的口径。
查询语言标签与语料同尺（lang_class CJK/ASCII 比）。
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "recall-len"
CORPUS_VERSION = "recall-v3-len"

# qid 顺序 = targets.jsonl 顺序（L01..L72），L73+ 为跨语言探针（复用 target）
QUERIES = {
    "L01": "What did the AgentLane run monitor UI push include and why did MCP results go plaintext-only?",
    "L02": "When was memory-arbiter 0.14.1 released and what did the semantic timing fix cover?",
    "L03": "What is the plan-mode policy for low-risk continuation and the execution review loop?",
    "L04": "How did the JD socrates local proxy switch to the v1 compatibility layer?",
    "L05": "What is the route-split setup of the JD socrates proxy on port 47821?",
    "L06": "How was the Claude Code context length 40001 error through the JD proxy fixed?",
    "L07": "What caused the ZCode proxy 40002 extended thinking failures and how were they fixed?",
    "L08": "What was the update notice UX decision about 7-day suppression without ack?",
    "L09": "mema Team v1 工程实施方案经过两轮对抗审查后的结论是什么？",
    "L10": "memory-arbiter v0.3.0 公司群推广文案定稿时有哪些要点？",
    "L11": "README 的 The Bigger Picture 概念段是随哪个版本发的？",
    "L12": "飞天小虾私有桥接自检那条记录是干什么的？",
    "L13": "plan-mode-mcp v0.1.0 的代码审查结论如何，有没有 P0 级问题？",
    "L14": "Agnes 图片和视频 skill 的本地下载输出规则改成了什么样？",
    "L15": "0.9 本地规则文档更新加了哪些 structured claims 冲突处理口径？",
    "L16": "mema 和迷码指的是什么工具？",
    "L17": "workspace 归一为什么不能只靠向量相似度，还需要什么？",
    "L18": "patent-writer 的工具与环境里模型 fallback 链是怎样的？",
    "L19": "给其他系统装 memory-arbiter 应该优先引用哪份安装文档？",
    "L20": "v0.8 工具收敛后日常只用哪几个接口，memory_split 保留成什么定位？",
    "L21": "memory-arbiter 宣传物料的事实核验快照核到了哪些版本状态？",
    "L22": "MCP 工具描述英文化在哪个版本上线，护栏怎么处理的？",
    "L23": "plan-mode-mcp 的 SKILL.md 精简版为什么刻意不写 approve_plan？",
    "L24": "doctor 报的最新已知版本 0.9.5 比当前 0.9.8 旧，根因查出来是什么？",
    "L25": "v0.10.3 和 v0.10.4 的发布顺序为什么对调？",
    "L26": "workspace 语义归一方案里 strict 隔离下拿不准的写入怎么处理？",
    "L27": "workspace resolver 的评测里为什么最终选了 rule-first 而不是 Qwen 裁决？",
    "L28": "mema 腾讯云部署做了哪些安全加固？",
    "L29": "mema-core 的 CI 强制门覆盖哪些 Python 版本？",
    "L30": "新建子 Agent 时配置注册要动哪两个地方？",
    "L31": "v0.2.3 发版时 README 的一句话定位改成了什么？",
    "L32": "patent-writer 的文档入库规范里四层架构各放什么？",
    "L33": "memory_edit 的版本链机制包含哪些工具和数据结构？",
    "L34": "PyPI token 撤销跟踪最后的建议是什么？",
    "L35": "dogfooding 方法论里为什么说诊断工具自闭环是黄金形态？",
    "L36": "v0.8 分段和向量层是怎么绑定的，split_enabled 开关去哪了？",
    "L37": "连续三个版本漏建 GitHub Release 的坑是怎么发现和补救的？",
    "L38": "plan-mode 二期的 plan 持久化和跨 session resume 实现了什么？",
    "L39": "v0.8.6 修了 memory_edit 的什么问题？",
    "L40": "inactive 向量的幽灵召回问题是怎么修的？",
    "L41": "jd-opus-proxy 怎么拿到并注入 JingleAI 的 Cookie？",
    "L42": "mema v0.10.0 发布后复评的结论是什么？",
    "L43": "Qwen 单模型语义冲突测试里 1.5B gate 策略表现如何？",
    "L44": "M1 方案（id=647）的就绪度审查结论是什么？",
    "L45": "mema 开源与商业化的边界是怎么定的？",
    "L46": "mema 备案下来后 SEO 要做哪些站长平台收录？",
    "L47": "memory_govern 为什么所有状态变更都要求 authorized=true？",
    "L48": "营销链路里各业务线的活动推广页都用什么系统？",
    "L49": "v0.2.5 修的 CJK 查询被 strict phrase 勒死是什么 bug？",
    "L50": "v0.3.0 的宽召回加软重排是为了治理什么问题？",
    "L51": "AgentRail 为什么决定保持独立 CLI，OpenClaw 扮演什么角色？",
    "L52": "plan-mode-mcp v0.2.1 的竞品调研结论有哪些？",
    "L53": "银行营销的新需求和续期分别走哪个系统？",
    "L54": "国补活动为什么金营不做一键配置？",
    "L55": "金营平台提需人提交后还能撤回吗？",
    "L56": "用户希望被怎么称呼，职业背景是什么？",
    "L57": "架构工作规范里工程实现固定委托给谁？",
    "L58": "金营平台的 PO 分别是谁，二期截止什么时候？",
    "L59": "小金库-消费和小金库超级攒怎么区分？",
    "L60": "金营二期资源位搭建和投放每月总耗时多少？",
    "L61": "金营需求统一收口切量的批次是怎么规划的？",
    "L62": "金融带货知识库里 VOP、VSP、锦鲤三种模式有什么区别？",
    "L63": "patent-writer 冲突处理规范里带 ✅ 的条目怎么处理？",
    "L64": "金融带货角色确认要和领导对齐哪几个核心问题？",
    "L65": "金营一期建设成果里哪些还没完成？",
    "L66": "智能配券 12 个横向场景里量级排第一的是哪个？",
    "L67": "金营二期切量 v4 底表里切量原则是什么？",
    "L68": "京东 VOP 的定位是什么？",
    "L69": "金融带货专项的内部管控收口目标包括什么？",
    "L70": "金融带货工作路线图里张志维和潘韵佳怎么分工？",
    "L71": "专利交底书的技术联系人默认填什么？",
    "L72": "VOP 实操手册里 API 限流的错误码是多少？",
    # 跨语言探针（query 语言 × target 语言交叉）
    "L73": "计划模式里低风险任务的继续执行和执行后审查是什么策略？",       # zh → en (L03)
    "L74": "更新提醒为什么 7 天内不重复弹，ack 是怎么定的？",               # zh → en (L08)
    "L75": "Claude Code 过代理报上下文长度超限是怎么解决的？",             # zh → en (L06)
    "L76": "Which system do bank marketing renewals go through versus new requirements?",  # en → zh (L53)
}
CROSS_LANG_TARGET = {"L73": "L03", "L74": "L08", "L75": "L06", "L76": "L53"}


def lang_class(text: str) -> str:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cjk + lat
    if total == 0:
        return "other"
    r = cjk / total
    return "zh" if r >= 0.7 else ("en" if r <= 0.15 else "mixed")


def main() -> int:
    targets = [json.loads(l) for l in (OUT / "targets.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(targets) == 72, f"expected 72 targets, got {len(targets)}"

    queries = []
    labels = []
    for i, t in enumerate(targets, 1):
        qid = f"L{i:02d}"
        text = QUERIES[qid]
        queries.append({"qid": qid, "kind": "paraphrase", "query": text,
                        "lang": lang_class(text)})
        labels.append({"qid": qid, "fixture_key": t["fixture_key"], "label": "relevant"})
    for qid, src in CROSS_LANG_TARGET.items():
        idx = int(src[1:]) - 1
        t = targets[idx]
        queries.append({"qid": qid, "kind": "paraphrase_crosslang", "query": QUERIES[qid],
                        "lang": lang_class(QUERIES[qid])})
        labels.append({"qid": qid, "fixture_key": t["fixture_key"], "label": "relevant"})

    (OUT / "queries.json").write_text(
        json.dumps({"corpus_version": CORPUS_VERSION, "queries": queries},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with (OUT / "labels.jsonl").open("w", encoding="utf-8") as f:
        for l in labels:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    import collections
    q_lang = collections.Counter(q["lang"] for q in queries)
    print(f"queries={len(queries)} labels={len(labels)} query_lang={dict(q_lang)}")
    print(f"-> {OUT/'queries.json'} + labels.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
