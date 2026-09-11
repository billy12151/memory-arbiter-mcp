#!/usr/bin/env python3
"""eval_grammar_free_matrix — 0.15.14 A2 去语法约束校准矩阵（复现脚本）.

2026-09-11 已跑的全量结果（n=62 真实对，温度 0）存档于
docs/eval/eval-grammar-free-matrix-2026-09-11.json；结论与决策规则见
ZCodeProject/docs/mema-01514-qwen-perf-and-cleanup-plan-2026-09-11.md §A2：
G0（语法基线）/ L0（纯 prompt）/ L2（assistant 前缀播种）三条件 JSON 与
四字段全 62/62，墙钟 L0/L2 ≈ 0.46/0.43s vs G0 1.54s（−72%），采纳
L2 + L3 后置截断。

本脚本独立于产品代码（自嵌 prompt 与 schema 副本），可在任意 pair 集上
复现同一矩阵，供下游库换模型/换提示词时重标定：

  pairs 文件格式（JSON 数组，每项两个 envelope，字段同产品 envelope）：
  [{"left": {"quote": "...", "subject": "...", ...}, "right": {...}}, ...]

  运行：
  python scripts/eval_grammar_free_matrix.py --model <gguf> --pairs pairs.json \
      [--conditions G0,L0,L2] [--out result.json]

只读模型与输入文件；不触碰任何数据库。

实施日独立复验（2026-09-11，16 对活库样本——比校准集更重的 plan 文档型
长引用）：G0 16/16（语法强制）；L0 12/16 四字段 + 2 超字（L3 可救），
2 例为「额外字段」型（模型抄 metadata 键入 JSON），走产品既有 schema
反馈重试；墙钟 L0 ≈ G0 的一半（方向与档案一致）。

**L2 播种被产品集成证伪（2026-09-11 实施日裁定）**：矩阵只测协议有效
性/墙钟，未测语义门质量；真模型 e2e 发布门（Tier1 校准对）上 L2 的
反向抽取确定性错位（「高 ROI 候选分析」vs「重排版」），L0 与 G0 同
质量。产品落地取 **L0 + L3**（无播种，create_chat_completion 去
response_format），墙钟收益保持 −70% 档。脚本保留三条件供复现与再评
估。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# --- pair-v6 prompt（与产品 _PAIR_PROMPT 逐字一致；措辞冻结，勿改） --------
_PAIR_PROMPT = """你只做条件抽槽，直接以 { 开头输出一个 JSON 对象，不要解释、复述输入或裁决。
对象必须恰好包含四个字符串字段：attribute_a、value_a、attribute_b、value_b。
attribute 是两侧正在回答的最小可比较问题，不包含具体值、时间、环境或版本；value 是原证据中该属性的具体取值，取原文中的连续片段，长度不超过 64 字、不超过 12 个词；原句过长时截取最能体现取值差异的连续片段，禁止整句照抄，value 不得以句号、叹号、分号等句末标点结尾。
无论是否能可靠抽取，都必须输出全部四个字符串字段，不得省略字段。无法可靠抽取时将对应字段写成字符串 "__unknown__"；不要输出 null、conflict、coexistence、winner、confidence 或额外字段。
例：A=生产数据库使用 MySQL。B=生产数据库使用 SQLite。
输出：{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型","value_b":"SQLite"}"""

_PAIR_RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "attribute_a": {"type": "string", "maxLength": 80},
            "value_a": {"type": "string", "maxLength": 64},
            "attribute_b": {"type": "string", "maxLength": 80},
            "value_b": {"type": "string", "maxLength": 64},
        },
        "required": ["attribute_a", "value_a", "attribute_b", "value_b"],
        "additionalProperties": False,
    },
}

# L2 播种前缀：assistant 轮从 JSON 第一个字段的中途开始续写
_PAIR_SEED = '{"attribute_a":"'


def _memory_text(record: dict[str, Any]) -> str:
    tags = ", ".join(record.get("tags") or []) if isinstance(record.get("tags"), list) else str(record.get("tags") or "")
    raw_metadata = record.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    fields = (
        ("subject", record.get("subject")), ("tags", tags),
        ("workspace_canonical", record.get("workspace_canonical")),
        ("memory_id", record.get("memory_id")), ("version", record.get("version")),
        ("event_time", record.get("event_time")),
        ("entity", metadata.get("entity")), ("scope", metadata.get("scope")),
    )
    return "; ".join(f"{key}={value}" for key, value in fields if value not in (None, "", []))


def _pair_text(left: dict[str, Any], right: dict[str, Any], *, quote_cap: int = 400) -> str:
    left_quote = str(left.get("quote") or left.get("content") or "")[:quote_cap]
    right_quote = str(right.get("quote") or right.get("content") or "")[:quote_cap]
    return (
        f"A metadata: {_memory_text(left)}\n"
        f"B metadata: {_memory_text(right)}\n"
        "只根据以下证据原文抽取 attribute/value：\n"
        f"A证据原文={left_quote}\nB证据原文={right_quote}"
    )


def _user_turn(left: dict[str, Any], right: dict[str, Any]) -> str:
    return f"输入: {_pair_text(left, right)}\n输出:"


def _extract_first_json_object(raw: str) -> str | None:
    text = raw or ""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _validate(raw: str) -> dict[str, Any]:
    """返回 {json, fields, over, words}；与产品 extraction 协议同口径。"""
    snippet = _extract_first_json_object(raw)
    out: dict[str, Any] = {"json": False, "fields": False, "over": False, "words": False}
    if not snippet:
        return out
    try:
        parsed = json.loads(snippet)
    except ValueError:
        return out
    out["json"] = True
    names = {"attribute_a", "value_a", "attribute_b", "value_b"}
    if not isinstance(parsed, dict) or set(parsed) != names:
        return out
    if not all(isinstance(parsed[name], str) for name in names):
        return out
    out["fields"] = True
    out["over"] = (
        len(parsed["value_a"]) > 64 or len(parsed["value_b"]) > 64
        or len(parsed["attribute_a"]) > 80 or len(parsed["attribute_b"]) > 80
    )
    out["words"] = any(len(str(parsed[name]).split()) > 12 for name in ("value_a", "value_b"))
    return out


def run_condition(llm: Any, condition: str, left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], str]:
    started = time.perf_counter()
    if condition in {"G0", "L0"}:
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": _PAIR_PROMPT},
                {"role": "user", "content": _user_turn(left, right)},
            ],
            "max_tokens": 384,
            "temperature": 0.0,
            "top_p": 0.9,
            "stop": ["\n\n"],
        }
        if condition == "G0":
            kwargs["response_format"] = _PAIR_RESPONSE_FORMAT
        out = llm.create_chat_completion(**kwargs)
        raw = out["choices"][0]["message"]["content"] or ""
    elif condition == "L2":
        # 手工 ChatML（Qwen2.5 内嵌模板同形）+ assistant 前缀播种：
        # 模型在 JSON 第一个字段中途续写，create_chat_completion 不支持
        # 前缀续写（模板会闭合 assistant 轮），必须走裸 completion。
        prompt = (
            f"<|im_start|>system\n{_PAIR_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{_user_turn(left, right)}<|im_end|>\n"
            f"<|im_start|>assistant\n{_PAIR_SEED}"
        )
        out = llm.create_completion(
            prompt=prompt, max_tokens=384, temperature=0.0, top_p=0.9,
            stop=["<|im_end|>", "\n\n"],
        )
        raw = _PAIR_SEED + (out["choices"][0]["text"] or "")
    else:
        raise ValueError(f"unknown condition: {condition}")
    elapsed = time.perf_counter() - started
    verdict = _validate(raw)
    verdict["t"] = elapsed
    verdict["raw"] = raw
    return verdict, raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="GGUF 模型路径")
    parser.add_argument("--pairs", required=True, help="pairs JSON 文件")
    parser.add_argument("--conditions", default="G0,L0,L2")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--out", default="eval-grammar-free-matrix-result.json")
    args = parser.parse_args()

    from llama_cpp import Llama

    pairs: list[dict[str, Any]] = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    llm = Llama(
        model_path=args.model, n_ctx=args.n_ctx, n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers, verbose=False,
    )
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    stats: dict[str, dict[str, Any]] = {}
    detail: list[dict[str, Any]] = []
    for condition in conditions:
        times: list[float] = []
        counters = {"json": 0, "fields": 0, "over": 0, "words": 0, "n": 0}
        for index, pair in enumerate(pairs):
            verdict, _raw = run_condition(llm, condition, pair["left"], pair["right"])
            times.append(verdict["t"])
            for key in ("json", "fields", "over", "words"):
                counters[key] += int(bool(verdict[key]))
            counters["n"] += 1
            detail.append({
                "pair": index, "cond": condition,
                "json": verdict["json"], "fields": verdict["fields"],
                "over": verdict["over"], "words": verdict["words"],
                "t": round(verdict["t"], 4),
            })
        times.sort()
        p95 = times[max(0, int(0.95 * len(times) + 0.5) - 1)]
        stats[condition] = {
            **counters,
            "t_mean": round(sum(times) / len(times), 3),
            "t_p95": round(p95, 3),
        }
        print(f"{condition}: {stats[condition]}")
    Path(args.out).write_text(
        json.dumps({"stats": stats, "detail": detail}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
