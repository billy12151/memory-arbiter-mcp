#!/usr/bin/env python3
"""验证「带定语的 key 向量是否分得开」：embeddinggemma-300m 实测余弦距离.

对照组设计：
- 定语差异（生产 vs 测试环境）：用户预期"向量一定不一样"
- 字面差异同义（生产环境超时 vs 生产环境超时时间）：期望相似
- 同定语不同指标（超时 vs 并发上限）：期望不同
- 同 key 不同值（500ms vs 200ms）：期望相似
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.runner import default_embed_model  # noqa: E402
from memory_arbiter.embedder import build_embedder  # noqa: E402

GROUPS = [
    # 基线
    ("字面差异同义", "生产环境超时", "生产环境超时时间"),
    ("定语差异(生产/测试)", "生产环境超时", "测试环境超时"),
    ("完全不同主题", "生产环境超时", "包装材料供应商准入"),
    # 反义/对立定语：能否越过 0.95/0.97 阈值（用户挑战的反例候选）
    ("反义定语(昨天/今天)", "昨天的订单量", "今天的订单量"),
    ("反义定语(旧版/新版)", "旧版本的超时配置", "新版本的超时配置"),
    ("反义定语(内部/外部)", "内部 API 的超时", "外部 API 的超时"),
    ("反义定语(上行/下行)", "上行带宽限制", "下行带宽限制"),
    ("反义定语(预售/正式)", "预售商品的退款规则", "正式商品的退款规则"),
    ("反义定语(测试/正式环境)", "测试环境的发布流程", "正式环境的发布流程"),
    # 反向风险：同 key 释义变体能否守住 0.95 以上（守不住则漏进 Qwen 层，可接受）
    ("同key释义漂移", "数据库选型", "库使用架构"),
    ("同key同义定语(生产/线上)", "生产环境的超时", "线上环境的超时"),
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    embedder, warnings = build_embedder(str(default_embed_model()))
    assert embedder is not None, warnings
    cache: dict[str, list[float]] = {}

    def vec(text: str) -> list[float]:
        if text not in cache:
            cache[text] = list(embedder.embed_text(prefix="", body=text).embedding)
        return cache[text]

    for label, a, b in GROUPS:
        print(f"{label:24s} cos={_cosine(vec(a), vec(b)):.4f}   {a!r} vs {b!r}")


if __name__ == "__main__":
    main()
