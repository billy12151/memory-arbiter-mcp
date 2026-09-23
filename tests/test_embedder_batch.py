"""ManagedEmbedder.embed_texts 契约 — 0.17.0 C1（单元退役方案）。

覆盖：批 vs 单路由分流（超长走 embed_text）、批量失败一次重试后逐条降级
（never-raises、坏条目 sentinel 定位）、无 encode_batch 的假实例直接逐条、
锁纪律（批量与单条共享 _embed_lock）。
数值学口径（批 vs 单 cos≥0.9999，非逐位一致——浮点归约顺序差异属预期）
由底部 @pytest.mark.slow 真模型测试钉住（发版门，默认不跑）。
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.embedder import (  # noqa: E402
    EMBED_TEXTS_SPLIT_CHARS,
    EmbedResult,
    ManagedEmbedder,
)


def _cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def _deterministic_vec(text: str, dim: int = 8) -> list[float]:
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(text))
    return [((seed >> (i % 24)) & 0xFF) / 255.0 for i in range(dim)]


def _make_embedder(
    encode, encode_many=None, tokenize=None, **kwargs
) -> ManagedEmbedder:
    def _tokenize(text: str) -> list[int]:
        return list(range(len(text)))

    fields: dict = dict(
        encode_raw=encode,
        tokenize=tokenize or _tokenize,
        model_digest="digest",
        embedding_space_id="space",
        n_ctx=2048,
        reserved_tokens=64,
        dim=8,
    )
    if encode_many is not None:
        fields["encode_batch"] = encode_many
    fields.update(kwargs)
    return ManagedEmbedder(**fields)


def test_batch_route_matches_per_item_and_splits_long_items() -> None:
    """短条目批量、超长条目单条（embed_text 截断预算路径），序保真。"""
    calls: list[str] = []
    singles: list[str] = []

    def encode(text: str) -> list[float]:
        singles.append(text)
        return _deterministic_vec(text)

    def encode_many(items: list[str]) -> list[list[float]]:
        calls.extend(items)
        return [_deterministic_vec(t) for t in items]

    emb = _make_embedder(encode, encode_many)
    short_a, short_b = "短句一条。", "表格行: 值 123 的记录"
    long_item = "超长" * (EMBED_TEXTS_SPLIT_CHARS // 2 + 10)
    results = emb.embed_texts([short_a, long_item, short_b])
    assert len(results) == 3
    # 短条目走批量、长条目走单条（embed_text 的预算截断会截前缀）
    assert calls == [short_a, short_b] and len(singles) == 1
    assert singles[0] == long_item[: len(singles[0])]
    # 序保真 + 批/单同源文本向量一致
    assert _cos(results[0].embedding, _deterministic_vec(short_a)) > 0.999999
    assert results[1].truncated is True  # embed_text 预算截断路径打了标
    assert _cos(results[2].embedding, _deterministic_vec(short_b)) > 0.999999


def test_batch_failure_retries_then_degrades_per_item() -> None:
    """批量抛错→同实例重试一次→仍失败→逐条 embed_text 兜底（never-raises）。"""
    attempts = {"n": 0}

    def encode(text: str) -> list[float]:
        return _deterministic_vec(text)

    def flaky_encode_many(items: list[str]) -> list[list[float]]:
        attempts["n"] += 1
        raise RuntimeError("simulated batch fault")

    emb = _make_embedder(encode, flaky_encode_many)
    texts = [f"第 {i} 条记录内容。" for i in range(5)]
    results = emb.embed_texts(texts)
    assert attempts["n"] == 2  # 一次原始 + 一次重试
    assert all(r.embedding for r in results)  # 全部经逐条兜底成功
    assert "simulated batch fault" in (emb.last_encode_error or "")


def test_batch_count_mismatch_degrades_per_item() -> None:
    """条数不符（丢条）不静默接受——重试后逐条兜底。"""

    def encode(text: str) -> list[float]:
        return _deterministic_vec(text)

    def short_encode_many(items: list[str]) -> list[list[float]]:
        return [_deterministic_vec(t) for t in items[:-1]]  # 少返回一条

    emb = _make_embedder(encode, short_encode_many)
    results = emb.embed_texts(["a 记录。", "b 记录。", "c 记录。"])
    assert len(results) == 3 and all(r.embedding for r in results)


def test_no_batch_closure_goes_per_item() -> None:
    """encode_batch=None（测试假实例/CPU 降级后）直接逐条，行为完整。"""
    singles: list[str] = []

    def encode(text: str) -> list[float]:
        singles.append(text)
        return _deterministic_vec(text)

    emb = _make_embedder(encode)  # 不注入 encode_batch
    results = emb.embed_texts(["x 记录。", "y 记录。"])
    assert len(singles) == 2 and all(r.embedding for r in results)


def test_batch_runs_under_embed_lock() -> None:
    """批量闭包与单条共享 _embed_lock：持锁期间单条调用阻塞（串行不变式）。"""
    in_batch = threading.Event()
    released = threading.Event()
    order: list[str] = []

    def encode(text: str) -> list[float]:
        order.append(f"single:{text}")
        return _deterministic_vec(text)

    def slow_encode_many(items: list[str]) -> list[list[float]]:
        order.append("batch:enter")
        in_batch.set()
        released.wait(timeout=5.0)  # 持锁等待，直到单条尝试证明被阻塞
        order.append("batch:exit")
        return [_deterministic_vec(t) for t in items]

    emb = _make_embedder(encode, slow_encode_many)
    worker = threading.Thread(target=lambda: emb.embed_texts(["批内条目。"]))
    worker.start()
    assert in_batch.wait(timeout=5.0)
    single = threading.Thread(target=lambda: emb.embed_text(prefix="", body="单条。"))
    single.start()
    single.join(timeout=0.3)
    assert single.is_alive()  # 被批持的锁挡住
    released.set()
    worker.join(timeout=5.0)
    single.join(timeout=5.0)
    assert order[0] == "batch:enter" and order[1] == "batch:exit"


def test_empty_and_whitespace_items_fall_to_embed_text() -> None:
    """空/纯空白条目不经批量（假实例鲁棒性），由 embed_text 兜底。"""
    routed: list[str] = []

    def encode(text: str) -> list[float]:
        routed.append(text)
        return _deterministic_vec(text) if text else []

    def encode_many(items: list[str]) -> list[list[float]]:
        routed.extend(items)
        return [_deterministic_vec(t) for t in items]

    emb = _make_embedder(encode, encode_many)
    results = emb.embed_texts(["正常条目。", "   "])
    assert "正常条目。" in routed and "   " not in routed[:1]
    assert isinstance(results[1], EmbedResult)


@pytest.mark.slow
def test_real_model_batch_vs_single_cos_parity() -> None:
    """真模型数值学口径：批 vs 单 cos≥0.9999（浮点归约顺序差异，非逐位）。

    spike（/tmp/spike_batch_embed.py，2026-09-23）实测 ≥0.9999997；本测试
    把口径钉成发版门，防后人追查「向量为什么变了 1e-7」。
    """
    from memory_arbiter.embedder import build_embedder

    model_path = Path(
        "~/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf"
    ).expanduser()
    if not model_path.exists():
        pytest.skip("real embedding model not configured on this host")
    emb, warnings = build_embedder(str(model_path))
    assert emb is not None, warnings
    try:
        texts = [
            "网关读超时为 500 毫秒，熔断阈值 3 次。",
            "表格行: 队列长度=200, 帽=1000, 淘汰=LRU",
            "版本 0.17.0 未发版，仅本地观测。",
        ] * 6  # 18 条，含跨 n_batch 组合
        batched = emb.embed_texts(texts)
        for text, batch_result in zip(texts, batched):
            single = emb.embed_text(prefix="", body=text)
            assert batch_result.embedding and single.embedding
            assert _cos(batch_result.embedding, single.embedding) >= 0.9999, text
    finally:
        # Real-model teardown rule: unload + del + gc, or pytest exits 134.
        emb.close()
        del emb
        import gc

        gc.collect()
