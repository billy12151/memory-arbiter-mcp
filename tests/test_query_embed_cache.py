"""P1-T1 query-embed LRU cache（0.16.12 性能 Part 1）.

免模型路径：假 embedder 计数 encode 调用，验证同一 (space, lineage, epoch,
query) 只 embed 一次、换 lineage/space/epoch 不命中、vec mismatch 时缓存
不被误用（返回 None + warning）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.config import Settings  # noqa: E402
from memory_arbiter.db import MemoryDB  # noqa: E402
from memory_arbiter.tools import MemoryTools  # noqa: E402


class _FakeEmbedder:
    embed_epoch = 0

    def __init__(self) -> None:
        self.calls = 0

    def embed_text(self, prefix: str, body: str, max_body_chars: int | None = None):
        self.calls += 1
        return SimpleNamespace(embedding=[0.1, 0.2, 0.3], last_encode_error=None)


def _tools_with_fake_embedder(tmp_path: Path, monkeypatch) -> tuple[MemoryTools, _FakeEmbedder]:
    settings = Settings(
        db_path=tmp_path / "t.sqlite3", backup_jsonl=tmp_path / "b.jsonl",
        client="t-client", agent_id="t-agent",
    )
    tools = MemoryTools(settings, MemoryDB(settings))
    fake = _FakeEmbedder()
    monkeypatch.setattr(tools, "_ensure_embedder", lambda: (fake, []))
    return tools, fake


def test_same_query_embeds_once_across_finds(tmp_path: Path, monkeypatch) -> None:
    tools, fake = _tools_with_fake_embedder(tmp_path, monkeypatch)
    first = tools.memory("find", {"query": "金营平台需求管理"})
    second = tools.memory("find", {"query": "金营平台需求管理"})
    assert first["ok"] and second["ok"]
    assert fake.calls == 1  # 第二次命中缓存
    assert tools._query_embed_cache  # 缓存有条目


def test_distinct_queries_each_embed(tmp_path: Path, monkeypatch) -> None:
    tools, fake = _tools_with_fake_embedder(tmp_path, monkeypatch)
    tools.memory("find", {"query": "金营平台需求管理"})
    tools.memory("find", {"query": "mema 发版流程"})
    assert fake.calls == 2


def test_epoch_bump_invalidates_cache(tmp_path: Path, monkeypatch) -> None:
    tools, fake = _tools_with_fake_embedder(tmp_path, monkeypatch)
    tools.memory("find", {"query": "同一个查询"})
    assert fake.calls == 1
    fake.embed_epoch += 1  # GPU→CPU 降级换代
    tools.memory("find", {"query": "同一个查询"})
    assert fake.calls == 2  # 换代后不命中


def test_build_counter_bump_invalidates_cache(tmp_path: Path, monkeypatch) -> None:
    tools, fake = _tools_with_fake_embedder(tmp_path, monkeypatch)
    tools.memory("find", {"query": "同一个查询"})
    assert fake.calls == 1
    fake.embed_epoch += 1
    tools._embedder_builds += 1  # embedder 重建换代（随清空一并失效）
    tools.memory("find", {"query": "同一个查询"})
    assert fake.calls == 2


def test_space_flip_invalidates_cache(tmp_path: Path, monkeypatch) -> None:
    tools, fake = _tools_with_fake_embedder(tmp_path, monkeypatch)
    original = tools.db.get_vec_index_state
    state = {"state": "ok", "active_space_id": "space-a"}
    monkeypatch.setattr(tools.db, "get_vec_index_state", lambda: dict(state))
    tools.memory("find", {"query": "同一个查询"})
    assert fake.calls == 1
    state["active_space_id"] = "space-b"  # vec 空间换代
    tools.memory("find", {"query": "同一个查询"})
    assert fake.calls == 2
    monkeypatch.setattr(tools.db, "get_vec_index_state", original)


def test_vec_mismatch_never_serves_cache(tmp_path: Path, monkeypatch) -> None:
    tools, fake = _tools_with_fake_embedder(tmp_path, monkeypatch)
    original = tools.db.get_vec_index_state
    state = {"state": "ok", "active_space_id": "space-a"}
    monkeypatch.setattr(tools.db, "get_vec_index_state", lambda: dict(state))
    tools.memory("find", {"query": "查询"})  # 健康态写入缓存
    assert fake.calls == 1
    state["state"] = "mismatch"  # 空间失配
    result = tools.memory("find", {"query": "查询"})
    assert fake.calls == 1  # 未 embed——被首道 vec-state 门拦截
    warnings = " ".join(str(w) for w in (result.get("warnings") or []))
    assert "vec_disabled=embedding_space_mismatch" in warnings
    monkeypatch.setattr(tools.db, "get_vec_index_state", original)


def test_cache_capacity_evicts_lru(tmp_path: Path, monkeypatch) -> None:
    tools, fake = _tools_with_fake_embedder(tmp_path, monkeypatch)
    tools._query_embed_cache_capacity = 2
    for i in range(3):
        tools.memory("find", {"query": f"查询 {i}"})
    assert len(tools._query_embed_cache) == 2
    tools.memory("find", {"query": "查询 0"})  # 0 已被淘汰 → 重嵌入
    assert fake.calls == 4
