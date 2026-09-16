"""eval/runner.py unit coverage — P0-1 c2 (plan §2.6).

Uses the no-embedder degraded path (lexical fallback) so tests stay fast and
model-free; the real-model path is exercised by the harness run itself
(publish-gate tier, same as test_qwen_perf_gates).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from runner import (  # noqa: E402
    VALID_SOURCE_TYPES,
    _remember,
    build_settings,
    replay_fixtures,
    run_recall_queries,
    run_self_recall,
    temp_library,
)


def _envelope(key: str, subject: str, content: str, source_type: str = "user_confirmed") -> dict:
    return {
        "fixture_key": key,
        "content_sha": key.removeprefix("t-").ljust(64, "0"),
        "content": content,
        "subject": subject,
        "tags": ["eval"],
        "workspace": "eval-ws",
        "workspace_canonical": "eval-ws",
        "event_time": "2026-09-16T00:00:00+00:00",
        "source_type": source_type,
        "metadata": {},
    }


def test_settings_are_harness_pinned() -> None:
    import tempfile

    from memory_arbiter.config import Settings

    with tempfile.TemporaryDirectory() as tmp:
        settings = build_settings(Path(tmp), embed_model=None)
        assert isinstance(settings, Settings)
        assert settings.semantic_conflict_enabled is False  # recall 套件不起 Qwen
        assert settings.isolation == "none"
        assert settings.client == "mema-eval"
        assert "harness.sqlite3" in str(settings.db_path)


def test_temp_library_creates_and_destroys() -> None:
    with temp_library(embed_model=None) as tools:
        db_file = Path(tools.settings.db_path)
        assert db_file.exists()
        result = tools.memory("remember", {"content": "x", "subject": "s"})
        assert result["ok"] is True
        workdir = db_file.parent
    assert not workdir.exists()


def test_replay_is_idempotent_on_identical_content() -> None:
    envelopes = [
        _envelope("t-aaa1", "主题甲", "内容一：金营平台需求管理"),
        _envelope("t-bbb2", "主题乙", "内容二：mema 发版流程"),
        # 与第一条同内容不同 key —— 0.16.6 防重门应幂等映射到同一 id
        _envelope("t-ccc3", "主题甲", "内容一：金营平台需求管理"),
    ]
    with temp_library(embed_model=None) as tools:
        id_map = replay_fixtures(tools, envelopes)
        assert len(set(id_map.values())) == 2
        assert id_map["t-aaa1"] == id_map["t-ccc3"]


def test_legacy_source_type_falls_back_to_unknown() -> None:
    assert "agent_observed" not in VALID_SOURCE_TYPES
    with temp_library(embed_model=None) as tools:
        new_id, replayed = _remember(tools, _envelope("t-ddd4", "主题丙", "内容三", source_type="agent_observed"))
        assert new_id is not None and replayed is False


def test_recall_collection_shape() -> None:
    envelopes = [_envelope("t-aaa1", "金营平台需求管理中枢", "内容一：金营平台需求管理"),
                 _envelope("t-bbb2", "mema 发版流程", "内容二：mema 发版流程检查单")]
    queries = [{"qid": "T01", "kind": "lookup", "query": "金营平台需求管理"}]
    with temp_library(embed_model=None) as tools:
        id_map = replay_fixtures(tools, envelopes)
        collected = run_recall_queries(tools, queries, id_map)
        self_recall = run_self_recall(tools, envelopes, id_map)
    row = collected[0]
    assert row["qid"] == "T01" and row["ok"] and row["elapsed_ms"] >= 0
    assert row["hits"] and row["hits"][0]["fixture_key"] == "t-aaa1"
    assert row["hits"][0]["rank"] == 1
    assert json.dumps(row, ensure_ascii=False)  # collector 产物可序列化
    assert all(item["in_top10"] for item in self_recall)
