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
    FIXTURES,
    VALID_SOURCE_TYPES,
    _load_jsonl,
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
        id_map, perf_rows = replay_fixtures(tools, envelopes)
        assert len(set(id_map.values())) == 2
        assert id_map["t-aaa1"] == id_map["t-ccc3"]
        # 0.16.12 perf 采集：逐条写入计时行，字段齐且非负，产物可序列化
        assert len(perf_rows) == len(envelopes)
        assert perf_rows[2]["duplicate_replay"] is True
        assert all(row["elapsed_ms"] >= 0 for row in perf_rows)
        assert all(
            {"fixture_key", "elapsed_ms", "duplicate_replay"} <= set(row) for row in perf_rows
        )
        assert json.dumps(perf_rows, ensure_ascii=False)


def test_legacy_source_type_falls_back_to_unknown() -> None:
    assert "agent_observed" not in VALID_SOURCE_TYPES
    with temp_library(embed_model=None) as tools:
        new_id, replayed, elapsed_ms = _remember(
            tools, _envelope("t-ddd4", "主题丙", "内容三", source_type="agent_observed"),
        )
        assert new_id is not None and replayed is False
        assert elapsed_ms >= 0


def test_pairs_large_fixture_schema() -> None:
    """P0-T2 中大型用例组：schema/体量/单元数/标签/埋点唯一性守卫。"""
    from memory_arbiter.evidence import local_text_units

    rows = _load_jsonl(FIXTURES / "conflict" / "pairs_large.jsonl")
    regular_ids = {row["pair_id"] for row in _load_jsonl(FIXTURES / "conflict" / "pairs.jsonl")}
    assert 6 <= len(rows) <= 8
    conflict = [row for row in rows if row["label"] == "true_conflict"]
    coexist = [row for row in rows if row["label"] == "coexist"]
    assert 2 <= len(conflict) <= 5 and 1 <= len(coexist) <= 3
    for row in rows:
        assert row["shape"] == "large_unit"
        assert row["label"] in {"true_conflict", "coexist", "noise"}
        assert row["pair_id"] not in regular_ids  # 不与常规对集撞 id
        units_seen: dict[str, int] = {}
        for side in ("left", "right"):
            member = row[side]
            for field in ("content", "subject", "tags", "workspace", "event_time",
                          "source_type", "metadata", "memory_id"):
                assert field in member
            content = member["content"]
            assert 4 <= len(content.encode("utf-8")) / 1024 <= 8, (row["pair_id"], side)
            units = local_text_units(member["subject"], content)
            assert 30 <= len(units) <= 60, (row["pair_id"], side, len(units))
            units_seen[side] = len(units)
        # 左右单元数一致（分歧以等位替换单元实现，便于逐单元对齐）
        left_units = [u.text for u in local_text_units(row["left"]["subject"], row["left"]["content"])]
        right_units = [u.text for u in local_text_units(row["right"]["subject"], row["right"]["content"])]
        assert len(left_units) == len(right_units)
        diffs = [i for i, (a, b) in enumerate(zip(left_units, right_units)) if a != b]
        if row["label"] == "true_conflict":
            assert len(diffs) == 1, (row["pair_id"], diffs)  # 单一冲突埋点，可归因
        else:
            assert len(diffs) >= 1  # 共存对：有差异但无对立决策


def test_label_overrides_covers_large_pairs_audit() -> None:
    import json

    overrides = json.loads((FIXTURES / "conflict" / "label_overrides.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(FIXTURES / "conflict" / "pairs_large.jsonl")
    for row in rows:
        assert row["pair_id"] in overrides  # 自标审计已登记（null=不改 label）


def test_recall_collection_shape() -> None:
    envelopes = [_envelope("t-aaa1", "金营平台需求管理中枢", "内容一：金营平台需求管理"),
                 _envelope("t-bbb2", "mema 发版流程", "内容二：mema 发版流程检查单")]
    queries = [{"qid": "T01", "kind": "lookup", "query": "金营平台需求管理"}]
    with temp_library(embed_model=None) as tools:
        id_map, _ = replay_fixtures(tools, envelopes)
        collected = run_recall_queries(tools, queries, id_map)
        self_recall = run_self_recall(tools, envelopes, id_map)
    row = collected[0]
    assert row["qid"] == "T01" and row["ok"] and row["elapsed_ms"] >= 0
    assert row["hits"] and row["hits"][0]["fixture_key"] == "t-aaa1"
    assert row["hits"][0]["rank"] == 1
    assert json.dumps(row, ensure_ascii=False)  # collector 产物可序列化
    assert all(item["in_top10"] for item in self_recall)
