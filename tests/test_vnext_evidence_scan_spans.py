from __future__ import annotations
from pathlib import Path
import sqlite3
import contextlib
import uuid
from typing import Any
import pytest
from memory_arbiter.config import Settings
from memory_arbiter.db_generation import CONFLICT_DETECTOR_VERSION
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.evidence import evidence_content_hash, local_text_units
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryRecord
from memory_arbiter.semantic_conflict import (
    decide_evidence,
    normalize_value,
)
from memory_arbiter.tools import MemoryTools
from memory_arbiter.vnext_migration import build, inspect

from tests.test_vnext_evidence import (
    CountingEmbedder, FakeEmbedder, ModelSignal,
    _conflict_member, _hits_with_metadata, _job_snapshot, _pass_cos_gate,
    _seed_notice, _strict_pair_backend, _structured_conflict_payload,
    make_tools,
)  # noqa: F401  # facade 基建（无状态），域文件统一带包 import

# ── test_vnext_evidence scan candidates/read spans domain (commit 12 split) ──

def test_scan_candidates_enumerates_filters_and_paginates(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    dup1 = tools.memory_write(content="完全一样的配置说明文字。", subject="same", tags=[])["data"]
    dup2 = tools.memory_write(content="完全一样的配置说明文字。", subject="same", tags=[])["data"]
    far = tools.memory_write(content="PostgreSQL 数据库生产环境配置。", subject="db", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    result = tools.memory_repair(
        "scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True},
    )
    assert result["ok"] is True
    data = result["data"]
    pairs = {(c["left_id"], c["right_id"]): c for c in data["candidates"]}
    # The numeric-change pair is surfaced as a notify clue with snippets.
    key = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    assert key in pairs
    clue = pairs[key]
    assert clue["route"] == "review_candidate"
    assert "numeric_value_candidate" in clue["reasons"]
    assert clue["left_snippet"] and clue["right_snippet"]
    # Equivalent duplicates and unrelated memories are not surfaced.
    dup_key = (min(dup1["id"], dup2["id"]), max(dup1["id"], dup2["id"]))
    assert dup_key not in pairs
    for c in data["candidates"]:
        assert far["id"] not in (c["left_id"], c["right_id"])

    # Enumeration alone does not claim persistence: an external loop may retry
    # until record_conflict succeeds, with the same frozen candidate identity.
    repeated = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True})
    repeated_clue = next(c for c in repeated["data"]["candidates"] if (c["left_id"], c["right_id"]) == key)
    assert repeated_clue["candidate_key_hash"] == clue["candidate_key_hash"]

    # A successfully recorded candidate snapshot suppresses re-surfacing. Use
    # the group API and exact candidate_key rather than removed pair columns.
    registered = tools.db.record_conflict_group(
        workspace_canonical="default", slot_key=None, members=clue["members"], value_groups=[],
        candidate_key=clue["candidate_key"], detection_reason="已分诊", source="scheduled_scan",
        detector_version=CONFLICT_DETECTOR_VERSION, status="not_a_conflict",
    )
    assert registered["outcome"] == "inserted"
    dismissed_scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True})
    assert key not in {(c["left_id"], c["right_id"]) for c in dismissed_scan["data"]["candidates"]}
    tools.memory("update", {"memory_id": a["id"], "new_content": "接口超时为 60 秒，已修订。", "reason": "r"})
    assert tools.wait_semantic_worker_drained(timeout=5)
    reopened = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True})
    assert key in {(c["left_id"], c["right_id"]) for c in reopened["data"]["candidates"]}


def test_scan_candidates_pagination_and_check_gate(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    ids = [
        tools.memory_write(content=f"扫描锚点 {i}：连接池 {i}0。", subject=f"scan{i}", tags=[])["data"]["id"]
        for i in range(3)
    ]
    # Real rule check pair: structurally similar, no deterministic signal.
    chk1 = tools.memory_write(content="服务使用数据库甲存储数据。", subject="store", tags=[])["data"]
    chk2 = tools.memory_write(content="服务采用数据库乙保存数据。", subject="store", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    # Cursor pagination: one anchor per batch, union covers everything.
    collected, anchor, batches = set(), 0, 0
    while True:
        r = tools.memory_repair("scan_candidates", {"anchor_memory_id": anchor, "batch": 1, "k": 10})
        assert r["ok"] is True
        batches += 1
        for c in r["data"]["candidates"]:
            collected.add((c["left_id"], c["right_id"]))
        if r["data"]["next_anchor_memory_id"] is None:
            break
        anchor = r["data"]["next_anchor_memory_id"]
    assert batches == len(ids) + 2

    # include_check gates rule-level check clues in and out.
    chk_key = (min(chk1["id"], chk2["id"]), max(chk1["id"], chk2["id"]))
    base = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 10, "k": 10, "include_quotes": True})
    with_check = tools.memory_repair(
        "scan_candidates", {"anchor_memory_id": 0, "batch": 10, "k": 10, "include_check": True},
    )
    base_pairs = {(c["left_id"], c["right_id"]) for c in base["data"]["candidates"]}
    check_pairs = {(c["left_id"], c["right_id"]) for c in with_check["data"]["candidates"]}
    assert all(c["route"] in {"review_candidate", "notice_ready"} for c in base["data"]["candidates"])
    assert chk_key not in base_pairs
    assert chk_key in check_pairs
    chk_clue = next(c for c in with_check["data"]["candidates"] if (c["left_id"], c["right_id"]) == chk_key)
    assert chk_clue["route"] == "review_candidate"
    assert len(check_pairs) > len(base_pairs)


def test_record_conflict_not_a_conflict_registration(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="阈值 100。", subject="th", tags=[])["data"]
    b = tools.memory_write(content="阈值 200。", subject="th", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True})
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)

    payload = {
        "slot_key": None,
        "members": clue["members"],
        "value_groups": [],
        "candidate_key": clue["candidate_key"],
        "status": "not_a_conflict",
        "detector_version": CONFLICT_DETECTOR_VERSION,
        "source": "scheduled_scan",
        "reason": "同主题演进，无需治理",
        # not_a_conflict dispositions require explicit authorization (B-C3).
        "authorized": True,
    }
    marked = tools.memory_repair("record_conflict", payload)
    assert marked["ok"] is True
    row = tools.db.get_conflict(marked["data"]["conflict_id"])
    assert row["status"] == "not_a_conflict"
    assert row["slot_key"] is None
    assert row["candidate_key_hash"]

    repeat = tools.memory_repair("record_conflict", payload)
    assert repeat["ok"] is True and repeat["data"]["outcome"] == "deduped"
    after = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10, "include_quotes": True})
    assert pair not in {(c["left_id"], c["right_id"]) for c in after["data"]["candidates"]}

    invalid_open = tools.memory_repair("record_conflict", {**payload, "status": "open"})
    assert invalid_open["ok"] is False
    assert invalid_open["data"]["outcome"] == "invalid_slot_key"


def test_scan_candidates_review_regression_batch(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    # batch=0 / k=0 are falsy but must be rejected, not coerced to defaults.
    zero = tools.memory_repair("scan_candidates", {"batch": 0})
    assert zero["ok"] is False
    zero_k = tools.memory_repair("scan_candidates", {"k": 0})
    assert zero_k["ok"] is False

    # Subject-only version progression must not become a numeric clue:
    # only body-text units participate (matching the write-time path).
    v1 = tools.memory_write(content="服务部署文档正文内容，部署步骤说明。", subject="服务 2024 规划", tags=[])["data"]
    v2 = tools.memory_write(content="服务部署文档正文内容，回滚步骤说明。", subject="服务 2025 规划", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10, "include_quotes": True})
    subject_pair = (min(v1["id"], v2["id"]), max(v1["id"], v2["id"]))
    assert subject_pair not in {(c["left_id"], c["right_id"]) for c in scan["data"]["candidates"]}

    # A memory pair whose equivalent unit precedes its numeric-change unit
    # must still surface the numeric clue (peer-blacklisting regression).
    mixed = tools.memory_write(
        content="完全相同的开场说明。\n重试次数为 3 次。", subject="mixed", tags=[],
    )["data"]
    twin = tools.memory_write(
        content="完全相同的开场说明。\n重试次数为 5 次。", subject="mixed", tags=[],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    scan2 = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10, "include_quotes": True})
    mixed_pair = (min(mixed["id"], twin["id"]), max(mixed["id"], twin["id"]))
    clue = next(
        (c for c in scan2["data"]["candidates"] if (c["left_id"], c["right_id"]) == mixed_pair), None,
    )
    assert clue is not None
    assert "numeric_value_candidate" in clue["reasons"]
    # Snippets show the numeric-bearing text, not the equivalent preamble.
    snippets = clue["left_snippet"] + clue["right_snippet"]
    assert "重试次数" in snippets


def test_scan_candidates_strict_workspace_does_not_leak(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    tools.settings.isolation = "strict"
    # Workspace names whose fake embeddings differ (one contains "pgsql"),
    # so vector resolution keeps them canonically separate.
    tools.memory_write(content="主库配置说明与参数清单。", subject="db", tags=[], workspace="dbapgsql")["data"]
    tools.memory_write(
        content="接口超时为 5 秒。", subject="timeout", tags=[], workspace="apisvc",
    )["data"]
    tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[], workspace="apisvc")["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    with tools.db.connection() as conn:
        canonicals = {
            row["id"]: row["workspace_canonical"]
            for row in conn.execute("SELECT id, workspace_canonical FROM memories")
        }
    assert set(canonicals.values()) == {"dbapgsql", "apisvc"}
    # A later active row in another workspace must not create a phantom next
    # cursor for strict workspace pagination.
    tools.memory_write(content="末尾外部工作区。", subject="tail", tags=[], workspace="dbapgsql")
    assert tools.wait_semantic_worker_drained(timeout=5)

    result = tools.memory_repair(
        "scan_candidates", {"anchor_memory_id": 0, "batch": 50, "k": 10, "workspace": "apisvc"},
    )
    assert result["ok"] is True
    assert result["data"]["next_anchor_memory_id"] is None
    for clue in result["data"]["candidates"]:
        assert "postgres" not in clue["left_snippet"] and "postgres" not in clue["right_snippet"]
        for deep_read in clue["deep_read"].values():
            assert deep_read["workspace"] == "apisvc"
            executed = tools.memory("read", deep_read)
            assert executed["ok"] is True, executed
        for side in (clue["left_id"], clue["right_id"]):
            record = tools.db.get_memory(side)
            canonical = record.get("workspace_canonical") or record.get("workspace")
            assert canonical == "apisvc", f"leaked anchor from workspace {canonical}"


def test_read_span_window_and_clue_deep_read(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    content = "第一段背景说明文字。\n重试次数为 3 次。\n第三段运维备注信息。"
    a = tools.memory_write(content=content, subject="span", tags=[])["data"]
    b = tools.memory_write(content="第一段背景说明文字。\n重试次数为 5 次。\n第三段运维备注信息。", subject="span", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    # 0.16.0 unit-aligned span read (plan §3): the window returns the COMPLETE
    # evidence units overlapping [start, end) — never a half-sentence slice.
    full = tools.memory("read", {"memory_id": a["id"]})["data"]["memory"]
    total = len(full["content"])
    windowed = tools.memory("read", {"memory_id": a["id"], "span": {"start": 0, "end": 12}})
    assert windowed["ok"] is True
    aligned = windowed["data"]["memory"]["content"]
    # Unit-aligned: whole units only — every sentence survives intact even
    # though the raw window ends mid-unit at char 12.
    assert "重试次数为 3 次。" in aligned
    assert windowed["data"]["span"]["unit_aligned"] is True
    assert windowed["data"]["span"]["total_chars"] == total
    assert windowed["data"]["span"]["end"] == 12
    clipped = tools.memory("read", {"memory_id": a["id"], "span": {"start": total - 3, "end": 10_000}})
    assert clipped["data"]["span"]["end"] == total
    assert clipped["data"]["memory"]["content"] in content
    for bad in ({"start": -1, "end": 5}, {"start": 5, "end": 5}, {"start": total + 5, "end": total + 9}):
        rejected = tools.memory("read", {"memory_id": a["id"], "span": bad})
        assert rejected["ok"] is False
    not_dict = tools.memory("read", {"memory_id": a["id"], "span": [0, 5]})
    assert not_dict["ok"] is False

    # The clue's deep_read spans round-trip: reading each span returns the
    # triggering numeric text.
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10, "include_quotes": True})
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    assert "numeric_value_candidate" in clue["reasons"]
    for side in ("left", "right"):
        call = clue["deep_read"][side]
        assert call["memory_id"] in (a["id"], b["id"])
        window = tools.memory("read", {"memory_id": call["memory_id"], "span": call["span"]})
        assert window["ok"] is True
        assert "重试次数为" in window["data"]["memory"]["content"]


def test_notice_snapshot_carries_evidence_spans(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_notice(tools, a["id"], b["id"], complete_slot=True)
    notice = tools.memory_repair("notice", {"action": "read", "notice_id": notice_id})["data"]["notice"]
    for member in notice["member_versions"]:
        assert member["evidence_span"][0] < member["evidence_span"][1]
        window = tools.memory("read", {
            "memory_id": member["memory_id"],
            "span": {"start": member["evidence_span"][0], "end": member["evidence_span"][1]},
        })
        assert window["ok"] is True
        assert "接口超时" in window["data"]["memory"]["content"]


def test_deep_read_spans_follow_clue_upgrade_and_drifted_offsets(tmp_path: Path) -> None:
    filler = "这条是中性填充内容，用来把两个区域拉开超过窗口宽度。" * 8  # ~360 chars
    # Empty lines keep the filler and the numeric value as separate text
    # units, so the clue upgrades from a similarity check to a numeric
    # notify across two distinct regions.
    content_a = f"部署架构完全相同的描述文字甲。\n\n{filler}\n\n重试次数为 3 次。"
    content_b = f"部署架构完全相同的描述文字甲。\n\n{filler}\n\n重试次数为 5 次。"
    tools = make_tools(tmp_path)
    a = tools.memory_write(content=content_a, subject="upg", tags=[])["data"]
    b = tools.memory_write(content=content_b, subject="upg", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)

    scan = tools.memory_repair(
        "scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10, "include_check": True, "include_quotes": True},
    )
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    assert clue["route"] == "review_candidate"
    assert "numeric_value_candidate" in clue["reasons"]
    # The deep-read span must track the UPGRADED (numeric) discovery, not
    # the first (similarity) one — and re-reading it must contain the text.
    for side in ("left", "right"):
        call = clue["deep_read"][side]
        window = tools.memory("read", {"memory_id": call["memory_id"], "span": call["span"]})
        assert window["ok"] is True
        assert "重试次数为" in window["data"]["memory"]["content"]


def test_deep_read_survives_offset_drift_on_dense_content(tmp_path: Path) -> None:
    # Semicolon-dense repeated-token lines make stored unit offsets drift;
    # deep_read must locate the trigger by text, not trust the offsets.
    def dense(value: str) -> str:
        return "\n".join([
            f"pool.max=50; pool.timeout={value}; pool.min_idle=5;",
            "cache.ttl=60; cache.size=1000; cache.backend=memory;",
            f"retry.count=3; retry.backoff=200; retry.value={value};",
            "log.level=info; log.file=/var/log/app.log; log.rotation=daily;",
            "queue.depth=100; queue.workers=8; queue.strategy=fifo;",
            f"rate.limit=1000; rate.burst=50; rate.window={value};",
        ])

    tools = make_tools(tmp_path)
    a = tools.memory_write(content=dense("30"), subject="cfg", tags=[])["data"]
    b = tools.memory_write(content=dense("90"), subject="cfg", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    scan = tools.memory_repair("scan_candidates", {"anchor_memory_id": 0, "batch": 20, "k": 10, "include_quotes": True})
    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
    clue = next(c for c in scan["data"]["candidates"] if (c["left_id"], c["right_id"]) == pair)
    for side in ("left", "right"):
        call = clue["deep_read"][side]
        assert call["span"] is not None
        window = tools.memory("read", {"memory_id": call["memory_id"], "span": call["span"]})
        assert window["ok"] is True
        assert "timeout=30" in window["data"]["memory"]["content"] or "timeout=90" in window["data"]["memory"]["content"]


def test_notice_snapshot_is_frozen_when_member_drifts(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_notice(tools, a["id"], b["id"], complete_slot=True)
    before = tools.db.read_semantic_notice(notice_id)
    original_members = before["member_versions"]

    tools.memory("update", {"memory_id": a["id"], "new_content": "短。", "reason": "缩短"})
    after = tools.db.read_semantic_notice(notice_id)
    assert after["freshness"]["fresh"] is False
    assert after["member_versions"] == original_members
    assert after["member_versions"][0]["version"] == 1


def _assert_exact_text_unit_spans(subject: str, content: str) -> None:
    from memory_arbiter.evidence import _clean

    for unit in local_text_units(subject, content):
        if unit.kind != "text":
            continue
        assert 0 <= unit.start_offset < unit.end_offset <= len(content)
        restored = _clean(content[unit.start_offset:unit.end_offset])
        assert restored == unit.text, (
            unit.unit_index, unit.start_offset, unit.end_offset,
            unit.text, restored,
        )


def test_evidence_offsets_exactly_map_normalized_source_text() -> None:
    # Regression: repeated tokens + semicolon-dense multiline config used to
    # make the reverse locator overshoot by hundreds of chars and cascade.
    dense = "\n".join([
        "pool.max=50; pool.timeout=30; pool.min_idle=5;",
        "cache.ttl=60; cache.size=1000; cache.backend=memory;",
        "retry.count=3; retry.backoff=200; retry.jitter=50;",
        "log.level=info; log.file=/var/log/app.log; log.rotation=daily;",
        "queue.depth=100; queue.workers=8; queue.strategy=fifo;",
        "rate.limit=1000; rate.burst=50; rate.window=1s;",
    ])
    _assert_exact_text_unit_spans("service config", dense)

    # Cross-line whitespace, CRLF, repeated sentence text, headings, short
    # paragraph merge, and long overlapping windows.
    corpus = [
        "重复句子。重复句子。值为 3。重复句子。值为 5。",
        "第一行  多空格\r\n第二行\t制表符\r\n\r\n尾段。",
        "# 标题 2025\n正文一。\n\n短。\n相邻长段落用于合并验证。",
        ("超长无标点文本abc重复词" * 80),
        "A; B; C; A; B; C; A; B; C; 数值=10; 数值=20;",
    ]
    for content in corpus:
        _assert_exact_text_unit_spans("subject", content)


def test_evidence_offset_mapping_deterministic_fuzz() -> None:
    import random

    random.seed(20260820)
    tokens = [
        "alpha", "beta", "key=30;", "key=90;", "重复", "接口", "超时",
        "。", "；", ";", "!", "?", "  ", "\t", "\n", "\r\n",
    ]
    for _ in range(1000):
        content = "".join(random.choice(tokens) for _ in range(random.randint(1, 80)))
        if random.random() < 0.25:
            content = "# 标题 2025\n" + content
        _assert_exact_text_unit_spans("主题" if random.random() < 0.5 else "", content)


def test_short_paragraph_merge_never_crosses_heading_barrier() -> None:
    from memory_arbiter.evidence import _clean

    content = "短。\n# Runtime Heading\n后面的正文足够长用于单独索引。"
    units = local_text_units("", content)
    headings = [u for u in units if u.kind == "heading"]
    texts = [u for u in units if u.kind == "text"]
    assert [u.text for u in headings] == ["Runtime Heading"]
    assert texts
    assert all("Runtime Heading" not in u.text and "#" not in u.text for u in texts)
    for unit in texts:
        assert _clean(content[unit.start_offset:unit.end_offset]) == unit.text

    # Symmetric case: long paragraph before the heading, short one after it.
    content2 = "前面的正文足够长用于单独索引。\n# Boundary\n短。"
    units2 = local_text_units("", content2)
    text_units2 = [u for u in units2 if u.kind == "text"]
    assert all("Boundary" not in u.text and "#" not in u.text for u in text_units2)
    for unit in text_units2:
        assert _clean(content2[unit.start_offset:unit.end_offset]) == unit.text


def test_embedding_pipeline_version_rotated_for_exact_offsets() -> None:
    from memory_arbiter.embedder import EMBEDDING_PIPELINE_VERSION

    # 0.17.0 终局（owner 2026-09-26）：存储侧裸文本=已发版 v2 语义，版本钉 2
    # 不轮换——升级用户零全量重嵌；查询侧前缀是运行时行为。前缀值变更不进
    # space_id 组成，未来改存储侧嵌入文本必须手动 bump。
    assert EMBEDDING_PIPELINE_VERSION == 2


# ── 2026-08-21 review round: notice-pipeline and scan wide-gate fixes ────────