"""0.16.0 batch read tests (plan §1.5/§6⑨/§6⑭/§6⑰): three-mode unification,
per-mode caps, unit-aligned span hits, over-budget structured response, ACL,
and read's backward-compatible content_mode default."""
from __future__ import annotations

import json
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    return MemoryTools(Settings(db_path=tmp_path / "m.db", backup_jsonl=tmp_path / "b.jsonl"))


def _write(tools: MemoryTools, subject: str, content: str, workspace: str = "w") -> int:
    res = tools.memory_write(
        content=content, subject=subject, workspace=workspace,
        source_type="agent_generated", tags=[],
    )
    assert res.get("ok"), res
    return int(res["data"]["id"])


def _batch_read(tools: MemoryTools, **payload):
    return tools.memory(action="batch_read", data=payload)


def _drain_evidence(tools: MemoryTools) -> None:
    assert tools.wait_evidence_worker_drained(timeout=30.0)


def _insert_units(tools: MemoryTools, memory_id: int, units: list[tuple[str, int, int]]) -> None:
    """Insert evidence units directly (tests run without an embedder, so the
    async index never publishes rows on its own)."""
    import hashlib

    from memory_arbiter.models import utc_now_iso

    memory = tools.db.get_memory(memory_id)
    content_hash = hashlib.sha256(str(memory["content"]).encode()).hexdigest()
    with tools.db.write_transaction() as conn:
        for index, (text, start, end) in enumerate(units):
            conn.execute(
                """INSERT INTO memory_evidence(
                     memory_id,memory_version,content_hash,unit_index,kind,text,
                     start_offset,end_offset,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (memory_id, int(memory["version"]), content_hash, index, "text",
                 text, start, end, utc_now_iso()),
            )


# ── validation / caps ───────────────────────────────────────────────────────

def test_batch_read_requires_ids(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    res = _batch_read(tools)
    assert not res.get("ok")
    res = _batch_read(tools, memory_ids=[])
    assert not res.get("ok")


def test_batch_read_rejects_unknown_content_mode(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "甲主题", "甲内容")
    res = _batch_read(tools, memory_ids=[mid], content_mode="span")
    assert not res.get("ok")
    assert "content_mode" in str(res["data"])


def test_batch_read_caps_per_mode(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    ids = [_write(tools, f"主题{i}", f"内容{i}") for i in range(12)]
    # full cap = 10
    res = _batch_read(tools, memory_ids=ids, content_mode="full")
    assert not res.get("ok")
    assert "at most 10" in str(res["data"])
    # preview cap = 50
    fifty_one = ids + [10**9] * 39 + [10**9 + 1, 10**9 + 2]
    res = _batch_read(tools, memory_ids=fifty_one, content_mode="preview")
    assert not res.get("ok")
    assert "at most 50" in str(res["data"])
    # exactly at the cap is fine
    res = _batch_read(tools, memory_ids=ids[:10], content_mode="full")
    assert res.get("ok"), res


def test_batch_read_deduplicates_ids(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "去重主题", "去重内容")
    res = _batch_read(tools, memory_ids=[mid, mid, mid])
    assert res.get("ok"), res
    assert res["data"]["count"] == 1


# ── three modes ─────────────────────────────────────────────────────────────

def test_batch_read_preview_carries_outline_no_content(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "预览主题", "预览正文内容")
    res = _batch_read(tools, memory_ids=[mid])
    assert res.get("ok"), res
    item = res["data"]["results"][0]
    assert item["found"] is True
    assert "content" not in item["memory"]
    assert item["memory"]["content_chars"] == len("预览正文内容")
    assert "outline" in item["memory"]


def test_batch_read_full_returns_content(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "全文主题", "全文正文内容")
    res = _batch_read(tools, memory_ids=[mid], content_mode="full")
    assert res.get("ok"), res
    item = res["data"]["results"][0]
    assert item["memory"]["content"] == "全文正文内容"
    assert "size" in res["data"]


def test_batch_read_not_found_is_per_item(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "存在主题", "存在内容")
    res = _batch_read(tools, memory_ids=[mid, 999999])
    assert res.get("ok"), res
    by_id = {item["memory_id"]: item for item in res["data"]["results"]}
    assert by_id[mid]["found"] is True
    assert by_id[999999]["found"] is False
    assert by_id[999999]["error"] == "not_found"


def test_batch_read_hits_unit_aligned_zero_truncation(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    body = "第一段落讲述催收规范。第二段落记录债务转移流程。第三段落是别的主题。"
    mid = _write(tools, "单元对齐主题", body)
    _insert_units(tools, mid, [
        ("第一段落讲述催收规范。", 0, body.index("第二段落")),
        ("第二段落记录债务转移流程。", body.index("第二段落"), body.index("第三段落")),
        ("第三段落是别的主题。", body.index("第三段落"), len(body)),
    ])
    # Pick a span that starts mid-unit; the response must still carry the
    # COMPLETE unit text, never a half-sentence slice.
    span_start = body.index("第二段落") + 2
    res = _batch_read(
        tools, memory_ids=[mid], content_mode="hits",
        spans={str(mid): {"start": span_start, "end": span_start + 5}},
    )
    assert res.get("ok"), res
    record = res["data"]["results"][0]["memory"]
    spans = record.get("hit_spans") or []
    assert spans, "expected unit-aligned hit spans"
    for span in spans:
        assert span["text"] in body
        assert body[span["start_offset"]:span["end_offset"]] == span["text"]


def test_batch_read_hits_without_evidence_falls_back_full(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "无索引主题", "无索引正文")
    # No evidence drain: rows may not exist yet; hits degrades to the full
    # record instead of returning an empty page.
    res = _batch_read(tools, memory_ids=[mid], content_mode="hits")
    assert res.get("ok"), res
    item = res["data"]["results"][0]
    assert item["found"] is True


# ── over-budget (full mode) ────────────────────────────────────────────────

def test_batch_read_full_over_budget_returns_structured_prompt(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    big = "长" * 45_000  # ~135KB UTF-8 across two items > 80KB budget
    a = _write(tools, "超长甲", big)
    b = _write(tools, "超长乙", big)
    res = _batch_read(tools, memory_ids=[a, b], content_mode="full")
    assert res.get("ok"), res
    data = res["data"]
    assert data["over_budget"] is True
    assert data["total_bytes"] > data["budget_bytes"]
    for item in data["results"]:
        assert "content" not in item["memory"]
        assert item["memory"]["content_chars"] > 0
    assert "individually" in data["hint"]


def test_batch_read_full_single_item_over_hard_cap(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    big = "巨" * 40_000  # ~120KB single item > 100KB hard ceiling
    mid = _write(tools, "单条超长", big)
    res = _batch_read(tools, memory_ids=[mid], content_mode="full")
    assert res.get("ok"), res
    assert res["data"]["over_budget"] is True


# ── read backward compatibility ────────────────────────────────────────────

def test_read_default_full_unchanged(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "兼容主题", "兼容正文")
    res = tools.memory(action="read", data={"memory_id": mid})
    assert res.get("ok"), res
    assert res["data"]["memory"]["content"] == "兼容正文"


def test_read_preview_mode(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "预览读主题", "预览读正文")
    res = tools.memory(action="read", data={"memory_id": mid, "content_mode": "preview"})
    assert res.get("ok"), res
    assert "content" not in res["data"]["memory"]
    assert res["data"]["memory"]["content_chars"] == len("预览读正文")


def test_read_span_unit_aligned_with_evidence(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    body = "第一段说甲规则。第二段说乙规则。"
    mid = _write(tools, "窗口主题", body)
    _insert_units(tools, mid, [
        ("第一段说甲规则。", 0, body.index("第二段")),
        ("第二段说乙规则。", body.index("第二段"), len(body)),
    ])
    res = tools.memory(action="read", data={
        "memory_id": mid,
        "span": {"start": body.index("第二段"), "end": body.index("第二段") + 3},
    })
    assert res.get("ok"), res
    memory = res["data"]["memory"]
    span_meta = res["data"]["span"]
    # Unit-aligned: the covered unit comes back whole, verbatim in the body.
    assert memory["content"] in body
    assert span_meta["unit_aligned"] is True
    assert span_meta["units"] >= 1


def test_read_span_legacy_fallback_without_evidence(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    body = "第一段说甲规则。第二段说乙规则。"
    mid = _write(tools, "旧窗口主题", body)
    res = tools.memory(action="read", data={
        "memory_id": mid,
        "span": {"start": 0, "end": 5},
    })
    assert res.get("ok"), res
    memory = res["data"]["memory"]
    assert memory["content"] == body[:5]
    assert "unit_aligned" not in res["data"]["span"]


def test_batch_read_rejects_bad_span_shape(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "坏窗口主题", "坏窗口正文")
    res = _batch_read(
        tools, memory_ids=[mid], content_mode="hits",
        spans={str(mid): {"start": 10, "end": 2}},
    )
    assert not res.get("ok")
    res = _batch_read(tools, memory_ids=[mid], spans={"not-an-id": {"start": 0, "end": 2}})
    assert not res.get("ok")


def test_batch_read_validation_registry_accepts_fields(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    mid = _write(tools, "注册表主题", "注册表正文")
    res = _batch_read(tools, memory_ids=[mid], content_mode="preview", workspace="w")
    assert res.get("ok"), res
    # unknown fields are stripped with a warning, not fatal
    res = _batch_read(tools, memory_ids=[mid], not_a_field=1)
    assert res.get("ok"), res
