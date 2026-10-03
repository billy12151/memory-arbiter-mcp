"""D1 双通道 intake（0.17.1，方案 §1.2/§1.6，owner 选项 1）。

钉死的契约：
- D1 校验（conflicts._normalize_members）二选一放行：normalized_value ==
  normalize_value(value_raw)（机械规整形态）或 normalized_value ==
  value_raw（§3.4 快照原文形态——判定型 notice 值=行文本）；转述（两不靠）
  仍拒——D1 只校验两字段关系，不校验内容；
- 判定型 notice escalate 端到端：真判定形态播种（值=行文本、slot attribute
  =12-hex pair anchor）→ escalate 升 open 组、成员值=行文本、notice 原地
  candidate→open（notice_delivery_status=resolved）；
- promote-in-place 照抄判定 notice 成套快照（slot_key+member_versions+
  value_groups+detector_version）→ deduped 且原行升 open；转述提交仍拒；
- 防回归：agent 自由建组（memory_repair record_conflict）的转述成员仍被
  通道一拒绝；value_raw=None / 空串豁免不变。
"""
from __future__ import annotations

from typing import Any

import pytest

from memory_arbiter.db.conflicts import ConflictStore
from memory_arbiter.pipeline.evidence import _conflict_notice_payload, _retired_gate_slot_key
from memory_arbiter.semantic_conflict import normalize_value

import tests.test_vnext_evidence as tv


# ── 单元层：_normalize_members 双通道 ────────────────────────────────────────

def _member(**overrides: Any) -> dict[str, Any]:
    base = {
        "memory_id": 12, "version": 1, "attribute_raw": "timeout",
        "value_raw": "5000ms", "normalized_attribute": "timeout",
        "normalized_value": normalize_value("5000ms"),
        "evidence_quote": "timeout 5000ms", "evidence_span": [0, 14],
        "content_hash": "0" * 64, "direction": "a_to_b",
        "prompt_version": "p1", "detector_version": "d1",
    }
    base.update(overrides)
    return base


def test_d1_accepts_mechanical_form() -> None:
    normalized = ConflictStore._normalize_members([
        _member(), _member(memory_id=34, value_raw="30 秒", normalized_value=normalize_value("30 秒")),
    ])
    assert normalized[0]["normalized_value"] == normalize_value("5000ms")


def test_d1_accepts_snapshot_verbatim_form() -> None:
    row_text = "接口超时为 5 秒。"
    normalized = ConflictStore._normalize_members([
        _member(value_raw=row_text, normalized_value=row_text),
        _member(memory_id=34, value_raw="接口超时为 30 秒。", normalized_value="接口超时为 30 秒。"),
    ])
    assert normalized[0]["normalized_value"] == row_text


def test_d1_rejects_paraphrase() -> None:
    raw = "老版本 MySQL 数据库"
    paraphrase = "mysql"
    assert paraphrase != raw and paraphrase != normalize_value(raw)
    with pytest.raises(ValueError, match="value_raw verbatim"):
        ConflictStore._normalize_members([_member(value_raw=raw, normalized_value=paraphrase)])


def test_d1_none_and_empty_value_raw_still_exempt() -> None:
    normalized = ConflictStore._normalize_members([
        _member(value_raw=None, normalized_value=""),
        _member(memory_id=34, value_raw="", normalized_value="whatever"),
    ])
    assert normalized[0]["normalized_value"] == ""
    assert normalized[1]["normalized_value"] == "whatever"


# ── 端到端：判定型 notice 播种 → escalate ────────────────────────────────────

_JUDGED_ANCHOR = "a3f9c2d41b76"  # 12-hex pair-diff anchor 形态（§3.4）


def _seed_judged_notice(tools: tv.MemoryTools, left_id: int, right_id: int) -> int:
    """按 A-cross 判定型 notice 的真实组装形态播种（evidence._land_dispatch_notice
    → _conflict_notice_payload：值=两侧行文本、member_versions 为轻量 value 形态、
    slot attribute=12-hex anchor）。"""
    left_text = "接口超时为 5 秒。"
    right_text = "接口超时为 30 秒。"
    payload = _conflict_notice_payload(
        reason="model_classified_conflict",
        attribute=_JUDGED_ANCHOR,
        slot_key=_retired_gate_slot_key("default", _JUDGED_ANCHOR, "timeout"),
        left_id=left_id, left_version=1,
        left_value_norm=left_text, left_display=left_text, left_quote=left_text,
        right_id=right_id, right_version=1,
        right_value_norm=right_text, right_display=right_text, right_quote=right_text,
        left_content=left_text, right_content=right_text,
        extra={"model_signal": {"label": "conflict", "probs": {
            "conflict": 0.9, "no_conflict": 0.05, "possible_conflict": 0.05,
        }, "mechanism": None, "model_version": "mdeberta:test"}},
    )
    created = tools.db.record_semantic_notice(
        memory_id=left_id, peer_id=right_id, severity="normal",
        notice_type="semantic_evidence",
        title=f"Possible memory change with #{right_id}",
        message="model_classified_conflict",
        payload=payload,
        dedupe_key=f"judged:{left_id}:{right_id}",
        left_version=1, right_version=1,
    )
    return int(created["notice_id"])


def test_judged_notice_escalates_to_open_group(tmp_path) -> None:
    tools = tv.make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_judged_notice(tools, left["id"], right["id"])

    escalated = tools.memory_repair(
        "notice", {"action": "escalate", "notice_id": notice_id, "reason": "verified"},
    )
    assert escalated["ok"] is True, escalated
    detail = tools.memory_review("conflict_detail", {"conflict_id": notice_id})["data"]["conflict"]
    assert detail["status"] == "open"
    assert detail["notice_delivery_status"] == "resolved"
    # 成员值 = 行文本原文（§3.4 quote 即值），不是机械规整值
    member_values = {str(m["normalized_value"]) for m in detail["member_versions"]}
    assert member_values == {"接口超时为 5 秒。", "接口超时为 30 秒。"}


def test_judged_notice_promote_via_verbatim_snapshot_copy(tmp_path) -> None:
    tools = tv.make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_judged_notice(tools, left["id"], right["id"])

    notice = tools.memory_repair("notice", {"action": "read", "notice_id": notice_id})["data"]["notice"]
    # 快照全套在 notice 顶层字段（payload.member_versions 是轻量 value 形态）
    submitted = {
        "slot_key": dict(notice["slot_key"]),
        "members": [dict(m) for m in notice["member_versions"]],
        "value_groups": [dict(g) for g in notice["value_groups"]],
        "status": "open",
        "detector_version": notice["detector_version"],
        "source": "scheduled_scan",
        "reason": "agent triage confirms the judged contradiction",
    }
    promoted = tools.memory_repair("record_conflict", submitted)
    assert promoted["ok"] is True, promoted
    assert promoted["data"]["outcome"] == "deduped"
    detail = tools.memory_review("conflict_detail", {"conflict_id": notice_id})["data"]["conflict"]
    assert detail["status"] == "open"


def test_promote_paraphrased_copy_still_rejected(tmp_path) -> None:
    tools = tv.make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_judged_notice(tools, left["id"], right["id"])

    notice = tools.memory_repair("notice", {"action": "read", "notice_id": notice_id})["data"]["notice"]
    members = [dict(m) for m in notice["member_versions"]]
    raw = str(members[0]["value_raw"])
    paraphrase = "五秒超时"
    assert paraphrase != raw and paraphrase != normalize_value(raw)
    members[0]["normalized_value"] = paraphrase
    submitted = {
        "slot_key": dict(notice["slot_key"]),
        "members": members,
        "value_groups": [dict(g) for g in notice["value_groups"]],
        "status": "open",
        "detector_version": notice["detector_version"],
        "source": "scheduled_scan",
        "reason": "paraphrased retelling must stay rejected",
    }
    rejected = tools.memory_repair("record_conflict", submitted)
    assert rejected["ok"] is False
    assert "value_raw verbatim" in str(rejected)


def test_record_conflict_paraphrase_still_rejected_on_free_intake(tmp_path) -> None:
    """防回归：自由 intake 场景转述仍拒（D1 通道一仍咬；owner 选项 1 只放宽
    verbatim 自洽形态）。"""
    tools = tv.make_tools(tmp_path)
    left = tools.memory_write(content="数据库是 MySQL。", subject="db", tags=[])["data"]
    right = tools.memory_write(content="数据库是 SQLite。", subject="db", tags=[])["data"]
    raw = "老版本 MySQL 数据库"
    paraphrase = "mysql"
    assert paraphrase != raw and paraphrase != normalize_value(raw)
    submitted = {
        "slot_key": {"entity": "project-x", "attribute": "database", "scope": "production"},
        "members": [
            tv.ConflictMember(
                memory_id=left["id"], version=1, attribute_raw="database",
                value_raw=raw, normalized_attribute="database",
                normalized_value=paraphrase, evidence_quote="数据库是 MySQL。",
                evidence_span=[0, 10], content_hash="0" * 64, direction="a_to_b",
                prompt_version="p1", detector_version="d1",
            ).to_dict(),
            {
                "memory_id": right["id"], "version": 1, "attribute_raw": "database",
                "value_raw": "SQLite", "normalized_attribute": "database",
                "normalized_value": normalize_value("SQLite"), "evidence_quote": "数据库是 SQLite。",
                "evidence_span": [0, 11], "content_hash": "1" * 64, "direction": "b_to_a",
                "prompt_version": "p1", "detector_version": "d1",
            },
        ],
        "value_groups": [
            {"normalized_value": paraphrase, "display_value": raw, "members": [f"{left['id']}@1"]},
            {"normalized_value": normalize_value("SQLite"), "display_value": "SQLite", "members": [f"{right['id']}@1"]},
        ],
        "status": "open",
        "detector_version": "d1",
        "prompt_version": "p1",
        "source": "scheduled_scan",
        "reason": "paraphrase must be rejected at intake",
    }
    rejected = tools.memory_repair("record_conflict", submitted)
    assert rejected["ok"] is False
    assert "value_raw verbatim" in str(rejected)
