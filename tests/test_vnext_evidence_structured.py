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

# ── test_vnext_evidence unified notice/structured/wave8 domain (commit 12 split) ──



def test_unified_conflicts_notice_columns_and_api(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_notice(tools, left["id"], right["id"], complete_slot=True)

    with tools.db.connection() as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantic_notices'"
        ).fetchone() is None
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(conflicts)")}
    assert {
        "notice_type", "notice_payload", "notice_task_id", "notice_dedupe_key",
        "notice_delivery_status", "notice_slot_provenance",
    } <= columns

    listed = tools.memory_repair("notice", {"action": "list", "status": "open"})
    assert listed["ok"] is True
    assert notice_id in {item["notice_id"] for item in listed["data"]["notices"]}
    read = tools.memory_repair("notice", {"action": "read", "notice_id": notice_id})
    notice = read["data"]["notice"]
    assert notice["conflict_id"] == notice_id
    assert notice["payload"]["slot_key"]["entity"] == "checkout-api"
    assert len(notice["member_versions"]) == 2


def test_notice_escalation_requires_structured_slot_snapshot(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    incomplete_id = _seed_notice(tools, left["id"], right["id"], complete_slot=False)

    rejected = tools.memory_repair("notice", {"action": "escalate", "notice_id": incomplete_id})
    assert rejected["ok"] is False
    assert rejected["data"]["outcome"] == "escalate_failed"
    assert rejected["data"]["detail"]["outcome"] == "structured_group_required"

    complete_id = _seed_notice(tools, left["id"], right["id"], complete_slot=True)
    notice = tools.memory_repair("notice", {"action": "read", "notice_id": complete_id})["data"]["notice"]
    assert notice["slot_key"] == {
        "attribute": "接口超时", "entity": "checkout-api", "scope": "production",
    }
    assert notice["notice_slot_provenance"]["entity"] == "metadata"
    escalated = tools.memory_repair(
        "notice", {"action": "escalate", "notice_id": complete_id, "reason": "verified"},
    )
    assert escalated["ok"] is True
    assert escalated["data"]["conflict_id"] == complete_id
    detail = tools.memory_review(
        "conflict_detail", {"conflict_id": complete_id},
    )["data"]["conflict"]
    assert detail["status"] == "open"
    assert detail["source"] == "semantic_notice"
    assert detail["notice_delivery_status"] == "resolved"
    assert detail["slot_key"]["entity"] == "checkout-api"
    assert {group["normalized_value"] for group in detail["value_groups"]} == {
        normalize_value("5 秒"), normalize_value("30 秒"),
    }


def test_notice_freshness_and_terminal_lifecycle_use_api(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    notice_id = _seed_notice(tools, left["id"], right["id"], complete_slot=True)
    assert tools.db.read_semantic_notice(notice_id)["freshness"]["fresh"] is True

    tools.memory("update", {"memory_id": left["id"], "new_content": "接口超时为 10 秒。", "reason": "修订"})
    assert tools.db.read_semantic_notice(notice_id)["freshness"]["fresh"] is False
    stale = tools.memory_repair("notice", {"action": "escalate", "notice_id": notice_id})
    assert stale["ok"] is False and stale["data"]["outcome"] == "stale_notice"

    fresh_id = _seed_notice(tools, left["id"], right["id"], left_version=2, complete_slot=True)
    dismissed = tools.memory_repair(
        "notice", {"action": "dismiss", "notice_id": fresh_id, "reason": "可共存"},
    )
    assert dismissed["ok"] is True
    assert tools.db.read_semantic_notice(fresh_id)["status"] == "dismissed"


def test_structured_record_conflict_group_dedupes_and_rejects_pair_payload(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    payload = _structured_conflict_payload(left["id"], right["id"])

    inserted = tools.memory_repair("record_conflict", payload)
    assert inserted["ok"] is True and inserted["data"]["outcome"] == "inserted"
    deduped = tools.memory_repair("record_conflict", payload)
    assert deduped["ok"] is True and deduped["data"]["outcome"] == "deduped"
    assert deduped["data"]["conflict_id"] == inserted["data"]["conflict_id"]

    pair = tools.memory_repair(
        "record_conflict", {"left_id": left["id"], "right_id": right["id"], "reason": "legacy"},
    )
    assert pair["ok"] is False
    assert any("left_id" in warning for warning in pair.get("warnings", []))


def test_structured_record_conflict_validates_slot_members_and_revision(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    payload = _structured_conflict_payload(left["id"], right["id"])
    inserted = tools.memory_repair("record_conflict", payload)
    assert inserted["ok"] is True

    third = tools.memory_write(content="接口超时为 60 秒。", subject="timeout", tags=[])["data"]
    third_member = _conflict_member(third["id"], "60秒", "接口超时为 60 秒。")
    appended_payload = {
        **payload,
        "members": payload["members"] + [third_member],
        "value_groups": payload["value_groups"] + [
            ConflictValueGroup(normalize_value("60 秒"), "60 秒", (f"{third['id']}@1",)).to_dict(),
        ],
        "expected_revision": 1,
    }
    appended = tools.memory_repair("record_conflict", appended_payload)
    assert appended["ok"] is True and appended["data"]["outcome"] == "appended"
    stale = tools.memory_repair("record_conflict", appended_payload)
    assert stale["ok"] is False and stale["data"]["outcome"] == "stale_conflict"


def test_judge_uses_group_revision_and_apply_plan(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    left = tools.memory_write(content="接口超时为 5 秒。", subject="timeout", tags=[])["data"]
    right = tools.memory_write(content="接口超时为 30 秒。", subject="timeout", tags=[])["data"]
    conflict_id = tools.memory_repair(
        "record_conflict", _structured_conflict_payload(left["id"], right["id"]),
    )["data"]["conflict_id"]
    judged = tools.memory("judge", {
        "conflict_id": conflict_id,
        "expected_revision": 1,
        "chosen_value": normalize_value("30 秒"),
        "decided_by": "user",
        "ref": "test",
        "reason": "用户确认生产超时",
        "authorized": True,
        "resolution_memory_id": right["id"],
        "apply_plan": [
            {"memory_id": left["id"], "action": "update_current_claim"},
            {"memory_id": right["id"], "action": "use_as_resolution"},
        ],
    })
    assert judged["ok"] is True
    assert judged["data"]["revision"] == 2
    assert judged["data"]["next_action"]["action"] == "apply_conflict_action"
    legacy = tools.memory("judge", {
        "conflict_id": conflict_id,
        "expected_left_version": 1,
        "expected_right_version": 1,
        "verdict": "contradiction",
        "reason": "legacy",
    })
    assert legacy["ok"] is False


def test_wave8_regression_basics(tmp_path: Path) -> None:
    from memory_arbiter.evidence import local_text_units

    content = "\n".join([
        "pool.max=50; pool.timeout=30; pool.min_idle=5;",
        "cache.ttl=60; cache.size=1000; cache.backend=memory;",
        "retry.count=3; retry.backoff=200; retry.jitter=50;",
        "log.level=info; log.file=/var/log/app.log; log.rotation=daily;",
        "queue.depth=100; queue.workers=8; queue.strategy=fifo;",
        "rate.limit=1000; rate.burst=50; rate.window=1s;",
    ])
    units = local_text_units("cfg", content)
    assert units
    assert all(u.start_offset < u.end_offset or u.kind == "subject" for u in units)

    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="s", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="s2", tags=[])["data"]
    huge = "x" * 5000
    payload = _structured_conflict_payload(a["id"], b["id"])
    rejected = tools.memory_repair("record_conflict", {**payload, "reason": huge})
    assert rejected["ok"] is False
    assert rejected["data"]["error"] == "invalid_input"

    bogus = tools.memory_govern(
        "resolve_conflict", {"conflict_id": 99999, "reason": "x", "authorized": True},
    )
    assert bogus["ok"] is False
    assert bogus["data"]["outcome"] == "invalid_input"
    assert bogus["data"]["field"] == "expected_revision"

    ids = [tools.memory_write(content=f"修复 {i}。", subject=f"fix{i}", tags=[])["data"]["id"] for i in range(4)]
    result = tools.memory_repair(
        "rebuild_evidence", {"dry_run": False, "memory_ids": ids, "batch_size": 2},
    )
    assert result["data"]["queued"] == 4


def test_judge_group_schema_rejects_legacy_fields_and_bad_types(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="接口超时为 5 秒。", subject="t", tags=[])["data"]
    b = tools.memory_write(content="接口超时为 30 秒。", subject="t", tags=[])["data"]
    conflict_id = tools.memory_repair(
        "record_conflict", _structured_conflict_payload(a["id"], b["id"]),
    )["data"]["conflict_id"]
    base = {
        "conflict_id": conflict_id, "expected_revision": 1, "chosen_value": "30秒",
        "decided_by": "user", "ref": "test", "reason": "两说", "authorized": True,
        "resolution_memory_id": b["id"],
        "apply_plan": [{"memory_id": a["id"], "action": "update_current_claim"}],
    }
    assert tools.memory("judge", {**base, "expected_revision": "banana"})["ok"] is False
    assert tools.memory("judge", {**base, "decided_by": "model"})["ok"] is False
    legacy = tools.memory("judge", {
        "conflict_id": conflict_id, "expected_left_version": 1,
        "expected_right_version": 1, "verdict": "contradiction", "reason": "legacy",
    })
    assert legacy["ok"] is False
    assert tools.memory("judge", base)["ok"] is True


def test_wave8_malformed_tags_do_not_zero_filter_recall(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)
    healthy = tools.memory_write(content="健康条目。", subject="h", tags=["alpha"])["data"]
    tools.memory_write(content="另一条。", subject="h2", tags=[])
    with tools.db.write_transaction() as conn:
        conn.execute("UPDATE memories SET tags='not-json{' WHERE subject='h2'")
    result = tools.memory("find", {"tags_filter": ["alpha"]})
    assert healthy["id"] in [r["id"] for r in result["data"]["results"]]
    assert result["data"]["total_estimate"] >= 1


def test_wave8_new_database_permissions(tmp_path: Path) -> None:
    import os as _os
    tools = make_tools(tmp_path)
    mode = _os.stat(tools.settings.db_path).st_mode & 0o777
    assert mode == 0o600


def test_wave8_doctor_reports_attention_volume(tmp_path: Path) -> None:
    import json as _json

    tools = make_tools(tmp_path)
    a = tools.memory_write(content="限流 10。", subject="r", tags=[])["data"]
    b = tools.memory_write(content="限流 99。", subject="r", tags=[])["data"]
    reg = tools.memory_repair(
        "record_conflict", _structured_conflict_payload(a["id"], b["id"]),
    )
    assert reg["ok"] is True
    log_path = tools.db.settings.db_path.parent / "attention_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(_json.dumps({"ts": "2026-01-01T00:00:00+00:00", "trigger": "search", "source": "open_table", "ids": [1]}) + "\n")
    report = tools.memory_review("doctor", {})["data"]
    finding = next(f for f in report["findings"] if f["check_id"] == "capacity.attention_volume")
    assert finding["status"] == "pass"
    assert finding["evidence"]["total_lines"] == 1

