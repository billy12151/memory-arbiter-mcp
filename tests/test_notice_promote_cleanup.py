"""0.17.1 修复批 A2：promote 后正式冲突组可被 notice dismiss 撤回（修复钉）.

背景（产品面端到端复现）：record_conflict(status=open) 命中同一 notice 快照
时走 conflicts.py 的 promote-in-place 分支，此前只置 status='open'、不清
notice_delivery_status → 该行仍出现在 notice list(open)，且 notice dismiss
会把它改写回 not_a_conflict（正式冲突组被 notice 通道静默撤回，绕过
judge/apply 治理链）。escalate 三处本就写 'resolved'，promote 漏了。

钉死契约：
- promote 后行 notice_delivery_status='resolved' + resolution reason 前缀；
- notice list(open) 不再含该 id；
- notice dismiss 返回 already_terminal（核心钉）；
- bare candidate（无 notice_type）promote 路径不受影响；
- 第二阶语义（显式声明）：promote 后同对写时 notice 被 is_semantic_pair_closed
  抑制（与 escalate 同形态，"升组即结案"）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path) -> MemoryTools:
    return MemoryTools(Settings(db_path=tmp_path / "m.db", backup_jsonl=tmp_path / "b.jsonl",
                                workspace="w"))


def _seed_notice(tools: MemoryTools) -> tuple[int, int, int]:
    """播种一条 notice（candidate + notice_type + delivered）。返回 (notice_id, a, b)。"""
    a = tools.memory_write(content="生产数据库使用 MySQL 8", subject="db",
                           workspace="w", source_type="agent_generated")
    b = tools.memory_write(content="生产数据库使用 SQLite 3", subject="db2",
                           workspace="w", source_type="agent_generated")
    aid, bid = int(a["data"]["id"]), int(b["data"]["id"])
    h = hashlib.sha256(b"x").hexdigest()
    slot = {"entity": "生产数据库", "attribute": "数据库选型", "scope": "默认"}
    mv = [
        {"memory_id": aid, "version": 1, "value_raw": "MySQL 8", "normalized_value": "MySQL 8",
         "attribute_raw": "数据库选型", "normalized_attribute": "数据库选型", "direction": "a_to_b",
         "content_hash": h, "evidence_quote": "使用 MySQL 8", "evidence_span": [0, 10],
         "prompt_version": "v1", "detector_version": "semantic-evidence-v1"},
        {"memory_id": bid, "version": 1, "value_raw": "SQLite 3", "normalized_value": "SQLite 3",
         "attribute_raw": "数据库选型", "normalized_attribute": "数据库选型", "direction": "b_to_a",
         "content_hash": h, "evidence_quote": "使用 SQLite 3", "evidence_span": [0, 11],
         "prompt_version": "v1", "detector_version": "semantic-evidence-v1"},
    ]
    groups = [{"normalized_value": "MySQL 8", "display_value": "MySQL 8", "members": [f"{aid}@1"]},
              {"normalized_value": "SQLite 3", "display_value": "SQLite 3", "members": [f"{bid}@1"]}]
    from memory_arbiter.semantic_conflict import notice_dedupe_key

    res = tools.db.semantic_notices.record_semantic_notice(
        memory_id=aid, peer_id=bid, severity="normal", notice_type="semantic_evidence",
        title="t", message="m",
        payload={"slot_key": slot, "member_versions": mv, "value_groups": groups, "task_id": "t1"},
        dedupe_key=notice_dedupe_key(aid, bid, 1, 1, "semantic_evidence"),
        left_version=1, right_version=1,
    )
    return int(res["conflict_id"]), aid, bid


def _promote(tools: MemoryTools, aid: int, bid: int):
    h = hashlib.sha256(b"x").hexdigest()
    slot = {"entity": "生产数据库", "attribute": "数据库选型", "scope": "默认"}
    mv = [
        {"memory_id": aid, "version": 1, "value_raw": "MySQL 8", "normalized_value": "MySQL 8",
         "attribute_raw": "数据库选型", "normalized_attribute": "数据库选型", "direction": "a_to_b",
         "content_hash": h, "evidence_quote": "使用 MySQL 8", "evidence_span": [0, 10],
         "prompt_version": "v1", "detector_version": "semantic-evidence-v1"},
        {"memory_id": bid, "version": 1, "value_raw": "SQLite 3", "normalized_value": "SQLite 3",
         "attribute_raw": "数据库选型", "normalized_attribute": "数据库选型", "direction": "b_to_a",
         "content_hash": h, "evidence_quote": "使用 SQLite 3", "evidence_span": [0, 11],
         "prompt_version": "v1", "detector_version": "semantic-evidence-v1"},
    ]
    groups = [{"normalized_value": "MySQL 8", "display_value": "MySQL 8", "members": [f"{aid}@1"]},
              {"normalized_value": "SQLite 3", "display_value": "SQLite 3", "members": [f"{bid}@1"]}]
    return tools.memory_repair("record_conflict", {
        "members": mv, "value_groups": groups, "slot_key": slot,
        "detector_version": "semantic-evidence-v1", "source": "agent_recorded",
        "reason": "manual promote", "workspace": "w", "status": "open",
    })


def _row(tools: MemoryTools, cid: int) -> dict:
    with tools.db.connection() as conn:
        r = conn.execute(
            "SELECT status, notice_delivery_status, notice_resolution_reason FROM conflicts WHERE id=?",
            (cid,),
        ).fetchone()
    return dict(r)


def test_promote_clears_notice_delivery_state(tmp_path: Path) -> None:
    """核心钉：promote 后 delivery='resolved' 且 reason 带前缀。"""
    tools = make_tools(tmp_path)
    cid, aid, bid = _seed_notice(tools)
    res = _promote(tools, aid, bid)
    assert res.get("ok"), res
    row = _row(tools, cid)
    assert row["status"] == "open"
    assert row["notice_delivery_status"] == "resolved"
    assert str(row["notice_resolution_reason"]).startswith("promoted_to_conflict:")


def test_promote_notice_not_listed_as_open(tmp_path: Path) -> None:
    """promote 后 notice list(open) 不再含该 id。"""
    tools = make_tools(tmp_path)
    cid, aid, bid = _seed_notice(tools)
    _promote(tools, aid, bid)
    notices = tools.db.list_semantic_notices(status="open") or []
    assert cid not in [int(n["id"]) for n in notices]


def test_promoted_conflict_cannot_be_dismissed_via_notice(tmp_path: Path) -> None:
    """核心钉：notice dismiss 对已 promote 的正式冲突组返回 already_terminal。

    此前它会返回 updated 并把 open 组改写成 not_a_conflict（绕过 judge/apply）。
    """
    tools = make_tools(tmp_path)
    cid, aid, bid = _seed_notice(tools)
    _promote(tools, aid, bid)
    d = tools.memory_repair("notice", {"action": "dismiss", "notice_id": cid, "reason": "fp"})
    assert (d.get("data") or {}).get("outcome") == "already_terminal", d
    row = _row(tools, cid)
    assert row["status"] == "open", "正式冲突组不得被 notice 通道撤回"


def test_bare_candidate_promote_untouched(tmp_path: Path) -> None:
    """bare candidate（无 notice_type）promote 路径不受影响（仍 not_applicable）。"""
    tools = make_tools(tmp_path)
    a = tools.memory_write(content="X 是 1", subject="s1", workspace="w", source_type="agent_generated")
    b = tools.memory_write(content="X 是 2", subject="s2", workspace="w", source_type="agent_generated")
    aid, bid = int(a["data"]["id"]), int(b["data"]["id"])
    h = hashlib.sha256(b"y").hexdigest()
    slot = {"entity": "X", "attribute": "值", "scope": "默认"}
    mv = [
        {"memory_id": aid, "version": 1, "value_raw": "1", "normalized_value": "1",
         "attribute_raw": "值", "normalized_attribute": "值", "direction": "a_to_b",
         "content_hash": h, "evidence_quote": "X 是 1", "evidence_span": [0, 5],
         "prompt_version": "v1", "detector_version": "semantic-evidence-v1"},
        {"memory_id": bid, "version": 1, "value_raw": "2", "normalized_value": "2",
         "attribute_raw": "值", "normalized_attribute": "值", "direction": "b_to_a",
         "content_hash": h, "evidence_quote": "X 是 2", "evidence_span": [0, 5],
         "prompt_version": "v1", "detector_version": "semantic-evidence-v1"},
    ]
    groups = [{"normalized_value": "1", "display_value": "1", "members": [f"{aid}@1"]},
              {"normalized_value": "2", "display_value": "2", "members": [f"{bid}@1"]}]
    # 先落 bare candidate（not_a_conflict 无 slot；用 open 建组前先建 candidate 不可行，
    # 故直接走 record_conflict 两阶段：先 status=not_a_conflict 不占 slot，再 open 建组）
    res = tools.memory_repair("record_conflict", {
        "members": mv, "value_groups": groups, "slot_key": slot,
        "detector_version": "semantic-evidence-v1", "source": "agent_recorded",
        "reason": "r", "workspace": "w", "status": "open",
    })
    assert res.get("ok"), res
    cid = int((res["data"] or {})["conflict_id"])
    row = _row(tools, cid)
    assert row["notice_delivery_status"] == "not_applicable"


def test_promote_suppresses_pair_repeat_notice(tmp_path: Path) -> None:
    """第二阶语义（显式声明）：promote 后同对写时 notice 被 pair-closed 抑制。

    与 escalate 同形态（"升组即结案"）；本测试钉住该语义不漂移。
    notice_dedupe_key 走生产派生（semantic_conflict.notice_dedupe_key），
    notice_type 用生产值 semantic_evidence。
    """
    from memory_arbiter.semantic_conflict import notice_dedupe_key

    tools = make_tools(tmp_path)
    cid, aid, bid = _seed_notice(tools)
    _promote(tools, aid, bid)
    key = notice_dedupe_key(aid, bid, 1, 1, "semantic_evidence")
    closed = tools.db.semantic_notices.is_semantic_pair_closed(
        aid, bid, left_version=1, right_version=1, notice_type="semantic_evidence",
    )
    assert closed is True, (
        "promote 后该对应视为已结案（不重发 notice）——"
        f"dedupe_key={key[:12]}…"
    )
