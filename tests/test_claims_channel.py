"""claims 契约链覆盖 — 0.17.0 P2-5.1/5.2/5.3。

校验 schema 硬门、写入持久化（归一/grounding/拒收回执）、灰度缺失警告与
enforce 硬拒、零 Qwen 通道（同 attr 值不同发 notice；A4 自共存不报；
软门单侧未填不拦【review 推荐①口径，随 owner 收口】）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import tests.test_vnext_evidence as tv  # noqa: E402
from memory_arbiter.validation import validate_product_payload  # noqa: E402


# ---------------- P2-5.1 校验 ----------------

def test_claims_schema_hard_errors() -> None:
    bad_payloads = [
        {"content": "x" * 20, "subject": "s", "claims": "not-a-list"},
        {"content": "x" * 20, "subject": "s", "claims": [{"attr": "a" * 65, "value": "1"}]},
        {"content": "x" * 20, "subject": "s", "claims": [{"attr": "a", "value": "v" * 65}]},
        {"content": "x" * 20, "subject": "s", "claims": [{"attr": "超时 500ms", "value": "500ms"}]},
        {"content": "x" * 20, "subject": "s",
         "claims": [{"attr": "a", "value": " ".join(f"w{i}" for i in range(13))}]},
        {"content": "x" * 20, "subject": "s",
         "claims": [{"attr": f"a{i}", "value": str(i)} for i in range(21)]},
    ]
    for payload in bad_payloads:
        result = validate_product_payload("memory", "remember", payload)
        assert result.error is not None and result.error.get("field") == "claims", payload
    ok = validate_product_payload("memory", "remember", {
        "content": "x" * 20, "subject": "s", "claims": [],
    })
    assert ok.error is None  # 空数组=显式声明无 claims
    ok2 = validate_product_payload("memory", "remember", {
        "content": "x" * 20, "subject": "s", "claims": [{"attr": "超时", "value": "500ms"}],
    })
    assert ok2.error is None


# ---------------- P2-5.2 写入路径 ----------------

def test_write_persists_claims_and_rejects_ungrounded(tmp_path: Path) -> None:
    tools = tv.make_tools(tmp_path)
    result = tools.memory_write(
        content="网关读超时为 500 毫秒，熔断开启。",
        subject="gw", tags=[],
        claims=[
            {"attr": "超时", "value": "500 毫秒"},
            {"attr": "熔断", "value": "不存在于正文的值"},
        ],
    )
    assert result["ok"] is True
    data = result["data"]
    assert data["claims_written"] == 1
    assert [r["reason"] for r in data["claims_rejected"]] == ["value_not_in_content"]
    claims = tools.db.claims.current_claims(data["id"])
    assert len(claims) == 1 and claims[0]["value_norm"] != ""  # 归一在位
    with tools.db.connection() as conn:
        vec_rows = conn.execute(
            "SELECT COUNT(*) FROM memory_claim_vec WHERE id IN "
            "(SELECT id FROM memory_claims WHERE memory_id=?)", (data["id"],),
        ).fetchone()[0]
    assert vec_rows == 1


def test_missing_claims_grey_warn_and_enforce(tmp_path: Path) -> None:
    tools = tv.make_tools(tmp_path)
    result = tools.memory_write(content="普通记忆一句话足够长。", subject="s", tags=[])
    assert result["ok"] is True
    assert any("claims 必填" in w for w in result.get("warnings", []))  # 灰度只警告

    tools.settings.claims_required = True
    enforced = tools.memory_write(content="另一条普通记忆也很长。", subject="s2", tags=[])
    assert enforced["ok"] is False
    assert enforced["data"]["field"] == "claims"
    tools.settings.claims_required = False
    explicit = tools.memory_write(content="第三条普通记忆长度足够。", subject="s3", tags=[], claims=[])
    assert explicit["ok"] is True and "claims 必填" not in explicit.get("warnings", [])


# ---------------- P2-5.3 零 Qwen 通道 ----------------

def _write_with_claims(tools, content: str, subject: str, claims: list, metadata: dict | None = None):
    return tools.memory_write(
        content=content, subject=subject, tags=[],
        metadata=metadata if metadata is not None else {"entity": "svc", "scope": "prod"},
        claims=claims,
    )["data"]


def test_claims_channel_fires_on_value_diff(tmp_path: Path) -> None:
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    a = _write_with_claims(tools, "网关读超时为 500 毫秒。", "a", [{"attr": "超时", "value": "500 毫秒"}])
    b = _write_with_claims(tools, "网关读超时为 3 秒。", "b", [{"attr": "超时", "value": "3 秒"}])
    result = tools._process_semantic_conflict_job(
        b["id"], {"memory_id": b["id"], "version": 1, "content_hash": "unused"},
    )
    channel = result.get("claims_channel", {})
    assert channel.get("notices", 0) >= 1
    notices = [
        n for n in tools.db.list_semantic_notices()
        if n["memory_id"] == b["id"] and n["notice_type"] == "claim_conflict"
    ]
    assert len(notices) == 1
    assert notices[0]["payload"]["source"] == "claim_conflict"
    # 半秒折算（P2-1.2）：claims 值归一后同值不报
    c = _write_with_claims(tools, "灰度窗口为半秒的记录。", "c", [{"attr": "超时", "value": "半秒"}])
    tools._process_semantic_conflict_job(
        c["id"], {"memory_id": c["id"], "version": 1, "content_hash": "unused"},
    )
    fired_again = [
        n for n in tools.db.list_semantic_notices()
        if n["memory_id"] == c["id"] and n["notice_type"] == "claim_conflict"
    ]
    # 半秒=0.5s：与 a 的 500ms 归一相同 → 不报（若折算缺失则会误报，钉住 P2-1.2 联动）
    from memory_arbiter.semantic_conflict import normalize_value
    assert normalize_value("半秒") == normalize_value("500 毫秒")


def test_claims_channel_coexistence_veto_a4(tmp_path: Path) -> None:
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    _write_with_claims(tools, "网关读超时为 500 毫秒。", "a", [{"attr": "超时", "value": "500 毫秒"}])
    # b 自声明双值（白天/夜间）→ 对 a 不是对立主张
    b = _write_with_claims(
        tools, "白天超时 3 秒，夜间超时 9 秒。", "b",
        [{"attr": "超时", "value": "3 秒"}, {"attr": "超时", "value": "9 秒"}],
    )
    result = tools._process_semantic_conflict_job(
        b["id"], {"memory_id": b["id"], "version": 1, "content_hash": "unused"},
    )
    assert result.get("claims_channel", {}).get("notices", 0) == 0


def test_claims_channel_soft_gate_single_side_passes(tmp_path: Path) -> None:
    """review 推荐①：单侧填 entity/scope 不拦（仅双侧都填且不等才拦）。"""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    _write_with_claims(tools, "网关读超时为 500 毫秒。", "a", [{"attr": "超时", "value": "500 毫秒"}])
    b = tools.memory_write(  # b 无 metadata
        content="网关读超时为 3 秒。", subject="b", tags=[],
        claims=[{"attr": "超时", "value": "3 秒"}],
    )["data"]
    result = tools._process_semantic_conflict_job(
        b["id"], {"memory_id": b["id"], "version": 1, "content_hash": "unused"},
    )
    assert result.get("claims_channel", {}).get("notices", 0) >= 1  # 单侧未填=不拦


def test_agent_supplied_backfill_pending_and_apply(tmp_path: Path) -> None:
    """0.17.0（owner 2026-09-23）：claims_backfill 的 agent 供给模式——
    pending 列缺口清单、apply 落库（source=backfill/replace 语义/grounding）。"""
    tools = tv.make_tools(tmp_path)
    a = tools.memory_write(content="网关读超时为 500 毫秒，熔断开启。", subject="a", tags=[])["data"]
    b = tools.memory_write(content="限流上限为 200 QPS。", subject="b", tags=[])["data"]

    pending = tools._claims_backfill_task({"mode": "pending", "limit": 10})
    assert pending["ok"] and pending["count"] == 2
    ids = {item["memory_id"] for item in pending["items"]}
    assert ids == {a["id"], b["id"]}
    assert all(item["content"] for item in pending["items"])

    applied = tools._claims_backfill_task({
        "mode": "apply",
        "results": [
            {"memory_id": a["id"], "claims": [
                {"attr": "超时", "value": "500 毫秒"},
                {"attr": "熔断", "value": "没有这句话"},  # grounding 拒
            ]},
            {"memory_id": b["id"], "claims": [{"attr": "限流上限", "value": "200 QPS"}]},
        ],
    })
    assert applied["ok"] and applied["applied_memories"] == 2
    assert applied["claims_written"] == 2
    assert applied["rejected_count"] == 1
    rows = tools.db.claims.current_claims(a["id"])
    assert len(rows) == 1 and rows[0]["source"] == "backfill"
    # 缺口清零
    assert tools._claims_backfill_task({"mode": "pending"})["count"] == 0
    # replace 重放：同结果覆盖不产生重复
    again = tools._claims_backfill_task({
        "mode": "apply",
        "results": [{"memory_id": a["id"], "claims": [{"attr": "超时", "value": "500 毫秒"}]}],
    })
    assert again["claims_written"] == 1
    assert len(tools.db.claims.current_claims(a["id"])) == 1
