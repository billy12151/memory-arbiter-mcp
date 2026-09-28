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


def _write_with_claims(tools, content: str, subject: str, claims: list, metadata: dict | None = None):
    return tools.memory_write(
        content=content, subject=subject, tags=[],
        metadata=metadata if metadata is not None else {"entity": "svc", "scope": "prod"},
        claims=claims,
    )["data"]



def test_edit_claims_inheritance_and_dropped(tmp_path: Path) -> None:
    """R2 方案 C（owner 拍板）：内容编辑不传 claims 时旧版本 claims 自动
    继承到新版本（重过 grounding）；值已不在新正文的剔除进
    claims_inherited_dropped 回执——通道不再被编辑静默打死。"""
    tools = tv.make_tools(tmp_path)
    a = tools.memory_write(
        content="网关读超时为 500 毫秒，熔断开启。", subject="inherit", tags=[],
        claims=[{"attr": "超时", "value": "500 毫秒"}],
    )["data"]
    edited = tools.memory_edit(
        memory_id=a["id"],
        new_content="网关读超时仍为 500 毫秒，另补限流说明一句话。",
        reason="r1",
    )
    assert edited["ok"], edited
    data = edited["data"]
    assert data["claims_inherited"] == 1
    assert not data.get("claims_inherited_dropped")
    rows = tools.db.claims.current_claims(a["id"])
    assert len(rows) == 1
    assert rows[0]["memory_version"] == data["new_version"]

    edited2 = tools.memory_edit(
        memory_id=a["id"], new_content="正文已完全改写为别的话题内容。", reason="r2",
    )
    assert edited2["ok"], edited2
    data2 = edited2["data"]
    assert data2.get("claims_inherited", 0) == 0
    dropped = data2.get("claims_inherited_dropped") or []
    assert [r["reason"] for r in dropped] == ["value_not_in_content"]
    assert tools.db.claims.current_claims(a["id"]) == []


def test_status_flip_repins_claims_without_unique_collision(tmp_path: Path) -> None:
    """R2 P1 回归：继承让「跨版本同值行共存」（v1 审计行 + v2 继承行）成为
    稳态，snapshot 语义翻转（confirm 改 confidence/protection）的 +1 重钉在
    旧行为下逐行查 UNIQUE 必撞、炸掉整个翻转事务；负空间两步平移后通过。"""
    tools = tv.make_tools(tmp_path)
    a = tools.memory_write(
        content="网关读超时为 500 毫秒，熔断开启。", subject="repin", tags=[],
        claims=[{"attr": "超时", "value": "500 毫秒"}],
    )["data"]
    tools.memory_edit(
        memory_id=a["id"], new_content="网关读超时仍是 500 毫秒。", reason="r",
    )
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT memory_version, COUNT(*) FROM memory_claims"
            " WHERE memory_id=? GROUP BY memory_version", (a["id"],),
        ).fetchall()
    assert {int(r[0]): int(r[1]) for r in rows} == {1: 1, 2: 1}, "跨版本同值行必须共存"
    confirmed = tools.memory_confirm(memory_id=a["id"], confidence=0.9, authorized=True)
    assert confirmed["ok"], confirmed
    with tools.db.connection() as conn:
        rows = conn.execute(
            "SELECT memory_version, COUNT(*) FROM memory_claims"
            " WHERE memory_id=? GROUP BY memory_version", (a["id"],),
        ).fetchall()
    versions = sorted(int(r[0]) for r in rows)
    assert len(versions) == len(set(versions)), "重钉后不得出现版本内重复行"
