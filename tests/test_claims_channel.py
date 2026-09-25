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
    assert channel.get("channel_b_notices", 0) >= 1
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
    assert result.get("claims_channel", {}).get("channel_b_notices", 0) == 0


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
    assert result.get("claims_channel", {}).get("channel_b_notices", 0) >= 1  # 单侧未填=不拦


def test_claims_channel_versional_attr_exempt(tmp_path: Path) -> None:
    """D1（owner 2026-09-23）：版本类 attr 值差异=预期演进，豁免不报且计数可见；
    同记忆非版本 attr（超时）不受豁免影响照报——证明豁免范围没扩大。"""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    _write_with_claims(
        tools, "内核版本为 0.15.8，超时为 500 毫秒。", "a",
        [{"attr": "版本", "value": "0.15.8"}, {"attr": "超时", "value": "500 毫秒"}],
    )
    b = _write_with_claims(
        tools, "内核版本为 0.16.0，超时为 3 秒。", "b",
        [{"attr": "版本", "value": "0.16.0"}, {"attr": "超时", "value": "3 秒"}],
    )
    result = tools._process_semantic_conflict_job(
        b["id"], {"memory_id": b["id"], "version": 1, "content_hash": "unused"},
    )
    channel = result.get("claims_channel", {})
    assert channel.get("channel_b_versional_vetoed", 0) >= 1  # 版本 claim 被豁免且可见
    assert channel.get("channel_b_notices", 0) == 1  # 超时值不同照报（豁免不外溢）
    fired = [
        n for n in tools.db.list_semantic_notices()
        if n["notice_type"] == "claim_conflict"
    ]
    assert len(fired) == 1 and "超时" in fired[0]["message"]


def test_claims_channel_versional_hit_level_fallback(tmp_path: Path, monkeypatch) -> None:
    """D1 hit 级兜底：own attr 非版本、peer attr 版本类（τ 近似过门）同样豁免。"""
    import memory_arbiter.constants as constants
    monkeypatch.setattr(constants, "CLAIM_ATTR_TAU", 0.0)
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    _write_with_claims(tools, "构建说明见发布说明，版本 1.0 已出。", "a",
                       [{"attr": "发布说明", "value": "版本 1.0"}])
    b = _write_with_claims(tools, "构建说明 release notes 更新到 2.0。", "b",
                           [{"attr": "release notes", "value": "2.0"}])
    result = tools._process_semantic_conflict_job(
        b["id"], {"memory_id": b["id"], "version": 1, "content_hash": "unused"},
    )
    channel = result.get("claims_channel", {})
    assert channel.get("channel_b_notices", 0) == 0
    assert channel.get("channel_b_versional_vetoed", 0) >= 1
    assert not [
        n for n in tools.db.list_semantic_notices()
        if n["notice_type"] == "claim_conflict"
    ]


def test_claims_exact_lane_hit_level_versional_norm_collision(tmp_path: Path) -> None:
    """D1 hit 级（精确通道）：own attr "releasenotes" raw 非版本al、peer
    attr "release notes" raw 版本al，但 attr_norm 归一相等（剥空格）→
    精确通道候选命中后必须按 peer RAW attr 豁免，不得产 notice，也不得
    记 fired_attrs 封死 KNN 侧救济（R2 复评 P2：修复前该对在精确通道
    误报）。"""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    _write_with_claims(tools, "构建说明 release notes 更新到 1.0。", "a",
                       [{"attr": "release notes", "value": "1.0"}])
    b = _write_with_claims(tools, "构建说明 releasenotes 更新到 2.0。", "b",
                           [{"attr": "releasenotes", "value": "2.0"}])
    result = tools._process_semantic_conflict_job(
        b["id"], {"memory_id": b["id"], "version": 1, "content_hash": "unused"},
    )
    channel = result.get("claims_channel", {})
    assert channel.get("channel_b_notices", 0) == 0
    assert channel.get("channel_b_versional_vetoed", 0) >= 1
    assert not [
        n for n in tools.db.list_semantic_notices()
        if n["notice_type"] == "claim_conflict"
    ]


def test_claims_vectors_pending_branch_carries_exact_lane(tmp_path: Path) -> None:
    """R2 复评 P1：claims 向量缺失（embed 失败/backfill replace 后未重建）
    时 KNN 腿跑不了，但精确通道先跑且可能已创建 notice——vectors_pending
    早退分支必须携带精确通道真实结果（修复前硬编码 channel_b_notices: 0
    并丢 _surfaced_peers：回执误报 0、A-cross 对同一对二次通知）。"""
    tools = tv.make_tools(tmp_path)
    tools.settings.semantic_conflict_on_write = "off"
    _write_with_claims(tools, "网关读超时为 500 毫秒。", "a", [{"attr": "超时", "value": "500 毫秒"}])
    b = _write_with_claims(tools, "网关读超时为 3 秒。", "b", [{"attr": "超时", "value": "3 秒"}])
    # 模拟 b 的 claims 向量缺失（等价于 embed 失败/replace 清空后的形态）
    with tools.db.connection() as conn:
        conn.execute(
            "DELETE FROM memory_claim_vec WHERE id IN"
            " (SELECT id FROM memory_claims WHERE memory_id=?)",
            (b["id"],),
        )
        conn.commit()
    result = tools._process_semantic_conflict_job(
        b["id"], {"memory_id": b["id"], "version": 1, "content_hash": "unused"},
    )
    channel = result.get("claims_channel", {})
    assert channel.get("reason") == "vectors_pending"
    assert channel.get("channel_b_notices", 0) >= 1  # 精确通道的 notice 必须可见
    assert channel.get("channel_b_exact_checked", 0) >= 1
    notices = [
        n for n in tools.db.list_semantic_notices()
        if n["memory_id"] == b["id"] and n["notice_type"] == "claim_conflict"
    ]
    assert len(notices) == 1  # 精确通道创建，未被丢报
    assert "_surfaced_peers" not in channel  # wrapper pop 契约不破


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


def test_claims_backfill_no_mode_returns_guidance(tmp_path: Path) -> None:
    """D2（owner 2026-09-23）：无人值守 Qwen 通道退役——无 mode 返回引导语。"""
    tools = tv.make_tools(tmp_path)
    result = tools._claims_backfill_task({})
    assert result["ok"] is False
    assert "mode='pending'" in result["error"] and "mode='apply'" in result["error"]
    # model_path 残留参数不再复活 Qwen 通道
    result2 = tools._claims_backfill_task({"model_path": "/nonexistent.gguf"})
    assert result2["ok"] is False and "mode='pending'" in result2["error"]


# ---------------- 0.17.0 review R2 ----------------

def test_backfill_apply_overflow_receipt_and_source_fallback(tmp_path: Path) -> None:
    """R2：backfill apply 是唯一绕过 schema ≤20 硬门的入口——溢出 claims
    逐条进 rejected（reason=exceeds_max_per_memory），不再被静默截断蒸发；
    逐条非法 source 兜底 agent（memory_claims CHECK 约束防炸）。"""
    tools = tv.make_tools(tmp_path)
    content = "正文锚点：" + " ".join(f"token{i}" for i in range(24)) + "。"
    a = tools.memory_write(content=content, subject="overflow", tags=[])["data"]
    claims = [
        {"attr": f"attr{i}", "value": f"token{i}",
         **({"source": "hacker"} if i == 0 else {})}
        for i in range(22)
    ]
    applied = tools._claims_backfill_task({
        "mode": "apply", "results": [{"memory_id": a["id"], "claims": claims}],
    })
    assert applied["ok"], applied
    assert applied["claims_written"] == 20
    assert applied["rejected_count"] == 2
    overflow = [
        r for r in applied["rejected_sample"]
        if r["reason"] == "exceeds_max_per_memory"
    ]
    assert [r["index"] for r in overflow] == [20, 21]
    rows = tools.db.claims.current_claims(a["id"])
    assert len(rows) == 20
    assert rows[0]["source"] == "agent", "非法 source 必须兜底 agent"
    assert rows[1]["source"] == "backfill"


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
