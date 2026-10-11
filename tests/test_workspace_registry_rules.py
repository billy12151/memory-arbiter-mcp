

# ── from test_workspace_rules.py ──
# helper make_tools renamed: rules_make_tools (collision)

"""Rule-first workspace decision layer (design 636 §2, §3, §5)."""
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools
from memory_arbiter import workspace_rules as wr
import json
from contextlib import contextmanager
from types import SimpleNamespace
import sqlite3
import threading
import pytest
from memory_arbiter.db import MemoryDB, _normalize_alias_key
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryStatus
from tests._helpers import alias_governance_make_tools, register
from tests._helpers import record_conflict


# ── quality classification ───────────────────────────────────────────────────

def test_quality_empty_default():
    assert wr.classify_workspace_quality("") == "empty"
    assert wr.classify_workspace_quality("   ") == "empty"
    assert wr.classify_workspace_quality("default") == "default"
    assert wr.classify_workspace_quality("默认") == "default"


def test_quality_generic():
    assert wr.classify_workspace_quality("实施计划") == "generic"
    assert wr.classify_workspace_quality("月报") == "generic"
    assert wr.classify_workspace_quality("notes") == "generic"


def test_quality_specific():
    assert wr.classify_workspace_quality("金营项目") == "specific"
    assert wr.classify_workspace_quality("project-x") == "specific"


def test_quality_suspicious():
    assert wr.classify_workspace_quality("/etc/passwd") == "suspicious"
    assert wr.classify_workspace_quality("http://x.com") == "suspicious"
    assert wr.classify_workspace_quality("a" * 90) == "suspicious"


# ── evidence extraction ──────────────────────────────────────────────────────

def test_extract_evidence_from_dict():
    ev = wr.extract_evidence({
        "subject": "金营项目周报",
        "content": "# 概述\n\n本周完成了 X。还做了 Y。\n\n## 细节\n更多内容。",
    })
    assert ev["subject"] == "金营项目周报"
    assert ev["title"] == "金营项目周报"
    assert "概述" in ev["headings"]
    assert ev["key_sentences"]


# ── rule decision ────────────────────────────────────────────────────────────

def test_decision_auto_on_confirmed_alias():
    resolved = {"matched_by": "confirmed_alias", "canonical": "金营项目", "similar": []}
    d = wr.rule_decision("金营二期", resolved)
    assert d["decision"] == "AUTO" and d["canonical"] == "金营项目"


def test_decision_keep_reference_material():
    resolved = {"matched_by": "vector", "canonical": "金营项目",
                "similar": [{"name": "金营项目", "distance": 0.1}]}
    ev = {"title": "参考金营项目的月报模板", "first_para": "借鉴其结构"}
    d = wr.rule_decision("模板库", resolved, ev)
    assert d["decision"] == "KEEP" and d["reason"] == "reference_material"


def test_decision_keep_rejected_pair():
    resolved = {"matched_by": "vector", "canonical": "金营项目",
                "rejected_canonicals": ["金营项目"],
                "similar": [{"name": "金营项目", "distance": 0.1}]}
    d = wr.rule_decision("金营培训", resolved)
    assert d["decision"] == "KEEP" and d["reason"] == "rejected_pair"


def test_decision_ask_on_generic():
    resolved = {"matched_by": "new", "canonical": "月报", "similar": []}
    d = wr.rule_decision("月报", resolved)
    assert d["decision"] == "ASK"


def test_decision_ask_on_near_tie():
    resolved = {"matched_by": "vector", "canonical": "A",
                "similar": [{"name": "A", "distance": 0.20}, {"name": "B", "distance": 0.22}]}
    d = wr.rule_decision("金营", resolved)
    assert d["decision"] == "ASK" and d["reason"] == "candidate_near_tie"


def test_decision_auto_new_specific():
    resolved = {"matched_by": "new", "canonical": "赛博项目", "similar": []}
    d = wr.rule_decision("赛博项目", resolved)
    assert d["decision"] == "AUTO" and d["reason"] == "new_specific_canonical"


# ── review regression: rejected candidate at similar[1] must NOT block a valid
#    merge into the resolver's chosen non-rejected canonical (workspace_rules:143)

def test_rejected_at_similar0_does_not_block_valid_chosen_canonical():
    # resolver skipped rejected ProjectC (similar[0]) and chose ProjectD.
    resolved = {
        "matched_by": "vector", "canonical": "ProjectD",
        "rejected_canonicals": ["ProjectC"],
        "similar": [{"name": "ProjectC", "distance": 0.10},
                    {"name": "ProjectD", "distance": 0.20}],
    }
    d = wr.rule_decision("aliasX", resolved)
    # must merge into the valid ProjectD, NOT keep-separate on the rejected pair
    assert d["decision"] == "AUTO"
    assert d["canonical"] == "ProjectD"


def test_rejected_at_similar1_does_not_trigger_spurious_near_tie():
    # ProjectD is the clean winner; rejected ProjectC sits at similar[1].
    resolved = {
        "matched_by": "vector", "canonical": "ProjectD",
        "rejected_canonicals": ["ProjectC"],
        "similar": [{"name": "ProjectD", "distance": 0.20},
                    {"name": "ProjectC", "distance": 0.22}],
    }
    d = wr.rule_decision("ProjectDvariant", resolved)
    # only one non-rejected candidate → no tie → AUTO, not a re-prompt
    assert d["decision"] == "AUTO"
    assert d["reason"] == "vector_strong"


def test_chosen_canonical_equal_to_rejected_keeps_separate():
    # defensive: if the chosen canonical itself is a rejected name (via a
    # non-confirmed/non-exact path), keep apart rather than merge.
    resolved = {
        "matched_by": "fallback", "canonical": "ProjectC",
        "rejected_canonicals": ["ProjectC"], "similar": [],
    }
    d = wr.rule_decision("ProjectC", resolved)
    assert d["decision"] == "KEEP" and d["reason"] == "rejected_pair"


# ── integration: write path surfaces decision ───────────────────────────────

def rules_make_tools(tmp_path: Path, isolation: str = "weak") -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "rules.sqlite3",
        backup_jsonl=tmp_path / "rules.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_write_surfaces_ask_for_generic_workspace(tmp_path):
    t = rules_make_tools(tmp_path)
    r = t.memory_write(content="some plan", workspace="月报", source_type="agent_generated", subject="test")
    data = r["data"]
    assert data["workspace_decision"] == "ASK"
    assert data.get("write_hints", {}).get("workspace_review")


def test_write_auto_for_specific_workspace(tmp_path):
    t = rules_make_tools(tmp_path)
    r = t.memory_write(content="alpha", workspace="金营项目", source_type="agent_generated", subject="test")
    assert r["data"]["workspace_decision"] == "AUTO"


# ── mechanical variant of an existing canonical (2026-08-21) ─────────────────
#
# A real-library dry-run showed agent-lane failing to reach its existing
# AgentLane canonical: exact match is case/separator sensitive and the vector
# tier could miss a low-frequency canonical. A deterministic case/hyphen/
# underscore/whitespace fold reuses the registered spelling without vector/Qwen.

def test_mechanical_variant_reuses_existing_canonical(tmp_path):
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute("INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES('AgentLane',datetime('now'))")

    for raw in ["agent-lane", "AGENTLANE", "agent_lane", "Agent Lane"]:
        r = db.resolve_workspace_canonical(raw, None)
        assert r["canonical"] == "AgentLane", raw
        assert r["matched_by"] == "mechanical_variant", raw

    # Exact spelling still takes the exact tier.
    exact = db.resolve_workspace_canonical("AgentLane", None)
    assert exact["matched_by"] == "exact"

    # A genuinely different name is NOT folded into the canonical.
    other = db.resolve_workspace_canonical("agentlanes-cli", None)
    assert other["matched_by"] == "new"
    assert other["canonical"] == "agentlanes-cli"


def test_mechanical_variant_does_not_collapse_blanks(tmp_path):
    # Empty/whitespace keys must never collide via the mechanical fold.
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute("INSERT OR IGNORE INTO workspace_canonicals(name,created_at) VALUES('AgentLane',datetime('now'))")
    r = db.resolve_workspace_canonical("   ", None)
    assert r["canonical"] == "default"
    assert r["matched_by"] == "fallback"



# ── 0.17.1 追加（2026-10-01）：mechanical_variant AUTO 补判 + rejected 机械通道免疫 ──
#
# 方案：ZCodeProject/docs/mema-ws-normalization-cleanup-plan-2026-10-01.md（v4）。
# G 守卫（变体对拒分）之后，新的 twin rejected 行无法再经治理产生；本节的
# rejected 行一律直插表内，模拟的是存量行（legacy）——免疫逻辑消费的正是它们。


def _insert_legacy_rejected_row(tools, alias: str, canonical: str) -> None:
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO workspace_aliases(alias_workspace,canonical,status,updated_at) "
            "VALUES(?,?,'rejected',datetime('now'))",
            (_normalize_alias_key(alias), canonical),
        )


def test_rule_decision_mechanical_variant_is_auto():
    resolved = {"matched_by": "mechanical_variant", "canonical": "AgentLane",
                "similar": [], "rejected_canonicals": []}
    d = wr.rule_decision("agent-lane", resolved, {"title": "t", "first_para": "内容"})
    assert d["decision"] == "AUTO"
    assert d["reason"] == "mechanical_variant"
    assert d["canonical"] == "AgentLane"


def test_rejected_legacy_row_blocks_mechanical_fold(tmp_path):
    """被拒原名（legacy rejected 行）不再被 1b 机械折叠进被拒桶（C 层1）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    _insert_legacy_rejected_row(t, "agent-lane", "AgentLane")
    r = db.resolve_workspace_canonical("agent-lane", None)
    assert r["matched_by"] == "new"
    assert r["canonical"] == "agent-lane"
    assert "AgentLane" in r["rejected_canonicals"]


def test_rejected_ghost_variant_second_hop_blocks_fold(tmp_path):
    """幽灵变体（agent-lane 被拒后写 agent_lane）经机械键第二跳吃到免疫（C 层2）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    _insert_legacy_rejected_row(t, "agent-lane", "AgentLane")
    r = db.resolve_workspace_canonical("agent_lane", None)
    assert r["matched_by"] == "new"
    assert r["canonical"] == "agent_lane"
    assert "AgentLane" in r["rejected_canonicals"]


def test_rejected_twin_does_not_block_exact_canonical(tmp_path):
    """桶本名 exact 命中不受 rejected 影响，规则层 AUTO（合法写入）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    _insert_legacy_rejected_row(t, "agent-lane", "AgentLane")
    r = db.resolve_workspace_canonical("AgentLane", None)
    assert r["matched_by"] == "exact"
    d = wr.rule_decision("AgentLane", r, {"title": "t", "first_para": "x"})
    assert d["decision"] == "AUTO"


def test_confirmed_ghost_variant_still_folds(tmp_path):
    """confirmed 幽灵变体不参与第二跳，仍走 1b 折叠（第二跳只消费 rejected）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    ok, errors = db.workspaces.record_workspace_decision(
        "agent-lane", "AgentLane", status="confirmed",
    )
    assert ok, errors
    r = db.resolve_workspace_canonical("agent_lane", None)
    assert r["matched_by"] == "mechanical_variant"
    assert r["canonical"] == "AgentLane"


def test_multiple_rejected_twin_rows_all_aggregated(tmp_path):
    """同机械键多行 rejected 全量聚合（真实库 agent-chancellor 双行形状，R2-P0c）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        for name in ("AgentLane", "MemoryBank"):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
                "VALUES(?,datetime('now'))",
                (name,),
            )
    _insert_legacy_rejected_row(t, "agent-chancellor", "AgentLane")
    _insert_legacy_rejected_row(t, "agent-chancellor", "MemoryBank")
    r = db.resolve_workspace_canonical("agent_chancellor", None)
    assert "AgentLane" in r["rejected_canonicals"]
    assert "MemoryBank" in r["rejected_canonicals"]
    assert r["matched_by"] == "new"
    assert r["canonical"] == "agent_chancellor"


def test_direct_rejected_hit_aggregates_sibling_spellings(tmp_path):
    """P2 #5：直中（alias key 精确命中拒绝行）也聚合机械兄弟拼写的拒绝——
    separate "agent-lane"→X 与 "agent_lane"→Y 后再解析 "agent-lane"，
    X/Y 同时进抑制名单，向量并桶被拒。"""
    pytest.importorskip("sqlite_vec")
    t = alias_governance_make_tools(tmp_path, vec=True)
    if not t.db.state.sqlite_vec_available:
        pytest.skip("sqlite-vec unavailable")

    class Embedder:
        embedding_space_id = "test"
        dim = 2

        def embed_text(self, prefix="", body=""):
            return SimpleNamespace(embedding=[1.0, 0.0])

    embedder = Embedder()
    with t.db.write_transaction() as conn:
        for target in ("AlphaBucket", "BetaBucket"):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
                "VALUES(?,datetime('now'))", (target,),
            )
    for target in ("AlphaBucket", "BetaBucket"):
        t.db.workspaces.publish_workspace_canonical_vector(target, [1.0, 0.0])
    ok1, e1 = t.db.workspaces.record_workspace_decision(
        "agent-lane", "AlphaBucket", status="rejected",
    )
    ok2, e2 = t.db.workspaces.record_workspace_decision(
        "agent_lane", "BetaBucket", status="rejected",
    )
    assert ok1 and ok2, (e1, e2)

    r = t.db.resolve_workspace_canonical("agent-lane", embedder)
    # both spellings' rejections aggregated on the direct hit
    assert "AlphaBucket" in r["rejected_canonicals"]
    assert "BetaBucket" in r["rejected_canonicals"]
    # the vector merge into the (distance-0) rejected targets is refused
    assert r["matched_by"] == "new"
    assert r["canonical"] == "agent-lane"
    assert {s["name"] for s in r["similar"]} == set()


def test_ghost_hop_does_not_redirect_unrelated_registered_twin(tmp_path):
    """跨身份 rejected（foo-bar→ProjectX）不误伤已注册的孪生拼写 foo_bar（exact 优先）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('foo_bar',datetime('now'))"
        )
    ok, errors = db.workspaces.record_workspace_decision(
        "foo-bar", "ProjectX", status="rejected",
    )
    assert ok, errors
    r = db.resolve_workspace_canonical("foo_bar", None)
    assert r["matched_by"] == "exact"
    assert r["canonical"] == "foo_bar"


def test_separate_refuses_mechanical_twin_pair(tmp_path):
    """G 守卫：同一机械身份的变体对 separate 直接拒绝（owner 2026-10-01 拍板）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    result = t.memory_govern("separate_workspace_alias", {
        "workspace": "default",
        "alias": "agent-lane", "canonical": "AgentLane",
        "reason": "try to split", "authorized": True,
    })
    assert result["ok"] is False, result["data"]
    assert "spelling variants" in result["data"]["error"]
    # 幽灵拼写同样被拒
    ok2, errors2 = db.workspaces.record_workspace_decision(
        "agent_lane", "AgentLane", status="rejected",
    )
    assert not ok2 and errors2
    with db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspace_aliases").fetchone()[0] == 0
    # confirmed 方向不受守卫影响（ twin 确认=折叠语义，合法）
    ok3, errors3 = db.workspaces.record_workspace_decision(
        "agent-lane", "AgentLane", status="confirmed",
    )
    assert ok3, errors3


def test_strict_rejected_name_goes_pending(tmp_path):
    """strict 下被拒原名落 new → strict_block → PENDING + confirm 流程（新桶须确认）。"""
    t = rules_make_tools(tmp_path, "strict")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    _insert_legacy_rejected_row(t, "agent-lane", "AgentLane")
    r = t.memory_write(
        content="x", subject="s", workspace="agent-lane",
        source_type="agent_generated",
    )
    assert r["ok"], r
    data = r["data"]
    assert data["workspace_canonical"] == "agent-lane"
    assert data.get("action_required") == "confirm_new_workspace"
    assert db.get_memory(data["id"])["status"] == MemoryStatus.PENDING.value


def test_strict_mechanical_variant_reuses_active(tmp_path):
    """strict 下机械变体（确定性身份）直接 ACTIVE 复用，不 ASK 不 PENDING。"""
    t = rules_make_tools(tmp_path, "strict")
    db = t.db
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
            "VALUES('AgentLane',datetime('now'))"
        )
    r = t.memory_write(
        content="x", subject="s", workspace="agent-lane",
        source_type="agent_generated",
    )
    assert r["ok"], r
    data = r["data"]
    assert data["workspace_decision"] == "AUTO"
    assert data["workspace_canonical"] == "AgentLane"
    assert db.get_memory(data["id"])["status"] == MemoryStatus.ACTIVE.value


def test_migrate_drops_twin_contradicting_rejection_with_warning(tmp_path):
    """R2-P1：migrate repoint 不得制造孪生 rejected 行——会咬合劈桶的行丢弃+警告。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    register(t, "OldProj", "agent-lane")
    # 跨身份合法 rejected（agent_lane ↛ OldProj，G 允许创建）
    _insert_legacy_rejected_row(t, "agent_lane", "OldProj")
    updated, warnings, _committed = db.workspaces.migrate_workspace("OldProj", "agent-lane")
    assert updated >= 0
    assert any("dropped while repointing" in w and "spelling variant" in w for w in warnings), warnings
    # 行已丢弃（不再存在会咬合的孪生 rejected），折叠语义恢复
    r = db.resolve_workspace_canonical("agent_lane", None)
    assert r["matched_by"] == "mechanical_variant"
    assert r["canonical"] == "agent-lane"
    # 非孪生 rejected 行照常跟随迁移
    _insert_legacy_rejected_row(t, "unrelated", "agent-lane")
    _u2, w2, _c2 = db.workspaces.migrate_workspace("agent-lane", "MemoryBank")
    assert not any("dropped while repointing" in w for w in w2), w2
    with db.connection() as conn:
        rows = [
            (str(row["alias_workspace"]), str(row["canonical"]), str(row["status"]))
            for row in conn.execute(
                "SELECT alias_workspace,canonical,status FROM workspace_aliases"
            )
        ]
    assert ("unrelated", "MemoryBank", "rejected") in rows


def test_multiple_rejected_rows_across_spellings_all_aggregated(tmp_path):
    """同机械键**不同拼写** alias 的多行 rejected 全量聚合（R2-P2 变异实测钉性缺口）。"""
    t = rules_make_tools(tmp_path, "none")
    db = t.db
    with db.write_transaction() as conn:
        for name in ("AgentLane", "MemoryBank"):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name,created_at) "
                "VALUES(?,datetime('now'))",
                (name,),
            )
    _insert_legacy_rejected_row(t, "agent-chancellor", "AgentLane")
    _insert_legacy_rejected_row(t, "agent_chancellor", "MemoryBank")
    # 原名带空格形态：精确键双 miss，只可能经第二跳聚合
    r = db.resolve_workspace_canonical("agent chancellor", None)
    assert "AgentLane" in r["rejected_canonicals"]
    assert "MemoryBank" in r["rejected_canonicals"]
    assert r["matched_by"] == "new"