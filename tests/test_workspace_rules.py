"""Rule-first workspace decision layer (design 636 §2, §3, §5)."""
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools
from memory_arbiter import workspace_rules as wr


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

def make_tools(tmp_path: Path, isolation: str = "weak") -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "rules.sqlite3",
        backup_jsonl=tmp_path / "rules.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        enable_sqlite_vec=False, vec_dim=2, isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def test_write_surfaces_ask_for_generic_workspace(tmp_path):
    t = make_tools(tmp_path)
    r = t.memory_write(content="some plan", workspace="月报", source_type="agent_generated", subject="test")
    data = r["data"]
    assert data["workspace_decision"] == "ASK"
    assert data.get("write_hints", {}).get("workspace_review")


def test_write_auto_for_specific_workspace(tmp_path):
    t = make_tools(tmp_path)
    r = t.memory_write(content="alpha", workspace="金营项目", source_type="agent_generated", subject="test")
    assert r["data"]["workspace_decision"] == "AUTO"
