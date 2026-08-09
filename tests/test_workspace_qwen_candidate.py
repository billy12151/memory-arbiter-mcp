"""Qwen/local-model workspace candidate suggester (design 636 §6, §7, §9).

The real GGUF model is optional; these tests exercise the parser and the
per-isolation policy with a stub backend so they run without a model file.
"""
from pathlib import Path

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools
from memory_arbiter.semantic_conflict import (
    WorkspaceCandidateSignal,
    workspace_candidate_from_text,
)


# ── parser ───────────────────────────────────────────────────────────────────

def test_parse_valid_candidate():
    raw = '{"candidate": "金营项目", "relation": "alias", "confidence": 0.92, "evidence": "同一项目不同写法"}'
    sig = workspace_candidate_from_text(raw, ["金营项目", "其他项目"])
    assert sig.candidate == "金营项目"
    assert sig.relation == "alias"
    assert sig.confidence == 0.92


def test_parse_drops_hallucinated_candidate():
    # model returns a candidate not in the offered list → dropped
    raw = '{"candidate": "不存在项目", "relation": "alias", "confidence": 0.9}'
    sig = workspace_candidate_from_text(raw, ["金营项目"])
    assert sig.candidate is None
    assert sig.relation == "uncertain"  # downgraded since no candidate


def test_parse_missing_json_is_uncertain():
    sig = workspace_candidate_from_text("no json here", ["a"])
    assert sig.candidate is None and sig.relation == "uncertain"
    assert sig.error == "missing_json"


def test_parse_unknown_relation_normalized():
    raw = '{"candidate": "a", "relation": "bogus", "confidence": 0.5}'
    sig = workspace_candidate_from_text(raw, ["a"])
    assert sig.relation == "uncertain"


def test_parse_clamps_out_of_range_confidence():
    hi = workspace_candidate_from_text('{"candidate":"a","relation":"alias","confidence":5.0}', ["a"])
    assert hi.confidence == 1.0
    lo = workspace_candidate_from_text('{"candidate":"a","relation":"alias","confidence":-3}', ["a"])
    assert lo.confidence == 0.0


# ── per-isolation policy (stub backend) ──────────────────────────────────────

class _StubBackend:
    """Minimal stand-in for LocalGGUFSemanticBackend."""
    def __init__(self, signal: WorkspaceCandidateSignal):
        self._signal = signal

    def suggest_workspace_candidate(self, ws_raw, evidence, candidates):
        return self._signal


def make_tools(tmp_path: Path, isolation: str) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "qwen.sqlite3",
        backup_jsonl=tmp_path / "qwen.jsonl",
        client="codex", agent_id="agent-a", workspace="default",
        enable_sqlite_vec=False, vec_dim=2, isolation=isolation,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _force_undecided_with_candidate(t: MemoryTools, backend, similar_name="金营项目"):
    """Patch resolver → undecided + a candidate, and inject a stub backend."""
    def fake_resolve(ws_raw, embedder=None, *, match_distance=None, register_new=True):
        return {
            "canonical": ws_raw, "is_new": True, "matched_by": "new",
            "distance": None, "similar": [{"name": similar_name, "distance": 0.4}],
            "rejected_canonicals": [],
        }
    t.db.resolve_workspace_canonical = fake_resolve  # type: ignore
    t._ensure_semantic_backend = lambda: backend  # type: ignore


def test_weak_high_confidence_silent_merge(tmp_path):
    t = make_tools(tmp_path, "weak")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.95, "同项目"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    assert d["workspace_decision"] == "AUTO"
    assert d["workspace_canonical"] == "金营项目"
    assert d["workspace_matched_by"] == "qwen"


def test_weak_low_confidence_asks_not_merge(tmp_path):
    t = make_tools(tmp_path, "weak")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "related", 0.5, "可能相关"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    assert d["workspace_decision"] == "ASK"
    # NOT silently merged
    assert d["workspace_canonical"] != "金营项目"
    assert d.get("write_hints", {}).get("workspace_review")


def test_strict_never_silent_merges_even_high_conf(tmp_path):
    t = make_tools(tmp_path, "strict")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.99, "同项目"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    # strict: high-conf candidate does NOT auto-merge; memory stays pending
    assert d["workspace_canonical"] != "金营项目"
    assert d.get("action_required") == "confirm_new_workspace"


def test_no_backend_falls_back_to_ask(tmp_path):
    t = make_tools(tmp_path, "weak")
    _force_undecided_with_candidate(t, None)  # no backend
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    assert r["data"]["workspace_decision"] == "ASK"


# ── strict-switch governance advisory (636 §9) ───────────────────────────────

def test_strict_emits_governance_advisory(monkeypatch):
    monkeypatch.setenv("MEMORY_ARBITER_ISOLATION", "strict")
    s = Settings.from_env()
    assert any("memory_govern" in w for w in s.config_warnings)
