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


def _force_undecided_with_candidate(t: MemoryTools, backend, similar_name="金营项目", distance=0.2):
    """Patch resolver → undecided + a candidate, and inject a stub backend.

    The candidate distance defaults to 0.2 (inside workspace_match_distance): a
    real near-miss the vector brought within range, which is the only situation
    where Qwen is allowed to arbitrate an AUTO merge. Over-distance candidates
    (e.g. 0.4) are filtered out before Qwen sees them by design."""
    def fake_resolve(ws_raw, embedder=None, *, match_distance=None, register_new=True):
        return {
            "canonical": ws_raw, "is_new": True, "matched_by": "new",
            "distance": None, "similar": [{"name": similar_name, "distance": distance}],
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


def test_none_high_confidence_normalizes_without_acl_or_confirmed_alias(tmp_path):
    t = make_tools(tmp_path, "none")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.95, "同项目"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    assert r["data"]["workspace_canonical"] == "金营项目"
    assert t.db.get_workspace_alias("金营") is None
    # none stays ACL-free: an unscoped search spans all workspaces, while an
    # explicit filter canonicalizes then scopes that one query (spec §15.6).
    # (The stub backend normalizes both writes to the same canonical.)
    t.memory_write(content="y", workspace="其他", source_type="agent_generated", subject="other")
    assert len(t.memory_search(query="")["data"]["results"]) == 2
    scoped = t.memory_search(query="", workspace="金营项目")["data"]["results"]
    assert {item.get("workspace_canonical") or item["workspace"] for item in scoped} == {"金营项目"}
    assert t.memory_search(query="", workspace="别的项目")["data"]["results"] == []


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


def test_near_miss_registers_only_final_canonical(tmp_path):
    t = make_tools(tmp_path, "weak")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.95, "同项目"))
    _force_undecided_with_candidate(t, backend)
    t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    with t.db.connection() as conn:
        names = {row["name"] for row in conn.execute("SELECT name FROM workspace_canonicals")}
    assert "金营项目" in names
    assert "金营" not in names


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


# ── decision-reason distinguishes model-absent from low-confidence (spec §8) ──

def test_no_backend_reason_is_qwen_unavailable_not_low_conf(tmp_path):
    t = make_tools(tmp_path, "weak")
    _force_undecided_with_candidate(t, None)  # backend absent, candidates exist
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    assert d["workspace_decision"] == "ASK"
    assert d["workspace_decision_reason"] == "qwen_unavailable"


def test_genuine_low_confidence_reason_is_qwen_low_conf(tmp_path):
    t = make_tools(tmp_path, "weak")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.5, "可能相关"))
    _force_undecided_with_candidate(t, backend)
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    assert d["workspace_decision"] == "ASK"
    assert d["workspace_decision_reason"] == "qwen_low_conf"


def test_rejected_candidate_reason_is_qwen_rejected(tmp_path):
    t = make_tools(tmp_path, "none")
    # Qwen suggests a candidate the user already rejected. The rule stays
    # undecided (a second, non-rejected near-miss keeps it a near_miss), so the
    # write path reaches the Qwen branch and must refuse the rejected merge.
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "alias", 0.95, "同项目"))

    def fake_resolve(ws_raw, embedder=None, *, match_distance=None, register_new=True):
        return {
            "canonical": ws_raw, "is_new": True, "matched_by": "new", "distance": None,
            "similar": [
                {"name": "金营项目", "distance": 0.18},
                {"name": "别的项目", "distance": 0.22},
            ],
            "rejected_canonicals": ["金营项目"],
        }
    t.db.resolve_workspace_canonical = fake_resolve  # type: ignore
    t._ensure_semantic_backend = lambda: backend  # type: ignore
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="test")
    d = r["data"]
    # A user-rejected candidate immediately overrides the high-confidence auto.
    assert d["workspace_decision"] == "ASK"
    assert d["workspace_canonical"] == "金营"
    assert d["workspace_decision_reason"] == "qwen_rejected_candidate"


# ── constrained decoding at the inference call site (2026-08-21 dry-run) ─────
#
# A real-library dry-run showed every workspace suggestion failing with
# missing_json: Qwen2.5-0.5B answered in prose ("candidate: AgentLane\nrelation:
# same_family") because suggest_workspace_candidate never passed a
# response_format, unlike classify_pair. The parser tests above could not catch
# it — only the call site can — so assert the schema is wired in and bounded.

def test_workspace_suggester_uses_constrained_decoding() -> None:
    from memory_arbiter.semantic_conflict import (
        LocalGGUFSemanticBackend, _WORKSPACE_PROMPT, _WORKSPACE_RESPONSE_FORMAT,
    )

    schema = _WORKSPACE_RESPONSE_FORMAT["schema"]
    assert _WORKSPACE_RESPONSE_FORMAT["type"] == "json_object"
    assert set(schema["required"]) == {"candidate", "relation", "confidence", "evidence"}
    assert schema["additionalProperties"] is False
    # relation is constrained to the spec's enum, so the model cannot invent one.
    assert set(schema["properties"]["relation"]["enum"]) == {
        "alias", "typo", "same_project", "same_family", "related", "unrelated", "uncertain",
    }
    # evidence is bounded: an unbounded field let the model paste whole memory
    # bodies in and blow past max_tokens, truncating the JSON.
    assert schema["properties"]["evidence"]["maxLength"] == 200
    assert "只输出 JSON" in _WORKSPACE_PROMPT

    captured: dict = {}

    class _Llm:
        @staticmethod
        def create_chat_completion(**kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": '{"candidate":"金营项目","relation":"alias","confidence":0.9,"evidence":"同项目"}'}}]}

    backend = LocalGGUFSemanticBackend.__new__(LocalGGUFSemanticBackend)
    import threading
    backend._infer_lock = threading.Lock()
    backend._cond = threading.Condition(threading.Lock())
    backend._acquire_llm_for_call = lambda: _Llm()          # type: ignore[method-assign]
    backend._release_llm_for_call = lambda: None            # type: ignore[method-assign]

    signal = backend.suggest_workspace_candidate(
        "金营", {"title": "金营项目排期", "key_sentences": ["交付计划"]}, ["金营项目"],
    )
    assert captured.get("response_format") is _WORKSPACE_RESPONSE_FORMAT
    assert signal.candidate == "金营项目"
    assert signal.relation == "alias"
    assert signal.error is None


# ── Qwen only arbitrates in-threshold candidates (2026-08-21 real-lib A/B) ───
#
# A dry-run over the real 547-memory library had Qwen answer
# "same_project@0.95" merging openclaw into proto-test at cosine 0.357 — far
# past the 0.25 threshold — because the AUTO gate looked at Qwen confidence but
# not vector distance. Fix: _suggest_workspace_candidate drops candidates beyond
# workspace_qwen_candidate_distance before Qwen sees them, so an over-distance
# name can never be resurrected into an AUTO merge.

def test_over_distance_candidate_is_filtered_before_qwen(tmp_path):
    t = make_tools(tmp_path, "none")
    calls = []

    class _SpyBackend:
        def suggest_workspace_candidate(self, ws_raw, evidence, candidates, **kw):
            calls.append(list(candidates))
            return WorkspaceCandidateSignal(candidates[0] if candidates else None,
                                            "same_project", 0.95, "hallucinated")

    # Only over-distance neighbors (0.357 > 0.25): Qwen must not even be asked.
    _force_undecided_with_candidate(t, _SpyBackend(), similar_name="proto-test", distance=0.357)
    r = t.memory_write(content="x", workspace="openclaw", source_type="agent_generated", subject="s")
    assert calls == []  # no candidate survived the distance bound
    assert r["data"]["workspace_canonical"] == "openclaw"  # stays NEW, not merged
    assert r["data"]["workspace_decision"] == "ASK"


def test_in_threshold_candidate_still_reaches_qwen_and_auto_merges(tmp_path):
    t = make_tools(tmp_path, "none")
    backend = _StubBackend(WorkspaceCandidateSignal("金营项目", "same_project", 0.95, "同项目"))
    _force_undecided_with_candidate(t, backend, distance=0.2)  # inside threshold
    r = t.memory_write(content="x", workspace="金营", source_type="agent_generated", subject="s")
    assert r["data"]["workspace_canonical"] == "金营项目"
    assert r["data"]["workspace_decision"] == "AUTO"


def test_qwen_candidate_pool_capped_at_top_k(tmp_path):
    t = make_tools(tmp_path, "none")
    seen = []

    class _SpyBackend:
        def suggest_workspace_candidate(self, ws_raw, evidence, candidates, **kw):
            seen.append(list(candidates))
            return WorkspaceCandidateSignal(None, "uncertain", None, "")

    def fake_resolve(ws_raw, embedder=None, *, match_distance=None, register_new=True):
        # Five in-threshold neighbors; only top-3 should reach Qwen.
        return {"canonical": ws_raw, "is_new": True, "matched_by": "new", "distance": None,
                "similar": [{"name": f"c{i}", "distance": 0.10 + i * 0.02} for i in range(5)],
                "rejected_canonicals": []}
    t.db.resolve_workspace_canonical = fake_resolve  # type: ignore
    t._ensure_semantic_backend = lambda: _SpyBackend()  # type: ignore
    t.memory_write(content="x", workspace="w", source_type="agent_generated", subject="s")
    assert seen and len(seen[0]) == 3
    assert seen[0] == ["c0", "c1", "c2"]
