"""weak 连续向量加权 (mema 721 期1) + §2a 共享准入 helper.

Covers the pure helpers (guarded distance, admission predicate, weight
curve), the _workspace_bonus curve/binary fallbacks, and the search-path
wiring: distance_map precompute behind the workspace_weak_vector_weight flag,
ranking lift for a near workspace, flag-off binary regression, and magnitude
discipline (the nudge never overrides a subject/tags hit).
"""
from pathlib import Path

import pytest

from memory_arbiter import workspace_rules as wr
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.search import _workspace_bonus
from memory_arbiter.tools import MemoryTools


def make_tools(tmp_path: Path, isolation: str = "weak", *, weak_vector: bool = True) -> MemoryTools:
    settings = Settings(
        db_path=tmp_path / "wv.sqlite3",
        backup_jsonl=tmp_path / "wv.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        enable_sqlite_vec=False,
        vec_dim=2,
        isolation=isolation,
        workspace_weak_vector_weight=weak_vector,
    )
    return MemoryTools(settings=settings, db=MemoryDB(settings))


def _write(tools: MemoryTools, content: str, workspace: str, subject: str) -> int:
    return tools.memory_write(
        content=content, workspace=workspace, subject=subject,
        source_type="agent_generated",
    )["data"]["id"]


# ── §2a pure helpers ─────────────────────────────────────────────────────────

def test_workspace_vector_distance_guards():
    dmap = {"agent-rail": 0.142}
    assert wr.workspace_vector_distance("agent-lane", "agent-lane", dmap) == 0.0
    assert wr.workspace_vector_distance("agent-lane", "agent-rail", dmap) == pytest.approx(0.142)
    # no map / missing entry → exact-equality fallback for the caller
    assert wr.workspace_vector_distance("agent-lane", "agent-rail", None) is None
    assert wr.workspace_vector_distance("agent-lane", "agent-rail", {}) is None
    assert wr.workspace_vector_distance("agent-lane", "unknown-ws", dmap) is None
    # default insulation (both sides, every synonym)
    for term in ("default", "默认", "none", "null", "unknown", "未知"):
        assert wr.workspace_vector_distance(term, "agent-rail", dmap) is None
        assert wr.workspace_vector_distance("agent-lane", term, dmap) is None
    # short-name guard
    assert wr.workspace_vector_distance("w", "agent-rail", dmap) is None
    assert wr.workspace_vector_distance("agent-lane", "w", dmap) is None
    # substring / generic-token proximity guard (721 §3d hazards)
    assert wr.workspace_vector_distance(
        "main", "openclaw-main", {"openclaw-main": 0.132},
    ) is None
    assert wr.workspace_vector_distance(
        "project-alpha", "project-beta", {"project-beta": 0.1},
    ) is None
    # real same-project pair is NOT suppressed (no containment, no generic-only overlap)
    assert wr.workspace_vector_distance(
        "agent-rail", "agent-lane", {"agent-lane": 0.142},
    ) == pytest.approx(0.142)
    assert wr.workspace_vector_distance(
        "金营项目", "金科营销项目", {"金科营销项目": 0.16},
    ) == pytest.approx(0.16)
    # configurable min_name_len
    assert wr.workspace_vector_distance("abc", "wxyz", {"wxyz": 0.2}, min_name_len=4) is None
    assert wr.workspace_vector_distance("abcd", "wxyz", {"wxyz": 0.2}, min_name_len=4) == pytest.approx(0.2)


def test_workspace_admit_cutoff():
    dmap = {"agent-rail": 0.142, "openclaw-main": 0.132, "far-ws": 0.364}
    assert wr.workspace_admit("agent-lane", "agent-rail", dmap, 0.25) is True
    assert wr.workspace_admit("agent-lane", "agent-lane", dmap, 0.25) is True
    assert wr.workspace_admit("agent-lane", "far-ws", dmap, 0.25) is False
    assert wr.workspace_admit("main", "openclaw-main", dmap, 0.25) is False
    assert wr.workspace_admit("default", "agent-rail", dmap, 0.25) is False
    assert wr.workspace_admit("agent-lane", "w", dmap, 0.25) is False
    assert wr.workspace_admit("agent-lane", "missing", dmap, 0.25) is False


def test_weak_curve_anchor_points():
    assert wr.weak_workspace_vector_weight(0.0) == pytest.approx(0.30)
    assert wr.weak_workspace_vector_weight(0.10) == pytest.approx(0.30)
    assert wr.weak_workspace_vector_weight(0.142) == pytest.approx(0.30)  # full-bonus zone
    assert wr.weak_workspace_vector_weight(0.15) == pytest.approx(0.30)
    assert wr.weak_workspace_vector_weight(0.20) == pytest.approx(0.20)  # decayed, in (0, 0.30)
    assert wr.weak_workspace_vector_weight(0.225) == pytest.approx(0.15)
    assert wr.weak_workspace_vector_weight(0.30) == pytest.approx(0.0)
    assert wr.weak_workspace_vector_weight(0.364) == 0.0
    assert wr.weak_workspace_vector_weight(0.9) == 0.0


def test_weak_curve_cap_below_subject_medium():
    # 满分锚 0.30 不盖过 subject-medium (6.0)
    assert wr.WEAK_VECTOR_WEIGHT_MAX < 6.0 / 10


# ── _workspace_bonus curve + fallbacks ───────────────────────────────────────

def test_workspace_bonus_curve_and_fallbacks():
    dmap = {"agent-rail": 0.20, "far-ws": 0.40}
    rec_rail = {"workspace_canonical": "agent-rail"}
    rec_far = {"workspace_canonical": "far-ws"}
    rec_def = {"workspace_canonical": "default"}

    assert _workspace_bonus(rec_rail, "agent-lane", "weak", distance_map=dmap) == pytest.approx(0.20)
    # known-far cross-workspace: ≈0, no -0.15 hard penalty
    assert _workspace_bonus(rec_far, "agent-lane", "weak", distance_map=dmap) == 0.0
    # no map → exact v0.9.7 binary step
    assert _workspace_bonus(rec_rail, "agent-lane", "weak") == -0.15
    assert _workspace_bonus(rec_rail, "agent-rail", "weak") == 0.30
    # default insulation → binary fallback even with a map entry
    assert _workspace_bonus(rec_def, "agent-lane", "weak", distance_map={"default": 0.0}) == -0.15
    assert _workspace_bonus(rec_def, "default", "weak", distance_map={"default": 0.0}) == 0.30
    # guarded pair inside the map → binary fallback
    assert _workspace_bonus(
        {"workspace_canonical": "openclaw-main"}, "main", "weak",
        distance_map={"openclaw-main": 0.132},
    ) == -0.15
    # 0 outside weak mode
    assert _workspace_bonus(rec_rail, "agent-lane", "strict", distance_map=dmap) == 0.0
    assert _workspace_bonus(rec_rail, "agent-lane", "none", distance_map=dmap) == 0.0
    assert _workspace_bonus(rec_rail, None, "weak", distance_map=dmap) == 0.0


# ── search wiring ────────────────────────────────────────────────────────────

def test_search_lifts_near_workspace_with_vector_weight(tmp_path, monkeypatch):
    tools = make_tools(tmp_path, "weak", weak_vector=True)
    near = _write(tools, "release notes near", "agent-rail", "shared release notes")
    _write(tools, "release notes far", "unrelated-ws", "shared release notes")
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map",
        lambda query, names: {"agent-rail": 0.20, "unrelated-ws": 0.9},
    )

    r = tools.memory_search(query="shared release notes", workspace="agent-lane", debug_ranking=True)
    results = r["data"]["results"]
    assert results[0]["id"] == near
    boosted = next(x for x in results if x["id"] == near)
    assert boosted["_workspace_bonus"] == pytest.approx(0.20)  # decayed positive weight


def test_search_full_bonus_zone_uses_max(tmp_path, monkeypatch):
    tools = make_tools(tmp_path, "weak", weak_vector=True)
    near = _write(tools, "rail memory", "agent-rail", "shared release notes")
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map",
        lambda query, names: {"agent-rail": 0.142},
    )
    r = tools.memory_search(query="shared release notes", workspace="agent-lane", debug_ranking=True)
    boosted = next(x for x in r["data"]["results"] if x["id"] == near)
    assert boosted["_workspace_bonus"] == pytest.approx(0.30)


def test_search_never_computes_map_when_flag_off(tmp_path, monkeypatch):
    tools = make_tools(tmp_path, "weak", weak_vector=False)
    _write(tools, "release notes", "agent-rail", "shared release notes")
    calls: list[str] = []

    def spy(query, names):
        calls.append(str(query))
        return {"agent-rail": 0.2}

    monkeypatch.setattr(tools.db.workspaces, "canonical_distance_map", spy)
    r = tools.memory_search(query="shared release notes", workspace="agent-lane", debug_ranking=True)
    assert calls == []  # off = exact binary behaviour, map never requested
    boosted = next(x for x in r["data"]["results"])
    assert boosted["_workspace_bonus"] in (0.30, -0.15)


def test_degraded_map_falls_back_to_binary(tmp_path, monkeypatch):
    tools = make_tools(tmp_path, "weak", weak_vector=True)
    _write(tools, "release notes", "agent-rail", "shared release notes")
    # Flag on but sqlite-vec degraded → canonical_distance_map returns {} →
    # per-record guard fallback to the binary step.
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map", lambda query, names: {},
    )
    r = tools.memory_search(query="shared release notes", workspace="agent-lane", debug_ranking=True)
    assert r["data"]["results"]
    assert r["data"]["results"][0]["_workspace_bonus"] == -0.15


def test_default_query_never_uses_vector_weight(tmp_path, monkeypatch):
    tools = make_tools(tmp_path, "weak", weak_vector=True)
    _write(tools, "global memory", "", "shared release notes")
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map",
        lambda query, names: pytest.fail("default query must not enter the vector system"),
    )
    r = tools.memory_search(query="shared release notes", workspace="默认")
    assert r["data"]["results"]
    assert r["data"]["results"][0]["id"]


def test_vector_weight_never_overrides_subject_hit(tmp_path, monkeypatch):
    tools = make_tools(tmp_path, "weak", weak_vector=True)
    subject_hit = _write(tools, "plain content", "unrelated-ws", "deploy checklist")
    content_only = _write(tools, "mentions deploy checklist deep inside", "agent-rail", "misc notes")
    monkeypatch.setattr(
        tools.db.workspaces, "canonical_distance_map",
        lambda query, names: {"agent-rail": 0.10, "unrelated-ws": 0.9},
    )
    r = tools.memory_search(query="deploy checklist", workspace="agent-lane")
    ids = [x["id"] for x in r["data"]["results"]]
    assert subject_hit in ids and content_only in ids
    assert ids.index(subject_hit) < ids.index(content_only)


# ── canonical_distance_map (2b) against a real vec table ────────────────────

def test_canonical_distance_map_one_query(tmp_path):
    import json as _json

    settings = Settings(
        db_path=tmp_path / "dm.sqlite3",
        backup_jsonl=tmp_path / "dm.jsonl",
        enable_sqlite_vec=True,
        vec_dim=2,
        client="codex",
        agent_id="agent-a",
        workspace="default",
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    if not tools.db.state.sqlite_vec_available:  # pragma: no cover - env without sqlite-vec
        pytest.skip("sqlite-vec unavailable")
    with tools.db.write_transaction() as conn:
        for name, vector in (
            ("agent-lane", [1.0, 0.0]),
            ("agent-rail", [0.99, 0.141]),   # cosine distance ≈ 0.01-0.02
            ("orthogonal-ws", [0.0, 1.0]),   # cosine distance = 1.0
        ):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
                (name,),
            )
            row = conn.execute("SELECT id FROM workspace_canonicals WHERE name = ?", (name,)).fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                (int(row["id"]), _json.dumps(vector)),
            )

    dmap = tools.db.workspaces.canonical_distance_map("agent-lane", ["agent-rail", "orthogonal-ws"])
    assert set(dmap) == {"agent-rail", "orthogonal-ws"}
    assert dmap["agent-rail"] == pytest.approx(0.0, abs=0.05)
    assert dmap["orthogonal-ws"] == pytest.approx(1.0, abs=0.05)
    # query canonical without a vector → degraded {}
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES ('vecless', '2026-01-01T00:00:00Z')",
        )
    assert tools.db.workspaces.canonical_distance_map("vecless", ["agent-rail"]) == {}
    # default query never enters
    assert tools.db.workspaces.canonical_distance_map("default", ["agent-rail"]) == {}


def test_distance_map_skips_null_distance_instead_of_raising(tmp_path):
    """Round-1 review fix: degenerate (all-zero) vectors make sqlite-vec return
    SQL NULL for vec_distance_cosine — the map must skip that canonical
    (vectorless → binary fallback), never raise TypeError through search."""
    import json as _json

    settings = Settings(
        db_path=tmp_path / "null.sqlite3",
        backup_jsonl=tmp_path / "null.jsonl",
        enable_sqlite_vec=True,
        vec_dim=2,
        client="codex",
        agent_id="agent-a",
        workspace="default",
        isolation="weak",
        workspace_weak_vector_weight=True,
    )
    tools = MemoryTools(settings=settings, db=MemoryDB(settings))
    if not tools.db.state.sqlite_vec_available:  # pragma: no cover
        pytest.skip("sqlite-vec unavailable")
    with tools.db.write_transaction() as conn:
        for name, vector in (
            ("agent-lane", [1.0, 0.0]),
            ("zero-ws", [0.0, 0.0]),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
                (name,),
            )
            row = conn.execute("SELECT id FROM workspace_canonicals WHERE name = ?", (name,)).fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
                (int(row["id"]), _json.dumps(vector)),
            )

    dmap = tools.db.workspaces.canonical_distance_map("agent-lane", ["zero-ws"])
    assert dmap == {}  # NULL distance skipped, not raised

    tools.memory_write(
        content="zero workspace memory", workspace="zero-ws",
        subject="null distance subject", source_type="agent_generated",
    )
    result = tools.memory_search(query="null distance subject", workspace="agent-lane")
    assert result["ok"] is True  # search never raises; zero-ws record still reachable
