from pathlib import Path
import hashlib
import pytest
from memory_arbiter.acl import raw_workspace
from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import ConflictMember, ConflictValueGroup, MemoryStatus
from memory_arbiter.tools import MemoryTools
from tests.test_workspace_recall_isolation import _VEC_AVAILABLE

@pytest.fixture(autouse=True)
def _relevance_floor_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0.15.9: the tests in this file exercise scoping/preview/metering with
    # content-only fixtures; the query-recall relevance floor is orthogonal to
    # what they pin (its behavior lives in tests/test_recall_quality_0159.py).
    # Disable it here so weak fixtures stay recallable.
    monkeypatch.setattr("memory_arbiter.search.QUERY_RECALL_SCORE_FLOOR", -1.0)

"""default 双向绝缘 (mema 721 期0).

The reserved default pool ("", default/默认/none/null/unknown/未知, case-
insensitive) must be insulated from the vector/alias system in BOTH
directions: no workspace may be merged INTO default by KNN AUTO-merge, no
canonical vector may ever be published for a default term, default synonyms
resolve to the single global pool, and alias governance (accept/reject/
rename/migrate) refuses any pair touching default.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import (
    DEFAULT_TERMS,
    DEFAULT_WORKSPACE_NAME,
    is_default_workspace_term,
    )
from memory_arbiter.db import MemoryDB
from memory_arbiter.tools import MemoryTools

NON_EMPTY_DEFAULT_TERMS = sorted(t for t in DEFAULT_TERMS if t)

try:
    import sqlite_vec  # type: ignore  # noqa: F401
    _VEC_AVAILABLE = True
except Exception:
    _VEC_AVAILABLE = False

requires_vec = pytest.mark.skipif(not _VEC_AVAILABLE, reason="sqlite-vec not installed")


def default_insulation_make_tools(tmp_path: Path, isolation: str = "none", *, vec: bool = False) -> MemoryTools:
    # vec=True points at a (fake) GGUF model — the model path IS the intent
    # since 0.15.0 — and mirrors the first successful embedder build by
    # creating the lazy vec0 tables at dim 2.
    model = tmp_path / "fake.gguf"
    settings = Settings(
        db_path=tmp_path / "ins.sqlite3",
        backup_jsonl=tmp_path / "ins.jsonl",
        client="codex",
        agent_id="agent-a",
        workspace="default",
        embedding_model_path=model if vec else None,
        isolation=isolation,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    if vec:
        model.write_bytes(b"fake")
        assert db.ensure_vec_tables(2) == []
    return tools


class FixedEmbedder:
    def __init__(self, vector: list[float]):
        self.vector = vector

    def embed_text(self, prefix: str = "", body: str = ""):
        return SimpleNamespace(embedding=list(self.vector))


def _default_insulation_write(tools: MemoryTools, content: str, workspace: str = "", subject: str = "test") -> dict:
    return tools.memory_write(
        content=content, workspace=workspace, subject=subject,
        source_type="agent_generated",
    )


def _register_canonical_with_vector(tools: MemoryTools, name: str, vector: list[float]) -> None:
    with tools.db.write_transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
            (name,),
        )
        row = conn.execute(
            "SELECT id FROM workspace_canonicals WHERE name = ?", (name,)
        ).fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO workspace_canonicals_vec(id, embedding) VALUES (?, ?)",
            (int(row["id"]), json.dumps(vector)),
        )


def _canonical_vec_row(tools: MemoryTools, name: str):
    with tools.db.connection() as conn:
        return conn.execute(
            "SELECT v.id FROM workspace_canonicals c "
            "JOIN workspace_canonicals_vec v ON v.id = c.id WHERE c.name = ?",
            (name,),
        ).fetchone()


# ── KNN exclusion (改动1) ────────────────────────────────────────────────────

@requires_vec
def test_knn_never_merges_into_default_even_with_published_vector(tmp_path):
    tools = default_insulation_make_tools(tmp_path, vec=True)
    assert tools.db.state.sqlite_vec_available
    # Simulate a legacy/foreign backfill that force-published a default vector.
    _register_canonical_with_vector(tools, "default", [1.0, 0.0])
    assert _canonical_vec_row(tools, "default") is not None

    # Control: a normal canonical at distance 0 does attract the AUTO merge —
    # proving the embedder/distances work and default is excluded by name.
    _register_canonical_with_vector(tools, "claw", [0.0, 1.0])
    merged = tools.db.resolve_workspace_canonical(
        "clawproj", FixedEmbedder([0.0, 1.0]),
    )
    assert merged["matched_by"] == "vector"
    assert merged["canonical"] == "claw"

    # A name embedding to distance ~0 FROM DEFAULT must stay NEW: the only
    # ≤cutoff neighbour is the excluded default row (claw sits at 1.0).
    resolved = tools.db.resolve_workspace_canonical(
        "defaultproj", FixedEmbedder([1.0, 0.0]),
    )
    assert resolved["matched_by"] == "new"
    assert resolved["canonical"] == "defaultproj"
    assert resolved["is_new"] is True
    assert "default" not in [s["name"] for s in resolved["similar"]]


@requires_vec
@pytest.mark.parametrize("term", NON_EMPTY_DEFAULT_TERMS)
def test_knn_excludes_every_default_synonym_canonical(tmp_path, term):
    tools = default_insulation_make_tools(tmp_path, vec=True)
    # Even a legacy DB that registered a synonym as its own canonical (with a
    # vector) can never attract merges: the candidate SQL excludes all terms.
    _register_canonical_with_vector(tools, term, [1.0, 0.0])
    resolved = tools.db.resolve_workspace_canonical(
        f"{term}project", FixedEmbedder([1.0, 0.0]),
    )
    assert resolved["matched_by"] == "new"
    assert term not in [s["name"] for s in resolved["similar"]]
# ── synonym resolution (2c) ──────────────────────────────────────────────────

@pytest.mark.parametrize("term", NON_EMPTY_DEFAULT_TERMS + ["Default", "DEFAULT", " None ", "未知 "])
def test_default_synonyms_resolve_to_single_pool(tmp_path, term):
    tools = default_insulation_make_tools(tmp_path)
    resolved = tools.db.resolve_workspace_canonical(term, None)
    assert resolved["canonical"] == DEFAULT_WORKSPACE_NAME
    assert resolved["matched_by"] == "fallback"
    assert resolved["is_new"] is False
    with tools.db.connection() as conn:
        names = [r["name"] for r in conn.execute("SELECT name FROM workspace_canonicals").fetchall()]
    assert term.strip() not in names  # no phantom synonym canonical


# ── vector publish insulation (改动2) ────────────────────────────────────────

@requires_vec
def test_vector_publish_paths_skip_default_terms(tmp_path):
    tools = default_insulation_make_tools(tmp_path, vec=True)
    store = tools.db.workspaces
    embedder = FixedEmbedder([1.0, 0.0])
    with tools.db.write_transaction() as conn:
        for name in ("default", "默认", "null", "projx"):
            conn.execute(
                "INSERT OR IGNORE INTO workspace_canonicals(name, created_at) VALUES (?, '2026-01-01T00:00:00Z')",
                (name,),
            )
    result = {"warnings": [], "vector_publish_pending": False}

    # 2026-10-05：一步式 _publish_missing_workspace_canonical_vector 已随死代码
    # 清理移除（生产零调用，写路径自 P2 #9 起走 prepare+publish 两步）；
    # default 绝缘断言由下方两步 API 覆盖。
    assert store.prepare_missing_workspace_canonical_embedding("默认", embedder) is None
    assert store.prepare_workspace_canonical_embedding("null", embedder) is None
    assert store.publish_workspace_canonical_vector("default", [1.0, 0.0]) == []
    for name in ("default", "默认", "null"):
        assert _canonical_vec_row(tools, name) is None

    # Control: a normal canonical publishes through the same path.
    assert store.publish_workspace_canonical_vector("projx", [1.0, 0.0]) == []
    assert _canonical_vec_row(tools, "projx") is not None


# ── governance guards (改动3) ────────────────────────────────────────────────

@pytest.mark.parametrize("term", NON_EMPTY_DEFAULT_TERMS)
def test_rename_refuses_default_in_both_directions(tmp_path, term):
    tools = default_insulation_make_tools(tmp_path)
    _default_insulation_write(tools, "projX memory", "projX")

    updated, warnings, _committed = tools.db.rename_workspace_canonical("projX", term)
    assert updated == 0
    assert warnings and "reserved" in warnings[0]
    updated, warnings, _committed = tools.db.rename_workspace_canonical(term, "projY")
    assert updated == 0
    assert warnings and "reserved" in warnings[0]

    r = tools.memory_govern("rename_workspace_canonical", {
        "workspace": "default",
        "old": "projX", "new": term, "reason": "try merge into default", "authorized": True,
    })
    assert r["ok"] is False
    assert r["data"]["renamed"] is False
    assert any("reserved" in w for w in r["warnings"])


@pytest.mark.parametrize("term", NON_EMPTY_DEFAULT_TERMS)
def test_removed_pairwise_actions_and_internal_decisions_never_touch_default(tmp_path, term):
    tools = default_insulation_make_tools(tmp_path)
    _default_insulation_write(tools, "projX memory", "projX")

    for action, alias, canonical in (
        ("accept_workspace_alias", term, "projX"),
        ("accept_workspace_alias", "projX", term),
        ("reject_workspace_alias", term, "projX"),
    ):
        result = tools.memory_govern(action, {
            "alias": alias, "canonical": canonical, "authorized": True,
        })
        assert result["ok"] is False
        assert result["data"]["error_code"] == "workspace_alias_action_removed"

    assert tools.db.record_workspace_decision(term, "projX", status="confirmed")[0] is False
    assert tools.db.record_workspace_decision("projX", term, status="confirmed")[0] is False
    with tools.db.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM workspace_aliases WHERE canonical = ? OR alias_workspace = ?",
            (term, term),
        ).fetchone()[0]
    assert count == 0


def test_migrate_refuses_default_in_both_directions(tmp_path):
    tools = default_insulation_make_tools(tmp_path)
    _default_insulation_write(tools, "projX memory", "projX")
    updated, warnings, _committed = tools.db.migrate_workspace("projX", "default")
    assert updated == 0 and warnings
    updated, warnings, _committed = tools.db.migrate_workspace("默认", "projX")
    assert updated == 0 and warnings


# ── regression: default writes/recall keep working ──────────────────────────

def test_default_write_and_synonym_recall_regression(tmp_path):
    tools = default_insulation_make_tools(tmp_path)
    w1 = _default_insulation_write(tools, "global fact one", "")
    w2 = _default_insulation_write(tools, "synonym fact two", "默认")
    assert w1["data"]["workspace_canonical"] == DEFAULT_WORKSPACE_NAME
    assert w2["data"]["workspace_canonical"] == DEFAULT_WORKSPACE_NAME

    found = tools.memory_search(query="fact", workspace="默认")
    ids = [r["id"] for r in found["data"]["results"]]
    assert w1["data"]["id"] in ids
    assert w2["data"]["id"] in ids


@requires_vec
def test_placement_hint_still_fires_for_default_synonym(tmp_path, monkeypatch):
    tools = default_insulation_make_tools(tmp_path, vec=True)  # placement hint requires sqlite_vec
    proj = _default_insulation_write(tools, "projA memory", "projA", subject="projA subject")
    monkeypatch.setattr(tools, "_ensure_embedder", lambda: (FixedEmbedder([1.0, 0.0]), []))
    monkeypatch.setattr(
        tools.db, "row_knn",
        lambda emb, k=8: [{"memory_id": proj["data"]["id"], "distance": 0.1,
                            "status": "active", "workspace": "projA",
                            "workspace_canonical": "projA"}],
    )

    w = _default_insulation_write(tools, "editor preference", "默认", subject="editor preference")
    assert w["data"]["workspace_canonical"] == DEFAULT_WORKSPACE_NAME
    hints = w["data"].get("write_hints") or {}
    assert hints["placement_suggestion"]["suggested_workspace"] == "projA"


# ── regression: full-width IME spellings fold into the global pool ──────────

def test_full_width_default_spellings_are_default_terms():
    # NFKC folds full-width IME spellings onto their ASCII twins before the
    # synonym comparison; without it ｄｅｆａｕｌｔ registers a phantom second
    # default pool instead of landing in the global one.
    assert is_default_workspace_term("ｄｅｆａｕｌｔ")
    assert is_default_workspace_term("ＮＵＬＬ")
    assert is_default_workspace_term("　默认　")  # U+3000 spaces are stripped
    # boundary: full-width PROJECT names are still real workspaces, and
    # supersets of "default" are not synonyms
    assert not is_default_workspace_term("ｐｒｏｊ")
    assert not is_default_workspace_term("defaulted")


def test_full_width_default_write_lands_in_global_pool(tmp_path):
    tools = default_insulation_make_tools(tmp_path)
    written = _default_insulation_write(tools, "full-width default write", "ｄｅｆａｕｌｔ", subject="nfkc")
    assert written["ok"]
    assert written["data"]["workspace_canonical"] == DEFAULT_WORKSPACE_NAME
    with tools.db.connection() as conn:
        names = [str(row["name"]) for row in conn.execute("SELECT name FROM workspace_canonicals")]
    assert "ｄｅｆａｕｌｔ" not in names


def test_move_refuses_full_width_default_destination(tmp_path):
    tools = default_insulation_make_tools(tmp_path)
    memory_id = int(_default_insulation_write(tools, "x", "proj-a", subject="s")["data"]["id"])
    outcome = tools.memory_govern("move_memories_workspace", {
        "workspace": "default",
        "memory_ids": [memory_id], "new_workspace": "ＮＵＬＬ", "authorized": True,
    })
    assert not outcome["ok"]
    assert "reserved global pool" in outcome["data"]["error"]
    assert tools.db.get_memory(memory_id)["workspace"] == "proj-a"


# ── from test_workspace_weak_vector_weight.py ──
# helpers renamed: weak_vector_weight_make_tools, _weak_vector_weight_write (collisions)
