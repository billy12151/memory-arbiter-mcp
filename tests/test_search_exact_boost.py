"""Gate-v2 G2: the exact-hit fusion guarantee (owner 拍板 6).

A memory whose subject IS the query (casefold + whitespace-stripped), or
whose best evidence row's TRUE cosine clears COS_EXACT_BOOST, must win the
page — RRF rank-flattening (#91: cosine 1.0, KNN first, find-rank 15) may
not bury it. The boost lands in the fusion ring AND is exempt from the
lexical/evidence quota trim (original-rank admission would cut an
exact match that entered the pool through a late channel).

Embedder discipline: a directed dim-2 fake — "alpha" bodies point east,
"beta" bodies point north, everything else sits on the diagonal. Query
embeddings ride embedding_auto_query (default True) so the real search
path computes them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.db import MemoryDB
from memory_arbiter.embedder import EmbedResult
from memory_arbiter.models import MemoryRecord
from memory_arbiter.tools import MemoryTools

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _hashed_unit_vector(text: str) -> "list[float]":
    # Deterministic pseudo-random direction: identical text → identical
    # vector (cosine 1.0), different text → near-orthogonal (a 2-dim fake
    # would collapse unrelated texts onto one direction and fire the
    # cosine leg spuriously; 8 hashed dims keep discrimination real).
    import math
    import zlib
    seed = zlib.crc32(text.encode("utf-8"))
    raw = [((seed >> (3 * i)) & 7) + 1 for i in range(8)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


class _DirectedEmbedder:
    embedding_space_id = "gate-v2-g2-exact-space"
    dim = 8
    last_encode_error = None

    @staticmethod
    def embed_text(prefix: str, body: str, max_body_chars: "int | None" = None) -> EmbedResult:
        text = f"{prefix} {body}".casefold()
        if "alpha" in text:
            vector = [1.0] + [0.0] * 7
        elif "beta" in text:
            vector = [0.0, 1.0] + [0.0] * 6
        else:
            vector = _hashed_unit_vector(text)
        return EmbedResult(vector, False, 1, 1)

    @classmethod
    def embed_texts(cls, bodies: "list[str]", max_body_chars: "int | None" = None) -> list[EmbedResult]:
        return [cls.embed_text("", body) for body in bodies]


def _make_tools(tmp_path: Path) -> MemoryTools:
    pytest.importorskip("sqlite_vec")
    model = tmp_path / "fake-g2.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "g2.sqlite3",
        backup_jsonl=tmp_path / "g2-backup.jsonl",
        embedding_model_path=model,
        client="c", agent_id="a",
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    tools._embedder = _DirectedEmbedder()
    tools._embedder_loaded = True
    assert db.ensure_vec_tables(_DirectedEmbedder.dim) == []
    db.init_vec_index_state(_DirectedEmbedder.embedding_space_id, True, _DirectedEmbedder.dim)
    return tools


def _write_and_index(tools: MemoryTools, subject: str, content: str) -> int:
    mid, _warnings = tools.db.insert_memory(MemoryRecord(
        content=content, agent_id="a", workspace="main", tags=[],
        source_type="agent_generated", subject=subject,
    ))
    assert mid is not None
    outcome = tools._evidence.index_memory(int(mid))
    assert outcome.get("status") == "indexed", outcome
    return int(mid)


def test_exact_subject_match_ranks_first(tmp_path: Path) -> None:
    tools = _make_tools(tmp_path)
    exact = _write_and_index(tools, "Deployment timeout threshold", "alpha body line")
    other_a = _write_and_index(tools, "Release checklist notes", "beta body line")
    other_b = _write_and_index(tools, "Unrelated storage policy", "plain gamma text")
    # The diagonal query vector keeps cosine ≈0.71 against the alpha row —
    # the subject-equality leg fires, the cosine leg does not.
    result = tools.memory_search(query="Deployment Timeout  Threshold")
    results = result["data"]["results"]
    ids = [int(row["id"]) for row in results]
    assert ids and ids[0] == exact
    # Unrelated memories share no token and no vector direction with the
    # query — not recalled at all is correct, the page is exact alone.
    exact_row = results[0]
    assert exact_row.get("lexical_rank") is not None


def test_cosine_exact_boost_ranks_first(tmp_path: Path) -> None:
    tools = _make_tools(tmp_path)
    alpha = _write_and_index(tools, "Completely different heading one", "alpha body line")
    _write_and_index(tools, "Completely different heading two", "beta body line")
    # Query vector points east; the alpha row is the same direction → true
    # cosine 1.0 ≥ COS_EXACT_BOOST fires even though no subject matches.
    result = tools.memory_search(query="alpha query")
    results = result["data"]["results"]
    ids = [int(row["id"]) for row in results]
    assert ids[0] == alpha
    assert results[0].get("evidence_best_score") == 1.0


def test_plain_query_ordering_unchanged_and_no_exact_keys(tmp_path: Path) -> None:
    tools = _make_tools(tmp_path)
    strong = _write_and_index(tools, "Guideline renewal draft", "beta strong match text")
    _write_and_index(tools, "Server room inventory", "beta weak text")
    # The strong memory's subject/tags carry the only query-term overlap in
    # this library, so the plain RRF order it produces IS the unchanged
    # baseline ordering.
    # "renewal draft" overlaps the subject lexically (plain FTS hit) but
    # is neither an exact subject match nor a >0.98 vector twin.
    result = tools.memory_search(query="renewal draft")
    results = result["data"]["results"]
    ids = [int(row["id"]) for row in results]
    assert ids[0] == strong
    # No exact leg fired: no transparency keys, no boost marker effects.
    for row in results:
        assert "evidence_best_score" not in row
