from __future__ import annotations
import hashlib
import json
import re
import sqlite3
import sys
import types
from pathlib import Path
import pytest
from memory_arbiter.arbitration import compare_memories
from memory_arbiter.config import Settings, parse_bool
from memory_arbiter.db import MemoryDB
from memory_arbiter.models import SourceType
from memory_arbiter.tools import MemoryTools

@pytest.fixture(autouse=True)
def _relevance_floor_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0.15.9: the tests in this file exercise scoping/preview/metering with
    # content-only fixtures; the query-recall relevance floor is orthogonal to
    # what they pin (its behavior lives in tests/test_recall_quality_0159.py).
    # Disable it here so weak fixtures stay recallable.
    monkeypatch.setattr("memory_arbiter.search.QUERY_RECALL_SCORE_FLOOR", -1.0)

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_arbiter.config import Settings
from memory_arbiter.constants import (
    EMBEDDING_DEFAULT_DIM,
    EMBEDDING_MAX_SECTION_CHARS,
    EMBEDDING_N_CTX,
    EMBEDDING_RESERVED_TOKENS,
    REMOVED_ENV_NAMES,
    SEMANTIC_PRELOAD,
)
from memory_arbiter.db import MemoryDB
from memory_arbiter.db.meta import vec_table_dimension
from memory_arbiter.embedder import (
    EMBEDDING_PIPELINE_VERSION,
    ManagedEmbedder,
    build_embedder,
    compute_embedding_space_id,
    compute_model_digest,
)
from memory_arbiter.tools import MemoryTools

try:
    import sqlite_vec  # type: ignore  # noqa: F401

    _VEC_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _VEC_AVAILABLE = False

requires_vec = pytest.mark.skipif(not _VEC_AVAILABLE, reason="sqlite-vec not installed")




# ---------------------------------------------------------------------------
# (d) active_dim fact source
# ---------------------------------------------------------------------------


class _DimProbeEmbedder:
    """Fake ManagedEmbedder reporting a fixed dim (no GGUF runtime needed)."""

    def __init__(self, dim: int) -> None:
        self.embedding_space_id = f"fake-dim-{dim}-space"
        self.dim = dim

    def embed_text(self, prefix: str = "", body: str = "", max_body_chars=None):
        from memory_arbiter.embedder import EmbedResult

        return EmbedResult([0.5] * self.dim, False, 0, 0)


@requires_vec
def test_fake_embedder_dim_creates_lazy_tables_and_records_active_dim(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "dim.sqlite3",
        backup_jsonl=tmp_path / "dim.jsonl",
        embedding_model_path=model,
    )
    db = MemoryDB(settings)
    tools = MemoryTools(settings=settings, db=db)
    # Fresh library: no vec0 tables yet (lazy creation — schema init must not
    # have built them without knowing the dim).
    with db.connection() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_row_vec'"
            ).fetchone()
            is None
        )

    # First successful embedder build at dim 4: tables + space state together.
    fake = _DimProbeEmbedder(4)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            "memory_arbiter.embedder.build_embedder",
            lambda *_a, **_k: (fake, []),
        )
        tools._embedder_loaded = False
        embedder, warnings = tools._ensure_embedder()
        assert embedder is fake and warnings == []
    finally:
        monkeypatch.undo()

    with db.connection() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_row_vec'"
            ).fetchone()
            is not None
        )
        assert vec_table_dimension(conn, "memory_row_vec") == 4
        assert vec_table_dimension(conn, "workspace_canonicals_vec") == 4
    assert db.meta.get_active_dim() == 4
    assert db.get_vec_index_state()["state"] == "ready"


@requires_vec
def test_dim_change_drops_and_recreates_vec_tables(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "swap.sqlite3",
        backup_jsonl=tmp_path / "swap.jsonl",
        embedding_model_path=model,
    )
    db = MemoryDB(settings)
    assert db.ensure_vec_tables(4) == []
    db.init_vec_index_state("old-space", True, active_dim=4)

    # A model swap to dim 2: init_vec_index_state must drop the dim-4 tables
    # and re-create them at the new dim atomically with the mismatch flip.
    db.init_vec_index_state("new-space", True, active_dim=2)
    with db.connection() as conn:
        assert vec_table_dimension(conn, "memory_row_vec") == 2
        assert vec_table_dimension(conn, "workspace_canonicals_vec") == 2
    assert db.meta.get_active_dim() == 2
    state = db.get_vec_index_state()
    assert state["state"] == "mismatch"
    assert state["target_space_id"] == "new-space"


@requires_vec
def test_dim_swap_back_arms_rebuild_on_native_dim_tables(tmp_path: Path) -> None:
    # Swapping the model to dim B and back to A keeps the A space id active
    # in meta while the forward flip already rebuilt the tables at B. The
    # revert branch rebuilds them at A — but because that rebuild wiped all
    # vectors, it must NOT flip ready (surviving evidence rows would lose
    # coverage behind a passing gate); it arms the standard mismatch
    # rebuild toward A instead, and repeated inits stay armed (round-2
    # adversarial review finding).
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "back.sqlite3",
        backup_jsonl=tmp_path / "back.jsonl",
        embedding_model_path=model,
    )
    db = MemoryDB(settings)
    assert db.ensure_vec_tables(4) == []
    db.init_vec_index_state("space-a", True, active_dim=4)
    db.init_vec_index_state("space-b", True, active_dim=2)  # forward swap
    db.init_vec_index_state("space-a", True, active_dim=4)  # swap back

    with db.connection() as conn:
        assert vec_table_dimension(conn, "memory_row_vec") == 4
        assert vec_table_dimension(conn, "workspace_canonicals_vec") == 4
        # A native-dim vector must be insertable — the wedged state failed here.
        conn.execute(
            "INSERT INTO memory_row_vec(id, parent_status, embedding) VALUES(1,'active',?)",
            ("[" + ", ".join("0.1" for _ in range(4)) + "]",),
        )
    state = db.get_vec_index_state()
    assert state["state"] == "mismatch"
    assert state["target_space_id"] == "space-a"
    assert state["active_dim"] == 4
    # Idempotency: the armed rebuild must survive repeated inits (the
    # same-id revert flip must not bypass it back to ready).
    db.init_vec_index_state("space-a", True, active_dim=4)
    state = db.get_vec_index_state()
    assert state["state"] == "mismatch"
    assert state["target_space_id"] == "space-a"


def test_default_model_space_id_unchanged_vs_literal_former_defaults(
    tmp_path: Path,
) -> None:
    # The frozen constants must reproduce the exact former-default space id
    # (2048/64/3600 engine params, 768-dim default model): any drift would
    # force a pointless full evidence rebuild for default users.
    model = tmp_path / "default.gguf"
    model.write_bytes(b"default-model-bytes")
    digest = compute_model_digest(str(model))

    from_constants = compute_embedding_space_id(
        digest,
        EMBEDDING_DEFAULT_DIM,
        EMBEDDING_PIPELINE_VERSION,
        {
            "n_ctx": EMBEDDING_N_CTX,
            "reserved_tokens": EMBEDDING_RESERVED_TOKENS,
            "max_section_chars": EMBEDDING_MAX_SECTION_CHARS,
        },
    )
    from_literals = compute_embedding_space_id(
        digest,
        768,
        2,  # 0.17.0 终局（owner 2026-09-26）：存储侧裸文本=已发版 v2 语义，版本不轮换
        {"n_ctx": 2048, "reserved_tokens": 64, "max_section_chars": 3600},
    )
    assert from_constants == from_literals

    from memory_arbiter.vnext_migration import _configured_embedding_space_id

    settings = Settings(
        db_path=tmp_path / "space.sqlite3",
        backup_jsonl=tmp_path / "space.jsonl",
        embedding_model_path=model,
    )
    assert _configured_embedding_space_id(settings, 768) == from_constants
    # Without a dim there is nothing trustworthy to derive an identity from.
    assert _configured_embedding_space_id(settings, None) is None
    assert _configured_embedding_space_id(None, 768) is None


# ---------------------------------------------------------------------------
# (f) hybrid-only search
# ---------------------------------------------------------------------------


def test_search_source_has_no_bm25_path() -> None:
    source = Path(__file__).parents[1] / "memory_arbiter" / "search.py"
    text = source.read_text(encoding="utf-8")
    assert "_search_bm25" not in text
    assert "_get_ranking_mode" not in text
    # The surviving helpers the hybrid path still uses. v0.15.9: _recent_fallback
    # is browse-only (empty query); the query fallback and its warning prefix
    # are gone — empty is honest.
    assert "_recent_fallback" in text
    assert "No direct memory match" not in text
    from memory_arbiter.search import _recent_fallback  # noqa: F401

    import memory_arbiter.constants as _constants
    assert not hasattr(_constants, "NO_DIRECT_MATCH_PREFIX")


# ---------------------------------------------------------------------------
# (g) lazy schema probe
# ---------------------------------------------------------------------------


@requires_vec
def test_model_configured_library_without_vec_tables_is_healthy(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    settings = Settings(
        db_path=tmp_path / "probe.sqlite3",
        backup_jsonl=tmp_path / "probe.jsonl",
        embedding_model_path=model,
    )
    db = MemoryDB(settings)
    # The extension is loadable and the library is pre-first-embed: the probe
    # must pass (no "sqlite-vec unavailable" degradation) even though the
    # derived vec0 tables do not exist yet.
    assert db.state.sqlite_vec_available is True
    assert db.db_available is True
    assert not any("sqlite-vec unavailable" in w for w in db.state.warnings)
    with db.connection() as conn:
        for table in ("memory_row_vec", "workspace_canonicals_vec"):
            assert (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                is None
            )
    # The repair surface only counts missing tables once the vec state says
    # the index should be live (ready + model configured): a pre-first-embed
    # library is healthy, not broken.
    tools = MemoryTools(settings=settings, db=db)
    preview = tools.memory_repair("rebuild_evidence", {"dry_run": True})
    assert preview["ok"] is True
    assert preview["data"]["vector_table_repair"]["required"] is False


def test_config_surface_examples_and_registry_aligned() -> None:
    """R3 盘点 P2 pin: the reference example must cover every registry key
    and the surface count is pinned — 0.17.1 aligned them by hand and 0.17.2
    drifted again (mdeberta_possible_min_susp missing from five prose spots
    and the example); a test is cheaper than a third realignment. Bump the
    count together with the INTEGRATION prose and the example config and prose when adding a key."""
    from memory_arbiter.config_registry import CONFIG_DESCRIPTORS

    assert len(CONFIG_DESCRIPTORS) == 28

    example = json.loads(
        (Path(__file__).resolve().parents[1] / "examples" / "memory-arbiter.config.example.json").read_text()
    )
    missing: list[str] = []
    for descriptor in CONFIG_DESCRIPTORS:
        node: object = example
        for part in descriptor["path"].split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                missing.append(descriptor["path"])
                break
    assert not missing, f"example config missing registry keys: {missing}"
