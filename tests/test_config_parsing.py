
# ── from test_config_slim.py ──

"""0.15.0 config-slimming regression guards (plan §7).

Pins the new configuration contract:
  a) dataclass defaults == empty-env from_env() field by field (one source of
     truth — the dataclass IS the default surface now);
  b) auto-enable: embedding.model_path alone turns embedding on (no vec
     section); semantic_conflict.model_path alone turns semantic conflict on;
  c) deprecation: removed file keys warn "no longer configurable"; removed env
     exports warn "no longer read";
  d) active_dim fact source: a dim-4 fake embedder lazily creates the vec0
     tables at dim 4 and records meta active_dim=4; the default-model space id
     computed from the frozen constants equals the literal former-default
     combination (2048/64/3600/768) — no space drift for default users;
  e) the Settings field set is frozen at exactly the 20 slim fields;
  f) bm25 is gone: search source has no _search_bm25/_get_ranking_mode and
     ranking is hybrid-only (no ranking-mode concept in responses);
  g) lazy schema: a model-configured library with no vec tables starts without
     an "sqlite-vec unavailable" warning (the probe must not judge a healthy
     pre-first-embed library as broken).
"""


import dataclasses
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
# No autouse _relevance_floor_off copy here: parsing-domain tests never touch the
# search recall path (grep-verified), so the floor stays at its real default.

try:
    import sqlite_vec  # type: ignore  # noqa: F401

    _VEC_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _VEC_AVAILABLE = False

requires_vec = pytest.mark.skipif(not _VEC_AVAILABLE, reason="sqlite-vec not installed")

# Launch-context env (retained) that must be cleared so from_env() sees a
# truly empty launch context.
_LAUNCH_ENV = (
    "MEMORY_ARBITER_CONFIG",
    "MEMORY_ARBITER_DB_PATH",
    "MEMORY_ARBITER_BACKUP_JSONL",
    "MEMORY_ARBITER_MCP_TRANSPORT",
    "MEMORY_ARBITER_CLIENT",
    "MEMORY_ARBITER_AGENT_ID",
)


def _hermetic_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict
) -> Path:
    """Empty launch context + an explicit (possibly empty) config file."""
    for key in _LAUNCH_ENV:
        monkeypatch.delenv(key, raising=False)
    for key in REMOVED_ENV_NAMES:
        monkeypatch.delenv(key, raising=False)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("MEMORY_ARBITER_CONFIG", str(cfg))
    return cfg


def test_policy_path_key_warns_after_agent_policy_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B1 (0.15.14): AgentPolicy is gone; a configured policy_path must warn
    (removed-feature wording), not fail startup or silently no-op."""
    _hermetic_env(monkeypatch, tmp_path, {"policy_path": "/tmp/policy.json"})
    settings = Settings.from_env()
    assert not hasattr(settings, "policy_path")
    assert not hasattr(settings, "policy")
    assert any(
        "policy_path is no longer read" in warning
        and "AgentPolicy was removed in 0.15.14" in warning
        for warning in settings.config_warnings
    )


# ---------------------------------------------------------------------------
# (e) frozen field set — the slim dataclass is the contract
# ---------------------------------------------------------------------------

SLIM_SETTINGS_FIELDS = frozenset(
    {
        "db_path",
        "backup_jsonl",
        "client",
        "agent_id",
        "workspace",
        "mcp_transport",
        "mcp_http_host",
        "mcp_http_port",
        "embedding_model_path",
        "embedding_auto_query",
        "embedding_auto_write",
        "isolation",
        "update_check_enabled",
        "include_size",
        "semantic_conflict_enabled",
        "semantic_conflict_model_path",
        "semantic_conflict_mdeberta_ckpt",
        "semantic_conflict_mdeberta_model_dir",
        "semantic_conflict_mdeberta_notice_min_prob",
        "semantic_conflict_mdeberta_possible_min_susp",
        "semantic_conflict_mdeberta_batch",
        "semantic_conflict_on_write",
        "semantic_conflict_gpu_layers",
        "semantic_conflict_notice_sync_wait_ms",
        "config_warnings",
        "config_file_loaded",
        "llm_base_url",
        "llm_api_key",
        "llm_model",
        "llm_max_context_tokens",
        "llm_max_output_tokens",
        "llm_timeout_seconds",
    }
)


def test_settings_field_set_matches_slim_contract() -> None:
    # 0.15.14: 21 — semantic_conflict_gpu_layers added (A3),
    # semantic_conflict_max_notice_pairs removed (A5) and policy_path/policy
    # removed (B1: AgentPolicy deleted). config_file_loaded stays a
    # runtime-injected field (never a file key, not in the registry).
    # 0.17.0 P2-5.2: claims_required added (grey-period switch, default off).
    # 0.17.1: mdeberta judge keys added (ckpt/model_dir/notice_min_prob/batch).
    # 0.18.0: llm.* six keys added (console knowledge workbench, plan §B.2).
    # 0.17.2: semantic_conflict_mdeberta_possible_min_susp added (possible 灰区门).
    assert len(SLIM_SETTINGS_FIELDS) == 32
    assert set(Settings.__dataclass_fields__) == SLIM_SETTINGS_FIELDS


def test_settings_rejects_removed_kwargs() -> None:
    # Every former knob must fail loudly at construction, not silently no-op.
    for removed in (
        "enable_sqlite_vec",
        "vec_dim",
        "embedding_provider",
        "semantic_conflict_backend",
        "semantic_conflict_preload",
        "notice_sync_wait_ms",
        "workspace_recall_cutoff",
        "recall_pool_cap",
        "tool_profile",
        "mcp_http_path",
    ):
        with pytest.raises(TypeError):
            Settings(
                db_path=Path("x.sqlite3"),
                backup_jsonl=Path("x.jsonl"),
                **{removed: True},
            )


# ---------------------------------------------------------------------------
# (a) dataclass defaults vs empty-env from_env()
# ---------------------------------------------------------------------------


def test_from_env_empty_config_matches_dataclass_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hermetic_env(monkeypatch, tmp_path, {})
    from_env_settings = Settings.from_env()

    defaults = Settings(db_path=Path("d.sqlite3"), backup_jsonl=Path("b.jsonl"))
    # Environment-dependent fields are excluded by design: db/backup paths are
    # cwd-based, client/agent_id/mcp_transport read launch env, config_warnings
    # collects parse-time diagnostics, and config_file_loaded records whether
    # a real config file was found (runtime provenance, not a setting).
    excluded = {
        "db_path",
        "backup_jsonl",
        "client",
        "agent_id",
        "mcp_transport",
        "config_warnings",
        "config_file_loaded",
    }
    for field in dataclasses.fields(Settings):
        if field.name in excluded:
            continue
        assert getattr(from_env_settings, field.name) == getattr(
            defaults, field.name
        ), f"from_env default drifted from dataclass default on {field.name}"


# ---------------------------------------------------------------------------
# (b) auto-enable (one intent, one knob)
# ---------------------------------------------------------------------------


def test_embedding_model_path_alone_enables_vec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")
    _hermetic_env(
        monkeypatch,
        tmp_path,
        {
            "db_path": str(tmp_path / "b.sqlite3"),
            "backup_jsonl": str(tmp_path / "b.jsonl"),
            "embedding": {"model_path": str(model)},
        },
    )

    settings = Settings.from_env()
    # No vec section, no provider — pointing at the model IS the intent.
    assert settings.embedding_model_path == model
    tools = MemoryTools(settings)
    assert tools._embedding_configured() is True


def test_no_embedding_model_path_means_embedding_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hermetic_env(
        monkeypatch,
        tmp_path,
        {
            "db_path": str(tmp_path / "b.sqlite3"),
            "backup_jsonl": str(tmp_path / "b.jsonl"),
        },
    )
    settings = Settings.from_env()
    assert settings.embedding_model_path is None
    assert MemoryTools(settings)._embedding_configured() is False


def test_semantic_mdeberta_ckpt_alone_auto_enables_and_preloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "mdeberta-v4m.pt"
    ckpt.write_bytes(b"fake")
    (tmp_path / "mdeberta-base").mkdir()
    _hermetic_env(
        monkeypatch,
        tmp_path,
        {
            "db_path": str(tmp_path / "b.sqlite3"),
            "backup_jsonl": str(tmp_path / "b.jsonl"),
            "semantic_conflict": {"mdeberta_ckpt": str(ckpt)},
        },
    )

    settings = Settings.from_env()
    assert settings.semantic_conflict_enabled is True
    assert settings.semantic_conflict_mdeberta_ckpt == ckpt
    # model_dir default = ckpt-adjacent mdeberta-base (§3.6)
    assert settings.semantic_conflict_mdeberta_model_dir == tmp_path / "mdeberta-base"
    assert settings.semantic_conflict_mdeberta_notice_min_prob == 0.8  # 0.17.2 换装 v58_ep3_fp16 定稿（owner 2026-10-10）；v52 时代 T=0.5 随 ckpt 退役
    assert any("auto-enabled" in warning for warning in settings.config_warnings)

    tools = MemoryTools(settings)
    status = tools._semantic_status()
    assert status["enabled"] is True
    assert status["configured"] is True
    assert status["engine"] == "mdeberta"
    # preload froze to true: a configured checkpoint loads at startup.
    assert SEMANTIC_PRELOAD is True


def test_semantic_legacy_model_path_warns_and_does_not_enable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"fake")
    _hermetic_env(
        monkeypatch,
        tmp_path,
        {
            "db_path": str(tmp_path / "b.sqlite3"),
            "backup_jsonl": str(tmp_path / "b.jsonl"),
            "semantic_conflict": {"model_path": str(model)},
        },
    )

    settings = Settings.from_env()
    # 0.17.1: the Qwen/GGUF path is retired — configured legacy key warns and
    # does NOT enable; intent now lives in mdeberta_ckpt.
    assert settings.semantic_conflict_enabled is False
    assert MemoryTools(settings)._semantic_configured() is False
    assert any("model_path is retired" in w for w in settings.config_warnings)
    assert any("mdeberta_ckpt" in w for w in settings.config_warnings)


def test_semantic_explicit_false_wins_over_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"fake")
    _hermetic_env(
        monkeypatch,
        tmp_path,
        {
            "db_path": str(tmp_path / "b.sqlite3"),
            "backup_jsonl": str(tmp_path / "b.jsonl"),
            "semantic_conflict": {"enabled": False, "model_path": str(model)},
        },
    )
    settings = Settings.from_env()
    assert settings.semantic_conflict_enabled is False
    assert MemoryTools(settings)._semantic_configured() is False


# ---------------------------------------------------------------------------
# (c) deprecation warnings
# ---------------------------------------------------------------------------


def test_removed_file_keys_warn_and_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hermetic_env(
        monkeypatch,
        tmp_path,
        {
            "semantic_conflict": {
                "preload": False,
                "resident": False,
                "backend": "local",
                "job_timeout_ms": 1,
                # notice_sync_wait_ms left this set in 0.15.8 (live key again);
                # a value here must now take effect, not warn.
                "notice_sync_wait_ms": 1,
            },
            "embedding": {"provider": "gguf", "n_ctx": 128},
            "vec": {"enabled": False, "dim": 8},
            "tool_profile": "legacy_full",
            "workspace_recall_cutoff": 0.9,
        },
    )
    settings = Settings.from_env()
    joined = "\n".join(settings.config_warnings)
    for key in (
        "semantic_conflict.preload",
        "semantic_conflict.resident",
        "semantic_conflict.backend",
        "semantic_conflict.job_timeout_ms",
        "embedding.provider",
        "embedding.n_ctx",
        "vec.enabled",
        "vec.dim",
        "tool_profile",
        "workspace_recall_cutoff",
    ):
        assert f"{key} is no longer configurable" in joined, key
    # Ignored, not applied: the removed vec.enabled=false must not disable
    # an embedding model that IS configured.
    assert settings.semantic_conflict_enabled is False
    # 0.15.8: the restored key is applied (clamped into [0, 5000]).
    assert settings.semantic_conflict_notice_sync_wait_ms == 1


def test_removed_env_exports_warn_no_longer_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _hermetic_env(monkeypatch, tmp_path, {})
    monkeypatch.setenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_PRELOAD", "false")
    monkeypatch.setenv("MEMORY_ARBITER_ENABLE_SQLITE_VEC", "true")
    monkeypatch.setenv("MEMORY_ARBITER_RANKING_MODE", "bm25")
    monkeypatch.setenv("MEMORY_ARBITER_UPDATE_CHECK_ENABLED", "false")
    monkeypatch.setenv("MEMORY_ARBITER_GGUF", "/tmp/legacy.gguf")

    settings = Settings.from_env()
    joined = "\n".join(settings.config_warnings)
    for name in (
        "MEMORY_ARBITER_SEMANTIC_CONFLICT_PRELOAD",
        "MEMORY_ARBITER_ENABLE_SQLITE_VEC",
        "MEMORY_ARBITER_RANKING_MODE",
        "MEMORY_ARBITER_UPDATE_CHECK_ENABLED",
        "MEMORY_ARBITER_GGUF",
    ):
        assert f"{name} is no longer read" in joined, name
    # A stale ranking-mode export cannot resurrect bm25, and the dead update
    # switch must not flip the file default.
    assert settings.update_check_enabled is True
    assert cfg.exists()