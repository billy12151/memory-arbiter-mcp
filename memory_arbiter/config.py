from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


@dataclass
class AgentPolicy:
    # Per-client overrides. Unlisted clients use ``default_enabled``.
    client_defaults: dict[str, bool] = field(default_factory=dict)
    default_enabled: bool = True
    allow_agents: list[str] = field(default_factory=list)
    deny_agents: list[str] = field(default_factory=list)

    def enabled_for(self, client: str, agent_id: str) -> bool:
        if agent_id in self.deny_agents:
            return False
        if agent_id in self.allow_agents:
            return True
        normalized = (client or "").lower()
        if normalized in self.client_defaults:
            return self.client_defaults[normalized]
        return self.default_enabled


@dataclass
class Settings:
    db_path: Path
    backup_jsonl: Path
    policy_path: Optional[Path] = None
    client: str = "codex"
    agent_id: str = "default"
    workspace: str = "default"
    enable_sqlite_vec: bool = False
    vec_dim: int = 768
    recall_pool_cap: int = 50
    content_like_cap: int = 30
    superseded_limit: int = 20
    policy: AgentPolicy = field(default_factory=AgentPolicy)
    embedding_provider: Optional[str] = None
    embedding_model_path: Optional[Path] = None
    embedding_auto_query: bool = True
    embedding_auto_write: bool = True
    # ── Embedding pipeline params (v0.6.0: part of embedding_space_id) ──
    embedding_n_ctx: int = 2048
    embedding_reserved_tokens: int = 64
    # Maximum text passed to one embedding call.
    max_section_chars: int = 3600
    # Workspace isolation level: none (default, ws ignored) | weak (soft rerank) | strict (hard filter).
    isolation: str = "none"
    # Cosine-distance cutoff for workspace alias canonicalization (vec KNN).
    # Lower = stricter. ~0.16 merges 金营项目/金科营销项目; ~0.43 keeps unrelated
    # projects distinct, so 0.25 cleanly separates synonyms from distinct workspaces.
    workspace_match_distance: float = 0.25
    update_check_enabled: bool = True
    tool_profile: str = "product"
    semantic_conflict_enabled: bool = False
    semantic_conflict_backend: str = "local_gguf"
    semantic_conflict_model_path: Optional[Path] = None
    semantic_conflict_on_write: str = "async"
    semantic_conflict_max_concurrency: int = 1
    semantic_conflict_queue_max_size: int = 100
    semantic_conflict_n_ctx: int = 1024
    semantic_conflict_n_threads: int = 4
    semantic_conflict_n_batch: int = 128
    semantic_conflict_resident: bool = True
    semantic_conflict_preload: bool = False
    semantic_conflict_job_timeout_ms: int = 5000
    semantic_conflict_inference_timeout_ms: int = 30000
    semantic_conflict_load_timeout_ms: int = 120000
    semantic_conflict_min_pair_budget_ms: int = 1000
    config_warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        config_warnings: list[str] = []
        config_path = _find_config_file(config_warnings)
        cfg = load_config_file(config_path, config_warnings)
        cwd = Path.cwd()
        policy_raw = os.getenv("MEMORY_ARBITER_POLICY")
        vec_cfg = cfg.get("vec") or {}
        emb_cfg = cfg.get("embedding") or {}
        if not isinstance(vec_cfg, dict):
            config_warnings.append(f"vec={vec_cfg!r} invalid; using env/defaults")
            vec_cfg = {}
        if not isinstance(emb_cfg, dict):
            config_warnings.append(f"embedding={emb_cfg!r} invalid; using env/defaults")
            emb_cfg = {}
        emb_cfg = {str(k): v for k, v in emb_cfg.items() if not str(k).startswith("_")}
        semantic_cfg = cfg.get("semantic_conflict") or {}
        if not isinstance(semantic_cfg, dict):
            config_warnings.append(f"semantic_conflict={semantic_cfg!r} invalid; using env/defaults")
            semantic_cfg = {}
        semantic_cfg = {str(k): v for k, v in semantic_cfg.items() if not str(k).startswith("_")}
        update_cfg_raw = cfg.get("update_check", {})
        update_check_enabled = True
        if isinstance(update_cfg_raw, dict):
            update_check_enabled = parse_bool_warn(
                update_cfg_raw.get("enabled", True), True, name="update_check.enabled", warnings=config_warnings
            )
        elif update_cfg_raw is not None:
            update_check_enabled = parse_bool_warn(
                update_cfg_raw, True, name="update_check", warnings=config_warnings
            )
        def pick_str(cfg_key: str, env_key: str, default: str) -> str:
            try:
                if cfg.get(cfg_key) is not None:
                    return str(cfg[cfg_key])
                return os.getenv(env_key, default)
            except Exception:
                return default

        def pick_path(cfg_key: str, env_key: str, default_path: Path) -> Path:
            try:
                val = cfg.get(cfg_key)
                if val is not None:
                    return Path(str(val)).expanduser()
                return Path(os.getenv(env_key, str(default_path))).expanduser()
            except Exception:
                return default_path

        def pick_int_field(cfg_val: Any, env_key: str, default: int, name: str) -> int:
            if cfg_val is not None:
                return parse_int(cfg_val, default, name=name, warnings=config_warnings)
            env_val = os.getenv(env_key)
            if env_val is not None:
                return parse_int(env_val, default, name=name, warnings=config_warnings)
            return default

        def pick_bool_field(
            cfg_val: Any,
            env_key: str,
            default_str: str = "false",
            name: str = "",
            default_bool: bool = False,
        ) -> bool:
            if cfg_val is not None:
                return parse_bool_warn(cfg_val, default_bool, name=name, warnings=config_warnings)
            env_val = os.getenv(env_key, default_str)
            return parse_bool_warn(env_val, default_bool, name=name, warnings=config_warnings)

        def pick_float_field(cfg_val: Any, env_key: str, default: float, name: str) -> float:
            if cfg_val is not None:
                return parse_float(cfg_val, default, name=name, warnings=config_warnings)
            env_val = os.getenv(env_key)
            if env_val is not None:
                return parse_float(env_val, default, name=name, warnings=config_warnings)
            return default

        embedding_model_raw = emb_cfg.get("model_path") or os.getenv("MEMORY_ARBITER_EMBEDDING_MODEL_PATH") or os.getenv("MEMORY_ARBITER_GGUF")
        embedding_provider_raw = emb_cfg.get("provider") or os.getenv("MEMORY_ARBITER_EMBEDDING_PROVIDER") or ("gguf" if embedding_model_raw else None)
        embedding_provider = str(embedding_provider_raw).lower() if embedding_provider_raw else None
        if embedding_provider and embedding_provider != "gguf":
            config_warnings.append(f"embedding.provider={embedding_provider!r} unsupported; auto-embedding disabled.")

        isolation = pick_str(
            "isolation", "MEMORY_ARBITER_ISOLATION", "none"
        ).strip().lower()
        if isolation not in {"none", "weak", "strict"}:
            config_warnings.append(
                f"isolation={isolation!r} invalid; using none"
            )
            isolation = "none"
        if isolation == "strict":
            # 636 §9: strict decides visibility, so unconfirmed workspace
            # aliases silently hide memories. Nudge the operator to govern
            # workspaces (accept/reject/pending) before relying on strict.
            config_warnings.append(
                "isolation=strict: confirm workspace aliases via "
                "memory_govern (accept/reject_workspace_alias, "
                "confirm_pending_workspace) so new/aliased workspaces are not "
                "silently excluded from recall."
            )

        tool_profile = pick_str(
            "tool_profile", "MEMORY_ARBITER_TOOL_PROFILE", "product"
        ).strip().lower()
        if tool_profile != "product":
            config_warnings.append(
                f"tool_profile={tool_profile!r} invalid; using product"
            )
            tool_profile = "product"

        semantic_backend = str(
            semantic_cfg.get("backend")
            or os.getenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_BACKEND")
            or "local_gguf"
        ).strip().lower()
        if semantic_backend not in {"local_gguf"}:
            config_warnings.append(
                f"semantic_conflict.backend={semantic_backend!r} unsupported; using local_gguf"
            )
            semantic_backend = "local_gguf"
        semantic_on_write = str(
            semantic_cfg.get("on_write")
            or os.getenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_ON_WRITE")
            or "async"
        ).strip().lower()
        if semantic_on_write not in {"async", "off"}:
            config_warnings.append(
                f"semantic_conflict.on_write={semantic_on_write!r} invalid; using async"
            )
            semantic_on_write = "async"
        semantic_model_raw = (
            semantic_cfg.get("model_path")
            or os.getenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_MODEL_PATH")
        )

        # model_path configured but enabled not explicitly set → auto-enable.
        # One intent shouldn't need two knobs; the user expressed intent by
        # pointing at a model. Explicit enabled=false still wins (pick_bool_field
        # honours a non-None cfg_val / a set env var over the default).
        _semantic_auto_enable = bool(semantic_model_raw)
        if (
            _semantic_auto_enable
            and semantic_cfg.get("enabled") is None
            and not os.getenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED")
        ):
            config_warnings.append(
                "semantic_conflict.enabled not set; model_path configured → "
                "auto-enabled. Set enabled=false to disable."
            )

        if semantic_cfg.get("max_concurrency") not in (None, 1, "1") or os.getenv("MEMORY_ARBITER_SEMANTIC_CONFLICT_MAX_CONCURRENCY") not in (None, "1"):
            config_warnings.append("semantic_conflict.max_concurrency is reserved in this version; MVP semantic worker uses max_concurrency=1")

        settings = cls(
            db_path=pick_path("db_path", "MEMORY_ARBITER_DB_PATH", cwd / "memory_arbiter.sqlite3"),
            backup_jsonl=pick_path("backup_jsonl", "MEMORY_ARBITER_BACKUP_JSONL", cwd / "memory_arbiter.backup.jsonl"),
            policy_path=Path(str(cfg.get("policy_path"))).expanduser() if cfg.get("policy_path") else (Path(policy_raw).expanduser() if policy_raw else None),
            client=pick_str("client", "MEMORY_ARBITER_CLIENT", "codex"),
            agent_id=pick_str("agent_id", "MEMORY_ARBITER_AGENT_ID", "default"),
            workspace=pick_str("workspace", "MEMORY_ARBITER_WORKSPACE", "default"),
            enable_sqlite_vec=pick_bool_field(
                vec_cfg.get("enabled"), "MEMORY_ARBITER_ENABLE_SQLITE_VEC", "false", name="vec.enabled", default_bool=False
            ),
            vec_dim=pick_int_field(vec_cfg.get("dim"), "MEMORY_ARBITER_VEC_DIM", 768, name="vec.dim"),
            recall_pool_cap=pick_int_field(cfg.get("recall_pool_cap"), "MEMORY_ARBITER_RECALL_POOL_CAP", 50, name="recall_pool_cap"),
            content_like_cap=pick_int_field(cfg.get("content_like_cap"), "MEMORY_ARBITER_CONTENT_LIKE_CAP", 30, name="content_like_cap"),
            superseded_limit=clamp_int(
                pick_int_field(cfg.get("superseded_limit"), "MEMORY_ARBITER_SUPERSEDED_LIMIT", 20, name="superseded_limit"),
                1, 50, name="superseded_limit", warnings=config_warnings,
            ),
            embedding_provider=embedding_provider,
            embedding_model_path=Path(str(embedding_model_raw)).expanduser() if embedding_model_raw else None,
            embedding_auto_query=pick_bool_field(
                emb_cfg.get("auto_query"), "MEMORY_ARBITER_EMBEDDING_AUTO_QUERY", "true", name="embedding.auto_query", default_bool=True
            ),
            embedding_auto_write=pick_bool_field(
                emb_cfg.get("auto_write"), "MEMORY_ARBITER_EMBEDDING_AUTO_WRITE", "true", name="embedding.auto_write", default_bool=True
            ),
            embedding_n_ctx=clamp_int(
                pick_int_field(emb_cfg.get("n_ctx"), "MEMORY_ARBITER_EMBEDDING_N_CTX", 2048, name="embedding.n_ctx"),
                128, 131072, name="embedding.n_ctx", warnings=config_warnings,
            ),
            embedding_reserved_tokens=clamp_int(
                pick_int_field(emb_cfg.get("reserved_tokens"), "MEMORY_ARBITER_EMBEDDING_RESERVED_TOKENS", 64, name="embedding.reserved_tokens"),
                0, 4096, name="embedding.reserved_tokens", warnings=config_warnings,
            ),
            max_section_chars=clamp_int(
                pick_int_field(emb_cfg.get("max_unit_chars"), "MEMORY_ARBITER_EMBEDDING_MAX_UNIT_CHARS", 3600, name="embedding.max_unit_chars"),
                100, 1_000_000, name="embedding.max_unit_chars", warnings=config_warnings,
            ),
            isolation=isolation,
            workspace_match_distance=clamp_float(
                pick_float_field(cfg.get("workspace_match_distance"), "MEMORY_ARBITER_WORKSPACE_MATCH_DISTANCE", 0.25, name="workspace_match_distance"),
                0.0, 2.0, name="workspace_match_distance", warnings=config_warnings,
            ),
            update_check_enabled=update_check_enabled,
            tool_profile=tool_profile,
            semantic_conflict_enabled=pick_bool_field(
                semantic_cfg.get("enabled"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_ENABLED",
                "true" if _semantic_auto_enable else "false",
                name="semantic_conflict.enabled", default_bool=_semantic_auto_enable,
            ),
            semantic_conflict_backend=semantic_backend,
            semantic_conflict_model_path=Path(str(semantic_model_raw)).expanduser() if semantic_model_raw else None,
            semantic_conflict_on_write=semantic_on_write,
            semantic_conflict_max_concurrency=1,
            semantic_conflict_queue_max_size=clamp_int(
                pick_int_field(semantic_cfg.get("queue_max_size"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_QUEUE_MAX_SIZE", 100, name="semantic_conflict.queue_max_size"),
                1, 10000, name="semantic_conflict.queue_max_size", warnings=config_warnings,
            ),
            semantic_conflict_n_ctx=clamp_int(
                pick_int_field(semantic_cfg.get("n_ctx"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_N_CTX", 1024, name="semantic_conflict.n_ctx"),
                128, 8192, name="semantic_conflict.n_ctx", warnings=config_warnings,
            ),
            semantic_conflict_n_threads=clamp_int(
                pick_int_field(semantic_cfg.get("n_threads"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_N_THREADS", 4, name="semantic_conflict.n_threads"),
                1, 64, name="semantic_conflict.n_threads", warnings=config_warnings,
            ),
            semantic_conflict_n_batch=clamp_int(
                pick_int_field(semantic_cfg.get("n_batch"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_N_BATCH", 128, name="semantic_conflict.n_batch"),
                1, 2048, name="semantic_conflict.n_batch", warnings=config_warnings,
            ),
            semantic_conflict_resident=pick_bool_field(
                semantic_cfg.get("resident"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_RESIDENT", "true", name="semantic_conflict.resident", default_bool=True
            ),
            semantic_conflict_preload=pick_bool_field(
                semantic_cfg.get("preload"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_PRELOAD", "false", name="semantic_conflict.preload", default_bool=False
            ),
            semantic_conflict_job_timeout_ms=clamp_int(
                pick_int_field(semantic_cfg.get("job_timeout_ms"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_JOB_TIMEOUT_MS", 5000, name="semantic_conflict.job_timeout_ms"),
                100, 600000, name="semantic_conflict.job_timeout_ms", warnings=config_warnings,
            ),
            semantic_conflict_inference_timeout_ms=clamp_int(
                pick_int_field(semantic_cfg.get("inference_timeout_ms"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_INFERENCE_TIMEOUT_MS", 30000, name="semantic_conflict.inference_timeout_ms"),
                100, 600000, name="semantic_conflict.inference_timeout_ms", warnings=config_warnings,
            ),
            semantic_conflict_load_timeout_ms=clamp_int(
                pick_int_field(semantic_cfg.get("load_timeout_ms"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_LOAD_TIMEOUT_MS", 120000, name="semantic_conflict.load_timeout_ms"),
                100, 600000, name="semantic_conflict.load_timeout_ms", warnings=config_warnings,
            ),
            semantic_conflict_min_pair_budget_ms=clamp_int(
                pick_int_field(semantic_cfg.get("min_pair_budget_ms"), "MEMORY_ARBITER_SEMANTIC_CONFLICT_MIN_PAIR_BUDGET_MS", 1000, name="semantic_conflict.min_pair_budget_ms"),
                50, 300000, name="semantic_conflict.min_pair_budget_ms", warnings=config_warnings,
            ),
        )
        settings.config_warnings = config_warnings
        settings.policy = load_policy(settings.policy_path, config_warnings)
        return settings

    def defaults(self) -> dict[str, str]:
        return {"agent_id": self.agent_id, "workspace": self.workspace}


def load_policy(path: Optional[Path], warnings: Optional[list[str]] = None) -> AgentPolicy:
    if not path or not path.exists():
        return AgentPolicy()
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        # Malformed/missing policy must not crash startup — fall back to default allow-all.
        if warnings is not None:
            warnings.append(f"Policy file {path} parse failed: {exc}; using default allow-all policy.")
        return AgentPolicy()
    # parse_bool so a hand-edited JSON string like "false" isn't truthy (#5).
    return AgentPolicy(
        client_defaults={str(k): parse_bool(v, True) for k, v in (raw.get("client_defaults") or {}).items()},
        default_enabled=parse_bool(raw.get("default_enabled", True), True),
        allow_agents=list(raw.get("allow_agents") or []),
        deny_agents=list(raw.get("deny_agents") or []),
    )


def parse_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        low = val.strip().lower()
        if low in _TRUE_STRINGS:
            return True
        if low in _FALSE_STRINGS:
            return False
    return default


def parse_bool_warn(val: Any, default: bool, name: str = "", warnings: Optional[list[str]] = None) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        low = val.strip().lower()
        if low in _TRUE_STRINGS:
            return True
        if low in _FALSE_STRINGS:
            return False
        if warnings is not None:
            warnings.append(f"{name}={val!r} invalid; using default {default}")
    return default


def parse_int(val: Any, default: int, name: str = "", warnings: Optional[list[str]] = None) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        if warnings is not None and val is not None:
            warnings.append(f"{name}={val!r} invalid; using default {default}")
        return default


def parse_float(val: Any, default: float, name: str = "", warnings: Optional[list[str]] = None) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        if warnings is not None and val is not None:
            warnings.append(f"{name}={val!r} invalid; using default {default}")
        return default


def clamp_int(val: int, lo: int, hi: int, name: str = "", warnings: Optional[list[str]] = None) -> int:
    """Clamp an int to [lo, hi], emitting a warning when out of range."""
    if val < lo:
        if warnings is not None:
            warnings.append(f"{name}={val} below minimum {lo}; clamped to {lo}")
        return lo
    if val > hi:
        if warnings is not None:
            warnings.append(f"{name}={val} above maximum {hi}; clamped to {hi}")
        return hi
    return val


def clamp_float(val: float, lo: float, hi: float, name: str = "", warnings: Optional[list[str]] = None) -> float:
    """Clamp a float to [lo, hi], emitting a warning when out of range."""
    if val < lo:
        if warnings is not None:
            warnings.append(f"{name}={val} below minimum {lo}; clamped to {lo}")
        return lo
    if val > hi:
        if warnings is not None:
            warnings.append(f"{name}={val} above maximum {hi}; clamped to {hi}")
        return hi
    return val


def _find_config_file(warnings: list[str]) -> Optional[Path]:
    env_path = os.getenv("MEMORY_ARBITER_CONFIG")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
        warnings.append(f"MEMORY_ARBITER_CONFIG={env_path} does not exist; falling back to XDG config.")
    xdg = Path.home() / ".config" / "memory-arbiter" / "config.json"
    return xdg if xdg.exists() else None


def load_config_file(path: Optional[Path], warnings: list[str]) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data or {}
    except json.JSONDecodeError as exc:
        warnings.append(f"Config file {path} JSON parse failed: {exc}; falling back to env.")
        return {}
    except OSError as exc:
        warnings.append(f"Config file {path} read failed: {exc}; falling back to env.")
        return {}
