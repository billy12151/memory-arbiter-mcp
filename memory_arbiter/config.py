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
    # Per-client overrides. Any client *not* listed here defaults to enabled.
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
        # Default-allow: any unrecognised client is enabled.
        return True


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
    policy: AgentPolicy = field(default_factory=AgentPolicy)
    embedding_provider: Optional[str] = None
    embedding_model_path: Optional[Path] = None
    embedding_auto_query: bool = True
    embedding_auto_write: bool = True
    # ── Embedding pipeline params (v0.6.0: part of embedding_space_id) ──
    embedding_n_ctx: int = 2048
    embedding_reserved_tokens: int = 64
    # ── Section split (v0.6.0, advanced feature, all default off) ──
    split_enabled: bool = False
    split_threshold: int = 4000
    section_vec_distance_threshold: float = 0.42
    section_fulltext_threshold: float = 0.8
    max_sections: int = 50
    max_section_chars: int = 3600
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
        split_cfg = cfg.get("split") or {}
        if not isinstance(split_cfg, dict):
            config_warnings.append(f"split={split_cfg!r} invalid; using env/defaults")
            split_cfg = {}
        split_cfg = {str(k): v for k, v in split_cfg.items() if not str(k).startswith("_")}

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
            split_enabled=pick_bool_field(
                split_cfg.get("enabled"), "MEMORY_ARBITER_SPLIT_ENABLED", "false", name="split.enabled", default_bool=False
            ),
            split_threshold=clamp_int(
                pick_int_field(split_cfg.get("threshold"), "MEMORY_ARBITER_SPLIT_THRESHOLD", 4000, name="split.threshold"),
                100, 1_000_000, name="split.threshold", warnings=config_warnings,
            ),
            section_vec_distance_threshold=clamp_float(
                pick_float_field(split_cfg.get("section_vec_distance_threshold"), "MEMORY_ARBITER_SECTION_VEC_DISTANCE_THRESHOLD", 0.42, name="split.section_vec_distance_threshold"),
                0.0, 2.0, name="split.section_vec_distance_threshold", warnings=config_warnings,
            ),
            section_fulltext_threshold=clamp_float(
                pick_float_field(split_cfg.get("section_fulltext_threshold"), "MEMORY_ARBITER_SECTION_FULLTEXT_THRESHOLD", 0.8, name="split.section_fulltext_threshold"),
                0.0, 1.0, name="split.section_fulltext_threshold", warnings=config_warnings,
            ),
            max_sections=clamp_int(
                pick_int_field(split_cfg.get("max_sections"), "MEMORY_ARBITER_MAX_SECTIONS", 50, name="split.max_sections"),
                2, 500, name="split.max_sections", warnings=config_warnings,
            ),
            max_section_chars=clamp_int(
                pick_int_field(split_cfg.get("max_section_chars"), "MEMORY_ARBITER_MAX_SECTION_CHARS", 3600, name="split.max_section_chars"),
                100, 1_000_000, name="split.max_section_chars", warnings=config_warnings,
            ),
        )
        settings.config_warnings = config_warnings
        settings.policy = load_policy(settings.policy_path)
        return settings

    def defaults(self) -> dict[str, str]:
        return {"agent_id": self.agent_id, "workspace": self.workspace}


def load_policy(path: Optional[Path]) -> AgentPolicy:
    if not path or not path.exists():
        return AgentPolicy()
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)
    return AgentPolicy(
        client_defaults=dict(raw.get("client_defaults") or AgentPolicy().client_defaults),
        default_enabled=bool(raw.get("default_enabled", True)),
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
