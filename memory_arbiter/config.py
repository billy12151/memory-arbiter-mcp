from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import REMOVED_ENV_NAMES

_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


@dataclass
class AgentPolicy:
    # Per-client overrides. Unlisted clients use ``default_enabled``.
    client_defaults: dict[str, bool] = field(default_factory=dict)
    default_enabled: bool = True
    allow_agents: list[str] = field(default_factory=list)
    deny_agents: list[str] = field(default_factory=list)

    def enabled_for(self, client: str | None, agent_id: str | None) -> bool:
        # Identity may be absent (no trusted request identity); an unknown
        # caller is neither denied nor allowed by name and lands on the
        # client/default policy.
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
    """Runtime settings — file-driven since 0.15.0.

    Everything not on this dataclass is a frozen constant (see
    memory_arbiter.constants): engine params, timeouts, thresholds and caps
    were user knobs through 0.14.x and are no longer configurable.
    """

    db_path: Path
    backup_jsonl: Path
    policy_path: Path | None = None
    # No built-in identity: the MCP server refuses to start without an
    # explicitly configured client/agent_id (see server.build_runtime).
    client: str = ""
    agent_id: str = ""
    workspace: str = "default"
    mcp_transport: str = "stdio"
    mcp_http_host: str = "127.0.0.1"
    mcp_http_port: int = 8000
    policy: AgentPolicy = field(default_factory=AgentPolicy)
    # Pointing at a GGUF model IS the intent to embed: no provider or
    # vec.enabled knob any more (one intent, one knob).
    embedding_model_path: Path | None = None
    embedding_auto_query: bool = True
    embedding_auto_write: bool = True
    # Workspace isolation: none (omitted workspace spans the library; an explicit
    # workspace scopes that read) | weak (soft rerank) | strict (hard scope).
    isolation: str = "none"
    update_check_enabled: bool = True
    # semantic_conflict: model_path → auto-enabled (explicit enabled=false
    # wins); preload/resident are frozen true — a configured model loads at
    # startup and stays resident.
    semantic_conflict_enabled: bool = False
    semantic_conflict_model_path: Path | None = None
    semantic_conflict_on_write: str = "async"
    semantic_conflict_max_notice_pairs: int = 2
    config_warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the config file plus launch-context env only.

        Since 0.15.0 every tunable is file-driven. The retained env vars are
        process launch context (which config file, which DB instance, which
        transport, which client identity), not settings; frozen former knobs
        live in memory_arbiter.constants.
        """
        config_warnings: list[str] = []
        for name in REMOVED_ENV_NAMES:
            if os.getenv(name) is not None:
                config_warnings.append(
                    f"{name} is no longer read (config is file-only since 0.15.0); value ignored"
                )
        config_path = _find_config_file(config_warnings)
        cfg = load_config_file(config_path, config_warnings)
        cwd = Path.cwd()

        def section(name: str) -> dict[str, Any]:
            raw = cfg.get(name)
            if isinstance(raw, dict):
                return {str(k): v for k, v in raw.items() if not str(k).startswith("_")}
            if raw is not None:
                config_warnings.append(f"{name}={raw!r} invalid; using defaults")
            return {}

        def warn_removed(prefix: str, data: dict[str, Any], removed: frozenset[str]) -> None:
            for key in sorted(removed.intersection(data)):
                config_warnings.append(
                    f"{prefix}{key} is no longer configurable (frozen at its former default); value ignored"
                )

        warn_removed("", cfg, _REMOVED_TOP_LEVEL_KEYS)
        vec_cfg = section("vec")
        warn_removed("vec.", vec_cfg, _REMOVED_VEC_KEYS)
        emb_cfg = section("embedding")
        warn_removed("embedding.", emb_cfg, _REMOVED_EMBEDDING_KEYS)
        semantic_cfg = section("semantic_conflict")
        warn_removed("semantic_conflict.", semantic_cfg, _REMOVED_SEMANTIC_KEYS)
        mcp_cfg = section("mcp")
        mcp_http_cfg = mcp_cfg.get("http")
        if not isinstance(mcp_http_cfg, dict):
            if mcp_http_cfg is not None:
                config_warnings.append(f"mcp.http={mcp_http_cfg!r} invalid; using defaults")
            mcp_http_cfg = {}
        mcp_http_cfg = {str(k): v for k, v in mcp_http_cfg.items() if not str(k).startswith("_")}
        warn_removed("mcp.http.", mcp_http_cfg, _REMOVED_HTTP_KEYS)

        def pick_str(cfg_key: str, default: str) -> str:
            value = cfg.get(cfg_key)
            if value is None:
                return default
            return str(value)

        def pick_env_str(cfg_key: str, env_key: str, default: str) -> str:
            value = cfg.get(cfg_key)
            if value is not None:
                return str(value)
            return os.getenv(env_key, default)

        def pick_env_path(cfg_key: str, env_key: str, default_path: Path) -> Path:
            value = cfg.get(cfg_key)
            if value is not None:
                return Path(str(value)).expanduser()
            env_value = os.getenv(env_key)
            if env_value:
                return Path(env_value).expanduser()
            return default_path

        def pick_bool_field(cfg_val: Any, name: str, default_bool: bool) -> bool:
            if cfg_val is not None:
                return parse_bool_warn(cfg_val, default_bool, name=name, warnings=config_warnings)
            return default_bool

        def pick_int_field(cfg_val: Any, default: int, name: str) -> int:
            if cfg_val is not None:
                return parse_int(cfg_val, default, name=name, warnings=config_warnings)
            return default

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

        isolation = pick_str("isolation", "none").strip().lower()
        if isolation not in {"none", "weak", "strict"}:
            config_warnings.append(f"isolation={isolation!r} invalid; using none")
            isolation = "none"
        if isolation == "strict":
            config_warnings.append(
                "isolation=strict: review new workspaces before relying on scoped recall; "
                "use confirm_pending_workspace for pending writes, rename/migrate to merge "
                "duplicates, and confirm_workspaces after reviewing the registry."
            )

        mcp_transport = str(
            mcp_cfg.get("transport") or os.getenv("MEMORY_ARBITER_MCP_TRANSPORT") or "stdio"
        ).strip().lower().replace("_", "-")
        if mcp_transport not in {"stdio", "streamable-http"}:
            config_warnings.append(f"mcp.transport={mcp_transport!r} invalid; using stdio")
            mcp_transport = "stdio"

        embedding_model_raw = emb_cfg.get("model_path")

        semantic_on_write = str(semantic_cfg.get("on_write") or "async").strip().lower()
        if semantic_on_write not in {"async", "off"}:
            config_warnings.append(f"semantic_conflict.on_write={semantic_on_write!r} invalid; using async")
            semantic_on_write = "async"
        semantic_model_raw = semantic_cfg.get("model_path")

        # model_path configured but enabled not explicitly set -> auto-enable.
        # One intent shouldn't need two knobs; the user expressed intent by
        # pointing at a model. Explicit enabled=false still wins.
        _semantic_auto_enable = bool(semantic_model_raw)
        if (
            _semantic_auto_enable
            and semantic_cfg.get("enabled") is None
        ):
            config_warnings.append(
                "semantic_conflict.enabled not set; model_path configured -> "
                "auto-enabled (and preloaded at startup). Set enabled=false to disable."
            )

        settings = cls(
            db_path=pick_env_path("db_path", "MEMORY_ARBITER_DB_PATH", cwd / "memory_arbiter.sqlite3"),
            backup_jsonl=pick_env_path("backup_jsonl", "MEMORY_ARBITER_BACKUP_JSONL", cwd / "memory_arbiter.backup.jsonl"),
            policy_path=Path(str(cfg.get("policy_path"))).expanduser() if cfg.get("policy_path") else None,
            client=pick_env_str("client", "MEMORY_ARBITER_CLIENT", ""),
            agent_id=pick_env_str("agent_id", "MEMORY_ARBITER_AGENT_ID", ""),
            workspace=pick_str("workspace", "default"),
            mcp_transport=mcp_transport,
            mcp_http_host=str(mcp_http_cfg.get("host") or "127.0.0.1").strip(),
            mcp_http_port=clamp_int(
                pick_int_field(mcp_http_cfg.get("port"), 8000, name="mcp.http.port"),
                1, 65535, name="mcp.http.port", warnings=config_warnings,
            ),
            embedding_model_path=Path(str(embedding_model_raw)).expanduser() if embedding_model_raw else None,
            embedding_auto_query=pick_bool_field(
                emb_cfg.get("auto_query"), name="embedding.auto_query", default_bool=True
            ),
            embedding_auto_write=pick_bool_field(
                emb_cfg.get("auto_write"), name="embedding.auto_write", default_bool=True
            ),
            isolation=isolation,
            update_check_enabled=update_check_enabled,
            semantic_conflict_enabled=pick_bool_field(
                semantic_cfg.get("enabled"), name="semantic_conflict.enabled",
                default_bool=_semantic_auto_enable,
            ),
            semantic_conflict_model_path=Path(str(semantic_model_raw)).expanduser() if semantic_model_raw else None,
            semantic_conflict_on_write=semantic_on_write,
            semantic_conflict_max_notice_pairs=clamp_int(
                pick_int_field(semantic_cfg.get("max_notice_pairs"), 2, name="semantic_conflict.max_notice_pairs"),
                1, 3, name="semantic_conflict.max_notice_pairs", warnings=config_warnings,
            ),
        )
        settings.config_warnings = config_warnings
        settings.policy = load_policy(settings.policy_path, config_warnings)
        return settings

    def defaults(self) -> dict[str, str]:
        return {"agent_id": self.agent_id, "workspace": self.workspace}


# Former config keys that are frozen constants now (0.15.0). Present in a
# config file they produce a one-line deprecation warning and are ignored.
_REMOVED_TOP_LEVEL_KEYS = frozenset({
    "tool_profile", "recall_pool_cap", "content_like_cap", "superseded_limit",
    "workspace_match_distance", "workspace_qwen_candidate_distance",
    "workspace_qwen_candidate_top_k", "workspace_weak_vector_weight",
    "workspace_min_name_len", "workspace_recall_admission", "workspace_recall_cutoff",
})
_REMOVED_VEC_KEYS = frozenset({"enabled", "dim"})
_REMOVED_EMBEDDING_KEYS = frozenset({"provider", "n_ctx", "reserved_tokens", "max_unit_chars"})
_REMOVED_HTTP_KEYS = frozenset({"path", "stateless", "json_response", "max_request_body_size"})
_REMOVED_SEMANTIC_KEYS = frozenset({
    "backend", "max_concurrency", "queue_max_size", "n_ctx", "n_threads", "n_batch",
    "resident", "preload", "job_timeout_ms", "inference_timeout_ms", "load_timeout_ms",
    "min_pair_budget_ms", "max_evidence_units", "scan_enhance", "scan_max_pairs",
    "scan_budget_ms", "notice_sync_wait_ms", "workspace_qwen_budget_ms",
})


def load_policy(path: Path | None, warnings: list[str] | None = None) -> AgentPolicy:
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


def parse_bool_warn(val: Any, default: bool, name: str = "", warnings: list[str] | None = None) -> bool:
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


def parse_int(val: Any, default: int, name: str = "", warnings: list[str] | None = None) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        if warnings is not None and val is not None:
            warnings.append(f"{name}={val!r} invalid; using default {default}")
        return default


def parse_float(val: Any, default: float, name: str = "", warnings: list[str] | None = None) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        if warnings is not None and val is not None:
            warnings.append(f"{name}={val!r} invalid; using default {default}")
        return default


def clamp_int(val: int, lo: int, hi: int, name: str = "", warnings: list[str] | None = None) -> int:
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


def clamp_float(val: float, lo: float, hi: float, name: str = "", warnings: list[str] | None = None) -> float:
    """Clamp a finite float to [lo, hi], warning and using lo for NaN/Inf."""
    if not math.isfinite(val):
        if warnings is not None:
            warnings.append(f"{name}={val} is not finite; using minimum {lo}")
        return lo
    if val < lo:
        if warnings is not None:
            warnings.append(f"{name}={val} below minimum {lo}; clamped to {lo}")
        return lo
    if val > hi:
        if warnings is not None:
            warnings.append(f"{name}={val} above maximum {hi}; clamped to {hi}")
        return hi
    return val


def _find_config_file(warnings: list[str]) -> Path | None:
    env_path = os.getenv("MEMORY_ARBITER_CONFIG")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
        warnings.append(f"MEMORY_ARBITER_CONFIG={env_path} does not exist; falling back to XDG config.")
    xdg = Path.home() / ".config" / "memory-arbiter" / "config.json"
    return xdg if xdg.exists() else None


def load_config_file(path: Path | None, warnings: list[str]) -> dict[str, Any]:
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
