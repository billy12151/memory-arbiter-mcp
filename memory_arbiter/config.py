from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


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

    @classmethod
    def from_env(cls) -> "Settings":
        cwd = Path.cwd()
        policy_raw = os.getenv("MEMORY_ARBITER_POLICY")
        settings = cls(
            db_path=Path(os.getenv("MEMORY_ARBITER_DB_PATH", cwd / "memory_arbiter.sqlite3")).expanduser(),
            backup_jsonl=Path(os.getenv("MEMORY_ARBITER_BACKUP_JSONL", cwd / "memory_arbiter.backup.jsonl")).expanduser(),
            policy_path=Path(policy_raw).expanduser() if policy_raw else None,
            client=os.getenv("MEMORY_ARBITER_CLIENT", "codex"),
            agent_id=os.getenv("MEMORY_ARBITER_AGENT_ID", "default"),
            workspace=os.getenv("MEMORY_ARBITER_WORKSPACE", "default"),
            enable_sqlite_vec=os.getenv("MEMORY_ARBITER_ENABLE_SQLITE_VEC", "false").lower() in {"1", "true", "yes"},
            vec_dim=int(os.getenv("MEMORY_ARBITER_VEC_DIM", "768")),
            recall_pool_cap=int(os.getenv("MEMORY_ARBITER_RECALL_POOL_CAP", "50")),
            content_like_cap=int(os.getenv("MEMORY_ARBITER_CONTENT_LIKE_CAP", "30")),
        )
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
