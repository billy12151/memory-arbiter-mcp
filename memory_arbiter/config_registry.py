"""Read-only descriptors for current configuration fields.

Since 0.15.0 the user-facing config surface is the 18-file-key slim face;
everything else is a frozen constant (memory_arbiter.constants) and no longer
appears here.
"""
from __future__ import annotations
from typing import Any

GROUPS = [
    {"key": "paths", "label_en": "Paths", "label_zh": "基础路径"},
    {"key": "identity", "label_en": "Identity", "label_zh": "身份"},
    {"key": "server", "label_en": "MCP server", "label_zh": "MCP 服务"},
    {"key": "workspace", "label_en": "Workspace", "label_zh": "工作区"},
    {"key": "embedding", "label_en": "Evidence embedding", "label_zh": "Evidence 向量"},
    {"key": "semantic", "label_en": "Conflict filtering", "label_zh": "冲突降噪"},
    {"key": "update", "label_en": "Update check", "label_zh": "更新检查"},
]


def _item(path: str, group: str, attr: str | None = None, default: Any = None) -> dict[str, Any]:
    label = path.replace("_", " ").replace(".", " / ").title()
    return {
        "path": path, "settings_attr": attr, "default": default, "group": group,
        "label_en": label, "label_zh": label,
        "desc_en": f"Current setting for {path}.", "desc_zh": f"{path} 的当前配置。",
        "editable": False, "restart_required": True, "risk": "caution",
    }


CONFIG_DESCRIPTORS = [
    _item("db_path", "paths"),
    _item("backup_jsonl", "paths"),
    _item("policy_path", "paths"),
    _item("client", "identity"),
    _item("agent_id", "identity"),
    _item("mcp.transport", "server", "mcp_transport", "stdio"),
    _item("mcp.http.host", "server", "mcp_http_host", "127.0.0.1"),
    _item("mcp.http.port", "server", "mcp_http_port", 8000),
    _item("workspace", "workspace", default="default"),
    _item("isolation", "workspace", default="none"),
    _item("embedding.model_path", "embedding", "embedding_model_path"),
    _item("embedding.auto_query", "embedding", "embedding_auto_query", True),
    _item("embedding.auto_write", "embedding", "embedding_auto_write", True),
    _item("semantic_conflict.enabled", "semantic", "semantic_conflict_enabled", False),
    _item("semantic_conflict.model_path", "semantic", "semantic_conflict_model_path"),
    _item("semantic_conflict.on_write", "semantic", "semantic_conflict_on_write", "async"),
    _item("semantic_conflict.max_notice_pairs", "semantic", "semantic_conflict_max_notice_pairs", 2),
    _item("update_check.enabled", "update", "update_check_enabled", True),
]

for _descriptor in CONFIG_DESCRIPTORS:
    if _descriptor["path"] == "isolation":
        _descriptor["label_zh"] = "工作区隔离等级"


def grouped_descriptors() -> list[dict[str, Any]]:
    return [{**group, "items": [item for item in CONFIG_DESCRIPTORS if item["group"] == group["key"]]} for group in GROUPS]
