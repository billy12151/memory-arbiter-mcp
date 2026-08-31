"""Read-only descriptors for current configuration fields."""
from __future__ import annotations
from typing import Any

GROUPS = [
    {"key": "paths", "label_en": "Paths", "label_zh": "基础路径"},
    {"key": "server", "label_en": "MCP server", "label_zh": "MCP 服务"},
    {"key": "retrieval", "label_en": "Retrieval", "label_zh": "召回"},
    {"key": "embedding", "label_en": "Evidence embedding", "label_zh": "Evidence 向量"},
    {"key": "semantic", "label_en": "Conflict filtering", "label_zh": "冲突降噪"},
    {"key": "workspace", "label_en": "Workspace", "label_zh": "工作区"},
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
    _item("db_path", "paths"), _item("backup_jsonl", "paths"),
    _item("mcp.transport", "server", "mcp_transport", "stdio"),
    _item("mcp.http.host", "server", "mcp_http_host", "127.0.0.1"),
    _item("mcp.http.port", "server", "mcp_http_port", 8000),
    _item("mcp.http.path", "server", "mcp_http_path", "/mcp"),
    _item("mcp.http.stateless", "server", "mcp_http_stateless", True),
    _item("recall_pool_cap", "retrieval", default=50), _item("content_like_cap", "retrieval", default=30),
    _item("vec.enabled", "embedding", "enable_sqlite_vec", False), _item("vec.dim", "embedding", "vec_dim", 768),
    _item("embedding.provider", "embedding", "embedding_provider"), _item("embedding.model_path", "embedding", "embedding_model_path"),
    _item("embedding.auto_query", "embedding", "embedding_auto_query", True), _item("embedding.auto_write", "embedding", "embedding_auto_write", True),
    _item("embedding.n_ctx", "embedding", "embedding_n_ctx", 2048), _item("embedding.max_unit_chars", "embedding", "max_section_chars", 3600),
    _item("semantic_conflict.enabled", "semantic", "semantic_conflict_enabled", False),
    _item("semantic_conflict.model_path", "semantic", "semantic_conflict_model_path"),
    _item("semantic_conflict.n_ctx", "semantic", "semantic_conflict_n_ctx", 1024),
    _item("semantic_conflict.job_timeout_ms", "semantic", "semantic_conflict_job_timeout_ms", 5000),
    _item("semantic_conflict.max_notice_pairs", "semantic", "semantic_conflict_max_notice_pairs", 2),
    _item("semantic_conflict.notice_sync_wait_ms", "semantic", "notice_sync_wait_ms", 5000),
    _item("semantic_conflict.workspace_qwen_budget_ms", "semantic", "workspace_qwen_budget_ms", 750),
    _item("isolation", "workspace", default="none"),
    _item("workspace_match_distance", "workspace", default=0.25),
    _item("workspace_weak_vector_weight", "workspace", default=False),
    _item("workspace_min_name_len", "workspace", default=3),
    _item("workspace_recall_admission", "workspace", default=True),
    _item("workspace_recall_cutoff", "workspace", default=0.25),
]

for _descriptor in CONFIG_DESCRIPTORS:
    if _descriptor["path"] == "isolation":
        _descriptor["label_zh"] = "工作区隔离等级"


def grouped_descriptors() -> list[dict[str, Any]]:
    return [{**group, "items": [item for item in CONFIG_DESCRIPTORS if item["group"] == group["key"]]} for group in GROUPS]
