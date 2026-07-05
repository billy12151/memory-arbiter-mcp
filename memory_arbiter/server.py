from __future__ import annotations

import sys
from typing import Any, Optional

from .config import Settings
from .tools import MemoryTools


def build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError(
            "MCP Python SDK is not installed. Install with `pip install -r requirements.txt` "
            "or `pip install mcp`, then run `memory-arbiter-mcp` again."
        ) from exc

    app = FastMCP("memory-arbiter-mcp")
    tools = MemoryTools(Settings.from_env())

    @app.tool()
    def memory_write(
        content: str,
        agent_id: Optional[str] = None,
        workspace: Optional[str] = None,
        tags: Optional[list[str]] = None,
        source_type: str = "unknown",
        source_ref: Optional[str] = None,
        event_time: Optional[str] = None,
        ingest_time: Optional[str] = None,
        confidence: float = 0.5,
        protection_level: str = "normal",
        status: str = "active",
        subject: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """写入一条结构化记忆到跨工具共享记忆库。必填 content（正文），建议填 subject（标题）、tags（标签）、source_type（来源类型：agent_generated/user_confirmed/document_extracted）。"""
        return tools.memory_write(
            content=content,
            agent_id=agent_id,
            workspace=workspace,
            tags=tags or [],
            source_type=source_type,
            source_ref=source_ref,
            event_time=event_time,
            ingest_time=ingest_time,
            confidence=confidence,
            protection_level=protection_level,
            status=status,
            subject=subject,
            metadata=metadata or {},
        )

    @app.tool()
    def memory_search(query: str = "", workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10) -> dict[str, Any]:
        """按关键词搜索跨工具共享记忆库。项目知识、历史决策、偏好、文档摘要类问题应先查记忆，再读源文件。搜索技巧：优先用 2-4 个核心词，不要只用整句；一次搜不到先换同义词/短关键词重试；空 query 或 memory_recent 可列出最近记忆；命中 user_confirmed/高置信记忆时优先采用；仅当记忆缺失、过期或冲突时再回原始文件。"""
        return tools.memory_search(query=query, workspace=workspace, tags=tags or [], limit=limit)

    @app.tool()
    def memory_recent(workspace: Optional[str] = None, limit: int = 20) -> dict[str, Any]:
        """列出指定 workspace 最近记忆，不按关键词过滤。用于关键词不确定、memory_search 直接命中为空、或需要先浏览库存再决定是否读源文件的场景。"""
        return tools.memory_recent(workspace=workspace, limit=limit)

    @app.tool()
    def memory_compare(left_id: int, right_id: int) -> dict[str, Any]:
        """比较两条记忆是否冲突，返回可解释的比较理由，不落冲突记录。"""
        return tools.memory_compare(left_id=left_id, right_id=right_id)

    @app.tool()
    def memory_arbitrate(left_id: int, right_id: int, mark_conflict: bool = True, apply: bool = False) -> dict[str, Any]:
        """仲裁两条冲突记忆的胜者与败者。mark_conflict=true 记录冲突，apply=true 自动将非保护败方标记为 superseded。"""
        return tools.memory_arbitrate(left_id=left_id, right_id=right_id, mark_conflict=mark_conflict, apply=apply)

    @app.tool()
    def memory_list_conflicts(status: str = "open", limit: int = 50) -> dict[str, Any]:
        """列出记忆冲突记录，默认只看 open 状态。"""
        return tools.memory_list_conflicts(status=status, limit=limit)

    @app.tool()
    def memory_confirm(memory_id: int, source_ref: Optional[str] = None, confidence: float = 1.0) -> dict[str, Any]:
        """将一条记忆标记为用户确认，提升为 user_confirmed + locked 保护级别，禁止自动覆盖。"""
        return tools.memory_confirm(memory_id=memory_id, source_ref=source_ref, confidence=confidence)

    @app.tool()
    def memory_supersede(
        memory_id: int,
        reason: str,
        superseded_by: Optional[int] = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """显式废弃一条记忆，可突破 user_confirmed/locked 保护（memory_arbitrate 被挡时用）。必须 authorized=true 才执行；联动降保护级别并把相关 open 冲突标记为 resolved，审计记录写入 conflicts 表。废弃后不可逆。"""
        return tools.memory_supersede(
            memory_id=memory_id,
            reason=reason,
            superseded_by=superseded_by,
            authorized=authorized,
        )

    @app.tool()
    def memory_status() -> dict[str, Any]:
        """查看 memory-arbiter 运行状态：数据库路径、降级模式、客户端标识、策略配置。"""
        return tools.memory_status()

    @app.tool()
    def memory_audit_summary() -> dict[str, Any]:
        """返回各 workspace 的记忆统计概览：条目数、最旧/最新条目时间、open 冲突数、各 source_type 分布。纯 SQL 聚合，不做语义判断，用于快速判断是否需要深入审查。"""
        return tools.memory_audit_summary()

    return app


def main() -> None:
    try:
        build_server().run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
