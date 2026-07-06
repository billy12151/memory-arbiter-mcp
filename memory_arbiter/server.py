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
    def memory_search(query: str = "", workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, include_superseded: bool = False, debug_ranking: bool = False, query_embedding: Optional[list[float]] = None) -> dict[str, Any]:
        """按关键词搜索跨工具共享记忆库。项目知识、历史决策、偏好、文档摘要类问题应先查记忆，再读源文件。搜索技巧：优先用 2-4 个核心词，不要只用整句；一次搜不到先换同义词/短关键词重试；空 query 或 memory_recent 可列出最近记忆；命中 user_confirmed/高置信记忆时优先采用；仅当记忆缺失、过期或冲突时再回原始文件。默认不返回 superseded（已废弃）记忆；审计/梳理历史决策演化链时传 include_superseded=true，废弃记录会排到 active 之后。debug_ranking=true 返回排序调试字段（_match_reason / _subject_level 等），用于评估排序质量。v0.3.1：传 query_embedding（与已灌入 memory_store_embedding 的向量同维度）会在宽召回阶段加一路 vec0 语义近邻，召回字面不重合但语义相近的记忆；不传则只走字面召回（默认行为）。"""
        return tools.memory_search(query=query, workspace=workspace, tags=tags or [], limit=limit, include_superseded=include_superseded, debug_ranking=debug_ranking, query_embedding=query_embedding)

    @app.tool()
    def memory_store_embedding(memory_id: int, embedding: list[float]) -> dict[str, Any]:
        """为指定记忆存入或替换语义向量（v0.3.1 可选语义检索）。memory-arbiter 不内置 embedding 模型——调用方需用任意同维度模型（默认 768 维，可用环境变量 MEMORY_ARBITER_VEC_DIM 配置）自行生成向量后传入。典型用法：跑 docs/semantic_example.py 之类的脚本，把记忆正文批量灌向量。灌完后，memory_search 传 query_embedding 即可走语义召回。"""
        return tools.memory_store_embedding(memory_id=memory_id, embedding=embedding)

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
