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
        """写入一条结构化记忆到跨工具共享记忆库。必填 content，建议填 subject/tags/source_type。v0.5.0：如果配置了 GGUF embedding + sqlite-vec，写入成功后会自动存向量；响应里仅在尝试过向量化时返回 embedding_stored。"""
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
        """搜索跨工具共享记忆库。项目知识、历史决策、偏好、文档摘要类问题应先查记忆，再读源文件。优先用 2-4 个核心词；一次搜不到先换同义词/短关键词重试；空 query 或 memory_recent 可列最近记忆。默认不返回 superseded；审计历史链路时传 include_superseded=true。debug_ranking=true 返回排序调试字段。v0.5.0：配置 GGUF embedding + sqlite-vec 后，不传 query_embedding 也会自动对 query 向量化；显式 query_embedding 仍优先。"""
        return tools.memory_search(query=query, workspace=workspace, tags=tags or [], limit=limit, include_superseded=include_superseded, debug_ranking=debug_ranking, query_embedding=query_embedding)

    @app.tool()
    def memory_store_embedding(memory_id: int, embedding: list[float]) -> dict[str, Any]:
        """为指定记忆手动存入或替换语义向量。v0.5.0 配置 GGUF embedding 后，新写入/普通查询可自动向量化；这个工具仍适合 backfill、非 GGUF 模型、远程 API 或自定义向量流程。向量维度必须匹配 vec.dim。"""
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
        """查看 memory-arbiter 运行状态：数据库路径、降级模式、客户端标识、策略配置、配置解析 warning、自动 embedding 是否已配置。"""
        return tools.memory_status()

    @app.tool()
    def memory_audit_summary() -> dict[str, Any]:
        """返回各 workspace 的记忆统计概览：条目数、最旧/最新条目时间、open 冲突数、各 source_type 分布。纯 SQL 聚合，不做语义判断，用于快速判断是否需要深入审查。"""
        return tools.memory_audit_summary()

    @app.tool()
    def memory_edit(
        memory_id: int,
        new_content: Optional[str] = None,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        new_subject: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
        reason: str = "",
        authorized: bool = False,
    ) -> dict[str, Any]:
        """原地编辑记忆正文，旧版本自动存入 memory_history 版本链并同步 FTS。两种模式：传 new_content 整体替换，或 old_text+new_text 精确局部替换。normal 记忆可直接编辑；locked/user_confirmed 需 authorized=true。v0.5.0：若配置自动 embedding，编辑成功后会重算向量；重算/写入失败会删除旧向量，避免语义召回仍按旧内容命中。"""
        return tools.memory_edit(
            memory_id=memory_id,
            new_content=new_content,
            old_text=old_text,
            new_text=new_text,
            new_subject=new_subject,
            new_tags=new_tags,
            reason=reason,
            authorized=authorized,
        )

    @app.tool()
    def memory_history(memory_id: int) -> dict[str, Any]:
        """查看一条记忆的版本演化轨迹（memory_history 表的历史快照，按版本号倒序）。只读，不动任何表。配合 memory_edit 使用：每次编辑前的旧正文都存在这里，必要时可人工恢复。"""
        return tools.memory_history(memory_id=memory_id)

    @app.tool()
    def memory_cleanup_history(
        memory_id: Optional[int] = None,
        older_than_days: Optional[int] = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        """清理 memory_history 表的历史快照（不碰 memories 活跃记录）。三种粒度：传 memory_id 只清指定记忆的历史；传 older_than_days 只清 N 天前的快照；两者都不传=全量瘦身，必须 authorized=true 作为确认门。绝对安全：无论传什么参数，只 DELETE FROM memory_history，memories 表一条都不动。"""
        return tools.memory_cleanup_history(
            memory_id=memory_id,
            older_than_days=older_than_days,
            authorized=authorized,
        )

    return app


def main() -> None:
    try:
        build_server().run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
