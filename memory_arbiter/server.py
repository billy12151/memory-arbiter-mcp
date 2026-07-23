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
        """写入一条结构化记忆到跨工具共享记忆库。必填 content，建议填 subject/tags/source_type。

tags 是排序和过滤的关键信号（tag 精确命中权重高于 content）。建议打"查询意图词"——用户将来可能用什么词查这条记忆？
  例：发版记录 → tags 含 "发版" + 版本号
      技术决策 → tags 含 "决策" + 主题词
      用户偏好 → tags 含 "偏好" + 偏好类型
避免打 subject 已有的描述词（冗余，不增加检索价值）。

v0.5.0：如果配置了 GGUF embedding + sqlite-vec，写入成功后会自动存向量；响应里仅在尝试过向量化时返回 embedding_stored。"""
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
    def memory_search(query: str = "", workspace: Optional[str] = None, tags: Optional[list[str]] = None, limit: int = 10, include_superseded: bool = False, debug_ranking: bool = False, query_embedding: Optional[list[float]] = None, tags_filter: Optional[list[str]] = None, after_time: Optional[str] = None, before_time: Optional[str] = None, source_type: Optional[str] = None, include_linked_open_items: bool = True, include_conflict_signal: bool = True) -> dict[str, Any]:
        """按相关性检索记忆。limit 是单页大小（默认 10），不是结果上限。返回的 has_more=true 表示还有更多未返回的结果——本版不提供翻页机制，调用方可换更精确的 query、放大 limit（上限 100）、或加 tags_filter 收窄范围。

项目知识、历史决策、偏好、文档摘要类问题应先查记忆，再读源文件。优先用 2-4 个核心词；一次搜不到先换同义词/短关键词重试；空 query 或 memory_recent 可列最近记忆。默认不返回 superseded；审计历史链路时传 include_superseded=true。debug_ranking=true 返回排序调试字段。

参数：
  query: 检索词。为空时仍可调用（返回最近记忆走 fallback），但 tags_filter / after_time / before_time / source_type 在 query 为空时不会独立召回——这些过滤只对 query 召回的 pool 做后过滤，不单独查询。要"列出所有带 X tag 的"用 memory_recent + 客户端过滤。含 ASCII 标识符 + CJK 词时**应空格分隔**(如 "v0.7.2 发版" 而非 "v0.7.2发版")，否则混合 token 走 equality 路径可能漏匹配。
  tags_filter: 严格过滤（AND 语义），记忆的 tags 必须包含这里列出的所有标签。注意：开启 tags_filter 时 vec 语义召回大概率失效（vec 候选 tags 与 query 字面无关，会被精确匹配的 post-filter 砍掉）。
  after_time / before_time: ISO 8601 时间区间（按 ingest_time 过滤，naive 当 UTC）。无效格式会被忽略并返回 warning。
  source_type: 按来源类型过滤（user_confirmed / agent_generated / document_extracted 等）
  include_linked_open_items: 默认 true。命中查询时，在 linked_open_items 字段附最多 5 条同主题的 active 待办（tags 含 todo）。高噪声场景可传 false 关闭。仅在真实命中（retrieval_mode=direct）时触发。

include_conflict_signal: 默认 true。命中结果如涉及 open conflicts，在 result 上附 conflict_signal 字段（来源分 open_table：scan/record 落表的结构化冲突；runtime_metadata_hint：运行时启发式，未经 LLM 验证）。仅在真实命中（retrieval_mode=direct）时触发。

注意：tags_filter 是 AND 语义——必须同时含所有列出的 tag。适合：找最相关的 N 条 / 带过滤条件的穷举式查询。不适合：纯结构化查询（query 为空）。

v0.5.0：配置 GGUF embedding + sqlite-vec 后，不传 query_embedding 也会自动对 query 向量化；显式 query_embedding 仍优先。"""
        return tools.memory_search(query=query, workspace=workspace, tags=tags or [], limit=limit, include_superseded=include_superseded, debug_ranking=debug_ranking, query_embedding=query_embedding, tags_filter=tags_filter, after_time=after_time, before_time=before_time, source_type=source_type, include_linked_open_items=include_linked_open_items, include_conflict_signal=include_conflict_signal)

    @app.tool()
    def memory_get(
        memory_id: int,
        sections: str = "catalog",
        section_ids: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """获取指定 section 的完整原文片段（content[start_offset:end_offset]）+ 元数据。section_ids 为空时返回该 memory 的全部 sections。"""
        return tools.memory_get(memory_id=memory_id, sections=sections, section_ids=section_ids)

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
        """低频诊断工具：比较两条记忆的规则优先级（保护→event_time→source_type→confidence→ingest_time），返回可解释的比较理由，不落冲突记录。日常冲突发现请依赖 memory_search 的 conflict_signal（open_table / runtime_metadata_hint）或 scan_conflict_candidates → record_conflict 工作流。"""
        return tools.memory_compare(left_id=left_id, right_id=right_id)

    @app.tool()
    def memory_arbitrate(left_id: int, right_id: int, mark_conflict: bool = True, apply: bool = False) -> dict[str, Any]:
        """兼容保留的手动仲裁工具。mark_conflict=true 使用旧版 record_conflict 路径（不带 v0.7.5 富化字段），apply=true 自动将非保护败方标记为 superseded。新冲突系统推荐使用 scan_conflict_candidates → record_conflict → list_conflicts → supersede/resolve 工作流；本工具不作为日常入口。"""
        return tools.memory_arbitrate(left_id=left_id, right_id=right_id, mark_conflict=mark_conflict, apply=apply)

    @app.tool()
    def memory_list_conflicts(status: str = "open", limit: int = 50) -> dict[str, Any]:
        """列出记忆冲突记录，默认只看 open 状态。"""
        return tools.memory_list_conflicts(status=status, limit=limit)

    @app.tool()
    def memory_scan_conflict_candidates(
        workspace: Optional[str] = None,
        top_k: int = 8,
        max_pairs: int = 200,
        max_distance: float = 12.0,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """向量召回候选冲突对（无 LLM）。增量扫描（只扫新增 + 最近编辑的记忆）、组对去重、同 workspace 过滤、按 distance 截断。每条记忆取 embedding 跑 top-K 近邻，配对后规范化（left<right）。向量不可用时正常返回 scanned=False + hint（非报错）。agent 拿到候选对后应逐对跑 LLM 比对，再用 memory_record_conflict 落表。**注意：本工具设计用于 agent 侧定时/手动扫描闭环，不建议普通对话主动调用。**"""
        return tools.memory_scan_conflict_candidates(
            workspace=workspace, top_k=top_k, max_pairs=max_pairs,
            max_distance=max_distance, incremental=incremental,
        )

    @app.tool()
    def memory_record_conflict(
        left_id: int,
        right_id: int,
        reason: str,
        conflict_type: Optional[str] = None,
        conflict_point: Optional[str] = None,
        suggested_winner: Optional[int] = None,
        confidence_hint: Optional[str] = None,
        source: Optional[str] = None,
        refresh: bool = False,
        left_version: Optional[int] = None,
        right_version: Optional[int] = None,
        scan_prompt_version: Optional[str] = None,
        scan_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """记录一条冲突，带扫描富化字段（conflict_type/conflict_point/suggested_winner/confidence_hint/source）。规范 pair（left<right）+ 幂等（已有 open 同 pair 返回 deduped=True 不重写）。refresh=true 时更新已有行的富化字段（返回 refreshed），用于 scan 定时任务重判后回写。source 标注建议来源（如 llm_informed）。conflict_type 可为 contradiction/evolution 等；evolution 特指同主题演进残留（stale_active_memory）：新版应 supersede 旧版但两条仍 active。**注意：本工具设计用于 agent 侧定时/手动扫描闭环，不建议普通对话主动调用。**"""
        return tools.memory_record_conflict(
            left_id=left_id, right_id=right_id, reason=reason,
            conflict_type=conflict_type, conflict_point=conflict_point,
            suggested_winner=suggested_winner, confidence_hint=confidence_hint,
            source=source, refresh=refresh,
            left_version=left_version, right_version=right_version,
            scan_prompt_version=scan_prompt_version, scan_model=scan_model,
        )

    @app.tool()
    def memory_resolve_conflict(conflict_id: int, reason: str = "") -> dict[str, Any]:
        """按 conflict_id 关闭单条 open 冲突（dismiss 误报用）。零 schema 变更。与 memory_supersede 的 resolve_conflicts_for（按 memory 关所有相关冲突）不同，本工具只关指定那一条。**注意：本工具设计用于 agent 侧扫描闭环，不建议普通对话主动调用。**"""
        return tools.memory_resolve_conflict(conflict_id=conflict_id, reason=reason)

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
    def memory_doctor_overview(deep: bool = False) -> dict[str, Any]:
        """对 memory-arbiter 做一次健康体检，返回分级诊断报告（只读）。覆盖配置完整性、向量化启用链、分段、数据一致性、容量堆积。每条诊断带 severity 与针对当前配置的 fix_hint。deep=true 时额外实际加载 GGUF 模型做维度探针（秒级开销）。"""
        return tools.memory_doctor_overview(deep=deep)

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
        tags_only: bool = False,
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """原地编辑记忆正文或 tags。三种模式：new_content 整体替换、old_text+new_text 精确局部替换、tags_only=true+add_tags/remove_tags 仅更新 tags。tags-only 模式不写 memory_history、不增加 version、不重算 embedding、不触发重分段（FTS 仍同步 tags）；locked/user_confirmed 需 authorized=true。v0.7.6 起：完成 todo 用 tags_only=true+remove_tags=["todo"]。"""
        return tools.memory_edit(
            memory_id=memory_id,
            new_content=new_content,
            old_text=old_text,
            new_text=new_text,
            new_subject=new_subject,
            new_tags=new_tags,
            reason=reason,
            authorized=authorized,
            tags_only=tags_only,
            add_tags=add_tags,
            remove_tags=remove_tags,
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

    # ── v0.8.0: Section split (Agent continuation/repair entry) ──
    # 日常写入请用 memory_write（规则文档会自动分段；无结构长文返回 split_request
    #  供 Agent 续接）。本工具只用于：收到 split_request 后的内部续接、历史
    #  NULL/failed/declined 修复、active 记录 rebuild。普通写入不要预先调用。

    @app.tool()
    def memory_split(
        memory_id: int,
        split_decision: Optional[str] = None,
        decision_content_hash: Optional[str] = None,
        decision_memory_version: Optional[int] = None,
        decision_split_status: Optional[str] = None,
        decision_split_revision: Optional[int] = None,
        sections: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """长文分段内部续接/修复入口（v0.8.0 重新定位）。两阶段：split_decision=None 返回完整原文+schema+snapshot 供 Agent 用自身 LLM 生成段落元数据；split_decision="split" 校验 anchor/offset/section-size 并原子发布段落+向量。split_decision="rebuild" 重建已 active 记录的派生索引（失败保留旧索引）。日常写入用 memory_write（有规则 heading 的文档会自动分段，无结构长文返回 split_request 由 Agent 续接）；本工具只在收到 split_request、历史 NULL/failed/declined 修复或 active rebuild 时调用，普通写入不要预先调用。"""
        return tools.memory_split(
            memory_id=memory_id,
            split_decision=split_decision,
            decision_content_hash=decision_content_hash,
            decision_memory_version=decision_memory_version,
            decision_split_status=decision_split_status,
            decision_split_revision=decision_split_revision,
            sections=sections,
        )

    @app.tool()
    def memory_rebuild_embeddings(
        memory_ids: Optional[list[int]] = None,
        dry_run: bool = True,
        batch_size: Optional[int] = 50,
    ) -> dict[str, Any]:
        """批量重建向量（v0.6.0）。用于 embedding 模型切换后的向量层迁移，或 ready 状态下的局部修复。不需要 LLM 调用——只重算向量。dry_run=True 只返回清单不执行。"""
        return tools.memory_rebuild_embeddings(
            memory_ids=memory_ids,
            dry_run=dry_run,
            batch_size=batch_size,
        )

    return app


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from .doctor_cli import run_cli
        run_cli(sys.argv[2:])
        return
    try:
        build_server().run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
