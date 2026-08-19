# Memory Arbiter MCP

Memory Arbiter（迷码，命令 `mema`）让多个 AI 客户端共享同一个本地 SQLite 记忆库，并显式保存来源、事实时间、版本历史、workspace 边界和治理记录。

## 单一 Evidence 主线

每条记忆只保存一份完整原文。后台从 `subject`、Markdown heading、局部语句和必要的重叠窗口生成统一 evidence 向量。语义召回与冲突候选召回都使用这一套索引。

冲突候选按以下流程处理：

1. evidence KNN 找到局部语义相近的文本；
2. 窄规则分为 `notify`、`check`、`ignore`；
3. `notify` 直接生成 advisory notice，不受 Qwen 否决；
4. `check` 只把短 pair 交给可选的本地 Qwen；Qwen 不可用时降级为 `ignore`；
5. Agent 读取两侧完整记忆后终审。

Qwen 只负责降噪，不会编辑记忆、确认冲突或自动废弃任何记录。正式 conflict judgment 使用左右 memory version 作为 CAS 快照固定。

## 四个产品工具

- `memory`：写入、查询、读取、更新、judgment、状态
- `memory_review`：只读检查、历史、冲突、审计
- `memory_govern`：需要用户明确授权的治理操作
- `memory_repair`：evidence 重建、备份恢复、notice 和运行时维护

## 升级与旁路迁移

历史用户升级本版本后，旧库会让 MCP 拒绝启动且不会被修改；推荐用交互式 `mema upgrade`（默认 dry-run 预览，确认后构建新库、核对行数与 fingerprint、原子切换标准 JSON 配置）。等价的底层入口是 `mema migrate-vnext`。迁移只复制当前保留表的公共列，重建 FTS 与 evidence；旧库保持不变，便于回滚。

搜索召回为 lexical 与 evidence 双通道融合（RRF）：FTS/subject/tags 与局部文本向量各自排序后融合，单个长文档最多占一个结果位。

详见 [README.md](README.md) 和 [docs/INTEGRATION.md](docs/INTEGRATION.md)。
