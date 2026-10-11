# 从旧版本升级（重要，先看再动）

> 操作指南，已从 README 移出（README 只做产品门面）。适用于 schema 世代早于 `workspace_state_v1` 的数据库（0.14.8 之前的 claim/向量世代数据库启动时直接拒绝，必须走迁移）。



> ⚠️ **0.14.8 升级警告：** 当前 schema generation 仍为 `workspace_state_v1`；更早的数据库**直接启动会被拒绝**，必须走迁移命令。迁移会**丢弃旧的冲突/裁决/提醒历史**（记忆原文和历史都保留），升级完成后需要跑一遍全库扫描重新发现冲突。结构迁移与向量兼容现在独立判断：空间不一致会禁用向量并等待显式重建，不再阻断结构升级。

```bash
# 1. 先预览，看看会动什么（不动真格）
mema upgrade --dry-run

# 2. 停掉所有连着迷码的 AI 客户端，然后做备份：
sqlite3 /绝对路径/memory.sqlite3 "PRAGMA wal_checkpoint(TRUNCATE);"
cp /绝对路径/memory.sqlite3 /绝对路径/memory.pre-0.14.sqlite3

# 3. 正式迁移（会自动切换配置；旧数据库文件永远不会被删）
mema upgrade

# 4. 重启客户端，验证
mema doctor --json
```

为什么要先跑那条 `sqlite3` 命令？因为数据库的"最新流水"可能还在 WAL 暂存文件里，只拷主文件会拷丢。这条命令先把流水并入主文件，再拷贝才是完整的。

**升级后要做点什么吗？** 不用。第一次启动迷码会自己开个后台线程把语句向量补齐（记忆的"每句话向量"，冲突检测用）。进度看 `mema doctor` 的 `rows.coverage`；没装向量模型时它会先挂起，装好后自动继续，全程不用手动跑脚本。



## 迁移的两种路径

并排迁移分 preserve（保留）与 rebuild（重建）两种，由每条 schema 迁移自己声明 `vector_effect=preserve|rebuild`。preserve 路线原样克隆 FTS/证据/向量载荷，只在事务内重建冲突域；rebuild 路线重新生成证据和向量。向量健康或空间不匹配从不改变结构迁移的结果——空间不匹配单独记录为 `state=mismatch`、禁用向量读取，之后用 `memory_repair(rebuild_evidence)` 修复即可。两条路线都有意从空的 `conflicts`/通知状态起步：不拷旧 `conflicts`、追加式 `conflict_judgments`、`semantic_notices` 历史——现行矛盾由一次全库扫描重新发现。

## 重建路径的前置条件

完整重建需要：sqlite-vec、可读的本地 GGUF embedding 模型、`llama-cpp-python`（装 `semantic-local` extra 即可）、可写的目标目录和足够磁盘。preserve 迁移不加载任何模型、不要求向量完备。判定模型 checkpoint 从来不是迁移的前置条件。命令会在请求确认前报告所选模式、向量效果/兼容性、记忆数、预计向量工作量、剩余磁盘、源与目标。

## `--no-switch` 与 `--yes`

`--no-switch`：只构建并校验新库，不动配置——适合先看看结果再决定。`--yes`：跳过交互确认**和**"所有写入者已停止、旧冲突/裁决/提醒历史将被永久舍弃"的确认——但它不会替你停进程、不会 checkpoint 源库、也不会创建回滚备份。低层的 `mema migrate-vnext` 命令保留用于诊断。

## 旧库怎么办

源数据库永远不会被删除。标准 JSON 配置在完整校验通过后才备份并切换；环境变量形式的 `db_path` 覆盖会作为手动操作上报。请在新库正常使用一段时间之后再处理旧库；如果新库已经接收了写入，不要直接切回——先盘点这批更新的记录。
