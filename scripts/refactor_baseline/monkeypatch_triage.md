# R4 — monkeypatch 处置清单（阶段 3/4/5 的共同前置）

> 生成于 2026-08-09，基于 `pre-refactor-baseline`。由 `monkeypatch_inventory.py` 重新生成原始命中。
> 原则：facade 委托必须让旧 patch 点仍作用于真实执行路径（选 a）；只有当实现位置迁移导致旧委托消失时，才同步迁 patch 目标并在该提交 PR 中显式标注（选 b）。验收标准 §9.5：不改断言逻辑，patch 目标可随实现位置迁移但须证明真实生效。

## 处置决策表

| 测试位置 | patch 目标 | 移动到 | 处置 | 理由 / 执行动作 |
|---|---|---|---|---|
| test_doctor_unit.py:141,146 | `doc._check_db_size`（doctor 模块级函数） | doctor_checks/integrity.py | **a** | `doctor/__init__`（或瘦壳 doctor.py）保留 `_check_db_size` 名字绑定到 integrity 实现；monkeypatch 改的是模块属性，瘦壳导入处需 `from .doctor_checks.integrity import _check_db_size` 使 `doc._check_db_size` 可写。**注意**：`run_all_checks` 必须经模块属性解析 `_check_db_size`（而非提前绑定引用），否则 patch 打不中——注册表 CHECKS 用惰性解析。 |
| test_memory_arbiter.py:1828,1857 | `memory_arbiter.tools.search_memories`（tools 模块内 import 的函数） | search/ 包入口 | **a** | tools.py 顶部 `from .search import search_memories` 的兼容 import 保留（§6.1 已列）。monkeypatch 替换的是 tools 模块命名空间里的 `search_memories`，memory_search 内部必须以模块级名字调用 `search_memories(...)` 才能被 patch 命中。**约束**：拆分时 memory_search 不得改成 `from .search import ...` 后以本地别名或直接 `search.recall.search_memories` 调用绕过模块命名空间。 |
| test_update_monitor.py:248 | `UpdateMonitor._load_state`（类方法） | 不移动 | — | update_monitor.py 不在拆分范围。无动作。 |
| test_v074_linked_open_items.py:497 | `memory_arbiter.server.Settings.from_env` | 不移动 | — | server.py 仅 re-export Settings 用法，不拆分。无动作。 |
| test_v076_conflict_signals.py:598 | `tools.db.is_pair_dismissed`（MemoryDB 实例方法） | db/conflicts.py store | **a** | `MemoryDB.is_pair_dismissed` 以 facade 委托保留（委托到 conflicts store）。monkeypatch 改的是实例属性 `tools.db.is_pair_dismissed`，**实例属性遮蔽类方法**——只要 signals 管线经 `self.db.is_pair_dismissed(...)`（实例调用）而非 `MemoryDB.is_pair_dismissed(self.db,...)`（类调用）即可命中。**约束**：signals.py 内部必须用实例调用形式。 |
| test_v080_protocol.py:296,321 | `tools._publish_sections`（MemoryTools 实例方法） | pipeline/sections.py | **a** | `MemoryTools._publish_sections` 以 facade 委托保留（委托到 SectionPipeline）。同上是实例属性遮蔽，SplitReindexWorker._process_one 经 `self._tools._publish_sections(...)` 调用即可命中。**约束**：worker 与 _after_write_split 必须经 `self._tools._publish_sections`（实例）调用，不得直接持有 SectionPipeline 引用后绕过 tools。 |
| test_v090_structured_claims.py:750,881 | `tools.db.publish_memory_claims` | db 去委托 → db.claims | **b** | §4.1 删除该 26 个转发方法，调用点改 `db.claims.publish_memory_claims`。patch 目标需迁为 `tools.db.claims.publish_memory_claims`（实例属性遮蔽）。**提交时显式标注**。 |
| test_v090_structured_claims.py:789,795 | `tools.db.find_structured_claim_pairs` | db 去委托 → db.claims | **b** | 同上，迁为 `tools.db.claims.find_structured_claim_pairs`。 |
| test_v090_structured_claims.py:853,864 | `tools.db.record_conflict_enriched` | db/conflicts.py store | **a** | `MemoryDB.record_conflict_enriched` 属 conflicts store，但**不在 §4.1 的 26 个删除清单**（那 26 个是 claim/judgment 转发）。facade 委托保留，实例调用即可命中。若最终决定也归入 store 去委托，则降级为 **b** 并迁目标。 |
| test_v090_structured_claims.py:766 | `memory_arbiter.tools.extract_claims`（tools 模块内 import 的函数） | claims.py（不移动，tools 仅 import） | **a** | tools.py 顶部 `from .claims import extract_claims` 保留。monkeypatch 替换 tools 命名空间的 `extract_claims`；`_index_and_reconcile_claims(_impl)` 必须以模块级名字 `extract_claims(...)` 调用。**约束**：pipeline/write.py 搬运 claims 索引逻辑时，仍经 `self._tools` 所在模块命名空间解析 extract_claims，或在 write.py 内 `from ..claims import extract_claims` 并以模块名调用，确保 patch 到 tools 命名空间仍生效——若 write.py 自成命名空间则此处需改为 **b** 并把 patch 迁到 `memory_arbiter.pipeline.write.extract_claims`。 |

## 关键执行约束（跨阶段通用）

1. **实例属性遮蔽原则**：monkeypatch 设置实例属性会遮蔽同名类方法。facade 委托要有效，**调用方必须用实例调用**（`obj.method(...)`），不能用类名调用（`Cls.method(obj,...)`）——否则实例 patch 失效。
2. **模块命名空间原则**：对模块级函数 patch（search_memories / extract_claims），调用方必须以**被 patch 模块的模块级名字**调用，不能 `from X import f` 后在别处缓存引用。
3. **b 类迁移**：去委托的 claim/judgment 方法（publish_memory_claims / find_structured_claim_pairs / 其余 24 个转发）一旦改调用点为 `db.claims.*`/`db.judgments.*`，对应 patch 目标同步迁移，且在同一提交内完成（不改断言逻辑，只改 patch 路径）。
4. 每次阶段 3/4/5 提交前重跑 `monkeypatch_inventory.py` 对比，确认无遗漏新 patch 点。
