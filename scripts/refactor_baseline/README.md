# 阶段 -1/0 安全网基线（2026-08-09，tag: pre-refactor-baseline）

重构实施前的安全网产物。所有工具均可重跑；阶段 3/4/5 每次提交前应重新核对。

## 测试基线

- 命令：`.venv/bin/python -m pytest -q`
- 结果：**609 passed in 26.73s**（interpreter: `.venv/bin/python` = Python 3.12.13，pytest 9.1.1；**勿用 `.venv-test`（3.14）**）
- 收集数：609 tests collected
- tag：`pre-refactor-baseline`

## 产物清单

| 文件 | 用途 | 对应风险 |
|---|---|---|
| `snapshot_symbols.py` → `symbols_before.json` | 固化 db/tools/search/doctor 四个 facade 模块（204 个模块符号）+ MemoryDB/MemoryTools（194 个方法）的名字、`inspect.signature`、`__module__`、`__qualname__`。阶段后用 `diff_symbols.py` 比对，名字消失/签名漂移/`__module__` 非白名单变化即失败 | R6 |
| `monkeypatch_inventory.py` → `monkeypatch_inventory.md` | 扫出 17 处 monkeypatch/mock.patch 命中 | R4 |
| `monkeypatch_triage.md` | **人工处置表**：17 处逐条标注 (a) 保留 facade 委托 或 (b) 迁 patch 目标，附"实例属性遮蔽/模块命名空间"两条执行约束 | R4 |
| `getattr_attribution.py` → `getattr_attribution.md` | 55 处 `getattr(settings,…,default)` 归因：0 NOT-A-FIELD，唯一 MISMATCH（db.py:881）经核实为真等价。阶段 2 可全部清零 | R5 |
| `mypy_strict_baseline.txt` + `check_mypy.py` | 175 个存量 strict error 白名单；`check_mypy.py` 在重构期间检测**新增** type error（存量消失不算失败） | §9.4 |

## 关键结论（已核实）

1. **getattr 可全清零**：55 处中 0 处读了非字段属性；db.py:881 的 `None` 兜底被后续 `0.25 if configured is None` 中和，与字段默认 0.25 等价。
2. **monkeypatch 主要风险**：`tools.db.publish_memory_claims` / `find_structured_claim_pairs` 等 claim/judgment 转发方法在 §4.1 被去委托改 `db.claims.*`，对应 patch 目标必须同步迁移（b 类）；`tools.db.is_pair_dismissed` / `tools._publish_sections` 等保留 facade 委托即可靠实例属性遮蔽命中（a 类），但**调用方必须用实例调用形式**。
3. **类型基线**：175 个存量 error 集中在 server.py(69, MCP 装饰器)/tools.py(42)。check_mypy.py 用于防新增。

## 阶段 3/4/5 提交前 checklist

```bash
.venv/bin/python -m pytest -q                                   # 全绿
.venv/bin/python scripts/refactor_baseline/check_mypy.py        # 无新增 type error
.venv/bin/python scripts/refactor_baseline/monkeypatch_inventory.py  # 无遗漏 patch 点
.venv/bin/python scripts/refactor_baseline/snapshot_symbols.py --out /tmp/symbols_after.json
.venv/bin/python scripts/refactor_baseline/diff_symbols.py scripts/refactor_baseline/symbols_before.json /tmp/symbols_after.json
```
