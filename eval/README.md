# mema eval harness — 内部质量回归考试台

> 内部设施：`eval/` 不进 pip 发行包（pyproject 只打包 `memory_arbiter*`），
> 克隆仓库即可运行。定位与决策见
> `ZCodeProject/docs/mema-eval-harness-and-demo-plan-2026-09-16.md`（mema #999 转译）。

## 快速跑

```bash
# 全量三套件（recall+similarity 共库无 Qwen ≈5min；conflict 独立库开 Qwen ≈8min）
.venv/bin/python eval/runner.py --suite all --label <标签>

# 只跑召回+自召回 / 只跑相似 / 只跑冲突
.venv/bin/python eval/runner.py --suite recall --label <标签>
.venv/bin/python eval/runner.py --suite similarity --label <标签>
.venv/bin/python eval/runner.py --suite conflict --label <标签>

# 打分 + 报告（JSON+Markdown 落 eval/results/）
.venv/bin/python eval/score.py --run eval/results/all-<标签>.json

# 回归门（对比基线，超相对下降 10% exit 1）
.venv/bin/python eval/score.py --run eval/results/all-<标签>.json \
    --baseline eval/baselines/baseline-0.16.6.json

# 写新基线（大版本或语料换版后）
.venv/bin/python eval/score.py --run eval/results/all-<标签>.json \
    --baseline-write eval/baselines/baseline-<版本>.json
```

模型：embedder 与 Qwen 默认读本机 `~/.config/memory-arbiter/config.json` 的
`embedding.model_path` / `semantic_conflict.model_path` 单值（只取路径，不继承
其他配置——Settings 直构，防配置漂移）；可用 `--embed-model/--qwen-model` 覆盖。

## 组成

```
eval/
  export_fixtures.py        召回语料导出（真库→快照，只读，content_sha 寻址）
  export_conflict_pairs.py  冲突对集导出（治理历史→草稿，只读）
  runner.py                 临时库生命周期 + 三套件执行与采集
  score.py                  指标计算（计数+占比成对）+ 报告 + 回归门
  fixtures/recall/          召回考卷：34 query + 98 target + 200 陪衬 + 标注
  fixtures/similarity/      相似考卷：48 例自然梯度（不凑门）
  fixtures/conflict/        冲突考卷：66 对三形态（scan_evolution /
                            governed_negative / write_opposition）+
                            pairs.raw.jsonl 草稿 + label_overrides.json 自标审计
  baselines/                基线（进 git）；results/ 为每次跑的原始产物（不进 git）
```

## 纪律

- **语料不凑门**：考卷保持自然形态，触不触发交机制表现（owner 2026-09-16
  纠偏）。c3 教训：把 near 类改到必过 0.95/0.8 双门等于送分卷，发现不了
  「判定过严」——首跑 4/12 的真近似提示率正是调阈值的依据。
- **notice 未产出 = FAILED**，不是 skip（owner D2）。
- **每 N 个大版本重抽语料**（防旧考卷刷分）：重跑两个 export 工具、复核
  自标、bump corpus_version、重建基线。
- 真模型进程退出纪律：runner 收尾 shutdown 已处理；直接调
  `temp_library()` 的脚本勿绕过 with。
- conflict 对集注入「对内唯一、对间互斥」的 metadata.entity/scope——
  写时配对硬前提（evidence.py provenance 门），不是可选项。

## 发布门

发版 checklist 增加：`--suite all` 全量跑一次 + `score.py --baseline` 门绿
（provisional 10% 相对下降阈值；正式门槛待 owner 依基线报告拍板）。

## 已知发现（0.16.6 基线，待 owner 裁量后续调优线）

1. 相似提示 0.95/0.8 双门漏掉自然后缀档（「（终版）/细则/修订」ratio
   0.83~0.93 全部不触发），真近似提示率 4/12。
2. 写时冲突检测对 scan 形态（演进/取代未标注）全盲 0/10——两条链路检出
   形态错配，scan 兜底是必要补位，不是冗余。
3. 写时在自己设计的值对立形态上 3/12：4 对被 no_difference 过滤器误杀、
   1 对 provenance 边缘、4 对候选/Qwen 层未成对。
