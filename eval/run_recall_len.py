#!/usr/bin/env python3
"""run_recall_len — recall-v3-len harness 跑分包装：文档侧前缀变体.

用法: run_recall_len.py {qxq|bare|official}
查询侧保持产品默认（EMBED_PREFIX_SEARCH 不动），只改写入侧 EMBED_PREFIX_STS——
即「未来若拍板改文档侧前缀」的最小差异实验。floor/池/融合全部产品默认。
"""
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

tag = sys.argv[1]
DOC = {
    # 历史 qxq 消融臂（EMBED_PREFIX_STS 曾=query 前缀）。产品侧该常量已改
    # 空串（裸文本，0.17.0 终局），qxq 与 bare 两臂现恒等——保留仅为结果
    # 文件名兼容。
    "qxq": None,
    "bare": "",                        # 写入侧裸文本
    "official": "title: none | text: ",  # EmbeddingGemma 官方文档提示
}
if tag not in DOC:
    raise SystemExit(f"unknown tag: {tag}")
val = DOC[tag]
if val is not None:
    import memory_arbiter.constants as C
    C.EMBED_PREFIX_STS = val
    for name in (
        "memory_arbiter.pipeline.write",
        "memory_arbiter.pipeline.evidence",
        "memory_arbiter.db.workspaces",
        "memory_arbiter.tools",
        "memory_arbiter.semantic_conflict",
        "memory_arbiter.vnext_migration",
    ):
        mod = importlib.import_module(name)
        if hasattr(mod, "EMBED_PREFIX_STS"):
            mod.EMBED_PREFIX_STS = val
    print(f"[prefix patch] doc side EMBED_PREFIX_STS={val!r}; query side = product default")
else:
    print("[prefix patch] none (product default = bare text; same as the qxq arm)")

sys.argv = [
    "runner", "--suite", "recall",
    "--recall-dir", str(REPO / "eval" / "fixtures" / "recall-len"),
    "--label", f"len3-{tag}",
    "--out", str(REPO / "eval" / "results"),
]
import importlib.util
spec = importlib.util.spec_from_file_location("harness_runner", REPO / "eval" / "runner.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
raise SystemExit(m.main())
