#!/usr/bin/env python3
"""split_tool.py — 大文件拆分机械抽取器（拆分批自用，不进发行包）。

用法（对单个类做 mixin 拆分）：
  python3 scripts/split_tool.py <source.py> <ClassName> <target.py> <MixinName> \\
      <method1,method2,...> [--header-comment "..."]

行为：
1. 在 source.py 里定位 `class ClassName:`，按 4 空格缩进的 def 边界找出
   每个列名方法的完整块（含 decorator 与紧邻的类内注释行）。
2. 剪出块写入 target.py：模块 docstring + 拆分批标记 + source 的完整
   import 块 + mixin 类壳 + 方法块（缩进不变，mixin 内层级相同）。
3. source.py 改为：import target 的 Mixin + ClassName 基类列表加 Mixin。
4. 输出摘要：每个方法的行数、target 总行数、source 剩余行数。

不做的事（人工做）：import 修剪（ruff 对新文件 --fix）、跨 mixin 静态互调
改写、TYPE_CHECKING 声明、re-export。纯文本操作，不执行任何被拆代码。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def find_class_block(lines: list[str], class_name: str) -> tuple[int, int]:
    """返回 (class 行号, 类体结束行号 exclusive)。"""
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^class {class_name}\b", ln):
            start = i
            break
    if start is None:
        sys.exit(f"class {class_name} not found")
    # 类体结束 = 下一个顶层语句（非空且缩进为 0）
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln.startswith((" ", "\t", ")")) and not ln.startswith(")"):
            # 顶层 def/class/赋值
            if not ln.startswith("#"):
                end = j
                break
    return start, end


def find_method_spans(
    lines: list[str], class_start: int, class_end: int, names: list[str]
) -> dict[str, tuple[int, int]]:
    """在类体内找每个方法的 (start, end) 行号（0-based, end exclusive）。"""
    # 收集类体内所有 4 缩进的 def/@ 行
    marks: list[tuple[int, str, bool]] = []  # (line_idx, name, is_decor_or_comment)
    for i in range(class_start + 1, class_end):
        ln = lines[i]
        m = re.match(r"^    (?:@|def )", ln)
        if not m:
            continue
        marks.append((i, ln.strip(), ln.strip().startswith("@")))
    # 展开成方法段（decor 行归属后续 def）
    spans: dict[str, tuple[int, int]] = {}
    entries: list[tuple[int, int, str]] = []  # (start, end, name)
    pending_decor_start: int | None = None
    for k, (i, text, is_decor) in enumerate(marks):
        if is_decor:
            if pending_decor_start is None:
                pending_decor_start = i
            continue
        m = re.match(r"def (\w+)", text)
        if not m:
            continue
        name = m.group(1)
        start = pending_decor_start if pending_decor_start is not None else i
        pending_decor_start = None
        # end = 下一个 mark 的行（decor 归属它）或类体尾
        end = class_end
        if k + 1 < len(marks):
            end = marks[k + 1][0]
            # 保留方法间紧邻的空行给前一个方法：回退连续空行
            while end > start + 1 and lines[end - 1].strip() == "":
                end -= 1
            if k + 1 < len(marks) and marks[k + 1][2]:  # 下一是 decorator
                pass  # end 停在 decorator 行
        entries.append((start, end, name))
    for start, end, name in entries:
        spans[name] = (start, end)
    missing = [n for n in names if n not in spans]
    if missing:
        sys.exit(f"methods not found in class body: {missing}")
    return spans


def extract_import_block(lines: list[str]) -> tuple[int, int]:
    """文件头到第一个非 import/注释/docstring 顶层的行。"""
    end = 0
    in_docstring = False
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if i == 0 and stripped.startswith('"""'):
            if stripped.count('"""') >= 2:
                continue
            in_docstring = True
            continue
        if in_docstring:
            if '"""' in stripped:
                in_docstring = False
            continue
        if (
            stripped.startswith(("import ", "from "))
            or stripped.startswith("#")
            or stripped == ""
            or stripped.startswith(")")
            or stripped.startswith(("\"", "'"))
            or (stripped.endswith("(") and stripped.startswith(("from ", "import ")))
            or (lines[i - 1].rstrip().endswith(("(", ",")) and not stripped.startswith(("def ", "class ", "@")))
        ):
            end = i + 1
            continue
        if stripped.startswith(("from ", "import ")) or stripped == "":
            end = i + 1
            continue
        break
    return 0, end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("class_name")
    ap.add_argument("target")
    ap.add_argument("mixin_name")
    ap.add_argument("methods")  # 逗号分隔
    ap.add_argument("--doc", default="")
    ap.add_argument("--typechecking-extra", default="")  # 逗号分隔的额外 TYPE_CHECKING import 源
    args = ap.parse_args()

    src_path = Path(args.source)
    tgt_path = Path(args.target)
    lines = src_path.read_text().splitlines()
    names = [n.strip() for n in args.methods.split(",") if n.strip()]

    cstart, cend = find_class_block(lines, args.class_name)
    spans = find_method_spans(lines, cstart, cend, names)

    # 按出现顺序收集行块
    ordered = sorted((spans[n][0], spans[n][1], n) for n in names)
    blocks: list[str] = []
    for start, end, name in ordered:
        blocks.append("\n".join(lines[start:end]))

    # import 块 = 文件头到 class 前
    imp_end = cstart
    # 回退 class 前的空行
    while imp_end > 0 and lines[imp_end - 1].strip() == "":
        imp_end -= 1
    import_block = "\n".join(lines[:imp_end])

    doc = args.doc or f"拆分批 mixin（从 {src_path.name} 搬出，纯移动无行为变化）。"
    tc_extra = ""
    if args.typechecking_extra:
        tc_extra = "\n".join(f"    {x.strip()}" for x in args.typechecking_extra.split(",") if x.strip())

    header = f'"""{doc}"""\nfrom __future__ import annotations\n\n'
    # source 的 import 块里已含 from __future__ 与 docstring 相关——直接原样复制更安全
    body = (
        f"{import_block}\n\n\n"
        f"class {args.mixin_name}:\n"
        f'    """{doc}"""\n\n'
        + "\n\n".join(blocks)
        + "\n"
    )
    tgt_path.write_text(header + body.replace("from __future__ import annotations\n\n\n", "", 1) if False else body)

    # source 剪出
    cut = set()
    for start, end, name in ordered:
        cut.update(range(start, end))
    new_lines = [ln for i, ln in enumerate(lines) if i not in cut]
    src_path.write_text("\n".join(new_lines) + "\n")

    total = sum(e - s for s, e, _ in ordered)
    print(f"extracted {len(names)} methods, {total} lines")
    print(f"target: {tgt_path} ({len(tgt_path.read_text().splitlines())} lines)")
    print(f"source now: {len(new_lines)} lines")


if __name__ == "__main__":
    main()
