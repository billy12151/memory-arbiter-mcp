"""CLI shell for ``memory-arbiter doctor`` (design doc §10.2).

The ambulance entry: works even when the MCP process is down, because it
opens its own read-only connection and never constructs ``MemoryDB``.
Dispatch is wired in ``server.main`` by intercepting ``argv[1]=="doctor"``;
no new console script is added (pyproject unchanged).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import Settings
from .doctor import OverviewReport, Severity, doctor_overview_cli, report_to_dict


# ANSI color codes (no external dependency; design doc §10.2).
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"

_SEV_COLOR = {
    Severity.CRITICAL: _RED,
    Severity.WARNING: _YELLOW,
    Severity.INFO: _GREEN,
}
_SEV_LABEL = {
    Severity.CRITICAL: "CRITICAL",
    Severity.WARNING: "WARNING",
    Severity.INFO: "INFO",
}


def _color(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{_RESET}" if use_color else text


def _render_text(report: OverviewReport, use_color: bool) -> str:
    """Plain-text rendering with optional ANSI color (§10.2)."""
    lines: list[str] = []
    sev = report.overall
    lines.append(_color("=" * 60, _BOLD, use_color))
    lines.append(_color("memory-arbiter doctor — 健康体检报告", _BOLD, use_color))
    lines.append(_color("=" * 60, _BOLD, use_color))
    lines.append(f"快照时间: {report.snapshot_ts}  (CLI 快照可能略旧于 MCP 实时)")
    lines.append(f"总体评级: {_color(_SEV_LABEL[sev], _SEV_COLOR[sev], use_color)}")
    s = report.summary
    lines.append(
        f"模式: {s.get('mode')}  |  记忆数: {s.get('total_memories')}  |  "
        f"向量生效: {s.get('vec_effective')}  |  分段: {s.get('split_enabled')}"
    )
    lines.append("")

    # Group findings by dimension, preserving order.
    dims: dict[str, list] = {}
    for f in report.findings:
        dims.setdefault(f.dimension, []).append(f)

    dim_order = ["config", "vector", "split", "consistency", "capacity"]
    for dim in dim_order:
        items = dims.get(dim)
        if not items:
            continue
        lines.append(_color(f"[{dim}]", _CYAN, use_color))
        for f in items:
            mark = {"pass": "✓", "fail": "✗", "warn": "!", "n/a": "·", "error": "?"}.get(f.status, " ")
            sev_color = _SEV_COLOR.get(f.severity, "")
            prefix = _color(f"{mark} [{_SEV_LABEL[f.severity]}] {f.title}", sev_color, use_color)
            lines.append(f"  {prefix}")
            if f.detail:
                lines.append(_color(f"      {f.detail}", _DIM, use_color))
            if f.fix_hint:
                lines.append(f"      修复: {f.fix_hint}")
        lines.append("")

    if report.overall == Severity.INFO:
        lines.append(_color("所有检查通过，未发现问题。", _GREEN, use_color))
    return "\n".join(lines)


def run_cli(argv: list[str]) -> None:
    """CLI entry: parse args, run doctor, render."""
    parser = argparse.ArgumentParser(
        prog="memory-arbiter doctor",
        description="memory-arbiter 健康体检（只读，救护车入口）",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON（同 MCP data 结构）")
    parser.add_argument("--deep", action="store_true", help="实际加载 GGUF 模型做维度探针（秒级）")
    parser.add_argument("--db", type=str, default=None, help="覆盖 DB 路径（救护车场景）")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    db_path = Path(args.db).expanduser() if args.db else settings.db_path
    # If --db overrides, build a Settings copy with the new path so all
    # downstream code (incl. build_unopenable_report) uses it.
    if args.db:
        settings = replace(settings, db_path=db_path)

    report = doctor_overview_cli(db_path, settings, deep=args.deep)

    if args.json:
        import json
        print(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2))
    else:
        use_color = sys.stdout.isatty()
        print(_render_text(report, use_color=use_color))

    # Exit code: non-zero if any critical/warning, for scripting/CI use.
    if report.overall == Severity.CRITICAL:
        sys.exit(2)
    if report.overall == Severity.WARNING:
        sys.exit(1)
