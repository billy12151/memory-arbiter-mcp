"""CLI shell for ``memory-arbiter setup``.

Semi-automatic one-shot setup helper: generates ``~/.config/memory-arbiter/config.json``
from an inline template, then runs read-only environment checks and prints the
*exact* commands / download URLs the user still needs to run.

Design stance (deliberate): this command does **not** call ``pip``, does **not**
download the model, does **not** touch the network. Failing installs of
``llama-cpp-python`` or a blocked model download are environment problems the
user must handle; setup only tells them precisely what to do. Dispatch is
wired in ``server.main`` by intercepting ``argv[1]=="setup"``; no new console
script is added (pyproject unchanged).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Model + download sources — kept as module constants so they are easy to update.
DEFAULT_MODEL_FILENAME = "embeddinggemma-300m-qat-Q8_0.gguf"
EXPECTED_MODEL_BYTES = 329 * 1024 * 1024  # ~329 MB; we tolerate ±20% (see _size_ok)
MODEL_SIZE_TOLERANCE = 0.20

HF_DOWNLOAD_URL = (
    "https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"
    "/resolve/main/embeddinggemma-300m-qat-Q8_0.gguf"
)
MODELSCOPE_DOWNLOAD_URL = (
    "https://modelscope.cn/models/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"
    "/resolve/master/embeddinggemma-300m-qat-Q8_0.gguf"
)
LLAMA_CPP_CPU_EXTRA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

# llama-cpp-python prebuilt CPU wheels cover this Python range. Outside it pip
# falls back to source build (needs VS Build Tools on Windows).
LLAMA_CPP_SUPPORTED_PY = (3, 10), (3, 12)

# ANSI color codes (no external dependency; mirrors doctor_cli.py).
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"


def _color(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{_RESET}" if use_color else text


def _default_config_dict(model_path: Path, db_path: Path, backup_jsonl: Path) -> dict[str, Any]:
    """Inline config template.

    Mirrors ``examples/memory-arbiter.config.example.json`` field structure but
    drops the long ``_readme`` tutorials (runtime config does not need the
    lesson) and writes absolute paths. Kept here (not loaded from the wheel)
    because pyproject has no ``package-data`` declaration — examples/ is in
    sdist but not in wheel. Inlining makes the behaviour testable and
    install-path-independent.
    """
    return {
        "db_path": str(db_path),
        "backup_jsonl": str(backup_jsonl),
        "vec": {"enabled": True, "dim": 768},
        "embedding": {
            "provider": "gguf",
            "model_path": str(model_path),
            "auto_query": True,
            "auto_write": True,
        },
        "recall_pool_cap": 50,
        "content_like_cap": 30,
        "split": {
            "threshold": 4000,
            "section_vec_distance_threshold": 0.42,
            "section_fulltext_threshold": 0.8,
            "max_sections": 50,
            "max_section_chars": 3600,
        },
    }


def _default_paths() -> tuple[Path, Path, Path, Path]:
    """Compute platform-correct default paths via Path.home().

    Returns (config_dir, config_path, model_path, db_path, backup_jsonl) —
    actually (config_path, config_dir, model_path, db_path) tuple of four:
    config_path, model_path, db_path, backup_jsonl.
    """
    home = Path.home()
    config_dir = home / ".config" / "memory-arbiter"
    config_path = config_dir / "config.json"
    data_dir = home / ".local" / "share" / "memory-arbiter"
    model_path = data_dir / "models" / DEFAULT_MODEL_FILENAME
    db_path = data_dir / "memory.sqlite3"
    backup_jsonl = data_dir / "memory.backup.jsonl"
    return config_path, model_path, db_path, backup_jsonl


def _python_version_supported() -> tuple[bool, str]:
    """Check if current Python is in llama-cpp-python CPU wheel range."""
    cur = (sys.version_info.major, sys.version_info.minor)
    lo, hi = LLAMA_CPP_SUPPORTED_PY
    ok = lo <= cur <= hi
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return ok, version_str


def _check_sqlite_vec() -> bool:
    try:
        import sqlite_vec  # noqa: F401
        return True
    except Exception:
        return False


def _check_llama_cpp() -> bool:
    try:
        import llama_cpp  # noqa: F401
        return True
    except Exception:
        return False


def _model_size_ok(path: Path, *, expected: int = EXPECTED_MODEL_BYTES) -> tuple[bool, int]:
    """Return (size_looks_right, actual_bytes). Missing → (False, 0).

    When the user supplies their own model (different from the bundled
    embeddinggemma), pass its expected size via ``expected``; otherwise we
    only sanity-check against the embeddinggemma baseline.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return False, 0
    low = expected * (1 - MODEL_SIZE_TOLERANCE)
    high = expected * (1 + MODEL_SIZE_TOLERANCE)
    return low <= size <= high, size


def _detect_existing_model_path(config_path: Path) -> tuple[Path | None, str]:
    """If an existing config.json points at a real model file, honour it.

    Returns (resolved_model_path, note) where note is a short human-readable
    explanation for the setup log, or (None, "") when there is nothing to
    preserve. We only preserve when the file actually exists on disk — a
    stale path from a long-ago uninstall should not block the embeddinggemma
    default.
    """
    if not config_path.exists():
        return None, ""
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, ""
    if not isinstance(parsed, dict):
        return None, ""
    raw = parsed.get("embedding", {}).get("model_path") if isinstance(parsed.get("embedding"), dict) else None
    if not raw:
        return None, ""
    resolved = Path(str(raw)).expanduser()
    if not resolved.is_file():
        return None, ""
    # Don't treat the embeddinggemma default as "user-supplied" — that path
    # is what we'd write anyway, so there's nothing to preserve.
    if resolved.name == DEFAULT_MODEL_FILENAME:
        return None, ""
    return resolved, f"检测到你已配置的模型: {resolved.name}（沿用，未覆盖）"


# ── Rendering ──────────────────────────────────────────────────────────────

def _render_step_header(title: str, use_color: bool) -> str:
    return (
        "\n"
        + _color("=" * 60, _BOLD, use_color)
        + "\n"
        + _color(title, _BOLD, use_color)
        + "\n"
        + _color("=" * 60, _BOLD, use_color)
    )


def _render_config_step(
    config_path: Path,
    config_dict: dict[str, Any],
    *,
    written: bool,
    backup_path: Path | None,
    print_only: bool,
    use_color: bool,
) -> list[str]:
    lines: list[str] = []
    lines.append(_render_step_header("Step 1 — config.json", use_color))
    if print_only:
        lines.append(_color("--print-config: 仅打印，不写盘", _DIM, use_color))
        lines.append("")
        lines.append(f"目标路径: {config_path}")
        lines.append("")
        lines.append(_color("内容:", _CYAN, use_color))
        lines.append(json.dumps(config_dict, ensure_ascii=False, indent=2))
        return lines
    if not written:
        lines.append(_color("✗ config.json 未写入（--no-config 跳过，或写入失败见上方错误）", _RED, use_color))
        return lines
    lines.append(_color(f"✓ config.json 已写入: {config_path}", _GREEN, use_color))
    if backup_path is not None:
        lines.append(_color(f"  原文件已备份为: {backup_path}", _DIM, use_color))
    return lines


def _render_check_step(
    checks: dict[str, Any],
    model_path: Path,
    use_color: bool,
) -> tuple[list[str], bool]:
    """Render environment checks + remediation hints. Returns (lines, all_ok)."""
    lines: list[str] = []
    lines.append(_render_step_header("Step 2 — 环境自检", use_color))
    all_ok = True

    def mark(ok: bool) -> str:
        return _color("✓", _GREEN, use_color) if ok else _color("✗", _RED, use_color)

    # sqlite-vec
    sv = checks["sqlite_vec"]
    lines.append(f"{mark(sv)} sqlite-vec: {'已装' if sv else '未装'}")
    if not sv:
        all_ok = False
        lines.append(_color("  → pip install sqlite-vec", _DIM, use_color))
        lines.append(_color("  → 或: pip install memory-arbiter-mcp[vec]", _DIM, use_color))

    # llama-cpp-python
    lc = checks["llama_cpp"]
    py_ok, py_str = _python_version_supported()
    lines.append(f"{mark(lc)} llama-cpp-python: {'已装' if lc else '未装'}")
    if not lc:
        all_ok = False
        lines.append(
            _color(
                f"  → pip install llama-cpp-python --extra-index-url {LLAMA_CPP_CPU_EXTRA_INDEX}",
                _DIM,
                use_color,
            )
        )
        if not py_ok:
            lines.append(
                _color(
                    f"  ⚠ 你的 Python 是 {py_str}，但 llama-cpp-python CPU 预构建 wheel "
                    f"只支持 {LLAMA_CPP_SUPPORTED_PY[0][0]}.{LLAMA_CPP_SUPPORTED_PY[0][1]}–"
                    f"{LLAMA_CPP_SUPPORTED_PY[1][0]}.{LLAMA_CPP_SUPPORTED_PY[1][1]}。"
                    "装不上时可改用远程 API embedding（见 README “Optional: Semantic Recall”）。",
                    _YELLOW,
                    use_color,
                )
            )

    # GGUF model file
    exists = checks["model_exists"]
    size_ok, size_bytes = checks["model_size_ok"], checks["model_size_bytes"]
    if exists:
        size_mb = size_bytes / (1024 * 1024)
        size_label = f"存在 ({size_mb:.0f} MB)"
        # If present but wrong size, flag warning but keep all_ok true (user may have a different quant).
        if not size_ok:
            size_label += _color("  ⚠ 大小异常，可能是不同量化版本——若维度非 768 需改 vec.dim", _YELLOW, use_color)
            lines.append(f"{_color('✓', _GREEN, use_color)} GGUF 模型: {size_label}")
            lines.append(_color(f"     路径: {model_path}", _DIM, use_color))
        else:
            lines.append(f"{_color('✓', _GREEN, use_color)} GGUF 模型: {size_label}")
            lines.append(_color(f"     路径: {model_path}", _DIM, use_color))
    else:
        all_ok = False
        lines.append(f"{mark(False)} GGUF 模型: 未找到")
        lines.append(_color(f"     预期路径: {model_path}", _DIM, use_color))
        lines.append(_color("  → 下载（HuggingFace）:", _DIM, use_color))
        lines.append(_color(f"     {HF_DOWNLOAD_URL}", _CYAN, use_color))
        lines.append(_color("  → 国内镜像（ModelScope，访问 HF 不稳时用）:", _DIM, use_color))
        lines.append(_color(f"     {MODELSCOPE_DOWNLOAD_URL}", _CYAN, use_color))
        lines.append(_color("  → 下完放到上述路径，或改 config.json 的 embedding.model_path 指向实际位置", _DIM, use_color))

    # config load
    cl = checks["config_load_ok"]
    warnings = checks["config_warnings"]
    lines.append(f"{mark(cl)} config.json 加载: {'OK' if cl else '失败'}")
    if not cl:
        all_ok = False
        lines.append(_color(f"  错误: {checks['config_load_error']}", _YELLOW, use_color))
    elif warnings:
        # warnings are non-fatal but worth surfacing.
        for w in warnings[:3]:
            lines.append(_color(f"  ⚠ {w}", _YELLOW, use_color))
        if len(warnings) > 3:
            lines.append(_color(f"  …（还有 {len(warnings) - 3} 条警告）", _DIM, use_color))

    return lines, all_ok


def _render_summary(all_ok: bool, use_color: bool, config_written: bool, *, suppress_warning: bool = False) -> list[str]:
    lines: list[str] = []
    lines.append(_render_step_header("Step 3 — 汇总", use_color))
    if all_ok:
        lines.append(_color("✓ 环境就绪。重启 MCP 客户端即可生效（embedding 首次调用会惰性加载模型）。", _GREEN, use_color))
    elif suppress_warning:
        # Dry-run / check-only: just report readiness without the "do these steps" framing.
        lines.append(_color("（预览/检查模式：以上自检结果仅供参考，未执行写入或安装。）", _DIM, use_color))
    else:
        lines.append(_color("⚠ 有缺失项。完成上述步骤后重新运行 `memory-arbiter setup` 验证。", _YELLOW, use_color))
        if config_written:
            lines.append(_color("  config.json 已生成，但 embedding 还没就绪 —— 此时 memory-arbiter 退化为", _DIM, use_color))
            lines.append(_color("  纯关键词检索（FTS5），其余功能（写入/版本链/冲突/分段）不受影响。", _DIM, use_color))
    return lines


# ── Main entry ─────────────────────────────────────────────────────────────

def run_cli(argv: list[str]) -> int:
    """CLI entry: generate config, run checks, print remediation. Return exit code."""
    parser = argparse.ArgumentParser(
        prog="memory-arbiter setup",
        description="memory-arbiter 一键配置（生成 config.json + 环境自检 + 精确指引，不替你装包/下载）",
    )
    parser.add_argument("--force", action="store_true", help="config.json 已存在时直接覆盖（默认备份不覆盖）")
    parser.add_argument("--config-path", type=str, default=None, help="自定义 config.json 写入路径")
    parser.add_argument("--print-config", action="store_true", help="只打印将生成的 config 内容，不写盘")
    parser.add_argument("--no-config", action="store_true", help="跳过 config 生成，只跑环境自检")
    args = parser.parse_args(argv)

    use_color = sys.stdout.isatty()

    # Resolve paths (all platform-correct via Path.home()).
    default_config_path, default_model_path, default_db_path, default_backup_jsonl = _default_paths()
    config_path = Path(args.config_path).expanduser() if args.config_path else default_config_path

    # Honour a user-supplied model already present in an existing config:
    # if embedding.model_path points at a real file (and isn't our default
    # embeddinggemma), keep it instead of overwriting with the bundled path.
    # --force bypasses this: it means "reset to defaults, including model".
    if args.force:
        preserved_model, preserve_note = None, ""
    else:
        preserved_model, preserve_note = _detect_existing_model_path(config_path)
    model_path = preserved_model or default_model_path

    out_lines: list[str] = []
    out_lines.append(_color("memory-arbiter setup — 配置助手（半自动）", _BOLD, use_color))
    out_lines.append(_color("生成 config + 检测环境 + 给出可复制的命令。不调 pip、不下模型。", _DIM, use_color))
    if preserved_model is not None:
        out_lines.append(_color(f"  ℹ {preserve_note}", _CYAN, use_color))

    # ── Step 1: config.json ──
    config_dict = _default_config_dict(model_path, default_db_path, default_backup_jsonl)
    backup_path: Path | None = None
    written = False
    config_write_error: str | None = None

    if args.print_config:
        # print-only; nothing written.
        pass
    elif args.no_config:
        # skip entirely.
        pass
    else:
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file first, then atomically swap it into place — so a
            # write failure never leaves config_path missing or half-written (#6).
            tmp_path = config_path.with_name(f"{config_path.name}.tmp")
            tmp_path.write_text(
                json.dumps(config_dict, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if config_path.exists() and not args.force:
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                backup_path = config_path.with_name(f"{config_path.name}.bak.{ts}")
                # On case-insensitive filesystems (macOS default) rename is atomic and safe.
                config_path.rename(backup_path)
            tmp_path.replace(config_path)
            written = True
        except OSError as exc:
            config_write_error = f"{type(exc).__name__}: {exc}"

    out_lines.extend(
        _render_config_step(
            config_path,
            config_dict,
            written=written,
            backup_path=backup_path,
            print_only=args.print_config,
            use_color=use_color,
        )
    )
    if config_write_error is not None:
        out_lines.append(_color(f"✗ config.json 写入失败: {config_write_error}", _RED, use_color))

    # ── Step 2: environment checks ──
    # Config-load check: try to load via Settings.from_env() AFTER we may have
    # just written the file. If user passed --config-path to a non-default
    # location, Settings.from_env() won't find it unless MEMORY_ARBITER_CONFIG
    # is set — in that case we skip the load check rather than false-alarm.
    config_load_ok = True
    config_load_error_str = ""
    config_warnings: list[str] = []
    skip_load_check = bool(args.config_path) and not os.getenv("MEMORY_ARBITER_CONFIG")
    if skip_load_check:
        # Custom config path without env override — can't verify via from_env().
        config_load_ok = True  # neutral; we just wrote a valid JSON.
        config_warnings = ["使用了 --config-path 但未设 MEMORY_ARBITER_CONFIG，跳过加载验证"]
    else:
        try:
            # Force re-read by calling from_env fresh (it reads disk each call).
            from .config import Settings
            Settings.from_env()
            # from_env does not raise on missing fields (uses defaults), so we
            # additionally confirm the file parses as JSON we recognise.
            if config_path.exists():
                parsed = json.loads(config_path.read_text(encoding="utf-8"))
                if not isinstance(parsed, dict) or "vec" not in parsed:
                    config_load_ok = False
                    config_load_error_str = "config.json 解析成功但缺少必要字段（vec 段）"
        except Exception as exc:
            config_load_ok = False
            config_load_error_str = f"{type(exc).__name__}: {exc}"

    model_exists = model_path.exists()
    # For a user-supplied model we can't know the right size, so only run the
    # baseline comparison against embeddinggemma; otherwise just report size.
    if preserved_model is not None:
        # User's own model: exists check is enough; size is informational only.
        size_ok = True
        try:
            size_bytes = model_path.stat().st_size
        except OSError:
            size_bytes = 0
    else:
        size_ok, size_bytes = _model_size_ok(model_path)

    checks = {
        "sqlite_vec": _check_sqlite_vec(),
        "llama_cpp": _check_llama_cpp(),
        "model_exists": model_exists,
        "model_size_ok": size_ok,
        "model_size_bytes": size_bytes,
        "config_load_ok": config_load_ok,
        "config_load_error": config_load_error_str,
        "config_warnings": config_warnings,
    }
    check_lines, all_ok = _render_check_step(checks, model_path, use_color)
    out_lines.extend(check_lines)

    # ── Step 3: summary ──
    # In --print-config (dry-run) or --no-config modes we don't make strong
    # "ready / not ready" claims: the user only asked to preview or to check
    # an existing environment. Suppress the missing-items warning there.
    suppress_warning = args.print_config or args.no_config
    out_lines.extend(_render_summary(all_ok, use_color, written, suppress_warning=suppress_warning))

    print("\n".join(out_lines))

    # Exit codes: 0 all ok (or dry-run), 1 missing items, 2 config write failed.
    if config_write_error is not None:
        return 2
    if suppress_warning:
        return 0
    return 0 if all_ok else 1
