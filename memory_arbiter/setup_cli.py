"""CLI shell for ``memory-arbiter setup``.

One-shot setup helper: generates a current starter
``~/.config/memory-arbiter/config.json``, then runs read-only environment checks
and prints concrete commands and download URLs for remaining prerequisites.

Network stance: the bare command does **not** call ``pip``, does **not**
download models, does **not** touch the network — failing installs of
``llama-cpp-python`` or a blocked model download are environment problems the
user must handle, and setup only tells them precisely what to do. Passing
``--install`` flips the command into execution mode: it pip-installs the
optional dependencies, downloads both GGUF models (embedding + semantic,
HuggingFace with ModelScope fallback, resumable), and writes the finished
config itself — one command an agent can run end-to-end. Dispatch is
wired in ``server.main`` by intercepting ``argv[1]=="setup"``; no new console
script is added (pyproject unchanged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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

# Semantic-conflict (Qwen) model — the second half of a "full" install. URLs
# verified 2026-09-09: HF honours Range (206), ModelScope answers 200 to a
# ranged probe (the downloader treats an ignored Range as restart-from-zero).
QWEN_MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
QWEN_MODEL_DIRNAME = "Qwen2.5-0.5B-Instruct"
EXPECTED_QWEN_BYTES = 491_400_032  # q4_k_m; same ±20% tolerance as embedding
QWEN_HF_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    "/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
QWEN_MODELSCOPE_URL = (
    "https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    "/resolve/master/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB
_DOWNLOAD_TIMEOUT_S = 60
_DOWNLOAD_USER_AGENT = "memory-arbiter-setup"

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


def _default_config_dict(
    model_path: Path,
    db_path: Path,
    backup_jsonl: Path,
    *,
    qwen_model_path: Path | None = None,
) -> dict[str, Any]:
    """Return the 0.15.0 slim starter config (18 user keys, file-only).

    Everything else the 0.14.x config carried is a frozen constant now.
    Identity (client/agent_id) is intentionally left empty: the MCP server
    refuses to start until it is filled in. ``qwen_model_path`` is only
    supplied by ``--install`` (execution mode points semantic_conflict at the
    downloaded model); guidance mode leaves it None.
    """
    return {
        "db_path": str(db_path),
        "backup_jsonl": str(backup_jsonl),
        "client": "",
        "agent_id": "",
        "workspace": "default",
        "isolation": "none",
        "policy_path": None,
        "update_check": {"enabled": True},
        "embedding": {
            "model_path": str(model_path),
            "auto_query": True,
            "auto_write": True,
        },
        "semantic_conflict": {
            "model_path": str(qwen_model_path) if qwen_model_path is not None else None,
            "on_write": "async",
            "max_notice_pairs": 2,
        },
        "mcp": {
            "transport": "stdio",
            "http": {"host": "127.0.0.1", "port": 8000},
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


def _default_qwen_path() -> Path:
    """The runtime's default semantic-conflict model location."""
    return (
        Path.home() / ".local" / "share" / "memory-arbiter" / "models"
        / "semantic-conflict" / QWEN_MODEL_DIRNAME / QWEN_MODEL_FILENAME
    )


# ── Execution mode (--install) ─────────────────────────────────────────────

def _download_with_resume(
    url: str,
    part_path: Path,
    *,
    log: Callable[[str], None] = print,
) -> bool:
    """Download *url* to *part_path*, resuming a partial file when possible.

    Returns True once the file is fully downloaded (caller validates the size
    and atomically renames). Servers that ignore the Range header (ModelScope
    answers 200 to a ranged probe) make us restart from zero — the .part file
    is truncated and refilled, never served as-is.
    """
    resume_from = 0
    try:
        if part_path.exists():
            resume_from = part_path.stat().st_size
    except OSError:
        resume_from = 0
    headers = {"User-Agent": _DOWNLOAD_USER_AGENT}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        log(f"  断点续传: 已有 {resume_from / (1024 * 1024):.0f} MB，继续下载")
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            status = int(getattr(response, "status", 200) or 200)
            if resume_from > 0 and status != 206:
                # Range ignored → full body coming; restart from scratch.
                resume_from = 0
            part_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "ab" if resume_from > 0 else "wb"
            with part_path.open(mode) as handle:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
    except (OSError, ValueError) as exc:
        log(f"  ✗ 下载失败: {type(exc).__name__}: {exc}")
        return False
    return True


def _part_path_for(dest: Path, url: str) -> Path:
    """Per-URL partial file: resume only ever continues bytes from the SAME
    mirror. Two mirrors serving different content would otherwise produce a
    hybrid file that passes the size gate but is corrupt (observed live:
    an HF attempt truncated at 47MB, then ModelScope resumed over it).
    """
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return dest.with_name(f"{dest.name}.{digest}.part")


def _install_model(
    label: str,
    dest: Path,
    urls: list[str],
    *,
    expected_bytes: int,
    log: Callable[[str], None] = print,
) -> bool:
    """Ensure the GGUF at *dest* exists and passes the size check.

    Already-good files are kept (idempotent re-runs). Otherwise each mirror is
    tried in order with same-URL resume; the first download passing the size
    check is atomically renamed into place and stale partials from other
    mirrors are cleaned up.
    """
    size_ok, size_bytes = _model_size_ok(dest, expected=expected_bytes)
    if size_ok:
        log(f"✓ {label}: 已存在 ({size_bytes / (1024 * 1024):.0f} MB)，跳过下载")
        return True
    for url in urls:
        part_path = _part_path_for(dest, url)
        log(f"→ 下载 {label}: {url}")
        if not _download_with_resume(url, part_path, log=log):
            continue
        ok, actual = _model_size_ok(part_path, expected=expected_bytes)
        if not ok:
            log(f"  ✗ 大小校验失败（{actual} bytes，预期约 {expected_bytes}），尝试下一个镜像")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part_path.replace(dest)
        for stale in dest.parent.glob(f"{dest.name}.*.part"):
            try:
                stale.unlink()
            except OSError:
                pass
        log(f"✓ {label}: 下载完成 → {dest}")
        return True
    log(f"✗ {label}: 所有镜像均失败，可手动下载后放入 {dest}")
    return False


def _pip_install(packages: list[str], *, log: Callable[[str], None] = print) -> bool:
    """pip-install *packages* into the current interpreter. Never raises."""
    cmd = [
        sys.executable, "-m", "pip", "install",
        *packages,
        "--extra-index-url", LLAMA_CPP_CPU_EXTRA_INDEX,
    ]
    log(f"→ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"  ✗ pip 执行失败: {type(exc).__name__}: {exc}")
        return False
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        for line in tail:
            log(f"  {line}")
        log("  ✗ pip 安装失败，可手动重试上述命令")
        return False
    return True


def _pyenv_guidance_lines(py_str: str) -> list[str]:
    """Executable remediation for Python versions without llama.cpp wheels."""
    return [
        f"  ⚠ 你的 Python 是 {py_str}，llama-cpp-python CPU 预构建 wheel 只支持 "
        f"{LLAMA_CPP_SUPPORTED_PY[0][0]}.{LLAMA_CPP_SUPPORTED_PY[0][1]}–"
        f"{LLAMA_CPP_SUPPORTED_PY[1][0]}.{LLAMA_CPP_SUPPORTED_PY[1][1]}。可执行修复：",
        "     pyenv install 3.12",
        "     pyenv local 3.12   # 或: pyenv shell 3.12",
        "     python3.12 -m pip install memory-arbiter-mcp[vec,semantic-local]",
        "   （无 pyenv 时：brew install pyenv，或改用远程 API embedding，见 README）",
    ]


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
    qwen_path: Path,
    use_color: bool,
    *,
    require_qwen: bool = False,
) -> tuple[list[str], bool]:
    """Render environment checks + remediation hints. Returns (lines, all_ok).

    The semantic (qwen) model is informational in guidance mode — a minimal
    install is legitimate — but counts toward readiness under ``--install``,
    which promises a full install.
    """
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
            lines.extend(_color(line, _YELLOW, use_color) for line in _pyenv_guidance_lines(py_str))

    # GGUF model file
    exists = checks["model_exists"]
    size_ok, size_bytes = checks["model_size_ok"], checks["model_size_bytes"]
    if exists:
        size_mb = size_bytes / (1024 * 1024)
        size_label = f"存在 ({size_mb:.0f} MB)"
        # If present but wrong size, flag warning but keep all_ok true (user may have a different quant).
        if not size_ok:
            size_label += _color("  ⚠ 大小异常，可能是不同量化版本（维度会自动从模型探测，无需配置）", _YELLOW, use_color)
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
        lines.append(_color("  → 或运行 `mema setup --install` 自动下载（含断点续传/镜像切换）", _DIM, use_color))

    # semantic-conflict (qwen) model — informational unless --install asked for
    # a full install.
    qwen_exists = checks["qwen_exists"]
    if qwen_exists:
        qwen_mb = checks["qwen_size_bytes"] / (1024 * 1024)
        lines.append(f"{_color('✓', _GREEN, use_color)} Qwen 语义模型: 存在 ({qwen_mb:.0f} MB)")
        lines.append(_color(f"     路径: {qwen_path}", _DIM, use_color))
    else:
        if require_qwen:
            all_ok = False
            lines.append(f"{mark(False)} Qwen 语义模型: 未找到（冲突检测不可用）")
        else:
            lines.append(f"{_color('⚠', _YELLOW, use_color)} Qwen 语义模型: 未找到（冲突检测不可用，属可选能力）")
        lines.append(_color(f"     预期路径: {qwen_path}", _DIM, use_color))
        lines.append(_color("  → 运行 `mema setup --install` 自动下载并写 config（推荐）", _DIM, use_color))
        lines.append(_color(f"  → 手动（HuggingFace）: {QWEN_HF_URL}", _CYAN, use_color))
        lines.append(_color(f"  → 国内镜像（ModelScope）: {QWEN_MODELSCOPE_URL}", _CYAN, use_color))

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
        # Preview/check-only modes report observations without claiming readiness.
        lines.append(_color("（预览/检查模式：以上仅为当前环境检查；未写入配置、安装依赖或下载模型。）", _DIM, use_color))
    else:
        lines.append(_color("⚠ 有缺失项。完成上述步骤后重新运行 `memory-arbiter setup` 验证。", _YELLOW, use_color))
        if config_written:
            lines.append(_color("  config.json 已生成，但 embedding 还没就绪 —— 此时 memory-arbiter 退化为", _DIM, use_color))
            lines.append(_color("  纯关键词检索（FTS5），其余功能（写入/版本链/冲突治理）不受影响。", _DIM, use_color))
    return lines


# ── Main entry ─────────────────────────────────────────────────────────────

def run_cli(argv: list[str]) -> int:
    """CLI entry: generate config, run checks, print remediation. Return exit code.

    ``--install`` switches from guidance to execution: pip-install the optional
    dependencies, download both GGUF models (resumable, mirror fallback), and
    write the finished config (including semantic_conflict.model_path) itself.
    """
    parser = argparse.ArgumentParser(
        prog="memory-arbiter setup",
        description="memory-arbiter 一键配置（生成 config.json + 环境自检 + 精确指引；--install 直接执行安装）",
    )
    parser.add_argument("--force", action="store_true", help="config.json 已存在时直接覆盖（默认备份不覆盖）")
    parser.add_argument("--config-path", type=str, default=None, help="自定义 config.json 写入路径")
    parser.add_argument("--print-config", action="store_true", help="只打印将生成的 config 内容，不写盘")
    parser.add_argument("--no-config", action="store_true", help="跳过 config 生成，只跑环境自检")
    parser.add_argument("--install", action="store_true", help="执行模式：装依赖 + 下载两个模型 + 回写 config（默认只指导不执行）")
    args = parser.parse_args(argv)

    use_color = sys.stdout.isatty()

    # Resolve paths (all platform-correct via Path.home()).
    default_config_path, default_model_path, default_db_path, default_backup_jsonl = _default_paths()
    config_path = Path(args.config_path).expanduser() if args.config_path else default_config_path
    qwen_path = _default_qwen_path()

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
    if args.install:
        out_lines.append(_color("--install 执行模式：装依赖 + 下载 embedding/qwen 模型 + 回写 config。", _DIM, use_color))
    else:
        out_lines.append(_color("生成 config + 检测环境 + 给出可复制的命令。--install 可直接执行全部安装。", _DIM, use_color))
    if preserved_model is not None:
        out_lines.append(_color(f"  ℹ {preserve_note}", _CYAN, use_color))

    # ── Step 1: config.json ──
    config_dict = _default_config_dict(
        model_path, default_db_path, default_backup_jsonl,
        qwen_model_path=qwen_path if args.install else None,
    )
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

    # ── Step 1.5 (--install only): execute pip + model downloads ──
    if args.install:
        out_lines.append(_render_step_header("Step 1.5 — 执行安装（--install）", use_color))
        py_ok, py_str = _python_version_supported()
        if not _check_sqlite_vec():
            if _pip_install(["sqlite-vec"]):
                out_lines.append(_color("✓ sqlite-vec 安装完成", _GREEN, use_color))
            else:
                out_lines.append(_color("✗ sqlite-vec 安装失败（见上方 pip 输出）", _RED, use_color))
        else:
            out_lines.append(_color("✓ sqlite-vec: 已装，跳过", _GREEN, use_color))
        if not _check_llama_cpp():
            if py_ok:
                if _pip_install(["llama-cpp-python"]):
                    out_lines.append(_color("✓ llama-cpp-python 安装完成", _GREEN, use_color))
                else:
                    out_lines.append(_color("✗ llama-cpp-python 安装失败（见上方 pip 输出）", _RED, use_color))
            else:
                out_lines.extend(_color(line, _YELLOW, use_color) for line in _pyenv_guidance_lines(py_str))
        else:
            out_lines.append(_color("✓ llama-cpp-python: 已装，跳过", _GREEN, use_color))
        _install_model(
            "embedding 模型", model_path,
            [HF_DOWNLOAD_URL, MODELSCOPE_DOWNLOAD_URL],
            expected_bytes=EXPECTED_MODEL_BYTES,
        )
        _install_model(
            "Qwen 语义模型", qwen_path,
            [QWEN_HF_URL, QWEN_MODELSCOPE_URL],
            expected_bytes=EXPECTED_QWEN_BYTES,
        )

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
                if not isinstance(parsed, dict) or "embedding" not in parsed:
                    config_load_ok = False
                    config_load_error_str = "config.json 解析成功但缺少必要字段（embedding 段）"
        except Exception as exc:
            config_load_ok = False
            config_load_error_str = f"{type(exc).__name__}: {exc}"

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
        "model_exists": model_path.exists(),
        "model_size_ok": size_ok,
        "model_size_bytes": size_bytes,
        "qwen_exists": qwen_path.exists(),
        "qwen_size_bytes": qwen_path.stat().st_size if qwen_path.exists() else 0,
        "config_load_ok": config_load_ok,
        "config_load_error": config_load_error_str,
        "config_warnings": config_warnings,
    }
    check_lines, all_ok = _render_check_step(
        checks, model_path, qwen_path, use_color, require_qwen=args.install,
    )
    out_lines.extend(check_lines)

    # ── Step 3: summary ──
    # Preview/check-only modes report observed state without claiming readiness
    # or implying that this command changed the environment.
    suppress_warning = args.print_config or args.no_config
    out_lines.extend(_render_summary(all_ok, use_color, written, suppress_warning=suppress_warning))

    # ── Step 4: scheduled tasks guidance ──
    from .scan_tasks import SCHEDULED_TASKS_SPEC

    out_lines.append("")
    out_lines.append("Recommended scheduled tasks (see the `scheduled_tasks` help topic for the full spec):")
    for task in SCHEDULED_TASKS_SPEC["tasks"]:
        out_lines.append(f"  - {task['name']} ({task['cadence']}): {task['purpose']}")

    print("\n".join(out_lines))

    # Exit codes: 0 all ok (or dry-run), 1 missing items, 2 config write failed.
    if config_write_error is not None:
        return 2
    if suppress_warning:
        return 0
    return 0 if all_ok else 1
