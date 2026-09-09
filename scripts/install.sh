#!/usr/bin/env bash
# memory-arbiter-mcp one-line installer. Run with bash (not sh):
#
#   curl -fsSL https://memarbiter.cn/install.sh | bash
#
# Does everything the multi-step manual flow used to require:
#   1. install the package with [vec,semantic-local] extras (uv > pipx > pip)
#   2. mema setup --install  (deps check + download both GGUF models + config)
#   3. mema doctor           (verify the install end-to-end)
#   4. print the MCP client config snippet to paste
set -euo pipefail

PKG='memory-arbiter-mcp[vec,semantic-local]'

log() { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m⚠\033[0m %s\n' "$*"; }

# ── 1. install the package ──────────────────────────────────────────────────
# llama-cpp-python ships prebuilt CPU wheels only for Python 3.10-3.12, so the
# uv path pins 3.12 (uv would otherwise pick the newest CPython it manages).
if command -v uv >/dev/null 2>&1; then
    log "uv tool install --python 3.12 $PKG"
    uv tool install --force --python 3.12 "$PKG" || {
        warn "uv --python 3.12 失败（本机无 3.12？uv 会自动下载）；回退默认解释器"
        uv tool install --force "$PKG"
    }
elif command -v pipx >/dev/null 2>&1; then
    log "pipx install $PKG"
    pipx install --force "$PKG"
elif command -v python3 >/dev/null 2>&1; then
    log "python3 -m pip install --user $PKG"
    python3 -m pip install --user --upgrade "$PKG" || {
        warn "pip --user 失败（PEP 668 externally-managed 系统的典型症状）。建议安装 uv 后重跑："
        warn "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        warn "  uv tool install --python 3.12 $PKG"
        exit 1
    }
    warn "若提示 'mema: command not found'，请把 python user bin 加入 PATH 后重开终端"
else
    echo "未找到 python3。请先安装 Python 3.10–3.12（推荐 uv 或 pyenv）。" >&2
    exit 1
fi

# uv/pipx install into ~/.local/bin, which is often absent from a curl|bash
# session's PATH; pip --user adds the interpreter's user base bin on top.
export PATH="$HOME/.local/bin:$PATH"
if command -v python3 >/dev/null 2>&1; then
    USER_BASE_BIN="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts", "posix_user"))' 2>/dev/null || true)"
    [ -n "${USER_BASE_BIN:-}" ] && export PATH="$USER_BASE_BIN:$PATH"
fi

# ── 2. full setup: deps + both models + config.json ─────────────────────────
# Exit 1 from setup means "missing items" (e.g. a 3.13 interpreter without
# llama.cpp wheels) — report it instead of dying before doctor runs.
log "mema setup --install（下载约 800MB 模型，支持断点续传/国内镜像）"
mema setup --install || warn "mema setup --install 未完全就绪，请查看上方输出（常见：Python 3.13+ 无 llama.cpp wheel，见 pyenv 提示）"

# ── 3. verify ────────────────────────────────────────────────────────────────
log "mema doctor"
mema doctor || warn "doctor 有警告项，请查看上方输出"

# ── 4. MCP client snippet ────────────────────────────────────────────────────
cat <<'EOF'

✅ 安装完成。把下面这段加到你的 MCP 客户端配置（client/agent_id 改成你的名字）：

{
  "mcpServers": {
    "memory-arbiter": {
      "command": "mema",
      "env": {
        "MEMORY_ARBITER_CLIENT": "your-client",
        "MEMORY_ARBITER_AGENT_ID": "your-agent"
      }
    }
  }
}

重启 MCP 客户端后，第一次工具调用会收到 onboarding 健康卡。
EOF
