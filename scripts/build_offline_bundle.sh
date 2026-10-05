#!/usr/bin/env bash
# build_offline_bundle.sh — 组装 mema 自包含离线安装包（发版时跑；方案见
# BillyProject/docs/mema-打包分发方案-2026-10-05.md）。
#
# 产物:
#   dist/mema-offline-<version>/            bundle 展开形态
#   dist/mema-offline-<version>.tar.gz     单 asset 上传 GitHub Release（~860MB）
#   dist/mema-offline-<version>.sha256     校验和
#
# 内容: wheel + embeddinggemma GGUF + mdeberta-base(config/tokenizer,不含
# safetensors——加载代码只读 config+tokenizer) + mDeBERTa fp16 ckpt + install.sh。
#
# 用法:
#   scripts/build_offline_bundle.sh               # 全量组装
#   scripts/build_offline_bundle.sh --dry-run     # 只列清单+算体积,不拷贝不打包
#   MDEBERTA_CKPT=/path/to/ckpt.pt scripts/build_offline_bundle.sh   # 覆盖默认 ckpt
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# 模型件路径（默认本机现状;发版时可环境变量覆盖）
EMBED_GGUF="${EMBED_GGUF:-$HOME/.node-llama-cpp/models/hf_ggml-org_embeddinggemma-300m-qat-Q8_0.gguf}"
MDEBERTA_BASE="${MDEBERTA_BASE:-/Users/zhangzhiwei17/BillyProject/mini-clash/models/mdeberta-base}"
MDEBERTA_CKPT="${MDEBERTA_CKPT:-/Users/zhangzhiwei17/BillyProject/mini-clash/models/mdeberta-v52_ep3_fp16.pt}"

VERSION="$("$REPO/.venv/bin/python" -c 'import memory_arbiter; print(memory_arbiter.__version__)')"
OUT_DIR="$REPO/dist/mema-offline-$VERSION"
ARCHIVE="$REPO/dist/mema-offline-$VERSION.tar.gz"

# ---- 前置检查 -----------------------------------------------------------
for f in "$EMBED_GGUF" "$MDEBERTA_CKPT" \
         "$MDEBERTA_BASE/config.json" "$MDEBERTA_BASE/tokenizer.json" \
         "$MDEBERTA_BASE/tokenizer_config.json"; do
  [[ -f "$f" ]] || { echo "缺少模型件: $f" >&2; exit 1; }
done
# 明确拒收 safetensors:558MB 死重,加载代码不读（semantic_judge.py:205）。
if [[ -f "$MDEBERTA_BASE/model.safetensors" ]]; then
  SAFETENSORS_NOTE="（已排除 model.safetensors,加载不读）"
fi

size_of() { du -h "$1" | cut -f1; }

echo "bundle: mema-offline-$VERSION  $SAFETENSORS_NOTE"
echo "  [1/4] wheel               (构建时产物)"
echo "  [2/4] embedder GGUF        $(size_of "$EMBED_GGUF")  $EMBED_GGUF"
echo "  [3/4] mdeberta-base-min    ~16MB  config.json+tokenizer.json+tokenizer_config.json"
echo "  [4/4] mdeberta ckpt fp16   $(size_of "$MDEBERTA_CKPT")  $MDEBERTA_CKPT"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "dry-run: 不拷贝不打包。"
  exit 0
fi

# ---- wheel -------------------------------------------------------------
"$REPO/.venv/bin/python" -m build --wheel --outdir "$REPO/dist" "$REPO" \
  || { echo "wheel 构建失败（.venv 缺 build? pip install build）" >&2; exit 1; }
WHEEL="$(ls -t "$REPO"/dist/memory_arbiter_mcp-*.whl | head -1)"

# ---- 组装 --------------------------------------------------------------
rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR/models/mdeberta-base"
cp "$WHEEL" "$OUT_DIR/"
cp "$EMBED_GGUF" "$OUT_DIR/models/"
cp "$MDEBERTA_BASE/config.json" \
   "$MDEBERTA_BASE/tokenizer.json" \
   "$MDEBERTA_BASE/tokenizer_config.json" \
   "$OUT_DIR/models/mdeberta-base/"
cp "$MDEBERTA_CKPT" "$OUT_DIR/models/"

cat > "$OUT_DIR/install.sh" <<'INSTALL'
#!/usr/bin/env bash
# mema 离线包安装（root 权限不需要;模型放用户目录,config 不覆盖已存在的）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${MEMA_MODELS_DIR:-$HOME/.local/share/memory-arbiter/models}"
CFG="${MEMA_CONFIG:-$HOME/.config/memory-arbiter/config.json}"

echo "[1/4] 安装 wheel（含 torch/transformers 依赖约 200MB,几分钟）"
pip install "${MEMA_EXTRA_ARGS:-}" "$HERE"/memory_arbiter_mcp-*.whl

echo "[2/4] 安装模型到 $MODELS_DIR"
mkdir -p "$MODELS_DIR" "$HOME/.config/memory-arbiter"
cp -n "$HERE/models/"*.gguf "$MODELS_DIR/" 2>/dev/null || true
cp -Rn "$HERE/models/mdeberta-base" "$MODELS_DIR/" 2>/dev/null || true
CKPT="$(ls "$HERE/models/"mdeberta-*.pt | head -1)"
CKPT_NAME="$(basename "$CKPT")"
cp -n "$CKPT" "$MODELS_DIR/"

echo "[3/4] 配置 $CFG"
if [[ -f "$CFG" ]]; then
  echo "  已存在,不覆盖。请自行确认 semantic_conflict.mdeberta_ckpt = $MODELS_DIR/$CKPT_NAME"
else
  cat > "$CFG" <<CFGJSON
{
  "db_path": "$HOME/.local/share/memory-arbiter/memory.sqlite3",
  "backup_jsonl": "$HOME/.local/share/memory-arbiter/memory.backup.jsonl",
  "client": "CHANGE_ME", "agent_id": "CHANGE_ME",
  "embedding": {"model_path": "$MODELS_DIR/$(ls "$HERE/models/"*.gguf | xargs basename)"},
  "semantic_conflict": {
    "mdeberta_ckpt": "$MODELS_DIR/$CKPT_NAME",
    "mdeberta_notice_min_prob": 0.5
  }
}
CFGJSON
  echo "  已生成,client/agent_id 需改填。"
fi

echo "[4/4] 验证"
mema doctor || true
echo "完成。MCP 客户端配置命令: mema（= memory-arbiter-mcp）"
INSTALL
chmod +x "$OUT_DIR/install.sh"

# ---- 打包 + 校验和 -----------------------------------------------------
( cd "$REPO/dist" && tar -czf "mema-offline-$VERSION.tar.gz" "mema-offline-$VERSION" )
( cd "$REPO/dist" && shasum -a 256 "mema-offline-$VERSION.tar.gz" > "mema-offline-$VERSION.sha256" )
echo "产物:"
ls -lh "$ARCHIVE" "$REPO/dist/mema-offline-$VERSION.sha256"
