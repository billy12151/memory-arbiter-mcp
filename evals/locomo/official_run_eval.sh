#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

EVAL_ARGS=()
USER_SCORED_PATH=""
USER_SPECIFIED_METRICS=0
USER_EVALUATOR_MODEL=""
SHOW_HELP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      SHOW_HELP=1
      shift
      ;;
    --output-path)
      if [[ $# -lt 2 ]]; then
        echo "error: --output-path requires a value" >&2
        exit 2
      fi
      USER_SCORED_PATH="$2"
      shift 2
      ;;
    --output-path=*)
      USER_SCORED_PATH="${1#*=}"
      shift
      ;;
    --metrics|--metrics=*)
      USER_SPECIFIED_METRICS=1
      EVAL_ARGS+=("$1")
      shift
      ;;
    --evaluator-model)
      if [[ $# -lt 2 ]]; then
        echo "error: --evaluator-model requires a value" >&2
        exit 2
      fi
      USER_EVALUATOR_MODEL="$2"
      EVAL_ARGS+=("$1" "$2")
      shift 2
      ;;
    --evaluator-model=*)
      USER_EVALUATOR_MODEL="${1#*=}"
      EVAL_ARGS+=("$1")
      shift
      ;;
    *)
      EVAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${SHOW_HELP}" -eq 1 ]]; then
  cat <<'EOF'
Usage: ./scripts/run_eval.sh [wrapper-options] [evaluate-options]

Wrapper options:
  --output-path PATH   Override the scored predictions output path.
  -h, --help           Show this help and exit.

This wrapper runs the LoCoMo scorer first and then writes JSON + Markdown summaries.
All other arguments are forwarded to the underlying evaluate CLI shown below.

EOF
  "${LOCOMO_PYTHON_BIN}" -c 'from evaluate import build_parser; build_parser().print_help()'
  exit 0
fi

DEFAULT_SCORED_PATH="$("${LOCOMO_PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

path = Path(os.environ["LOCOMO_PREDICTIONS_PATH"])
if path.suffix == ".jsonl":
    print(path.with_name(f"{path.stem}_scored.jsonl"))
else:
    suffix = path.suffix or ".json"
    print(path.with_name(f"{path.stem}_scored{suffix}"))
PY
)"

export LOCOMO_SCORED_PATH="${USER_SCORED_PATH:-${LOCOMO_SCORED_PATH:-${DEFAULT_SCORED_PATH}}}"

DEFAULT_SUMMARY_PATH="$("${LOCOMO_PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

path = Path(os.environ["LOCOMO_SCORED_PATH"])
print(path.with_name(f"{path.stem}_summary.json"))
PY
)"

export LOCOMO_SUMMARY_PATH="${LOCOMO_SUMMARY_PATH:-${DEFAULT_SUMMARY_PATH}}"

DEFAULT_MARKDOWN_SUMMARY_PATH="$("${LOCOMO_PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

path = Path(os.environ["LOCOMO_SUMMARY_PATH"])
print(path.with_suffix(".md"))
PY
)"

export LOCOMO_MARKDOWN_SUMMARY_PATH="${LOCOMO_MARKDOWN_SUMMARY_PATH:-${DEFAULT_MARKDOWN_SUMMARY_PATH}}"

DEFAULT_METRICS=(--metrics llm f1 bleu)

SHOULD_RUN_LLM=0
if [[ "${USER_SPECIFIED_METRICS}" -eq 0 ]]; then
  for token in "${DEFAULT_METRICS[@]}"; do
    lowered="$(printf '%s' "${token}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${lowered}" == "llm" || "${lowered}" == "all" ]]; then
      SHOULD_RUN_LLM=1
      break
    fi
  done
else
  for token in "${EVAL_ARGS[@]}"; do
    lowered="$(printf '%s' "${token}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${lowered}" == "llm" || "${lowered}" == "all" ]]; then
      SHOULD_RUN_LLM=1
      break
    fi
    if [[ "${lowered}" == --metrics=* ]]; then
      metrics_inline="${lowered#--metrics=}"
      metrics_inline="${metrics_inline//,/ }"
      for metric in ${metrics_inline}; do
        if [[ "${metric}" == "llm" || "${metric}" == "all" ]]; then
          SHOULD_RUN_LLM=1
          break 2
        fi
      done
    fi
  done
fi

if [[ "${SHOULD_RUN_LLM}" -eq 1 ]]; then
  # LoCoMo-Refined's official judge baseline uses Qwen3-14B.
  RESOLVED_EVALUATOR_MODEL="${USER_EVALUATOR_MODEL:-${EVALUATOR_MODEL:-qwen3-14b}}"
  NORMALIZED_EVALUATOR_MODEL="$(printf '%s' "${RESOLVED_EVALUATOR_MODEL}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')"
  if [[ ! "${NORMALIZED_EVALUATOR_MODEL}" =~ (qwen314b|qwenqwen314b)$ ]]; then
    echo "[LoCoMo-Refined] WARNING: evaluator model is '${RESOLVED_EVALUATOR_MODEL}'." >&2
    echo "[LoCoMo-Refined] Official judge model is Qwen3-14B; non-Qwen results may be inconsistent." >&2
    if [[ -t 0 ]]; then
      read -r -p "Type yes to continue anyway: " CONFIRM_NON_QWEN
      if [[ "${CONFIRM_NON_QWEN}" != "yes" ]]; then
        echo "Aborted: non-Qwen evaluator model not confirmed." >&2
        exit 1
      fi
    else
      echo "[LoCoMo-Refined] Continuing with non-Qwen model in non-interactive session." >&2
    fi
  fi
fi

echo "[1/2] Scoring predictions..." >&2
"${LOCOMO_PYTHON_BIN}" -c 'from evaluate import main; main()' \
  --questions-path "${LOCOMO_QUESTIONS_PATH}" \
  --predictions-path "${LOCOMO_PREDICTIONS_PATH}" \
  "${DEFAULT_METRICS[@]}" \
  --output-path "${LOCOMO_SCORED_PATH}" \
  "${EVAL_ARGS[@]}"

echo "[2/2] Summarizing scores..." >&2
"${LOCOMO_PYTHON_BIN}" -c 'from summarize import main; main()' \
  --input-path "${LOCOMO_SCORED_PATH}" \
  --output-path "${LOCOMO_SUMMARY_PATH}" \
  --markdown-output-path "${LOCOMO_MARKDOWN_SUMMARY_PATH}"
