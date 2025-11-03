#!/bin/bash
# 直接 sbatch 提出可能な LLM 単発推論ジョブ
#
# Usage (例):
#   sbatch shell_scripts/run_llm_once.sh --model meta-llama/Llama-3.1-8B-Instruct
#   sbatch shell_scripts/run_llm_once.sh -m Qwen/Qwen2.5-7B-Instruct --engine vllm
#
#SBATCH --partition public
#SBATCH --ntasks=1
# 標準出力・標準エラーを同一ログに集約
#SBATCH --output=./slurm_logs/slurm-%j.out
#SBATCH --error=./slurm_logs/slurm-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

# 実行ディレクトリを sbatch 提出ディレクトリに固定し、ローカルパッケージを優先
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

show_usage() {
  cat <<'USAGE'
run_llm_once.sh — 単発のLLM推論ジョブ（sbatch 直投げ用）

使い方:
  sbatch shell_scripts/run_llm_once.sh \
    --model MODEL [--engine {normal|vllm}] [--prompt TEXT | --prompt-file PATH] [--python PATH] \
    [--max-new-tokens N] [--repetition-penalty X] [--temperature X] [--top-p X] [--top-k N] [--do-sample|--no-do-sample] \
    [--tensor-parallel-size N] [--gpu-memory-utilization X]

必須:
  -m, --model MODEL           使用するモデル名

任意:
  -e, --engine {normal|vllm}  推論エンジン（既定: normal）
  -p, --prompt TEXT           入力プロンプト文字列
  -f, --prompt-file PATH      プロンプトファイル（指定時は --prompt より優先）
  -P, --python PATH           Python 実行コマンド（既定: python）
  --max-new-tokens N          生成トークン数の上限
  --repetition-penalty X      反復抑制係数
  --temperature X             サンプリング温度
  --top-p X                   nucleus サンプリングの確率質量
  --top-k N                   上位kサンプリング
  --do-sample | --no-do-sample サンプリング有効/無効
  --tensor-parallel-size N    vLLM テンソル並列数
  --gpu-memory-utilization X  vLLM GPUメモリ使用率(0-1)
  --max-model-len N           vLLM の最大シーケンス長
  -h, --help                  このヘルプを表示

例:
  sbatch --gres=gpu:1 shell_scripts/run_llm_once.sh -m meta-llama/Llama-3.1-8B-Instruct
  sbatch --gres=gpu:1 shell_scripts/run_llm_once.sh -m Qwen/Qwen2.5-7B-Instruct -e vllm -p "こんにちは"
  sbatch --gres=gpu:1 shell_scripts/run_llm_once.sh -m meta-llama/Llama-3.1-8B-Instruct -f prompt.txt -P /opt/conda/bin/python
USAGE
}

# 入力パラメータ（最小: model のみ）
MODEL=""
ENGINE_OPT=""
PROMPT_OPT=""
PROMPT_FILE_OPT=""
PYTHON_OPT=""
MAX_NEW_TOKENS_OPT=""
REPETITION_PENALTY_OPT=""
TEMPERATURE_OPT=""
TOP_P_OPT=""
TOP_K_OPT=""
DO_SAMPLE_FLAG=""   # --do-sample または --no-do-sample
TP_SIZE_OPT=""
GPU_MEM_UTIL_OPT=""
MAX_MODEL_LEN_OPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="${2:-}"; shift 2;;
    -e|--engine) ENGINE_OPT="${2:-}"; shift 2;;
    -p|--prompt) PROMPT_OPT="${2:-}"; PROMPT_FILE_OPT=""; shift 2;;
    -f|--prompt-file) PROMPT_FILE_OPT="${2:-}"; shift 2;;
    -P|--python) PYTHON_OPT="${2:-}"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS_OPT="${2:-}"; shift 2;;
    --repetition-penalty) REPETITION_PENALTY_OPT="${2:-}"; shift 2;;
    --temperature) TEMPERATURE_OPT="${2:-}"; shift 2;;
    --top-p) TOP_P_OPT="${2:-}"; shift 2;;
    --top-k) TOP_K_OPT="${2:-}"; shift 2;;
    --do-sample) DO_SAMPLE_FLAG="--do-sample"; shift 1;;
    --no-do-sample) DO_SAMPLE_FLAG="--no-do-sample"; shift 1;;
    --tensor-parallel-size) TP_SIZE_OPT="${2:-}"; shift 2;;
    --gpu-memory-utilization) GPU_MEM_UTIL_OPT="${2:-}"; shift 2;;
    --max-model-len) MAX_MODEL_LEN_OPT="${2:-}"; shift 2;;
    -h|--help)  show_usage; exit 0;;
    -*) echo "Error: 不明なオプション: $1" >&2; show_usage >&2; exit 2;;
    *)  if [[ -z "${MODEL}" ]]; then MODEL="$1"; else echo "Error: 余分な引数: $1" >&2; exit 2; fi; shift;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "Error: モデル名が指定されていません（-m/--model）。" >&2
  show_usage >&2
  exit 2
fi

# 環境設定（ENGINE/PYTHON/PROMPT は環境変数で受ける）
ENGINE="${ENGINE_OPT:-normal}"

case "${ENGINE}" in
  normal|vllm) :;;
  *) echo "Error: --engine は normal|vllm のいずれかを指定してください（今: ${ENGINE}）。" >&2; exit 2;;
esac
PYTHON="${PYTHON:-python}"

PYTHON="${PYTHON_OPT:-python}"

# PROMPT の決定（ファイルがあれば優先）
if [[ -n "${PROMPT_FILE_OPT}" ]]; then
  if [[ -r "${PROMPT_FILE_OPT}" ]]; then
    PROMPT="$(<"${PROMPT_FILE_OPT}")"
  else
    echo "Error: 指定された --prompt-file が読み取れません: ${PROMPT_FILE_OPT}" >&2
    exit 2
  fi
else
  PROMPT="${PROMPT_OPT:-日本語で3行自己紹介して}"
fi

echo "---- machinefile ----"
srun -n ${SLURM_NTASKS:-1} hostname | sort > my_hosts

echo "---- Slurm Environment Variables ----"
cat <<ETX
JOB_ID="${SLURM_JOB_ID:-}"
JOB_NAME="${SLURM_JOB_NAME:-}"
PARTITION_NAME="${SLURM_JOB_PARTITION:-}"
NODE_LIST="${SLURM_JOB_NODELIST:-}"
NTASKS="${SLURM_NTASKS:-}"
ETX

echo "---- 実行を開始します ----"

# GPUログをバックグラウンドでslurmログに書き込み
(
  while true; do
    nvidia-smi || true
    echo "========================"
    sleep 10
  done
) &
GPU_MON_PID=$!
trap 'kill ${GPU_MON_PID} 2>/dev/null || true' EXIT SIGTERM SIGINT

# setup.sh がある場合のみ環境準備（任意）
if [[ -f setup.sh ]]; then
  bash setup.sh --no-pre-commit --no-kernel --no-enter
fi

echo "[RUNNER] Engine     : ${ENGINE}"
echo "[RUNNER] Model      : ${MODEL}"
echo "[RUNNER] PromptLen  : ${#PROMPT} chars"
echo "[RUNNER] Python(bin): $(command -v "${PYTHON}" || which python || true)"

# 実行（-u: 行単位フラッシュ）— オプションを適宜付与
ARGS=( -e "${ENGINE}" "${MODEL}" -p "${PROMPT}" )
[[ -n "${MAX_NEW_TOKENS_OPT}" ]] && ARGS+=( --max-new-tokens "${MAX_NEW_TOKENS_OPT}" )
[[ -n "${REPETITION_PENALTY_OPT}" ]] && ARGS+=( --repetition-penalty "${REPETITION_PENALTY_OPT}" )
[[ -n "${TEMPERATURE_OPT}" ]] && ARGS+=( --temperature "${TEMPERATURE_OPT}" )
[[ -n "${TOP_P_OPT}" ]] && ARGS+=( --top-p "${TOP_P_OPT}" )
[[ -n "${TOP_K_OPT}" ]] && ARGS+=( --top-k "${TOP_K_OPT}" )
[[ -n "${DO_SAMPLE_FLAG}" ]] && ARGS+=( ${DO_SAMPLE_FLAG} )
[[ -n "${TP_SIZE_OPT}" ]] && ARGS+=( --tensor-parallel-size "${TP_SIZE_OPT}" )
[[ -n "${GPU_MEM_UTIL_OPT}" ]] && ARGS+=( --gpu-memory-utilization "${GPU_MEM_UTIL_OPT}" )
[[ -n "${MAX_MODEL_LEN_OPT}" ]] && ARGS+=( --max-model-len "${MAX_MODEL_LEN_OPT}" )

srun -u "${PYTHON}" run_llm.py "${ARGS[@]}"

echo "---- 実行が終了しました ----"
