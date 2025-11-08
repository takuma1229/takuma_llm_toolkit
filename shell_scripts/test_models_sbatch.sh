#!/usr/bin/env bash
#
# SLURM batch script: takuma-llm-toolkit のスモークテストを GPU ノードで実行します。
# 既定では `smoke` マーカーのみを実行します。追加の pytest 引数はそのまま渡せます。
#
# 使い方:
#   sbatch shell_scripts/test_models_sbatch.sh              # smoke マーカーのみ
#   sbatch --gres=gpu:1 -p <partition> shell_scripts/test_models_sbatch.sh "-q -m smoke"
#   sbatch shell_scripts/test_models_sbatch.sh "-q -k gemma"
#
# モデル上書き（例）:
#   TAKUMA_TEST_QWEN="Qwen/Qwen2.5-7B-Instruct" \
#   sbatch shell_scripts/test_models_sbatch.sh
#
# 必要に応じて以下の #SBATCH 行を環境やポリシーに合わせて変更してください。
#
#SBATCH --job-name=takuma-llm-tests
#SBATCH --gres=gpu:1
# 注意: 出力先ディレクトリが存在しないと Submit 直後に CANCELLED になります。
# 既定ではカレントに出力し、必要なら sbatch の -o/-e で上書きしてください。
# 例: sbatch -o logs/%x_%j.out -e logs/%x_%j.err shell_scripts/test_models_sbatch.sh
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# リポジトリルートへ移動
cd "$(dirname "$0")/.."

mkdir -p logs || true

echo "[INFO] Hostname: $(hostname)" | tee -a logs/tests_env.log
echo "[INFO] Date: $(date -Is)" | tee -a logs/tests_env.log
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

# HF キャッシュの既定（任意）
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

# pytest オプションの既定（smoke マーカーのみ）。引数があればそれを優先。
MARKERS="${TAKUMA_TEST_MARKERS:-smoke}"
if [[ $# -gt 0 ]]; then
  # すべて引数を pytest に丸投げ
  PYTEST_OPTS=("$@")
else
  PYTEST_OPTS=(-q test/test_models.py -m "$MARKERS")
fi

# 実行コマンドの解決
if command -v uv >/dev/null 2>&1; then
  EXEC=(uv run pytest)
elif [[ -x .venv/bin/python ]]; then
  # venv がある場合はそれを使う
  source .venv/bin/activate
  EXEC=(pytest)
else
  EXEC=(python -m pytest)
fi

set -x
if command -v srun >/dev/null 2>&1; then
  srun --ntasks=1 "${EXEC[@]}" "${PYTEST_OPTS[@]}"
else
  "${EXEC[@]}" "${PYTEST_OPTS[@]}"
fi
