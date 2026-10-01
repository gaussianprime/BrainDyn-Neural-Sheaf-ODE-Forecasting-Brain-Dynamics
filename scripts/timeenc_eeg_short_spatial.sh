#!/usr/bin/env bash
# LEMON EEG -- SHORT horizon, sinusoidal time encoding, PLAIN sheaf settings.
#
#SBATCH --job-name=bd_timeenc_eeg_short_spatial
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/slurm/%x_%A.out
#SBATCH --error=./logs/slurm/%x_%A.err

set -euo pipefail
trap 'echo "[$(date)] ERROR: failed at line ${LINENO} (exit $?)" >&2' ERR

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/BrainDyn-SUMRY}"
cd "${REPO_DIR}"
mkdir -p logs/slurm logs/timeenc_eeg checkpoints/timeenc_eeg
source .venv/bin/activate
source scripts/_timeenc_base.sh

MANIFEST="${MANIFEST:-data/lemon_manifest.csv}"
EPOCHS="${EPOCHS:-120}"
LR="${LR:-1e-3}"
SEED="${SEED:-2}"

timeenc_graph_flags 305

SAVE_PATH="checkpoints/timeenc_eeg/timeenc_short${RUN_TAG}.pt"
LOG_PATH="logs/timeenc_eeg/timeenc_short${RUN_TAG}.log"

CMD=(
  python main.py
  --dataset lemon_eeg
  --lemon_manifest_csv "${MANIFEST}"
  --condition EC
  --max_subjects 202
  --max_windows_per_subject 150
  --lemon_data_seed 42
  --cache
  --x 32 --y 8 --stride 8
  --forecast_mode short
  "${TIMEENC_GRAPH_FLAGS[@]}"
  --epochs "${EPOCHS}"
  --shuffle_seeds "${TIMEENC_SHUFFLE_SEEDS[@]}"
  --batch_size 64 --num_workers 2
  --hidden_dim 16 --map_hidden_dim 16 --map_mlp_hidden_dim 64 --vf_hidden_dim 128
  "${TIMEENC_FLAGS[@]}"
  --lr "${LR}"
  "${TIMEENC_OPT_FLAGS[@]}"
  --seed "${SEED}"
  --graph_mode "spatial"
  --save_path "${SAVE_PATH}"
)

echo "[$(date)] EEG timeenc SHORT"
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}"
echo "branch=$(git branch --show-current 2>/dev/null || true)  commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
nvidia-smi -L || true
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"