#!/usr/bin/env bash
# LEMON EEG -- LONG horizon (autoregressive), sinusoidal time encoding.
#
#SBATCH --job-name=bd_timeenc_eeg_long_ar
#SBATCH --partition=gpu,gpu_h200,gpu_rtx6000,gpu_b200,gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --array=0-5
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err

set -euo pipefail
trap 'echo "[$(date)] ERROR: shuffle_index=${SHUFFLE_INDEX:-<unset>} seed=${SEED_VAL:-<unset>} failed at line ${LINENO} (exit $?)" >&2' ERR

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/BrainDyn-SUMRY}"
cd "${REPO_DIR}"
mkdir -p logs/slurm logs/timeenc_eeg checkpoints/timeenc_eeg
source .venv/bin/activate
source scripts/_timeenc_base.sh

MANIFEST="${MANIFEST:-data/lemon_manifest.csv}"
EPOCHS="${EPOCHS:-120}"
LR="${LR:-1e-3}"
SEED="${SEED:-2}"
USE_AMP="${USE_AMP:-1}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHUFFLE_INDEX="${SLURM_ARRAY_TASK_ID}"
else
  SHUFFLE_INDEX="${SHUFFLE_INDEX:-0}"
fi

SEED_VAL="${TIMEENC_SHUFFLE_SEEDS[${SHUFFLE_INDEX}]}"

timeenc_graph_flags 305

SAVE_PATH="checkpoints/timeenc_eeg/timeenc_long_ar_seed${SEED_VAL}${RUN_TAG}.pt"
LOG_PATH="logs/timeenc_eeg/timeenc_long_ar_seed${SEED_VAL}${RUN_TAG}.log"

AMP_ARGS=()
[[ "${USE_AMP}" == "1" ]] && AMP_ARGS=(--amp)

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
  --forecast_mode long_ar_train
  --ar_chunk_size 8
  --ar_segment_len 64
  --ar_segments_per_subject 38
  --tbptt_chunks 3
  --test_rollout_steps 32
  --run_batch_size 128
  --ss_start 1.0
  --ss_end 0.0
  --ss_decay_epochs 30
  --val_every 1
  "${TIMEENC_GRAPH_FLAGS[@]}"
  --epochs "${EPOCHS}"
  --shuffle_seeds "${TIMEENC_SHUFFLE_SEEDS[@]}"
  --shuffle_index "${SHUFFLE_INDEX}"
  --batch_size 64 --num_workers 2
  --hidden_dim 16 --map_hidden_dim 16 --map_mlp_hidden_dim 64 --vf_hidden_dim 128
  "${TIMEENC_FLAGS[@]}"
  --lr "${LR}" --lr_patience 10 --lr_factor 0.5 --lr_min 1e-5
  --weight_decay 1e-5 --grad_clip 1.0
  "${AMP_ARGS[@]}"
  --no_pin_memory
  --seed "${SEED}"
  --save_path "${SAVE_PATH}"
)

echo "[$(date)] EEG timeenc LONG_AR: shuffle_index=${SHUFFLE_INDEX} seed=${SEED_VAL} (task ${SLURM_ARRAY_TASK_ID:-n/a})"
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}"
echo "branch=$(git branch --show-current 2>/dev/null || true)  commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
nvidia-smi -L || true
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
