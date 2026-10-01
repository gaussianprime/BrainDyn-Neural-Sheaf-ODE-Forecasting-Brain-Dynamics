#!/usr/bin/env bash
# fMRI (PNC) -- SHORT horizon time-encoding ABLATIONS: {no_sheaf, no_lstm}.
#
#SBATCH --job-name=bd_timeenc_fmri_short_ablation
#SBATCH --partition=gpu,gpu_h200,gpu_rtx6000,gpu_b200,gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --array=0-2
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err

set -euo pipefail
trap 'echo "[$(date)] ERROR: arm=${ARM:-<unset>} failed at line ${LINENO} (exit $?)" >&2' ERR

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/BrainDyn-SUMRY}"
cd "${REPO_DIR}"
mkdir -p logs/slurm logs/timeenc_fmri checkpoints/timeenc_fmri
source .venv/bin/activate
source scripts/_timeenc_base.sh

MANIFEST_CSV="${MANIFEST_CSV:-data/manifest_fmri.csv}"
EPOCHS="${EPOCHS:-120}"
LR="${LR:-1e-3}"
SEED="${SEED:-2}"

timeenc_graph_flags 2000
timeenc_resolve_arm no_sheaf no_lstm no_time_encoding

case "${ARM}" in
  no_sheaf) ARM_FLAGS=(--no_sheaf) ;;
  no_lstm)  ARM_FLAGS=(--ablation_no_lstm) ;;
  no_time_encoding) ARM_FLAGS=(--time_embed_dim 0) ;;
  *) echo "ERROR: unknown ARM '${ARM}' (expected no_sheaf|no_lstm|no_time_encoding)" >&2; exit 1 ;;
esac

SAVE_PATH="checkpoints/timeenc_fmri/timeenc_short_${ARM}${RUN_TAG}.pt"
LOG_PATH="logs/timeenc_fmri/timeenc_short_${ARM}${RUN_TAG}.log"

CMD=(
  python main.py
  --dataset fmri
  --cohort PNC
  --manifest_csv "${MANIFEST_CSV}"
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
  --save_path "${SAVE_PATH}"
  "${ARM_FLAGS[@]}"
)

echo "[$(date)] fMRI timeenc SHORT ablation: arm=${ARM}  (task ${SLURM_ARRAY_TASK_ID:-n/a})"
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}"
echo "branch=$(git branch --show-current 2>/dev/null || true)  commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
echo "manifest=${MANIFEST_CSV}"
nvidia-smi -L || true
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
