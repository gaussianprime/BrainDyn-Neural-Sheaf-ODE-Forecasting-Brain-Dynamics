#!/usr/bin/env bash
# Functional-connectivity (|Pearson corr|, top-k) graph prior across all four
# timeenc arms -- fMRI/EEG x short/long_ar -- one SLURM task per shuffle.
#
# Array layout (size 20): task = arm*5 + shuffle_index
#   arm 0: fmri_short   tasks 0-4
#   arm 1: fmri_long    tasks 5-9
#   arm 2: eeg_short    tasks 10-14
#   arm 3: eeg_long     tasks 15-19
#
#SBATCH --job-name=bd_timeenc_fc_all
#SBATCH --partition=gpu,gpu_h200,gpu_rtx6000,gpu_b200,gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --array=0-19
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err

set -euo pipefail
trap 'echo "[$(date)] ERROR: arm=${ARM:-<unset>} shuffle_index=${SHUFFLE_INDEX:-<unset>} seed=${SEED_VAL:-<unset>} failed at line ${LINENO} (exit $?)" >&2' ERR

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/BrainDyn-SUMRY}"
cd "${REPO_DIR}"
mkdir -p logs/slurm logs/timeenc_fmri logs/timeenc_eeg \
         checkpoints/timeenc_fmri checkpoints/timeenc_eeg
source .venv/bin/activate

# 5 shuffles per arm x 4 arms = 20 tasks.
SHUFFLE_SEEDS="${SHUFFLE_SEEDS:-2 3 4 5 6}"
source scripts/_timeenc_base.sh

N_SHUFFLE="${N_SHUFFLE:-5}"
ARMS=(fmri_short fmri_long eeg_short eeg_long)
N_ARM=${#ARMS[@]}
N_TASKS=$(( N_ARM * N_SHUFFLE ))

if [[ ${#TIMEENC_SHUFFLE_SEEDS[@]} -lt ${N_SHUFFLE} ]]; then
  echo "ERROR: need >= ${N_SHUFFLE} shuffle seeds, got ${#TIMEENC_SHUFFLE_SEEDS[@]} (SHUFFLE_SEEDS='${SHUFFLE_SEEDS}')" >&2
  exit 1
fi

TASK="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK < 0 || TASK >= N_TASKS )); then
  echo "ERROR: task ${TASK} out of range: need --array=0-$(( N_TASKS - 1 )) for ${N_ARM} arms x ${N_SHUFFLE} shuffles" >&2
  exit 1
fi
ARM="${ARMS[$(( TASK / N_SHUFFLE ))]}"
SHUFFLE_INDEX=$(( TASK % N_SHUFFLE ))
SEED_VAL="${TIMEENC_SHUFFLE_SEEDS[${SHUFFLE_INDEX}]}"

EPOCHS="${EPOCHS:-120}"
LR="${LR:-1e-3}"
SEED="${SEED:-2}"
USE_AMP="${USE_AMP:-1}"
FMRI_MANIFEST_CSV="${FMRI_MANIFEST_CSV:-data/manifest_fmri.csv}"
EEG_MANIFEST="${EEG_MANIFEST:-data/lemon_manifest.csv}"

# FC graph prior: same top-k selection knobs as the Granger arms, but scored by
# |Pearson corr|. Honors TOPK_PER_NODE (falls back to a total-edge budget).
fc_graph_flags() {
  local total_edges="$1"
  if [[ -n "${TOPK_PER_NODE:-}" ]]; then
    FC_GRAPH_FLAGS=(
      --graph_mode fc
      --granger_threshold_mode topk_per_node
      --granger_topk_per_node "${TOPK_PER_NODE}"
    )
    GRAPH_TAG="_tpn${TOPK_PER_NODE}"
  else
    FC_GRAPH_FLAGS=(
      --graph_mode fc
      --granger_threshold_mode topk --granger_topk_edges "${total_edges}"
    )
    GRAPH_TAG=""
  fi
  RUN_TAG="_fc${GRAPH_TAG}${SHEAF_TAG}"
}

# Per-arm dataset + horizon flags.
case "${ARM}" in
  fmri_short|fmri_long)
    DOMAIN=fmri
    fc_graph_flags 2000
    DATA_FLAGS=(
      --dataset fmri --cohort PNC
      --manifest_csv "${FMRI_MANIFEST_CSV}" --cache
    )
    ;;
  eeg_short|eeg_long)
    DOMAIN=eeg
    fc_graph_flags 305
    DATA_FLAGS=(
      --dataset lemon_eeg --lemon_manifest_csv "${EEG_MANIFEST}"
      --condition EC --max_subjects 202 --max_windows_per_subject 150
      --lemon_data_seed 42 --cache
    )
    ;;
  *)
    echo "ERROR: unknown arm '${ARM}'" >&2; exit 1 ;;
esac

case "${ARM}" in
  fmri_short|eeg_short)
    HORIZON=short
    HORIZON_FLAGS=(--x 32 --y 8 --stride 8 --forecast_mode short)
    ;;
  fmri_long)
    HORIZON=long_ar
    HORIZON_FLAGS=(
      --x 32 --y 8 --stride 8
      --forecast_mode long_ar_train --ar_chunk_size 8 --tbptt_chunks 3
      --test_rollout_steps 32 --run_batch_size 8
      --ss_start 1.0 --ss_end 0.0 --val_every 1
    )
    ;;
  eeg_long)
    HORIZON=long_ar
    HORIZON_FLAGS=(
      --x 32 --y 8 --stride 8
      --forecast_mode long_ar_train --ar_chunk_size 8
      --ar_segment_len 64 --ar_segments_per_subject 38 --tbptt_chunks 3
      --test_rollout_steps 32 --run_batch_size 128
      --ss_start 1.0 --ss_end 0.0 --ss_decay_epochs 30 --val_every 1
    )
    ;;
esac

AMP_ARGS=()
[[ "${USE_AMP}" == "1" ]] && AMP_ARGS=(--amp)

SAVE_PATH="checkpoints/timeenc_${DOMAIN}/timeenc_${HORIZON}_seed${SEED_VAL}${RUN_TAG}.pt"
LOG_PATH="logs/timeenc_${DOMAIN}/timeenc_${HORIZON}_seed${SEED_VAL}${RUN_TAG}.log"

CMD=(
  python main.py
  "${DATA_FLAGS[@]}"
  "${HORIZON_FLAGS[@]}"
  "${FC_GRAPH_FLAGS[@]}"
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

echo "[$(date)] timeenc FC: arm=${ARM} (${DOMAIN}/${HORIZON}) shuffle_index=${SHUFFLE_INDEX} seed=${SEED_VAL} (task ${SLURM_ARRAY_TASK_ID:-n/a}/$(( N_TASKS - 1 )))"
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}"
echo "branch=$(git branch --show-current 2>/dev/null || true)  commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
echo "graph=FC top-k  run_tag=${RUN_TAG}"
nvidia-smi -L || true
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
