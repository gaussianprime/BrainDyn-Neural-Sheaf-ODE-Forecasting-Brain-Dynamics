#!/usr/bin/env bash
#SBATCH --job-name=bd_nest_f01
#SBATCH --partition=scavenge_gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/slurm/%x_%A_%a.out
#SBATCH --error=./logs/slurm/%x_%A_%a.err
#
# NEST perturbed forecasting with the structural graph prior (4 edges per node),
# trained and evaluated with 10% of the 32-bin context at or after the
# perturbation onset. Train, val and test are drawn from the same subject pool.
#
# Grid (task = cell * 5 + shuffle), identical at both horizons:
#   0-19   BrainDyn: full nosheaf nolstm notime
#
# Usage:
#   HORIZON=short  sbatch --array=0-19 --time=00:40:00 scripts/nest_f01_train.sh
#   HORIZON=arlong sbatch --array=0-19 scripts/nest_f01_train.sh
# Set NEST_NPZ_PATH to the silence_dc dataset.npz if it is not at the default.
# A finished task is skipped on resubmission; an interrupted one resumes from
# its per-epoch checkpoint.
set -uo pipefail

_bd_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd || true)"
REPO_DIR="${REPO_DIR:-}"
for _bd_c in "${REPO_DIR}" "${SLURM_SUBMIT_DIR:-}" "${_bd_root}"; do
  if [[ -n "$_bd_c" && -d "$_bd_c/train" ]]; then REPO_DIR="$_bd_c"; break; fi
done
unset _bd_root _bd_c
[[ -d "$REPO_DIR/train" ]] || { echo "ERROR: set REPO_DIR to the repo root" >&2; exit 1; }
cd "$REPO_DIR"
mkdir -p logs/slurm logs/benchmarks
source .venv/bin/activate
export PYTHONUNBUFFERED=1

HORIZON="${HORIZON:?set HORIZON=arlong or HORIZON=short}"
FRAC="${FRAC:-0.1}"
N_SHUFFLE=5
NPZ="${NEST_NPZ_PATH:-nest_simulated_neurons_silencedc/dataset.npz}"
GRAPH_FLAGS=(--graph_mode structural --fc_threshold_mode topk_per_node --fc_topk_per_node 4)

CELLS=(full nosheaf nolstm notime)
N_CELL=${#CELLS[@]}
case "${HORIZON}" in
  arlong) HORIZON_FLAGS=(--forecast_mode long_ar_train
                         --ar_chunk_size 8 --tbptt_chunks 3 --ar_stride 72
                         --test_rollout_steps 32 --run_batch_size 8
                         --ss_start 0.0 --ss_end 0.0 --val_every 1) ;;
  short)  HORIZON_FLAGS=(--forecast_mode short) ;;
  *) echo "ERROR: HORIZON must be arlong or short, got '${HORIZON}'" >&2; exit 1 ;;
esac
TAG="nest_f${FRAC//./}_structural_${HORIZON}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
(( TASK >= 0 && TASK < N_CELL * N_SHUFFLE )) || {
  echo "ERROR: task ${TASK} outside 0-$(( N_CELL * N_SHUFFLE - 1 ))" >&2; exit 1; }
CELL="${CELLS[$(( TASK / N_SHUFFLE ))]}"
S=$(( TASK % N_SHUFFLE ))

[[ -f "${NPZ}" ]] || { echo "ERROR: dataset not found: ${NPZ} (set NEST_NPZ_PATH)" >&2; exit 1; }
grep -q "\"${TAG}\"" scripts/summarize_benchmark_results.py || {
  echo "ERROR: ${TAG} is not registered in scripts/summarize_benchmark_results.py" >&2; exit 1; }

BRAINDYN=(--model braindyn
          --time_embed_max_period 16.0 --time_learn_freqs
          --time_max_cycles_per_step 0.5 --time_rate_mode fixed
          --sheaf_norm sym --sheaf_map_scale none
          --coupling_block none --diffusion_step 1.0
          --map_hidden_dim 16 --map_mlp_hidden_dim 64 --vf_hidden_dim 128
          --vf_layers 2)
case "${CELL}" in
  full)      MODEL_FLAGS=("${BRAINDYN[@]}" --sheaf_layers 1 --time_embed_dim 16) ;;
  nosheaf)   MODEL_FLAGS=("${BRAINDYN[@]}" --sheaf_layers 1 --time_embed_dim 16 --no_sheaf) ;;
  nolstm)    MODEL_FLAGS=("${BRAINDYN[@]}" --sheaf_layers 1 --time_embed_dim 16 --ablation_no_lstm) ;;
  notime)    MODEL_FLAGS=("${BRAINDYN[@]}" --sheaf_layers 1 --time_embed_dim 0) ;;
esac

RUN_DIR="checkpoints/${TAG}/${CELL}"; mkdir -p "${RUN_DIR}"
LOG="logs/benchmarks/${TAG}_${CELL}_fold${S}.log"
echo "[$(date)] ${TAG} cell=${CELL} shuffle=${S} commit=$(git rev-parse --short HEAD)"

if [[ -f "${LOG}" ]] && grep -q "=== Benchmark Summary (perturbed horizon) ===" "${LOG}"; then
  echo "already complete: ${LOG}"; exit 0
fi
[[ -f "${LOG}" ]] && mv "${LOG}" "${LOG}.prev.$(date +%s)"

python -m train.train_nest_braindyn \
  "${MODEL_FLAGS[@]}" \
  --dataset nest --nest_npz_path "${NPZ}" \
  --nest_task_mode perturb_forecast \
  --perturb_post_onset_frac "${FRAC}" \
  --x 32 --y 8 --stride 8 \
  "${HORIZON_FLAGS[@]}" \
  --norm_mode train_global \
  --sheaf_map_pe none \
  "${GRAPH_FLAGS[@]}" \
  --epochs 120 \
  --num_shuffles "${N_SHUFFLE}" --shuffle_index "${S}" \
  --train_frac 0.7 --val_frac 0.2 \
  --batch_size 64 --num_workers 2 \
  --hidden_dim 16 \
  --lr 1e-3 --lr_patience 3 --lr_factor 0.5 --lr_min 1e-6 \
  --weight_decay 1e-5 --grad_clip 1.0 \
  --seed 2 --amp --no_pin_memory \
  --save_path "${RUN_DIR}/braindyn.pt" \
  > "${LOG}" 2>&1
st=$?
echo "[$(date)] exit ${st}: ${LOG}"
exit "${st}"
