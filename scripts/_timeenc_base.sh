#!/bin/bash
# Shared flag definitions for the sinusoidal-time-encoding experiments.

SHUFFLE_SEEDS="${SHUFFLE_SEEDS:-2 3 4 5 6 8}"
read -ra TIMEENC_SHUFFLE_SEEDS <<< "${SHUFFLE_SEEDS}"

SHEAF_NORM="${SHEAF_NORM:-sym}"
DIFFUSION_STEP="${DIFFUSION_STEP:-1.0}"
SHEAF_TAG=""
[[ "${SHEAF_NORM}" != "sym" ]] && SHEAF_TAG="${SHEAF_TAG}_norm${SHEAF_NORM}"
[[ "${DIFFUSION_STEP}" != "1.0" ]] && SHEAF_TAG="${SHEAF_TAG}_ds${DIFFUSION_STEP}"

# The ODE + plain-sheaf configuration, shared by every script in this family.
TIMEENC_FLAGS=(
  --sheaf_map_pe learned --sheaf_map_pe_dim 8
  --sheaf_norm "${SHEAF_NORM}"
  --diffusion_step "${DIFFUSION_STEP}"
  --no-identity_restriction_init
  --sheaf_map_scale none
  --vf_layers=4
  --no-freeze_map_scale
  --coupling_block none
  --sheaf_layers 1
  --time_learn_freqs
)

TIMEENC_OPT_FLAGS=(
  --lr_patience 10 --lr_factor 0.5 --lr_min 1e-5
  --weight_decay 1e-5 --grad_clip 1.0
  --amp --no_pin_memory
)

timeenc_graph_flags() {
  local total_edges="$1"
  if [[ -n "${TOPK_PER_NODE:-}" ]]; then
    TIMEENC_GRAPH_FLAGS=(
      --graph_mode granger --granger_lag 1
      --granger_threshold_mode topk_per_node
      --granger_topk_per_node "${TOPK_PER_NODE}"
    )
    GRAPH_TAG="_tpn${TOPK_PER_NODE}"
  else
    TIMEENC_GRAPH_FLAGS=(
      --graph_mode granger --granger_lag 1
      --granger_threshold_mode topk --granger_topk_edges "${total_edges}"
    )
    GRAPH_TAG=""
  fi
  RUN_TAG="${GRAPH_TAG}${SHEAF_TAG}"
}

# Resolve an arm name from the SLURM array index.
timeenc_resolve_arm() {
  local -a arms=("$@")
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    ARM="${arms[${SLURM_ARRAY_TASK_ID}]}"
  else
    ARM="${ARM:-${arms[0]}}"
  fi
}
