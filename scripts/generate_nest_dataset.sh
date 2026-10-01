#!/usr/bin/env bash
#SBATCH --job-name=nest_gen
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

set -euo pipefail
trap 'ec=$?; echo "[$(date)] ERROR: failed at line ${LINENO} (exit $ec)" >&2' ERR

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ ! -d "$REPO_DIR/data" ]]; then
  echo "ERROR: REPO_DIR=$REPO_DIR doesn't look like the repo root (no data/ found) -- Slurm may have run a relocated copy of this script. Override explicitly: REPO_DIR=\$HOME/BrainDyn-SUMRY sbatch <script>" >&2
  exit 1
fi
cd "$REPO_DIR"
mkdir -p logs/slurm

OUT_DIR="${OUT_DIR:-data/simulated_neuron_dataset}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-1000}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ACTIVATE_CMD="${ACTIVATE_CMD:-source .venv/bin/activate}"
echo "[$(date)] Activating: ${ACTIVATE_CMD}"
eval "${ACTIVATE_CMD}"

python -c "import nest" || {
  echo "[$(date)] ERROR: \`import nest\` still fails after '${ACTIVATE_CMD}'." >&2
  echo "Set ACTIVATE_CMD to whatever environment actually has NEST installed." >&2
  exit 1
}

echo "[$(date)] Generating NEST dataset"
echo "  out_dir=${OUT_DIR}  num_simulations=${NUM_SIMULATIONS}  seed=${SEED}"
echo "  commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# shellcheck disable=SC2086
python data/simulate_neuron_dataset.py \
  --num-simulations "${NUM_SIMULATIONS}" \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}" \
  ${EXTRA_ARGS}

echo "[$(date)] Done. Provenance + seed formulas are recorded in ${OUT_DIR}/run_config.json"
