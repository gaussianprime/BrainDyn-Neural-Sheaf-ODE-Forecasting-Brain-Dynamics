# BrainDyn datasets

BrainDyn trains on three datasets: **PNC fMRI**, **LEMON EEG**, and **NEST (simulated)**. This document covers how each dataset's manifest and derived files are built, what a training sample looks like, and which `main.py` flags apply.

The raw source data (PNC imaging, raw LEMON EEG recordings) lives on the shared HPC store, not in git. Only manifests and small derived arrays are expected under `data/`.

---

## PNC fMRI

Philadelphia Neurodevelopmental Cohort resting-state fMRI, drawn from the RBC (ReproBrainChart) PNC+HBN release, parcellated into 400 ROIs with the Schaefer-400 / 7-Networks atlas via the CPAC pipeline.

### Manifest

`data/rbc_dataset.py` reads a CSV manifest (default path `data/manifest.csv`, passed via `--manifest_csv`), one row per scan run, with columns:

| Column | Meaning |
|---|---|
| `cohort` | `PNC` or `HBN` |
| `subject_id`, `session`, `run` | Identify the scan |
| `site` | Acquisition site |
| `split` | `train` / `val` / `test`, assigned at the subject level |
| `T` | Number of timepoints (TRs) in the run |
| `path` | Path to the run's parcellated timeseries (`.1D`, AFNI format — comma-delimited, `#`-commented header, shape `(T, 400)`) |

This repo does not include the `build_manifest.py` that produces this CSV from a CPAC output tree — the manifest is expected to already exist (built upstream) and only needs pointing to via `--manifest_csv`. The experiment scripts default to `data/manifest_fmri.csv`, overridable via the `MANIFEST_CSV` environment variable.

### Sample

Each item from `RBCDataset` is a sliding-window `(context, horizon)` pair:

```python
x    : Tensor[L_x, 400]   # context window
y    : Tensor[L_y, 400]   # horizon to predict
meta : dict               # cohort, subject_id, session, run, site, split, T, path
```

Window length/horizon/stride are set by `--x` / `--y` / `--stride` (TRs); `--cohort PNC` restricts the manifest to the PNC rows. A `"within"` split mode (one sample per run, `context = [T-x-y : T-y]`, `horizon = [T-y : T]`) is also available for within-subject held-out evaluation, independent of the manifest's between-subject `split` column.

### Running

```bash
python main.py \
  --dataset fmri --cohort PNC \
  --manifest_csv data/manifest.csv \
  --x 32 --y 8 --stride 8 --forecast_mode short \
  --graph_mode granger --granger_lag 1 --granger_threshold_mode topk --granger_topk_edges 2000
```

See [`scripts/timeenc_fmri_short_spatial.sh`](../scripts/timeenc_fmri_short_spatial.sh) / [`scripts/timeenc_fmri_long_ar_spatial.sh`](../scripts/timeenc_fmri_long_ar_spatial.sh) for the paper's main configurations (spatial graph; the Granger example above matches [`scripts/timeenc_fmri_short.sh`](../scripts/timeenc_fmri_short.sh)), and the root [README's Experiments section](../README.md#experiments) for the full command list.

---

## LEMON EEG

MPI Leipzig Mind-Brain-Body (LEMON) database, sensor-space resting-state EEG, eyes-closed (`EC`) condition. Each EEG channel is a node, exactly as each fMRI ROI is a node — the entire training/eval/graph machinery is shared with the fMRI pipeline via `--dataset lemon_eeg`.

### Preprocessing pipeline

1. **`data/lemon_build_npy.py`** loads each subject's `sub-XXXXXX_EC.set` EEGLAB file once with MNE, splits it on boundary-annotation **onsets** into its boundary-free blocks (typically 8 one-minute EC blocks per subject), and writes, per block:
   ```
   <out_dir>/<subject>_<cond>_block{k:02d}.npy   # (T_block, N) float32
   <out_dir>/<subject>_<cond>.json                # sidecar: channel names, sfreq, ...
   ```
   Channel count is read from the data, not hardcoded — LEMON drops bad channels per subject, so N varies (58–61 out of a nominal 62-channel montage). With `--interpolate` (used for the paper), each subject's missing channels are instead spherical-spline interpolated up to the union montage, so every subject has the same channels.

2. **`data/lemon_make_manifest.py`** scans that directory and emits one manifest row per subject:
   ```
   subject_id, condition, split, n_blocks, block_paths (';'-joined),
   n_channels, ch_names (';'-joined), n_interpolated, sfreq, T_total, has_mri, in_overlap
   ```
   Rows with fewer than `--min_channels` (default 58) original channels are flagged. Split is assigned at the **subject** level, seeded/deterministic, so the test split is subject-disjoint (window-level CV in `main.py` re-mixes only the train+val pool). `has_mri` / `in_overlap` flag subjects with a matching structural-MRI (FreeSurfer) directory.

### Channel harmonization

Because per-subject channel counts vary, `data/lemon_dataset.py` reindexes every subject onto a **canonical montage**: the intersection of channel names common to all subjects in the manifest, sorted. Node *j* is then the same electrode across every subject.

### Sample

Same tensor contract as the fMRI pipeline — `{"x": (L_x, N), "y": (L_y, N), "meta": dict}`, N = channel count.

### Running

```bash
python main.py \
  --dataset lemon_eeg \
  --lemon_manifest_csv data/lemon_manifest.csv \
  --condition EC --max_subjects 202 --max_windows_per_subject 150 --lemon_data_seed 42 \
  --x 32 --y 8 --stride 8 --forecast_mode short \
  --graph_mode granger --granger_lag 1 --granger_threshold_mode topk --granger_topk_edges 305
```

Raw LEMON data is staged on the shared HPC filesystem; build `data/lemon_manifest.csv` locally with `data/lemon_make_manifest.py` (it is not tracked because it holds absolute paths).

The graph prior in this example is `--graph_mode granger`: a directed graph from per-subject Granger scores, averaged over the train fold. The paper's main results use `--graph_mode spatial` (nearest electrodes by scalp position).

See [`scripts/timeenc_eeg_short_spatial.sh`](../scripts/timeenc_eeg_short_spatial.sh) / [`scripts/timeenc_eeg_long_ar_spatial.sh`](../scripts/timeenc_eeg_long_ar_spatial.sh) for the paper's main configurations (the Granger example above matches [`scripts/timeenc_eeg_short.sh`](../scripts/timeenc_eeg_short.sh)).

---

## NEST (simulated)

A synthetic multi-neuron spiking dataset, generated with the [NEST](https://www.nest-simulator.org/) spiking-network simulator.

### Generation

`data/simulate_neuron_dataset.py` builds a neuron-connectivity graph, simulates each "subject" as an independent network realization, and bins spike output into a rate timeseries. Defaults:

| Parameter | Default | Flag |
|---|---|---|
| Nodes (neurons) | `100` | `--n-nodes` |
| Neuron graph | `small_world` (k=8, β=0.1 rewiring) | `--small-world-k`, `--small-world-beta` |
| Neuron model | `iaf_psc_alpha` | `--neuron-model` |
| Simulation length | `2000` ms | `--simulation-time-ms` |
| Solver resolution | `0.1` ms | `--resolution-ms` |
| Recurrent synapse weight | `10` pA | `--synapse-weight` |
| Input | Poisson, `1000` Hz, `50` pA | `--input-type`, `--poisson-rate-hz`, `--poisson-weight` |
| Intrinsic noise | `5` pA std | `--noise-std-pa` |
| Output bin size | `10` ms | `--bin-size-ms` |
| Subjects | `1000` | `--num-simulations` |
| Perturbation | `mute_bins` | `--perturbation-mode` |

The paper's NEST experiments use `--perturbation-mode silence_dc`, which silences one neuron inside the simulation. The default `mute_bins` only edits binned counts after the simulation and cannot be used with the `perturb_forecast` task.

**NEST is not importable from the main training `.venv`.** Generate the dataset from a separate environment:

```bash
uv venv ~/envs/nest && uv pip install --python ~/envs/nest/bin/python nest-simulator numpy scipy matplotlib tqdm
ACTIVATE_CMD="source ~/envs/nest/bin/activate" \
OUT_DIR=nest_simulated_neurons_silencedc EXTRA_ARGS="--perturbation-mode silence_dc" \
  sbatch scripts/generate_nest_dataset.sh
```

Output: `$OUT_DIR/dataset.npz` (default `data/simulated_neuron_dataset/`) + `run_config.json` (provenance and seed values) + `check/` (sanity-check plots).

### Sample

`dataset.npz` holds rate arrays shaped `[subjects, channels, time]` plus perturbation metadata, indexed by `data/sn_dataset.py`'s `SNDataset`. Three task modes:

- **`forecasting`** — unperturbed rates, context-only z-scoring.
- **`perturb_forecast`** — aligned perturbed futures (same windows), horizon normalized with the **original** (unperturbed) context's mean/std, for counterfactual evaluation. This is the mode used for the paper's NEST experiments.
- **`perturbation`** — full original and perturbed trajectories, both normalized with statistics from the original.

Splits: `train` / `val` / `test` (cross-subject) and `within`.

### Running

NEST is trained through a standalone entrypoint, `train/train_nest_braindyn.py`, not through `main.py`. The paper runs are launched by [`scripts/nest_f01_train.sh`](../scripts/nest_f01_train.sh):

```bash
HORIZON=short  sbatch --array=0-19 --time=00:40:00 scripts/nest_f01_train.sh
HORIZON=arlong sbatch --array=0-19 scripts/nest_f01_train.sh
```

`train/train_nest_braindyn.py` mirrors `main.py`'s model/ablation flags (`--hidden_dim`, `--no_sheaf`, `--ablation_no_lstm`, ...) and its subject-grouped shuffle-split scheme. Its graph prior is `--graph_mode structural` (used for the paper) or `granger`. The older `train/train_nest.py` is not used by any script.
