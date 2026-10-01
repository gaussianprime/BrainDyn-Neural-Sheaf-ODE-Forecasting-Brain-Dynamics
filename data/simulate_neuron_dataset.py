"""Generate simulated neuron-graph trajectories and save paired rate datasets.

Outputs (all under ``--out-dir``, default ``data/simulated_neuron_dataset/``):
- ``dataset.npz``
- ``run_config.json``
- Subject-0 check artifacts in ``check/``
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import nest
except ImportError:
    # NEST (the simulator) is only available on the cluster where this script
    # actually runs -- it's a compiled simulator provided via `module load nest`
    # or a conda env, not a plain `pip install`. Keeping the import guarded lets
    # the config dataclasses and pure helpers stay importable and testable
    # anywhere; setup_nest raises a clear error if a real simulation is attempted.
    nest = None

# dirname(dirname(...)) -- must resolve to the repo root, matching
# data/sn_dataset.py's own ROOT_DIR.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "data", "simulated_neuron_dataset")
CHECK_SUBDIR_NAME = "check"

# ---------------------------------------------------------------------------
# Config (neuron-graph run only)
# ---------------------------------------------------------------------------

_DEFAULT_IAF_NEURON_PARAMS: dict[str, Any] = {
    "C_m": 250.0,
    "tau_m": 20.0,
    "t_ref": 2.0,
    "E_L": -70.0,
    "V_reset": -70.0,
    "V_th": -55.0,
}


def _default_iaf_neuron_params() -> dict[str, Any]:
    return dict(_DEFAULT_IAF_NEURON_PARAMS)


@dataclass
class PerturbationSpec:
    mode: str
    nodes: list[int]
    start_ms: float
    end_ms: float
    extra_rate_hz: float = 120.0
    extra_weight: float | None = None
    scale: float = 1.0
    # silence_dc: hyperpolarizing current (pA, must be negative) injected into the
    # target for [start_ms, end_ms). None means "read NestConfig.silence_dc_amplitude_pa".
    silence_amplitude_pa: float | None = None


@dataclass
class GraphConfig:
    n_nodes: int = 100
    graph_rule: str = "small_world"
    dim: int = 2
    seed: int = 0
    k: int = 8
    radius: float = 0.25
    p_connect: float = 0.05
    small_world_k: int = 8
    small_world_beta: float = 0.1
    directed: bool = True
    allow_self_edges: bool = False


@dataclass
class NestConfig:
    """NEST wiring: neuron model, Poisson/dc drive, recurrent synapses, optional perturbation tuning."""

    neuron_model: str = "iaf_psc_alpha"
    neuron_params: dict[str, Any] = field(default_factory=_default_iaf_neuron_params)
    resolution_ms: float = 0.1
    local_num_threads: int = 1
    # NEST kernel RNG seed. None leaves NEST's built-in default in place, which is
    # what makes an unseeded run non-reproducible across NEST versions and makes
    # every subject reuse the same stochastic drive. simulate_bulk_neuron_dataset
    # always sets this per subject via derive_kernel_seed().
    rng_seed: Optional[int] = None
    simulation_time_ms: float = 2000.0
    # All weights below are currents in pA -- the native unit of iaf_psc_alpha's
    # syn_spec weight. There is deliberately no mV/PSP convention here.
    synapse_weight: float = 10.0
    synapse_delay_ms: float = 1.5
    use_graph_weights: bool = False
    input_type: str = "poisson"
    poisson_rate_hz: float = 1000.0
    poisson_weight: float = 50.0
    poisson_delay_ms: float = 1.0
    dc_amplitude_pa: float = 0.0  # only read when input_type == "dc"
    noise_mean_pa: float = 0.0
    noise_std_pa: float = 5.0  # 0 disables; >0 adds intrinsic current noise
    noise_dt_ms: float = 1.0
    record_to: str = "memory"
    perturbations: list[PerturbationSpec] = field(default_factory=list)
    extra_poisson_rate_hz: float = 1000.0
    extra_poisson_weight: float | None = None
    # Must not be 1.0: scale_bins multiplies counts by this, so 1.0 makes the
    # perturbed trajectory identical to the original. See
    # _assert_perturbation_is_effective.
    scale_bins_factor: float = 0.9
    # silence_dc: the hyperpolarizing current held on the target neuron for the
    # intervention window. Must be negative (see _assert_perturbation_is_effective).
    #
    # Calibrate DOWN, not up. iaf_psc_alpha leaves V_m unbounded below, so an
    # over-strong current drives V_m far past V_reset and the neuron then needs
    # tens of ms to climb back to threshold after the current stops -- an artificial
    # silence tail extending beyond the intervention window. Raising V_min is not an
    # escape: neuron_params apply to both runs of the pair and would break the
    # baseline's bit-reproducibility.
    silence_dc_amplitude_pa: float = -250.0


@dataclass
class ProcessingConfig:
    # counts_to_smoothed_rates_hz always converts to Hz (the npz keys are named
    # smoothed_rates_hz_*), so there is deliberately no normalize_rate_hz toggle
    # -- a False branch would make those key names lie.
    bin_size_ms: float = 10.0
    smoothing_sigma_ms: float = 20.0


@dataclass
class FullConfig:
    mode: str = "neuron"
    graph: GraphConfig = field(default_factory=GraphConfig)
    nest: NestConfig = field(default_factory=NestConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

NEST_RNG_SEED_MAX = 2**32 - 1

NEST_KERNEL_SEED_FORMULA = (
    "1 + SeedSequence([seed, subject_index]).generate_state(1, uint32)[0] % (2**32 - 1)"
)


def derive_kernel_seed(base_seed: int, subject_index: int) -> int:
    """Deterministic NEST kernel seed in ``[1, 2**32-1]`` for one subject.

    Derived from ``(base_seed, subject_index)`` through a SeedSequence so that:

    * it is a well-mixed independent draw per subject (a raw ``seed + s`` is not),
    * it is keyed *positionally*, so a subject's seed does not shift when
      ``--num-simulations`` changes, and
    * it can never be 0 or >= 2**32, both of which NEST rejects.

    Deliberately a separate stream from the ``SeedSequence(seed).spawn(n)`` used
    for perturbation-window sampling, so the stochastic drive and the
    perturbation window stay independently reproducible.
    """
    state = np.random.SeedSequence([int(base_seed), int(subject_index)]).generate_state(
        1, dtype=np.uint32
    )
    return 1 + int(state[0]) % NEST_RNG_SEED_MAX


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_provenance(repo_dir: str = ROOT_DIR) -> dict[str, Any]:
    """``{commit, short_commit, branch, dirty}``; all ``None`` outside a checkout.

    Never raises -- a missing git binary, a tarball export, or a detached/odd
    repo state must not take down a multi-hour simulation run.
    """
    out: dict[str, Any] = {"commit": None, "short_commit": None, "branch": None, "dirty": None}
    try:
        def _git(*args: str) -> str:
            return subprocess.run(
                ("git", "-C", repo_dir, *args),
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()

        out["commit"] = _git("rev-parse", "HEAD")
        out["short_commit"] = out["commit"][:12]
        out["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        out["dirty"] = bool(_git("status", "--porcelain"))
    except Exception:
        pass
    return out


def _nest_version() -> Optional[str]:
    if nest is None:
        return None
    version = getattr(nest, "__version__", None)
    if version is None:
        try:
            version = nest.version()  # older PyNEST
        except Exception:
            version = None
    return None if version is None else str(version)


def collect_provenance(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Everything needed to tell whether a rerun is comparable to this run.

    Importable and callable with ``nest is None`` -- ``nest_version`` is simply
    ``None`` in that case.
    """
    import scipy  # local: only needed for the version string

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_provenance(),
        "nest_version": _nest_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "command_line": list(argv) if argv is not None else list(sys.argv),
        "library_versions": {
            "numpy": getattr(np, "__version__", None),
            "scipy": getattr(scipy, "__version__", None),
            "matplotlib": getattr(matplotlib, "__version__", None),
        },
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _generate_positions(n_nodes: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n_nodes, dim))


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _symmetrize(adj: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adj = np.logical_or(adj, adj.T).astype(np.int8)
    weights = np.maximum(weights, weights.T)
    return adj, weights


def _build_knn_graph(cfg: GraphConfig, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = cfg.n_nodes
    distances = _pairwise_distances(positions)
    adj = np.zeros((n, n), dtype=np.int8)
    weights = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        order = np.argsort(distances[i])
        neighbors = [j for j in order if j != i][: cfg.k]
        for j in neighbors:
            adj[i, j] = 1
            weights[i, j] = 1.0 / (distances[i, j] + 1e-8)

    if not cfg.directed:
        adj, weights = _symmetrize(adj, weights)

    return adj, weights


def _build_distance_graph(cfg: GraphConfig, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = _pairwise_distances(positions)
    adj = (distances <= cfg.radius).astype(np.int8)

    if not cfg.allow_self_edges:
        np.fill_diagonal(adj, 0)

    weights = np.where(adj > 0, 1.0 / (distances + 1e-8), 0.0).astype(np.float32)

    if not cfg.directed:
        adj, weights = _symmetrize(adj, weights)

    return adj, weights


def _build_erdos_renyi_graph(cfg: GraphConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    adj = (rng.random((cfg.n_nodes, cfg.n_nodes)) < cfg.p_connect).astype(np.int8)

    if not cfg.allow_self_edges:
        np.fill_diagonal(adj, 0)

    if not cfg.directed:
        adj = np.logical_or(adj, adj.T).astype(np.int8)

    weights = adj.astype(np.float32)
    return adj, weights


def _build_small_world_graph(cfg: GraphConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_nodes
    k = cfg.small_world_k

    if k >= n:
        raise ValueError("small_world_k must be less than n_nodes.")
    if k % 2 != 0:
        raise ValueError("small_world_k must be even.")

    adj = np.zeros((n, n), dtype=np.int8)
    half_k = k // 2

    for i in range(n):
        for offset in range(1, half_k + 1):
            j = (i + offset) % n
            adj[i, j] = 1
            if not cfg.directed:
                adj[j, i] = 1

    # Watts–Strogatz: revisit each lattice edge-slot once with probability beta rewire endpoint.
    if cfg.directed:
        for i in range(n):
            for offset in range(1, half_k + 1):
                old_j = (i + offset) % n
                if rng.random() >= cfg.small_world_beta:
                    continue
                candidates = [x for x in range(n) if x != i and adj[i, x] == 0]
                if not candidates:
                    continue
                adj[i, old_j] = 0
                adj[i, int(rng.choice(candidates))] = 1
    else:
        for i in range(n):
            for offset in range(1, half_k + 1):
                old_j = (i + offset) % n
                if rng.random() >= cfg.small_world_beta:
                    continue
                if adj[i, old_j] == 0:
                    continue
                candidates = [
                    x
                    for x in range(n)
                    if x != i and adj[i, x] == 0 and adj[x, i] == 0
                ]
                if not candidates:
                    continue
                adj[i, old_j] = 0
                adj[old_j, i] = 0
                new_j = int(rng.choice(candidates))
                adj[i, new_j] = 1
                adj[new_j, i] = 1

    if not cfg.allow_self_edges:
        np.fill_diagonal(adj, 0)

    weights = adj.astype(np.float32)
    return adj, weights


def build_graph(cfg: GraphConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = _generate_positions(cfg.n_nodes, cfg.dim, cfg.seed)

    if cfg.graph_rule == "knn":
        adj, weights = _build_knn_graph(cfg, positions)
    elif cfg.graph_rule == "distance":
        adj, weights = _build_distance_graph(cfg, positions)
    elif cfg.graph_rule == "erdos_renyi":
        adj, weights = _build_erdos_renyi_graph(cfg)
    elif cfg.graph_rule == "small_world":
        adj, weights = _build_small_world_graph(cfg)
    else:
        raise ValueError(f"Unknown graph_rule: {cfg.graph_rule}")

    return adj, weights, positions


# ---------------------------------------------------------------------------
# NEST wiring (neuron graph)
# ---------------------------------------------------------------------------


def setup_nest(nest_cfg: NestConfig) -> None:
    if nest is None:
        raise RuntimeError(
            "The `nest` Python package failed to import (run `python -c 'import nest'` "
            "directly to see the underlying error). NEST is a compiled simulator, "
            "typically provided via `module load nest` or a conda env built against a "
            "real NEST install -- not a plain `pip install`. Point ACTIVATE_CMD in "
            "scripts/generate_nest_dataset.sh at an environment that has it."
        )
    nest.ResetKernel()
    # One atomic SetKernelStatus rather than three separate assignments: in NEST 3,
    # setting local_num_threads re-creates the per-virtual-process RNGs, so
    # `nest.rng_seed = ...` followed by `nest.local_num_threads = ...` silently
    # discards the seed. Order-independence here is the whole point.
    kernel: dict[str, Any] = {
        "resolution": nest_cfg.resolution_ms,
        "local_num_threads": nest_cfg.local_num_threads,
    }
    if nest_cfg.rng_seed is not None:
        s = int(nest_cfg.rng_seed)
        if not (1 <= s <= NEST_RNG_SEED_MAX):
            raise ValueError(f"NEST rng_seed must be in [1, {NEST_RNG_SEED_MAX}], got {s}")
        kernel["rng_seed"] = s
    nest.SetKernelStatus(kernel)


def create_neurons(n_nodes: int, nest_cfg: NestConfig):
    params = nest_cfg.neuron_params or {}
    return nest.Create(nest_cfg.neuron_model, n_nodes, params=params)


def connect_graph_edges(
    neurons, adjacency: np.ndarray, graph_weights: np.ndarray, nest_cfg: NestConfig
) -> None:
    """One ``nest.Connect`` per source neuron, all_to_all over its target set.

    Batching by source is valid because the weight depends only on the source
    (a scalar), or under ``use_graph_weights`` on the (source, target) pair,
    which all_to_all accepts as a ``(len(post), 1)`` array.

    Connections are created in ``np.where`` order (row-major, i.e. sorted by
    source then target). ``all_to_all`` is a deterministic rule that consumes no
    NEST randomness, so batching affects neither the RNG stream nor the
    connection set.
    """
    rows, cols = np.where(adjacency > 0)
    if rows.size == 0:
        return

    for src_idx in np.unique(rows):
        targets = cols[rows == src_idx]
        if targets.size == 0:
            continue

        if nest_cfg.use_graph_weights:
            weight: Any = (
                nest_cfg.synapse_weight
                * np.asarray(graph_weights[src_idx, targets], dtype=float)
            ).reshape(-1, 1)
        else:
            weight = nest_cfg.synapse_weight

        nest.Connect(
            neurons[int(src_idx)],
            neurons[[int(t) for t in targets]],
            syn_spec={"weight": weight, "delay": nest_cfg.synapse_delay_ms},
        )


def add_input_drive(neurons, nest_cfg: NestConfig):
    if nest_cfg.input_type == "none":
        return None

    if nest_cfg.input_type == "poisson":
        generator = nest.Create("poisson_generator", params={"rate": nest_cfg.poisson_rate_hz})
        nest.Connect(
            generator,
            neurons,
            syn_spec={"weight": nest_cfg.poisson_weight, "delay": nest_cfg.poisson_delay_ms},
        )
        return generator

    if nest_cfg.input_type == "dc":
        generator = nest.Create("dc_generator", params={"amplitude": nest_cfg.dc_amplitude_pa})
        nest.Connect(generator, neurons)
        return generator

    raise ValueError(f"Unknown input_type: {nest_cfg.input_type}")


def _nest_grid_dt_ms(dt_ms: float, resolution_ms: float) -> float:
    """NEST ``noise_generator`` ``dt`` must be a multiple of the simulation resolution."""
    res = float(resolution_ms)
    if res <= 0.0:
        return float(dt_ms)
    n = max(1, int(round(float(dt_ms) / res)))
    return n * res


def add_noise_current(neurons, nest_cfg: NestConfig) -> None:
    if nest_cfg.noise_std_pa <= 0.0:
        return
    dt = _nest_grid_dt_ms(float(nest_cfg.noise_dt_ms), float(nest_cfg.resolution_ms))
    gen = nest.Create(
        "noise_generator",
        params={
            "mean": float(nest_cfg.noise_mean_pa),
            "std": float(nest_cfg.noise_std_pa),
            "dt": dt,
        },
    )
    nest.Connect(gen, neurons)


def _nest_grid_time_ms(t_ms: float, resolution_ms: float) -> float:
    """NEST device times (e.g. poisson_generator start/stop) must fall on the resolution grid."""
    res = float(resolution_ms)
    if res <= 0.0:
        return float(t_ms)
    return float(round(t_ms / res) * res)


def add_perturbation_inputs(neurons, nest_cfg: NestConfig) -> None:
    """Attach in-simulation perturbation devices (post-hoc modes are no-ops here).

    Called LAST in nest_simulation_raw_counts, after the background Poisson drive and
    the noise generator. That ordering matters for silence_dc: the stochastic devices
    have already claimed their per-VP RNG streams by this point, and a dc_generator
    draws nothing, so the perturbed run stays bit-identical to the baseline until the
    current switches on. That is the property extra_poisson cannot offer -- its
    per-connection Poisson draws interleave into the same per-VP stream as the
    background drive, so its pair is matched only in distribution (see
    build_run_config's reproducibility_notes).
    """
    if not nest_cfg.perturbations:
        return

    res = float(nest_cfg.resolution_ms)
    for spec in nest_cfg.perturbations:
        if spec.mode not in ("extra_poisson", "silence_dc"):
            continue
        start = _nest_grid_time_ms(float(spec.start_ms), res)
        stop = _nest_grid_time_ms(float(spec.end_ms), res)
        if stop <= start:
            stop = start + res

        if spec.mode == "silence_dc":
            amp = float(
                spec.silence_amplitude_pa
                if spec.silence_amplitude_pa is not None
                else nest_cfg.silence_dc_amplitude_pa
            )
            for idx in spec.nodes:
                g = nest.Create(
                    "dc_generator",
                    params={"amplitude": amp, "start": start, "stop": stop},
                )
                # No syn_spec: a current generator injects into the neuron directly,
                # and the default static_synapse weight of 1.0 passes the amplitude
                # through unscaled. A delay here would shift the intervention off the
                # window the metadata records.
                nest.Connect(g, neurons[int(idx)])
            continue

        w = float(spec.extra_weight if spec.extra_weight is not None else nest_cfg.poisson_weight)
        for idx in spec.nodes:
            g = nest.Create(
                "poisson_generator",
                params={
                    "rate": float(spec.extra_rate_hz),
                    "start": start,
                    "stop": stop,
                },
            )
            nest.Connect(
                g,
                neurons[int(idx)],
                syn_spec={"weight": w, "delay": float(nest_cfg.poisson_delay_ms)},
            )


def attach_spike_recorder(neurons, nest_cfg: NestConfig):
    if nest_cfg.record_to == "none":
        return None

    sr = nest.Create("spike_recorder", params={"record_to": nest_cfg.record_to})
    nest.Connect(neurons, sr)
    return sr


def get_spike_events(spike_recorder) -> tuple[np.ndarray, np.ndarray]:
    if spike_recorder is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    events = spike_recorder.get("events")
    senders = np.asarray(events["senders"], dtype=np.int64)
    times = np.asarray(events["times"], dtype=np.float64)
    return senders, times


# ---------------------------------------------------------------------------
# Spike binning and rates
# ---------------------------------------------------------------------------


def bin_spikes(
    senders: np.ndarray,
    times_ms: np.ndarray,
    nest_ids: np.ndarray,
    simulation_time_ms: float,
    bin_size_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_units = len(nest_ids)
    n_bins = int(np.ceil(simulation_time_ms / bin_size_ms))
    bin_edges = np.arange(n_bins + 1, dtype=np.float64) * bin_size_ms

    counts = np.zeros((n_units, n_bins), dtype=np.float32)
    ids = np.asarray(nest_ids, dtype=np.int64)
    senders = np.asarray(senders, dtype=np.int64)

    bin_idx = np.floor(np.asarray(times_ms, dtype=np.float64) / bin_size_ms).astype(np.int64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)

    # Map NEST node IDs to unit indices with a lookup table. IDs are small and
    # contiguous because ResetKernel runs per subject, so the table is a few
    # hundred entries. -1 marks "not one of our neurons".
    if ids.size:
        lut = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
        lut[ids] = np.arange(ids.size, dtype=np.int64)
        in_range = (senders >= 0) & (senders < lut.size)
        unit_idx = np.where(in_range, lut[np.clip(senders, 0, lut.size - 1)], -1)
        valid &= unit_idx >= 0
        # Repeated additions of exactly 1.0 into float32 are exact below 2**24,
        # so np.add.at's accumulation order cannot change the counts.
        np.add.at(counts, (unit_idx[valid], bin_idx[valid]), 1.0)

    return counts, bin_edges


def _gaussian_kernel_1d(sigma_bins: float, truncate: float = 4.0) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0], dtype=np.float32)

    radius = int(truncate * sigma_bins + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma_bins * sigma_bins))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def smooth_counts(counts: np.ndarray, bin_size_ms: float, smoothing_sigma_ms: float) -> np.ndarray:
    sigma_bins = smoothing_sigma_ms / bin_size_ms
    kernel = _gaussian_kernel_1d(sigma_bins)

    smoothed = np.zeros_like(counts, dtype=np.float32)
    for i in range(counts.shape[0]):
        smoothed[i] = np.convolve(counts[i], kernel, mode="same")

    return smoothed


def counts_to_rate_hz(counts_or_smoothed: np.ndarray, bin_size_ms: float) -> np.ndarray:
    bin_size_s = bin_size_ms / 1000.0
    return counts_or_smoothed / bin_size_s


def counts_to_smoothed_rates_hz(counts: np.ndarray, processing: ProcessingConfig) -> np.ndarray:
    sm = smooth_counts(counts, processing.bin_size_ms, processing.smoothing_sigma_ms)
    return counts_to_rate_hz(sm, processing.bin_size_ms).astype(np.float32)


# ---------------------------------------------------------------------------
# Save output artifacts
# ---------------------------------------------------------------------------

def save_bulk_dataset_npz(arrays: dict[str, Any], out_dir: str = DEFAULT_OUT_DIR) -> str:
    """Write ``dataset.npz`` under ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dataset.npz")
    np.savez_compressed(path, **arrays)
    return path


def save_run_config_json(meta: dict[str, Any], out_dir: str = DEFAULT_OUT_DIR) -> str:
    path = os.path.join(out_dir, "run_config.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def _prepare_out_dir(out_dir: str, overwrite: bool) -> str:
    """Create ``out_dir``, refusing to clobber an existing ``dataset.npz``.

    Called *before* the simulation loop -- discovering the collision after an
    hour of NEST time is precisely the failure being prevented.
    """
    out_dir = os.path.abspath(out_dir)
    npz = os.path.join(out_dir, "dataset.npz")
    if os.path.exists(npz) and not overwrite:
        raise FileExistsError(
            f"{npz} already exists. Pass --overwrite to replace it, or "
            f"--out-dir <new dir> to write elsewhere."
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Perturbations (post-hoc mute/scale; extra_poisson handled in NEST)
# ---------------------------------------------------------------------------

PERTURBATION_MODES = frozenset({"extra_poisson", "silence_dc", "mute_bins", "scale_bins"})

# Modes that enter the simulation, so their effect propagates through the recurrent
# graph. The complement (mute_bins/scale_bins) only edits binned counts after a single
# baseline run, leaving every non-target neuron bit-identical -- usable as masking, but
# NOT as a causal intervention. data/sn_dataset.py refuses to pair those with
# task_mode="perturb_forecast" for exactly that reason.
IN_SIMULATION_PERTURBATION_MODES = frozenset({"extra_poisson", "silence_dc"})
POSTHOC_PERTURBATION_MODES = PERTURBATION_MODES - IN_SIMULATION_PERTURBATION_MODES


def sample_perturbation_spec(
    rng: np.random.Generator,
    n_nodes: int,
    simulation_time_ms: float,
    mode: str,
) -> PerturbationSpec:
    """One random neuron and a mid-run time window (duration ~12 to 20% of the simulation)."""
    if mode not in PERTURBATION_MODES:
        raise ValueError(f"Unknown mode {mode!r}")

    nodes = [int(rng.integers(0, n_nodes))]

    duration = float(simulation_time_ms * rng.uniform(0.12, 0.20))
    margin_lo = 0.10 * simulation_time_ms
    margin_hi = simulation_time_ms - 0.10 * simulation_time_ms - duration
    if margin_hi <= margin_lo:
        start_ms = 0.0
        end_ms = min(duration, simulation_time_ms)
    else:
        start_ms = float(rng.uniform(margin_lo, margin_hi))
        end_ms = float(start_ms + duration)
        end_ms = min(end_ms, simulation_time_ms)
        if end_ms <= start_ms:
            end_ms = min(simulation_time_ms, start_ms + max(1.0, duration * 0.5))

    return PerturbationSpec(mode=mode, nodes=nodes, start_ms=start_ms, end_ms=end_ms)


def _validate_perturbations(
    specs: Sequence[PerturbationSpec],
    n_nodes: int,
    simulation_time_ms: float,
) -> None:
    for spec in specs:
        if spec.mode not in PERTURBATION_MODES:
            raise ValueError(f"Unknown perturbation mode {spec.mode!r}; expected one of {PERTURBATION_MODES}")
        if spec.end_ms <= spec.start_ms:
            raise ValueError(f"Perturbation window invalid: start_ms={spec.start_ms} end_ms={spec.end_ms}")
        if spec.start_ms < 0 or spec.end_ms > simulation_time_ms:
            raise ValueError(
                f"Perturbation [{spec.start_ms}, {spec.end_ms}] ms outside [0, {simulation_time_ms}]"
            )
        for i in spec.nodes:
            if i < 0 or i >= n_nodes:
                raise ValueError(f"Perturbation node index {i} out of range for n_nodes={n_nodes}")
        if spec.mode == "extra_poisson" and spec.extra_rate_hz < 0:
            raise ValueError("extra_poisson requires extra_rate_hz >= 0")
        if spec.mode == "scale_bins" and spec.scale < 0:
            raise ValueError("scale_bins requires scale >= 0")
        if (
            spec.mode == "silence_dc"
            and spec.silence_amplitude_pa is not None
            and spec.silence_amplitude_pa >= 0
        ):
            raise ValueError(
                "silence_dc requires a negative (hyperpolarizing) silence_amplitude_pa; "
                f"got {spec.silence_amplitude_pa}"
            )


def _assert_perturbation_is_effective(mode: str, nest_cfg: NestConfig) -> None:
    """Reject configs where the perturbed trajectory equals the original by construction.

    A degenerate perturbation is not a crash -- it silently produces a dataset in
    which ``smoothed_rates_hz_perturbed == smoothed_rates_hz_original``
    bit-for-bit, making every counterfactual metric trivially perfect. Far cheaper
    to refuse up front than to discover it downstream.
    """
    if mode == "scale_bins" and float(nest_cfg.scale_bins_factor) == 1.0:
        raise ValueError(
            "scale_bins with scale_bins_factor=1.0 is a no-op: the perturbed rates would "
            "be identical to the original. Pass --scale-bins-factor != 1.0."
        )
    if mode == "extra_poisson" and float(nest_cfg.extra_poisson_rate_hz) <= 0.0:
        raise ValueError(
            "extra_poisson with extra_poisson_rate_hz <= 0 injects nothing. "
            "Pass --extra-poisson-rate-hz > 0."
        )
    if mode == "silence_dc" and float(nest_cfg.silence_dc_amplitude_pa) >= 0.0:
        raise ValueError(
            "silence_dc needs a NEGATIVE (hyperpolarizing) current to silence the "
            f"target; got silence_dc_amplitude_pa={nest_cfg.silence_dc_amplitude_pa}. "
            "A zero or depolarizing amplitude would leave the neuron firing (and a "
            "positive one would excite it, which is what --perturbation-mode "
            "extra_poisson is for). Pass --silence-dc-amplitude-pa < 0."
        )


def _bin_mask_for_interval(bin_edges_ms: np.ndarray, t0: float, t1: float) -> np.ndarray:
    a = bin_edges_ms[:-1]
    b = bin_edges_ms[1:]
    return (b > t0) & (a < t1)


def _apply_posthoc_perturbations_to_counts(
    counts: np.ndarray,
    bin_edges_ms: np.ndarray,
    specs: Sequence[PerturbationSpec],
) -> np.ndarray:
    out = np.array(counts, dtype=np.float32, copy=True)
    for spec in specs:
        # In-simulation modes already had their effect during nest.Simulate; touching
        # the counts again here would double-apply the intervention.
        if spec.mode in IN_SIMULATION_PERTURBATION_MODES:
            continue
        mask = _bin_mask_for_interval(bin_edges_ms, spec.start_ms, spec.end_ms)
        if not np.any(mask):
            continue
        for idx in spec.nodes:
            if spec.mode == "mute_bins":
                out[idx, mask] = 0.0
            elif spec.mode == "scale_bins":
                out[idx, mask] *= float(spec.scale)
            else:
                raise ValueError(f"Unknown post-hoc perturbation mode: {spec.mode}")
    return out


# ---------------------------------------------------------------------------
# Full NEST run → binned counts (post-hoc mute/scale not applied here)
# ---------------------------------------------------------------------------


def nest_simulation_raw_counts(cfg: FullConfig) -> dict[str, Any]:
    setup_nest(cfg.nest)

    adjacency, graph_weights, positions = build_graph(cfg.graph)
    _validate_perturbations(cfg.nest.perturbations, cfg.graph.n_nodes, cfg.nest.simulation_time_ms)

    neurons = create_neurons(cfg.graph.n_nodes, cfg.nest)
    connect_graph_edges(neurons, adjacency, graph_weights, cfg.nest)
    add_input_drive(neurons, cfg.nest)
    add_noise_current(neurons, cfg.nest)
    add_perturbation_inputs(neurons, cfg.nest)
    spike_recorder = attach_spike_recorder(neurons, cfg.nest)

    nest.Simulate(cfg.nest.simulation_time_ms)

    senders, times_ms = get_spike_events(spike_recorder)
    nest_ids = np.asarray(neurons.tolist(), dtype=np.int64)

    counts, bin_edges_ms = bin_spikes(
        senders=senders,
        times_ms=times_ms,
        nest_ids=nest_ids,
        simulation_time_ms=cfg.nest.simulation_time_ms,
        bin_size_ms=cfg.processing.bin_size_ms,
    )

    return {
        "adjacency": adjacency.astype(np.int8),
        "graph_weights": graph_weights.astype(np.float32),
        "positions": positions.astype(np.float32),
        "nest_ids": nest_ids,
        "spike_senders": senders,
        "spike_times_ms": times_ms,
        "bin_edges_ms": bin_edges_ms,
        "binned_counts": counts,
    }


def run_paired_smoothed_rates_hz(
    cfg: FullConfig, spec: PerturbationSpec, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return original/perturbed smoothed rates, shared bin edges, and adjacency.

    ``mute_bins`` and ``scale_bins`` apply perturbations to counts from one baseline run,
    so every non-target neuron is bit-identical between the two trajectories and nothing
    propagates. ``extra_poisson`` and ``silence_dc`` run baseline and perturbed
    simulations separately, with the intervention inside nest.Simulate, so the effect
    does propagate through the recurrent graph. Both runs of the pair share one kernel
    seed (see the main loop) -- for ``silence_dc`` that makes them bit-identical until
    the current switches on, since a dc_generator consumes no RNG.
    """
    if mode in POSTHOC_PERTURBATION_MODES:
        cfg0 = replace(cfg, nest=replace(cfg.nest, perturbations=[]))
        d = nest_simulation_raw_counts(cfg0)
        counts = d["binned_counts"]
        bin_edges_ms = d["bin_edges_ms"]
        rates_orig = counts_to_smoothed_rates_hz(counts, cfg.processing)
        counts_pert = _apply_posthoc_perturbations_to_counts(counts, bin_edges_ms, [spec])
        rates_pert = counts_to_smoothed_rates_hz(counts_pert, cfg.processing)
        adjacency = d["adjacency"]
        return rates_orig, rates_pert, bin_edges_ms, adjacency

    if mode in IN_SIMULATION_PERTURBATION_MODES:
        cfg0 = replace(cfg, nest=replace(cfg.nest, perturbations=[]))
        d0 = nest_simulation_raw_counts(cfg0)
        bin_edges_ms = d0["bin_edges_ms"]
        rates_orig = counts_to_smoothed_rates_hz(d0["binned_counts"], cfg.processing)
        adjacency = d0["adjacency"]

        cfg1 = replace(cfg, nest=replace(cfg.nest, perturbations=[spec]))
        d1 = nest_simulation_raw_counts(cfg1)
        rates_pert = counts_to_smoothed_rates_hz(d1["binned_counts"], cfg.processing)
        return rates_orig, rates_pert, bin_edges_ms, adjacency

    raise ValueError(f"Unknown perturbation mode {mode!r}")


# ---------------------------------------------------------------------------
# Sanity-check plots
# ---------------------------------------------------------------------------

def _load_dataset_arrays(output: str | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        if output.endswith(".npz"):
            return dict(np.load(output, allow_pickle=True))
        if os.path.isdir(output):
            npz = os.path.join(output, "dataset.npz")
            if not os.path.isfile(npz):
                raise FileNotFoundError(f"Expected {npz}")
            return dict(np.load(npz, allow_pickle=True))
    raise ValueError(f"Not a directory, .npz, or dict: {output!r}")


def _dataset_dir_for_config(output: str | dict[str, np.ndarray]) -> Optional[str]:
    if isinstance(output, dict):
        return None
    if isinstance(output, str) and os.path.isdir(output):
        return output
    if isinstance(output, str) and output.endswith(".npz"):
        return os.path.dirname(output) or None
    return None


def _graph_title_from_config(config_dir: Optional[str]) -> str:
    if config_dir is None:
        return "Graph structure"
    cfg_path = os.path.join(config_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return "Graph structure"
    with open(cfg_path, encoding="utf-8") as f:
        c = json.load(f)
    g = c.get("graph", {})
    rule = g.get("graph_rule", "?")
    n = g.get("n_nodes", "?")
    if rule == "knn":
        return f"Neuron graph ({rule}, k={g.get('k', '?')}, n={n})"
    return f"Neuron graph ({rule}, n={n})"


def _bin_centers_ms(bin_edges_ms: np.ndarray) -> np.ndarray:
    return 0.5 * (bin_edges_ms[:-1] + bin_edges_ms[1:])


def _central_time_slice(n_bins: int, center_fraction: float) -> slice:
    if n_bins < 1:
        return slice(0, 0)
    if center_fraction >= 1.0 or center_fraction <= 0.0:
        return slice(0, n_bins)
    trim = (1.0 - center_fraction) / 2.0
    a = int(round(n_bins * trim))
    b = n_bins - int(round(n_bins * trim))
    if b <= a:
        return slice(0, n_bins)
    return slice(a, b)


def write_check_original_vs_perturbed_panels(
    rates_orig: np.ndarray,
    rates_pert: np.ndarray,
    bin_edges_ms: np.ndarray,
    check_dir: str,
    n_plot_nodes: int = 5,
    seed: int = 0,
    center_fraction: float = 0.5,
    pin_nodes: Optional[Sequence[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 120,
    panels_name: str = "smoothed_sample_panels.png",
) -> str:
    """Solid = original, dashed = perturbed. If ``pin_nodes`` is set, those units are always plotted."""
    rates_orig = np.asarray(rates_orig, dtype=np.float64)
    rates_pert = np.asarray(rates_pert, dtype=np.float64)
    n_units, n_bins = rates_orig.shape
    if rates_pert.shape != (n_units, n_bins):
        raise ValueError("rates_orig and rates_pert must have the same shape")
    if n_plot_nodes > n_units:
        raise ValueError(f"n_plot_nodes ({n_plot_nodes}) > n_units ({n_units})")

    rng = np.random.default_rng(seed)
    k = min(n_plot_nodes, n_units)
    pinned: list[int] = []
    for x in pin_nodes or []:
        xi = int(x)
        if 0 <= xi < n_units:
            pinned.append(xi)
    pinned = list(dict.fromkeys(pinned))
    if len(pinned) >= k:
        node_indices = np.sort(np.asarray(pinned[:k], dtype=int))
    elif pinned:
        rest_pool = [i for i in range(n_units) if i not in pinned]
        need = k - len(pinned)
        if need > 0 and len(rest_pool) > 0:
            extra = rng.choice(rest_pool, size=min(need, len(rest_pool)), replace=False)
            node_indices = np.sort(np.concatenate([np.asarray(pinned, dtype=int), extra]))
        else:
            node_indices = np.sort(np.asarray(pinned, dtype=int))
    else:
        node_indices = np.sort(rng.choice(n_units, size=k, replace=False))

    t_full = _bin_centers_ms(np.asarray(bin_edges_ms, dtype=np.float64))
    if t_full.shape[0] != n_bins:
        raise ValueError(f"Time axis length {t_full.shape[0]} != number of rate bins {n_bins}")

    s = _central_time_slice(n_bins, center_fraction)
    t_ms = t_full[s]
    ro = rates_orig[node_indices, :][:, s]
    rp = rates_pert[node_indices, :][:, s]
    if t_ms.shape[0] < 1:
        raise ValueError("No time bins in selected range (empty slice).")

    if figsize is None:
        figsize = (10.0, 2.4 * n_plot_nodes + 0.3)

    fig, axs = plt.subplots(n_plot_nodes, 1, sharex=True, sharey=False, figsize=figsize)
    ax_list = [axs] if n_plot_nodes == 1 else [axs[i] for i in range(n_plot_nodes)]

    for i, (ax, idx) in enumerate(zip(ax_list, node_indices)):
        c = f"C{i % 10}"
        ax.plot(t_ms, ro[i], color=c, alpha=0.95, linewidth=1.2, linestyle="-", label="original")
        ax.plot(t_ms, rp[i], color=c, alpha=0.85, linewidth=1.2, linestyle="--", label="perturbed")
        ax.set_ylabel("Smoothed rate (Hz)", fontsize=9)
        ax.set_title(f"Node {int(idx)}", fontsize=10)
    ax_list[-1].set_xlabel("Time (ms)")
    ax_list[0].legend(loc="upper right", fontsize=8)
    win_desc = "full timeline" if center_fraction >= 0.999 else "middle of timeline"
    fig.suptitle(
        f"Smoothed rate — original (solid) vs perturbed (dashed), {win_desc}",
        y=0.999,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    os.makedirs(check_dir, exist_ok=True)
    out_path = os.path.join(check_dir, panels_name)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_voltage_sample_panels(
    vm_diag: dict[str, Any],
    check_dir: str,
    n_plot_exc: int = 3,
    n_plot_inh: int = 2,
    figsize: Tuple[float, float] = (10.0, 7.0),
    dpi: int = 120,
    panels_name: str = "voltage_sample_traces.png",
) -> str:
    """Plot sample ``V_m`` traces for excitatory (first indices) and inhibitory units."""
    t = np.asarray(vm_diag["voltage_times_ms"], dtype=np.float64)
    V = np.asarray(vm_diag["voltage_vm_mv"], dtype=np.float64)
    n_exc = int(vm_diag["n_exc"])
    n_inh = int(vm_diag["n_inh"])
    n_exc_plot = min(n_plot_exc, n_exc)
    n_inh_plot = min(n_plot_inh, n_inh)
    n_rows = n_exc_plot + n_inh_plot
    if n_rows < 1:
        raise ValueError("voltage_vm_mv has no rows to plot")

    fig, axs = plt.subplots(n_rows, 1, sharex=True, figsize=figsize)
    ax_list = [axs] if n_rows == 1 else list(axs)
    row = 0
    for i in range(n_exc_plot):
        ax_list[row].plot(t, V[i], color="C0", lw=0.8, alpha=0.9)
        ax_list[row].set_ylabel("mV", fontsize=8)
        ax_list[row].set_title(f"Excitatory index {i}", fontsize=9)
        row += 1
    for j in range(n_inh_plot):
        idx = n_exc + j
        ax_list[row].plot(t, V[idx], color="C3", lw=0.8, alpha=0.9)
        ax_list[row].set_ylabel("mV", fontsize=8)
        ax_list[row].set_title(f"Inhibitory index {idx}", fontsize=9)
        row += 1
    ax_list[-1].set_xlabel("Time (ms)")
    fig.suptitle("Membrane potential (baseline simulation)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(check_dir, exist_ok=True)
    out_path = os.path.join(check_dir, panels_name)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_graph_structure(
    output: str | dict[str, np.ndarray],
    max_edges_sample: int = 400,
    seed: Optional[int] = None,
    figsize: Tuple[float, float] = (11.0, 5.0),
    config_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    arrays = _load_dataset_arrays(output)
    if "adjacency" not in arrays or "positions" not in arrays:
        raise KeyError("adjacency and positions required")
    adj = np.asarray(arrays["adjacency"])
    pos = np.asarray(arrays["positions"], dtype=np.float64)
    if pos.shape[1] < 2:
        raise ValueError("positions must have at least 2 columns")
    pos = pos[:, :2]

    cfg_dir = config_dir if config_dir is not None else _dataset_dir_for_config(output)
    graph_directed = True
    if cfg_dir is not None:
        cfg_path = os.path.join(cfg_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                graph_directed = bool(json.load(f).get("graph", {}).get("directed", True))

    n = adj.shape[0]
    edges_rc = np.stack(np.where(adj > 0), axis=1)
    n_edges = len(edges_rc)

    rng = np.random.default_rng(seed)
    focal = int(rng.integers(0, n)) if n > 0 else 0

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    shown_l = min(max_edges_sample, n_edges)
    if shown_l > 0:
        pick = (
            np.arange(n_edges)
            if shown_l >= n_edges
            else rng.choice(n_edges, size=shown_l, replace=False)
        )
        for e in pick:
            i, j = int(edges_rc[e, 0]), int(edges_rc[e, 1])
            ax_l.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                color="0.45",
                alpha=0.22,
                linewidth=0.7,
                zorder=1,
            )
    ax_l.scatter(pos[:, 0], pos[:, 1], s=14, c="0.15", zorder=3, linewidths=0)
    ax_l.set_aspect("equal", adjustable="box")
    ax_l.set_xlabel("x")
    ax_l.set_ylabel("y")
    ax_l.set_title(
        f"Sample of edges ({shown_l} of {n_edges})",
        fontsize=10,
    )
    ax_l.grid(True, alpha=0.2)

    out_j = np.where(adj[focal] > 0)[0]
    in_i = np.where(adj[:, focal] > 0)[0]
    in_i = in_i[in_i != focal]

    ax_r.scatter(pos[:, 0], pos[:, 1], s=10, c="0.75", zorder=2, linewidths=0)
    ax_r.scatter(
        pos[focal, 0],
        pos[focal, 1],
        s=120,
        c="crimson",
        zorder=5,
        edgecolors="k",
        linewidths=0.6,
    )

    def draw_seg(i: int, j: int, color: str, lw: float, z: int, alpha: float) -> None:
        ax_r.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color=color,
            alpha=alpha,
            linewidth=lw,
            zorder=z,
        )

    if graph_directed:
        for j in out_j:
            draw_seg(focal, j, "#1f77b4", 1.4, 3, 0.75)
        for i in in_i:
            draw_seg(i, focal, "#ff7f0e", 1.4, 3, 0.75)
        ax_r.set_title(
            f"Node {focal}: out (blue) / in (orange)",
            fontsize=10,
        )
    else:
        for j in out_j:
            if j == focal:
                continue
            draw_seg(focal, j, "#2ca02c", 1.2, 3, 0.8)
        ax_r.set_title(
            f"Node {focal}: neighbors (undirected)",
            fontsize=10,
        )

    ax_r.set_aspect("equal", adjustable="box")
    ax_r.set_xlabel("x")
    ax_r.set_ylabel("y")
    ax_r.grid(True, alpha=0.2)

    title = _graph_title_from_config(cfg_dir)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()

    return fig, (ax_l, ax_r)


def _write_graph_structure_plot(
    output: str | dict[str, np.ndarray],
    out_path: str,
    max_edges_sample: int = 400,
    seed: Optional[int] = None,
    figsize: Tuple[float, float] = (11.0, 5.0),
    dpi: int = 120,
    config_dir: Optional[str] = None,
) -> str:
    fig, _ax = _plot_graph_structure(
        output,
        max_edges_sample=max_edges_sample,
        seed=seed,
        figsize=figsize,
        config_dir=config_dir,
    )
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# End-to-end run (many subjects / graphs in one invocation)
# ---------------------------------------------------------------------------

def _perturbation_spec_to_jsonable(spec: PerturbationSpec) -> dict[str, Any]:
    return {
        "mode": spec.mode,
        "nodes": list(spec.nodes),
        "start_ms": spec.start_ms,
        "end_ms": spec.end_ms,
        "extra_rate_hz": spec.extra_rate_hz,
        "extra_weight": spec.extra_weight,
        "scale": spec.scale,
        "silence_amplitude_pa": spec.silence_amplitude_pa,
    }


def _write_check_artifacts_subject_zero(
    base_cfg: FullConfig,
    check_dir: str,
    graph_seed: int,
    spec: PerturbationSpec,
    perturbation_mode: str,
    rates_orig: np.ndarray,
    rates_pert: np.ndarray,
    bin_edges_ms: np.ndarray,
) -> tuple[str, str]:
    n_nodes = base_cfg.graph.n_nodes
    os.makedirs(check_dir, exist_ok=True)
    graph_cfg = replace(base_cfg.graph, seed=graph_seed)
    adj, _gw, pos = build_graph(graph_cfg)
    check_config = {
        "graph": asdict(graph_cfg),
        "nest": asdict(base_cfg.nest),
        "processing": asdict(base_cfg.processing),
        "subject_index": 0,
        "graph_seed": int(graph_seed),
        "perturbation": _perturbation_spec_to_jsonable(spec),
        "perturbation_mode": perturbation_mode,
    }
    with open(os.path.join(check_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(check_config, f, indent=2)
    check_arrays = {
        "adjacency": adj.astype(np.int8),
        "positions": pos.astype(np.float32),
    }
    p_panels = write_check_original_vs_perturbed_panels(
        rates_orig,
        rates_pert,
        bin_edges_ms,
        check_dir,
        n_plot_nodes=min(5, n_nodes),
        seed=int(graph_seed),
        center_fraction=1.0,
        pin_nodes=spec.nodes,
    )
    p_graph = _write_graph_structure_plot(
        check_arrays,
        os.path.join(check_dir, "graph_structure.png"),
        seed=int(graph_seed),
        config_dir=check_dir,
    )
    return p_panels, p_graph


REQUIRED_NPZ_KEYS = frozenset({
    # Mirrors what data/sn_dataset.py::SNDataset.__init__ reads. Renaming any of
    # these silently breaks every NEST trainer; adding new keys is safe.
    "smoothed_rates_hz_original",
    "smoothed_rates_hz_perturbed",
    "bin_edges_ms",
    "bin_size_ms",
    "graph_seeds",
    "adjacency",
    "perturbation_start_ms",
    "perturbation_end_ms",
    "perturbation_n_nodes",
    "perturbation_nodes",
})


def build_run_config(
    base_cfg: FullConfig,
    *,
    num_simulations: int,
    perturbation_mode: str,
    seed: int,
    kernel_seeds: np.ndarray,
    npz_path: str,
    out_dir: str,
    check_dir: Optional[str],
    npz_keys: Sequence[str],
    argv: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Assemble ``run_config.json``.

    The first five keys are the historical schema and must stay put:
    ``data/sn_dataset.py`` reads top-level ``perturbation_mode`` from this file
    (found as a sibling of the npz) and silently degrades to ``""`` if it is
    missing or renamed.
    """
    return {
        "schema_version": 2,
        "perturbation_mode": perturbation_mode,
        "num_simulations": int(num_simulations),
        "seed": int(seed),
        "dataset_npz": npz_path,
        "base_config": base_cfg.to_dict(),
        "rng": {
            "base_seed": int(seed),
            "local_num_threads": int(base_cfg.nest.local_num_threads),
            "graph_seed_formula": "seed + subject_index",
            "nest_kernel_seed_formula": NEST_KERNEL_SEED_FORMULA,
            "perturbation_window_seed_formula":
                "SeedSequence(seed).spawn(num_simulations)[subject_index]",
            "nest_kernel_seeds_npz_key": "nest_kernel_seeds",
            "nest_kernel_seed_shared_across_paired_runs": True,
            "reproducibility_notes": [
                "Bit-identical replay requires the same NEST version AND the same "
                "local_num_threads: NEST partitions its RNG streams per virtual process.",
                "For extra_poisson, a matched kernel seed does not make the "
                "baseline/perturbed pair bit-matched -- the perturbation generator's "
                "per-connection Poisson draws interleave into the same per-VP stream as "
                "the background drive. The pair is matched only in distribution.",
                "For silence_dc the pair IS bit-matched up to the intervention onset: "
                "a dc_generator is deterministic and draws no RNG, and it is created "
                "after the stochastic devices have claimed their per-VP streams. Any "
                "difference between the trajectories before start_ms is therefore a "
                "bug, not noise.",
                "mute_bins and scale_bins edit binned counts after a single baseline "
                "run. Every non-target neuron stays bit-identical, so the perturbation "
                "does NOT propagate and the pair is not a causal counterfactual.",
            ],
        },
        "provenance": collect_provenance(argv),
        "outputs": {
            "out_dir": os.path.abspath(out_dir),
            "dataset_npz": npz_path,
            "run_config_json": os.path.join(os.path.abspath(out_dir), "run_config.json"),
            "check_dir": check_dir,
            "npz_keys": sorted(npz_keys),
        },
    }


def simulate_bulk_neuron_dataset(
    base_cfg: FullConfig,
    num_simulations: int,
    perturbation_mode: str,
    seed: int,
    *,
    out_dir: str = DEFAULT_OUT_DIR,
    check_dir: Optional[str] = None,
    overwrite: bool = False,
    write_check_artifacts: bool = True,
    argv: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Simulate many subjects and persist paired original/perturbed rate arrays."""
    if perturbation_mode not in PERTURBATION_MODES:
        raise ValueError(f"perturbation_mode must be one of {sorted(PERTURBATION_MODES)}")
    if num_simulations < 1:
        raise ValueError("num_simulations must be >= 1")
    _assert_perturbation_is_effective(perturbation_mode, base_cfg.nest)

    out_dir = _prepare_out_dir(out_dir, overwrite)
    if check_dir is None:
        check_dir = os.path.join(out_dir, CHECK_SUBDIR_NAME)

    n_nodes = base_cfg.graph.n_nodes
    n_bins = int(np.ceil(base_cfg.nest.simulation_time_ms / base_cfg.processing.bin_size_ms))

    rates_o = np.zeros((num_simulations, n_nodes, n_bins), dtype=np.float32)
    rates_p = np.zeros((num_simulations, n_nodes, n_bins), dtype=np.float32)
    graph_seeds = np.zeros(num_simulations, dtype=np.int64)
    kernel_seeds = np.zeros(num_simulations, dtype=np.int64)
    pert_start_ms = np.zeros(num_simulations, dtype=np.float64)
    pert_end_ms = np.zeros(num_simulations, dtype=np.float64)
    pert_n_nodes = np.zeros(num_simulations, dtype=np.int8)
    pert_nodes = np.full((num_simulations, 1), -1, dtype=np.int64)
    adjacency = np.zeros((num_simulations, n_nodes, n_nodes), dtype=np.int8)

    bin_edges_ms: np.ndarray | None = None
    check_panels_path: str | None = None
    check_graph_path: str | None = None

    seq = np.random.SeedSequence(seed)
    child_seeds = seq.spawn(num_simulations)

    for s in tqdm(range(num_simulations), desc="simulations", unit="subj"):
        graph_seed = int(seed + s)
        graph_seeds[s] = graph_seed
        graph_cfg = replace(base_cfg.graph, seed=graph_seed)
        # The kernel seed rides on the config rather than being passed as an
        # argument, so that run_paired_smoothed_rates_hz's two replace() copies
        # (baseline + perturbed, in extra_poisson mode) automatically share one
        # seed -- their difference must be attributable to the perturbation.
        kernel_seed = derive_kernel_seed(seed, s)
        kernel_seeds[s] = kernel_seed
        cfg = replace(
            base_cfg,
            graph=graph_cfg,
            nest=replace(base_cfg.nest, rng_seed=kernel_seed),
        )

        spec_rng = np.random.default_rng(child_seeds[s])
        spec = sample_perturbation_spec(
            spec_rng,
            n_nodes,
            cfg.nest.simulation_time_ms,
            perturbation_mode,
        )
        if perturbation_mode == "extra_poisson":
            spec = replace(spec, extra_rate_hz=float(cfg.nest.extra_poisson_rate_hz))
            if cfg.nest.extra_poisson_weight is not None:
                spec = replace(spec, extra_weight=float(cfg.nest.extra_poisson_weight))
        elif perturbation_mode == "silence_dc":
            spec = replace(spec, silence_amplitude_pa=float(cfg.nest.silence_dc_amplitude_pa))
        elif perturbation_mode == "scale_bins":
            spec = replace(spec, scale=float(cfg.nest.scale_bins_factor))
        _validate_perturbations([spec], n_nodes, cfg.nest.simulation_time_ms)

        ro, rp, edges, adj = run_paired_smoothed_rates_hz(cfg, spec, perturbation_mode)
        rates_o[s] = ro
        rates_p[s] = rp
        if bin_edges_ms is None:
            bin_edges_ms = np.asarray(edges, dtype=np.float64)

        pert_start_ms[s] = spec.start_ms
        pert_end_ms[s] = spec.end_ms
        kn = len(spec.nodes)
        pert_n_nodes[s] = kn
        pert_nodes[s, :kn] = np.asarray(spec.nodes, dtype=np.int64)
        adjacency[s] = adj

        if s == 0 and write_check_artifacts:
            be = bin_edges_ms
            assert be is not None
            # `cfg`, not `base_cfg`: the check config.json should record the
            # graph seed and kernel seed subject 0 actually ran with.
            check_panels_path, check_graph_path = _write_check_artifacts_subject_zero(
                cfg,
                check_dir,
                graph_seed,
                spec,
                perturbation_mode,
                ro,
                rp,
                be,
            )

    assert bin_edges_ms is not None

    arrays: dict[str, Any] = {
        "smoothed_rates_hz_original": rates_o,
        "smoothed_rates_hz_perturbed": rates_p,
        "bin_edges_ms": bin_edges_ms.astype(np.float64),
        "graph_seeds": graph_seeds,
        # Additive key -- SNDataset reads by explicit name, so this is inert to
        # every existing reader. Enables single-subject replay.
        "nest_kernel_seeds": kernel_seeds,
        "adjacency": adjacency.astype(np.int8),
        "bin_size_ms": base_cfg.processing.bin_size_ms,
        "perturbation_start_ms": pert_start_ms,
        "perturbation_end_ms": pert_end_ms,
        "perturbation_n_nodes": pert_n_nodes,
        "perturbation_nodes": pert_nodes,
    }
    missing = REQUIRED_NPZ_KEYS - set(arrays)
    if missing:
        raise RuntimeError(f"dataset.npz is missing keys SNDataset requires: {sorted(missing)}")

    npz_path = save_bulk_dataset_npz(arrays, out_dir)

    run_meta = build_run_config(
        base_cfg,
        num_simulations=num_simulations,
        perturbation_mode=perturbation_mode,
        seed=seed,
        kernel_seeds=kernel_seeds,
        npz_path=npz_path,
        out_dir=out_dir,
        check_dir=check_dir if write_check_artifacts else None,
        npz_keys=list(arrays),
        argv=argv,
    )
    save_run_config_json(run_meta, out_dir)

    return {
        "dataset_npz": npz_path,
        "run_config_json": os.path.join(out_dir, "run_config.json"),
        "out_dir": out_dir,
        "check_dir": check_dir if write_check_artifacts else None,
        "smoothed_sample_panels": check_panels_path,
        "graph_structure": check_graph_path,
        "num_simulations": num_simulations,
        "nest_kernel_seeds": kernel_seeds,
    }


def _full_config_from_args(args: argparse.Namespace) -> FullConfig:
    graph = GraphConfig(
        n_nodes=args.n_nodes,
        graph_rule=args.graph_rule,
        dim=args.dim,
        seed=args.seed,
        k=args.k,
        radius=args.radius,
        p_connect=args.p_connect,
        small_world_k=args.small_world_k,
        small_world_beta=args.small_world_beta,
        directed=not args.undirected,
        allow_self_edges=args.allow_self_edges,
    )
    nest_cfg = NestConfig(
        neuron_model=args.neuron_model,
        resolution_ms=args.resolution_ms,
        local_num_threads=args.threads,
        simulation_time_ms=args.simulation_time_ms,
        synapse_weight=args.synapse_weight,
        synapse_delay_ms=args.synapse_delay_ms,
        use_graph_weights=args.use_graph_weights,
        input_type=args.input_type,
        poisson_rate_hz=args.poisson_rate_hz,
        poisson_weight=args.poisson_weight,
        poisson_delay_ms=args.poisson_delay_ms,
        dc_amplitude_pa=args.dc_amplitude_pa,
        noise_mean_pa=args.noise_mean_pa,
        noise_std_pa=args.noise_std_pa,
        noise_dt_ms=args.noise_dt_ms,
        record_to=args.record_to,
        perturbations=[],
        extra_poisson_rate_hz=args.extra_poisson_rate_hz,
        extra_poisson_weight=args.extra_poisson_weight,
        scale_bins_factor=args.scale_bins_factor,
        silence_dc_amplitude_pa=args.silence_dc_amplitude_pa,
    )
    processing = ProcessingConfig(
        bin_size_ms=args.bin_size_ms,
        smoothing_sigma_ms=args.smoothing_sigma_ms,
    )
    return FullConfig(mode="neuron", graph=graph, nest=nest_cfg, processing=processing)


# Every CLI default below is read off these instances rather than repeated as a
# literal, so the CLI and the dataclass defaults cannot drift apart.
_G_DEFAULTS = GraphConfig()
_N_DEFAULTS = NestConfig()
_P_DEFAULTS = ProcessingConfig()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a NEST dataset where each graph node is one neuron.")

    p.add_argument("--n-nodes", type=int, default=_G_DEFAULTS.n_nodes)
    p.add_argument("--graph-rule", type=str, default=_G_DEFAULTS.graph_rule, choices=["knn", "distance", "erdos_renyi", "small_world"])
    p.add_argument("--dim", type=int, default=_G_DEFAULTS.dim)
    p.add_argument("--seed", type=int, default=_G_DEFAULTS.seed, help="Base seed: subject s uses graph seed (seed + s), NEST kernel seed derive_kernel_seed(seed, s), and a spawned stream for the perturbation window.")
    p.add_argument("--k", type=int, default=_G_DEFAULTS.k)
    p.add_argument("--radius", type=float, default=_G_DEFAULTS.radius)
    p.add_argument("--p-connect", type=float, default=_G_DEFAULTS.p_connect)
    p.add_argument("--small-world-k", type=int, default=_G_DEFAULTS.small_world_k)
    p.add_argument("--small-world-beta", type=float, default=_G_DEFAULTS.small_world_beta)
    p.add_argument("--undirected", action="store_true")
    p.add_argument("--allow-self-edges", action="store_true")

    p.add_argument("--neuron-model", type=str, default=_N_DEFAULTS.neuron_model)
    p.add_argument("--resolution-ms", type=float, default=_N_DEFAULTS.resolution_ms)
    p.add_argument("--threads", type=int, default=_N_DEFAULTS.local_num_threads, help="NEST local_num_threads. Bit-identical replay requires the same value: NEST partitions its RNG streams per virtual process.")
    p.add_argument("--simulation-time-ms", type=float, default=_N_DEFAULTS.simulation_time_ms)
    p.add_argument("--use-graph-weights", action="store_true")

    p.add_argument("--input-type", type=str, default=_N_DEFAULTS.input_type, choices=["poisson", "dc", "none"])
    p.add_argument("--poisson-rate-hz", type=float, default=_N_DEFAULTS.poisson_rate_hz, help="shared Poisson rate (Hz) onto each neuron; lower can silence output")
    p.add_argument("--poisson-weight", type=float, default=_N_DEFAULTS.poisson_weight, help="pA")
    p.add_argument("--synapse-weight", type=float, default=_N_DEFAULTS.synapse_weight, help="recurrent synapse weight (pA)")
    p.add_argument("--poisson-delay-ms", type=float, default=_N_DEFAULTS.poisson_delay_ms)
    p.add_argument("--synapse-delay-ms", type=float, default=_N_DEFAULTS.synapse_delay_ms)
    p.add_argument("--dc-amplitude-pa", type=float, default=_N_DEFAULTS.dc_amplitude_pa, help="only used if --input-type dc")
    p.add_argument("--noise-mean-pa", type=float, default=_N_DEFAULTS.noise_mean_pa, help="noise_generator mean current (pA); only used if --noise-std-pa > 0")
    p.add_argument("--noise-std-pa", type=float, default=_N_DEFAULTS.noise_std_pa, help="noise_generator std (pA); 0 disables; >0 adds intrinsic current noise (default: %(default)s)")
    p.add_argument("--noise-dt-ms", type=float, default=_N_DEFAULTS.noise_dt_ms, help="noise_generator update interval (ms); snapped to a multiple of --resolution-ms.")

    p.add_argument("--record-to", type=str, default=_N_DEFAULTS.record_to, choices=["memory", "ascii", "none"])
    p.add_argument("--bin-size-ms", type=float, default=_P_DEFAULTS.bin_size_ms)
    p.add_argument("--smoothing-sigma-ms", type=float, default=_P_DEFAULTS.smoothing_sigma_ms)

    p.add_argument("--num-simulations", type=int, default=1000)
    p.add_argument("--perturbation-mode", type=str, default="mute_bins", choices=sorted(PERTURBATION_MODES))
    p.add_argument(
        "--silence-dc-amplitude-pa",
        type=float,
        default=_N_DEFAULTS.silence_dc_amplitude_pa,
        help="silence_dc hyperpolarizing current (pA); must be < 0. Use the smallest "
        "magnitude that yields zero spikes in the window -- an over-strong current "
        "leaves the target recovering long after the intervention ends.",
    )
    p.add_argument("--extra-poisson-rate-hz", type=float, default=_N_DEFAULTS.extra_poisson_rate_hz)
    p.add_argument("--extra-poisson-weight", type=float, default=_N_DEFAULTS.extra_poisson_weight)
    p.add_argument("--scale-bins-factor", type=float, default=_N_DEFAULTS.scale_bins_factor, help="scale_bins multiplier; must not be 1.0 (that is a no-op).")

    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory for dataset.npz + run_config.json + check/ (default: %(default)s).")
    p.add_argument("--check-dir", type=str, default=None, help="Override the subject-0 check artifact dir (default: <out-dir>/check).")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing dataset.npz in --out-dir.")
    p.add_argument("--no-check-artifacts", action="store_true", help="Skip the subject-0 PNG/config.json artifacts.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = _full_config_from_args(args)
    try:
        result = simulate_bulk_neuron_dataset(
            cfg,
            num_simulations=args.num_simulations,
            perturbation_mode=args.perturbation_mode,
            seed=args.seed,
            out_dir=args.out_dir,
            check_dir=args.check_dir,
            overwrite=args.overwrite,
            write_check_artifacts=not args.no_check_artifacts,
            argv=list(sys.argv) if argv is None else list(argv),
        )
    except FileExistsError as e:
        # Clean one-line failure for sbatch rather than a traceback.
        raise SystemExit(str(e))
    print(f"Saved dataset.npz to:     {result['dataset_npz']}")
    print(f"Saved run_config.json to: {result['run_config_json']}")
    if result["check_dir"] is None:
        print("Subject-0 check artifacts: skipped (--no-check-artifacts)")
    else:
        print(f"Subject-0 check artifacts: {result['check_dir']}")
        print(f"  panels: {result['smoothed_sample_panels']}")
        print(f"  graph:  {result['graph_structure']}")


if __name__ == "__main__":
    main()
