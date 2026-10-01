"""PyTorch Dataset and DataLoader utilities for bulk simulated neuron data.

Expects ``dataset.npz`` from ``simulate_neuron_dataset.py``: rate arrays shaped
``[subjects, channels, time]`` plus perturbation metadata. Forecasting uses
unperturbed rates; ``perturb_forecast`` adds aligned perturbed futures (same
windows, every tensor normalised with the **original context**'s mean/std so
the counterfactual difference is attributable to the perturbation rather than
to a change of units); ``perturbation`` mode returns full original and
perturbed trajectories normalised with statistics from the original.

Normalization follows the repo convention (see ``data/rbc_dataset.py``,
``data/lemon_dataset.py``): ``norm_mode="context"``
z-scores each window against its own **context** inside ``__getitem__``, while
``norm_mode="train_global"`` returns RAW windows and leaves per-channel scaling
to ``main.py::compute_train_global_stats -> batch_to_model_tensors``, which can
only run once the train split is known. Only the context may set the statistics:
using the whole run leaks the horizon's scale and offset back into the context
the model is allowed to see.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, cast
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_NPZ_PATH = os.path.join(ROOT_DIR, "data", "simulated_neuron_dataset", "dataset.npz")
SPLITS = Literal["train", "val", "test", "within"]
TASK_MODES = Literal["forecasting", "perturbation", "perturb_forecast"]
NormMode = Literal["context", "train_global"]
CROSS_SUBJECT_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})

# Perturbation modes that edit binned counts AFTER a single baseline run, so the
# intervention never propagates (see the guard in SNDataset.__init__). Mirrors
# data/simulate_neuron_dataset.py::POSTHOC_PERTURBATION_MODES, duplicated rather than
# imported because that module pulls in matplotlib and this one sits in the trainer's
# import path. Keep the two in agreement.
_POSTHOC_PERTURBATION_MODES: frozenset[str] = frozenset({"mute_bins", "scale_bins"})

# See _zscore_pair_from_reference for the reasoning behind both values; hoisted
# to module level so they read as one convention rather than two magic numbers.
_ZSCORE_STD_FLOOR = 1e-6
_ZSCORE_CLAMP = 20.0


def _read_run_config(npz_path: str) -> dict[str, Any]:
    path = os.path.join(os.path.dirname(npz_path), "run_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _partition_subjects(
    n: int, train_frac: float, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= train_frac <= 1.0 and 0.0 <= val_frac <= 1.0):
        raise ValueError("train_frac and val_frac must be in [0, 1].")
    if train_frac + val_frac > 1.0 + 1e-9:
        raise ValueError("train_frac + val_frac must not exceed 1.")
    if n < 1:
        raise ValueError("n must be at least 1.")

    perm = np.random.default_rng(seed).permutation(n)
    n_train = max(0, min(int(round(train_frac * n)), n))
    n_val = max(0, min(int(round(val_frac * n)), n - n_train))
    i1 = n_train + n_val
    return perm[:n_train], perm[n_train:i1], perm[i1:]


def _raw_pair(a: np.ndarray, b: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """RAW (unscaled) float32 tensors, for ``norm_mode="train_global"``.

    Per-channel mean/std are applied downstream by
    ``main.py::compute_train_global_stats -> batch_to_model_tensors`` once the
    train split is known -- identical contract to ``data/rbc_dataset.py``'s
    train_global branch. Normalizing here too would double-normalize, silently:
    the loss would still descend.
    """
    return (
        torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)),
    )


def _zscore_pair_from_reference(
    reference: np.ndarray, other: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = reference.mean(axis=0, keepdims=True)
    std = reference.std(axis=0, keepdims=True).clip(_ZSCORE_STD_FLOOR)
    a = (reference - mean) / std
    b = (other - mean) / std
    # A near-silent NEST channel can have its context-window std floored all
    # the way down to the 1e-6 clip above, at which point even a tiny
    # absolute deviation in `other` (a single extra/missing spike) blows up
    # to an enormous normalized magnitude -- easily overflowing fp16 under
    # --amp once squared in the loss. Real z-scores essentially
    # never exceed +-20 for a non-degenerate channel, so this only clips the
    # degenerate case. clip() alone would NOT be enough if reference/other
    # ever contains an actual NaN/Inf (e.g. from corrupted source data) --
    # clip leaves NaN untouched (NaN compares False against both bounds), so
    # nan_to_num runs first to turn any NaN/Inf into a finite number before
    # the clip bounds it into range.
    a = np.clip(np.nan_to_num(a), -_ZSCORE_CLAMP, _ZSCORE_CLAMP)
    b = np.clip(np.nan_to_num(b), -_ZSCORE_CLAMP, _ZSCORE_CLAMP)
    return (
        torch.from_numpy(a.astype(np.float32).copy()),
        torch.from_numpy(b.astype(np.float32).copy()),
    )


class SNDataset(Dataset):
    """Index over ``dataset.npz`` for forecasting windows, perturb-aligned forecasting windows, or full-sequence perturbation pairs."""

    def __init__(
        self,
        npz_path: str = DATA_NPZ_PATH,
        *,
        task_mode: TASK_MODES = "forecasting",
        split: SPLITS = "train",
        x: int = 90,
        y: int = 30,
        stride: int = 1,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        split_seed: int = 0,
        cache: bool = False,
        norm_mode: NormMode = "context",
        perturb_post_onset_frac: float | None = None,
        perturb_context_gap_bins: int = 0,
    ) -> None:
        """
        Args:
            x: Number of context time bins given to the model as input
                (`x[t : t + x]`) in forecasting mode.
            y: Number of future time bins predicted by the model
                (`y[t + x : t + x + y]`) in forecasting mode.
                In perturbation mode, full trajectories are returned and
                `x`/`y` are ignored. In ``perturb_forecast``, `x`/`y`/stride
                match forecasting; each sample also includes ``y_perturbed``.
            perturb_post_onset_frac: fraction of the ``perturb_forecast`` context
                that falls at or after the perturbation onset; negative values
                leave a gap of ``|frac|*x`` bins before the onset. ``None`` keeps
                the default ``x/10`` anchoring. When set, subjects whose window
                does not fit in the run are dropped rather than re-anchored.
            perturb_context_gap_bins: bins skipped between the end of the context
                and the start of the horizon.
        """
        if split not in CROSS_SUBJECT_SPLITS and split != "within":
            raise ValueError(f"invalid split: {split!r}")
        if task_mode not in ("forecasting", "perturbation", "perturb_forecast"):
            raise ValueError(f"invalid task_mode: {task_mode!r}")
        if norm_mode not in ("context", "train_global"):
            raise ValueError(
                f"norm_mode must be 'context' or 'train_global', got {norm_mode!r}"
            )
        if task_mode == "perturbation" and norm_mode == "train_global":
            raise ValueError(
                "task_mode='perturbation' returns full original/perturbed "
                "trajectories with no context/horizon split, so a train-fold "
                "*context* statistic is undefined for it; use norm_mode='context'."
            )

        path = os.path.abspath(os.fspath(npz_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        self.npz_path = path
        self.task_mode = task_mode
        self.split: SPLITS = split
        self.norm_mode: NormMode = norm_mode
        self.x = x
        self.y = y
        self.stride = stride
        if perturb_post_onset_frac is not None and not (
            -1.0 <= float(perturb_post_onset_frac) <= 1.0
        ):
            raise ValueError(
                "perturb_post_onset_frac must be in [-1, 1], got "
                f"{perturb_post_onset_frac!r}"
            )
        self.perturb_post_onset_frac = (
            None if perturb_post_onset_frac is None else float(perturb_post_onset_frac)
        )
        if int(perturb_context_gap_bins) < 0:
            raise ValueError(
                f"perturb_context_gap_bins must be >= 0, got {perturb_context_gap_bins!r}"
            )
        self.perturb_context_gap_bins = int(perturb_context_gap_bins)
        self._cache_subject_tc = cache
        self._tc_cache: dict[int, np.ndarray] = {}

        mmap: str | None = None if cache else "r"
        self._npz = np.load(path, mmap_mode=mmap, allow_pickle=True)
        self._rates_o = self._npz["smoothed_rates_hz_original"]
        self._rates_p = self._npz["smoothed_rates_hz_perturbed"]
        self._bin_size_ms = self._npz["bin_size_ms"]

        self.n_subjects, self.n_channels, self.n_bins = (int(self._rates_o.shape[i]) for i in range(3))

        self._pert_start = np.asarray(self._npz["perturbation_start_ms"], dtype=np.float64)
        self._pert_end = np.asarray(self._npz["perturbation_end_ms"], dtype=np.float64)
        self._pert_n = np.asarray(self._npz["perturbation_n_nodes"], dtype=np.int64)
        self._pert_nodes = np.asarray(self._npz["perturbation_nodes"], dtype=np.int64)
        self._graph_seeds = np.asarray(self._npz["graph_seeds"], dtype=np.int64)
        self._adjacency = np.asarray(self._npz["adjacency"], dtype=np.int64)

        run_cfg = _read_run_config(path)
        self._perturbation_mode: str | None = run_cfg.get("perturbation_mode")

        if (
            self.task_mode == "perturb_forecast"
            and self._perturbation_mode in _POSTHOC_PERTURBATION_MODES
        ):
            raise ValueError(
                f"task_mode='perturb_forecast' on a dataset generated with "
                f"perturbation_mode={self._perturbation_mode!r}: that mode masks binned "
                "counts after the fact, so the perturbation never propagated through the "
                "recurrent graph and the counterfactual is degenerate. Regenerate with "
                "`--perturbation-mode silence_dc` (see data/simulate_neuron_dataset.py). "
                f"Dataset: {path}"
            )

        train_i, val_i, test_i = _partition_subjects(
            self.n_subjects, train_frac, val_frac, split_seed
        )
        by_split = {
            "train": train_i.astype(np.int64, copy=False),
            "val": val_i.astype(np.int64, copy=False),
            "test": test_i.astype(np.int64, copy=False),
        }
        if split in CROSS_SUBJECT_SPLITS:
            self._subject_ids = by_split[split]
        else:
            self._subject_ids = np.arange(self.n_subjects, dtype=np.int64)

        self._index: list[tuple[int, ...]] = []
        # (subject_index, meta, t_start) protocol -- matches RBCDataset/
        # LemonDataset's `_samples`, which main.py::subject_groups_for and
        # compute_train_global_stats both require for grouped CV. Derived
        # from self._index right after it's built (see _build_sample_index).
        self._samples: list[tuple[int, dict[str, Any], int]] = []
        self._build_sample_index()

    def _build_sample_index(self) -> None:
        self._index.clear()
        if self.task_mode == "perturbation":
            self._index.extend((int(s),) for s in self._subject_ids.tolist())
            self._finalize_samples()
            return

        if self.task_mode not in ("forecasting", "perturb_forecast"):
            raise RuntimeError(f"unexpected task_mode for window index: {self.task_mode!r}")

        T = self.n_bins
        for s in self._subject_ids.tolist():
            if self.task_mode == "perturb_forecast":
                # Exactly one window per subject, anchored just before the perturbation
                # onset. The `x / 10` backset is deliberate and is NOT `x`: it puts the
                # onset a few bins into the context, so the model SEES the intervention
                # begin and the horizon lands on the network's response and recovery.
                # At the benchmark's x=32, y=8, bin=10ms that is a 3-bin backset, with
                # the 24-40 bin perturbation window spanning the rest of the context and
                # the horizon covering post-offset recovery. Changing this redefines the
                # task.
                pert_start = self._pert_start[s]
                if self.perturb_post_onset_frac is None:
                    t0 = max(0, int(pert_start / self._bin_size_ms - self.x / 10))
                    t0 = min(t0, T - self.x - self.y)
                else:
                    # `pre` context bins fall strictly before the onset.
                    pre = int(round(self.x * (1.0 - self.perturb_post_onset_frac)))
                    t0 = int(round(pert_start / self._bin_size_ms)) - pre
                    if t0 < 0 or t0 + self.x + self.perturb_context_gap_bins + self.y > T:
                        continue
                self._index.append((s, t0))

            else:
                if self.split == "within":
                    t0 = T - self.x - self.y
                    if t0 >= 0:
                        self._index.append((s, t0))
                else:
                    last_t = T - self.x - self.y
                    if last_t < 0:
                        continue
                    for t in range(0, last_t + 1, self.stride):
                        self._index.append((s, t))
        self._finalize_samples()

    def _finalize_samples(self) -> None:
        self._samples = [
            (
                int(entry[0]),
                {"subject_id": str(int(entry[0])), "subject_index": int(entry[0])},
                int(entry[1]) if len(entry) > 1 else 0,
            )
            for entry in self._index
        ]

    def _load_cached(self, subject_index: int) -> np.ndarray:
        """(T, N) raw (non-z-scored) original rate array for one subject --
        satisfies the _samples/_load_cached protocol main.py's
        compute_train_global_stats expects (see RBCDataset/LemonDataset).
        Returns raw data in either norm_mode, by design: under train_global
        this is the array the fold statistics are computed from, and under
        context it is the array __getitem__ z-scores per window."""
        return self._rates_orig_tc(int(subject_index))

    def _rates_orig_tc(self, subject: int) -> np.ndarray:
        if self._cache_subject_tc:
            hit = self._tc_cache.get(subject)
            if hit is not None:
                return hit
            tc = np.asarray(self._rates_o[subject], dtype=np.float32).T.copy()
            self._tc_cache[subject] = tc
            return tc
        return np.asarray(self._rates_o[subject], dtype=np.float32).T

    def _rates_pert_tc(self, subject: int) -> np.ndarray:
        return np.asarray(self._rates_p[subject], dtype=np.float32).T

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.task_mode == "forecasting":
            s, t = self._index[idx]
            ts = self._rates_orig_tc(s)
            ctx_o = ts[t : t + self.x]
            hrz = ts[t + self.x : t + self.x + self.y]
            if self.norm_mode == "context":
                x_t, y_t = _zscore_pair_from_reference(ctx_o, hrz)
            else:
                x_t, y_t = _raw_pair(ctx_o, hrz)
            meta = {
                "path": str(int(s)),
                # Same key _finalize_samples writes: build_granger_graph regroups
                # the loader by meta["subject_id"], so the batch meta has to carry
                # it too, not just the _samples metadata used for grouped CV.
                "subject_id": str(int(s)),
                "subject_index": int(s),
                "graph_seed": int(self._graph_seeds[s]),
                "t_start": int(t),
                "split": self.split,
                "T": self.n_bins,
                "n_channels": self.n_channels,
                "adjacency": torch.from_numpy(self._adjacency[s].copy()),
            }
            return {"x": x_t, "y": y_t, "meta": meta}

        if self.task_mode == "perturb_forecast":
            s, t = self._index[idx]
            ts_o = self._rates_orig_tc(s)
            ts_p = self._rates_pert_tc(s)
            ctx_o = ts_o[t : t + self.x]
            ctx_p = ts_p[t : t + self.x]
            _h0 = t + self.x + self.perturb_context_gap_bins
            hrz_o = ts_o[_h0 : _h0 + self.y]
            hrz_p = ts_p[_h0 : _h0 + self.y]
            if self.norm_mode == "context":
                # Reference is ctx_o -- the ORIGINAL context -- for all four
                # tensors, and both halves of that matter:
                #   * context, not the full run: only timepoints the model may
                #     observe can set the statistics. Referencing ts_o would
                #     fold the horizon's scale and offset back into the
                #     context, so the run's future would leak into the input.
                #   * original, not ctx_p: one shared scale is what makes
                #     (y_perturbed - y) attributable to the perturbation rather
                #     than to a change of units. Normalizing the perturbed pair
                #     against its own context would be causal but would destroy
                #     the counterfactual.
                x_t, y_t = _zscore_pair_from_reference(ctx_o, hrz_o)
                _, x_pert_t = _zscore_pair_from_reference(ctx_o, ctx_p)
                _, y_pert_t = _zscore_pair_from_reference(ctx_o, hrz_p)
            else:
                # train_global: one fold constant scales everything downstream,
                # so the shared-scale property above holds trivially.
                x_t, y_t = _raw_pair(ctx_o, hrz_o)
                x_pert_t, y_pert_t = _raw_pair(ctx_p, hrz_p)
            meta = {
                "path": str(int(s)),
                "subject_id": str(int(s)),
                "subject_index": int(s),
                "graph_seed": int(self._graph_seeds[s]),
                "t_start": int(t),
                "split": self.split,
                "T": self.n_bins,
                "n_channels": self.n_channels,
                "adjacency": torch.from_numpy(self._adjacency[s].copy()),
                "perturbed_node": int(self._pert_nodes[s, 0]),
            }
            return {"x": x_t, "y": y_t, "x_perturbed": x_pert_t, "y_perturbed": y_pert_t, "meta": meta}

        else:
            (s,) = self._index[idx]
            orig = self._rates_orig_tc(s)
            pert = self._rates_pert_tc(s)
            o_t, p_t = _zscore_pair_from_reference(orig, pert)
            kn = int(self._pert_n[s])
            nodes = torch.from_numpy(self._pert_nodes[s].astype(np.int64).copy())
            mode = self._perturbation_mode or ""
            meta = {
                "subject_id": str(int(s)),
                "subject_index": int(s),
                "graph_seed": int(self._graph_seeds[s]),
                "split": self.split,
                "T": self.n_bins,
                "n_channels": self.n_channels,
                "adjacency": torch.from_numpy(self._adjacency[s].copy()),
                "perturbation_start_ms": float(self._pert_start[s]),
                "perturbation_end_ms": float(self._pert_end[s]),
                "perturbation_mode": mode,
                "perturbed_n_nodes": torch.tensor(kn, dtype=torch.long),
                "perturbed_nodes": nodes,
            }
            return {"x_original": o_t, "x_perturbed": p_t, "meta": meta}

    def summary(self) -> str:
        return (
            f"SNDataset({self.task_mode=}, {self.split=}, {self.x=}, {self.y=}, "
            f"{self.stride=}, norm_mode={self.norm_mode!r}) "
            f"n={len(self)} pool={len(self._subject_ids)} shape=({len(self)},{self.n_bins},{self.n_channels})"
        )

    def close(self) -> None:
        if getattr(self, "_npz", None) is not None:
            self._npz.close()


def make_dataloaders(
    npz_path: str = DATA_NPZ_PATH,
    *,
    task_mode: TASK_MODES = "forecasting",
    x: int = 90,
    y: int = 30,
    stride: int = 1,
    batch_size: int = 32,
    num_workers: int = 2,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    split_seed: int = 0,
    cache: bool = False,
    pin_memory: bool = True,
    verbose: bool = False,
    norm_mode: NormMode = "context",
    perturb_post_onset_frac: float | None = None,
    perturb_context_gap_bins: int = 0,
) -> dict[str, DataLoader]:
    pin = pin_memory and torch.cuda.is_available()
    out: dict[str, DataLoader] = {}
    for sp in ("train", "val", "test"):
        split: SPLITS = cast(SPLITS, sp)
        ds = SNDataset(
            npz_path,
            task_mode=task_mode,
            split=split,
            x=x,
            y=y,
            stride=stride,
            perturb_post_onset_frac=perturb_post_onset_frac,
            perturb_context_gap_bins=perturb_context_gap_bins,
            train_frac=train_frac,
            val_frac=val_frac,
            split_seed=split_seed,
            cache=cache,
            norm_mode=norm_mode,
        )
        out[sp] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(sp == "train"),
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
        )
        if verbose:
            print(ds.summary())
    return out


def make_nest_run_loader(npz_path: str = DATA_NPZ_PATH):
    """Callable run_loader(subject_index) -> (T, N) raw original rate array,
    for SubjectRunDataset / --forecast_mode long_ar_train (mirrors
    data/lemon_dataset.py::make_lemon_run_loader's factory pattern -- the
    only difference is the "path" key is a subject index into the shared
    npz, not a per-subject file path)."""
    path = os.path.abspath(os.fspath(npz_path))
    npz = np.load(path, mmap_mode="r", allow_pickle=True)
    rates_o = npz["smoothed_rates_hz_original"]

    def run_loader(subject_index: int) -> np.ndarray:
        return np.asarray(rates_o[int(subject_index)], dtype=np.float32).T

    return run_loader


def make_nest_run_loader_perturbed(npz_path: str = DATA_NPZ_PATH):
    """Same as make_nest_run_loader, but over smoothed_rates_hz_perturbed --
    the full-length counterfactual trace paired with the original run.
    Stands in for run_loader in an autoregressive test rollout so the
    (normally-trained) model is evaluated on the perturbed trace as an
    alternate test set: perturbed context in, forecast, score against the
    perturbed continuation -- same treatment as SNDataset's perturb_forecast
    mode gives the direct (non-AR) models via x_perturbed/y_perturbed."""
    path = os.path.abspath(os.fspath(npz_path))
    npz = np.load(path, mmap_mode="r", allow_pickle=True)
    rates_p = npz["smoothed_rates_hz_perturbed"]

    def run_loader(subject_index: int) -> np.ndarray:
        return np.asarray(rates_p[int(subject_index)], dtype=np.float32).T

    return run_loader


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unit tests for SNDataset.")
    ap.add_argument("npz_path", nargs="?", default=str(DATA_NPZ_PATH))
    ap.add_argument("--x", type=int, default=90)
    ap.add_argument("--y", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-batches", type=int, default=2)
    args = ap.parse_args()

    for task_mode in ("forecasting", "perturb_forecast", "perturbation"):
        print(f"\nSanity checking task mode: {task_mode}...\n\n")

        loaders = make_dataloaders(
            args.npz_path,
            task_mode=task_mode,
            x=args.x,
            y=args.y,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            verbose=True,
        )

        for name, loader in loaders.items():
            n = len(loader.dataset)
            if n == 0:
                print(f"{name}: empty")
                continue
            t0 = time.perf_counter()
            for i, batch in enumerate(loader):
                if i == 0:
                    if task_mode in ("forecasting", "perturb_forecast"):
                        extra = ""
                        if task_mode == "perturb_forecast":
                            extra = f" x_perturbed={tuple(batch['x_perturbed'].shape)}"
                            extra = f" y_perturbed={tuple(batch['y_perturbed'].shape)}"
                        print(
                            f"{name}: x={tuple(batch['x'].shape)} y={tuple(batch['y'].shape)}{extra} "
                            f"subject={batch['meta']['subject_index']}"
                        )
                    else:
                        print(
                            f"{name}: x_original={tuple(batch['x_original'].shape)} "
                            f"{name}: x_perturbed={tuple(batch['x_perturbed'].shape)} "
                            f"metadata={batch['meta']}"
                        )
                if i + 1 >= args.n_batches:
                    break
            print(f"{name}: {args.n_batches} batches in {time.perf_counter() - t0:.2f}s")
