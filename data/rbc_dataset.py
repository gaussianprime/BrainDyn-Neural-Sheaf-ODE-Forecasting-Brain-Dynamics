"""rbc_dataset.py
================
PyTorch Dataset and DataLoader factory for the RBC resting-state fMRI
dataset (PNC + HBN), using Schaefer-400 / 36-Parameter parcellated
timeseries produced by the CPAC pipeline.

Overview
--------
Each sample is a (context, horizon) pair:

    x  : Tensor[L_x, 400]   – context window given to the model
    y  : Tensor[L_y, 400]   – horizon the model must predict
    meta : dict              – subject / cohort metadata (no tensors)

Normalisation
~~~~~~~~~~~~~
The dataset returns raw windows. The training entrypoint computes statistics
from each fold's training contexts and applies them to train, validation, and
test data.

Split modes
~~~~~~~~~~~
"train" / "val" / "test"
    Sliding windows with configurable stride.  Only windows whose
    horizon falls entirely within [0, T - y) are included, so the
    final y timepoints are never part of any training/eval horizon.

"within"
    One sample per run for every subject (regardless of between-subject
    partition).  Context = [T - x - y : T - y], horizon = [T - y : T].
    Used for within-subject held-out evaluation.

Quick start
-----------
    from rbc_dataset import make_dataloaders

    loaders = make_dataloaders(
        manifest_csv = "manifest.csv",
        x            = 90,   # context TRs
        y            = 30,   # horizon TRs
        stride       = 10,
        batch_size   = 32,
        num_workers  = 2,
    )
    for batch in loaders["train"]:
        ctx  = batch["x"]     # (B, 90, 400)
        hrz  = batch["y"]     # (B, 30, 400)
        meta = batch["meta"]  # dict of lists (strings / ints)

The manifest is built upstream; the builder is not part of this repo.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Split = Literal["train", "val", "test", "within"]
NormMode = Literal["train_global"]
_BETWEEN_SPLITS = frozenset({"train", "val", "test"})

_DEFAULT_SCHAEFER_CENTROIDS_JSON = (
    Path(__file__).parent / "atlases" / "schaefer400_7networks" / "centroids.json"
)

# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------


def _load_1d(path: str | Path) -> np.ndarray:
    """Load an AFNI .1D file into a float32 array of shape (T, 400).

    The first line is a comment header starting with '#'; numpy's
    ``comments`` parameter strips it automatically.
    """
    return np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)


def schaefer_centroids(
    positions_json: str | Path | None = None,
    expected_nodes: int = 400,
) -> np.ndarray:
    """Schaefer-400 parcel centroid coordinates (MNI mm), in canonical parcel
    order -> ``(N, 3)``.

    Used to build the spatial (parcel-proximity) prior graph. Positions come
    from a precomputed ``centroids.json`` (see
    ``scripts/build_schaefer_centroids.py``, run once) so training needs no
    nilearn atlas download -- mirrors how ``lemon_dataset.montage_positions``
    reads a cached ``montage_positions.json`` instead of importing MNE.
    Defaults to ``data/atlases/schaefer400_7networks/centroids.json``.
    """
    p = Path(positions_json) if positions_json is not None else _DEFAULT_SCHAEFER_CENTROIDS_JSON
    if not p.exists():
        raise FileNotFoundError(
            f"Schaefer centroid file not found: {p}. Run "
            "scripts/build_schaefer_centroids.py once to generate it."
        )
    payload = json.loads(p.read_text())
    centroids = np.asarray(payload["centroids"], dtype=np.float64)
    if centroids.shape != (expected_nodes, 3):
        raise ValueError(
            f"{p} holds centroids shaped {tuple(centroids.shape)}, expected "
            f"({expected_nodes}, 3)."
        )
    return centroids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RBCDataset(Dataset):
    """Sliding-window dataset over RBC (PNC + HBN) fMRI parcellated timeseries.

    Parameters
    ----------
    manifest_csv : str | Path
        Path to the fMRI manifest CSV.
    split : "train" | "val" | "test" | "within"
        Determines which rows from the manifest are used and how windows
        are generated (see module docstring).
    x : int
        Context length in TRs (timepoints given to the model).
    y : int
        Horizon length in TRs (timepoints the model predicts).
    stride : int
        Window stride for "train" / "val" / "test" splits.
        Ignored for "within".
    cohort : str | None
        If given, restrict to "PNC" or "HBN".
    min_t : int
        Skip runs with fewer than this many timepoints.  Should match
        the value used when building the manifest.
    cache : bool
        If True, timeseries arrays are held in RAM after first load.
        Useful when the dataset fits in memory and num_workers > 0
        (each worker gets its own copy).
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: Split,
        x: int = 90,
        y: int = 30,
        stride: int = 1,
        cohort: Optional[str] = None,
        min_t: int = 0,
        cache: bool = True,
        norm_mode: NormMode = "train_global",
    ) -> None:
        if split not in (_BETWEEN_SPLITS | {"within"}):
            raise ValueError(
                f"split must be one of 'train','val','test','within'; got {split!r}"
            )
        if norm_mode != "train_global":
            raise ValueError("only fold-training-global normalization is supported")
        self.x      = x
        self.y      = y
        self.stride = stride
        self.split  = split
        self.cache  = cache
        self.norm_mode = norm_mode
        self._cache: Dict[str, np.ndarray] = {}

        rows = self._read_manifest(manifest_csv)

        # --- filter by cohort and by between-subject partition --------------
        if cohort is not None:
            rows = [r for r in rows if r["cohort"] == cohort]
        if split in _BETWEEN_SPLITS:
            rows = [r for r in rows if r["split"] == split]
        if min_t > 0:
            rows = [r for r in rows if int(r["T"]) >= min_t]

        # --- build flat sample index ----------------------------------------
        # Each entry: (path, metadata_dict, window_start_t)
        self._samples: List[tuple[str, dict, int]] = []
        for r in rows:
            T    = int(r["T"])
            meta = {
                "cohort":     r["cohort"],
                "subject_id": r["subject_id"],
                "session":    r["session"],
                "run":        r["run"],
                "site":       r["site"],
                "between_split": r["split"],
                "T":          T,
                "path":       r["path"],
            }

            if split == "within":
                # context = [T-x-y : T-y],  horizon = [T-y : T]
                t0 = T - x - y
                if t0 >= 0:
                    self._samples.append((r["path"], meta, t0))
            else:
                # Sliding windows; horizon must end strictly before T-y
                # so the last y TRs are never seen as a training target.
                # Constraint: t + x + y <= T - y  →  t <= T - x - 2*y
                max_t = T - x - 2 * y
                if max_t < 0:
                    continue
                for t in range(0, max_t + 1, stride):
                    self._samples.append((r["path"], meta, t))

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        path, meta, t = self._samples[idx]
        ts = self._load_cached(path)          # (T, 400)

        ctx_np = ts[t       : t + self.x]     # (x, 400)
        hrz_np = ts[t+self.x : t+self.x+self.y]  # (y, 400)

        ctx = torch.from_numpy(ctx_np.copy())
        hrz = torch.from_numpy(hrz_np.copy())

        meta_out = dict(meta)
        meta_out["t_start"] = int(t)
        return {"x": ctx, "y": hrz, "meta": meta_out}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_manifest(path: str | Path) -> List[dict]:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    def _load_cached(self, path: str) -> np.ndarray:
        if self.cache:
            if path not in self._cache:
                self._cache[path] = _load_1d(path)
            return self._cache[path]
        return _load_1d(path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_runs = len({s[0] for s in self._samples})
        return (
            f"RBCDataset(split={self.split!r}, x={self.x}, y={self.y}, "
            f"stride={self.stride}) | "
            f"{len(self)} windows from {n_runs} runs"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def make_dataloaders(
    manifest_csv: str | Path,
    x: int = 90,
    y: int = 30,
    stride: int = 1,
    batch_size: int = 32,
    num_workers: int = 2,
    cohort: Optional[str] = None,
    min_t: int = 0,
    cache: bool = False,
    pin_memory: bool = True,
    norm_mode: NormMode = "train_global",
) -> Dict[str, DataLoader]:
    """Return a dict of DataLoaders for all four split modes.

    Keys: "train", "val", "test", "within".

    Parameters
    ----------
    manifest_csv : str | Path
        Path to the fMRI manifest CSV.
    x, y, stride : int
        Window parameters (see RBCDataset).
    batch_size : int
        Samples per mini-batch.
    num_workers : int
        Parallel data-loading workers.  Start with 0–2 on HPC GPFS and
        increase if I/O is the bottleneck.
    cohort : str | None
        Restrict to "PNC" or "HBN"; None uses both.
    min_t : int
        Minimum run length (passed to RBCDataset).
    cache : bool
        Cache timeseries in RAM per worker (see RBCDataset).
    pin_memory : bool
        Enable pinned memory transfer when a CUDA GPU is available.
    """
    use_pin = pin_memory and torch.cuda.is_available()

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test", "within"):
        ds = RBCDataset(
            manifest_csv, split=split,          # type: ignore[arg-type]
            x=x, y=y, stride=stride,
            cohort=cohort, min_t=min_t, cache=cache, norm_mode=norm_mode,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=(num_workers > 0),
            # Collate dicts: tensors stack normally; string meta fields
            # become lists of strings automatically via the default collate.
        )
        print(ds.summary())

    return loaders


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, time

    ap = argparse.ArgumentParser(
        description="Smoke-test RBCDataset against a manifest CSV."
    )
    ap.add_argument("manifest_csv")
    ap.add_argument("--x",           type=int, default=90)
    ap.add_argument("--y",           type=int, default=30)
    ap.add_argument("--stride",      type=int, default=10)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cohort",      default=None)
    ap.add_argument(
        "--norm_mode",
        choices=["train_global"],
        default="train_global",
    )
    ap.add_argument("--n_batches",   type=int, default=3,
                    help="How many batches to time per split")
    args = ap.parse_args()

    loaders = make_dataloaders(
        args.manifest_csv,
        x=args.x, y=args.y, stride=args.stride,
        batch_size=args.batch_size, num_workers=args.num_workers,
        cohort=args.cohort,
        norm_mode=args.norm_mode,
    )

    for split, loader in loaders.items():
        if len(loader.dataset) == 0:           # type: ignore[arg-type]
            print(f"  [{split}] EMPTY – skipping")
            continue
        t0 = time.perf_counter()
        for i, batch in enumerate(loader):
            if i == 0:
                print(
                    f"  [{split}] x={tuple(batch['x'].shape)}  "
                    f"y={tuple(batch['y'].shape)}  "
                    f"cohort={batch['meta']['cohort'][:2]}"
                )
            if i + 1 >= args.n_batches:
                break
        elapsed = time.perf_counter() - t0
        print(f"  [{split}] {args.n_batches} batches in {elapsed:.2f}s")
