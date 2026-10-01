"""lemon_dataset.py
================
PyTorch Dataset + DataLoader factory for the LEMON sensor-space EEG arm
(eyes-closed). A drop-in parallel to ``data/rbc_dataset.py`` (fMRI): it produces
the **identical tensor contract**, so ``main.py`` treats an EEG channel exactly
as it treats an fMRI ROI — each channel is a node.

Contract mirrored from ``RBCDataset`` (verified against the real file):
  * ``__getitem__`` -> ``{"x": (Lx, N), "y": (Ly, N), "meta": dict}``; N (channels)
    is the last axis; the default collate stacks to ``(B, Lx, N)`` / ``(B, Ly, N)``.
  * ``._samples``: ``list[tuple[path, meta, t]]`` — one entry per window, ``t`` the
    window start index into the array returned by ``_load_cached(path)``.
  * ``.x``: context length attribute.
  * ``._load_cached(path) -> np.ndarray (T, N)``.
  These three are the tightest coupling: ``compute_train_global_stats`` in
  ``main.py`` reaches through them (``ts = ds._load_cached(path); ctx = ts[t:t+ds.x]``)
  for ``--norm_mode train_global`` — the mode this arm actually uses.

LEMON-specific structure:
  * ``path`` is a **single boundary-free block** ``.npy`` (written by
    ``lemon_build_npy.py``). Because each block is its own array, a window can
    never straddle a block seam — the "boundary onsets only" rule from the probe
    is enforced structurally, in one place: the ``_samples`` enumeration.
  * ``--max_subjects``: seeded, deterministic subject selection over the sorted
    subject universe, so every graph-mode variant sees the SAME subjects.
  * ``--max_windows_per_subject``: per-subject seeded subsample of the enumerated
    boundary-safe window START indices (we sample starts, never truncate time, so
    windows still span all blocks).

Normalization is fold-training-global: this dataset returns raw windows and
``main.py`` fits statistics using training contexts only.
"""

from __future__ import annotations

import csv
import json
import zlib
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Split = Literal["train", "val", "test", "within"]
NormMode = Literal["train_global"]
_BETWEEN_SPLITS = frozenset({"train", "val", "test"})


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------


def _load_npy(path: str | Path) -> np.ndarray:
    """Load a per-block LEMON array as float32 of shape (T_block, N)."""
    arr = np.load(path)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _block_len(path: str | Path) -> int:
    """Cheap block length (reads the .npy header only, not the payload)."""
    return int(np.load(path, mmap_mode="r").shape[0])


def _subject_rng(seed: int, subject_id: str) -> np.random.Generator:
    """Deterministic, order-independent per-subject RNG for window subsampling."""
    mixed = (int(seed) ^ zlib.crc32(subject_id.encode("utf-8"))) & 0xFFFFFFFF
    return np.random.default_rng(mixed)


# ---------------------------------------------------------------------------
# Channel harmonization
# ---------------------------------------------------------------------------
# LEMON drops bad channels per subject, so the channel COUNT varies (58-61 from a
# nominal 62). The model and the pooled train_global stats need a fixed node set,
# so every subject is reindexed onto a canonical montage: the intersection of
# channel NAMES common to all subjects, in a stable (sorted) order. Node j is then
# the same electrode across all subjects.


def _row_ch_names(r: dict) -> List[str]:
    names = [c for c in r.get("ch_names", "").split(";") if c]
    if not names:
        raise ValueError(
            "Manifest row is missing the 'ch_names' column needed to harmonize "
            "per-subject channel sets. Rebuild the manifest with "
            "data/lemon_make_manifest.py (the sidecar JSONs already carry ch_names)."
        )
    return names


def _canonical_channels(rows: List[dict]) -> List[str]:
    """Canonical montage = channel names common to every subject, sorted."""
    per_subject = [set(_row_ch_names(r)) for r in rows]
    if not per_subject:
        return []
    common = set.intersection(*per_subject)
    if not common:
        raise RuntimeError(
            "No channels are common to all subjects, so no fixed montage exists. "
            "Restrict the manifest to subjects sharing a montage, or interpolate "
            "bad channels back to the full cap at preprocessing time."
        )
    return sorted(common)


def _reorder_indices(subject_ch_names: List[str], canonical: List[str]) -> np.ndarray:
    """Column indices that map a subject's block (in its own order) to canonical."""
    pos = {c: i for i, c in enumerate(subject_ch_names)}
    return np.array([pos[c] for c in canonical], dtype=np.int64)


def _build_path_reorder(
    rows: List[dict], canonical: List[str]
) -> Dict[str, np.ndarray]:
    """Map every block path to the column-index array that harmonizes it."""
    reorder: Dict[str, np.ndarray] = {}
    for r in rows:
        idx = _reorder_indices(_row_ch_names(r), canonical)
        for bp in r["block_paths"].split(";"):
            if bp:
                reorder[bp] = idx
    return reorder


def montage_positions(
    channels: List[str],
    montage_name: str = "standard_1005",
    positions_json: str | Path | None = None,
) -> np.ndarray:
    """3D electrode coordinates for ``channels``, in the given order → ``(N, 3)``.

    Used to build the spatial (electrode-proximity) prior graph. Positions come
    from ``montage_positions.json`` (written by ``lemon_build_npy.py``) when
    available — so training needs no MNE — otherwise MNE is lazily imported to read
    the standard montage (a one-time setup step, never the training loop). Matching
    is case-insensitive; any channel absent from the montage raises.
    """
    name_to_xyz: dict | None = None
    if positions_json is not None:
        p = Path(positions_json)
        if p.exists():
            payload = json.loads(p.read_text())
            name_to_xyz = payload.get("ch_pos", payload)

    if name_to_xyz is None:
        import mne  # lazy: only needed when no positions file is present

        montage = mne.channels.make_standard_montage(montage_name)
        name_to_xyz = {k: list(v) for k, v in montage.get_positions()["ch_pos"].items()}

    lower = {str(k).lower(): v for k, v in name_to_xyz.items()}
    pos: List = []
    missing: List[str] = []
    for c in channels:
        v = name_to_xyz.get(c)
        if v is None:
            v = lower.get(c.lower())
        if v is None:
            missing.append(c)
        else:
            pos.append(v)
    if missing:
        raise ValueError(
            f"{len(missing)} channel(s) not found in montage {montage_name!r}: {missing}"
        )
    return np.asarray(pos, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LemonDataset(Dataset):
    """Sliding-window dataset over LEMON EC per-block EEG timeseries.

    Parameters
    ----------
    manifest_csv : str | Path
        CSV produced by ``lemon_make_manifest.py``.
    split : "train" | "val" | "test" | "within"
        Between-subject split (from the manifest's per-subject ``split`` column),
        or "within" for the last-window-per-block held-out eval.
    x, y, stride : int
        Context / horizon / window stride (same semantics as ``RBCDataset``).
    condition : str
        Condition filter (default "EC").
    max_subjects : int | None
        Cap the subject universe to a seeded deterministic subset (shared across
        all splits and graph-mode variants). None = all subjects.
    max_windows_per_subject : int | None
        Per-subject seeded subsample of enumerated window starts. None = keep all.
    overlap_only : bool
        Restrict to subjects flagged ``in_overlap`` (EEG∩MRI).
    seed : int
        Seed for subject selection and per-subject window subsampling.
    norm_mode : "context" | "train_global"
    cache : bool
        Hold per-block arrays in RAM after first load.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: Split,
        x: int = 30,
        y: int = 10,
        stride: int = 10,
        condition: str = "EC",
        max_subjects: Optional[int] = None,
        max_windows_per_subject: Optional[int] = None,
        overlap_only: bool = False,
        seed: int = 42,
        norm_mode: NormMode = "train_global",
        cache: bool = False,
    ) -> None:
        if split not in (_BETWEEN_SPLITS | {"within"}):
            raise ValueError(
                f"split must be one of 'train','val','test','within'; got {split!r}"
            )
        if norm_mode != "train_global":
            raise ValueError("only fold-training-global normalization is supported")
        self.x = x
        self.y = y
        self.stride = stride
        self.split = split
        self.condition = condition
        self.norm_mode = norm_mode
        self.cache = cache
        self._cache: Dict[str, np.ndarray] = {}

        rows = self._read_manifest(manifest_csv)
        rows = [r for r in rows if r["condition"] == condition]
        if overlap_only:
            rows = [r for r in rows if str(r.get("in_overlap", "0")) in ("1", "True", "true")]

        # --- canonical montage (fixed node set) -------------------------------
        # Computed over ALL condition/overlap-filtered subjects so num_nodes is
        # stable regardless of split or --max_subjects.
        self.channels = _canonical_channels(rows)

        # --- seeded subject selection (shared across splits / graph modes) ----
        selected = self._select_subjects(rows, max_subjects, seed)

        # --- keep rows for this split whose subject is selected ---------------
        split_rows = [
            r for r in rows if r["subject_id"] in selected and r["split"] == split
        ] if split in _BETWEEN_SPLITS else [
            r for r in rows if r["subject_id"] in selected
        ]

        # Per-block column-reorder onto the canonical montage, applied in
        # _load_cached so every subject presents the same nodes in the same order.
        self._path_reorder = _build_path_reorder(split_rows, self.channels)

        # --- build flat sample index: (block_path, meta, t) -------------------
        self._samples: List[tuple[str, dict, int]] = []
        for r in split_rows:
            subject_id = r["subject_id"]
            block_paths = [p for p in r["block_paths"].split(";") if p]

            # Enumerate all boundary-safe starts for this subject across blocks,
            # then (optionally) subsample START INDICES — never truncating time.
            subject_starts: List[tuple[str, int, int, int]] = []  # (path, t, block_idx, T_block)
            for block_idx, bp in enumerate(block_paths):
                T = _block_len(bp)
                if split == "within":
                    t0 = T - x - y
                    if t0 >= 0:
                        subject_starts.append((bp, t0, block_idx, T))
                else:
                    # Horizon must end strictly before T-y so the last y samples
                    # of a block are never a training/eval target:
                    #   t + x + y <= T - y  ->  t <= T - x - 2*y
                    max_t = T - x - 2 * y
                    if max_t < 0:
                        continue
                    for t in range(0, max_t + 1, stride):
                        subject_starts.append((bp, t, block_idx, T))

            if (
                max_windows_per_subject is not None
                and len(subject_starts) > max_windows_per_subject
            ):
                rng = _subject_rng(seed, subject_id)
                keep = rng.choice(
                    len(subject_starts), size=max_windows_per_subject, replace=False
                )
                subject_starts = [subject_starts[i] for i in sorted(keep.tolist())]

            for bp, t, block_idx, T in subject_starts:
                meta = {
                    "subject_id": subject_id,
                    "condition": condition,
                    "between_split": r["split"],
                    "block_idx": block_idx,
                    "T": T,             # block length (what _load_cached returns)
                    "path": bp,         # per-block .npy path
                    "has_mri": int(str(r.get("has_mri", "0")) in ("1", "True", "true")),
                }
                self._samples.append((bp, meta, t))

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        path, meta, t = self._samples[idx]
        ts = self._load_cached(path)  # (T_block, N)

        ctx_np = ts[t : t + self.x]                       # (x, N)
        hrz_np = ts[t + self.x : t + self.x + self.y]     # (y, N)

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

    # Salt that decouples subject SELECTION from the manifest's split ASSIGNMENT.
    # Both steps otherwise do `default_rng(seed).shuffle(sorted(ids))`; if the
    # manifest was built with the same seed used here (both default to 42), the
    # two permutations are identical, so `shuffled[:max_subjects]` lands entirely
    # inside the manifest's `shuffled[:n_test]` test block and the train split
    # comes out empty. Feeding a distinct SeedSequence stream guarantees the two
    # orderings are independent even at equal seeds.
    _SELECT_SALT = 0x5E1EC7

    @staticmethod
    def _select_subjects(
        rows: List[dict], max_subjects: Optional[int], seed: int
    ) -> set:
        """Seeded deterministic subject subset over the sorted subject universe.

        Stratified by split so the cap is spread across train/val/test in
        proportion to their sizes — the held-out test split and the train pool
        are both represented for any reasonable ``max_subjects``. Deterministic
        and independent of graph mode, so every ablation arm sees the same
        subjects.
        """
        universe = sorted({r["subject_id"] for r in rows})
        if max_subjects is None or max_subjects >= len(universe):
            return set(universe)

        by_split: Dict[str, list] = {}
        for r in rows:
            by_split.setdefault(r["split"], set()).add(r["subject_id"])
        splits_sorted = sorted(by_split)
        sizes = {s: len(by_split[s]) for s in splits_sorted}
        total = sum(sizes.values())

        # Largest-remainder allocation so the per-split takes sum to exactly
        # max_subjects.
        raw = {s: max_subjects * sizes[s] / total for s in splits_sorted}
        alloc = {s: int(np.floor(raw[s])) for s in splits_sorted}
        remainder = max_subjects - sum(alloc.values())
        by_frac = sorted(splits_sorted, key=lambda s: raw[s] - alloc[s], reverse=True)
        for i in range(remainder):
            alloc[by_frac[i % len(by_frac)]] += 1

        rng = np.random.default_rng([seed, LemonDataset._SELECT_SALT])
        selected: set = set()
        for s in splits_sorted:
            ids = sorted(by_split[s])
            rng.shuffle(ids)
            selected.update(ids[: min(alloc[s], len(ids))])
        return selected

    def _load_cached(self, path: str) -> np.ndarray:
        """Return the block harmonized to the canonical montage, shape (T, N_canon).

        The reorder is applied here (not in __getitem__) so that main.py's
        train_global reach-through — ``ts = ds._load_cached(path); ctx = ts[t:t+x]``
        — sees the same fixed node set for every subject.
        """
        if self.cache and path in self._cache:
            return self._cache[path]
        arr = _load_npy(path)
        reorder = self._path_reorder.get(path)
        if reorder is not None:
            arr = np.ascontiguousarray(arr[:, reorder])
        if self.cache:
            self._cache[path] = arr
        return arr

    def summary(self) -> str:
        n_subjects = len({s[1]["subject_id"] for s in self._samples})
        n_blocks = len({s[0] for s in self._samples})
        return (
            f"LemonDataset(split={self.split!r}, x={self.x}, y={self.y}, "
            f"stride={self.stride}) | {len(self)} windows from "
            f"{n_blocks} blocks / {n_subjects} subjects"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def make_lemon_dataloaders(
    manifest_csv: str | Path,
    x: int = 30,
    y: int = 10,
    stride: int = 10,
    batch_size: int = 64,
    num_workers: int = 1,
    condition: str = "EC",
    max_subjects: Optional[int] = None,
    max_windows_per_subject: Optional[int] = None,
    overlap_only: bool = False,
    seed: int = 42,
    cache: bool = False,
    pin_memory: bool = True,
    norm_mode: NormMode = "train_global",
) -> Dict[str, DataLoader]:
    """Return a dict of DataLoaders keyed "train"/"val"/"test"/"within".

    Same signature shape as ``rbc_dataset.make_dataloaders`` plus the LEMON
    budget knobs (``max_subjects``, ``max_windows_per_subject``, ``overlap_only``,
    ``seed``, ``condition``).
    """
    use_pin = pin_memory and torch.cuda.is_available()

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test", "within"):
        ds = LemonDataset(
            manifest_csv,
            split=split,  # type: ignore[arg-type]
            x=x,
            y=y,
            stride=stride,
            condition=condition,
            max_subjects=max_subjects,
            max_windows_per_subject=max_windows_per_subject,
            overlap_only=overlap_only,
            seed=seed,
            norm_mode=norm_mode,
            cache=cache,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            # A shuffled DataLoader over 0 samples raises in RandomSampler; only
            # shuffle when there is something to shuffle. Empty val/test/within is
            # normal (e.g. no held-out test subjects); an empty train pool is
            # surfaced with a diagnostic below and by main.py's combined check.
            shuffle=(split == "train") and len(ds) > 0,
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=(num_workers > 0 and len(ds) > 0),
        )
        print(ds.summary())

    if len(loaders["train"].dataset) == 0:  # type: ignore[arg-type]
        _diagnose_empty_train(
            manifest_csv,
            condition=condition,
            overlap_only=overlap_only,
            max_subjects=max_subjects,
            seed=seed,
        )

    return loaders


def _diagnose_empty_train(
    manifest_csv: str | Path,
    condition: str,
    overlap_only: bool,
    max_subjects: Optional[int],
    seed: int,
) -> None:
    """Print an actionable breakdown when the train split has zero windows.

    Empty train is almost always a manifest/filter mismatch, not a code bug, so
    show exactly where the subjects went (condition filter, overlap filter,
    per-split counts) instead of leaving the caller with a torch sampler error.
    """
    rows = LemonDataset._read_manifest(manifest_csv)
    conds = sorted({r.get("condition", "") for r in rows})
    cond_rows = [r for r in rows if r.get("condition") == condition]
    if overlap_only:
        cond_rows = [
            r for r in cond_rows if str(r.get("in_overlap", "0")) in ("1", "True", "true")
        ]
    split_counts: Dict[str, int] = {}
    for r in cond_rows:
        split_counts[r.get("split", "?")] = split_counts.get(r.get("split", "?"), 0) + 1
    selected = LemonDataset._select_subjects(cond_rows, max_subjects, seed)
    sel_train = sum(
        1 for r in cond_rows if r["subject_id"] in selected and r.get("split") == "train"
    )
    print(
        "\n[LEMON] WARNING: the 'train' split has 0 windows. Diagnostics:\n"
        f"  manifest: {manifest_csv}\n"
        f"  total rows: {len(rows)} | conditions present: {conds}\n"
        f"  rows matching --condition={condition!r}"
        f"{' and in_overlap' if overlap_only else ''}: {len(cond_rows)}\n"
        f"  per-split subject counts (post-filter): {split_counts}\n"
        f"  subjects kept by max_subjects={max_subjects} (seed={seed}): {len(selected)}\n"
        f"  -> selected subjects labeled split=='train': {sel_train}\n"
        "  Likely causes: no rows labeled split=='train' in the manifest, a "
        "condition/in_overlap filter removing everything, or max_subjects "
        "excluding all train subjects. Rebuild with data/lemon_make_manifest.py "
        "(check --test_frac/--val_frac) or raise/clear --max_subjects.\n"
    )


def make_lemon_run_loader(
    manifest_csv: str | Path,
    condition: str = "EC",
    overlap_only: bool = False,
):
    """Build a per-run loader that harmonizes blocks onto the canonical montage.

    Used by ``SubjectRunDataset`` / the test rollout in main.py (long modes). It
    recomputes the same canonical montage + per-path reorder as ``LemonDataset``
    (over the same condition/overlap-filtered rows), so a run loaded here has the
    identical fixed node set as the windowed batches.
    """
    rows = LemonDataset._read_manifest(manifest_csv)
    rows = [r for r in rows if r["condition"] == condition]
    if overlap_only:
        rows = [
            r for r in rows if str(r.get("in_overlap", "0")) in ("1", "True", "true")
        ]
    canonical = _canonical_channels(rows)
    reorder = _build_path_reorder(rows, canonical)

    def _load(path: str) -> np.ndarray:
        arr = _load_npy(path)
        idx = reorder.get(path)
        if idx is not None:
            arr = np.ascontiguousarray(arr[:, idx])
        return arr

    return _load


def run_loader(path: str) -> np.ndarray:
    """Unharmonized per-run loader (kept for the smoke CLI). main.py uses
    ``make_lemon_run_loader`` so runs share the windowed batches' node set."""
    return _load_npy(path)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Smoke-test LemonDataset against a manifest.")
    ap.add_argument("manifest_csv")
    ap.add_argument("--x", type=int, default=30)
    ap.add_argument("--y", type=int, default=10)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_subjects", type=int, default=None)
    ap.add_argument("--max_windows_per_subject", type=int, default=None)
    ap.add_argument("--overlap_only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--norm_mode", choices=["train_global"], default="train_global")
    ap.add_argument("--n_batches", type=int, default=3)
    args = ap.parse_args()

    loaders = make_lemon_dataloaders(
        args.manifest_csv,
        x=args.x,
        y=args.y,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_subjects=args.max_subjects,
        max_windows_per_subject=args.max_windows_per_subject,
        overlap_only=args.overlap_only,
        seed=args.seed,
        norm_mode=args.norm_mode,
    )
    total = sum(len(l.dataset) for l in loaders.values())  # type: ignore[arg-type]
    print(f"Realized total windows across splits: {total}")

    for split, loader in loaders.items():
        if len(loader.dataset) == 0:  # type: ignore[arg-type]
            print(f"  [{split}] EMPTY - skipping")
            continue
        t0 = time.perf_counter()
        for i, batch in enumerate(loader):
            if i == 0:
                print(
                    f"  [{split}] x={tuple(batch['x'].shape)}  "
                    f"y={tuple(batch['y'].shape)}  "
                    f"subj={batch['meta']['subject_id'][:2]}"
                )
            if i + 1 >= args.n_batches:
                break
        print(f"  [{split}] {args.n_batches} batches in {time.perf_counter() - t0:.2f}s")
