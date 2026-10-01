"""lemon_make_manifest.py
======================
Manifest builder for the LEMON sensor-EEG arm. Mirrors the role of the fMRI
``manifest.csv``: one row per subject, consumed by ``data/lemon_dataset.py``.

It scans a directory of per-block ``.npy`` derivatives + sidecar JSONs produced
by ``data/lemon_build_npy.py`` and emits a CSV with, per subject:

    subject_id, condition, split, n_blocks, block_paths (';'-joined),
    n_channels, sfreq, T_total, has_mri, in_overlap

Design decisions (handoff §5.4, §7.2):
  * **Split is assigned at the SUBJECT level**, seeded/deterministic, so the
    test split is subject-disjoint from train/val (a clean held-out null). The
    window-level CV in ``main.py`` re-mixes only the train+val pool.
  * The full EC set is emitted with an ``in_overlap`` flag (EEG∩MRI); subject
    *subsetting* for a given run (``--max_subjects``) is a run-time decision made
    in the dataset, NOT here — so every graph-mode variant sees the same subjects.
  * ``has_mri`` / ``in_overlap`` are computed by intersecting the EEG subject IDs
    with the FreeSurfer subject directories, when ``--freesurfer_root`` is given.

Usage
-----
    python -m data.lemon_make_manifest \
        --npy_dir data/lemon_ec_npy \
        --out data/lemon_manifest.csv \
        --test_frac 0.2 --val_frac 0.0 --seed 42 \
        [--freesurfer_root /path/to/freesurfer/ds000221]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np


def _load_sidecars(npy_dir: Path, condition: str) -> List[dict]:
    sidecars: List[dict] = []
    for jp in sorted(npy_dir.glob(f"*_{condition}.json")):
        sc = json.loads(jp.read_text())
        # Resolve block paths relative to the sidecar location if they moved.
        resolved = []
        for bp in sc["block_files"]:
            p = Path(bp)
            if not p.exists():
                cand = npy_dir / Path(bp).name
                if cand.exists():
                    p = cand
            resolved.append(str(p))
        sc["block_files"] = resolved
        sidecars.append(sc)
    return sidecars


def _freesurfer_subjects(freesurfer_root: Path | None) -> Set[str]:
    if freesurfer_root is None:
        return set()
    if not freesurfer_root.exists():
        raise FileNotFoundError(f"FreeSurfer root not found: {freesurfer_root}")
    return {p.name for p in freesurfer_root.glob("sub-*") if p.is_dir()}


def _assign_splits(
    subject_ids: List[str], test_frac: float, val_frac: float, seed: int
) -> Dict[str, str]:
    """Deterministic subject-level split assignment (test is subject-disjoint)."""
    if test_frac < 0 or val_frac < 0 or (test_frac + val_frac) >= 1.0:
        raise ValueError(
            f"Require test_frac,val_frac >= 0 and test_frac+val_frac < 1; "
            f"got test={test_frac}, val={val_frac}"
        )
    rng = np.random.default_rng(seed)
    order = sorted(subject_ids)  # stable base order before the seeded shuffle
    shuffled = list(order)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    split_of: Dict[str, str] = {}
    for i, sid in enumerate(shuffled):
        if i < n_test:
            split_of[sid] = "test"
        elif i < n_test + n_val:
            split_of[sid] = "val"
        else:
            split_of[sid] = "train"
    return split_of


def build_manifest(
    npy_dir: str | Path,
    out_csv: str | Path,
    conditions: List[str] = ("EC",),
    test_frac: float = 0.2,
    val_frac: float = 0.0,
    seed: int = 42,
    freesurfer_root: str | Path | None = None,
    min_channels: int = 58,
    require_equal_subjects: bool = False,
) -> int:
    """Build a manifest for one or more conditions.

    With multiple conditions (e.g. ``["EC", "EO"]``) this emits **one row per
    (subject, condition)** into a single CSV, selected at train time by
    ``--condition``. Only subjects present in *every* requested condition are kept
    (a clean paired design), and the train/val/test split is assigned **per
    subject** so a subject has the SAME split in all conditions. Because the split
    RNG keys off ``seed`` and the sorted common-subject list, an EC-only manifest
    and this combined manifest agree on splits when the subject set matches.
    """
    npy_dir = Path(npy_dir)
    conditions = list(conditions)

    # --- load sidecars per condition ------------------------------------------
    by_cond: Dict[str, Dict[str, dict]] = {}
    for cond in conditions:
        sidecars = _load_sidecars(npy_dir, cond)
        if not sidecars:
            raise FileNotFoundError(
                f"No *_{cond}.json sidecars found in {npy_dir}. Run lemon_build_npy "
                f"--condition {cond} first."
            )
        by_cond[cond] = {sc["subject_id"]: sc for sc in sidecars}

    # --- subject-count check across conditions --------------------------------
    subj_sets = {c: set(m) for c, m in by_cond.items()}
    common = set.intersection(*subj_sets.values())
    print("Subject counts per condition:")
    for c in conditions:
        extra = len(subj_sets[c] - common)
        print(f"  {c}: {len(subj_sets[c])} subjects" + (f"  ({extra} not in all conditions)" if extra else ""))
    if len(conditions) > 1 and any(subj_sets[c] != common for c in conditions):
        msg = (f"Conditions do not share the same subjects: common={len(common)}, "
               + ", ".join(f"{c}={len(subj_sets[c])}" for c in conditions))
        if require_equal_subjects:
            raise ValueError(msg + " (omit --require_equal_subjects to keep the intersection).")
        print(f"  WARNING: {msg}. Keeping the {len(common)} shared subjects.")

    # --- QC warning: original good-channel count ------------------------------
    low = [(c, sid, sc.get("n_channels_original", sc["n_channels"]))
           for c in conditions for sid, sc in by_cond[c].items()
           if sid in common and sc.get("n_channels_original", sc["n_channels"]) < min_channels]
    if low:
        print(f"  WARNING: {len(low)} (subject,condition) below {min_channels} original "
              f"channels (should have been QC-dropped at build time): {low[:5]}...")

    fs_subjects = _freesurfer_subjects(Path(freesurfer_root) if freesurfer_root else None)
    split_of = _assign_splits(sorted(common), test_frac, val_frac, seed)

    fieldnames = [
        "subject_id", "condition", "split", "n_blocks", "block_paths",
        "n_channels", "ch_names", "n_interpolated", "sfreq", "T_total",
        "has_mri", "in_overlap",
    ]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cond in conditions:
            for sid in sorted(common):
                sc = by_cond[cond][sid]
                has_mri = sid in fs_subjects
                writer.writerow({
                    "subject_id": sid,
                    "condition": sc["condition"],
                    "split": split_of[sid],           # same split across conditions
                    "n_blocks": sc["n_blocks"],
                    "block_paths": ";".join(sc["block_files"]),
                    "n_channels": sc["n_channels"],
                    "ch_names": ";".join(sc["ch_names"]),
                    "n_interpolated": len(sc.get("interpolated_channels", [])),
                    "sfreq": sc["sfreq"],
                    "T_total": sc["n_times_total"],
                    "has_mri": int(has_mri),
                    "in_overlap": int(has_mri),
                })
                n_rows += 1

    n_test = sum(1 for s in split_of.values() if s == "test")
    n_val = sum(1 for s in split_of.values() if s == "val")
    n_overlap = sum(1 for sid in common if sid in fs_subjects)
    print(
        f"Wrote {out_csv}: {n_rows} rows = {len(common)} subjects x {len(conditions)} "
        f"condition(s) {conditions} | split (train={len(common) - n_test - n_val}, "
        f"val={n_val}, test={n_test}); {n_overlap} in EEG-MRI overlap."
    )
    return n_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the LEMON sensor-EEG manifest CSV from .npy derivatives."
    )
    ap.add_argument("--npy_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument(
        "--conditions",
        type=str,
        default="EC",
        help="Comma-separated conditions to include, e.g. 'EC' or 'EC,EO'. Multiple "
        "conditions emit one combined manifest (one row per subject x condition), "
        "selected at train time by --condition.",
    )
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min_channels",
        type=int,
        default=58,
        help="Warn if any subject's original good-channel count is below this.",
    )
    ap.add_argument(
        "--require_equal_subjects",
        action="store_true",
        help="Error (instead of intersecting) if conditions don't share the same subjects.",
    )
    ap.add_argument(
        "--freesurfer_root",
        type=str,
        default=None,
        help="Optional: dir containing sub-* FreeSurfer subjects, for has_mri/in_overlap.",
    )
    args = ap.parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    build_manifest(
        npy_dir=args.npy_dir,
        out_csv=args.out,
        conditions=conditions,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        freesurfer_root=args.freesurfer_root,
        min_channels=args.min_channels,
        require_equal_subjects=args.require_equal_subjects,
    )


if __name__ == "__main__":
    main()
