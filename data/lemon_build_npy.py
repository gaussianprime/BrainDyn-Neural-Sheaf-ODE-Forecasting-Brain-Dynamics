"""lemon_build_npy.py
===================
Preprocessing / derivative builder for the LEMON resting-state EEG arm
(eyes-closed, sensor space).

For each ``sub-XXXXXX_EC.set`` EEGLAB file this loads the recording **once**
with MNE, splits it on the EEGLAB ``boundary`` annotation ONSETS into its
boundary-free blocks (8 one-minute EC blocks for a typical subject), and writes:

    <out_dir>/<subject>_<cond>_block{k:02d}.npy   (T_block, N)  float32
    <out_dir>/<subject>_<cond>.json               tiny sidecar

The per-block ``.npy`` files are the **shared starting point** for both this
sensor arm and the future source-projection arm — MNE never runs inside the
training loop; the dataset reads these arrays with ``np.load``.

Gotchas honored (see handoff §4/§9):
  * Split on boundary **onsets only** — EEGLAB→MNE boundary events carry garbage
    durations ("annotation expanding outside data range"), so durations are ignored.
  * Channel count is **read from the data**, never hardcoded (59 here, varies).
  * The ``.set`` needs its ``.fdt`` co-located to load.

Usage
-----
    # one subject (smoke test)
    python -m data.lemon_build_npy --set path/to/sub-032301_EC.set --out_dir data/lemon_ec_npy

    # a whole download tree (recurses for *_<cond>.set)
    python -m data.lemon_build_npy --eeg_root /path/to/lemon_ec --out_dir data/lemon_ec_npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _boundary_block_edges(onsets_samples: List[int], n_times: int) -> List[int]:
    """Return sorted block edges ``[0, b1, b2, ..., n_times]`` from boundary onsets.

    Boundary DURATIONS are deliberately ignored (they are garbage in the
    EEGLAB→MNE conversion). Only onsets define where one boundary-free block ends
    and the next begins.
    """
    edges = [0] + sorted(int(s) for s in onsets_samples) + [int(n_times)]
    # Deduplicate / clamp so we never emit a zero- or negative-length block.
    cleaned = [edges[0]]
    for e in edges[1:]:
        e = max(cleaned[-1], min(e, n_times))
        if e > cleaned[-1]:
            cleaned.append(e)
    return cleaned


def _interpolate_to_target(raw, target_channels: List[str], montage_name: str):
    """Reconstruct missing channels so ``raw`` carries exactly ``target_channels``.

    Missing channels (dropped as bad in LEMON preprocessing) are added as zeros,
    marked bad, and spherical-spline interpolated from their neighbors, then the
    channels are reordered to the target montage order. This removes per-subject
    channel dropout so every subject presents the same fixed node set. Returns
    ``(raw, interpolated_channel_names)``.
    """
    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="raise")

    target = list(target_channels)
    target_set = set(target)
    present = set(raw.ch_names)

    # Channels outside the target (shouldn't happen for a union target, but a
    # fixed nominal target could omit one) are dropped so the node set is exact.
    extra = [c for c in raw.ch_names if c not in target_set]
    if extra:
        raw.drop_channels(extra)

    missing = [c for c in target if c not in present]
    if missing:
        info_missing = mne.create_info(
            missing, raw.info["sfreq"], ["eeg"] * len(missing)
        )
        data_missing = np.zeros((len(missing), raw.n_times), dtype=float)
        raw_missing = mne.io.RawArray(
            data_missing, info_missing, first_samp=raw.first_samp
        )
        # Base object is `raw`, so its boundary annotations are preserved.
        raw.add_channels([raw_missing], force_update_info=True)
        raw.set_montage(montage, match_case=False, on_missing="raise")
        raw.info["bads"] = list(missing)
        raw.interpolate_bads(reset_bads=True, mode="accurate")

    raw.reorder_channels(target)
    return raw, missing


class LowChannelCount(ValueError):
    """Raised when a subject has fewer good channels than the QC minimum."""


def build_subject(
    set_path: str | Path,
    out_dir: str | Path,
    target_channels: List[str] | None = None,
    montage_name: str = "standard_1005",
    min_channels: int = 0,
) -> dict:
    """Load one ``*_<cond>.set``, split on boundaries, write per-block ``.npy`` + sidecar.

    If ``target_channels`` is given, each subject's missing channels are
    interpolated up to that montage first (see ``_interpolate_to_target``), so
    every subject shares the same fixed node set. ``min_channels`` is a QC gate on
    the subject's *original* (pre-interpolation) good-channel count — below it the
    subject is rejected (``LowChannelCount``) so we never impute a large fraction
    of a subject's montage. Returns the sidecar dict.
    """
    import mne  # imported lazily so the training loop never needs MNE

    mne.set_log_level("ERROR")

    set_path = Path(set_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse "sub-XXXXXX_<cond>" from the filename stem.
    stem = set_path.stem  # e.g. "sub-032301_EC"
    if "_" in stem:
        subject_id, condition = stem.rsplit("_", 1)
    else:
        subject_id, condition = stem, "NA"

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True)
    sfreq = float(raw.info["sfreq"])
    n_channels_original = len(raw.ch_names)
    if min_channels and n_channels_original < min_channels:
        raise LowChannelCount(
            f"{subject_id}_{condition}: {n_channels_original} good channels "
            f"< --min_channels {min_channels}"
        )

    interpolated_channels: List[str] = []
    if target_channels is not None:
        raw, interpolated_channels = _interpolate_to_target(
            raw, target_channels, montage_name
        )

    ch_names = list(raw.ch_names)
    n_channels = len(ch_names)  # read from data; never hardcoded

    # Boundary onsets in SAMPLES (onsets only — durations are garbage).
    onsets_samples = [
        round(onset * sfreq)
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if "boundary" in str(desc).lower()
    ]
    edges = _boundary_block_edges(onsets_samples, raw.n_times)

    # (T, N) float32 — transpose from MNE's (N, T).
    data = raw.get_data().T.astype(np.float32, copy=False)

    block_files: List[str] = []
    block_offsets: List[list[int]] = []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        block = np.ascontiguousarray(data[a:b])  # (T_block, N)
        fname = f"{subject_id}_{condition}_block{k:02d}.npy"
        fpath = out_dir / fname
        np.save(fpath, block)
        block_files.append(str(fpath.resolve()))
        block_offsets.append([int(a), int(b)])

    sidecar = {
        "subject_id": subject_id,
        "condition": condition,
        "sfreq": sfreq,
        "n_channels": n_channels,
        "n_channels_original": n_channels_original,
        "ch_names": ch_names,
        "interpolated": target_channels is not None,
        "interpolated_channels": interpolated_channels,
        "n_blocks": len(block_files),
        "block_files": block_files,
        "block_offsets": block_offsets,  # [start, end) into the full series
        "n_times_total": int(raw.n_times),
        "source_set": str(set_path.resolve()),
    }
    sidecar_path = out_dir / f"{subject_id}_{condition}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    return sidecar


def _discover_sets(eeg_root: Path, condition: str) -> List[Path]:
    return sorted(eeg_root.rglob(f"*_{condition}.set"))


def scan_channel_union(set_paths: List[Path]) -> List[str]:
    """Pass-1 union of channel names across recordings (reads headers only)."""
    return sorted(set().union(*scan_headers(set_paths).values())) if set_paths else []


def scan_headers(set_paths: List[Path]) -> Dict[str, List[str]]:
    """Read each recording's channel names from the header only (``preload=False``).

    Cheap pass-1 used to QC-filter subjects by good-channel count and to compute
    the interpolation-target union without loading the ``.fdt`` payloads.
    """
    import mne

    mne.set_log_level("ERROR")
    headers: Dict[str, List[str]] = {}
    for sp in set_paths:
        raw = mne.io.read_raw_eeglab(str(sp), preload=False)
        headers[str(sp)] = list(raw.ch_names)
    return headers


def _write_montage_positions(
    channels: List[str], out_dir: Path, montage_name: str
) -> None:
    """Write ``montage_positions.json`` (name→xyz) so the spatial graph and the
    comparison notebook need no MNE at train time."""
    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    ch_pos = montage.get_positions()["ch_pos"]
    lower = {str(k).lower(): (k, v) for k, v in ch_pos.items()}
    positions: dict = {}
    for c in channels:
        hit = ch_pos.get(c)
        if hit is None and c.lower() in lower:
            hit = lower[c.lower()][1]
        if hit is not None:
            positions[c] = [float(x) for x in hit]
    path = out_dir / "montage_positions.json"
    path.write_text(json.dumps({"montage": montage_name, "ch_pos": positions}, indent=2))
    print(f"  electrode positions: {len(positions)} channels -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build per-block LEMON EEG .npy derivatives from EEGLAB .set files."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--set", type=str, help="Path to a single *_<cond>.set file.")
    src.add_argument(
        "--eeg_root",
        type=str,
        help="Root dir to recurse for *_<condition>.set files (co-located .fdt required).",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write .npy + .json.")
    ap.add_argument(
        "--condition",
        type=str,
        default="EC",
        help="Condition suffix to discover under --eeg_root (default EC).",
    )
    ap.add_argument(
        "--interpolate",
        action="store_true",
        help="Spherical-spline interpolate each subject's missing channels up to a "
        "common montage (the union of channel names across all subjects), so every "
        "subject shares the same fixed node set. Reviewer-standard; removes the "
        "per-subject channel dropout that otherwise forces an intersection.",
    )
    ap.add_argument(
        "--montage",
        type=str,
        default="standard_1005",
        help="MNE standard montage used for channel positions during interpolation.",
    )
    ap.add_argument(
        "--target_channels_json",
        type=str,
        default=None,
        help="With --interpolate: JSON providing the target montage (a list, or "
        "{'channels': [...]}) — typically the montage_union.json emitted by a prior "
        "--eeg_root run. Use it to interpolate a second condition (e.g. EO) onto the "
        "SAME montage as the first (EC). Required for single --set; optional for "
        "--eeg_root (overrides the per-run union when given).",
    )
    ap.add_argument(
        "--min_channels",
        type=int,
        default=58,
        help="QC gate: reject subjects with fewer than this many original good "
        "channels (before interpolation), so we never impute a large fraction of a "
        "montage. Set 0 to disable.",
    )
    args = ap.parse_args()

    if args.set:
        set_paths = [Path(args.set)]
    else:
        set_paths = _discover_sets(Path(args.eeg_root), args.condition)
        if not set_paths:
            raise FileNotFoundError(
                f"No *_{args.condition}.set files found under {args.eeg_root}"
            )

    def _load_target(path: str) -> List[str]:
        payload = json.loads(Path(path).read_text())
        return payload["channels"] if isinstance(payload, dict) else list(payload)

    # Resolve the interpolation target montage (data-driven union, or a provided one).
    target_channels: List[str] | None = None
    if args.interpolate:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.eeg_root:
            # Pass 1: read headers, QC-filter by --min_channels, then either adopt a
            # provided target montage (e.g. EC's, for EO) or compute this run's union.
            print(f"Pass 1/2: scanning headers across {len(set_paths)} recordings "
                  f"(QC: >= {args.min_channels} channels)...")
            headers = scan_headers(set_paths)
            kept, dropped = [], []
            for sp in set_paths:
                (kept if len(headers[str(sp)]) >= args.min_channels else dropped).append(sp)
            for sp in dropped:
                print(f"  QC drop: {Path(sp).stem} has {len(headers[str(sp)])} "
                      f"< {args.min_channels} channels")
            print(f"  kept {len(kept)}/{len(set_paths)} recordings")
            set_paths = kept

            if args.target_channels_json:
                target_channels = _load_target(args.target_channels_json)
                print(f"  using PROVIDED target montage: {len(target_channels)} channels "
                      f"from {args.target_channels_json} (not overwriting montage_*.json)")
            else:
                target_channels = sorted(set().union(*(headers[str(p)] for p in kept)))
                union_path = out_dir / "montage_union.json"
                union_path.write_text(
                    json.dumps({"channels": target_channels, "n": len(target_channels)}, indent=2)
                )
                print(f"  union montage: {len(target_channels)} channels -> {union_path}")
                _write_montage_positions(target_channels, out_dir, args.montage)
        else:
            if not args.target_channels_json:
                raise ValueError(
                    "--interpolate with single --set requires --target_channels_json "
                    "(e.g. the montage_union.json from a prior --eeg_root run)."
                )
            target_channels = _load_target(args.target_channels_json)
            print(f"Using target montage of {len(target_channels)} channels from "
                  f"{args.target_channels_json}")

    pass_tag = "Pass 2/2: building" if args.interpolate and args.eeg_root else "Building"
    print(f"{pass_tag} .npy derivatives for {len(set_paths)} recording(s) -> {args.out_dir}")
    n_skipped = 0
    for i, sp in enumerate(set_paths, 1):
        try:
            sc = build_subject(
                sp, args.out_dir,
                target_channels=target_channels,
                montage_name=args.montage,
                min_channels=args.min_channels,
            )
            interp = (
                f", interpolated {len(sc['interpolated_channels'])} "
                f"(from {sc['n_channels_original']})"
                if sc["interpolated"] else ""
            )
            print(
                f"  [{i}/{len(set_paths)}] {sc['subject_id']}_{sc['condition']}: "
                f"{sc['n_blocks']} blocks, {sc['n_channels']} ch{interp}, "
                f"{sc['n_times_total']} samples @ {sc['sfreq']:.0f} Hz"
            )
        except LowChannelCount as exc:  # QC rejection (e.g. single --set below min)
            n_skipped += 1
            print(f"  [{i}/{len(set_paths)}] QC skip: {exc}")
        except Exception as exc:  # keep going; a single bad file shouldn't halt a batch
            print(f"  [{i}/{len(set_paths)}] FAILED {sp}: {exc}")

    if n_skipped:
        print(f"Skipped {n_skipped} recording(s) below --min_channels={args.min_channels}.")


if __name__ == "__main__":
    main()
