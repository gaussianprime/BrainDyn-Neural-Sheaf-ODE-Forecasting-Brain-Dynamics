#!/usr/bin/env python

"""

inspect_lemon_subject.py

========================



Single-subject inspection for the LEMON (ds000221 / MPI-Leipzig Mind-Brain-Body)

*BIDS-ID preprocessed* resting-state EEG, run BEFORE committing to a manifest

schema or a dataloader. It answers, empirically and from the data itself:



  1. What is the sampling rate and channel count/names? (do NOT hardcode 62)

  2. Is this a single concatenated file (S200/S210 + `boundary` events) or a

     pre-split _EC/_EO file? How is eyes-closed (EC) actually delimited?

  3. Where are the EEGLAB `boundary` events (concatenation points between the

     1-minute blocks)? A forecasting window must never straddle one.

  4. Which marker really is EC? The LEMON descriptor says S210=eyes-closed,

     S200=eyes-open, but this is a known source of confusion, so we cross-check

     against occipital alpha power (higher for EC; the Berger effect).

  5. How many boundary-respecting windows does EC yield at (x, y, stride), and

     how much *physical time* does one window span at 250 Hz?



Nothing here writes derivatives or commits to a schema. It only prints, so you

can eyeball whether the assumptions baked into the loader will hold.



Usage

-----

    python inspect_lemon_subject.py --set-file /path/to/sub-010002.set

    python inspect_lemon_subject.py --eeg-root /data/LEMON/EEG_Preprocessed_BIDS_ID \

                                    --subject sub-010002



Only mne + numpy + scipy are needed for sensor-space inspection; nibabel /

nilearn / mne-bids are NOT required until you touch MRI or the source arm.

On an HPC compute node with no network egress, install the env ONCE on a login

node or in a container — a runtime pip install will fail inside a job.

"""



from __future__ import annotations



import argparse

import glob

import os

import sys





# --------------------------------------------------------------------------- #

# Self-contained dependency check (no-op if already present).                  #

# Trimmed to what sensor-space inspection actually needs.                      #

# --------------------------------------------------------------------------- #

def _ensure_deps() -> None:

    import importlib.util

    import subprocess



    req = {"mne": "mne>=1.6", "pymatreader": "pymatreader", "scipy": "scipy",

           "numpy": "numpy"}

    missing = [pip for mod, pip in req.items()

               if importlib.util.find_spec(mod) is None]

    if missing:

        print("installing:", missing)

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

    print("dependencies ready")





_ensure_deps()



import numpy as np                      # noqa: E402

from scipy.signal import welch          # noqa: E402

import mne                              # noqa: E402



mne.set_log_level("WARNING")





# --------------------------------------------------------------------------- #

# Marker configuration.                                                        #

# Documented (Babayan et al. 2019): S200 = eyes OPEN, S210 = eyes CLOSED.      #

# We match on substring so 'S210', 'Stimulus/S210', ' 210', '210' all resolve. #

# The alpha check below is the source of truth if the label disagrees.         #

# --------------------------------------------------------------------------- #

EC_MARKER_SUBSTR = "210"   # eyes closed  (per data descriptor)

EO_MARKER_SUBSTR = "200"   # eyes open

BOUNDARY_SUBSTR = "boundary"



# Occipital / parieto-occipital channels for the alpha sanity check.

OCC_CANDIDATES = ["O1", "O2", "Oz", "POz", "PO3", "PO4", "PO7", "PO8", "PO9",

                  "PO10", "O9", "O10", "Iz"]

ALPHA_BAND = (8.0, 13.0)





def _sep() -> None:

    print("=" * 74)





# --------------------------------------------------------------------------- #

# Loading                                                                      #

# --------------------------------------------------------------------------- #

# Preprocessed EEGLAB (.set) is what the sensor arm should consume; raw

# BrainVision (.vhdr) is the un-cleaned lineage inside the OpenNeuro MRI tree.

EEG_EXTS = (".set", ".vhdr")





def _report_hits(hits: list[str]) -> None:

    print(f"Found {len(hits)} EEG file(s):")

    for h in hits:

        kind = "preprocessed .set" if h.lower().endswith(".set") else "RAW .vhdr"

        print(f"  - [{kind}] {h}")





def _choose(hits: list[str]) -> str:

    """Prefer preprocessed .set over raw .vhdr; prefer an explicit _EC file."""

    sets = [h for h in hits if h.lower().endswith(".set")]

    pool = sets if sets else hits

    ec = [h for h in pool if "_ec" in os.path.basename(h).lower()

          or os.path.basename(h).lower().endswith("ec.set")]

    chosen = (ec or pool)[0]

    if not sets:

        print("  ! No .set found — only raw BrainVision. This is the MRI/raw "

              "lineage, NOT the preprocessed derivative the loader should use.")

    print(f"Inspecting: {chosen}")

    return chosen





def _glob_eeg(root: str, subject: str | None) -> list[str]:

    hits: list[str] = []

    stem = f"{subject}*" if subject else "*"

    for ext in EEG_EXTS:

        hits += glob.glob(os.path.join(root, "**", f"{stem}{ext}"), recursive=True)

    return sorted(set(hits))





def resolve_eeg_path(args: argparse.Namespace) -> str:

    # Direct path: may be a single file OR a directory to search.

    if args.set_file:

        if os.path.isfile(args.set_file):

            return args.set_file

        if os.path.isdir(args.set_file):

            hits = _glob_eeg(args.set_file, None)

            if not hits:

                sys.exit(

                    f"No .set/.vhdr found under directory:\n  {args.set_file}\n"

                    "  If this is the OpenNeuro ds000221 tree, EEG lives at "

                    "sub-XXXXXX/RSEEG/*.vhdr (raw), and the preprocessed .set "

                    "distribution has to be downloaded separately.")

            _report_hits(hits)

            return _choose(hits)

        sys.exit(f"--set-file path does not exist: {args.set_file}")



    # eeg-root + subject.

    hits = _glob_eeg(args.eeg_root, args.subject)

    if not hits:

        sys.exit(f"No EEG files for {args.subject} under {args.eeg_root}")

    _report_hits(hits)

    return _choose(hits)





def load_raw(path: str) -> mne.io.BaseRaw:

    ext = os.path.splitext(path)[1].lower()

    if ext == ".set":

        return mne.io.read_raw_eeglab(path, preload=True)

    if ext == ".vhdr":

        print("  ! Loading RAW BrainVision — expect ~2500 Hz, ~62 ch, no ICA "

              "cleaning. Useful as a format/marker probe only; the sensor arm "

              "should consume the preprocessed .set instead.")

        return mne.io.read_raw_brainvision(path, preload=True)

    sys.exit(f"Unsupported EEG format '{ext}' (want .set or .vhdr): {path}")





# --------------------------------------------------------------------------- #

# Reports                                                                      #

# --------------------------------------------------------------------------- #

def report_basics(raw: mne.io.BaseRaw, set_path: str) -> float:

    _sep()

    print("BASICS")

    _sep()

    sfreq = float(raw.info["sfreq"])

    n_ch = len(raw.ch_names)

    dur_s = raw.n_times / sfreq

    print(f"file            : {set_path}")

    print(f"basename        : {os.path.basename(set_path)}")

    print(f"sfreq           : {sfreq:g} Hz   (expected 250)")

    print(f"n_channels      : {n_ch}         (expected ~59-62; varies per subject)")

    print(f"n_times         : {raw.n_times}")

    print(f"duration        : {dur_s:.1f} s  ({dur_s/60:.2f} min)")

    ch_types = sorted(set(raw.get_channel_types()))

    print(f"channel types   : {ch_types}")

    if sfreq == 2500:

        print("  ! 2500 Hz = RAW LEMON EEG. The preprocessed derivative is "

              "250 Hz — you are pointed at the raw lineage.")

    elif sfreq != 250:

        print("  ! sfreq is neither 250 (preprocessed) nor 2500 (raw) — a few "

              "LEMON subjects differ. Flag or resample.")

    if any(t != "eeg" for t in raw.get_channel_types()):

        print("  ! non-EEG channels present (e.g. EOG). Decide whether they are "

              "nodes or should be dropped before windowing.")

    return sfreq





def report_channels(raw: mne.io.BaseRaw) -> None:

    _sep()

    print("CHANNELS")

    _sep()

    names = raw.ch_names

    for i in range(0, len(names), 8):

        print("  " + "  ".join(f"{n:>6}" for n in names[i:i + 8]))

    has_pos = raw.get_montage() is not None

    print(f"montage present : {has_pos}")

    occ = [c for c in raw.ch_names if c in OCC_CANDIDATES]

    print(f"occipital chans : {occ if occ else '(none of the usual O/PO labels found)'}")





def report_annotations(raw: mne.io.BaseRaw) -> None:

    _sep()

    print("ANNOTATIONS / EVENTS")

    _sep()

    ann = raw.annotations

    if len(ann) == 0:

        print("  (no annotations at all — condition is likely encoded in the "

              "filename, not in markers. Confirm _EC/_EO naming.)")

        return

    descs = list(ann.description)

    uniq, counts = np.unique(descs, return_counts=True)

    print("unique descriptions (count):")

    for d, c in sorted(zip(uniq, counts), key=lambda t: -t[1]):

        tag = ""

        if BOUNDARY_SUBSTR in d.lower():

            tag = "  <- boundary (block/concat seam)"

        elif EC_MARKER_SUBSTR in d:

            tag = "  <- EC marker? (documented S210=eyes-closed)"

        elif EO_MARKER_SUBSTR in d:

            tag = "  <- EO marker? (documented S200=eyes-open)"

        print(f"  {d!r:>24} : {c}{tag}")



    n_boundary = sum(1 for d in descs if BOUNDARY_SUBSTR in d.lower())

    print(f"\nboundary events : {n_boundary}  "

          f"(expect >0: blocks were segmented then concatenated)")



    print("\nfirst events (onset_s, dur_s, description):")

    for onset, dur, d in list(zip(ann.onset, ann.duration, ann.description))[:40]:

        print(f"  {onset:8.2f}  {dur:6.2f}  {d!r}")

    if len(ann) > 40:

        print(f"  ... ({len(ann) - 40} more)")





# --------------------------------------------------------------------------- #

# Condition segmentation (the load-bearing part for the loader)                #

# --------------------------------------------------------------------------- #

def contiguous_segments(raw: mne.io.BaseRaw) -> list[tuple[int, int]]:

    """Split the recording into contiguous runs bounded by `boundary` events.



    Returns a list of (start_sample, end_sample_exclusive). A forecasting window

    must live entirely inside one of these — this is the whole 'don't cross a

    boundary' requirement, isolated in one place.

    """

    sfreq = raw.info["sfreq"]

    n = raw.n_times

    cut_samples = sorted(

        int(round(onset * sfreq))

        for onset, d in zip(raw.annotations.onset, raw.annotations.description)

        if BOUNDARY_SUBSTR in d.lower()

    )

    edges = [0] + [c for c in cut_samples if 0 < c < n] + [n]

    segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)

            if edges[i + 1] > edges[i]]

    return segs





def condition_intervals(raw: mne.io.BaseRaw, want: str,

                        set_path: str) -> list[tuple[int, int]]:

    """Return sample intervals belonging to the requested condition ('EC'/'EO').



    Three cases, resolved in priority order:

      (a) filename says the whole file is one condition (pre-split _EC/_EO)

      (b) S200/S210 markers present -> segments tagged by the nearest preceding

          condition marker

      (c) neither -> cannot determine; caller must decide

    """

    substr = EC_MARKER_SUBSTR if want == "EC" else EO_MARKER_SUBSTR

    base = os.path.basename(set_path).lower()

    segs = contiguous_segments(raw)



    # (a) pre-split file

    if f"_{want.lower()}" in base or base.endswith(f"{want.lower()}.set"):

        print(f"  condition source: FILENAME says whole file is {want} "

              f"({len(segs)} boundary-free segment(s))")

        return segs



    # (b) marker-tagged single concatenated file

    sfreq = raw.info["sfreq"]

    cond_marks = sorted(

        (int(round(o * sfreq)), d)

        for o, d in zip(raw.annotations.onset, raw.annotations.description)

        if substr in d

    )

    if cond_marks:

        print(f"  condition source: MARKERS ({len(cond_marks)} '{substr}' "

              f"onsets found)")

        # A block runs from each condition marker to the next boundary/segment end.

        seg_starts = np.array([s for s, _ in segs])

        seg_ends = np.array([e for _, e in segs])

        out: list[tuple[int, int]] = []

        for start, _ in cond_marks:

            j = np.searchsorted(seg_ends, start, side="right")

            if j < len(segs):

                out.append((max(start, seg_starts[j]), seg_ends[j]))

        return out



    # (c) unknown

    print(f"  condition source: UNKNOWN — no '_{want}' in filename and no "

          f"'{substr}' markers. Cannot delimit {want} automatically.")

    return []





# --------------------------------------------------------------------------- #

# Alpha sanity check — resolves the S200/S210 ambiguity from physiology.       #

# --------------------------------------------------------------------------- #

def _band_ratio(data: np.ndarray, sfreq: float) -> float:

    """Mean occipital alpha-band power fraction over a set of channels."""

    if data.shape[1] < int(sfreq):  # need at least ~1 s

        return float("nan")

    freqs, psd = welch(data, fs=sfreq, nperseg=min(int(sfreq * 2), data.shape[1]))

    alpha = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])

    broad = (freqs >= 1.0) & (freqs <= 45.0)

    with np.errstate(invalid="ignore", divide="ignore"):

        frac = psd[:, alpha].sum(axis=1) / psd[:, broad].sum(axis=1)

    return float(np.nanmean(frac))





def _gather(raw: mne.io.BaseRaw, intervals: list[tuple[int, int]],

            picks: list[str]) -> np.ndarray:

    if not intervals or not picks:

        return np.empty((len(picks), 0))

    idx = mne.pick_channels(raw.ch_names, picks, ordered=True)

    full = raw.get_data(picks=idx)

    return np.concatenate([full[:, s:e] for s, e in intervals], axis=1)





def alpha_check(raw: mne.io.BaseRaw, set_path: str) -> None:

    _sep()

    print("ALPHA SANITY CHECK  (EC should have HIGHER occipital alpha)")

    _sep()

    occ = [c for c in raw.ch_names if c in OCC_CANDIDATES]

    if not occ:

        occ = raw.ch_names  # fall back to all channels, with a caveat

        print("  (no occipital labels found; using ALL channels — the EC>EO "

              "contrast will be weaker.)")

    sfreq = raw.info["sfreq"]

    ec = _gather(raw, condition_intervals(raw, "EC", set_path), occ)

    eo = _gather(raw, condition_intervals(raw, "EO", set_path), occ)

    ec_a = _band_ratio(ec, sfreq) if ec.shape[1] else float("nan")

    eo_a = _band_ratio(eo, sfreq) if eo.shape[1] else float("nan")

    print(f"  EC alpha fraction : {ec_a:.3f}  ({ec.shape[1]} samples)")

    print(f"  EO alpha fraction : {eo_a:.3f}  ({eo.shape[1]} samples)")

    if np.isnan(ec_a) or np.isnan(eo_a):

        print("  ! Could not compute both conditions — likely a pre-split file "

              "or missing markers. Verify EC identity manually.")

    elif ec_a > eo_a:

        print("  OK: EC > EO alpha — the marker->condition mapping looks correct.")

    else:

        print("  ! EC <= EO alpha — the S200/S210 labels may be SWAPPED in this "

              "file. Trust the alpha result, not the marker string.")





# --------------------------------------------------------------------------- #

# Windowing preview                                                            #

# --------------------------------------------------------------------------- #

def _windows_in_segment(seg_len: int, win: int, stride: int) -> int:

    return 0 if seg_len < win else (seg_len - win) // stride + 1





def windowing_preview(raw: mne.io.BaseRaw, set_path: str, x: int, y: int,

                      stride: int, decimate: int) -> None:

    _sep()

    print("WINDOWING PREVIEW (EC only, boundary-respecting)")

    _sep()

    sfreq = raw.info["sfreq"]

    win = x + y

    ec_intervals = condition_intervals(raw, "EC", set_path)

    if not ec_intervals:

        print("  No EC intervals resolved — cannot preview windows.")

        return



    seg_lens = [e - s for s, e in ec_intervals]

    total_windows = sum(_windows_in_segment(L, win, stride) for L in seg_lens)

    span_ms = 1000.0 * win / sfreq

    print(f"  x={x} context, y={y} horizon, window={win}, stride={stride}")

    print(f"  EC segments        : {len(seg_lens)}  "

          f"(lengths in samples: {seg_lens[:8]}{' ...' if len(seg_lens) > 8 else ''})")

    print(f"  EC windows (subj)  : {total_windows}")

    print(f"  one window spans   : {span_ms:.0f} ms at {sfreq:g} Hz")

    if span_ms < 500:

        print(f"    ! {span_ms:.0f} ms is under one alpha cycle-ish of context; "

              f"compare against fMRI's ~120 s window. Consider decimation.")



    if decimate > 1:

        dec_sfreq = sfreq / decimate

        dec_lens = [L // decimate for L in seg_lens]

        dec_windows = sum(_windows_in_segment(L, win, stride) for L in dec_lens)

        print(f"\n  --- with decimate x{decimate} (effective {dec_sfreq:g} Hz) ---")

        print(f"  EC windows (subj)  : {dec_windows}")

        print(f"  one window spans   : {1000.0 * win / dec_sfreq:.0f} ms")



    # The actual guarantee we care about: prove no window crosses a boundary.

    print("\n  boundary-crossing check: windows are cut strictly within each "

          "segment above, so none straddle a boundary. PASS by construction.")





# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:

    ap = argparse.ArgumentParser(description=__doc__,

                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    src = ap.add_mutually_exclusive_group(required=True)

    src.add_argument("--set-file", type=str, help="direct path to one .set file")

    src.add_argument("--eeg-root", type=str,

                     help="BIDS-ID preprocessed EEG root (use with --subject)")

    ap.add_argument("--subject", type=str, help="e.g. sub-010002")

    ap.add_argument("--x", type=int, default=30, help="context length")

    ap.add_argument("--y", type=int, default=10, help="forecast horizon")

    ap.add_argument("--stride", type=int, default=10)

    ap.add_argument("--decimate", type=int, default=1,

                    help="preview window budget at 250/decimate Hz")

    args = ap.parse_args()

    if args.eeg_root and not args.subject:

        ap.error("--eeg-root requires --subject")

    return args





def main() -> None:

    args = parse_args()

    set_path = resolve_eeg_path(args)

    raw = load_raw(set_path)



    report_basics(raw, set_path)

    report_channels(raw)

    report_annotations(raw)

    alpha_check(raw, set_path)

    windowing_preview(raw, set_path, args.x, args.y, args.stride, args.decimate)



    _sep()

    print("DONE — verify before designing the manifest:")

    print("  [ ] sfreq == 250 (else flag/resample)")

    print("  [ ] channel count read from data, not hardcoded")

    print("  [ ] boundary events present and EC segments delimited correctly")

    print("  [ ] alpha check says EC > EO (marker mapping is right)")

    print("  [ ] EC window budget per subject is sane for your compute plan")

    _sep()





if __name__ == "__main__":

    main()
