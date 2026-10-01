from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data.rbc_dataset import make_dataloaders, schaefer_centroids
from data.lemon_dataset import (
    make_lemon_dataloaders,
    make_lemon_run_loader,
    montage_positions,
)

from model.braindyn import BrainDyn, BrainDynConfig
from model.graph_builders import (
    adjacency_to_edge_index,
    build_corr_graph_from_series,
    build_granger_graph_from_series,
    build_lap_pe,
    build_sc_graph,
    build_spatial_graph_from_positions,
    corr_score_from_series,
    granger_score_from_series,
    threshold_granger_scores,
)
from model.losses import dtw_mean_normalized, total_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_node_series(loader, max_batches, norm_stats=None):
    """Collect independent context windows as (B_total, T, N) for Granger."""
    chunks: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        chunks.append(
            normalize_with_stats(batch["x"].float(), norm_stats)
        )  # (B, Lx, N)

    if not chunks:
        raise RuntimeError("Unable to build graph: train loader produced zero batches.")

    return torch.cat(chunks, dim=0)


def _granger_prior_from_subject_series(
    subject_series,
    lag: int,
    threshold: float,
    threshold_mode: str,
    topk_edges: int,
    topk_per_node: int = 0,
):
    score_sum = None
    n_subjects = 0
    total_windows = 0
    for _sid, series in subject_series:
        if series.shape[0] == 0:
            continue
        score_sum = (
            granger_score_from_series(series, lag=lag)
            if score_sum is None
            else score_sum + granger_score_from_series(series, lag=lag)
        )
        n_subjects += 1
        total_windows += int(series.shape[0])

    if score_sum is None:
        raise RuntimeError(
            "Granger graph: no subject windows collected from the train fold."
        )

    score = score_sum / float(n_subjects)
    adjacency, threshold_used = threshold_granger_scores(
        score=score,
        threshold=threshold,
        threshold_mode=threshold_mode,
        topk_edges=topk_edges,
        topk_per_node=topk_per_node,
    )
    edge_index = adjacency_to_edge_index(adjacency)
    if edge_index.shape[1] == 0:
        raise RuntimeError(
            f"Granger graph has no edges at threshold={threshold:.4f}. "
            "Lower --granger_threshold or use --granger_threshold_mode topk."
        )
    info: dict[str, float | int | str] = {
        "mode": threshold_mode,
        "threshold": float(threshold_used),
        "lag": int(lag),
        "topk_edges": int(topk_edges),
        "topk_per_node": int(topk_per_node),
        "selected_edges": int(edge_index.shape[1]),
        "n_subjects": int(n_subjects),
        "total_windows": int(total_windows),
    }
    return edge_index, score, info


def _grouped_subject_series(dataset, indices, groups, max_windows_per_subject):
    """Yield ``(subject_id, (n_win, x, N))`` window stacks in deterministic order.

    Run with ``--cache`` or the disk I/O will dominate the score computation.
    """
    groups = np.asarray(groups)

    by_subject: dict[str, list[int]] = {}
    for i in indices:
        by_subject.setdefault(str(groups[int(i)]), []).append(int(i))

    for sid in sorted(by_subject):
        # Ascending global index == window order within a subject (the dataset
        # appends windows per subject in block/time order); a fixed cap keeps
        # the first-K in that order, so the draw is deterministic.
        sidx = sorted(by_subject[sid])
        if max_windows_per_subject is not None:
            sidx = sidx[:max_windows_per_subject]
        wins = []
        for i in sidx:
            ds, local = _resolve_concat_sample(dataset, i)
            path, _meta, t = ds._samples[local]
            ts = ds._load_cached(path)
            wins.append(
                torch.from_numpy(ts[t : t + ds.x].astype(np.float32, copy=True))
            )
        yield sid, torch.stack(wins, dim=0)  # (n_win, x, N)


def build_granger_graph_grouped(
    dataset,
    indices,
    groups,
    threshold: float,
    lag: int,
    threshold_mode: str,
    topk_edges: int,
    topk_per_node: int = 0,
    max_windows_per_subject: int | None = None,
):
    """computes from all subjects
    run with ``--cache`` or the disk I/O will dominate the score computation
    """
    return _granger_prior_from_subject_series(
        _grouped_subject_series(dataset, indices, groups, max_windows_per_subject),
        lag=lag,
        threshold=threshold,
        threshold_mode=threshold_mode,
        topk_edges=topk_edges,
        topk_per_node=topk_per_node,
    )


def _corr_prior_from_subject_series(
    subject_series,
    threshold: float,
    threshold_mode: str,
    topk_edges: int,
    topk_per_node: int = 0,
    absolute: bool = True,
):
    """Subject-averaged |Pearson| functional-connectivity prior (mirror of granger)."""
    score_sum = None
    n_subjects = 0
    total_windows = 0
    for _sid, series in subject_series:
        if series.shape[0] == 0:
            continue
        s = corr_score_from_series(series, absolute=absolute)
        score_sum = s if score_sum is None else score_sum + s
        n_subjects += 1
        total_windows += int(series.shape[0])

    if score_sum is None:
        raise RuntimeError(
            "Correlation graph: no subject windows collected from the train fold."
        )

    score = score_sum / float(n_subjects)
    adjacency, threshold_used = threshold_granger_scores(
        score=score,
        threshold=threshold,
        threshold_mode=threshold_mode,
        topk_edges=topk_edges,
        topk_per_node=topk_per_node,
    )
    edge_index = adjacency_to_edge_index(adjacency)
    if edge_index.shape[1] == 0:
        raise RuntimeError(
            f"Correlation graph has no edges at threshold={threshold:.4f}. "
            "Lower --granger_threshold or use --granger_threshold_mode topk."
        )
    info: dict[str, float | int | str] = {
        "mode": threshold_mode,
        "threshold": float(threshold_used),
        "topk_edges": int(topk_edges),
        "topk_per_node": int(topk_per_node),
        "absolute": bool(absolute),
        "selected_edges": int(edge_index.shape[1]),
        "n_subjects": int(n_subjects),
        "total_windows": int(total_windows),
    }
    return edge_index, score, info


def build_corr_graph_grouped(
    dataset,
    indices,
    groups,
    threshold: float,
    threshold_mode: str,
    topk_edges: int,
    topk_per_node: int = 0,
    absolute: bool = True,
    max_windows_per_subject: int | None = None,
):
    """Subject-fair functional-connectivity prior over the whole train fold."""
    return _corr_prior_from_subject_series(
        _grouped_subject_series(dataset, indices, groups, max_windows_per_subject),
        threshold=threshold,
        threshold_mode=threshold_mode,
        topk_edges=topk_edges,
        topk_per_node=topk_per_node,
        absolute=absolute,
    )


def build_granger_graph(
    loader,
    threshold: float,
    lag: int,
    threshold_mode: str,
    topk_edges: int,
    topk_per_node: int = 0,
    norm_stats=None,
    max_batches=None,
    max_windows_per_subject: int | None = None,
):
    if max_batches is not None:
        warnings.warn(
            "build_granger_graph: max_batches is ignored; the prior is now built "
            "from the whole loader (subject-grouped) to remove shuffle-seed wobble.",
            stacklevel=2,
        )

    by_subject: dict[str, list[tuple[tuple, torch.Tensor]]] = {}
    for batch in loader:
        xb = normalize_with_stats(batch["x"].float(), norm_stats)  # (B, T, N)
        meta = batch["meta"]
        sids = meta["subject_id"]
        paths = meta.get("path", [None] * len(sids))
        tstarts = meta.get("t_start", [0] * len(sids))
        for i in range(len(sids)):
            key = (paths[i], int(tstarts[i]))  # deterministic within-subject order
            by_subject.setdefault(str(sids[i]), []).append((key, xb[i]))

    def _subject_series():
        for sid in sorted(by_subject):
            items = sorted(by_subject[sid], key=lambda kv: kv[0])
            if max_windows_per_subject is not None:
                items = items[:max_windows_per_subject]
            yield sid, torch.stack([w for _k, w in items], dim=0)

    return _granger_prior_from_subject_series(
        _subject_series(),
        lag=lag,
        threshold=threshold,
        threshold_mode=threshold_mode,
        topk_edges=topk_edges,
        topk_per_node=topk_per_node,
    )


def load_square_matrix_csv(path: str) -> torch.Tensor:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{path} must be a square (N, N) matrix, got shape {arr.shape}"
        )
    return torch.from_numpy(arr).float()


def load_vector_csv(path: str) -> torch.Tensor:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64).reshape(-1)
    return torch.from_numpy(arr).float()


def _resolve_concat_sample(dataset, idx: int):
    if hasattr(dataset, "datasets") and hasattr(dataset, "cumulative_sizes"):
        dataset_idx = 0
        for cumsum in dataset.cumulative_sizes:
            if idx < cumsum:
                break
            dataset_idx += 1
        prev = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        return dataset.datasets[dataset_idx], idx - prev
    return dataset, idx


def compute_train_global_stats(dataset, indices) -> dict[str, torch.Tensor]:
    sums = None
    sq_sums = None
    count = 0
    for idx in (
        indices
    ):  # indices only from train fold to avoid leakage in mean/std computation
        ds, local_idx = _resolve_concat_sample(dataset, int(idx))
        path, _meta, t = ds._samples[local_idx]
        ts = ds._load_cached(path)
        ctx = ts[t : t + ds.x].astype(np.float64, copy=False)
        ctx_sum = ctx.sum(axis=0)
        ctx_sq_sum = np.square(ctx).sum(axis=0)
        sums = ctx_sum if sums is None else sums + ctx_sum
        sq_sums = ctx_sq_sum if sq_sums is None else sq_sums + ctx_sq_sum
        count += ctx.shape[0]

    if count == 0 or sums is None or sq_sums is None:
        raise RuntimeError("Cannot compute train-global stats from an empty fold.")

    mean = sums / count
    var = np.maximum(sq_sums / count - np.square(mean), 1e-12)
    std = np.sqrt(var).clip(1e-6)
    return {
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "std": torch.from_numpy(std.astype(np.float32)),
    }


def normalize_with_stats(
    x: torch.Tensor,
    norm_stats: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if norm_stats is None:
        return x
    mean = norm_stats["mean"].to(device=x.device, dtype=x.dtype)
    std = norm_stats["std"].to(device=x.device, dtype=x.dtype)
    view_shape = [1] * x.ndim
    view_shape[-1] = mean.numel()
    return (x - mean.view(*view_shape)) / std.view(*view_shape)


def batch_to_model_tensors(
    batch: dict,
    device: torch.device,
    norm_stats: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_ctx = batch["x"].to(device=device, dtype=torch.float32)
    y_future = batch["y"].to(device=device, dtype=torch.float32)
    if norm_stats is not None:
        x_ctx = normalize_with_stats(x_ctx, norm_stats)
        y_future = normalize_with_stats(y_future, norm_stats)

    x_history = x_ctx.permute(0, 2, 1).unsqueeze(-1)
    y_true = y_future.permute(1, 0, 2).unsqueeze(-1)
    return x_history, y_true


def channel_exclusion_mask(
    node_idx: torch.Tensor, num_nodes: int, device: torch.device
) -> torch.Tensor:
    """``(1, B, N, 1)`` bool mask, False at each sample's excluded channel.

    ``node_idx`` is a ``(B,)`` integer tensor -- one channel index per sample, so the
    excluded channel differs across the batch and a plain slice will not do.

    The rank-4 shape is what ``model.losses.total_loss`` needs: it does
    ``mask.expand_as(x_pred)`` against ``(T, B, N, F)``, which requires the mask to
    already be 4-D with every dim either 1 or matching. It also broadcasts unchanged in
    ``run_test_rollout_chunks``, where B == 1.
    """
    idx = node_idx.to(device=device, dtype=torch.long).reshape(-1)
    keep = torch.ones(idx.shape[0], num_nodes, dtype=torch.bool, device=device)
    keep[torch.arange(idx.shape[0], device=device), idx] = False
    return keep[None, :, :, None]


def apply_channel_mask(
    y_pred_np: np.ndarray, y_true_np: np.ndarray, mask_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop masked ``(sample, channel, feature)`` sequences, keeping the time axis.

    ``(T, B, N, F) -> (T, K, 1, 1)`` where K is the number of surviving sequences.

    Rank 4 out is deliberate rather than tidy: every downstream metric then works
    untouched. ``dtw_mean_normalized`` reshapes to ``(T, prod(shape[1:]))`` and the
    caller's ``batch_sequences = prod(shape[1:])`` both come out as K, and the
    ``.reshape(-1)`` feeding pcc/scc stays correct. Selecting sequences (not elements)
    is valid because the mask is constant along time and feature.

    Mirrors the selection idiom in ``run_epoch_ar_train::_flush``, the only other place
    in this repo that computes metrics over a subset.
    """
    T = y_pred_np.shape[0]
    keep = np.broadcast_to(mask_np, y_pred_np.shape).reshape(T, -1)[0]
    p = y_pred_np.reshape(T, -1)[:, keep]
    t = y_true_np.reshape(T, -1)[:, keep]
    return p[:, :, None, None], t[:, :, None, None]


def _clamp_ar_feedback(t: torch.Tensor) -> torch.Tensor:
    """Clamp a value before it's recycled as future context in an
    autoregressive rollout. With --ss_start/--ss_end 0.0 (every production
    long-horizon script's setting), training is fully free-running from
    epoch 1 of a randomly-initialized model -- its own predictions, not
    ground truth, get fed back as history every step. An early, poorly-
    trained prediction that drifts outside the ~[-20, 20] normalized range
    (see data/sn_dataset.py::_zscore_pair_from_reference) would otherwise
    compound across many chained steps and overflow fp16 under --amp. This
    only clips an already-diverging rollout; it's a no-op for any prediction
    within a normal range. nan_to_num first: clamp alone leaves an actual
    NaN untouched, and a NaN fed back as "history" would poison every
    subsequent step forever, not just the one that produced it."""
    return torch.nan_to_num(t).clamp(-20.0, 20.0)


def _safe_corr(pred_flat: np.ndarray, true_flat: np.ndarray) -> tuple[float, float]:
    """Pearson/Spearman correlation, treating a constant array as "no signal"
    (0.0) rather than NaN. Real data (fMRI/EEG) rarely hits this, but sparse
    spike-derived NEST rates can legitimately be all-zero over a short
    window/batch, where correlation is mathematically undefined -- that's a
    degenerate-batch artifact, not evidence of a training blowup, so it
    shouldn't poison the epoch's running average or trip the finite-metric
    guard the way a real NaN loss should."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        pcc_val, _ = pearsonr(pred_flat, true_flat)
        scc_val, _ = spearmanr(pred_flat, true_flat)
    pcc_val = 0.0 if not np.isfinite(pcc_val) else float(pcc_val)
    scc_val = 0.0 if not np.isfinite(scc_val) else float(scc_val)
    return pcc_val, scc_val


class SubjectRunDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        indices,
        run_loader,
        use_cache: bool = False,
        segment_len: int | None = None,
        segments_per_subject: int | None = None,
        segment_seed: int = 0,
    ):
        seen_paths = {}
        for idx in sorted(int(i) for i in indices):
            ds, local_idx = _resolve_concat_sample(base_dataset, idx)
            path, meta, _ = ds._samples[local_idx]
            if path not in seen_paths:
                seen_paths[path] = meta
        self._runs = [(path, meta) for path, meta in seen_paths.items()]
        self._run_loader = run_loader
        self._use_cache = use_cache
        self._cache: dict[str, torch.Tensor] = {}

        self._segment_len = segment_len
        self._seg_index: list[tuple[int, int]] | None = None
        if segment_len is not None and segments_per_subject is not None:
            self._build_segment_index(
                int(segment_len), int(segments_per_subject), int(segment_seed)
            )

    def _load_run(self, path) -> torch.Tensor:
        if self._use_cache and path in self._cache:
            return self._cache[path]
        ts_t = torch.from_numpy(self._run_loader(path))  # (T, N) float32
        if self._use_cache:
            self._cache[path] = ts_t
        return ts_t

    def _build_segment_index(self, L: int, K: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._seg_index = []
        short_runs = 0
        for run_idx, (path, _meta) in enumerate(self._runs):
            T = int(self._load_run(path).shape[0])
            if T < L:
                short_runs += 1
                continue
            k_run = min(K, T // L)  # cap so the k_run segments stay non-overlapping
            if k_run < K:
                short_runs += 1
            bin_size = T // k_run  # >= L by construction
            for b in range(k_run):
                lo = b * bin_size
                hi = lo + bin_size - L  # inclusive last valid start in this bin
                start = int(rng.integers(lo, hi + 1)) if hi > lo else lo
                self._seg_index.append((run_idx, start))
        if short_runs:
            print(
                f"[SubjectRunDataset] {short_runs} run(s) shorter than "
                f"{K} x {L} non-overlapping timepoints; used all that fit."
            )

    def __len__(self):
        if self._seg_index is not None:
            return len(self._seg_index)
        return len(self._runs)

    def __getitem__(self, idx):
        if self._seg_index is not None:
            run_idx, start = self._seg_index[idx]
            path, meta = self._runs[run_idx]
            seg = self._load_run(path)[start : start + self._segment_len]
            return {"ts": seg, "meta": meta}
        path, meta = self._runs[idx]
        return {"ts": self._load_run(path), "meta": meta}


def pad_collate_runs(batch):
    # collects variable length batches and enforces max length limits (used in eeg data)
    lengths = torch.tensor([b["ts"].shape[0] for b in batch], dtype=torch.long)
    n = int(batch[0]["ts"].shape[1])
    t_max = int(lengths.max())
    ts = torch.zeros(len(batch), t_max, n, dtype=torch.float32)
    for i, b in enumerate(batch):
        li = int(b["ts"].shape[0])
        ts[i, :li] = b["ts"].to(torch.float32)
    return {"ts": ts, "lengths": lengths, "meta": [b["meta"] for b in batch]}


def normalize_ar_run(
    ts: torch.Tensor,
    norm_stats: dict[str, torch.Tensor] | None,
    *,
    x: int | None = None,
    norm_mode: str = "train_global",
) -> torch.Tensor:
    """Normalize a full AR run.

    ``train_global`` z-scores each node with statistics from the training fold.
    It is the default, the setting every paper run uses, and the only mode
    main.py's own CLI exposes. ``context`` instead z-scores against the run's
    first ``x`` steps; it is reachable only through
    train/train_nest_braindyn.py's --norm_mode.
    """
    if norm_mode == "context":
        if x is None:
            raise ValueError("x (context length) is required for norm_mode='context'")
        ctx = ts[:, :x, :]
        mean = ctx.mean(dim=1, keepdim=True)
        std = ctx.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        # See data/sn_dataset.py::_zscore_pair_from_reference for why this is
        # clamped: a near-constant context window floors std at 1e-6, and a
        # small subsequent deviation then blows up to a magnitude that
        # overflows fp16 under --amp. nan_to_num first: clamp alone leaves an
        # actual NaN/Inf (e.g. from corrupted source data) untouched.
        return torch.nan_to_num((ts - mean) / std).clamp(-20.0, 20.0)
    if norm_mode == "train_global":
        if norm_stats is None:
            raise ValueError("fold-training normalization statistics are required")
        return normalize_with_stats(ts, norm_stats)
    raise ValueError(f"Unknown norm_mode={norm_mode!r}")


def teacher_forcing_probability(
    start: float,
    end: float,
    epoch: int,
    epochs: int,
    decay_epochs: int | None = None,
) -> float:
    if not (0.0 <= start <= 1.0):
        raise ValueError(f"ss_start must be in [0, 1], got {start}")
    if not (0.0 <= end <= 1.0):
        raise ValueError(f"ss_end must be in [0, 1], got {end}")
    schedule_epochs = epochs if decay_epochs is None else int(decay_epochs)
    if schedule_epochs < 0:
        raise ValueError(f"ss_decay_epochs must be >= 0, got {schedule_epochs}")
    if schedule_epochs == 0:
        return float(end)
    if schedule_epochs <= 1:
        return float(start if epoch <= 1 else end)
    mix = min(max((epoch - 1) / float(schedule_epochs - 1), 0.0), 1.0)
    return float(start + (end - start) * mix)


def run_epoch_ar_train(
    model,
    subject_loader,
    edge_index,
    dt,
    x,
    chunk_size,
    tbptt_chunks,
    optimizer,
    lambda_mse,
    lambda_mae,
    grad_clip,
    desc,
    scaler=None,
    teacher_forcing_prob: float = 1.0,
    norm_mode: str = "train_global",
    norm_stats: dict[str, torch.Tensor] | None = None,
    stride: int | None = None,
):
    """Full run-level autoregressive training with truncated BPTT and scheduled sampling.

    For each subject run:
      - Use first x timepoints as initial context.
      - Roll forward in TBPTT windows of chunk_size*tbptt_chunks timepoints,
        `stride` timepoints apart.
      - With probability teacher_forcing_prob at each time point/sample, feed
        ground truth back into context; otherwise feed model predictions.

    `stride` controls how far the next TBPTT window starts past the previous
    one. It defaults to chunk_size*tbptt_chunks, i.e. back-to-back windows
    that walk every timepoint of every run exactly once. Passing a larger
    stride skips the gap between windows instead of visiting it, which is
    what actually bounds per-epoch compute on long runs.
    """
    if not (0.0 <= teacher_forcing_prob <= 1.0):
        raise ValueError(
            f"teacher_forcing_prob must be in [0, 1], got {teacher_forcing_prob}"
        )
    burst_span = chunk_size * tbptt_chunks
    if stride is None:
        stride = burst_span
    if stride < chunk_size:
        raise ValueError(f"stride must be >= chunk_size, got stride={stride} chunk_size={chunk_size}")
    extra_skip = max(stride - burst_span, 0)

    is_train = optimizer is not None
    use_amp = scaler is not None
    model.train(is_train)

    device = edge_index.device
    total_running = torch.zeros((), device=device)
    mse_running = torch.zeros((), device=device)
    mae_running = torch.zeros((), device=device)
    dtw_running = 0.0
    n_chunks = 0
    n_elements = 0
    n_dtw = 0
    # Accumulate flattened valid predictions/targets for epoch-level PCC/SCC.
    pred_accum: list[np.ndarray] = []
    true_accum: list[np.ndarray] = []

    # Chunk buffers flushed to the host in bulk (one sync per ~FLUSH_CHUNKS
    # chunks instead of per chunk) so the batched GPU work is not serialised by
    # per-step .cpu() copies. Also bounds transient memory to FLUSH_CHUNKS.
    FLUSH_CHUNKS = 256
    pred_buf: list[torch.Tensor] = []
    true_buf: list[torch.Tensor] = []
    mask_buf: list[torch.Tensor] = []

    def _flush():
        nonlocal dtw_running, n_dtw
        if not pred_buf:
            return
        p_np = torch.stack(pred_buf).cpu().numpy()  # (C, chunk, B, N, 1)
        t_np = torch.stack(true_buf).cpu().numpy()
        m_np = torch.stack(mask_buf).cpu().numpy()  # (C, chunk, B, 1, 1) bool
        valid_flat = np.broadcast_to(m_np, p_np.shape).reshape(-1)
        pred_accum.append(p_np.reshape(-1)[valid_flat])
        true_accum.append(t_np.reshape(-1)[valid_flat])
        # Per-chunk DTW over the fully-in-length sub-batch (diagnostic only).
        for c in range(p_np.shape[0]):
            col = m_np[c, :, :, 0, 0].all(axis=0)  # (B,) runs whose whole chunk is real
            if not col.any():
                continue
            dtw_running += dtw_mean_normalized(p_np[c][:, col], t_np[c][:, col])
            n_dtw += 1
        pred_buf.clear()
        true_buf.clear()
        mask_buf.clear()

    pbar = tqdm(subject_loader, desc=desc, leave=False)
    for batch in pbar:
        ts = batch["ts"].to(device, dtype=torch.float32)  # (B, T, N)
        ts = normalize_ar_run(ts, norm_stats=norm_stats, x=x, norm_mode=norm_mode)
        B = ts.shape[0]
        T = ts.shape[1]
        N = ts.shape[2]

        lengths = batch.get("lengths")
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
        else:
            lengths = lengths.to(device)

        if T < x + chunk_size:
            continue  # even the longest run in this batch is too short

        chunk_idx = 0
        # Randomize the starting phase (only while training) so that, across
        # epochs, the fixed skip pattern below eventually walks the whole
        # run instead of always skipping the exact same gaps -- otherwise
        # anything in a skipped gap would never be trained on, ever. Eval
        # stays at a fixed phase (0) so val metrics are comparable epoch to
        # epoch for the scheduler/checkpoint selection.
        phase = 0
        if is_train and extra_skip > 0:
            # extra_skip is derived from --ar_stride, which can be set far
            # larger than any real run's length (e.g. --ar_stride 100000 to
            # make long_ar_train visit ~1 window per run instead of walking
            # the whole thing). Drawing phase from the full [0, extra_skip]
            # range ignores that and overwhelmingly lands t = x + phase past
            # T, so the while loop below never runs even once -- silently
            # contributing zero chunks for that batch. Cap it to what this
            # run can actually support.
            max_phase = min(extra_skip, T - x - chunk_size)
            if max_phase > 0:
                phase = int(torch.randint(0, max_phase + 1, (1,)).item())
        t = x + phase  # current position in the run

        # Initial context: (B, N, x, 1) -- the x real timepoints immediately
        # before the (possibly phase-shifted) start position.
        hist = ts[:, t - x : t, :].permute(0, 2, 1).unsqueeze(-1)

        while t + chunk_size <= T:
            # Zero gradients and re-anchor the window's chunk count at the
            # start of a TBPTT window; held constant for every chunk in it
            # (a shrinking per-chunk divisor would over-weight the last chunk
            # of every window relative to the first).
            if chunk_idx % tbptt_chunks == 0:
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                chunks_in_window = min(tbptt_chunks, (T - t) // chunk_size)

            steps = torch.arange(t, t + chunk_size, device=device)
            valid = steps[:, None] < lengths[None, :]  # (chunk_size, B)
            if not bool(valid.any()):
                break  # every run in the batch is exhausted past here
            mask = valid[:, :, None, None]

            # Ground truth for this chunk: (chunk_size, B, N, 1)
            y_chunk = ts[:, t : t + chunk_size, :]
            y_chunk = y_chunk.permute(1, 0, 2).unsqueeze(-1)

            with torch.set_grad_enabled(is_train):
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(
                        x_history=hist,
                        edge_index=edge_index,
                        pred_steps=chunk_size,
                        dt=dt,
                        autoregressive=False,
                    )
                    y_pred = out["x_pred"]  # (chunk_size, B, N, 1)
                    losses = total_loss(
                        y_pred,
                        y_chunk,
                        lambda_mse=lambda_mse,
                        lambda_mae=lambda_mae,
                        mask=mask,
                    )
                    # chunks_in_window is fixed once per TBPTT window (set at the
                    # top of the loop), so every chunk in the window is weighted
                    # equally AND a short final window is normalized by its real
                    # width rather than the full tbptt_chunks.
                    loss = losses["total"] / chunks_in_window

                if is_train:
                    # Keep graph for intermediate chunks so gradients can flow
                    # through the full TBPTT window; free it at the window end.
                    will_step_now = ((chunk_idx + 1) % tbptt_chunks == 0) or (
                        t + 2 * chunk_size > T
                    )
                    retain = not will_step_now
                    if use_amp:
                        scaler.scale(loss).backward(retain_graph=retain)
                    else:
                        loss.backward(retain_graph=retain)

            # Accumulate metric sums on-device; no host sync in the hot loop.
            valid_elements = int(mask.expand_as(y_pred).sum())
            total_running = total_running + losses["total"].detach() * valid_elements
            mse_running = mse_running + losses["mse"].detach() * valid_elements
            mae_running = mae_running + losses["mae"].detach() * valid_elements
            n_elements += valid_elements
            n_chunks += 1

            pred_buf.append(y_pred.detach())
            true_buf.append(y_chunk.detach())
            mask_buf.append(mask)
            if len(pred_buf) >= FLUSH_CHUNKS:
                _flush()

            if is_train and teacher_forcing_prob > 0.0:
                if teacher_forcing_prob >= 1.0:
                    feed = y_chunk.detach()
                else:
                    tf_mask = (
                        torch.rand(
                            y_pred.shape[0],
                            y_pred.shape[1],
                            1,
                            1,
                            device=device,
                        )
                        < teacher_forcing_prob
                    )
                    feed = torch.where(tf_mask, y_chunk.detach(), y_pred)
            else:
                # Eval is always deterministic/free-running.
                feed = y_pred
            # Pin padded (out-of-length) positions to 0 so an exhausted run
            # cannot free-run into divergence and corrupt shared-parameter
            # gradients via 0*inf; its loss is masked out regardless.
            feed = torch.where(mask, feed, torch.zeros_like(feed))
            feed = _clamp_ar_feedback(feed)
            next_hist = feed.permute(1, 2, 0, 3)
            hist = torch.cat([hist[:, :, chunk_size:, :], next_hist], dim=2)

            chunk_idx += 1
            t += chunk_size
            at_tbptt_boundary = chunk_idx % tbptt_chunks == 0

            # Update params at the end of a TBPTT window
            if is_train and (at_tbptt_boundary or t + chunk_size > T):
                if grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    )
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                # Detach context after update to break gradient chain
                hist = hist.detach()

            # Skip the gap between TBPTT windows instead of walking every
            # timepoint of the run: each window still trains/evaluates a
            # real contiguous chunk_size*tbptt_chunks span, but consecutive
            # windows are now `stride` apart rather than back-to-back.
            # Context is rebuilt from ground truth at the jump since the
            # rolling buffer no longer covers what's immediately before it.
            if at_tbptt_boundary and extra_skip > 0 and t + chunk_size <= T:
                t = min(t + extra_skip, T - chunk_size)
                hist = ts[:, t - x : t, :].permute(0, 2, 1).unsqueeze(-1)

        _flush()  # drain this batch's remaining chunks (one host sync)
        pbar.set_postfix(
            {
                "total": f"{float(total_running) / max(n_elements, 1):.4f}",
                "mse": f"{float(mse_running) / max(n_elements, 1):.4f}",
            }
        )

    if n_chunks == 0:
        return {
            "total": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "pcc": float("nan"),
            "scc": float("nan"),
            "dtw": float("nan"),
        }

    all_pred = np.concatenate(pred_accum)
    all_true = np.concatenate(true_accum)
    pcc_val, scc_val = _safe_corr(all_pred, all_true)

    return {
        "total": float(total_running) / n_elements,
        "mse": float(mse_running) / n_elements,
        "mae": float(mae_running) / n_elements,
        "pcc": float(pcc_val),
        "scc": float(scc_val),
        "dtw": (dtw_running / n_dtw) if n_dtw else float("nan"),
    }


def rollout_autoregressive(
    model,
    x_history,
    edge_index,
    dt,
    pred_steps,
    chunk_size,
):
    """Autoregressively roll out predictions via feeding chunks back in to context."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    hist = x_history
    remaining = pred_steps
    chunks = []
    while remaining > 0:
        step = min(chunk_size, remaining)
        out = model(
            x_history=hist,
            edge_index=edge_index,
            pred_steps=step,
            dt=dt,
            autoregressive=False,
        )
        pred_chunk = out["x_pred"]
        chunks.append(pred_chunk)

        pred_hist = _clamp_ar_feedback(pred_chunk).permute(1, 2, 0, 3)
        hist = torch.cat([hist[:, :, step:, :], pred_hist], dim=2)
        remaining -= step

    return torch.cat(chunks, dim=0)


def run_epoch(
    model,
    loader,
    edge_index,
    dt,
    optimizer,
    lambda_mse,
    lambda_mae,
    grad_clip,
    desc,
    forecast_mode,
    ar_chunk_size,
    scaler=None,
    norm_stats: dict[str, torch.Tensor] | None = None,
    batch_transform=batch_to_model_tensors,
    channel_mask_fn=None,
):
    """``batch_transform`` defaults to the standard (x, y) extraction; pass a
    different callable (same signature) to score against a different target
    -- e.g. NEST's counterfactual perturbed-horizon evaluation, which needs
    (x, y_perturbed) instead of (x, y) but otherwise reuses this whole loop.

    ``channel_mask_fn(batch, device) -> (1, B, N, 1) bool`` restricts every reported
    metric to the True channels. Default None leaves the arithmetic bit-for-bit
    identical, so no other caller is affected."""
    is_train = optimizer is not None
    use_amp = scaler is not None
    model.train(is_train)

    total_running = 0.0
    mse_running = 0.0
    mae_running = 0.0
    pcc_running = 0.0
    scc_running = 0.0
    dtw_running = 0.0
    n_batches = 0
    n_elements = 0
    n_sequences = 0
    n_nonfinite_batches = 0
    pred_accum = []
    true_accum = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x_history, y_true = batch_transform(
            batch, edge_index.device, norm_stats=norm_stats
        )
        pred_steps = y_true.shape[0]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                if forecast_mode == "short":
                    out = model(
                        x_history=x_history,
                        edge_index=edge_index,
                        pred_steps=pred_steps,
                        dt=dt,
                        autoregressive=False,
                    )
                    y_pred = out["x_pred"]
                elif forecast_mode == "long":
                    y_pred = rollout_autoregressive(
                        model=model,
                        x_history=x_history,
                        edge_index=edge_index,
                        dt=dt,
                        pred_steps=pred_steps,
                        chunk_size=ar_chunk_size,
                    )
                else:
                    raise ValueError(
                        f"Unknown forecast_mode='{forecast_mode}'. Use 'short' or 'long'."
                    )

                chan_mask = (
                    channel_mask_fn(batch, y_pred.device)
                    if channel_mask_fn is not None
                    else None
                )
                losses = total_loss(
                    y_pred,
                    y_true,
                    lambda_mse=lambda_mse,
                    lambda_mae=lambda_mae,
                    mask=chan_mask,
                )
                objective = losses["total"]
                # Model-derived penalties (model.regularization_loss). Added to the
                # OPTIMIZED loss only, deliberately: `losses["total"]` is the
                # number reported and compared against `best_val`, and folding a
                # regularizer into it would make runs with different penalty
                # weights incomparable and would move the checkpoint criterion.
                # hasattr, not a direct call: keeps this loop usable by any plain
                # nn.Module that has no such hook.
                if is_train and hasattr(model, "regularization_loss"):
                    reg = model.regularization_loss()
                    if reg is not None:
                        objective = objective + reg
                loss = objective

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip
                        )
                    optimizer.step()

        # Under AMP a batch can overflow fp16 and produce a non-finite loss.
        # GradScaler already handles that correctly -- scaler.step() SKIPS the
        # optimizer update and lowers the loss scale -- so the weights are
        # untouched and the run is healthy. But an unguarded running sum turns
        # that single skipped batch into a NaN epoch metric forever after,
        # which then looks like divergence. Exclude what the scaler discarded
        # and count it instead, so the reported loss reflects the steps that
        # actually happened.
        #
        # Gated on use_amp: without a scaler there is nothing that "handled" the
        # overflow, so a non-finite loss is real divergence and must still poison
        # the metric -- that is what trips train_nest_braindyn's _require_finite_metrics.
        batch_total = float(losses["total"].detach().cpu())
        if use_amp and not math.isfinite(batch_total):
            n_nonfinite_batches += 1
            continue
        # Weight by the number of elements the loss actually averaged over. Under a
        # mask total_loss divides by valid.sum(), so weighting by the full numel()
        # would silently bias the epoch mean toward the masked batches.
        batch_elements = (
            y_pred.numel()
            if chan_mask is None
            else int(chan_mask.expand_as(y_pred).sum())
        )
        total_running += batch_total * batch_elements
        mse_running += float(losses["mse"].detach().cpu()) * batch_elements
        mae_running += float(losses["mae"].detach().cpu()) * batch_elements
        n_elements += batch_elements

        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        if chan_mask is not None:
            # pcc/scc/dtw have no mask parameter, so drop the excluded sequences
            # outright; everything below then operates on the kept subset unchanged.
            y_pred_np, y_true_np = apply_channel_mask(
                y_pred_np, y_true_np, chan_mask.detach().cpu().numpy()
            )
        # _safe_corr, not pearsonr/spearmanr directly: a constant prediction (or a
        # constant target on a masked-flat window) makes both return NaN with a
        # RuntimeWarning, which then poisons the running sum for the whole epoch.
        pcc_val, scc_val = _safe_corr(y_pred_np.ravel(), y_true_np.ravel())
        pcc_running += pcc_val
        scc_running += scc_val
        pred_accum.append(y_pred_np.reshape(-1))
        true_accum.append(y_true_np.reshape(-1))
        batch_sequences = int(np.prod(y_pred_np.shape[1:]))
        dtw_running += dtw_mean_normalized(y_pred_np, y_true_np) * batch_sequences
        n_sequences += batch_sequences
        n_batches += 1

        pbar.set_postfix(
            {
                "total": f"{total_running / n_elements:.4f}",
                "mse": f"{mse_running / n_elements:.4f}",
                "mae": f"{mae_running / n_elements:.4f}",
                "pcc": f"{pcc_running / n_batches:.4f}",
                "scc": f"{scc_running / n_batches:.4f}",
                "dtw": f"{dtw_running / n_sequences:.4f}",
            }
        )

    if n_batches == 0:
        return {
            "total": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "pcc": float("nan"),
            "scc": float("nan"),
            "dtw": float("nan"),
        }

    if n_elements == 0:
        # Every batch overflowed -- that IS divergence, not a transient skip.
        return {
            "total": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "pcc": float("nan"),
            "scc": float("nan"),
            "dtw": float("nan"),
            "n_nonfinite_batches": n_nonfinite_batches,
        }
    if n_nonfinite_batches:
        print(
            # n_batches counts only the batches that made it past the `continue`,
            # so it is the wrong denominator on its own.
            f"  [amp] {n_nonfinite_batches}/{n_batches + n_nonfinite_batches} "
            f"batches had a non-finite "
            f"loss; GradScaler skipped their optimizer step, so they are excluded "
            f"from the epoch metric rather than poisoning it."
        )
    all_pred = np.concatenate(pred_accum)
    all_true = np.concatenate(true_accum)
    pcc, _ = pearsonr(all_pred, all_true)
    scc, _ = spearmanr(all_pred, all_true)
    return {
        "total": total_running / n_elements,
        "mse": mse_running / n_elements,
        "mae": mae_running / n_elements,
        "pcc": float(pcc),
        "scc": float(scc),
        "dtw": dtw_running / n_sequences,
        "n_nonfinite_batches": n_nonfinite_batches,
    }


def run_test_rollout_chunks(
    model,
    loader,
    edge_index,
    dt,
    chunk_steps,
    context_len,
    rollout_steps,
    lambda_mse,
    lambda_mae,
    desc,
    run_loader,
    norm_mode: str = "train_global",
    norm_stats: dict[str, torch.Tensor] | None = None,
    channel_mask_fn=None,
):
    """Evaluate test split by autoregressive chunk rollout over full runs.

    ``channel_mask_fn(path, device) -> (1, 1, N, 1) bool`` restricts every reported
    metric to the True channels; ``path`` is the run key this loop already carries (for
    NEST, the subject index). Default None leaves the arithmetic untouched."""
    if chunk_steps <= 0:
        raise ValueError(f"chunk_steps must be positive, got {chunk_steps}")
    if rollout_steps is not None and rollout_steps <= 0:
        raise ValueError(
            f"rollout_steps must be positive when set, got {rollout_steps}"
        )

    # Sample one seeded-random available context window per run (instead of
    # always the earliest) so the rollout isn't anchored at the same point in
    # every recording.
    run_starts_all: dict[str, set[tuple[int, int]]] = {}
    for batch in loader:
        meta = batch["meta"]
        for path, t_start, run_len in zip(meta["path"], meta["t_start"], meta["T"]):
            run_starts_all.setdefault(path, set()).add((int(t_start), int(run_len)))
    rng = random.Random(0)
    run_starts = {
        path: rng.choice(sorted(starts)) for path, starts in run_starts_all.items()
    }

    total_running = 0.0
    mse_running = 0.0
    mae_running = 0.0
    pcc_running = 0.0
    scc_running = 0.0
    dtw_running = 0.0
    n_chunks = 0
    n_elements = 0
    n_sequences = 0
    n_nonfinite_chunks = 0
    nonfinite_runs: set = set()
    pred_accum = []
    true_accum = []

    pbar = tqdm(run_starts.items(), desc=desc, leave=False)
    for path, (t0, T) in pbar:
        ts = run_loader(path)
        if t0 + context_len >= T:
            continue
        # Keyed on the run identity this loop already has -- meta is out of scope by
        # now (it was consumed building run_starts), so the lookup goes through `path`.
        chan_mask = (
            channel_mask_fn(path, edge_index.device)
            if channel_mask_fn is not None
            else None
        )

        ctx_raw = ts[t0 : t0 + context_len]
        if norm_mode == "context":
            mean = ctx_raw.mean(axis=0, keepdims=True)
            std = ctx_raw.std(axis=0, keepdims=True).clip(1e-6)
        elif norm_mode == "train_global":
            if norm_stats is None:
                raise ValueError("fold-training normalization statistics are required")
            mean = norm_stats["mean"].cpu().numpy().reshape(1, -1)
            std = norm_stats["std"].cpu().numpy().reshape(1, -1)
        else:
            raise ValueError(f"Unknown norm_mode={norm_mode!r}")

        hist_norm = ((ctx_raw - mean) / std).astype(np.float32)
        hist = torch.from_numpy(hist_norm).to(device=edge_index.device)
        hist = hist.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, N, x, 1)

        current_t = t0 + context_len
        max_pred_steps = T - current_t
        if rollout_steps is not None:
            max_pred_steps = min(max_pred_steps, rollout_steps)
        remaining = max_pred_steps
        while remaining > 0:
            step = min(chunk_steps, remaining)
            out = model(
                x_history=hist,
                edge_index=edge_index,
                pred_steps=step,
                dt=dt,
                autoregressive=False,
            )
            y_pred = out["x_pred"]  # (step, 1, N, 1)

            gt_raw = ts[current_t : current_t + step]
            # Ground truth again -- unclamped for the reason given at hist_norm.
            gt_norm = ((gt_raw - mean) / std).astype(np.float32)
            y_true = torch.from_numpy(gt_norm).to(device=edge_index.device)
            y_true = y_true.unsqueeze(1).unsqueeze(-1)  # (step, 1, N, 1)

            losses = total_loss(
                y_pred,
                y_true,
                lambda_mse=lambda_mse,
                lambda_mae=lambda_mae,
                mask=chan_mask,
            )
            chunk_total = float(losses["total"].detach().cpu())
            if not math.isfinite(chunk_total):
                # Same guard as run_epoch: one non-finite chunk must not poison
                # the whole fold's reported metric (that is how a run with a
                # healthy best-val printed "Test | total=nan"). Unlike run_epoch
                # the chunks are NOT independent -- this is an autoregressive
                # rollout, so y_pred is fed back into `hist`. Once it is
                # non-finite every later chunk of THIS run is non-finite too, so
                # abandon the rest of the run rather than accumulating garbage.
                # Other runs continue and still contribute.
                #
                # Ungated, unlike run_epoch's twin: this function is eval-only and
                # takes no scaler, so there is no use_amp to condition on, and the
                # per-run report below already names what diverged.
                n_nonfinite_chunks += 1
                nonfinite_runs.add(path)
                break
            # No regularizer here at all: this function is eval-only (no
            # optimizer, called under torch.no_grad), so a penalty could serve no
            # optimization purpose and would only make the reported test `total`
            # incomparable across runs.
            # Weight by what the loss averaged over -- see the twin in run_epoch.
            chunk_elements = (
                y_pred.numel()
                if chan_mask is None
                else int(chan_mask.expand_as(y_pred).sum())
            )
            total_running += chunk_total * chunk_elements
            mse_running += float(losses["mse"].detach().cpu()) * chunk_elements
            mae_running += float(losses["mae"].detach().cpu()) * chunk_elements
            n_elements += chunk_elements

            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()
            if chan_mask is not None:
                y_pred_np, y_true_np = apply_channel_mask(
                    y_pred_np, y_true_np, chan_mask.detach().cpu().numpy()
                )
            pcc_val, scc_val = _safe_corr(y_pred_np.ravel(), y_true_np.ravel())
            pcc_running += pcc_val
            scc_running += scc_val
            pred_accum.append(y_pred_np.reshape(-1))
            true_accum.append(y_true_np.reshape(-1))
            chunk_sequences = int(np.prod(y_pred_np.shape[1:]))
            dtw_running += dtw_mean_normalized(y_pred_np, y_true_np) * chunk_sequences
            n_sequences += chunk_sequences
            n_chunks += 1

            pred_hist = _clamp_ar_feedback(y_pred).permute(1, 2, 0, 3)
            hist = torch.cat([hist[:, :, step:, :], pred_hist], dim=2)

            current_t += step
            remaining -= step

            pbar.set_postfix(
                {
                    "total": f"{total_running / n_elements:.4f}",
                    "mse": f"{mse_running / n_elements:.4f}",
                    "mae": f"{mae_running / n_elements:.4f}",
                    "pcc": f"{pcc_running / n_chunks:.4f}",
                    "scc": f"{scc_running / n_chunks:.4f}",
                    "dtw": f"{dtw_running / n_sequences:.4f}",
                }
            )

    if n_nonfinite_chunks:
        # Say WHICH runs diverged and how many chunks were dropped. A bare NaN
        # (or, worse, a silently shortened rollout) gives no way to tell "the
        # model is broken" from "one run of 40 hit a bad chunk at step 300".
        print(
            f"  [warn] {desc}: {n_nonfinite_chunks} non-finite chunk(s) across "
            f"{len(nonfinite_runs)}/{len(run_starts)} run(s); those rollouts were "
            f"truncated at the first non-finite chunk and excluded from the metrics."
        )

    if n_chunks == 0:
        return {
            "total": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "pcc": float("nan"),
            "scc": float("nan"),
            "dtw": float("nan"),
            "n_nonfinite_chunks": n_nonfinite_chunks,
        }

    all_pred = np.concatenate(pred_accum)
    all_true = np.concatenate(true_accum)
    pcc, _ = pearsonr(all_pred, all_true)
    scc, _ = spearmanr(all_pred, all_true)
    return {
        "total": total_running / n_elements,
        "mse": mse_running / n_elements,
        "mae": mae_running / n_elements,
        "pcc": float(pcc),
        "scc": float(scc),
        "dtw": dtw_running / n_sequences,
        "n_nonfinite_chunks": n_nonfinite_chunks,
    }


@torch.no_grad()
def graph_health_line(model, loader, edge_index, norm_stats=None) -> str:

    gl = getattr(getattr(model, "dynamics", None), "graph_laplacian", None)
    if gl is None or not hasattr(gl, "apply_precomputed"):
        return ""
    try:
        batch = next(iter(loader))
    except (StopIteration, TypeError):
        return ""

    was_training = model.training
    model.eval()
    try:
        x_hist, _ = batch_to_model_tensors(
            batch, edge_index.device, norm_stats=norm_stats
        )
        sheaf_h, aux = model.dynamics.compute_sheaf_h(x_hist[:2], edge_index)
        h_t = aux["h_t"]
        rel = float((sheaf_h - h_t).norm() / (h_t.norm() + 1e-12))

        parts = [f"|Lh|={rel:.2e}"]
        rho = aux.get("rho_src")
        if rho is not None:
            d, de = rho.shape[-2], rho.shape[-1]
            sv = torch.linalg.svdvals(rho.reshape(-1, d, de).float())
            parts.insert(0, f"sig={float(sv.mean()):.3f}")
        if getattr(gl, "raw_map_scale", None) is not None:
            # exp(s): the learned map magnitude
            parts.append(f"s={gl.map_scale():.3f}")
        # learned output amplitude R
        rate_fn = getattr(model.dynamics.ode, "effective_rate", None)
        if (
            rate_fn is not None
            and getattr(model.dynamics.ode, "rate_mode", "fixed") != "fixed"
        ):
            R = rate_fn()
            parts.append(
                f"R={float(R.mean()):.3f}[{float(R.min()):.3f},{float(R.max()):.3f}]"
            )
        te = getattr(model.dynamics.ode, "time_embed", None)
        if te is not None and getattr(te, "learn_freqs", False):
            # learned sinusoid frequencies, in cycles/step
            c = te.cycles_per_step()
            parts.append(f"w={float(c.min()):.4f}-{float(c.max()):.4f}c/s")
        sat_fn = getattr(model.dynamics.ode, "field_saturation", None)
        if sat_fn is not None:
            # fraction of the |dx/dt| <= R budget the tanh actually uses
            _s = sat_fn(x_hist[:2][:, :, -1, :].float(), sheaf_h.float())
            parts.append(
                f"sat={_s['mean']:.3f}/{_s['max']:.3f}(>{_s['frac_over_0.9']:.2f})"
            )
        return " | graph " + " ".join(parts)
    except Exception:
        return ""
    finally:
        model.train(was_training)


def save_dynamics_plot(
    model,
    loader,
    edge_index,
    dt,
    forecast_mode,
    ar_chunk_size,
    out_path: Path,
    max_nodes: int = 4,
    norm_stats: dict[str, torch.Tensor] | None = None,
) -> bool:
    """Save a quick prediction-vs-truth dynamics plot from one loader batch."""
    try:
        batch = next(iter(loader))
    except StopIteration:
        return False

    x_history, y_true = batch_to_model_tensors(
        batch, edge_index.device, norm_stats=norm_stats
    )
    pred_steps = y_true.shape[0]
    model.eval()
    with torch.no_grad():
        if forecast_mode in {"short", "long_ar_train"}:
            out = model(
                x_history=x_history,
                edge_index=edge_index,
                pred_steps=pred_steps,
                dt=dt,
                autoregressive=False,
            )
            y_pred = out["x_pred"]
        elif forecast_mode == "long":
            y_pred = rollout_autoregressive(
                model=model,
                x_history=x_history,
                edge_index=edge_index,
                dt=dt,
                pred_steps=pred_steps,
                chunk_size=ar_chunk_size,
            )
        else:
            raise ValueError(f"Unknown forecast_mode='{forecast_mode}'.")

    # use first sample in batch and first few nodes to keep plots readable.
    y_true_np = y_true[:, 0, :, 0].detach().cpu().numpy()  # (Ly, N)
    y_pred_np = y_pred[:, 0, :, 0].detach().cpu().numpy()  # (Ly, N)

    node_count = min(max_nodes, y_true_np.shape[1])
    fig, axes = plt.subplots(node_count, 1, figsize=(10, 2.4 * node_count), sharex=True)
    if node_count == 1:
        axes = [axes]

    t = np.arange(y_true_np.shape[0])
    for node_idx in range(node_count):
        ax = axes[node_idx]
        ax.plot(t, y_true_np[:, node_idx], label="true", linewidth=2.0, color="#1f77b4")
        ax.plot(
            t,
            y_pred_np[:, node_idx],
            label="pred",
            linewidth=1.8,
            linestyle="--",
            color="#d62728",
        )
        ax.set_ylabel(f"node {node_idx}")
        ax.grid(alpha=0.3)
        if node_idx == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("forecast step")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_subset_loader(
    dataset, indices, batch_size, num_workers, pin_memory, shuffle, seed=None
):
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=gen,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )


def subject_groups_for(dataset, group_key: str = "subject_id") -> np.ndarray:
    labels = []
    for idx in range(len(dataset)):
        ds, local_idx = _resolve_concat_sample(dataset, idx)
        meta = ds._samples[local_idx][1]
        if group_key not in meta:
            raise KeyError(
                f"sample metadata has no {group_key!r}; grouped CV needs it on every "
                f"sample. Got keys: {sorted(meta)}"
            )
        labels.append(str(meta[group_key]))
    return np.asarray(labels)


def compute_shuffle_split_indices(
    groups: np.ndarray, seed: int, train_frac: float, val_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # shuffle with integer numbers of subject in split
    if isinstance(groups, (int, np.integer)):
        raise TypeError(
            "compute_shuffle_split_indices takes per-sample group labels, not "
            "n_samples. Build them with subject_groups_for(combined_dataset)."
        )
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1 to leave a test split; got "
            f"{train_frac} + {val_frac} = {train_frac + val_frac}"
        )

    labels = np.asarray(groups)
    # np.unique returns a sorted array, so the subject ordering is canonical and
    # never depends on sample order or dict/set iteration order.
    uniq = np.unique(labels)
    n = len(uniq)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            f"{n} subjects cannot fill all three splits at train_frac="
            f"{train_frac}, val_frac={val_frac} (got train={n_train}, "
            f"val={n_val}, test={n_test}); need more subjects or less extreme "
            "fractions."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_subj = uniq[perm[:n_train]]
    val_subj = uniq[perm[n_train : n_train + n_val]]
    test_subj = uniq[perm[n_train + n_val :]]

    train_idx = np.flatnonzero(np.isin(labels, train_subj))
    val_idx = np.flatnonzero(np.isin(labels, val_subj))
    test_idx = np.flatnonzero(np.isin(labels, test_subj))
    return train_idx, val_idx, test_idx


def resolve_shuffle_seeds(
    base_seed: int, num_shuffles: int, explicit: list[int] | None
) -> list[int]:
    # Per-shuffle seeds to help comparability, if not explicit, becomes base_seed * 1000 + i

    if explicit:
        return [int(s) for s in explicit]
    if num_shuffles < 1:
        raise ValueError(f"num_shuffles must be >= 1, got {num_shuffles}")
    return [base_seed * 1000 + i for i in range(num_shuffles)]


def split_fingerprint(train_idx: np.ndarray, groups: np.ndarray) -> str:

    subjects = sorted(set(np.asarray(groups)[train_idx].tolist()))
    digest = hashlib.sha1("|".join(subjects).encode()).hexdigest()[:12]
    return f"n={len(train_idx)} nsub={len(subjects)} sha1={digest}"


def assert_disjoint_runs(train_run_ds, val_run_ds) -> None:
    # Fail if two SubjectRunDatasets share a run
    shared = {p for p, _ in train_run_ds._runs} & {p for p, _ in val_run_ds._runs}
    if shared:
        sample = sorted(shared)[:5]
        raise RuntimeError(
            f"{len(shared)} run(s) appear in both train and val after dedup, e.g. {sample}. "
            "The fold split is not subject-grouped."
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train BrainDyn on fMRI or EEG timeseries."
    )

    ap.add_argument(
        "--dataset",
        type=str,
        default="fmri",
        choices=["fmri", "lemon_eeg"],
        help="dataset backend: fmri (RBC manifest CSV) or lemon_eeg (LEMON "
        "sensor-space EEG manifest CSV)",
    )
    ap.add_argument("--manifest_csv", type=str, default="data/manifest.csv")

    # --- LEMON EEG arm ---------
    ap.add_argument(
        "--lemon_manifest_csv",
        type=str,
        default="data/lemon_manifest.csv",
        help="LEMON EC manifest CSV (from data/lemon_make_manifest.py).",
    )
    ap.add_argument(
        "--condition",
        type=str,
        default="EC",
        help="LEMON condition to train on (EC only for now; forward-compat).",
    )
    ap.add_argument(
        "--max_subjects",
        type=int,
        default=100,
        help="LEMON: cap the subject universe to a seeded deterministic subset so "
        "every graph-mode variant sees the same subjects (0/None = all).",
    )
    ap.add_argument(
        "--max_windows_per_subject",
        type=int,
        default=150,
        help="LEMON: per-subject seeded subsample of enumerated boundary-safe "
        "window starts (0/None = keep all; native 250 Hz yields ~12k/subject).",
    )
    ap.add_argument(
        "--lemon_overlap_only",
        action="store_true",
        help="LEMON: restrict to subjects flagged in_overlap (EEG-MRI) for the "
        "matched sensor-vs-source comparison.",
    )
    ap.add_argument(
        "--lemon_data_seed",
        type=int,
        default=42,
        help="LEMON: seed for subject selection + per-subject window subsampling. "
        "Keep FIXED across graph-mode variants or the control is confounded.",
    )

    # ---cohort flag for fMRI ----
    ap.add_argument(
        "--cohort", type=str, default="PNC", help="PNC, HBN, or None for both"
    )

    # --- general flags ----
    ap.add_argument("--x", type=int, default=30, help="context length")
    ap.add_argument("--y", type=int, default=10, help="forecast horizon length")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--min_t", type=int, default=0)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument(
        "--num_shuffles",
        type=int,
        default=5,
        help="number of random subject-shuffle splits to run",
    )
    ap.add_argument(
        "--shuffle_index",
        type=int,
        default=-1,
        help="run only this single 0-based shuffle and skip the rest; -1 (default) runs all of them",
    )
    ap.add_argument(
        "--shuffle_seeds",
        type=int,
        nargs="*",
        default=None,
        help="explicit per-shuffle seeds for split and training rng",
    )
    ap.add_argument(
        "--train_frac",
        type=float,
        default=0.7,
        help="fraction of SUBJECTS in each shuffle's train split",
    )
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="fraction of SUBJECTS in each shuffle's val split.",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau multiplicative decay factor",
    )
    ap.add_argument(
        "--lr_patience",
        type=int,
        default=2,
        help="Epochs with no val improvement before reducing LR",
    )
    ap.add_argument(
        "--lr_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for ReduceLROnPlateau",
    )
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden_dim", type=int, default=16)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--lstm_dropout", type=float, default=0.0)
    ap.add_argument("--map_hidden_dim", type=int, default=16)
    ap.add_argument("--vf_hidden_dim", type=int, default=128)
    ap.add_argument(
        "--vf_layers",
        type=int,
        default=2,
        help="number of LINEAR layers in the ODE vector-field MLP",
    )

    ap.add_argument(
        "--sheaf_layers",
        type=int,
        default=1,
        help="rounds of (I - step*L_F) sheaf diffusion",
    )
    ap.add_argument(
        "--diffusion_step",
        type=float,
        default=1.0,
        help="step size in the (I - step*L_F) update",
    )
    ap.add_argument(
        "--sheaf_map_pe",
        type=str,
        default="learned",
        choices=["none", "lappe", "learned"],
        help="Positional encoding appended to the restriction-map MLP input (edge-identity "
        "channel), routed only into that MLP — not the LSTM encoder or ODE field.",
    )
    ap.add_argument(
        "--time_embed_dim",
        type=int,
        default=16,
        help="width of the sinusoidal time embedding",
    )
    ap.add_argument(
        "--time_embed_max_period",
        type=float,
        default=16.0,
        help="longest wavelength in the fixed frequency schedule in solver time",
    )
    ap.add_argument(
        "--time_rate_mode",
        choices=["fixed", "global", "per_node"],
        default="fixed",
        help="learn the output amplitude R",
    )
    ap.add_argument(
        "--time_learn_freqs",
        action="store_true",
        help="learn the sinusoid frequencies w_k in the time embedding ",
    )
    ap.add_argument(
        "--time_max_cycles_per_step",
        type=float,
        default=0.5,
        help="cap on learned frequencies, in cycles per forecast step",
    )
    ap.add_argument(
        "--sheaf_map_pe_dim",
        type=int,
        default=8,
        help="PE width for --sheaf_map_pe (embedding dim for 'learned'; LapPE rank "
        "for 'lappe').",
    )
    ap.add_argument(
        "--sheaf_map_scale",
        choices=["none", "norm", "orth"],
        default="norm",
        help="reparametrize the restriction maps as rho = exp(s) * direction(M)",
    )
    ap.add_argument(
        "--coupling_block",
        choices=["none", "block", "complex"],
        default="block",
        help="restrict the restriction maps to be block-diagonal",
    )
    ap.add_argument(
        "--freeze_map_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pin the learned map scale s at its init",
    )
    ap.add_argument(
        "--learn_diffusion_gain",
        action="store_true",
        help="learn a scalar multiplier on --diffusion_step instead of fixing it.",
    )
    ap.add_argument(
        "--sheaf_norm",
        choices=["none", "sym", "row"],
        default="sym",
        help="degree normalization of the sheaf Laplacian",
    )

    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_mae", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=1.0)

    ap.add_argument(
        "--graph_mode",
        type=str,
        default="granger",
        choices=["granger", "spatial", "sc", "fc"],
        help=("graph prior choices"),
    )
    ap.add_argument(
        "--spatial_k",
        type=int,
        default=8,
        help="kNN neighbors per node for --graph_mode spatial (ignored if "
        "--spatial_radius > 0).",
    )
    ap.add_argument(
        "--spatial_radius",
        type=float,
        default=0.0,
        help="Distance threshold for --graph_mode spatial (0 = use kNN via "
        "--spatial_k). Units depend on --dataset: montage units (~meters) for "
        "lemon_eeg, MNI millimeters for fmri.",
    )
    ap.add_argument(
        "--sc_connectome_csv",
        type=str,
        default=None,
        help="Required for --graph_mode sc. CSV of a square (N, N) group-average "
        "structural-connectome matrix (e.g. streamline counts), in manifest node "
        "order, IDENTICAL across every fold/subject by construction.",
    )
    ap.add_argument(
        "--sc_volumes_csv",
        type=str,
        default=None,
        help="Optional for --graph_mode sc. Single-column CSV of per-parcel "
        "volumes (same order as --sc_connectome_csv), used to remove the size "
        "bias in raw streamline counts: SC_ij /= sqrt(volume_i * volume_j). "
        "Without it, --sc_connectome_csv is used as-is (symmetrized, thresholded).",
    )
    ap.add_argument(
        "--spatial_positions_json",
        type=str,
        default=None,
        help="Optional node-positions JSON for --graph_mode spatial. Default for "
        "lemon_eeg: montage_positions.json beside the .npy derivatives, else MNE "
        "fallback. Default for fmri: "
        "data/atlases/schaefer400_7networks/centroids.json (see "
        "scripts/build_schaefer_centroids.py).",
    )
    ap.add_argument(
        "--graph_max_batches",
        type=int,
        default=8,
        help="deprecated",
    )
    ap.add_argument(
        "--granger_threshold",
        type=float,
        default=0.01,
        help="manual Granger score threshold",
    )
    ap.add_argument(
        "--granger_lag",
        type=int,
        default=1,
        help="lag used for pairwise Granger graph estimation",
    )
    ap.add_argument(
        "--granger_threshold_mode",
        type=str,
        default="manual",
        choices=["manual", "topk", "topk_per_node"],
        help="how to convert Granger scores into edges",
    )
    ap.add_argument(
        "--granger_topk_per_node",
        type=int,
        default=0,
        help="edges kept per node when --granger_threshold_mode=topk_per_node",
    )
    ap.add_argument(
        "--granger_topk_edges",
        type=int,
        default=0,
        help="number of directed edges to keep when --granger_threshold_mode=topk",
    )
    ap.add_argument(
        "--fc_absolute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="for --graph_mode fc: rank edges by |Pearson corr| (default). Use "
        "--no-fc_absolute to keep signed correlation, dropping anticorrelated pairs.",
    )
    ap.add_argument("--save_path", type=str, default="checkpoints/braindyn_rbc_best.pt")
    ap.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (float16) to reduce GPU memory",
    )
    ap.add_argument(
        "--no_pin_memory",
        action="store_true",
        help="Disable pin_memory in DataLoader to reduce CPU RAM usage",
    )
    ap.add_argument(
        "--forecast_mode",
        type=str,
        default="short",
        choices=["short", "long", "long_ar_train"],
        help=(
            "short: direct horizon prediction; "
            "long: window training + autoregressive inference; "
            "long_ar_train: full run-level autoregressive training + inference"
        ),
    )
    ap.add_argument(
        "--ar_chunk_size",
        type=int,
        default=1,
        help="chunk size for autoregressive rollout when --forecast_mode=long or long_ar_train",
    )
    ap.add_argument(
        "--test_rollout_steps",
        type=int,
        default=None,
        help="when --forecast_mode=long, cap test autoregressive rollout to this many future timepoints (default: full remaining run)",
    )
    ap.add_argument(
        "--tbptt_chunks",
        type=int,
        default=5,
        help="truncated BPTT: detach gradients every N chunks during long_ar_train",
    )
    ap.add_argument(
        "--ss_start",
        type=float,
        default=1.0,
        help="scheduled sampling: teacher forcing probability at epoch 1 (1.0 = full teacher forcing)",
    )
    ap.add_argument(
        "--ss_end",
        type=float,
        default=0.0,
        help="scheduled sampling: teacher forcing probability at final epoch (0.0 = full free-running)",
    )
    ap.add_argument(
        "--ss_decay_epochs",
        type=int,
        default=None,
        help=(
            "Number of epochs over which to linearly decay teacher forcing from "
            "--ss_start to --ss_end. Default: decay across all --epochs. Use 15 "
            "to finish the decay at epoch 15 and keep --ss_end afterward."
        ),
    )
    ap.add_argument(
        "--ablation_gcn",
        action="store_true",
        help="Ablation 1: use GCN neighborhood aggregation instead of sheaf operator",
    )
    ap.add_argument(
        "--ablation_no_lstm",
        action="store_true",
        help="Ablation 2: disable LSTM temporal encoder and use last-step linear projection",
    )
    ap.add_argument(
        "--static_edge_maps",
        action="store_true",
        help=(
            "Use one learned static restriction-map pair per directed edge instead "
            "of the default data-dependent shared restriction-map MLP."
        ),
    )
    ap.add_argument(
        "--map_mlp_hidden_dim",
        type=int,
        default=64,
        help="Internal hidden width of the default restriction-map MLP.",
    )
    ap.add_argument(
        "--identity_restriction_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="initialize the sheaf restriction maps at the identity",
    )
    ap.add_argument(
        "--no_sheaf",
        action="store_true",
        help="trivial-sheaf ablation",
    )
    ap.add_argument(
        "--run_batch_size",
        type=int,
        default=1,
        help="batch size for the per-run AR training DataLoader",
    )
    ap.add_argument(
        "--ar_segment_len",
        type=int,
        default=None,
        help="long_ar_train only. If set (with --ar_segments_per_subject), train on "
        "fixed-length NON-OVERLAPPING segments of this many timepoints per run ",
    )
    ap.add_argument(
        "--ar_segments_per_subject",
        type=int,
        default=None,
        help="number of non-overlapping segments to draw per run when "
        "--ar_segment_len is set",
    )
    ap.add_argument(
        "--val_every",
        type=int,
        default=1,
        help="Run validation every N epochs (default 1 = every epoch). Skipped epochs still train and log train metrics.",
    )
    args = ap.parse_args()
    _validate_new_flag_combinations(args)
    return args


def _validate_new_flag_combinations(args) -> None:
    """Reject flag combinations that are wrong before any data is touched."""
    if args.sheaf_map_scale != "none" and args.ablation_gcn:
        raise ValueError(
            "--sheaf_map_scale reparametrizes the sheaf's restriction maps; "
            "--ablation_gcn replaces the sheaf with a GCN, which has none."
        )
    if args.no_sheaf:
        _neutralized = []
        if args.identity_restriction_init:
            args.identity_restriction_init = False
            _neutralized.append("--identity_restriction_init")
        if args.sheaf_map_pe != "none":
            args.sheaf_map_pe = "none"
            _neutralized.append("--sheaf_map_pe")
        if args.sheaf_map_scale != "none":
            args.sheaf_map_scale = "none"
            _neutralized.append("--sheaf_map_scale")
        if args.freeze_map_scale:
            args.freeze_map_scale = False
            _neutralized.append("--freeze_map_scale")
        if args.coupling_block != "none":
            args.coupling_block = "none"
            _neutralized.append("--coupling_block")
        if _neutralized:
            print(
                f"[--no_sheaf] identity maps: {', '.join(_neutralized)} disabled "
                f"(they parameterize trained maps, which this ablation has none of)."
            )
    if args.time_embed_dim == 0:
        if args.time_learn_freqs:
            args.time_learn_freqs = False
            print(
                "[--time_embed_dim 0] --time_learn_freqs disabled: there are no "
                "sinusoids to learn frequencies for"
            )
    if args.identity_restriction_init and args.ablation_gcn:
        raise ValueError(
            "--identity_restriction_init has no effect with --ablation_gcn: GCN "
            "aggregation has no sheaf restriction maps to initialize."
        )
    if args.graph_mode == "spatial" and args.dataset not in ("lemon_eeg", "fmri"):
        raise ValueError(
            "--graph_mode spatial is only supported for --dataset lemon_eeg "
            "(electrode positions) or --dataset fmri (Schaefer-400 parcel centroids)."
        )
    if args.graph_mode == "sc" and args.dataset != "fmri":
        raise ValueError(
            "--graph_mode sc is only supported for --dataset fmri. This "
            "structural-connectome prior (Schaefer-400 SC) is fMRI-only"
        )
    if args.sheaf_map_pe != "none" and args.static_edge_maps:
        # PE is appended to the shared restriction MLP input; the static-map paths
        # have no such MLP to route it into.
        raise ValueError(
            "--sheaf_map_pe routes positional encoding into the restriction MLP, so it "
            "requires the MLP restriction maps (i.e. without --static_edge_maps)."
        )
    if args.no_sheaf:
        if args.ablation_gcn:
            raise ValueError(
                "--no_sheaf and --ablation_gcn are mutually exclusive graph "
                "operators (trivial sheaf vs. GCN aggregation)"
            )
        if args.static_edge_maps:
            raise ValueError(
                "--no_sheaf pins identity maps and builds no restriction parameters"
            )
    if args.freeze_map_scale and args.sheaf_map_scale == "none":
        raise ValueError(
            "--freeze_map_scale pins the learned scale s, but --sheaf_map_scale none "
            "has no such parameter"
        )
    if args.time_rate_mode == "per_node" and args.ablation_gcn:
        raise ValueError(
            "--time_rate_mode per_node needs the node count, which the GCN "
            "ablation path does not expose"
        )


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dataset == "fmri":
        manifest_csv = Path(args.manifest_csv)
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

        loaders = make_dataloaders(
            manifest_csv=manifest_csv,
            x=args.x,
            y=args.y,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cohort=args.cohort,
            min_t=args.min_t,
            cache=args.cache,
            pin_memory=not args.no_pin_memory,
            norm_mode="train_global",
        )

        def run_loader(path: str) -> np.ndarray:
            return np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)

    elif args.dataset == "lemon_eeg":
        lemon_manifest_csv = Path(args.lemon_manifest_csv)
        if not lemon_manifest_csv.exists():
            raise FileNotFoundError(
                f"LEMON manifest not found: {lemon_manifest_csv} "
                "(build it with data/lemon_build_npy.py + data/lemon_make_manifest.py)."
            )

        loaders = make_lemon_dataloaders(
            manifest_csv=lemon_manifest_csv,
            x=args.x,
            y=args.y,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            condition=args.condition,
            max_subjects=(args.max_subjects or None),
            max_windows_per_subject=(args.max_windows_per_subject or None),
            overlap_only=args.lemon_overlap_only,
            seed=args.lemon_data_seed,
            cache=args.cache,
            pin_memory=not args.no_pin_memory,
            norm_mode="train_global",
        )
        realized = sum(len(l.dataset) for l in loaders.values())  # type: ignore[arg-type]
        n_canon = len(loaders["train"].dataset.channels)  # type: ignore[attr-defined]
        print(
            f"LEMON window budget realized: {realized} total windows "
            f"(max_subjects={args.max_subjects}, "
            f"max_windows_per_subject={args.max_windows_per_subject}, "
            f"seed={args.lemon_data_seed}) | canonical montage: {n_canon} channels "
            f"(intersection across subjects; per-subject counts vary)"
        )

        # MNE never runs in the training loop: blocks are pre-built .npy arrays.
        run_loader = make_lemon_run_loader(
            lemon_manifest_csv,
            condition=args.condition,
            overlap_only=args.lemon_overlap_only,
        )

    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

    train_dataset = loaders["train"].dataset
    val_dataset = loaders["val"].dataset
    test_dataset = loaders["test"].dataset
    use_pin = (not args.no_pin_memory) and torch.cuda.is_available()

    # shuffle-split scheme
    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    if len(combined_dataset) == 0:  # type: ignore[arg-type]
        raise RuntimeError(
            "Combined dataset is empty. Adjust x/y/stride/cohort/min_t settings."
        )
    combined_groups = subject_groups_for(combined_dataset)
    n_subjects_total = len(set(combined_groups.tolist()))
    print(
        f"Shuffle-split pool: {len(combined_dataset)} windows from "
        f"{n_subjects_total} subjects "
        "(manifest train/val/test labels ignored; split grouped by subject)"
    )

    if args.dataset == "fmri":
        cohort_tag = (args.cohort or "all").lower()
    else:
        cohort_tag = "all"
    forecast_tag = args.forecast_mode
    ablation_parts = []
    if args.ablation_gcn:
        ablation_parts.append("ablation_gcn")
    if args.ablation_no_lstm:
        ablation_parts.append("ablation_no_lstm")
    if args.no_sheaf:
        ablation_parts.append("no_sheaf")
    if args.identity_restriction_init:
        ablation_parts.append("identity_maps")
    ablation_tag = "main" if len(ablation_parts) == 0 else "_".join(ablation_parts)

    default_save_path = "checkpoints/braindyn_rbc_best.pt"
    resolved_save_path = (
        f"checkpoints/braindyn_{args.dataset}_{cohort_tag}_{forecast_tag}_{ablation_tag}_dt02_trainviz_best.pt"
        if args.save_path == default_save_path
        else args.save_path
    )
    save_path = Path(resolved_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    shuffle_seeds = resolve_shuffle_seeds(
        args.seed, args.num_shuffles, args.shuffle_seeds
    )
    num_shuffles = len(shuffle_seeds)

    if args.shuffle_index >= 0:
        if args.shuffle_index >= num_shuffles:
            raise ValueError(
                f"--shuffle_index {args.shuffle_index} out of range for "
                f"{num_shuffles} shuffle(s) (valid: 0..{num_shuffles - 1})"
            )
        runs_to_do = [args.shuffle_index]
        print(
            f"Single-shuffle mode: running only shuffle "
            f"{args.shuffle_index + 1}/{num_shuffles} "
            "(seed-determined, so identical to that shuffle of a full run)."
        )
    else:
        runs_to_do = list(range(num_shuffles))

    run_val_scores = []
    run_val_metrics = []
    run_test_scores = []

    for run_idx in runs_to_do:
        # each shuffle's seed drives both its split and its training RNG
        run_seed = shuffle_seeds[run_idx]
        set_seed(run_seed)
        train_idx, val_idx, test_idx = compute_shuffle_split_indices(
            combined_groups, run_seed, args.train_frac, args.val_frac
        )
        train_subjects = set(combined_groups[train_idx].tolist())
        val_subjects = set(combined_groups[val_idx].tolist())
        test_subjects = set(combined_groups[test_idx].tolist())
        assert (
            not (train_subjects & val_subjects)
            and not (train_subjects & test_subjects)
            and not (val_subjects & test_subjects)
        ), (
            f"shuffle {run_idx}: subjects overlap across splits — the split is "
            "not subject-grouped"
        )
        run_fp = split_fingerprint(train_idx, combined_groups)
        print(
            f"Shuffle {run_idx + 1}/{num_shuffles} (seed {run_seed}) | "
            f"train {len(train_idx)} win / {len(train_subjects)} subj, "
            f"val {len(val_idx)} win / {len(val_subjects)} subj, "
            f"test {len(test_idx)} win / {len(test_subjects)} subj | "
            f"fingerprint: {run_fp}"
        )

        train_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=train_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin,
            shuffle=True,
            seed=run_seed,
        )
        val_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=val_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin,
            shuffle=False,
            seed=run_seed + 1,
        )
        test_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=test_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin,
            shuffle=False,
            seed=run_seed + 2,
        )
        run_norm_stats = compute_train_global_stats(combined_dataset, train_idx)

        if args.graph_mode == "spatial":
            # physical-proximity prior: electrode positions (EEG) or Schaefer-400
            # parcel centroids (fMRI)
            if args.dataset == "lemon_eeg":
                train_ds = loaders["train"].dataset  # type: ignore[assignment]
                channels = list(getattr(train_ds, "channels", []) or [])
                if not channels:
                    raise RuntimeError(
                        "--graph_mode spatial requires the dataset to expose channel names "
                        "(LemonDataset.channels)."
                    )
                pos_json = args.spatial_positions_json
                if pos_json is None and getattr(train_ds, "_samples", None):
                    cand = (
                        Path(train_ds._samples[0][0]).parent / "montage_positions.json"
                    )
                    pos_json = str(cand) if cand.exists() else None
                positions = montage_positions(channels, positions_json=pos_json)
                num_nodes = len(channels)
            elif args.dataset == "fmri":
                sample_batch = next(iter(train_loader))
                num_nodes = int(sample_batch["x"].shape[-1])
                positions = schaefer_centroids(
                    positions_json=args.spatial_positions_json,
                    expected_nodes=num_nodes,
                )
            else:
                raise AssertionError(
                    f"Unhandled --graph_mode spatial dataset: {args.dataset}"
                )
            edge_index_cpu, graph_score, graph_info = (
                build_spatial_graph_from_positions(
                    positions,
                    k=args.spatial_k,
                    radius=(args.spatial_radius or None),
                )
            )
            graph_tag = "Spatial"
            graph_desc = (
                f"kNN k={graph_info['k']}"
                if graph_info["radius"] == 0.0
                else f"radius={graph_info['radius']:.3f}"
            ) + f", edges={graph_info['selected_edges']}"
        elif args.graph_mode == "granger":
            # deterministic, subject-fair prior
            edge_index_cpu, graph_score, graph_info = build_granger_graph_grouped(
                dataset=combined_dataset,
                indices=train_idx,
                groups=combined_groups,
                threshold=args.granger_threshold,
                lag=args.granger_lag,
                threshold_mode=args.granger_threshold_mode,
                topk_edges=args.granger_topk_edges,
                topk_per_node=args.granger_topk_per_node,
                max_windows_per_subject=(args.max_windows_per_subject or None),
            )
            graph_tag = "Granger"
            graph_desc = (
                f"mode={graph_info['mode']}, threshold={graph_info['threshold']:.4f}, "
                f"lag={graph_info['lag']}, topk_edges={graph_info['topk_edges']}, "
                f"subjects={graph_info['n_subjects']}, windows={graph_info['total_windows']}"
            )
        elif args.graph_mode == "fc":
            # data-driven functional-connectivity prior: subject-averaged
            # |Pearson corr| over the train fold, thresholded like Granger.
            edge_index_cpu, graph_score, graph_info = build_corr_graph_grouped(
                dataset=combined_dataset,
                indices=train_idx,
                groups=combined_groups,
                threshold=args.granger_threshold,
                threshold_mode=args.granger_threshold_mode,
                topk_edges=args.granger_topk_edges,
                topk_per_node=args.granger_topk_per_node,
                absolute=args.fc_absolute,
                max_windows_per_subject=(args.max_windows_per_subject or None),
            )
            graph_tag = "FC"
            graph_desc = (
                f"mode={graph_info['mode']}, threshold={graph_info['threshold']:.4f}, "
                f"absolute={graph_info['absolute']}, topk_edges={graph_info['topk_edges']}, "
                f"subjects={graph_info['n_subjects']}, windows={graph_info['total_windows']}"
            )
        elif args.graph_mode == "sc":
            # fixed structural-connectome prior
            if not args.sc_connectome_csv:
                raise ValueError("--graph_mode sc requires --sc_connectome_csv.")
            sample_batch = next(iter(train_loader))
            num_nodes = int(sample_batch["x"].shape[-1])
            sc_matrix = load_square_matrix_csv(args.sc_connectome_csv)
            if sc_matrix.shape[0] != num_nodes:
                raise ValueError(
                    f"--sc_connectome_csv is {tuple(sc_matrix.shape)} but the data has "
                    f"{num_nodes} nodes."
                )
            volumes = (
                load_vector_csv(args.sc_volumes_csv) if args.sc_volumes_csv else None
            )
            edge_index_cpu, graph_score, graph_info = build_sc_graph(
                sc_matrix, topk_edges=args.granger_topk_edges, volumes=volumes
            )
            graph_tag = "SC"
            graph_desc = (
                f"volume_normalized={graph_info['volume_normalized']}, "
                f"threshold={graph_info['threshold']:.4f}, edges={graph_info['selected_edges']}"
            )
        else:
            raise AssertionError(f"Unhandled graph mode: {args.graph_mode}")
        edge_index = edge_index_cpu.to(device)

        if graph_score is not None:
            num_nodes = graph_score.shape[0]
        print(
            f"Shuffle {run_idx + 1}/{num_shuffles} | {graph_tag} graph built: "
            f"N={num_nodes}, E={edge_index.shape[1]}, {graph_desc}"
        )
        # map-PE for the sheaf module: a separate LapPE routed only into the
        # sheaf MLP
        sheaf_node_pe = None
        if args.sheaf_map_pe == "lappe":
            sheaf_node_pe = build_lap_pe(
                edge_index_cpu, num_nodes=num_nodes, rank=args.sheaf_map_pe_dim
            )
            print(
                f"Shuffle {run_idx + 1}/{num_shuffles} | gate LapPE (map-PE): "
                f"shape={tuple(sheaf_node_pe.shape)} from symmetrized {graph_tag} graph"
            )

        config = BrainDynConfig(
            signal_dim=1,
            hidden_dim=args.hidden_dim,
            num_nodes=num_nodes,
            window_size=args.x,
            lstm_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
            map_hidden_dim=args.map_hidden_dim,
            vf_hidden_dim=args.vf_hidden_dim,
            vf_layers=args.vf_layers,
            use_gcn=args.ablation_gcn,
            use_lstm_encoder=(not args.ablation_no_lstm),
            edge_specific_maps=args.static_edge_maps,
            sheaf_layers=args.sheaf_layers,
            diffusion_step=args.diffusion_step,
            sheaf_mlp_maps=(not args.static_edge_maps),
            map_mlp_hidden_dim=args.map_mlp_hidden_dim,
            identity_restriction_init=args.identity_restriction_init,
            sheaf_map_pe=args.sheaf_map_pe,
            sheaf_map_pe_dim=args.sheaf_map_pe_dim,
            frozen_identity=args.no_sheaf,
            sheaf_norm=args.sheaf_norm,
            learn_diffusion_gain=args.learn_diffusion_gain,
            sheaf_map_scale=args.sheaf_map_scale,
            freeze_map_scale=args.freeze_map_scale,
            coupling_block=args.coupling_block,
            time_embed_dim=args.time_embed_dim,
            time_embed_max_period=args.time_embed_max_period,
            time_rate_mode=args.time_rate_mode,
            time_learn_freqs=args.time_learn_freqs,
            time_max_cycles_per_step=args.time_max_cycles_per_step,
        )
        model = BrainDyn(config, sheaf_node_pe=sheaf_node_pe).to(device)
        # Register the edge universe for edge-specific restriction maps before
        # building the optimizer (no-op when --edge_specific_maps is not set).
        model.register_restriction_edges(edge_index)
        print(
            f"Shuffle {run_idx + 1}/{num_shuffles} | model ablations: "
            f"gcn={args.ablation_gcn}, no_lstm={args.ablation_no_lstm}, "
            f"no_sheaf={args.no_sheaf}, "
            f"static_edge_maps={args.static_edge_maps}, norm_mode=train_global, "
            f"graph_mode={args.graph_mode}"
        )
        checkpoint_config = vars(args).copy()
        checkpoint_config["norm_mode"] = "train_global"
        checkpoint_config["edge_specific_maps"] = args.static_edge_maps
        checkpoint_config["sheaf_mlp_maps"] = not args.static_edge_maps

        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )

        scaler = (
            torch.amp.GradScaler("cuda")
            if (args.amp and torch.cuda.is_available())
            else None
        )
        if scaler is not None and run_idx == 0:
            print("AMP enabled: using float16 for forward pass")

        # For long_ar_train, build subject run loaders (full timeseries, no windowing)
        if args.forecast_mode == "long_ar_train":
            train_run_dataset = SubjectRunDataset(
                combined_dataset,
                train_idx,
                run_loader=run_loader,
                use_cache=args.cache,
                segment_len=args.ar_segment_len,
                segments_per_subject=args.ar_segments_per_subject,
                segment_seed=args.lemon_data_seed,
            )
            val_run_dataset = SubjectRunDataset(
                combined_dataset,
                val_idx,
                run_loader=run_loader,
                use_cache=args.cache,
                segment_len=args.ar_segment_len,
                segments_per_subject=args.ar_segments_per_subject,
                segment_seed=args.lemon_data_seed,
            )
            # Dedup collapses windows to whole runs; if a run reached both sides
            # the two items would be the identical full timeseries.
            assert_disjoint_runs(train_run_dataset, val_run_dataset)
            train_run_gen = torch.Generator()
            train_run_gen.manual_seed(run_seed + 2)
            train_run_loader = DataLoader(
                train_run_dataset,
                batch_size=args.run_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=use_pin,
                persistent_workers=(args.num_workers > 0),
                generator=train_run_gen,
                worker_init_fn=_seed_worker if args.num_workers > 0 else None,
                collate_fn=pad_collate_runs,
            )
            val_run_gen = torch.Generator()
            val_run_gen.manual_seed(run_seed + 3)
            val_run_loader = DataLoader(
                val_run_dataset,
                batch_size=args.run_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=use_pin,
                persistent_workers=(args.num_workers > 0),
                generator=val_run_gen,
                worker_init_fn=_seed_worker if args.num_workers > 0 else None,
                collate_fn=pad_collate_runs,
            )

        run_save_path = save_path.with_name(
            f"{save_path.stem}_shuffle{run_idx + 1}{save_path.suffix}"
        )
        run_plot_dir = Path("training") / f"{save_path.stem}_shuffle{run_idx + 1}"
        best_val = float("inf")

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            if args.forecast_mode == "long_ar_train":
                # Linear decay of teacher forcing probability over epochs
                tf_prob = teacher_forcing_probability(
                    args.ss_start,
                    args.ss_end,
                    epoch,
                    args.epochs,
                    decay_epochs=args.ss_decay_epochs,
                )
                train_start = time.perf_counter()
                train_metrics = run_epoch_ar_train(
                    model=model,
                    subject_loader=train_run_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    x=args.x,
                    chunk_size=args.ar_chunk_size,
                    tbptt_chunks=args.tbptt_chunks,
                    optimizer=optimizer,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    grad_clip=args.grad_clip,
                    desc=f"shuffle {run_idx + 1} train {epoch}/{args.epochs} [tf={tf_prob:.2f}]",
                    scaler=scaler,
                    teacher_forcing_prob=tf_prob,
                    norm_stats=run_norm_stats,
                )
                train_time_s = time.perf_counter() - train_start
            else:
                train_start = time.perf_counter()
                train_metrics = run_epoch(
                    model=model,
                    loader=train_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    optimizer=optimizer,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    grad_clip=args.grad_clip,
                    desc=f"shuffle {run_idx + 1} train {epoch}/{args.epochs}",
                    forecast_mode=args.forecast_mode,
                    ar_chunk_size=args.ar_chunk_size,
                    scaler=scaler,
                    norm_stats=run_norm_stats,
                )
                train_time_s = time.perf_counter() - train_start

            run_val_this_epoch = (epoch % args.val_every == 0) or (epoch == args.epochs)
            val_metrics = None
            val_time_s = 0.0
            if run_val_this_epoch:
                val_start = time.perf_counter()
                with torch.no_grad():
                    if args.forecast_mode == "long_ar_train":
                        val_metrics = run_epoch_ar_train(
                            model=model,
                            subject_loader=val_run_loader,
                            edge_index=edge_index,
                            dt=args.dt,
                            x=args.x,
                            chunk_size=args.ar_chunk_size,
                            tbptt_chunks=args.tbptt_chunks,
                            optimizer=None,
                            lambda_mse=args.lambda_mse,
                            lambda_mae=args.lambda_mae,
                            grad_clip=args.grad_clip,
                            desc=f"shuffle {run_idx + 1} val {epoch}/{args.epochs}",
                            teacher_forcing_prob=0.0,
                            norm_stats=run_norm_stats,
                        )
                    else:
                        val_metrics = run_epoch(
                            model=model,
                            loader=val_loader,
                            edge_index=edge_index,
                            dt=args.dt,
                            optimizer=None,
                            lambda_mse=args.lambda_mse,
                            lambda_mae=args.lambda_mae,
                            grad_clip=args.grad_clip,
                            desc=f"shuffle {run_idx + 1} val {epoch}/{args.epochs}",
                            forecast_mode=args.forecast_mode,
                            ar_chunk_size=args.ar_chunk_size,
                            norm_stats=run_norm_stats,
                        )
                val_time_s = time.perf_counter() - val_start

            prev_lr = optimizer.param_groups[0]["lr"]
            if val_metrics is not None:
                scheduler.step(val_metrics["total"])
            new_lr = optimizer.param_groups[0]["lr"]
            lr_msg = f" | lr={new_lr:.2e}"
            if new_lr < prev_lr:
                lr_msg += " (reduced)"

            epoch_time_s = time.perf_counter() - epoch_start

            graph_msg = graph_health_line(
                model, val_loader or train_loader, edge_index, run_norm_stats
            )

            if val_metrics is not None:
                print(
                    f"Shuffle {run_idx + 1}/{num_shuffles} Epoch {epoch:03d} | "
                    f"train total={train_metrics['total']:.6f} mse={train_metrics['mse']:.6f} mae={train_metrics['mae']:.6f} "
                    f"pcc={train_metrics['pcc']:.4f} scc={train_metrics['scc']:.4f} | "
                    f"val total={val_metrics['total']:.6f} mse={val_metrics['mse']:.6f} mae={val_metrics['mae']:.6f} "
                    f"pcc={val_metrics['pcc']:.4f} scc={val_metrics['scc']:.4f}"
                    f" | t_train={train_time_s:.1f}s t_val={val_time_s:.1f}s t_epoch={epoch_time_s:.1f}s"
                    f"{lr_msg}{graph_msg}"
                )
            else:
                print(
                    f"Shuffle {run_idx + 1}/{num_shuffles} Epoch {epoch:03d} | "
                    f"train total={train_metrics['total']:.6f} mse={train_metrics['mse']:.6f} mae={train_metrics['mae']:.6f} "
                    f"pcc={train_metrics['pcc']:.4f} scc={train_metrics['scc']:.4f}"
                    f" | t_train={train_time_s:.1f}s t_epoch={epoch_time_s:.1f}s (no val)"
                    f"{lr_msg}{graph_msg}"
                )

            val_bad = val_metrics is not None and not math.isfinite(
                val_metrics["total"]
            )
            train_bad = not math.isfinite(train_metrics["total"])
            if val_bad or (train_bad and val_metrics is None):
                raise RuntimeError(
                    f"Shuffle {run_idx + 1} diverged at epoch {epoch}: "
                    f"train total={train_metrics['total']}, "
                    f"val total={val_metrics['total'] if val_metrics else 'n/a'}. "
                    "Aborting: a non-finite loss never beats `best_val`, so no "
                    "checkpoint would be written and the run would waste its "
                    "remaining epochs. Reduce --lr, or if using "
                    "check the graph magnitude."
                )
            if train_bad:
                print(
                    f"  [warn] shuffle {run_idx + 1} epoch {epoch}: every train batch "
                    f"was non-finite, but val={val_metrics['total']:.6f} is healthy. "
                    "Continuing; check --amp / loss scale if this persists."
                )

            if val_metrics is not None and val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": checkpoint_config,
                        "best_val_total": best_val,
                        "edge_index": edge_index_cpu,
                        "graph_mode": args.graph_mode,
                        "graph_info": graph_info,
                        "graph_score": (
                            graph_score.detach().cpu()
                            if graph_score is not None
                            else None
                        ),
                        "shuffle": run_idx + 1,
                        "norm_stats": run_norm_stats,
                    },
                    run_save_path,
                )
                print(
                    f"Saved shuffle {run_idx + 1} best checkpoint to {run_save_path} (val total={best_val:.6f})"
                )

            if epoch % 5 == 0:
                plot_path = run_plot_dir / f"epoch_{epoch:03d}.png"
                saved = save_dynamics_plot(
                    model=model,
                    loader=val_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    forecast_mode=args.forecast_mode,
                    ar_chunk_size=args.ar_chunk_size,
                    out_path=plot_path,
                    norm_stats=run_norm_stats,
                )
                if saved:
                    print(f"Saved dynamics plot: {plot_path}")
                else:
                    print(
                        f"Skipped dynamics plot at epoch {epoch}: validation loader had no batches"
                    )

        run_val_scores.append(best_val)

        print(
            f"Evaluating shuffle {run_idx + 1} best checkpoint on val and test splits..."
        )
        ckpt = torch.load(run_save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            if args.forecast_mode == "long_ar_train":
                best_val_metrics = run_epoch_ar_train(
                    model=model,
                    subject_loader=val_run_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    x=args.x,
                    chunk_size=args.ar_chunk_size,
                    tbptt_chunks=args.tbptt_chunks,
                    optimizer=None,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    grad_clip=args.grad_clip,
                    desc=f"shuffle {run_idx + 1} best-val",
                    teacher_forcing_prob=0.0,
                    norm_stats=run_norm_stats,
                )
                test_metrics = run_test_rollout_chunks(
                    model=model,
                    loader=test_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    chunk_steps=args.y,
                    context_len=args.x,
                    rollout_steps=args.test_rollout_steps,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    desc=f"shuffle {run_idx + 1} test-rollout",
                    run_loader=run_loader,
                    norm_stats=run_norm_stats,
                )
            else:
                best_val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    optimizer=None,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    grad_clip=args.grad_clip,
                    desc=f"shuffle {run_idx + 1} best-val",
                    forecast_mode=args.forecast_mode,
                    ar_chunk_size=args.ar_chunk_size,
                    norm_stats=run_norm_stats,
                )
            if args.forecast_mode == "long":
                test_metrics = run_test_rollout_chunks(
                    model=model,
                    loader=test_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    chunk_steps=args.y,
                    context_len=args.x,
                    rollout_steps=args.test_rollout_steps,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    desc=f"shuffle {run_idx + 1} test-rollout",
                    run_loader=run_loader,
                    norm_stats=run_norm_stats,
                )
            elif args.forecast_mode == "short":
                test_metrics = run_epoch(
                    model=model,
                    loader=test_loader,
                    edge_index=edge_index,
                    dt=args.dt,
                    optimizer=None,
                    lambda_mse=args.lambda_mse,
                    lambda_mae=args.lambda_mae,
                    grad_clip=args.grad_clip,
                    desc=f"shuffle {run_idx + 1} test",
                    forecast_mode=args.forecast_mode,
                    ar_chunk_size=args.ar_chunk_size,
                    norm_stats=run_norm_stats,
                )
        run_val_metrics.append(best_val_metrics)
        run_test_scores.append(test_metrics)

        print(
            f"Shuffle {run_idx + 1} Val  | total={best_val_metrics['total']:.6f} "
            f"mse={best_val_metrics['mse']:.6f} mae={best_val_metrics['mae']:.6f} "
            f"pcc={best_val_metrics['pcc']:.4f} scc={best_val_metrics['scc']:.4f}"
        )
        print(
            f"Shuffle {run_idx + 1} Test | total={test_metrics['total']:.6f} "
            f"mse={test_metrics['mse']:.6f} mae={test_metrics['mae']:.6f} "
            f"pcc={test_metrics['pcc']:.4f} scc={test_metrics['scc']:.4f}"
        )

    def _ms(metrics_list, key):
        vals = [m[key] for m in metrics_list]
        return float(np.mean(vals)), float(np.std(vals))

    print(f"\nShuffle Val Summary (mean ± std across {num_shuffles} shuffles):")
    print(
        f"  MSE   : {_ms(run_val_metrics, 'mse')[0]:.6f} ± {_ms(run_val_metrics, 'mse')[1]:.6f}"
    )
    print(
        f"  MAE   : {_ms(run_val_metrics, 'mae')[0]:.6f} ± {_ms(run_val_metrics, 'mae')[1]:.6f}"
    )
    print(
        f"  PCC   : {_ms(run_val_metrics, 'pcc')[0]:.4f}  ± {_ms(run_val_metrics, 'pcc')[1]:.4f}"
    )
    print(
        f"  SCC   : {_ms(run_val_metrics, 'scc')[0]:.4f}  ± {_ms(run_val_metrics, 'scc')[1]:.4f}"
    )
    print(
        f"  DTW   : {_ms(run_val_metrics, 'dtw')[0]:.6f} ± {_ms(run_val_metrics, 'dtw')[1]:.6f}"
    )

    print(f"\nShuffle Test Summary (mean ± std across {num_shuffles} shuffles):")
    print(
        f"  MSE   : {_ms(run_test_scores, 'mse')[0]:.6f} ± {_ms(run_test_scores, 'mse')[1]:.6f}"
    )
    print(
        f"  MAE   : {_ms(run_test_scores, 'mae')[0]:.6f} ± {_ms(run_test_scores, 'mae')[1]:.6f}"
    )
    print(
        f"  PCC   : {_ms(run_test_scores, 'pcc')[0]:.4f}  ± {_ms(run_test_scores, 'pcc')[1]:.4f}"
    )
    print(
        f"  SCC   : {_ms(run_test_scores, 'scc')[0]:.4f}  ± {_ms(run_test_scores, 'scc')[1]:.4f}"
    )
    print(
        f"  DTW   : {_ms(run_test_scores, 'dtw')[0]:.6f} ± {_ms(run_test_scores, 'dtw')[1]:.6f}"
    )


if __name__ == "__main__":
    main()
