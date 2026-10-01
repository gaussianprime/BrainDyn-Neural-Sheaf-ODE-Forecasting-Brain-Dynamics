from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from data.sn_dataset import DATA_NPZ_PATH, make_dataloaders
from model.braindyn import BrainDyn, BrainDynConfig

# Only the numerical tolerances are shared with the LEMON/fMRI Granger builder
from model.graph_builders import _GRANGER_MAX_RHO2, _granger_collinear_tol
from model.losses import dtw_mean_normalized, total_loss


# NEST uses subject-grouped k-fold CV (main.py uses repeated shuffle splits).
def compute_cv_fold_indices(
    groups: np.ndarray, cv_folds: int, seed: int
) -> list[np.ndarray]:
    """Deterministic subject-grouped CV fold index arrays (window-balanced).

    Every sample sharing a group label lands in the same fold. Folds are balanced
    on window count via a seeded longest-processing-time-first greedy assignment.
    """
    if isinstance(groups, (int, np.integer)):
        raise TypeError("compute_cv_fold_indices takes per-sample group labels.")
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")
    labels = np.asarray(groups)
    uniq, codes = np.unique(labels, return_inverse=True)
    n_groups = len(uniq)
    if n_groups < cv_folds:
        raise ValueError(
            f"only {n_groups} distinct groups for {cv_folds}-fold grouped CV"
        )
    counts = np.bincount(codes, minlength=n_groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    order = perm[np.argsort(-counts[perm], kind="stable")]
    fold_load = np.zeros(cv_folds, dtype=np.int64)
    fold_of_group = np.empty(n_groups, dtype=np.int64)
    for g in order:
        f = int(np.argmin(fold_load))
        fold_of_group[g] = f
        fold_load[f] += counts[g]
    fold_of_sample = fold_of_group[codes]
    folds = [np.flatnonzero(fold_of_sample == f) for f in range(cv_folds)]
    assert all(len(f) > 0 for f in folds), "grouped split produced an empty fold"
    return folds


def fold_train_val_indices(
    fold_indices: list[np.ndarray], fold_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx)`` for one fold from the shared split."""
    val_idx = fold_indices[fold_idx]
    train_idx = np.concatenate(
        [fold_indices[i] for i in range(len(fold_indices)) if i != fold_idx]
    )
    return train_idx, val_idx


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU.

    AMP is gated to CUDA only by the caller, so the cuda-specific autocast /
    GradScaler paths stay disabled on MPS. Set PYTORCH_ENABLE_MPS_FALLBACK=1
    when running so any op not yet implemented on MPS falls back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_to_model_tensors_nest(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert NEST forecasting batch to BrainDyn shapes.

    Input batch:
      x: (B, Lx, N)
      y: (B, Ly, N)

    Returns:
      x_history: (B, N, Lx, 1)
      y_true:    (Ly, B, N, 1)
    """
    x_ctx = batch["x"].to(device=device, dtype=torch.float32)
    y_future = batch["y"].to(device=device, dtype=torch.float32)

    x_history = x_ctx.permute(0, 2, 1).unsqueeze(-1)
    y_true = y_future.permute(1, 0, 2).unsqueeze(-1)
    return x_history, y_true


def adjacency_to_edge_index(adjacency: torch.Tensor) -> torch.Tensor:
    """Convert adjacency matrix (N, N) to edge_index (2, E).

    Args:
        adjacency: (N, N) binary or continuous adjacency matrix where adj[i,j] represents edge i->j

    Returns:
        edge_index: (2, E) where E is the number of edges, format is [source, destination]
    """
    # Clone so fill_diagonal_ never mutates the caller's (possibly cached) tensor.
    adj = adjacency.bool().clone()
    adj.fill_diagonal_(False)
    src_idx, dst_idx = torch.where(
        adj
    )  # torch.where returns (row_indices, col_indices)
    if src_idx.shape[0] == 0:
        # Return empty edge index with correct shape if no edges
        return torch.zeros((2, 0), dtype=torch.long, device=adjacency.device)
    edge_index = torch.stack([src_idx, dst_idx], dim=0).long()
    return edge_index


def get_subject_edge_indices_from_batch(
    batch: dict,
    device: torch.device,
) -> list[torch.Tensor]:
    """Extract per-subject edge indices from batch metadata.

    Args:
        batch: dictionary with "meta" containing "adjacency" (B, N, N)
        device: torch device

    Returns:
        List of edge_index tensors, one per subject in batch
    """
    if "meta" not in batch or "adjacency" not in batch["meta"]:
        raise ValueError("Batch must contain 'meta' with 'adjacency' field")

    adjacency_batch = batch["meta"]["adjacency"].to(device=device)  # (B, N, N)

    edge_indices = []
    for b in range(adjacency_batch.shape[0]):
        edge_idx = adjacency_to_edge_index(adjacency_batch[b])
        edge_indices.append(edge_idx)

    return edge_indices


def collect_node_features_nest(loader: DataLoader, max_batches: int) -> torch.Tensor:
    """Collect node-wise features from NEST forecasting context windows.

    Returns:
        (N, S) tensor with concatenated samples over batch/time.
    """
    chunks: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x_ctx = batch["x"].float()  # (B, Lx, N)
        node_series = x_ctx.permute(2, 0, 1).reshape(x_ctx.shape[2], -1)  # (N, B*Lx)
        chunks.append(node_series)

    if not chunks:
        raise RuntimeError(
            "Unable to build Granger graph: train loader produced zero batches."
        )

    features = torch.cat(chunks, dim=1)
    features = (features - features.mean(dim=1, keepdim=True)) / (
        features.std(dim=1, keepdim=True) + 1e-6
    )
    return features


def build_granger_graph_nest(
    loader: DataLoader,
    max_batches: int,
    threshold: float,
    lag: int,
    threshold_mode: str,
    extra_edges: int,
    gt_ref_adj: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build directed graph from pairwise Granger causality scores (src -> dst)."""
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if extra_edges < 0:
        raise ValueError(f"extra_edges must be >= 0, got {extra_edges}")

    features = collect_node_features_nest(loader=loader, max_batches=max_batches)
    x = features.detach().cpu().numpy().astype(np.float64).T  # (T, N)
    T, N = x.shape
    if T <= lag + 1:
        raise RuntimeError(f"Not enough samples for Granger graph: T={T}, lag={lag}")

    x = x - x.mean(axis=0, keepdims=True)
    y_t = x[lag:, :]
    x_lag = x[:-lag, :]

    # Pairwise log-RSS-ratio scores via Frisch-Waugh-Lovell on two second-moment matrices
    y_c = y_t - y_t.mean(axis=0, keepdims=True)
    x_c = x_lag - x_lag.mean(axis=0, keepdims=True)

    # Unit-norm each column first. The score is a partial correlation, so it is exactly
    # invariant to per-column scaling; rescaling changes no result but keeps the Gram
    # O(1) and makes the projection divisor exactly 1.
    # All degeneracy tests below compare *energies* (squared norms), matching
    # model.graph_builders -- comparing norms instead would silently apply the
    # tolerance squared.
    collinear_tol = _granger_collinear_tol(y_t.shape[0])
    tiny = np.finfo(np.float64).tiny
    x_energy = np.einsum("ij,ij->j", x_c, x_c)
    y_energy = np.einsum("ij,ij->j", y_c, y_c)
    dead = x_energy <= collinear_tol * np.maximum(
        np.einsum("ij,ij->j", x_lag, x_lag), tiny
    )
    y_dead = y_energy <= collinear_tol * np.maximum(
        np.einsum("ij,ij->j", y_t, y_t), tiny
    )
    x_c = x_c / np.where(dead, 1.0, np.sqrt(x_energy))
    y_c = y_c / np.where(y_dead, 1.0, np.sqrt(y_energy))

    # Indexed [src, dst] throughout: src is the row axis, dst the column axis, so the matmuls yield all N**2 pairs at once with no loop over either index.
    gram = x_c.T @ x_c
    cross = x_c.T @ y_c
    xy = np.diag(cross).copy()

    rss_r = np.where(y_dead, 0.0, np.maximum(1.0 - xy * xy, 0.0))
    num = cross - gram * xy[None, :]
    den = np.maximum(1.0 - gram * gram, 0.0)

    # den -> 0 means src is collinear with the dst-lag and rss_r -> 0 means dst is already fully explained by its own lag; both mean src adds nothing, i.e. 0.
    degenerate = (
        (den <= collinear_tol)
        | dead[:, None]
        | (rss_r <= collinear_tol)[None, :]
    )
    rho2 = np.divide(
        num * num,
        den * rss_r[None, :],
        out=np.zeros_like(num),
        where=~degenerate,
    )
    granger = -np.log1p(-np.clip(rho2, 0.0, _GRANGER_MAX_RHO2))
    np.maximum(granger, 0.0, out=granger)
    np.fill_diagonal(granger, 0.0)

    score = torch.from_numpy(granger).float()
    n_nodes = score.shape[0]
    diag_mask = torch.eye(n_nodes, dtype=torch.bool)

    info = {
        "mode": threshold_mode,
        "threshold": float(threshold),
        "extra_edges": int(extra_edges),
        "gt_edges_total": 0,
        "gt_edges_kept": 0,
        "selected_edges": 0,
        "auto_strategy": "",
    }

    if threshold_mode == "manual":
        adjacency = score > float(threshold)
    elif threshold_mode == "gt_superset_auto":
        if gt_ref_adj is None:
            raise RuntimeError(
                "gt_superset_auto requires a ground-truth reference adjacency"
            )
        gt_mask = gt_ref_adj.bool().clone()
        gt_mask.fill_diagonal_(False)
        gt_scores = score[gt_mask]
        if gt_scores.numel() == 0:
            raise RuntimeError(
                "Ground-truth union adjacency has zero edges; cannot calibrate threshold."
            )
        # Robust auto threshold: avoid thr=0 turning the graph nearly complete.
        # If GT contains positive scores, use smallest positive GT score and force GT edges in.
        # If all GT scores are zero, fall back to GT-only (no extra threshold-based edges).
        gt_pos = gt_scores[gt_scores > 0]
        if gt_pos.numel() > 0:
            thr = float(gt_pos.min().item())
            adjacency = (score >= thr) | gt_mask
            info["auto_strategy"] = "min_positive_gt_score_with_gt_force"
        else:
            thr = float("inf")
            adjacency = gt_mask.clone()
            info["auto_strategy"] = "all_gt_scores_zero_gt_only"
        info["threshold"] = thr
        info["gt_edges_total"] = int(gt_mask.sum().item())
    elif threshold_mode == "gt_superset_topk":
        if gt_ref_adj is None:
            raise RuntimeError(
                "gt_superset_topk requires a ground-truth reference adjacency"
            )
        gt_mask = gt_ref_adj.bool().clone()
        gt_mask.fill_diagonal_(False)
        adjacency = gt_mask.clone()
        cand_mask = (~gt_mask) & (~diag_mask)
        cand_scores = score[cand_mask]
        k = min(int(extra_edges), int(cand_scores.numel()))
        if k > 0:
            top_idx = torch.topk(cand_scores, k=k, largest=True).indices
            take = torch.zeros_like(cand_scores, dtype=torch.bool)
            take[top_idx] = True
            adjacency[cand_mask] = take
        info["gt_edges_total"] = int(gt_mask.sum().item())
    else:
        raise ValueError(f"Unknown granger threshold mode: {threshold_mode}")

    adjacency.fill_diagonal_(False)
    if gt_ref_adj is not None:
        gt_mask = gt_ref_adj.bool().clone()
        gt_mask.fill_diagonal_(False)
        info["gt_edges_total"] = int(gt_mask.sum().item())
        info["gt_edges_kept"] = int((adjacency & gt_mask).sum().item())
    info["selected_edges"] = int(adjacency.sum().item())

    # granger[src, dst] scores src->dst, so adjacency[i, j] is edge i->j. torch.where
    # returns (row=src, col=dst): keep edge_index = [src, dst] so it matches both the
    # sheaf convention (edge_index[0]=src) and the granger_prior[src, dst] lookup used
    # by the prior-delta regularizer.
    src_idx, dst_idx = torch.where(adjacency)
    edge_index = torch.stack([src_idx, dst_idx], dim=0).long()
    if edge_index.shape[1] == 0:
        raise RuntimeError(
            f"Granger graph has no edges at threshold={threshold:.4f}. Lower --granger_threshold."
        )

    return edge_index, score, info


def collect_gt_reference_adjacency_nest(
    loader: DataLoader,
    max_batches: int,
    reference: str,
) -> torch.Tensor:
    """Collect GT reference adjacency for calibration on training data.

    reference:
      - first_subject: first observed subject adjacency (stable sparse template)
      - union: edgewise union over seen subjects (can be very dense)
      - intersection: edgewise intersection over seen subjects (very strict)
    """
    ref_adj: torch.Tensor | None = None
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        meta = batch.get("meta", {})
        adj = meta.get("adjacency", None)
        if adj is None:
            raise RuntimeError(
                "Batch meta does not contain adjacency; cannot build GT reference graph."
            )
        adj_t = adj.detach().cpu().bool()
        if reference == "first_subject":
            ref_adj = adj_t[0].clone()
            break
        batch_any = adj_t.any(dim=0)
        if ref_adj is None:
            ref_adj = batch_any
        elif reference == "union":
            ref_adj |= batch_any
        elif reference == "intersection":
            ref_adj &= batch_any
        else:
            raise ValueError(f"Unknown granger_gt_reference: {reference}")

    if ref_adj is None:
        raise RuntimeError("No batches available to collect GT reference adjacency.")
    ref_adj.fill_diagonal_(False)
    return ref_adj


def run_forecasting_epoch(
    model: BrainDyn,
    loader: DataLoader,
    edge_index: torch.Tensor | None,
    dt: float,
    optimizer: torch.optim.Optimizer | None,
    lambda_mse: float,
    lambda_mae: float,
    grad_clip: float,
    use_amp: bool,
    desc: str,
    use_subject_graphs: bool = True,
    lambda_l1_delta: float = 0.0,
    lambda_prior_delta: float = 0.0,
    granger_score_mat: torch.Tensor | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train(is_train)

    total_running = 0.0
    mse_running = 0.0
    mae_running = 0.0
    pcc_running = 0.0
    scc_running = 0.0
    dtw_running = 0.0
    l1_delta_running = 0.0
    prior_delta_running = 0.0
    n_batches = 0
    device = select_device()
    use_delta_reg = lambda_l1_delta > 0.0 or lambda_prior_delta > 0.0

    # Pre-normalize Granger score matrix to [0, 1] once per epoch for efficiency
    granger_prior: torch.Tensor | None = None
    if granger_score_mat is not None and lambda_prior_delta > 0.0:
        gs = granger_score_mat.float().to(device)
        gs_min, gs_max = gs.min(), gs.max()
        granger_prior = (gs - gs_min) / (gs_max - gs_min).clamp(min=1e-8)

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x_history, y_true = batch_to_model_tensors_nest(batch, device)

        # Extract per-subject edge indices from batch metadata if using subject-specific graphs
        if use_subject_graphs:
            subject_edge_indices = get_subject_edge_indices_from_batch(batch, device)
        else:
            # Fall back to provided edge_index if not using subject graphs
            if edge_index is None:
                raise ValueError("edge_index required when use_subject_graphs=False")
            subject_edge_indices = [edge_index] * x_history.shape[0]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Process each subject in batch with their own edge_index
                y_pred_list = []
                l1_reg = torch.zeros(1, device=device)
                prior_reg = torch.zeros(1, device=device)

                for b in range(x_history.shape[0]):
                    x_hist_b = x_history[b : b + 1]  # (1, N, T, F)
                    edge_idx_b = subject_edge_indices[b]

                    out = model(
                        x_history=x_hist_b,
                        edge_index=edge_idx_b,
                        pred_steps=y_true.shape[0],
                        dt=dt,
                        autoregressive=False,
                        return_aux=use_delta_reg,
                    )
                    y_pred_list.append(out["x_pred"])

                    # Delta regularization (only when lambdas > 0)
                    if use_delta_reg and "aux_seq" in out:
                        delta_list = [
                            aux["delta"] for aux in out["aux_seq"] if "delta" in aux
                        ]
                        if delta_list:
                            # Average delta magnitude over predicted time steps: (1, E)
                            delta_avg = torch.stack(delta_list, dim=0).mean(dim=0)
                            delta_norm_b = delta_avg.norm(p=2, dim=-1)  # (1, E)

                            # L1 sparsity: encourage most edges to have small activation
                            if lambda_l1_delta > 0.0:
                                l1_reg = l1_reg + lambda_l1_delta * delta_norm_b.mean()

                            # Prior-distance: pull delta magnitudes towards the Granger prior scores.
                            # Edges with high Granger score -> high delta, low Granger score -> low delta.
                            if lambda_prior_delta > 0.0 and granger_prior is not None:
                                src_nodes = edge_idx_b[0]
                                dst_nodes = edge_idx_b[1]
                                prior_labels = granger_prior[
                                    src_nodes, dst_nodes
                                ]  # (E,) in [0,1]
                                dn = delta_norm_b[0]  # (E,)
                                # Normalize to [0,1] so it can be compared with Granger prior
                                dn_scaled = dn / (dn.detach().max().clamp(min=1e-6))
                                prior_reg = prior_reg + lambda_prior_delta * torch.mean(
                                    (dn_scaled - prior_labels) ** 2
                                )

                # Concatenate all predictions at once and compute losses on full batch
                y_pred = torch.cat(y_pred_list, dim=1)  # (Ly, B, N, F)
                batch_losses = total_loss(
                    y_pred, y_true, lambda_mse=lambda_mse, lambda_mae=lambda_mae
                )
                loss = batch_losses["total"] + l1_reg.squeeze() + prior_reg.squeeze()

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

        total_running += float(batch_losses["total"].detach().cpu())
        mse_running += float(batch_losses["mse"].detach().cpu())
        mae_running += float(batch_losses["mae"].detach().cpu())
        l1_delta_running += float(l1_reg.detach().cpu())
        prior_delta_running += float(prior_reg.detach().cpu())

        y_pred_np = y_pred.detach().cpu().numpy().ravel()
        y_true_np = y_true.detach().cpu().numpy().ravel()
        try:
            pcc_val = float(pearsonr(y_pred_np, y_true_np).statistic)
        except Exception:
            pcc_val = float("nan")
        try:
            scc_val = float(spearmanr(y_pred_np, y_true_np).statistic)
        except Exception:
            scc_val = float("nan")

        dtw_val = dtw_mean_normalized(
            y_pred.detach().cpu().numpy(),
            y_true.detach().cpu().numpy(),
        )

        pcc_running += pcc_val
        scc_running += scc_val
        dtw_running += float(dtw_val)
        n_batches += 1

        pbar.set_postfix(
            {
                "total": f"{total_running / n_batches:.4f}",
                "mse": f"{mse_running / n_batches:.4f}",
                "pcc": f"{pcc_running / n_batches:.4f}",
                "scc": f"{scc_running / n_batches:.4f}",
                "dtw": f"{dtw_running / n_batches:.4f}",
                "l1d": f"{l1_delta_running / n_batches:.4f}",
                "prd": f"{prior_delta_running / n_batches:.4f}",
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
            "l1_delta": float("nan"),
            "prior_delta": float("nan"),
        }

    return {
        "total": total_running / n_batches,
        "mse": mse_running / n_batches,
        "mae": mae_running / n_batches,
        "pcc": pcc_running / n_batches,
        "scc": scc_running / n_batches,
        "dtw": dtw_running / n_batches,
        "l1_delta": l1_delta_running / n_batches,
        "prior_delta": prior_delta_running / n_batches,
    }


def evaluate_forecasting_split(
    model: BrainDyn,
    loader: DataLoader,
    edge_index: torch.Tensor | None,
    dt: float,
    lambda_mse: float,
    lambda_mae: float,
    grad_clip: float,
    use_amp: bool,
    split_name: str,
    use_subject_graphs: bool = True,
    lambda_l1_delta: float = 0.0,
    lambda_prior_delta: float = 0.0,
    granger_score_mat: torch.Tensor | None = None,
) -> Dict[str, float]:
    metrics = run_forecasting_epoch(
        model=model,
        loader=loader,
        edge_index=edge_index,
        dt=dt,
        optimizer=None,
        lambda_mse=lambda_mse,
        lambda_mae=lambda_mae,
        grad_clip=grad_clip,
        use_amp=use_amp,
        desc=f"forecast {split_name}",
        use_subject_graphs=use_subject_graphs,
        lambda_l1_delta=lambda_l1_delta,
        lambda_prior_delta=lambda_prior_delta,
        granger_score_mat=granger_score_mat,
    )
    print(
        f"{split_name.capitalize()} forecast | total={metrics['total']:.4f} "
        f"mse={metrics['mse']:.4f} mae={metrics['mae']:.4f} "
        f"pcc={metrics['pcc']:.4f} scc={metrics['scc']:.4f} dtw={metrics['dtw']:.4f}"
    )
    return metrics


def make_subset_loader(
    dataset: ConcatDataset | Subset | torch.utils.data.Dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def metric_mean_std(metrics: list[Dict[str, float]], key: str) -> tuple[float, float]:
    values = np.asarray([m[key] for m in metrics], dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))


def print_cv_summary(title: str, metrics: list[Dict[str, float]]) -> None:
    print(f"\n{title} (mean +/- std across folds):")
    print(
        f"  total : {metric_mean_std(metrics, 'total')[0]:.4f} +/- {metric_mean_std(metrics, 'total')[1]:.4f}"
    )
    print(
        f"  mse   : {metric_mean_std(metrics, 'mse')[0]:.4f} +/- {metric_mean_std(metrics, 'mse')[1]:.4f}"
    )
    print(
        f"  mae   : {metric_mean_std(metrics, 'mae')[0]:.4f} +/- {metric_mean_std(metrics, 'mae')[1]:.4f}"
    )
    print(
        f"  pcc   : {metric_mean_std(metrics, 'pcc')[0]:.4f} +/- {metric_mean_std(metrics, 'pcc')[1]:.4f}"
    )
    print(
        f"  scc   : {metric_mean_std(metrics, 'scc')[0]:.4f} +/- {metric_mean_std(metrics, 'scc')[1]:.4f}"
    )
    print(
        f"  dtw   : {metric_mean_std(metrics, 'dtw')[0]:.4f} +/- {metric_mean_std(metrics, 'dtw')[1]:.4f}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train BrainDyn on unperturbed simulated NEST trajectories for forecasting.",
    )

    ap.add_argument("--npz_path", type=str, default=str(DATA_NPZ_PATH))

    ap.add_argument(
        "--x",
        type=int,
        default=30,
        help="Context length for unperturbed forecasting windows.",
    )
    ap.add_argument("--y", type=int, default=10, help="Forecast horizon length.")
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument(
        "--cv_folds",
        type=int,
        default=3,
        help="Number of cross-validation folds over the combined forecasting train+val pool.",
    )

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--cache",
        action="store_true",
        help="Cache arrays in RAM (instead of memory mapping).",
    )
    ap.add_argument("--no_pin_memory", action="store_true")

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--lstm_dropout", type=float, default=0.0)
    ap.add_argument("--map_hidden_dim", type=int, default=16)
    ap.add_argument("--vf_hidden_dim", type=int, default=128)
    ap.add_argument(
        "--ode_method",
        type=str,
        default="rk4",
        choices=["rk4", "dopri5", "euler", "midpoint"],
    )
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--ablation_gcn", action="store_true")
    ap.add_argument("--ablation_no_lstm", action="store_true")
    ap.add_argument(
        "--static_edge_maps",
        action="store_true",
        help="Use static per-edge maps instead of the default restriction-map MLP.",
    )
    ap.add_argument("--graph_mode", type=str, default="granger", choices=["granger"])
    ap.add_argument("--graph_max_batches", type=int, default=30)
    ap.add_argument(
        "--granger_threshold",
        type=float,
        default=0.00763873802497983,
        help="Manual Granger threshold. Default yields about 7833 candidate edges on train folds and retains about 320/400 GT edges.",
    )
    ap.add_argument("--granger_lag", type=int, default=1)
    ap.add_argument(
        "--granger_threshold_mode",
        type=str,
        default="manual",
        choices=["manual", "gt_superset_auto", "gt_superset_topk"],
        help="How to convert Granger scores into edges.",
    )
    ap.add_argument(
        "--granger_extra_edges",
        type=int,
        default=200,
        help="Extra non-GT edges to add when --granger_threshold_mode=gt_superset_topk.",
    )
    ap.add_argument(
        "--granger_gt_reference",
        type=str,
        default="first_subject",
        choices=["first_subject", "union", "intersection"],
        help="GT reference used by Granger superset modes.",
    )

    ap.add_argument(
        "--epochs", "--epochs_forecast", dest="epochs", type=int, default=60
    )
    ap.add_argument("--lr", "--lr_forecast", dest="lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_mae", type=float, default=0.0)
    ap.add_argument(
        "--lambda_l1_delta",
        type=float,
        default=1.0,
        help="L1 sparsity on edge delta norms: penalizes mean ||delta|| across all candidate edges.",
    )
    ap.add_argument(
        "--lambda_prior_delta",
        type=float,
        default=1.0,
        help="Prior-distance regularization: MSE between normalized ||delta|| per edge and the subject GT binary mask. "
        "Encourages high delta for GT edges and low delta for non-GT edges.",
    )

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save_path",
        "--save_forecast_path",
        dest="save_path",
        type=str,
        default="checkpoints/braindyn_nest_unperturbed_best.pt",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = select_device()
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}")

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset.npz not found: {npz_path}")

    forecast_loaders = make_dataloaders(
        npz_path=str(npz_path),
        task_mode="forecasting",
        x=args.x,
        y=args.y,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
        cache=args.cache,
        pin_memory=(not args.no_pin_memory),
        verbose=True,
    )

    train_loader = forecast_loaders["train"]
    val_loader = forecast_loaders["val"]
    test_loader = forecast_loaders["test"]
    use_pin = (not args.no_pin_memory) and torch.cuda.is_available()

    n_nodes = int(train_loader.dataset.n_channels)

    combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    if args.cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {args.cv_folds}")
    if len(combined_dataset) < args.cv_folds:
        raise ValueError(
            f"Not enough combined train+val samples ({len(combined_dataset)}) for {args.cv_folds}-fold CV"
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Subject-grouped CV: build a per-window subject-id array over the combined
    # pool, then split with the same subject-grouped helper the main entrypoint
    # uses so a subject's windows never straddle a fold's train/val boundary.
    # ConcatDataset concatenates in order (train_ds windows, then val_ds windows),
    # which matches the _index ordering below. SNDataset stores the subject id as
    # element [0] of each _index tuple (== meta["subject_index"]).
    train_ds, val_ds = combined_dataset.datasets
    groups = np.array(
        [idx[0] for idx in train_ds._index] + [idx[0] for idx in val_ds._index],
        dtype=np.int64,
    )  # groups[g] = subject id of combined_dataset[g]
    n_subjects_pool = int(np.unique(groups).size)
    if n_subjects_pool < args.cv_folds:
        raise ValueError(
            f"Only {n_subjects_pool} subjects in train+val pool for "
            f"{args.cv_folds}-fold subject-grouped CV."
        )
    fold_indices = compute_cv_fold_indices(groups, args.cv_folds, args.seed)

    print(f"\n=== Unperturbed Forecasting (NEST, {args.cv_folds}-fold CV) ===")

    fold_val_metrics: list[Dict[str, float]] = []
    fold_test_metrics: list[Dict[str, float]] = []

    for fold_idx in range(args.cv_folds):
        train_idx, val_idx = fold_train_val_indices(fold_indices, fold_idx)
        assert not (
            set(groups[train_idx].tolist()) & set(groups[val_idx].tolist())
        ), (
            f"fold {fold_idx + 1}: subjects shared between train and val — "
            "split is not subject-grouped"
        )
        # Reseed per fold so fold k is independent of RNG consumed by folds 0..k-1
        # (weight init + batch order); the split itself was fixed above from
        # args.seed and is unaffected. Matches main.py's CV reproducibility.
        set_seed(args.seed * 1000 + fold_idx)

        fold_train_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=train_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin,
            shuffle=True,
        )
        fold_val_loader = make_subset_loader(
            dataset=combined_dataset,
            indices=val_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin,
            shuffle=False,
        )

        if args.graph_mode == "granger":
            gt_ref_adj = None
            if args.granger_threshold_mode in ("gt_superset_auto", "gt_superset_topk"):
                gt_ref_adj = collect_gt_reference_adjacency_nest(
                    loader=fold_train_loader,
                    max_batches=args.graph_max_batches,
                    reference=args.granger_gt_reference,
                )

            edge_index_cpu, graph_score, graph_info = build_granger_graph_nest(
                loader=fold_train_loader,
                max_batches=args.graph_max_batches,
                threshold=args.granger_threshold,
                lag=args.granger_lag,
                threshold_mode=args.granger_threshold_mode,
                extra_edges=args.granger_extra_edges,
                gt_ref_adj=gt_ref_adj,
            )
            graph_tag = "Granger"
            graph_thresh_str = (
                f"mode={args.granger_threshold_mode}, threshold={graph_info['threshold']:.4f}, "
                f"lag={args.granger_lag}, extra_edges={args.granger_extra_edges}, "
                f"gt_ref={args.granger_gt_reference}"
            )
        else:
            raise AssertionError(f"Unhandled graph mode: {args.graph_mode}")

        edge_index = edge_index_cpu.to(device)

        n_edges = edge_index.shape[1]
        density = n_edges / max(n_nodes * (n_nodes - 1), 1)
        print(
            f"Fold {fold_idx + 1}/{args.cv_folds} | {graph_tag} graph: {n_edges} directed edges "
            f"| density={density:.3f} | {graph_thresh_str}"
        )
        if graph_score is not None:
            print(
                f"Fold {fold_idx + 1}/{args.cv_folds} | {graph_tag} score mean={graph_score.abs().mean().item():.3f}, "
                f"max={graph_score.abs().max().item():.3f}"
            )
        if args.graph_mode == "granger" and graph_info["gt_edges_total"] > 0:
            print(
                f"Fold {fold_idx + 1}/{args.cv_folds} | GT coverage in {graph_tag}: "
                f"{graph_info['gt_edges_kept']}/{graph_info['gt_edges_total']} edges"
            )
            if graph_info.get("auto_strategy"):
                print(
                    f"Fold {fold_idx + 1}/{args.cv_folds} | {graph_tag} auto strategy: "
                    f"{graph_info['auto_strategy']}"
                )
        if density > 0.95:
            print(
                f"WARNING: fold {fold_idx + 1} graph density is {density:.3f} (very dense). "
                "Consider --granger_threshold_mode gt_superset_topk with smaller --granger_extra_edges."
            )

        model_cfg = BrainDynConfig(
            signal_dim=1,
            hidden_dim=args.hidden_dim,
            num_nodes=n_nodes,
            window_size=args.x,
            lstm_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
            map_hidden_dim=args.map_hidden_dim,
            vf_hidden_dim=args.vf_hidden_dim,
            ode_method=args.ode_method,
            use_gcn=args.ablation_gcn,
            use_lstm_encoder=(not args.ablation_no_lstm),
            edge_specific_maps=args.static_edge_maps,
            sheaf_mlp_maps=(not args.static_edge_maps),
        )
        model = BrainDyn(model_cfg).to(device)
        # Register the per-fold global edge universe for edge-specific restriction
        # maps before the optimizer (no-op when --edge_specific_maps is not set).
        # NEST training shares this graph across subjects (use_subject_graphs=False).
        model.register_restriction_edges(edge_index)

        optimizer_forecast = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler_forecast = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_forecast,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

        best_val_forecast = float("inf")
        fold_save_path = save_path.with_name(
            f"{save_path.stem}_fold{fold_idx + 1}{save_path.suffix}"
        )
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_forecasting_epoch(
                model=model,
                loader=fold_train_loader,
                edge_index=edge_index,
                dt=args.dt,
                optimizer=optimizer_forecast,
                lambda_mse=args.lambda_mse,
                lambda_mae=args.lambda_mae,
                grad_clip=args.grad_clip,
                use_amp=use_amp,
                desc=f"fold {fold_idx + 1} train [{epoch}/{args.epochs}]",
                use_subject_graphs=False,
                lambda_l1_delta=args.lambda_l1_delta,
                lambda_prior_delta=args.lambda_prior_delta,
                granger_score_mat=graph_score,
            )
            val_metrics = run_forecasting_epoch(
                model=model,
                loader=fold_val_loader,
                edge_index=edge_index,
                dt=args.dt,
                optimizer=None,
                lambda_mse=args.lambda_mse,
                lambda_mae=args.lambda_mae,
                grad_clip=args.grad_clip,
                use_amp=use_amp,
                desc=f"fold {fold_idx + 1} val [{epoch}/{args.epochs}]",
                use_subject_graphs=False,
                lambda_l1_delta=args.lambda_l1_delta,
                lambda_prior_delta=args.lambda_prior_delta,
                granger_score_mat=graph_score,
            )
            scheduler_forecast.step(val_metrics["total"])

            print(
                f"Fold {fold_idx + 1}/{args.cv_folds} Epoch {epoch:03d} | "
                f"train total={train_metrics['total']:.4f} mse={train_metrics['mse']:.4f} mae={train_metrics['mae']:.4f} "
                f"pcc={train_metrics['pcc']:.4f} scc={train_metrics['scc']:.4f} dtw={train_metrics['dtw']:.4f} "
                f"l1d={train_metrics['l1_delta']:.4f} prd={train_metrics['prior_delta']:.4f} | "
                f"val total={val_metrics['total']:.4f} mse={val_metrics['mse']:.4f} mae={val_metrics['mae']:.4f} "
                f"pcc={val_metrics['pcc']:.4f} scc={val_metrics['scc']:.4f} dtw={val_metrics['dtw']:.4f} "
                f"l1d={val_metrics['l1_delta']:.4f} prd={val_metrics['prior_delta']:.4f}"
            )

            if val_metrics["total"] < best_val_forecast:
                best_val_forecast = val_metrics["total"]
                torch.save(
                    {
                        "stage": "unperturbed_forecasting",
                        "model_state": model.state_dict(),
                        "edge_index": edge_index.detach().cpu(),
                        "use_subject_graphs": False,
                        "args": vars(args),
                        "val_metrics": val_metrics,
                        "fold": fold_idx + 1,
                    },
                    fold_save_path,
                )
                print(
                    f"Saved fold {fold_idx + 1} best forecasting checkpoint -> {fold_save_path}"
                )

        state = torch.load(fold_save_path, map_location=device)
        model.load_state_dict(state["model_state"])

        best_val_metrics = evaluate_forecasting_split(
            model=model,
            loader=fold_val_loader,
            edge_index=edge_index,
            dt=args.dt,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            split_name=f"fold {fold_idx + 1} val",
            use_subject_graphs=False,
            lambda_l1_delta=args.lambda_l1_delta,
            lambda_prior_delta=args.lambda_prior_delta,
            granger_score_mat=graph_score,
        )
        test_metrics = evaluate_forecasting_split(
            model=model,
            loader=test_loader,
            edge_index=edge_index,
            dt=args.dt,
            lambda_mse=args.lambda_mse,
            lambda_mae=args.lambda_mae,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            split_name=f"fold {fold_idx + 1} test",
            use_subject_graphs=False,
            lambda_l1_delta=args.lambda_l1_delta,
            lambda_prior_delta=args.lambda_prior_delta,
            granger_score_mat=graph_score,
        )
        fold_val_metrics.append(best_val_metrics)
        fold_test_metrics.append(test_metrics)

    print_cv_summary("CV Val Summary", fold_val_metrics)
    print_cv_summary("CV Test Summary", fold_test_metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
