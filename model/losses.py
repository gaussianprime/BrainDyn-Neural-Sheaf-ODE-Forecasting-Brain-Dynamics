from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mse_forecast_loss(x_pred, x_true):
    """
    x_pred: (T_pred, B, N, F)
    x_true: (T_pred, B, N, F)
    """
    return F.mse_loss(x_pred, x_true)


def mae_forecast_loss(x_pred, x_true):
    return F.l1_loss(x_pred, x_true)


def total_loss(
    x_pred,
    x_true,
    lambda_mse=1.0,
    lambda_mae=0.0,
    mask=None,
):
    """Forecast loss with optional validity mask."""
    if mask is None:
        mse = mse_forecast_loss(x_pred, x_true)
        mae = mae_forecast_loss(x_pred, x_true)
    else:
        valid = mask.to(torch.bool).expand_as(x_pred)
        diff = torch.where(valid, x_pred - x_true, torch.zeros_like(x_pred))
        # nan_to_num's default posinf/neginf is the dtype's max representable
        # value (~3.4e38 for fp32) -- squaring that two lines down overflows
        # straight back to inf, so an already-diverged prediction at one
        # valid timestep was never actually being sanitised. Clamp to a
        # value that stays finite (and fp16-safe) even after squaring,
        # consistent with the +-20 normalized-value convention used
        # elsewhere (see main.py::_clamp_ar_feedback).
        diff = torch.nan_to_num(diff, nan=0.0, posinf=40.0, neginf=-40.0)
        denom = valid.sum().clamp_min(1).to(x_pred.dtype)
        mse = (diff * diff).sum() / denom
        mae = diff.abs().sum() / denom
    total = lambda_mse * mse + lambda_mae * mae

    return {
        "total": total,
        "mse": mse,
        "mae": mae,
    }


def dtw_mean_normalized(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute mean normalised DTW across B*N node-sequences."""
    # Collapse to (T, M) where M = B*N*F
    arr_p = np.asarray(pred, dtype=np.float64)
    arr_t = np.asarray(true, dtype=np.float64)
    T = arr_p.shape[0]
    if T == 0:
        return float("nan")
    # Reshape: (T, M)
    p = arr_p.reshape(T, -1)  # (T, M)
    t = arr_t.reshape(T, -1)  # (T, M)
    M = p.shape[1]

    # Cost matrix: (M, T, T) via broadcasting
    # |p[r, m] - t[c, m]| for all r, c, m
    cost = np.abs(p.T[:, :, None] - t.T[:, None, :])  # (M, T, T)

    # Vectorised DP across all M sequences simultaneously
    dp = np.full((M, T, T), np.inf, dtype=np.float64)
    dp[:, 0, 0] = cost[:, 0, 0]
    for r in range(1, T):
        dp[:, r, 0] = dp[:, r - 1, 0] + cost[:, r, 0]
    for c in range(1, T):
        dp[:, 0, c] = dp[:, 0, c - 1] + cost[:, 0, c]
    for r in range(1, T):
        for c in range(1, T):
            dp[:, r, c] = cost[:, r, c] + np.minimum(
                np.minimum(dp[:, r - 1, c], dp[:, r, c - 1]),
                dp[:, r - 1, c - 1],
            )

    # Normalise by the maximum possible path length (2T - 1) and average
    return float(dp[:, T - 1, T - 1].mean() / max(2 * T - 1, 1))
