from __future__ import annotations

import numpy as np
import torch


def adjacency_to_edge_index(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = adjacency.bool().clone()
    adjacency.fill_diagonal_(False)
    src_idx, dst_idx = torch.where(adjacency)
    if src_idx.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=adjacency.device)
    return torch.stack([src_idx, dst_idx], dim=0).long()


def symmetrize_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return a binary symmetric adjacency ``(N, N)`` from a directed edge_index."""
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if edge_index.numel() > 0:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        A[src, dst] = 1.0
        A[dst, src] = 1.0
    A.fill_diagonal_(0.0)
    return A


def build_lap_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    rank: int,
    eigval_tol: float = 1e-6,
) -> torch.Tensor:
    """Laplacian positional encodings (LapPE) from the symmetrized graph."""
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if num_nodes < 2:
        raise ValueError(f"Need at least 2 nodes for LapPE, got {num_nodes}")

    A = symmetrize_edge_index(edge_index.cpu(), num_nodes)
    deg = A.sum(dim=1)
    dinv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    lap = torch.eye(num_nodes) - dinv_sqrt[:, None] * A * dinv_sqrt[None, :]
    lap = 0.5 * (lap + lap.T)  # guard against tiny asymmetry before eigh

    eigvals, eigvecs = torch.linalg.eigh(lap)
    pe = eigvecs[:, eigvals > eigval_tol][:, :rank]

    if pe.shape[1] > 0:
        col_idx = torch.arange(pe.shape[1])
        pivot = pe.abs().argmax(dim=0)
        signs = torch.sign(pe[pivot, col_idx])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        pe = pe * signs[None, :]

    if pe.shape[1] < rank:
        pad = torch.zeros((num_nodes, rank - pe.shape[1]), dtype=pe.dtype)
        pe = torch.cat([pe, pad], dim=1)
    return pe.float().contiguous()


def build_spatial_graph_from_positions(
    positions,
    k: int = 8,
    radius: float | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, None, dict[str, float | int | str]]:
    """Physical-proximity prior graph from node coordinates."""
    if isinstance(positions, torch.Tensor):
        pos = positions.detach().cpu().numpy().astype(np.float64)
    else:
        pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(pos.shape)}")
    n_nodes = pos.shape[0]
    if n_nodes < 2:
        raise ValueError(f"Need at least 2 nodes, got {n_nodes}")

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.square(diff).sum(axis=-1))  # (N, N)

    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    if radius is not None:
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        adjacency = dist <= float(radius)
        np.fill_diagonal(adjacency, False)
        k_used = 0
    else:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        k_used = min(int(k), n_nodes - 1)
        # k nearest neighbors per node (column 0 of argsort is self at dist 0).
        order = np.argsort(dist, axis=1)
        for i in range(n_nodes):
            neigh = order[i][order[i] != i][:k_used]
            adjacency[i, neigh] = True

    if symmetric:
        adjacency = adjacency | adjacency.T

    edge_index = adjacency_to_edge_index(torch.from_numpy(adjacency))
    if edge_index.shape[1] == 0:
        raise RuntimeError(
            "Spatial graph has no edges. Increase --spatial_k or --spatial_radius."
        )

    info: dict[str, float | int | str] = {
        "mode": "spatial",
        "k": int(k_used),
        "radius": float(radius) if radius is not None else 0.0,
        "threshold": float(radius) if radius is not None else 0.0,
        "selected_edges": int(edge_index.shape[1]),
    }
    return edge_index, None, info


def build_granger_graph_from_features(
    features: torch.Tensor,
    threshold: float,
    lag: int,
    threshold_mode: str,
    topk_edges: int,
    topk_per_node: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    """Build a directed edge_index from one continuous feature sequence.

    ``features`` is shaped ``(N, S)``. The returned score matrix is indexed as
    ``score[src, dst]``.
    """
    if features.ndim != 2:
        raise ValueError(
            f"features must have shape (N, S), got {tuple(features.shape)}"
        )
    series = features.T.unsqueeze(0)
    return build_granger_graph_from_series(
        series=series,
        threshold=threshold,
        lag=lag,
        threshold_mode=threshold_mode,
        topk_edges=topk_edges,
        topk_per_node=topk_per_node,
    )


def build_granger_graph_from_series(
    series: torch.Tensor,
    threshold: float,
    lag: int,
    threshold_mode: str,
    topk_edges: int,
    topk_per_node: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    """Build a directed graph from independent windows shaped ``(B, T, N)``.

    Lagged regression rows are stacked within each window only, so shuffled or
    unrelated windows are not treated as consecutive timepoints.
    """
    if series.ndim != 3:
        raise ValueError(f"series must have shape (B, T, N), got {tuple(series.shape)}")
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if topk_edges < 0:
        raise ValueError(f"topk_edges must be >= 0, got {topk_edges}")

    score = granger_score_from_series(series=series, lag=lag)
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
    }
    return edge_index, score, info


# collinearity guard, as a fraction of a column's own energy
_GRANGER_COLLINEAR_TOL_FLOOR = 1e-10
_GRANGER_COLLINEAR_TOL_EPS_MULT = 100.0

# ceiling on the squared partial correlation, keeps a numerically-perfect fit from producing inf and poisoning the topk selection
_GRANGER_MAX_RHO2 = 1.0 - 1e-12


def _granger_collinear_tol(n_obs: int) -> float:
    """Relative tolerance for "this column adds no independent variance"."""
    return max(
        _GRANGER_COLLINEAR_TOL_FLOOR,
        _GRANGER_COLLINEAR_TOL_EPS_MULT
        * float(n_obs)
        * float(np.finfo(np.float64).eps),
    )


def granger_score_from_series(series: torch.Tensor, lag: int) -> torch.Tensor:
    """Return pairwise RSS/log-ratio Granger scores from ``(B, T, N)`` windows.

    Computed via Frisch-Waugh-Lovell from two second-moment matrices rather than
    one least-squares fit per ordered pair, which is algebraically identical.
    """
    if series.ndim != 3:
        raise ValueError(f"series must have shape (B, T, N), got {tuple(series.shape)}")
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")

    x = series.detach().cpu().numpy().astype(np.float64)
    n_windows, t_steps, n_nodes = x.shape
    if n_nodes < 2:
        raise ValueError(f"Need at least 2 nodes, got {n_nodes}")
    if t_steps <= lag + 1:
        raise RuntimeError(
            f"Not enough samples for Granger graph: T={t_steps}, lag={lag}"
        )

    x = x - x.mean(axis=(0, 1), keepdims=True)
    x = x / (x.std(axis=(0, 1), keepdims=True) + 1e-6)
    # Lagged rows are stacked within each window only, so shuffled or unrelated
    # windows are never treated as consecutive timepoints.
    y_t = x[:, lag:, :].reshape(-1, n_nodes)
    x_lag = x[:, :-lag, :].reshape(-1, n_nodes)
    n_obs = y_t.shape[0]
    if n_obs <= 2:
        raise RuntimeError(
            f"Not enough lagged observations for Granger graph: windows={n_windows}, "
            f"T={t_steps}, lag={lag}"
        )

    granger = _granger_scores_fwl(y_t=y_t, x_lag=x_lag)
    return torch.from_numpy(granger).float()


def _granger_scores_fwl(y_t: np.ndarray, x_lag: np.ndarray) -> np.ndarray:
    """Pairwise Granger log-RSS-ratio scores for aligned ``(n_obs, N)`` designs."""
    collinear_tol = _granger_collinear_tol(y_t.shape[0])

    y_c = y_t - y_t.mean(axis=0, keepdims=True)
    x_c = x_lag - x_lag.mean(axis=0, keepdims=True)

    gram = x_c.T @ x_c
    cross = x_c.T @ y_c
    yy = np.einsum("ij,ij->j", y_c, y_c)

    xx = np.diag(gram).copy()
    xy = np.diag(cross).copy()

    # compare the centered energy against the column's own uncentered energy so the test is scale-free
    uncentered = np.einsum("ij,ij->j", x_lag, x_lag)
    dead = xx <= collinear_tol * np.maximum(uncentered, np.finfo(np.float64).tiny)
    xx_safe = np.where(dead, 1.0, xx)

    ratio = xy / xx_safe
    rss_r = np.maximum(yy - xy * ratio, 0.0)
    num = cross - gram * ratio[None, :]
    den = np.maximum(xx[:, None] - (gram * gram) / xx_safe[None, :], 0.0)

    degenerate = (
        (den <= collinear_tol * xx[:, None])
        | dead[:, None]
        | dead[None, :]
        | (rss_r <= collinear_tol * yy)[None, :]
    )
    rho2 = np.divide(
        num * num,
        den * rss_r[None, :],
        out=np.zeros_like(num),
        where=~degenerate,
    )
    # log1p is more accurate than log(1 - rho2) for the small scores that sit near the threshold

    granger = -np.log1p(-np.clip(rho2, 0.0, _GRANGER_MAX_RHO2))
    np.maximum(granger, 0.0, out=granger)
    np.fill_diagonal(granger, 0.0)
    return granger


def threshold_granger_scores(
    score: torch.Tensor,
    threshold: float,
    threshold_mode: str,
    topk_edges: int,
    topk_per_node: int = 0,
) -> tuple[torch.Tensor, float]:
    """Binarize a directed ``score[src, dst]`` matrix into an adjacency."""
    n_nodes = score.shape[0]
    diag_mask = torch.eye(n_nodes, dtype=torch.bool, device=score.device)

    if threshold_mode == "manual":
        adjacency = score > float(threshold)
        threshold_used = float(threshold)
    elif threshold_mode == "topk_per_node":
        k = int(topk_per_node)
        if k <= 0:
            raise ValueError(
                "--granger_topk_per_node must be > 0 when "
                "--granger_threshold_mode=topk_per_node"
            )
        k = min(k, n_nodes - 1)
        masked = score.masked_fill(diag_mask, float("-inf"))
        idx = torch.topk(masked, k=k, dim=1, largest=True).indices  # (N, k)
        adjacency = torch.zeros_like(score, dtype=torch.bool)
        adjacency.scatter_(1, idx, True)
        # Report the weakest edge actually kept
        threshold_used = float(score.gather(1, idx).min().item())
    elif threshold_mode == "topk":
        adjacency = torch.zeros_like(score, dtype=torch.bool)
        cand_mask = ~diag_mask
        cand_scores = score[cand_mask]
        k = min(int(topk_edges), int(cand_scores.numel()))
        if k <= 0:
            raise ValueError(
                "--granger_topk_edges must be > 0 when --granger_threshold_mode=topk"
            )
        top_idx = torch.topk(cand_scores, k=k, largest=True).indices
        take = torch.zeros_like(cand_scores, dtype=torch.bool)
        take[top_idx] = True
        adjacency[cand_mask] = take
        threshold_used = float(cand_scores[top_idx].min().item())
    else:
        raise ValueError(f"Unknown granger threshold mode: {threshold_mode}")

    adjacency.fill_diagonal_(False)
    return adjacency, threshold_used


def corr_score_from_series(series: torch.Tensor, absolute: bool = True) -> torch.Tensor:
    """Symmetric Pearson functional-connectivity score matrix from ``(B, T, N)`` windows.

    Each window is mean-centered over time before the observations are pooled, so a
    slow drift in a window's mean cannot masquerade as co-fluctuation. ``absolute``
    scores edges by correlation magnitude (the usual FC edge strength); set it False
    to keep signed correlation, which drops anticorrelated pairs under a topk cut.
    """
    if series.ndim != 3:
        raise ValueError(f"series must have shape (B, T, N), got {tuple(series.shape)}")

    x = series.detach().cpu().numpy().astype(np.float64)
    n_windows, t_steps, n_nodes = x.shape
    if n_nodes < 2:
        raise ValueError(f"Need at least 2 nodes, got {n_nodes}")
    if t_steps < 2:
        raise RuntimeError(f"Not enough samples for correlation graph: T={t_steps}")

    # Center each window over time so cross-window mean offsets do not leak into the
    # pooled covariance, then pool every timepoint before one correlation estimate.
    x = x - x.mean(axis=1, keepdims=True)
    flat = x.reshape(-1, n_nodes)  # (B*T, N)
    corr = np.corrcoef(flat, rowvar=False)  # (N, N)
    # A node that is constant across every pooled sample yields NaN rows/cols.
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    score = np.abs(corr) if absolute else corr
    np.fill_diagonal(score, 0.0)
    return torch.from_numpy(score).float()


def build_corr_graph_from_series(
    series: torch.Tensor,
    threshold: float,
    threshold_mode: str,
    topk_edges: int,
    topk_per_node: int = 0,
    absolute: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    """Build a functional-connectivity graph from ``(B, T, N)`` windows.

    The score matrix is symmetric (Pearson correlation), so ``topk`` selection keeps
    both directions of each undirected edge and the returned graph is undirected.
    """
    if topk_edges < 0:
        raise ValueError(f"topk_edges must be >= 0, got {topk_edges}")

    score = corr_score_from_series(series, absolute=absolute)
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
    }
    return edge_index, score, info


def build_sc_graph(
    sc_matrix: torch.Tensor,
    topk_edges: int,
    volumes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    """Fixed structural-connectome (SC) graph."""
    if sc_matrix.ndim != 2 or sc_matrix.shape[0] != sc_matrix.shape[1]:
        raise ValueError(
            f"sc_matrix must be square (N, N), got {tuple(sc_matrix.shape)}"
        )
    sc = sc_matrix.float().clone()
    if volumes is not None:
        if volumes.ndim != 1 or volumes.shape[0] != sc.shape[0]:
            raise ValueError(
                f"volumes must be (N,) matching sc_matrix, got {tuple(volumes.shape)}"
            )
        vol = volumes.float().clamp(min=1e-8)
        sc = sc / torch.sqrt(vol[:, None] * vol[None, :])
    sc = 0.5 * (sc + sc.T)
    sc.fill_diagonal_(0.0)

    adjacency, threshold_used = threshold_granger_scores(
        score=sc, threshold=0.0, threshold_mode="topk", topk_edges=topk_edges
    )
    edge_index = adjacency_to_edge_index(adjacency)
    if edge_index.shape[1] == 0:
        raise RuntimeError("SC graph has no edges. Increase --granger_topk_edges.")

    info: dict[str, float | int | str] = {
        "mode": "sc",
        "volume_normalized": volumes is not None,
        "threshold": float(threshold_used),
        "topk_edges": int(topk_edges),
        "selected_edges": int(edge_index.shape[1]),
    }
    return edge_index, sc, info
