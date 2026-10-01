"""Restriction-map geometry analysis for the MLP sheaf (fMRI + EEG)."""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import torch

from model.braindyn import BrainDyn, BrainDynConfig
from model.sheaf import SheafLaplacian

__all__ = [
    "config_from_ckpt",
    "num_nodes_from_ckpt",
    "build_model_from_ckpt",
    "harvest_maps",
    "harvest_over_loader",
    "harvest_forecasts",
    "sheaf_step_decomposition",
    "sheaf_step_decomposition_over_loader",
    "sheaf_edge_contributions",
    "harvest_edge_contributions_over_loader",
    "residual_series",
    "residual_cross_channel",
    "residual_neighbor_probe",
    "frob",
    "edge_norm_stats",
    "svdvals",
    "effective_rank",
    "ranks",
    "orthogonality_defect",
    "svd_polar",
    "flatten_maps",
    "pairwise_distance",
    "set_effective_rank",
    "random_baseline",
    "mlp_pair_inputs",
    "mlp_jacobian_norms",
    "linear_cka",
    "neighbor_cosine",
    "sheaf_relative_change",
    "electrode_groups",
    "net_attention_operator",
    "delta_energy",
    "transport_ratio",
    "transport_ratio_null",
    "message_energy",
    "node_state_change",
    "edge_distance",
    "edge_transport_twist",
    "principal_axes",
    "bimodal_threshold",
    "orthogonal_content",
    "polar_orthogonal",
    "align_edge_maps",
    "cross_condition_gram",
    "map_field_smoothness",
    "edge_cluster_separation",
    "save_scalp_animation",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rebuild a model from a checkpoint (MLP-map mode)
# ─────────────────────────────────────────────────────────────────────────────
def config_from_ckpt(cfg: dict, num_nodes: int) -> BrainDynConfig:
    """Reconstruct a ``BrainDynConfig`` from a saved training-config dict."""
    g = cfg.get
    return BrainDynConfig(
        signal_dim=1,
        hidden_dim=g("hidden_dim"),
        num_nodes=num_nodes,
        window_size=g("x"),
        lstm_layers=g("lstm_layers", 1),
        lstm_dropout=g("lstm_dropout", 0.0),
        map_hidden_dim=g("map_hidden_dim", 16),
        vf_hidden_dim=g("vf_hidden_dim", 128),
        use_gcn=bool(
            g("ablation_gcn") or g("ablation_gat") or g("ablation_simple_graph")
        ),
        use_lstm_encoder=not g("ablation_no_lstm", False),
        edge_specific_maps=g("edge_specific_maps", False),
        sheaf_layers=g("sheaf_layers", 1),
        diffusion_step=g("diffusion_step", 1.0),
        sheaf_mlp_maps=g("sheaf_mlp_maps", not g("static_edge_maps", False)),
        map_mlp_hidden_dim=g("map_mlp_hidden_dim", 64),
        # Init-only flag: only affects the map *initialization*, which loaded weights
        # overwrite, so threading it is a no-op for learned-map harvesting. It matters
        # for reconstructing the true initial map field (harvest_initial_baseline).
        identity_restriction_init=g("identity_restriction_init", False),
        # Sheaf map-PE fields: default off so pre-PE checkpoints are unaffected, but
        # carried through when present (the PE run widens the restriction-MLP input to
        # 2*(hidden+pe) and adds a map_pe buffer).
        sheaf_map_pe=g("sheaf_map_pe", "none"),
        sheaf_map_pe_dim=g("sheaf_map_pe_dim", 8),
        # Time-encoding fields, so a checkpoint rebuilds the field it trained.
        time_embed_dim=g("time_embed_dim", 16),
        time_embed_max_period=g("time_embed_max_period", 16.0),
        time_rate_mode=g("time_rate_mode", "fixed"),
        time_learn_freqs=g("time_learn_freqs", False),
        time_max_cycles_per_step=g("time_max_cycles_per_step", 0.5),
        vf_layers=g("vf_layers", 2),
    )


def num_nodes_from_ckpt(ckpt: dict) -> int:
    """Recover node count from ``graph_score`` (N x N) else ``edge_index.max()+1``."""
    gs = ckpt.get("graph_score")
    if gs is not None:
        return int(gs.shape[0])
    ei = ckpt["edge_index"]
    if ei is not None and ei.numel():
        return int(ei.max()) + 1
    raise ValueError(
        "Cannot infer num_nodes: checkpoint has neither graph_score nor a non-empty "
        "edge_index. Pass num_nodes explicitly."
    )


_ANALYSIS_CRITICAL = (
    "dynamics.temporal_encoder",
    "dynamics.graph_laplacian",
    "dynamics.no_lstm_proj",
)
_ODE_HEAD = ("dynamics.hopf", "dynamics.vector_field")


def _load_analysis_subset(model: BrainDyn, sd: dict, ckpt_name: str) -> dict:
    """Load every checkpoint tensor whose name and shape match the model; skip the rest."""
    msd = model.state_dict()
    loadable, shape_mismatch = {}, []
    for k, v in sd.items():
        if k in msd:
            if tuple(msd[k].shape) == tuple(v.shape):
                loadable[k] = v
            else:
                shape_mismatch.append(k)
    model.load_state_dict(loadable, strict=False)

    loaded = set(loadable)
    mismatch_set = set(shape_mismatch)
    # "missing" = model params left at init for any reason OTHER than a shape clash (those are
    # reported via shape_mismatch) so a key is never counted in both buckets.
    missing = [k for k in msd if k not in loaded and k not in mismatch_set]
    extra = [k for k in sd if k not in msd]  # checkpoint tensors with no home

    def _hits(keys, prefixes):
        return [k for k in keys if k.startswith(prefixes)]

    critical = _hits(missing, _ANALYSIS_CRITICAL) + _hits(
        shape_mismatch, _ANALYSIS_CRITICAL
    )
    if critical:
        hint = ""
        if _hits(shape_mismatch, ("dynamics.temporal_encoder",)):
            hint = (
                " The encoder input width differs — most likely a node-PE-concat encoder "
                "(LSTM input = signal_dim + node PE) the current model has no path for."
            )
        warnings.warn(
            f"{ckpt_name}: ANALYSIS-CRITICAL weights did not load "
            f"({len(critical)} tensor(s), e.g. {critical[:3]}). Harvested embeddings/maps would "
            f"come from UNTRAINED weights and are NOT valid.{hint} Use a checkpoint matching the "
            "current encoder/sheaf, or analyze this one with the code it was trained on.",
            stacklevel=3,
        )
    elif missing or shape_mismatch or extra:
        head_missing = bool(
            _hits(missing, _ODE_HEAD) or _hits(shape_mismatch, _ODE_HEAD)
        )
        note = (
            " The ODE dynamics head was left at init, so map/geometry harvesting is valid "
            "but forecasting (curves-of-fit, residuals) is NOT for this checkpoint."
            if head_missing
            else ""
        )
        warnings.warn(
            f"{ckpt_name}: loaded {len(loadable)} tensor(s); left {len(missing)} model tensor(s) "
            f"at init and dropped {len(extra)} checkpoint tensor(s) from modules the analysis "
            f"does not use.{note}",
            stacklevel=3,
        )
    return {
        "loaded": len(loadable),
        "missing": missing,
        "extra": extra,
        "shape_mismatch": shape_mismatch,
    }


def build_model_from_ckpt(
    ckpt: dict,
    edge_index: torch.Tensor,
    device: torch.device | str = "cpu",
    num_nodes: Optional[int] = None,
    ckpt_name: str = "<checkpoint>",
    require_mlp: bool = True,
) -> BrainDyn:
    """Rebuild a trained MLP-map model ready for map harvesting."""
    if num_nodes is None:
        num_nodes = num_nodes_from_ckpt(ckpt)
    conf = config_from_ckpt(ckpt["config"], num_nodes)

    sd = ckpt["model_state_dict"]

    has_static_table = any(k.endswith("graph_laplacian.restriction_maps") for k in sd)
    has_mlp = any("graph_laplacian.restriction_mlp." in k for k in sd)
    if require_mlp and (not conf.sheaf_mlp_maps or has_static_table or not has_mlp):
        raise ValueError(
            f"{ckpt_name} is not an MLP-map checkpoint "
            f"(sheaf_mlp_maps={conf.sheaf_mlp_maps}, static_table={has_static_table}, "
            f"mlp_present={has_mlp}). This analysis targets sheaf_mlp_maps=True models; "
            "point it at an MLP-trained run (one trained without --static_edge_maps)."
        )
    sheaf_node_pe = None
    if getattr(conf, "sheaf_map_pe", "none") == "lappe":
        from model.graph_builders import build_lap_pe

        # The saved buffer's width is authoritative for the LapPE rank (matches the
        # trained restriction-MLP input exactly); fall back to the config field.
        map_pe_w = sd.get("dynamics.graph_laplacian.map_pe")
        rank = int(map_pe_w.shape[1]) if map_pe_w is not None else conf.sheaf_map_pe_dim
        sheaf_node_pe = build_lap_pe(
            edge_index.detach().cpu(), num_nodes=num_nodes, rank=rank
        )

    model = BrainDyn(conf, sheaf_node_pe=sheaf_node_pe)
    model.register_restriction_edges(edge_index)
    # Load the intersection of the checkpoint and the current model (by name AND shape).
    _load_analysis_subset(model, sd, ckpt_name)
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Harvest the data-dependent map population
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def harvest_maps(
    model: BrainDyn,
    x_history: torch.Tensor,
    edge_index: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Run one forward pass and return the instantiated restriction maps."""
    sheaf_h, aux = model.dynamics.compute_sheaf_h(x_history, edge_index)
    src = edge_index[0]
    dst = edge_index[1]
    h_t = aux["h_t"]
    B = h_t.shape[0]
    # Sheaf map-PE (cell B: --sheaf_map_pe): part of the restriction-MLP input, so it
    # must be harvested for any analysis that re-runs that MLP.
    gl = getattr(model.dynamics, "graph_laplacian", None)
    pe_pair = getattr(gl, "_edge_pe", None)
    if pe_pair is not None and getattr(gl, "sheaf_map_pe", "none") != "none":
        pe_s, pe_d = gl._edge_pe(src, dst)  # (E, pe_dim) each
        pe_src = pe_s.unsqueeze(0).expand(B, -1, -1).detach().cpu().numpy()
        pe_dst = pe_d.unsqueeze(0).expand(B, -1, -1).detach().cpu().numpy()
    else:
        pe_src = np.empty((B, int(src.numel()), 0), dtype=np.float32)
        pe_dst = np.empty((B, int(dst.numel()), 0), dtype=np.float32)
    out = {
        "rho_src": aux["rho_src"].detach().cpu().numpy(),
        "rho_dst": aux["rho_dst"].detach().cpu().numpy(),
        "h_t": h_t.detach().cpu().numpy(),
        "sheaf_h": sheaf_h.detach().cpu().numpy(),
        "h_src": h_t[:, src, :].detach().cpu().numpy(),
        "h_dst": h_t[:, dst, :].detach().cpu().numpy(),
        "pe_src": pe_src,
        "pe_dst": pe_dst,
        "delta": aux["delta"].detach().cpu().numpy(),
        "src": src.detach().cpu().numpy(),
        "dst": dst.detach().cpu().numpy(),
    }
    return out


def harvest_over_loader(
    model: BrainDyn,
    loader: Any,
    edge_index: torch.Tensor,
    norm_stats: Optional[dict] = None,
    max_batches: Optional[int] = None,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Pool a larger map population across probe batches."""
    from main import (
        batch_to_model_tensors,
    )  # local import: avoids CLI import at module load

    edge_index = edge_index.to(device)
    keys = (
        "rho_src",
        "rho_dst",
        "h_t",
        "sheaf_h",
        "h_src",
        "h_dst",
        "pe_src",
        "pe_dst",
        "delta",
    )
    acc: dict[str, list] = {k: [] for k in keys}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x_history, _ = batch_to_model_tensors(batch, device, norm_stats=norm_stats)
        h = harvest_maps(model, x_history, edge_index)
        for k in keys:
            acc[k].append(h[k])
        src, dst = h["src"], h["dst"]
    out = {k: np.concatenate(acc[k], axis=0) for k in keys}
    out["src"], out["dst"] = src, dst
    return out


def harvest_initial_baseline(
    model: BrainDyn,
    loader: Any,
    edge_index: torch.Tensor,
    norm_stats: Optional[dict] = None,
    max_batches: Optional[int] = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Harvest the *true initial* restriction maps on the trained model's embeddings.

    The honest "initial" condition for an MLP-map sheaf is **not** a Xavier-random
    matrix population: the maps are ``rho = MLP([h_src || h_dst || pe])``, so at
    initialization they are whatever the freshly-initialized sheaf-learner produces
    on the current embeddings -- either the identity (``identity_restriction_init``)
    or the small ``gain=0.1`` readout. This reconstructs exactly that by deep-copying
    the trained model, re-initializing *only* the restriction-map parameters
    (``SheafLaplacian.reset_restriction_parameters``), and re-harvesting over the same
    probe batches. The LSTM/temporal encoder is left trained, so the operating points
    (embeddings) match the learned harvest one-for-one and only the map-generating
    function is rolled back to init.
    """
    import copy

    init_model = copy.deepcopy(model)
    torch.manual_seed(seed)
    init_model.dynamics.graph_laplacian.reset_restriction_parameters()
    init_model.to(device).eval()
    return harvest_over_loader(
        init_model, loader, edge_index, norm_stats=norm_stats,
        max_batches=max_batches, device=device,
    )


@torch.no_grad()
def sheaf_step_decomposition(
    model: BrainDyn,
    x_history: torch.Tensor,
    edge_index: torch.Tensor,
    round_idx: int = 0,
) -> dict[str, torch.Tensor]:
    """Split one diffusion round's ``L_F h`` into its diagonal (self) and off-diagonal
    (neighbor) blocks — exactly as ``SheafLaplacian.forward`` builds it."""
    dyn = model.dynamics
    gl = getattr(dyn, "graph_laplacian", None)
    if not isinstance(gl, SheafLaplacian):
        raise TypeError(
            "sheaf_step_decomposition requires a SheafLaplacian graph op (the diagonal/"
            f"off-diagonal split is sheaf-specific); got {type(gl).__name__}. This does not "
            "apply to the GCN ablation."
        )
    step = float(dyn.diffusion_step)
    n_layers = int(dyn.sheaf_layers)
    if not (0 <= round_idx < max(n_layers, 1)):
        raise ValueError(
            f"round_idx={round_idx} out of range for sheaf_layers={n_layers} "
            "(valid rounds are 0..sheaf_layers-1)."
        )

    # Encode exactly as compute_sheaf_h (model/dynamics.py) does, then run the real
    # diffusion recurrence so the target round sees the true updated stalks.
    if dyn.use_lstm_encoder:
        h = dyn.temporal_encoder(x_history)  # (B, N, H)
    else:
        h = torch.tanh(dyn.no_lstm_proj(x_history[:, :, -1, :]))

    src = edge_index[0]
    dst = edge_index[1]
    diag = off = lap = None
    for r in range(round_idx + 1):
        lap, aux = gl.forward(h, edge_index)
        if r == round_idx:
            rho_src, rho_dst = aux["rho_src"], aux["rho_dst"]
            tilde_src, tilde_dst = aux["tilde_src"], aux["tilde_dst"]
            # Diagonal (self) block: pull each endpoint's OWN restricted stalk back through
            # its OWN map — i.e. Σ_e ρ_{i→e}ᵀ ρ_{i→e} h_i, no neighbor term.
            diag = torch.zeros_like(h)
            diag.index_add_(1, src, gl._pullback(tilde_src, rho_src))
            diag.index_add_(1, dst, gl._pullback(tilde_dst, rho_dst))
            # Off-diagonal (neighbor) block: pull the NEIGHBOR's restricted stalk back
            # through this node's map — i.e. −Σ_e ρ_{i→e}ᵀ ρ_{j→e} h_j. This is the only
            # message-passing channel. diag + off == lap by construction.
            off = torch.zeros_like(h)
            off.index_add_(1, src, gl._pullback(-tilde_dst, rho_src))
            off.index_add_(1, dst, gl._pullback(-tilde_src, rho_dst))
            # Degree normalization (sheaf_norm != "none") scales the aggregated
            # Laplacian on the way out. The input scaling is already baked into
            # tilde_src/tilde_dst, but the OUTPUT scaling is applied after the
            # index_add in SheafLaplacian.forward, so it must be mirrored here
            # or the documented `diag + off == lap` invariant breaks.
            if getattr(gl, "sheaf_norm", "none") != "none":
                nscale = gl._node_scale(edge_index, h.device, h.dtype).view(1, -1, 1)
                diag = diag * nscale
                off = off * nscale
            break
        h = h - step * lap

    update = -step * lap
    lap_norm = lap.norm(dim=-1)
    diag_norm = diag.norm(dim=-1)
    off_norm = off.norm(dim=-1)
    return {
        "lap": lap,
        "update": update,
        "diag": diag,
        "offdiag": off,
        "h": h,  # the stalks the round acts on (encoder output at round 0)
        "lap_norm": lap_norm,
        "update_norm": update.norm(dim=-1),
        "diag_norm": diag_norm,
        "off_norm": off_norm,
        "h_norm": h.norm(dim=-1),  # per-node ‖h‖ — denominator for the relative update
        "ratio": off_norm / (diag_norm + 1e-12),
        "step": step,
        "round_idx": int(round_idx),
        "sheaf_layers": n_layers,
    }


@torch.no_grad()
def sheaf_step_decomposition_over_loader(
    model: BrainDyn,
    loader: Any,
    edge_index: torch.Tensor,
    norm_stats: Optional[dict] = None,
    max_batches: Optional[int] = None,
    device: torch.device | str = "cpu",
    round_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Pool :func:`sheaf_step_decomposition` over probe batches → per-(window, node) arrays."""
    from main import (
        batch_to_model_tensors,
    )  # local import: avoids CLI import at module load

    edge_index = edge_index.to(device)
    keys = ("lap_norm", "update_norm", "diag_norm", "off_norm", "h_norm", "ratio")
    acc: dict[str, list] = {k: [] for k in keys}
    step = 1.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x_history, _ = batch_to_model_tensors(batch, device, norm_stats=norm_stats)
        d = sheaf_step_decomposition(model, x_history, edge_index, round_idx=round_idx)
        step = d["step"]
        for k in keys:
            acc[k].append(d[k].detach().cpu().numpy())
    if not acc["ratio"]:
        raise RuntimeError(
            "sheaf_step_decomposition_over_loader: loader yielded no batches."
        )
    out = {k: np.concatenate(acc[k], axis=0) for k in keys}  # (P, N)
    out["step"] = step
    return out


@torch.no_grad()
def sheaf_edge_contributions(
    model: BrainDyn,
    x_history: torch.Tensor,
    edge_index: torch.Tensor,
    round_idx: int = 0,
) -> dict[str, torch.Tensor]:
    """Per-edge messages that build the one-step update, split by endpoint."""
    dyn = model.dynamics
    gl = getattr(dyn, "graph_laplacian", None)
    if not isinstance(gl, SheafLaplacian):
        raise TypeError(
            "sheaf_edge_contributions requires a SheafLaplacian graph op; got "
            f"{type(gl).__name__}."
        )
    step = float(dyn.diffusion_step)
    n_layers = int(dyn.sheaf_layers)
    if not (0 <= round_idx < max(n_layers, 1)):
        raise ValueError(
            f"round_idx={round_idx} out of range for sheaf_layers={n_layers}."
        )
    if dyn.use_lstm_encoder:
        h = dyn.temporal_encoder(x_history)
    else:
        h = torch.tanh(dyn.no_lstm_proj(x_history[:, :, -1, :]))

    for r in range(round_idx + 1):
        lap, aux = gl.forward(h, edge_index)
        if r == round_idx:
            rho_src, rho_dst = aux["rho_src"], aux["rho_dst"]
            delta = aux["delta"]
            msg_to_src = gl._pullback(-delta, rho_src)  # (B, E, H)
            msg_to_dst = gl._pullback(delta, rho_dst)  # (B, E, H)
            break
        h = h - step * lap

    def _frob(rho: torch.Tensor) -> torch.Tensor:
        # rho is (B, E, H, d_e) in MLP mode, or (E, H, d_e) shared across the batch.
        if rho.ndim == 4:
            return rho.reshape(rho.shape[0], rho.shape[1], -1).norm(dim=-1)  # (B, E)
        f = rho.reshape(rho.shape[0], -1).norm(dim=-1)  # (E,)
        return f.unsqueeze(0).expand(msg_to_src.shape[0], -1)

    return {
        "c_to_src": step * msg_to_src.norm(dim=-1),
        "c_to_dst": step * msg_to_dst.norm(dim=-1),
        "rho_src_frob": _frob(rho_src),
        "rho_dst_frob": _frob(rho_dst),
        "delta_norm": delta.norm(dim=-1),
        "step": step,
        "round_idx": int(round_idx),
    }


@torch.no_grad()
def harvest_edge_contributions_over_loader(
    model: BrainDyn,
    loader: Any,
    edge_index: torch.Tensor,
    norm_stats: Optional[dict] = None,
    max_batches: Optional[int] = None,
    device: torch.device | str = "cpu",
    round_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Pool :func:`sheaf_edge_contributions` over probe windows → per-(window, edge) arrays."""
    from main import batch_to_model_tensors

    edge_index = edge_index.to(device)
    keys = ("c_to_src", "c_to_dst", "rho_src_frob", "rho_dst_frob", "delta_norm")
    acc: dict[str, list] = {k: [] for k in keys}
    step = 1.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x_history, _ = batch_to_model_tensors(batch, device, norm_stats=norm_stats)
        d = sheaf_edge_contributions(model, x_history, edge_index, round_idx=round_idx)
        step = d["step"]
        for k in keys:
            acc[k].append(d[k].detach().cpu().numpy())
    if not acc["c_to_src"]:
        raise RuntimeError(
            "harvest_edge_contributions_over_loader: loader yielded no batches."
        )
    out = {k: np.concatenate(acc[k], axis=0) for k in keys}  # (S, E)
    out["src"] = edge_index[0].detach().cpu().numpy()
    out["dst"] = edge_index[1].detach().cpu().numpy()
    out["step"] = step
    return out


def diffusion_directedness(
    src: np.ndarray,
    dst: np.ndarray,
    c_to_src: np.ndarray,
    c_to_dst: np.ndarray,
    num_nodes: Optional[int] = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Does the sheaf diffusion's action recover the *directed* Granger arrow?"""
    src = np.asarray(src).astype(int)
    dst = np.asarray(dst).astype(int)
    to_src = np.asarray(c_to_src, dtype=float)
    to_dst = np.asarray(c_to_dst, dtype=float)
    if to_src.ndim == 2:
        to_src = to_src.mean(axis=0)
    if to_dst.ndim == 2:
        to_dst = to_dst.mean(axis=0)
    E = int(src.shape[0])
    if num_nodes is None:
        num_nodes = int(max(src.max(), dst.max())) + 1 if E else 0

    # per-edge imbalance toward the dst (Granger-effect) endpoint
    edge_imbalance = (to_dst - to_src) / (to_dst + to_src + eps)

    # reciprocity: is the reverse directed edge also present?
    edge_set = set(zip(src.tolist(), dst.tolist()))
    reciprocal = np.array(
        [(int(d), int(s)) in edge_set for s, d in zip(src, dst)], dtype=bool
    )

    # per unordered pair: accumulate the update delivered to each of its two endpoints.
    # For edge (s, d): node d receives to_dst, node s receives to_src.
    pair_flow: dict[tuple, list] = {}
    for e in range(E):
        s, d = int(src[e]), int(dst[e])
        ts, td = float(to_src[e]), float(to_dst[e])
        a, b = (s, d) if s < d else (d, s)
        rec = pair_flow.setdefault(
            (a, b), [0.0, 0.0, False, False]
        )  # flow_a, flow_b, a->b, b->a
        if s == a:  # directed edge a -> b
            rec[0] += ts
            rec[1] += td
            rec[2] = True
        else:  # directed edge b -> a
            rec[1] += ts
            rec[0] += td
            rec[3] = True

    if pair_flow:
        pairs = np.array(list(pair_flow.keys()), dtype=int).reshape(-1, 2)
        fl = np.array([pair_flow[tuple(p)][:2] for p in pairs], dtype=float)  # (P, 2)
        has = np.array([pair_flow[tuple(p)][2:] for p in pairs], dtype=bool)  # (P, 2)
    else:
        pairs = np.zeros((0, 2), dtype=int)
        fl = np.zeros((0, 2), dtype=float)
        has = np.zeros((0, 2), dtype=bool)
    bidirectional = has.all(axis=1)
    uni = ~bidirectional
    pair_asym = np.abs(fl[:, 0] - fl[:, 1]) / (fl.sum(axis=1) + eps)

    # directional agreement, uni pairs only: does the effect (dst) endpoint receive more?
    # for a uni pair the arrow is a->b if has[:,0] else b->a; the effect node is its dst.
    eff_flow = np.where(has[:, 0], fl[:, 1], fl[:, 0])  # flow into the effect node
    cause_flow = np.where(has[:, 0], fl[:, 0], fl[:, 1])  # flow into the cause node
    agree = eff_flow > cause_flow
    agreement_rate = float(agree[uni].mean()) if uni.any() else float("nan")

    # per-node flux. edge (s,d): d is driven by s (to_dst), s is driven by d (to_src).
    received = np.zeros(num_nodes)
    drive_out = np.zeros(num_nodes)
    if E:
        np.add.at(received, dst, to_dst)
        np.add.at(received, src, to_src)
        np.add.at(drive_out, src, to_dst)  # s drives d
        np.add.at(drive_out, dst, to_src)  # d drives s
    net_source = drive_out - received

    summary = dict(
        n_edges=E,
        n_pairs=int(pairs.shape[0]),
        n_uni=int(uni.sum()),
        n_bi=int(bidirectional.sum()),
        frac_reciprocal_edges=(float(reciprocal.mean()) if E else float("nan")),
        agreement_rate_uni=agreement_rate,
        median_asym_uni=(
            float(np.median(pair_asym[uni])) if uni.any() else float("nan")
        ),
        median_asym_bi=(
            float(np.median(pair_asym[bidirectional]))
            if bidirectional.any()
            else float("nan")
        ),
        mean_edge_imbalance=(float(edge_imbalance.mean()) if E else float("nan")),
    )
    return dict(
        edge_imbalance=edge_imbalance,
        reciprocal=reciprocal,
        to_src=to_src,
        to_dst=to_dst,
        pairs=pairs,
        pair_asym=pair_asym,
        bidirectional=bidirectional,
        received=received,
        drive_out=drive_out,
        net_source=net_source,
        summary=summary,
    )


@torch.no_grad()
def harvest_forecasts(
    model: BrainDyn,
    loader: Any,
    edge_index: torch.Tensor,
    norm_stats: Optional[dict] = None,
    dt: float = 1.0,
    max_batches: Optional[int] = None,
    device: torch.device | str = "cpu",
    ctx_show: int = 15,
) -> dict[str, np.ndarray]:
    """Short-horizon test forecasts → per-(window, node) traces for curves-of-fit."""
    from main import batch_to_model_tensors

    edge_index = edge_index.to(device)
    ctx_l, tru_l, prd_l = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x_history, y_true = batch_to_model_tensors(batch, device, norm_stats=norm_stats)
        pred_steps = int(y_true.shape[0])  # (Ly, B, N, 1)
        y_pred = model(
            x_history=x_history,
            edge_index=edge_index,
            pred_steps=pred_steps,
            dt=dt,
            autoregressive=False,
        )["x_pred"]  # (Ly, B, N, 1)
        ctx_l.append(
            x_history[:, :, -ctx_show:, 0].detach().cpu().numpy()
        )  # (B, N, ctx_show)
        tru_l.append(
            y_true[..., 0].permute(1, 2, 0).detach().cpu().numpy()
        )  # (B, N, Ly)
        prd_l.append(
            y_pred[..., 0].permute(1, 2, 0).detach().cpu().numpy()
        )  # (B, N, Ly)
    if not ctx_l:
        raise RuntimeError("harvest_forecasts: loader yielded no batches.")
    ctx = np.concatenate(ctx_l, 0)
    tru = np.concatenate(tru_l, 0)
    prd = np.concatenate(prd_l, 0)
    return {"ctx": ctx, "true": tru, "pred": prd, "mse": ((prd - tru) ** 2).mean(-1)}


def residual_series(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Stack forecast residuals into a per-channel sample matrix."""
    r = np.asarray(pred, dtype=np.float64) - np.asarray(
        true, dtype=np.float64
    )  # (S, N, Ly)
    if r.ndim != 3:
        raise ValueError(f"expected (S, N, Ly) residuals, got {r.shape}")
    S, N, Ly = r.shape
    return r.transpose(0, 2, 1).reshape(S * Ly, N)  # (S*Ly, N)


def residual_cross_channel(
    resid: np.ndarray, edge_index: Optional[np.ndarray] = None
) -> dict:
    """Same-time cross-channel correlation of residuals."""
    X = np.asarray(resid, dtype=np.float64)
    C = np.nan_to_num(np.corrcoef(X, rowvar=False))  # (N, N); zero-variance chans -> 0
    N = C.shape[0]
    off = ~np.eye(N, dtype=bool)
    out = {"corr": C, "offdiag_abs_mean": float(np.abs(C[off]).mean())}
    if edge_index is not None:
        ei = np.asarray(edge_index)
        if ei.shape[1] > 0:
            adj = np.zeros((N, N), dtype=bool)
            adj[ei[0], ei[1]] = True
            adj[ei[1], ei[0]] = True
            np.fill_diagonal(adj, False)
            nb, nn = adj, (~adj) & off
            out["neighbor_abs_corr"] = (
                float(np.abs(C[nb]).mean()) if nb.any() else float("nan")
            )
            out["nonneighbor_abs_corr"] = (
                float(np.abs(C[nn]).mean()) if nn.any() else float("nan")
            )
            out["n_edges"] = int(nb.sum() // 2)
    return out


def residual_neighbor_probe(
    resid: np.ndarray,
    edge_index: Optional[np.ndarray] = None,
    test_frac: float = 0.3,
    ridge: float = 1.0,
    seed: int = 0,
) -> dict:
    """Linear probe: predict each channel's residual from the *other* channels' residuals."""
    X = np.asarray(resid, dtype=np.float64)
    n, N = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(n * (1.0 - test_frac))
    tr, te = perm[:cut], perm[cut:]
    null_map = rng.permutation(n)  # row permutation that decouples target from features

    if edge_index is not None and np.asarray(edge_index).shape[1] > 0:
        ei = np.asarray(edge_index)
        nbrs: list[list[int]] = [[] for _ in range(N)]
        seen = [set() for _ in range(N)]
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            if s != d:
                if d not in seen[s]:
                    nbrs[s].append(int(d))
                    seen[s].add(d)
                if s not in seen[d]:
                    nbrs[d].append(int(s))
                    seen[d].add(s)
    else:
        nbrs = [[j for j in range(N) if j != i] for i in range(N)]

    def _r2(feat_idx, target):
        if not feat_idx:
            return np.nan
        Xtr, Xte = X[np.ix_(tr, feat_idx)], X[np.ix_(te, feat_idx)]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ytr, yte = target[tr], target[te]
        my = ytr.mean()
        p = Xtr.shape[1]
        w = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(p), Xtr.T @ (ytr - my))
        pred = Xte @ w + my
        ss_res = float(((yte - pred) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum())
        return 1.0 - ss_res / (ss_tot + 1e-12)

    r2 = np.full(N, np.nan)
    r2_null = np.full(N, np.nan)
    for i in range(N):
        r2[i] = _r2(nbrs[i], X[:, i])
        r2_null[i] = _r2(nbrs[i], X[null_map, i])
    return {
        "r2": r2,
        "r2_mean": float(np.nanmean(r2)),
        "r2_null_mean": float(np.nanmean(r2_null)),
        "n_features_mean": float(np.mean([len(s) for s in nbrs])),
    }


def _as_stack(R: np.ndarray) -> np.ndarray:
    """Coerce ``(..., m, n)`` to ``(P, m, n)`` by flattening all leading axes."""
    R = np.asarray(R)
    if R.ndim < 2:
        raise ValueError(f"expected map stack with >=2 dims, got shape {R.shape}")
    m, n = R.shape[-2], R.shape[-1]
    return R.reshape(-1, m, n)


def frob(R: np.ndarray, normalize: Optional[str] = None) -> np.ndarray:
    """Per-map Frobenius norm. Returns ``(P,)``."""
    Rs = _as_stack(R)
    f = np.linalg.norm(Rs, ord="fro", axis=(1, 2))
    if normalize is None:
        return f
    m, n = Rs.shape[-2], Rs.shape[-1]
    if normalize == "isometry":
        return f / np.sqrt(min(m, n))
    if normalize == "rms":
        return f / np.sqrt(m * n)
    raise ValueError(
        f"unknown normalize={normalize!r} (use None, 'isometry', or 'rms')"
    )


def edge_norm_stats(
    R: np.ndarray, normalize: Optional[str] = "isometry"
) -> dict[str, Any]:
    """Separate the two sources of norm variance for a ``(S, E, m, n)`` population."""
    R = np.asarray(R)
    if R.ndim < 3:
        raise ValueError(f"expected (S, E, m, n) population, got shape {R.shape}")
    S, E = R.shape[0], R.shape[1]
    fro = frob(R, normalize=normalize).reshape(S, E)
    edge_mean = fro.mean(0)
    between_var = float(edge_mean.var())
    within_var = float(fro.var(0).mean())
    return {
        "per_map": fro.reshape(-1),
        "edge_mean": edge_mean,
        "edge_std": fro.std(0),
        "between_var": between_var,
        "within_var": within_var,
        "total_var": float(fro.reshape(-1).var()),
        "between_frac": between_var / (between_var + within_var + 1e-12),
        "normalize": normalize,
    }


def svdvals(R: np.ndarray) -> np.ndarray:
    """Per-map singular values, ``(P, min(m, n))`` (descending)."""
    return np.linalg.svd(_as_stack(R), compute_uv=False)


def effective_rank(R: np.ndarray, kind: str = "participation") -> np.ndarray:
    """Per-map effective rank from the singular spectrum. Returns ``(P,)``."""
    s = svdvals(R)
    if kind == "participation":
        num = s.sum(-1) ** 2
        den = (s**2).sum(-1) + 1e-12
        return num / den
    if kind == "entropy":
        p = s**2
        p = p / (p.sum(-1, keepdims=True) + 1e-12)
        ent = -(p * np.log(p + 1e-12)).sum(-1)
        return np.exp(ent)
    raise ValueError(f"unknown kind={kind!r}")


def ranks(R: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Per-map numerical rank at absolute tolerance ``tol``. Returns ``(P,)``."""
    return np.array([np.linalg.matrix_rank(M, tol=tol) for M in _as_stack(R)])


def orthogonality_defect(R: np.ndarray) -> np.ndarray:
    """Per-map ``||R^T R - I_n||_F`` (column orthonormality defect). Returns ``(P,)``."""
    Rs = _as_stack(R)
    n = Rs.shape[-1]
    G = np.einsum("pmi,pmj->pij", Rs, Rs)  # R^T R : (P, n, n)
    return np.linalg.norm(G - np.eye(n)[None], ord="fro", axis=(1, 2))


def svd_polar(R: np.ndarray) -> dict[str, np.ndarray]:
    """SVD-based polar decomposition, splitting each map into rotation vs scaling."""
    Rs = _as_stack(R)
    P, m, n = Rs.shape
    U, S, Vt = np.linalg.svd(Rs, full_matrices=False)  # U:(P,m,r) S:(P,r) Vt:(P,r,n)
    Q = U @ Vt  # (P, m, n)  nearest partial isometry
    # SPD stretch P = V S V^T = Vt^T diag(S) Vt
    Pspd = np.einsum("pri,pr,prj->pij", Vt, S, Vt)  # (P, n, n)
    eye_mn = np.eye(m, n)[None]
    eye_n = np.eye(n)[None]

    rotation_dev = np.linalg.norm(Q - eye_mn, ord="fro", axis=(1, 2))
    stretch_dev = np.linalg.norm(Pspd - eye_n, ord="fro", axis=(1, 2))
    return {
        "sing": S,
        "scale_dev": np.linalg.norm(S - 1.0, axis=-1),
        "rotation_dev": rotation_dev,
        "stretch_dev": stretch_dev,
        "iso_dev": np.linalg.norm(Rs - Q, ord="fro", axis=(1, 2)),
        "total_dev": np.linalg.norm(Rs - eye_mn, ord="fro", axis=(1, 2)),
        "rotation_frac": rotation_dev / (rotation_dev + stretch_dev + 1e-12),
    }


def flatten_maps(R: np.ndarray) -> np.ndarray:
    """``(P, m, n)`` -> ``(P, m*n)`` (vectorized maps for distance / PCA / CKA)."""
    Rs = _as_stack(R)
    return Rs.reshape(Rs.shape[0], -1)


def pairwise_distance(Xf: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Full ``(P, P)`` pairwise-distance matrix over vectorized maps/embeddings."""
    from scipy.spatial.distance import cdist

    return cdist(Xf, Xf, metric)


def set_effective_rank(Xf: np.ndarray, center: bool = True) -> float:
    """Participation-ratio effective dimensionality of a *set* of vectors."""
    X = np.asarray(Xf, dtype=np.float64)
    if center:
        X = X - X.mean(0, keepdims=True)
    # covariance eigenvalues via singular values of X (lambda_i = s_i^2 / (P-1))
    s = np.linalg.svd(X, compute_uv=False)
    lam = s**2
    return float(lam.sum() ** 2 / ((lam**2).sum() + 1e-12))


def random_baseline(
    shape: tuple[int, int],
    n: int,
    kind: str = "xavier",
    seed: int = 0,
) -> np.ndarray:
    """Reference map population for the discriminability comparison. ``(n, m, n_cols)``."""
    m, ncol = shape
    rng = np.random.default_rng(seed)
    if kind == "xavier":
        limit = np.sqrt(6.0 / (m + ncol))
        return rng.uniform(-limit, limit, size=(n, m, ncol))
    if kind == "orthogonal":
        out = np.empty((n, m, ncol))
        for i in range(n):
            A = rng.standard_normal((m, ncol))
            Q, _ = np.linalg.qr(A)
            out[i] = Q[:, :ncol]
        return out
    raise ValueError(f"unknown kind={kind!r}")


def mlp_pair_inputs(H: dict, shuffle: Optional[np.ndarray] = None) -> np.ndarray:
    """Rebuild the restriction-MLP input ``[h_src ‖ h_dst (‖ pe_src ‖ pe_dst)]``. ``(P, in_dim)``."""
    hs = np.asarray(H["h_src"]).reshape(-1, H["h_src"].shape[-1])
    hd = np.asarray(H["h_dst"]).reshape(-1, H["h_dst"].shape[-1])
    if shuffle is not None:
        hd = hd[shuffle]
    parts = [hs, hd]
    pe_s, pe_d = H.get("pe_src"), H.get("pe_dst")
    if pe_s is not None and np.asarray(pe_s).shape[-1] > 0:
        ps = np.asarray(pe_s).reshape(-1, pe_s.shape[-1])
        pd = np.asarray(pe_d).reshape(-1, pe_d.shape[-1])
        parts += [ps, pd[shuffle] if shuffle is not None else pd]
    return np.concatenate(parts, axis=-1).astype(np.float32)


@torch.no_grad()
def _restriction_mlp(model: BrainDyn) -> torch.nn.Module:
    gl = model.dynamics.graph_laplacian
    if not hasattr(gl, "restriction_mlp"):
        raise AttributeError(
            "graph_laplacian has no restriction_mlp — model is not in MLP-map mode."
        )
    return gl.restriction_mlp


def mlp_jacobian_norms(
    model: BrainDyn,
    h_pairs: torch.Tensor,
    batch: int = 256,
) -> np.ndarray:
    """Frobenius norm of the sheaf-learner Jacobian at each operating point."""
    mlp = _restriction_mlp(model)
    device = next(mlp.parameters()).device
    h_pairs = h_pairs.to(device)

    def _single(x):  # (2H,) -> (H*d_e,)
        return mlp(x)

    norms: list[np.ndarray] = []
    try:
        # Per-sample Jacobian via vmap(jacrev): yields (b, out, in) with no cross-sample
        # terms, so memory stays O(b * out * in) instead of the O(b^2 * ...) blow-up of
        # a batched autograd.functional.jacobian.
        from torch.func import jacrev, vmap

        jfn = vmap(jacrev(_single))
        for start in range(0, h_pairs.shape[0], batch):
            xb = h_pairs[start : start + batch]
            J = jfn(xb)  # (b, out, in)
            norms.append(J.reshape(xb.shape[0], -1).norm(dim=-1).detach().cpu().numpy())
    except Exception:
        # Fallback for older torch: one small Jacobian per sample.
        for x in h_pairs:
            J = torch.autograd.functional.jacobian(_single, x)  # (out, in)
            norms.append(np.array([float(J.reshape(-1).norm())]))
    return np.concatenate(norms, axis=0)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA similarity between two sets of representations."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # HSIC via Gram matrices: <X X^T, Y Y^T>_F = ||X^T Y||_F^2 (linear kernel).
    xty = X.T @ Y
    hsic_xy = np.sum(xty**2)
    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def neighbor_cosine(h_src: np.ndarray, h_dst: np.ndarray) -> np.ndarray:
    """Cosine similarity between the two endpoints' LSTM embeddings, per (window, edge)."""
    a = np.asarray(h_src, dtype=np.float64)
    b = np.asarray(h_dst, dtype=np.float64)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
    return num / den


def sheaf_relative_change(sheaf_h: np.ndarray, h_t: np.ndarray) -> np.ndarray:
    """Relative magnitude of the sheaf diffusion per window: ``‖sheaf_h − h_t‖ / ‖h_t‖``."""
    s = np.asarray(sheaf_h, dtype=np.float64)
    h = np.asarray(h_t, dtype=np.float64)
    d = (s - h).reshape(s.shape[0], -1)
    return np.linalg.norm(d, axis=1) / (
        np.linalg.norm(h.reshape(h.shape[0], -1), axis=1) + 1e-9
    )


def electrode_groups(channels: list) -> dict[str, np.ndarray]:
    """Biology labels for 10-20/10-10 EEG channel names → per-node ``(N,)`` string arrays."""
    import re

    lobe_order = [
        ("FP", "frontal"),
        ("AF", "frontal"),
        ("FC", "central"),
        ("FT", "temporal"),
        ("CP", "parietal"),
        ("TP", "temporal"),
        ("PO", "occipital"),
        ("F", "frontal"),
        ("C", "central"),
        ("T", "temporal"),
        ("P", "parietal"),
        ("O", "occipital"),
        ("I", "occipital"),
    ]
    hemi, lobe = [], []
    for ch in channels:
        c = str(ch).strip()
        m = re.match(r"^([A-Za-z]+)([0-9]+|[zZ])?$", c)
        prefix = (m.group(1) if m else c).upper()
        suf = (m.group(2) or "") if m else ""
        if suf == "" or suf.lower() == "z":
            hemi.append("M")
        else:
            hemi.append("L" if int(suf) % 2 == 1 else "R")
        lobe.append(
            next((lb for pre, lb in lobe_order if prefix.startswith(pre)), "other")
        )
    return {"hemisphere": np.array(hemi), "lobe": np.array(lobe)}


def net_attention_operator(R: np.ndarray) -> np.ndarray:
    """Per-map singular-value-weighted left-vector operator ``P_L = U S U^T``. ``(P, m, m)``."""
    Rs = _as_stack(R)
    U, S, _ = np.linalg.svd(Rs, full_matrices=False)  # U: (P, m, k)
    return np.einsum("pik,pk,pjk->pij", U, S, U)  # (P, m, m)


def delta_energy(H: dict) -> np.ndarray:
    """Per-edge edge-space message energy ``‖delta_e‖``, window-averaged. ``(E,)``."""
    d = np.asarray(H["delta"], dtype=np.float64)  # (S, E, d_e)
    return np.linalg.norm(d, axis=-1).mean(0)


def transport_ratio(H: dict, eps: float = 1e-8) -> np.ndarray:
    """Per-edge transport ratio ``r_e = ‖delta_e‖ / ‖h_dst − h_src‖``, window-averaged. ``(E,)``."""
    d = np.linalg.norm(np.asarray(H["delta"], dtype=np.float64), axis=-1)  # (S, E)
    raw = np.linalg.norm(
        np.asarray(H["h_dst"], dtype=np.float64)
        - np.asarray(H["h_src"], dtype=np.float64),
        axis=-1,
    )  # (S, E)
    return (d / (raw + eps)).mean(0)


def transport_ratio_null(
    H: dict,
    kind: str = "orthogonal",
    match_scale: bool = True,
    seed: int = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-edge transport ratio under RANDOM maps — the chance floor for :func:`transport_ratio`."""
    hs = np.asarray(H["h_src"], dtype=np.float64)
    hd = np.asarray(H["h_dst"], dtype=np.float64)  # (S, E, H)
    S, E, Hd = hs.shape
    scale = (
        float(frob(np.asarray(H["rho_src"], float), "isometry").reshape(S, E).mean())
        if match_scale
        else 1.0
    )
    Rs = random_baseline((Hd, Hd), E, kind, seed=seed) * scale  # (E, H, H)
    Rd = random_baseline((Hd, Hd), E, kind, seed=seed + 1) * scale
    ts = np.einsum("sed,edh->seh", hs, Rs)
    td = np.einsum("sed,edh->seh", hd, Rd)
    d_null = np.linalg.norm(td - ts, axis=-1)  # (S, E)
    raw = np.linalg.norm(hd - hs, axis=-1)
    return (d_null / (raw + eps)).mean(0)


def message_energy(H: dict) -> np.ndarray:
    """Per-edge pulled-back node-space message ``‖msg_e‖ = ‖delta_e · ρ_dstᵀ‖``, window-avg. ``(E,)``."""
    delta = np.asarray(H["delta"], dtype=np.float64)  # (S, E, d_e)
    rho_dst = np.asarray(H["rho_dst"], dtype=np.float64)  # (S, E, H, d_e)
    msg = np.einsum("seh,sedh->sed", delta, rho_dst)  # (S, E, H) node space
    return np.linalg.norm(msg, axis=-1).mean(0)


def node_state_change(H: dict) -> np.ndarray:
    """Per-node relative change the sheaf makes to the encoding ``‖sheaf_h − h_t‖ / ‖h_t‖``. ``(N,)``."""
    s = np.asarray(H["sheaf_h"], dtype=np.float64)
    h = np.asarray(H["h_t"], dtype=np.float64)  # (S, N, H)
    return (np.linalg.norm(s - h, axis=-1) / (np.linalg.norm(h, axis=-1) + 1e-9)).mean(
        0
    )


def edge_distance(pairs: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Euclidean electrode distance for each ``(src, dst)`` pair. ``(E,)``."""
    pos = np.asarray(positions, dtype=np.float64)
    return np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)


def edge_transport_twist(H: dict) -> np.ndarray:
    """Per-edge parallel-transport twist ``‖Q_dstᵀ Q_src − I‖_F`` on the window-mean maps. ``(E,)``."""
    Qs = polar_orthogonal(
        np.asarray(H["rho_src"], dtype=np.float64).mean(0)
    )  # (E, H, d_e)
    Qd = polar_orthogonal(np.asarray(H["rho_dst"], dtype=np.float64).mean(0))
    T = np.einsum("eki,ekj->eij", Qd, Qs)  # Q_dstᵀ Q_src : (E, d_e, d_e)
    eye = np.eye(T.shape[-1])[None]
    return np.linalg.norm(T - eye, ord="fro", axis=(1, 2))


def principal_axes(M: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` eigenvectors + eigenvalue fractions of a symmetric operator ``M`` ``(m, m)``."""
    ev, V = np.linalg.eigh(0.5 * (M + M.T))
    ev, V = ev[::-1], V[:, ::-1]
    for i in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, i])), i] < 0:
            V[:, i] = -V[:, i]
    frac = ev / (ev.sum() + 1e-12)
    k = min(k, V.shape[1])
    return V[:, :k], frac[:k]


def bimodal_threshold(x: np.ndarray, bins: int = 128) -> float:
    """Otsu threshold splitting a 1-D distribution into two modes (dependency-free)."""
    x = np.asarray(x, dtype=np.float64)
    hist, edges = np.histogram(x, bins=bins)
    p = hist / max(hist.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    csum = np.cumsum(p * centers)
    m0 = csum / np.clip(w0, 1e-12, None)
    m1 = (csum[-1] - csum) / np.clip(w1, 1e-12, None)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def orthogonal_content(R: np.ndarray) -> dict[str, np.ndarray]:
    """Absolute orthogonal magnitude of each map — clarifies a polar ``rotation_frac``."""
    s = svdvals(R).astype(np.float64)  # (P, k)
    mean_sv = s.mean(-1)
    iso = mean_sv**2 / ((s**2).mean(-1) + 1e-12)
    return {
        "orthogonal_scale": mean_sv,
        "max_sv": s.max(-1),
        "min_sv": s.min(-1),
        "isometric_energy_frac": iso,
        "anisotropy": 1.0 - iso,
    }


def polar_orthogonal(R: np.ndarray) -> np.ndarray:
    """The orthogonal (partial-isometry) component ``Q = U V^T`` of ``rho = Q P``. ``(P, m, n)``."""
    Rs = _as_stack(R)
    U, _, Vt = np.linalg.svd(Rs, full_matrices=False)
    return U @ Vt


def align_edge_maps(
    pairs_a: np.ndarray, Ra: np.ndarray, pairs_b: np.ndarray, Rb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict two per-edge map sets to their shared ``(src, dst)`` edges, aligned in order."""
    key = lambda P: {(int(u), int(v)): i for i, (u, v) in enumerate(P)}
    ka, kb = key(pairs_a), key(pairs_b)
    common = [k for k in ka if k in kb]
    ia = [ka[k] for k in common]
    ib = [kb[k] for k in common]
    return Ra[ia], Rb[ib], np.array(common)


def cross_condition_gram(
    R1: np.ndarray, R2: np.ndarray, normalize: bool = True
) -> dict:
    """Cosine Gram between two edge-aligned map sets — do two conditions recover the same maps?"""
    A, B = flatten_maps(R1).astype(np.float64), flatten_maps(R2).astype(np.float64)
    if normalize:
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    G = A @ B.T  # (E, E) cosine similarity
    E = G.shape[0]
    diag = np.diag(G)
    off = G[~np.eye(E, dtype=bool)]
    return {
        "gram": G,
        "diag": diag,
        "diag_mean": float(diag.mean()),
        "offdiag_mean": float(off.mean()),
        "separation": float(diag.mean() - off.mean()),
    }


def map_field_smoothness(
    Q: np.ndarray, pairs: np.ndarray, n_null: int = 5, seed: int = 0
) -> dict:
    """Smoothness of the (orthogonal) map field over the graph vs a shuffled-map null."""
    from collections import defaultdict

    inc: dict = defaultdict(list)
    for i, (u, v) in enumerate(pairs):
        inc[int(u)].append(i)
        inc[int(v)].append(i)
    adj = set()
    for es in inc.values():
        for a in range(len(es)):
            for b in range(a + 1, len(es)):
                adj.add((es[a], es[b]))
    if not adj:
        return {
            "learned": float("nan"),
            "null": float("nan"),
            "smoothness_ratio": float("nan"),
            "n_adj": 0,
        }
    A = np.array(sorted(adj))

    def energy(M):
        d = (M[A[:, 0]] - M[A[:, 1]]).reshape(len(A), -1)
        return float(np.linalg.norm(d, axis=1).mean())

    learned = energy(Q)
    rng = np.random.default_rng(seed)
    null = float(
        np.mean([energy(Q[rng.permutation(Q.shape[0])]) for _ in range(n_null)])
    )
    return {
        "learned": learned,
        "null": null,
        "smoothness_ratio": learned / (null + 1e-12),
        "n_adj": len(A),
    }


def edge_cluster_separation(
    R: np.ndarray, subsample: int = 3000, seed: int = 0
) -> dict:
    """How well the maps distinguish edges — the PE-diversity / discriminability readout."""
    R = np.asarray(R, dtype=np.float64)
    S, E = R.shape[0], R.shape[1]
    X = R.reshape(S, E, -1)  # (S, E, D)
    edge_mean = X.mean(0)  # (E, D)
    grand = edge_mean.mean(0)
    between = float(((edge_mean - grand) ** 2).sum(1).mean())
    within = float(((X - edge_mean[None]) ** 2).sum(-1).mean())

    flat = X.reshape(S * E, -1)
    labels = np.tile(np.arange(E), S)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], size=min(subsample, flat.shape[0]), replace=False)
    sub, sub_lab = flat[idx], labels[idx]
    # brute-force nearest neighbor (exclude self) — small subsample keeps this O(n^2) cheap
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(1)
    acc = float((sub_lab[nn] == sub_lab).mean())
    return {
        "between": between,
        "within": within,
        "separation_ratio": between / (within + 1e-12),
        "nn_edge_acc": acc,
        "chance": 1.0 / E,
    }


def save_scalp_animation(
    edge_vals: np.ndarray,
    ei_np: np.ndarray,
    pos2d: np.ndarray,
    out_path: str,
    fps: int = 10,
    scale: str = "global",
    cmap: str = "viridis",
    stride: int = 1,
    max_frames: Optional[int] = None,
    dpi: int = 110,
    title: str = "edge value",
    node_color: str = "k",
) -> str:
    """Animate a per-window edge field on the scalp and save a video."""
    import shutil
    import matplotlib.pyplot as plt
    import matplotlib.animation as manim
    from matplotlib.collections import LineCollection

    V = np.asarray(edge_vals, dtype=float)  # (T, E)
    if V.ndim != 2:
        raise ValueError(f"edge_vals must be (T, E), got {V.shape}")
    T = V.shape[0]
    frames = list(range(0, T, max(1, stride)))
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError("no frames selected (check stride/max_frames)")

    seg = [
        [(pos2d[s, 0], pos2d[s, 1]), (pos2d[d, 0], pos2d[d, 1])]
        for s, d in zip(ei_np[0], ei_np[1])
    ]

    if scale == "per_frame":
        norm = plt.Normalize(0.0, 1.0)
        cbar_label = "min–max per frame"

        def vals_for(f):
            v = V[f]
            lo, hi = np.nanmin(v), np.nanmax(v)
            return (v - lo) / (hi - lo + 1e-12)
    else:
        sub = V[frames]
        vmin, vmax = (
            (float(np.nanmin(sub)), float(np.nanmax(sub)))
            if scale == "global01"
            else (0.0, float(np.nanmax(sub)))
        )
        norm = plt.Normalize(vmin, vmax)
        cbar_label = title

        def vals_for(f):
            return V[f]

    fig, ax = plt.subplots(figsize=(6, 6))
    r = np.abs(pos2d).max() * 1.15
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(th), r * np.sin(th), color="k", lw=1)
    ax.plot([-0.09 * r, 0, 0.09 * r], [r * 0.99, r * 1.12, r * 0.99], color="k", lw=1)
    lc = LineCollection(seg, cmap=cmap, norm=norm, lw=1.6, alpha=0.9)
    lc.set_array(vals_for(frames[0]))
    ax.add_collection(lc)
    ax.scatter(pos2d[:, 0], pos2d[:, 1], s=18, c=node_color, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(lc, ax=ax, fraction=0.046, label=cbar_label)
    ttl = ax.set_title(f"{title} — window {frames[0]}", fontsize=11)

    def update(f):
        lc.set_array(vals_for(f))
        ttl.set_text(f"{title} — window {f}")
        return lc, ttl

    anim = manim.FuncAnimation(
        fig, update, frames=frames, interval=1000 / fps, blit=False
    )

    out_path = str(out_path)
    wrote = out_path
    try:
        if out_path.lower().endswith(".mp4"):
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg not found")
            anim.save(
                out_path, writer=manim.FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi
            )
        else:
            anim.save(out_path, writer=manim.PillowWriter(fps=fps), dpi=dpi)
    except Exception:
        wrote = out_path.rsplit(".", 1)[0] + ".gif"  # ffmpeg unavailable → GIF fallback
        anim.save(wrote, writer=manim.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return wrote
