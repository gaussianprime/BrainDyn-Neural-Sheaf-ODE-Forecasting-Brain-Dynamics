from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.sn_dataset import (
    DATA_NPZ_PATH as NEST_DATA_NPZ_PATH,
    make_dataloaders as make_nest_dataloaders,
    make_nest_run_loader,
)
from main import (
    SubjectRunDataset,
    _validate_new_flag_combinations,
    assert_disjoint_runs,
    batch_to_model_tensors,
    build_granger_graph,
    compute_train_global_stats,
    compute_shuffle_split_indices,
    graph_health_line,
    make_subset_loader,
    pad_collate_runs,
    resolve_shuffle_seeds,
    run_epoch,
    run_epoch_ar_train,
    run_test_rollout_chunks,
    set_seed,
    split_fingerprint,
    subject_groups_for,
    teacher_forcing_probability,
)
from model.braindyn import BrainDyn, BrainDynConfig
from model.graph_builders import (
    adjacency_to_edge_index,
    build_lap_pe,
    build_spatial_graph_from_positions,
    threshold_granger_scores,
)


def _require_finite_metrics(metrics: dict[str, float], label: str) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise RuntimeError(f"Non-finite {label} metric {key}={value}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="BrainDyn trainer for the NEST benchmark."
    )
    ap.add_argument(
        "--model",
        choices=["braindyn"],
        default="braindyn",
    )
    ap.add_argument(
        "--dataset",
        choices=["nest"],
        default="nest",
    )
    ap.add_argument("--nest_npz_path", default=str(NEST_DATA_NPZ_PATH))
    ap.add_argument(
        "--nest_task_mode",
        choices=["forecasting", "perturb_forecast"],
        default="forecasting",
        help="perturb_forecast additionally evaluates the test fold's context "
        "against the perturbed horizon after normal training/testing.",
    )
    ap.add_argument("--nest_train_frac", type=float, default=0.8)
    ap.add_argument("--nest_val_frac", type=float, default=0.1)
    ap.add_argument("--nest_split_seed", type=int, default=0)
    ap.add_argument(
        "--perturb_post_onset_frac",
        type=float,
        default=None,
        help="Fraction of the perturb_forecast context at or after the "
        "perturbation onset. Unset keeps the default x/10 anchoring.",
    )
    ap.add_argument(
        "--perturb_context_gap_bins",
        type=int,
        default=0,
        help="Bins skipped between the end of the context and the horizon.",
    )

    ap.add_argument("--x", type=int, default=32)
    ap.add_argument("--y", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--norm_mode", choices=["context", "train_global"], default="train_global")
    ap.add_argument("--forecast_mode", choices=["short", "long", "long_ar_train"], default="long")
    ap.add_argument("--ar_chunk_size", type=int, default=8)
    ap.add_argument(
        "--ar_stride",
        type=int,
        default=None,
        help=(
            "Spacing (in timepoints) between TBPTT windows in long_ar_train "
            "mode. Defaults to ar_chunk_size*tbptt_chunks (every timepoint "
            "of every run walked once per epoch); set larger to skip the "
            "gap between windows and cut per-epoch compute."
        ),
    )
    ap.add_argument("--test_rollout_steps", type=int, default=32)
    ap.add_argument("--tbptt_chunks", type=int, default=3)
    ap.add_argument("--ss_start", type=float, default=0.0)
    ap.add_argument("--ss_end", type=float, default=0.0)
    ap.add_argument("--ss_decay_epochs", type=int, default=None)
    ap.add_argument("--run_batch_size", type=int, default=8)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--no_pin_memory", action="store_true")
    ap.add_argument(
        "--num_shuffles",
        type=int,
        default=5,
        help="number of random subject-shuffle splits to run. Each shuffle draws a "
        "fresh subject-grouped train/val/test split (see --train_frac/--val_frac) and "
        "trains one model; metrics are summarized as mean +/- std across shuffles. "
        "Ignored if --shuffle_seeds is given (count becomes that list's length).",
    )
    ap.add_argument(
        "--shuffle_index",
        type=int,
        default=-1,
        help="run only this single 0-based shuffle and skip the rest; -1 (default) "
        "runs all of them. The shuffle's seed fixes its split AND its RNG stream "
        "(weights, batch order, dropout), so shuffle k here reproduces shuffle k of a "
        "full run. Lets a SLURM array own one shuffle each.",
    )
    ap.add_argument(
        "--shuffle_seeds",
        type=int,
        nargs="*",
        default=None,
        help="explicit per-shuffle seeds, e.g. --shuffle_seeds 0 1 2 3 4. Each seed "
        "drives both its split and its training RNG. Overrides --num_shuffles. Omit to "
        "derive seeds as --seed * 1000 + i.",
    )
    ap.add_argument(
        "--train_frac",
        type=float,
        default=0.7,
        help="fraction of SUBJECTS in each shuffle's train split (integer subject "
        "counts; test_frac = 1 - train_frac - val_frac).",
    )
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="fraction of SUBJECTS in each shuffle's val split.",
    )
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training entirely and load the existing best checkpoint "
        "at --save_path's derived fold path, then run eval (normal + "
        "perturbed if --nest_task_mode perturb_forecast) only. Errors if no "
        "checkpoint exists for that (model, fold) yet.",
    )
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2)

    ap.add_argument("--hidden_dim", type=int, default=16)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--lstm_dropout", type=float, default=0.0)

    ap.add_argument(
        "--graph_mode",
        choices=["granger", "structural"],
        default="granger",
        help="granger: pooled Granger graph over the train fold. structural: "
             "edge frequency of the ground-truth connectome over the train "
             "subjects, cut with --fc_threshold_mode.",
    )
    ap.add_argument("--granger_threshold", type=float, default=0.01)
    ap.add_argument("--granger_lag", type=int, default=1)
    ap.add_argument(
        "--granger_threshold_mode",
        choices=["manual", "topk", "topk_per_node"],
        default="topk",
    )
    ap.add_argument("--granger_topk_edges", type=int, default=2000)
    ap.add_argument(
        "--granger_topk_per_node",
        type=int,
        default=0,
        help="edges kept per node when --granger_threshold_mode=topk_per_node. "
             "Set this to --granger_topk_edges // num_nodes to match the global "
             "mode's TOTAL edge count exactly (500/100 = 5 for NEST), so the two "
             "graph constructions are compared at equal sparsity. Global top-k "
             "leaves nodes isolated, and an isolated node's Laplacian row is "
             "identically zero, so the ODE coupling term never reaches it.",
    )
    ap.add_argument(
        "--fc_threshold_mode",
        choices=["topk", "topk_per_node"],
        default="topk",
        help="Cut applied to the structural prior.",
    )
    ap.add_argument(
        "--fc_topk_edges",
        type=int,
        default=0,
        help="Global edge budget for the structural prior (0: --granger_topk_edges).",
    )
    ap.add_argument(
        "--fc_topk_per_node",
        type=int,
        default=0,
        help="Per-node edge budget for the structural prior (0: --granger_topk_per_node).",
    )

    # BrainDyn configuration used in the paper.
    ap.add_argument("--map_hidden_dim", type=int, default=16)
    ap.add_argument("--map_mlp_hidden_dim", type=int, default=64)
    ap.add_argument("--vf_hidden_dim", type=int, default=128)
    ap.add_argument("--vf_layers", type=int, default=2)
    ap.add_argument("--edge_specific_maps", action="store_true", default=True)
    # Both are read DIRECTLY (not via getattr) by main.py's shared
    # _validate_new_flag_combinations, so this CLI must define them or the
    # validator raises AttributeError before training starts.
    ap.add_argument("--ablation_gcn", action="store_true",
                    help="Replace the sheaf Laplacian with a plain GCN aggregator.")
    ap.add_argument("--static_edge_maps", action="store_true",
                    help="Per-edge restriction-map tables instead of the shared map MLP.")
    ap.add_argument("--sheaf_layers", type=int, default=1)
    ap.add_argument("--diffusion_step", type=float, default=1.0)
    ap.add_argument("--sheaf_map_pe", choices=["none", "lappe", "learned"], default="lappe")
    ap.add_argument("--sheaf_map_pe_dim", type=int, default=8)

    # Sheaf-operator knobs, mirroring main.py so this trainer can run the same
    # BrainDyn configs.
    ap.add_argument(
        "--ablation_no_lstm",
        action="store_true",
        help="Replace the LSTM temporal encoder with a linear projection of the "
             "LAST time step (model/dynamics.py's no_lstm_proj).",
    )
    ap.add_argument(
        "--identity_restriction_init",
        action="store_true",
        help="Initialize the sheaf restriction maps at the identity (torch.eye) "
             "instead of Xavier random, so the sheaf starts as a plain "
             "graph-diffusion operator and learns to deviate. For the shared "
             "restriction-map MLP this is a zero-weight final layer + identity bias.",
    )
    ap.add_argument(
        "--sheaf_norm",
        choices=["none", "sym", "row"],
        default="none",
        help="Degree normalization of the sheaf Laplacian. 'none' (default) is the "
             "original unnormalized operator, whose diagonal block grows LINEARLY in "
             "node degree, pushing lambda_max(L_F) far past the 2.0 that "
             "(I - step*L_F) needs -- the diffusion then amplifies instead of "
             "diffusing. 'sym' applies D^-1/2 L D^-1/2 (the sheaf analogue of GCN's "
             "D^-1/2 A D^-1/2): bounds lambda_max and rescales each node by its OWN "
             "degree, which a single --diffusion_step cannot do. 'row' is D^-1 L "
             "(non-symmetric, so L is no longer PSD) for comparison only.",
    )
    ap.add_argument(
        "--sheaf_map_scale",
        choices=["none", "norm", "orth"],
        default="none",
        help="Reparametrize the restriction maps as rho = exp(s) * direction(M) with a "
             "single LEARNED scalar s, instead of letting the MLP set the magnitude. "
             "'norm' normalizes M's magnitude; 'orth' uses its polar (orthogonal) "
             "direction.",
    )
    ap.add_argument(
        "--freeze_map_scale",
        action="store_true",
        help="Pin the learned map scale s at its init. Requires --sheaf_map_scale.",
    )
    ap.add_argument(
        "--learn_diffusion_gain",
        action="store_true",
        help="Learn a scalar multiplier on --diffusion_step instead of fixing it. "
             "Parameterized in log space, init exactly 1.0.",
    )
    ap.add_argument(
        "--no_sheaf",
        action="store_true",
        help="Trivial-sheaf ablation: pin the restriction maps at the identity and "
             "freeze them, collapsing the sheaf to the plain graph Laplacian L = D - A. "
             "Isolates the effect of the LEARNED restriction map. Distinct from "
             "--identity_restriction_init (identity only at init, then trained).",
    )
    # ---- observation model / ODE variant --------------------------------
    # Defaults here MUST track main.py's, or the same flags mean different
    # models in the two trainers. _validate_new_flag_combinations (imported
    # from main.py and called below) covers the cross-flag rules for all of
    # these, so nothing needs restating.
    ap.add_argument(
        "--coupling_block",
        choices=["none", "block", "complex"],
        default="block",
        help="Restrict the COUPLING restriction maps to be block-diagonal, pairing "
             "stalk dimension s with s + d/2 ('complex' keeps only the "
             "rotation-scaling part of each 2x2 block). The coupling maps are only "
             "reported in the sheaf's aux output; the DIFFUSION maps used in the "
             "forward pass stay dense either way.",
    )
    ap.add_argument(
        "--time_embed_dim",
        type=int, default=16,
        help="Width of the sinusoidal time embedding. "
             "Half sin, half cos; an odd value zero-pads the last "
             "channel.",
    )
    ap.add_argument(
        "--time_embed_max_period",
        type=float, default=16.0,
        help="Longest wavelength in the fixed frequency schedule, in SOLVER-TIME "
             "units (forecast steps). "
             "Deliberately NOT the Transformer default of 10000: the clock restarts "
             "at t=0 every forward call and t never exceeds the rollout length, at "
             "which 10000 leaves the slow channels numerically constant. Scale with "
             "the ROLLOUT LENGTH, not with TR or sampling rate.",
    )
    ap.add_argument(
        "--time_rate_mode",
        choices=["fixed", "global", "per_node"],
        default="fixed",
        help="Learn the OUTPUT amplitude R in dx_i/dt = R*tanh(MLP(...)). "
             "'fixed' (default) keeps the constant "
             "SinusoidalTimeVariant.RATE (2.0). 'global' learns one scalar; 'per_node' learns "
             "one per region (N params). Parameterized as R = 2*rate*sigmoid(rho) with "
             "rho init 0, so R == rate EXACTLY at initialization and |dx/dt| <= 2*rate "
             "always -- the a-priori no-divergence bound survives, 2x looser. This is "
             "the only place a learned amplitude adds anything: an amplitude on the "
             "TIME EMBEDDING is absorbed exactly by the field's first nn.Linear.",
    )
    ap.add_argument(
        "--time_learn_freqs",
        action="store_true",
        help="Learn the sinusoid FREQUENCIES w_k in the time embedding, "
             "initialized AT the fixed geometric "
             "schedule so training starts from the fixed encoding exactly. The "
             "frequency is the only part of a1*sin(2*pi*w*t) + a2*cos(2*pi*w*t) worth "
             "learning here -- a1/a2 already exist as the field's first nn.Linear "
             "columns. Capped at --time_max_cycles_per_step (Nyquist).",
    )
    ap.add_argument(
        "--time_max_cycles_per_step",
        type=float, default=0.5,
        help="Nyquist cap on learned frequencies, in cycles per forecast step "
             "(default 0.5 = one full cycle per two steps, the fastest the dt=1 output "
             "grid can represent). Only used with --time_learn_freqs.",
    )

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--lr_patience", type=int, default=3)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument(
        "--patience",
        type=int,
        default=None,
        help="stop training a shuffle once this many consecutive --val_every "
        "checks pass with no improvement in val total loss (train-to-convergence "
        "instead of always running the full --epochs). Distinct from --lr_patience, "
        "which only controls the LR schedule. Default None = disabled (unchanged "
        "fixed-epoch behavior).",
    )
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=1.0)
    ap.add_argument("--lambda_mae", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_path", default="checkpoints/benchmark_best.pt")
    args = ap.parse_args()
    _validate_new_flag_combinations(args)
    return args


def load_dataset(args: argparse.Namespace):
    if args.dataset != "nest":
        raise ValueError(
            f"--dataset {args.dataset!r} is not supported by this trainer; "
            "only 'nest' is. Use main.py for fMRI/LEMON EEG."
        )
    loaders = make_nest_dataloaders(
        npz_path=args.nest_npz_path,
        task_mode=args.nest_task_mode,
        x=args.x,
        y=args.y,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=args.nest_train_frac,
        val_frac=args.nest_val_frac,
        split_seed=args.nest_split_seed,
        cache=args.cache,
        pin_memory=not args.no_pin_memory,
        norm_mode=args.norm_mode,
        perturb_post_onset_frac=args.perturb_post_onset_frac,
        perturb_context_gap_bins=args.perturb_context_gap_bins,
    )
    run_loader = make_nest_run_loader(args.nest_npz_path)
    return loaders, run_loader


def _batch_to_model_tensors_perturbed(batch, device, norm_stats=None):
    """Same transform as main.py::batch_to_model_tensors, but runs the model
    on batch["x_perturbed"]/batch["y_perturbed"] instead of batch["x"]/
    batch["y"] -- evaluates the (normally-trained) model on the perturbed
    trace as an alternate test set: perturbed context in, forecast, score
    against the perturbed continuation. Only valid for --nest_task_mode
    perturb_forecast batches (which carry x_perturbed/y_perturbed)."""
    from main import normalize_with_stats

    x_ctx = batch["x_perturbed"].to(device=device, dtype=torch.float32)
    y_future = batch["y_perturbed"].to(device=device, dtype=torch.float32)
    if norm_stats is not None:
        x_ctx = normalize_with_stats(x_ctx, norm_stats)
        y_future = normalize_with_stats(y_future, norm_stats)
    x_history = x_ctx.permute(0, 2, 1).unsqueeze(-1)
    y_true = y_future.permute(1, 0, 2).unsqueeze(-1)
    return x_history, y_true


def build_structural_group_graph(
    npz_path, train_subjects, topk_edges, threshold_mode="topk", topk_per_node=0
):
    """Structural prior: directed edge frequency of the ground-truth connectome
    over the train subjects, binarized with a global or per-node top-k cut."""
    adjacency = np.load(npz_path)["adjacency"]
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError(f"expected a (S, N, N) adjacency, got {adjacency.shape}")
    n_subjects, num_nodes = adjacency.shape[0], adjacency.shape[1]
    idx = np.unique(np.asarray([int(s) for s in train_subjects], dtype=np.int64))
    if idx.size == 0:
        raise RuntimeError("structural graph: no train subjects in this fold.")
    if idx.min() < 0 or idx.max() >= n_subjects:
        raise IndexError(f"train subject index outside [0, {n_subjects})")
    freq = (adjacency[idx] != 0).mean(axis=0).astype(np.float64)
    np.fill_diagonal(freq, 0.0)
    score = torch.from_numpy(freq).float()
    adjacency_mask, threshold_used = threshold_granger_scores(
        score=score,
        threshold=0.0,
        threshold_mode=threshold_mode,
        topk_edges=int(topk_edges),
        topk_per_node=int(topk_per_node),
    )
    edge_index = adjacency_to_edge_index(adjacency_mask)
    if edge_index.shape[1] == 0:
        raise RuntimeError("structural graph selected no edges.")
    info = {
        "mode": threshold_mode,
        "graph_kind": "structural",
        "topk_per_node": int(topk_per_node),
        "threshold": float(threshold_used),
        "topk_edges": int(topk_edges),
        "selected_edges": int(edge_index.shape[1]),
        "n_subjects": int(idx.size),
    }
    return edge_index, score, info, int(num_nodes)


def build_fold_graph(
    args, train_loader, loaders, fold_norm_stats, fold_idx, device, train_subjects=None
):
    if args.graph_mode == "granger":
        edge_index_cpu, graph_score, graph_info = build_granger_graph(
            loader=train_loader,
            # No max_batches: the Granger prior is built from the whole loader,
            # subject-grouped.
            threshold=args.granger_threshold,
            lag=args.granger_lag,
            threshold_mode=args.granger_threshold_mode,
            topk_edges=args.granger_topk_edges,
            topk_per_node=args.granger_topk_per_node,
            norm_stats=fold_norm_stats,
        )
        graph_tag = "Granger"
        num_nodes = int(graph_score.shape[0])
    elif args.graph_mode == "structural":
        if not train_subjects:
            raise RuntimeError("--graph_mode structural needs the fold's train subjects.")
        edge_index_cpu, graph_score, graph_info, num_nodes = build_structural_group_graph(
            args.nest_npz_path,
            train_subjects,
            topk_edges=args.fc_topk_edges or args.granger_topk_edges,
            threshold_mode=args.fc_threshold_mode,
            topk_per_node=args.fc_topk_per_node or args.granger_topk_per_node,
        )
        graph_tag = "Structural"
    else:
        raise ValueError(f"no graph builder matched: graph_mode={args.graph_mode!r}")

    edge_index = edge_index_cpu.to(device)
    print(
        f"Shuffle {fold_idx + 1}/{args.num_shuffles} | {graph_tag} graph: "
        f"N={num_nodes}, E={edge_index.shape[1]}, info={graph_info}"
    )
    return edge_index, edge_index_cpu, graph_score, graph_info, graph_tag, num_nodes


def make_model(
    args,
    num_nodes: int,
    edge_index_cpu: torch.Tensor,
    graph_tag: str,
    device,
):
    sheaf_node_pe = None
    if args.sheaf_map_pe == "lappe":
        sheaf_node_pe = build_lap_pe(
            edge_index_cpu,
            num_nodes=num_nodes,
            rank=args.sheaf_map_pe_dim,
        )
        print(
            f"BrainDyn sheaf-map LapPE: shape={tuple(sheaf_node_pe.shape)} "
            f"from symmetrized {graph_tag} graph"
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
        edge_specific_maps=args.edge_specific_maps,
        sheaf_layers=args.sheaf_layers,
        diffusion_step=args.diffusion_step,
        sheaf_mlp_maps=(not args.static_edge_maps),
        map_mlp_hidden_dim=args.map_mlp_hidden_dim,
        use_lstm_encoder=not args.ablation_no_lstm,
        identity_restriction_init=args.identity_restriction_init,
        sheaf_map_pe=args.sheaf_map_pe,
        sheaf_map_pe_dim=args.sheaf_map_pe_dim,
        frozen_identity=args.no_sheaf,
        sheaf_norm=args.sheaf_norm,
        sheaf_map_scale=args.sheaf_map_scale,
        freeze_map_scale=args.freeze_map_scale,
        learn_diffusion_gain=args.learn_diffusion_gain,
        coupling_block=args.coupling_block,
        time_embed_dim=args.time_embed_dim,
        time_embed_max_period=args.time_embed_max_period,
        time_rate_mode=args.time_rate_mode,
        time_learn_freqs=args.time_learn_freqs,
        time_max_cycles_per_step=args.time_max_cycles_per_step,
    )
    model = BrainDyn(config, sheaf_node_pe=sheaf_node_pe).to(device)
    model.register_restriction_edges(edge_index_cpu.to(device))
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin = (not args.no_pin_memory) and torch.cuda.is_available()
    print(f"Device: {device}")
    print(
        f"Benchmark protocol: model={args.model} dataset={args.dataset} "
        f"x={args.x} y={args.y} mode={args.forecast_mode} "
        f"rollout={args.test_rollout_steps} shuffles={args.num_shuffles} seed={args.seed}"
    )

    loaders, run_loader = load_dataset(args)
    # Shuffle-split scheme: pool ALL subjects -- the manifest's own train/val/test
    # labels are ignored here -- and re-draw a subject-grouped train/val/test split
    # per shuffle (see compute_shuffle_split_indices).
    combined_dataset = ConcatDataset(
        [loaders["train"].dataset, loaders["val"].dataset, loaders["test"].dataset]
    )
    combined_groups = subject_groups_for(combined_dataset)
    n_subjects = len(set(combined_groups.tolist()))
    print(
        f"Shuffle-split pool: {len(combined_dataset)} windows from {n_subjects} subjects "
        "(manifest train/val/test labels ignored; split grouped by subject)"
    )
    shuffle_seeds = resolve_shuffle_seeds(args.seed, args.num_shuffles, args.shuffle_seeds)
    num_shuffles = len(shuffle_seeds)
    if args.shuffle_index >= 0:
        if args.shuffle_index >= num_shuffles:
            raise ValueError(
                f"--shuffle_index {args.shuffle_index} out of range for "
                f"{num_shuffles} shuffle(s) (valid: 0..{num_shuffles - 1})"
            )
        folds_to_run = [args.shuffle_index]
        print(
            f"Single-shuffle mode: running only shuffle "
            f"{args.shuffle_index + 1}/{num_shuffles} "
            "(seed-determined, so identical to that shuffle of a full run)."
        )
    else:
        folds_to_run = list(range(num_shuffles))
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fold_tests = []
    fold_tests_perturbed = []
    fold_tests_excl = []
    fold_tests_perturbed_excl = []
    run_perturbed_eval = args.dataset == "nest" and args.nest_task_mode == "perturb_forecast"
    perturbed_run_loader = None
    batch_channel_mask = None
    path_channel_mask = None
    if run_perturbed_eval:
        import numpy as _np

        from data.sn_dataset import make_nest_run_loader_perturbed
        from main import channel_exclusion_mask

        perturbed_run_loader = make_nest_run_loader_perturbed(args.nest_npz_path)

        _pert_npz = _np.load(args.nest_npz_path, mmap_mode="r", allow_pickle=True)
        _pert_nodes = _np.asarray(_pert_npz["perturbation_nodes"], dtype=_np.int64)
        # Channel count from the dataset itself: the fold-loop local `num_nodes`
        # is not bound until build_fold_graph runs.
        _n_channels = int(_pert_npz["smoothed_rates_hz_original"].shape[1])

        def batch_channel_mask(batch, device):
            """run_epoch hook -- per-sample target index rides in the batch meta."""
            return channel_exclusion_mask(
                batch["meta"]["perturbed_node"], _n_channels, device
            )

        def path_channel_mask(path, device):
            """run_test_rollout_chunks hook -- meta is out of scope by then, so look the
            target up from the run key, which for NEST is the subject index."""
            return channel_exclusion_mask(
                torch.tensor([int(_pert_nodes[int(path), 0])]), _n_channels, device
            )
    for fold_idx in folds_to_run:
        # Each shuffle's seed drives BOTH its split and its training RNG (init,
        # batch order, dropout), so running only --shuffle_index k reproduces
        # shuffle k of a full run bit-for-bit. `* 1000` in the derived seeds keeps
        # a base-seed sweep from aliasing (see resolve_shuffle_seeds).
        fold_seed = shuffle_seeds[fold_idx]
        set_seed(fold_seed)
        train_idx, val_idx, test_idx = compute_shuffle_split_indices(
            combined_groups, fold_seed, args.train_frac, args.val_frac
        )
        train_subjects = set(combined_groups[train_idx].tolist())
        val_subjects = set(combined_groups[val_idx].tolist())
        test_subjects = set(combined_groups[test_idx].tolist())
        # The single assert that would catch a window-level split: subjects must
        # be disjoint across all three splits.
        assert (
            not (train_subjects & val_subjects)
            and not (train_subjects & test_subjects)
            and not (val_subjects & test_subjects)
        ), (
            f"shuffle {fold_idx}: subjects overlap across splits — the split is "
            "not subject-grouped"
        )
        print(
            f"Shuffle {fold_idx + 1}/{num_shuffles} (seed {fold_seed}) | "
            f"train {len(train_idx)} win / {len(train_subjects)} subj, "
            f"val {len(val_idx)} win / {len(val_subjects)} subj, "
            f"test {len(test_idx)} win / {len(test_subjects)} subj | "
            f"fingerprint: {split_fingerprint(train_idx, combined_groups)}"
        )
        train_loader = make_subset_loader(
            combined_dataset,
            train_idx,
            args.batch_size,
            args.num_workers,
            use_pin,
            True,
        )
        val_loader = make_subset_loader(
            combined_dataset,
            val_idx,
            args.batch_size,
            args.num_workers,
            use_pin,
            False,
        )
        test_loader = make_subset_loader(
            combined_dataset,
            test_idx,
            args.batch_size,
            args.num_workers,
            use_pin,
            False,
        )
        fold_norm_stats = (
            compute_train_global_stats(combined_dataset, train_idx)
            if args.norm_mode == "train_global"
            else None
        )
        edge_index, edge_index_cpu, graph_score, graph_info, graph_tag, num_nodes = build_fold_graph(
            args, train_loader, loaders, fold_norm_stats, fold_idx, device,
            train_subjects=train_subjects,
        )
        model = make_model(args, num_nodes, edge_index_cpu, graph_tag, device)
        # One FIXED probe batch for the per-epoch graph-health line, materialized
        # once. graph_health_line only does next(iter(loader)), so a one-element
        # list is a valid "loader" -- and re-drawing a random batch every epoch
        # would both fork/join the DataLoader workers 120x per fold and make
        # |L_F h|/|h| wobble with batch content rather than read as a trend.
        # long_ar_train's run loader yields a different batch shape that
        # batch_to_model_tensors cannot read, so skip it there.
        graph_probe = None
        if args.forecast_mode != "long_ar_train":
            graph_probe = [next(iter(train_loader))]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
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

        train_run_loader = val_run_loader = None
        if args.forecast_mode == "long_ar_train":
            train_run_dataset = SubjectRunDataset(
                combined_dataset, train_idx, run_loader, args.cache
            )
            val_run_dataset = SubjectRunDataset(
                combined_dataset, val_idx, run_loader, args.cache
            )
            assert_disjoint_runs(train_run_dataset, val_run_dataset)
            train_run_loader = DataLoader(
                train_run_dataset,
                batch_size=args.run_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=use_pin,
                persistent_workers=(args.num_workers > 0),
                collate_fn=pad_collate_runs,
            )
            val_run_loader = DataLoader(
                val_run_dataset,
                batch_size=args.run_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=use_pin,
                persistent_workers=(args.num_workers > 0),
                collate_fn=pad_collate_runs,
            )

        fold_save_path = save_path.with_name(
            f"{save_path.stem}_{args.model}_fold{fold_idx + 1}{save_path.suffix}"
        )
        resume_path = fold_save_path.with_name(
            f"{fold_save_path.stem}_resume{fold_save_path.suffix}"
        )
        best_val = float("inf")
        best_val_metrics = None
        start_epoch = 1
        epochs_without_improve = 0
        if args.eval_only:
            if not fold_save_path.exists():
                raise RuntimeError(
                    f"--eval_only requires an existing checkpoint at "
                    f"{fold_save_path}, but none found -- train this fold "
                    "normally first."
                )
            start_epoch = args.epochs + 1  # skips the epoch loop below entirely
            print(
                f"Fold {fold_idx + 1}: --eval_only set, skipping training, "
                f"using {fold_save_path}"
            )
        elif resume_path.exists():
            resume_ckpt = torch.load(resume_path, map_location=device)
            if resume_ckpt["model"] != args.model:
                raise RuntimeError(
                    f"Resume checkpoint {resume_path} was trained with "
                    f"model={resume_ckpt['model']!r}, not {args.model!r} -- "
                    "delete the stale checkpoint if this mismatch is expected."
                )
            model.load_state_dict(resume_ckpt["model_state_dict"])
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            if scaler is not None and resume_ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
            best_val = resume_ckpt["best_val"]
            best_val_metrics = resume_ckpt["best_val_metrics"]
            epochs_without_improve = resume_ckpt.get("epochs_without_improve", 0)
            start_epoch = resume_ckpt["epoch"] + 1
            print(
                f"Fold {fold_idx + 1}: resumed from {resume_path} at epoch "
                f"{start_epoch}/{args.epochs} (best_val={best_val:.6f})"
            )
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.perf_counter()
            if args.forecast_mode == "long_ar_train":
                tf_prob = teacher_forcing_probability(
                    args.ss_start,
                    args.ss_end,
                    epoch,
                    args.epochs,
                    decay_epochs=args.ss_decay_epochs,
                )
                train_metrics = run_epoch_ar_train(
                    model,
                    train_run_loader,
                    edge_index,
                    args.dt,
                    args.x,
                    args.ar_chunk_size,
                    args.tbptt_chunks,
                    optimizer,
                    args.lambda_mse,
                    args.lambda_mae,
                    args.grad_clip,
                    f"fold {fold_idx + 1} train {epoch}/{args.epochs} [tf={tf_prob:.2f}]",
                    scaler=scaler,
                    teacher_forcing_prob=tf_prob,
                    norm_mode=args.norm_mode,
                    norm_stats=fold_norm_stats,
                    stride=args.ar_stride,
                )
            else:
                train_metrics = run_epoch(
                    model,
                    train_loader,
                    edge_index,
                    args.dt,
                    optimizer,
                    args.lambda_mse,
                    args.lambda_mae,
                    args.grad_clip,
                    f"fold {fold_idx + 1} train {epoch}/{args.epochs}",
                    args.forecast_mode,
                    args.ar_chunk_size,
                    scaler=scaler,
                    norm_stats=fold_norm_stats,
                )
            _require_finite_metrics(train_metrics, f"fold {fold_idx + 1} epoch {epoch} train")

            val_metrics = None
            if epoch % args.val_every == 0 or epoch == args.epochs:
                with torch.no_grad():
                    if args.forecast_mode == "long_ar_train":
                        val_metrics = run_epoch_ar_train(
                            model,
                            val_run_loader,
                            edge_index,
                            args.dt,
                            args.x,
                            args.ar_chunk_size,
                            args.tbptt_chunks,
                            None,
                            args.lambda_mse,
                            args.lambda_mae,
                            args.grad_clip,
                            f"fold {fold_idx + 1} val {epoch}/{args.epochs}",
                            teacher_forcing_prob=0.0,
                            norm_mode=args.norm_mode,
                            norm_stats=fold_norm_stats,
                            stride=args.ar_stride,
                        )
                    else:
                        val_metrics = run_epoch(
                            model,
                            val_loader,
                            edge_index,
                            args.dt,
                            None,
                            args.lambda_mse,
                            args.lambda_mae,
                            args.grad_clip,
                            f"fold {fold_idx + 1} val {epoch}/{args.epochs}",
                            args.forecast_mode,
                            args.ar_chunk_size,
                            norm_stats=fold_norm_stats,
                                )
                _require_finite_metrics(val_metrics, f"fold {fold_idx + 1} epoch {epoch} val")
                scheduler.step(val_metrics["total"])
                if val_metrics["total"] < best_val:
                    best_val = val_metrics["total"]
                    best_val_metrics = val_metrics
                    epochs_without_improve = 0
                    checkpoint_config = vars(args).copy()
                    # Annotations the analysis tooling reads off the checkpoint
                    # to rebuild a config. Mirrors main.py: both are implied by
                    # make_model and neither is an argparse flag.
                    checkpoint_config["sheaf_mlp_maps"] = True
                    checkpoint_config["neural_ode"] = "sinusoidal_time"
                    torch.save(
                        {
                            "model": args.model,
                            "model_state_dict": model.state_dict(),
                            "config": checkpoint_config,
                            "fold": fold_idx + 1,
                            "best_val_total": best_val,
                            "best_val_metrics": best_val_metrics,
                            "edge_index": edge_index_cpu,
                            "graph_info": graph_info,
                            "graph_score": graph_score.detach().cpu()
                            if torch.is_tensor(graph_score)
                            else graph_score,
                            "norm_stats": fold_norm_stats,
                        },
                        fold_save_path,
                    )
                else:
                    epochs_without_improve += 1
            torch.save(
                {
                    "model": args.model,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "best_val": best_val,
                    "best_val_metrics": best_val_metrics,
                    "epochs_without_improve": epochs_without_improve,
                },
                resume_path,
            )
            # A dead sheaf is otherwise silent: the loss curve looks normal while
            # no gradient reaches the restriction maps. Returns "" for a
            # non-BrainDyn model or if anything in the probe raises.
            graph_msg = graph_health_line(
                model, graph_probe, edge_index, fold_norm_stats
            )
            print(
                f"Shuffle {fold_idx + 1}/{num_shuffles} Epoch {epoch:03d} | "
                f"train total={train_metrics['total']:.6f} mse={train_metrics['mse']:.6f} "
                f"val total={(val_metrics or {'total': float('nan')})['total']:.6f} "
                f"t={time.perf_counter() - t0:.1f}s{graph_msg}"
            )
            if args.patience is not None and epochs_without_improve >= args.patience:
                print(
                    f"Shuffle {fold_idx + 1}/{num_shuffles}: stopping early at epoch "
                    f"{epoch}/{args.epochs} -- {epochs_without_improve} val checks "
                    f"with no improvement (--patience {args.patience})."
                )
                break

        if resume_path.exists():
            resume_path.unlink()
        ckpt = torch.load(fold_save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            if args.forecast_mode == "long_ar_train":
                test_metrics = run_test_rollout_chunks(
                    model,
                    test_loader,
                    edge_index,
                    args.dt,
                    args.ar_chunk_size,
                    args.x,
                    args.test_rollout_steps,
                    args.lambda_mse,
                    args.lambda_mae,
                    f"fold {fold_idx + 1} test-rollout",
                    run_loader,
                    args.norm_mode,
                    fold_norm_stats,
                )
            else:
                test_metrics = run_epoch(
                    model,
                    test_loader,
                    edge_index,
                    args.dt,
                    None,
                    args.lambda_mse,
                    args.lambda_mae,
                    args.grad_clip,
                    f"fold {fold_idx + 1} test",
                    args.forecast_mode,
                    args.ar_chunk_size,
                    norm_stats=fold_norm_stats,
                )
        fold_tests.append(test_metrics)
        print(
            f"Fold {fold_idx + 1} Test | total={test_metrics['total']:.6f} "
            f"mse={test_metrics['mse']:.6f} mae={test_metrics['mae']:.6f} "
            f"pcc={test_metrics['pcc']:.4f} scc={test_metrics['scc']:.4f} "
            f"dtw={test_metrics['dtw']:.4f}"
        )
        
        if run_perturbed_eval:
            with torch.no_grad():
                if args.forecast_mode == "long_ar_train":
                    # Same rollout mechanics, run a second time against the
                    # perturbed trace for both context and ground truth --
                    # perturbed_run_loader stands in for run_loader.
                    test_metrics_perturbed = run_test_rollout_chunks(
                        model,
                        test_loader,
                        edge_index,
                        args.dt,
                        args.ar_chunk_size,
                        args.x,
                        args.test_rollout_steps,
                        args.lambda_mse,
                        args.lambda_mae,
                        f"fold {fold_idx + 1} test-rollout-perturbed",
                        perturbed_run_loader,
                        args.norm_mode,
                        fold_norm_stats,
                    )
                else:
                    test_metrics_perturbed = run_epoch(
                        model,
                        test_loader,
                        edge_index,
                        args.dt,
                        None,
                        args.lambda_mse,
                        args.lambda_mae,
                        args.grad_clip,
                        f"fold {fold_idx + 1} test-perturbed",
                        args.forecast_mode,
                        args.ar_chunk_size,
                        norm_stats=fold_norm_stats,
                        batch_transform=_batch_to_model_tensors_perturbed,
                        )
            fold_tests_perturbed.append(test_metrics_perturbed)
            print(
                f"Fold {fold_idx + 1} Test (perturbed) | "
                f"total={test_metrics_perturbed['total']:.6f} "
                f"mse={test_metrics_perturbed['mse']:.6f} "
                f"mae={test_metrics_perturbed['mae']:.6f} "
                f"pcc={test_metrics_perturbed['pcc']:.4f} "
                f"scc={test_metrics_perturbed['scc']:.4f} "
                f"dtw={test_metrics_perturbed['dtw']:.4f}"
            )

            with torch.no_grad():
                if args.forecast_mode == "long_ar_train":
                    test_metrics_excl = run_test_rollout_chunks(
                        model, test_loader, edge_index, args.dt,
                        args.ar_chunk_size, args.x, args.test_rollout_steps,
                        args.lambda_mse, args.lambda_mae,
                        f"fold {fold_idx + 1} test-rollout-excl",
                        run_loader, args.norm_mode, fold_norm_stats,
                        channel_mask_fn=path_channel_mask,
                    )
                    test_metrics_perturbed_excl = run_test_rollout_chunks(
                        model, test_loader, edge_index, args.dt,
                        args.ar_chunk_size, args.x, args.test_rollout_steps,
                        args.lambda_mse, args.lambda_mae,
                        f"fold {fold_idx + 1} test-rollout-perturbed-excl",
                        perturbed_run_loader, args.norm_mode, fold_norm_stats,
                        channel_mask_fn=path_channel_mask,
                    )
                else:
                    test_metrics_excl = run_epoch(
                        model, test_loader, edge_index, args.dt, None,
                        args.lambda_mse, args.lambda_mae, args.grad_clip,
                        f"fold {fold_idx + 1} test-excl",
                        args.forecast_mode, args.ar_chunk_size,
                        norm_stats=fold_norm_stats,
                        channel_mask_fn=batch_channel_mask,
                        )
                    test_metrics_perturbed_excl = run_epoch(
                        model, test_loader, edge_index, args.dt, None,
                        args.lambda_mse, args.lambda_mae, args.grad_clip,
                        f"fold {fold_idx + 1} test-perturbed-excl",
                        args.forecast_mode, args.ar_chunk_size,
                        norm_stats=fold_norm_stats,
                        batch_transform=_batch_to_model_tensors_perturbed,
                        channel_mask_fn=batch_channel_mask,
                        )
            fold_tests_excl.append(test_metrics_excl)
            fold_tests_perturbed_excl.append(test_metrics_perturbed_excl)
            print(
                f"Fold {fold_idx + 1} Test (target excluded) | "
                f"mse={test_metrics_excl['mse']:.6f} -> "
                f"perturbed mse={test_metrics_perturbed_excl['mse']:.6f}"
            )

    keys = ["total", "mse", "mae", "pcc", "scc", "dtw"]
    print("\n=== Benchmark Summary ===")
    for key in keys:
        vals = np.array([m[key] for m in fold_tests], dtype=float)
        print(f"{key}: mean={np.nanmean(vals):.6f} std={np.nanstd(vals):.6f}")

    if run_perturbed_eval:
        # The leading "\n" on each header is load-bearing: the blank line it produces is
        # what stops summarize_benchmark_results.py's greedy body regex from running
        # past the end of one block and swallowing the next block's header.
        for title, folds in (
            ("(perturbed horizon)", fold_tests_perturbed),
            ("(target excluded)", fold_tests_excl),
            ("(perturbed horizon, target excluded)", fold_tests_perturbed_excl),
        ):
            print(f"\n=== Benchmark Summary {title} ===")
            for key in keys:
                vals = np.array([m[key] for m in folds], dtype=float)
                print(f"{key}: mean={np.nanmean(vals):.6f} std={np.nanstd(vals):.6f}")


if __name__ == "__main__":
    main()
