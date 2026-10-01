from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError:  # pragma: no cover - runtime guard for optional dependency
    GCNConv = None


def union_edge_index(
    edge_indices: list[torch.Tensor], num_nodes: int | None = None
) -> torch.Tensor:
    """Combine several ``(2, E_k)`` edge_index tensors into unique pairs."""
    if len(edge_indices) == 0:
        raise ValueError("union_edge_index requires at least one edge_index")

    cat = torch.cat([e.to(torch.long) for e in edge_indices], dim=1)  # (2, sum_k E_k)
    if cat.shape[1] == 0:
        return cat
    if num_nodes is None:
        num_nodes = int(cat.max()) + 1

    keys = cat[0] * num_nodes + cat[1]
    uniq = torch.unique(keys, sorted=True)
    src = uniq // num_nodes
    dst = uniq % num_nodes
    return torch.stack([src, dst], dim=0).long()


class SheafLaplacian(nn.Module):
    """(L_F h)_v = sum_{u in N_v} rho_{v->e}^T (tilde{h}_{v->e} - tilde{h}_{u->e})"""

    def __init__(
        self,
        hidden_dim,
        num_nodes,
        map_hidden_dim=128,
        edge_specific_maps=False,
        sheaf_mlp_maps=False,
        map_mlp_hidden_dim=64,
        identity_restriction_init=False,
        sheaf_map_pe="none",
        sheaf_map_pe_dim=8,
        sheaf_node_pe=None,
        frozen_identity=False,
        sheaf_norm="none",
        learn_diffusion_gain=False,
        sheaf_map_scale="none",
        freeze_map_scale=False,
        coupling_block="none",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.map_hidden_dim = map_hidden_dim
        self.edge_specific_maps = edge_specific_maps
        # ---- Degree normalization ----
        if sheaf_norm not in ("none", "sym", "row"):
            raise ValueError(
                f"sheaf_norm must be one of none|sym|row, got {sheaf_norm!r}"
            )
        self.sheaf_norm = sheaf_norm
        # Degrees depend only on edge_index, which is fixed for a run, so cache
        self._deg_cache_key = None
        self._deg_cache = None
        # log-space so the gain is positive; zero-init => gain == 1.0 exactly,
        # i.e. training starts at the fixed-step model. See diffusion_gain().
        self.raw_diffusion_gain = (
            nn.Parameter(torch.zeros(())) if learn_diffusion_gain else None
        )
        # ---- Scale-free restriction maps----
        if sheaf_map_scale not in ("none", "norm", "orth"):
            raise ValueError(
                f"sheaf_map_scale must be one of none|norm|orth, got {sheaf_map_scale!r}"
            )
        if coupling_block not in ("none", "block", "complex"):
            raise ValueError(
                f"coupling_block must be one of none|block|complex, got {coupling_block!r}"
            )
        self.coupling_block = coupling_block
        self._mask_key = None
        self._mask_cache = None
        self.sheaf_map_scale = sheaf_map_scale
        self.raw_map_scale = (
            nn.Parameter(torch.zeros(())) if sheaf_map_scale != "none" else None
        )
        # ---- Freezing the scale ----
        self.freeze_map_scale = bool(freeze_map_scale)
        if self.raw_map_scale is not None and self.freeze_map_scale:
            self.raw_map_scale.requires_grad_(False)
        # identity_restriction_init=True seeds the restriction maps at the
        self.identity_restriction_init = identity_restriction_init
        self.sheaf_mlp_maps = sheaf_mlp_maps
        self.map_mlp_hidden_dim = map_mlp_hidden_dim

        # ---- Trivial-sheaf ablation ----
        self.frozen_identity = frozen_identity
        if frozen_identity:
            if sheaf_map_pe != "none":
                raise ValueError(
                    "frozen_identity sheaf has no restriction MLP, so map "
                    f"positional encoding is meaningless; got sheaf_map_pe={sheaf_map_pe!r}."
                )
            self.sheaf_mlp_maps = False
            self.edge_specific_maps = False
            self.sheaf_map_pe = "none"
            self.map_pe_dim = 0
            self.sheaf_map_scale = "none"
            self.raw_map_scale = None
            self.freeze_map_scale = False
            self.coupling_block = "none"
            return

        # ---- Map positional encoding ----
        self.sheaf_map_pe = sheaf_map_pe
        pe_dim = 0
        if sheaf_map_pe != "none":
            if not sheaf_mlp_maps:
                raise ValueError(
                    "sheaf_map_pe routes positional encoding into the restriction MLP, "
                )
            if sheaf_map_pe == "lappe":
                if sheaf_node_pe is None:
                    raise ValueError(
                        "sheaf_map_pe='lappe' requires sheaf_node_pe (the LapPE matrix)."
                    )
                pe = torch.as_tensor(sheaf_node_pe, dtype=torch.float32)
                if pe.ndim != 2 or pe.shape[0] != num_nodes:
                    raise ValueError(
                        f"sheaf_node_pe must be (num_nodes, pe_dim); got {tuple(pe.shape)}"
                    )
                # standardize to zero-mean/unit-variance before use
                pe = (pe - pe.mean()) / pe.std().clamp(min=1e-8)
                # Non-learnable structural buffer (generalizes to unseen edges/subjects).
                self.register_buffer("map_pe", pe.contiguous(), persistent=True)
                pe_dim = pe.shape[1]
            elif sheaf_map_pe == "learned":
                pe_dim = sheaf_map_pe_dim
                self.map_pe_emb = nn.Embedding(num_nodes, pe_dim)
            else:
                raise ValueError(
                    f"sheaf_map_pe must be one of none|lappe|learned, got {sheaf_map_pe!r}"
                )
        self.map_pe_dim = pe_dim

        if sheaf_mlp_maps:
            # shared sheaf-learner MLP: [h_self || h_neighbor (|| pe_self || pe_neighbor)]
            # -> flattened (hidden_dim x map_hidden_dim) restriction matrix
            self.restriction_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim + 2 * pe_dim, map_mlp_hidden_dim),
                nn.Tanh(),
                nn.Linear(map_mlp_hidden_dim, hidden_dim * map_hidden_dim),
            )
        elif edge_specific_maps:
            # per-edge tables are allocated lazily in register_edges() once the edge universe is known
            self._edges_registered = False
            self.rho_src_default = nn.Parameter(torch.empty(hidden_dim, map_hidden_dim))
            self.rho_dst_default = nn.Parameter(torch.empty(hidden_dim, map_hidden_dim))
        else:
            self.restriction_maps = nn.Parameter(
                torch.empty(num_nodes, hidden_dim, map_hidden_dim)
            )

        # Single source of truth for the initial map field; see the method docstring.
        self.reset_restriction_parameters()

    def reset_restriction_parameters(self) -> None:
        """(Re)apply the initial values of the restriction-map parameters.

        ``__init__`` calls this after allocating the map modules, so it is the one
        place that defines how the maps start: identity when
        ``identity_restriction_init``, otherwise the small ``gain=0.1`` readout
        (or Xavier for the static/edge-specific tables). Because the MLP maps are a
        *function* of the (LSTM) embeddings rather than stored parameters, there is
        no map tensor to initialize -- the "initialization" lives entirely in these
        weights. Analysis code can call this on a *trained* model to recover the
        true initial map field on the model's own embeddings, instead of standing it
        in with a Xavier-random population.
        """
        if self.frozen_identity:
            return
        if self.raw_map_scale is not None:
            with torch.no_grad():
                self.raw_map_scale.zero_()
        map_pe_emb = getattr(self, "map_pe_emb", None)
        if map_pe_emb is not None:
            map_pe_emb.reset_parameters()
        if self.sheaf_mlp_maps:
            nn.init.xavier_uniform_(self.restriction_mlp[0].weight)
            nn.init.zeros_(self.restriction_mlp[0].bias)
            if self.identity_restriction_init:
                nn.init.zeros_(self.restriction_mlp[2].weight)
                with torch.no_grad():
                    self.restriction_mlp[2].bias.copy_(
                        torch.eye(self.hidden_dim, self.map_hidden_dim).reshape(-1)
                    )
            else:
                # so the sheaf Laplacian starts well-conditioned
                nn.init.xavier_uniform_(self.restriction_mlp[2].weight, gain=0.1)
                nn.init.zeros_(self.restriction_mlp[2].bias)
        elif self.edge_specific_maps:
            self._init_restriction_(self.rho_src_default)
            self._init_restriction_(self.rho_dst_default)
            if getattr(self, "_edges_registered", False):
                self._init_restriction_(self.rho_src_table)
                self._init_restriction_(self.rho_dst_table)
        else:
            self._init_restriction_(self.restriction_maps)

    def _init_restriction_(self, tensor: torch.Tensor) -> None:
        """Initialize a ``(..., hidden_dim, map_hidden_dim)`` restriction table."""
        if self.identity_restriction_init:
            eye = torch.eye(
                self.hidden_dim,
                self.map_hidden_dim,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            with torch.no_grad():
                tensor.copy_(eye.expand_as(tensor))
        else:
            nn.init.xavier_uniform_(tensor)

    _NS_QUINTIC = (3.4445, -4.7750, 2.0315)
    _NS_QUINTIC_STEPS = 12
    _NS_CUBIC_STEPS = 4

    @classmethod
    def _polar(cls, m: torch.Tensor) -> torch.Tensor:
        """Nearest orthogonal matrix to ``m`` (``U V^T``), by Newton-Schulz."""
        in_dtype = m.dtype
        m = m.float()
        eps = torch.finfo(m.dtype).eps
        x = m / (m.norm(dim=(-2, -1), keepdim=True) + eps)
        wide = x.shape[-1] > x.shape[-2]
        if wide:  # iterate on the tall orientation
            x = x.transpose(-2, -1)
        a, b, c = cls._NS_QUINTIC
        for _ in range(cls._NS_QUINTIC_STEPS):
            g = x.transpose(-2, -1) @ x
            x = a * x + b * (x @ g) + c * (x @ (g @ g))
        for _ in range(cls._NS_CUBIC_STEPS):
            x = 1.5 * x - 0.5 * (x @ (x.transpose(-2, -1) @ x))
        if wide:
            x = x.transpose(-2, -1)
        return x.to(in_dtype)

    def _rescale_maps(
        self, rho: torch.Tensor, apply_block: bool = True
    ) -> torch.Tensor:
        """Split rho into a learned scale and a normalized direction."""
        if apply_block:
            rho = self._apply_block_structure(rho)
        if self.sheaf_map_scale == "none":
            return rho
        d, d_e = rho.shape[-2], rho.shape[-1]
        target = float(min(d, d_e)) ** 0.5
        if self.sheaf_map_scale == "orth":
            direction = self._polar(rho)
        else:
            eps = torch.finfo(rho.dtype).eps
            direction = rho / (rho.norm(dim=(-2, -1), keepdim=True) + eps) * target
        return torch.exp(self.raw_map_scale).to(rho.dtype) * direction

    def _apply_block_structure(self, rho: torch.Tensor) -> torch.Tensor:
        """Restrict rho to be block-diagonal, pairing stalk dimension s with s + d/2."""
        mode = getattr(self, "coupling_block", "none")
        if mode == "none":
            return rho
        d, d_e = rho.shape[-2], rho.shape[-1]
        if d != d_e or d % 2 != 0:
            raise ValueError(
                f"coupling_block={mode!r} needs a square, even-dimensional stalk; "
                f"got {d}x{d_e}."
            )
        S = d // 2
        if mode == "block":
            # Zero every entry outside the {s, S+s} x {s, S+s} blocks.
            return rho * self._block_mask(rho.device, rho.dtype, S)
        # complex-linear: keep only the rotation-scaling part of each 2x2 block
        idx_r = torch.arange(S, device=rho.device)
        idx_i = idx_r + S
        p = rho[..., idx_r, idx_r]
        q = rho[..., idx_r, idx_i]
        u = rho[..., idx_i, idx_r]
        v = rho[..., idx_i, idx_i]
        c = 0.5 * (p + v)
        sn = 0.5 * (u - q)
        out = torch.zeros_like(rho)
        out[..., idx_r, idx_r] = c
        out[..., idx_i, idx_i] = c
        out[..., idx_r, idx_i] = -sn
        out[..., idx_i, idx_r] = sn
        return out

    def _block_mask(self, device, dtype, S: int) -> torch.Tensor:
        """(2S, 2S) 0/1 mask selecting the per-mode {s, S+s} blocks. Cached."""
        key = (S, device, dtype)
        if getattr(self, "_mask_key", None) == key and self._mask_cache is not None:
            return self._mask_cache
        m = torch.zeros(2 * S, 2 * S, device=device, dtype=dtype)
        idx = torch.arange(S, device=device)
        for a in (idx, idx + S):
            for b in (idx, idx + S):
                m[a, b] = 1.0
        self._mask_key = key
        self._mask_cache = m
        return m

    def map_scale(self) -> float:
        """Current ``||rho||_F / sqrt(min(d, d_e))``; 1.0 at init."""
        if self.raw_map_scale is None:
            return 1.0
        return float(torch.exp(self.raw_map_scale))

    @staticmethod
    def _restrict(h_e: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Apply restriction map: ``tilde = h_e @ rho``."""
        eq = "bed,bedh->beh" if rho.ndim == 4 else "bed,edh->beh"
        return torch.einsum(eq, h_e, rho)

    @staticmethod
    def _pullback(delta: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Pull an edge-space signal back to node space: ``msg = delta @ rho^T``."""
        eq = "beh,bedh->bed" if rho.ndim == 4 else "beh,edh->bed"
        return torch.einsum(eq, delta, rho)

    def register_edges(self, edge_index: torch.Tensor) -> None:
        """Allocate per-edge restriction-map tables for the given edge universe."""
        if self.sheaf_mlp_maps or not self.edge_specific_maps:
            return
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
            )

        device = self.rho_src_default.device
        dtype = self.rho_src_default.dtype

        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)
        if src.numel() > 0 and (
            int(src.max()) >= self.num_nodes or int(dst.max()) >= self.num_nodes
        ):
            raise ValueError(
                f"edge_index references a node id >= num_nodes ({self.num_nodes})"
            )

        # Deterministic, compact slot ids for each unique ordered (src, dst) pair.
        pair_key = src * self.num_nodes + dst  # (E,)
        unique_keys = torch.unique(pair_key, sorted=True)  # (P,)
        num_pairs = int(unique_keys.numel())

        pair_slot = torch.full(
            (self.num_nodes, self.num_nodes), -1, dtype=torch.long, device=device
        )
        u_src = (unique_keys // self.num_nodes).to(torch.long)
        u_dst = (unique_keys % self.num_nodes).to(torch.long)
        pair_slot[u_src, u_dst] = torch.arange(
            num_pairs, device=device, dtype=torch.long
        )

        rho_src_table = torch.empty(
            num_pairs, self.hidden_dim, self.map_hidden_dim, device=device, dtype=dtype
        )
        rho_dst_table = torch.empty(
            num_pairs, self.hidden_dim, self.map_hidden_dim, device=device, dtype=dtype
        )
        self._init_restriction_(rho_src_table)
        self._init_restriction_(rho_dst_table)

        # (Re)register the buffer and parameters, overwriting any prior universe.
        if "pair_slot" in self._buffers:
            del self._buffers["pair_slot"]
        self.register_buffer("pair_slot", pair_slot, persistent=True)
        self.rho_src_table = nn.Parameter(rho_src_table)
        self.rho_dst_table = nn.Parameter(rho_dst_table)
        self.num_pairs = num_pairs
        self._edges_registered = True

    def _edge_pe(
        self, src: torch.Tensor, dst: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Per-edge map positional encodings ``(pe_src, pe_dst)`` or ``(None, None)``."""
        if self.sheaf_map_pe == "lappe":
            return self.map_pe[src], self.map_pe[dst]
        if self.sheaf_map_pe == "learned":
            return self.map_pe_emb(src), self.map_pe_emb(dst)
        return None, None

    def _gather_restriction_maps(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rho_src, rho_dst)``."""
        if self.sheaf_mlp_maps:
            B = h_src.shape[0]
            # Ordered concatenation makes the two endpoint maps asymmetric while
            # sharing one set of MLP weights (NSD sheaf-learner).
            pe_src, pe_dst = self._edge_pe(src, dst)
            if pe_src is not None:
                pe_src_b = pe_src.unsqueeze(0).expand(B, -1, -1)
                pe_dst_b = pe_dst.unsqueeze(0).expand(B, -1, -1)
                pair_sd = torch.cat([h_src, h_dst, pe_src_b, pe_dst_b], dim=-1)
                pair_ds = torch.cat([h_dst, h_src, pe_dst_b, pe_src_b], dim=-1)
            else:
                pair_sd = torch.cat([h_src, h_dst], dim=-1)  # (B, E, 2H)
                pair_ds = torch.cat([h_dst, h_src], dim=-1)  # (B, E, 2H)
            rho_src = self.restriction_mlp(pair_sd).reshape(
                B, -1, self.hidden_dim, self.map_hidden_dim
            )
            rho_dst = self.restriction_mlp(pair_ds).reshape(
                B, -1, self.hidden_dim, self.map_hidden_dim
            )
        elif not self.edge_specific_maps:
            rho_src = self.restriction_maps[src]
            rho_dst = self.restriction_maps[dst]
        else:
            if not getattr(self, "_edges_registered", False):
                raise RuntimeError(
                    "edge_specific_maps=True but the edge universe has not been "
                    "registered. Call model.register_restriction_edges(edge_index) "
                    "after constructing the model and before the optimizer."
                )

            slots = self.pair_slot[src, dst]  # (E,)
            valid = slots >= 0
            safe = slots.clamp(min=0)
            rho_src = self.rho_src_table[safe]  # (E, d, d_e)
            rho_dst = self.rho_dst_table[safe]
            if not bool(valid.all()):
                # Pairs unseen at registration fall back to shared default maps.
                inv = ~valid
                rho_src = rho_src.clone()
                rho_dst = rho_dst.clone()
                rho_src[inv] = self.rho_src_default
                rho_dst[inv] = self.rho_dst_default

        dense = (
            self._rescale_maps(rho_src, apply_block=False),
            self._rescale_maps(rho_dst, apply_block=False),
        )
        if getattr(self, "coupling_block", "none") == "none":
            return dense, dense
        coupling = (
            self._rescale_maps(rho_src, apply_block=True),
            self._rescale_maps(rho_dst, apply_block=True),
        )
        return dense, coupling

    def _node_scale(self, edge_index, device, dtype):
        """Per-node scale factor applied on each side of L (cached per graph)."""
        key = (int(edge_index.shape[1]), int(edge_index.data_ptr()))
        if self._deg_cache_key == key and self._deg_cache is not None:
            cached = self._deg_cache
            if cached.device == device and cached.dtype == dtype:
                return cached

        deg = torch.zeros(self.num_nodes, device=device, dtype=dtype)
        ones = torch.ones(edge_index.shape[1], device=device, dtype=dtype)
        deg.index_add_(0, edge_index[0], ones)
        deg.index_add_(0, edge_index[1], ones)

        if self.sheaf_norm == "sym":
            scale = deg.clamp(min=1e-12).pow(-0.5)
        elif self.sheaf_norm == "row":
            scale = deg.clamp(min=1e-12).reciprocal()
        else:
            scale = torch.ones_like(deg)
        scale = torch.where(deg > 0, scale, torch.zeros_like(scale))
        scale = scale.to(dtype)

        self._deg_cache_key = key
        self._deg_cache = scale
        return scale

    def forward(self, h, edge_index):
        """
        h: (B, N, D)
        edge_index: (2, E)

        returns:
            lap: (B, N, D)
            aux: dict
        """
        if h.ndim != 3:
            raise ValueError(f"h must have shape (B, N, D), got {tuple(h.shape)}")
        if h.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {h.shape[1]}")

        src = edge_index[0]
        dst = edge_index[1]

        if src.numel() == 0:
            return torch.zeros_like(h), {}

        # D^-1/2 L D^-1/2 factorizes as: scale the stalks going in, scale the aggregated result coming out
        h_in = h
        if self.sheaf_norm == "sym":
            scale = self._node_scale(edge_index, h.device, h.dtype)
            h_in = h * scale.view(1, -1, 1)

        h_src = h_in[:, src, :]
        h_dst = h_in[:, dst, :]

        if self.frozen_identity:
            # Trivial sheaf: identity restriction maps -> plain (unnormalized) graph Laplacian.
            delta = h_dst - h_src
            lap = torch.zeros_like(h)
            if delta.dtype != lap.dtype:
                delta = delta.to(lap.dtype)
            lap.index_add_(1, src, -delta)
            lap.index_add_(1, dst, delta)
            if self.sheaf_norm != "none":
                nscale = self._node_scale(edge_index, h.device, h.dtype)
                lap = lap * nscale.view(1, -1, 1)
            return lap.to(h.dtype), {"graph_mode": "frozen_identity", "delta": delta}

        (rho_src, rho_dst), (rho_src_c, rho_dst_c) = self._gather_restriction_maps(
            src, dst, h_src, h_dst
        )

        tilde_src = self._restrict(h_src, rho_src)
        tilde_dst = self._restrict(h_dst, rho_dst)

        # Linear message passing: unit endpoint weights, no rescale, so delta is
        # the plain difference of the restricted stalks (pure sheaf Laplacian).
        delta = tilde_dst - tilde_src
        msg_src = self._pullback(-delta, rho_src)
        msg_dst = self._pullback(delta, rho_dst)

        lap = torch.zeros_like(h)
        lap.index_add_(1, src, msg_src)
        lap.index_add_(1, dst, msg_dst)

        if self.sheaf_norm != "none":
            scale = self._node_scale(edge_index, h.device, h.dtype)
            lap = lap * scale.view(1, -1, 1)

        aux = {
            "rho_src": rho_src,
            "rho_dst": rho_dst,
            "rho_src_coupling": rho_src_c,
            "rho_dst_coupling": rho_dst_c,
            "tilde_src": tilde_src,
            "tilde_dst": tilde_dst,
            "delta": delta,
        }
        return lap.to(h.dtype), aux

    def diffuse(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        num_layers: int,
        step: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """L rounds of sheaf diffusion ``H^(l) = H^(l-1) - step * L_F H^(l-1)``."""
        rounds = int(num_layers)
        if rounds < 0:
            raise ValueError(f"num_layers must be >= 0, got {num_layers}")
        eff_step = step * self.diffusion_gain()
        aux: dict = {}
        for _ in range(rounds):
            lap, aux = self.forward(h, edge_index)
            h = h - eff_step * lap
        return h, aux

    def diffusion_gain(self):
        """Multiplier on ``step``; 1.0 unless ``learn_diffusion_gain``."""
        if self.raw_diffusion_gain is None:
            return 1.0
        return torch.exp(self.raw_diffusion_gain)

    @torch.jit.unused
    def apply_precomputed(self, v, edge_index, rho_src, rho_dst):
        """Apply ``L_F`` to an arbitrary node signal using already-computed maps."""
        src, dst = edge_index[0], edge_index[1]
        if src.numel() == 0:
            return torch.zeros_like(v)

        v_in = v
        if self.sheaf_norm == "sym":
            scale = self._node_scale(edge_index, v.device, v.dtype)
            v_in = v * scale.view(1, -1, 1)

        if rho_src is None or rho_dst is None:
            # Trivial sheaf (frozen_identity): maps are the identity, so the
            # operator is the plain graph Laplacian on v.
            delta = v_in[:, dst, :] - v_in[:, src, :]
            out = torch.zeros_like(v)
            out.index_add_(1, src, -delta)
            out.index_add_(1, dst, delta)
        else:
            tilde_src = self._restrict(v_in[:, src, :], rho_src)
            tilde_dst = self._restrict(v_in[:, dst, :], rho_dst)
            delta = tilde_dst - tilde_src
            out = torch.zeros_like(v)
            out.index_add_(1, src, self._pullback(-delta, rho_src))
            out.index_add_(1, dst, self._pullback(delta, rho_dst))

        if self.sheaf_norm != "none":
            scale = self._node_scale(edge_index, v.device, v.dtype)
            out = out * scale.view(1, -1, 1)
        return out


class GCNAggregator(nn.Module):
    """GCN neighborhood aggregation implemented with PyG GCNConv."""

    def __init__(self, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        if GCNConv is None:
            raise ImportError(
                "GCN ablation requires torch-geometric. Install it with: pip install torch-geometric"
            )
        self.gcn = GCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            add_self_loops=False,
            bias=True,
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        """
        h: (B, N, D)
        edge_index: (2, E)
        returns:
            agg: (B, N, D)
            aux: dict
        """
        if h.ndim != 3:
            raise ValueError(f"h must have shape (B, N, D), got {tuple(h.shape)}")
        if h.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {h.shape[1]}")

        batch_outputs = []
        for b in range(h.shape[0]):
            # PyG GCNConv expects node features of shape (N, D) per graph.
            out_b = self.gcn(h[b], edge_index)
            batch_outputs.append(out_b)
        agg = torch.stack(batch_outputs, dim=0)

        aux = {
            "graph_mode": "gcn",
        }
        return agg, aux
