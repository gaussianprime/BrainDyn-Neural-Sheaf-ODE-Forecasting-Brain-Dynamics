from __future__ import annotations

import torch
import torch.nn as nn

from .temporal_encoder import TemporalEncoder
from .sheaf import GCNAggregator, SheafLaplacian
from .time_encoding_ode import SinusoidalTimeVariant


class BrainDynDynamics(nn.Module):
    """Computes dx/dt given a history window and current signal."""

    def __init__(
        self,
        signal_dim,
        hidden_dim,
        num_nodes,
        window_size,
        lstm_layers=1,
        lstm_dropout=0.0,
        map_hidden_dim=128,
        vf_hidden_dim=128,
        vf_layers=2,
        use_gcn=False,
        use_lstm_encoder=True,
        edge_specific_maps=False,
        sheaf_layers=1,
        diffusion_step=1.0,
        sheaf_mlp_maps=True,
        map_mlp_hidden_dim=64,
        identity_restriction_init=False,
        sheaf_map_pe="none",
        sheaf_map_pe_dim=8,
        sheaf_node_pe=None,
        frozen_identity=False,
        sheaf_norm="none",
        sheaf_map_scale="none",
        freeze_map_scale=False,
        coupling_block="none",
        learn_diffusion_gain=False,
        time_embed_dim=16,
        time_embed_max_period=16.0,
        time_rate_mode="fixed",
        time_learn_freqs=False,
        time_max_cycles_per_step=0.5,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.use_gcn = use_gcn
        self.use_lstm_encoder = use_lstm_encoder
        self.edge_specific_maps = edge_specific_maps
        self.sheaf_layers = sheaf_layers
        self.diffusion_step = diffusion_step

        if self.use_lstm_encoder:
            self.temporal_encoder = TemporalEncoder(
                input_dim=signal_dim,
                hidden_dim=hidden_dim,
                num_layers=lstm_layers,
                dropout=lstm_dropout,
                num_nodes=num_nodes,
            )
        else:
            self.no_lstm_proj = nn.Linear(signal_dim, hidden_dim)

        if self.use_gcn:
            self.graph_laplacian = GCNAggregator(
                hidden_dim=hidden_dim, num_nodes=num_nodes
            )
        else:
            self.graph_laplacian = SheafLaplacian(
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                map_hidden_dim=map_hidden_dim,
                edge_specific_maps=edge_specific_maps,
                sheaf_mlp_maps=sheaf_mlp_maps,
                map_mlp_hidden_dim=map_mlp_hidden_dim,
                identity_restriction_init=identity_restriction_init,
                sheaf_map_pe=sheaf_map_pe,
                sheaf_map_pe_dim=sheaf_map_pe_dim,
                sheaf_node_pe=sheaf_node_pe,
                frozen_identity=frozen_identity,
                sheaf_norm=sheaf_norm,
                sheaf_map_scale=sheaf_map_scale,
                freeze_map_scale=freeze_map_scale,
                coupling_block=coupling_block,
                learn_diffusion_gain=learn_diffusion_gain,
            )

        self.ode = SinusoidalTimeVariant(
            signal_dim,
            hidden_dim,
            vf_hidden_dim,
            time_embed_dim=time_embed_dim,
            time_embed_max_period=time_embed_max_period,
            rate=None,
            rate_mode=time_rate_mode,
            num_nodes=num_nodes,
            learn_freqs=time_learn_freqs,
            max_cycles_per_step=time_max_cycles_per_step,
            vf_layers=vf_layers,
        )

    def register_restriction_edges(self, edge_index: torch.Tensor) -> None:
        """Register the edge universe for edge-specific restriction maps."""
        if not self.use_gcn and hasattr(self.graph_laplacian, "register_edges"):
            self.graph_laplacian.register_edges(edge_index)

    def compute_sheaf_h(
        self,
        x_hist: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the history-dependent sheaf-diffused encoding."""
        if self.use_lstm_encoder:
            h_t = self.temporal_encoder(x_hist)  # (B, N, H)
        else:
            h_t = torch.tanh(self.no_lstm_proj(x_hist[:, :, -1, :]))
        if not self.use_gcn:
            sheaf_h, graph_aux = self.graph_laplacian.diffuse(
                h_t, edge_index, self.sheaf_layers, self.diffusion_step
            )
        else:
            sheaf_h, graph_aux = self.graph_laplacian(h_t, edge_index)  # (B, N, H)
        return sheaf_h, {"h_t": h_t, **graph_aux}
