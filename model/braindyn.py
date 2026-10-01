from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .dynamics import BrainDynDynamics


@dataclass
class BrainDynConfig:
    signal_dim: int
    hidden_dim: int
    num_nodes: int
    window_size: int
    lstm_layers: int = 1
    lstm_dropout: float = 0.0
    map_hidden_dim: int = 16
    vf_hidden_dim: int = 128
    vf_layers: int = 2
    use_gcn: bool = False
    use_lstm_encoder: bool = True
    edge_specific_maps: bool = False
    sheaf_layers: int = 1
    diffusion_step: float = 1.0
    sheaf_mlp_maps: bool = True
    map_mlp_hidden_dim: int = 64
    identity_restriction_init: bool = False
    sheaf_map_pe: str = "none"
    sheaf_map_pe_dim: int = 8
    frozen_identity: bool = False
    sheaf_norm: str = "none"
    learn_diffusion_gain: bool = False
    sheaf_map_scale: str = "none"
    freeze_map_scale: bool = False
    coupling_block: str = "none"
    time_embed_dim: int = 16
    time_embed_max_period: float = 16.0
    time_rate_mode: str = "fixed"
    time_learn_freqs: bool = False
    time_max_cycles_per_step: float = 0.5


class BrainDyn(nn.Module):
    """
    BrainDyn with ODE solver stepping.

    Per forward call, integrates over the full requested horizon from
    a fixed context window. Longer rollouts are handled outside this
    module by feeding predicted chunks back into context.
    """

    def __init__(self, config, node_pe=None, sheaf_node_pe=None) -> None:
        super().__init__()
        self.config = config

        self.dynamics = BrainDynDynamics(
            signal_dim=config.signal_dim,
            hidden_dim=config.hidden_dim,
            num_nodes=config.num_nodes,
            window_size=config.window_size,
            lstm_layers=config.lstm_layers,
            lstm_dropout=config.lstm_dropout,
            map_hidden_dim=config.map_hidden_dim,
            vf_hidden_dim=config.vf_hidden_dim,
            vf_layers=getattr(config, "vf_layers", 2),
            use_gcn=config.use_gcn,
            use_lstm_encoder=config.use_lstm_encoder,
            edge_specific_maps=getattr(config, "edge_specific_maps", False),
            sheaf_layers=getattr(config, "sheaf_layers", 1),
            diffusion_step=getattr(config, "diffusion_step", 1.0),
            sheaf_mlp_maps=getattr(config, "sheaf_mlp_maps", True),
            map_mlp_hidden_dim=getattr(config, "map_mlp_hidden_dim", 64),
            identity_restriction_init=getattr(
                config, "identity_restriction_init", False
            ),
            sheaf_map_pe=getattr(config, "sheaf_map_pe", "none"),
            sheaf_map_pe_dim=getattr(config, "sheaf_map_pe_dim", 8),
            sheaf_node_pe=sheaf_node_pe,
            frozen_identity=getattr(config, "frozen_identity", False),
            sheaf_norm=getattr(config, "sheaf_norm", "none"),
            sheaf_map_scale=getattr(config, "sheaf_map_scale", "none"),
            freeze_map_scale=getattr(config, "freeze_map_scale", False),
            coupling_block=getattr(config, "coupling_block", "none"),
            learn_diffusion_gain=getattr(config, "learn_diffusion_gain", False),
            time_embed_dim=getattr(config, "time_embed_dim", 16),
            time_embed_max_period=getattr(config, "time_embed_max_period", 16.0),
            time_rate_mode=getattr(config, "time_rate_mode", "fixed"),
            time_learn_freqs=getattr(config, "time_learn_freqs", False),
            time_max_cycles_per_step=getattr(config, "time_max_cycles_per_step", 0.5),
        )

    def regularization_loss(self):
        return None

    def register_restriction_edges(self, edge_index: torch.Tensor) -> None:
        """Register the edge universe for edge-specific restriction maps."""
        self.dynamics.register_restriction_edges(edge_index)

    def forward(
        self,
        x_history: torch.Tensor,
        edge_index: torch.Tensor,
        pred_steps: int,
        dt: float = 1.0,
        autoregressive: bool = False,
        return_aux: bool = False,
    ) -> dict[str, Any]:

        if pred_steps <= 0:
            raise ValueError(f"pred_steps must be positive, got {pred_steps}")

        if autoregressive:
            raise ValueError("BrainDyn.forward does not perform autoregression.")
        return self._forward_ode(
            x_history, edge_index, pred_steps, dt, return_aux=return_aux
        )

    def _forward_ode(
        self,
        x_history: torch.Tensor,
        edge_index: torch.Tensor,
        pred_steps: int,
        dt: float = 1.0,
        return_aux: bool = False,
    ) -> dict[str, Any]:
        """Integrate the time-encoded ODE (model/time_encoding_ode.py)."""
        x0 = x_history[:, :, -1, :]
        x_prev = x_history[:, :, -2, :] if x_history.shape[2] >= 2 else x0

        # Encoder + sheaf run under any ambient AMP autocast (fast); frozen after.
        _sheaf_h, _enc = self.dynamics.compute_sheaf_h(x_history, edge_index)
        _h_t = _enc["h_t"]
        _graph_aux = {k: v for k, v in _enc.items() if k != "h_t"}

        ode = self.dynamics.ode
        dev = x_history.device.type
        with torch.autocast(device_type=dev, enabled=False):
            sheaf_f = _sheaf_h.float()
            h_t_f = _h_t.float()
            x0f = x0.float()
            x_prevf = x_prev.float()

            z0 = ode.init_state(x0f, x_prevf, h_t_f, float(dt))

            def rhs(_t: torch.Tensor, z_eval: torch.Tensor) -> torch.Tensor:
                return ode.rhs(
                    _t,
                    z_eval,
                    sheaf_h=sheaf_f,
                    h_t=h_t_f,
                    chunk_index=0,
                    chunk_size=pred_steps,
                    dt=float(dt),
                    slow=None,
                    abs_t0=0.0,
                    coupling=None,
                )

            # Sample times. return_aux tags every predicted step with its time,
            # so this is built regardless of whether aux is requested.
            t_eval = torch.arange(
                pred_steps + 1, device=x_history.device, dtype=torch.float32
            ) * float(dt)
            z_traj = odeint(rhs, z0, t_eval, method="rk4")
            x_pred = ode.decode(z_traj[1:])
        out: dict[str, Any] = {"x_pred": x_pred}
        if return_aux:
            base_aux = {
                "h_t": _h_t,
                "sheaf_h": _sheaf_h,
                "lap_h": _sheaf_h,
                "encoder_mode": "lstm"
                if self.dynamics.use_lstm_encoder
                else "last-step-linear",
                **_graph_aux,
            }
            out["aux_seq"] = [
                {**base_aux, "ode_t": t, "dxdt": step}
                for t, step in zip(t_eval[1:], x_pred)
            ]
        return out
