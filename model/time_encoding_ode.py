"""Traditional sinusoidal time encoding, concatenated into a non-autonomous ODE."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of a scalar time: ``[sin(2*pi*w_k*t), cos(2*pi*w_k*t)]``."""

    def __init__(
        self,
        dim: int,
        max_period: float = 10000.0,
        learn_freqs: bool = False,
        max_cycles_per_step: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.learn_freqs = bool(learn_freqs)
        self.max_cycles_per_step = float(max_cycles_per_step)
        half = dim // 2
        if self.learn_freqs:
            # Init at the fixed schedule: convert it to cycles/step, then invert
            # w = max_cycles*sigmoid(raw) so training starts at the fixed model.
            rad = torch.exp(
                -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
            )
            cycles = (rad / (2.0 * math.pi)).clamp(
                1e-6, self.max_cycles_per_step - 1e-6
            )
            frac = cycles / self.max_cycles_per_step
            self.raw_omega = nn.Parameter(torch.log(frac / (1.0 - frac)))

    def frequencies(self, device=None, dtype=None) -> torch.Tensor:
        """Angular frequencies (radians per step), shape ``(dim // 2,)``."""
        if self.learn_freqs:
            w = self.max_cycles_per_step * torch.sigmoid(self.raw_omega)
            return 2.0 * math.pi * w
        half = self.dim // 2
        return torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=device, dtype=dtype)
            / half
        )

    def cycles_per_step(self) -> torch.Tensor:
        """The same frequencies in cycles/step."""
        return self.frequencies() / (2.0 * math.pi)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = self.frequencies(device=t.device, dtype=t.dtype).to(t.dtype)
        args = t[..., None] * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class SinusoidalTimeVariant(nn.Module):
    """dx_i/dt = RATE * tanh(MLP([x_i || h^(L)_i || TimeEmbedding(t)]))."""

    RATE = 2.0  # upper bound
    RATE_MODES = ("fixed", "global", "per_node")

    def __init__(
        self,
        signal_dim,
        hidden_dim,
        vf_hidden_dim,
        time_embed_dim=16,
        time_embed_max_period=16.0,
        rate=None,
        rate_mode="fixed",
        num_nodes=None,
        learn_freqs=False,
        max_cycles_per_step=0.5,
        vf_layers=2,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        self.state_dim = signal_dim  # the state is the signal -- no latent
        self.time_embed_dim = int(time_embed_dim)
        if self.time_embed_dim < 0:
            raise ValueError(f"time_embed_dim must be >= 0, got {time_embed_dim}")
        # dim == 0 is the no time encoding ablation
        self.no_time_encoding = self.time_embed_dim == 0
        self.time_embed = TimeEmbedding(
            self.time_embed_dim,
            max_period=time_embed_max_period,
            learn_freqs=learn_freqs,
            max_cycles_per_step=max_cycles_per_step,
        )
        self.rate = float(self.RATE if rate is None else rate)
        if self.rate <= 0:
            raise ValueError(f"rate must be > 0, got {self.rate}")

        # ---- Learned output amplitude ------------------------------------
        if rate_mode not in self.RATE_MODES:
            raise ValueError(
                f"rate_mode must be one of {self.RATE_MODES}, got {rate_mode!r}"
            )
        self.rate_mode = rate_mode
        if rate_mode == "global":
            self.raw_rate = nn.Parameter(torch.zeros(()))
        elif rate_mode == "per_node":
            if not num_nodes:
                raise ValueError("rate_mode='per_node' requires num_nodes")
            # (N, 1) broadcasts against dx of shape (B, N, signal_dim).
            self.raw_rate = nn.Parameter(torch.zeros(int(num_nodes), 1))

        # ---- Vector-field MLP --------------------------------------------
        if int(vf_layers) < 2:
            raise ValueError(f"vf_layers must be >= 2, got {vf_layers}")
        field_in = signal_dim + hidden_dim + self.time_embed_dim
        blocks: list[nn.Module] = [nn.Linear(field_in, vf_hidden_dim), nn.Tanh()]
        for _ in range(int(vf_layers) - 2):
            blocks += [nn.Linear(vf_hidden_dim, vf_hidden_dim), nn.Tanh()]
        blocks.append(nn.Linear(vf_hidden_dim, signal_dim))
        self.field = nn.Sequential(*blocks)

    def effective_rate(self):
        """``R``: scalar under 'fixed', else 2*rate*sigmoid(rho) (learned)."""
        if self.rate_mode == "fixed":
            return self.rate
        return 2.0 * self.rate * torch.sigmoid(self.raw_rate)

    def coupling_gain(self):
        """No coupling term exists for this variant -- always uncoupled."""
        return None

    def drive_graph_share(self, sheaf_h, h_t):
        return None

    @torch.no_grad()
    def field_saturation(self, x, sheaf_h, t=0.0):
        """How much of the |dx/dt| <= R budget the tanh is actually using."""
        t_scalar = torch.as_tensor(float(t), device=x.device, dtype=x.dtype)
        emb = self.time_embed(t_scalar).to(x.dtype)
        emb = emb.reshape(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
        a = torch.tanh(self.field(torch.cat([x, sheaf_h, emb], dim=-1))).abs()
        return {
            "mean": float(a.mean()),
            "max": float(a.max()),
            "frac_over_0.9": float((a > 0.9).float().mean()),
        }

    def init_state(self, x0, x_prev, h_t, dt):
        return x0

    def decode(self, z):
        return z

    def rhs(
        self,
        t,
        z,
        *,
        sheaf_h,
        h_t,
        chunk_index,
        chunk_size,
        dt,
        slow,
        abs_t0,
        coupling=None,
    ):
        # One embedding per solver stage, identical across the batch and every node. Broadcast to match z's leading dims before concatenating.
        t_scalar = (
            t.reshape(())
            if torch.is_tensor(t)
            else torch.tensor(float(t), device=z.device, dtype=z.dtype)
        )
        emb = self.time_embed(t_scalar)  # (time_embed_dim,)
        emb = emb.reshape(1, 1, -1).expand(z.shape[0], z.shape[1], -1)
        x_in = torch.cat([z, sheaf_h, emb], dim=-1)
        return self.effective_rate() * torch.tanh(self.field(x_in))
