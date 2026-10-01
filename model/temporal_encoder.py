from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Shared LSTM temporal-history encoder.

    Input:
        x_hist: (B, N, T, F)
    Output:
        h: (B, N, H)

    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.0,
        num_nodes=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def forward(self, x_hist):
        """
        x_hist: (B, N, T, F)
        returns: (B, N, H)
        """
        if x_hist.ndim != 4:
            raise ValueError(
                f"x_hist must have shape (B, N, T, F), got {tuple(x_hist.shape)}"
            )

        B, N, T, F = x_hist.shape
        if F != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {F}")

        x_flat = x_hist.reshape(B * N, T, x_hist.shape[-1])
        z_seq, _ = self.lstm(x_flat)
        z_seq = z_seq.reshape(B, N, T, self.hidden_dim)
        return z_seq[:, :, -1, :]
