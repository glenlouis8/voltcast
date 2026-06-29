"""
Two from-scratch PyTorch models with identical I/O shapes for a fair comparison:
LSTMBaseline (recurrent) and TemporalTransformer (attention, our main model).

    Input:  (batch, seq_len=168, num_features=13)
    Output: (batch, forecast_horizon=24)
"""

import math
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    LSTM baseline: reads 168 hours left to right, predicts next 24 from the
    final hidden state. If the Transformer can't beat this, something's wrong.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_horizon: int = 24,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.output_head = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # last timestep's hidden state summarizes all 168 hours
        last_hidden = out[:, -1, :]
        return self.output_head(last_hidden)


class PositionalEncoding(nn.Module):
    """
    Adds position info to each timestep. A Transformer sees all 168 hours at
    once and has no inherent sense of order, so we add a unique sine/cosine
    pattern per position to let it tell hour 1 from hour 100.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # buffer, not a parameter: moves with the model but the optimizer ignores it
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Encoder-only Transformer for forecasting:
        13 -> d_model projection -> positional encoding -> N encoder layers
        -> take last timestep -> linear head -> 24 predictions.

    Encoder-only (no decoder) because we emit all 24 hours at once from a fixed
    input, not autoregressively. We read off the last timestep because after
    attention it has seen the whole sequence and is the most recent context.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,             # d_model must be divisible by nhead
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        forecast_horizon: int = 24,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output_head(x)


def main():
    """Push a random batch through both models to check output shapes."""
    batch_size, seq_len, num_features = 64, 168, 13
    fake_batch = torch.randn(batch_size, seq_len, num_features)
    print("Input shape:", fake_batch.shape)

    lstm = LSTMBaseline(input_dim=num_features)
    print(f"LSTM output: {lstm(fake_batch).shape}  (expect (64, 24))")
    print(f"LSTM params: {sum(p.numel() for p in lstm.parameters()):,}")

    transformer = TemporalTransformer(input_dim=num_features)
    print(f"Transformer output: {transformer(fake_batch).shape}  (expect (64, 24))")
    print(f"Transformer params: {sum(p.numel() for p in transformer.parameters()):,}")


if __name__ == "__main__":
    main()
