"""
src/model.py

Two models built from scratch in PyTorch:
    1. LSTMBaseline   — reads sequence left to right, simpler, older approach
    2. TemporalTransformer — attention-based, our main model

Both take identical input/output shapes so comparison is fair:
    Input:  (batch_size, seq_len=168, num_features=13)
    Output: (batch_size, forecast_horizon=24)

No HuggingFace. No pretrained weights. Every layer written by hand.
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# LSTM BASELINE
# ─────────────────────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """
    Simple LSTM that reads 168 hours left to right and predicts next 24.

    LSTM = Long Short-Term Memory. An older recurrent architecture (2014 era).
    It processes one timestep at a time, maintaining a "hidden state" — a
    small memory vector that summarizes everything seen so far.

    After reading all 168 hours, the final hidden state is fed into a
    linear layer to produce 24 predictions.

    We build this as a baseline. If our Transformer can't beat this,
    something is wrong with the Transformer.
    """

    def __init__(
        self,
        input_dim: int,        # number of input features (13)
        hidden_dim: int = 128, # size of LSTM's internal memory vector
        num_layers: int = 2,   # stack 2 LSTMs on top of each other
        dropout: float = 0.1,  # randomly zero out 10% of neurons during training
        forecast_horizon: int = 24,
    ):
        # super().__init__() calls nn.Module's constructor.
        # Required boilerplate for every PyTorch model class.
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # input shape: (batch, seq, features) not (seq, batch, features)
            dropout=dropout,   # applied between LSTM layers (not after last layer)
        )

        # Linear layer: maps hidden_dim → forecast_horizon
        # This is the "output head" — takes the final LSTM state and produces predictions
        self.output_head = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, 168, 13)

        forward() defines what happens when you pass data through the model.
        PyTorch calls this automatically — you never call forward() directly.
        You just do: predictions = model(x)
        """
        # out shape: (batch_size, 168, hidden_dim)
        # out contains the hidden state at every timestep
        # _ contains the final cell state — we don't need it
        out, _ = self.lstm(x)

        # We only want the hidden state at the LAST timestep (hour 168).
        # That single vector summarizes all 168 hours.
        # out[:, -1, :] means: all batches, last timestep, all hidden dims
        last_hidden = out[:, -1, :]  # shape: (batch_size, hidden_dim)

        # Map to 24 predictions
        return self.output_head(last_hidden)  # shape: (batch_size, 24)


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Adds position information to each timestep's embedding.

    Problem: Transformer processes all 168 hours simultaneously.
    It has no built-in sense of order — hour 1 and hour 100 look identical
    to it unless we tell it which is which.

    Solution: add a unique pattern (sine/cosine waves) to each position.
    Hour 1 gets pattern A, hour 2 gets pattern B, etc.
    Now the model can tell positions apart.

    This uses sine and cosine waves of different frequencies.
    Think of it like a unique radio frequency for each position.

    The math:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    pos = position in sequence (0 to 167)
    i   = dimension index
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the positional encoding matrix once at initialization.
        # Shape: (max_len, d_model) — one row per position, one column per dimension
        pe = torch.zeros(max_len, d_model)

        # position: column vector [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term: the frequency denominators
        # exp(log(10000) * -2i/d_model) = 10000^(-2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even indices → sine, odd indices → cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer: saves pe as part of the model but NOT as a trainable parameter.
        # It moves with the model (to GPU/MPS if needed) but doesn't get updated by optimizer.
        # unsqueeze(0) adds a batch dimension: shape becomes (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        Adds the positional encoding to x.
        pe[:, :x.size(1)] slices to the actual sequence length (168).
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Transformer encoder for time series forecasting.

    Architecture (in order):
        1. Input projection:   13 → d_model=64   (linear layer)
        2. Positional encoding: adds position info
        3. Transformer encoder: 2 layers of multi-head attention
        4. Take last timestep:  (batch, 168, 64) → (batch, 64)
        5. Output head:         64 → 24           (linear layer)

    Why encoder-only (no decoder)?
        Decoder is needed for autoregressive generation (like GPT generating
        one token at a time). We don't do that — we output all 24 hours at once
        from a fixed input. Encoder-only is simpler and faster for this task.

    Why take only the last timestep?
        The Transformer encoder processes all 168 hours in parallel.
        After attention, each timestep's representation has "seen" all other
        timesteps. The last timestep (hour 168) is the most recent — its
        representation summarizes the full context most naturally.
        We use it to make predictions, similar to how LSTM uses final hidden state.
    """

    def __init__(
        self,
        input_dim: int,             # number of input features (13)
        d_model: int = 64,          # internal dimension — all layers work in this space
        nhead: int = 4,             # number of attention heads (d_model must be divisible by nhead)
        num_layers: int = 2,        # how many Transformer encoder layers to stack
        dim_feedforward: int = 256, # size of the feedforward layer inside each encoder layer
        dropout: float = 0.1,
        forecast_horizon: int = 24,
    ):
        super().__init__()

        # Step 1: project input from 13 features → d_model=64
        # Why? Transformer internally works with d_model-sized vectors.
        # Every timestep must be converted to this fixed size first.
        self.input_projection = nn.Linear(input_dim, d_model)

        # Step 2: positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Step 3: Transformer encoder
        # TransformerEncoderLayer = one layer of: multi-head attention + feedforward + layer norm
        # TransformerEncoder = stack of num_layers such layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # input shape: (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Step 4+5: output head — maps d_model → forecast_horizon
        self.output_head = nn.Linear(d_model, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, 168, 13)
        """
        # Step 1: 13 → 64
        x = self.input_projection(x)   # (batch, 168, 64)

        # Step 2: add position info
        x = self.pos_encoding(x)       # (batch, 168, 64)

        # Step 3: attention — each hour attends to all other hours
        x = self.transformer(x)        # (batch, 168, 64)

        # Step 4: take last timestep only
        x = x[:, -1, :]               # (batch, 64)

        # Step 5: predict 24 hours
        return self.output_head(x)     # (batch, 24)


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Pass a fake batch through both models to verify shapes are correct.
    No real data needed — just random tensors of the right shape.
    """
    batch_size   = 64
    seq_len      = 168
    num_features = 13

    # torch.randn creates random numbers from a normal distribution.
    # Shape: (64 examples, 168 hours, 13 features)
    fake_batch = torch.randn(batch_size, seq_len, num_features)

    print("Input shape:", fake_batch.shape)
    print()

    # Test LSTM
    lstm = LSTMBaseline(input_dim=num_features)
    lstm_out = lstm(fake_batch)
    print(f"LSTMBaseline output: {lstm_out.shape}  ← should be (64, 24)")
    total_params = sum(p.numel() for p in lstm.parameters())
    print(f"LSTM parameters: {total_params:,}")
    print()

    # Test Transformer
    transformer = TemporalTransformer(input_dim=num_features)
    transformer_out = transformer(fake_batch)
    print(f"TemporalTransformer output: {transformer_out.shape}  ← should be (64, 24)")
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Transformer parameters: {total_params:,}")


if __name__ == "__main__":
    main()
