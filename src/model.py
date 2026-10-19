"""
The two neural networks we compare. Both take the same input and give the same
output, so the comparison is fair:

    input:  (batch, 168 hours, 10 features)
    output: (batch, 24)   the predicted demand for the next 24 hours (scaled)

    python src/model.py   # quick check: push fake data through both models
"""

import math

import torch
import torch.nn as nn

NUM_FEATURES = 10
SEQ_LEN = 168
FORECAST_HORIZON = 24


class LSTMBaseline(nn.Module):
    """
    Reads the 168 hours one by one, left to right, keeping a running "memory".
    After the last hour, that memory is turned into 24 predictions.
    """

    def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,   # input shape is (batch, hours, features)
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, FORECAST_HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)        # memory after every hour: (batch, 168, 128)
        last = out[:, -1, :]         # memory after the LAST hour (it has read everything)
        return self.head(last)       # (batch, 24)


class PositionalEncoding(nn.Module):
    """
    A Transformer looks at all 168 hours at once, so it has no sense of order.
    This adds a unique wave pattern to each position, so it can tell hour 1
    from hour 100.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()  # 0, 1, 2, ...
        # different positions get different wave speeds (fast waves, slow waves)
        speed = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * speed)
        pe[:, 1::2] = torch.cos(position * speed)

        # a buffer is saved with the model but is not learned
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TemporalTransformer(nn.Module):
    """
    10 features -> a wider internal size -> add positions -> attention layers
    -> take the last hour -> 24 predictions.
    """

    def __init__(self, input_dim: int = NUM_FEATURES, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)  # 10 numbers -> 64 numbers
        self.positions = PositionalEncoding(d_model, dropout=dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,                       # 4 "attention heads", each looks for its own pattern
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, FORECAST_HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)   # (batch, 168, 64)
        x = self.positions(x)
        x = self.encoder(x)            # every hour now carries info from all other hours
        x = x[:, -1, :]                # the last hour: it has "looked at" the whole week
        return self.head(x)            # (batch, 24)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    fake = torch.randn(64, SEQ_LEN, NUM_FEATURES)
    print("input:", tuple(fake.shape))
    for name, model in (("LSTM", LSTMBaseline()), ("Transformer", TemporalTransformer())):
        out = model(fake)
        print(f"{name:<12} output {tuple(out.shape)} | {count_parameters(model):,} learnable numbers")


if __name__ == "__main__":
    main()
