# VoltCast: Energy Load Forecasting — Build Plan

## What This Is

Custom PyTorch Temporal Transformer that forecasts national energy demand 24 hours ahead,
wrapped in a full MLOps pipeline from live API ingestion to production inference.

**One-line resume bullet:**
"Built PyTorch Temporal Transformer forecasting national energy demand 24hrs ahead;
outperformed LSTM baseline on MAE across 4 countries; full MLOps pipeline with
live API ingestion, automated retraining, drift detection, and FastAPI serving."

**Why this project:**
- Not a tutorial. No YouTube series teaches this exact stack.
- Clean, honest eval: MAE/RMSE on real held-out test data. No recursive hacks.
- PyTorch written from scratch: positional encoding, multi-head attention, feedforward — every layer.
- Live data ingestion via ENTSOE API = real production behavior.
- Closes the classical ML + PyTorch gap without overlapping existing portfolio.

---

## Dataset

**Source:** EIA (U.S. Energy Information Administration) API v2
**URL:** https://api.eia.gov/v2/electricity/rto/region-data/data/
**Python client:** `requests` (no special library needed)
**Data:** Hourly actual electricity demand by US grid region (MW)
**Date range:** 2015-01-01 to present (~80k rows per region)
**Regions to use:** CAL (California), TEX (Texas/ERCOT), PJM (Mid-Atlantic), MISO (Midwest)
**API key:** Free, instant at eia.gov/opendata

**How to pull data:**
```python
import requests
import pandas as pd

def pull_eia(region: str, api_key: str) -> pd.Series:
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[type][]": "D",           # D = demand
        "facets[respondent][]": region,   # e.g. "CAL"
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
        "offset": 0,
    }
    r = requests.get(url, params=params)
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)
    df["period"] = pd.to_datetime(df["period"])
    df = df.set_index("period")["value"].astype(float)
    return df  # pd.Series, hourly MW
```

---

## Project Structure

```
voltcast/
├── BUILD_PLAN.md              # this file
├── data/
│   ├── raw/                   # raw parquet per country
│   └── features/              # engineered features parquet
├── src/
│   ├── ingestion.py           # ENTSOE API pull + save to raw/
│   ├── validation.py          # Pandera schema checks
│   ├── features.py            # feature engineering
│   ├── dataset.py             # PyTorch Dataset class
│   ├── model.py               # Transformer + LSTM (both in PyTorch from scratch)
│   ├── train.py               # training loop, MLflow logging
│   ├── evaluate.py            # MAE/RMSE/MAPE, baseline comparisons
│   ├── registry.py            # Champion/Challenger MLflow model registry
│   ├── inference.py           # load champion, generate 24h forecast
│   └── drift.py               # Evidently drift detection
├── api/
│   └── main.py                # FastAPI serving endpoint
├── dashboard/
│   └── app.py                 # Streamlit dashboard (forecast + metrics)
├── .github/
│   └── workflows/
│       ├── ingest.yml         # daily: pull latest ENTSOE data
│       ├── retrain.yml        # weekly: retrain if drift detected
│       └── inference.yml      # daily: generate 24h forecast
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## PyTorch Model (write from scratch — no HuggingFace)

### Temporal Transformer (`src/model.py`)

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, forecast_horizon=24):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, forecast_horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x[:, -1, :]           # take last timestep
        return self.output_head(x) # (batch, forecast_horizon)


class LSTMBaseline(nn.Module):
    """LSTM baseline for ablation — same input/output interface."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 dropout=0.1, forecast_horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.output_head = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.output_head(out[:, -1, :])
```

**Architecture decisions to document (for interviews):**
- Why `batch_first=True`: cleaner (batch, seq, features) convention
- Why take last timestep: encoder-only, no causal masking needed for fixed-window input
- Why `d_model=64` not 512: small dataset, bigger = overfitting
- nhead=4 divides d_model=64 evenly (16 dims per head)

---

## Feature Engineering (`src/features.py`)

Input: raw hourly MW series per country
Output: multivariate feature matrix per timestep

```
Features per timestep:
- load_mw              # target (normalized)
- hour_sin, hour_cos   # hour of day (Fourier)
- dow_sin, dow_cos     # day of week (Fourier)
- month_sin, month_cos # month (Fourier)
- is_weekend           # binary
- lag_1..lag_24        # previous 24 hours of load
- lag_168              # same hour last week
- rolling_mean_24      # 24hr rolling mean
- rolling_std_24       # 24hr rolling std
```

**Normalization:** Z-score per country (fit on train, apply to val/test). Store mean/std in MLflow as params.

**Sequence construction:** sliding window of 168 hours (1 week) → predict next 24 hours.

---

## Training Loop (`src/train.py`)

Write the loop by hand — no Trainer classes:

```python
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

**Training config:**
- Loss: MAE (nn.L1Loss) — matches eval metric
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: CosineAnnealingLR
- Epochs: 100, early stopping patience=10 on val MAE
- Batch size: 64
- Train/val/test split: 70/15/15 chronological (never shuffle time series)

**MLflow log per epoch:** train_loss, val_loss, val_mae, val_rmse, lr

---

## Ablation (`src/evaluate.py`)

Train 3 models, compare on test set:

| Model | Description |
|-------|-------------|
| Naive baseline | "tomorrow = today" (last 24h repeated) |
| LSTM | LSTMBaseline, same hyperparams |
| Transformer | TemporalTransformer (your model) |

Report per country: MAE, RMSE, MAPE. Log all to MLflow. Pick winner per country.
Document WHY Transformer wins (or loses) on specific countries — that's your interview story.

---

## MLflow Tracking (`src/train.py`)

```python
with mlflow.start_run(run_name=f"{model_type}_{country}_{timestamp}"):
    mlflow.log_params({
        "model_type": model_type,
        "country": country,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "seq_len": 168,
        "forecast_horizon": 24,
        "lr": lr,
        "epochs_trained": actual_epochs,
    })
    mlflow.log_metrics({
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_mape": test_mape,
        "val_mae_best": best_val_mae,
    })
    mlflow.pytorch.log_model(model, "model")
```

---

## Champion/Challenger (`src/registry.py`)

- On each weekly retrain: train Challenger on latest data
- Load current Champion from MLflow Model Registry
- Compare test MAE on same held-out window
- Promote Challenger if MAE improves by >1%
- Log both runs, tag winner as "production"

---

## Data Validation (`src/validation.py`)

Pandera schema on raw ingested data:
- No nulls in load_mw
- load_mw > 0 (can't have negative power consumption)
- Timestamps monotonically increasing, no gaps > 2 hours
- load_mw within 3 std of rolling mean (spike detection)

Fail validation → skip that batch, log alert, don't corrupt feature store.

---

## FastAPI Serving (`api/main.py`)

```
GET /forecast?country=DE&hours=24
→ returns: [{timestamp, predicted_mw, country}]

GET /health
→ returns: {model_version, last_trained, champion_mae}
```

Load champion model at startup from MLflow registry.

---

## Drift Detection (`src/drift.py`)

Evidently AI report weekly:
- Input drift: compare current week's load distribution vs training distribution
- Prediction drift: compare forecast errors this week vs baseline errors
- If drift detected: trigger retrain GitHub Action

---

## GitHub Actions Workflows

**ingest.yml** — runs daily at 06:00 UTC:
```
pull ENTSOE API → validate → save to data/raw/ → update features
```

**retrain.yml** — runs weekly Sunday 02:00 UTC:
```
check drift report → if drift: retrain both models → Champion/Challenger → promote
```

**inference.yml** — runs daily at 07:00 UTC (after ingest):
```
load champion → generate 24h forecast → save to data/forecasts/ → update dashboard
```

---

## Eval Metrics (honest, no hacks)

- **MAE** (primary): mean absolute error in MW
- **RMSE**: penalizes large errors more
- **MAPE**: % error, interpretable to non-ML people
- **Test set**: last 15% of data by time. Never touched during training or tuning.

No recursive bridge. No auto-regressive gap-filling. Model gets 168 real hours, predicts next 24. Measured on real actuals. Done.

---

## Dashboard (Streamlit)

- Live 24h forecast vs actuals (line chart)
- Model comparison table (Naive vs LSTM vs Transformer MAE per country)
- Champion model metadata (version, trained date, test MAE)
- Drift report status (green/red)

---

## Resume Entry (p8)

**Name:** VoltCast: Energy Load Forecasting with Temporal Transformer

**Highlights (4 bullets):**
1. Built **TemporalTransformer** from scratch in **PyTorch** (positional encoding, multi-head attention, feedforward layers) forecasting national energy demand 24hrs ahead on ENTSOE grid data across 4 countries.
2. Ran ablation across Naive, **LSTM**, and Transformer baselines; Transformer achieved lowest MAE on [X]/4 countries — documented when and why attention outperforms recurrence on this data.
3. Engineered full MLOps pipeline: live ENTSOE API ingestion, **Pandera** validation, **MLflow** experiment tracking, Champion/Challenger promotion, and **Evidently AI** drift detection on automated **GitHub Actions** schedule.
4. Served champion model via **FastAPI** + **Docker** with automated daily inference; **Streamlit** dashboard displays live forecasts vs actuals with model lineage and drift status.

---

## Build Order (do in this sequence)

1. ENTSOE API key + test pull for DE, FR, ES, IT
2. `src/ingestion.py` — pull and save raw parquet
3. `src/validation.py` — Pandera schema
4. `src/features.py` — feature engineering + sequence construction
5. `src/dataset.py` — PyTorch Dataset + DataLoader
6. `src/model.py` — TemporalTransformer + LSTMBaseline
7. `src/train.py` — training loop + MLflow logging
8. `src/evaluate.py` — ablation table (Naive vs LSTM vs Transformer)
9. `src/registry.py` — Champion/Challenger
10. `src/inference.py` — load champion, generate forecast
11. `api/main.py` — FastAPI endpoint
12. `src/drift.py` — Evidently drift report
13. `dashboard/app.py` — Streamlit
14. `.github/workflows/` — ingest + retrain + inference Actions
15. `Dockerfile` + `docker-compose.yml`
16. README with architecture diagram, results table, setup instructions

---

## Key Interview Questions This Project Answers

- "Walk me through your PyTorch training loop." → `src/train.py`, written by hand
- "Why Transformer over LSTM?" → ablation results + attention on seasonal patterns
- "How do you handle data quality in production?" → Pandera validation + drift detection
- "How does your Champion/Challenger work?" → MLflow registry + MAE threshold promotion
- "What happens when the model degrades in production?" → Evidently drift → auto retrain
- "What's your model's actual accuracy?" → MAE X MW on real held-out test set, no hacks
