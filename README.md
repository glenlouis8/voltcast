# ⚡ VoltCast — 24-Hour US Electricity Demand Forecasting

End-to-end MLOps system that predicts the next **24 hours** of electricity demand for four major US grid regions — **California (CAL)**, **Texas (ERCOT/TEX)**, **PJM (Mid-Atlantic)**, and **MISO (Midwest)** — using a Transformer built from scratch in PyTorch.

It is **serverless**: no always-on backend. Scheduled GitHub Actions pull fresh data, run the model, and publish forecasts. Models live in a hosted registry, forecasts in object storage, the frontend reads them directly.

---

## Why this project

Grid operators must match electricity supply to demand every hour. Too little → blackouts. Too much → wasted money. VoltCast forecasts demand 24 hours ahead so that decision has data behind it — the same problem real ISOs solve, packaged as a clean, automated ML system.

The goal was not just a model, but the **full production loop**: ingest → validate → train → evaluate → register → serve → monitor for drift → retrain. Every layer is built explicitly (no `Trainer`, no AutoML) so the engineering is visible.

---

## Results

Ablation on the **untouched test set** (latest 15% of data, never seen in training). Primary metric is **MAE in megawatts** — average absolute error per predicted hour.

### California (CAL)

| Model | Test MAE | MAPE | vs. Naive |
|-------|---------:|-----:|----------:|
| Naive (repeat last 24h) | 1,562 MW | 4.89% | baseline |
| LSTM (from scratch) | 2,502 MW | 7.68% | −60% (worse) |
| **Transformer (from scratch)** | **1,050 MW** | **3.24%** | **+32.8% better** |

The Transformer is the served **champion** in every region. The ablation is the proof: a strong naive baseline keeps the model honest, and the Transformer clearly beats it.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   GitHub Actions   │  pipeline.py  (build · forecast · retrain)    │
   (cron, no server)│                                              │
                    │  ingestion → validation → features →         │
                    │  train → registry → inference / drift        │
                    └───────────────┬──────────────────────────────┘
                                    │ writes
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        EIA API (data)      DagsHub (MLflow)          AWS S3
        re-pulled,          models + champion         forecasts + drift
        nothing stored      registry                  references (JSON)
                                                            │ reads
                                                            ▼
                                                   Vercel frontend
                                                   (fetches S3 JSON)
```

**Storage split — each tool does one job:**

| Thing | Home | Why |
|-------|------|-----|
| Code | GitHub | version control + CI |
| Models + champion alias | DagsHub (hosted MLflow) | versioning, registry, promotion |
| Forecasts + drift references | AWS S3 (parquet + JSON) | cheap files; CI writes, frontend reads |
| Raw data | nowhere — re-pulled from EIA each run | always fresh, rolling 5-year window |

---

## The Model

A **TemporalTransformer**, built layer by layer in PyTorch (`src/model.py`):

```
Input  (batch, 168 hours, 13 features)
  → Linear projection           13 → d_model=64
  → Positional encoding         (sin/cos, so order is known)
  → TransformerEncoder × 2      nhead=4, dim_feedforward=256, dropout=0.1
  → take last timestep
  → Linear output head          64 → 24
Output (batch, 24)              next 24 hours of MW
```

It is a **168 → 24 sequence-to-sequence** problem solved in **one shot** — the model sees 168 real hours (one week) and outputs all 24 future hours at once. No recursive feeding of predictions back as inputs (which compounds error).

An **LSTM baseline** (`hidden_dim=128`, 2 layers) and a **naive baseline** are trained/run alongside it for the ablation.

### Features (13)

- **Raw load** (`load_mw`)
- **Fourier time encodings** — sin/cos of hour, day-of-week, month (so hour 23 and hour 0 are "close")
- **Lags** — load 1h, 24h, and 168h (same hour last week) ago
- **Rolling stats** — 24h mean and std
- **is_weekend** flag

All features are z-score normalized with the scaler **fit on training data only** (no leakage).

---

## MLOps Highlights

- **From-scratch models** — every layer written by hand; raw training loop with AdamW, `CosineAnnealingLR`, gradient clipping, and early stopping. No HuggingFace, no `Trainer`.
- **Champion / Challenger registry** — `registry.py` evaluates both models on the test set, registers the winner in DagsHub, and only promotes a challenger that beats the champion by **>1%** test MAE.
- **Data validation** — Pandera contract (`validation.py`) rejects nulls, spikes, and gaps before any data reaches training.
- **Drift detection** — Evidently (`drift.py`) compares a fresh EIA pull against the champion's pinned training distribution (K-S test). The reference snapshot is saved at crowning time, so drift always measures against what the live model actually learned.
- **Hybrid retraining** — `pipeline.py retrain` retrains a region when **drift fires OR** the champion exceeds a 30-day age cap (catches slow shifts the statistical test misses).
- **Chronological splits** — 70/15/15 by time, never shuffled. The test set is the most recent data and is never touched during training or tuning.

---

## Automation (GitHub Actions)

| Workflow | Schedule | Command | Purpose |
|----------|----------|---------|---------|
| `forecast.yml` | hourly (`0 * * * *`) | `pipeline.py forecast` | rolling 24h forecast → S3 |
| `retrain.yml` | weekly (`0 2 * * 0`) | `pipeline.py retrain` | drift/staleness check → retrain if needed |
| `build.yml` | manual | `pipeline.py build` | full rebuild — train all, crown champions |

All run on free public-repo runners (CPU; the models are small). Cost ≈ **$0**.

---

## Project Structure

```
voltcast/
├── src/
│   ├── ingestion.py    # pull hourly demand from EIA (rolling 5-year window)
│   ├── validation.py   # Pandera data contract
│   ├── features.py     # Fourier / lag / rolling features + normalization
│   ├── dataset.py      # sliding windows (168→24), PyTorch DataLoaders
│   ├── model.py        # TemporalTransformer + LSTMBaseline (from scratch)
│   ├── train.py        # hand-written training loop + MLflow logging
│   ├── evaluate.py     # ablation: Naive vs LSTM vs Transformer
│   ├── registry.py     # champion/challenger promotion + drift reference
│   ├── inference.py    # champion → next 24h forecast
│   ├── drift.py        # Evidently drift report
│   ├── storage.py      # S3-or-local I/O (parquet + JSON)
│   ├── mlflow_setup.py # DagsHub-or-local MLflow config
│   └── pipeline.py     # orchestrator: build / forecast / retrain
├── frontend/           # Next.js dashboard (Vercel); reads private S3 via a server route
├── .github/workflows/  # forecast · retrain · build
└── pyproject.toml      # deps (managed with uv)
```

---

## Running Locally

Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync                                   # install everything from uv.lock

# one command per pipeline mode:
uv run python src/pipeline.py build       # full rebuild (trains all models)
uv run python src/pipeline.py forecast    # daily 24h forecast for all regions
uv run python src/pipeline.py retrain     # drift/staleness-gated retrain

# or individual stages:
uv run python src/ingestion.py
uv run python src/train.py --model transformer --country CAL
uv run python src/evaluate.py --country CAL
uv run python src/drift.py --country CAL
```

The dashboard is a Next.js app in `frontend/`, deployed on Vercel. It fetches
forecasts through a server route (`/api/forecast/<region>`) that reads the
private S3 bucket with server-side AWS credentials — keys never reach the browser.

```bash
cd frontend
npm install
npm run dev    # http://localhost:3000 (needs AWS creds in frontend/.env.local)
```

### Configuration (`.env`)

```bash
EIA_API_KEY=...               # free key from eia.gov/opendata
DAGSHUB_REPO_OWNER=...         # hosted MLflow (models + registry)
DAGSHUB_REPO_NAME=...
DAGSHUB_TOKEN=...
S3_BUCKET=...                  # forecasts + drift references
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

Everything has a **local fallback** — with no cloud config, MLflow uses local SQLite and storage writes to `data/`. The same code runs offline or in CI.

---

## Tech Stack

**PyTorch** · **MLflow / DagsHub** · **Evidently** · **Pandera** · **AWS S3 (boto3)** · **GitHub Actions** · **Next.js / Vercel** · **uv** · **EIA Open Data API**

---

## Hard Rules (design constraints)

1. PyTorch from scratch — no HuggingFace, no `Trainer`.
2. One-shot forecasting — 168 real hours in, 24 out. No recursive feeding.
3. Never touch the test set during training or tuning.
4. Never shuffle a time series — splits are chronological.
5. Fit scalers on training data only — no leakage.
