# Stock Transformer — Full-Stack App

Turns the `Stock_Transformer_Colab` research notebook (Transformer encoder forecasting
INFY and HUL next-day closes) into a deployable full-stack app:

- **backend/** — FastAPI service. Serves precomputed research results (metrics, charts,
  monthly error breakdown) and a **live** `/api/predict/{ticker}` endpoint that pulls the
  latest data and runs the trained model.
- **frontend/** — React (Vite) dashboard. Ticker switcher, live prediction panel, and the
  full set of research charts (price history, predicted-vs-actual, MC-dropout uncertainty
  bands, monthly error table, train/val loss curves).

## How it maps to the notebook

| Notebook section                                | Backend file            |
|--------------------------------------------------|--------------------------|
| §2 Data acquisition (yfinance + synthetic fallback) | `app/data.py`         |
| §4 Feature engineering                            | `app/features.py`      |
| §5 `StockTransformer` model                       | `app/model.py`          |
| §6 Training & walk-forward evaluation             | `app/train.py`          |
| §12 Monte Carlo dropout                           | `app/train.py`, `app/inference.py` |

`train.py` runs the exact same pipeline as the notebook, then saves:
- a model checkpoint (`backend/models/<TICKER>.pt`) for live inference
- a precomputed results JSON (`backend/precomputed/<TICKER>.json`) for the dashboard,
  so the frontend doesn't need to re-run training on every page load.

`inference.py` loads that checkpoint and predicts the *next* trading day's close using
the freshest data it can fetch — real NSE data via `yfinance` when the server has
outbound internet, else the same statistically-calibrated synthetic generator used in
training.

Model checkpoints and precomputed JSON for INFY and HUL are already included in this
repo (trained during setup), so the app works out of the box without retraining.

## Run locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend** (separate terminal)
```bash
cd frontend
npm install
npm run dev
```
The frontend reads the API base URL from `VITE_API_URL` (see `frontend/.env`, defaults
to `http://localhost:8000`).

To retrain from scratch (e.g. to pull real yfinance data instead of the bundled
checkpoints):
```bash
cd backend
python -m app.train
```

## Deploy

**Backend → Render**
1. Push `backend/` to a GitHub repo (or the whole `stock-app/` folder as a monorepo).
2. On Render: New → Web Service → point at the repo, root directory `backend`.
3. Render will pick up `render.yaml` automatically (CPU-only torch install + uvicorn
   start command). Free tier is enough — the model is ~70k parameters.
4. Note the deployed URL, e.g. `https://stock-transformer-api.onrender.com`.

**Frontend → Vercel**
1. Push `frontend/` (same repo, root directory `frontend`).
2. On Vercel: New Project → import the repo, root directory `frontend`.
3. Set environment variable `VITE_API_URL` to your Render backend URL.
4. Deploy. Vercel picks up `vercel.json` automatically.

**CORS**: `backend/app/main.py` currently allows all origins (`allow_origins=["*"]`) so
this works immediately. Once you know your Vercel domain, tighten that list.

## API reference

- `GET /api/health` — liveness check
- `GET /api/tickers` — `{"tickers": ["INFY", "HUL"]}`
- `GET /api/results/{ticker}` — full precomputed dashboard payload
- `GET /api/predict/{ticker}` — live next-day prediction with an 80% MC-dropout CI
