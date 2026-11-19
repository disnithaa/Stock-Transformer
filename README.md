# Stock Transformer

A real-data, one-session stock forecasting research app for INFY and HUL. The backend fetches Yahoo Finance prices, trains the Transformer model from the real-data notebook, and serves dated evaluation results. The Transformer directly predicts signed next-session log returns.

The real-data model notebook is included under [`notebooks/Real_Stock_Transformer_Colab.ipynb`](notebooks/Real_Stock_Transformer_Colab.ipynb). [Open it directly in Google Colab](https://colab.research.google.com/drive/1q77xK1usuv78F00fAkTmXhV8957GBZKK).

## Architecture and targets

| Area | File | Role |
|---|---|---|
| Data | `backend/app/data.py` | Yahoo Finance daily close input; synthetic data only by explicit opt-in |
| Features | `backend/app/features.py` | Nine causal features, 20-session samples, next-session log-return target |
| Transformer | `backend/app/model.py` | Two-layer Transformer encoder with scalar normalized log-return output |
| Direction | Transformer output sign | The notebook predicts signed next-session log return directly; positive is up, negative is down |
| Evaluation | `backend/app/evaluation.py` | Train-only reference models, confusion metrics, logistic-baseline confidence buckets, delayed backtest diagnostic |
| Training | `backend/app/train.py` | Notebook-matched 80/20 chronological training/validation and untouched 2025-01-01–2026-06-30 holdout |
| Inference | `backend/app/inference.py` | Current real-data window and direct Transformer return prediction |
| API | `backend/app/main.py` | Results, health, ticker and live prediction endpoints |
| UI | `frontend/src/` | Summary, scorecard, charts and monthly results |
| Company profiles | `backend/app/company.py`, `frontend/src/components/CompanyPage.jsx` | Yahoo Finance company profile, quote, valuation, annual financials and one-year price chart |

For each target session, the input window ends at the preceding known close. The target is `log(next_close / prior_close)`. The Transformer directly predicts this signed return; its sign determines direction. Directional accuracy is computed from the predicted return sign, not from close-level R².

## Evaluation design

- The report-aligned split is chronological: training data runs through 2024-12-31 and the untouched test period runs from 2025-01-01 through 2026-06-30. Market dates without a trading session are naturally absent.
- Epoch selection and direction-model selection use a chronological tail of pre-test training observations only. After epoch selection, the Transformer refits through the full pre-test period.
- Preprocessing is fitted within each training fit. Reference regression and direction-baseline models use the same test dates as the forecast model.
- Regression scorecards include previous-close persistence, a five-close moving average, linear log-return regression, percentage-return Ridge, volatility-normalized Ridge, and the notebook Transformer forecast.
- Direction scorecards include always-up, training majority class, logistic regression, the notebook Transformer’s raw-return sign, and an expanding logistic refit at each test origin. Metrics include accuracy with a Wilson 95% interval, balanced accuracy, up precision/recall/F1, MCC, confusion matrix, and actual class mix.
- The three-class experiment uses a causal threshold `max(10 bp, 0.25 × trailing 20-session volatility)`. The rule is fixed before test evaluation.
- The optional scorecard strategy is long/flat and delays action one session. It uses close-only returns and charges 10 bp per position turnover unit. This is a diagnostic proxy; it omits opening-price fills, liquidity, borrow, taxes and market impact, so its results are not a claim of executable profitability.
- R² is computed on close levels. It can be high because prices trend and must be interpreted alongside persistence errors and return-direction metrics.

The expanded comparison fields are written to `backend/precomputed/<TICKER>.json` under `evaluation` and `evaluation_metadata`. Old precomputed files without those keys are identified by the UI as needing regeneration; the app does not fabricate comparison metrics.

On each company page, the test calendar opens a dedicated date analysis with actual and predicted closes, returns, direction result, price error, dropout range, and a nearby actual-versus-predicted chart. These date pages are opened from the company page and are not top-level navigation items.

## Setup and run

Backend, from `backend/`:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend, in another terminal from `frontend/`:

```bash
npm install
npm run dev
```

Set `VITE_API_URL` to the backend URL if it is not `http://localhost:8000`.

## Reproduce the evaluation

From `backend/`, with Yahoo Finance access:

```bash
python -m app.train
```

This fetches current real data, refits the models, evaluates the predeclared holdout and baselines, then updates the two `.pt` checkpoints and JSON payloads. It does not substitute synthetic prices if the Yahoo Finance request fails.

Run deterministic unit checks from `backend/`:

```bash
python -m unittest discover -s tests -v
```

Frontend quality checks from `frontend/`:

```bash
npm run lint
npm run build
```

## API

- `GET /api/health`
- `GET /api/tickers`
- `GET /api/company/{ticker}` — live quote, company profile, valuation ratios, annual statements, and one-year closing-price history
- `GET /api/results/{ticker}` — saved real-data evaluation payload
- `GET /api/predict/{ticker}` — live next-session estimate; returns an error if real prices or a verified checkpoint are unavailable

## Findings and limitations

See [MODEL_AUDIT.md](MODEL_AUDIT.md) for the initial audit of the earlier classifier-based pipeline. This project now follows the attached real-data notebook: the Transformer predicts the signed return directly. High price-level R² does not imply predictive return direction. Daily price data and a single holdout provide limited evidence, and dropout intervals are not calibrated. These models are for research and education, not investment advice.

The dashboard deploys as a Vite frontend and FastAPI backend. Configure `VITE_API_URL` to the deployed backend and restrict CORS in `backend/app/main.py` to the intended frontend origin before public deployment.
