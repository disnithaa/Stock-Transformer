import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .company import get_company_profile
from .data import TEST_END, TICKERS, TRAIN_END, TRAIN_START, TEST_START
from .inference import predict_next_day

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "precomputed")

app = FastAPI(title="Stock Transformer API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/tickers")
def get_tickers():
    return {"tickers": list(TICKERS.keys())}


@app.get("/api/company/{ticker}")
def get_company(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker {ticker}")
    try:
        return get_company_profile(ticker)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Company data for {ticker} is currently unavailable from Yahoo Finance: {e}",
        )


@app.get("/api/results/{ticker}")
def get_results(ticker: str):
    ticker = ticker.upper()
    path = os.path.join(PRECOMPUTED_DIR, f"{ticker}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No precomputed results for {ticker}")
    with open(path) as f:
        payload = json.load(f)
    if payload.get("data_source") != "yfinance":
        raise HTTPException(
            status_code=503,
            detail=f"Verified real-data evaluation for {ticker} is not available yet. Run the training pipeline to fetch Yahoo Finance data and regenerate results.",
        )
    metadata = payload.get("evaluation_metadata", {})
    aligned = (
        metadata.get("model") == "Stock Transformer next-session log-return regression"
        and metadata.get("training_start") == TRAIN_START
        and metadata.get("training_end") == TRAIN_END
        and metadata.get("test_start") == TEST_START
        and metadata.get("test_end") == TEST_END
        and metadata.get("last_test_session", "") <= TEST_END
    )
    if not aligned:
        raise HTTPException(
            status_code=503,
            detail=f"Saved {ticker} results use an older model or date range. From backend/, run `python -m app.train` to regenerate the notebook-aligned real-data evaluation.",
        )
    return payload


@app.get("/api/predict/{ticker}")
def get_prediction(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker {ticker}")
    try:
        return predict_next_day(ticker)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
