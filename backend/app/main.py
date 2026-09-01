import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .data import TICKERS
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


@app.get("/api/results/{ticker}")
def get_results(ticker: str):
    ticker = ticker.upper()
    path = os.path.join(PRECOMPUTED_DIR, f"{ticker}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No precomputed results for {ticker}")
    with open(path) as f:
        return json.load(f)


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
