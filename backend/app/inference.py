import os

import numpy as np
import torch

from .data import load_stock_data, TEST_END
from .features import latest_window
from .model import StockTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_cache = {}


def _load_checkpoint(ticker: str):
    if ticker in _cache:
        return _cache[ticker]
    path = os.path.join(MODELS_DIR, f"{ticker}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained checkpoint for {ticker}. Run training first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = StockTransformer(n_features=ckpt["n_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ckpt["model"] = model
    _cache[ticker] = ckpt
    return ckpt


def predict_next_day(ticker: str, mc_passes: int = 80):
    ckpt = _load_checkpoint(ticker)
    model = ckpt["model"]
    x_min, x_max = ckpt["x_min"], ckpt["x_max"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    # Pull the freshest data available (real via yfinance when the deployment has internet,
    # else the same synthetic generator used for training) and build the latest feature window.
    df = load_stock_data(ticker, use_real=True, end=TEST_END)
    X, last_close, last_date = latest_window(df)

    Xs = (X - x_min) / (x_max - x_min)
    Xt = torch.tensor(Xs, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_ret = (model(Xt).numpy() * y_std + y_mean)[0]

    model.train()
    mc_preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            mc_preds.append((model(Xt).numpy() * y_std + y_mean)[0])
    mc_preds = np.array(mc_preds)
    p10, p90 = np.percentile(mc_preds, [10, 90])

    pred_close = float(last_close * np.exp(pred_ret))
    p10_close = float(last_close * np.exp(p10))
    p90_close = float(last_close * np.exp(p90))

    return {
        "ticker": ticker,
        "as_of_date": last_date.strftime("%Y-%m-%d"),
        "last_close": round(float(last_close), 2),
        "predicted_next_close": round(pred_close, 2),
        "predicted_return_pct": round(float(pred_ret) * 100, 3),
        "ci80_low": round(min(p10_close, p90_close), 2),
        "ci80_high": round(max(p10_close, p90_close), 2),
    }
