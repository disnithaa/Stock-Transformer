import os
from datetime import datetime, timezone

import numpy as np
import torch

from .data import REAL_DATA_SOURCE, load_stock_data
from .direction import classifier_input
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
    if ckpt.get("data_source") != REAL_DATA_SOURCE:
        raise FileNotFoundError(f"The {ticker} checkpoint was not trained on verified Yahoo Finance data. Retrain the model first.")
    if ckpt.get("model_name") != "Stock Transformer · notebook architecture":
        raise FileNotFoundError(f"The {ticker} checkpoint uses an older model. Run the notebook-aligned training pipeline first.")
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

    # Live inference requires current real prices and never falls back to generated data.
    df = load_stock_data(ticker, use_real=True, end=datetime.now(timezone.utc).date().isoformat())
    X, last_close, last_date = latest_window(df)

    Xs = (X - x_min) / (x_max - x_min)
    Xt = torch.tensor(Xs, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_ret = (model(Xt).numpy() * y_std + y_mean)[0]
    direction_model = ckpt.get("direction_model")
    direction_constant = ckpt.get("direction_constant")
    if direction_model is not None:
        direction_class = int(
            direction_model.predict(classifier_input(X, ckpt["direction_flatten"]))[0]
        )
        direction_sign = 1.0 if direction_class == 1 else -1.0
        pred_ret = direction_sign * abs(float(pred_ret))
    elif direction_constant is not None:
        direction_sign = 1.0 if int(direction_constant) == 1 else -1.0
        pred_ret = direction_sign * abs(float(pred_ret))
    else:
        direction_sign = 1.0 if pred_ret > 0 else -1.0

    model.train()
    mc_preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            raw_return = float((model(Xt).numpy() * y_std + y_mean)[0])
            mc_preds.append(direction_sign * abs(raw_return) if (direction_model is not None or direction_constant is not None) else raw_return)
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
        "predicted_direction": "up" if pred_ret > 0 else "down" if pred_ret < 0 else "flat",
        "direction_model_name": ckpt.get("direction_model_name", "Transformer raw-return sign"),
        "ci80_low": round(min(p10_close, p90_close), 2),
        "ci80_high": round(max(p10_close, p90_close), 2),
        "data_source": REAL_DATA_SOURCE,
    }
