import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import REAL_DATA_SOURCE, TICKERS, TRAIN_END, TRAIN_START, TEST_END, TEST_START, load_stock_data
from .direction import fit_direction_classifier
from .evaluation import (
    apply_minmax_scaler,
    chronological_train_validation_indices,
    evaluation_suite,
    fit_minmax_scaler,
)
from .features import FEATURES, WINDOW, compute_features, make_sequences
from .model import StockTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "precomputed")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PRECOMPUTED_DIR, exist_ok=True)


def run_pipeline(ticker, df, epochs=40, batch_size=32, mc_passes=80, use_real_flag=True, seed=0):
    if use_real_flag and df.attrs.get("data_source") != REAL_DATA_SOURCE:
        raise ValueError("Real-data training requires a dataset fetched from Yahoo Finance.")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Keep the final dated row for test targets; its next-return label is not
    # needed because each sequence takes the label from the preceding row.
    feat_df = compute_features(df, require_target=False)
    train_mask = feat_df["Date"] <= TRAIN_END
    train_df = feat_df[train_mask].reset_index(drop=True)
    test_df = feat_df[feat_df["Date"] >= (pd.Timestamp(TEST_START) - pd.Timedelta(days=60))].reset_index(drop=True)

    Xtr_full, ytr_full, dtr_full, ctr_full = make_sequences(train_df, WINDOW)
    Xte_full, yte_full, dte_full, cte_full = make_sequences(test_df, WINDOW)
    test_target_dates = pd.to_datetime(dte_full)
    keep = (test_target_dates >= pd.Timestamp(TEST_START)) & (test_target_dates <= pd.Timestamp(TEST_END))
    Xte, yte, dte, cte = Xte_full[keep], yte_full[keep], dte_full[keep], cte_full[keep]
    train_idx, val_idx = chronological_train_validation_indices(len(Xtr_full), 0.8)
    Xtr, ytr = Xtr_full[train_idx], ytr_full[train_idx]
    Xval, yval = Xtr_full[val_idx], ytr_full[val_idx]

    n_feat = Xtr.shape[2]
    x_min, x_max = fit_minmax_scaler(Xtr)
    scale_x = lambda X: apply_minmax_scaler(X, x_min, x_max)
    y_mean, y_std = ytr.mean(), ytr.std()
    scale_y = lambda y: (y - y_mean) / y_std
    unscale_y = lambda y: y * y_std + y_mean

    Xtr_t = torch.tensor(scale_x(Xtr), dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(scale_y(ytr), dtype=torch.float32, device=DEVICE)
    Xval_t = torch.tensor(scale_x(Xval), dtype=torch.float32, device=DEVICE)
    yval_t = torch.tensor(scale_y(yval), dtype=torch.float32, device=DEVICE)
    Xte_t = torch.tensor(scale_x(Xte), dtype=torch.float32, device=DEVICE)

    model = StockTransformer(n_features=n_feat).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    n_train = len(Xtr_t)
    best_val, best_state = float("inf"), None
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n_train
        sched.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), yval_t).item()
        train_losses.append(ep_loss)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"[{ticker}] epoch {epoch:3d} train={ep_loss:.4f} val={val_loss:.4f}")

    best_epoch = int(np.argmin(val_losses))
    # Match the notebook: evaluate with the best validation checkpoint, without
    # refitting on the validation tail.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        transformer_raw_ret = unscale_y(model(Xte_t).cpu().numpy())
        validation_raw_ret = unscale_y(model(Xval_t).cpu().numpy())
    transformer_val_accuracy = float(np.mean((validation_raw_ret > 0) == (yval > 0)) * 100)
    majority_label = int(np.mean(ytr > 0) >= 0.5)
    majority_val_accuracy = float(np.mean((yval > 0) == majority_label) * 100)
    direction_classifier = fit_direction_classifier(Xtr_full, ytr_full, Xte)
    validation_candidates = [
        (majority_val_accuracy, 0, "Training majority class"),
        (transformer_val_accuracy, 1, "Transformer raw-return sign"),
        (direction_classifier["validation_accuracy"], 2, direction_classifier["name"]),
    ]
    selected_accuracy, _, selected_name = max(validation_candidates, key=lambda row: (row[0], -row[1]))
    if selected_name == direction_classifier["name"] and selected_accuracy > transformer_val_accuracy and selected_accuracy > majority_val_accuracy:
        direction = {
            "model": selected_name,
            "estimator": direction_classifier["model"],
            "flatten": direction_classifier["flatten"],
            "predictions": direction_classifier["predictions"],
            "confidence": direction_classifier["confidence"],
            "validation_accuracy": direction_classifier["validation_accuracy"],
        }
        direction_sign = np.where(direction["predictions"] == 1, 1.0, -1.0)
        pred_ret = direction_sign * np.abs(transformer_raw_ret)
    elif selected_name == "Training majority class":
        direction = {
            "model": selected_name,
            "estimator": None,
            "flatten": None,
            "predictions": np.full(len(Xte), majority_label, dtype=int),
            "confidence": None,
            "validation_accuracy": selected_accuracy,
        }
        direction_sign = np.where(direction["predictions"] == 1, 1.0, -1.0)
        pred_ret = direction_sign * np.abs(transformer_raw_ret)
    else:
        direction = {
            "model": "Transformer raw-return sign",
            "estimator": None,
            "flatten": None,
            "predictions": (transformer_raw_ret > 0).astype(int),
            "confidence": None,
            "validation_accuracy": transformer_val_accuracy,
        }
        pred_ret = transformer_raw_ret.copy()
    actual_ret = yte
    pred_close = cte * np.exp(pred_ret)
    actual_close = cte * np.exp(actual_ret)
    mae = float(np.mean(np.abs(pred_close - actual_close)))
    mse = float(np.mean((pred_close - actual_close) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((actual_close - pred_close) ** 2)
    ss_tot = np.sum((actual_close - actual_close.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    # Binary direction is defined as up iff return > 0; zero belongs to down.
    da = float(np.mean((pred_ret > 0) == (actual_ret > 0)) * 100)
    direction_validation_accuracy = direction["validation_accuracy"]

    # Aligned, train-only baseline suite; all rows share the exact test dates above.
    raw = df.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["ma5_prior"] = raw["Close"].shift(1).rolling(5).mean()
    ma5 = raw.set_index("Date")["ma5_prior"].reindex(pd.to_datetime(dte)).to_numpy()
    if not np.isfinite(ma5).all():
        raise ValueError("Moving-average baseline is missing values on evaluation dates")
    evaluation = evaluation_suite(
        Xtr_full, ytr_full, Xte, yte, cte, actual_close, ma5,
        transformer_raw_ret, direction,
        Xtr_full[:, -1, FEATURES.index("vol20")],
        Xte[:, -1, FEATURES.index("vol20")],
    )

    model.train()
    mc_preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            mc_preds.append(unscale_y(model(Xte_t).cpu().numpy()))
    mc_preds = np.array(mc_preds)
    if direction["model"] != "Transformer raw-return sign":
        mc_preds = np.abs(mc_preds) * direction_sign[None, :]
    mc_p10 = np.percentile(mc_preds, 10, axis=0)
    mc_p90 = np.percentile(mc_preds, 90, axis=0)

    # Save checkpoint + scalers for live inference
    ckpt_path = os.path.join(MODELS_DIR, f"{ticker}.pt")
    torch.save(
        {
            "state_dict": best_state,
            "n_features": n_feat,
            "x_min": x_min,
            "x_max": x_max,
            "y_mean": y_mean,
            "y_std": y_std,
            "data_source": df.attrs.get("data_source", "synthetic"),
            "data_fetched_at": df.attrs.get("data_fetched_at"),
            "model_name": "Stock Transformer · notebook architecture",
            "direction_validation_accuracy": direction_validation_accuracy,
            "direction_model": direction["estimator"],
            "direction_flatten": direction["flatten"],
            "direction_model_name": direction["model"],
            "direction_constant": int(direction["predictions"][0]) if direction["model"] == "Training majority class" else None,
        },
        ckpt_path,
    )

    # ---- Precompute dashboard JSON ----
    test_dates = pd.to_datetime(dte)
    monthly_rows = []
    dfm = pd.DataFrame({"date": test_dates, "pred_c": pred_close, "act_c": actual_close, "pred_r": pred_ret, "act_r": actual_ret})
    dfm["ym"] = dfm["date"].dt.to_period("M")
    for ym, g in dfm.groupby("ym"):
        if len(g) < 5:
            continue
        m_mae = float(np.mean(np.abs(g.pred_c - g.act_c)))
        m_mse = float(np.mean((g.pred_c - g.act_c) ** 2))
        m_rmse = float(np.sqrt(m_mse))
        ss_res_m = np.sum((g.act_c - g.pred_c) ** 2)
        ss_tot_m = np.sum((g.act_c - g.act_c.mean()) ** 2)
        m_r2 = float(1 - ss_res_m / ss_tot_m) if ss_tot_m > 0 else None
        m_da = float(np.mean((g.pred_r > 0) == (g.act_r > 0)) * 100)
        monthly_rows.append({"month": str(ym), "mae": m_mae, "mse": m_mse, "rmse": m_rmse, "r2": m_r2, "da": m_da})

    # Downsample full historical timeline for payload size (~every 3rd trading day)
    full_dates = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d").tolist()
    full_close = df["Close"].round(2).tolist()
    step = max(1, len(full_dates) // 1500)
    hist_dates = full_dates[::step]
    hist_close = full_close[::step]
    if hist_dates[-1] != full_dates[-1]:
        hist_dates.append(full_dates[-1])
        hist_close.append(full_close[-1])

    payload = {
        "ticker": ticker,
        "data_source": df.attrs.get("data_source", "synthetic"),
        "data_fetched_at": df.attrs.get("data_fetched_at"),
        "data_start_date": pd.to_datetime(df["Date"]).min().strftime("%Y-%m-%d"),
        "data_end_date": pd.to_datetime(df["Date"]).max().strftime("%Y-%m-%d"),
        "metrics": {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "DA": da},
        "evaluation": evaluation,
        "evaluation_metadata": {
            "model": "Stock Transformer next-session log-return regression",
            "target": "next-session close-to-close log return; up iff return > 0",
            "horizon": "1 trading session",
            "training_start": TRAIN_START,
            "training_end": TRAIN_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "split": "chronological holdout",
            "evaluation_start": TEST_START,
            "evaluation_end": TEST_END,
            "first_test_session": test_dates.min().strftime("%Y-%m-%d"),
            "last_test_session": test_dates.max().strftime("%Y-%m-%d"),
            "observations": int(len(test_dates)),
            "forecasting_method": "fixed pre-test Transformer magnitude with validation-selected direction; rolling input windows",
        },
        "direction_model_name": direction["model"],
        "direction_validation_accuracy": direction_validation_accuracy,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_dates": test_dates.strftime("%Y-%m-%d").tolist(),
        "prior_close": cte.round(2).tolist(),
        "actual_close": actual_close.round(2).tolist(),
        "pred_close": pred_close.round(2).tolist(),
        "actual_return": actual_ret.tolist(),
        "predicted_return": pred_ret.tolist(),
        "direction_confidence": direction["confidence"].tolist() if direction["confidence"] is not None else None,
        "mc_p10_close": (cte * np.exp(mc_p10)).round(2).tolist(),
        "mc_p90_close": (cte * np.exp(mc_p90)).round(2).tolist(),
        "monthly": monthly_rows,
        "history_dates": hist_dates,
        "history_close": hist_close,
        "train_test_split_date": TRAIN_END,
    }
    with open(os.path.join(PRECOMPUTED_DIR, f"{ticker}.json"), "w") as f:
        json.dump(payload, f)

    print(f"[{ticker}] DONE  MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f} DA={da:.2f}%")
    return payload


def main():
    # yfinance's end date is exclusive, so add one day to include June 30.
    end_exclusive = (pd.Timestamp(TEST_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for ticker in TICKERS:
        df = load_stock_data(ticker, use_real=True, end=end_exclusive)
        run_pipeline(ticker, df)


if __name__ == "__main__":
    main()
