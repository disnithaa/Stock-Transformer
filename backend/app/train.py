import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import TICKERS, TRAIN_END, TRAIN_START, TEST_END, TEST_START, load_stock_data
from .features import FEATURES, WINDOW, compute_features, make_sequences
from .model import StockTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "precomputed")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PRECOMPUTED_DIR, exist_ok=True)


def run_pipeline(ticker, df, epochs=40, batch_size=32, mc_passes=80, use_real_flag=True, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    feat_df = compute_features(df)
    train_mask = feat_df["Date"] <= TRAIN_END
    train_df = feat_df[train_mask].reset_index(drop=True)
    test_df = feat_df[feat_df["Date"] >= (pd.Timestamp(TEST_START) - pd.Timedelta(days=60))].reset_index(drop=True)

    Xtr_full, ytr_full, dtr_full, ctr_full = make_sequences(train_df, WINDOW)
    Xte_full, yte_full, dte_full, cte_full = make_sequences(test_df, WINDOW)
    keep = pd.to_datetime(dte_full) >= pd.Timestamp(TEST_START)
    Xte, yte, dte, cte = Xte_full[keep], yte_full[keep], dte_full[keep], cte_full[keep]

    n = len(Xtr_full)
    split = int(n * 0.8)
    Xtr, ytr = Xtr_full[:split], ytr_full[:split]
    Xval, yval = Xtr_full[split:], ytr_full[split:]

    n_feat = Xtr.shape[2]
    x_min = Xtr.reshape(-1, n_feat).min(axis=0)
    x_max = Xtr.reshape(-1, n_feat).max(axis=0)
    x_max = np.where(x_max == x_min, x_min + 1e-6, x_max)
    scale_x = lambda X: (X - x_min) / (x_max - x_min)
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
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_ret = unscale_y(model(Xte_t).cpu().numpy())
    actual_ret = yte
    pred_close = cte * np.exp(pred_ret)
    actual_close = cte * np.exp(actual_ret)
    mae = float(np.mean(np.abs(pred_close - actual_close)))
    mse = float(np.mean((pred_close - actual_close) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((actual_close - pred_close) ** 2)
    ss_tot = np.sum((actual_close - actual_close.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    da = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)) * 100)

    model.train()
    mc_preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            mc_preds.append(unscale_y(model(Xte_t).cpu().numpy()))
    mc_preds = np.array(mc_preds)
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
        m_da = float(np.mean(np.sign(g.pred_r) == np.sign(g.act_r)) * 100)
        monthly_rows.append({"month": str(ym), "mae": m_mae, "mse": m_mse, "rmse": m_rmse, "r2": m_r2, "da": m_da})

    # Downsample full historical timeline for payload size (~every 3rd trading day)
    full_dates = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d").tolist()
    full_close = df["Close"].round(2).tolist()
    step = max(1, len(full_dates) // 1500)
    hist_dates = full_dates[::step]
    hist_close = full_close[::step]

    payload = {
        "ticker": ticker,
        "data_source": "synthetic" if not use_real_flag else "real_or_synthetic_fallback",
        "metrics": {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "DA": da},
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_dates": test_dates.strftime("%Y-%m-%d").tolist(),
        "actual_close": actual_close.round(2).tolist(),
        "pred_close": pred_close.round(2).tolist(),
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
    for ticker in TICKERS:
        df = load_stock_data(ticker, use_real=True)
        run_pipeline(ticker, df)


if __name__ == "__main__":
    main()
