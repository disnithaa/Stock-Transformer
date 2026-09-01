import numpy as np
import pandas as pd

FEATURES = ["LogReturn", "lag1_ret", "lag2_ret", "vol20", "mean_ret20", "rsi14", "rel_ma50", "dow_sin", "dow_cos"]
WINDOW = 20


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nine causal (no look-ahead) features per trading day. Target: next-day log-return."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["lag1_ret"] = df["LogReturn"].shift(1)
    df["lag2_ret"] = df["LogReturn"].shift(2)
    df["vol20"] = df["LogReturn"].rolling(20).std()
    df["mean_ret20"] = df["LogReturn"].rolling(20).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - (100 / (1 + rs))).fillna(50)
    ma50 = df["Close"].rolling(50).mean()
    df["rel_ma50"] = (df["Close"] - ma50) / ma50
    df["dow_sin"] = np.sin(2 * np.pi * df["Date"].dt.dayofweek / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df["Date"].dt.dayofweek / 5)
    df["target_ret"] = df["LogReturn"].shift(-1)
    return df.dropna().reset_index(drop=True)


def make_sequences(df: pd.DataFrame, window: int = WINDOW):
    X, y, dates, closes = [], [], [], []
    feat = df[FEATURES].values
    target = df["target_ret"].values
    close = df["Close"].values
    dt = df["Date"].values
    for i in range(window, len(df)):
        X.append(feat[i - window : i])
        y.append(target[i - 1])
        dates.append(dt[i])
        closes.append(close[i - 1])
    return np.array(X), np.array(y), np.array(dates), np.array(closes)


def latest_window(df: pd.DataFrame, window: int = WINDOW):
    """Build the single most recent feature window for a live next-day prediction."""
    feat_df = compute_features(df)
    feat = feat_df[FEATURES].values
    if len(feat) < window:
        raise ValueError("Not enough history to build a feature window")
    X = feat[-window:][None, :, :]
    last_close = float(feat_df["Close"].values[-1])
    last_date = pd.Timestamp(feat_df["Date"].values[-1])
    return X, last_close, last_date
