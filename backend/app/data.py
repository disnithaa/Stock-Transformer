import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

TICKERS = {"INFY": "INFY.NS", "HUL": "HINDUNILVR.NS"}
TRAIN_START, TRAIN_END = "2000-01-01", "2024-12-31"
TEST_START, TEST_END = "2025-01-01", "2026-06-30"

INFY_ANCHORS = {
    2000.0: 130, 2000.9: 260, 2001.9: 90, 2003.9: 140, 2005.9: 260,
    2007.9: 430, 2008.9: 190, 2009.9: 330, 2011.9: 300, 2013.9: 420,
    2015.9: 560, 2017.9: 500, 2019.9: 720, 2020.15: 520, 2020.9: 950,
    2021.9: 1550, 2022.9: 1380, 2023.9: 1440, 2024.99: 1850,
}
HUL_ANCHORS = {
    2000.0: 48, 2001.9: 58, 2003.9: 62, 2005.9: 78, 2007.9: 105,
    2008.9: 90, 2009.9: 130, 2011.9: 200, 2013.9: 320, 2015.9: 420,
    2017.9: 620, 2018.9: 850, 2019.9: 1050, 2020.15: 900, 2020.9: 1180,
    2021.9: 1250, 2022.9: 1330, 2023.9: 1580, 2024.99: 2350,
}
INFY_TEST_SHAPE = {2025.0: 1.00, 2025.25: 0.90, 2025.45: 0.80, 2025.65: 0.92,
                   2025.9: 1.00, 2026.15: 1.10, 2026.35: 1.00, 2026.5: 1.05}
HUL_TEST_SHAPE = {2025.0: 1.00, 2025.25: 0.97, 2025.45: 0.93, 2025.65: 0.99,
                  2025.9: 1.05, 2026.15: 1.10, 2026.35: 1.07, 2026.5: 1.12}
ELEVATED_TRAIN = [
    ("2000-03-01", "2001-12-31", 2.2), ("2008-06-01", "2009-03-31", 2.4),
    ("2013-05-01", "2013-09-30", 1.5), ("2016-11-01", "2017-01-31", 1.3),
    ("2020-02-15", "2020-06-30", 3.0), ("2022-01-01", "2022-10-31", 1.5),
]


def fetch_real_data(yf_ticker, start, end):
    import yfinance as yf

    df = yf.download(yf_ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {yf_ticker}")
    df = df[["Close"]].reset_index()
    df.columns = ["Date", "Close"]
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def _build_trend(anchors, yrs):
    xs = np.array(sorted(anchors.keys()))
    ys = np.log(np.array([anchors[x] for x in xs]))
    cs = PchipInterpolator(xs, ys)
    return np.exp(cs(np.clip(yrs, xs.min(), xs.max())))


def _garch_vol(n, omega, alpha, beta, seed):
    rng = np.random.default_rng(seed)
    vol = np.zeros(n)
    eps = np.zeros(n)
    vol[0] = np.sqrt(omega / (1 - alpha - beta))
    for t in range(1, n):
        vol[t] = np.sqrt(omega + alpha * eps[t - 1] ** 2 + beta * vol[t - 1] ** 2)
        eps[t] = vol[t] * rng.standard_t(6)
    return vol


def _vol_multiplier(dt, windows):
    m = 1.0
    for s, e, mult in windows:
        if pd.Timestamp(s) <= dt <= pd.Timestamp(e):
            m = max(m, mult)
    return m


def _simulate_path(trend, dates, base_omega, alpha, beta, seed, noise_scale, elevated_windows, start_price=None):
    n = len(trend)
    vol = _garch_vol(n, base_omega, alpha, beta, seed)
    rng = np.random.default_rng(seed + 999)
    vmult = np.array([_vol_multiplier(d, elevated_windows) for d in dates])
    raw_shock = np.clip(vol * vmult * rng.standard_t(6, size=n) * noise_scale, -0.08, 0.08)
    raw_shock -= raw_shock.mean()
    log_trend = np.log(trend)
    trend_daily_ret = np.diff(log_trend, prepend=log_trend[0])
    price = np.zeros(n)
    price[0] = start_price if start_price is not None else trend[0]
    dev = np.log(price[0]) - log_trend[0]
    for t in range(1, n):
        ret = trend_daily_ret[t] + raw_shock[t] - 0.03 * dev
        price[t] = price[t - 1] * np.exp(ret)
        dev = np.log(price[t]) - log_trend[t]
    return price


def _build_synthetic_stock(anchors, test_shape, base_omega, alpha, beta, seed, noise_scale):
    train_dates = pd.bdate_range(TRAIN_START, TRAIN_END)
    test_dates = pd.bdate_range(TEST_START, TEST_END)
    train_years = train_dates.year + (train_dates.dayofyear - 1) / 365.25
    train_trend = _build_trend(anchors, train_years)
    train_price = _simulate_path(train_trend, train_dates, base_omega, alpha, beta, seed, noise_scale, ELEVATED_TRAIN)
    last_train_price = train_price[-1]
    frac = (test_dates - test_dates[0]).days / 365.25 + 2025.0
    xs = np.array(sorted(test_shape.keys()))
    ys = np.log(np.array([test_shape[x] for x in xs]) * last_train_price)
    cs = PchipInterpolator(xs, ys)
    test_trend = np.exp(cs(np.clip(frac, xs.min(), xs.max())))
    test_price = _simulate_path(
        test_trend, test_dates, base_omega, alpha, beta, seed + 500, noise_scale,
        [(TEST_START, TEST_END, 1.5)], start_price=last_train_price,
    )
    all_dates = train_dates.append(test_dates)
    full_price = np.concatenate([train_price, test_price])
    df = pd.DataFrame({"Date": all_dates, "Close": np.round(full_price, 2)})
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def generate_synthetic(ticker):
    if ticker == "INFY":
        return _build_synthetic_stock(INFY_ANCHORS, INFY_TEST_SHAPE, 0.000045, 0.08, 0.87, 101, 0.28)
    elif ticker == "HUL":
        return _build_synthetic_stock(HUL_ANCHORS, HUL_TEST_SHAPE, 0.000015, 0.06, 0.88, 202, 0.34)
    raise ValueError(ticker)


def load_stock_data(ticker: str, use_real: bool = True, start: str = TRAIN_START, end: str = TEST_END) -> pd.DataFrame:
    """Real yfinance data if available and use_real=True, else synthetic fallback."""
    if use_real:
        try:
            return fetch_real_data(TICKERS[ticker], start, end)
        except Exception:
            pass
    return generate_synthetic(ticker)
