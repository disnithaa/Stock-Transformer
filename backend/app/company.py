"""Live company profile and market fundamentals from Yahoo Finance."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import TICKERS


def _number(value, digits=2):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, digits) if np.isfinite(value) else None


def _integer(value):
    number = _number(value, 0)
    return int(number) if number is not None else None


def _latest_statement_value(statement, labels):
    if statement is None or statement.empty:
        return {"value": None, "period": None}
    for label in labels:
        if label not in statement.index:
            continue
        series = statement.loc[label].dropna()
        if series.empty:
            continue
        period = max(series.index)
        period_text = pd.Timestamp(period).strftime("%Y-%m-%d")
        return {"value": _number(series.loc[period], 0), "period": period_text}
    return {"value": None, "period": None}


def get_company_profile(ticker: str):
    """Fetch a live quote, one-year price history, profile and latest annual fundamentals."""
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise ValueError(f"Unknown ticker {ticker}")

    import yfinance as yf

    symbol = TICKERS[ticker]
    instrument = yf.Ticker(symbol)
    prices = instrument.history(period="1y", auto_adjust=False, actions=False)
    if prices.empty or "Close" not in prices:
        raise RuntimeError(f"Yahoo Finance returned no recent prices for {ticker}")

    close = pd.to_numeric(prices["Close"], errors="coerce").dropna()
    prices = prices.loc[close.index]
    if len(close) < 2:
        raise RuntimeError(f"Yahoo Finance returned too little recent price history for {ticker}")

    try:
        info = instrument.info or {}
    except Exception:
        info = {}

    last_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    first_close = float(close.iloc[0])
    latest_date = pd.Timestamp(close.index[-1])
    if latest_date.tzinfo is not None:
        latest_date = latest_date.tz_localize(None)

    # Fetch frames once above; support common Yahoo Finance statement label variants.
    try:
        income_stmt_frame = instrument.income_stmt
    except Exception:
        income_stmt_frame = pd.DataFrame()
    try:
        balance_sheet_frame = instrument.balance_sheet
    except Exception:
        balance_sheet_frame = pd.DataFrame()
    annual_financials = {
        "revenue": _latest_statement_value(income_stmt_frame, ["Total Revenue"]),
        "gross_profit": _latest_statement_value(income_stmt_frame, ["Gross Profit"]),
        "operating_income": _latest_statement_value(income_stmt_frame, ["Operating Income"]),
        "net_income": _latest_statement_value(income_stmt_frame, ["Net Income", "Net Income Common Stockholders"]),
        "diluted_eps": _latest_statement_value(income_stmt_frame, ["Diluted EPS", "Basic EPS"]),
        "operating_cash_flow": _latest_statement_value(income_stmt_frame, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
        "free_cash_flow": _latest_statement_value(income_stmt_frame, ["Free Cash Flow"]),
        "total_assets": _latest_statement_value(balance_sheet_frame, ["Total Assets"]),
        "total_debt": _latest_statement_value(balance_sheet_frame, ["Total Debt", "Long Term Debt"]),
        "stockholders_equity": _latest_statement_value(balance_sheet_frame, ["Stockholders Equity", "Stockholders' Equity"]),
        "cash_and_equivalents": _latest_statement_value(balance_sheet_frame, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
    }

    latest_bar = prices.iloc[-1]
    high_series = pd.to_numeric(prices.get("High", close), errors="coerce").dropna()
    low_series = pd.to_numeric(prices.get("Low", close), errors="coerce").dropna()
    volume_series = pd.to_numeric(prices.get("Volume", pd.Series(dtype=float)), errors="coerce").dropna()
    period_dates = pd.to_datetime(close.index)
    if getattr(period_dates, "tz", None) is not None:
        period_dates = period_dates.tz_localize(None)

    return {
        "ticker": ticker,
        "symbol": symbol,
        "company_name": info.get("longName") or info.get("shortName"),
        "currency": info.get("currency") or "INR",
        "financial_currency": info.get("financialCurrency") or info.get("currency") or "INR",
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        "country": info.get("country"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "summary": info.get("longBusinessSummary"),
        "employees": _integer(info.get("fullTimeEmployees")),
        "quote": {
            "as_of_date": latest_date.strftime("%Y-%m-%d"),
            "last_close": _number(last_close),
            "previous_close": _number(previous_close),
            "change": _number(last_close - previous_close),
            "change_pct": _number((last_close / previous_close - 1) * 100) if previous_close else None,
            "one_year_return_pct": _number((last_close / first_close - 1) * 100) if first_close else None,
            "day_high": _number(latest_bar.get("High")),
            "day_low": _number(latest_bar.get("Low")),
            "week_52_high": _number(high_series.max()) if not high_series.empty else None,
            "week_52_low": _number(low_series.min()) if not low_series.empty else None,
            "moving_average_50d": _number(close.tail(50).mean()) if len(close) >= 50 else None,
            "moving_average_200d": _number(close.tail(200).mean()) if len(close) >= 200 else None,
            "average_volume_20d": _integer(volume_series.tail(20).mean()) if not volume_series.empty else None,
            "average_volume_3m": _integer(volume_series.tail(60).mean()) if not volume_series.empty else None,
        },
        "valuation": {
            "market_cap": _integer(info.get("marketCap")),
            "enterprise_value": _integer(info.get("enterpriseValue")),
            "trailing_pe": _number(info.get("trailingPE")),
            "forward_pe": _number(info.get("forwardPE")),
            "price_to_book": _number(info.get("priceToBook")),
            "enterprise_to_ebitda": _number(info.get("enterpriseToEbitda")),
            "beta": _number(info.get("beta")),
            "dividend_yield_pct": _number(info.get("dividendYield") * 100) if isinstance(info.get("dividendYield"), (int, float)) else None,
        },
        "annual_financials": annual_financials,
        "history_dates": [d.strftime("%Y-%m-%d") for d in period_dates],
        "history_close": [_number(value) for value in close.to_numpy()],
        "data_source": "Yahoo Finance",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
