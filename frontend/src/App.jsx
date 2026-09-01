import { useEffect, useState } from "react";
import "./index.css";
import "./dashboard.css";
import Sidebar from "./components/Sidebar";
import { TimelineChart, PredictedVsActualChart, UncertaintyChart, LossChart } from "./components/Charts";
import MonthlyTable from "./components/MonthlyTable";
import { TICKER_META } from "./tickerMeta";
import { api } from "./api";

const FALLBACK_TICKERS = ["INFY", "HUL"];

export default function App() {
  const [tickers, setTickers] = useState(FALLBACK_TICKERS);
  const [ticker, setTicker] = useState("INFY");
  const [results, setResults] = useState(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    api.tickers().then((r) => setTickers(r.tickers)).catch(() => {});
  }, []);

  useEffect(() => {
    setStatus("loading");
    api
      .results(ticker)
      .then((r) => {
        setResults(r);
        setStatus("done");
      })
      .catch(() => setStatus("error"));
  }, [ticker]);

  const meta = TICKER_META[ticker] || { name: ticker, color: "var(--gold)" };

  return (
    <div className="shell">
      <Sidebar ticker={ticker} onSelect={setTicker} tickers={tickers} />

      <div className="main">
          <div className="hero">
            <div className="hero-eyebrow">Out-of-sample test period · Jan 2025 – Jun 2026</div>
            <h1 className="hero-title">
              Can attention see past {ticker === "INFY" ? "IT services'" : "FMCG's"} noise to the next tick?
            </h1>
            <p className="hero-desc">
              A 2-layer Transformer encoder reads 20 trading days of returns, volatility, RSI and moving-average
              deviation to forecast {meta.name}'s next close — walk-forward, one day at a time, with uncertainty
              from Monte Carlo dropout.
            </p>
          </div>

          {status === "loading" && <div className="loading">Loading results for {ticker}…</div>}
          {status === "error" && <div className="error-box">Couldn't reach the API. Is the backend running?</div>}

          {status === "done" && results && (
            <>
              <div className="metrics-row">
                <MetricCell label="MAE (₹)" value={results.metrics.MAE.toFixed(2)} />
                <MetricCell label="RMSE (₹)" value={results.metrics.RMSE.toFixed(2)} />
                <MetricCell label="R²" value={results.metrics.R2.toFixed(4)} />
                <MetricCell label="Directional accuracy" value={`${results.metrics.DA.toFixed(1)}%`} />
              </div>

              <section className="block">
                <div className="block-title">Full price history</div>
                <div className="block-sub">2000–2024 training window, with the walk-forward test period appended.</div>
                <div className="chart-wrap">
                  <TimelineChart
                    dates={results.history_dates}
                    closes={results.history_close}
                    splitDate={results.train_test_split_date}
                    color={meta.color}
                  />
                </div>
              </section>

              <section className="block">
                <div className="block-title">Predicted vs actual close</div>
                <div className="block-sub">One-step walk-forward forecasts against the realised closing price.</div>
                <div className="chart-wrap">
                  <PredictedVsActualChart
                    dates={results.test_dates}
                    actual={results.actual_close}
                    predicted={results.pred_close}
                    color={meta.color}
                  />
                </div>
              </section>

              <section className="block">
                <div className="block-title">Monte Carlo dropout uncertainty</div>
                <div className="block-sub">80% confidence band from 80 stochastic forward passes.</div>
                <div className="chart-wrap">
                  <UncertaintyChart
                    dates={results.test_dates}
                    actual={results.actual_close}
                    p10={results.mc_p10_close}
                    p90={results.mc_p90_close}
                    color={meta.color}
                  />
                </div>
              </section>

              <section className="block">
                <div className="block-title">Monthly error distribution</div>
                <div className="block-sub">Test-period metrics recomputed within each calendar month.</div>
                <MonthlyTable monthly={results.monthly} />
              </section>

              <section className="block">
                <div className="block-title">Training vs validation loss</div>
                <div className="block-sub">Chronological 80/20 split, checkpointed at lowest validation MSE (epoch {results.best_epoch}).</div>
                <div className="chart-wrap">
                  <LossChart trainLosses={results.train_losses} valLosses={results.val_losses} bestEpoch={results.best_epoch} />
                </div>
              </section>

              <div className="footer-note">
                Training data is real NSE daily closes via yfinance where the deployment has outbound internet
                access; it falls back to a statistically calibrated synthetic series (matching sector growth,
                GARCH-style volatility clustering, and known market-stress windows) when it doesn't.
              </div>
            </>
          )}
      </div>
    </div>
  );
}

function MetricCell({ label, value }) {
  return (
    <div className="metric-cell">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}
