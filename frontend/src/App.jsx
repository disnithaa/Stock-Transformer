import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { TICKER_META } from "./tickerMeta";
import { PredictedVsActualChart, UncertaintyChart, LossChart } from "./components/Charts";
import MonthlyTable from "./components/MonthlyTable";
import "./index.css";
import "./dashboard.css";

const money = (value) => Number.isFinite(value) ? `₹${Number(value).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—";
const dateFmt = (value) => value ? new Date(`${value}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" }) : "—";

export default function App() {
  const [tickers, setTickers] = useState(["INFY", "HUL"]);
  const [ticker, setTicker] = useState("INFY");
  const [page, setPage] = useState("overview");
  const [results, setResults] = useState(null);
  const [status, setStatus] = useState("loading");
  const [prediction, setPrediction] = useState({ status: "idle", data: null, error: "" });
  const requestId = useRef(0);

  useEffect(() => { api.tickers().then((r) => setTickers(r.tickers)).catch(() => {}); }, []);
  useEffect(() => {
    let active = true;
    requestId.current += 1;
    api.results(ticker).then((data) => { if (active) { setResults(data); setStatus("done"); } })
      .catch(() => { if (active) setStatus("error"); });
    return () => { active = false; };
  }, [ticker]);

  const meta = TICKER_META[ticker] || { name: ticker, symbol: ticker, desc: "NSE listed equity" };
  const latest = useMemo(() => results?.history_close?.length ? results.history_close.length - 1 : -1, [results]);
  function selectTicker(nextTicker) {
    requestId.current += 1;
    setTicker(nextTicker);
    setResults(null);
    setStatus("loading");
    setPrediction({ status: "idle", data: null, error: "" });
  }
  async function runPrediction() {
    const id = ++requestId.current;
    setPrediction({ status: "loading", data: null, error: "" });
    try { const data = await api.predict(ticker); if (id === requestId.current && data.ticker === ticker) setPrediction({ status: "done", data, error: "" }); }
    catch (e) { if (id === requestId.current) setPrediction({ status: "error", data: null, error: e.message }); }
  }

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#overview" aria-label="Stock Transformer home"><span className="brand-mark">ST</span><span><b>ST <i>/</i> RESEARCH</b><small>QUANTITATIVE FORECASTING</small></span></a>
      <nav className="primary-nav" aria-label="Primary navigation">
        <button className={page === "overview" ? "selected" : ""} onClick={() => setPage("overview")}>Overview</button>
        <button className={page === "lab" ? "selected" : ""} onClick={() => setPage("lab")}>Model Lab</button>
      </nav>
      <div className="top-actions"><label className="ticker-select"><span className="sr-only">Select stock</span><select value={ticker} onChange={(e) => selectTicker(e.target.value)}>{tickers.map((t) => <option key={t}>{t}</option>)}</select><span>⌄</span></label><span className={`status-pill ${status}`}><i />{status === "done" ? "Research data" : status === "loading" ? "Loading" : "API unavailable"}</span><a className="github-link" href="https://github.com/disnithaa/Stock-Transformer" target="_blank" rel="noreferrer">GitHub ↗</a></div>
    </header>
    <main className="page-content">
      {status === "loading" && <div className="loading-state"><span className="spinner"/> Loading research results for {ticker}…</div>}
      {status === "error" && <div className="error-state"><b>Couldn’t load research results</b><p>Check that the Stock Transformer API is available, then try again.</p><button onClick={() => { setStatus("loading"); api.results(ticker).then((data) => { setResults(data); setStatus("done"); }).catch(() => setStatus("error")); }}>Retry</button></div>}
      {status === "done" && results && (page === "overview" ? <>
        <div className="page-heading"><div><div className="eyebrow">NSE · RESEARCH WORKSPACE</div><h1>{meta.name} <span>{ticker}</span></h1><p>{meta.desc} <span className="dot-sep">·</span> Evaluation period {dateFmt(results.test_dates[0])} — {dateFmt(results.test_dates.at(-1))}</p></div><div className="data-badge"><span className="badge-dot"/><div><b>{results.data_source === "yfinance" ? "Yahoo Finance data" : "Source not verified"}</b><small>{results.data_source === "yfinance" ? `Coverage through ${dateFmt(results.data_end_date)}` : "Precomputed evaluation"}</small></div></div></div>
        <div className="price-strip"><div><small>LATEST EVALUATION CLOSE</small><strong>{money(results.history_close[latest])}</strong><span>{dateFmt(results.history_dates[latest])} · {results.data_source === "yfinance" ? "Yahoo Finance daily close" : "source metadata unavailable"}</span></div><div className="price-divider"/><div><small>OUT-OF-SAMPLE PERIOD</small><strong>{dateFmt(results.test_dates[0])} <em>→</em> {dateFmt(results.test_dates.at(-1))}</strong><span>{results.test_dates.length.toLocaleString()} daily observations</span></div><div className="price-divider"/><div><small>MODEL STATUS</small><strong className="ready-text">Evaluation available</strong><span>Walk-forward, one-step predictions</span></div></div>
        <div className="section-caption"><span>01</span> MODEL PERFORMANCE <i/></div>
        <div className="metrics-row">
          <Metric label="Mean absolute error" value={money(results.metrics.MAE)} detail="Price error · INR"/>
          <Metric label="Root mean squared error" value={money(results.metrics.RMSE)} detail="Price error · INR"/>
          <Metric label="Directional accuracy" value={`${results.metrics.DA.toFixed(1)}%`} detail="Sign of next-day return" tone="mint"/>
          <Metric label="R²" value={results.metrics.R2.toFixed(4)} detail="Out-of-sample fit"/>
        </div>
        <section className="hero-chart-card"><div className="card-head"><div><div className="eyebrow">WALK-FORWARD EVALUATION</div><h2>Actual vs predicted close</h2><p>One-step forecasts compared with realised closing prices across the test period.</p></div><div className="chart-legend"><span><i className="legend-actual"/>Actual</span><span><i className="legend-pred"/>Model prediction</span></div></div><div className="hero-chart"><PredictedVsActualChart dates={results.test_dates} actual={results.actual_close} predicted={results.pred_close}/></div><div className="chart-foot"><span>Daily close · INR</span><span>{dateFmt(results.test_dates[0])} — {dateFmt(results.test_dates.at(-1))}</span></div></section>
        <div className="lower-grid"><section className="panel prediction-card"><div className="panel-head"><div><div className="eyebrow">ON-DEMAND INFERENCE</div><h2>Next-session estimate</h2></div><span className="neutral-tag">Not a trading signal</span></div>
          {prediction.status === "idle" && <p className="muted-copy">Run the trained model against the latest available data. This request is separate from the cached evaluation.</p>}
          {prediction.status === "loading" && <p className="muted-copy"><span className="spinner small"/> Fetching data and running inference…</p>}
          {prediction.status === "error" && <p className="error-copy">{prediction.error}</p>}
          {prediction.status === "done" && prediction.data && <><div className="prediction-number">{money(prediction.data.predicted_next_close)}</div><div className={`prediction-change ${prediction.data.predicted_return_pct > 0 ? "positive" : prediction.data.predicted_return_pct < 0 ? "negative" : ""}`}>{prediction.data.predicted_return_pct > 0 ? "▲" : prediction.data.predicted_return_pct < 0 ? "▼" : "—"} {Math.abs(prediction.data.predicted_return_pct).toFixed(3)}% vs {money(prediction.data.last_close)}</div><div className="range-row"><span>MC dropout 80% range</span><b>{money(prediction.data.ci80_low)} — {money(prediction.data.ci80_high)}</b></div><div className="range-row"><span>Input data through</span><b>{dateFmt(prediction.data.as_of_date)}</b></div><p className="muted-copy">The model estimates a {prediction.data.predicted_return_pct > 0 ? "positive" : prediction.data.predicted_return_pct < 0 ? "negative" : "near-zero"} next-session return relative to the latest available close. The interval reflects stochastic model predictions; calibration has not been established.</p></>}
          <button className="primary-button" onClick={runPrediction} disabled={prediction.status === "loading"}>{prediction.status === "done" ? "Run again" : "Run next-session estimate"}</button></section>
          <section className="panel summary-card"><div className="eyebrow">MODEL SNAPSHOT</div><h2>Transformer encoder</h2><p className="muted-copy">A compact model forecasts the next log return from a rolling window of engineered daily features.</p><div className="fact-list"><div><span>Input window</span><b>20 trading sessions</b></div><div><span>Input features</span><b>9 engineered features</b></div><div><span>Architecture</span><b>2 encoder layers · 4 heads</b></div><div><span>Uncertainty method</span><b>Monte Carlo dropout</b></div><div><span>Best validation epoch</span><b>{results.best_epoch + 1} / {results.train_losses.length}</b></div></div><button className="text-button" onClick={() => setPage("lab")}>Explore model evaluation <span>↗</span></button></section></div>
        <footer className="disclaimer">For research and educational purposes only. Forecasts are model estimates, not investment advice. <span>·</span> Historical evaluation uses daily closing prices fetched from Yahoo Finance.</footer>
      </> : <ModelLab results={results} ticker={ticker}/>)}</main>
  </div>;
}

function Metric({ label, value, detail, tone }) { return <div className="metric-card"><small>{label}</small><strong className={tone || ""}>{value}</strong><span>{detail}</span></div>; }

function ModelLab({ results, ticker }) {
  const features = ["Log return", "Lagged returns ×2", "20d volatility", "20d mean return", "RSI-14", "Deviation from MA-50", "Weekday encoding ×2"];
  return <><div className="page-heading lab-heading"><div><div className="eyebrow">RESEARCH / MODEL LAB</div><h1>Model evaluation <span>{ticker}</span></h1><p>Inspect the architecture, training behaviour, forecast errors, and uncertainty estimates.</p></div><div className="data-badge"><span className="badge-dot"/><div><b>Walk-forward test</b><small>{dateFmt(results.test_dates[0])} — {dateFmt(results.test_dates.at(-1))}</small></div></div></div>
    <div className="section-caption"><span>01</span> ARCHITECTURE <i/></div><section className="panel architecture"><div className="architecture-flow">{["Input features", "Linear embedding", "Positional encoding", "Transformer encoder", "Mean pooling", "Regression head", "Next-day log return"].map((step, i) => <div className="arch-step" key={step}><small>{String(i + 1).padStart(2, "0")}</small><b>{step}</b>{i < 6 && <span>→</span>}</div>)}</div><div className="feature-chips">{features.map((f) => <span key={f}>{f}</span>)}</div><div className="config-row"><div><small>Window length</small><b>20</b></div><div><small>Features</small><b>9</b></div><div><small>Embedding</small><b>64</b></div><div><small>Attention heads</small><b>4</b></div><div><small>Encoder layers</small><b>2</b></div><div><small>Output</small><b>1 return</b></div></div></section>
    <div className="lab-grid"><LabPanel title="Training & validation loss" subtitle={`Chronological 80/20 training split · best checkpoint at epoch ${results.best_epoch + 1}`}><LossChart trainLosses={results.train_losses} valLosses={results.val_losses} bestEpoch={results.best_epoch}/></LabPanel><LabPanel title="Actual vs predicted" subtitle="Walk-forward closing price estimates · INR"><PredictedVsActualChart dates={results.test_dates} actual={results.actual_close} predicted={results.pred_close}/></LabPanel><LabPanel title="Uncertainty analysis" subtitle="80th percentile interval from stochastic dropout passes"><UncertaintyChart dates={results.test_dates} actual={results.actual_close} p10={results.mc_p10_close} p90={results.mc_p90_close}/><p className="chart-explain">Monte Carlo dropout runs repeated stochastic forward passes to estimate prediction variation. The displayed interval has not been assessed for statistical calibration.</p></LabPanel><LabPanel title="Residual distribution" subtitle="Predicted close minus actual close · INR"><ResidualChart results={results}/></LabPanel></div>
    <div className="section-caption"><span>02</span> ERROR ANALYSIS <i/></div><div className="metrics-row lab-metrics"><Metric label="MAE · price" value={money(results.metrics.MAE)} detail="Mean absolute error"/><Metric label="MSE · price²" value={Number(results.metrics.MSE).toLocaleString("en-IN", { maximumFractionDigits: 2 })} detail="Mean squared error"/><Metric label="RMSE · price" value={money(results.metrics.RMSE)} detail="Root mean squared error"/><Metric label="Directional accuracy" value={`${results.metrics.DA.toFixed(1)}%`} detail="Return direction" tone="mint"/></div>
    <LabPanel title="Monthly evaluation" subtitle="Metrics recomputed within each calendar month"><MonthlyTable monthly={results.monthly}/></LabPanel><footer className="disclaimer">Evaluation metrics describe the supplied test period and do not imply future performance.</footer></>;
}
function LabPanel({ title, subtitle, children }) { return <section className="panel lab-panel"><div className="panel-head"><div><h2>{title}</h2><p>{subtitle}</p></div></div><div className="lab-chart">{children}</div></section>; }
function ResidualChart({ results }) { const residuals = results.actual_close.map((v, i) => v - results.pred_close[i]); const n = 28, lo = Math.min(...residuals), hi = Math.max(...residuals), width = (hi - lo) / n || 1; const bins = Array(n).fill(0); residuals.forEach((x) => bins[Math.min(n - 1, Math.floor((x - lo) / width))]++); const max = Math.max(...bins); return <div className="histogram" role="img" aria-label={`Distribution of ${residuals.length} price residuals`}><div className="hist-bars">{bins.map((v, i) => <div key={i} title={`${v} observations`} style={{ height: `${Math.max(2, v / max * 100)}%` }}/>)}</div><div className="hist-axis"><span>{money(lo)}</span><span>Residual · INR</span><span>{money(hi)}</span></div><p className="chart-explain">{residuals.length.toLocaleString()} residuals derived as actual close minus predicted close. Values below zero indicate overprediction.</p></div>; }
