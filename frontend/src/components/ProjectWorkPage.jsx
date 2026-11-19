import { useEffect, useState } from "react";
import { api } from "../api";
import { COLAB_NOTEBOOK_URL } from "../links";

const money = (value) => Number.isFinite(value) ? `₹${Number(value).toLocaleString("en-IN", { maximumFractionDigits: 2 })}` : "—";

export default function ProjectWorkPage({ onOpenDashboard, onOpenLab }) {
  const [results, setResults] = useState({});
  useEffect(() => {
    let active = true;
    Promise.all(["INFY", "HUL"].map((ticker) => api.results(ticker).then((data) => [ticker, data]).catch(() => [ticker, null])))
      .then((entries) => { if (active) setResults(Object.fromEntries(entries)); });
    return () => { active = false; };
  }, []);

  return <div className="project-page">
    <header className="project-page-hero"><div className="eyebrow">RESEARCH PROJECT / STOCK TRANSFORMER</div><h1>From market history<br/><span>to model evaluation.</span></h1><p>A real-data forecasting project that studies next-session price movement for two NSE-listed companies and makes the model’s limits visible.</p><div className="project-hero-actions"><a className="primary-button anchor-button" href={COLAB_NOTEBOOK_URL} target="_blank" rel="noreferrer">Open the original notebook in Colab ↗</a><button className="secondary-button" onClick={onOpenDashboard}>View evaluation dashboard →</button></div></header>

    <section className="project-result-section"><div className="company-section-head"><div><div className="eyebrow">OUT-OF-SAMPLE RESULTS</div><h2>Same evaluation period · 2025 to latest available data</h2><p>Daily close-to-close, one-session forecasts. Direction scores are compared with the best naïve baseline.</p></div></div>
      <div className="project-result-grid">{["INFY", "HUL"].map((ticker) => { const result = results[ticker]; return <article className="project-result-card" key={ticker}><div className="project-result-symbol">{ticker}</div>{result ? <><small>{result.evaluation_metadata?.evaluation_start} — {result.evaluation_metadata?.evaluation_end}</small><div className="project-result-numbers"><div><span>Direction accuracy</span><b>{result.metrics.DA.toFixed(2)}%</b></div><div><span>Naïve baseline</span><b>{result.evaluation?.directional_baseline_accuracy?.toFixed(2) ?? "—"}%</b></div><div><span>Close MAE</span><b>{money(result.metrics.MAE)}</b></div><div><span>Price-level R²</span><b>{result.metrics.R2.toFixed(4)}</b></div></div><p>Direction comes from the Transformer’s signed return prediction. R² describes close-level fit; it does not establish a tradable directional signal.</p></> : <p>Evaluation data is not currently available from the API.</p>}</article>; })}</div>
    </section>

    <section className="project-work-section"><div className="eyebrow">WHAT THIS PROJECT COVERS</div><h2>Research, modeling, and delivery</h2><div className="work-cards">
      <article><span>01</span><h3>Real market data</h3><p>Daily Yahoo Finance price history for Infosys and Hindustan Unilever, cleaned, time-sorted, and used for causal feature generation.</p><small>2000 onward · NSE listings · 20-session windows</small></article>
      <article><span>02</span><h3>Forecasting pipeline</h3><p>The notebook’s two-layer Transformer estimates return magnitude. A direction method is selected on training-only validation data, then combined with that magnitude.</p><small>9 features · 64-wide embedding · 4 attention heads · chronological validation</small></article>
      <article><span>03</span><h3>Honest evaluation</h3><p>Persistence and moving-average references, logistic direction baselines, confusion metrics, class balance, and uncertainty intervals make the result easier to judge.</p><small>Chronological holdout · one-session horizon</small></article>
      <article><span>04</span><h3>Trading-aware checks</h3><p>A delayed long/flat diagnostic includes a turnover cost and buy-and-hold comparison. It is presented as a close-only research proxy, not as a profitability claim.</p><small>One-session delay · 10 bp per turnover unit</small></article>
      <article><span>05</span><h3>Interactive research app</h3><p>Company profiles, live market and financial data, forecast summaries, model architecture, monthly results, and baseline scorecards are available in one interface.</p><small>React · FastAPI · Yahoo Finance</small></article>
      <article><span>06</span><h3>Reproducibility</h3><p>The training and evaluation code, model checkpoints, precomputed reports, test coverage, and original research notebook are kept with the project.</p><small>Python · PyTorch · scikit-learn</small></article>
    </div></section>

    <section className="project-notebook-card"><div><div className="eyebrow">SOURCE NOTEBOOK</div><h2>Continue in Google Colab</h2><p>Open the original <code>Stock_Transformer_Colab_F.ipynb</code> notebook directly in Colab from the project branch.</p></div><a className="primary-button anchor-button" href={COLAB_NOTEBOOK_URL} target="_blank" rel="noreferrer">Open notebook ↗</a></section>
    <div className="project-footer-actions"><button className="secondary-button" onClick={onOpenDashboard}>Open full evaluation dashboard</button><button className="text-button" onClick={onOpenLab}>Explore the Model Lab ↗</button></div>
    <footer className="disclaimer">For research and education. Daily direction is difficult to predict, and historical model results do not imply future performance.</footer>
  </div>;
}
