import { TICKER_META } from "../tickerMeta";
import PredictionPanel from "./PredictionPanel";

export default function Sidebar({ ticker, onSelect, tickers }) {
  return (
    <div className="sidebar">
      <div>
        <div className="brand">Stock Transformer</div>
        <div className="brand-sub">Attention-based next-day forecasting for NSE-listed equities</div>
      </div>

      <div className="ticker-switch">
        {tickers.map((t) => {
          const meta = TICKER_META[t] || { name: t, desc: "", color: "var(--gold)" };
          return (
            <button
              key={t}
              className={`ticker-btn ${ticker === t ? "active" : ""}`}
              style={{ "--dot-color": meta.color }}
              onClick={() => onSelect(t)}
            >
              <div className="ticker-name">
                <span className="dot" style={{ background: meta.color }} />
                {t}
              </div>
              <div className="ticker-desc">
                {meta.name} · {meta.desc}
              </div>
            </button>
          );
        })}
      </div>

      <PredictionPanel ticker={ticker} />
    </div>
  );
}
