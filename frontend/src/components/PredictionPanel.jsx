import { useState } from "react";
import { api } from "../api";

export default function PredictionPanel({ ticker }) {
  const [state, setState] = useState({ status: "idle", data: null, error: null });

  async function runPrediction() {
    setState({ status: "loading", data: null, error: null });
    try {
      const data = await api.predict(ticker);
      setState({ status: "done", data, error: null });
    } catch (e) {
      setState({ status: "error", data: null, error: e.message });
    }
  }

  const d = state.data;
  const delta = d ? d.predicted_next_close - d.last_close : null;
  const deltaClass = delta == null ? "" : delta >= 0 ? "up" : "down";

  return (
    <div className="predict-panel">
      <div className="predict-label">Live next-day prediction — {ticker}</div>

      {state.status === "idle" && (
        <div className="predict-note">Runs the trained model against the latest available close to forecast tomorrow's price, with an 80% confidence band from Monte Carlo dropout.</div>
      )}

      {state.status === "loading" && <div className="predict-note">Fetching latest data and running inference…</div>}

      {state.status === "error" && <div className="predict-note" style={{ color: "var(--down)" }}>{state.error}</div>}

      {state.status === "done" && d && (
        <>
          <div className="predict-value mono">₹{d.predicted_next_close.toLocaleString()}</div>
          <div className={`predict-delta mono ${deltaClass}`}>
            {delta >= 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(2)} ({d.predicted_return_pct}%) vs last close ₹{d.last_close}
          </div>
          <div className="predict-range">
            80% CI: ₹{d.ci80_low} – ₹{d.ci80_high}
            <br />
            as of {d.as_of_date}
          </div>
        </>
      )}

      <button className="predict-btn" onClick={runPrediction} disabled={state.status === "loading"}>
        {state.status === "done" ? "Run again" : "Run prediction"}
      </button>
    </div>
  );
}
