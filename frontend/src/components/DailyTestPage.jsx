import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { TICKER_META } from "../tickerMeta";

const money = (value) => Number.isFinite(value) ? `₹${Number(value).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—";
const dateFmt = (value) => value ? new Date(`${value}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" }) : "—";
const returnPct = (value) => Number.isFinite(value) ? `${(Math.expm1(value) * 100) > 0 ? "+" : ""}${(Math.expm1(value) * 100).toFixed(3)}%` : "—";

export default function DailyTestPage({ ticker, results, date, onBack }) {
  const meta = TICKER_META[ticker] || { name: ticker };
  const index = results?.test_dates?.indexOf(date) ?? -1;
  if (index < 0) {
    return <div className="daily-test-page"><button className="back-link" onClick={onBack}>← <span>Back to {meta.name}</span></button><div className="company-data-error"><div><b>Test data not found for this date</b><p>Choose a highlighted session from the company page calendar.</p></div></div></div>;
  }

  const actual = results.actual_close[index];
  const predicted = results.pred_close[index];
  const priorClose = results.prior_close?.[index] ?? (index > 0 ? results.actual_close[index - 1] : undefined);
  const actualReturn = results.actual_return?.[index] ?? (Number.isFinite(priorClose) ? Math.log(actual / priorClose) : undefined);
  const predictedReturn = results.predicted_return?.[index] ?? (Number.isFinite(priorClose) ? Math.log(predicted / priorClose) : undefined);
  const actualDirection = actualReturn > 0 ? "Up" : actualReturn < 0 ? "Down" : "Flat";
  const predictedDirection = predictedReturn > 0 ? "Up" : predictedReturn < 0 ? "Down" : "Flat";
  const directionCorrect = Number.isFinite(actualReturn) && Number.isFinite(predictedReturn) && actualDirection === predictedDirection;
  const error = predicted - actual;
  const p10 = results.mc_p10_close?.[index];
  const p90 = results.mc_p90_close?.[index];
  const confidence = results.direction_confidence?.[index];
  const intervalContainsActual = Number.isFinite(p10) && Number.isFinite(p90) && actual >= p10 && actual <= p90;
  const windowStart = Math.max(0, index - 10);
  const windowEnd = Math.min(results.test_dates.length, index + 11);
  const chartData = results.test_dates.slice(windowStart, windowEnd).map((day, offset) => {
    const row = windowStart + offset;
    return { date: day, actual: results.actual_close[row], predicted: results.pred_close[row] };
  });

  return <div className="daily-test-page">
    <button className="back-link" onClick={onBack}>← <span>Back to {meta.name} company page</span></button>
    <header className="daily-test-heading"><div><div className="eyebrow">{ticker} / HELD-OUT TEST SESSION</div><h1>{dateFmt(date)}<span>1-day forecast</span></h1><p>Forecast made from the previous known close using the model frozen before the test period.</p></div><div className={`daily-direction-pill ${directionCorrect ? "correct" : "incorrect"}`}><small>DIRECTION RESULT</small><b>{directionCorrect ? "Correct" : Number.isFinite(actualReturn) ? "Incorrect" : "Unavailable"}</b><span>{Number.isFinite(actualReturn) ? `Predicted ${predictedDirection.toLowerCase()} · actual ${actualDirection.toLowerCase()}` : "Previous-session close is unavailable"}</span></div></header>

    <section className="daily-stat-grid">
      <DailyStat label="Actual close" value={money(actual)} detail={`Observed ${dateFmt(date)}`}/>
      <DailyStat label="Predicted close" value={money(predicted)} detail="Direct Transformer return forecast"/>
      <DailyStat label="Absolute price error" value={money(Math.abs(error))} detail={`${error > 0 ? "Over" : error < 0 ? "Under" : "Exact"} prediction by ${money(Math.abs(error))}`}/>
      <DailyStat label="Previous known close" value={money(priorClose)} detail="Forecast origin price"/>
      <DailyStat label="Actual next-session return" value={returnPct(actualReturn)} detail="Close-to-close log return"/>
      <DailyStat label="Predicted return" value={returnPct(predictedReturn)} detail="Signed return from the Transformer"/>
      <DailyStat label="80% dropout range" value={`${money(p10)} — ${money(p90)}`} detail={intervalContainsActual ? "Actual close is inside this uncalibrated range" : "Actual close is outside this uncalibrated range"}/>
      <DailyStat label="Direction confidence" value={Number.isFinite(confidence) ? `${(confidence * 100).toFixed(1)}%` : "Not estimated"} detail={Number.isFinite(confidence) ? "Classifier probability; not calibrated" : `${results.direction_model_name} does not output probabilities`}/>
    </section>

    <section className="daily-chart-card"><div className="company-section-head"><div><div className="eyebrow">LOCAL TEST WINDOW</div><h2>Actual vs predicted closing price</h2><p>Selected session marked on the line chart · daily close in INR</p></div><span className="neutral-tag">{chartData.length} sessions</span></div><div className="daily-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData} margin={{ top: 8, right: 18, bottom: 3, left: 3 }}><CartesianGrid stroke="#26313d" vertical={false}/><XAxis dataKey="date" tick={{ fontSize: 9, fill: "#8290a0" }} minTickGap={28}/><YAxis tick={{ fontSize: 9, fill: "#8290a0" }} width={62} domain={["auto", "auto"]}/><Tooltip contentStyle={{ background: "#19222d", border: "1px solid #33404e", borderRadius: 7, color: "#f2f5f8", fontSize: 11 }} formatter={(value, name) => [money(value), name]}/><Legend wrapperStyle={{ fontSize: 10 }}/><ReferenceLine x={date} stroke="#f2be68" strokeDasharray="4 3" label={{ value: "Selected date", fill: "#f2be68", fontSize: 9 }}/><Line type="monotone" dataKey="actual" stroke="#62d7b0" dot={false} strokeWidth={2} name="Actual close"/><Line type="monotone" dataKey="predicted" stroke="#a5b4fc" dot={false} strokeWidth={1.7} strokeDasharray="5 4" name="Predicted close"/></LineChart></ResponsiveContainer></div></section>
    <p className="daily-test-note">Evaluation period: {results.evaluation_metadata?.test_start} to {results.evaluation_metadata?.test_end}. The uncertainty interval is based on Monte Carlo dropout and has not been calibrated. This single-session result is descriptive, not investment advice.</p>
  </div>;
}

function DailyStat({ label, value, detail }) { return <article className="daily-stat-card"><small>{label}</small><b>{value}</b><span>{detail}</span></article>; }
