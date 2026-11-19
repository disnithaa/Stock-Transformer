import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";

const axisStyle = { fontSize: 10, fontFamily: "DM Mono", fill: "#8290a0" };
const gridColor = "#26313d";

function tooltipStyle() {
  return {
    background: "#19222d",
    border: "1px solid #33404e",
    borderRadius: 7,
    color: "#f2f5f8",
    fontFamily: "DM Mono",
    fontSize: 11,
  };
}

export function TimelineChart({ dates, closes, splitDate }) {
  const data = dates.map((d, i) => ({ date: d, close: closes[i] }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={gridColor} vertical={false} />
        <XAxis dataKey="date" tick={axisStyle} minTickGap={60} />
        <YAxis tick={axisStyle} width={56} domain={["auto", "auto"]} />
        <Tooltip contentStyle={tooltipStyle()} labelStyle={{ color: "#c4ced9" }} formatter={(v) => [`₹${Number(v).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, "Close"]} />
        {splitDate && <ReferenceLine x={splitDate} stroke="#9aa3ad" strokeDasharray="3 3" />}
        <Line type="monotone" dataKey="close" stroke="#62d7b0" dot={false} strokeWidth={1.5} name="Close" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PredictedVsActualChart({ dates, actual, predicted }) {
  const data = dates.map((d, i) => ({ date: d, actual: actual[i], predicted: predicted[i] }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={gridColor} vertical={false} />
        <XAxis dataKey="date" tick={axisStyle} minTickGap={60} />
        <YAxis tick={axisStyle} width={56} domain={["auto", "auto"]} />
        <Tooltip contentStyle={tooltipStyle()} labelStyle={{ color: "#c4ced9" }} formatter={(v, name) => [`₹${Number(v).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, name]} />
        <Legend wrapperStyle={{ fontSize: 10, fontFamily: "DM Sans" }} />
        <Line type="monotone" dataKey="actual" stroke="#62d7b0" dot={false} strokeWidth={1.6} name="Actual close" />
        <Line type="monotone" dataKey="predicted" stroke="#a5b4fc" dot={false} strokeWidth={1.3} strokeDasharray="5 4" name="Predicted (1-step)" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function UncertaintyChart({ dates, actual, p10, p90 }) {
  const data = dates.map((d, i) => ({ date: d, actual: actual[i], band: [p10[i], p90[i]] }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={gridColor} vertical={false} />
        <XAxis dataKey="date" tick={axisStyle} minTickGap={60} />
        <YAxis tick={axisStyle} width={56} domain={["auto", "auto"]} />
        <Tooltip contentStyle={tooltipStyle()} labelStyle={{ color: "#c4ced9" }} />
        <Area dataKey="band" stroke="#a5b4fc" strokeWidth={1} fill="#a5b4fc" fillOpacity={0.16} name="MC dropout interval" />
        <Line type="monotone" dataKey="actual" stroke="#62d7b0" dot={false} strokeWidth={1.4} name="Actual close" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function LossChart({ trainLosses, valLosses, bestEpoch }) {
  const data = trainLosses.map((t, i) => ({ epoch: i, train: t, val: valLosses[i] }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={gridColor} vertical={false} />
        <XAxis dataKey="epoch" tick={axisStyle} />
        <YAxis tick={axisStyle} width={48} />
        <Tooltip contentStyle={tooltipStyle()} labelStyle={{ color: "#e8e6df" }} />
        <Legend wrapperStyle={{ fontSize: 10, fontFamily: "DM Sans" }} />
        <ReferenceLine x={bestEpoch} stroke="#f2be68" strokeDasharray="3 3" label={{ value: "best", fill: "#f2be68", fontSize: 10 }} />
        <Line type="monotone" dataKey="train" stroke="#62d7b0" dot={false} strokeWidth={1.4} name="Train MSE" />
        <Line type="monotone" dataKey="val" stroke="#a5b4fc" dot={false} strokeWidth={1.4} name="Validation MSE" />
      </LineChart>
    </ResponsiveContainer>
  );
}
