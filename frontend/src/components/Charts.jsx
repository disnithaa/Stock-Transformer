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

const axisStyle = { fontSize: 11, fontFamily: "IBM Plex Mono", fill: "#9aa3ad" };
const gridColor = "#262e38";

function tooltipStyle() {
  return {
    background: "#182029",
    border: "1px solid #262e38",
    borderRadius: 4,
    fontFamily: "IBM Plex Mono",
    fontSize: 12,
  };
}

export function TimelineChart({ dates, closes, splitDate, color }) {
  const data = dates.map((d, i) => ({ date: d, close: closes[i] }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={gridColor} vertical={false} />
        <XAxis dataKey="date" tick={axisStyle} minTickGap={60} />
        <YAxis tick={axisStyle} width={56} domain={["auto", "auto"]} />
        <Tooltip contentStyle={tooltipStyle()} labelStyle={{ color: "#e8e6df" }} />
        {splitDate && <ReferenceLine x={splitDate} stroke="#9aa3ad" strokeDasharray="3 3" />}
        <Line type="monotone" dataKey="close" stroke={color} dot={false} strokeWidth={1.4} name="Close" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PredictedVsActualChart({ dates, actual, predicted, color }) {
  const data = dates.map((d, i) => ({ date: d, actual: actual[i], predicted: predicted[i] }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={gridColor} vertical={false} />
        <XAxis dataKey="date" tick={axisStyle} minTickGap={60} />
        <YAxis tick={axisStyle} width={56} domain={["auto", "auto"]} />
        <Tooltip contentStyle={tooltipStyle()} labelStyle={{ color: "#e8e6df" }} />
        <Legend wrapperStyle={{ fontSize: 12, fontFamily: "IBM Plex Sans" }} />
        <Line type="monotone" dataKey="actual" stroke={color} dot={false} strokeWidth={1.6} name="Actual close" />
        <Line type="monotone" dataKey="predicted" stroke="#c9a227" dot={false} strokeWidth={1.2} strokeDasharray="4 3" name="Predicted (1-step)" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function UncertaintyChart({ dates, actual, p10, p90, color }) {
  const data = dates.map((d, i) => ({ date: d, actual: actual[i], band: [p10[i], p90[i]] }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={gridColor} vertical={false} />
        <XAxis dataKey="date" tick={axisStyle} minTickGap={60} />
        <YAxis tick={axisStyle} width={56} domain={["auto", "auto"]} />
        <Tooltip contentStyle={tooltipStyle()} labelStyle={{ color: "#e8e6df" }} />
        <Area dataKey="band" stroke="none" fill="#c9a227" fillOpacity={0.18} name="80% MC-dropout CI" />
        <Line type="monotone" dataKey="actual" stroke={color} dot={false} strokeWidth={1.4} name="Actual close" />
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
        <Legend wrapperStyle={{ fontSize: 12, fontFamily: "IBM Plex Sans" }} />
        <ReferenceLine x={bestEpoch} stroke="#c9a227" strokeDasharray="3 3" label={{ value: "best", fill: "#c9a227", fontSize: 11 }} />
        <Line type="monotone" dataKey="train" stroke="#5b9bd9" dot={false} strokeWidth={1.4} name="Train MSE" />
        <Line type="monotone" dataKey="val" stroke="#e5484d" dot={false} strokeWidth={1.4} name="Validation MSE" />
      </LineChart>
    </ResponsiveContainer>
  );
}
