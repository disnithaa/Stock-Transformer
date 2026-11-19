import React from "react";

const weekDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const monthKey = (date) => `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
const fromIso = (value) => new Date(`${value}T00:00:00`);
const dateLabel = (value) => value ? fromIso(value).toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" }) : "—";

export default function TestingCalendar({ dates = [], trainingStart, trainingEnd, testStart, testEnd, onOpenDate }) {
  const [selectedDate, setSelectedDate] = React.useState("");
  const [visibleMonth, setVisibleMonth] = React.useState("");
  // The report's fixed holdout ends on June 30, 2026. Ignore any later rows
  // still present in an older precomputed payload until it is regenerated.
  const reportDates = dates.filter((date) => date >= "2025-01-01" && date <= "2026-06-30");
  const dateSet = new Set(reportDates);
  const firstMonth = reportDates.length ? monthKey(fromIso(reportDates[0])) : "";
  const lastMonth = reportDates.length ? monthKey(fromIso(reportDates.at(-1))) : "";
  const currentMonth = visibleMonth && visibleMonth >= firstMonth && visibleMonth <= lastMonth ? visibleMonth : lastMonth;
  const [year, month] = currentMonth ? currentMonth.split("-").map(Number) : [0, 0];
  const firstWeekday = currentMonth ? (new Date(year, month - 1, 1).getDay() + 6) % 7 : 0;
  const daysInMonth = currentMonth ? new Date(year, month, 0).getDate() : 0;
  const cells = [...Array(firstWeekday).fill(null), ...Array.from({ length: daysInMonth }, (_, index) => index + 1)];
  const previousMonth = currentMonth ? monthKey(new Date(year, month - 2, 1)) : "";
  const nextMonth = currentMonth ? monthKey(new Date(year, month, 1)) : "";
  const canGoPrevious = previousMonth >= firstMonth;
  const canGoNext = nextMonth <= lastMonth;

  return <section className="test-calendar-panel">
    <div className="company-section-head"><div><div className="eyebrow">HELD-OUT TEST DATA</div><h2>Choose a test session</h2><p>Only market sessions with saved actual and predicted values are selectable.</p></div><span className="neutral-tag">{reportDates.length} sessions</span></div>
    <div className="data-period-strip"><div><small>TRAINING</small><b>{dateLabel(trainingStart || "2000-01-01")} — {dateLabel(trainingEnd || "2024-12-31")}</b></div><div><small>TESTING</small><b>{dateLabel(testStart || "2025-01-01")} — {dateLabel(testEnd || "2026-06-30")}</b></div></div>
    {reportDates.length ? <div className="calendar-analysis-layout">
      <div className="test-calendar">
        <div className="calendar-month-nav"><button onClick={() => setVisibleMonth(previousMonth)} disabled={!canGoPrevious} aria-label="Previous month">←</button><b>{currentMonth ? new Date(year, month - 1, 1).toLocaleDateString("en-IN", { month: "long", year: "numeric" }) : "Test period"}</b><button onClick={() => setVisibleMonth(nextMonth)} disabled={!canGoNext} aria-label="Next month">→</button></div>
        <div className="calendar-grid calendar-weekdays">{weekDays.map((day) => <span key={day}>{day}</span>)}</div>
        <div className="calendar-grid">{cells.map((day, index) => {
          const iso = day ? `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}` : "";
          const enabled = Boolean(iso && dateSet.has(iso));
          return <button key={iso || `empty-${index}`} className={`${enabled ? "has-test" : ""} ${selectedDate === iso ? "selected" : ""}`} disabled={!enabled} onClick={() => setSelectedDate(iso)} aria-label={enabled ? `View test data for ${dateLabel(iso)}` : undefined}>{day || ""}</button>;
        })}</div>
        <div className="calendar-legend"><span><i/> Test observation available</span><small>Non-trading days and dates outside the test split are disabled.</small></div>
      </div>
      <div className="calendar-selection-card"><div className="eyebrow">SELECTED SESSION</div><strong>{selectedDate ? dateLabel(selectedDate) : "Choose a highlighted date"}</strong><p>{selectedDate ? "Open a dedicated page with the actual-versus-predicted chart and session statistics." : "The calendar only enables dates that have an out-of-sample prediction."}</p><button className="primary-button" disabled={!selectedDate} onClick={() => onOpenDate(selectedDate)}>View date analysis <span>→</span></button></div>
    </div> : <div className="calendar-empty">No saved test sessions are available for the report period yet.</div>}
  </section>;
}
