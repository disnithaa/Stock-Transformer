import { TICKER_META } from "../tickerMeta";
import { TimelineChart } from "./Charts";
import TestingCalendar from "./TestingCalendar";
import { COMPANY_VISUALS } from "../companyVisuals";

const numeric = (value) => Number.isFinite(value) ? value : null;
const plainNumber = (value) => numeric(value) == null ? "—" : Number(value).toLocaleString("en-IN", { maximumFractionDigits: 2 });
const compactMoney = (value, currency = "INR") => numeric(value) == null ? "—" : new Intl.NumberFormat("en-IN", { style: "currency", currency, notation: "compact", maximumFractionDigits: 2 }).format(value);
const closeMoney = (value, currency = "INR") => numeric(value) == null ? "—" : new Intl.NumberFormat("en-IN", { style: "currency", currency, minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
const signedPct = (value) => numeric(value) == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
const dateLabel = (value) => value ? new Date(`${value}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" }) : "—";

export default function CompanyPage({ ticker, data, status, error, evaluationResults, onBack, onRetry, onOpenTestDate }) {
  const meta = TICKER_META[ticker];
  const quote = data?.quote || {};
  const valuation = data?.valuation || {};
  const financials = data?.annual_financials || {};
  const financialCurrency = data?.financial_currency || data?.currency || "INR";
  const evaluation = evaluationResults?.ticker === ticker ? evaluationResults : null;
  const evaluationMeta = evaluation?.evaluation_metadata;
  const visual = COMPANY_VISUALS[ticker];
  const changeTone = quote.change_pct > 0 ? "up" : quote.change_pct < 0 ? "down" : "";

  return <div className="company-page">
    <button className="back-link" onClick={onBack}>← <span>All companies</span></button>
    <header className="company-page-heading">
      <div><div className="eyebrow">COMPANY PROFILE / NSE EQUITY</div><h1>{data?.company_name || meta.name}<span>{ticker}</span></h1><p>{data?.sector || meta.desc}{data?.industry ? ` · ${data.industry}` : ""}{data?.exchange ? ` · ${data.exchange}` : ""}</p></div>
      {data?.website && <a className="secondary-button company-site-link" href={data.website} target="_blank" rel="noreferrer">Official website ↗</a>}
    </header>

    <section className="company-visual" aria-label={`${meta.name} company image`}>
      <img src={visual.image} alt={visual.alt} loading="lazy" referrerPolicy="no-referrer"/>
      <div className="company-visual-shade"><div className="eyebrow">COMPANY CONTEXT</div><h2>{visual.caption}</h2></div>
      <a className="company-visual-credit" href={visual.source} target="_blank" rel="noreferrer">Photo: {visual.credit} ↗</a>
    </section>

    {status === "loading" && <div className="company-loading"><span className="spinner"/> Loading live company and market data…</div>}
    {status === "error" && <div className="company-data-error"><div><b>Live financial data is unavailable</b><p>{error || "Yahoo Finance did not return a current quote. No estimates are shown in its place."}</p></div><button className="secondary-button" onClick={onRetry}>Try again</button></div>}

    {data && <>
      <section className="company-quote-panel">
        <div className="company-last-price"><small>LATEST CLOSE · {dateLabel(quote.as_of_date)}</small><strong>{closeMoney(quote.last_close, data.currency)}</strong><span className={`quote-change ${changeTone}`}>{quote.change > 0 ? "+" : ""}{closeMoney(quote.change, data.currency)} <b>({signedPct(quote.change_pct)})</b> vs previous close</span></div>
        <div className="quote-panel-divider"/>
        <div className="quote-mini-stat"><small>ONE-YEAR CHANGE</small><b className={quote.one_year_return_pct > 0 ? "up" : quote.one_year_return_pct < 0 ? "down" : ""}>{signedPct(quote.one_year_return_pct)}</b><span>Trailing 12 months</span></div>
        <div className="quote-mini-stat"><small>52-WEEK RANGE</small><b>{closeMoney(quote.week_52_low, data.currency)} — {closeMoney(quote.week_52_high, data.currency)}</b><span>Daily traded-price range</span></div>
        <div className="quote-mini-stat"><small>MARKET CAPITALIZATION</small><b>{compactMoney(valuation.market_cap, data.currency)}</b><span>Yahoo Finance reported value</span></div>
      </section>

      <section className="company-chart-card">
        <div className="company-section-head"><div><div className="eyebrow">PRICE HISTORY</div><h2>One-year closing prices</h2><p>Daily close · {data.currency} · data through {dateLabel(quote.as_of_date)}</p></div><span className="neutral-tag">Yahoo Finance</span></div>
        <div className="company-chart"><TimelineChart dates={data.history_dates} closes={data.history_close}/></div>
      </section>

      <TestingCalendar dates={evaluation?.test_dates || []} trainingStart={evaluationMeta?.training_start} trainingEnd={evaluationMeta?.training_end} testStart={evaluationMeta?.test_start} testEnd={evaluationMeta?.test_end} onOpenDate={onOpenTestDate}/>

      <div className="company-content-grid">
        <section className="company-info-card">
          <div className="eyebrow">BUSINESS PROFILE</div><h2>About {data.company_name || meta.name}</h2>
          <p className="business-summary">{data.summary || "Yahoo Finance did not provide a company description for this listing."}</p>
          <div className="profile-facts"><Fact label="Sector" value={data.sector}/><Fact label="Industry" value={data.industry}/><Fact label="Country" value={data.country}/><Fact label="Full-time employees" value={data.employees?.toLocaleString("en-IN")}/></div>
        </section>
        <section className="company-info-card">
          <div className="eyebrow">VALUATION & TRADING</div><h2>Market statistics</h2>
          <div className="finance-stat-grid">
            <FinanceStat label="Trailing P/E" value={plainNumber(valuation.trailing_pe)}/><FinanceStat label="Forward P/E" value={plainNumber(valuation.forward_pe)}/>
            <FinanceStat label="Price / book" value={plainNumber(valuation.price_to_book)}/><FinanceStat label="Enterprise value" value={compactMoney(valuation.enterprise_value, data.currency)}/>
            <FinanceStat label="EV / EBITDA" value={plainNumber(valuation.enterprise_to_ebitda)}/><FinanceStat label="Beta" value={plainNumber(valuation.beta)}/>
            <FinanceStat label="Dividend yield" value={numeric(valuation.dividend_yield_pct) == null ? "—" : `${valuation.dividend_yield_pct.toFixed(2)}%`}/><FinanceStat label="Average volume · 20d" value={quote.average_volume_20d?.toLocaleString("en-IN") || "—"}/>
            <FinanceStat label="50-day average" value={closeMoney(quote.moving_average_50d, data.currency)}/><FinanceStat label="200-day average" value={closeMoney(quote.moving_average_200d, data.currency)}/>
          </div>
        </section>
      </div>

      <section className="company-info-card annual-card">
          <div className="company-section-head"><div><div className="eyebrow">LATEST REPORTED FINANCIALS</div><h2>Annual statement snapshot</h2><p>Figures shown in {financialCurrency}; statement period and values are supplied by Yahoo Finance.</p></div></div>
        <div className="annual-grid">
          <AnnualStat label="Revenue" entry={financials.revenue} currency={financialCurrency}/><AnnualStat label="Gross profit" entry={financials.gross_profit} currency={financialCurrency}/><AnnualStat label="Operating income" entry={financials.operating_income} currency={financialCurrency}/><AnnualStat label="Net income" entry={financials.net_income} currency={financialCurrency}/><AnnualStat label="Diluted EPS" entry={financials.diluted_eps} currency={financialCurrency} perShare/><AnnualStat label="Operating cash flow" entry={financials.operating_cash_flow} currency={financialCurrency}/><AnnualStat label="Free cash flow" entry={financials.free_cash_flow} currency={financialCurrency}/><AnnualStat label="Total assets" entry={financials.total_assets} currency={financialCurrency}/><AnnualStat label="Total debt" entry={financials.total_debt} currency={financialCurrency}/><AnnualStat label="Stockholders’ equity" entry={financials.stockholders_equity} currency={financialCurrency}/><AnnualStat label="Cash & equivalents" entry={financials.cash_and_equivalents} currency={financialCurrency}/>
        </div>
      </section>
      <p className="company-data-note">Data source: {data.data_source}. Quote as of {dateLabel(quote.as_of_date)}; page refreshed {new Date(data.fetched_at).toLocaleString("en-IN")}. Financial statement values may be reported for an earlier fiscal period. This page is for research, not investment advice.</p>
    </>}
  </div>;
}

function Fact({ label, value }) { return <div><small>{label}</small><b>{value || "—"}</b></div>; }
function FinanceStat({ label, value }) { return <div className="finance-stat"><small>{label}</small><b>{value || "—"}</b></div>; }
function AnnualStat({ label, entry, currency, perShare = false }) {
  const value = entry?.value;
  const display = numeric(value) == null ? "—" : perShare ? closeMoney(value, currency) : compactMoney(value, currency);
  return <div className="annual-stat"><small>{label}</small><b>{display}</b><span>{entry?.period ? `Period ending ${dateLabel(entry.period)}` : "Period unavailable"}</span></div>;
}
