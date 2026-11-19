import { TICKER_META } from "../tickerMeta";
import { COLAB_NOTEBOOK_URL } from "../links";
import { COMPANY_VISUALS } from "../companyVisuals";

const COMPANIES = [
  { ticker: "INFY", number: "01", focus: "Technology · IT services", prompt: "Explore Infosys", logo: "https://www.infosys.com/content/dam/infosys-web/en/global-resource/media-resources/2026/infosys-logo-svg.svg" },
  { ticker: "HUL", number: "02", focus: "Consumer goods · FMCG", prompt: "Explore HUL", logo: "https://www.companieslogo.com/img/orig/HINDUNILVR.NS_BIG-656edf29.svg" },
];

export default function HomePage({ onSelectCompany, onOpenProject }) {
  return <div className="home-page">
    <section className="home-hero">
      <div className="eyebrow"><span className="home-live-dot"/> STOCK TRANSFORMER / COMPANY RESEARCH</div>
      <h1>Two companies.<br/><span>One research workspace.</span></h1>
      <p>Explore company profiles, current market data, financial fundamentals, and the forecasting work behind this project.</p>
      <div className="home-hero-meta"><span>INDIA · NSE</span><i/> <span>REAL MARKET DATA</span><i/> <span>TRANSFORMER RESEARCH</span></div>
      <div className="home-orbit home-orbit-one"/><div className="home-orbit home-orbit-two"/>
    </section>

    <div className="home-section-head"><div><div className="eyebrow">CHOOSE A COMPANY</div><h2>Company profiles</h2></div><span>01 — 02</span></div>
    <section className="company-choice-grid">
      {COMPANIES.map((company) => {
        const meta = TICKER_META[company.ticker];
        const visual = COMPANY_VISUALS[company.ticker];
        return <button className={`company-choice-card ${company.ticker.toLowerCase()}`} key={company.ticker} onClick={() => onSelectCompany(company.ticker)}>
          <img className="choice-photo" src={visual.image} alt="" aria-hidden="true" loading="lazy" referrerPolicy="no-referrer"/>
          <div className="choice-card-top"><span>{company.number} / COMPANY</span><span className="choice-arrow">↗</span></div>
          <div className="choice-logo-plate"><img src={company.logo} alt={`${meta.name} logo`} loading="lazy" referrerPolicy="no-referrer"/></div>
          <div className="choice-card-bottom"><div><small>{company.focus}</small><h3>{meta.name}</h3><span className="choice-symbol">NSE:{company.ticker === "HUL" ? "HINDUNILVR" : company.ticker}</span></div><b>{company.prompt} <span>→</span></b></div>
        </button>;
      })}
    </section>

    <section className="project-entry-card">
      <div className="project-entry-icon">ST</div>
      <div className="project-entry-copy"><div className="eyebrow">THE RESEARCH BEHIND THE APP</div><h2>Explore the project work</h2><p>Model design, real-data pipeline, evaluation results, baselines, and implementation notes in one place.</p></div>
      <button className="secondary-button" onClick={onOpenProject}>View project details <span>↗</span></button>
    </section>
    <div className="home-notebook-line"><span>Prefer the original notebook workflow?</span><a href={COLAB_NOTEBOOK_URL} target="_blank" rel="noreferrer">Open the Colab notebook ↗</a></div>
  </div>;
}
