const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  tickers: () => get("/api/tickers"),
  results: (ticker) => get(`/api/results/${ticker}`),
  predict: (ticker) => get(`/api/predict/${ticker}`),
};
