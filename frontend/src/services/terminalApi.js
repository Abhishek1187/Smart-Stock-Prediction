import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 45000,
});

const predictApi = axios.create({
  baseURL: "/predict",
  timeout: 45000,
});

export const fetchMarketOverview = async () => {
  const { data } = await api.get("/market-overview/");
  return data;
};

export const fetchStocksUniverse = async () => {
  const { data } = await api.get("/stocks/");
  return Array.isArray(data) ? data : [];
};

export const fetchPriceHistory = async (symbol, days = 365) => {
  const { data } = await api.get("/price-history/", { params: { symbol, days } });
  return data;
};

export const fetchTechnicalIndicators = async (symbol) => {
  const { data } = await api.get("/technical-indicators/", { params: { symbol } });
  return data;
};

export const fetchSentiment = async (symbol, days = 180) => {
  const { data } = await api.get("/sentiment/", { params: { symbol, days } });
  return data;
};

export const fetchPredictionHistory = async (symbol, horizon_days = 1) => {
  const { data } = await api.get("/prediction/", { params: { symbol, horizon_days } });
  return data;
};

export const fetchAdvancedAnalytics = async (symbol, days = 365) => {
  const { data } = await api.get("/advanced-analytics/", { params: { symbol, days } });
  return data;
};

export const fetchModelComparison = async (symbol) => {
  const { data } = await api.get("/model_comparison/", { params: { symbol } });
  return data;
};

export const fetchLivePrediction = async (symbol, model = "transformer") => {
  const { data } = await predictApi.get(`/${encodeURIComponent(symbol)}/`, {
    params: { model },
  });
  return data;
};
