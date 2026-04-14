import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactApexChart from "react-apexcharts";
import {
  ResponsiveContainer,
  ComposedChart,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Bar,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
} from "recharts";
import {
  fetchAdvancedAnalytics,
  fetchLivePrediction,
  fetchMarketOverview,
  fetchPriceHistory,
  fetchStocksUniverse,
  fetchPredictionHistory,
  fetchSentiment,
  fetchTechnicalIndicators,
} from "../services/terminalApi";

const TIMEFRAMES = [
  { label: "6M", days: 180 },
  { label: "1Y", days: 365 },
  { label: "2Y", days: 730 },
  { label: "5Y", days: 1825 },
];

const POSITIVE = "#22c55e";
const NEGATIVE = "#fb7185";
const NEUTRAL = "#f59e0b";

const formatPct = (value) => {
  if (value === null || value === undefined) return "-";
  return `${(value * 100).toFixed(2)}%`;
};

const miniNumber = (value) => {
  if (value === null || value === undefined) return "-";
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(value);
};

const formatDateTime = (value) => {
  if (!value) return "-";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return String(value);
  return dt.toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const numOrNull = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const buildCurrentComparisonSeries = (latestActual, lstmPrediction, transformerPrediction) => {
  const actual = numOrNull(latestActual);
  const lstm = numOrNull(lstmPrediction);
  const transformer = numOrNull(transformerPrediction);

  return [
    { model: "Actual", price: actual, color: "#2dd4bf" },
    { model: "LSTM", price: lstm, color: "#f59e0b" },
    { model: "Transformer", price: transformer, color: "#38bdf8" },
  ];
};

const downsampleSeries = (rows, maxPoints) => {
  if (!Array.isArray(rows) || rows.length <= maxPoints) return rows || [];
  const step = Math.ceil(rows.length / maxPoints);
  return rows.filter((_, index) => index % step === 0 || index === rows.length - 1);
};

const formatNumberCompact = (value) => {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 10000000) return `${(value / 10000000).toFixed(2)}Cr`;
  if (Math.abs(value) >= 100000) return `${(value / 100000).toFixed(2)}L`;
  return miniNumber(value);
};

const mergeTickerUniverse = (overviewTickers = [], universe = []) => {
  const map = new Map();
  universe.forEach((item) => {
    if (item?.symbol) map.set(item.symbol, { ...item });
  });
  overviewTickers.forEach((item) => {
    if (!item?.symbol) return;
    map.set(item.symbol, { ...(map.get(item.symbol) || {}), ...item });
  });
  return Array.from(map.values());
};

const TerminalDashboard = () => {
  const [timeframe, setTimeframe] = useState(365);
  const [overview, setOverview] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState("RELIANCE.NS");
  const [allStocks, setAllStocks] = useState([]);
  const [priceHistory, setPriceHistory] = useState([]);
  const [sentiment, setSentiment] = useState([]);
  const [technical, setTechnical] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [livePredictions, setLivePredictions] = useState({ lstm: null, transformer: null });
  const [advancedAnalytics, setAdvancedAnalytics] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const refreshRef = useRef({ requestId: 0 });
  const lastLoadedKeyRef = useRef("");
  const modelRefreshRef = useRef({ id: 0 });

  const loadOverview = async () => {
    const [overviewResult, universeResult] = await Promise.allSettled([
      fetchMarketOverview(),
      fetchStocksUniverse(),
    ]);

    const overviewData = overviewResult.status === "fulfilled" ? overviewResult.value : null;
    const universeData = universeResult.status === "fulfilled" ? universeResult.value : [];
    const mergedTickers = mergeTickerUniverse(overviewData?.tickers || [], universeData || []);

    if (overviewData) {
      setOverview({ ...overviewData, tickers: mergedTickers });
      if (!selectedSymbol && mergedTickers.length) {
        setSelectedSymbol(mergedTickers[0].symbol);
      }
    }
    setAllStocks(universeData || []);

    if (!overviewData && !overview) {
      throw new Error("Market overview unavailable");
    }
  };

  const loadSymbolData = async (symbol, days) => {
    const [priceResult, sentimentResult, indicatorResult, predictionResult, advancedResult] = await Promise.allSettled([
      fetchPriceHistory(symbol, days),
      fetchSentiment(symbol, days),
      fetchTechnicalIndicators(symbol),
      fetchPredictionHistory(symbol, 1),
      fetchAdvancedAnalytics(symbol, days),
    ]);

    return {
      priceResult,
      sentimentResult,
      indicatorResult,
      predictionResult,
      advancedResult,
    };
  };

  const applySymbolPayload = (payload, { preserveOnFailure }) => {
    const { priceResult, sentimentResult, indicatorResult, predictionResult, advancedResult } = payload;

    if (priceResult.status === "fulfilled") {
      const normalizedPrice = (priceResult.value?.series || []).map((row) => ({
        ...row,
        open: numOrNull(row.open),
        high: numOrNull(row.high),
        low: numOrNull(row.low),
        close: numOrNull(row.close),
        volume: numOrNull(row.volume),
        return_1d: numOrNull(row.return_1d),
        return_7d: numOrNull(row.return_7d),
        return_14d: numOrNull(row.return_14d),
        return_30d: numOrNull(row.return_30d),
        volatility_14d: numOrNull(row.volatility_14d),
        relative_strength: numOrNull(row.relative_strength),
      }));
      setPriceHistory(normalizedPrice);
    } else if (!preserveOnFailure) {
      setPriceHistory([]);
    }

    if (sentimentResult.status === "fulfilled") {
      const normalizedSentiment = (sentimentResult.value?.series || []).map((row) => ({
        ...row,
        sentiment_mean: numOrNull(row.sentiment_mean),
        sentiment_std: numOrNull(row.sentiment_std),
        news_count: numOrNull(row.news_count),
        positive_ratio: numOrNull(row.positive_ratio),
      }));
      setSentiment(normalizedSentiment);
    } else if (!preserveOnFailure) {
      setSentiment([]);
    }

    if (indicatorResult.status === "fulfilled") {
      const latest = indicatorResult.value?.latest;
      const normalizedTechnical = latest
        ? {
            ...latest,
            close: numOrNull(latest.close),
            return_1d: numOrNull(latest.return_1d),
            return_7d: numOrNull(latest.return_7d),
            return_14d: numOrNull(latest.return_14d),
            return_30d: numOrNull(latest.return_30d),
            volatility_14d: numOrNull(latest.volatility_14d),
            momentum_10d: numOrNull(latest.momentum_10d),
            volume_change_1d: numOrNull(latest.volume_change_1d),
            relative_strength: numOrNull(latest.relative_strength),
            atr_14: numOrNull(latest.atr_14),
          }
        : null;
      setTechnical(normalizedTechnical);
    } else if (!preserveOnFailure) {
      setTechnical(null);
    }

    if (predictionResult.status === "fulfilled") {
      setPredictionHistory(predictionResult.value?.series || []);
    } else if (!preserveOnFailure) {
      setPredictionHistory([]);
    }

    if (advancedResult.status === "fulfilled") {
      setAdvancedAnalytics(advancedResult.value || null);
    } else if (!preserveOnFailure) {
      setAdvancedAnalytics(null);
    }

    if (
      priceResult.status !== "fulfilled" &&
      sentimentResult.status !== "fulfilled" &&
      indicatorResult.status !== "fulfilled" &&
      predictionResult.status !== "fulfilled" &&
      advancedResult.status !== "fulfilled"
    ) {
      throw new Error("All symbol services unavailable");
    }
  };

  const runLivePrediction = async (modelKey) => {
    try {
      setPredictionLoading(true);
      setError("");
      const result = await fetchLivePrediction(selectedSymbol, modelKey);
      setLivePredictions((prev) => ({ ...prev, [modelKey]: { ...result, _runAt: new Date().toISOString() } }));
      setPredictionHistory((prev) => {
        const next = [
          {
            date: new Date().toISOString().slice(0, 10),
            model_name: (result.model_name || modelKey).toUpperCase(),
            predicted_close: result.predicted_close,
            actual_close: result.actual_close,
            rmse: result?.model_metadata?.training_rmse ?? null,
            mae: result?.model_metadata?.training_mae ?? null,
            mape: null,
            directional_accuracy: null,
          },
          ...(prev || []),
        ];
        return next.slice(0, 200);
      });
    } catch (e) {
      setError(e?.response?.data?.error || "Prediction request failed");
    } finally {
      setPredictionLoading(false);
    }
  };

  const refreshBothModels = async (symbol) => {
    const modelReqId = modelRefreshRef.current.id + 1;
    modelRefreshRef.current = { id: modelReqId };

    try {
      setPredictionLoading(true);
      const [lstm, transformer] = await Promise.allSettled([
        fetchLivePrediction(symbol, "lstm"),
        fetchLivePrediction(symbol, "transformer"),
      ]);

      if (modelReqId !== modelRefreshRef.current.id || symbol !== selectedSymbol) {
        return;
      }

      setLivePredictions((prev) => ({
        lstm:
          lstm.status === "fulfilled"
            ? { ...lstm.value, _runAt: lstm.value?.served_at || new Date().toISOString() }
            : prev.lstm,
        transformer:
          transformer.status === "fulfilled"
            ? { ...transformer.value, _runAt: transformer.value?.served_at || new Date().toISOString() }
            : prev.transformer,
      }));
    } finally {
      if (modelReqId === modelRefreshRef.current.id) {
        setPredictionLoading(false);
      }
    }
  };

  useEffect(() => {
    let isMounted = true;

    const run = async (showSpinner) => {
      const requestId = refreshRef.current.requestId + 1;
      refreshRef.current = { requestId };
      const key = `${selectedSymbol}:${timeframe}`;
      const keyChanged = key !== lastLoadedKeyRef.current;

      try {
        if (showSpinner) {
          setLoading(true);
          setError("");
        }

        await loadOverview();
        if (!isMounted || requestId !== refreshRef.current.requestId) return;
        const payload = await loadSymbolData(selectedSymbol, timeframe);
        if (!isMounted || requestId !== refreshRef.current.requestId) return;
        applySymbolPayload(payload, { preserveOnFailure: !keyChanged });
        lastLoadedKeyRef.current = key;
      } catch (e) {
        if (!isMounted || requestId !== refreshRef.current.requestId) return;
        if (!overview && !priceHistory.length) {
          setError(e?.response?.data?.error || e?.message || "Unable to load terminal data");
        }
      } finally {
        if (isMounted && requestId === refreshRef.current.requestId) {
          if (showSpinner) setLoading(false);
        }
      }
    };

    run(true);

    const timer = setInterval(() => run(false), 60000);
    return () => {
      isMounted = false;
      clearInterval(timer);
    };
  }, [selectedSymbol, timeframe]);

  useEffect(() => {
    const universeSymbols = new Set((allStocks || []).map((s) => s.symbol));
    if (selectedSymbol && universeSymbols.size && !universeSymbols.has(selectedSymbol)) {
      const fallback = allStocks.find((s) => s.symbol)?.symbol;
      if (fallback) {
        setSelectedSymbol(fallback);
      }
    }
  }, [allStocks, selectedSymbol]);

  useEffect(() => {
    // Do not auto-run heavy model inference on every symbol switch; keep terminal responsive.
    setLivePredictions({ lstm: null, transformer: null });
    setPredictionLoading(false);
  }, [selectedSymbol]);

  const latestCandle = priceHistory[priceHistory.length - 1];

  const overviewBySymbol = useMemo(() => {
    const map = {};
    (overview?.stocks || []).forEach((x) => {
      map[x.symbol] = x;
    });
    (overview?.gainers || []).forEach((x) => {
      map[x.symbol] = x;
    });
    (overview?.losers || []).forEach((x) => {
      map[x.symbol] = x;
    });
    (overview?.indices || []).forEach((x) => {
      map[x.symbol] = x;
    });
    return map;
  }, [overview]);

  const selectedOverview = overviewBySymbol[selectedSymbol] || latestCandle || {};
  const marketRefreshDate = selectedOverview?.date || latestCandle?.date || null;
  const watchlistTickers = useMemo(
    () => mergeTickerUniverse(overview?.tickers || [], allStocks || []),
    [overview, allStocks]
  );

  const sentimentMeter = useMemo(() => {
    const latest = sentiment[sentiment.length - 1];
    const score = latest?.sentiment_mean || 0;
    const positive = Math.max(0, Math.min(100, ((score + 1) / 2) * 100));
    return [
      { name: "Positive", value: positive, color: POSITIVE },
      { name: "Negative", value: 100 - positive, color: NEGATIVE },
    ];
  }, [sentiment]);

  const latestSentiment = sentiment[sentiment.length - 1] || null;
  const marketMovers = useMemo(() => {
    const stocks = (overview?.stocks || []).filter((row) => numOrNull(row?.return_1d) !== null);
    const sortedDesc = [...stocks].sort((a, b) => (numOrNull(b.return_1d) || 0) - (numOrNull(a.return_1d) || 0));
    const sortedAsc = [...stocks].sort((a, b) => (numOrNull(a.return_1d) || 0) - (numOrNull(b.return_1d) || 0));

    const gainersFromApi = (overview?.gainers || [])
      .filter((row) => (numOrNull(row?.return_1d) || 0) > 0)
      .sort((a, b) => (numOrNull(b.return_1d) || 0) - (numOrNull(a.return_1d) || 0));

    const losersFromApi = (overview?.losers || [])
      .filter((row) => (numOrNull(row?.return_1d) || 0) < 0)
      .sort((a, b) => (numOrNull(a.return_1d) || 0) - (numOrNull(b.return_1d) || 0));

    const gainers = (gainersFromApi.length ? gainersFromApi : sortedDesc.filter((row) => (numOrNull(row?.return_1d) || 0) > 0)).slice(0, 6);
    const gainerSymbols = new Set(gainers.map((row) => row.symbol));
    const losersBase = losersFromApi.length ? losersFromApi : sortedAsc.filter((row) => (numOrNull(row?.return_1d) || 0) < 0);
    const losers = losersBase.filter((row) => !gainerSymbols.has(row.symbol)).slice(0, 6);

    return { gainers, losers };
  }, [overview]);

  const sentimentPriceAlignment = useMemo(() => {
    const score = numOrNull(latestSentiment?.sentiment_mean) ?? 0;
    const move = numOrNull(selectedOverview?.return_1d) ?? 0;
    if (!score || !move) return "Neutral";
    const sameDirection = (score > 0 && move > 0) || (score < 0 && move < 0);
    return sameDirection ? "Aligned" : "Divergent";
  }, [latestSentiment, selectedOverview]);

  const returnsDistribution = advancedAnalytics?.returns_distribution || [];
  const volatilitySeries = advancedAnalytics?.volatility_series || priceHistory;
  const hasPriceSeries = (priceHistory || []).length > 0;
  const maxCandlePoints = timeframe <= 180 ? 260 : timeframe <= 365 ? 320 : timeframe <= 730 ? 420 : 520;
  const displayPriceHistory = useMemo(
    () => downsampleSeries(priceHistory, maxCandlePoints),
    [priceHistory, maxCandlePoints]
  );

  const candleSeries = useMemo(() => {
    return (displayPriceHistory || [])
      .map((row) => {
        const ts = Date.parse(row.date);
        const o = numOrNull(row.open);
        const h = numOrNull(row.high);
        const l = numOrNull(row.low);
        const c = numOrNull(row.close);
        if (!Number.isFinite(ts) || !Number.isFinite(o) || !Number.isFinite(h) || !Number.isFinite(l) || !Number.isFinite(c)) {
          return null;
        }
        return { x: ts, y: [o, h, l, c] };
      })
      .filter(Boolean);
  }, [displayPriceHistory]);

  const volumeSeries = useMemo(() => {
    return (displayPriceHistory || [])
      .map((row) => {
        const ts = Date.parse(row.date);
        const vol = numOrNull(row.volume);
        const o = numOrNull(row.open);
        const c = numOrNull(row.close);
        if (!Number.isFinite(ts) || !Number.isFinite(vol)) return null;
        return {
          x: ts,
          y: vol,
          fillColor: Number.isFinite(o) && Number.isFinite(c) && c >= o ? "#22c55e55" : "#fb718555",
        };
      })
      .filter(Boolean);
  }, [displayPriceHistory]);

  const priceChartOptions = useMemo(
    () => ({
      chart: {
        type: "candlestick",
        height: 340,
        background: "transparent",
        id: "price-volume-chart",
        zoom: {
          enabled: true,
          type: "x",
          autoScaleYaxis: true,
        },
        toolbar: {
          show: true,
          tools: {
            download: false,
            selection: true,
            zoom: true,
            zoomin: true,
            zoomout: true,
            pan: true,
            reset: true,
          },
        },
        animations: { enabled: false },
      },
      theme: { mode: "dark" },
      grid: {
        borderColor: "#1f3448",
        strokeDashArray: 3,
      },
      xaxis: {
        type: "datetime",
        labels: {
          datetimeUTC: false,
          style: { colors: "#8fa7bf", fontSize: "10px" },
        },
        axisBorder: { color: "#26415a" },
        axisTicks: { color: "#26415a" },
      },
      yaxis: [
        {
          seriesName: "OHLC",
          opposite: true,
          labels: {
            style: { colors: "#8fa7bf", fontSize: "10px" },
            formatter: (val) => miniNumber(val),
          },
          tooltip: { enabled: true },
          decimalsInFloat: 2,
        },
        {
          seriesName: "Volume",
          labels: {
            style: { colors: "#8fa7bf", fontSize: "10px" },
            formatter: (val) => formatNumberCompact(val),
          },
        },
      ],
      plotOptions: {
        candlestick: {
          colors: {
            upward: "#22c55e",
            downward: "#fb7185",
          },
          wick: {
            useFillColor: false,
          },
        },
        bar: {
          columnWidth: "70%",
        },
      },
      stroke: {
        width: [1, 0],
      },
      tooltip: {
        theme: "dark",
        x: { format: "dd MMM yyyy" },
      },
      legend: {
        show: true,
        labels: { colors: "#8fa7bf" },
      },
      dataLabels: { enabled: false },
    }),
    []
  );

  const priceChartSeries = useMemo(
    () => [
      { name: "OHLC", type: "candlestick", data: candleSeries },
      { name: "Volume", type: "column", data: volumeSeries },
    ],
    [candleSeries, volumeSeries]
  );
  const currentComparisonSeries = useMemo(
    () =>
      buildCurrentComparisonSeries(
        latestCandle?.close,
        livePredictions?.lstm?.predicted_close,
        livePredictions?.transformer?.predicted_close
      ),
    [latestCandle?.close, livePredictions]
  );

  const panelMetrics = useMemo(() => {
    if (!priceHistory.length) {
      return {
        close: selectedOverview?.close,
        return_1d: selectedOverview?.return_1d,
        return_30d: selectedOverview?.return_30d,
        volatility_14d: selectedOverview?.volatility_14d,
      };
    }

    const latest = priceHistory[priceHistory.length - 1];
    const prev = priceHistory.length > 1 ? priceHistory[priceHistory.length - 2] : null;
    const d30 = priceHistory.length > 30 ? priceHistory[priceHistory.length - 31] : null;
    const tailReturns = priceHistory
      .map((row) => numOrNull(row.return_1d))
      .filter((v) => Number.isFinite(v));
    const last14 = tailReturns.slice(-14);
    const mean = last14.length ? last14.reduce((s, v) => s + v, 0) / last14.length : null;
    const variance =
      last14.length && mean !== null
        ? last14.reduce((s, v) => s + (v - mean) ** 2, 0) / last14.length
        : null;

    return {
      close: numOrNull(latest?.close),
      return_1d:
        prev && Number.isFinite(latest?.close) && Number.isFinite(prev?.close) && prev.close !== 0
          ? (latest.close / prev.close) - 1
          : numOrNull(latest?.return_1d),
      return_30d:
        d30 && Number.isFinite(latest?.close) && Number.isFinite(d30?.close) && d30.close !== 0
          ? (latest.close / d30.close) - 1
          : numOrNull(latest?.return_30d),
      volatility_14d:
        variance !== null
          ? Math.sqrt(variance)
          : numOrNull(latest?.volatility_14d),
    };
  }, [priceHistory, selectedOverview]);

  const correlationMatrix = useMemo(() => {
    const rows = priceHistory.filter((x) => Number.isFinite(x.return_1d));
    const columns = ["return_1d", "return_7d", "return_30d", "volatility_14d", "relative_strength"];

    const corr = (a, b) => {
      const pairs = rows
        .map((r) => [r[a], r[b]])
        .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));

      if (pairs.length < 3) return 0;

      const xa = pairs.map((p) => p[0]);
      const ya = pairs.map((p) => p[1]);
      const xMean = xa.reduce((s, v) => s + v, 0) / xa.length;
      const yMean = ya.reduce((s, v) => s + v, 0) / ya.length;

      const numerator = xa.reduce((s, _, i) => s + (xa[i] - xMean) * (ya[i] - yMean), 0);
      const xVar = xa.reduce((s, v) => s + (v - xMean) ** 2, 0);
      const yVar = ya.reduce((s, v) => s + (v - yMean) ** 2, 0);

      return numerator / Math.sqrt((xVar || 1) * (yVar || 1));
    };

    return columns.map((row) => ({
      metric: row,
      values: columns.map((col) => ({ metric: col, value: corr(row, col) })),
    }));
  }, [priceHistory]);

  const panelTitle = (title, subtitle) => (
    <div className="mb-3 flex items-end justify-between">
      <div>
        <h3 className="font-display text-sm uppercase tracking-[0.16em] text-terminal-muted">{title}</h3>
        <p className="font-mono text-xs text-terminal-text/75">{subtitle}</p>
      </div>
    </div>
  );

  if (loading && !overview) {
    return <div className="p-8 font-mono text-terminal-text">Loading terminal...</div>;
  }

  if (error && !overview) {
    return <div className="p-8 font-mono text-rose-300">{error}</div>;
  }

  const hasRenderableData = Boolean((overview?.tickers || []).length || (priceHistory || []).length);

  return (
    <div className="min-h-screen px-4 py-4 md:px-6 md:py-6">
      <div className="mx-auto max-w-[1600px] space-y-4">
        <header className="terminal-card rounded-xl p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="font-mono text-xs uppercase tracking-[0.22em] text-terminal-muted">Smart Stock Terminal</p>
              <h1 className="font-display text-2xl text-terminal-text md:text-3xl">Quant + Sentiment Market Workstation</h1>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="rounded-md border border-terminal-border bg-terminal-panel px-3 py-1.5 font-mono text-xs text-terminal-text"
              >
                {watchlistTickers.map((stock) => (
                  <option key={stock.symbol} value={stock.symbol}>
                    {stock.symbol}
                  </option>
                ))}
              </select>
              {TIMEFRAMES.map((tf) => (
                <button
                  key={tf.days}
                  onClick={() => setTimeframe(tf.days)}
                  className={`rounded-md border px-3 py-1.5 font-mono text-xs transition ${
                    timeframe === tf.days
                      ? "border-terminal-accent bg-terminal-accent/20 text-terminal-text"
                      : "border-terminal-border bg-terminal-panel text-terminal-muted hover:text-terminal-text"
                  }`}
                >
                  {tf.label}
                </button>
              ))}
            </div>
          </div>
          <div className="mt-3 grid grid-cols-1 gap-2 border-t border-terminal-border/60 pt-3 md:grid-cols-3">
            <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 px-3 py-2">
              <p className="font-mono text-[11px] uppercase tracking-[0.08em] text-terminal-muted">Market Refresh</p>
              <p className="font-mono text-xs text-terminal-text">{selectedSymbol}: {marketRefreshDate || "-"}</p>
              <p className="font-mono text-[11px] text-terminal-muted">Overview snapshot: {formatDateTime(overview?.as_of)}</p>
              <p className="font-mono text-[11px] text-terminal-muted">Data source: {selectedOverview?.source || "-"}</p>
            </div>
            <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 px-3 py-2">
              <p className="font-mono text-[11px] uppercase tracking-[0.08em] text-terminal-muted">LSTM Run Time</p>
              <p className="font-mono text-xs text-terminal-text">{formatDateTime(livePredictions?.lstm?._runAt || livePredictions?.lstm?.served_at)}</p>
            </div>
            <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 px-3 py-2">
              <p className="font-mono text-[11px] uppercase tracking-[0.08em] text-terminal-muted">Transformer Run Time</p>
              <p className="font-mono text-xs text-terminal-text">{formatDateTime(livePredictions?.transformer?._runAt || livePredictions?.transformer?.served_at)}</p>
            </div>
          </div>
        </header>

        {error && !hasRenderableData ? (
          <div className="rounded-md border border-terminal-danger/60 bg-terminal-danger/10 px-3 py-2 font-mono text-xs text-rose-200">
            {error}
          </div>
        ) : null}

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-12">
          <aside className="space-y-4 xl:col-span-3">
            <div className="terminal-card rounded-xl p-4">
              {panelTitle("Watchlist", "Symbols + daily move")}
              <div className="max-h-[280px] space-y-2 overflow-auto pr-1">
                {watchlistTickers.map((ticker) => {
                  const row = overviewBySymbol[ticker.symbol] || {};
                  const isActive = selectedSymbol === ticker.symbol;
                  const move = row.return_1d;
                  return (
                    <button
                      key={ticker.symbol}
                      onClick={() => {
                        setError("");
                        setSelectedSymbol(ticker.symbol);
                      }}
                      className={`w-full rounded-md border px-3 py-2 text-left transition ${
                        isActive
                          ? "border-terminal-accent bg-terminal-accent/15"
                          : "border-terminal-border bg-terminal-panelSoft/60 hover:border-terminal-accent/50"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <p className="font-mono text-xs text-terminal-text">{ticker.symbol}</p>
                        <p className={`font-mono text-xs ${move >= 0 ? "ticker-positive" : "ticker-negative"}`}>
                          {formatPct(move)}
                        </p>
                      </div>
                      <p className="truncate text-xs text-terminal-muted">{ticker.name}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="terminal-card rounded-xl p-4">
              {panelTitle("Index Tracker", "NIFTY and broad market view")}
              <div className="space-y-2">
                {(overview?.indices || []).map((indexRow) => (
                  <div
                    key={indexRow.symbol}
                    className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 px-2 py-1.5"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-mono text-terminal-text">{indexRow.symbol}</span>
                      <span className={(indexRow.return_1d || 0) >= 0 ? "ticker-positive" : "ticker-negative"}>
                        {formatPct(indexRow.return_1d)}
                      </span>
                    </div>
                    <p className="font-mono text-[11px] text-terminal-muted">{miniNumber(indexRow.close)}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="terminal-card rounded-xl p-4">
              {panelTitle("Market Movers", "Top gainers and losers")}
              <div className="space-y-3">
                <div>
                  <p className="mb-1 font-mono text-xs uppercase tracking-wide text-terminal-success">Gainers</p>
                  {(marketMovers.gainers || []).slice(0, 4).map((g) => (
                    <div key={g.symbol} className="flex items-center justify-between py-0.5 text-xs">
                      <span className="font-mono text-terminal-text">{g.symbol}</span>
                      <span className="ticker-positive">{formatPct(g.return_1d)}</span>
                    </div>
                  ))}
                </div>
                <div>
                  <p className="mb-1 font-mono text-xs uppercase tracking-wide text-terminal-danger">Losers</p>
                  {(marketMovers.losers || []).slice(0, 4).map((g) => (
                    <div key={g.symbol} className="flex items-center justify-between py-0.5 text-xs">
                      <span className="font-mono text-terminal-text">{g.symbol}</span>
                      <span className="ticker-negative">{formatPct(g.return_1d)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="terminal-card rounded-xl p-4">
              {panelTitle("Sector Performance", "1D average returns")}
              <div className="space-y-2">
                {(overview?.sector_performance || []).slice(0, 6).map((s) => (
                  <div key={s.sector} className="rounded-md border border-terminal-border bg-terminal-panelSoft/50 px-2 py-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-terminal-text">{s.sector}</span>
                      <span className={s.avg_return_1d >= 0 ? "ticker-positive" : "ticker-negative"}>
                        {formatPct(s.avg_return_1d)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </aside>

          <main className="space-y-4 xl:col-span-6">
            <div className="terminal-card rounded-xl p-4">
              {panelTitle("Price + Volume", `${selectedSymbol} | candlestick + volume`)}
              <div className="mb-3 grid grid-cols-2 gap-2 md:grid-cols-4">
                <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 p-2">
                  <p className="font-mono text-[11px] text-terminal-muted">Close</p>
                  <p className="font-mono text-sm text-terminal-text">{miniNumber(panelMetrics.close)}</p>
                </div>
                <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 p-2">
                  <p className="font-mono text-[11px] text-terminal-muted">1D</p>
                  <p className={`font-mono text-sm ${(panelMetrics.return_1d || 0) >= 0 ? "ticker-positive" : "ticker-negative"}`}>
                    {formatPct(panelMetrics.return_1d)}
                  </p>
                </div>
                <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 p-2">
                  <p className="font-mono text-[11px] text-terminal-muted">30D</p>
                  <p className={`font-mono text-sm ${(panelMetrics.return_30d || 0) >= 0 ? "ticker-positive" : "ticker-negative"}`}>
                    {formatPct(panelMetrics.return_30d)}
                  </p>
                </div>
                <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 p-2">
                  <p className="font-mono text-[11px] text-terminal-muted">Volatility</p>
                  <p className="font-mono text-sm text-terminal-warning">{formatPct(panelMetrics.volatility_14d)}</p>
                </div>
              </div>
              <div className="h-[360px] w-full">
                {hasPriceSeries ? (
                  <ReactApexChart
                    key={`${selectedSymbol}-${timeframe}-${candleSeries.length}`}
                    options={priceChartOptions}
                    series={priceChartSeries}
                    type="candlestick"
                    height={340}
                  />
                ) : (
                  <div className="flex h-full items-center justify-center rounded-md border border-terminal-border bg-terminal-panelSoft/50">
                    <p className="font-mono text-xs text-terminal-muted">No price data available for selected timeframe.</p>
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="terminal-card rounded-xl p-4 md:col-span-2">
                {panelTitle("Model Price Compare", `current price vs model estimates | ${selectedSymbol}`)}
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <p className="font-mono text-[11px] uppercase tracking-[0.1em] text-terminal-muted">Run Models</p>
                  <button
                    onClick={() => runLivePrediction("lstm")}
                    disabled={predictionLoading}
                    className="rounded-md border border-terminal-border bg-terminal-panel px-2.5 py-1 font-mono text-[11px] text-terminal-text transition hover:border-terminal-accent disabled:opacity-50"
                  >
                    Run LSTM
                  </button>
                  <button
                    onClick={() => runLivePrediction("transformer")}
                    disabled={predictionLoading}
                    className="rounded-md border border-terminal-border bg-terminal-panel px-2.5 py-1 font-mono text-[11px] text-terminal-text transition hover:border-terminal-accent disabled:opacity-50"
                  >
                    Run Transformer
                  </button>
                  <button
                    onClick={() => refreshBothModels(selectedSymbol)}
                    disabled={predictionLoading}
                    className="rounded-md border border-terminal-accent bg-terminal-accent/15 px-2.5 py-1 font-mono text-[11px] text-terminal-text transition hover:bg-terminal-accent/25 disabled:opacity-50"
                  >
                    {predictionLoading ? "Refreshing..." : "Run Both"}
                  </button>
                </div>
                <div className="h-[240px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={currentComparisonSeries}>
                      <CartesianGrid strokeDasharray="2 2" stroke="#1f3448" />
                      <XAxis dataKey="model" tick={{ fill: "#8fa7bf", fontSize: 10 }} />
                      <YAxis tick={{ fill: "#8fa7bf", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "#08131f", border: "1px solid #26415a" }} />
                      <Bar dataKey="price" fill="#2dd4bf" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
                <p className="mt-2 font-mono text-[11px] text-terminal-muted">
                  Panel compares current market price with latest LSTM and Transformer predictions. No future projection curve is used.
                </p>
              </div>

              <div className="terminal-card rounded-xl p-4">
                {panelTitle("Volatility Regime", "volatility vs returns")}
                <div className="h-[220px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="2 2" stroke="#1f3448" />
                      <XAxis dataKey="volatility_14d" tick={{ fill: "#8fa7bf", fontSize: 10 }} />
                      <YAxis dataKey="return_1d" tick={{ fill: "#8fa7bf", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "#08131f", border: "1px solid #26415a" }} />
                        <Scatter data={volatilitySeries} fill="#2dd4bf" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="terminal-card rounded-xl p-4">
                {panelTitle("Returns Distribution", "daily return histogram")}
                <div className="h-[220px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={returnsDistribution}>
                      <CartesianGrid strokeDasharray="2 2" stroke="#1f3448" />
                      <XAxis dataKey="label" tick={{ fill: "#8fa7bf", fontSize: 10 }} />
                      <YAxis tick={{ fill: "#8fa7bf", fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "#08131f", border: "1px solid #26415a" }} />
                      <Bar dataKey="count" fill="#0ea5e9" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </main>

          <aside className="space-y-4 xl:col-span-3">
            <div className="terminal-card rounded-xl p-4">
              {panelTitle("Sentiment Timeline", "daily aggregated sentiment")}
              <div className="h-[160px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={sentiment}>
                    <CartesianGrid strokeDasharray="2 2" stroke="#1f3448" />
                    <XAxis dataKey="date" tick={{ fill: "#8fa7bf", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#8fa7bf", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#08131f", border: "1px solid #26415a" }} />
                    <Area type="monotone" dataKey="sentiment_mean" stroke="#2dd4bf" fill="#2dd4bf33" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 h-[120px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={sentiment}>
                    <CartesianGrid strokeDasharray="2 2" stroke="#1f3448" />
                    <XAxis dataKey="date" tick={{ fill: "#8fa7bf", fontSize: 9 }} />
                    <YAxis yAxisId="news" tick={{ fill: "#8fa7bf", fontSize: 9 }} />
                    <YAxis yAxisId="ratio" orientation="right" domain={[0, 1]} tick={{ fill: "#8fa7bf", fontSize: 9 }} />
                    <Tooltip contentStyle={{ background: "#08131f", border: "1px solid #26415a" }} />
                    <Bar yAxisId="news" dataKey="news_count" fill="#0ea5e9" opacity={0.55} />
                    <Line yAxisId="ratio" type="monotone" dataKey="positive_ratio" stroke="#22c55e" dot={false} strokeWidth={1.8} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 h-[90px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={sentimentMeter} innerRadius={32} outerRadius={50} dataKey="value">
                      {sentimentMeter.map((entry) => (
                        <Cell key={entry.name} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                <div className="rounded border border-terminal-border bg-terminal-panelSoft/60 px-2 py-1.5">
                  <p className="font-mono text-terminal-muted">Latest News Count</p>
                  <p className="font-mono text-terminal-text">{miniNumber(latestSentiment?.news_count)}</p>
                </div>
                <div className="rounded border border-terminal-border bg-terminal-panelSoft/60 px-2 py-1.5">
                  <p className="font-mono text-terminal-muted">Positive Ratio</p>
                  <p className="font-mono text-terminal-text">{formatPct(latestSentiment?.positive_ratio)}</p>
                </div>
                <div className="rounded border border-terminal-border bg-terminal-panelSoft/60 px-2 py-1.5">
                  <p className="font-mono text-terminal-muted">Sentiment Mean</p>
                  <p className="font-mono text-terminal-text">{miniNumber(latestSentiment?.sentiment_mean)}</p>
                </div>
                <div className="rounded border border-terminal-border bg-terminal-panelSoft/60 px-2 py-1.5">
                  <p className="font-mono text-terminal-muted">Price Alignment</p>
                  <p className={`font-mono ${sentimentPriceAlignment === "Aligned" ? "ticker-positive" : sentimentPriceAlignment === "Divergent" ? "ticker-negative" : "text-terminal-warning"}`}>
                    {sentimentPriceAlignment}
                  </p>
                </div>
              </div>
            </div>

            <div className="terminal-card rounded-xl p-4">
              {panelTitle("Prediction Tape", "latest stored model outputs")}
              <div className="max-h-[200px] overflow-auto">
                {(predictionHistory || []).slice(0, 12).map((row, idx) => (
                  <div key={`${row.date}-${idx}`} className="grid grid-cols-3 gap-2 border-b border-terminal-border/50 py-1 text-xs">
                    <span className="font-mono text-terminal-muted">{row.date}</span>
                    <span className="font-mono text-terminal-text">{row.model_name}</span>
                    <span className="font-mono text-terminal-text">{miniNumber(row.predicted_close)}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="terminal-card rounded-xl p-4">
              {panelTitle("Forecast Controls", "execution mode + notes")}
              <div className="space-y-2 text-xs">
                <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 p-2">
                  <p className="font-mono text-terminal-muted">Actual Close</p>
                  <p className="font-mono text-terminal-text">{miniNumber(latestCandle?.close)}</p>
                </div>
                <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 p-2">
                  <p className="font-mono text-terminal-muted">LSTM Predicted Close</p>
                  <p className="font-mono text-terminal-text">
                    {livePredictions?.lstm?.predicted_close ? miniNumber(livePredictions.lstm.predicted_close) : "Run LSTM"}
                  </p>
                </div>
                <div className="rounded-md border border-terminal-border bg-terminal-panelSoft/60 p-2">
                  <p className="font-mono text-terminal-muted">Transformer Predicted Close</p>
                  <p className="font-mono text-terminal-warning">
                    {livePredictions?.transformer?.predicted_close ? miniNumber(livePredictions.transformer.predicted_close) : "Run Transformer"}
                  </p>
                </div>
              </div>
            </div>

            <div className="terminal-card rounded-xl p-4">
              {panelTitle("Correlation Matrix", "feature co-movement")}
              <div className="space-y-1 text-[11px]">
                {correlationMatrix.map((row) => (
                  <div key={row.metric} className="grid grid-cols-6 gap-1">
                    <div className="font-mono text-terminal-muted">{row.metric.replace("_", " ")}</div>
                    {row.values.map((item) => {
                      const value = Number(item.value || 0);
                      const color = value > 0.35 ? "#22c55e" : value < -0.35 ? "#fb7185" : "#f59e0b";
                      return (
                        <div
                          key={`${row.metric}-${item.metric}`}
                          className="rounded px-1 py-0.5 text-center font-mono"
                          style={{ background: `${color}26`, color }}
                        >
                          {value.toFixed(2)}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          </aside>
        </section>
      </div>
    </div>
  );
};

export default TerminalDashboard;
