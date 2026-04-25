# Smart Stock Predictor: Comprehensive Project Documentation

## 1. Executive Summary

This project is an end-to-end stock analytics and forecasting platform focused on Indian equities and indices.

It combines:

- A Django backend for data ingestion, feature engineering, sentiment aggregation, model inference APIs, and analytics APIs.
- A React frontend that renders a terminal-style analytics dashboard with charts, model outputs, and market breadth panels.
- Deep learning models (LSTM and Transformer) trained on engineered market features plus sentiment context.

Primary goal:

- Provide a decision-support terminal for market monitoring and model-assisted price expectation, not guaranteed trading advice.

Secondary goal:

- Provide a research-friendly codebase where model behavior, engineered factors, and sentiment effects can be studied and compared.


## 2. What The System Tracks

### 2.1 Tracked Asset Universe

The current tracked universe (from backend metadata seed and frontend symbol utilities) includes:

Stocks:

- RELIANCE.NS
- AXISBANK.NS
- HDFCBANK.NS
- ONGC.NS
- SBIN.NS
- INFY.NS
- TCS.NS
- ICICIBANK.NS
- KOTAKBANK.NS
- ADANIPORTS.NS
- ADANIENT.NS
- BAJFINANCE.NS
- BHARTIARTL.NS

Indices:

- ^NSEI (NIFTY 50)
- ^NSEBANK (BANK NIFTY)
- ^NSEMDCP50 (NIFTY MIDCAP 50)
- ^CNXAUTO (NIFTY AUTO)


### 2.2 Data Domains

The platform works with four core data domains:

- Ticker metadata: symbol identity, sector, asset type, benchmark mapping.
- Market OHLCV + engineered features: daily bars with returns, volatility, ATR, momentum, relative strength.
- Sentiment aggregates: daily sentiment mean/std, article count, positive ratio from news articles.
- Prediction records: model outputs and evaluation fields (predicted/actual close, error metrics).


## 3. Tech Stack

### 3.1 Backend

- Python + Django 5.1
- Django REST Framework (JSON APIs)
- django-cors-headers
- Data/ML stack: pandas, numpy, scikit-learn, tensorflow, joblib, pandas-ta-classic
- Market data libraries/services: yfinance, requests to external APIs
- NLP sentiment: TextBlob
- Task hook: Celery task definition present
- DB: SQLite by default, PostgreSQL when `DATABASE_URL` is configured
- Caching: Django LocMemCache


### 3.2 Frontend

- React 18 + Vite
- axios for API calls
- react-router-dom for routing
- Visualization:
  - ApexCharts (`react-apexcharts`) for high-density candlestick + volume chart
  - Recharts for bar/line/area/pie/scatter/composed visualizations
  - Chart.js (`react-chartjs-2`) in legacy model comparison page
- Styling:
  - Tailwind CSS + custom theme tokens
  - Additional CSS modules for older pages/components


### 3.3 Dev Proxy and Routing

Vite proxy forwards:

- `/api` -> Django
- `/predict` -> Django
- `/static` -> Django


## 4. Backend Architecture

Top-level backend project layout:

- `backend/stockproject/stockproject`: Django config (`settings.py`, `urls.py`)
- `backend/stockproject/stockapi`: API namespace for market + analytics endpoints
- `backend/stockproject/predictor`: core analytics, model serving, data services, training scripts


### 4.1 URL Routing

Global routes:

- `/api/...` -> `stockapi.urls`
- `/predict/...` -> `predictor.urls`


### 4.2 API Endpoints (Internal APIs)

#### A) `stockapi` namespace (`/api`)

- `GET /api/health/`
  - Basic API health ping.

- `GET /api/ohlcv/<symbol>/`
  - Source: `stockapi.views.get_ohlcv`
  - Fetches recent intraday OHLCV (1-minute bars over 7-day horizon) using yfinance.
  - Supports `limit` and optional `end_time` filtering.

- `GET /api/stocks/`
  - Source: `predictor.views.get_stocks` (imported into stockapi urls)
  - Returns stock/index universe list.

- `GET /api/model_comparison/?symbol=...`
  - Source: `predictor.views.model_comparison`
  - Returns side-by-side LSTM vs Transformer result payload for selected symbol.

- `GET /api/price-history/?symbol=...&days=...`
  - Source: `predictor.analytics_views.price_history`
  - Returns historical daily series with both market and sentiment overlays.

- `GET /api/technical-indicators/?symbol=...`
  - Source: `predictor.analytics_views.technical_indicators`
  - Returns latest technical snapshot and monthly aggregate stats.

- `GET /api/sentiment/?symbol=...&days=...`
  - Source: `predictor.analytics_views.sentiment_view`
  - Returns daily sentiment series and latest sentiment snapshot.

- `GET /api/prediction/?symbol=...&horizon_days=1`
  - Source: `predictor.analytics_views.prediction_view`
  - Returns stored prediction history from DB (`PredictionRecord`).

- `GET /api/market-overview/`
  - Source: `predictor.analytics_views.market_overview`
  - Returns market snapshot with tickers, stocks, gainers, losers, sector performance, and indices.

- `GET /api/advanced-analytics/?symbol=...&days=...`
  - Source: `predictor.analytics_views.advanced_analytics`
  - Returns volatility series and returns histogram bins.


#### B) `predictor` namespace (`/predict`)

- `GET /predict/health/`
  - Source: `predictor.views.health_check`
  - Reports model system readiness (asset-aware predictor, base model, available models).

- `GET /predict/stocks/`
  - Source: `predictor.views.get_stocks`
  - Universe metadata for symbols.

- `GET /predict/model-comparison/?symbol=...`
  - Source: `predictor.views.model_comparison`
  - Full comparison payload with per-model metrics and chart-ready arrays.

- `GET or POST /predict/<symbol>/?model=lstm|transformer`
  - Source: `predictor.views.predict_price`
  - Runs live prediction via `AssetAwarePredictor`.
  - Also enriches response with sentiment context (average sentiment, article count).


### 4.3 External APIs and Data Providers

The backend integrates with these external providers:

- Yahoo Finance via yfinance
  - Primary market data source for daily and intraday retrieval.

- TwelveData API (`https://api.twelvedata.com/time_series`)
  - Fallback daily data source in `MultiSourceMarketDataFetcher`.
  - Requires `TWELVEDATA_API_KEY` for use.

- NSE India API (historical endpoint)
  - Secondary fallback in `NSEDataFetcher` after yfinance.

- NewsAPI (`https://newsapi.org/v2/everything`)
  - News retrieval for sentiment aggregation.

- GNews (`https://gnews.io/api/v4/search`)
  - Additional news retrieval in parallel with NewsAPI.

Notes:

- News retrieval uses deduplication by URL/title.
- Sentiment scoring uses TextBlob polarity on title/description/content.


## 5. Database Design and Data Lifecycle

Core tables are defined in `predictor.models` and created in migration `0001_initial`.


### 5.1 `TickerMetadata`

Purpose:

- Canonical list of tracked assets and their descriptors.

Key fields:

- `symbol` (unique)
- `name`
- `asset_type` (`stock`/`index`)
- `sector`
- `exchange`
- `benchmark_symbol`
- `is_active`
- `created_at`, `updated_at`


### 5.2 `MarketData`

Purpose:

- Daily OHLCV plus engineered quantitative features.

Key fields:

- Raw: `open`, `high`, `low`, `close`, `volume`
- Engineered:
  - `log_return`
  - `return_1d`, `return_7d`, `return_14d`, `return_30d`
  - `volatility_14d`
  - `atr_14`
  - `momentum_10d`
  - `volume_change_1d`
  - `relative_strength`
- `source`
- `created_at`, `updated_at`

Constraints:

- Unique by (`symbol`, `date`)


### 5.3 `SentimentData`

Purpose:

- Day-level sentiment aggregates for each symbol.

Key fields:

- `sentiment_mean`
- `sentiment_std`
- `news_count`
- `positive_ratio`
- `source`
- Timestamps

Constraints:

- Unique by (`symbol`, `date`)


### 5.4 `PredictionRecord`

Purpose:

- Stores model outputs and evaluation metrics over time.

Key fields:

- `symbol`, `date`, `horizon_days`, `model_name`
- `predicted_close`, `actual_close`
- `rmse`, `mae`, `mape`, `directional_accuracy`
- `metadata` (JSON)
- `created_at`


### 5.5 Data Ingestion and Update Paths

There are two practical update paths:

1. On-demand API warmup path:

- Analytics endpoints call `_ensure_symbol_ready` and `_ensure_sentiment_ready` when needed.
- Missing/sparse data triggers fetch, feature engineering, and DB upsert.

2. Batch sync path:

- Management command: `python manage.py sync_market_data --years 5`
- Seeds metadata, fetches market data and sentiment, computes features, and upserts all rows.


### 5.6 Caching Strategy

- Django LocMem cache is used for expensive endpoints (`market-overview`, `price-history`, etc.).
- Throttle keys avoid repeated rapid refresh for symbol/sentiment updates.
- Goal: keep terminal UI responsive while limiting redundant upstream API calls.


## 6. Financial Feature Engineering and Calculations

Feature logic is primarily in:

- `predictor/services/feature_engineering.py`
- `predictor/utils.py` (technical indicators for model pipelines)

Main engineered metrics:

- Log return:
  - $\log(P_t / P_{t-1})$

- Simple returns:
  - $r_{1d} = P_t / P_{t-1} - 1$
  - Similar for 7/14/30-day windows.

- Rolling volatility:
  - Std dev of daily returns over a 14-day rolling window.

- Momentum:
  - $P_t - P_{t-10}$

- ATR (Average True Range, 14):
  - True range from max of high-low and gap terms, then rolling mean.

- Relative strength vs benchmark:
  - `asset_close / benchmark_close` (after forward/back fill of benchmark)

Technical indicators used in training/inference preprocessing include:

- SMA, EMA (9, 12, 21, 26)
- RSI(14)
- Bollinger bands (MA20 ± 2σ)
- MACD + signal line
- Stochastic oscillator `%K`, `%D`
- ATR(14)
- `news_sentiment`
- Additional engineered `% change` features for OHLC and normalized volume changes.


## 7. Model System: LSTM and Transformer

### 7.1 Overall Model Philosophy

The codebase contains two model tracks:

1. Asset-aware models (primary runtime path)

- Per-symbol LSTM and Transformer artifacts.
- Separate handling for stocks vs indices.
- Different validation constraints and scaling behavior by asset type.

2. Global models (research/alternative path)

- Trained across multiple symbols with universal scalers.
- Designed for cross-asset generalization.


### 7.2 Runtime Inference (`AssetAwarePredictor`)

Inference flow:

1. Normalize/detect symbol type (stock/index).
2. Load symbol-specific model + scalers if present.
3. Fallback to generic improved model if symbol-specific files are absent.
4. Fetch recent market data (daily default; intraday optional fallback path exists).
5. Preprocess with same features as training.
6. Build rolling sequences (default length 60 timesteps).
7. Predict scaled outputs and inverse-transform to price space.
8. Validate/clamp prediction within configured deviation bounds.
9. Return prediction and metadata (confidence proxy, model metrics, counts).

Comparison mode (`compare_models`) runs both LSTM and Transformer and computes:

- Predicted vs actual values
- Time-series arrays (`predictions`, `actuals`)
- Metrics: MSE, RMSE, MAE, R²


### 7.3 Training Scripts

Main training scripts:

- `asset_aware_trainer.py`
  - Trains per-symbol LSTM and Transformer with asset-specific config.

- `train_lstm_improved.py`
  - Single-model improved LSTM training pipeline.

- `train_transformer_improved.py`
  - Single-model improved Transformer pipeline.

- `global_trainer.py`
  - Trains global LSTM/Transformer across many symbols and saves universal scalers.

Model artifacts are saved under `predictor/models/` (for example):

- `lstm_stocks_<SYMBOL>_model.keras`
- `transformer_stocks_<SYMBOL>_model.keras`
- `lstm_indices_<SYMBOL>_model.keras`
- `global_lstm_model.keras`, `global_transformer_model.keras`
- Feature and target scaler `.pkl` files
- JSON metadata files with training/evaluation summary


### 7.4 What The Models Output

Primary outputs:

- Predicted next close price
- Current/actual close price reference
- Confidence proxy based on recent prediction stability
- Historical backtest-like metrics in comparison payload

Practical interpretation:

- These are model-derived expectations from historical dynamics + engineered factors + sentiment context.
- They should be interpreted as probabilistic guidance, not deterministic truth.


## 8. Sentiment Pipeline

Sentiment flow:

1. Build query terms from symbol/company mapping.
2. Fetch from NewsAPI and GNews in parallel.
3. Deduplicate by URL/title.
4. Score text using TextBlob polarity.
5. Group by published date and aggregate:
   - mean polarity
   - std deviation
   - article count
   - positive ratio
6. Persist daily aggregates into `SentimentData`.

How it is used:

- Sentiment time series is displayed in terminal charts.
- Latest sentiment can be merged into live prediction responses.
- `news_sentiment` is an explicit feature in the model input set.


## 9. Frontend Architecture and Component-by-Component Explanation

Current primary route:

- `/` and `/terminal` both render `TerminalDashboard`.


### 9.1 API Client Layer

`frontend/src/services/terminalApi.js` centralizes all terminal API calls.

Functions and endpoint usage:

- `fetchMarketOverview()` -> `GET /api/market-overview/`
- `fetchStocksUniverse()` -> `GET /api/stocks/`
- `fetchPriceHistory(symbol, days)` -> `GET /api/price-history/`
- `fetchTechnicalIndicators(symbol)` -> `GET /api/technical-indicators/`
- `fetchSentiment(symbol, days)` -> `GET /api/sentiment/`
- `fetchPredictionHistory(symbol, horizon_days)` -> `GET /api/prediction/`
- `fetchAdvancedAnalytics(symbol, days)` -> `GET /api/advanced-analytics/`
- `fetchModelComparison(symbol)` -> `GET /api/model_comparison/`
- `fetchLivePrediction(symbol, model)` -> `GET /predict/<symbol>/?model=...`


### 9.2 Main Dashboard: `TerminalDashboard.jsx`

This is the core presentation layer and currently the most important frontend file.

High-level behavior:

- Loads market overview + stock universe.
- Loads symbol-specific datasets (price, sentiment, indicators, prediction history, advanced analytics).
- Polls refresh every 60 seconds.
- Keeps lightweight resilience by applying partial successes when some endpoints fail.

Panels and what they represent:

1. Header panel

- Selected symbol, timeframe, data source, refresh times.

2. Watchlist

- All symbols from merged universe + overview metadata.
- Shows day move per symbol.

3. Index tracker

- Key index snapshots with close and daily return.

4. Market movers

- Top gainers and losers (with API-provided lists and fallback sorting logic).

5. Sector performance

- Average 1D return per sector.

6. Price + Volume chart

- Candlestick + volume columns from `price-history` series.
- Includes derived quick metrics (close, 1D, 30D, volatility).

7. Model price compare

- Current price vs latest LSTM/Transformer predicted close.
- Manual model execution controls.

8. Volatility regime scatter

- Relationship between volatility and return.

9. Returns distribution histogram

- Bucketed daily return frequencies.

10. Sentiment timeline + bar/line + pie

- Time-series sentiment mean, news count, positive ratio, sentiment share.

11. Prediction tape

- Recent stored prediction records.

12. Forecast controls

- Current close and latest live model outputs.

13. Correlation matrix

- Pairwise correlations across selected engineered features.


### 9.3 Graphs: Axis Definitions and Meaning

1. Candlestick + volume chart (ApexCharts)

- X-axis: datetime (market date)
- Primary Y-axis: price (OHLC)
- Secondary Y-axis: volume
- Interpretation: price action and volume participation over selected timeframe.

2. Volatility regime scatter

- X-axis: `volatility_14d`
- Y-axis: `return_1d`
- Interpretation: return behavior under low/high volatility conditions.

3. Returns distribution bar chart

- X-axis: predefined return buckets (`<-2%`, `-2 to -1%`, ..., `>2%`)
- Y-axis: count of days in each bucket
- Interpretation: empirical return distribution shape and tail behavior.

4. Sentiment area chart

- X-axis: date
- Y-axis: `sentiment_mean`
- Interpretation: directional sentiment trend.

5. Sentiment composed chart

- X-axis: date
- Left Y-axis: `news_count`
- Right Y-axis: `positive_ratio` (0 to 1)
- Interpretation: sentiment quality alongside article volume.

6. Sentiment pie chart

- Slice values: positive vs negative share from transformed latest sentiment score.

7. Model compare bar chart

- X-axis: categories (`Actual`, `LSTM`, `Transformer`)
- Y-axis: price
- Interpretation: immediate side-by-side model gap vs current observed close.

8. Legacy model comparison line charts (`ModelComparison.jsx`)

- X-axis: sequence index / time_series array
- Y-axis: price
- Two lines: predicted vs actual
- Interpretation: model fit behavior over sampled sequence window.


### 9.4 Other Frontend Components (Legacy/Secondary)

Even though terminal route is primary, these components still exist and may be used in future route expansions:

- `HomePage.jsx`
  - Landing summary with links to terminal/model lab.
  - No direct API calls.

- `StockPage.jsx`
  - Symbol-specific page using legacy components.
  - Uses `StockChart` and `PredictionDisplay` components.

- `StockChart.jsx`
  - Calls `GET /api/ohlcv/<symbol>/` for intraday bars.
  - Renders candlestick chart.

- `PredictionDisplay.jsx`
  - Calls `POST /predict/<symbol>/` to get prediction and sentiment context.

- `Header.jsx`, `DropDown.jsx`, `SearchBar.jsx`
  - UI navigation and symbol selection; no backend data fetching except navigation target updates.

- `stockSymbols.js`
  - Symbol mapping layer from display key to NSE/BSE code.


## 10. End-to-End Request Flow (Typical Dashboard Session)

1. User opens terminal page.
2. Frontend fetches market overview and stock universe.
3. For selected symbol, frontend requests:
   - price history
   - sentiment
   - technical indicators
   - prediction history
   - advanced analytics
4. Backend returns DB-backed data; if missing, it may trigger on-demand refresh.
5. User optionally triggers live LSTM/Transformer inference.
6. Backend predicts with asset-aware model and returns latest predicted close.
7. Frontend updates compare panel and prediction tape.


## 11. Operational Commands

From `backend/stockproject`:

- Run migrations:
  - `python manage.py migrate`

- Run server:
  - `python manage.py runserver`

- Sync market + sentiment data:
  - `python manage.py sync_market_data --years 5`

From `frontend`:

- Install deps:
  - `npm install`

- Start dev server:
  - `npm run dev`


## 12. Important Implementation Notes and Limitations

- Sentiment/news API keys are currently hardcoded in code and should be moved to environment variables for secure deployment.
- Some legacy frontend pages remain in repository while terminal dashboard is the active route.
- `PredictionRecord` retrieval exists; persistent recording behavior should be verified/expanded if long-term model tracking is required.
- Database schema readiness is handled in analytics responses with explicit 503 guidance when tables are missing.


## 13. What The Project Is Trying To Achieve Overall

The project aims to unify:

- Market state monitoring (price, returns, sector breadth, volatility),
- Sentiment state monitoring (news-driven market mood),
- Model-based expected price estimates (LSTM/Transformer),

into one operator-friendly interface for analysis, experimentation, and communication.

It is effectively a quant + sentiment assisted market intelligence workstation for NSE-focused assets.


## 14. Agent-Ready Literature Review and Study Guide

This section is designed so autonomous agents can quickly discover relevant research papers and technical references.


### 14.1 Problem Statements (for paper search)

Use these as search intents:

- Stock price forecasting with LSTM and Transformer using technical indicators.
- Hybrid market prediction using OHLCV + news sentiment features.
- Relative strength and volatility features in equity forecasting.
- Daily sentiment aggregation effects on short-horizon stock prediction.
- Cross-asset (stock + index) scaling and transferability in deep time-series models.


### 14.2 Keywords and Query Strings

Core keywords:

- `LSTM stock prediction technical indicators`
- `Transformer time series stock forecasting`
- `OHLCV sentiment fusion model finance`
- `TextBlob sentiment financial news reliability`
- `volatility regime returns distribution equity`
- `relative strength feature benchmark forecasting`
- `NSE Indian market deep learning forecasting`

Suggested compound queries:

- `("LSTM" OR "Transformer") AND stock AND (OHLCV OR technical indicators) AND sentiment`
- `equity forecasting daily returns volatility ATR RSI MACD`
- `financial news sentiment aggregation daily polarity prediction`
- `benchmark-relative strength feature stock return prediction`
- `Indian stock market deep learning forecast NSE NIFTY`


### 14.3 Agent Prompt Template (Copy-Use)

Use this template with research agents:

"Analyze literature for an NSE-focused stock forecasting system using daily OHLCV, technical indicators (RSI, MACD, Bollinger, ATR), and daily aggregated news sentiment. Prioritize papers that compare LSTM and Transformer architectures, discuss feature engineering for volatility/returns, and report robust evaluation metrics (MAE, RMSE, MAPE, directional accuracy, R²). Return: (1) seminal papers, (2) recent SOTA papers, (3) datasets and benchmarks, (4) modeling pitfalls, (5) evaluation best practices, (6) reproducibility checklists, and (7) ideas directly applicable to this architecture." 


### 14.4 Evaluation Dimensions for Literature Mapping

When reviewing papers, map findings to these project dimensions:

- Data frequency: intraday vs daily
- Feature set: raw OHLCV vs engineered indicators vs sentiment
- Model class: recurrent vs attention-based vs hybrid
- Training strategy: per-asset vs global/multi-asset
- Scaling strategy: local vs global scalers
- Target horizon: 1-day vs multi-step
- Metrics: MAE/RMSE/MAPE/R²/directional accuracy
- Robustness: regime shifts, outliers, data-source drift


### 14.5 Immediate Research Backlog (Practical)

1. Compare explicit multi-step forecasting vs single-step recursive forecasting.
2. Evaluate sentiment lag windows (same-day vs 1-day lag vs rolling aggregate).
3. Test probabilistic outputs (prediction intervals) instead of point-only forecasts.
4. Benchmark against classical baselines (ARIMA/XGBoost/LightGBM) for sanity checks.
5. Study impact of sector/benchmark-aware features on directional accuracy.


## 15. File-Level Quick Reference

Backend core:

- `backend/stockproject/stockproject/settings.py`
- `backend/stockproject/stockproject/urls.py`
- `backend/stockproject/stockapi/urls.py`
- `backend/stockproject/stockapi/views.py`
- `backend/stockproject/predictor/urls.py`
- `backend/stockproject/predictor/views.py`
- `backend/stockproject/predictor/analytics_views.py`
- `backend/stockproject/predictor/models.py`
- `backend/stockproject/predictor/services/data_sources.py`
- `backend/stockproject/predictor/services/feature_engineering.py`
- `backend/stockproject/predictor/services/sentiment_pipeline.py`
- `backend/stockproject/predictor/news_sentiment.py`
- `backend/stockproject/predictor/asset_aware_predictor.py`
- `backend/stockproject/predictor/asset_aware_trainer.py`
- `backend/stockproject/predictor/global_trainer.py`
- `backend/stockproject/predictor/management/commands/sync_market_data.py`

Frontend core:

- `frontend/src/App.jsx`
- `frontend/src/services/terminalApi.js`
- `frontend/src/pages/TerminalDashboard.jsx`
- `frontend/src/pages/HomePage.jsx`
- `frontend/src/pages/StockPage.jsx`
- `frontend/src/pages/ModelComparison.jsx`
- `frontend/src/components/StockChart.jsx`
- `frontend/src/components/PredictionDisplay.jsx`
- `frontend/src/utils/stockSymbols.js`


## 16. Presentation-Ready Narrative (Short Script)

"This platform is a market intelligence terminal for NSE assets. We ingest price data from market providers, compute quantitative features such as returns, volatility, ATR, momentum, and benchmark-relative strength, and combine that with daily news sentiment from NewsAPI and GNews. The backend stores everything in normalized tables and serves analytics and model endpoints through Django REST APIs. On the frontend, a terminal-style React dashboard visualizes market structure, sector breadth, sentiment flow, and LSTM/Transformer model outputs. The architecture supports both asset-aware per-symbol models and global models, making it useful for both operational monitoring and research-driven experimentation." 
