# Data and Database Documentation

## 1. Scope

This document explains the full data lifecycle in this project:

- where data is fetched from,
- how it is transformed,
- where and how it is stored,
- how model outputs are evaluated,
- and the SQL-level table structure used by Django.

Project root: `d:/major-project`
Backend root: `d:/major-project/backend/stockproject`

---

## 2. Database Engine and Runtime Configuration

Source: `backend/stockproject/stockproject/settings.py`

The backend supports two database modes:

1. PostgreSQL when environment variable `DATABASE_URL` is present.
2. SQLite otherwise (default local mode), with DB file at:
   - `backend/stockproject/db.sqlite3`

Cache layer:

- Django LocMemCache (`stock-terminal-cache`) is used for API response caching and refresh throttling.
- This is memory-only and does not persist to database.

---

## 3. Data Sources (Fetch Layer)

### 3.1 Market Data Sources

Primary source:

- Yahoo Finance via `yfinance`
- Implementation: `backend/stockproject/predictor/services/data_sources.py`
- Class: `YahooFinanceProvider.fetch_daily(...)`

Fallback source:

- TwelveData API (`https://api.twelvedata.com/time_series`)
- Same file/class family:
  - `TwelveDataProvider.fetch_daily(...)`
- Uses `TWELVEDATA_API_KEY` when configured.

Fetcher orchestrator:

- `MultiSourceMarketDataFetcher.fetch_daily_prices(...)`
- Attempts Yahoo first, then TwelveData.
- Returns both dataframe and provenance source label (`yahoo_finance` or `twelvedata`).

### 3.2 Sentiment Data Sources

Sources:

- NewsAPI (`https://newsapi.org/v2/everything`)
- GNews (`https://gnews.io/api/v4/search`)

Implementation:

- `backend/stockproject/predictor/news_sentiment.py`
- `fetch_news_articles_newsapi(...)`
- `fetch_news_articles_gnews(...)`
- `fetch_news_articles(...)` (parallelized fetch + dedupe)
- `analyze_sentiment(...)` using TextBlob polarity

Aggregation pipeline:

- `backend/stockproject/predictor/services/sentiment_pipeline.py`
- `aggregate_daily_sentiment(symbol, company_name)`
- Groups article sentiment by date and computes:
  - `sentiment_mean`
  - `sentiment_std`
  - `news_count`
  - `positive_ratio`

### 3.3 Intraday OHLCV Endpoint (Not Persisted)

Endpoint:

- `GET /api/ohlcv/<symbol>/`

Implementation:

- `backend/stockproject/stockapi/views.py`
- Pulls intraday 1-minute bars via `yfinance` (`period=7d`, `interval=1m`).
- Returns API response directly; does not write into DB tables.

---

## 4. Transformation Layer (Feature Engineering)

Implementation: `backend/stockproject/predictor/services/feature_engineering.py`

For market series, these engineered fields are computed before persistence:

- `log_return`
- `return_1d`
- `return_7d`
- `return_14d`
- `return_30d`
- `volatility_14d` (rolling std of daily return)
- `momentum_10d`
- `volume_change_1d`
- `atr_14` (Average True Range style rolling mean)
- `relative_strength` (asset close / benchmark close)

Cleaning step:

- `clean_ohlcv(...)`
- Deduplicates by date, enforces numeric OHLCV, drops invalid rows.

---

## 5. Storage Layer (Write Path)

### 5.1 Primary batch sync command

Management command:

- `python manage.py sync_market_data --years 5`
- File: `backend/stockproject/predictor/management/commands/sync_market_data.py`

Behavior:

1. Seeds tracked symbols into metadata table.
2. Fetches historical daily prices for each symbol.
3. Computes engineered features and relative strength.
4. Upserts daily rows into market table.
5. Aggregates and upserts sentiment rows (unless `--skip-sentiment`).

Write semantics:

- Uses `update_or_create` with key `(symbol, date)` for both market and sentiment.
- Ensures idempotent sync and update-safe backfills.

### 5.2 On-demand lazy refresh from API endpoints

Implementation: `backend/stockproject/predictor/analytics_views.py`

If DB rows are missing/stale during API calls:

- `_ensure_symbol_ready(...)` triggers market refresh flow.
- `_ensure_sentiment_ready(...)` triggers sentiment refresh flow.

Throttling keys in cache:

- `market_refresh:<symbol>`
- `sentiment_refresh:<symbol>`

This prevents repeated heavy fetches during high API traffic.

### 5.3 Async hook

Celery task:

- `backend/stockproject/predictor/tasks.py`
- `run_market_sync(...)` calls `sync_market_data` command.

---

## 6. Data Model and SQL Structure

Core models are in:

- `backend/stockproject/predictor/models.py`

Initial migration:

- `backend/stockproject/predictor/migrations/0001_initial.py`

> Note: SQL shown below is schema-equivalent documentation based on Django migration/model definitions.

### 6.1 Table: `predictor_tickermetadata`

Purpose:

- Master symbol registry and business metadata.

Columns:

- `id` BIGINT PK (auto)
- `symbol` VARCHAR(20), UNIQUE, NOT NULL
- `name` VARCHAR(120), NOT NULL
- `asset_type` VARCHAR(20), NOT NULL, default `stock`
- `sector` VARCHAR(80), NOT NULL, default empty string
- `exchange` VARCHAR(20), NOT NULL, default `NSE`
- `benchmark_symbol` VARCHAR(20), NOT NULL, default `^NSEI`
- `is_active` BOOLEAN, NOT NULL, default true
- `created_at` DATETIME/TIMESTAMP, auto now add
- `updated_at` DATETIME/TIMESTAMP, auto now

Constraints/indexes:

- Unique: `symbol`

Representative SQL (SQLite style):

```sql
CREATE TABLE predictor_tickermetadata (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol VARCHAR(20) NOT NULL UNIQUE,
  name VARCHAR(120) NOT NULL,
  asset_type VARCHAR(20) NOT NULL DEFAULT 'stock',
  sector VARCHAR(80) NOT NULL DEFAULT '',
  exchange VARCHAR(20) NOT NULL DEFAULT 'NSE',
  benchmark_symbol VARCHAR(20) NOT NULL DEFAULT '^NSEI',
  is_active BOOLEAN NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL
);
```

### 6.2 Table: `predictor_marketdata`

Purpose:

- Daily OHLCV and engineered technical features for each symbol-date.

Columns:

- `id` BIGINT PK (auto)
- `symbol` VARCHAR(20), indexed, NOT NULL
- `date` DATE, indexed, NOT NULL
- `open` FLOAT, NOT NULL
- `high` FLOAT, NOT NULL
- `low` FLOAT, NOT NULL
- `close` FLOAT, NOT NULL
- `volume` BIGINT, NOT NULL, default 0
- `log_return` FLOAT, NULL
- `return_1d` FLOAT, NULL
- `return_7d` FLOAT, NULL
- `return_14d` FLOAT, NULL
- `return_30d` FLOAT, NULL
- `volatility_14d` FLOAT, NULL
- `atr_14` FLOAT, NULL
- `momentum_10d` FLOAT, NULL
- `volume_change_1d` FLOAT, NULL
- `relative_strength` FLOAT, NULL
- `source` VARCHAR(30), NOT NULL, default `yahoo_finance`
- `created_at` DATETIME/TIMESTAMP, auto now add
- `updated_at` DATETIME/TIMESTAMP, auto now

Constraints/indexes:

- Unique constraint: `(symbol, date)` named `uniq_symbol_date_market_data`
- Indexes on `symbol`, `date`

Representative SQL:

```sql
CREATE TABLE predictor_marketdata (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol VARCHAR(20) NOT NULL,
  date DATE NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume BIGINT NOT NULL DEFAULT 0,
  log_return REAL NULL,
  return_1d REAL NULL,
  return_7d REAL NULL,
  return_14d REAL NULL,
  return_30d REAL NULL,
  volatility_14d REAL NULL,
  atr_14 REAL NULL,
  momentum_10d REAL NULL,
  volume_change_1d REAL NULL,
  relative_strength REAL NULL,
  source VARCHAR(30) NOT NULL DEFAULT 'yahoo_finance',
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  CONSTRAINT uniq_symbol_date_market_data UNIQUE (symbol, date)
);
CREATE INDEX predictor_marketdata_symbol_idx ON predictor_marketdata(symbol);
CREATE INDEX predictor_marketdata_date_idx ON predictor_marketdata(date);
```

### 6.3 Table: `predictor_sentimentdata`

Purpose:

- Daily aggregated sentiment per symbol-date.

Columns:

- `id` BIGINT PK (auto)
- `symbol` VARCHAR(20), indexed, NOT NULL
- `date` DATE, indexed, NOT NULL
- `sentiment_mean` FLOAT, NOT NULL, default 0.0
- `sentiment_std` FLOAT, NOT NULL, default 0.0
- `news_count` INTEGER, NOT NULL, default 0
- `positive_ratio` FLOAT, NOT NULL, default 0.0
- `source` VARCHAR(30), NOT NULL, default `newsapi_gnews`
- `created_at` DATETIME/TIMESTAMP, auto now add
- `updated_at` DATETIME/TIMESTAMP, auto now

Constraints/indexes:

- Unique constraint: `(symbol, date)` named `uniq_symbol_date_sentiment_data`
- Indexes on `symbol`, `date`

Representative SQL:

```sql
CREATE TABLE predictor_sentimentdata (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol VARCHAR(20) NOT NULL,
  date DATE NOT NULL,
  sentiment_mean REAL NOT NULL DEFAULT 0.0,
  sentiment_std REAL NOT NULL DEFAULT 0.0,
  news_count INTEGER NOT NULL DEFAULT 0,
  positive_ratio REAL NOT NULL DEFAULT 0.0,
  source VARCHAR(30) NOT NULL DEFAULT 'newsapi_gnews',
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  CONSTRAINT uniq_symbol_date_sentiment_data UNIQUE (symbol, date)
);
CREATE INDEX predictor_sentimentdata_symbol_idx ON predictor_sentimentdata(symbol);
CREATE INDEX predictor_sentimentdata_date_idx ON predictor_sentimentdata(date);
```

### 6.4 Table: `predictor_predictionrecord`

Purpose:

- Historical model prediction/evaluation records (if/when persisted).

Columns:

- `id` BIGINT PK (auto)
- `symbol` VARCHAR(20), indexed, NOT NULL
- `date` DATE, indexed, NOT NULL
- `horizon_days` SMALLINT UNSIGNED/positive small int, NOT NULL, default 1
- `model_name` VARCHAR(50), NOT NULL
- `predicted_close` FLOAT, NOT NULL
- `actual_close` FLOAT, NULL
- `rmse` FLOAT, NULL
- `mae` FLOAT, NULL
- `mape` FLOAT, NULL
- `directional_accuracy` FLOAT, NULL
- `metadata` JSON, NOT NULL, default empty object
- `created_at` DATETIME/TIMESTAMP, auto now add

Constraints/indexes:

- Indexes on `symbol`, `date`
- No uniqueness constraint declared in initial migration.

Representative SQL:

```sql
CREATE TABLE predictor_predictionrecord (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol VARCHAR(20) NOT NULL,
  date DATE NOT NULL,
  horizon_days INTEGER NOT NULL DEFAULT 1,
  model_name VARCHAR(50) NOT NULL,
  predicted_close REAL NOT NULL,
  actual_close REAL NULL,
  rmse REAL NULL,
  mae REAL NULL,
  mape REAL NULL,
  directional_accuracy REAL NULL,
  metadata JSON NOT NULL DEFAULT '{}',
  created_at DATETIME NOT NULL
);
CREATE INDEX predictor_predictionrecord_symbol_idx ON predictor_predictionrecord(symbol);
CREATE INDEX predictor_predictionrecord_date_idx ON predictor_predictionrecord(date);
```

---

## 7. Evaluation Logic (How Data Is Evaluated)

### 7.1 Inference-time evaluation

Main prediction runtime:

- `backend/stockproject/predictor/asset_aware_predictor.py`

Computed from sequence predictions vs actuals:

- MSE
- RMSE
- MAE
- R2 score
- confidence score (based on std/mean of recent predictions)

Prediction validation step:

- `validate_prediction(...)` clamps extreme predictions based on asset-type-specific max deviation from current price.

Important note:

- These metrics are returned in API responses for model comparison/prediction contexts.
- Current code paths do not show a write operation that persists these runtime metrics into `predictor_predictionrecord` automatically.

### 7.2 Training-time evaluation

Training pipeline:

- `backend/stockproject/predictor/asset_aware_trainer.py`

Training evaluates:

- MAE and RMSE on recent test prediction slice.
- Saves metadata JSON files with:
  - `test_mae`
  - `test_rmse`
  - final train/validation loss
  - sequence length, data size, price range

Artifacts location:

- `backend/stockproject/predictor/models/`
- Includes `.keras` model files, scaler `.pkl` files, metadata `.json` files.

---

## 8. API Endpoints and Their Data Dependencies

Route source: `backend/stockproject/stockapi/urls.py`

- `/api/price-history/`
  - Reads `predictor_marketdata` and overlays `predictor_sentimentdata` by date.
  - Can trigger lazy market refresh if rows missing.

- `/api/technical-indicators/`
  - Reads latest + monthly aggregates from `predictor_marketdata`.

- `/api/sentiment/`
  - Reads `predictor_sentimentdata` and can trigger lazy sentiment refresh.

- `/api/prediction/`
  - Reads `predictor_predictionrecord` history.

- `/api/market-overview/`
  - Reads `predictor_tickermetadata` + latest `predictor_marketdata` snapshots.

- `/api/advanced-analytics/`
  - Reads return/volatility series from `predictor_marketdata`.

- `/api/ohlcv/<symbol>/`
  - Fetch-only from Yahoo intraday; no DB writes.

### 8.1 Graph Plotting: Exactly Where Chart Data Comes From

This section maps dashboard visualizations to storage source and persistence behavior.

#### A) Terminal Dashboard (`frontend/src/pages/TerminalDashboard.jsx`)

1. Main price candlestick chart
  - API: `/api/price-history/`
  - Backend source: `predictor_marketdata`
  - Columns used: `date`, `open`, `high`, `low`, `close`, `volume`
  - Persisted? Yes (daily rows in DB)

2. Price-derived overlays on chart payload
  - API: `/api/price-history/`
  - Backend source: `predictor_marketdata` (+ sentiment join)
  - Columns used: `return_1d`, `return_7d`, `return_14d`, `return_30d`, `volatility_14d`, `atr_14`, `relative_strength`
  - Persisted? Yes (from `predictor_marketdata`), plus merged sentiment fields from `predictor_sentimentdata`

3. Sentiment timeline area chart
  - API: `/api/sentiment/`
  - Backend source: `predictor_sentimentdata`
  - Columns used: `date`, `sentiment_mean`
  - Persisted? Yes

4. News volume + positive ratio composed chart
  - API: `/api/sentiment/`
  - Backend source: `predictor_sentimentdata`
  - Columns used: `date`, `news_count`, `positive_ratio`
  - Persisted? Yes

5. Sentiment donut (positive vs negative meter)
  - API: `/api/sentiment/`
  - Source fields: latest `sentiment_mean`
  - Persisted input? Yes (from `predictor_sentimentdata`)
  - Donut split itself is computed in frontend state (derived, not stored)

6. Volatility vs return scatter chart
  - API: `/api/advanced-analytics/` (fallback to `/api/price-history/` data in frontend)
  - Backend source: `predictor_marketdata`
  - Fields used: `volatility_14d`, `return_1d`
  - Persisted? Yes

7. Return distribution histogram
  - API: `/api/advanced-analytics/`
  - Backend source: derived from `predictor_marketdata.return_1d`
  - Persisted bins? No (histogram bins are computed on request)

8. Current comparison bar (Actual vs LSTM vs Transformer)
  - Inputs:
    - Actual: latest close from `priceHistory` (`/api/price-history/` => `predictor_marketdata.close`)
    - LSTM/Transformer predictions: `/predict/<symbol>/?model=...`
  - Persisted?
    - Actual close: Yes (`predictor_marketdata`)
    - Live prediction result: No, held in frontend state unless separately written by custom code

9. Prediction history table/list in terminal panel
  - API: `/api/prediction/`
  - Backend source: `predictor_predictionrecord`
  - Persisted? Yes if table has rows
  - Important behavior: current live prediction call path appends temporary rows in frontend state for display; this does not automatically persist to DB.

10. Market movers, sector performance, index cards
  - API: `/api/market-overview/`
  - Backend source: `predictor_tickermetadata` + latest snapshots from `predictor_marketdata`
  - Persisted? Yes (source rows are DB rows)
  - Sector aggregates are computed at response time (derived, not stored as a separate table)

#### B) Legacy intraday chart component (`frontend/src/components/StockChart.jsx`)

1. Intraday OHLCV candlestick chart
  - API: `/api/ohlcv/<symbol>/`
  - Backend source: live Yahoo Finance pull in `stockapi/views.py`
  - Persisted? No (response-only data; not inserted into predictor tables)

### 8.2 What Is Stored vs What Is Computed

Stored in DB:

- Daily OHLCV and engineered technical features (`predictor_marketdata`)
- Daily sentiment aggregates (`predictor_sentimentdata`)
- Symbol master metadata (`predictor_tickermetadata`)
- Prediction history records when written (`predictor_predictionrecord`)

Computed at request/runtime (not separately persisted):

- Histogram bins for return distribution
- Sector aggregate performance summaries
- Sentiment meter positive/negative split
- Frontend temporary comparison rows from live prediction responses

Stored on filesystem (not DB):

- Trained model files (`.keras`)
- Scalers (`.pkl`)
- Training metadata JSON (`.json`)
  - Location: `backend/stockproject/predictor/models/`

---

## 9. Data Integrity and Idempotency Rules

Implemented safeguards:

- Unique constraints on `(symbol, date)` for market and sentiment data prevent duplicate daily rows.
- Upsert strategy (`update_or_create`) makes re-syncs deterministic.
- Refresh throttling via cache avoids repeated expensive external API pulls.
- Missing-table handler in analytics views returns explicit operational guidance (`run migrate`).

Current gaps/considerations:

- No explicit FK constraints across core predictor tables (symbol is denormalized string key).
- `PredictionRecord` persistence path is not evident in current backend code flow.
- API keys are currently embedded in `news_sentiment.py` and should be moved to environment variables for production hardening.

---

## 10. Non-project Django Tables (Framework-managed)

When migrations run, Django also creates framework tables such as:

- `django_migrations`
- `django_content_type`
- `auth_user`, `auth_group`, related auth join tables
- `django_admin_log`
- `django_session`

These are standard Django operational tables and are separate from the project-specific analytics schema.

---

## 11. Quick Operational Commands

From `backend/stockproject`:

```bash
python manage.py migrate
python manage.py sync_market_data --years 5
python manage.py runserver
```

Optional targeted sync:

```bash
python manage.py sync_market_data --symbols RELIANCE.NS ^NSEI --years 3
```

---

## 12. End-to-end Data Flow Summary

1. Source fetch:
   - Market prices from Yahoo (fallback TwelveData)
   - News from NewsAPI + GNews
2. Transform:
   - Clean OHLCV
   - Compute engineered features and relative strength
   - Aggregate daily sentiment stats
3. Persist:
   - Upsert to market/sentiment tables keyed by symbol+date
   - Metadata seeded/updated in ticker table
4. Serve:
   - Analytics endpoints read from DB and cache response
   - Optional lazy refresh if gaps are detected
5. Evaluate:
   - Runtime prediction metrics in response payload
   - Training metrics persisted as model metadata files
