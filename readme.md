# Smart Stock Predictor

Smart Stock Predictor is a terminal-style market analytics and forecasting platform.
It combines a Django backend for market data, sentiment, and model-serving APIs with a React + Vite frontend that renders a high-density trading dashboard.

The latest version is terminal-first: the primary experience is the Terminal Dashboard with price/volume charts, market breadth, sector analytics, sentiment panels, and model comparison outputs.

## Live Architecture

- Frontend: React 18, Vite, Tailwind CSS, ApexCharts, Recharts, Axios
- Backend: Django 5.1, Django REST Framework, CORS, Celery-ready task hooks
- Data sources: Yahoo Finance (primary), TwelveData fallback
- Sentiment: NewsAPI + GNews ingestion, TextBlob scoring
- Storage: SQLite by default, PostgreSQL when DATABASE_URL is provided

## Key Capabilities

- Terminal-style market workstation UI
- Market overview with gainers, losers, sectors, and index trackers
- Price history, technical indicators, volatility and returns analytics
- Daily sentiment timeline, positive ratio, and article count overlays
- Prediction endpoints for LSTM and Transformer model outputs
- Model comparison endpoint for side-by-side forecasting
- Caching layer for responsive analytics endpoints

## Repository Structure

- [backend](backend): Django project and prediction services
- [backend/stockproject/predictor](backend/stockproject/predictor): ML prediction app, analytics APIs, sentiment, training artifacts
- [backend/stockproject/stockapi](backend/stockproject/stockapi): Market and analytics API router
- [frontend](frontend): React application with Terminal Dashboard UI
- [frontend/src/pages/TerminalDashboard.jsx](frontend/src/pages/TerminalDashboard.jsx): Main dashboard page

## Local Setup

### 1) Backend Setup

1. Go to [backend/stockproject](backend/stockproject)
2. Create a virtual environment
	python -m venv .venv
3. Activate it (Windows PowerShell)
	.\\.venv\\Scripts\\Activate.ps1
4. Install dependencies
	pip install -r predictor/requirements.txt
5. Run migrations
	python manage.py migrate
6. Start backend server
	python manage.py runserver

Backend runs at http://127.0.0.1:8000

### Backend First-Run Checks

If you pull the project fresh or switch branches with model changes, always run:

python manage.py makemigrations
python manage.py migrate

If you see an error like `no such table: predictor_tickermetadata`, it means migrations were not applied to your local database yet.

### 2) Frontend Setup

1. Go to [frontend](frontend)
2. Install packages
	npm install
3. Start dev server
	npm run dev

Frontend runs at http://127.0.0.1:5173

## API Overview

Base routes are configured in [backend/stockproject/stockproject/urls.py](backend/stockproject/stockproject/urls.py):

- /api/ for analytics and market APIs
- /predict/ for prediction and model-comparison APIs

### Market and Analytics Routes

Defined in [backend/stockproject/stockapi/urls.py](backend/stockproject/stockapi/urls.py):

- GET /api/health/
- GET /api/stocks/
- GET /api/market-overview/
- GET /api/price-history/?symbol=RELIANCE.NS&days=365
- GET /api/technical-indicators/?symbol=RELIANCE.NS
- GET /api/sentiment/?symbol=RELIANCE.NS&days=180
- GET /api/prediction/?symbol=RELIANCE.NS&horizon_days=1
- GET /api/advanced-analytics/?symbol=RELIANCE.NS&days=365
- GET /api/model_comparison/?symbol=RELIANCE.NS

### Prediction Routes

Defined in [backend/stockproject/predictor/urls.py](backend/stockproject/predictor/urls.py):

- GET /predict/health/
- GET /predict/stocks/
- GET /predict/model-comparison/?symbol=RELIANCE.NS
- GET /predict/RELIANCE.NS/?model=lstm
- GET /predict/RELIANCE.NS/?model=transformer

## Data and Persistence

- Default database is SQLite for local development.
- If DATABASE_URL is present, Django switches to PostgreSQL automatically.
- Core analytics tables include ticker metadata, market data, sentiment data, and prediction records.

## Optional Data Sync Command

To backfill market and sentiment data using management command:

python manage.py sync_market_data --years 5

Implementation lives in [backend/stockproject/predictor/management/commands/sync_market_data.py](backend/stockproject/predictor/management/commands/sync_market_data.py)

## Notes

- The frontend root route now points to the terminal dashboard in [frontend/src/App.jsx](frontend/src/App.jsx).
- Some external API integrations currently use hardcoded keys in [backend/stockproject/predictor/news_sentiment.py](backend/stockproject/predictor/news_sentiment.py). Move these to environment variables before production use.

