# Deployment Guide

Step-by-step instructions to reproduce the full deployment: local Docker Compose, Render (backend), and Vercel (frontend), wired together by GitHub Actions.

## 1. Local: Docker Compose

Prerequisites: Docker Desktop running.

```bash
cp backend/stockproject/.env.example backend/stockproject/.env
# fill in NEWS_API_KEY / GNEWS_API_KEY / TWELVEDATA_API_KEY if you have them
# (the app degrades gracefully without them - sentiment/fallback data sources just stay disabled)

docker compose up --build
```

This starts three containers (`time-series-system-db`, `-backend`, `-frontend`) on a private network:

| Service | URL | Notes |
|---|---|---|
| Frontend | http://localhost:5173 | Served by nginx, static Vite build |
| Backend | http://localhost:8000 | gunicorn + Django, migrations run automatically on start |
| Postgres | internal only (`db:5432`) | Data persists in the `postgres_data` named volume |

Override the host ports with `BACKEND_PORT=8001 FRONTEND_PORT=5174 docker compose up` if 8000/5173 are already taken locally.

Verify:

```bash
curl http://localhost:8000/api/health/
curl http://localhost:8000/predict/health/
```

Tear down: `docker compose down -v` (the `-v` also removes the Postgres volume, i.e. a full reset).

## 2. Render: backend hosting

Render reads `render.yaml` (a "Blueprint") from the repo root and provisions both the web service and its database in one step.

1. In the Render dashboard: **New → Blueprint**, point it at this GitHub repo.
2. Render provisions:
   - `time-series-system-db` — a free-tier managed Postgres instance.
   - `time-series-system-backend` — a `docker` runtime web service built from `backend/Dockerfile` with build context `backend/`.
3. Fill in the env vars marked `sync: false` in `render.yaml` (Render dashboard → service → Environment):
   - `ALLOWED_HOSTS` — the `*.onrender.com` domain Render assigns after first deploy.
   - `CORS_ALLOWED_ORIGINS` — the Vercel frontend URL from step 3 below.
   - `NEWS_API_KEY`, `GNEWS_API_KEY`, `TWELVEDATA_API_KEY` — your own keys (see [API key rotation](#api-key-rotation) below).
4. **Turn off Render's auto-deploy for this service** (Settings → Auto-Deploy → off). GitHub Actions is the deploy trigger (step 4 below) — leaving both on causes duplicate/racing deploys.
5. Create a Deploy Hook (Settings → Deploy Hook → Create Hook), copy the URL.
6. In the GitHub repo: **Settings → Secrets and variables → Actions**, add `RENDER_DEPLOY_HOOK_URL` with that URL.

## 3. Vercel: frontend hosting

1. Import the repo into Vercel, framework preset **Vite**, root directory `frontend/`.
2. Set the environment variable `VITE_API_BASE_URL` to the Render backend URL (e.g. `https://time-series-system-backend.onrender.com`), for both Production and Preview.
3. **Disconnect Vercel's Git integration** (Settings → Git → Disconnect) so it doesn't auto-deploy on push — GitHub Actions is the deploy trigger instead (step 4).
4. Generate the values needed for Actions: run `vercel link` locally once to create `.vercel/project.json` (gives you `orgId`/`projectId`), and create a token at vercel.com/account/tokens.
5. Add `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID` as GitHub Actions repo secrets.
6. Go back to Render and set `CORS_ALLOWED_ORIGINS` to this Vercel URL, then redeploy.

## 4. GitHub Actions: the deploy trigger

Once the secrets above are set, every push to `main` runs `.github/workflows/ci.yml`:

```
backend-tests ─┐
                ├─→ docker-build-and-scan ─→ deploy-backend (Render deploy hook)
frontend-tests ┴──────────────────────────→ deploy-frontend (Vercel CLI, --prod)
```

Deploys only fire on `push` to `main` (not on pull requests), and only after their upstream test/build/scan jobs succeed. `codeql.yml` and `gitleaks.yml` run independently on push/PR/schedule and report to the repo's Security tab.

## 5. Verifying the live deployment

```bash
curl https://<your-render-domain>/api/health/
curl https://<your-render-domain>/predict/health/
```

Then open the Vercel URL, load a symbol, and request a prediction — check the browser network tab for a clean `200` with no CORS errors.

## API key rotation

The repo's git history contains two now-revoked NewsAPI/GNews keys that were previously hardcoded in `predictor/news_sentiment.py`. If you fork or reuse this repo:

1. Generate your own free-tier keys at [newsapi.org](https://newsapi.org) and [gnews.io](https://gnews.io).
2. Set them as `NEWS_API_KEY` / `GNEWS_API_KEY` in your local `.env` and in the Render dashboard — never commit them.
3. The app runs fine without them; sentiment features just report empty/disabled until keys are present (see `NEWS_API_DISABLED` / `GNEWS_API_DISABLED` fallback logic in `news_sentiment.py`).
