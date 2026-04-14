from __future__ import annotations

from datetime import datetime, timedelta

from django.core.cache import cache
from django.db import OperationalError, ProgrammingError
from django.db.models import Avg
from django.db.models.functions import TruncMonth
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response

from predictor.models import MarketData, PredictionRecord, SentimentData, TickerMetadata
from predictor.services.data_sources import MultiSourceMarketDataFetcher
from predictor.services.feature_engineering import clean_ohlcv, compute_financial_features, compute_relative_strength
from predictor.services.sentiment_pipeline import aggregate_daily_sentiment


TRACKED_UNIVERSE = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "asset_type": "stock", "sector": "Energy", "benchmark_symbol": "^NSEI"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank", "asset_type": "stock", "sector": "Financial Services", "benchmark_symbol": "^NSEBANK"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "asset_type": "stock", "sector": "Financial Services", "benchmark_symbol": "^NSEBANK"},
    {"symbol": "ONGC.NS", "name": "Oil and Natural Gas Corporation", "asset_type": "stock", "sector": "Energy", "benchmark_symbol": "^NSEI"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "asset_type": "stock", "sector": "Financial Services", "benchmark_symbol": "^NSEBANK"},
    {"symbol": "INFY.NS", "name": "Infosys", "asset_type": "stock", "sector": "Information Technology", "benchmark_symbol": "^NSEI"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "asset_type": "stock", "sector": "Information Technology", "benchmark_symbol": "^NSEI"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "asset_type": "stock", "sector": "Financial Services", "benchmark_symbol": "^NSEBANK"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "asset_type": "stock", "sector": "Financial Services", "benchmark_symbol": "^NSEBANK"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports and SEZ", "asset_type": "stock", "sector": "Industrials", "benchmark_symbol": "^NSEI"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "asset_type": "stock", "sector": "Industrials", "benchmark_symbol": "^NSEI"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "asset_type": "stock", "sector": "Financial Services", "benchmark_symbol": "^NSEBANK"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "asset_type": "stock", "sector": "Telecommunications", "benchmark_symbol": "^NSEI"},
    {"symbol": "^NSEI", "name": "NIFTY 50", "asset_type": "index", "sector": "Broad Market", "benchmark_symbol": "^NSEI"},
    {"symbol": "^NSEBANK", "name": "BANK NIFTY", "asset_type": "index", "sector": "Financial Services", "benchmark_symbol": "^NSEBANK"},
    {"symbol": "^NSEMDCP50", "name": "NIFTY MIDCAP 50", "asset_type": "index", "sector": "Broad Market", "benchmark_symbol": "^NSEI"},
    {"symbol": "^CNXAUTO", "name": "NIFTY AUTO", "asset_type": "index", "sector": "Automobile", "benchmark_symbol": "^NSEI"},
]

_fetcher = MultiSourceMarketDataFetcher()


def _seed_universe_metadata():
    for row in TRACKED_UNIVERSE:
        TickerMetadata.objects.update_or_create(symbol=row["symbol"], defaults=row)


def _refresh_market_data(symbol, benchmark_symbol="^NSEI", min_years=1):
    latest = MarketData.objects.filter(symbol=symbol).order_by("-date").first()
    today = timezone.now().date()
    # Avoid blocking requests by refreshing too aggressively. A few days of staleness is acceptable
    # for this terminal, and deep backfills are still triggered when a window is missing.
    if latest and latest.date >= (today - timedelta(days=5)):
        return

    throttle_key = f"market_refresh:{symbol}"
    if cache.get(throttle_key):
        return
    cache.set(throttle_key, True, timeout=180)

    years = max(int(min_years or 1), 2 if latest is None else 1)
    price_result = _fetcher.fetch_daily_prices(symbol, years=years)
    asset_df = compute_financial_features(clean_ohlcv(price_result.data))

    if symbol.startswith("^"):
        asset_df["relative_strength"] = 1.0
    else:
        bench_result = _fetcher.fetch_daily_prices(benchmark_symbol, years=years)
        bench_df = clean_ohlcv(bench_result.data)
        rs_frame = compute_relative_strength(asset_df, bench_df)
        asset_df = asset_df.merge(rs_frame, on="date", how="left")

    for _, row in asset_df.iterrows():
        MarketData.objects.update_or_create(
            symbol=symbol,
            date=row["date"],
            defaults={
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0) or 0),
                "log_return": _to_float(row.get("log_return")),
                "return_1d": _to_float(row.get("return_1d")),
                "return_7d": _to_float(row.get("return_7d")),
                "return_14d": _to_float(row.get("return_14d")),
                "return_30d": _to_float(row.get("return_30d")),
                "volatility_14d": _to_float(row.get("volatility_14d")),
                "atr_14": _to_float(row.get("atr_14")),
                "momentum_10d": _to_float(row.get("momentum_10d")),
                "volume_change_1d": _to_float(row.get("volume_change_1d")),
                "relative_strength": _to_float(row.get("relative_strength")),
                "source": price_result.source,
            },
        )


def _ensure_symbol_ready(symbol, min_years=1):
    _seed_universe_metadata()
    metadata = TickerMetadata.objects.filter(symbol=symbol).first()
    benchmark_symbol = metadata.benchmark_symbol if metadata else "^NSEI"
    try:
        _refresh_market_data(symbol, benchmark_symbol=benchmark_symbol, min_years=min_years)
    except Exception:
        # Keep API responsive even if live refresh fails for one symbol.
        pass


def _ensure_sentiment_ready(symbol, days=365):
    start_date = timezone.now().date() - timedelta(days=days)
    if SentimentData.objects.filter(symbol=symbol, date__gte=start_date).exists():
        return

    throttle_key = f"sentiment_refresh:{symbol}"
    if cache.get(throttle_key):
        return
    cache.set(throttle_key, True, timeout=1800)

    metadata = TickerMetadata.objects.filter(symbol=symbol).first()
    company_name = metadata.name if metadata and metadata.name else symbol

    try:
        daily_sentiment = aggregate_daily_sentiment(symbol=symbol, company_name=company_name)
        for day, values in daily_sentiment.items():
            SentimentData.objects.update_or_create(
                symbol=symbol,
                date=datetime.fromisoformat(day).date(),
                defaults={
                    "sentiment_mean": values.get("sentiment_mean", 0.0),
                    "sentiment_std": values.get("sentiment_std", 0.0),
                    "news_count": values.get("news_count", 0),
                    "positive_ratio": values.get("positive_ratio", 0.0),
                },
            )
    except Exception:
        # Do not break the API if external news providers fail.
        pass


def _to_float(value):
    if value is None:
        return None
    try:
        if str(value).lower() == "nan":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_market_snapshot(symbol: str):
    rows = list(
        MarketData.objects.filter(symbol=symbol)
        .order_by("-date")
        .values(
            "date",
            "close",
            "return_1d",
            "return_7d",
            "return_30d",
            "volatility_14d",
            "volume",
            "source",
        )[:31]
    )
    if not rows:
        return None

    latest = dict(rows[0])
    if len(rows) > 1 and rows[1].get("close"):
        latest["return_1d"] = (latest["close"] / rows[1]["close"]) - 1
    if len(rows) > 7 and rows[7].get("close"):
        latest["return_7d"] = (latest["close"] / rows[7]["close"]) - 1
    if len(rows) > 30 and rows[30].get("close"):
        latest["return_30d"] = (latest["close"] / rows[30]["close"]) - 1
    return latest


def _get_window_days(request, default=365):
    try:
        return int(request.query_params.get("days", default))
    except ValueError:
        return default


def _missing_table_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "no such table" in message or "doesn't exist" in message


def _schema_not_ready_response(exc: Exception):
    return Response(
        {
            "error": "Database schema is not ready. Run 'python manage.py migrate' from backend/stockproject and restart the server.",
            "details": str(exc),
        },
        status=503,
    )


def _cached_response(cache_key, producer, ttl_seconds=90):
    payload = cache.get(cache_key)
    if payload is None:
        try:
            payload = producer()
        except (OperationalError, ProgrammingError) as exc:
            if _missing_table_error(exc):
                return _schema_not_ready_response(exc)
            raise
        cache.set(cache_key, payload, timeout=ttl_seconds)
    return Response(payload)


@api_view(["GET"])
def price_history(request):
    symbol = request.query_params.get("symbol")
    if not symbol:
        return Response({"error": "symbol query parameter is required"}, status=400)

    days = _get_window_days(request, default=730)
    start_date = timezone.now().date() - timedelta(days=days)
    years_needed = max(1, (days // 365) + 1)

    cache_key = f"price_history:{symbol}:{days}"

    def produce():
        def fetch_rows():
            return list(
                MarketData.objects.filter(symbol=symbol, date__gte=start_date)
                .order_by("date")
                .values(
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "return_1d",
                    "return_7d",
                    "return_14d",
                    "return_30d",
                    "volatility_14d",
                    "relative_strength",
                    "atr_14",
                )
            )

        market_rows = fetch_rows()
        if not market_rows:
            try:
                _ensure_symbol_ready(symbol, min_years=years_needed)
            except Exception:
                pass
            market_rows = fetch_rows()

        # Ensure chart integrity: if large windows return sparse rows, retry with a lightweight refresh.
        if len(market_rows) < 20:
            try:
                _ensure_symbol_ready(symbol, min_years=years_needed)
            except Exception:
                pass
            market_rows = fetch_rows()

        if market_rows:
            oldest = market_rows[0].get("date")
            if oldest and oldest > start_date:
                try:
                    _ensure_symbol_ready(symbol, min_years=years_needed)
                except Exception:
                    pass
                market_rows = fetch_rows()

        sentiment_rows = SentimentData.objects.filter(symbol=symbol, date__gte=start_date).values(
            "date", "sentiment_mean", "news_count", "positive_ratio"
        )
        sentiment_map = {row["date"]: row for row in sentiment_rows}

        for row in market_rows:
            sentiment = sentiment_map.get(row["date"])
            row["sentiment_mean"] = sentiment["sentiment_mean"] if sentiment else 0.0
            row["news_count"] = sentiment["news_count"] if sentiment else 0
            row["positive_ratio"] = sentiment["positive_ratio"] if sentiment else 0.0

        return {"symbol": symbol, "days": days, "series": market_rows}

    return _cached_response(cache_key, produce)


@api_view(["GET"])
def technical_indicators(request):
    symbol = request.query_params.get("symbol")
    if not symbol:
        return Response({"error": "symbol query parameter is required"}, status=400)

    cache_key = f"technical_indicators:{symbol}"

    def produce():
        latest = (
            MarketData.objects.filter(symbol=symbol)
            .order_by("-date")
            .values(
                "date",
                "close",
                "return_1d",
                "return_7d",
                "return_14d",
                "return_30d",
                "volatility_14d",
                "momentum_10d",
                "volume_change_1d",
                "relative_strength",
                "atr_14",
            )
            .first()
        )
        if not latest:
            try:
                _ensure_symbol_ready(symbol)
            except Exception:
                pass
            latest = (
                MarketData.objects.filter(symbol=symbol)
                .order_by("-date")
                .values(
                    "date",
                    "close",
                    "return_1d",
                    "return_7d",
                    "return_14d",
                    "return_30d",
                    "volatility_14d",
                    "momentum_10d",
                    "volume_change_1d",
                    "relative_strength",
                    "atr_14",
                )
                .first()
            )
        if not latest:
            return {"error": "No market data found for symbol", "status": 404}

        monthly = list(
            MarketData.objects.filter(symbol=symbol)
            .annotate(month=TruncMonth("date"))
            .values("month")
            .annotate(avg_volatility=Avg("volatility_14d"), avg_return=Avg("return_1d"))
            .order_by("month")
        )

        return {"symbol": symbol, "latest": latest, "monthly": monthly}

    payload = produce()
    if payload.get("status") == 404:
        return Response({"error": payload["error"]}, status=404)
    cache.set(cache_key, payload, timeout=90)
    return Response(payload)


@api_view(["GET"])
def sentiment_view(request):
    symbol = request.query_params.get("symbol")
    if not symbol:
        return Response({"error": "symbol query parameter is required"}, status=400)

    days = _get_window_days(request, default=365)
    start_date = timezone.now().date() - timedelta(days=days)

    cache_key = f"sentiment:{symbol}:{days}"

    def produce():
        rows = list(
            SentimentData.objects.filter(symbol=symbol, date__gte=start_date)
            .order_by("date")
            .values("date", "sentiment_mean", "sentiment_std", "news_count", "positive_ratio")
        )

        if not rows:
            _ensure_sentiment_ready(symbol, days=days)
            rows = list(
                SentimentData.objects.filter(symbol=symbol, date__gte=start_date)
                .order_by("date")
                .values("date", "sentiment_mean", "sentiment_std", "news_count", "positive_ratio")
            )

        latest = (
            SentimentData.objects.filter(symbol=symbol)
            .order_by("-date")
            .values("date", "sentiment_mean", "sentiment_std", "news_count", "positive_ratio")
            .first()
        )

        return {"symbol": symbol, "latest": latest, "series": rows}

    return _cached_response(cache_key, produce)


@api_view(["GET"])
def prediction_view(request):
    symbol = request.query_params.get("symbol")
    if not symbol:
        return Response({"error": "symbol query parameter is required"}, status=400)

    horizon_days = int(request.query_params.get("horizon_days", 1))
    cache_key = f"prediction:{symbol}:{horizon_days}"

    def produce():
        rows = list(
            PredictionRecord.objects.filter(symbol=symbol, horizon_days=horizon_days)
            .order_by("-date")
            .values(
                "date",
                "model_name",
                "predicted_close",
                "actual_close",
                "rmse",
                "mae",
                "mape",
                "directional_accuracy",
            )[:200]
        )
        return {"symbol": symbol, "horizon_days": horizon_days, "series": rows}

    return _cached_response(cache_key, produce)


@api_view(["GET"])
def market_overview(request):
    cache_key = "market_overview"

    def produce():
        _seed_universe_metadata()
        metadata = list(
            TickerMetadata.objects.filter(is_active=True)
            .values("symbol", "name", "asset_type", "sector", "benchmark_symbol")
            .order_by("asset_type", "symbol")
        )

        latest_market_rows = []
        missing_metadata = []

        for item in metadata:
            latest = _latest_market_snapshot(item["symbol"])
            if latest:
                latest_market_rows.append({**item, **latest})
            else:
                missing_metadata.append(item)

        # Prime symbols that have no snapshot yet so dashboard cards are populated
        # without forcing users to open each symbol page first.
        for item in missing_metadata:
            try:
                _ensure_symbol_ready(item["symbol"], min_years=1)
            except Exception:
                pass

        for item in missing_metadata:
            latest = _latest_market_snapshot(item["symbol"])
            if latest:
                latest_market_rows.append({**item, **latest})

        market_rows = [x for x in latest_market_rows if x["asset_type"] == "stock"]
        ranked_rows = [x for x in market_rows if x.get("return_1d") is not None]

        gainers = sorted(
            [x for x in ranked_rows if x.get("return_1d", 0) > 0],
            key=lambda x: x.get("return_1d"),
            reverse=True,
        )[:6]

        losers = sorted(
            [x for x in ranked_rows if x.get("return_1d", 0) < 0],
            key=lambda x: x.get("return_1d"),
        )[:6]

        # Fallback to best/worst ranked rows if one side has no sign-specific rows.
        if not gainers:
            gainers = sorted(ranked_rows, key=lambda x: x.get("return_1d"), reverse=True)[:6]
        if not losers:
            losers = sorted(ranked_rows, key=lambda x: x.get("return_1d"))[:6]

        gainer_symbols = {row["symbol"] for row in gainers}
        losers = [row for row in losers if row["symbol"] not in gainer_symbols][:6]

        sector_summary = {}
        for row in market_rows:
            sector = row.get("sector") or "Other"
            sector_summary.setdefault(sector, []).append(row)

        sector_perf = []
        for sector, rows in sector_summary.items():
            returns_1d = [r.get("return_1d") for r in rows if r.get("return_1d") is not None]
            returns_30d = [r.get("return_30d") for r in rows if r.get("return_30d") is not None]
            vol_14d = [r.get("volatility_14d") for r in rows if r.get("volatility_14d") is not None]

            if not returns_1d:
                continue

            avg_1d = sum(returns_1d) / len(returns_1d)
            avg_30d = sum(returns_30d) / len(returns_30d) if returns_30d else None
            avg_vol = sum(vol_14d) / len(vol_14d) if vol_14d else None
            sector_perf.append(
                {
                    "sector": sector,
                    "avg_return_1d": avg_1d,
                    "avg_return_30d": avg_30d,
                    "avg_volatility": avg_vol,
                    "count": len(returns_1d),
                }
            )

        indices = [x for x in latest_market_rows if x["asset_type"] == "index"]

        return {
            "as_of": timezone.now().isoformat(),
            "tickers": metadata,
            "stocks": market_rows,
            "gainers": gainers,
            "losers": losers,
            "sector_performance": sorted(sector_perf, key=lambda x: x["avg_return_1d"], reverse=True),
            "indices": indices,
        }

    return _cached_response(cache_key, produce)


@api_view(["GET"])
def advanced_analytics(request):
    symbol = request.query_params.get("symbol")
    if not symbol:
        return Response({"error": "symbol query parameter is required"}, status=400)

    days = _get_window_days(request, default=365)
    start_date = timezone.now().date() - timedelta(days=days)
    cache_key = f"advanced_analytics:{symbol}:{days}"

    def produce():
        series = list(
            MarketData.objects.filter(symbol=symbol, date__gte=start_date)
            .order_by("date")
            .values("date", "return_1d", "volatility_14d", "return_30d", "relative_strength")
        )

        values = [
            (row.get("return_1d") or 0) * 100
            for row in series
            if row.get("return_1d") is not None
        ]
        bins = [
            {"label": "<-2%", "min": float("-inf"), "max": -2, "count": 0},
            {"label": "-2 to -1%", "min": -2, "max": -1, "count": 0},
            {"label": "-1 to 0%", "min": -1, "max": 0, "count": 0},
            {"label": "0 to 1%", "min": 0, "max": 1, "count": 0},
            {"label": "1 to 2%", "min": 1, "max": 2, "count": 0},
            {"label": ">2%", "min": 2, "max": float("inf"), "count": 0},
        ]

        for value in values:
            for bucket in bins:
                if bucket["min"] <= value < bucket["max"]:
                    bucket["count"] += 1
                    break

        histogram = [{"label": item["label"], "count": item["count"]} for item in bins]

        return {
            "symbol": symbol,
            "days": days,
            "volatility_series": [
                {
                    "date": row["date"],
                    "volatility_14d": row.get("volatility_14d"),
                    "return_1d": row.get("return_1d"),
                    "return_30d": row.get("return_30d"),
                }
                for row in series
            ],
            "returns_distribution": histogram,
        }

    return _cached_response(cache_key, produce)
