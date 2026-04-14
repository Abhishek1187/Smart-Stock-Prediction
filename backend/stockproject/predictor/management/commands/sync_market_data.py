from __future__ import annotations

from datetime import datetime

from django.core.management.base import BaseCommand
from django.db import transaction

from predictor.models import MarketData, SentimentData, TickerMetadata
from predictor.services.data_sources import MultiSourceMarketDataFetcher
from predictor.services.feature_engineering import (
    clean_ohlcv,
    compute_financial_features,
    compute_relative_strength,
)
from predictor.services.sentiment_pipeline import aggregate_daily_sentiment


DEFAULT_TICKERS = [
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
    {"symbol": "^NSEBANK", "name": "NIFTY BANK", "asset_type": "index", "sector": "Financial Services", "benchmark_symbol": "^NSEBANK"},
    {"symbol": "^NSEMDCP50", "name": "NIFTY MIDCAP 50", "asset_type": "index", "sector": "Broad Market", "benchmark_symbol": "^NSEI"},
    {"symbol": "^CNXAUTO", "name": "NIFTY AUTO", "asset_type": "index", "sector": "Automobile", "benchmark_symbol": "^NSEI"},
]


class Command(BaseCommand):
    help = "Sync market data and sentiment into normalized analytics tables."

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            nargs="+",
            type=str,
            help="Optional symbols to sync. Defaults to active metadata symbols.",
        )
        parser.add_argument(
            "--years",
            type=int,
            default=5,
            help="Historical years to fetch for each symbol (default: 5).",
        )
        parser.add_argument(
            "--skip-sentiment",
            action="store_true",
            help="Skip sentiment aggregation during sync.",
        )

    def handle(self, *args, **options):
        symbols = options.get("symbols")
        years = int(options.get("years") or 5)
        skip_sentiment = bool(options.get("skip_sentiment"))
        fetcher = MultiSourceMarketDataFetcher()

        self._seed_metadata()

        if not symbols:
            symbols = list(
                TickerMetadata.objects.filter(is_active=True).values_list("symbol", flat=True)
            )

        benchmark_cache = {}

        for symbol in symbols:
            metadata = TickerMetadata.objects.filter(symbol=symbol).first()
            benchmark_symbol = metadata.benchmark_symbol if metadata else "^NSEI"

            self.stdout.write(f"Syncing {symbol}...")

            try:
                price_result = fetcher.fetch_daily_prices(symbol, years=years)
                ohlcv = clean_ohlcv(price_result.data)
                features = compute_financial_features(ohlcv)

                if symbol.startswith("^"):
                    features["relative_strength"] = 1.0
                else:
                    if benchmark_symbol not in benchmark_cache:
                        benchmark_result = fetcher.fetch_daily_prices(benchmark_symbol, years=years)
                        benchmark_cache[benchmark_symbol] = clean_ohlcv(benchmark_result.data)

                    rs_frame = compute_relative_strength(features, benchmark_cache[benchmark_symbol])
                    features = features.merge(rs_frame, on="date", how="left", suffixes=("", "_rs"))

                self._upsert_market_data(symbol, features, price_result.source)
                if not skip_sentiment:
                    self._upsert_sentiment(symbol, metadata.name if metadata else symbol)

                self.stdout.write(self.style.SUCCESS(f"Synced {symbol} successfully."))
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f"Failed syncing {symbol}: {exc}"))

    def _seed_metadata(self):
        for row in DEFAULT_TICKERS:
            TickerMetadata.objects.update_or_create(
                symbol=row["symbol"],
                defaults=row,
            )

    @transaction.atomic
    def _upsert_market_data(self, symbol, frame, source):
        for _, row in frame.iterrows():
            defaults = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0) or 0),
                "log_return": self._to_float(row.get("log_return")),
                "return_1d": self._to_float(row.get("return_1d")),
                "return_7d": self._to_float(row.get("return_7d")),
                "return_14d": self._to_float(row.get("return_14d")),
                "return_30d": self._to_float(row.get("return_30d")),
                "volatility_14d": self._to_float(row.get("volatility_14d")),
                "atr_14": self._to_float(row.get("atr_14")),
                "momentum_10d": self._to_float(row.get("momentum_10d")),
                "volume_change_1d": self._to_float(row.get("volume_change_1d")),
                "relative_strength": self._to_float(row.get("relative_strength")),
                "source": source,
            }
            MarketData.objects.update_or_create(
                symbol=symbol,
                date=row["date"],
                defaults=defaults,
            )

    @transaction.atomic
    def _upsert_sentiment(self, symbol, company_name):
        daily_sentiment = aggregate_daily_sentiment(symbol=symbol, company_name=company_name)
        for day, values in daily_sentiment.items():
            SentimentData.objects.update_or_create(
                symbol=symbol,
                date=datetime.fromisoformat(day).date(),
                defaults={
                    "sentiment_mean": values["sentiment_mean"],
                    "sentiment_std": values["sentiment_std"],
                    "news_count": values["news_count"],
                    "positive_ratio": values["positive_ratio"],
                },
            )

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            if str(value).lower() == "nan":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
