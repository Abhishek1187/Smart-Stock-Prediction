import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


@dataclass
class PriceFetchResult:
    data: pd.DataFrame
    source: str


class TwelveDataProvider:
    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TWELVEDATA_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def fetch_daily(self, symbol: str, outputsize: int = 1250) -> Optional[pd.DataFrame]:
        if not self.is_configured():
            return None

        params = {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": outputsize,
            "apikey": self.api_key,
            "timezone": "Asia/Kolkata",
            "format": "JSON",
        }
        response = requests.get(self.BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if "values" not in payload:
            return None

        df = pd.DataFrame(payload["values"])
        if df.empty:
            return None

        df.rename(columns={"datetime": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return (
            df[["date", "open", "high", "low", "close", "volume"]]
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )


class YahooFinanceProvider:
    def fetch_daily(self, symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
        ticker = yf.Ticker(symbol)
        frame = ticker.history(period=period, interval="1d", auto_adjust=False)
        if frame.empty:
            return None

        frame = frame.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        return (
            frame[["date", "open", "high", "low", "close", "volume"]]
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )


class MultiSourceMarketDataFetcher:
    def __init__(self):
        self.primary = YahooFinanceProvider()
        self.fallback = TwelveDataProvider()

    def fetch_daily_prices(self, symbol: str, years: int = 5) -> PriceFetchResult:
        normalized = self._normalize_symbol(symbol)
        outputsize = max(365, min(5000, years * 365))
        period = f"{max(1, years)}y"

        primary_df = self.primary.fetch_daily(symbol, period=period)
        if primary_df is not None and not primary_df.empty:
            return PriceFetchResult(data=primary_df, source="yahoo_finance")

        fallback_df = self.fallback.fetch_daily(normalized, outputsize=outputsize)
        if fallback_df is not None and not fallback_df.empty:
            return PriceFetchResult(data=fallback_df, source="twelvedata")

        raise ValueError(f"No market data found for symbol {symbol}")

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if symbol.startswith("^"):
            return symbol.replace("^", "")
        return symbol


def to_iso_date(value) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
