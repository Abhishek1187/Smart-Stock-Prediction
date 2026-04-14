from __future__ import annotations

import numpy as np
import pandas as pd


def compute_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered = engineered.sort_values("date").reset_index(drop=True)

    engineered["log_return"] = np.log(engineered["close"] / engineered["close"].shift(1))
    engineered["return_1d"] = engineered["close"].pct_change(1)
    engineered["return_7d"] = engineered["close"].pct_change(7)
    engineered["return_14d"] = engineered["close"].pct_change(14)
    engineered["return_30d"] = engineered["close"].pct_change(30)
    engineered["volatility_14d"] = engineered["return_1d"].rolling(14).std()
    engineered["momentum_10d"] = engineered["close"] - engineered["close"].shift(10)
    engineered["volume_change_1d"] = engineered["volume"].pct_change(1)

    high_low = engineered["high"] - engineered["low"]
    high_close = (engineered["high"] - engineered["close"].shift()).abs()
    low_close = (engineered["low"] - engineered["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    engineered["atr_14"] = tr.rolling(14).mean()

    return engineered


def compute_relative_strength(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    left = df[["date", "close"]].rename(columns={"close": "asset_close"})
    right = benchmark_df[["date", "close"]].rename(columns={"close": "benchmark_close"})

    merged = left.merge(right, on="date", how="left")
    merged["benchmark_close"] = merged["benchmark_close"].ffill().bfill()
    merged["relative_strength"] = merged["asset_close"] / merged["benchmark_close"]
    return merged[["date", "relative_strength"]]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=["date"]).sort_values("date")

    for col in ["open", "high", "low", "close", "volume"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=["open", "high", "low", "close"])
    cleaned["volume"] = cleaned["volume"].fillna(0)
    return cleaned.reset_index(drop=True)
