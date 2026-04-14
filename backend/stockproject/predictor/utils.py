import pandas as pd
import yfinance as yf
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import requests
import time
import json

# Patch numpy for pandas_ta compatibility
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if not hasattr(np, 'NaT'):
    import pandas._libs.tslibs.nattype as nattype
    np.NaT = nattype.NaT

import pandas_ta_classic as ta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

feature_scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
target_scaler_path = os.path.join(MODEL_DIR, "target_scaler.pkl")
model_path = os.path.join(MODEL_DIR, "lstm_stock_model_hybrid.keras")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# Add technical indicators
# =========================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[DEBUG] add_technical_indicators called with columns: {df.columns}")
    print(f"[DEBUG] Column types: {df.dtypes}")
    # Ensure lowercase columns
    df = df.copy()
    df.columns = df.columns.str.lower()
    # Guard against duplicate columns after lowercasing (e.g., Close + close)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for indicators: {col}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols)

    close = df['close']
    high = df['high']
    low = df['low']

    df['sma_10'] = close.rolling(window=10).mean()
    df['ema_9'] = close.ewm(span=9, adjust=False).mean()
    df['ema_21'] = close.ewm(span=21, adjust=False).mean()

    def compute_rsi(data, window=14):
        delta = data.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi)

    df['rsi_14'] = compute_rsi(close, 14)
    df['ma_20'] = close.rolling(window=20).mean()
    df['bb_upper'] = df['ma_20'] + 2 * close.rolling(window=20).std()
    df['bb_lower'] = df['ma_20'] - 2 * close.rolling(window=20).std()

    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['ema_26'] = close.ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Stochastic Oscillator
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    df['%k'] = ((close - low_14) / (high_14 - low_14)) * 100
    df['%d'] = df['%k'].rolling(window=3).mean()

    # ATR
    hl = high - low
    hc = abs(high - close.shift())
    lc = abs(low - close.shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()

    # Add news_sentiment with default value
    if 'news_sentiment' not in df.columns:
        df['news_sentiment'] = 0.0

    # Drop NaNs due to rolling indicators
    df = df.dropna()

    return df

# =========================
# Fetch last 60 minutes of stock data
# =========================
def fetch_last_60_minutes(symbol: str, use_daily_fallback: bool = True) -> pd.DataFrame:
    """
    Fetch last 60 minutes of stock data with multiple fallback APIs
    If use_daily_fallback is True, will fallback to daily data for consistency
    """
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_last_60_minutes.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            df = pd.DataFrame(cached_data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"[INFO] Loaded cached intraday data for {symbol}")
            return df
        except Exception as e:
            print(f"[WARNING] Failed to load cache for {symbol}: {str(e)}")

    # Try yfinance intraday first
    try:
        print(f"[INFO] Trying yfinance intraday for {symbol}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="21d", interval="5m")
        
        if not df.empty:
            df.reset_index(inplace=True)
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            df.rename(columns={'datetime': 'datetime'}, inplace=True)
            
            if 'datetime' not in df.columns and 'date' in df.columns:
                df.rename(columns={'date': 'datetime'}, inplace=True)
            
            df['volume'] = df['volume'].replace(0, df['volume'].mean())
            df['volume'] = df['volume'].fillna(df['volume'].mean())
            df = df.ffill().bfill()
            
            min_points = 60
            if len(df) >= min_points:
                print(f"[INFO] Successfully fetched {len(df)} intraday data points from yfinance")
                # Save to cache
                try:
                    df_to_cache = df.copy()
                    df_to_cache['datetime'] = df_to_cache['datetime'].astype(str)
                    df_to_cache.to_json(cache_file, orient='records', date_format='iso')
                    print(f"[INFO] Cached intraday data saved for {symbol}")
                except Exception as e:
                    print(f"[WARNING] Failed to save cache for {symbol}: {str(e)}")
                
                return df
            else:
                print(f"[WARNING] yfinance returned insufficient intraday data: {len(df)} points")
                # Add explicit warning for Axis Bank
                if symbol == "AXISBANK.NS":
                    print("[WARNING] Axis Bank intraday data insufficient, cache not saved.")
    except Exception as e:
        print(f"[WARNING] yfinance intraday failed for {symbol}: {str(e)}")

    # Fallback to Yahoo Finance API for intraday
    try:
        print(f"[INFO] Trying Yahoo Finance API for intraday {symbol}")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=21d&interval=5m"
        response = requests.get(url)
        response.raise_for_status()
        data_json = response.json()

        timestamps = data_json['chart']['result'][0]['timestamp']
        indicators = data_json['chart']['result'][0]['indicators']['quote'][0]

        df = pd.DataFrame({
            'datetime': pd.to_datetime(timestamps, unit='s'),
            'open': indicators['open'],
            'high': indicators['high'],
            'low': indicators['low'],
            'close': indicators['close'],
            'volume': indicators['volume']
        })

        df = df.dropna()
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        df['volume'] = df['volume'].replace(0, df['volume'].mean())
        df['volume'] = df['volume'].fillna(df['volume'].mean())
        df = df.ffill().bfill()

        min_points = 60
        if len(df) >= min_points:
            print(f"[INFO] Successfully fetched {len(df)} intraday data points from Yahoo Finance API")
            # Save to cache
            try:
                df_to_cache = df.copy()
                df_to_cache['datetime'] = df_to_cache['datetime'].astype(str)
                df_to_cache.to_json(cache_file, orient='records', date_format='iso')
                print(f"[INFO] Cached intraday data saved for {symbol}")
            except Exception as e:
                print(f"[WARNING] Failed to save cache for {symbol}: {str(e)}")
            
            time.sleep(1)  # Throttle requests
            return df
        else:
            print(f"[WARNING] Yahoo Finance API returned insufficient intraday data: {len(df)} points")

    except Exception as e:
        print(f"[WARNING] Yahoo Finance API intraday failed for {symbol}: {str(e)}")

    # Fallback to daily data if enabled and intraday fails
    if use_daily_fallback:
        print(f"[INFO] Falling back to daily data for {symbol} for consistency")
        return fetch_historical_data(symbol, period="3mo")  # Get 3 months of daily data

    print(f"[ERROR] All intraday data sources failed for {symbol}")
    return None

def fetch_historical_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical stock data for training using yfinance
    """
    print(f"[INFO] Fetching historical data for {symbol} with period {period}")
    
    # Try yfinance first
    try:
        print(f"[INFO] Trying yfinance for historical data of {symbol}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if not df.empty:
            df.reset_index(inplace=True)
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            
            # Rename date column to datetime for consistency
            if 'date' in df.columns:
                df.rename(columns={'date': 'datetime'}, inplace=True)
            
            df['volume'] = df['volume'].replace(0, df['volume'].mean())
            df['volume'] = df['volume'].fillna(df['volume'].mean())
            df = df.ffill().bfill()
            
            print(f"[INFO] Successfully fetched {len(df)} historical data points from yfinance")
            return df
    except Exception as e:
        print(f"[WARNING] yfinance failed for historical data of {symbol}: {str(e)}")

    print(f"[ERROR] All data sources failed for historical data of {symbol}")
    return None

def load_scalers_and_model():
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    model = load_model(model_path)
    return feature_scaler, target_scaler, model

def predict_stock_price(input_features):
    feature_scaler, target_scaler, model = load_scalers_and_model()
    # Scale input features
    input_scaled = feature_scaler.transform(input_features)
    # Reshape for LSTM input: (samples, timesteps, features)
    input_seq = np.expand_dims(input_scaled, axis=0)
    # Predict scaled output
    pred_scaled = model.predict(input_seq)
    # Check if prediction shape is correct for inverse transform
    if pred_scaled.ndim == 3:
        pred_scaled = pred_scaled.reshape(pred_scaled.shape[0], pred_scaled.shape[2])
    # Inverse transform to original scale
    pred_original = target_scaler.inverse_transform(pred_scaled)
    return pred_original[0][0]
