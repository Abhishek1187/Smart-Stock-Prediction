import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime, timedelta


class NSEDataFetcher:
    def __init__(self):
        self.nse_base_url = "https://www.nseindia.com/api"
        self.backup_base_url = "https://www.alphavantage.co/query"
        self.alpha_vantage_api_key = "OJGSGLAASRO31V80"  

    def fetch_yfinance_data(self, symbol, period="1y"):
        """
        Fetch stock data using yfinance as primary source.
        """
        try:
            print(f"[INFO] Trying yfinance for: {symbol} (period={period})")
            data = yf.download(symbol, period=period, interval="1d", auto_adjust=False)
            if data.empty:
                raise ValueError("No data returned from yfinance")
            data.reset_index(inplace=True)
            data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
            return data
        except Exception as e:
            print(f"[ERROR] yfinance fetch failed: {e}")
            return None

    def fetch_nse_data(self, symbol, years=1):
        """
        Fetch stock data from NSE India API.
        """
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        session = requests.Session()
        session.headers.update(headers)

        try:
            print(f"[INFO] Trying NSE API for: {symbol}")
            symbol_clean = symbol.replace(".NS", "")

            # Get cookie session initialized
            url = f"https://www.nseindia.com/get-quotes/equity?symbol={symbol_clean}"
            session.get(url, timeout=5)
            time.sleep(1.5) 

            to_date = datetime.today()
            from_date = to_date - timedelta(days=max(365, int(365 * years)))
            from_str = from_date.strftime('%d-%m-%Y')
            to_str = to_date.strftime('%d-%m-%Y')

            hist_url = (
                f"https://www.nseindia.com/api/historical/cm/equity?"
                f"symbol={symbol_clean}&series=[%22EQ%22]&from={from_str}&to={to_str}"
            )

            response = session.get(hist_url, timeout=10)
            if response.status_code == 429:
                print("[WARN] Rate limited by NSE. Retrying...")
                time.sleep(5)
                response = session.get(hist_url, timeout=10)

            response.raise_for_status()
            data = response.json()
            records = data.get("data", [])
            df = pd.DataFrame(records)
            if df.empty:
                raise ValueError("No data returned from NSE API")

            df['date'] = pd.to_datetime(df['CH_TIMESTAMP'])
            df.rename(columns={
                'CH_OPENING_PRICE': 'open',
                'CH_HIGH_PRICE': 'high',
                'CH_LOW_PRICE': 'low',
                'CH_CLOSING_PRICE': 'close',
                'CH_TOTTRDQTY': 'volume'
            }, inplace=True)
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

        except Exception as e:
            print(f"[ERROR] NSE API fetch failed: {e}")
            return None

    def fetch_alpha_vantage_data(self, symbol):
        # Removed Alpha Vantage API fetch as per user request
        print("[INFO] Alpha Vantage API fetch removed as per user request.")
        return None

    def fetch_data(self, symbol, period="1y"):
        """
        Tries yfinance → NSE API in order.
        Returns cleaned DataFrame or None if all fail.
        """
        print(f"\n[START] Fetching data for: {symbol} (period={period})")

        df = self.fetch_yfinance_data(symbol, period=period)
        if df is not None and not df.empty:
            print("[INFO] ✅ Data fetched from yfinance")
            return df

        period_year_map = {
            "1y": 1,
            "2y": 2,
            "5y": 5,
            "10y": 10,
            "max": 10,
        }
        years = period_year_map.get(str(period).lower(), 1)
        df = self.fetch_nse_data(symbol, years=years)
        if df is not None and not df.empty:
            print("[INFO] ✅ Data fetched from NSE API")
            return df

        print("[❌ ERROR] Data fetch failed from all sources.\n")
        return None
