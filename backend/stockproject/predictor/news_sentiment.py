import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import concurrent.futures
import os
import json
import sqlite3
from pathlib import Path

import pandas as pd

# NewsAPI configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "144036cfedee4e678875c0e2ea5bd16c")
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# GNews API configuration
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "3f95c7a0d831aee6463a767c85c35739")
GNEWS_API_ENDPOINT = "https://gnews.io/api/v4/search"

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache" / "sentiment"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Stock/Index name mappings for better news search
STOCK_NEWS_QUERIES = {
    "RELIANCE.NS": ["Reliance Industries", "Mukesh Ambani Reliance"],
    "HDFCBANK.NS": ["HDFC Bank", "HDFC Bank India"],
    "AXISBANK.NS": ["Axis Bank India", "Axis Bank"],
    "SBIN.NS": ["State Bank of India", "SBI Bank"],
    "INFY.NS": ["Infosys", "Infosys Limited"],
    "TCS.NS": ["TCS", "Tata Consultancy Services"],
    "ICICIBANK.NS": ["ICICI Bank", "ICICI Bank India"],
    "KOTAKBANK.NS": ["Kotak Mahindra Bank", "Kotak Bank"],
    "ADANIPORTS.NS": ["Adani Ports", "Adani Ports SEZ"],
    "ADANIENT.NS": ["Adani Enterprises", "Gautam Adani"],
    "BAJFINANCE.NS": ["Bajaj Finance", "Bajaj Finserv"],
    "BHARTIARTL.NS": ["Bharti Airtel", "Airtel India"],
    "ONGC.NS": ["ONGC", "Oil Natural Gas Corporation India"],
    "^NSEI": ["Nifty 50", "NSE India Nifty"],
    "^NSEBANK": ["Bank Nifty", "Nifty Bank Index"],
    "^NSEMDCP50": ["Nifty Midcap", "NSE Midcap India"],
    "^CNXAUTO": ["Nifty Auto", "Indian Auto Stocks"],
}


def _to_iso_date(date_value):
    if date_value is None:
        return None
    if isinstance(date_value, datetime):
        return date_value.strftime('%Y-%m-%d')
    try:
        return pd.to_datetime(date_value).strftime('%Y-%m-%d')
    except Exception:
        return None


def _parse_published_date(article):
    published = article.get("publishedAt") or article.get("published_at")
    if not published:
        return None
    try:
        return datetime.fromisoformat(str(published).replace("Z", "+00:00")).date()
    except Exception:
        return None


def _normalize_article(article):
    source_name = ""
    source_obj = article.get("source")
    if isinstance(source_obj, dict):
        source_name = source_obj.get("name") or ""

    title = article.get("title") or ""
    description = article.get("description") or ""
    content = article.get("content") or ""
    url = article.get("url") or article.get("link") or ""

    published_date = _parse_published_date(article)
    published_at = None
    if published_date:
        published_at = datetime.combine(published_date, datetime.min.time()).isoformat()

    return {
        "title": title,
        "description": description,
        "content": content,
        "url": url,
        "publishedAt": published_at,
        "source": source_name,
    }


def _article_relevance_score(article, query):
    """Simple relevance score so weakly related market headlines carry less weight."""
    title = (article.get("title") or "").lower()
    description = (article.get("description") or "").lower()
    query_tokens = [t for t in str(query).lower().split() if len(t) > 2]
    if not query_tokens:
        return 1.0

    hit_count = sum(token in title or token in description for token in query_tokens)
    return min(1.0, 0.5 + 0.2 * hit_count)


def _cache_file_path(symbol_or_query, start_date, end_date):
    safe_key = str(symbol_or_query).replace("/", "_").replace(" ", "_").replace(".", "_")
    return CACHE_DIR / f"{safe_key}_{start_date}_{end_date}.json"


def _load_cached_daily_sentiment(symbol_or_query, start_date, end_date):
    cache_path = _cache_file_path(symbol_or_query, start_date, end_date)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        cached_series = payload.get("daily_sentiment") or {}
        if cached_series:
            non_zero = sum(1 for v in cached_series.values() if abs(float(v)) > 1e-12)
            # Zero-only cache often indicates temporary API/rate-limit failure.
            if non_zero == 0:
                print(f"[INFO] Ignoring zero-only sentiment cache for {symbol_or_query} [{start_date} -> {end_date}]")
                return None
            return cached_series
    except Exception as exc:
        print(f"[WARN] Failed to read sentiment cache {cache_path}: {exc}")
    return None


def _save_cached_daily_sentiment(symbol_or_query, start_date, end_date, daily_sentiment):
    cache_path = _cache_file_path(symbol_or_query, start_date, end_date)
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "daily_sentiment": daily_sentiment,
    }
    try:
        with cache_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
    except Exception as exc:
        print(f"[WARN] Failed to write sentiment cache {cache_path}: {exc}")


def _load_daily_sentiment_from_db(symbol, start_date, end_date):
    """Best effort historical sentiment load from local sqlite DB if available."""
    db_path = BASE_DIR.parent / "db.sqlite3"
    if not db_path.exists():
        return {}

    query = """
        SELECT date, sentiment_mean
        FROM predictor_sentimentdata
        WHERE symbol = ? AND date BETWEEN ? AND ?
        ORDER BY date ASC
    """

    try:
        with sqlite3.connect(str(db_path)) as conn:
            frame = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
    except Exception as exc:
        print(f"[WARN] Failed to load sentiment from DB for {symbol}: {exc}")
        return {}

    if frame.empty:
        return {}

    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return {row["date"]: float(row["sentiment_mean"]) for _, row in frame.iterrows()}

def fetch_news_articles_newsapi(query, from_date=None, to_date=None, language="en", page_size=10):
    """Fetch from NewsAPI with timeout and error handling"""
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')

    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": language,
        "pageSize": min(10, page_size),
        "from": from_date,
        "to": to_date,
        "sortBy": "relevancy",
    }

    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            print(f"[DEBUG] NewsAPI: {len(articles)} articles for '{query}'")
            return articles
        else:
            print(f"[WARN] NewsAPI returned {response.status_code}")
            return []
    except requests.exceptions.Timeout:
        print(f"[WARN] NewsAPI timeout for '{query}'")
        return []
    except Exception as e:
        print(f"[ERROR] NewsAPI: {e}")
        return []

def fetch_news_articles_gnews(query, from_date=None, to_date=None, language="en", max_results=10):
    """Fetch from GNews with timeout and error handling"""
    if not from_date:
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%dT00:00:00Z')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%dT23:59:59Z')

    params = {
        "q": query,
        "token": GNEWS_API_KEY,
        "lang": language,
        "max": min(10, max_results),
        "from": from_date,
        "to": to_date,
    }

    try:
        response = requests.get(GNEWS_API_ENDPOINT, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            print(f"[DEBUG] GNews: {len(articles)} articles for '{query}'")
            return articles
        else:
            print(f"[WARN] GNews returned {response.status_code}")
            return []
    except requests.exceptions.Timeout:
        print(f"[WARN] GNews timeout for '{query}'")
        return []
    except Exception as e:
        print(f"[ERROR] GNews: {e}")
        return []

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    if not text or len(text.strip()) < 10:
        return 0.0
    try:
        # Limit text length for performance
        text = text[:1000]
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"[ERROR] Sentiment analysis: {e}")
        return 0.0

def get_average_sentiment(articles):
    """Calculate average sentiment from articles"""
    if not articles:
        return 0.0
    
    sentiments = []
    for article in articles[:20]:  # Limit to 20 articles for performance
        # Try multiple fields for content
        title = article.get("title") or ""
        description = article.get("description") or ""
        content_body = article.get("content") or ""
        # Weighted text payload improves stability over raw content-only sentiment.
        content = f"{title}. {description}. {content_body}".strip()
        if content and len(content) > 20:
            sentiment = analyze_sentiment(content)
            sentiment *= _article_relevance_score(article, title)
            if sentiment != 0.0:  # Only count non-neutral
                sentiments.append(sentiment)
    
    if sentiments:
        avg = sum(sentiments) / len(sentiments)
        print(f"[DEBUG] Average sentiment from {len(sentiments)} articles: {avg:.3f}")
        return avg
    return 0.0

def fetch_news_articles(query, from_date=None, to_date=None, language="en", page_size=20):
    """
    Fetch news articles with improved query building and parallel fetching
    """
    # Get better search queries for known symbols
    search_queries = STOCK_NEWS_QUERIES.get(query, [query])
    
    # Clean any remaining special chars
    clean_queries = []
    for q in search_queries:
        clean_q = q.replace('.NS', '').replace('^', '').replace('_', ' ')
        if clean_q:
            clean_queries.append(clean_q)
    
    if not clean_queries:
        clean_queries = [query.replace('.NS', '').replace('^', '')]
    
    print(f"[DEBUG] Searching news for: {clean_queries}")
    
    all_articles = []
    
    # Fetch from both APIs in parallel for speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for q in clean_queries[:2]:  # Limit to 2 queries
            futures.append(executor.submit(fetch_news_articles_newsapi, q, from_date, to_date, language, 10))
            futures.append(executor.submit(fetch_news_articles_gnews, q, from_date, to_date, language, 10))
        
        for future in concurrent.futures.as_completed(futures, timeout=10):
            try:
                articles = future.result()
                if articles:
                    all_articles.extend(articles)
            except Exception as e:
                print(f"[WARN] Future failed: {e}")
    
    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        url = article.get("url") or article.get("link")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
        elif not url:
            unique_articles.append(article)
    
    print(f"[DEBUG] Total unique articles: {len(unique_articles)}")
    return [_normalize_article(article) for article in unique_articles[:page_size]]


def get_daily_sentiment_series(symbol_or_query, start_date, end_date, use_cache=True):
    """
    Build a date-indexed sentiment series for [start_date, end_date].
    - Uses cached result when possible
    - Uses local DB historical sentiment if available (best source for long windows)
    - Falls back to live API aggregation
    Returns: dict[YYYY-MM-DD] -> float sentiment_mean
    """
    start_iso = _to_iso_date(start_date)
    end_iso = _to_iso_date(end_date)
    if not start_iso or not end_iso:
        return {}

    if use_cache:
        cached = _load_cached_daily_sentiment(symbol_or_query, start_iso, end_iso)
        if cached is not None:
            return {k: float(v) for k, v in cached.items()}

    date_index = pd.date_range(start=start_iso, end=end_iso, freq="D")
    daily_sentiment = {dt.strftime("%Y-%m-%d"): 0.0 for dt in date_index}

    # If symbol is known, prefer DB-backed historical sentiment first.
    if symbol_or_query in STOCK_NEWS_QUERIES:
        db_series = _load_daily_sentiment_from_db(symbol_or_query, start_iso, end_iso)
        for day, value in db_series.items():
            if day in daily_sentiment:
                daily_sentiment[day] = float(value)

    # Fetch API sentiment in rolling windows for missing dates.
    # Keep windows reasonably small to avoid API timeout/rate-limit bursts.
    missing_days = [day for day, value in daily_sentiment.items() if value == 0.0]
    if missing_days:
        start_dt = pd.to_datetime(start_iso)
        end_dt = pd.to_datetime(end_iso)
        window_days = 14
        cur = start_dt

        while cur <= end_dt:
            win_start = cur.strftime("%Y-%m-%d")
            win_end_dt = min(cur + timedelta(days=window_days - 1), end_dt)
            win_end = win_end_dt.strftime("%Y-%m-%d")

            articles = fetch_news_articles(
                symbol_or_query,
                from_date=win_start,
                to_date=win_end,
                language="en",
                page_size=40,
            )

            grouped = {}
            for article in articles:
                published_date = _parse_published_date(article)
                if not published_date:
                    continue
                day = published_date.strftime("%Y-%m-%d")
                if day not in daily_sentiment:
                    continue
                title = article.get("title") or ""
                description = article.get("description") or ""
                content_body = article.get("content") or ""
                text = f"{title}. {description}. {content_body}".strip()
                score = analyze_sentiment(text)
                score *= _article_relevance_score(article, symbol_or_query)
                grouped.setdefault(day, []).append(score)

            for day, scores in grouped.items():
                if scores:
                    daily_sentiment[day] = float(sum(scores) / len(scores))

            cur = win_end_dt + timedelta(days=1)

    if use_cache:
        _save_cached_daily_sentiment(symbol_or_query, start_iso, end_iso, daily_sentiment)

    return daily_sentiment
