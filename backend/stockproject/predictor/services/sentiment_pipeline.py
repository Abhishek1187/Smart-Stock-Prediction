from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List

from predictor.news_sentiment import analyze_sentiment, fetch_news_articles


def dedupe_articles(articles: List[dict]) -> List[dict]:
    seen = set()
    unique = []

    for article in articles:
        url = article.get("url") or article.get("link") or ""
        key = url.strip().lower() or (article.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(article)

    return unique


def aggregate_daily_sentiment(symbol: str, company_name: str | None = None) -> Dict[str, dict]:
    query = company_name or symbol
    raw_articles = fetch_news_articles(query, page_size=50)
    articles = dedupe_articles(raw_articles)

    grouped_scores = defaultdict(list)

    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        content = f"{title}. {description}".strip()
        score = analyze_sentiment(content)

        published = article.get("publishedAt") or article.get("published_at")
        if not published:
            continue
        try:
            day = datetime.fromisoformat(published.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            continue

        grouped_scores[day].append(score)

    daily = {}
    for day, values in grouped_scores.items():
        if not values:
            continue
        positive = [v for v in values if v > 0]
        daily[day] = {
            "sentiment_mean": float(mean(values)),
            "sentiment_std": float(pstdev(values)) if len(values) > 1 else 0.0,
            "news_count": len(values),
            "positive_ratio": float(len(positive) / len(values)),
        }

    return daily
