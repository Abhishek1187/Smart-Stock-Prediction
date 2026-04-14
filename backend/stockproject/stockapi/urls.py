from django.urls import path
from django.http import JsonResponse
from .views import get_ohlcv
from predictor.views import get_stocks, model_comparison
from predictor.analytics_views import (
    advanced_analytics,
    market_overview,
    prediction_view,
    price_history,
    sentiment_view,
    technical_indicators,
)

urlpatterns = [
    path('ohlcv/<str:symbol>/', get_ohlcv, name='get_ohlcv'),
    path('health/', lambda request: JsonResponse({"status": "ok", "message": "API is running"}), name='api_health'),
    path('stocks/', get_stocks, name='get_stocks'),
    path('model_comparison/', model_comparison, name='model_comparison'),
    path('price-history/', price_history, name='price_history'),
    path('technical-indicators/', technical_indicators, name='technical_indicators'),
    path('sentiment/', sentiment_view, name='sentiment'),
    path('prediction/', prediction_view, name='prediction_history'),
    path('market-overview/', market_overview, name='market_overview'),
    path('advanced-analytics/', advanced_analytics, name='advanced_analytics'),
]
