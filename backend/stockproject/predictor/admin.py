from django.contrib import admin
from .models import MarketData, SentimentData, PredictionRecord, TickerMetadata


@admin.register(TickerMetadata)
class TickerMetadataAdmin(admin.ModelAdmin):
	list_display = ("symbol", "name", "asset_type", "sector", "benchmark_symbol", "is_active")
	search_fields = ("symbol", "name", "sector")
	list_filter = ("asset_type", "exchange", "is_active")


@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
	list_display = ("symbol", "date", "close", "volume", "source", "updated_at")
	search_fields = ("symbol",)
	list_filter = ("source", "symbol")
	date_hierarchy = "date"


@admin.register(SentimentData)
class SentimentDataAdmin(admin.ModelAdmin):
	list_display = ("symbol", "date", "sentiment_mean", "news_count", "positive_ratio")
	search_fields = ("symbol",)
	list_filter = ("source", "symbol")
	date_hierarchy = "date"


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
	list_display = ("symbol", "date", "model_name", "horizon_days", "predicted_close", "actual_close")
	search_fields = ("symbol", "model_name")
	list_filter = ("model_name", "horizon_days")
	date_hierarchy = "date"
