from django.db import models


class TickerMetadata(models.Model):
	symbol = models.CharField(max_length=20, unique=True)
	name = models.CharField(max_length=120)
	asset_type = models.CharField(max_length=20, default="stock")
	sector = models.CharField(max_length=80, blank=True, default="")
	exchange = models.CharField(max_length=20, blank=True, default="NSE")
	benchmark_symbol = models.CharField(max_length=20, blank=True, default="^NSEI")
	is_active = models.BooleanField(default=True)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	def __str__(self):
		return f"{self.symbol} ({self.name})"


class MarketData(models.Model):
	symbol = models.CharField(max_length=20, db_index=True)
	date = models.DateField(db_index=True)
	open = models.FloatField()
	high = models.FloatField()
	low = models.FloatField()
	close = models.FloatField()
	volume = models.BigIntegerField(default=0)

	log_return = models.FloatField(null=True, blank=True)
	return_1d = models.FloatField(null=True, blank=True)
	return_7d = models.FloatField(null=True, blank=True)
	return_14d = models.FloatField(null=True, blank=True)
	return_30d = models.FloatField(null=True, blank=True)
	volatility_14d = models.FloatField(null=True, blank=True)
	atr_14 = models.FloatField(null=True, blank=True)
	momentum_10d = models.FloatField(null=True, blank=True)
	volume_change_1d = models.FloatField(null=True, blank=True)
	relative_strength = models.FloatField(null=True, blank=True)

	source = models.CharField(max_length=30, default="yahoo_finance")
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		constraints = [
			models.UniqueConstraint(fields=["symbol", "date"], name="uniq_symbol_date_market_data")
		]
		ordering = ["symbol", "date"]


class SentimentData(models.Model):
	symbol = models.CharField(max_length=20, db_index=True)
	date = models.DateField(db_index=True)
	sentiment_mean = models.FloatField(default=0.0)
	sentiment_std = models.FloatField(default=0.0)
	news_count = models.IntegerField(default=0)
	positive_ratio = models.FloatField(default=0.0)
	source = models.CharField(max_length=30, default="newsapi_gnews")
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		constraints = [
			models.UniqueConstraint(fields=["symbol", "date"], name="uniq_symbol_date_sentiment_data")
		]
		ordering = ["symbol", "date"]


class PredictionRecord(models.Model):
	symbol = models.CharField(max_length=20, db_index=True)
	date = models.DateField(db_index=True)
	horizon_days = models.PositiveSmallIntegerField(default=1)
	model_name = models.CharField(max_length=50)
	predicted_close = models.FloatField()
	actual_close = models.FloatField(null=True, blank=True)
	rmse = models.FloatField(null=True, blank=True)
	mae = models.FloatField(null=True, blank=True)
	mape = models.FloatField(null=True, blank=True)
	directional_accuracy = models.FloatField(null=True, blank=True)
	metadata = models.JSONField(default=dict, blank=True)
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ["-created_at"]
