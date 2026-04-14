from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="MarketData",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("symbol", models.CharField(db_index=True, max_length=20)),
                ("date", models.DateField(db_index=True)),
                ("open", models.FloatField()),
                ("high", models.FloatField()),
                ("low", models.FloatField()),
                ("close", models.FloatField()),
                ("volume", models.BigIntegerField(default=0)),
                ("log_return", models.FloatField(blank=True, null=True)),
                ("return_1d", models.FloatField(blank=True, null=True)),
                ("return_7d", models.FloatField(blank=True, null=True)),
                ("return_14d", models.FloatField(blank=True, null=True)),
                ("return_30d", models.FloatField(blank=True, null=True)),
                ("volatility_14d", models.FloatField(blank=True, null=True)),
                ("atr_14", models.FloatField(blank=True, null=True)),
                ("momentum_10d", models.FloatField(blank=True, null=True)),
                ("volume_change_1d", models.FloatField(blank=True, null=True)),
                ("relative_strength", models.FloatField(blank=True, null=True)),
                ("source", models.CharField(default="yahoo_finance", max_length=30)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["symbol", "date"],
                "constraints": [
                    models.UniqueConstraint(fields=("symbol", "date"), name="uniq_symbol_date_market_data")
                ],
            },
        ),
        migrations.CreateModel(
            name="PredictionRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("symbol", models.CharField(db_index=True, max_length=20)),
                ("date", models.DateField(db_index=True)),
                ("horizon_days", models.PositiveSmallIntegerField(default=1)),
                ("model_name", models.CharField(max_length=50)),
                ("predicted_close", models.FloatField()),
                ("actual_close", models.FloatField(blank=True, null=True)),
                ("rmse", models.FloatField(blank=True, null=True)),
                ("mae", models.FloatField(blank=True, null=True)),
                ("mape", models.FloatField(blank=True, null=True)),
                ("directional_accuracy", models.FloatField(blank=True, null=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="SentimentData",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("symbol", models.CharField(db_index=True, max_length=20)),
                ("date", models.DateField(db_index=True)),
                ("sentiment_mean", models.FloatField(default=0.0)),
                ("sentiment_std", models.FloatField(default=0.0)),
                ("news_count", models.IntegerField(default=0)),
                ("positive_ratio", models.FloatField(default=0.0)),
                ("source", models.CharField(default="newsapi_gnews", max_length=30)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["symbol", "date"],
                "constraints": [
                    models.UniqueConstraint(fields=("symbol", "date"), name="uniq_symbol_date_sentiment_data")
                ],
            },
        ),
        migrations.CreateModel(
            name="TickerMetadata",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("symbol", models.CharField(max_length=20, unique=True)),
                ("name", models.CharField(max_length=120)),
                ("asset_type", models.CharField(default="stock", max_length=20)),
                ("sector", models.CharField(blank=True, default="", max_length=80)),
                ("exchange", models.CharField(blank=True, default="NSE", max_length=20)),
                ("benchmark_symbol", models.CharField(blank=True, default="^NSEI", max_length=20)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
