import os
import traceback
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .news_sentiment import fetch_news_articles, get_average_sentiment
from .symbol_mapping import get_company_name_from_symbol

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_prediction_system = {
    "initialized": False,
    "asset_predictor": None,
    "base_model": None,
    "error": None,
}


def get_prediction_system(force_reload=False):
    if _prediction_system["initialized"] and not force_reload:
        return _prediction_system

    try:
        from .asset_aware_predictor import AssetAwarePredictor
        from .dynamic_model import load_base_model, initialize_scalers_for_popular_stocks

        print("[INFO] Initializing asset-aware prediction system...")
        _prediction_system["asset_predictor"] = AssetAwarePredictor()
        print("[INFO] Asset-aware prediction system ready.")

        print("[INFO] Loading base LSTM model for dynamic scaling...")
        _prediction_system["base_model"] = load_base_model()
        print("[INFO] Base LSTM model loaded successfully.")

        print("[INFO] Initializing scalers for popular stocks...")
        initialize_scalers_for_popular_stocks()
        print("[INFO] Dynamic scaling system ready.")

        _prediction_system["error"] = None
    except Exception as exc:
        print("[ERROR] Failed to initialize prediction systems:")
        traceback.print_exc()
        _prediction_system["asset_predictor"] = None
        _prediction_system["base_model"] = None
        _prediction_system["error"] = str(exc)
    finally:
        _prediction_system["initialized"] = True

    return _prediction_system

# Define feature set used in training
FEATURES = [
    'open', 'high', 'low', 'volume',
    'sma_10', 'ema_9', 'ema_21', 'rsi_14', 'ma_20',
    'bb_upper', 'bb_lower', 'ema_12', 'ema_26',
    'macd', 'signal_line', '%k', '%d', 'atr_14',
    'news_sentiment'
]


def _get_news_context(symbol: str):
    try:
        if symbol.startswith("^"):
            index_map = {
                "^NSEI": "NIFTY 50",
                "^NSEBANK": "BANK NIFTY",
                "^NSEMDCP50": "NIFTY MIDCAP 50",
                "^NSEFIN": "NIFTY FINANCE",
                "^CNXAUTO": "NIFTY AUTO",
            }
            news_query = index_map.get(symbol, symbol.lstrip("^"))
        else:
            news_query = get_company_name_from_symbol(symbol)

        articles = fetch_news_articles(news_query)
        avg_sentiment = get_average_sentiment(articles)
        article_count = len(articles) if articles else 0
        return avg_sentiment, article_count
    except Exception as news_exc:
        print(f"[ERROR] News sentiment fetch failed: {news_exc}")
        return 0.0, 0

@api_view(["GET"])
def health_check(request):
    try:
        system = get_prediction_system()
        asset_predictor = system["asset_predictor"]
        model = system["base_model"]

        status_info = {
            "asset_aware_predictor": asset_predictor is not None,
            "base_model": model is not None,
            "available_models": {},
            "initialization_error": system.get("error"),
        }
        
        if asset_predictor:
            status_info["available_models"] = asset_predictor.get_available_models()
        
        if asset_predictor is None and model is None:
            return Response({
                "status": "error", 
                "message": "No prediction systems loaded properly.",
                "details": status_info
            }, status=500)
        
        return Response({
            "status": "ok", 
            "message": "Prediction systems ready.",
            "details": status_info
        })
    except Exception as e:
        print("[EXCEPTION] Health check failed:")
        traceback.print_exc()
        return Response({"status": "error", "message": f"Internal server error: {str(e)}"}, status=500)

@api_view(["GET"])
def get_stocks(request):
    try:
        # Return all stocks and indices for which we have trained asset-aware models
        stocks = [
            # Stocks
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "type": "stock"},
            {"symbol": "AXISBANK.NS", "name": "Axis Bank", "type": "stock"},
            {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "type": "stock"},
            {"symbol": "ONGC.NS", "name": "Oil and Natural Gas Corporation", "type": "stock"},
            {"symbol": "SBIN.NS", "name": "State Bank of India", "type": "stock"},
            {"symbol": "INFY.NS", "name": "Infosys", "type": "stock"},
            {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "type": "stock"},
            {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "type": "stock"},
            {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "type": "stock"},
            {"symbol": "ADANIPORTS.NS", "name": "Adani Ports and SEZ", "type": "stock"},
            {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "type": "stock"},
            {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "type": "stock"},
            {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "type": "stock"},
            
            # Indices - All trained indices with asset-aware models
            {"symbol": "^NSEI", "name": "NIFTY 50", "type": "index"},
            {"symbol": "^NSEBANK", "name": "BANK NIFTY", "type": "index"},
            {"symbol": "^NSEMDCP50", "name": "NIFTY MIDCAP 50", "type": "index"},
            {"symbol": "^CNXAUTO", "name": "NIFTY AUTO", "type": "index"},
        ]
        return Response(stocks)
    except Exception as e:
        print("[EXCEPTION] Failed to get stocks:")
        traceback.print_exc()
        return Response({"error": f"Internal server error: {str(e)}"}, status=500)

@api_view(["GET" , "POST"])
def predict_price(request, symbol):
    try:
        system = get_prediction_system()
        asset_predictor = system["asset_predictor"]

        print(f"[DEBUG] Incoming request for symbol: {symbol}")

        # Handle index symbols differently - they don't need .NS suffix
        if not symbol.startswith("^") and not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            symbol += ".NS"

        print(f"[DEBUG] Processing prediction for symbol: {symbol}")

        requested_model = (request.query_params.get("model") or "transformer").strip().lower()
        if requested_model not in {"lstm", "transformer"}:
            requested_model = "transformer"

        # Try asset-aware prediction first
        prediction_result = {"error": "Prediction system unavailable"}
        if asset_predictor is not None:
            print(f"[INFO] Using asset-aware prediction for {symbol} with model={requested_model}")
            
            avg_sentiment, article_count = _get_news_context(symbol)

            # Use requested model with daily data for consistency
            prediction_result = asset_predictor.predict_price(
                symbol,
                requested_model,
                use_daily_data=True,
                news_sentiment_value=avg_sentiment,
            )
            
            if "error" not in prediction_result:
                # Add news sentiment to response
                prediction_result["average_sentiment"] = avg_sentiment
                prediction_result["article_count"] = article_count
                prediction_result["model_name"] = requested_model.upper()
                prediction_result["served_at"] = timezone.now().isoformat()
                
                print(f"[SUCCESS] Asset-aware prediction completed for {symbol}")
                print(f"[INFO] Predicted: {prediction_result.get('predicted_close')}, Actual: {prediction_result.get('actual_close')}")
                
                # Ensure proper JSON formatting by returning Response with dict, not string
                return Response(dict(prediction_result))
            else:
                print(f"[WARNING] Asset-aware prediction failed: {prediction_result.get('error')}")

        # No fallback available - asset-aware prediction is the only option
        return Response({
            "error": "Asset-aware prediction failed and no fallback models available.",
            "details": prediction_result.get('error', 'Unknown error')
        }, status=500)

    except Exception as e:
        print("[EXCEPTION] Prediction failed:")
        traceback.print_exc()
        return Response({"error": f"Internal server error: {str(e)}"}, status=500)

# Note: Old global models are no longer loaded. 
# All predictions now use the asset-aware prediction system.
print("[INFO] Using asset-aware prediction system exclusively.")

@api_view(["GET"])
def model_comparison(request):
    try:
        system = get_prediction_system()
        asset_predictor = system["asset_predictor"]

        symbol = request.query_params.get("symbol", "")
        if not symbol:
            return Response({"error": "Symbol query parameter is required."}, status=400)

        # Normalize symbol
        if not symbol.startswith("^") and not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            symbol += ".NS"

        print(f"[INFO] Model comparison requested for {symbol}")

        # Try asset-aware comparison first
        comparison_result = {"LSTM": {}, "Transformer": {}}
        if asset_predictor is not None:
            print(f"[INFO] Using asset-aware model comparison for {symbol}")
            
            avg_sentiment, article_count = _get_news_context(symbol)
            comparison_result = asset_predictor.compare_models(symbol, news_sentiment_value=avg_sentiment)
            comparison_result["sentiment_context"] = {
                "average_sentiment": avg_sentiment,
                "article_count": article_count,
            }
            
            if "error" not in comparison_result.get("LSTM", {}) and "error" not in comparison_result.get("Transformer", {}):
                # Use the complete asset-aware comparison result with all time series data and metrics
                print(f"[SUCCESS] Asset-aware model comparison completed for {symbol}")
                return Response(comparison_result)
            else:
                print(f"[WARNING] Asset-aware model comparison failed, falling back to original models")

        # No fallback available - asset-aware comparison is the only option
        return Response({
            "error": "Asset-aware model comparison failed and no fallback models available.",
            "details": {
                "LSTM": comparison_result.get("LSTM", {}),
                "Transformer": comparison_result.get("Transformer", {})
            }
        }, status=500)

    except Exception as e:
        print("[EXCEPTION] Model comparison failed:")
        traceback.print_exc()
        return Response({"error": f"Internal server error: {str(e)}"}, status=500)
