from django.urls import path
from .views import get_stocks, health_check, model_comparison, predict_price

urlpatterns = [
    path('health/', health_check, name='health_check'),
    path('stocks/', get_stocks, name='predictor_stocks'),
    path('model-comparison/', model_comparison, name='predictor_model_comparison'),
    path('<str:symbol>/', predict_price, name='predict_price'),  # Direct symbol access under /predict/
]
