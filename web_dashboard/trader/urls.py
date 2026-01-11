from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('toggle-bot/', views.toggle_bot_status, name='toggle_bot'),
    path('bot-status/', views.get_bot_status_view, name='bot_status'),
    path('update-trading-params/', views.update_trading_params, name='update_trading_params'),
    path('api/trading-data/', views.get_trading_data_api, name='trading_data_api'),
    path('sentiment_data/', views.get_news, name='sentiment_data'),
    path('update-active-symbol/', views.update_active_symbol, name='update_active_symbol'),
]

