import json
import os
import requests
import sys
from django.shortcuts import render
from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import logging
import torch
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)
ngrok_api_url = 'https://plumbaginaceous-mabelle-unelaborately.ngrok-free.dev/'  # Replace with your actual Colab ngrok URL
SYMBOLCOIN = "BTC_USDT"

@csrf_exempt
@require_POST
def toggle_bot_status(request):
    """Toggle the bot's running state on Colab"""
    try:
        # Get the current action from POST data
        action = request.POST.get('action', 'toggle')

        # Call Colab API to toggle bot status
        # IMPORTANT: Replace with your actual Colab ngrok URL!
        colab_api_url = ngrok_api_url + '/toggle_bot'

        response = requests.post(colab_api_url, json={'action': action}, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return JsonResponse({
                    'success': True,
                    'status': data.get('status'),
                    'message': data.get('message', 'Bot status updated')
                })

        return JsonResponse({
            'success': False,
            'message': f'Failed to toggle bot: {response.status_code}'
        })

    except Exception as e:
        logger.error(f"Error toggling bot status: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })

def get_bot_status():
    """Get current bot status from Colab"""
    try:
        # IMPORTANT: Replace with your actual Colab ngrok URL!
        colab_api_url = ngrok_api_url + '/bot_status'

        response = requests.get(colab_api_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data.get('status', 'unknown')

        return 'unknown'
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        return 'unknown'

def get_bot_status_view(request):
    """API endpoint to get bot status"""
    status = get_bot_status()
    return JsonResponse({
        'success': True,
        'status': status
    })

@csrf_exempt
@require_POST
def update_trading_params(request):
    """Update trading parameters in Colab predict_live.py"""
    try:
        import json

        # Parse the JSON data from request
        data = json.loads(request.body)
        settings = data.get('settings', data)  # Handle both nested and direct formats

        # Validate required parameters
        required_params = ['stop_loss', 'take_profit', 'confidence_trade', 'confidence_test', 'max_risk', 'max_leverage', 'max_time_red', 'opposite_signal', 'retrain_interval']
        for param in required_params:
            if param not in settings:
                return JsonResponse({
                    'success': False,
                    'message': f'Missing required parameter: {param}'
                })

        # Call Colab API to update parameters
        colab_api_url = ngrok_api_url + '/update_trading_params'

        response = requests.post(colab_api_url, json=settings, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return JsonResponse({
                    'success': True,
                    'message': 'Trading parameters updated successfully in Colab'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'message': result.get('message', 'Failed to update parameters in Colab')
                })
        else:
            return JsonResponse({
                'success': False,
                'message': f'Colab API error: {response.status_code}'
            })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'message': 'Invalid JSON data'
        })
    except Exception as e:
        logger.error(f"Error updating trading params: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })

def get_trading_data_api(request):
    """API endpoint to get real-time trading data for frontend updates"""
    trading_data = get_trading_data()
    global SYMBOLCOIN
    
    if trading_data:
        trades_history = trading_data.get('trades_history', [])
        active_trades = trading_data.get('active_trades', [])
        daily_stats = trading_data.get('daily_stats', {})

        # Fetch current price and calculate P&L for active trades
        current_price = 0
        try:
            # MEXC Futures Ticker
            url = "https://contract.mexc.com/api/v1/contract/ticker"
            params = {'symbol': SYMBOLCOIN}
            r = requests.get(url, params=params, timeout=3)
            print(r.json())
            if r.status_code == 200:
                data = r.json()
                if data['success'] and data['data']:
                    current_price = float(data['data']['lastPrice'])
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
        print(f"Current Price: {current_price}")
        # Calculate P&L for active trades
        if current_price > 0:
            for trade in trading_data.get('active_trades', []):
                try:
                    entry = float(trade.get('entry_price', 0))
                    qty = float(trade.get('quantity', 0))
                    side = trade.get('side', 'BUY')
                    trade['current_price'] = current_price
                    
                    if entry > 0:
                        if side == 'BUY':
                            pnl = (current_price - entry) * qty
                            pnl_pct = ((current_price - entry) / entry) * 100
                        else: # SELL
                            pnl = (entry - current_price) * qty
                            pnl_pct = ((entry - current_price) / entry) * 100
                        
                        trade['symbol'] = SYMBOLCOIN
                        trade['unrealized_pnl_usdt'] = pnl
                        trade['unrealized_pnl_percentage'] = pnl_pct
                    else:
                        trade['symbol'] = SYMBOLCOIN
                        trade['unrealized_pnl_usdt'] = 0
                        trade['unrealized_pnl_percentage'] = 0
                except Exception as e:
                    logger.error(f"Error calculating P&L: {e}")

        context = {
            'trades_history': trades_history[0:17],
            'active_trades': active_trades,
            'daily_stats': daily_stats,
        }
        
        return JsonResponse({
            'success': True,
            'trading_data': context
        })
    else:
        return JsonResponse({
            'success': False,
            'message': 'Failed to fetch trading data'
        })

def get_trading_data():
    # Get ALL trading data from Colab instead of local files
    try:
        # IMPORTANT: Replace with your actual Colab ngrok URL!
        colab_api_url = ngrok_api_url + '/trading_data'
        
        response = requests.get(colab_api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data['trading_data']
        
        logger.error(f"Colab trading data API call failed")
        return None
    except Exception as e:
        logger.error(f"Error calling Colab trading data API: {e}")
        return None

@csrf_exempt
@require_POST
def update_active_symbol(request):
    target_url = ngrok_api_url + '/update-active-symbol'

    try:
        data = json.loads(request.body)
        new_symbol = data.get('symbol')
        
        response = requests.post(target_url, json=new_symbol, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return JsonResponse({
                    'success': True,
                    'message': 'Symbol parameters updated successfully in Colab'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'message': result.get('message', 'Failed to update parameters in Colab')
                })
        else:
            return JsonResponse({
                'success': False,
                'message': f'Colab API error: {response.status_code}'
            })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def get_news(request):
    # 1. Check the URL being used
    # Ensure 'ngrok_api_url' is defined globally or imported!
    target_url = ngrok_api_url + '/sentiment_data'
    
    try:
        response = requests.get(target_url, timeout=30)
        
        # 2. Check if it's the Ngrok Warning Page
        if "ngrok-skip-browser-warning" in response.text:
            print("❌ Error: Ngrok 'Browser Warning' page detected!")
            print("   Fix: Add headers={'ngrok-skip-browser-warning': 'true'} to requests.get()")
        
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get('success'):
                    return JsonResponse({'success': True, 'sentiment_data': data.get('sentiment_data', {})})
                else:
                    print(f"⚠️ API returned 200 but success=False: {data}")
            except Exception as e:
                print(f"❌ JSON Decode Error: {e}")
                print(f"   Raw Content (First 100 chars): {response.text[:100]}")
        else:
            print(f"❌ Failed Response Content: {response.text[:200]}")

        # If we got here, something failed
        return JsonResponse({'success': False, 'error': f'Failed with status {response.status_code}'}, status=500)

    except Exception as e:
        print(f"🔥 CRASH: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
def get_news_data():
    """Fetch sentiment news data from Colab"""
    try:
        # IMPORTANT: Replace with your actual Colab ngrok URL!
        colab_api_url = ngrok_api_url + '/sentiment_data'

        response = requests.get(colab_api_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data.get('sentiment_data', {})
        
        logger.error(f"Colab sentiment data API call failed")
        return None
    except Exception as e:
        logger.error(f"Error calling Colab sentiment data API: {e}")
        return None

def dashboard(request):
    # Get ALL data from Colab instead of local JSON files
    trading_data = get_trading_data()
    news = get_news_data()

    context = {
        'trades_history': trading_data.get('trades_history', [])[0:17],
        'active_trades': trading_data.get('active_trades', []),
        'daily_stats': trading_data.get('daily_stats', {}),
        'sentiment_data': news
    }

    return render(request, 'trader/code.html', context)
