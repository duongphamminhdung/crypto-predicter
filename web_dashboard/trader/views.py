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
from .twitter_service import TwitterService

logger = logging.getLogger(__name__)
ngrok_api_url = 'https://plumbaginaceous-mabelle-unelaborately.ngrok-free.dev/'  # Replace with your actual Colab ngrok URL

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
        required_params = ['confidence_trade', 'confidence_test', 'max_risk', 'max_leverage', 'max_time_red', 'opposite_signal', 'retrain_interval']
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
    
    if trading_data:
        trades_history = trading_data.get('trades_history', [])
        active_trades = trading_data.get('active_trades', [])
        daily_stats = trading_data.get('daily_stats', {})

        # Fetch current price and calculate P&L for active trades
        current_price = 0
        try:
            # MEXC Futures Ticker
            url = "https://contract.mexc.com/api/v1/contract/ticker"
            params = {'symbol': 'BTC_USDT'}
            r = requests.get(url, params=params, timeout=3)
            if r.status_code == 200:
                data = r.json()
                if data['success'] and data['data']:
                    current_price = float(data['data']['lastPrice'])
        except Exception as e:
            logger.error(f"Error fetching price: {e}")

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
                        
                        trade['unrealized_pnl_usdt'] = pnl
                        trade['unrealized_pnl_percentage'] = pnl_pct
                    else:
                        trade['unrealized_pnl_usdt'] = 0
                        trade['unrealized_pnl_percentage'] = 0
                except Exception as e:
                    logger.error(f"Error calculating P&L: {e}")

        context = {
            'trades_history': trades_history,
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

def dashboard(request):
    # Get ALL data from Colab instead of local JSON files
    trading_data = get_trading_data()
    
    if trading_data:
        trades_history = trading_data.get('trades_history', [])
        active_trades = trading_data.get('active_trades', [])
        daily_stats = trading_data.get('daily_stats', {})
    else:
        # Fallback to local files if Colab API fails
        trades_history = []
        active_trades = []
        daily_stats = {}
        
        # [Keep the existing local file reading code here as fallback]
        base_dir = settings.BASE_DIR.parent
        trades_log_path = os.path.join(base_dir, 'trades_log.json')
        current_trades_path = os.path.join(base_dir, 'current_trades.json')
        stats_path = os.path.join(base_dir, 'trading_stats.json')

        # Read Trades Log
        if os.path.exists(trades_log_path):
            try:
                with open(trades_log_path, 'r') as f:
                    content = f.read().strip()
                    if content and content != 'w':
                        raw_history = json.loads(content)
                        # Clean keys for Django template
                        for t in raw_history:
                            trades_history.append({
                                'time_stamp': t.get('time_stamp'),
                                'entry_price': t.get('entry price'),
                                'profit_loss_result': t.get('Profit/loss'),
                                'pl_percentage': t.get('PL percentage'),
                                'pl_in_dollar': t.get('PL in $'),
                                'side': 'UNK' # Side not currently saved in log
                            })
                        
                        # Reverse to show newest first
                        if isinstance(trades_history, list):
                            trades_history.reverse()
            except Exception as e:
                print(f"Error reading trades log: {e}")

        # Read Active Trades
        if os.path.exists(current_trades_path):
            try:
                with open(current_trades_path, 'r') as f:
                    content = f.read().strip()
                    if content and content != 'w':
                        data = json.loads(content)
                        # Combine active and simulated
                        active_trades = data.get('active_trades', [])
                        simulated_trades = data.get('simulated_trades', [])
                        
                        # Add a flag to distinguish them
                        for t in active_trades: t['is_simulated'] = False
                        for t in simulated_trades: t['is_simulated'] = True
                        
                        active_trades.extend(simulated_trades)
            except Exception as e:
                print(f"Error reading active trades: {e}")

        # Read Stats
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        daily_stats = json.loads(content)
            except Exception as e:
                print(f"Error reading stats: {e}")

    # Fetch Current Price for PnL Calculation
    current_price = 0
    try:
        # MEXC Futures Ticker
        url = "https://contract.mexc.com/api/v1/contract/ticker"
        params = {'symbol': 'BTC_USDT'}
        r = requests.get(url, params=params, timeout=3)
        if r.status_code == 200:
            data = r.json()
            if data['success'] and data['data']:
                current_price = float(data['data']['lastPrice'])
    except Exception as e:
        print(f"Error fetching price: {e}")

    # Calculate PnL if we have price and trades
    if current_price > 0:
        for trade in active_trades:
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
                    
                    trade['unrealized_pnl_usdt'] = pnl
                    trade['unrealized_pnl_percentage'] = pnl_pct
                else:
                    trade['unrealized_pnl_usdt'] = 0
                    trade['unrealized_pnl_percentage'] = 0
            except Exception as e:
                 print(f"Error calc pnl: {e}")

    # STEP 7: Fetch Twitter News
    twitter_service = TwitterService()
    twitter_news = twitter_service.fetch_crypto_news(max_results=8)
    
    # Add sentiment to each tweet
    for tweet in twitter_news:
        tweet['sentiment'] = twitter_service.get_sentiment(tweet['text'])
    
    context = {
        'trades_history': trades_history,
        'active_trades': active_trades,
        'daily_stats': daily_stats,
        'twitter_news': twitter_news,  # Add this line
    }
    return render(request, 'trader/code.html', context)
