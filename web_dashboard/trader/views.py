import json
import os
import requests
from django.shortcuts import render
from django.conf import settings

def dashboard(request):
    # Paths to JSON files (located in the root of the repo, parent of web_dashboard)
    base_dir = settings.BASE_DIR.parent
    trades_log_path = os.path.join(base_dir, 'trades_log.json')
    current_trades_path = os.path.join(base_dir, 'current_trades.json')
    stats_path = os.path.join(base_dir, 'trading_stats.json')

    trades_history = []
    active_trades = []
    daily_stats = {}

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

    context = {
        'trades_history': trades_history,
        'active_trades': active_trades,
        'daily_stats': daily_stats,
    }
    return render(request, 'trader/dashboard.html', context)
