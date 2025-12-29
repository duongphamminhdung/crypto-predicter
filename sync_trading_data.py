#!/usr/bin/env python3
"""
Script to sync trading data from local bot to Google Colab API
Run this from your local machine after the trading bot updates local JSON files
"""

import requests
import json
import os
import sys
from datetime import datetime

# Colab API URL (update this with your ngrok URL)
COLAB_API_URL = 'https://your-ngrok-url.ngrok.io'

def load_local_trading_data():
    """Load trading data from local JSON files"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    trading_data = {
        'trades_history': [],
        'active_trades': [],
        'daily_stats': {},
        'last_updated': datetime.now().isoformat()
    }

    # Load trades log
    trades_log_path = os.path.join(base_dir, 'trades_log.json')
    if os.path.exists(trades_log_path):
        try:
            with open(trades_log_path, 'r') as f:
                content = f.read().strip()
                if content and content != 'w':
                    raw_history = json.loads(content)
                    # Clean keys for consistency
                    for t in raw_history:
                        trading_data['trades_history'].append({
                            'time_stamp': t.get('time_stamp'),
                            'entry_price': t.get('entry price'),
                            'profit_loss_result': t.get('Profit/loss'),
                            'pl_percentage': t.get('PL percentage'),
                            'pl_in_dollar': t.get('PL in $'),
                            'side': 'UNK'  # Side not currently saved in log
                        })
        except Exception as e:
            print(f"Error reading trades log: {e}")

    # Load current trades
    current_trades_path = os.path.join(base_dir, 'current_trades.json')
    if os.path.exists(current_trades_path):
        try:
            with open(current_trades_path, 'r') as f:
                content = f.read().strip()
                if content and content != 'w':
                    data = json.loads(content)
                    # Combine active and simulated
                    active_trades = data.get('active_trades', [])
                    simulated_trades = data.get('simulated_trades', [])

                    # Add flags
                    for t in active_trades: t['is_simulated'] = False
                    for t in simulated_trades: t['is_simulated'] = True

                    trading_data['active_trades'] = active_trades + simulated_trades
        except Exception as e:
            print(f"Error reading current trades: {e}")

    # Load stats
    stats_path = os.path.join(base_dir, 'trading_stats.json')
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                content = f.read().strip()
                if content:
                    trading_data['daily_stats'] = json.loads(content)
        except Exception as e:
            print(f"Error reading stats: {e}")

    return trading_data

def sync_to_colab():
    """Send trading data to Colab API"""
    trading_data = load_local_trading_data()

    try:
        url = f"{COLAB_API_URL}/update_trading_data"
        response = requests.post(url, json=trading_data, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Trading data synced to Colab successfully")
                return True
            else:
                print(f"❌ Colab API returned error: {result}")
        else:
            print(f"❌ HTTP error: {response.status_code}")

    except Exception as e:
        print(f"❌ Failed to sync data: {e}")

    return False

if __name__ == '__main__':
    print("🔄 Syncing trading data to Colab...")
    success = sync_to_colab()
    if success:
        print("✅ Sync complete!")
    else:
        print("❌ Sync failed!")
        sys.exit(1)