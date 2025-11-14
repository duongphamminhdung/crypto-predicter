import hmac
import hashlib
import time
import requests
from dotenv import load_dotenv
import os

load_dotenv('config.api')

class Exchange:
    def __init__(self):
          # Try multiple possible variable names from config.api
        self.api_key    = os.getenv('MEXC_API_KEY') or os.getenv('api_key')
        self.api_secret = os.getenv('MEXC_API_SECRET') or os.getenv('secret_key')
        self.base_url   = "https://contract.mexc.com"                              # Updated for futures
        
        # If still no keys, try reading directly from config.api
        if not self.api_key or not self.api_secret:
            try:
                with open('config.api', 'r') as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.strip().split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key == 'api_key':
                                self.api_key = value
                            elif key == 'secret_key':
                                self.api_secret = value
            except FileNotFoundError:
                print("⚠️  Warning: config.api file not found")
        
        if not self.api_key or not self.api_secret:
            print("⚠️  Warning: API keys not configured. Trading will not work (OK for TEST mode).")

    def _get_timestamp(self):
        return int(time.time() * 1000)

    def _sign_request(self, params):
        if not self.api_secret:
            raise ValueError("API secret not configured. Cannot sign requests.")
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    def get_account_balance(self):
        """Get account balance for all assets"""
        # If no API keys configured, return None immediately
        if not self.api_key or not self.api_secret:
            return None
        
        endpoint = "/api/v3/account"
        timestamp = self._get_timestamp()
        
        params = {
            'timestamp': timestamp
        }
        
        try:
            params['signature'] = self._sign_request(params)
        except ValueError as e:
            print(f"Cannot sign request: {e}")
            return None
        
        headers = {
            'X-MEXC-APIKEY': self.api_key
        }
        
        try:
            response = requests.get(self.base_url + endpoint, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting account balance: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Error response: {e.response.text[:200]}")
            return None

    def get_usdt_balance(self):
        """Get available USDT balance"""
        # If no API keys, return mock balance for TEST mode
        if not self.api_key or not self.api_secret:
            print("Using mock balance (TEST mode)")
            return 1000.0  # Mock balance for testing
        
        account_info = self.get_account_balance()
        if account_info and 'balances' in account_info:
            for balance in account_info['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
        
        # If API call failed but keys exist, return mock balance
        print("API call failed, using mock balance")
        return 1000.0

    def place_order(self, symbol, order_type, side, quantity):
        endpoint = "/api/v3/order"
        timestamp = self._get_timestamp()
        
        params = {
            'symbol': symbol,
            'side': side, # 'BUY' or 'SELL'
            'type': order_type, # 'MARKET' or 'LIMIT'
            'quantity': quantity,
            'timestamp': timestamp
        }
        
        params['signature'] = self._sign_request(params)
        
        headers = {
            'X-MEXC-APIKEY': self.api_key
        }
        
        try:
            response = requests.post(self.base_url + endpoint, headers=headers, params=params)
            response.raise_for_status()
            print(f"Successfully placed {side} order for {quantity} {symbol}.")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error placing order: {e}")
            if e.response:
                print(f"Error response: {e.response.json()}")
            return None

if __name__ == '__main__':
    # Example usage (use with caution!)
    exchange = Exchange()
    # Ensure your config.api has the correct keys
    if exchange.api_key and exchange.api_secret:
         # Example: place a market buy order for a small amount of BTC
         # Note: MEXC uses symbol format without underscore: BTCUSDT
         # exchange.place_order('BTCUSDT', 'MARKET', 'BUY', 0.0001)
         pass
    else:
        print("API Key or Secret not found. Please check your config.api file.")
