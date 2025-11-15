import hmac
import hashlib
import time
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv('config.api')

class Exchange:
    def __init__(self):
        # Try multiple possible variable names from config.api
        self.api_key    = os.getenv('MEXC_API_KEY') or os.getenv('api_key')
        self.api_secret = os.getenv('MEXC_API_SECRET') or os.getenv('secret_key')
        self.base_url   = "https://contract.mexc.com"  # MEXC Futures API
        
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
        """Get timestamp in milliseconds for MEXC Futures API"""
        return int(time.time() * 1000)

    def _sign_request(self, method, timestamp, params=None, json_body=None):
        """
        Sign request according to MEXC Futures API v1
        
        Signature format: accessKey + timestamp + parameterString
        - For GET/DELETE: parameterString = sorted params joined with &
        - For POST: parameterString = JSON string (no sorting)
        
        Args:
            method: HTTP method ('GET', 'POST', 'DELETE')
            timestamp: Request timestamp in milliseconds
            params: Query parameters (for GET/DELETE)
            json_body: JSON body string (for POST)
        """
        if not self.api_secret or not self.api_key:
            raise ValueError("API secret or key not configured. Cannot sign requests.")
        
        # Build parameter string based on method
        if method.upper() in ['GET', 'DELETE']:
            # For GET/DELETE: sort parameters by key, join with &
            if params:
                # Exclude 'signature' from signing params
                signing_params = {k: v for k, v in params.items() if k != 'signature'}
                sorted_params = sorted(signing_params.items())
                parameter_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
            else:
                parameter_string = ''
        elif method.upper() == 'POST':
            # For POST: use JSON string directly (no sorting)
            if json_body:
                parameter_string = json_body
            else:
                parameter_string = ''
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        # Build signature string: accessKey + timestamp + parameterString
        signature_string = f"{self.api_key}{timestamp}{parameter_string}"
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature

    def get_account_balance(self):
        """Get account balance for all assets using MEXC Futures API v1"""
        # If no API keys configured, return None immediately
        if not self.api_key or not self.api_secret:
            return None
        
        endpoint = "/api/v1/private/account/assets"
        timestamp = self._get_timestamp()
        
        params = {}
        
        try:
            signature = self._sign_request('GET', timestamp, params=params)
        except ValueError as e:
            print(f"Cannot sign request: {e}")
            return None
        
        headers = {
            'ApiKey': self.api_key,
            'Request-Time': str(timestamp),
            'Content-Type': 'application/json',
            'Signature': signature
        }
        
        try:
            response = requests.get(self.base_url + endpoint, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # MEXC Futures API v1 response format: {"code": 0, "data": {...}, "msg": "success"}
            if result.get('code') == 0:
                return result.get('data', {})
            else:
                error_msg = result.get('msg') or result.get('message') or 'Unknown error'
                print(f"API Error getting account balance: {error_msg} (code: {result.get('code')})")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error getting account balance: {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    error_data = e.response.json()
                    print(f"Full API response: {error_data}")
                except:
                    print(f"Error response: {e.response.text[:500]}")
            return None

    def get_usdt_balance(self):
        """Get available USDT balance"""
        # If no API keys, return mock balance for TEST mode
        if not self.api_key or not self.api_secret:
            print("Using mock balance (TEST mode)")
            return 1000.0  # Mock balance for testing
        
        account_info = self.get_account_balance()
        if account_info:
            # MEXC Futures API v1 returns balance in different structures
            # Try multiple possible response structures
            if 'assets' in account_info:
                for asset in account_info['assets']:
                    if asset.get('asset') == 'USDT' or asset.get('currency') == 'USDT':
                        return float(asset.get('availableBalance', asset.get('available', 0)))
            elif 'balances' in account_info:
                for balance in account_info['balances']:
                    if balance.get('asset') == 'USDT' or balance.get('currency') == 'USDT':
                        return float(balance.get('free', balance.get('available', 0)))
            elif 'USDT' in account_info:
                return float(account_info['USDT'].get('available', account_info['USDT'].get('free', 0)))
        
        # If API call failed but keys exist, return mock balance
        print("API call failed, using mock balance")
        return 1000.0

    def place_order(self, symbol, order_type, side, quantity, price=None):
        """
        Place order using MEXC Futures API v1
        
        Args:
            symbol: Contract symbol, e.g., 'XAUT_USDT' (Gold/USDT futures)
            order_type: 'MARKET' or 'LIMIT'
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
        """
        endpoint = "/api/v1/private/order/submit"
        timestamp = self._get_timestamp()
        
        # Remove underscore from symbol for order placement (XAUT_USDT -> XAUTUSDT)
        order_symbol = symbol.replace('_', '')
        
        # Build order parameters
        order_params = {
            'symbol': order_symbol,
            'side': side.upper(),  # BUY or SELL
            'type': order_type.upper(),  # MARKET or LIMIT
            'quantity': str(quantity)
        }
        
        # Add price for LIMIT orders
        if order_type.upper() == 'LIMIT' and price:
            order_params['price'] = str(price)
        
        # Convert to JSON string for signing
        json_body_string = json.dumps(order_params, separators=(',', ':'))
        
        try:
            signature = self._sign_request('POST', timestamp, json_body=json_body_string)
        except ValueError as e:
            print(f"Cannot sign request: {e}")
            return None
        
        headers = {
            'ApiKey': self.api_key,
            'Request-Time': str(timestamp),
            'Content-Type': 'application/json',
            'Signature': signature
        }
        
        try:
            response = requests.post(
                self.base_url + endpoint,
                headers=headers,
                data=json_body_string,  # Send exact JSON string used for signing
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # MEXC Futures API v1 response format: {"code": 0, "data": {...}, "msg": "success"}
            if result.get('code') == 0:
                print(f"Successfully placed {side} order for {quantity} {symbol}.")
                return result.get('data', {})
            else:
                error_msg = result.get('msg') or result.get('message') or 'Unknown error'
                print(f"Error placing order: {error_msg} (code: {result.get('code')})")
                print(f"Full API response: {result}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error placing order: {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    error_data = e.response.json()
                    print(f"Full API response: {error_data}")
                except:
                    print(f"Error response: {e.response.text[:500]}")
            return None

if __name__ == '__main__':
    # Example usage (use with caution!)
    exchange = Exchange()
    # Ensure your config.api has the correct keys
    if exchange.api_key and exchange.api_secret:
        # Example: place a market buy order for gold futures
        # Note: MEXC Futures uses symbol format with underscore: XAUT_USDT
        # exchange.place_order('XAUT_USDT', 'MARKET', 'BUY', 0.01)
        pass
    else:
        print("API Key or Secret not found. Please check your config.api file.")
