import hmac
import hashlib
import time
import json
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
        """Get account balance for all assets
        Endpoint: GET /api/v1/private/account/assets (MEXC Futures API v1)
        """
        # If no API keys configured, return None immediately
        if not self.api_key or not self.api_secret:
            return None
        
        # Correct endpoint for MEXC Futures API v1
        endpoint = "/api/v1/private/account/assets"
        timestamp = self._get_timestamp()
        
        params = {
            'timestamp': timestamp
        }
        
        try:
            # Generate signature (signature goes in header, not params)
            signature = self._sign_request('GET', timestamp, params=params)
        except ValueError as e:
            print(f"Cannot sign request: {e}")
            return None
        
        # MEXC Futures API v1 requires ApiKey, Request-Time, and Signature headers
        headers = {
            'ApiKey': self.api_key,
            'Request-Time': str(timestamp),
            'Signature': signature,
            'Content-Type': 'application/json'
        }
        
        try:
            # Increased timeout to 30 seconds
            response = requests.get(
                self.base_url + endpoint,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # MEXC Futures API v1 returns: {"code": 0, "data": {...}, "msg": "success"}
            # Debug: Print full response if there's an error
            if data.get('code') == 0:
                return data.get('data', {})
            else:
                # Print detailed error information
                error_msg = data.get('msg') or data.get('message') or data.get('error') or 'Unknown error'
                error_code = data.get('code') or data.get('status') or 'N/A'
                print(f"API Error (code: {error_code}): {error_msg}")
                # Print full response for debugging (truncated to 500 chars)
                print(f"Full API response: {str(data)[:500]}")
                return None
        except requests.exceptions.Timeout:
            print("Error getting account balance: Request timed out (30s)")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error getting account balance: {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    error_data = e.response.json()
                    # Try multiple possible error message fields
                    error_msg = error_data.get('msg') or error_data.get('message') or error_data.get('error') or error_data.get('detail') or 'Unknown error'
                    error_code = error_data.get('code') or error_data.get('status') or e.response.status_code
                    print(f"Error response (code: {error_code}): {error_msg}")
                    # Print full response for debugging
                    print(f"Full error response: {str(error_data)[:500]}")
                except:
                    print(f"Error response (raw): {e.response.text[:500]}")
            return None

    def get_usdt_balance(self):
        """Get available USDT balance"""
        # If no API keys, return mock balance for TEST mode
        if not self.api_key or not self.api_secret:
            print("Using mock balance (TEST mode)")
            return 1000.0  # Mock balance for testing
        
        account_info = self.get_account_balance()
        if account_info:
            # MEXC Futures API v1 response structure may vary
            # Try different possible response structures
            if 'assets' in account_info:
                for asset in account_info['assets']:
                    if asset.get('asset') == 'USDT':
                        return float(asset.get('available', asset.get('free', 0)))
            elif 'balances' in account_info:
                for balance in account_info['balances']:
                    if balance.get('asset') == 'USDT':
                        return float(balance.get('available', balance.get('free', 0)))
            elif 'USDT' in account_info:
                # If USDT is directly in the response
                usdt_data = account_info['USDT']
                return float(usdt_data.get('available', usdt_data.get('free', 0)))
        
        # If API call failed but keys exist, return mock balance
        print("API call failed or USDT not found, using mock balance")
        return 1000.0

    def place_order(self, symbol, order_type, side, quantity, price=None):
        """Place order
        Endpoint: POST /api/v1/private/order/submit (MEXC Futures API v1)
        Note: MEXC Futures API may have restrictions on order placement
        """
        # Correct endpoint for MEXC Futures API v1
        endpoint = "/api/v1/private/order/submit"
        timestamp = self._get_timestamp()
        
        # MEXC Futures uses symbol without underscore (BTCUSDT not BTC_USDT)
        symbol_clean = symbol.replace('_', '')
        
        # Build JSON body for POST request
        json_body_dict = {
            'symbol': symbol_clean,
            'side': side.upper(),  # BUY or SELL
            'type': order_type.upper(),  # MARKET or LIMIT
            'quantity': str(quantity)
        }
        
        # Add price for LIMIT orders
        if order_type.upper() == 'LIMIT' and price:
            json_body_dict['price'] = str(price)
        
        # Convert to JSON string for signing (no spaces, sorted keys for consistency)
        # This exact string must be used for both signing and sending
        json_body_string = json.dumps(json_body_dict, separators=(',', ':'), sort_keys=True)
        
        try:
            # Generate signature using JSON body string
            signature = self._sign_request('POST', timestamp, json_body=json_body_string)
        except ValueError as e:
            print(f"Cannot sign request: {e}")
            return None
        
        # MEXC Futures API v1 requires ApiKey, Request-Time, and Signature headers
        headers = {
            'ApiKey': self.api_key,
            'Request-Time': str(timestamp),
            'Signature': signature,
            'Content-Type': 'application/json'
        }
        
        try:
            # POST request with JSON body - use the exact JSON string used for signing
            response = requests.post(
                self.base_url + endpoint,
                headers=headers,
                data=json_body_string,  # Use exact JSON string to match signature
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # MEXC Futures API v1 returns: {"code": 0, "data": {...}, "msg": "success"}
            if data.get('code') == 0:
                print(f"Successfully placed {side} order for {quantity} {symbol_clean}.")
                return data.get('data', {})
            else:
                print(f"API Error placing order: {data.get('msg', 'Unknown error')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error placing order: {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    error_data = e.response.json()
                    print(f"Error response: {error_data.get('msg', e.response.text[:200])}")
                except:
                    print(f"Error response: {e.response.text[:200]}")
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
