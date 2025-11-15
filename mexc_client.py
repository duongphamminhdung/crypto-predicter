import requests
import pandas as pd
import time

class MEXCClient:
    """
    MEXC Futures/Contract API Client for Gold (XAUT/USDT) trading
    Based on: https://mexcdevelop.github.io/apidocs/contract_v1_en/
    """
    def __init__(self):
        self.base_url = "https://contract.mexc.com"
    
    def get_server_time(self):
        """
        Get server time
        Endpoint: GET /api/v1/contract/ping
        Weight: 1
        
        Returns server timestamp in milliseconds
        """
        endpoint = "/api/v1/contract/ping"
        try:
            response = requests.get(self.base_url + endpoint, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            # MEXC Futures API response format: {"success": true, "code": 0, "data": timestamp}
            if result.get('success') and result.get('code') == 0:
                return result.get('data')
            else:
                error_msg = result.get('message') or result.get('msg') or 'Unknown error'
                print(f"API Error getting server time: {error_msg} (code: {result.get('code')})")
                return None
        except Exception as e:
            print(f"Error getting server time: {e}")
            return None
    
    def get_contract_detail(self, symbol='XAUT_USDT'): 
        """
        Get contract information
        Endpoint: GET /api/v1/contract/detail
        Weight: 1
        
        :param symbol: Contract symbol, e.g., 'XAU_USDT' (Gold/USDT futures) or 'BTC_USDT'
        :return      : Contract details dict or None
        """
        endpoint = "/api/v1/contract/detail"
        params   = {'symbol': symbol}
        
        try:
            response = requests.get(self.base_url + endpoint, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            # MEXC Futures API response format: {"success": true, "code": 0, "data": {...}}
            if result.get('success') and result.get('code') == 0:
                return result.get('data')
            else:
                error_msg = result.get('message') or result.get('msg') or 'Unknown error'
                code = result.get('code', 'unknown')
                print(f"API Error getting contract detail: {error_msg} (code: {code})")
                return None
        except Exception as e:
            print(f"Error getting contract detail: {e}")
            return None
    
    def get_kline_data(self, symbol='XAUT_USDT', interval='Min1', start=None, end=None, limit=None):
        """
        Get K-line data for futures contract
        Endpoint: GET /api/v1/contract/kline/{symbol}
        
        According to MEXC Futures API docs:
        - Symbol format: XAUT_USDT (with underscore for futures) - Gold/USDT futures
        - Valid intervals: Min1, Min5, Min15, Min30, Min60, Hour4, Hour8, Day1, Week1, Month1
        - Limit: Default 2000, max 2000 records per request
        - Weight: 1
        
        :param symbol: Contract symbol, e.g., 'XAUT_USDT' (Gold/USDT futures) or 'BTC_USDT'
        :param interval: Min1, Min5, Min15, Min30, Min60, Hour4, Hour8, Day1, Week1, Month1
        :param start: Start timestamp in seconds (optional)
        :param end: End timestamp in seconds (optional)
        :param limit: Number of records to return (1-2000, default: 2000)
        :return: DataFrame with k-line data
        """
        endpoint = f"/api/v1/contract/kline/{symbol}"
        
        params = {
            'interval': interval
        }
        
        if start:
            params['start'] = int(start)
        if end:
            params['end'] = int(end)
        if limit:
            params['limit'] = min(int(limit), 2000)  # Cap at 2000
        
        try:
            response = requests.get(self.base_url + endpoint, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # MEXC Futures API response format: {"success": true, "code": 0, "data": {...}}
            if not result.get('success') or result.get('code') != 0:
                error_msg = result.get('message') or result.get('msg') or 'Unknown error'
                code = result.get('code', 'unknown')
                print(f"❌ API Error fetching kline: {error_msg} (code: {code})")
                return None
            
            data = result.get('data', {})
            
            # Futures API returns data in this structure:
            # {
            #   "time": [timestamp1, timestamp2, ...],      # Timestamp in seconds
            #   "open": [price1, price2, ...],              # Opening price
            #   "close": [price1, price2, ...],             # Closing price
            #   "high": [price1, price2, ...],              # Highest price
            #   "low": [price1, price2, ...],               # Lowest price
            #   "vol": [volume1, volume2, ...],             # Volume (contracts)
            #   "amount": [amount1, amount2, ...],          # Amount (USDT value)
            #   "realOpen": [price1, price2, ...],          # Real open (optional)
            #   "realClose": [price1, price2, ...],         # Real close (optional)
            #   "realHigh": [price1, price2, ...],          # Real high (optional)
            #   "realLow": [price1, price2, ...]            # Real low (optional)
            # }
            
            if not data or 'time' not in data or not isinstance(data.get('time'), list) or len(data.get('time', [])) == 0:
                print("Warning: API returned no kline data or empty time array")
                return None
            
            # Use real prices if available (more accurate), otherwise use regular prices
            use_real = all(key in data for key in ['realOpen', 'realClose', 'realHigh', 'realLow'])
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'open_time': data.get('time', []),
                'open': data.get('realOpen' if use_real else 'open', data.get('open', [])),
                'high': data.get('realHigh' if use_real else 'high', data.get('high', [])),
                'low': data.get('realLow' if use_real else 'low', data.get('low', [])),
                'close': data.get('realClose' if use_real else 'close', data.get('close', [])),
                'volume': data.get('vol', []),
                'amount': data.get('amount', [])
            })
            
            if df.empty:
                print("Warning: Created empty DataFrame from API data")
                return None
            
            # Convert timestamp to datetime (MEXC futures uses seconds, not milliseconds)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
            df['close_time'] = df['open_time']  # Set close_time same as open_time (can be adjusted if needed)
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by time (oldest to newest)
            df = df.sort_values('open_time').reset_index(drop=True)
            
            print(f"✅ Fetched {len(df)} kline records for {symbol} ({interval})")
            return df
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error fetching kline data: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   URL: {e.response.url}")
                print(f"   Response: {e.response.text[:200]}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching kline data: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error in get_kline_data: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == '__main__':
    client = MEXCClient()
    
    # Test server connection
    print("Testing MEXC Futures API connection...")
    server_time = client.get_server_time()
    print(f"Server time: {server_time}")
    
    # Test contract details
    contract_info = client.get_contract_detail('XAUT_USDT')
    if contract_info:
        print(f"Contract: {contract_info.get('symbol')}")
        print(f"Contract Size: {contract_info.get('contractSize')}")
    
    # Test kline data fetch
    print("\nFetching XAUT_USDT (Gold) kline data...")
    gold_data = client.get_kline_data(symbol='XAUT_USDT', interval='Min1')
    if gold_data is not None:
        print(gold_data.head())
        print(f"\nFetched {len(gold_data)} data points.")
        print(f"Date range: {gold_data['open_time'].iloc[0]} to {gold_data['open_time'].iloc[-1]}")
    else:
        print("Failed to fetch data.")
