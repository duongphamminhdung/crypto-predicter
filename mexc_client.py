import requests
import pandas as pd
import time

class MEXCClient:
    """
    MEXC Futures/Contract API Client
    Based on: https://mexcdevelop.github.io/apidocs/contract_v1_en/
    """
    def __init__(self):
        self.base_url = "https://contract.mexc.com"
    
    def get_server_time(self):
        """
        Get server time
        Endpoint: GET /api/v1/contract/ping
        Weight: 1
        """
        endpoint = "/api/v1/contract/ping"
        try:
            response = requests.get(self.base_url + endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('success'):
                return data.get('data')
            return None
        except Exception as e:
            print(f"Error getting server time: {e}")
            return None
    
    def get_contract_detail(self, symbol='BTC_USDT'):
        """
        Get contract information
        Endpoint: GET /api/v1/contract/detail
        Weight: 1
        
        :param symbol: Contract symbol, e.g., 'BTC_USDT'
        :return: Contract details
        """
        endpoint = "/api/v1/contract/detail"
        params   = {'symbol': symbol}
        
        try:
            response = requests.get(self.base_url + endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('success'):
                return data.get('data')
            else:
                print(f"API Error: {data.get('message', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Error getting contract detail: {e}")
            return None
    
    def get_kline_data(self, symbol='BTC_USDT', interval='Min1', start=None, end=None):
        """
        Get K-line data for futures contract
        Endpoint: GET /api/v1/contract/kline/{symbol}
        
        According to MEXC Futures API docs:
        - Symbol format: BTC_USDT (with underscore for futures)
        - Valid intervals: Min1, Min5, Min15, Min30, Min60, Hour4, Hour8, Day1, Week1, Month1
        - Limit: Default 2000, returns up to 2000 records
        - Weight: 1
        
        :param symbol: Contract symbol, e.g., 'BTC_USDT'
        :param interval: Min1, Min5, Min15, Min30, Min60, Hour4, Hour8, Day1, Week1, Month1
        :param start: Start timestamp in seconds (optional)
        :param end: End timestamp in seconds (optional)
        :return: DataFrame with k-line data
        """
        endpoint = f"/api/v1/contract/kline/{symbol}"
        
        params = {
            'interval': interval
        }
        
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        try:
            response = requests.get(self.base_url + endpoint, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Check if request was successful
            if not result.get('success'):
                print(f"❌ API Error: {result.get('message', 'Unknown error')}")
                return None
            
            data = result.get('data', {})
            
            # Futures API returns data in this structure:
            # {
            #   "time": [timestamp1, timestamp2, ...],
            #   "open": [price1, price2, ...],
            #   "high": [price1, price2, ...],
            #   "low": [price1, price2, ...],
            #   "close": [price1, price2, ...],
            #   "vol": [volume1, volume2, ...],
            #   "amount": [amount1, amount2, ...]
            # }
            
            if not data or 'time' not in data:
                print("Warning: API returned no kline data")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'open_time': data.get('time', []),
                'open': data.get('open', []),
                'high': data.get('high', []),
                'low': data.get('low', []),
                'close': data.get('close', []),
                'volume': data.get('vol', []),
                'amount': data.get('amount', [])
            })
            
            if df.empty:
                print("Warning: Created empty DataFrame from API data")
                return None
            
            # Convert timestamp to datetime (MEXC futures uses seconds, not milliseconds)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
            df['close_time'] = df['open_time']  # Set close_time same as open_time
            
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
    contract_info = client.get_contract_detail('BTC_USDT')
    if contract_info:
        print(f"Contract: {contract_info.get('symbol')}")
        print(f"Contract Size: {contract_info.get('contractSize')}")
    
    # Test kline data fetch
    print("\nFetching BTC_USDT kline data...")
    btc_data = client.get_kline_data(symbol='BTC_USDT', interval='Min1')
    if btc_data is not None:
        print(btc_data.head())
        print(f"\nFetched {len(btc_data)} data points.")
        print(f"Date range: {btc_data['open_time'].iloc[0]} to {btc_data['open_time'].iloc[-1]}")
    else:
        print("Failed to fetch data.")
