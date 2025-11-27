import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd

class DataProcessor:
    def __init__(self, look_back=60, future_horizon=15, tp_sl_ratio=0.5):
        self.look_back = look_back
        self.future_horizon = future_horizon  # Look 15 minutes ahead for TP/SL targets
        self.tp_sl_ratio = tp_sl_ratio  # Set to 0.5 for high sensitivity (0.5% moves)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators from OHLCV data.
        These are the features the model will learn from.
        """
        df = df.copy()
        
        # Basic price features
        df['open'] = df['open']  # OP - Open price
        df['high'] = df['high']  # HP - High price
        df['low'] = df['low']    # LP - Low price
        df['close'] = df['close']  # CP - Close price
        
        # Derived price features
        df['med'] = (df['high'] + df['low']) / 2  # MED - Median price (same as MID)
        df['mid'] = (df['high'] + df['low']) / 2  # MID - Midpoint price
        df['typ'] = (df['high'] + df['low'] + df['close']) / 3  # TYP - Typical price
        df['mean'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4  # MEAN - Mean price
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_diff'] = (df['close'] - df['open']) / df['open']
        
        # Highest High and Lowest Low (rolling windows)
        df['hh_14'] = df['high'].rolling(window=14).max()  # HH(14) - Highest High
        df['ll_14'] = df['low'].rolling(window=14).min()   # LL(14) - Lowest Low
        df['hh_20'] = df['high'].rolling(window=20).max()  # HH(20)
        df['ll_20'] = df['low'].rolling(window=20).min()   # LL(20)
        
                              # Moving Averages (SMA)
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages (EMA)
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Weighted Moving Average (WMA) - weights decrease linearly
        def wma(series, window):
            weights = np.arange(1, window + 1)
            return series.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        df['wma_14'] = wma(df['close'], 14)
        df['wma_20'] = wma(df['close'], 20)
        
        # Double EMA (DEMA) - 2*EMA - EMA(EMA)
        ema_14 = df['close'].ewm(span=14, adjust=False).mean()
        ema_ema_14 = ema_14.ewm(span=14, adjust=False).mean()
        df['dema_14'] = 2 * ema_14 - ema_ema_14
        
        # Triple EMA (TEMA) - 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
        ema_ema_ema_14 = ema_ema_14.ewm(span=14, adjust=False).mean()
        df['tema_14'] = 3 * ema_14 - 3 * ema_ema_14 + ema_ema_ema_14
        
        # Kaufman Adaptive Moving Average (KAMA)
        def kama(series, period=14, fast=2, slow=30):
            change = abs(series - series.shift(period))
            volatility = series.diff().abs().rolling(window=period).sum()
            efficiency_ratio = change / (volatility + 1e-10)
            smoothing_constant = (efficiency_ratio * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
            kama_values = pd.Series(index=series.index, dtype=float)
            kama_values.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                kama_values.iloc[i] = kama_values.iloc[i-1] + smoothing_constant.iloc[i] * (series.iloc[i] - kama_values.iloc[i-1])
            return kama_values
        df['kama_14'] = kama(df['close'], period=14)
        
        # Price position relative to moving averages (normalized)
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / (df['sma_20'] + 1e-10)
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 1e-10)
        df['price_vs_sma100'] = (df['close'] - df['sma_100']) / (df['sma_100'] + 1e-10)
        df['price_vs_sma200'] = (df['close'] - df['sma_200']) / (df['sma_200'] + 1e-10)
        df['price_vs_ema20'] = (df['close'] - df['ema_20']) / (df['ema_20'] + 1e-10)
        df['price_vs_ema50'] = (df['close'] - df['ema_50']) / (df['ema_50'] + 1e-10)
        df['price_vs_ema100'] = (df['close'] - df['ema_100']) / (df['ema_100'] + 1e-10)
        df['price_vs_ema200'] = (df['close'] - df['ema_200']) / (df['ema_200'] + 1e-10)
        
        # Moving average crossovers
        df['sma5_sma20_cross'] = (df['sma_5'] - df['sma_20']) / (df['sma_20'] + 1e-10)
        df['sma20_sma50_cross'] = (df['sma_20'] - df['sma_50']) / (df['sma_50'] + 1e-10)
        df['sma50_sma200_cross'] = (df['sma_50'] - df['sma_200']) / (df['sma_200'] + 1e-10)
        df['ema5_ema20_cross'] = (df['ema_5'] - df['ema_20']) / (df['ema_20'] + 1e-10)
        df['ema20_ema50_cross'] = (df['ema_20'] - df['ema_50']) / (df['ema_50'] + 1e-10)
        df['ema50_ema200_cross'] = (df['ema_50'] - df['ema_200']) / (df['ema_200'] + 1e-10)
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Prevent division by zero
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26  # MACD - Moving Average Convergence/Divergence
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()  # MACDS - MACD Signal
        df['macd_diff'] = df['macd'] - df['macd_signal']  # MACDH - MACD Histogram
        
        # Percentage Price Oscillator (PPO) - percentage difference between EMAs
        df['ppo'] = ((ema_12 - ema_26) / ema_26) * 100  # PPO - Percentage Price Oscillator
        df['ppo_signal'] = df['ppo'].ewm(span=9, adjust=False).mean()
        df['ppo_hist'] = df['ppo'] - df['ppo_signal']
        
        # Commodity Channel Index (CCI)
        typical_price = df['typ']
        sma_tp = typical_price.rolling(window=20).mean()
        mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci_20'] = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)  # CCI - Commodity Channel Index
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()  # MBB - Middle Bollinger Band
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)  # UBB - Upper Bollinger Band
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)  # LBB - Lower Bollinger Band
        # Prevent division by zero
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['pctbb_20'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10) * 100  # PCTBB - Percentage Bollinger Bands
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        # Prevent division by zero
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        
        # Money Flow Index (MFI) - volume-weighted RSI
        typical_price = df['typ']  # Use typical price
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
        mfi_ratio = positive_flow / (negative_flow + 1e-10)
        df['mfi_14'] = 100 - (100 / (1 + mfi_ratio))  # MFI - Money Flow Index
        
        # Chaikin A/D Line (AD) - Accumulation/Distribution
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)  # Close Location Value
        df['ad'] = (clv * df['volume']).cumsum()  # AD - Chaikin A/D Line
        
        # Chaikin A/D Oscillator (CO) - difference of EMAs of A/D
        ad_ema_fast = df['ad'].ewm(span=3, adjust=False).mean()
        ad_ema_slow = df['ad'].ewm(span=10, adjust=False).mean()
        df['co'] = ad_ema_fast - ad_ema_slow  # CO - Chaikin Oscillator
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['rate_of_change'] = df['close'].pct_change(periods=10)
        df['price_acceleration'] = df['close'].pct_change(periods=10).diff()  # Second derivative
        
        # Volatility - ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)  # TRANGE - True Range
        df['trange'] = true_range
        df['atr'] = true_range.rolling(window=14).mean()  # ATR - Average True Range
        df['atr_pct'] = df['atr'] / (df['close'] + 1e-10)  # ATR as percentage of price
        df['natr_14'] = (df['atr'] / (df['close'] + 1e-10)) * 100  # NATR - Normalized ATR
        
        # Traditional volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Stochastic Oscillator
        # Fast Stochastic %K and %D
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['fk_14'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 1e-10))  # FK - Fast %K
        df['fd_14'] = df['fk_14'].rolling(window=3).mean()  # FD - Fast %D
        
        # Slow Stochastic %K and %D (smoothed fast stochastic)
        df['sk_14'] = df['fk_14'].rolling(window=3).mean()  # SK - Slow %K
        df['sd_14'] = df['sk_14'].rolling(window=3).mean()  # SD - Slow %D
        
        # Keep old names for backward compatibility
        df['stoch_k'] = df['fk_14']
        df['stoch_d'] = df['fd_14']
        
        # Williams' %R (WILLR) - momentum oscillator
        df['willr_14'] = -100 * ((high_14 - df['close']) / (high_14 - low_14 + 1e-10))
        
        # DMI (Directional Movement Index) - Includes +DI, -DI, and ADX
        # Calculate +DM and -DM
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        # Ensure opposite direction is zero when same direction is positive
        plus_dm[minus_dm > plus_dm] = 0
        minus_dm[plus_dm > minus_dm] = 0
        
        # Store PLUS-DM and MINUS-DM
        df['plus_dm'] = plus_dm  # PLUS-DM - Plus Directional Movement
        df['minus_dm'] = minus_dm  # MINUS-DM - Minus Directional Movement
        
        # Calculate smoothed +DM, -DM, and TR
        period = 14
        tr_smoothed = true_range.rolling(window=period).mean()
        plus_dm_smoothed = plus_dm.rolling(window=period).mean()
        minus_dm_smoothed = minus_dm.rolling(window=period).mean()
        
        # Calculate +DI and -DI (Directional Indicators)
        df['plus_di'] = 100 * (plus_dm_smoothed / (tr_smoothed + 1e-10))  # PLUS-DI - Plus Directional Indicator
        df['minus_di'] = 100 * (minus_dm_smoothed / (tr_smoothed + 1e-10))  # MINUS-DI - Minus Directional Indicator
        
        # Calculate DX (Directional Index)
        dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
        df['dx'] = dx  # DX - Directional Movement Index
        df['adx'] = dx.rolling(window=period).mean()  # ADX - Average DX
        
        # ADXR - ADX Rating (average of current ADX and ADX from period ago)
        df['adxr'] = (df['adx'] + df['adx'].shift(period)) / 2  # ADXR - ADX Rating
        
        # DMI crossover signal (when +DI crosses -DI)
        df['dmi_cross'] = df['plus_di'] - df['minus_di']
        
        # Volume-weighted features
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
        
        # Local support/resistance (rolling min/max)
        df['local_high'] = df['high'].rolling(window=20).max()
        df['local_low'] = df['low'].rolling(window=20).min()
        df['dist_to_high'] = (df['close'] - df['local_high']) / (df['local_high'] + 1e-10)
        df['dist_to_low'] = (df['close'] - df['local_low']) / (df['local_low'] + 1e-10)
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        return df

    def get_train_test_data(self, df, train_split=0.8):
        """
        Process raw OHLCV data and create train/test datasets.
        
        Steps:
        1. Calculate technical indicators (drops NaN rows)
        2. Generate trading labels (buy/sell signals)
        3. Filter valid signals
        4. Scale features
        5. Create sequences for LSTM
        """
        # Calculate technical indicators first (this drops NaN rows)
        df = self.calculate_technical_indicators(df)
        
        # Generate labels based on future price movement
        df = self._generate_labels(df)
        
        # Filter out 'none' signals (-1)
        df_filtered = df[df['signal'] != -1].copy()

        # ----------------------------------------------------------------------
        # 💡 FIX: Add check for empty DataFrame after filtering.
        # ----------------------------------------------------------------------
        if df_filtered.empty:
            raise ValueError(
                "Could not find any training signals (buy/sell) in the provided data.\n"
                "This usually means the market was not volatile enough during the period.\n\n"
                "Possible solutions:\n"
                "1. Fetch a larger dataset by deleting 'training_data.csv' and running train.py again.\n"
                "2. Adjust the signal generation parameters (e.g., lower 'tp_sl_ratio') in data_processor.py."
            )

        print(f"Found {len(df_filtered)} valid trading signals from {len(df)} total data points")

        # Select feature columns (technical indicators)
        feature_columns = [
                          # Basic prices
            'open', 'high', 'low', 'close',
            # Derived prices
            'med', 'mid', 'typ', 'mean',
            # Price-based features
            'price_change', 'high_low_range', 'close_open_diff',
            # Highest High / Lowest Low
            'hh_14', 'll_14', 'hh_20', 'll_20',
            # Moving Averages (SMA)
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            # Exponential Moving Averages (EMA)
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            # Advanced Moving Averages
            'wma_14', 'wma_20', 'dema_14', 'tema_14', 'kama_14',
            # Price vs MA/EMA ratios
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma100', 'price_vs_sma200',
            'price_vs_ema20', 'price_vs_ema50', 'price_vs_ema100', 'price_vs_ema200',
            # MA/EMA Crossovers
            'sma5_sma20_cross', 'sma20_sma50_cross', 'sma50_sma200_cross',
            'ema5_ema20_cross', 'ema20_ema50_cross', 'ema50_ema200_cross',
            # Oscillators
            'rsi', 'willr_14',
            # MACD
            'macd', 'macd_signal', 'macd_diff',
            # PPO
            'ppo', 'ppo_signal', 'ppo_hist',
            # Stochastic
            'fk_14', 'fd_14', 'sk_14', 'sd_14', 'stoch_k', 'stoch_d',
            # DMI (Directional Movement Index)
            'plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx', 'adx', 'adxr', 'dmi_cross',
            # Bollinger Bands
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'pctbb_20',
            # CCI
            'cci_20',
            # Volume
            'volume_change', 'volume_ratio', 'vwap', 'price_vs_vwap',
            # Volume-based indicators
            'mfi_14', 'ad', 'co',
            # Momentum
            'momentum', 'rate_of_change', 'price_acceleration',
            # Volatility
            'volatility', 'trange', 'atr', 'atr_pct', 'natr_14',
            # Support/Resistance
            'local_high', 'local_low', 'dist_to_high', 'dist_to_low'
        ]
        
        # Extract and scale features
        data = df_filtered[feature_columns].values
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data)
        
        # Create a NEW DataFrame with scaled features and reset index
        df_filtered_scaled = pd.DataFrame(scaled_data, columns=feature_columns)
        df_filtered_scaled['signal'] = df_filtered['signal'].values
        df_filtered_scaled['take_profit'] = df_filtered['take_profit'].values
        df_filtered_scaled['stop_loss'] = df_filtered['stop_loss'].values
        
        # Reset index to ensure continuous indexing
        df_filtered_scaled = df_filtered_scaled.reset_index(drop=True)

        train_size = int(len(df_filtered_scaled) * train_split)
        
        print(f"Creating sequences with look_back={self.look_back}")
        print(f"Train set: {train_size} samples, Test set: {len(df_filtered_scaled) - train_size} samples")
        
        X_train, y_train_signal, y_train_tp, y_train_sl = self._create_dataset(
            df_filtered_scaled.iloc[:train_size], feature_columns
        )
        
        X_test, y_test_signal, y_test_tp, y_test_sl = self._create_dataset(
            df_filtered_scaled.iloc[train_size:], feature_columns
        )
        
        print(f"Created {len(X_train)} training sequences and {len(X_test)} test sequences")
        
        # Check if we have enough training data
        if len(X_train) == 0:
            raise ValueError(
                f"Not enough data to create training sequences.\n"
                f"Found {len(df_filtered)} signals, but need at least {self.look_back + 1} for training.\n"
                f"Try lowering tp_sl_ratio even further or fetching more historical data."
            )
        
        # Scale TP/SL using only close price for inverse transform compatibility
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaler.fit(df_filtered[['close']].values)
        self.close_scaler = close_scaler  # Save for later use in prediction

        # Handle case where test set might be empty
        if len(X_test) == 0:
            print("⚠️  Warning: Not enough data for test set. Using training data for both train and test.")
            X_test, y_test_signal, y_test_tp, y_test_sl = X_train, y_train_signal, y_train_tp, y_train_sl

        return (torch.from_numpy(X_train).float(), 
                torch.from_numpy(y_train_signal).long(), 
                torch.from_numpy(close_scaler.transform(y_train_tp.reshape(-1,1))).float(), 
                torch.from_numpy(close_scaler.transform(y_train_sl.reshape(-1,1))).float(),
                torch.from_numpy(X_test).float(),
                torch.from_numpy(y_test_signal).long(),
                torch.from_numpy(close_scaler.transform(y_test_tp.reshape(-1,1))).float(),
                torch.from_numpy(close_scaler.transform(y_test_sl.reshape(-1,1))).float())

    def _generate_labels(self, df):
        """
        Generate trading labels based on future price movement.
        Uses .iloc for position-based indexing to avoid index issues.
        """
        df = df.reset_index(drop=False)  # Keep the index as a column temporarily
        
        df['signal'] = -1  # -1 for none, 0 for sell, 1 for buy
        df['take_profit'] = df['close']
        df['stop_loss'] = df['close']

        for i in range(len(df) - self.future_horizon):
            # Use .iloc for position-based indexing
            future_prices = df['close'].iloc[i+1:i+1+self.future_horizon]
            current_price = df['close'].iloc[i]
            
            if future_prices.max() > current_price * (1 + 0.01 * self.tp_sl_ratio):
                df.loc[df.index[i], 'signal'] = 1
                  # BUY: TP = 90% of max price, SL = 20% below entry
                max_price = future_prices.max()
                df.loc[df.index[i], 'take_profit'] = current_price + (max_price - current_price) * 0.90
                df.loc[df.index[i], 'stop_loss'] = current_price * (1 - 0.20)

            elif future_prices.min() < current_price * (1 - 0.01 * self.tp_sl_ratio):
                df.loc[df.index[i], 'signal'] = 0
                # SELL: TP = min price, SL = 20% above entry
                df.loc[df.index[i], 'take_profit'] = future_prices.min()
                df.loc[df.index[i], 'stop_loss'] = current_price * (1 + 0.20)
        
        # Set the first column back as index if it was the original index
        if 'open_time' in df.columns:
            df.set_index('open_time', inplace=True)
        
        return df

    def _create_dataset(self, df_subset, feature_columns):
        X, y_signal, y_tp, y_sl = [], [], [], []
        
        # Get all feature values as a 2D array
        feature_values = df_subset[feature_columns].values
        signals = df_subset['signal'].values
        tps = df_subset['take_profit'].values
        sls = df_subset['stop_loss'].values

        for i in range(self.look_back, len(df_subset)):
            # Each sample is look_back timesteps x num_features
            X.append(feature_values[i-self.look_back:i])
            y_signal.append(signals[i])
            y_tp.append(tps[i])
            y_sl.append(sls[i])
            
        return np.array(X), np.array(y_signal), np.array(y_tp), np.array(y_sl)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()
        return self.scaler.inverse_transform(data.reshape(-1, 1))
