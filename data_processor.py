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
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_diff'] = (df['close'] - df['open']) / df['open']
        
        # Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        
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
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        # Prevent division by zero
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        # Prevent division by zero
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['rate_of_change'] = df['close'].pct_change(periods=10)
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        
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
            'close', 'price_change', 'high_low_range', 'close_open_diff',
            'sma_5', 'sma_10', 'sma_20', 'sma_100', 'ema_5', 'ema_10',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_position',
            'volume_change', 'volume_ratio',
            'momentum', 'rate_of_change', 'volatility'
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
