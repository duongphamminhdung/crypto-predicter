import pandas as pd
import numpy as np
from data_processor import DataProcessor

def debug_label_generation():
    print("Loading training_data.csv...")
    try:
        df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded {len(df)} rows.")
    
    # Initialize processor with the parameters used in train.py
    # look_back=60, future_horizon=240, tp_sl_ratio=0.3
    processor = DataProcessor(look_back=60, future_horizon=240, tp_sl_ratio=0.3)
    
    print(f"Processor config: Horizon={processor.future_horizon}, Threshold={processor.tp_sl_ratio}%")
    
    # Step 1: Check generation on raw data (simulating what happens in train.py before saving)
    print("\n--- Test 1: Generate labels on raw data ---")
    labeled_df = processor._generate_labels(df.copy())
    
    buy_signals = len(labeled_df[labeled_df['signal'] == 1])
    sell_signals = len(labeled_df[labeled_df['signal'] == 0])
    none_signals = len(labeled_df[labeled_df['signal'] == -1])
    
    print(f"Signals: BUY={buy_signals}, SELL={sell_signals}, NONE={none_signals}")
    
    if buy_signals + sell_signals == 0:
        print("❌ No signals generated on raw data!")
        # Debug why
        check_volatility(df, processor.future_horizon, processor.tp_sl_ratio)
    else:
        print("✅ Signals generated successfully on raw data.")

    # Step 2: Check generation inside get_train_test_data flow
    print("\n--- Test 2: Generate labels after technical indicators ---")
    try:
        df_indicators = processor.calculate_technical_indicators(df.copy())
        print(f"Data shape after indicators (dropped NaNs): {df_indicators.shape}")
        
        labeled_df_2 = processor._generate_labels(df_indicators.copy())
        
        buy_signals_2 = len(labeled_df_2[labeled_df_2['signal'] == 1])
        sell_signals_2 = len(labeled_df_2[labeled_df_2['signal'] == 0])
        
        print(f"Signals after indicators: BUY={buy_signals_2}, SELL={sell_signals_2}")
        
        if buy_signals_2 + sell_signals_2 == 0:
            print("❌ No signals generated after indicators!")
    except Exception as e:
        print(f"Error calculating indicators: {e}")

def check_volatility(df, horizon, threshold):
    print("\nDebug: Checking price moves manually...")
    df = df.reset_index(drop=True)
    max_up = 0
    max_down = 0
    
    threshold_factor = 0.01 * threshold
    
    for i in range(len(df) - horizon):
        current_price = df['close'].iloc[i]
        future_window = df['close'].iloc[i+1 : i+1+horizon]
        
        if future_window.empty:
            continue
            
        window_max = future_window.max()
        window_min = future_window.min()
        
        up_move = (window_max - current_price) / current_price
        down_move = (current_price - window_min) / current_price
        
        max_up = max(max_up, up_move)
        max_down = max(max_down, down_move)
        
        if up_move > threshold_factor:
            print(f"Found UP move at index {i}: {up_move*100:.2f}% (Threshold: {threshold}%)")
            return
        if down_move > threshold_factor:
            print(f"Found DOWN move at index {i}: {down_move*100:.2f}% (Threshold: {threshold}%)")
            return
            
    print(f"Max UP move found: {max_up*100:.2f}%")
    print(f"Max DOWN move found: {max_down*100:.2f}%")

if __name__ == "__main__":
    debug_label_generation()

