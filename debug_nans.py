import pandas as pd
import numpy as np
from data_processor import DataProcessor

def debug_nans():
    print("Loading training_data.csv...")
    try:
        df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded {len(df)} rows.")
    processor = DataProcessor(look_back=60, future_horizon=240, tp_sl_ratio=0.3)
    
    # Calculate indicators WITHOUT dropna first
    df_temp = df.copy()
    
    # We will step through calculate_technical_indicators manually to see where it breaks
    # Or just call it but comment out the dropna in the class (can't do that easily)
    # So we'll instantiate a processor and monkey-patch or just copy the logic?
    
    # Easier: Just run calculate_technical_indicators and before dropna check NaNs.
    # But calculate_technical_indicators has dropna at the end.
    
    # Let's verify if we can check it.
    # I'll modify DataProcessor temporarily or just reimplement the call here?
    # No, I can just check column by column if I paste the code, but that's verbose.
    
    # Wait, I can just edit data_processor.py to print debug info before dropna.
    # That's probably the fastest way.
    pass

if __name__ == "__main__":
    debug_nans()

