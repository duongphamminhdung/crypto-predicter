from mexc_client import MEXCClient
from data_processor import DataProcessor
from model import CryptoPredicter
import torch
import joblib
import pandas as pd
import os
import sys

# Model directory for saving/loading models and scalers
MODEL_DIR = "../model"

def main():
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"📁 Model directory: {os.path.abspath(MODEL_DIR)}")
    # Check if we're in test mode
    test_mode = '--test-mode' in sys.argv
    
    client = MEXCClient()
    
    df = None
    is_fine_tuning = False
    
    # Attempt to load recent training data first (for fine-tuning)
    if os.path.exists('recent_training_data.csv'):
        print("🔧 Fine-tuning mode: Using latest historical data for targeted retraining.")
        is_fine_tuning = True
        try:
            df = pd.read_csv('recent_training_data.csv', index_col=0, parse_dates=True)
            if df.empty:
                print("Warning: recent_training_data.csv was empty. Ignoring.")
                df = None
                is_fine_tuning = False
        except Exception as e:
            print(f"Error reading recent_training_data.csv, it may be corrupted. Error: {e}")
            df = None
            is_fine_tuning = False
        finally:
            # Always remove the temp file
            os.remove('recent_training_data.csv')
    
    # For fine-tuning with latest historical data, use more lenient parameters
    if is_fine_tuning:
            # Use smaller lookback and lower tp_sl_ratio to get more signals from latest data
        print("🔧 Fine-tuning parameters: look_back=30, tp_sl_ratio=0.3, future_horizon=15 (more lenient for better signal generation)")
        processor = DataProcessor(look_back=30, tp_sl_ratio=0.3, future_horizon=15)
    else:
        # Use 60-minute lookback window and 15-minute future horizon for full training
        print("🏗️  Training parameters: look_back=60, future_horizon=15 (predicting TP/SL for 15-minute window)")
        processor = DataProcessor(look_back=60, future_horizon=15)

    # If no recent data, use the main training data file
    if df is None and os.path.exists('training_data.csv'):
        print("Loading existing training data.")
        try: 
            df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
            
            # Check if dataframe has actual data rows (not just headers)
            if df.empty or len(df) < 100:
                print(f"Warning: training_data.csv has only {len(df)} rows. Need at least 100 rows.")
                raise ValueError("training_data.csv is too small or empty.")
            
            print(f"Loaded {len(df)} rows from training_data.csv")
            
            # Fetch new data to append
            last_timestamp = int(df.index[-1].timestamp())
            print("Fetching new data to append...")
            new_df              = client.get_kline_data(symbol='BTC_USDT', interval='Min1', start=last_timestamp)
            new_df['open_time'] = pd.to_datetime(new_df['open_time'])
            new_df.set_index('open_time', inplace=True)
            
            # Only concatenate if we got new data
            new_rows = new_df[~new_df.index.isin(df.index)]
            if len(new_rows) > 0:
                df = pd.concat([df, new_rows])
                print(f"Successfully appended {len(new_rows)} new records.")
            else:
                print("No new data to append. Using existing data.")

        except Exception as e:
            print(f"Error processing training_data.csv: {e}")
            print("Deleting corrupted/empty file and re-fetching fresh data...")
            try:
                os.remove('training_data.csv')
                print("Deleted training_data.csv")
            except OSError as remove_error:
                print(f"Error removing corrupted file: {remove_error}")
            df = None

    # If no data is loaded by this point (either absent, empty, or corrupted), fetch fresh data
    if df is None:
        print("=" * 60)
        print("Fetching initial training data")
        print("MEXC Futures API returns up to 2000 records per request, fetching multiple batches...")
        print(f"Target: 50000 records (~35 days of 1-minute data)")
        print("=" * 60)
        
        # MEXC Futures API returns up to 2000 records per request
        all_data = []
        total_to_fetch = 70000  # Use 500000 lines for initial training
        
        # Calculate batches needed (2000 records per batch)
        batches_needed = (total_to_fetch + 1999) // 2000  # Ceiling division
        
        for batch_num in range(batches_needed):
            print(f"Fetching batch {batch_num + 1}/{batches_needed}...")
            
            if batch_num == 0:
                # First batch - get most recent data (no time constraint)
                batch_df = client.get_kline_data(symbol='BTC_USDT', interval='Min1')
            else:
                # Subsequent batches - get data before the oldest timestamp from previous batch
                # Get the oldest time from the last batch (open_time is still a column at this point)
                oldest_time = int(all_data[-1]['open_time'].iloc[0].timestamp())
                    # Get data ending before the oldest point we have
                batch_df = client.get_kline_data(symbol='BTC_USDT', interval='Min1', end=oldest_time - 60)
            
            if batch_df is not None and not batch_df.empty:
                # Keep as raw dataframe for now (open_time is a column)
                all_data.append(batch_df)
            else:
                print(f"Warning: Failed to fetch batch {batch_num + 1}")
                break
        
        if len(all_data) == 0:
            raise RuntimeError("Failed to fetch any data from MEXC API. Please check your internet connection.")
        
        # Combine all batches
        df = pd.concat(all_data, ignore_index=True)
        
        if df is None or df.empty:
            raise RuntimeError("Failed to fetch data from MEXC API. Please check your internet connection and API configuration.")
        
        # Convert open_time to datetime (it's currently a column)
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.drop_duplicates(subset=['open_time'], keep='first')  # Remove duplicates
            df = df.sort_values('open_time')  # Sort by time
            df.set_index('open_time', inplace=True)
        else:
            # If open_time is already the index, just sort
            df = df.sort_index()
        
        print(f"✅ Successfully fetched {len(df)} records from MEXC")

    # Verify we have enough data
    if len(df) < 100:
        raise ValueError(f"Insufficient data: only {len(df)} rows. Need at least 100 rows for training.")

    print(f"Total data available: {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Generate labels
    print("Generating trading labels (buy/sell signals)...")
    labeled_df = processor._generate_labels(df.copy())
    
    # Save the labeled data (only for full training, not fine-tuning)
    if not is_fine_tuning:
        labeled_df.to_csv('training_data.csv')
        print(f"✅ Training data saved to training_data.csv ({len(labeled_df)} rows)")
    else:
        print(f"✅ Generated labels for {len(labeled_df)} rows (fine-tuning mode - not overwriting main training data)")

    # Process data for training (this also calculates technical indicators)
    print("\nProcessing data for training...")
    X_train, y_train_signal, y_train_tp, y_train_sl, _, _, _, _ = processor.get_train_test_data(labeled_df)
    print("✅ Data processing complete.")

    # Build and train model
    predicter = CryptoPredicter()
    
    # Check if this is fine-tuning mode
    if is_fine_tuning:
        # Fine-tuning: Load existing model and continue training with latest historical data
        print("🔧 Fine-tuning mode: Loading existing model...")
        model_path = os.path.join(MODEL_DIR, 'btc_predicter_model.pth')
        if os.path.exists(model_path):
            predicter.load_model(model_path)
            print("✅ Existing model loaded successfully.")
        else:
            print("⚠️  Warning: No existing model found. Training from scratch.")
        
        print("🎯 Fine-tuning with latest historical data (time-weighted)...")
        # 40 epochs for fine-tuning, with time weighting, lower learning rate
        predicter.train_model(X_train, y_train_signal, y_train_tp, y_train_sl, epochs=50, lr=0.0001, time_weighted=True)
        print("✅ Fine-tuning complete.")
    else:
        # Initial training from scratch
        print("🏗️  Training model from scratch...")
        predicter.train_model(X_train, y_train_signal, y_train_tp, y_train_sl, epochs=100, lr=0.0005, time_weighted=False)
        print("✅ Model training complete.")

    # Save the trained model and the scaler
    if test_mode:
        model_file = os.path.join(MODEL_DIR, 'btc_predicter_model_test.pth')
        print("Test mode: Saving model to separate file for testing...")
    else:
        model_file = os.path.join(MODEL_DIR, 'btc_predicter_model.pth')
    
    predicter.save_model(model_file)
    scaler_file = os.path.join(MODEL_DIR, 'scaler.gz')
    close_scaler_file = os.path.join(MODEL_DIR, 'close_scaler.gz')
    
    joblib.dump(processor.scaler, scaler_file)
    
    # Also save the close_scaler for TP/SL inverse transform
    if hasattr(processor, 'close_scaler'):
        joblib.dump(processor.close_scaler, close_scaler_file)
        print(f"✅ Model saved to {model_file}")
        print(f"✅ Feature scaler saved to {scaler_file}")
        print(f"✅ Close scaler saved to {close_scaler_file}")
    else:
        print(f"✅ Model saved to {model_file}")
        print(f"✅ Scaler saved to {scaler_file}")

if __name__ == '__main__':
    main()
