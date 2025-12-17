import torch
import joblib
import time
from mexc_client import MEXCClient
from model import CryptoPredicter
from exchange import Exchange
import numpy as np
import pandas as pd
import subprocess
import logging
from datetime import datetime
import json
import os
import sys
import shutil

# ═══════════════════════════════════════════════════════════
# 📝 LOGGING & STATS CONFIGURATION
# ═══════════════════════════════════════════════════════════
LOG_FILE            = "trading_bot.log"
STATS_FILE          = "trading_stats.json"
TRADES_LOG_FILE     = "trades_log.json"      # Separate file for individual trade logs
CURRENT_TRADES_FILE = "current_trades.json"  # File to track current/open trades for persistence
MODEL_DIR           = "../model"             # Directory for saving/loading models and scalers
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# 🧪 TEST MODE CONFIGURATION
# ═══════════════════════════════════════════════════════════
# Set TEST = True to simulate trades without using real money
# Set TEST = False for live trading with real money
TEST = True  # ⚠️ CHANGE THIS TO False FOR LIVE TRADING
# ═══════════════════════════════════════════════════════════

def get_feature_columns():
    return [
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
        'wma_14', 'wma_20', 'dema_14', 'tema_14',
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

                                    # ═══════════════════════════════════════════════════════════
                                    # ⚠️ FUTURES TRADING RISK CONFIGURATION
                                    # ═══════════════════════════════════════════════════════════
                                    # Futures trading uses leverage and is EXTREMELY risky!
                                    # Only trade with very high confidence to minimize risk
CONFIDENCE_THRESHOLD_TRADE  = 0.70  # Only trade when confidence > 70%
CONFIDENCE_THRESHOLD_TEST   = 0.55  # Trigger model testing when below 55%
MAX_POSITION_RISK           = 0.07  # Max 7% of balance at risk per trade (more aggressive)
MAX_LEVERAGE                = 75    # Exchange leverage cap (used more aggressively - max 15x in practice)
                                    # Early stop parameters
EARLY_STOP_MAX_TIME_MINUTES = 300   # Close trade if it's been red for this long (minutes) - 5 hours
EARLY_STOP_OPPOSITE_SIGNAL  = True  # Close losing trades if model predicts opposite signal
EARLY_STOP_LOSS_THRESHOLD   = 5.0    # Close trade when loss reaches $5 USD (standardized, tight risk control)
MIN_LOSS_FOR_OPPOSITE_SIGNAL = 3.0   # Require at least $3 loss before opposite signal can close trade
                                    # Global risk limits
MAX_DAILY_LOSS              = 500.0  # Stop trading if daily loss exceeds $500 USD
MAX_CONCURRENT_TRADES       = 5      # Maximum number of concurrent open positions
MAX_TOTAL_EXPOSURE_PCT      = 10.00   # Max 1000% of balance as total notional exposure (protected by early stops)
                                    # Early stop triggers when BOTH conditions are met: time limit AND opposite signal
                                    # Model refinement parameters
REFINEMENT_INTERVAL_SECONDS = 3600  # Trigger model refinement every 1 hour (3600 seconds)
FINE_TUNE_RECENT_ROWS      = 2000   # Minimum rows to include when preparing recent data for fine-tuning
FINE_TUNE_LOOKBACK         = 30
FINE_TUNE_FUTURE_HORIZON   = 15
FINE_TUNE_TP_SL_RATIO      = 0.3
                                    # ═══════════════════════════════════════════════════════════

def setup_logging():
    """Configure logging to file and console in append mode."""
    # Using basicConfig to set up handlers, ensuring it runs only once
    if not logging.getLogger().handlers:
        # Use append mode for log file
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set format for both handlers
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )

def main():
    setup_logging()
    logging.info("🚀 Starting trading bot...")
    
    feature_columns = get_feature_columns()
    model = CryptoPredicter(input_size=len(feature_columns))
    test_model = CryptoPredicter(input_size=len(feature_columns))
    client = MEXCClient()
    exchange = Exchange()
    
    look_back = 60
    future_horizon = 15  # Look 15 minutes ahead for TP/SL predictions 
    
    # Track active trades for loss detection (now a list to support multiple trades)
    # Load existing trades from file if available (for persistence across restarts)
    active_trades, simulated_trades, trade_counter = load_current_trades()
    logging.info(f"📂 Loaded {len(active_trades)} active trades and {len(simulated_trades)} simulated trades from {CURRENT_TRADES_FILE}")
    
    # Track model testing state
    testing_model = False
    testing_start_time = None
    testing_duration = 3 * 60  # 3 minutes in seconds
    current_model_predictions = []
    test_model_predictions = []
    
    # Track model refinement timing
    last_refinement_time = None  # Will be set on first refinement
    
    # Track consecutive confidence=1.0 predictions for refinement trigger
    consecutive_confidence_1_count = 0
    last_skip_time = None  # Track when we last skipped due to 4 consecutive conf=1.0

    # Initialize daily trading stats from trades_log.json
    daily_stats = initialize_stats_from_trades_log()  # This already saves all stats
    
    # Reconcile any leftover trades from previous run using the latest historical data
    active_trades, simulated_trades, daily_stats, trade_counter = reconcile_trades_on_startup(
        active_trades, simulated_trades, daily_stats, trade_counter
    )
    
    while True:
        try:
            # Check for day change to reset stats
            today = datetime.now().strftime("%Y-%m-%d")
            if daily_stats.get('date') != today:
                logging.info(f"New day ({today}). Resetting daily stats.")
                save_stats(daily_stats) # Save previous day's stats one last time
                # Load stats to get today's entry (or create new one)
                daily_stats = load_stats()

            # Display TEST mode warning
            if TEST:
                logging.info("\n" + "⚠️ "*30)
                logging.info("🧪 TEST MODE ACTIVE - SIMULATING TRADES (NO REAL MONEY)")
                logging.info("⚠️ "*30)
            
            # Check if testing period is over
            if testing_model:
                elapsed_time = time.time() - testing_start_time
                if elapsed_time >= testing_duration:
                    # Testing period complete - compare models and decide
                    if should_adopt_new_model(current_model_predictions, test_model_predictions):
                        logging.info("\n✅ New model performs BETTER! Adopting as primary model.")
                        adopt_new_model()
                    else:
                        logging.info("\n❌ New model does NOT perform better. Keeping current model.")
                    
                    testing_model = False
                    current_model_predictions = []
                    test_model_predictions = []
            
            # Load the current production model
            model_path = os.path.join(MODEL_DIR, 'btc_predicter_model.pth')
            scaler_path = os.path.join(MODEL_DIR, 'scaler.gz')
            close_scaler_path = os.path.join(MODEL_DIR, 'close_scaler.gz')
            
            model.load_model(model_path)
            model.eval()
            scaler = joblib.load(scaler_path)
            
            # Load close_scaler for TP/SL inverse transform
            if os.path.exists(close_scaler_path):
                close_scaler = joblib.load(close_scaler_path)
            else:
                close_scaler = None
                logging.warning(f"{close_scaler_path} not found. TP/SL predictions may be inaccurate.")
            
            if testing_model:
                # Also load the test model for comparison
                test_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model_test.pth')
                test_model.load_model(test_model_path)
                test_model.eval()
                elapsed_time = time.time() - testing_start_time
                remaining_min = int((testing_duration - elapsed_time) / 60)
                remaining_sec = int((testing_duration - elapsed_time) % 60)
                logging.info(f"\n🧪 TESTING MODE - Comparing both models ({remaining_min}m {remaining_sec}s remaining)")
                logging.info(f"   Current: {model_path} | Test: {test_model_path}")
            else:
                logging.info(f"\nModel loaded: {model_path}")

            # Fetch data for prediction - need extra for technical indicators
            # Futures API returns up to 2000 records, so we can get all we need in one request
            df = client.get_kline_data(symbol='BTC_USDT', interval='Min1')  # Gets recent BTC/USDT data
            
            if df is None or df.empty:
                logging.error("Failed to fetch market data. Skipping this cycle.")
                time.sleep(60)
                continue
            
            # Immediately append new data to training_data.csv
            append_data_to_training_file(df)
            
            # Check if it's time to trigger hourly model refinement
            current_time = time.time()
            should_refine = False
            if last_refinement_time is None:
                # First time - trigger refinement to initialize
                should_refine = True
                logging.info(f"\n⏰ First refinement cycle - initializing hourly refinement schedule")
            else:
                time_since_last_refinement = current_time - last_refinement_time
                if time_since_last_refinement >= REFINEMENT_INTERVAL_SECONDS:
                    should_refine = True
                    hours_elapsed = time_since_last_refinement / 3600
                    logging.info(f"\n⏰ Hourly refinement trigger: {hours_elapsed:.2f} hours since last refinement")
            
            # Trigger refinement if needed and not already testing
            if should_refine and not testing_model:
                logging.info("🔄 Triggering scheduled model refinement with recent data...")
                # Note: retrain_with_recent_data will need to be updated to match new training params if they changed significantly
                try:
                    retrain_with_recent_data(client)
                    # Start testing the new model only if refinement succeeded
                    testing_model = True
                    testing_start_time = time.time()
                    last_refinement_time = current_time  # Update last refinement time
                    current_model_predictions = []
                    test_model_predictions = []
                    logging.info(f"🧪 Starting 3-minute testing phase - comparing models...")
                except Exception as e:
                    logging.error(f"❌ Refinement failed due to error: {e}")
                    logging.info("⚠️ Continuing with current model without refinement")
                    last_refinement_time = current_time  # Still update to avoid immediate retry
            elif should_refine and testing_model:
                # Still in testing phase, reschedule for after testing completes
                logging.info(f"⏸️  Refinement scheduled but currently testing model. Will refine after testing completes.")
                last_refinement_time = current_time - (REFINEMENT_INTERVAL_SECONDS - testing_duration)  # Adjust to refine after testing
                
            current_price = df['close'].iloc[-1]
            
            # Check all active trades (real or simulated) for completion or loss
            trades_list = simulated_trades if TEST else active_trades
            trades_to_remove = []
            for i, trade in enumerate(trades_list):
                trade_loss   = check_trade_loss(trade, current_price)
                trade_profit = is_trade_complete(trade, current_price)
                
                if trade_loss:
                    # Use stored trade index if available, otherwise assign new one
                    trade_index = trade.get('index', None)
                    if trade_index is None:
                        trade_counter += 1
                        trade_index = trade_counter
                    pnl = print_trade_result(trade, current_price, result='LOSS', simulated=TEST, trade_index=trade_index)
                    if not TEST:
                        update_stats(daily_stats, 'LOSS', pnl)
                        save_stats(daily_stats)
                    trades_to_remove.append(i)
                    # Trigger retraining on first loss (don't retrain for every loss)
                    if not testing_model:
                        logging.info(f"\n⚠️ Trade resulted in a LOSS. Triggering model retraining with recent 1 hour data.")
                        try:
                            retrain_with_recent_data(client)
                            # Start testing the new model only if refinement succeeded
                            testing_model = True
                            testing_start_time = time.time()
                            last_refinement_time = time.time()  # Update last refinement time
                            current_model_predictions = []
                            test_model_predictions = []
                            logging.info(f"🧪 Starting 3-minute testing phase - comparing models...")
                        except Exception as e:
                            logging.error(f"❌ Refinement failed due to error: {e}")
                            logging.info("⚠️ Continuing with current model without refinement")
                elif trade_profit:
                    # Use stored trade index if available, otherwise assign new one
                    trade_index = trade.get('index', None)
                    if trade_index is None:
                        trade_counter += 1
                        trade_index = trade_counter
                    pnl = print_trade_result(trade, current_price, result='PROFIT', simulated=TEST, trade_index=trade_index)
                    if not TEST:
                        update_stats(daily_stats, 'PROFIT', pnl)
                        save_stats(daily_stats)
                    trades_to_remove.append(i)
            
            # Remove completed trades (filter list to avoid index issues)
            trades_list[:] = [trade for i, trade in enumerate(trades_list) if i not in trades_to_remove]
            
            # Log trade count for monitoring
            if trades_to_remove:
                logging.info(f"🧹 Removed {len(trades_to_remove)} completed trade(s). Remaining: {len(trades_list)}")
            
                # Save current trades to file after checking for completion
            save_current_trades(active_trades, simulated_trades, trade_counter)
            
            # Display active trades summary
            if trades_list:
                mode_label = "🧪 SIMULATED Trades" if TEST else "📊 Active Trades"
                # Update log message to reflect new active trade logging (more compact)
                logging.info(f"\n{mode_label}: {len(trades_list)}")
                # Log active trades to file for dashboard
                log_active_trades_to_file(trades_list, current_price, simulated=TEST)
                
                for trade in trades_list: 
                    pnl_pct, pnl_usdt = calculate_unrealized_pnl(trade, current_price)
                    status_emoji = "🟢" if pnl_pct > 0 else "🔴"
                    trade_index = trade.get('index', '?')  # Use stored trade index
                    logging.info(f"   {status_emoji} #{trade_index}: {trade['side']} @ ${trade['entry_price']:.1f} | "
                               f"PnL: {pnl_pct:+.2f}% (${pnl_usdt:+.2f}) | TP: ${trade['tp']:.1f} | SL: ${trade['sl']:.1f}")
            
            # Prediction logic for current model
            # Calculate technical indicators from raw OHLCV data
            from data_processor import DataProcessor
            temp_processor = DataProcessor(look_back=look_back, future_horizon=future_horizon)
            
            try:
                df_with_indicators = temp_processor.calculate_technical_indicators(df)
            except Exception as e:
                logging.error(f"Failed to calculate indicators: {e}", exc_info=True)
                time.sleep(60)
                continue
            
            # Select the same features used in training
            feature_columns = get_feature_columns()
            
            # Ensure we have enough data after indicator calculation
            if len(df_with_indicators) < look_back:
                logging.warning(f"Not enough data after indicators: {len(df_with_indicators)} < {look_back}. Skipping cycle.")
                time.sleep(60)
                continue
            
            # Take the last look_back rows
            features = df_with_indicators[feature_columns].tail(look_back).values
            
            # Scale features using the saved scaler
            try:
                features_scaled = scaler.transform(features)
            except Exception as e:
                logging.error(f"Failed to scale features: {e}", exc_info=True)
                time.sleep(60)
                continue
            
            # Create tensor with shape (1, look_back, num_features)
            try:
                X_pred = torch.from_numpy(features_scaled).float().unsqueeze(0)
                # Use Monte Carlo dropout with 15 samples for uncertainty estimation
                signal, signal_probs, tp_scaled, sl_scaled, uncertainty = model.predict(X_pred, mc_samples=15)
            except Exception as e:
                logging.error(f"Failed to make prediction: {e}", exc_info=True)
                time.sleep(60)
                continue
            
                                                    # Extract probabilities: signal_probs is softmax output [SELL_prob, BUY_prob]
            signal_map = {0: 'SELL', 1: 'BUY'}      # Define signal map early
            sell_prob  = signal_probs[0][0].item()  # Probability of SELL
            buy_prob   = signal_probs[0][1].item()  # Probability of BUY
            confidence = signal_probs.max().item()  # Maximum probability = confidence
            predicted_signal = signal.item()
            
            # Calculate uncertainty metric (standard deviation of predictions)
            uncertainty_score = uncertainty[0].max().item()  # Max uncertainty across classes
            
            # Diagnostic: Log if uncertainty is suspiciously low (MC dropout may not be working)
            if uncertainty_score < 1e-5:
                logging.warning(f"⚠️ MC Dropout Alert: Very low uncertainty ({uncertainty_score:.2e}). "
                              f"MC samples: 15, Model training mode: {model.training}, "
                              f"use_dropout_inference: {model.use_dropout_inference}, "
                              f"Dropout rate: {model.dropout_rate}")
            
            # Adjust confidence based on uncertainty
            # Higher uncertainty -> lower effective confidence
            effective_confidence = confidence * (1 - uncertainty_score)
            
            # Log detailed probability breakdown for debugging low confidence
            if effective_confidence < 0.75:  # Log when confidence is moderate/low
                logging.debug(f"🔍 Low confidence breakdown: SELL={sell_prob:.3f}, BUY={buy_prob:.3f}, "
                            f"Max (confidence)={confidence:.3f}, Uncertainty={uncertainty_score:.3f}, "
                            f"Effective Conf={effective_confidence:.3f}, Predicted={signal_map[predicted_signal]}")
            
            # Use effective confidence for trading decisions
            confidence = effective_confidence
            
            # Track consecutive confidence=1.0 predictions
            if confidence >= 0.9999:  # Consider 0.9999+ as confidence=1.0
                consecutive_confidence_1_count += 1
                logging.info(f"📊 Consecutive confidence=1.0 count: {consecutive_confidence_1_count}/4")
            else:
                # Reset counter if confidence drops below 1.0
                if consecutive_confidence_1_count > 0:
                    logging.info(f"🔄 Resetting consecutive confidence=1.0 counter (was {consecutive_confidence_1_count})")
                consecutive_confidence_1_count = 0
            
            # Check if we should trigger refinement due to 4 consecutive confidence=1.0
            if consecutive_confidence_1_count >= 4:
                # Check if enough time has passed since last skip (3 minutes)
                current_time = time.time()
                should_trigger_refinement = False
                
                if last_skip_time is None:
                    # First time hitting 4 consecutive - trigger refinement
                    should_trigger_refinement = True
                    logging.info("⚡ First time reaching 4 consecutive confidence=1.0 predictions")
                else:
                    time_since_skip = current_time - last_skip_time
                    if time_since_skip >= 180:  # 3 minutes = 180 seconds
                        should_trigger_refinement = True
                        logging.info(f"⚡ 3 minutes passed since last skip ({time_since_skip:.0f}s) - can trigger refinement again")
                    else:
                        remaining_time = 180 - time_since_skip
                        logging.info(f"⏸️  Skipping refinement - only {time_since_skip:.0f}s passed since last skip "
                                   f"(need 180s, {remaining_time:.0f}s remaining)")
                
                if should_trigger_refinement and not testing_model:
                    logging.info("🔄 Triggering model refinement due to 4 consecutive confidence=1.0 predictions")
                    try:
                        retrain_with_recent_data(client)
                        # Start testing the new model only if refinement succeeded
                        testing_model = True
                        testing_start_time = time.time()
                        last_refinement_time = time.time()
                        last_skip_time = time.time()  # Update skip time
                        consecutive_confidence_1_count = 0  # Reset counter
                        current_model_predictions = []
                        test_model_predictions = []
                        logging.info(f"🧪 Starting 3-minute testing phase - comparing models...")
                        logging.info(f"⏭️  Skipping trading for the next 3 minutes...")
                    except Exception as e:
                        logging.error(f"❌ Refinement failed due to error: {e}")
                        logging.info("⚠️ Continuing with current model without refinement")
                        # Still update skip time to avoid immediate retry
                        last_skip_time = time.time()
                        consecutive_confidence_1_count = 0  # Reset counter
                elif should_trigger_refinement and testing_model:
                    logging.info(f"⏸️  Refinement trigger detected but already testing model. Will skip after testing completes.")
                    last_skip_time = time.time()  # Update skip time even if not triggering
                    consecutive_confidence_1_count = 0  # Reset counter
            
                                                       # Use close_scaler for inverse transform of TP/SL (calculate early for early stop check)
            signal_mismatch        = False             # Default to no mismatch
            tp_based_signal        = predicted_signal  # Default to model signal
            model_predicted_signal = predicted_signal  # Keep original model prediction (set early for exception handling)
            try:
                if close_scaler is not None:
                    tp = close_scaler.inverse_transform(tp_scaled.detach().cpu().numpy().reshape(-1, 1))[0][0]
                    sl = close_scaler.inverse_transform(sl_scaled.detach().cpu().numpy().reshape(-1, 1))[0][0]
                else: 
                    # Fallback: use current price with percentage
                    tp = current_price * 1.01
                    sl = current_price * 0.99
                
                # signal_map already defined above
                # model_predicted_signal already set before try block
                
                # Focus trading on predicted TP: TP > entry → BUY, TP < entry → SELL
                if tp > current_price:
                    # TP is above entry → BUY signal based on TP
                    tp_based_signal = 1  # BUY
                    tp_pct = ((tp - current_price) / current_price) * 100  # Calculate TP percentage
                else:
                    # TP is below entry → SELL signal based on TP
                    tp_based_signal = 0  # SELL
                    tp_pct = ((current_price - tp) / current_price) * 100  # Calculate TP percentage
                
                # Use TP-based signal for trading
                predicted_signal = tp_based_signal
                
                # Check if model signal differs from TP-based signal
                signal_mismatch = (model_predicted_signal != tp_based_signal)
                
                if signal_mismatch:
                    # Model signal differs from TP-based signal - decrease TP percentage
                    logging.warning(f"⚠️ Signal mismatch: Model predicts {signal_map[model_predicted_signal]}, "
                                  f"but TP-based signal is {signal_map[tp_based_signal]} (TP: ${tp:.2f}). "
                                  f"Using TP-based signal with reduced TP %")
                    # Reduce TP percentage by half when signals differ
                    reduced_tp_pct = tp_pct * 0.5
                    
                    if tp_based_signal == 1:  # BUY
                        tp = current_price * (1 + reduced_tp_pct / 100)
                        sl = current_price * 0.80  # 20% below
                    else:  # SELL
                        tp = current_price * (1 - reduced_tp_pct / 100)
                        sl = current_price * 1.20  # 20% above
                    
                    logging.info(f"📊 Using TP-based signal ({signal_map[tp_based_signal]}) with reduced TP: "
                               f"${tp:.2f} ({reduced_tp_pct:.2f}% vs original {tp_pct:.2f}%)")
                else:
                    # Model signal matches TP-based signal - use normal TP/SL values
                    if tp_based_signal == 1:  # BUY
                        # Use model's TP or default to 10% above if too low
                        if tp <= current_price:
                            tp = current_price * 1.10  # 10% above
                        sl = current_price * 0.80  # 20% below
                    else:  # SELL
                        # Use model's TP or default to 10% below if too high
                        if tp >= current_price:
                            tp = current_price * 0.90  # 10% below
                        sl = current_price * 1.20  # 20% above
                    
                    logging.info(f"✅ Model signal matches TP-based signal ({signal_map[tp_based_signal]}) | "
                               f"TP: ${tp:.2f} ({tp_pct:.2f}%) | SL: ${sl:.2f}")
            except Exception as e:
                logging.error(f"Failed to calculate TP/SL: {e}", exc_info=True)
                tp = current_price * 1.01
                sl = current_price * 0.99
                # Default to BUY if TP calculation fails
                predicted_signal = 1 if tp > current_price else 0
                tp_based_signal = predicted_signal  # Update tp_based_signal if TP calc failed
                # model_predicted_signal already set before try block
            
            # Check for early stops on active trades (using TP-based signal with confidence >= 0.7)
            # Early stop triggers if trade has been losing for >5 hours AND TP-based signal (with conf >= 0.7) predicts opposite signal
            if trades_list:
                for i, trade in enumerate(trades_list):
                    # Only use TP-based signal for early stop if confidence is >= 0.7
                    if confidence >= 0.7:
                        early_stop_signal = tp_based_signal
                        signal_type = "tp-based"
                        logging.debug(f"🔍 Early stop check: Using TP-based signal ({tp_based_signal}) with confidence {confidence:.2f} >= 0.7")
                    else:
                        early_stop_signal = model_predicted_signal
                        signal_type = "model"
                        logging.debug(f"🔍 Early stop check: Using model signal ({model_predicted_signal}) with confidence {confidence:.2f} < 0.7")
                    
                    if check_early_stop(trade, current_price, early_stop_signal, signal_type):
                        # Use stored trade index if available, otherwise assign new one
                        trade_index = trade.get('index', None)
                        if trade_index is None:
                            trade_counter += 1
                            trade_index = trade_counter
                        
                        # Calculate actual P&L to determine if it's a profit or loss
                        pnl_pct, pnl_usdt = calculate_unrealized_pnl(trade, current_price)
                        result = 'PROFIT' if pnl_usdt > 0 else 'LOSS'
                        
                        pnl = print_trade_result(trade, current_price, result=result, simulated=TEST, trade_index=trade_index, early_stop=True)
                        if not TEST:
                            update_stats(daily_stats, result, pnl)
                            save_stats(daily_stats)
                        # Mark trade for removal
                        trades_to_remove_early = [i]
                        trades_list[:] = [trade for idx, trade in enumerate(trades_list) if idx not in trades_to_remove_early]
                        logging.info(f"🛑 Trade closed due to early stop condition ({result.lower()}: {pnl_pct:+.2f}%)")
                        # Save current trades after closing
                        save_current_trades(active_trades, simulated_trades, trade_counter)
                        break  # Still break after one early stop per cycle for stability
            
            # Early Profit-Taking: Close profitable trades (>= 0.15%) when TP-based signal is opposite
            # This check happens after early stop check and before risk management
            if trades_list:
                tp_signal_exit_index, tp_signal_exit_trade = check_profitable_trade_tp_signal_exit(
                    trades_list, current_price, tp_based_signal
                )
                if tp_signal_exit_index is not None and tp_signal_exit_trade is not None:
                    # Close the profitable trade
                    trade_index = tp_signal_exit_trade.get('index', None)
                    if trade_index is None:
                        trade_counter += 1
                        trade_index = trade_counter
                    
                    # Determine result based on P&L (should be PROFIT since threshold is >= 0.15%)
                    pnl_pct, pnl_usdt = calculate_unrealized_pnl(tp_signal_exit_trade, current_price)
                    result = 'PROFIT' if pnl_usdt > 0 else 'LOSS'
                    
                    pnl = print_trade_result(tp_signal_exit_trade, current_price, result=result, 
                                            simulated=TEST, trade_index=trade_index, early_stop=False)
                    if not TEST:
                        update_stats(daily_stats, result, pnl)
                        save_stats(daily_stats)
                    
                    trades_list.pop(tp_signal_exit_index)
                    logging.info(f"💰 Early profit-taking: Closed trade with {pnl_pct:+.2f}% profit due to opposite TP-based signal")
                    # Save current trades after closing
                    save_current_trades(active_trades, simulated_trades, trade_counter)
                    # Continue to next cycle after closing trade
                    continue
            
            # Risk Management: Check if profitable trades should be closed when model predicts opposite signal
            # This check happens after TP-based signal is calculated
            if trades_list:
                risk_mgmt_index, risk_mgmt_trade = check_profitable_trade_risk_management(
                    trades_list, current_price, predicted_signal
                )
                if risk_mgmt_index is not None and risk_mgmt_trade is not None:
                    # Close the profitable trade
                    trade_index = risk_mgmt_trade.get('index', None)
                    if trade_index is None:
                        trade_counter += 1
                        trade_index = trade_counter
                    
                    # Determine result based on P&L
                    pnl_pct, pnl_usdt = calculate_unrealized_pnl(risk_mgmt_trade, current_price)
                    result = 'PROFIT' if pnl_usdt > 0 else 'LOSS'
                    
                    pnl = print_trade_result(risk_mgmt_trade, current_price, result=result, 
                                            simulated=TEST, trade_index=trade_index, early_stop=False)
                    if not TEST:
                        update_stats(daily_stats, result, pnl)
                        save_stats(daily_stats)
                    
                    trades_list.pop(risk_mgmt_index)
                    logging.info(f"🛡️ Risk management: Closed profitable trade due to opposite signal prediction")
                    # Save current trades after closing
                    save_current_trades(active_trades, simulated_trades, trade_counter)
                    # Continue to next cycle after closing trade
                    continue
            
                                                                              # Calculate recommended trade amount based on confidence and balance
            try:
                            # Use simulated balance from stats in TEST mode, real balance in live mode
                if TEST:
                    usdt_balance = daily_stats.get('balance', 1000.0)
                else:
                    usdt_balance = exchange.get_usdt_balance()
                # Pass signal alignment info: higher investment when model signal matches TP-based signal
                signal_aligned = not signal_mismatch  # True if signals match
                trade_percentage, trade_margin_usdt, trade_quantity_btc, trade_notional_usdt, trade_leverage = calculate_trade_amount(
                    confidence, usdt_balance, current_price, sl, signal_aligned=signal_aligned
                )
            except Exception as e:
                logging.error(f"Failed to calculate trade amount: {e}", exc_info=True)
                time.sleep(60)
                continue
            
            # Display current model prediction with probability breakdown if confidence is low
            if confidence < 0.75:
                logging.info(f"\n📊 Signal: {signal_map[predicted_signal]} | Confidence: {confidence:.2f} (SELL: {sell_prob:.3f}, BUY: {buy_prob:.3f}) | "
                            f"TP: ${tp:.2f} | SL: ${sl:.2f} | Margin: ${trade_margin_usdt:.2f} ({trade_percentage:.1f}%) | "
                            f"Notional: ${trade_notional_usdt:.2f} (x{trade_leverage})")
            else:
                logging.info(f"\n📊 Signal: {signal_map[predicted_signal]} | Confidence: {confidence:.2f} | "
                            f"TP: ${tp:.2f} | SL: ${sl:.2f} | Margin: ${trade_margin_usdt:.2f} ({trade_percentage:.1f}%) | "
                            f"Notional: ${trade_notional_usdt:.2f} (x{trade_leverage})")

            # Track current model predictions during testing
            if testing_model:
                current_model_predictions.append({
                    'signal': predicted_signal,
                    'confidence': confidence,
                    'entry_price': current_price,
                    'tp': tp,
                    'sl': sl,
                    'timestamp': time.time()
                })
                
                # Also get test model prediction (same input features)
                test_signal, test_signal_probs, test_tp_scaled, test_sl_scaled, test_uncertainty = test_model.predict(X_pred, mc_samples=15)
                test_confidence_raw = test_signal_probs.max().item()
                test_uncertainty_score = test_uncertainty[0].max().item()
                test_confidence = test_confidence_raw * (1 - test_uncertainty_score)  # Adjust for uncertainty
                test_predicted_signal = test_signal.item()
                
                if close_scaler is not None:
                    test_tp = close_scaler.inverse_transform(test_tp_scaled.detach().cpu().numpy().reshape(-1, 1))[0][0]
                    test_sl = close_scaler.inverse_transform(test_sl_scaled.detach().cpu().numpy().reshape(-1, 1))[0][0]
                else:
                    test_tp = current_price * 1.01
                    test_sl = current_price * 0.99
                
                # Focus trading on predicted TP: TP > entry → BUY, TP < entry → SELL
                test_model_predicted_signal = test_predicted_signal  # Keep original model prediction
                
                if test_tp > current_price:
                    # TP is above entry → BUY signal based on TP
                    test_tp_based_signal = 1  # BUY
                    test_tp_pct = ((test_tp - current_price) / current_price) * 100
                else:
                    # TP is below entry → SELL signal based on TP
                    test_tp_based_signal = 0  # SELL
                    test_tp_pct = ((current_price - test_tp) / current_price) * 100
                
                # Use TP-based signal for trading
                test_predicted_signal = test_tp_based_signal
                
                # Check if test model signal differs from TP-based signal
                test_signal_mismatch = (test_model_predicted_signal != test_tp_based_signal)
                
                if test_signal_mismatch:
                    # Model signal differs from TP-based signal - decrease TP percentage
                    # Reduce TP percentage by half when signals differ
                    test_reduced_tp_pct = test_tp_pct * 0.5
                    
                    if test_tp_based_signal == 1:  # BUY
                        test_tp = current_price * (1 + test_reduced_tp_pct / 100)
                        test_sl = current_price * 0.80  # 20% below
                    else:  # SELL
                        test_tp = current_price * (1 - test_reduced_tp_pct / 100)
                        test_sl = current_price * 1.20  # 20% above
                else:
                    # Model signal matches TP-based signal - use normal TP/SL values
                    if test_tp_based_signal == 1:  # BUY
                        # Use model's TP or default to 10% above if too low
                        if test_tp <= current_price:
                            test_tp = current_price * 1.10  # 10% above
                        test_sl = current_price * 0.80  # 20% below
                    else:  # SELL
                        # Use model's TP or default to 10% below if too high
                        if test_tp >= current_price:
                            test_tp = current_price * 0.90  # 10% below
                        test_sl = current_price * 1.20  # 20% above
                
                # Continue with test model comparison after adjustments
                # Pass signal alignment info for test model
                test_signal_aligned = not test_signal_mismatch  # True if signals match
                # Use simulated balance from stats in TEST mode, real balance in live mode
                if TEST:
                    test_usdt_balance = daily_stats.get('balance', 1000.0)
                else:
                    test_usdt_balance = usdt_balance
                test_trade_percentage, test_trade_amount_usdt, test_trade_quantity_btc, test_trade_notional_usdt, test_trade_leverage = calculate_trade_amount(
                    test_confidence, test_usdt_balance, current_price, test_sl, signal_aligned=test_signal_aligned
                )
                
                # Display test model prediction
                logging.info("\n" + "="*60)
                logging.info("🧪 TEST MODEL PREDICTION")
                logging.info("="*60)
                print_prediction(test_predicted_signal, test_confidence, test_tp, test_sl, 
                               test_trade_percentage, test_trade_amount_usdt, test_trade_quantity_btc, test_trade_notional_usdt, test_trade_leverage, test_usdt_balance)
                
                test_model_predictions.append({
                    'signal'     : test_predicted_signal,
                    'confidence' : test_confidence,
                    'entry_price': current_price,
                    'tp'         : test_tp,
                    'sl'         : test_sl,
                    'timestamp'  : time.time()
                })

            # Check if we're in the 3-minute skip period after refinement trigger
            skip_trading_due_to_refinement = False
            if last_skip_time is not None:
                time_since_skip = time.time() - last_skip_time
                if time_since_skip < 180:  # Within 3 minutes of skip
                    skip_trading_due_to_refinement = True
                    remaining_time = 180 - time_since_skip
                    logging.info(f"⏭️  Skipping trading due to recent refinement trigger "
                               f"({time_since_skip:.0f}s elapsed, {remaining_time:.0f}s remaining)")
            
            # Global risk management checks before trading
            can_trade, risk_reason = check_global_risk_limits(daily_stats, trades_list, simulated=TEST)
            
            # Trading and Learning Logic - STRICT THRESHOLDS FOR FUTURES
            if skip_trading_due_to_refinement:
                logging.info("⏸️  In 3-minute skip period - no trading this cycle")
            elif not can_trade:
                logging.info(f"⏸️  Trading blocked by global risk management: {risk_reason}")
            elif testing_model:
                logging.info("\n🧪 Testing mode: Both models running in parallel. Using current model for trading.")
                # Still execute trades with current model during testing if VERY high confidence
                if confidence >= CONFIDENCE_THRESHOLD_TRADE:
                    # Check if we should skip this trade: only open new trade if entry is better OR confidence is higher (>=0.9)
                    should_skip = False
                    
                    # Count trades with confidence=1.0 from trades_log.json
                    confidence_1_count = 0
                    if os.path.exists(TRADES_LOG_FILE):
                        try:
                            with open(TRADES_LOG_FILE, 'r', encoding='utf-8') as f:
                                trades_log = json.load(f)
                                if isinstance(trades_log, list):
                                    confidence_1_count = sum(1 for t in trades_log if t.get('confidence', 0) >= 0.9999)
                        except:
                            pass
                    
                    for trade in trades_list:
                        if trade['signal'] == predicted_signal:  # Same signal type
                            existing_confidence = trade.get('confidence', 0.0)
                            entry_worse = False
                            
                            if predicted_signal == 1:  # BUY - better entry = lower price
                                entry_worse = current_price >= trade['entry_price']
                            else:  # SELL - better entry = higher price
                                entry_worse = current_price <= trade['entry_price']
                            
                            if entry_worse:
                                # Special check: skip if confidence=1.0, entry worse, and already 2+ trades with conf=1.0
                                if confidence >= 0.9999 and entry_worse and confidence_1_count >= 2:
                                    should_skip = True
                                    logging.info(f"⏭️  Skipping {signal_map[predicted_signal]} trade: Confidence=1.0, entry worse (${current_price:.2f} vs ${trade['entry_price']:.2f}), "
                                               f"and already {confidence_1_count} trades with confidence=1.0")
                                    break
                                # Entry is worse, but check if confidence is high enough to override
                                if confidence >= 0.9 and confidence > existing_confidence:
                                    # High confidence and higher than existing - allow trade
                                    logging.info(f"✅ Overriding worse entry: Confidence {confidence:.2f} >= 0.9 and higher than existing {existing_confidence:.2f}")
                                    should_skip = False
                                    break
                                else:
                                    # Entry worse and not enough confidence to override
                                    should_skip = True
                                    logging.info(f"⏭️  Skipping {signal_map[predicted_signal]} trade: Existing trade at ${trade['entry_price']:.2f} "
                                               f"(conf: {existing_confidence:.2f}), new entry would be ${current_price:.2f} "
                                               f"(conf: {confidence:.2f}) - entry not better and confidence not high enough")
                                    break
                    
                    if not should_skip:
                        # Assign trade index when creating the trade
                        trade_counter += 1
                        trade_index = trade_counter
                        logging.info(f"✅ VERY HIGH CONFIDENCE ({confidence:.2f}) - Executing trade")
                        if TEST:
                            new_trade = simulate_trade(
                                predicted_signal, trade_quantity_btc, current_price, tp, sl,
                                trade_margin_usdt, trade_notional_usdt, trade_leverage, confidence=confidence
                            )
                            new_trade['index'] = trade_index  # Store index in trade
                            simulated_trades.append(new_trade)
                        else:
                            new_trade = execute_trade(
                                predicted_signal, exchange, trade_quantity_btc, current_price, tp, sl,
                                trade_margin_usdt, trade_notional_usdt, trade_leverage, confidence=confidence
                            )
                            new_trade['index'] = trade_index  # Store index in trade
                            active_trades.append(new_trade)
                        # Save current trades after opening new trade
                        save_current_trades(active_trades, simulated_trades, trade_counter)
                else:
                    logging.info(f"⏸️  Confidence {confidence:.2f} below trading threshold ({CONFIDENCE_THRESHOLD_TRADE}). Waiting for better signal.")
            elif not skip_trading_due_to_refinement and can_trade and confidence >= CONFIDENCE_THRESHOLD_TRADE:
                # ONLY trade with high confidence (>70%) for futures
                # Check if we should skip this trade: only open new trade if entry is better OR confidence is higher (>=0.9)
                should_skip = False
                
                # Count trades with confidence=1.0 from trades_log.json
                confidence_1_count = 0
                if os.path.exists(TRADES_LOG_FILE):
                    try:
                        with open(TRADES_LOG_FILE, 'r', encoding='utf-8') as f:
                            trades_log = json.load(f)
                            if isinstance(trades_log, list):
                                confidence_1_count = sum(1 for t in trades_log if t.get('confidence', 0) >= 0.9999)
                    except:
                        pass
                
                for trade in trades_list:
                    if trade['signal'] == predicted_signal:  # Same signal type
                        existing_confidence = trade.get('confidence', 0.0)
                        entry_worse = False
                        
                        if predicted_signal == 1:  # BUY - better entry = lower price
                            entry_worse = current_price >= trade['entry_price']
                        else:  # SELL - better entry = higher price
                            entry_worse = current_price <= trade['entry_price']
                        
                        if entry_worse:
                            # Special check: skip if confidence=1.0, entry worse, and already 2+ trades with conf=1.0
                            if confidence >= 0.9999 and entry_worse and confidence_1_count >= 2:
                                should_skip = True
                                logging.info(f"⏭️  Skipping {signal_map[predicted_signal]} trade: Confidence=1.0, entry worse (${current_price:.2f} vs ${trade['entry_price']:.2f}), "
                                           f"and already {confidence_1_count} trades with confidence=1.0")
                                break
                            # Entry is worse, but check if confidence is high enough to override
                            if confidence >= 0.9 and confidence >= existing_confidence:
                                # High confidence and higher than existing - allow trade
                                logging.info(f"✅ Overriding worse entry: Confidence {confidence:.2f} >= 0.9 and higher than existing {existing_confidence:.2f}")
                                should_skip = False
                                break
                            else:
                                # Entry worse and not enough confidence to override
                                should_skip = True
                                logging.info(f"⏭️  Skipping {signal_map[predicted_signal]} trade: Existing trade at ${trade['entry_price']:.2f} "
                                           f"(conf: {existing_confidence:.2f}), new entry would be ${current_price:.2f} "
                                           f"(conf: {confidence:.2f}) - entry not better and confidence not high enough")
                                break
                
                if not should_skip:
                    # Assign trade index when creating the trade
                    trade_counter += 1
                    trade_index = trade_counter
                    logging.info(f"✅ VERY HIGH CONFIDENCE ({confidence:.2f}) - Executing trade")
                    if TEST:
                        new_trade = simulate_trade(
                            predicted_signal, trade_quantity_btc, current_price, tp, sl,
                            trade_margin_usdt, trade_notional_usdt, trade_leverage, confidence=confidence
                        )
                        new_trade['index'] = trade_index  # Store index in trade
                        simulated_trades.append(new_trade)
                    else:
                        new_trade = execute_trade(
                            predicted_signal, exchange, trade_quantity_btc, current_price, tp, sl,
                            trade_margin_usdt, trade_notional_usdt, trade_leverage, confidence=confidence
                        )
                        new_trade['index'] = trade_index  # Store index in trade
                        active_trades.append(new_trade)
                    # Save current trades after opening new trade
                    save_current_trades(active_trades, simulated_trades, trade_counter)
            elif not skip_trading_due_to_refinement and can_trade and confidence < CONFIDENCE_THRESHOLD_TEST:
                # Trigger model testing and fine-tuning when confidence is below threshold
                logging.info(f"\n⚠️ Confidence {confidence:.2f} below threshold ({CONFIDENCE_THRESHOLD_TEST})")
                logging.info("Triggering model fine-tuning with recent data...")
                if not testing_model:  # Don't start new test if already testing
                    try:
                        retrain_with_recent_data(client)
                        # Start testing the new model only if refinement succeeded
                        testing_model = True
                        testing_start_time = time.time()
                        last_refinement_time = time.time()  # Update last refinement time
                        current_model_predictions = []
                        test_model_predictions = []
                        logging.info(f"🧪 Starting 3-minute testing phase - comparing models...")
                    except Exception as e:
                        logging.error(f"❌ Refinement failed due to error: {e}")
                        logging.info("⚠️ Continuing with current model without refinement")
            else:
                # Confidence between thresholds - wait for better signal
                logging.info(f"⏸️  Confidence {confidence:.2f} is moderate. Waiting for higher confidence (>{CONFIDENCE_THRESHOLD_TRADE:.0%}) to trade.")

            # Save stats at the end of each cycle
            if not TEST:
                save_stats(daily_stats)
            
            # Save current trades at the end of each cycle to ensure persistence
            # (even if no trades changed, this updates the last_updated timestamp)
            save_current_trades(active_trades, simulated_trades, trade_counter)

            logging.info("Waiting for the next trading interval...")
            time.sleep(60)

        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            time.sleep(60)

def determine_leverage(confidence):
    """
    Determine leverage based on confidence - CONSERVATIVE approach
    Uses moderate leverage (5-15x) to balance profit potential with risk management
    """
    if confidence >= 0.98:
        return min(15, MAX_LEVERAGE)  # Max 15x for perfect confidence
    elif confidence >= 0.90:
        return min(12, MAX_LEVERAGE)  # 12x for very high confidence
    elif confidence >= 0.85:
        return min(10, MAX_LEVERAGE)  # 10x
    elif confidence >= 0.80:
        return min(8, MAX_LEVERAGE)   # 8x
    elif confidence >= 0.75:
        return min(6, MAX_LEVERAGE)   # 6x
    else:
        return min(5, MAX_LEVERAGE)   # Min 5x leverage

def calculate_trade_amount(confidence, balance, current_price, stop_loss, signal_aligned=True):
    """
    Calculate recommended trade amount (margin) based on confidence level and signal alignment.
    AGGRESSIVE position sizing for higher profit potential in leveraged trading.
    
    Base position sizing by confidence (% of balance as margin):
    - 1.0: 7% of balance (AGGRESSIVE)
    - 0.95-0.999: 6% of balance
    - 0.90-0.95: 5% of balance
    - 0.85-0.90: 4% of balance
    - 0.80-0.85: 3% of balance
    - 0.75-0.80: 2.5% of balance
    - 0.70-0.75: 2% of balance
    
    When signals are aligned (model signal matches TP-based signal):
    - Position size is increased by up to 50% based on confidence
    
    When signals are mismatched:
    - Position size is reduced by 40%
    
    Args:
        confidence: Model confidence (0.0 to 1.0)
        balance: Available USDT balance
        current_price: Current BTC price
        stop_loss: Stop loss price (for reference, not used in calculation)
        signal_aligned: True if model signal matches TP-based signal, False otherwise
    
    Returns:
        trade_percentage   : Percentage of balance to allocate as margin
        margin_usdt        : Margin placed on the trade
        trade_quantity_btc : BTC quantity controlled after applying leverage
        notional_usdt      : Effective notional size (margin * leverage)
        leverage_used      : Leverage multiplier applied to this trade
    """
    # AGGRESSIVE base position sizing (% of balance as margin)
    if confidence >= 1.0:
        base_percentage = 7.0    # Max 7% of balance for perfect confidence (AGGRESSIVE)
    elif confidence >= 0.95:
        base_percentage = 6.0    # 6% for very high confidence
    elif confidence >= 0.90:
        base_percentage = 5.0    # 5% of balance
    elif confidence >= 0.85:
        base_percentage = 4.0    # 4% of balance
    elif confidence >= 0.80:
        base_percentage = 3.0    # 3% of balance
    elif confidence >= 0.75:
        base_percentage = 2.5    # 2.5% of balance
    else:  # 0.70 - 0.75
        base_percentage = 2.0    # Min 2% of balance
    
    # Adjust position size based on signal alignment
    if signal_aligned:
        # Signals match - increase position size by up to 50%
        alignment_multiplier = 1.0 + (confidence * 0.5)  # 1.0 to 1.5 multiplier
        trade_percentage = base_percentage * alignment_multiplier
        # Cap at 10% of balance for safety
        if trade_percentage > 10.0:
            trade_percentage = 10.0
    else:
        # Signals mismatch - reduce position size by 40%
        trade_percentage = base_percentage * 0.6
    
    # Calculate margin in USDT
    margin_usdt = balance * (trade_percentage / 100)
    
    # Minimum trade size
    min_trade_amount = 10.0  # Minimum $10 USDT
    if margin_usdt < min_trade_amount:
        margin_usdt = min_trade_amount
        trade_percentage = (margin_usdt / balance * 100) if balance > 0 else 0
    
    # Use CONSERVATIVE leverage (much lower than before)
    leverage_used = determine_leverage(confidence)
    notional_usdt = margin_usdt * leverage_used
    
    # Additional safety check: Cap notional at 50% of balance to allow aggressive trading
    max_notional = balance * MAX_TOTAL_EXPOSURE_PCT  # Max 30% of balance as notional exposure
    if notional_usdt > max_notional:
        notional_usdt = max_notional
        leverage_used = notional_usdt / margin_usdt if margin_usdt > 0 else 1
        logging.warning(f"⚠️ Capping position: Reduced notional from ${margin_usdt * determine_leverage(confidence):.2f} to ${notional_usdt:.2f} (leverage adjusted to {leverage_used:.1f}x)")
    
    # Calculate quantity in BTC
    trade_quantity_btc = notional_usdt / current_price
    
    # Calculate actual risk based on stop loss distance
    sl_distance_pct = abs(current_price - stop_loss) / current_price * 100
    actual_risk_usdt = margin_usdt * leverage_used * (sl_distance_pct / 100)
    
    # Log risk management info
    alignment_status = "ALIGNED" if signal_aligned else "MISMATCHED"
    logging.debug(
        f"Position sizing: Conf={confidence:.2f}, Signals={alignment_status}, Margin={trade_percentage:.1f}%, "
        f"Margin=${margin_usdt:.2f}, Notional=${notional_usdt:.2f} (x{leverage_used}), SL_dist={sl_distance_pct:.2f}%, Risk=${actual_risk_usdt:.2f}"
    )
    
    return trade_percentage, margin_usdt, trade_quantity_btc, notional_usdt, leverage_used

def check_global_risk_limits(daily_stats, active_trades, simulated=False):
    """
    Global risk management checks before opening new trades.
    Prevents excessive risk exposure across multiple dimensions.
    
    Args:
        daily_stats: Current day's trading statistics
        active_trades: List of currently active trades
        simulated: Whether this is a simulated (TEST mode) check
    
    Returns:
        tuple: (bool: can_trade, str: reason) - True if safe to trade, False with reason if not
    """
    trade_mode = "SIMULATED" if simulated else "LIVE"
    
    # CHECK 1: Daily loss limit
    daily_pnl = daily_stats.get('total_profit_usdt', 0)
    if daily_pnl < -MAX_DAILY_LOSS:
        reason = f"Daily loss limit reached (${daily_pnl:.2f} < -${MAX_DAILY_LOSS:.2f})"
        logging.warning(f"🛑 [{trade_mode}] Global Risk: {reason}")
        return False, reason
    
    # CHECK 2: Maximum concurrent positions
    if len(active_trades) >= MAX_CONCURRENT_TRADES:
        reason = f"Maximum concurrent positions reached ({len(active_trades)}/{MAX_CONCURRENT_TRADES})"
        logging.warning(f"🛑 [{trade_mode}] Global Risk: {reason}")
        # Add detailed trade count logging for debugging
        trade_summary = [f"#{t.get('index', '?')}:{t.get('side', '?')}@${t.get('entry_price', 0):.0f}" for t in active_trades]
        logging.info(f"   Active trades breakdown: {trade_summary}")
        return False, reason
    
    # CHECK 3: Total exposure limit
    total_notional = sum(t.get('notional_usdt', 0) for t in active_trades)
    current_balance = daily_stats.get('balance', 1000.0) if simulated else None
    
    if current_balance is not None:  # Only check in TEST mode where we track balance
        max_exposure = current_balance * MAX_TOTAL_EXPOSURE_PCT
        if total_notional > max_exposure:
            reason = f"Total exposure limit reached (${total_notional:.2f} > ${max_exposure:.2f})"
            logging.warning(f"🛑 [{trade_mode}] Global Risk: {reason}")
            return False, reason
    
    # CHECK 4: Consecutive losses protection (optional enhancement)
    failed_trades = daily_stats.get('failed_trades', 0)
    successful_trades = daily_stats.get('successful_trades', 0)
    
    # If we have 3+ consecutive losses (no wins today and 3+ losses), reduce trading
    if failed_trades >= 3 and successful_trades == 0:
        reason = f"Consecutive losses detected ({failed_trades} losses, 0 wins today) - caution advised"
        logging.warning(f"⚠️ [{trade_mode}] Global Risk: {reason}")
        # This is a warning, not a hard stop - return True but log the concern
    
    # All checks passed
    logging.debug(f"✅ [{trade_mode}] Global risk checks passed (PnL: ${daily_pnl:.2f}, Positions: {len(active_trades)}/{MAX_CONCURRENT_TRADES}, Exposure: ${total_notional:.2f})")
    return True, "All risk checks passed"

def print_prediction(signal, confidence, tp, sl, trade_percentage, margin_usdt, trade_quantity_btc, notional_usdt, leverage, balance):
    signal_map = {0: 'SELL', 1: 'BUY'}
    logging.info(f"\n{'='*60}")
    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] PREDICTION")
    logging.info(f"{'='*60}")
    logging.info(f"Signal: {signal_map[signal]} | Confidence: {confidence:.2f}")
    logging.info(f"Take Profit: ${tp:.2f} | Stop Loss: ${sl:.2f}")
    logging.info(f"\n--- TRADE RECOMMENDATION ---")
    logging.info(f"Current USDT Balance: ${balance:.2f}")
    logging.info(f"Recommended Trade Size: {trade_percentage:.1f}% of balance")
    logging.info(f"Margin: ${margin_usdt:.2f} USDT ({trade_percentage:.1f}% of balance)")
    logging.info(f"Notional (x{leverage:.0f}): ${notional_usdt:.2f} | Size: {trade_quantity_btc:.6f} BTC")
    logging.info(f"{'='*60}\n")

def simulate_trade(signal, quantity, entry_price, tp, sl, margin_usdt, notional_usdt, leverage, confidence=None):
    """Simulate a trade without actually executing it (for TEST mode)"""
    side = {0: 'SELL', 1: 'BUY'}[signal]
    # Don't log simulated message - already shown at startup
    
    # Calculate trade amounts
    trade_amount_usdt = margin_usdt  # Store margin for backward compatibility
    
    # Calculate expected profit/loss amounts
    if signal == 1:  # BUY
        expected_profit_usdt = quantity * (tp - entry_price)
        expected_loss_usdt = quantity * (entry_price - sl)
    else:  # SELL
        expected_profit_usdt = quantity * (entry_price - tp)
        expected_loss_usdt = quantity * (sl - entry_price)
    
    # Return simulated trade info for tracking
    trade_info = {
        'signal': signal,
        'entry_price': entry_price,
        'tp': tp,
        'sl': sl,
        'side': side,
        'entry_time': time.time(),
        'quantity': quantity,
        'trade_amount_usdt': trade_amount_usdt,
        'margin_usdt': margin_usdt,
        'notional_usdt': notional_usdt,
        'leverage': leverage,
        'expected_profit_usdt': expected_profit_usdt,
        'expected_loss_usdt': expected_loss_usdt
    }
    # Store confidence if provided
    if confidence is not None:
        trade_info['confidence'] = confidence
    return trade_info

def log_trade_to_file(trade_index, trade, exit_price, result, actual_pnl_usdt, pnl_percentage, price_change_pct, duration_minutes, simulated=False, early_stop=False):
    """Log trade details to a separate JSON file"""
    try:
        # Guard required fields to ensure re-loadable structure
        required_fields = ['entry_price', 'entry_time', 'tp', 'sl', 'signal']
        missing = [f for f in required_fields if f not in trade or trade[f] is None]
        if missing:
            logging.warning(f"⚠️ Cannot log trade #{trade_index}: missing fields {missing}")
            return
        
        entry_time_dt = datetime.fromtimestamp(trade['entry_time'])
        
        # Load existing trades if file exists and is valid
        trades_log = []
        if os.path.exists(TRADES_LOG_FILE):
            try:
                with open(TRADES_LOG_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and content != 'w':  # Handle corrupted file
                        trades_log = json.loads(content)
                    else:
                        logging.warning(f"⚠️ {TRADES_LOG_FILE} appears corrupted. Starting fresh.")
                        trades_log = []
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"⚠️ Could not parse {TRADES_LOG_FILE}: {e}. Starting fresh.")
                trades_log = []
        
        # Ensure trades_log is a list
        if not isinstance(trades_log, list):
            logging.warning(f"⚠️ {TRADES_LOG_FILE} format invalid. Starting fresh.")
            trades_log = []
        
        # Remove any ACTIVE entry for this trade first (using old or new field names)
        trades_log = [t for t in trades_log if 
                     not (t.get('status') == 'ACTIVE' and 
                          (abs(t.get('entry_price', 0) - float(trade['entry_price'])) < 0.01 or
                           abs(t.get('entry price', 0) - float(trade['entry_price'])) < 0.01))]
        
        # Create trade log entry with simplified structure
        signal_map = {0: 'SELL', 1: 'BUY'}
        signal_value = trade.get('signal', -1)
        if signal_value is None:
            signal_value = -1
        trade_entry = {
            'index'        : trade_index,
            'time_stamp'   : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Profit/loss'  : result,                                                               # 'PROFIT' or 'LOSS'
            'PL percentage': float(pnl_percentage),
            'PL in $'      : float(actual_pnl_usdt),
            'entry price'  : float(trade['entry_price']),
            'entry_time'   : entry_time_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'tp'           : float(trade['tp']),
            'sl'           : float(trade['sl']),
            'leverage'     : float(trade.get('leverage', 1)),
            'margin'       : float(trade.get('margin_usdt', trade.get('trade_amount_usdt', 0))),
            'signal'       : signal_map.get(signal_value, 'UNKNOWN'),                              # BUY or SELL
            'signal_value' : int(signal_value),                                                    # 0 for SELL, 1 for BUY
            'early_stop'   : bool(early_stop)                                                      # True if trade was closed due to early stop
        }
        
        # Add confidence if available
        if 'confidence' in trade and trade['confidence'] is not None:
            trade_entry['confidence'] = float(trade['confidence'])
        
        # Append to log
        trades_log.append(trade_entry)
        
        # Save back to file with proper formatting
        with open(TRADES_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(trades_log, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📝 Trade #{trade_index} logged to {TRADES_LOG_FILE}")
    except Exception as e:
        logging.error(f"Error logging trade to file: {e}", exc_info=True)

def log_active_trades_to_file(trades_list, current_price, simulated=False):
    """Log active trades with unrealized P&L to JSON file"""
    try:
        # Load existing trades if file exists and is valid
        trades_log = []
        if os.path.exists(TRADES_LOG_FILE):
            try:
                with open(TRADES_LOG_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and content != 'w':
                        trades_log = json.loads(content)
                    else:
                        trades_log = []
            except (json.JSONDecodeError, ValueError):
                trades_log = []
        
        if not isinstance(trades_log, list):
            trades_log = []
        
        # Update or add active trades
        for i, trade in enumerate(trades_list, 1):
            # Guard required fields to ensure re-loadable structure
            required_fields = ['entry_price', 'entry_time', 'tp', 'sl', 'signal']
            missing = [f for f in required_fields if f not in trade or trade[f] is None]
            if missing:
                logging.warning(f"⚠️ Skipping active trade log (missing fields {missing})")
                continue

            pnl_pct, unrealized_pnl_usdt = calculate_unrealized_pnl(trade, current_price)
            entry_time_str = datetime.fromtimestamp(trade['entry_time']).strftime('%Y-%m-%d %H:%M:%S')
            duration_minutes = int((time.time() - trade['entry_time']) / 60)
            
            # Check if this trade already exists in log (by timestamp and entry_price)
            trade_exists = False
            for existing_trade in trades_log:
                if (existing_trade.get('entry_price') == float(trade['entry_price']) and
                    existing_trade.get('timestamp') == entry_time_str and
                    existing_trade.get('status') == 'ACTIVE'):
                    # Update existing active trade
                    existing_trade['unrealized_pnl_usdt'] = float(unrealized_pnl_usdt)
                    existing_trade['unrealized_pnl_percentage'] = float(pnl_pct)
                    existing_trade['current_price'] = float(current_price)
                    existing_trade['duration_minutes'] = duration_minutes
                    existing_trade['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    trade_exists = True
                    break
            
            if not trade_exists:
                # Add new active trade
                active_trade_entry = {
                    'trade_index': f"ACTIVE_{i}",
                    'timestamp': entry_time_str,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'ACTIVE',
                    'simulated': simulated,
                    'side': trade['side'],
                    'entry_price': float(trade['entry_price']),
                    'current_price': float(current_price),
                    'take_profit': float(trade['tp']),
                    'stop_loss': float(trade['sl']),
                    'entry_time': entry_time_str,
                    'tp': float(trade['tp']),
                    'sl': float(trade['sl']),
                    'quantity_btc': float(trade['quantity']),
                    'trade_amount_usdt': float(trade['trade_amount_usdt']),
                    'margin_usdt': float(trade.get('margin_usdt', trade.get('trade_amount_usdt', 0))),
                    'notional_usdt': float(trade.get('notional_usdt', 0)),
                    'leverage': float(trade.get('leverage', 1)),
                    'unrealized_pnl_usdt': float(unrealized_pnl_usdt),
                    'unrealized_pnl_percentage': float(pnl_pct),
                    'duration_minutes': duration_minutes,
                    'expected_profit_usdt': float(trade.get('expected_profit_usdt', 0)),
                    'expected_loss_usdt': float(trade.get('expected_loss_usdt', 0))
                }
                trades_log.append(active_trade_entry)
        
        # Remove old ACTIVE entries that are no longer in trades_list
        # (they were closed but we keep them as completed trades)
        active_entry_prices = {float(t['entry_price']) for t in trades_list}
        trades_log = [t for t in trades_log if 
                     not (t.get('status') == 'ACTIVE' and 
                          t.get('entry_price') not in active_entry_prices)]
        
                      # Save back to file
        with open(TRADES_LOG_FILE, 'w', encoding='utf-8') as f: 
            json.dump(trades_log, f, indent=2, ensure_ascii=False)
            
    except Exception as e: 
        logging.error(f"Error logging active trades to file: {e}", exc_info=True)

def print_trade_result(trade, exit_price, result, simulated=False, trade_index=None, early_stop=False):
    """Print detailed trade result summary"""
    duration = time.time() - trade['entry_time']
    duration_minutes = int(duration / 60)
    
    # Calculate actual profit/loss
    leverage = trade.get('leverage', 1)
    margin_used = trade.get('margin_usdt', trade.get('trade_amount_usdt', 1))
    if trade['signal'] == 1:  # BUY trade
        actual_pnl_usdt = trade['quantity'] * (exit_price - trade['entry_price'])
        price_change_pct = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
    else:  # SELL trade
        actual_pnl_usdt = trade['quantity'] * (trade['entry_price'] - exit_price)
        price_change_pct = ((trade['entry_price'] - exit_price) / trade['entry_price']) * 100
    
    pnl_percentage = (actual_pnl_usdt / margin_used) * 100 if margin_used else 0
    
    # Log to separate file if trade_index is provided
    if trade_index is not None:
        log_trade_to_file(trade_index, trade, exit_price, result, actual_pnl_usdt, pnl_percentage, price_change_pct, duration_minutes, simulated, early_stop)
    
    # Result emoji and color
    if result == 'PROFIT':
       emoji        = "✅"
       result_text  = "PROFIT (TP REACHED)"
    elif early_stop:
        emoji       = "🛑"
        result_text = "LOSS (EARLY STOP)"
    else:
        emoji       = "❌"
        result_text = "LOSS (SL HIT)"
    
    # Simplified logging - show only essential info (no "simulated" marker - shown at startup)
    if actual_pnl_usdt > 0:
        pnl_sign = "+"
        pnl_emoji = "💰"
    else:
        pnl_sign = ""
        pnl_emoji = "📉"
    
    logging.info(f"\n{emoji} Trade #{trade_index if trade_index else '?'} {result_text} | "
                 f"{trade['side']} | Entry: ${trade['entry_price']:.2f} → Exit: ${exit_price:.2f} | "
                 f"Margin: ${margin_used:.2f} | Lev: x{leverage:.0f} | "
                 f"Expected: ${trade['expected_profit_usdt']:.2f} | "
                 f"Actual: {pnl_sign}${actual_pnl_usdt:.2f} ({pnl_percentage:+.2f}% on margin) | "
                 f"Duration: {duration_minutes}m")
    
    return actual_pnl_usdt

def execute_trade(signal, exchange, quantity, entry_price, tp, sl, margin_usdt, notional_usdt, leverage, confidence=None):
    """Execute trade and return trade info for tracking"""
    side = {0: 'SELL', 1: 'BUY'}[signal]
    logging.info(f"\n💰 Confidence >= 0.70. Placing {side} order.")
    # Note: For futures, you may need to use a different order placement method
    # This will depend on your Exchange class implementation for futures
    exchange.place_order('BTC_USDT', 'MARKET', side, quantity)

    # Calculate trade amounts
    trade_amount_usdt = margin_usdt
    
    # Calculate expected profit/loss amounts
    if signal               == 1:                             # BUY
       expected_profit_usdt  = quantity * (tp - entry_price)
       expected_loss_usdt    = quantity * (entry_price - sl)
    else:  # SELL
        expected_profit_usdt = quantity * (entry_price - tp)
        expected_loss_usdt = quantity * (sl - entry_price)
    
    # Return trade info for tracking
    trade_info = {
        'signal'              : signal,
        'entry_price'         : entry_price,
        'tp'                  : tp,
        'sl'                  : sl,
        'side'                : side,
        'entry_time'          : time.time(),
        'quantity'            : quantity,
        'trade_amount_usdt'   : trade_amount_usdt,
        'margin_usdt'         : margin_usdt,
        'notional_usdt'       : notional_usdt,
        'leverage'            : leverage,
        'expected_profit_usdt': expected_profit_usdt,
        'expected_loss_usdt'  : expected_loss_usdt
    }
    # Store confidence if provided
    if confidence is not None:
        trade_info['confidence'] = confidence
    return trade_info

def calculate_unrealized_pnl(trade, current_price):
    """Calculate unrealized P&L percentage and dollar amount for an active trade
    
    For leveraged futures trading:
    - quantity stored is the full leveraged position (notional_usdt / entry_price)
    - PnL in USD = quantity * price_change
    - PnL % = (PnL in USD / margin) * 100
    """
    if trade['signal'] == 1:  # BUY trade
        pnl_usdt = trade['quantity'] * (current_price - trade['entry_price'])
    else:  # SELL trade
        pnl_usdt = trade['quantity'] * (trade['entry_price'] - current_price)
    
    # Calculate PnL percentage relative to margin invested (not price change %)
    margin_used = trade.get('margin_usdt', trade.get('trade_amount_usdt', 1))
    pnl_pct = (pnl_usdt / margin_used) * 100 if margin_used > 0 else 0
    
    return pnl_pct, pnl_usdt

def check_trade_loss(trade, current_price):
    """Check if the active trade hit stop loss (resulted in a loss)"""
    if trade['signal'] == 1:  # BUY trade
        # Loss if price went below stop loss
        if current_price <= trade['sl']:
            logging.info(f"BUY trade hit SL: Entry={trade['entry_price']:.2f}, SL={trade['sl']:.2f}, Current={current_price:.2f}")
            return True
    else:  # SELL trade
        # Loss if price went above stop loss
        if current_price >= trade['sl']:
            logging.info(f"SELL trade hit SL: Entry={trade['entry_price']:.2f}, SL={trade['sl']:.2f}, Current={current_price:.2f}")
            return True
    return False

def is_trade_complete(trade, current_price):
    """Check if the trade reached take profit (successful completion)"""
    if trade['signal'] == 1:  # BUY trade
        return current_price >= trade['tp']
    else:   # SELL trade
        return current_price <= trade['tp']

def check_early_stop(trade, current_price, signal_for_early_stop=None, signal_type="model"):
    """
    SIMPLIFIED early stop logic - checks three independent conditions:
    1. Hard loss limit: Close immediately if loss >= $5 (STANDARDIZED, tight risk control)
    2. Time-based exit: Close if trade has been losing for > 5 hours
    3. Signal reversal: Close if opposite signal detected with minimum loss >= $3
    
    Each condition is independent and can trigger trade closure.
    
    Args:
        trade: The active trade dictionary
        current_price: Current market price
        signal_for_early_stop: Signal to use for early stop decision (0=SELL, 1=BUY)
        signal_type: Type of signal being used ("model" or "tp-based")
    
    Returns:
        bool: True if trade should be closed, False otherwise
    """
    pnl_pct, pnl_usdt = calculate_unrealized_pnl(trade, current_price)
    duration_minutes = (time.time() - trade['entry_time']) / 60
    
    # CONDITION 1: Hard loss limit - IMMEDIATE CLOSE (STANDARDIZED to $5 for tight risk control)
    if pnl_usdt <= -EARLY_STOP_LOSS_THRESHOLD:
        logging.warning(f"🛑 Hard Stop: Loss threshold reached (${pnl_usdt:.2f} <= -${EARLY_STOP_LOSS_THRESHOLD:.2f})")
        return True
    
    # Skip remaining checks if trade is profitable
    if pnl_pct >= 0:
        return False
    
    # CONDITION 2: Time-based exit for losing trades (> 5 hours)
    if duration_minutes >= EARLY_STOP_MAX_TIME_MINUTES:
        logging.warning(f"🛑 Time Stop: Trade losing for {duration_minutes/60:.1f} hours (${pnl_usdt:.2f}, {pnl_pct:.2f}%)")
        return True
    
    # CONDITION 3: Signal reversal with minimum loss threshold (STANDARDIZED to $3)
    if EARLY_STOP_OPPOSITE_SIGNAL and signal_for_early_stop is not None:
        # Check if signal is opposite to current trade direction
        is_opposite = (
            (trade['signal'] == 1 and signal_for_early_stop == 0) or  # BUY trade, SELL signal
            (trade['signal'] == 0 and signal_for_early_stop == 1)     # SELL trade, BUY signal
        )
        
        if is_opposite and pnl_usdt <= -MIN_LOSS_FOR_OPPOSITE_SIGNAL:
            logging.warning(f"🛑 Signal Reversal: {signal_type.upper()} signal opposite with loss (${pnl_usdt:.2f} <= -${MIN_LOSS_FOR_OPPOSITE_SIGNAL:.2f})")
            return True
    
    # No stop conditions met
    return False

def check_profitable_trade_tp_signal_exit(trades_list, current_price, tp_based_signal):
    """
    Early profit-taking: Close profitable trades when TP-based signal is opposite.
    
    Rules:
    - If TP-based signal is opposite to trade direction
    - AND trade has profit PnL >= 0.15%
    - Then close the trade to lock in profits
    
    Args:
        trades_list: List of active trades
        current_price: Current market price
        tp_based_signal: TP-based signal (0=SELL, 1=BUY)
    
    Returns:
        tuple: (trade_to_close_index, trade_to_close) or (None, None) if no trade should be closed
    """
    if not trades_list or tp_based_signal is None:
        return None, None
    
    signal_map = {0: 'SELL', 1: 'BUY'}
    PROFIT_THRESHOLD_PCT = 0.15  # Close trades with >= 0.15% profit
    
    # Check all trades for opposite TP-based signal and sufficient profit
    for i, trade in enumerate(trades_list):
        # Check if TP-based signal is opposite to trade direction
        is_opposite = False
        if trade['signal'] == 1 and tp_based_signal == 0:  # BUY trade, TP-based signal is SELL
            is_opposite = True
        elif trade['signal'] == 0 and tp_based_signal == 1:  # SELL trade, TP-based signal is BUY
            is_opposite = True
        
        if is_opposite:
            # Calculate PnL
            pnl_pct, pnl_usdt = calculate_unrealized_pnl(trade, current_price)
            
            # Check if trade is profitable and meets threshold
            if pnl_pct >= PROFIT_THRESHOLD_PCT:
                logging.warning(f"💰 Early Profit-Taking: Closing {signal_map[trade['signal']]} trade "
                              f"(entry: ${trade['entry_price']:.2f}) with profit {pnl_pct:+.2f}% "
                              f"(${pnl_usdt:+.2f}) because TP-based signal is opposite ({signal_map[tp_based_signal]})")
                return i, trade
    
    return None, None

def check_profitable_trade_risk_management(trades_list, current_price, predicted_signal):
    """
    Risk management: Close profitable trades when model predicts opposite signal.
    
    Rules:
    - For BUY trades: Close the highest entry price (worst entry) if it's profitable AND model predicts SELL
    - For SELL trades: Close the lowest entry price (worst entry) if it's profitable AND model predicts BUY
    
    Args:
        trades_list: List of active trades
        current_price: Current market price
        predicted_signal: Model's predicted signal (0=SELL, 1=BUY)
    
    Returns:
        tuple: (trade_to_close_index, trade_to_close) or (None, None) if no trade should be closed
    """
    if not trades_list or predicted_signal is None:
        return None, None
    
    # Separate BUY and SELL trades
    buy_trades = [t for t in trades_list if t['signal'] == 1]
    sell_trades = [t for t in trades_list if t['signal'] == 0]
    
    # Check BUY trades: find highest entry (worst entry) that is profitable
    if buy_trades and predicted_signal == 0:  # Model predicts SELL
        # Find the highest entry price among BUY trades
        highest_buy_trade = max(buy_trades, key=lambda t: t['entry_price'])
        pnl_pct, pnl_usdt = calculate_unrealized_pnl(highest_buy_trade, current_price)
        
        # Check if this trade is profitable
        if pnl_pct > 0:
            # Find the index in the original trades_list
            for i, trade in enumerate(trades_list):
                if (trade['signal'] == 1 and 
                    abs(trade['entry_price'] - highest_buy_trade['entry_price']) < 0.01):
                    logging.warning(f"🛡️ Risk Management: Closing profitable BUY trade (highest entry ${highest_buy_trade['entry_price']:.2f}) "
                                  f"with unrealized profit {pnl_pct:+.2f}% (${pnl_usdt:+.2f}) because model predicts SELL")
                    return i, highest_buy_trade
    
    # Check SELL trades: find lowest entry (worst entry) that is profitable
    if sell_trades and predicted_signal == 1:  # Model predicts BUY
        # Find the lowest entry price among SELL trades
        lowest_sell_trade = min(sell_trades, key=lambda t: t['entry_price'])
        pnl_pct, pnl_usdt = calculate_unrealized_pnl(lowest_sell_trade, current_price)
        
        # Check if this trade is profitable
        if pnl_pct > 0:
            # Find the index in the original trades_list
            for i, trade in enumerate(trades_list):
                if (trade['signal'] == 0 and 
                    abs(trade['entry_price'] - lowest_sell_trade['entry_price']) < 0.01):
                    logging.warning(f"🛡️ Risk Management: Closing profitable SELL trade (lowest entry ${lowest_sell_trade['entry_price']:.2f}) "
                                  f"with unrealized profit {pnl_pct:+.2f}% (${pnl_usdt:+.2f}) because model predicts BUY")
                    return i, lowest_sell_trade
    
    return None, None

def should_adopt_new_model(current_predictions, test_predictions):
    """
    Compare performance of current model vs test model to decide adoption.
    
    Compares:
    - Average confidence levels
    - Number of high-confidence predictions
    - Consistency of predictions
    
    Args:
        current_predictions: List of predictions from current model
        test_predictions: List of predictions from test model
    
    Returns:
        bool: True if test model performs better, False otherwise
    """
    if len(test_predictions) < 2 or len(current_predictions) < 2:
        logging.info(f"❌ Insufficient predictions during test (Current: {len(current_predictions)}, Test: {len(test_predictions)}). Need at least 2 each.")
        return False
    
    # Calculate metrics for current model
    current_avg_confidence = sum(p['confidence'] for p in current_predictions) / len(current_predictions)
    current_high_conf_count = sum(1 for p in current_predictions if p['confidence'] >= 0.65)
    current_high_conf_ratio = current_high_conf_count / len(current_predictions)
    
    # Calculate metrics for test model
    test_avg_confidence = sum(p['confidence'] for p in test_predictions) / len(test_predictions)
    test_high_conf_count = sum(1 for p in test_predictions if p['confidence'] >= 0.65)
    test_high_conf_ratio = test_high_conf_count / len(test_predictions)
    
    # Calculate confidence variance (lower is better - more consistent)
    current_conf_variance = np.var([p['confidence'] for p in current_predictions])
    test_conf_variance = np.var([p['confidence'] for p in test_predictions])
    
    logging.info(f"\n{'='*60}")
    logging.info(f"📊 MODEL COMPARISON RESULTS")
    logging.info(f"{'='*60}")
    logging.info(f"\n🔵 CURRENT MODEL:")
    logging.info(f"   Total Predictions: {len(current_predictions)}")
    logging.info(f"   Average Confidence: {current_avg_confidence:.3f}")
    logging.info(f"   High Confidence (≥0.65): {current_high_conf_count} ({current_high_conf_ratio*100:.1f}%)")
    logging.info(f"   Confidence Variance: {current_conf_variance:.4f}")
    
    logging.info(f"\n🟢 TEST MODEL:")
    logging.info(f"   Total Predictions: {len(test_predictions)}")
    logging.info(f"   Average Confidence: {test_avg_confidence:.3f}")
    logging.info(f"   High Confidence (≥0.65): {test_high_conf_count} ({test_high_conf_ratio*100:.1f}%)")
    logging.info(f"   Confidence Variance: {test_conf_variance:.4f}")
    
    logging.info(f"\n📈 COMPARISON:")
    
    # Score each model (higher is better)
    current_score = 0
    test_score = 0
    
    # 1. Average confidence (40% weight)
    if test_avg_confidence > current_avg_confidence:
        improvement = ((test_avg_confidence - current_avg_confidence) / current_avg_confidence) * 100
        logging.info(f"   ✅ Avg Confidence: Test model is {improvement:.1f}% higher")
        test_score += 40
    else:
        decline = ((current_avg_confidence - test_avg_confidence) / current_avg_confidence) * 100
        logging.info(f"   ❌ Avg Confidence: Test model is {decline:.1f}% lower")
        current_score += 40
    
    # 2. High confidence ratio (35% weight)
    if test_high_conf_ratio > current_high_conf_ratio:
        logging.info(f"   ✅ High Conf Ratio: Test model has more ({test_high_conf_ratio*100:.1f}% vs {current_high_conf_ratio*100:.1f}%)")
        test_score += 35
    else:
        logging.info(f"   ❌ High Conf Ratio: Current model has more ({current_high_conf_ratio*100:.1f}% vs {test_high_conf_ratio*100:.1f}%)")
        current_score += 35
    
    # 3. Consistency - lower variance is better (25% weight)
    if test_conf_variance < current_conf_variance:
        improvement = ((current_conf_variance - test_conf_variance) / current_conf_variance) * 100
        logging.info(f"   ✅ Consistency: Test model is {improvement:.1f}% more consistent")
        test_score += 25
    else:
        decline = ((test_conf_variance - current_conf_variance) / test_conf_variance) * 100
        logging.info(f"   ❌ Consistency: Current model is {decline:.1f}% more consistent")
        current_score += 25
    
    logging.info(f"\n🎯 FINAL SCORES:")
    logging.info(f"   Current Model: {current_score}/100")
    logging.info(f"   Test Model: {test_score}/100")
    
    # Test model must score higher to be adopted
    if test_score > current_score:
        logging.info(f"\n✅ VERDICT: Test model performs BETTER (+{test_score - current_score} points)")
        return True
    else:
        logging.info(f"\n❌ VERDICT: Current model performs BETTER (+{current_score - test_score} points)")
        return False

def adopt_new_model():
    """Replace the current production model with the tested model"""
    import os
    import shutil
    
    current_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model.pth')
    test_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model_test.pth')
    backup_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model_backup.pth')
    
    # Backup current model
    if os.path.exists(current_model_path):
        shutil.copy(current_model_path, backup_model_path)
        logging.info(f"📦 Backed up current model to {backup_model_path}")
    
    # Replace with new model
    if os.path.exists(test_model_path):
        shutil.copy(test_model_path, current_model_path)
        logging.info("✅ New model adopted as primary model")
    else:
        logging.info("❌ Error: Test model file not found!")

def append_data_to_training_file(df):
    """Append new data to training_data.csv immediately when fetched"""
    if df is None or df.empty:
        return
    
    try:
        # Ensure open_time is datetime and set as index
        df_copy = df.copy()
        if 'open_time' in df_copy.columns:
            # open_time is a column - convert to datetime and set as index
            df_copy['open_time'] = pd.to_datetime(df_copy['open_time'])
            df_copy.set_index('open_time', inplace=True)
        elif df_copy.index.name == 'open_time' or isinstance(df_copy.index, pd.DatetimeIndex):
            # open_time is already the index - just ensure it's datetime
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
        else:
            # No open_time found - this shouldn't happen, but handle gracefully
            logging.warning("No 'open_time' column or index found in fetched data. Skipping append.")
            return
        
        # Load existing training data if it exists
        if os.path.exists('training_data.csv'):
            try:
                existing_df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
                
                # Find new rows that don't exist in training_data.csv
                new_rows = df_copy[~df_copy.index.isin(existing_df.index)]
                
                if len(new_rows) > 0:
                    # Append new rows
                    updated_df = pd.concat([existing_df, new_rows])
                    # Sort by index (time) and remove duplicates
                    updated_df = updated_df.sort_index()
                    updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                    # Save back to file
                    updated_df.to_csv('training_data.csv')
                    logging.info(f"✅ Appended {len(new_rows)} new records to training_data.csv (total: {len(updated_df)} rows)")
                else:
                    logging.debug("No new data to append to training_data.csv")
            except Exception as e:
                logging.warning(f"Could not append to training_data.csv: {e}. Creating new file.")
                # If append fails, create new file with current data
                df_copy.to_csv('training_data.csv')
                logging.info(f"✅ Created new training_data.csv with {len(df_copy)} records")
        else:
            # If file doesn't exist, create it with current data
            df_copy.to_csv('training_data.csv')
            logging.info(f"✅ Created training_data.csv with {len(df_copy)} records")
    except Exception as e:
        logging.error(f"Error appending data to training_data.csv: {e}", exc_info=True)

def retrain_with_recent_data(client):
    """Fine-tune the model with the latest portion of historical training data"""
    logging.info("Loading latest historical data for fine-tuning...")
    
    # Load the existing training dataset
    if not os.path.exists('training_data.csv'):
        error_msg = "No training_data.csv found. Cannot fine-tune. Run train.py first."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Load the full training dataset
        full_df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
        
        if full_df.empty or len(full_df) < 100:
            error_msg = f"Training data is too small ({len(full_df)} rows). Cannot fine-tune."
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Use the latest portion of data for fine-tuning (ensure enough samples for sequences)
        rows_target = max(FINE_TUNE_RECENT_ROWS, FINE_TUNE_LOOKBACK * 50)
        num_rows_to_use = min(len(full_df), rows_target)
        recent_df       = full_df.tail(num_rows_to_use)
        
        logging.info(f"Using latest {len(recent_df)} rows from training dataset (out of {len(full_df)} total) for fine-tuning")
        
        # Save as temporary training data for fine-tuning
        try:
            recent_df.to_csv('recent_training_data.csv')
            logging.info("✅ Created recent_training_data.csv for fine-tuning")
        except Exception as e:
            error_msg = f"Failed to save recent_training_data.csv: {e}"
            logging.error(error_msg)
            raise IOError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Failed to load training data for fine-tuning: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    logging.info("Starting model fine-tuning with recent data (time-weighted)...")
    # Run training script with test mode flag using the conda environment
    # The presence of 'recent_training_data.csv' automatically triggers fine-tuning mode
    try:
        result = subprocess.run(
            ['python', 'train.py', '--test-mode'],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False
        )
        
        if result.returncode != 0:
            # Check if this is an expected failure (exit code 0 means graceful abort due to insufficient signals)
            if "FINE-TUNING ABORTED" in result.stdout:
                logging.warning("⚠️ Fine-tuning aborted: Insufficient signals in recent data")
                logging.info("   This is expected when recent market was not volatile enough")
                logging.info("   Continuing with current model")
                # Don't raise exception - this is expected behavior
                # Still update last refinement time to avoid immediate retry
                raise ValueError("Insufficient signals for fine-tuning")
            else:
                error_msg = f"Fine-tuning failed with return code {result.returncode}. stderr: {result.stderr[:500]}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
        
        logging.info("Fine-tuning complete successfully.")
        test_model_path = os.path.join(MODEL_DIR, 'btc_predicter_model_test.pth')
        
        # Verify the test model was actually created
        if not os.path.exists(test_model_path):
            error_msg = f"Fine-tuning reported success but test model not found at {test_model_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logging.info(f"New model saved as {test_model_path}")
        logging.info("Fine-tuned model will enter testing phase.")
        
    except subprocess.TimeoutExpired:
        error_msg = "Fine-tuning process timed out after 5 minutes"
        logging.error(error_msg)
        # Clean up temporary file
        if os.path.exists('recent_training_data.csv'):
            os.remove('recent_training_data.csv')
            logging.info("🧹 Cleaned up recent_training_data.csv")
        raise TimeoutError(error_msg)
    except Exception as e:
        error_msg = f"Fine-tuning subprocess failed: {e}"
        logging.error(error_msg)
        # Clean up temporary file
        if os.path.exists('recent_training_data.csv'):
            os.remove('recent_training_data.csv')
            logging.info("🧹 Cleaned up recent_training_data.csv")
        raise RuntimeError(error_msg) from e
    finally:
        # Always clean up the temporary training data file after refinement attempt
        if os.path.exists('recent_training_data.csv'):
            try:
                os.remove('recent_training_data.csv')
                logging.info("🧹 Cleaned up recent_training_data.csv")
            except Exception as e:
                logging.warning(f"⚠️ Could not remove recent_training_data.csv: {e}")

def load_current_trades():
    """
    Load current/open trades from current_trades.json file.
    This allows the bot to restore active trades after a restart.
    
    Also fetches historical data since trade entry and appends to training_data.csv,
    checking when TP/SL was hit.
    
    Returns:
        tuple: (active_trades, simulated_trades, trade_counter)
    """
    active_trades = []
    simulated_trades = []
    trade_counter = 0
    
    if not os.path.exists(CURRENT_TRADES_FILE):
        logging.info(f"📂 {CURRENT_TRADES_FILE} not found. Starting with empty trade lists.")
        return active_trades, simulated_trades, trade_counter
    
    try:
        with open(CURRENT_TRADES_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content or content == 'w':  # Handle corrupted file
                logging.warning(f"⚠️ {CURRENT_TRADES_FILE} appears corrupted. Starting fresh.")
                return active_trades, simulated_trades, trade_counter
            
            data = json.loads(content)
            
            def _filter_valid_trades(trades, label):
                """Ensure trades contain required fields; drop and log otherwise."""
                required = ['entry_time', 'entry_price', 'tp', 'sl', 'signal']
                valid = []
                dropped = 0
                for t in trades:
                    missing = [k for k in required if t.get(k) is None]
                    if missing:
                        logging.warning(f"⚠️ Skipping {label} trade missing fields: {missing}")
                        dropped += 1
                        continue
                    valid.append(t)
                if dropped:
                    logging.info(f"🧹 Removed {dropped} invalid {label} trades missing required fields.")
                return valid
            
            # Load active trades (real trades)
            if 'active_trades' in data and isinstance(data['active_trades'], list):
                active_trades = data['active_trades']
                # Ensure entry_time is float (timestamp)
                for trade in active_trades:
                    if 'entry_time' in trade and isinstance(trade['entry_time'], str):
                        # Try to parse datetime string back to timestamp
                        try:
                            trade['entry_time'] = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S').timestamp()
                        except:
                            logging.warning(f"Could not parse entry_time for trade. Using current time.")
                            trade['entry_time'] = time.time()
                    elif 'entry_time' not in trade:
                        trade['entry_time'] = time.time()
                active_trades = _filter_valid_trades(active_trades, "active")
            
            # Load simulated trades (test mode)
            if 'simulated_trades' in data and isinstance(data['simulated_trades'], list):
                simulated_trades = data['simulated_trades']
                # Ensure entry_time is float (timestamp)
                for trade in simulated_trades:
                    if 'entry_time' in trade and isinstance(trade['entry_time'], str):
                        try:
                            trade['entry_time'] = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S').timestamp()
                        except:
                            logging.warning(f"Could not parse entry_time for simulated trade. Using current time.")
                            trade['entry_time'] = time.time()
                    elif 'entry_time' not in trade:
                        trade['entry_time'] = time.time()
                simulated_trades = _filter_valid_trades(simulated_trades, "simulated")
            
            # Load trade counter
            if 'trade_counter' in data:
                trade_counter = int(data['trade_counter'])
            
            logging.info(f"✅ Successfully loaded {len(active_trades)} active trades and {len(simulated_trades)} simulated trades from {CURRENT_TRADES_FILE}")
            if trade_counter > 0:
                logging.info(f"📊 Trade counter restored to {trade_counter}")
            
            # Process trades to append historical data and check TP/SL hit times
            if active_trades or simulated_trades:
                _process_historical_trade_data(active_trades + simulated_trades)
            
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logging.warning(f"⚠️ Could not parse {CURRENT_TRADES_FILE}: {e}. Starting with empty trade lists.")
        return active_trades, simulated_trades, trade_counter
    except Exception as e:
        logging.error(f"Error loading {CURRENT_TRADES_FILE}: {e}", exc_info=True)
        return active_trades, simulated_trades, trade_counter
    
    return active_trades, simulated_trades, trade_counter


def _process_historical_trade_data(trades):
    """
    For each loaded trade:
    1. Fetch historical data from entry time to now
    2. Find earliest time when TP or SL was hit
    3. Append new data to training_data.csv
    
    Args:
        trades: List of trade dictionaries
    """
    if not trades:
        return
    
    try:
        client = MEXCClient()
        
        # Load existing training data if available
        training_data_path = 'training_data.csv'
        if os.path.exists(training_data_path):
            existing_df = pd.read_csv(training_data_path, index_col=0, parse_dates=True)
            last_timestamp = int(existing_df.index[-1].timestamp())
            logging.info(f"📊 Loaded existing training data: {len(existing_df)} rows, last timestamp: {existing_df.index[-1]}")
        else:
            existing_df = None
            last_timestamp = None
        
        all_new_data = []
        
        for trade in trades:
            try:
                entry_time = trade.get('entry_time')
                entry_price = trade.get('entry_price')
                tp = trade.get('tp')
                sl = trade.get('sl')
                signal = trade.get('signal')  # 'BUY' or 'SELL'
                
                if not all([entry_time is not None, entry_price, tp, sl, signal is not None]):
                    logging.warning(f"⚠️ Trade missing required fields (entry_time={entry_time}, entry_price={entry_price}, tp={tp}, sl={sl}, signal={signal}), skipping")
                    continue
                
                # Only fetch data we don't already have
                start_time = int(entry_time)
                entry_dt = datetime.fromtimestamp(entry_time)
                
                # Check if entry time is in the future (shouldn't happen but handle it)
                current_time = time.time()
                if start_time > current_time:
                    logging.warning(f"⚠️ Trade entry time {entry_dt} is in the future. Skipping historical data fetch.")
                    continue
                
                # Check if entry is very recent (< 2 minutes ago) - not enough time for TP/SL
                if current_time - start_time < 120:  # Less than 2 minutes old
                    logging.info(f"ℹ️  Trade from {entry_dt} is very recent (< 2 min old). Skipping historical data fetch.")
                    continue
                
                if last_timestamp and start_time <= last_timestamp:
                    start_time = last_timestamp + 60  # Start 1 minute after last data
                    logging.debug(f"   Adjusting start time to {datetime.fromtimestamp(start_time)} (after last training data)")
                
                # Fetch data from entry time to now
                logging.info(f"📥 Fetching historical data for trade (entry: {entry_dt}, signal: {signal})")
                new_df = client.get_kline_data(symbol='BTC_USDT', interval='Min1', start=start_time)
                
                if new_df is None or new_df.empty:
                    logging.warning(f"⚠️ No historical data available for trade from {entry_dt}. Market data may not be available for this time period.")
                    continue
                
                if len(new_df) < 2:
                    logging.warning(f"⚠️ Insufficient historical data for trade from {entry_dt} (only {len(new_df)} rows). Need at least 2 data points.")
                    continue
                
                new_df['open_time'] = pd.to_datetime(new_df['open_time'])
                new_df.set_index('open_time', inplace=True)
                
                # Find when TP or SL was hit
                tp_hit_time = None
                sl_hit_time = None
                
                if signal == 'BUY' or signal == 1:
                    # For BUY: TP is above entry, SL is below
                    tp_hit = new_df[new_df['high'] >= tp]
                    sl_hit = new_df[new_df['low'] <= sl]
                elif signal == 'SELL' or signal == 0:
                    # For SELL: TP is below entry, SL is above
                    tp_hit = new_df[new_df['low'] <= tp]
                    sl_hit = new_df[new_df['high'] >= sl]
                else:
                    logging.warning(f"⚠️ Unknown signal type: {signal}")
                    continue
                
                if not tp_hit.empty:
                    tp_hit_time = tp_hit.index[0]
                if not sl_hit.empty:
                    sl_hit_time = sl_hit.index[0]
                
                # Log the earliest hit
                if tp_hit_time and sl_hit_time:
                    earliest_hit = min(tp_hit_time, sl_hit_time)
                    hit_type = "TP" if earliest_hit == tp_hit_time else "SL"
                    logging.info(f"✅ Trade {signal} from {datetime.fromtimestamp(entry_time)}: {hit_type} hit at {earliest_hit}")
                elif tp_hit_time:
                    logging.info(f"✅ Trade {signal} from {datetime.fromtimestamp(entry_time)}: TP hit at {tp_hit_time}")
                elif sl_hit_time:
                    logging.info(f"⚠️ Trade {signal} from {datetime.fromtimestamp(entry_time)}: SL hit at {sl_hit_time}")
                else:
                    logging.info(f"⏳ Trade {signal} from {datetime.fromtimestamp(entry_time)}: Neither TP nor SL hit yet")
                
                # Add to collection for appending
                all_new_data.append(new_df)
                
            except Exception as e:
                logging.error(f"❌ Error processing trade: {e}", exc_info=True)
                continue
        
        # Append all new data to training_data.csv
        if all_new_data:
            combined_new_df = pd.concat(all_new_data, ignore_index=False)
            combined_new_df = combined_new_df[~combined_new_df.index.duplicated(keep='first')]
            combined_new_df = combined_new_df.sort_index()
            
            if existing_df is not None:
                # Merge with existing, removing duplicates
                combined_df = pd.concat([existing_df, combined_new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                combined_df = combined_df.sort_index()
            else:
                combined_df = combined_new_df
            
            # Save back to training_data.csv
            combined_df.to_csv(training_data_path)
            logging.info(f"💾 Appended {len(combined_new_df)} new rows to training_data.csv (total: {len(combined_df)} rows)")
        else:
            logging.info(f"ℹ️ No new data to append to training_data.csv")
            
    except Exception as e:
        logging.error(f"❌ Error processing historical trade data: {e}", exc_info=True)

def save_current_trades(active_trades, simulated_trades, trade_counter):
    """
    Save current/open trades to current_trades.json file.
    This allows the bot to restore active trades after a restart.
    
    Args:
        active_trades: List of active real trades
        simulated_trades: List of active simulated trades (test mode)
        trade_counter: Current trade counter value
    """
    try:
        # Prepare data structure
        data = {
            'active_trades': [],
            'simulated_trades': [],
            'trade_counter': trade_counter,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Convert active trades to JSON-serializable format
        for trade in active_trades:
            trade_copy = trade.copy()
            # Convert entry_time to string for JSON serialization
            if 'entry_time' in trade_copy:
                trade_copy['entry_time'] = datetime.fromtimestamp(trade_copy['entry_time']).strftime('%Y-%m-%d %H:%M:%S')
            # Ensure all numeric values are float (not numpy types)
            for key in ['entry_price', 'tp', 'sl', 'quantity', 'trade_amount_usdt', 
                       'expected_profit_usdt', 'expected_loss_usdt', 'confidence']:
                if key in trade_copy and trade_copy[key] is not None:
                    trade_copy[key] = float(trade_copy[key])
            data['active_trades'].append(trade_copy)
        
        # Convert simulated trades to JSON-serializable format
        for trade in simulated_trades:
            trade_copy = trade.copy()
            # Convert entry_time to string for JSON serialization
            if 'entry_time' in trade_copy:
                trade_copy['entry_time'] = datetime.fromtimestamp(trade_copy['entry_time']).strftime('%Y-%m-%d %H:%M:%S')
            # Ensure all numeric values are float (not numpy types)
            for key in ['entry_price', 'tp', 'sl', 'quantity', 'trade_amount_usdt', 
                       'expected_profit_usdt', 'expected_loss_usdt', 'confidence']:
                if key in trade_copy and trade_copy[key] is not None:
                    trade_copy[key] = float(trade_copy[key])
            data['simulated_trades'].append(trade_copy)
        
        # Save to file
        with open(CURRENT_TRADES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.debug(f"💾 Saved {len(active_trades)} active trades and {len(simulated_trades)} simulated trades to {CURRENT_TRADES_FILE}")
        
    except Exception as e:
        logging.error(f"Error saving {CURRENT_TRADES_FILE}: {e}", exc_info=True)

def load_stats():
    """Load all daily stats from JSON file and return today's stats."""
    today = datetime.now().strftime("%Y-%m-%d")
    all_stats = {}
    
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            try:
                loaded_data = json.load(f)
                # Handle old format (single day object) - convert to new format
                if isinstance(loaded_data, dict):
                    # Check if it's old format (has 'date' but keys are not dates)
                    if 'date' in loaded_data and len(loaded_data) <= 5 and 'successful_trades' in loaded_data:
                        # Old format detected, convert
                        old_date = loaded_data.get('date', today)
                        all_stats = {old_date: loaded_data}
                    else:
                        # New format - dictionary of dates
                        all_stats = loaded_data
                else:
                    all_stats = {}
            except json.JSONDecodeError:
                logging.warning(f"Could not decode {STATS_FILE}. Starting with fresh stats.")
                all_stats = {}
    
    # Get or create today's stats
    if today not in all_stats:
        # Always initialize balance (both in TEST and LIVE mode to preserve continuity)
        balance_value = None
        # Get previous day's balance or initialize to $1000
        previous_balance = 1000.0  # Default starting balance
        if all_stats:
            # Get the most recent day's balance
            sorted_dates = sorted(all_stats.keys(), reverse=True)
            if sorted_dates:
                previous_balance = all_stats[sorted_dates[0]].get('balance', 1000.0)
        balance_value = previous_balance
        
        all_stats[today] = {
            "date": today,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit_usdt": 0.0,
            "win_rate_pct": 0.0
        }
        
        # Always add balance to preserve continuity between days
        all_stats[today]["balance"] = balance_value
    
    return all_stats[today]

def save_stats(today_stats): 
    """Save daily stats to JSON file. Updates today's stats in the all-days structure."""
    # Calculate win rate
    total_trades = today_stats['successful_trades'] + today_stats['failed_trades']
    if total_trades > 0:
        today_stats['win_rate_pct'] = (today_stats['successful_trades'] / total_trades) * 100
    else:
        today_stats['win_rate_pct'] = 0.0
    
    # Update timestamp
    today_stats['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Load all existing stats
    all_stats = {}
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            try:
                loaded_data = json.load(f)
                # Handle old format conversion
                if isinstance(loaded_data, dict):
                    if 'date' in loaded_data and len(loaded_data) <= 5 and 'successful_trades' in loaded_data:
                        # Old format detected, convert
                        old_date = loaded_data.get('date', today_stats['date'])
                        all_stats = {old_date: loaded_data}
                    else:
                        # New format - dictionary of dates
                        all_stats = loaded_data
            except json.JSONDecodeError:
                all_stats = {}
    
    # Update today's stats
    date_key = today_stats['date']
    all_stats[date_key] = today_stats
    
    # Save all stats
    with open(STATS_FILE, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    balance = today_stats.get('balance', 1000.0)
    log_msg = (f"📊 Daily stats saved for {date_key}: Wins={today_stats['successful_trades']}, Losses={today_stats['failed_trades']}, "
              f"P&L=${today_stats['total_profit_usdt']:.2f}, Balance=${balance:.2f}, Win Rate={today_stats['win_rate_pct']:.2f}%")
    logging.info(log_msg)

def update_stats(stats, result, pnl):
    """Update daily stats after a trade closes."""
    if result == 'PROFIT':
        stats['successful_trades'] += 1
    else: # LOSS
        stats['failed_trades'] += 1
    
    stats['total_profit_usdt'] += pnl
    
    # Update balance: Add P&L to current balance (both TEST and LIVE modes)
    if 'balance' not in stats:
        stats['balance'] = 1000.0  # Initialize if missing
    
    # Add P&L to balance (this represents actual account equity change)
    stats['balance'] += pnl
    
    # Ensure balance doesn't go negative (minimum $0)
    if stats['balance'] < 0:
        stats['balance'] = 0.0

def initialize_stats_from_trades_log():
    """
    Initialize trading stats from trades_log.json at bot startup. Processes all days.
    Recalculates balance from scratch in TEST mode only to ensure simulated balance accuracy.
    In LIVE mode, balance is tracked separately and not modified by this function.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Load existing stats to preserve previous days structure, but we'll recalculate balances
    all_stats = {}
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            try:
                loaded_data = json.load(f)
                # Handle old format conversion
                if isinstance(loaded_data, dict):
                    if 'date' in loaded_data and len(loaded_data) <= 5 and 'successful_trades' in loaded_data:
                        # Old format detected, convert
                        old_date = loaded_data.get('date', today)
                        all_stats = {old_date: loaded_data}
                    else:
                        # New format - dictionary of dates
                        all_stats = loaded_data
            except json.JSONDecodeError:
                all_stats = {}
    
    # Log that we're recalculating balances
    if all_stats:
        logging.info("🔄 Recalculating all balances using new algorithm from trades_log.json...")
    
    # Initialize today's stats
    if today not in all_stats:
        # Get previous day's balance or initialize to $1000
        previous_balance = 1000.0  # Default starting balance
        if all_stats:
            # Get the most recent day's balance
            sorted_dates = sorted(all_stats.keys(), reverse=True)
            if sorted_dates:
                previous_balance = all_stats[sorted_dates[0]].get('balance', 1000.0)
        
        all_stats[today] = {
            "date": today,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit_usdt": 0.0,
            "win_rate_pct": 0.0,
            "balance": previous_balance  # Carry forward balance from previous day
        }
    
    if not os.path.exists(TRADES_LOG_FILE):
        logging.info(f"📊 No {TRADES_LOG_FILE} found. Starting with fresh stats.")
        # Save today's stats even if no trades log exists
        with open(STATS_FILE, 'w') as f:
            json.dump(all_stats, f, indent=4)
        return all_stats[today]
    
    try:
        with open(TRADES_LOG_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content or content == 'w':
                logging.info(f"📊 {TRADES_LOG_FILE} is empty. Starting with fresh stats.")
                # Save today's stats even if trades log is empty
                with open(STATS_FILE, 'w') as f:
                    json.dump(all_stats, f, indent=4)
                return all_stats[today]
            
            trades_log = json.loads(content)
            
            if not isinstance(trades_log, list):
                logging.warning(f"📊 {TRADES_LOG_FILE} format invalid. Starting with fresh stats.")
                # Save today's stats even if trades log format is invalid
                with open(STATS_FILE, 'w') as f:
                    json.dump(all_stats, f, indent=4)
                return all_stats[today]
            
            # Process all completed trades (exclude ACTIVE trades) and group by day
            trades_by_day = {}
            for trade in trades_log:
                # Skip ACTIVE trades
                if trade.get('status') == 'ACTIVE':
                    continue
                
                trade_timestamp = trade.get('time_stamp', '')
                if not trade_timestamp:
                    continue
                
                # Extract date from timestamp (format: "YYYY-MM-DD HH:MM:SS")
                try:
                    trade_date = trade_timestamp.split(' ')[0]  # Get date part
                    if trade_date not in trades_by_day:
                        trades_by_day[trade_date] = {
                            "successful_trades": 0,
                            "failed_trades": 0,
                            "total_profit_usdt": 0.0
                        }
                    
                    result = trade.get('Profit/loss', '')
                    pnl = trade.get('PL in $', 0.0)
                    
                    if result == 'PROFIT':
                        trades_by_day[trade_date]['successful_trades'] += 1
                    elif result == 'LOSS':
                        trades_by_day[trade_date]['failed_trades'] += 1
                    
                    trades_by_day[trade_date]['total_profit_usdt'] += float(pnl)
                except (IndexError, ValueError):
                    continue
            
            # Calculate balance by processing trades chronologically
            sorted_trade_dates = sorted(trades_by_day.keys())
            
            # Start with initial balance of $1000 or carry forward from previous day
            initial_balance = 1000.0
            if all_stats:
                sorted_dates = sorted(all_stats.keys())
                if sorted_dates:
                    initial_balance = all_stats[sorted_dates[0]].get('balance', 1000.0)
            
            # Process trades chronologically to calculate balance
            running_balance = initial_balance
            
            for date_key in sorted_trade_dates:
                day_stats = trades_by_day[date_key]
                
                if date_key not in all_stats:
                    all_stats[date_key] = {
                        "date": date_key,
                        "timestamp": f"{date_key} 00:00:00",
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "successful_trades": 0,
                        "failed_trades": 0,
                        "total_profit_usdt": 0.0,
                        "win_rate_pct": 0.0,
                        "balance": running_balance  # Always track balance
                    }
                
                # Update stats for this day
                all_stats[date_key]['successful_trades'] = day_stats['successful_trades']
                all_stats[date_key]['failed_trades'] = day_stats['failed_trades']
                all_stats[date_key]['total_profit_usdt'] = day_stats['total_profit_usdt']
                
                # Update balance by adding P&L for this day (only in TEST mode)
                if TEST:
                    running_balance += day_stats['total_profit_usdt']
                    if running_balance < 0:
                        running_balance = 0.0
                    all_stats[date_key]['balance'] = running_balance
                
                # Calculate win rate
                total_trades = day_stats['successful_trades'] + day_stats['failed_trades']
                if total_trades > 0:
                    all_stats[date_key]['win_rate_pct'] = (day_stats['successful_trades'] / total_trades) * 100
                else:
                    all_stats[date_key]['win_rate_pct'] = 0.0
            
            # Save all stats to file
            with open(STATS_FILE, 'w') as f:
                json.dump(all_stats, f, indent=4)
            
            # Log summary with balance recalculation notice
            today_trades = trades_by_day.get(today, {})
            today_total = today_trades.get('successful_trades', 0) + today_trades.get('failed_trades', 0)
            balance = all_stats[today].get('balance', 1000.0)
            
            # Calculate total balance change for verification
            if len(sorted_trade_dates) > 1:
                first_date = sorted_trade_dates[0]
                last_date = sorted_trade_dates[-1]
                initial_balance = 1000.0
                if all_stats:
                    sorted_dates = sorted(all_stats.keys())
                    if sorted_dates and first_date in all_stats:
                        # Get balance before first trade
                        initial_balance = all_stats[first_date].get('balance', 1000.0) - all_stats[first_date].get('total_profit_usdt', 0)
                final_balance = all_stats[last_date].get('balance', 1000.0)
                total_change = final_balance - initial_balance
                logging.info(f"✅ Balance recalculated: ${initial_balance:.2f} → ${final_balance:.2f} (change: ${total_change:+.2f})")
            
            log_msg = (f"📊 Initialized stats from {TRADES_LOG_FILE}: "
                      f"Processed {len(trades_by_day)} day(s). Today: "
                      f"Wins={all_stats[today]['successful_trades']}, Losses={all_stats[today]['failed_trades']}, "
                      f"P&L=${all_stats[today]['total_profit_usdt']:.2f}, Balance=${balance:.2f}, "
                      f"Win Rate={all_stats[today]['win_rate_pct']:.2f}% ({today_total} trades)")
            logging.info(log_msg)
            
            return all_stats[today]
            
    except (json.JSONDecodeError, ValueError) as e:
        logging.warning(f"⚠️ Could not parse {TRADES_LOG_FILE}: {e}. Starting with fresh stats.")
        # Save today's stats even if parsing fails
        with open(STATS_FILE, 'w') as f:
            json.dump(all_stats, f, indent=4)
        return all_stats[today]
    except Exception as e:
        logging.error(f"Error reading {TRADES_LOG_FILE}: {e}. Starting with fresh stats.", exc_info=True)
        # Save today's stats even if there's an error
        with open(STATS_FILE, 'w') as f:
            json.dump(all_stats, f, indent=4)
        return all_stats[today]

def load_price_history(max_rows=None):
    """Load training_data.csv for trade reconciliation."""
    if not os.path.exists('training_data.csv'):
        logging.warning("training_data.csv not found. Cannot reconcile previous trades.")
        return None
    try:
        df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
        if df.empty:
            logging.warning("training_data.csv is empty. Cannot reconcile previous trades.")
            return None
        if max_rows:
            df = df.tail(max_rows)
        return df
    except Exception as e:
        logging.error(f"Failed to load training_data.csv for reconciliation: {e}")
        return None

def determine_trade_outcome_from_history(trade, price_history):
    """
    Check if a persisted trade should have been closed based on historical prices.
    
    A trade should be closed if:
    1. TP was reached (PROFIT)
    2. SL was hit (LOSS)
    3. PnL loss reached $5 USD (LOSS) - using EARLY_STOP_LOSS_THRESHOLD
    
    Returns:
        tuple: (exit_price, outcome) or (None, None) if trade should remain open
    """
    entry_ts = trade.get('entry_time')
    if entry_ts is None:
        return None, None
    entry_dt = datetime.fromtimestamp(entry_ts)
    history = price_history[price_history.index >= entry_dt]
    if history.empty:
        return None, None
    
    entry_price = trade['entry_price']
    tp = trade['tp']
    sl = trade['sl']
    signal = trade['signal']
    quantity = trade.get('quantity', 0)
    
    # Check each historical row chronologically
    for timestamp, row in history.iterrows():
        high = row.get('high')
        low = row.get('low')
        close = row.get('close')
        
        if pd.isna(high) or pd.isna(low) or pd.isna(close):
            continue
        
        if signal == 1:  # BUY trade
            # Check SL first (loss condition has priority)
            if low <= sl:
                logging.info(f"🔍 Reconciliation: BUY trade hit SL at {timestamp} (entry: ${entry_price:.2f}, SL: ${sl:.2f})")
                return sl, 'LOSS'
            
            # Check $5 loss threshold (using EARLY_STOP_LOSS_THRESHOLD)
            # For BUY: PnL_USD = quantity * (current_price - entry_price)
            # Loss of $5 means: quantity * (current_price - entry_price) <= -5
            # So: current_price <= entry_price - (5 / quantity)
            if quantity > 0:
                loss_threshold_price = entry_price - (EARLY_STOP_LOSS_THRESHOLD / quantity)
                if low <= loss_threshold_price:
                    # Use the actual low price that triggered the $200 loss threshold
                    exit_price = min(low, loss_threshold_price)
                    pnl_usdt = quantity * (exit_price - entry_price)
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    logging.info(f"🔍 Reconciliation: BUY trade hit ${EARLY_STOP_LOSS_THRESHOLD} loss threshold at {timestamp} (entry: ${entry_price:.2f}, exit: ${exit_price:.2f}, PnL: ${pnl_usdt:.2f}, {pnl_pct:.2f}%)")
                    return exit_price, 'LOSS'
            
            # Check TP (profit condition)
            if high >= tp:
                logging.info(f"🔍 Reconciliation: BUY trade hit TP at {timestamp} (entry: ${entry_price:.2f}, TP: ${tp:.2f})")
                return tp, 'PROFIT'
                
        else:  # SELL trade (signal == 0)
            # Check SL first (loss condition has priority)
            if high >= sl:
                logging.info(f"🔍 Reconciliation: SELL trade hit SL at {timestamp} (entry: ${entry_price:.2f}, SL: ${sl:.2f})")
                return sl, 'LOSS'
            
            # Check $5 loss threshold (using EARLY_STOP_LOSS_THRESHOLD)
            # For SELL: PnL_USD = quantity * (entry_price - current_price)
            # Loss of $5 means: quantity * (entry_price - current_price) <= -5
            # So: current_price >= entry_price + (5 / quantity)
            if quantity > 0:
                loss_threshold_price = entry_price + (EARLY_STOP_LOSS_THRESHOLD / quantity)
                if high >= loss_threshold_price:
                    # Use the actual high price that triggered the $200 loss threshold
                    exit_price = max(high, loss_threshold_price)
                    pnl_usdt = quantity * (entry_price - exit_price)
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    logging.info(f"🔍 Reconciliation: SELL trade hit ${EARLY_STOP_LOSS_THRESHOLD} loss threshold at {timestamp} (entry: ${entry_price:.2f}, exit: ${exit_price:.2f}, PnL: ${pnl_usdt:.2f}, {pnl_pct:.2f}%)")
                    return exit_price, 'LOSS'
            
            # Check TP (profit condition)
            if low <= tp:
                logging.info(f"🔍 Reconciliation: SELL trade hit TP at {timestamp} (entry: ${entry_price:.2f}, TP: ${tp:.2f})")
                return tp, 'PROFIT'
    
    return None, None

def reconcile_trades_on_startup(active_trades, simulated_trades, daily_stats, trade_counter):
    """
    When the bot restarts, check if any persisted trades should have been closed
    based on the latest historical data (training_data.csv).
    
    Checks for:
    1. Trades that reached TP (PROFIT)
    2. Trades that hit SL (LOSS)
    3. Trades with loss >= $5 USD (LOSS) - using EARLY_STOP_LOSS_THRESHOLD
    
    If any condition is met, close the trade, log the result, and update stats.
    """
    price_history = load_price_history()
    if price_history is None:
        logging.warning("⚠️ Cannot reconcile trades: no historical data available")
        return active_trades, simulated_trades, daily_stats, trade_counter
    
    total_active = len(active_trades)
    total_simulated = len(simulated_trades)
    
    if total_active + total_simulated == 0:
        logging.info("ℹ️  No persisted trades to reconcile.")
        return active_trades, simulated_trades, daily_stats, trade_counter
    
    logging.info(f"🔁 Reconciling {total_active + total_simulated} persisted trades against historical data...")
    logging.info(f"   Historical data range: {price_history.index[0]} to {price_history.index[-1]} ({len(price_history)} rows)")
    
    trades_closed = 0
    trades_kept = 0
    stats_updated = False
    
    for trades_list, simulated in ((active_trades, False), (simulated_trades, True)):
        if not trades_list:
            continue
            
        trade_type = "simulated" if simulated else "active"
        logging.info(f"   Checking {len(trades_list)} {trade_type} trades...")
        
        indexes_to_remove = []
        for idx, trade in enumerate(trades_list):
            # Log trade details for debugging
            entry_time = datetime.fromtimestamp(trade['entry_time']) if trade.get('entry_time') else None
            signal_map = {0: 'SELL', 1: 'BUY'}
            signal_str = signal_map.get(trade.get('signal'), 'UNKNOWN')
            
            exit_price, outcome = determine_trade_outcome_from_history(trade, price_history)
            
            if exit_price is None:
                # Trade should remain open
                logging.info(f"      ✓ {signal_str} trade from {entry_time} remains open (entry: ${trade['entry_price']:.2f}, TP: ${trade['tp']:.2f}, SL: ${trade['sl']:.2f})")
                trades_kept += 1
                continue
            
            # Trade should be closed
            if trade.get('index') is None:
                trade_counter += 1
                trade['index'] = trade_counter
            
            logging.info(f"      🔚 Closing {signal_str} trade #{trade['index']} from {entry_time} - {outcome}")
            
            pnl = print_trade_result(
                trade,
                exit_price,
                result='PROFIT' if outcome == 'PROFIT' else 'LOSS',
                simulated=simulated or TEST,
                trade_index=trade['index']
            )
            
            if not (simulated or TEST):
                update_stats(daily_stats, outcome, pnl)
                stats_updated = True
            
            indexes_to_remove.append(idx)
            trades_closed += 1
        
        for idx in reversed(indexes_to_remove):
            trades_list.pop(idx)
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"📊 RECONCILIATION SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"   Total trades checked: {total_active + total_simulated}")
    logging.info(f"   Trades closed: {trades_closed}")
    logging.info(f"   Trades kept open: {trades_kept}")
    logging.info(f"   Remaining active trades: {len(active_trades)}")
    logging.info(f"   Remaining simulated trades: {len(simulated_trades)}")
    logging.info(f"{'='*60}\n")
    
    if trades_closed:
        save_current_trades(active_trades, simulated_trades, trade_counter)
        if stats_updated:
            save_stats(daily_stats)
    
    return active_trades, simulated_trades, daily_stats, trade_counter

if __name__ == '__main__':
    main()
