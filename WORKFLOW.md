# Crypto Trading Bot - Complete Workflow

## 📋 Overview
This document describes the complete workflow of the crypto trading bot, including all decision logic, trade management, and model refinement processes.

---

## 🔄 Main Loop Flow

### 1. **Initialization**
- Load model from `../model/btc_predicter_model.pth`
- Load scalers (`scaler.gz`, `close_scaler.gz`)
- Initialize MEXC client and Exchange
- Set up logging to `trading_bot.log`
- Initialize trade tracking variables

### 2. **Data Fetching & Processing** (Every 5 seconds)
```
1. Fetch latest BTC/USDT 1-minute kline data from MEXC
2. Append data to training_data.csv for future training
3. Calculate 93 technical indicators (MA, EMA, RSI, MACD, DMI, etc.)
4. Scale features using saved scaler
5. Create input sequence (60 timesteps × 93 features)
```

### 3. **Model Prediction**
```
Input: (1, 60, 93) tensor of scaled features
Output:
  - signal_probs: [SELL_prob, BUY_prob] (softmax probabilities)
  - tp_scaled: Scaled take profit price
  - sl_scaled: Scaled stop loss price

Extract:
  - predicted_signal: argmax(signal_probs) → 0=SELL, 1=BUY
  - confidence: max(signal_probs) → 0.5 to 1.0
  - sell_prob, buy_prob: Individual probabilities
```

---

## 📊 Trade Management Flow

### 4. **Check Active Trades** (Before New Predictions)

#### 4.1 Check for Trade Completion/Loss
For each active trade:
- **Stop Loss Hit**: Close trade, log as LOSS
- **Take Profit Reached**: Close trade, log as PROFIT
- **Trade Index**: Use stored index (persists even if other trades closed)

#### 4.2 Check for Early Stop
Early stop triggers when **BOTH** conditions are met:
1. **Time Condition**: Trade has been losing (red) for ≥ 2 hours
2. **Opposite Signal**: Model predicts opposite signal (BUY→SELL or SELL→BUY)

If both true → Close trade early, log as LOSS (EARLY STOP)

#### 4.3 Display Active Trades
- Show trade index (persistent, e.g., #1, #2, #4, #5 even if #3 was closed)
- Show unrealized P&L (percentage and $ amount)
- Color: 🟢 green if profit, 🔴 red if loss

---

## 🎯 Signal Processing & TP/SL Calculation

### 5. **TP-Based Signal Determination**
```
1. Inverse transform TP/SL from scaled values to actual prices
2. Determine TP-based signal:
   - If TP > current_price → BUY signal (tp_based_signal = 1)
   - If TP < current_price → SELL signal (tp_based_signal = 0)
3. Use TP-based signal for trading (not model's original signal)
```

### 6. **Signal Mismatch Handling**
```
If model_predicted_signal ≠ tp_based_signal:
  - Log warning about mismatch
  - Reduce TP percentage by 50%
  - Still use TP-based signal for trading
  - Set signal_aligned = False (affects position sizing)

If model_predicted_signal = tp_based_signal:
  - Use normal TP/SL values
  - Set signal_aligned = True (affects position sizing)
```

### 7. **TP/SL Values**
**Normal (signals aligned):**
- BUY: TP = 110% of entry, SL = 80% of entry
- SELL: TP = 90% of entry, SL = 120% of entry

**Mismatch (signals differ):**
- TP reduced by 50% from calculated value
- SL remains: BUY = 80%, SELL = 120%

---

## 💰 Position Sizing

### 8. **Confidence-Based Position Sizing**
Base position size by confidence:
- 1.0: 75% of balance
- 0.95-0.999: 70% of balance
- 0.90-0.95: 65% of balance
- 0.85-0.90: 55% of balance
- 0.80-0.85: 45% of balance
- 0.75-0.80: 35% of balance
- 0.70-0.75: 25% of balance

### 9. **Signal Alignment Adjustment**
```
If signal_aligned = True:
  - Increase position by up to 25% based on confidence
  - Formula: base_percentage × (1.0 + confidence × 0.25)
  - Cap at 95% of balance

If signal_aligned = False:
  - Reduce position by 50%
  - Formula: base_percentage × 0.5
```

---

## 🚦 Trade Execution Logic

### 10. **Confidence Thresholds**
- **CONFIDENCE_THRESHOLD_TRADE = 0.70**: Minimum confidence to execute trade
- **CONFIDENCE_THRESHOLD_TEST = 0.70**: Trigger model refinement if below

### 11. **Trade Skipping Logic** (For Same Signal Type)
For each existing trade with same signal type:

**If entry is better:**
- Allow trade (no blocking)

**If entry is worse:**
- Check if confidence >= 0.9 AND confidence > existing trade's confidence
  - ✅ Yes: Allow trade (can override)
  - ❌ No: Block trade (add to blocking_trades list)

**Final Decision:**
- If blocking_trades list is empty → Execute trade
- If blocking_trades list has items → Skip trade

**Example:**
- Existing: BUY at $100, conf 0.85
- New: BUY at $105 (worse), conf 0.92
- Result: ✅ Execute (0.92 >= 0.9 and 0.92 > 0.85)

### 12. **Trade Execution**
```
If confidence >= 0.70:
  1. Check trade skipping logic
  2. If not skipped:
     - Assign trade index (increment trade_counter)
     - Store index in trade dictionary
     - Execute trade (real or simulated)
     - Add to active_trades list
```

---

## 🔄 Model Refinement Triggers

### 13. **Automatic Refinement Triggers**

#### 13.1 Hourly Refinement
- Triggered every 1 hour (3600 seconds)
- Uses recent 1 hour of data
- Creates test model for 3-minute comparison

#### 13.2 Trade Loss Trigger
- When any trade hits stop loss
- Retrains with recent 1 hour data
- Starts 3-minute testing phase

#### 13.3 Low Confidence Trigger
- When confidence < 0.70
- Retrains with recent 1 hour data
- Starts 3-minute testing phase

### 14. **Model Testing Phase** (3 minutes)
```
1. Both current and test models make predictions
2. Compare predictions side-by-side
3. After 3 minutes, evaluate which model performed better
4. If test model is better → Adopt it as new current model
5. If current model is better → Keep current model
```

---

## 📝 Logging & Tracking

### 15. **Log Files**

#### 15.1 `trading_bot.log`
- All console output
- Prediction details
- Trade execution logs
- Model refinement logs
- Error messages

#### 15.2 `trades_log.json`
- Closed trades only (not active trades)
- Structure:
  ```json
  {
    "index": 1,
    "time_stamp": "2025-01-15 10:30:00",
    "Profit/loss": "PROFIT",
    "PL percentage": 2.5,
    "PL in $": 125.50,
    "entry price": 95000.00
  }
  ```

#### 15.3 `trading_stats.json`
- Daily trading statistics
- Total profits/losses
- Win rate

### 16. **Trade Index Persistence**
- Each trade gets unique index when created
- Index stored in trade dictionary
- Indices persist even when trades are closed
- Example: Trades #1, #2, #3, #4, #5 → If #3 closes → Still shows #1, #2, #4, #5

---

## ⚙️ Configuration Parameters

### 17. **Risk Management**
- `MAX_POSITION_RISK = 0.10`: Max 10% of balance at risk per trade
- `MAX_LEVERAGE = 75`: Maximum leverage
- `EARLY_STOP_MAX_TIME_MINUTES = 120`: 2 hours for early stop
- `EARLY_STOP_OPPOSITE_SIGNAL = True`: Enable opposite signal check

### 18. **Model Parameters**
- `look_back = 60`: 60 timesteps (minutes) of historical data
- `future_horizon = 10`: Predict 10 minutes ahead for TP/SL
- `input_size = 93`: 93 technical indicators
- Transformer architecture: 4 layers, 8 attention heads, 128 d_model

---

## 🔍 Decision Tree Summary

### Trade Execution Decision Tree
```
1. Fetch data & make prediction
   ↓
2. Check active trades (SL/TP/Early Stop)
   ↓
3. Calculate TP-based signal
   ↓
4. Is confidence >= 0.70?
   ├─ No → Skip trade, check for refinement trigger
   └─ Yes → Continue
       ↓
5. Check trade skipping logic
   ├─ Entry better? → Execute
   ├─ Entry worse but confidence >= 0.9 and > all existing? → Execute
   └─ Entry worse and confidence not high enough? → Skip
       ↓
6. Calculate position size (confidence + signal alignment)
   ↓
7. Execute trade (real or simulated)
   ↓
8. Store trade with index
```

### Early Stop Decision Tree
```
For each active trade:
1. Is trade losing (red)?
   ├─ No → No early stop
   └─ Yes → Continue
       ↓
2. Has it been red for ≥ 2 hours?
   ├─ No → No early stop
   └─ Yes → Continue
       ↓
3. Is model predicting opposite signal?
   ├─ No → No early stop
   └─ Yes → Early stop (close trade)
```

### Model Refinement Decision Tree
```
1. Is it time for hourly refinement?
   ├─ Yes → Trigger refinement
   └─ No → Continue
       ↓
2. Did a trade hit stop loss?
   ├─ Yes → Trigger refinement
   └─ No → Continue
       ↓
3. Is confidence < 0.70?
   ├─ Yes → Trigger refinement
   └─ No → Continue normal operation
```

---

## 🎯 Key Features

1. **TP-Based Trading**: Always trades based on predicted TP direction, not model's signal
2. **Confidence Override**: Allows worse entry if confidence >= 0.9 and higher than existing
3. **Persistent Trade Indices**: Trade numbers stay consistent even after closures
4. **Early Stop Protection**: Closes losing trades after 2 hours if model predicts opposite
5. **Automatic Refinement**: Hourly + on loss + on low confidence
6. **Signal Alignment**: Adjusts position size based on model/TP signal match
7. **Comprehensive Logging**: All trades and predictions logged for analysis

---

## 📊 Model Architecture

- **Type**: Transformer-based (replaces LSTM)
- **Input**: 93 technical indicators
- **Sequence Length**: 60 timesteps
- **Architecture**:
  - 4 transformer encoder layers
  - 8 attention heads
  - 128 model dimension
  - 512 feed-forward dimension
  - Positional encoding for temporal information
  - Attention-weighted pooling with time decay bias

**Advantages over LSTM:**
- No forgetting phenomenon
- Parallel computation
- Better long-range dependency capture
- All-to-all attention mechanism

---

## 🔐 Safety Features

1. **TEST Mode**: Default enabled, simulates trades without real money
2. **Confidence Thresholds**: Only trades with high confidence (≥70%)
3. **Position Limits**: Max 95% of balance, minimum $10 per trade
4. **Early Stop**: Protects against prolonged losses
5. **Stop Loss**: Always set (20% for normal trades)
6. **Trade Skipping**: Prevents opening worse trades unless confidence is very high

---

## 📈 Expected Behavior

### Normal Operation
- Makes predictions every 60 seconds
- Executes trades when confidence >= 0.70
- Manages multiple active trades simultaneously
- Closes trades at TP or SL
- Refines model every hour

### High Confidence Override
- If entry is worse but confidence >= 0.9 and > existing → Still executes
- Logs override reason clearly

### Early Stop
- Closes losing trades after 2 hours if model predicts opposite signal
- Prevents holding losing positions too long

### Model Refinement
- Automatically retrains with recent data
- Tests new model for 3 minutes
- Adopts better performing model

---

## 🚨 Important Notes

1. **TEST Mode**: Always test in simulation mode first (`TEST = True`)
2. **Model Training**: Must train model before running (`./run.sh train`)
3. **API Keys**: Configure in `config.api` for live trading
4. **Balance**: Ensure sufficient USDT balance for trading
5. **Monitoring**: Regularly check `trading_bot.log` for errors
6. **Model Files**: Stored in `../model/` directory

---

## 📝 Workflow Summary

```
START
  ↓
Initialize (Load model, scalers, setup logging)
  ↓
[LOOP: Every 5 seconds]
  ↓
Fetch market data → Calculate indicators → Make prediction
  ↓
Check active trades (SL/TP/Early Stop)
  ↓
Process signal (TP-based) → Calculate TP/SL
  ↓
Calculate position size (confidence + alignment)
  ↓
Check trade skipping logic
  ↓
Execute trade if conditions met
  ↓
Check for model refinement triggers
  ↓
Wait 5 seconds → [LOOP]
```

---

**Last Updated**: Based on current codebase implementation
**Model**: Transformer-based with 93 technical indicators
**Trading Strategy**: TP-based signal with confidence override

