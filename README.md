# 🚀 AI Crypto High-Frequency Trading Bot

**A state-of-the-art high-frequency trading bot powered by a Transformer-based neural network for BTC/USDT futures trading on MEXC.**

This project utilizes a custom **Transformer** architecture to predict short-term price movements on the **1-minute timeframe**. It features an advanced **online learning** system that continuously adapts to changing market conditions by retraining itself in real-time with sophisticated risk management.

---

## ✨ Key Features

### 🧠 Advanced AI Model
* **Transformer Architecture**: Uses Multi-Head Self-Attention and Positional Encoding to capture complex temporal dependencies better than traditional RNNs/LSTMs.
* **Comprehensive Input**: Analyzes **93 technical indicators** including RSI, MACD, Bollinger Bands, Ichimoku Cloud, Volume patterns, and more.
* **Multi-Objective Output**: Simultaneously predicts:
  * **Direction**: BUY/SELL signal probability with uncertainty quantification via Monte Carlo Dropout.
  * **Targets**: Dynamic Take Profit (TP) and Stop Loss (SL) levels optimized for current volatility.
* **Confidence-Based Trading**: Only executes trades when model confidence exceeds 70% threshold.

### 🔄 Adaptive & Self-Healing
* **Online Learning**: Automatically retrains every hour or after losing trades using the most recent market data.
* **A/B Testing (Shadow Mode)**: New models run in parallel with the live model for 3 minutes before deployment.
* **Model Comparison**: Automatically adopts new models only if they empirically outperform based on:
  - Average confidence levels
  - High-confidence prediction ratio
  - Consistency (lower variance)
* **Crash Recovery**: Active trades persisted to disk (`current_trades.json`) for seamless restart recovery.
* **Trade Reconciliation**: On startup, checks historical data to close trades that hit TP/SL while bot was offline.

### 🛡️ Advanced Risk Management (Updated 2025-12-16)

#### Multi-Layer Protection System
* **Global Risk Limits**:
  - Maximum daily loss: $500 (circuit breaker)
  - Maximum concurrent trades: 3 positions
  - Total exposure cap: 50% of balance
  - Consecutive loss protection (warnings after 3+ losses)

* **Per-Trade Risk Controls**:
  - Hard stop loss: $7 per trade (immediate exit)
  - Signal reversal minimum: $3 loss threshold
  - Time-based exit: 5 hours maximum for losing trades
  - Position sizing: 2-7% of balance based on confidence

* **Dynamic Leverage** (Aggressive Profile):
  - Perfect confidence (≥0.98): 15x leverage
  - Very high (≥0.90): 12x leverage
  - High (≥0.85): 10x leverage
  - Moderate (≥0.80): 8x leverage
  - Good (≥0.75): 6x leverage
  - Minimum: 5x leverage

* **Smart Position Sizing**:
  - Scales with confidence: 2% to 7% of balance
  - Signal alignment bonus: +50% when model and TP signals agree
  - Signal mismatch penalty: -40% when signals disagree
  - Maximum position with bonus: 10% of balance

* **Intelligent Exit Strategies**:
  - **Hard Stop**: Immediate exit at $7 loss
  - **Time Stop**: Close losing trades after 5 hours
  - **Signal Reversal**: Exit on opposite signal with $3+ loss
  - **Profit Protection**: Lock in profits (≥0.15%) on opposite TP signal
  - **Risk Management**: Close worst-entry profitable trades on opposite signals

---

## 🏗️ Model Architecture

The core predictor is the [`CryptoPredicter`](model.py) class, a **Transformer-based sequence model**:

### Input Processing
* Sequence of 1-minute candles with 93 engineered features
* Shape: `(batch, seq_len, input_size)`
* Feature projection: ℝ^{input_size} → ℝ^{128} (d_model)

### Transformer Encoder
* **4 encoder layers** with:
  - Multi-Head Self-Attention (8 heads, batch-first)
  - Position-wise FFN (128 → 512 → 128) with ReLU + Dropout
  - Residual connections + LayerNorm
* **Positional Encoding**: Sinusoidal encoding (max_len=5000)
* **Attention-Weighted Pooling**: 
  - Learnable attention with exponential time-decay
  - Recent timesteps favored in predictions

### Output Heads
* **Signal Head**: 2-class softmax (SELL/BUY probabilities)
* **TP Head**: Regression for Take Profit price
* **SL Head**: Regression for Stop Loss price
* **Uncertainty**: Monte Carlo Dropout (15 samples) for confidence estimation

### Training
* Joint loss: CrossEntropy (signal) + MSE (TP + SL)
* Time-weighted training: Recent samples get higher loss weight
* Label smoothing: 0.1 for better generalization
* Fine-tuning: Hourly updates with latest 2000 rows

---

## 📊 Project Structure

```
crypto-predicter/
├── predict_live.py       # Main trading engine with risk management
├── train.py              # Model training and fine-tuning
├── model.py              # Transformer architecture definition
├── data_processor.py     # Technical indicator calculation
├── mexc_client.py        # MEXC Futures API client
├── exchange.py           # Order execution wrapper
├── run.sh                # Management script
├── requirements.txt      # Python dependencies
├── config.api            # API credentials (create this)
├── model/                # Saved models and scalers
│   ├── btc_predicter_model.pth
│   ├── btc_predicter_model_test.pth
│   ├── scaler.gz
│   └── close_scaler.gz
├── trading_bot.log       # Execution logs
├── trading_stats.json    # Daily performance statistics
├── trades_log.json       # Individual trade records
├── current_trades.json   # Active trade persistence
└── training_data.csv     # Historical market data
```

---

## 🚀 Quick Start

### Prerequisites
* Python 3.11.0+
* MEXC Futures Account with API access
* Minimum $100 balance recommended (starts with $1000 in TEST mode)

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd crypto-predicter

# Install dependencies
./run.sh install
```

### 2. Configuration

Create a `config.api` file in the root directory:

```ini
MEXC_API_KEY=your_api_key_here
MEXC_API_SECRET=your_api_secret_here
```

> **⚠️ Important**: The bot runs in **TEST MODE** (paper trading) by default.  
> To enable real trading, open [`predict_live.py`](predict_live.py:32) and set `TEST = False`.

### 3. Initial Training

Train the model with historical data:

```bash
./run.sh train
```

This process:
1. Fetches ~100,000 1-minute candles (~2 months) from MEXC
2. Calculates 93 technical indicators
3. Generates trading signals based on TP/SL thresholds
4. Trains the Transformer model (20 epochs)
5. Saves model and scalers to `model/` directory

**Training time**: ~5-10 minutes depending on hardware

### 4. Start Trading

Launch the live trading engine:

```bash
./run.sh start
```

The bot will:
- Load the trained model
- Connect to MEXC Futures API
- Start monitoring BTC/USDT 1-minute data
- Execute trades based on model predictions
- Auto-retrain every hour and after losses
- Persist all trades for recovery

### 5. Monitor Performance

```bash
# View live logs
./run.sh logs

# Check daily statistics
./run.sh stats

# View active trades
cat current_trades.json

# View trade history
cat trades_log.json
```

---

## 🎛️ Risk Configuration

Edit [`predict_live.py`](predict_live.py:84-108) to adjust risk parameters:

```python
# Trading Thresholds
CONFIDENCE_THRESHOLD_TRADE = 0.70  # Min confidence to trade (70%)
CONFIDENCE_THRESHOLD_TEST = 0.55   # Trigger retraining below 55%

# Position Sizing
MAX_POSITION_RISK = 0.07           # Max 7% of balance per trade
MAX_LEVERAGE = 75                  # Exchange cap (used up to 15x)

# Stop Loss
EARLY_STOP_LOSS_THRESHOLD = 7.0    # Hard stop at $7 loss
MIN_LOSS_FOR_OPPOSITE_SIGNAL = 3.0 # Min loss for signal reversal
EARLY_STOP_MAX_TIME_MINUTES = 300  # 5-hour time limit

# Global Limits
MAX_DAILY_LOSS = 500.0             # Daily loss circuit breaker
MAX_CONCURRENT_TRADES = 3          # Position limit
MAX_TOTAL_EXPOSURE_PCT = 0.50      # 50% max total exposure
```

---

## 📈 Performance Tracking

### Daily Statistics (`trading_stats.json`)
```json
{
  "2025-12-16": {
    "date": "2025-12-16",
    "successful_trades": 12,
    "failed_trades": 3,
    "total_profit_usdt": 45.30,
    "win_rate_pct": 80.0,
    "balance": 1045.30
  }
}
```

### Trade Log (`trades_log.json`)
Each trade records:
- Entry/exit prices and times
- Leverage and margin used
- Expected vs actual P&L
- Confidence level
- Signal type (BUY/SELL)
- Early stop flag

---

## 🔧 Advanced Features

### Monte Carlo Dropout
The model uses MC Dropout with 15 samples to estimate prediction uncertainty:
```python
signal, probs, tp, sl, uncertainty = model.predict(X, mc_samples=15)
effective_confidence = confidence * (1 - uncertainty_score)
```

### Signal Alignment
The bot uses two signals:
1. **Model Signal**: Direct classification output
2. **TP-Based Signal**: Inferred from predicted TP direction

When signals **align**: Position size increases by up to 50%  
When signals **mismatch**: Position size reduces by 40%

### Trade Reconciliation
On startup, the bot:
1. Loads persisted trades from `current_trades.json`
2. Fetches historical data since trade entry
3. Checks if TP/SL was hit while offline
4. Automatically closes completed trades
5. Updates statistics and balance

---

## 🆕 Recent Updates

### 2025-12-16: Risk Management Overhaul
* ✅ Standardized loss thresholds ($7 hard stop)
* ✅ Added global risk management system
* ✅ Implemented multi-layer protection (daily loss, exposure limits)
* ✅ Simplified early stop logic (3 independent conditions)
* ✅ Updated to aggressive profile (15x max leverage, 7% positions)
* ✅ Enhanced position sizing with signal alignment bonuses
* ✅ Added consecutive loss protection warnings

### 2025-12-02: Statistics & Balance Tracking
* Daily performance aggregation from trade logs
* Mock balance tracking in TEST mode ($1000 starting balance)
* Win rate and P&L calculations

### 2025-11-30: Smart Exit Strategies
* Early profit-taking on opposite TP signals (≥0.15% profit)
* Time-based stops for stagnant losing trades (5 hours)
* Risk management for profitable trades

### 2025-11-25: Model Architecture
* Migrated from LSTM to Transformer architecture
* Added shadow testing for model deployment
* Implemented hourly auto-retraining
* Enhanced logging and persistence

---

## 🛠️ Troubleshooting

### Model Not Trading
- Check confidence levels in logs (need ≥70%)
- Verify global risk limits aren't blocking trades
- Ensure daily loss hasn't exceeded $500
- Check if already at max 3 concurrent positions

### High Loss Rate
- Consider reducing leverage in [`determine_leverage()`](predict_live.py:956)
- Tighten stop loss (reduce `EARLY_STOP_LOSS_THRESHOLD`)
- Increase `CONFIDENCE_THRESHOLD_TRADE` to 0.75 or 0.80
- Reduce `MAX_POSITION_RISK` for smaller positions

### Model Not Retraining
- Check `REFINEMENT_INTERVAL_SECONDS` (default 3600 = 1 hour)
- Verify `training_data.csv` exists and is growing
- Look for errors in logs during retraining attempts

### Trades Not Recovering After Restart
- Verify `current_trades.json` exists and is valid JSON
- Check trade fields: entry_price, entry_time, tp, sl, signal
- Review reconciliation logs for TP/SL hit detection

---

## ⚠️ Disclaimer

**USE AT YOUR OWN RISK.**

This software is for **educational purposes only**. Cryptocurrency futures trading, especially with leverage, involves **extreme risk of financial loss**. You can lose more than your initial investment due to liquidation.

**Key Risks:**
- 15x leverage means 1% price move = 15% gain/loss on margin
- Multiple concurrent trades can compound losses quickly
- Market volatility can trigger rapid liquidation
- API failures can prevent timely exits
- Model predictions are not guarantees

**Recommendations:**
1. **Always start in TEST mode** to understand behavior
2. **Never invest more than you can afford to lose**
3. **Monitor trades actively**, especially with high leverage
4. **Start with lower leverage** (5x) and increase gradually
5. **Keep emergency fund** separate from trading balance

The authors and contributors are **not responsible** for any financial losses incurred through the use of this software. Trading results shown are historical and do not guarantee future performance.

**Trade responsibly. Test thoroughly. Risk only what you can afford to lose.**

---

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📧 Support

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Happy Trading! 🚀📈**
