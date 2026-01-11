# 🚀 AI Crypto High-Frequency Trading Bot

**A state-of-the-art high-frequency trading bot powered by a Transformer-based neural network for BTC/USDT futures.**

This project utilizes a custom **Transformer** architecture (replacing the previous LSTM model) to predict short-term price movements on the **1-minute timeframe**. It features an advanced **online learning** system that continuously adapts to changing market conditions by retraining itself in real-time.

---

## ✨ Key Features

### 🧠 Advanced AI Model
* **Transformer Architecture**: Uses Multi-Head Self-Attention and Positional Encoding to capture complex temporal dependencies better than traditional RNNs/LSTMs.
* **Comprehensive Input**     : Analyzes **93 technical indicators** including RSI, MACD, Bollinger Bands, Ichimoku Cloud, Volume patterns, and more.
* **Multi-Objective Output**  : Simultaneously predicts                           : 
* **Direction**               : BUY/SELL signal probability.
* **Targets**                 : Dynamic Take Profit (TP) and Stop Loss (SL) levels optimized for the current volatility.

### 🔄 Adaptive & Self-Healing
*   **Online Learning**: The bot automatically retrains itself every hour (or after a losing trade) using the most recent market data to adapt to new trends.
*   **A/B Testing (Shadow Mode)**: When a new model is trained, it runs in "shadow mode" alongside the live model. The bot only swaps to the new model if it empirically outperforms the current one over a testing period.
*   **Crash Recovery**: Active trades are persisted to disk (`current_trades.json`), allowing the bot to resume management seamlessly after a restart.

### 🛡️ Robust Risk Management
*   **Dynamic Position Sizing**: Adjusts trade size based on model confidence and signal alignment. Higher confidence = larger position.
*   **Smart Exits**:
    *   **Take Profit / Stop Loss**: Dynamic targets per trade.
    *   **Early Stop**: Automatically closes losing trades that stagnate or if the market sentiment flips (opposite signal detected).
    *   **Profit Protection**: Secures profits by closing positions early if the model predicts a reversal.
*   **Confidence Thresholds**: only executes trades when model confidence exceeds strict thresholds (default > 70%).

---

## 🏗️ Model Architecture

The core of the predictor is the `CryptoPredicter` class in `model.py`, a **Transformer-based sequence model**:

*   **Input**:
    *   Sequence of 1‑minute candles with ~93 engineered features.
    *   Shape: `(batch, seq_len, input_size)`.
*   **Feature Projection**:
    *   Linear layer `input_projection: ℝ^{input_size} → ℝ^{d_model}` with `d_model=128`.
*   **Positional Encoding**:
    *   Sinusoidal positional encoding (Vaswani et al. 2017) via `PositionalEncoding(d_model, max_len=5000)`.
    *   Injects time/ordering information into the feature embeddings.
*   **Transformer Encoder Stack**:
    *   `num_layers=4` encoder blocks.
    *   Each block has:
        *   Multi‑Head Self‑Attention (`nhead=8`, batch‑first).
        *   Position‑wise feed‑forward MLP (`128 → 512 → 128`) with ReLU + Dropout.
        *   Residual connections + LayerNorm around both attention and MLP.
*   **Attention‑Weighted Pooling with Time Decay**:
    *   Learnable attention scores per timestep (small MLP → scalar score).
    *   Added to an exponential time‑decay bias so **recent timesteps are favored**.
    *   Softmax over time → attention weights → weighted sum of encoder outputs → single vector per sequence.
*   **Multi‑Head Output** (from the pooled representation):
    *   `signal_head`: Linear → 2‑class softmax for **SELL / BUY** probabilities.
    *   `tp_head`: Linear → 1 value for **Take Profit price**.
    *   `sl_head`: Linear → 1 value for **Stop Loss price**.
    *   In `predict_live.py`, **confidence** is defined as `max(signal_probs)` (the higher of SELL/BUY probabilities).
*   **Training Objective**:
    *   Joint loss = classification loss (CrossEntropy for signal) + regression loss (MSE for TP + SL).
    *   Optional **time‑weighted training**: more recent samples get higher weight in the loss, so the model adapts faster to current market regimes.

In short, the model looks at a rolling window of 1‑minute data, attends over the whole sequence, favors more recent context, and outputs both **direction (BUY/SELL)** and **price targets (TP/SL)** for each new decision point.

---

## 🚀 How to Use

### Prerequisites
*   Python 3.11.0
*   MEXC Account (for API access)

### 1. Installation
Clone the repo and install dependencies using the helper script:

```bash
# Clone the repository
git clone <repository-url>
cd crypto-predicter

# Install dependencies
./run.sh install
```

### 2. Configuration
Create a `config.api` file in the root directory to store your MEXC credentials:

```ini
MEXC_API_KEY=your_api_key_here
MEXC_API_SECRET=your_api_secret_here
```

> **Note**: The bot runs in **TEST MODE** (simulated paper trading) by default. To enable real trading, open `predict_live.py` and set `TEST = False`.

### 3. Training
Before running the bot, you must train the initial model:

```bash
./run.sh train
```
This will:
1.  Fetch ~35 days of 1-minute historical data from MEXC.
2.  Generate technical indicators.
3.  Train the Transformer model.
4.  Save the model to `model/btc_predicter_model.pth`.

### 4. Start Trading
Launch the live trading engine:

```bash
./run.sh start
```

### Management Commands
The `run.sh` script helps you manage the bot:

*   `./run.sh logs`: View the live logs (`trading_bot.log`).
*   `./run.sh stats`: View daily performance statistics (`trading_stats.json`).
*   `./run.sh clean`: Delete all generated files (models, logs, data) to start fresh.

---

## 🆕 Recent Updates

*   **Architecture Overhaul**: Migrated from LSTM to **Transformer** for better long-range dependency capturing and parallel processing.
*   **Persistence**: Added `current_trades.json` to save active trade state, preventing loss of position management during restarts.
*   **Auto-Tuning**: Implemented an automated pipeline that fine-tunes the model on the latest 500 minutes of data every hour.
*   **Shadow Testing**: Added a parallel testing environment where new models compete against the active model before deployment.
*   **Enhanced Logging**: Detailed JSON logging for individual trades (`trades_log.json`) and active positions.
*   **Early-Stop Logic** (2025-11-30): Added smart early-stop rules that:
    *   Close trades when PnL ≥ **0.15%** in the opposite TP-based direction (profit lock-in).
    *   Close trades that stay in loss for **≥ 5 hours** *and* the model predicts the opposite direction.
*   **Risk Stats & PnL Tracking** (2025-12-02):
    *   `trading_stats.json` now aggregates daily results (wins, losses, PnL, win rate) from `trades_log.json`.
    *   In **TEST mode**, a mock **$1000 balance** is tracked and updated after each trade to simulate real account behavior.
*   **Stability & Dependency Fixes** (2025-12-01): Cleaned up indentation issues and updated dependencies to keep the environment reproducible.

---

## ⚠️ Disclaimer

**USE AT YOUR OWN RISK.**

This software is for educational purposes only. Cryptocurrency trading, especially high-frequency futures trading, involves significant risk of financial loss. The authors and contributors are not responsible for any financial losses incurred through the use of this bot. Always test thoroughly in simulation mode before using real funds.
