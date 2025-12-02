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

The core of the predictor is a `CryptoPredicter` class based on `torch.nn.TransformerEncoder`.

*   **Embedding**: Projects 93 input features into a high-dimensional space (`d_model=128`).
*   **Positional Encoding**: Injects temporal order information since Transformers process data in parallel.
*   **Encoder Layers**: Stack of Transformer Encoder layers with Multi-Head Attention (`nhead=8`) to weigh the importance of different past time steps.
*   **Attention-Weighted Pooling**: Aggregates the sequence output with a time-decay bias, focusing more on recent events.

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

---

## ⚠️ Disclaimer

**USE AT YOUR OWN RISK.**

This software is for educational purposes only. Cryptocurrency trading, especially high-frequency futures trading, involves significant risk of financial loss. The authors and contributors are not responsible for any financial losses incurred through the use of this bot. Always test thoroughly in simulation mode before using real funds.
