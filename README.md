# Crypto Price Predictor using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict the next day's closing price of a cryptocurrency using historical data from the MEXC exchange.

**Note**: This bot is currently configured for high-frequency trading on **1-minute (`M1`)** intervals. This is a more intensive and potentially riskier strategy than trading on longer timeframes.

## Features
- Fetches historical candlestick data from MEXC.
- Preprocesses data for time-series forecasting.
- Builds and trains an LSTM model using TensorFlow/Keras.
- Predicts the next day's closing price.
- Generates a simple 'BUY' or 'SELL' trading signal.

## How to Run

### 1. Clone the repository
```bash
git clone <repository-url>
cd crypto-predicter
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
First, you need to train the model using the historical data.
```bash
python train.py
```
This will fetch the latest historical data, train the model, and save `btc_predicter_model.pth`, `scaler.gz`, and `training_data.csv` to your project directory.

### 5. Run Live Predictions
Once the model is trained, you can start the live prediction engine.
```bash
python predict_live.py
```
This script will load the trained model and scaler, and then run in a continuous loop, fetching the latest data and making a new prediction at the start of each new day.

### Online Learning
This model has an online learning capability.
-   If the model's prediction confidence is high (≥ 0.65), it will place a trade.
-   If the confidence is low (< 0.4), it will not trade but will instead "observe" the market.
-   If it observes that its low-confidence prediction would have been unprofitable, it will automatically add the corrected data to its training set and retrain itself to improve future performance.

### 6. Set Up for Automated Trading
To enable automated trading, you need to provide your MEXC API key and secret.

1.  Create a file named `config.api` in the root of the project.
2.  Add your API key and secret to this file in the following format:

```
MEXC_API_KEY=your_api_key_here
MEXC_API_SECRET=your_api_secret_here
```

**Important**: This file is included in `.gitignore` to prevent you from accidentally committing your secret keys.

## Disclaimer
This project is for educational purposes only and should not be used for making real financial decisions. The cryptocurrency market is highly volatile, and this model does not guarantee any profits. **Automated trading is extremely risky and can lead to significant financial losses. Use this feature at your own risk and only with funds you are prepared to lose.**

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://your-repo-link.com/crypto-predicter.git
    cd crypto-predicter
    ```

2.  **Set up API Keys:**
    - Rename `config.api.example` to `config.api`.
    - Edit the `config.api` file and add your MEXC API Key and Secret.

3.  **Install Dependencies:**
    ```bash
    ./run.sh install
    ```

4.  **Train the Initial Model:**
    ```bash
    ./run.sh train
    ```
    This will fetch historical data, train the model, and save it.

5.  **Run the Bot:**
    - The bot starts in **TEST mode** by default (no real money).
    ```bash
    ./run.sh start
    ```
    - To enable live trading, edit `predict_live.py` and set `TEST = False`.

## Management Script Usage

The `run.sh` script provides a convenient way to manage the bot.

### Commands
```
Usage: ./run.sh [command]

Commands:
  install    - Install all required dependencies
  train      - Train the initial model (required before starting bot)
  start      - Start the live trading bot
  logs       - View the last 50 lines of bot logs
  stats      - View current trading statistics
  clean      - Remove all generated files (models, data, logs)
  help       - Show this help message
```

### Examples
```bash
# First time setup
./run.sh install

# Train the model from scratch
./run.sh train

# Start the bot (runs in TEST mode by default)
./run.sh start

# Check the bot's activity in another terminal
./run.sh logs

# View daily performance stats
./run.sh stats

# Clean up all generated files
./run.sh clean
```