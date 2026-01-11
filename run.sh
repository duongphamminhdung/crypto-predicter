#!/bin/bash

# ═══════════════════════════════════════════════════════════
# Crypto Trading Bot - Management Script
# ═══════════════════════════════════════════════════════════
# This script provides commands to manage your crypto trading bot:
# - Install dependencies
# - Train the initial model
# - Start the live trading bot
# - Clean up generated files
# ═══════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    printf "${BLUE}ℹ️  $1${NC}\n"
}

print_success() {
    printf "${GREEN}✅ $1${NC}\n"
}

print_warning() {
    printf "${YELLOW}⚠️  $1${NC}\n"
}

print_error() {
    printf "${RED}❌ $1${NC}\n"
}

print_header() {
    printf "\n${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
    printf "${BLUE}  $1${NC}\n"
    printf "${BLUE}═══════════════════════════════════════════════════════════${NC}\n\n"
}

  # Function to check if Python is installed
check_python() {
    
    # Use python3 if available, otherwise python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
    else
        PYTHON_CMD="python"
        PIP_CMD="pip"
    fi
    
    print_success "Python detected: $($PYTHON_CMD --version)"
}

# Function to install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    check_python
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    print_info "Installing packages from requirements.txt..."
    $PIP_CMD install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_success "All dependencies installed successfully!"
    else
        print_error "Failed to install dependencies. Please check your Python environment."
        exit 1
    fi
}

# Function to train the model
train_model() {
    print_header "Training Model"
    
    check_python
    
    print_info "Starting initial model training..."
    print_info "This will:"
    print_info "  • Fetch ~100,000 minutes (~2 months) of BTC/USDT Min1 data"
    print_info "  • Calculate 90+ technical indicators & features"
    print_info "  • Train a 4-layer Transformer model for 20 epochs"
    print_info "  • Save weights to ../model/btc_predicter_model.pth"
    echo ""
    
    $PYTHON_CMD train.py
    
    if [ $? -eq 0 ]; then
        print_success "Model training completed successfully!"
        print_success "Model saved as: ../model/btc_predicter_model.pth"
    else
        print_error "Training failed. Please check the logs for errors."
        exit 1
    fi
}

SCRIPT_NAME="predict_live.py"
PID_FILE="bot_process.pid"
LOG_FILE="trading_bot.log"

start_bot() {
    print_header "Starting Live Trading Bot"
    
    check_python
    
    # 1. Check if already running (Safety Mechanism)
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null; then
            print_warning "Bot is already running with PID: $OLD_PID"
            print_info "Stop it first using: ./run.sh stop"
            return
        else
            # File exists but process is dead -> Clean it up
            rm "$PID_FILE"
        fi
    fi

    # Check if model exists in ../model directory
    MODEL_DIR="../model"
    if [ ! -f "$MODEL_DIR/btc_predicter_model.pth" ]; then
        print_warning "Model file not found!"
        print_info "You need to train the model first. Run: ./run.sh train"
        exit 1
    fi
    
    # Check TEST mode
    if grep -q "TEST = True" predict_live.py; then
        print_warning "TEST MODE is ENABLED - No real money will be used"
        print_info "To enable live trading, edit predict_live.py and set: TEST = False"
    else
        print_warning "LIVE TRADING MODE - Real money will be used!"
        print_warning "Press Ctrl+C within 5 seconds to cancel..."
        sleep 5
    fi
    
    print_info "Starting the bot..."
    print_info "The bot will:"
    print_info "  • Make predictions every 60 seconds"
    print_info "  • Use 60-minute input window with 22 technical indicators"
    print_info "  • Execute trades when confidence ≥ 0.65"
    print_info "  • Auto fine-tune with time-weighted learning when confidence < 0.4 or trade loss"
    print_info "  • Log all activity to trading_bot.log"
    
    print_info "Starting the bot in BACKGROUND..."
    
    # 2. THE FIX: Run with nohup, in background (&), and save PID
    nohup $PYTHON_CMD "$SCRIPT_NAME" > "$LOG_FILE" 2>&1 &
    
    # 3. Save the PID immediately
    NEW_PID=$!
    echo $NEW_PID > "$PID_FILE"
    
    print_info "Bot started successfully! (PID: $NEW_PID)"
    print_info "You can now run other cells. To stop, run: !./run.sh stop"
}

stop_bot() {
    print_header "Stopping Trading Bot"
    
    # 1. Check for PID file
    if [ ! -f "$PID_FILE" ]; then
        print_warning "No PID file found ($PID_FILE)."
        print_info "Falling back to name search..."
        # Fallback to your old method if PID file is missing
        pkill -f "$SCRIPT_NAME" && print_success "Killed by name." || print_warning "No process found."
        return
    fi
    
    TARGET_PID=$(cat "$PID_FILE")
    
    # 2. Check if that specific process is running
    if ps -p "$TARGET_PID" > /dev/null; then
        print_info "Found bot process (PID: $TARGET_PID). Stopping..."
        
        # Polite kill (SIGTERM)
        kill "$TARGET_PID"
        
        # Wait up to 5 seconds for graceful shutdown
        for i in {1..5}; do
            if ! ps -p "$TARGET_PID" > /dev/null; then
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if ps -p "$TARGET_PID" > /dev/null; then
            print_warning "Bot refused to stop. Forcing kill..."
            kill -9 "$TARGET_PID"
        fi
        
        rm "$PID_FILE"
        print_success "Bot stopped."
    else
        print_warning "Process $TARGET_PID is not running. Cleaning up stale PID file."
        rm "$PID_FILE"
    fi
}

refresh_sentiment() {
    local script_path="./news/test_backend.py"

    echo ""
    echo "--- 🚀 Refreshing Sentiment Data ---"

    check_python

    # 1. Check if the file actually exists
    if [ ! -f "$script_path" ]; then
        echo "❌ Error: File not found."
        echo "   Please check the path."
        return 1
    fi

    # 2. Run the script using Python
    # We use 'python3' to be explicit, though 'python' usually works too
    $PYTHON_CMD "$script_path"

    # Capture the exit code of the python script (0 = success, anything else = error)
    local exit_code=$?

    # 3. Report the result
    if [ $exit_code -eq 0 ]; then
        echo "✅ Execution finished successfully."
    else
        echo "🔥 Script crashed with exit code: $exit_code"
    fi
    echo "------------------------------------"
}

# Function to clean up generated files
clean() {
    print_header "Cleaning Up Generated Files"
    
    print_warning "This will delete:"
    print_warning "  • Trained models (*.pth) in ../model/"
    print_warning "  • Training data (*.csv)"
    print_warning "  • Scaler files (*.gz)"
    print_warning "  • Log files (*.log)"
    print_warning "  • Stats files (*.json)"
    echo ""
    
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing files..."
        rm -f ../model/*.pth
        rm -f ../model/*.gz
        rm -f *.csv
        rm -f *.log
        rm -f trading_stats.json
        print_success "Cleanup completed!"
    else
        print_info "Cleanup cancelled."
    fi
}

# Function to view logs
view_logs() {
    print_header "Viewing Trading Bot Logs"
    
    if [ ! -f "trading_bot.log" ]; then
        print_warning "No log file found. The bot hasn't been run yet."
        exit 1
    fi
    
    print_info "Showing last 50 lines of trading_bot.log"
    print_info "Press 'q' to exit"
    echo ""
    
    tail -n 50 trading_bot.log
}

# Function to view stats
view_stats() {
    print_header "Viewing Trading Statistics"
    
    if [ ! -f "trading_stats.json" ]; then
        print_warning "No stats file found. No trades have been executed yet."
        exit 1
    fi
    
    print_info "Current trading statistics:"
    echo ""
    cat trading_stats.json | python3 -m json.tool
}

# Function to show help
show_help() {
    print_header "Crypto Trading Bot - Command Reference"
    
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  install    - Install all required dependencies"
    echo "  train      - Train the initial model (required before starting bot)"
    echo "  start      - Start the live trading bot"
    echo "  stop         - Stop the running trading bot"
    echo "  sentiment    - Refresh sentiment data from news sources"
    echo "  logs       - View the last 50 lines of bot logs"
    echo "  stats      - View current trading statistics"
    echo "  clean      - Remove all generated files (models, data, logs)"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh install    # First time setup"
    echo "  ./run.sh train      # Train the model"
    echo "  ./run.sh start      # Start trading"
    echo "  ./run.sh stop         # Stop the bot"
    echo "  ./run.sh logs       # Check what's happening"
    echo "  ./run.sh sentiment  # Refresh sentiment data"
    echo ""
    echo "Quick Start:"
    echo "  1. ./run.sh install"
    echo "  2. ./run.sh train"
    echo "  3. ./run.sh start"
    echo ""
}

# Main script logic
case "${1:-help}" in
    install)
        install_dependencies
        ;;
    train)
        train_model
        ;;
    start)
        start_bot
        ;;
    stop)
        stop_bot
        ;;
    sentiment)
        refresh_sentiment
        ;;
    logs)
        view_logs
        ;;
    stats)
        view_stats
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
