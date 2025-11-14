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
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

  # Function to check if Python is installed
check_python() {
    # if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    #     print_error "Python is not installed. Please install Python 3.8 or higher."
    #     exit 1
    # fi
    
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
    print_info "  • Fetch 5000 minutes (~3.5 days) of BTC/USDT data"
    print_info "  • Calculate 22 technical indicators"
    print_info "  • Train LSTM model with 20 epochs"
    print_info "  • Save model as btc_predicter_model.pth"
    echo ""
    
    $PYTHON_CMD train.py
    
    if [ $? -eq 0 ]; then
        print_success "Model training completed successfully!"
        print_success "Model saved as: btc_predicter_model.pth"
    else
        print_error "Training failed. Please check the logs for errors."
        exit 1
    fi
}

# Function to start the live bot
start_bot() {
    print_header "Starting Live Trading Bot"
    
    check_python
    
    # Check if model exists
    if [ ! -f "btc_predicter_model.pth" ]; then
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
    print_info "  • Save daily stats to trading_stats.json"
    echo ""
    print_info "Press Ctrl+C to stop the bot"
    echo ""
    
    $PYTHON_CMD predict_live.py
}

# Function to clean up generated files
clean() {
    print_header "Cleaning Up Generated Files"
    
    print_warning "This will delete:"
    print_warning "  • Trained models (*.pth)"
    print_warning "  • Training data (*.csv)"
    print_warning "  • Scaler files (*.gz)"
    print_warning "  • Log files (*.log)"
    print_warning "  • Stats files (*.json)"
    echo ""
    
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing files..."
        rm -f *.pth
        rm -f *.csv
        rm -f *.gz
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
    echo "  logs       - View the last 50 lines of bot logs"
    echo "  stats      - View current trading statistics"
    echo "  clean      - Remove all generated files (models, data, logs)"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh install    # First time setup"
    echo "  ./run.sh train      # Train the model"
    echo "  ./run.sh start      # Start trading"
    echo "  ./run.sh logs       # Check what's happening"
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
