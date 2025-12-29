#!/usr/bin/env python3
"""
Modify trading thresholds in predict_live.py
"""

import re
import os
from pathlib import Path

def modify_thresholds(file_path, new_trade_threshold=None, new_test_threshold=None,
                     new_risk=None, new_leverage=None):
    """Modify threshold values in predict_live.py"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match threshold assignments
    patterns = {
        'CONFIDENCE_THRESHOLD_TRADE': r'CONFIDENCE_THRESHOLD_TRADE\s*=\s*[\d.]+',
        'CONFIDENCE_THRESHOLD_TEST': r'CONFIDENCE_THRESHOLD_TEST\s*=\s*[\d.]+',
        'MAX_POSITION_RISK': r'MAX_POSITION_RISK\s*=\s*[\d.]+',
        'MAX_LEVERAGE': r'MAX_LEVERAGE\s*=\s*\d+'
    }

    replacements = {
        'CONFIDENCE_THRESHOLD_TRADE': f'CONFIDENCE_THRESHOLD_TRADE = {new_trade_threshold}' if new_trade_threshold else None,
        'CONFIDENCE_THRESHOLD_TEST': f'CONFIDENCE_THRESHOLD_TEST = {new_test_threshold}' if new_test_threshold else None,
        'MAX_POSITION_RISK': f'MAX_POSITION_RISK = {new_risk}' if new_risk else None,
        'MAX_LEVERAGE': f'MAX_LEVERAGE = {new_leverage}' if new_leverage else None
    }

    modified = False
    for key, pattern in patterns.items():
        if replacements[key]:
            if re.search(pattern, content):
                content = re.sub(pattern, replacements[key], content)
                print(f"✅ Updated {key} to {replacements[key].split(' = ')[1]}")
                modified = True
            else:
                print(f"⚠️  Could not find {key} in file")

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Successfully updated thresholds in {file_path}")
    else:
        print("\n⚠️  No thresholds were modified")

    return modified

def main():
    """Interactive threshold modifier"""
    file_path = 'predict_live.py'

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found in current directory")
        return

    print("Current Trading Thresholds Modifier")
    print("=" * 40)

    # Show current values
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    current_values = {}
    for line in content.split('\n'):
        if 'CONFIDENCE_THRESHOLD_TRADE' in line and '=' in line:
            current_values['trade'] = line.split('=')[1].strip()
        elif 'CONFIDENCE_THRESHOLD_TEST' in line and '=' in line:
            current_values['test'] = line.split('=')[1].strip()
        elif 'MAX_POSITION_RISK' in line and '=' in line:
            current_values['risk'] = line.split('=')[1].strip()
        elif 'MAX_LEVERAGE' in line and '=' in line:
            current_values['leverage'] = line.split('=')[1].strip()

    print("Current values:")
    for key, value in current_values.items():
        print(f"  {key.upper()}: {value}")

    print("\nEnter new values (press Enter to keep current):")

    try:
        trade_threshold = input("Trade Confidence Threshold (0.0-1.0): ").strip()
        trade_threshold = float(trade_threshold) if trade_threshold else None

        test_threshold = input("Test Confidence Threshold (0.0-1.0): ").strip()
        test_threshold = float(test_threshold) if test_threshold else None

        risk = input("Max Position Risk (0.0-1.0): ").strip()
        risk = float(risk) if risk else None

        leverage = input("Max Leverage (1-100): ").strip()
        leverage = int(leverage) if leverage else None

        if any([trade_threshold, test_threshold, risk, leverage]):
            modify_thresholds(file_path, trade_threshold, test_threshold, risk, leverage)
        else:
            print("No changes made.")

    except ValueError as e:
        print(f"Invalid input: {e}")

if __name__ == '__main__':
    main()