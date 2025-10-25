#!/usr/bin/env python3
"""
Ultra-minimal transformer-based stock trader for S&P 500.
Makes daily decisions: HOLD current stock, SWITCH to different stock, or go to CASH.
Uses anonymous features only - no ticker symbols to prevent memorization.

Main CLI interface that orchestrates the trading system components.
"""

import argparse
from datetime import datetime

from training_system import TrainingSystem


def main():
    """Command-line interface for the stock trader."""
    parser = argparse.ArgumentParser(description="Transformer Stock Trader")
    parser.add_argument('command', choices=['train', 'predict'], help='Train model or make prediction')
    parser.add_argument('--date', type=str, help='Date for prediction (YYYY-MM-DD)', 
                       default=datetime.now().strftime("%Y-%m-%d"))
    
    args = parser.parse_args()
    
    # Initialize the trading system
    trading_system = TrainingSystem()
    
    if args.command == 'train':
        trading_system.train_model()
    elif args.command == 'predict':
        action, target_stock = trading_system.predict_action(args.date)


if __name__ == "__main__":
    main()