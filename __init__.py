"""
StockTrader - Ultra-minimal transformer-based stock trading system.

A clean, modular implementation of a transformer model for S&P 500 trading decisions.
Uses anonymous features to prevent overfitting to specific tickers.

Modules:
    data_processor: Market data download and feature extraction
    model: Transformer architecture for trading decisions  
    training_system: Model training and prediction orchestration
    trader: CLI interface and main entry point

Usage:
    python trader.py train                    # Train the model
    python trader.py predict                  # Get today's action
    python trader.py predict --date 2024-10-24  # Specific date
"""

from .data_processor import MarketDataProcessor
from .model import TransformerStockTrader  
from .training_system import TrainingSystem

__version__ = "1.0.0"
__author__ = "StockTrader Development Team"

# Define what gets imported with "from stocktrader import *"
__all__ = [
    "MarketDataProcessor",
    "TransformerStockTrader", 
    "TrainingSystem"
]