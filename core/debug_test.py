"""
Simple 2025 Test - Debug Version
"""
import os
import torch
import numpy as np
from datetime import datetime, timedelta
from data_processor import MarketDataProcessor
from model import TransformerStockTrader

def simple_test():
    """Simple test to debug the 2025 future testing."""
    try:
        # Load model
        model_path = "trained_stock_trader.pth"
        if not os.path.exists(model_path):
            print("❌ No trained model found")
            return
            
        # Initialize components
        data_processor = MarketDataProcessor(lookback_days=30)
        actual_num_stocks = len(data_processor.sp500_tickers)
        model = TransformerStockTrader(num_stocks=actual_num_stocks)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        print(f"✅ Loaded model with {actual_num_stocks} stocks")
        
        # Test single prediction
        test_date = "2025-05-01"
        print(f"🧪 Testing single prediction for {test_date}")
        
        # Download data
        start_date = (datetime.strptime(test_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
        print(f"Downloading data from {start_date} to {test_date}")
        
        raw_market_data = data_processor.download_market_data(start_date, test_date)
        print(f"Downloaded data shape: {raw_market_data.shape if hasattr(raw_market_data, 'shape') else type(raw_market_data)}")
        
        # Extract features
        market_features = data_processor.extract_anonymous_features(raw_market_data)
        print(f"Features shape: {market_features.shape}")
        
        # Check if we have enough data
        if len(market_features) < 30:
            print(f"❌ Insufficient data: {len(market_features)} days, need 30")
            return
            
        # Prepare input
        input_sequence = market_features[-30:].reshape(1, 30, -1, 3)
        input_tensor = torch.FloatTensor(input_sequence)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            decision_logits = model(input_tensor)
            decision_probabilities = torch.softmax(decision_logits, dim=1)
            predicted_choice = torch.argmax(decision_probabilities, dim=1).item()
            confidence = torch.max(decision_probabilities).item()
        
        # Decode prediction
        if predicted_choice == 0:
            action = "HOLD"
            target_stock = None
        elif predicted_choice == 1:
            action = "CASH"
            target_stock = None
        else:
            stock_index = predicted_choice - 2
            if stock_index < len(data_processor.sp500_tickers):
                action = "SWITCH"
                target_stock = data_processor.sp500_tickers[stock_index]
            else:
                action = "CASH"
                target_stock = None
        
        print(f"✅ Prediction: {action} {target_stock if target_stock else ''} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()