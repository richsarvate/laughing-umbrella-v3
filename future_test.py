"""
2025 Future Testing Module for Stock Trading Transformer
Tests pattern learning by applying trained model to completely unseen 2025 data.
This validates whether the model learned generalizable market patterns vs. historical memorization.
"""

import os
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

# Import existing modules
from data_processor import MarketDataProcessor
from training_system import TrainingSystem
from model import TransformerStockTrader


class Future2025Tester:
    """Tests transformer model on unseen 2025 data to validate pattern learning."""
    
    def __init__(self):
        """Initialize tester with trained model and data processor."""
        # Load trained model
        self.model_path = "trained_stock_trader.pth"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at {self.model_path}. Train model first.")
        
        # Initialize components 
        self.data_processor = MarketDataProcessor(lookback_days=30)
        actual_num_stocks = len(self.data_processor.sp500_tickers)
        self.model = TransformerStockTrader(num_stocks=actual_num_stocks)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        
        print(f"✅ Loaded trained model with {actual_num_stocks} stocks")
    
    def get_random_2025_date(self) -> str:
        """Pick random date in 2025 with enough future data for 30-day test."""
        # Available range: Jan 1, 2025 to (current_date - 35 days) 
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 10, 25) - timedelta(days=35)  # Need 30+ days future data
        
        # Random date in valid range
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randint(0, days_between)
        
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%Y-%m-%d")
    
    def get_model_prediction(self, current_date: str) -> Tuple[str, str, float]:
        """Get model prediction for given date using sliding 30-day window."""
        try:
            # Download 2025 data up to current date (60 days buffer for features)
            start_date = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
            raw_market_data = self.data_processor.download_market_data(start_date, current_date)
            
            # Extract features using same method as training
            market_features = self.data_processor.extract_anonymous_features(raw_market_data)
            
            # Use last 30 days as input (same as training)
            if len(market_features) < 30:
                return "INSUFFICIENT_DATA", None, 0.0
                
            input_sequence = market_features[-30:].reshape(1, 30, -1, 3)
            input_tensor = torch.FloatTensor(input_sequence)
            
            # Make prediction
            with torch.no_grad():
                decision_logits = self.model(input_tensor)
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
                # Model chose specific stock
                stock_index = predicted_choice - 2
                if stock_index < len(self.data_processor.sp500_tickers):
                    action = "SWITCH"
                    target_stock = self.data_processor.sp500_tickers[stock_index]
                else:
                    action = "CASH"  # Fallback for out-of-range
                    target_stock = None
            
            return action, target_stock, confidence
            
        except Exception as e:
            print(f"⚠️  Error getting prediction for {current_date}: {str(e)}")
            return "ERROR", None, 0.0
    
    def calculate_5day_return(self, stock_ticker: str, start_date: str) -> float:
        """Calculate actual 5-day return for given stock and date."""
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = start_dt + timedelta(days=5)
            end_date = end_dt.strftime("%Y-%m-%d")
            
            # Download price data
            import yfinance as yf
            stock_data = yf.download(stock_ticker, start=start_date, end=end_date, progress=False)
            
            # Handle various data issues
            if stock_data is None or len(stock_data) == 0:
                return 0.0
            
            if hasattr(stock_data, 'empty') and stock_data.empty:
                return 0.0
                
            if len(stock_data) < 2:
                return 0.0
                
            # Extract prices safely
            start_price = float(stock_data['Close'].iloc[0])
            end_price = float(stock_data['Close'].iloc[-1])
            
            # Avoid division by zero
            if start_price == 0:
                return 0.0
            
            return float((end_price - start_price) / start_price)
            
        except Exception as e:
            print(f"⚠️  Error calculating return for {stock_ticker}: {str(e)}")
            return 0.0
    
    def test_30_days_from_date(self, start_date: str) -> Dict:
        """Test model performance over 30 days starting from given date."""
        print(f"🧪 Testing 2025 Pattern Learning: {start_date} → 30 days forward")
        print("⏳ Processing 30 days of predictions... (this may take a few minutes)")
        
        results = {
            'start_date': start_date,
            'daily_decisions': [],
            'daily_returns': [],
            'total_return': 0.0,
            'win_rate': 0.0,
            'profitable_days': 0,
            'total_days': 0
        }
        
        current_position = "CASH"
        cumulative_return = 0.0
        
        for day_offset in range(30):
            current_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            
            # Get model prediction
            action, target_stock, confidence = self.get_model_prediction(current_date)
            
            # Determine actual decision vs current position
            if action == "HOLD" and current_position != "CASH":
                decision_display = f"HOLD {current_position}"
                profit_stock = current_position
            elif action == "CASH":
                decision_display = "CASH"
                current_position = "CASH"
                profit_stock = None
            elif action == "SWITCH" and target_stock is not None:
                decision_display = f"SWITCH to {target_stock}"
                current_position = target_stock
                profit_stock = target_stock
            else:
                decision_display = "HOLD (default)"
                profit_stock = current_position if current_position != "CASH" else None
            
            # Calculate 5-day return
            day_return = 0.0
            if profit_stock is not None and profit_stock != "CASH":
                day_return = self.calculate_5day_return(profit_stock, current_date)
            
            # Create display string for file output
            return_display = f"{day_return:+.1%}" if day_return != 0 else "0.0%"
            day_result = f"Day {day_offset+1:2d} ({current_date}): {decision_display:<20} → {return_display}"
            
            # Track results
            results['daily_decisions'].append({
                'date': current_date,
                'decision': decision_display,
                'confidence': confidence,
                'return': day_return,
                'display': day_result
            })
            results['daily_returns'].append(day_return)
            
            cumulative_return += day_return
            if day_return > 0:
                results['profitable_days'] += 1
            results['total_days'] += 1
            

        
        # Calculate final metrics
        results['total_return'] = cumulative_return
        results['win_rate'] = results['profitable_days'] / results['total_days'] if results['total_days'] > 0 else 0
        
        return results
    
    def create_report_file(self, results: Dict):
        """Create a clean report file with daily actions and summary."""
        from datetime import datetime
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"2025_future_test_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("🧪 2025 FUTURE TEST REPORT - TRANSFORMER PATTERN LEARNING\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: TransformerStockTrader (trained on 2010-2024 data)\n")
            f.write(f"Test Period: {results['start_date']} → 30 trading days (completely unseen 2025 data)\n\n")
            
            f.write("DAILY ACTIONS & RESULTS:\n")
            f.write("-" * 70 + "\n")
            
            for decision in results['daily_decisions']:
                f.write(decision['display'] + "\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("📊 PERFORMANCE SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total Return: {results['total_return']:+.1%} (30 trading days)\n")
            f.write(f"Win Rate: {results['win_rate']:.1%} ({results['profitable_days']}/{results['total_days']} profitable days)\n")
            f.write(f"Average Daily Return: {np.mean(results['daily_returns']):.3%}\n")
            
            if len(results['daily_returns']) > 0:
                positive_returns = [r for r in results['daily_returns'] if r > 0]
                negative_returns = [r for r in results['daily_returns'] if r < 0]
                
                if positive_returns:
                    f.write(f"Average Gain: +{np.mean(positive_returns):.1%}\n")
                if negative_returns:
                    f.write(f"Average Loss: {np.mean(negative_returns):-.1%}\n")
            
            # Risk metrics
            max_return = max(results['daily_returns']) if results['daily_returns'] else 0
            min_return = min(results['daily_returns']) if results['daily_returns'] else 0
            f.write(f"Best Day: {max_return:+.1%}\n")
            f.write(f"Worst Day: {min_return:+.1%}\n")
            
            f.write("\n🎯 PATTERN LEARNING ASSESSMENT:\n")
            if results['total_return'] > 0.05:  # >5% over 30 days
                f.write("✅ EXCELLENT: Strong positive returns suggest good pattern learning\n")
                f.write("   Model successfully applied 2010-2024 learned patterns to unseen 2025 data.\n")
            elif results['total_return'] > 0.0:
                f.write("✅ GOOD: Positive returns indicate some pattern recognition\n")
            elif results['total_return'] > -0.02:  # Better than -2%
                f.write("⚠️  MARGINAL: Close to breakeven, mixed pattern learning\n")
            else:
                f.write("❌ POOR: Negative returns suggest pattern learning failed\n")
            
            f.write(f"\n📝 NOTES:\n")
            f.write("- This test validates whether the transformer learned generalizable market patterns\n")
            f.write("- All 2025 data was completely unseen during 2010-2024 training period\n")
            f.write("- Model uses anonymous features (momentum, volatility, RSI) to prevent ticker memorization\n")
            f.write("- Each decision uses 30-day sliding window, same as training methodology\n")
            f.write("- Returns calculated as actual 5-day forward performance\n")
        
        return filename
    
    def display_summary(self, results: Dict):
        """Display minimal console output and create detailed file report."""
        filename = self.create_report_file(results)
        
        print(f"\n✅ 2025 Future Test Completed!")
        print(f"📊 Total Return: {results['total_return']:+.1%} over 30 days")
        print(f"🎯 Win Rate: {results['win_rate']:.1%} ({results['profitable_days']}/{results['total_days']} profitable)")
        print(f"📄 Detailed report saved to: {filename}")
        
        if results['total_return'] > 0.05:
            print("🚀 EXCELLENT pattern learning performance!")
        elif results['total_return'] > 0.0:
            print("✅ Good pattern recognition results")
        else:
            print("⚠️  Pattern learning needs improvement")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Test 2025 future performance of trained stock trading model')
    parser.add_argument('--date', type=str, help='Specific start date (YYYY-MM-DD) instead of random')
    parser.add_argument('--runs', type=int, default=1, help='Number of random test runs')
    
    args = parser.parse_args()
    
    try:
        tester = Future2025Tester()
        
        if args.date:
            # Test specific date
            results = tester.test_30_days_from_date(args.date)
            tester.display_summary(results)
        else:
            # Test random date(s)
            for run in range(args.runs):
                if args.runs > 1:
                    print(f"\n🎲 Random Test Run {run + 1}/{args.runs}")
                
                random_date = tester.get_random_2025_date()
                results = tester.test_30_days_from_date(random_date)
                tester.display_summary(results)
                
                if run < args.runs - 1:
                    print("\n" + "-" * 70)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Train your model first using: python trader.py train")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()