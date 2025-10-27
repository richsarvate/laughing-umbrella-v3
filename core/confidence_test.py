#!/usr/bin/env python3
"""
Quick confidence test - Run model on random 10-day window in 2025.
Check if 99%+ confidence on ENPH is consistent or anomaly.
"""

import sys
import os
import random
from datetime import datetime, timedelta

# Add core to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training_system import TrainingSystem


def get_random_trading_days(start_date: str, end_date: str, num_days: int = 10):
    """Generate random consecutive trading days in 2025."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Pick random start point that allows num_days after it
    max_start = end - timedelta(days=num_days * 2)  # Buffer for weekends
    
    if max_start < start:
        max_start = start
    
    # Random start date
    days_range = (max_start - start).days
    random_offset = random.randint(0, max(days_range, 0))
    test_start = start + timedelta(days=random_offset)
    
    # Generate consecutive trading days
    trading_days = []
    current = test_start
    while len(trading_days) < num_days:
        # Skip weekends
        if current.weekday() < 5:  # Monday=0, Friday=4
            trading_days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
        
        # Safety check
        if current > end:
            break
    
    return trading_days[:num_days]


def run_confidence_test():
    """Run model on 10 random days and analyze confidence patterns."""
    print("=" * 70)
    print("🧪 CONFIDENCE TEST - Random 10-Day Window in 2025")
    print("=" * 70)
    
    # Get random 10 trading days in 2025
    test_days = get_random_trading_days("2025-01-01", "2025-10-26", num_days=10)
    
    print(f"\nTest Period: {test_days[0]} to {test_days[-1]}")
    print(f"Number of days: {len(test_days)}")
    print("\n" + "=" * 70)
    
    # Initialize trading system
    trading_system = TrainingSystem()
    
    # Track results
    predictions = []
    stock_counts = {}
    confidence_levels = []
    
    print("\nDAILY PREDICTIONS:")
    print("-" * 70)
    
    for i, test_date in enumerate(test_days, 1):
        try:
            # Get prediction
            action, target_stock, top3_choices = trading_system.predict_action(test_date)
            
            # Extract top choice details
            top_action, top_stock, top_confidence = top3_choices[0]
            
            # Track statistics
            stock_choice = top_stock if top_stock else top_action
            predictions.append({
                'date': test_date,
                'action': action,
                'stock': stock_choice,
                'confidence': top_confidence,
                'top3': top3_choices
            })
            
            # Count stock appearances
            stock_counts[stock_choice] = stock_counts.get(stock_choice, 0) + 1
            confidence_levels.append(top_confidence)
            
            # Display
            print(f"Day {i:2d} ({test_date}): {action:6s} | {stock_choice:6s} | "
                  f"Conf: {top_confidence:6.2%}")
            print(f"         Top 3: ", end="")
            for rank, (act, stk, conf) in enumerate(top3_choices, 1):
                stock_display = stk if stk else act
                print(f"#{rank}: {stock_display}({conf:.2%})  ", end="")
            print()
            
        except Exception as e:
            print(f"Day {i:2d} ({test_date}): ERROR - {e}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("📊 ANALYSIS")
    print("=" * 70)
    
    print(f"\nConfidence Statistics:")
    print(f"  Average: {sum(confidence_levels)/len(confidence_levels):.2%}")
    print(f"  Minimum: {min(confidence_levels):.2%}")
    print(f"  Maximum: {max(confidence_levels):.2%}")
    print(f"  Above 90%: {sum(1 for c in confidence_levels if c > 0.90)} days")
    print(f"  Above 50%: {sum(1 for c in confidence_levels if c > 0.50)} days")
    print(f"  Below 30%: {sum(1 for c in confidence_levels if c < 0.30)} days")
    
    print(f"\nStock Selection Frequency:")
    for stock, count in sorted(stock_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(test_days)) * 100
        print(f"  {stock:6s}: {count:2d} days ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("🎯 ASSESSMENT")
    print("=" * 70)
    
    # Evaluate results
    avg_conf = sum(confidence_levels) / len(confidence_levels)
    high_conf_days = sum(1 for c in confidence_levels if c > 0.90)
    dominant_stock = max(stock_counts.values())
    unique_stocks = len(stock_counts)
    
    print()
    if avg_conf > 0.80 and high_conf_days > 7:
        print("🚨 PROBLEM: Model shows extremely high confidence (>90%) most days")
        print("   This suggests overfitting or model issues.")
    elif avg_conf > 0.50:
        print("⚠️  WARNING: Model confidence is quite high (>50% average)")
        print("   This may indicate some overfitting.")
    else:
        print("✅ GOOD: Model shows reasonable confidence levels")
    
    print()
    if dominant_stock > 7:
        print(f"🚨 PROBLEM: One stock dominates ({dominant_stock}/10 days)")
        print("   Model is not adapting to changing conditions.")
    elif dominant_stock > 5:
        print(f"⚠️  WARNING: One stock appears frequently ({dominant_stock}/10 days)")
        print("   Model may be biased toward certain stocks.")
    else:
        print(f"✅ GOOD: Model shows variety ({unique_stocks} different choices)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_confidence_test()
