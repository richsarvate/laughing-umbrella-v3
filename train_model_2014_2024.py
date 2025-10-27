#!/usr/bin/env python3
"""
Train the position-agnostic transformer model on 2014-2024 data.
This uses the new position shuffling implementation to prevent position-based learning.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.training_system import TrainingSystem
import time

def train_model_2014_2024():
    """Train the model on 2014-2024 data with position shuffling."""
    print("="*60)
    print("TRAINING POSITION-AGNOSTIC TRANSFORMER STOCK TRADER")
    print("="*60)
    print("Training period: 2014-01-01 to 2024-01-01")
    print("Features: Position shuffling enabled")
    print("Anonymous features: No ticker symbols exposed")
    print("="*60)
    
    # Create training system
    print("Initializing training system...")
    training_system = TrainingSystem()
    
    print(f"Loaded {len(training_system.data_processor.sp500_tickers)} S&P 500 stocks")
    print("Position shuffling: ENABLED")
    print("Anonymous processing: ENABLED")
    print()
    
    # Start training
    print("Starting model training...")
    start_time = time.time()
    
    try:
        training_system.train_model(
            start_date="2014-01-01",
            end_date="2024-01-01"
        )
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        print()
        print("="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Training duration: {training_duration/60:.1f} minutes")
        print("Model saved to: models/trained_stock_trader.pth")
        print("Scaler saved to: models/feature_scaler.pkl")
        print()
        print("Key improvements in this training:")
        print("✓ Position shuffling prevents position-based learning")
        print("✓ Anonymous features prevent ticker symbol bias")
        print("✓ Multiple prediction shuffles for robust decisions")
        print("✓ Enhanced regularization for better generalization")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Please check the error details above.")
        return False
    
    return True

if __name__ == "__main__":
    success = train_model_2014_2024()
    if success:
        print("\nModel training completed successfully!")
        print("You can now use the trained model for predictions.")
    else:
        print("\nModel training failed. Please check the errors above.")