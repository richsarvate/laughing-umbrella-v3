"""
Step 4: Backtest the model on 2025 data.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from model import StockTransformer

def get_stock_prediction(model, features, device='cpu'):
    """Get model prediction for a single stock."""
    # Get last 60 days
    if len(features) < 60:
        return None
    
    sequence = features[-60:].values
    
    # Check for NaN
    if np.isnan(sequence).any():
        return None
    
    # Predict
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        pred = model(X).item()
    
    return pred

def backtest_cycle(model, processed_dir, start_date, hold_days=5, device='cpu'):
    """Run one backtest cycle."""
    predictions = {}
    
    # Get predictions for all stocks
    for file in processed_dir.glob("*.parquet"):
        ticker = file.stem
        df = pd.read_parquet(file)
        
        # Get data up to start_date
        df_until = df[df.index <= start_date]
        
        if len(df_until) < 60:
            continue
        
        # Get features (exclude target column)
        feature_cols = [col for col in df_until.columns if col != 'target_5d_return']
        features = df_until[feature_cols]
        
        # Predict
        pred = get_stock_prediction(model, features, device)
        if pred is not None:
            predictions[ticker] = pred
    
    if len(predictions) < 6:
        return None
    
    # Rank stocks
    sorted_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    long_stocks = [s[0] for s in sorted_stocks[:3]]
    short_stocks = [s[0] for s in sorted_stocks[-3:]]
    
    # Calculate actual returns
    returns = {}
    for file in processed_dir.glob("*.parquet"):
        ticker = file.stem
        df = pd.read_parquet(file)
        
        try:
            start_idx = df.index.get_loc(start_date)
            if start_idx + hold_days >= len(df):
                continue
            
            # Get actual 5-day return from raw close prices (need to load raw data)
            raw_file = Path('data/raw') / f"{ticker}.parquet"
            raw_df = pd.read_parquet(raw_file)
            
            start_price = raw_df.loc[start_date, 'Close']
            end_date = df.index[start_idx + hold_days]
            end_price = raw_df.loc[end_date, 'Close']
            
            returns[ticker] = (end_price - start_price) / start_price
        except:
            continue
    
    # Portfolio return
    long_return = np.mean([returns.get(s, 0) for s in long_stocks])
    short_return = np.mean([returns.get(s, 0) for s in short_stocks])
    portfolio_return = long_return - short_return
    
    return {
        'portfolio_return': portfolio_return,
        'long_stocks': long_stocks,
        'short_stocks': short_stocks,
        'long_return': long_return,
        'short_return': short_return
    }

def main():
    print("="*60)
    print("BACKTESTING ON 2025")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed_dir = Path('data/processed')
    
    # Load model
    print("\nLoading trained model...")
    model = StockTransformer(input_dim=26, hidden_dim=128, num_heads=2, num_layers=2)
    model.load_state_dict(torch.load('model_best.pth', map_location=device))
    model = model.to(device)
    
    # Get 2025 dates
    sample_file = list(processed_dir.glob("*.parquet"))[0]
    sample_df = pd.read_parquet(sample_file)
    dates_2025 = sample_df[sample_df.index >= '2025-01-01'].index
    
    if len(dates_2025) < 30:
        print("Not enough 2025 data for backtesting")
        return
    
    print(f"\nRunning 3 backtest cycles...")
    
    results = []
    for cycle in range(3):
        # Space out cycles by ~10 trading days
        if cycle * 10 >= len(dates_2025):
            break
        
        start_date = dates_2025[cycle * 10]
        
        print(f"\n--- Cycle {cycle + 1} ---")
        print(f"Date: {start_date.date()}")
        
        result = backtest_cycle(model, processed_dir, start_date, hold_days=5, device=device)
        
        if result is None:
            print("Insufficient data")
            continue
        
        print(f"Long: {result['long_stocks']}")
        print(f"Short: {result['short_stocks']}")
        print(f"Long return: {result['long_return']*100:.2f}%")
        print(f"Short return: {result['short_return']*100:.2f}%")
        print(f"Portfolio return: {result['portfolio_return']*100:.2f}%")
        
        results.append(result['portfolio_return'])
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Total cycles: {len(results)}")
        print(f"Average return: {np.mean(results)*100:.2f}%")
        print(f"Total return: {np.sum(results)*100:.2f}%")
        print(f"Win rate: {sum(1 for r in results if r > 0) / len(results) * 100:.1f}%")

if __name__ == '__main__':
    main()
