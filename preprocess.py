"""
Step 2: Preprocess raw data - calculate features, normalize, lag.
Run this after download.py completes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

def calculate_features(df):
    """Calculate all 25 features from OHLCV data."""
    features = pd.DataFrame(index=df.index)
    
    # Core price fields (5) - note: yfinance auto_adjust=True means Close is already adjusted
    features['open'] = df['Open']
    features['high'] = df['High']
    features['low'] = df['Low']
    features['close'] = df['Close']
    features['volume'] = df['Volume']
    
    # Returns (4)
    features['daily_return'] = df['Close'].pct_change()
    features['return_5d'] = df['Close'].pct_change(periods=5)
    features['return_10d'] = df['Close'].pct_change(periods=10)
    features['return_20d'] = df['Close'].pct_change(periods=20)
    
    # Moving averages (6)
    features['sma_5'] = df['Close'].rolling(5).mean()
    features['sma_10'] = df['Close'].rolling(10).mean()
    features['sma_20'] = df['Close'].rolling(20).mean()
    features['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    features['close_sma10_ratio'] = df['Close'] / features['sma_10']
    features['sma5_sma20_ratio'] = features['sma_5'] / features['sma_20']
    
    # Volatility (3)
    returns = df['Close'].pct_change()
    features['volatility_10'] = returns.rolling(10).std()
    features['volatility_20'] = returns.rolling(20).std()
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    features['atr_14'] = atr.average_true_range()
    
    # Momentum (5)
    rsi = RSIIndicator(close=df['Close'], window=14)
    features['rsi_14'] = rsi.rsi()
    macd = MACD(close=df['Close'])
    features['macd'] = macd.macd()
    features['macd_signal'] = macd.macd_signal()
    features['momentum_10'] = df['Close'] - df['Close'].shift(10)
    features['roc_10'] = df['Close'].pct_change(periods=10)
    
    # Volume (2)
    vol_mean = df['Volume'].rolling(10).mean()
    vol_std = df['Volume'].rolling(10).std()
    features['volume_zscore_10'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)
    features['volume_sma_ratio'] = df['Volume'] / vol_mean
    
    return features

def normalize_features(features, window=252):
    """Apply rolling z-score normalization."""
    normalized = features.copy()
    
    for col in features.columns:
        rolling_mean = features[col].rolling(window, min_periods=60).mean()
        rolling_std = features[col].rolling(window, min_periods=60).std()
        normalized[col] = (features[col] - rolling_mean) / (rolling_std + 1e-8)
    
    return normalized.fillna(0)

def preprocess_stock(ticker, raw_dir, processed_dir):
    """Preprocess a single stock."""
    # Load raw data
    raw_file = raw_dir / f"{ticker}.parquet"
    if not raw_file.exists():
        return False
    
    df = pd.read_parquet(raw_file)
    
    # Calculate features
    features = calculate_features(df)
    
    # Normalize
    features_norm = normalize_features(features)
    
    # Lag by 1 day to prevent lookahead
    features_lagged = features_norm.shift(1)
    
    # Add target (5-day forward return)
    features_lagged['target_5d_return'] = df['Close'].pct_change(periods=5).shift(-5)
    
    # Drop NaN rows
    features_clean = features_lagged.dropna()
    
    # Save
    if len(features_clean) > 60:  # Need at least 60 days for sequences
        processed_file = processed_dir / f"{ticker}.parquet"
        features_clean.to_parquet(processed_file)
        return True
    
    return False

def main():
    print("="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all raw files
    raw_files = list(raw_dir.glob("*.parquet"))
    print(f"\nProcessing {len(raw_files)} stocks...")
    
    success_count = 0
    for i, raw_file in enumerate(raw_files):
        ticker = raw_file.stem
        
        if preprocess_stock(ticker, raw_dir, processed_dir):
            success_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(raw_files)} ({success_count} successful)")
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Successfully preprocessed: {success_count} stocks")
    print(f"Features: 26 technical indicators")
    print(f"Normalization: Rolling z-score (252-day window)")
    print(f"Lag: 1 day to prevent lookahead bias")
    print(f"\nData saved to: {processed_dir}/")
    print("\nNext step: Run 'python train.py'")

if __name__ == '__main__':
    main()
