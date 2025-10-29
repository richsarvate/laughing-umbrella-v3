""""""

Step 2: Preprocess raw data - calculate 25 features, cross-sectional normalize, lag.Step 2: Preprocess raw data - calculate 25 features, cross-sectional normalize, lag.

""""""

import pandas as pdimport pandas as pd

import numpy as npimport numpy as np

from pathlib import Pathfrom pathlib import Path

from ta.momentum import RSIIndicatorfrom ta.momentum import RSIIndicator

from ta.trend import MACDfrom ta.trend import MACD

from ta.volatility import AverageTrueRangefrom ta.volatility import AverageTrueRange



def calculate_features(df):def calculate_features(df):

    """Calculate all 25 features from OHLCV data."""    """Calculate all 25 features from OHLCV data."""

    features = pd.DataFrame(index=df.index)    features = pd.DataFrame(index=df.index)

        

    # Core price fields (5)    # Core price fields (5)

    features['open'] = df['Open']    features['open'] = df['Open']

    features['high'] = df['High']    features['high'] = df['High']

    features['low'] = df['Low']    features['low'] = df['Low']

    features['close'] = df['Close']    features['close'] = df['Close']

    features['volume'] = df['Volume']    features['volume'] = df['Volume']

        

    # Returns (4)    # Returns (4)

    features['daily_return'] = df['Close'].pct_change()    features['daily_return'] = df['Close'].pct_change()

    features['return_5d'] = df['Close'].pct_change(periods=5)    features['return_5d'] = df['Close'].pct_change(periods=5)

    features['return_10d'] = df['Close'].pct_change(periods=10)    features['return_10d'] = df['Close'].pct_change(periods=10)

    features['return_20d'] = df['Close'].pct_change(periods=20)    features['return_20d'] = df['Close'].pct_change(periods=20)

        

    # Moving averages (6)    # Moving averages (6)

    features['sma_5'] = df['Close'].rolling(5).mean()    features['sma_5'] = df['Close'].rolling(5).mean()

    features['sma_10'] = df['Close'].rolling(10).mean()    features['sma_10'] = df['Close'].rolling(10).mean()

    features['sma_20'] = df['Close'].rolling(20).mean()    features['sma_20'] = df['Close'].rolling(20).mean()

    features['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()    features['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    features['close_sma10_ratio'] = df['Close'] / features['sma_10']    features['close_sma10_ratio'] = df['Close'] / features['sma_10']

    features['sma5_sma20_ratio'] = features['sma_5'] / features['sma_20']    features['sma5_sma20_ratio'] = features['sma_5'] / features['sma_20']

        

    # Volatility (3)    # Volatility (3)

    returns = df['Close'].pct_change()    returns = df['Close'].pct_change()

    features['volatility_10'] = returns.rolling(10).std()    features['volatility_10'] = returns.rolling(10).std()

    features['volatility_20'] = returns.rolling(20).std()    features['volatility_20'] = returns.rolling(20).std()

    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)

    features['atr_14'] = atr.average_true_range()    features['atr_14'] = atr.average_true_range()

        

    # Momentum (5)    # Momentum (5)

    rsi = RSIIndicator(close=df['Close'], window=14)    rsi = RSIIndicator(close=df['Close'], window=14)

    features['rsi_14'] = rsi.rsi()    features['rsi_14'] = rsi.rsi()

    macd = MACD(close=df['Close'])    macd = MACD(close=df['Close'])

    features['macd'] = macd.macd()    features['macd'] = macd.macd()

    features['macd_signal'] = macd.macd_signal()    features['macd_signal'] = macd.macd_signal()

    features['momentum_10'] = df['Close'] - df['Close'].shift(10)    features['momentum_10'] = df['Close'] - df['Close'].shift(10)

    features['roc_10'] = df['Close'].pct_change(periods=10)    features['roc_10'] = df['Close'].pct_change(periods=10)

        

    # Volume (2)    # Volume (2)

    vol_mean = df['Volume'].rolling(10).mean()    vol_mean = df['Volume'].rolling(10).mean()

    vol_std = df['Volume'].rolling(10).std()    vol_std = df['Volume'].rolling(10).std()

    features['volume_zscore_10'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)    features['volume_zscore_10'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)

    features['volume_sma_ratio'] = df['Volume'] / vol_mean    features['volume_sma_ratio'] = df['Volume'] / vol_mean

        

    return features    return features



def cross_sectional_normalize(all_stock_features):def cross_sectional_normalize(all_stock_features):

    """    """

    Normalize features across stocks for each date using vectorized operations.    Normalize features across stocks for each date using vectorized operations.

    all_stock_features: dict of {ticker: features_df}    all_stock_features: dict of {ticker: features_df}

    Returns: dict of {ticker: normalized_df}    Returns: dict of {ticker: normalized_df}

    """    """

    print(f"\nCross-sectional normalization...")    print(f"\nCross-sectional normalization...")

    print(f"  Total stocks: {len(all_stock_features)}")    print(f"  Total stocks: {len(all_stock_features)}")

        

    # Concatenate all stocks into a single DataFrame with MultiIndex    # Concatenate all stocks into a single DataFrame with MultiIndex

    print("  Concatenating data...")    print("  Concatenating data...")

    combined = pd.concat(all_stock_features, names=['ticker', 'date'])    combined = pd.concat(all_stock_features, names=['ticker', 'date'])

        

    # Get feature columns    # Get feature columns

    feature_cols = combined.columns.tolist()    feature_cols = combined.columns.tolist()

    print(f"  Features per stock: {len(feature_cols)}")    print(f"  Features per stock: {len(feature_cols)}")

        

    # Group by date and normalize across stocks    # Group by date and normalize across stocks

    print(f"  Normalizing across dates...")    print(f"  Normalizing across dates...")

    def normalize_group(group):    def normalize_group(group):

        # Calculate mean and std across stocks for this date        # Calculate mean and std across stocks for this date

        mean_vals = group.mean()        mean_vals = group.mean()

        std_vals = group.std()        std_vals = group.std()

        # Replace std of 0 with 1 to avoid division by zero        # Replace std of 0 with 1 to avoid division by zero

        std_vals = std_vals.replace(0, 1)        std_vals = std_vals.replace(0, 1)

        std_vals = std_vals.fillna(1)        std_vals = std_vals.fillna(1)

        # Z-score normalization        # Z-score normalization

        return (group - mean_vals) / std_vals        return (group - mean_vals) / std_vals

        

    normalized = combined.groupby(level='date', group_keys=False).apply(normalize_group)    normalized = combined.groupby(level='date', group_keys=False).apply(normalize_group)

        

    print("  Splitting back into individual stocks...")    print("  Splitting back into individual stocks...")

    # Split back into dictionary of individual stock DataFrames    # Split back into dictionary of individual stock DataFrames

    normalized_stocks = {}    normalized_stocks = {}

    for ticker in all_stock_features.keys():    for ticker in all_stock_features.keys():

        normalized_stocks[ticker] = normalized.loc[ticker].astype(float)        normalized_stocks[ticker] = normalized.loc[ticker].astype(float)

        

    print(f"  ✓ Normalization complete")    print(f"  ✓ Normalization complete")

        

    return normalized_stocks    return normalized_stocks



def main():def main():

    print("="*60)    print("="*60)

    print("PREPROCESSING DATA")    print("PREPROCESSING DATA")

    print("="*60)    print("="*60)

        

    raw_dir = Path('data/raw')    raw_dir = Path('data/raw')

    processed_dir = Path('data/processed')    processed_dir = Path('data/processed')

    processed_dir.mkdir(parents=True, exist_ok=True)    processed_dir.mkdir(parents=True, exist_ok=True)

        

    # Get all raw files    # Get all raw files

    raw_files = list(raw_dir.glob("*.parquet"))    raw_files = list(raw_dir.glob("*.parquet"))

    print(f"\nFound {len(raw_files)} raw stock files")    print(f"\nFound {len(raw_files)} raw stock files")

        

    # Step 1: Calculate features for all stocks    # Step 1: Calculate features for all stocks

    print("\nStep 1: Calculating features...")    print("\nStep 1: Calculating features...")

    all_features = {}    all_features = {}

    for i, raw_file in enumerate(raw_files, 1):    for i, raw_file in enumerate(raw_files, 1):

        ticker = raw_file.stem        ticker = raw_file.stem

        if (i) % 50 == 0:        if (i) % 50 == 0:

            print(f"  Progress: {i}/{len(raw_files)}")            print(f"  Progress: {i}/{len(raw_files)}")

                

        df = pd.read_parquet(raw_file)        df = pd.read_parquet(raw_file)

        features = calculate_features(df)        features = calculate_features(df)

                

        # Add target (5-day forward return, NOT lagged)        # Add target (5-day forward return, NOT lagged)

        features['target_5d_return'] = df['Close'].pct_change(periods=5).shift(-5)        features['target_5d_return'] = df['Close'].pct_change(periods=5).shift(-5)

                

        all_features[ticker] = features        all_features[ticker] = features

        

    # Step 2: Separate targets before normalization    # Step 2: Separate targets before normalization

    print("\nStep 2: Separating targets...")    print("\nStep 2: Separating targets...")

    targets = {}    targets = {}

    features_only = {}    features_only = {}

    for ticker, df in all_features.items():    for ticker, df in all_features.items():

        targets[ticker] = df['target_5d_return'].copy()        targets[ticker] = df['target_5d_return'].copy()

        features_only[ticker] = df.drop(columns=['target_5d_return'])        features_only[ticker] = df.drop(columns=['target_5d_return'])

        

    # Step 3: Cross-sectional normalization (features only, NOT target)    # Step 3: Cross-sectional normalization (features only, NOT target)

    print("\nStep 3: Cross-sectional normalization (features only)...")    print("\nStep 3: Cross-sectional normalization (features only)...")

    normalized_features = cross_sectional_normalize(features_only)    normalized_features = cross_sectional_normalize(features_only)

        

    # Step 4: Lag features by 1 day and add back unnormalized target    # Step 4: Lag features by 1 day and add back unnormalized target

    print("\nStep 4: Lagging features and adding targets...")    print("\nStep 4: Lagging features and adding targets...")

    for ticker in normalized_features:    for ticker in normalized_features:

        # Lag all feature columns        # Lag all feature columns

        normalized_features[ticker] = normalized_features[ticker].shift(1)        normalized_features[ticker] = normalized_features[ticker].shift(1)

        # Add back the unnormalized target        # Add back the unnormalized target

        normalized_features[ticker]['target_5d_return'] = targets[ticker]        normalized_features[ticker]['target_5d_return'] = targets[ticker]

        

    # Step 5: Save to data/processed    # Step 4: Save to data/processed

    print("\nStep 5: Saving processed files...")    print("\nStep 4: Saving processed files...")

    success_count = 0    success_count = 0

    for ticker, df in normalized_features.items():    for ticker, df in normalized_features.items():

        # Drop NaN rows        # Drop NaN rows

        df_clean = df.dropna()        df_clean = df.dropna()

                

        # Need at least 60 days for sequences        # Need at least 60 days for sequences

        if len(df_clean) >= 60:        if len(df_clean) >= 60:

            output_file = processed_dir / f"{ticker}.parquet"            output_file = processed_dir / f"{ticker}.parquet"

            df_clean.to_parquet(output_file)            df_clean.to_parquet(output_file)

            success_count += 1            success_count += 1

        

    # Summary    # Summary

    print("\n" + "="*60)    print("\n" + "="*60)

    print("PREPROCESSING COMPLETE")    print("PREPROCESSING COMPLETE")

    print("="*60)    print("="*60)

    print(f"Successfully processed: {success_count}/{len(raw_files)} stocks")    print(f"Successfully processed: {success_count}/{len(raw_files)} stocks")

    print(f"Features: 25 (5 price + 20 technical)")    print(f"Features: 25 (5 price + 20 technical)")

    print(f"Normalization: Cross-sectional z-score (features only)")    print(f"Normalization: Cross-sectional z-score (features only)")

    print(f"Target: Raw 5-day forward returns (NOT normalized)")    print(f"Target: Raw 5-day forward returns (NOT normalized)")

    print(f"Lag: 1 day (features only, not target)")    print(f"Lag: 1 day (features only, not target)")

    print(f"Data saved to: {processed_dir}/")    print(f"Data saved to: {processed_dir}/")

    print("\nNext step: Run 'python3 train_model.py'")    print("\nNext step: Run 'python3 train_model.py'")



if __name__ == '__main__':if __name__ == '__main__':

    main()    main()

