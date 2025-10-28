"""
Step 1: Download all S&P 500 data (2010-present) and cache it.
Run this once before preprocessing.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path

def get_sp500_tickers():
    """Get S&P 500 ticker list - using static list to avoid scraping issues."""
    # Top ~100 S&P 500 stocks for initial testing
    # Replace with full list or use a data provider API for production
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
        'ABBV', 'PFE', 'AVGO', 'KO', 'LLY', 'TMO', 'COST', 'MRK', 'ORCL',
        'ACN', 'DHR', 'VZ', 'ABT', 'WMT', 'CRM', 'NFLX', 'ADBE', 'NKE',
        'TXN', 'RTX', 'QCOM', 'NEE', 'PM', 'LOW', 'BMY', 'HON', 'UPS',
        'AMGN', 'T', 'COP', 'IBM', 'SPGI', 'CAT', 'MDT', 'SCHW', 'GS',
        'AXP', 'BLK', 'BKNG', 'SYK', 'DE', 'TJX', 'AMD', 'LMT', 'MDLZ',
        'ADP', 'GILD', 'CVS', 'MMC', 'C', 'LRCX', 'ADI', 'INTC', 'PYPL',
        'TMUS', 'CB', 'MO', 'SO', 'ZTS', 'EQIX', 'CME', 'FI', 'EOG',
        'WM', 'ITW', 'PNC', 'AON', 'CSX', 'CL', 'FCX', 'SBUX', 'DUK',
        'ICE', 'USB', 'BSX', 'NSC', 'SPG', 'HCA', 'PLD', 'GM', 'F', 'EMR'
    ]

def download_stock(ticker, start_date='2010-01-01', end_date='2025-12-31'):
    """Download single stock OHLCV data."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Flatten multi-index columns (yfinance returns ('Close', 'AAPL') format)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if len(data) > 300:  # Need at least ~1 year
            return data
    except Exception as e:
        print(f"  Failed {ticker}: {e}")
    return None

def main():
    print("="*60)
    print("DOWNLOADING S&P 500 DATA (2010-2025)")
    print("="*60)
    
    # Create cache directory
    cache_dir = Path('data/raw')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get tickers
    tickers = get_sp500_tickers()
    print(f"\nDownloading {len(tickers)} stocks...")
    
    # Download each stock
    success_count = 0
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        data = download_stock(ticker)
        
        if data is not None:
            # Save to cache
            cache_file = cache_dir / f"{ticker}.parquet"
            data.to_parquet(cache_file)
            success_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(tickers)} ({success_count} successful)")
        else:
            failed_tickers.append(ticker)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Successfully downloaded: {success_count} stocks")
    print(f"Failed: {len(failed_tickers)} stocks")
    
    if failed_tickers:
        print(f"\nFailed tickers: {', '.join(failed_tickers[:10])}")
        if len(failed_tickers) > 10:
            print(f"  ... and {len(failed_tickers) - 10} more")
    
    print(f"\nData saved to: {cache_dir}/")
    print("\nNext step: Run 'python preprocess.py'")

if __name__ == '__main__':
    main()
