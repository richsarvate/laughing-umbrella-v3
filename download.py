"""
Step 1: Download S&P 500 stock data from 2010 to present.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

def get_sp500_tickers():
    """Get current S&P 500 constituents."""
    # Using a reliable list of S&P 500 tickers
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]
    return table['Symbol'].str.replace('.', '-').tolist()

def download_stock_data(ticker, start_date='2010-01-01'):
    """Download OHLCV data for a single stock."""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return None
        
        # Clean up multi-level column names if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        return df[required_cols]
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

def main():
    print("="*60)
    print("DOWNLOADING S&P 500 DATA")
    print("="*60)
    
    raw_dir = Path('data/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Get S&P 500 tickers
    print("\nFetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers")
    
    # Download first 100 for faster testing
    tickers = tickers[:100]
    print(f"Downloading data for {len(tickers)} stocks (2010-present)...\n")
    
    success_count = 0
    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(tickers)}")
        
        output_file = raw_dir / f"{ticker}.parquet"
        
        # Skip if already downloaded
        if output_file.exists():
            success_count += 1
            continue
        
        df = download_stock_data(ticker)
        
        if df is not None and len(df) > 0:
            df.to_parquet(output_file)
            success_count += 1
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Successfully downloaded: {success_count}/{len(tickers)} stocks")
    print(f"Data saved to: {raw_dir}/")
    print("\nNext step: Run 'python3 preprocess.py'")

if __name__ == '__main__':
    main()
