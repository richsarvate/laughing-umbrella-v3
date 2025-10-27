"""
Market data processing module for S&P 500 stocks.
Handles data download, feature extraction, and anonymous representation.
"""

from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler


class MarketDataProcessor:
    """Handles S&P 500 data download and anonymous feature extraction."""
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.feature_scaler = StandardScaler()
        self.sp500_tickers = self._get_sp500_tickers()
        # Create anonymous mapping for training (prevents ticker leakage)
        self._create_anonymous_mapping()
    
    def _get_sp500_tickers(self) -> List[str]:
        """Get complete S&P 500 ticker list for maximum model performance."""
        # Complete S&P 500 stocks (approximately 500 stocks)
        sp500_tickers = [
            # Mega Cap (Top 50)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
            'ABBV', 'PFE', 'AVGO', 'KO', 'LLY', 'TMO', 'COST', 'MRK', 'ORCL',
            'ACN', 'DHR', 'VZ', 'ABT', 'WMT', 'CRM', 'NFLX', 'ADBE', 'NKE',
            'TXN', 'RTX', 'QCOM', 'NEE', 'PM', 'LOW', 'BMY', 'HON', 'UPS',
            'AMGN', 'T', 'COP', 'IBM', 'SPGI',
            
            # Large Cap (51-150)
            'CAT', 'MDT', 'SCHW', 'GS', 'AXP', 'BLK', 'BKNG', 'SYK', 'DE', 'TJX',
            'AMD', 'LMT', 'MDLZ', 'ADP', 'GILD', 'CVS', 'MMC', 'C', 'LRCX', 'ADI',
            'INTC', 'PYPL', 'TMUS', 'CB', 'MO', 'SO', 'ZTS', 'EQIX', 'CME', 'FI',
            'EOG', 'WM', 'ITW', 'PNC', 'AON', 'CSX', 'CL', 'FCX', 'SBUX', 'DUK',
            'ICE', 'USB', 'BSX', 'NSC', 'SPG', 'HCA', 'PLD', 'GM', 'F', 'EMR',
            'GE', 'NOW', 'ISRG', 'VRTX', 'BIDU', 'BDX', 'TGT', 'REGN', 'APD', 'SHW',
            'PANW', 'CMG', 'MU', 'AON', 'CCI', 'KLAC', 'AMAT', 'CDNS', 'SNPS', 'ORLY',
            'MCO', 'ECL', 'FTNT', 'MAR', 'MSI', 'ADSK', 'AJG', 'NXPI', 'ROP', 'PAYX',
            'ROST', 'KMB', 'EA', 'VRSK', 'CTSH', 'ODFL', 'CPRT', 'IEX', 'BK', 'GLW',
            'MCHP', 'KR', 'DXCM', 'CARR', 'WBA', 'HPQ', 'CSGP', 'ANSS', 'ON', 'BIIB',
            
            # Mid Cap (151-300) 
            'FAST', 'MPWR', 'IDXX', 'CTAS', 'CDW', 'FANG', 'EXC', 'XEL', 'WEC', 'ES',
            'AEE', 'LNT', 'EVRG', 'PEG', 'SRE', 'AEP', 'D', 'ED', 'ETR', 'FE',
            'PPL', 'AES', 'CNP', 'NI', 'LNT', 'PNW', 'SO', 'NEE', 'DUK', 'EXC',
            'PCG', 'EIX', 'AWK', 'ATO', 'CMS', 'DTE', 'NRG', 'VST', 'AEE', 'XEL',
            'WEC', 'ES', 'EVRG', 'PEG', 'SRE', 'AEP', 'D', 'ED', 'ETR', 'FE',
            'PPL', 'AES', 'CNP', 'NI', 'LNT', 'PNW', 'NEE', 'DUK', 'EXC', 'PCG',
            'EIX', 'AWK', 'ATO', 'CMS', 'DTE', 'NRG', 'VST', 'WMB', 'KMI', 'OKE',
            'EPD', 'ET', 'MPLX', 'TRGP', 'AM', 'SUN', 'PAA', 'WES', 'EQT', 'AR',
            'DVN', 'FANG', 'MRO', 'APA', 'MGY', 'SM', 'NOV', 'HAL', 'SLB', 'BKR',
            'VAL', 'MPC', 'PSX', 'VLO', 'HFC', 'DK', 'PBF', 'CVR', 'DINO', 'RRC',
            
            # Additional S&P 500 Companies (301-450)
            'ALLE', 'ARE', 'AOS', 'APH', 'ACGL', 'ANET', 'APA', 'AAON', 'AIZ', 'AFL',
            'A', 'APD', 'AKAM', 'ALK', 'ALB', 'AA', 'ALXN', 'ARE', 'ALKS', 'AEE',
            'AAP', 'AAPL', 'AMAT', 'APTV', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK',
            'ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BBWI', 'BAX', 'BDX',
            'BRK-B', 'BBY', 'BIO', 'BIIB', 'BLK', 'BK', 'BA', 'BKNG', 'BWA', 'BXP',
            'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF-B', 'CHRW', 'CDNS', 'CZR', 'CPT',
            'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE',
            'CDW', 'CE', 'CNC', 'CNP', 'CDAY', 'CERN', 'CF', 'CRL', 'SCHW', 'CHTR',
            'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG',
            'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG',
            
            # Final S&P 500 Companies (451-500)
            'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST', 'CTRA', 'CCI',
            'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY',
            'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR',
            'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN',
            'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR', 'EOG',
            'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'RE', 'EXC'
        ]
        
        # Remove any duplicates and return first 500
        return list(dict.fromkeys(sp500_tickers))[:500]
    
    def _create_anonymous_mapping(self):
        """Create anonymous labels for stocks to prevent ticker symbol leakage during training."""
        # Create anonymous identifiers that don't reveal company information
        self.anonymous_stock_ids = [f"STOCK_{i:03d}" for i in range(len(self.sp500_tickers))]
        
        # Mapping for debugging/prediction only (not used during training)
        self.ticker_to_anonymous = {ticker: anon_id for ticker, anon_id in 
                                  zip(self.sp500_tickers, self.anonymous_stock_ids)}
        self.anonymous_to_ticker = {anon_id: ticker for ticker, anon_id in 
                                  self.ticker_to_anonymous.items()}
        
    def get_anonymous_stock_info(self, stock_index: int) -> str:
        """Get anonymous stock identifier for logging purposes only."""
        if 0 <= stock_index < len(self.anonymous_stock_ids):
            return self.anonymous_stock_ids[stock_index]
        return f"UNKNOWN_STOCK_{stock_index}"
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI using simple method without talib dependency."""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)  # Neutral RSI
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Initialize RSI array
        rsi = np.full(len(prices), 50.0)
        
        # Calculate initial averages
        if len(gains) >= period:
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Calculate RSI for each point after the initial period
            for i in range(period, len(prices)):
                if i-period >= 0:
                    avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
                    avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
                    
                    if avg_loss == 0:
                        rsi[i] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def download_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download OHLCV data for all S&P stocks."""
        print(f"Downloading market data from {start_date} to {end_date}...")
        
        stock_data = yf.download(
            self.sp500_tickers, 
            start=start_date, 
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            prepost=True
        )
        return stock_data
    
    def extract_anonymous_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Convert raw price data to completely anonymous technical features.
        
        This method ensures that:
        1. No ticker symbols are exposed to the model
        2. Features are purely technical and position-agnostic
        3. All stocks are treated identically without company-specific bias
        """
        num_stocks = len(self.sp500_tickers)
        num_days = len(market_data)
        features_per_stock = 3  # momentum_5d, volatility_20d, rsi_14d
        
        # Initialize feature matrix: [days, stocks, features]
        feature_matrix = np.zeros((num_days, num_stocks, features_per_stock))
        
        # Process each stock anonymously - order doesn't matter since training will shuffle
        for i, ticker in enumerate(self.sp500_tickers):
            try:
                # Extract price series for this anonymous stock
                close_prices = market_data[ticker]['Close'].values
                
                # Skip if insufficient data for reliable feature calculation
                if len(close_prices) < 21:
                    # Fill with neutral values for missing data
                    feature_matrix[:, i, 0] = 0.0  # Neutral momentum
                    feature_matrix[:, i, 1] = 0.0  # Low volatility
                    feature_matrix[:, i, 2] = 0.5  # Neutral RSI
                    continue
                
                # Feature 1: 5-day momentum (return) - purely price-based, anonymous
                momentum_5d = np.zeros(len(close_prices))
                momentum_5d[5:] = (close_prices[5:] - close_prices[:-5]) / close_prices[:-5]
                
                # Feature 2: 20-day rolling volatility - purely statistical, anonymous
                returns = np.diff(close_prices) / close_prices[:-1]
                volatility_20d = np.zeros(len(close_prices))
                for j in range(20, len(close_prices)):
                    if j >= 20:
                        volatility_20d[j] = np.std(returns[j-20:j])
                
                # Feature 3: 14-day RSI - purely technical, anonymous
                rsi_14d = self._calculate_rsi(close_prices, period=14)
                rsi_14d = np.nan_to_num(rsi_14d, nan=50.0) / 100.0  # Normalize to [0,1]
                
                # Store anonymous features (no ticker information retained)
                feature_matrix[:, i, 0] = momentum_5d
                feature_matrix[:, i, 1] = volatility_20d  
                feature_matrix[:, i, 2] = rsi_14d
                
            except Exception as e:
                # Anonymous error logging - don't expose ticker information
                print(f"Warning: Could not process stock at index {i}: {str(e)[:50]}...")
                # Fill with neutral values for problematic stocks
                feature_matrix[:, i, 0] = 0.0  # Neutral momentum
                feature_matrix[:, i, 1] = 0.0  # Low volatility  
                feature_matrix[:, i, 2] = 0.5  # Neutral RSI
                continue
        
        # Remove NaN and standardize features globally
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        # Global standardization across all stocks and time periods
        # This ensures no stock-specific scaling bias can be learned
        original_shape = feature_matrix.shape
        flattened = feature_matrix.reshape(-1, features_per_stock)
        standardized = self.feature_scaler.fit_transform(flattened)
        feature_matrix = standardized.reshape(original_shape)
        
        return feature_matrix