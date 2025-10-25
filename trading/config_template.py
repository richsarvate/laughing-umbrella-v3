"""
Configuration template for Alpaca trading integration.
Copy this to config.py and fill in your API credentials.
"""

# Alpaca Paper Trading API Configuration
ALPACA_CONFIG = {
    'api_key': 'YOUR_ALPACA_API_KEY',
    'secret_key': 'YOUR_ALPACA_SECRET_KEY',
    'base_url': 'https://paper-api.alpaca.markets',  # Paper trading endpoint
    'api_version': 'v2'
}

# Trading Parameters
TRADING_CONFIG = {
    'cash_deployment_pct': 0.95,  # Use 95% of available cash for positions
    'trading_hours': {
        'execution_time': '17:00',  # 5 PM ET (after market close)
        'weekdays_only': True
    }
}

# Logging Configuration  
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_dir': 'logs',
    'log_file_pattern': 'trading_{year}{month:02d}.log'
}