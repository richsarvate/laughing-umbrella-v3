"""
Daily execution script for cron job automation.
Simple entry point for daily transformer trading.
"""

import os
import sys
import logging
from datetime import datetime

# Add trading module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from alpaca_trader import AlpacaTrader

def setup_logging():
    """Setup logging for daily execution."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"trading_{datetime.now().strftime('%Y%m')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Daily trading execution."""
    logger = setup_logging()
    
    try:
        logger.info("="*60)
        logger.info("🤖 DAILY TRANSFORMER TRADER STARTING")
        logger.info("="*60)
        
        # Execute trade (credentials loaded from config)
        trader = AlpacaTrader()
        trader.execute_daily_trade()
        
        logger.info("✅ Daily trading execution completed")
        
    except Exception as e:
        logger.error(f"❌ Daily trading failed: {e}")
        
    finally:
        logger.info("🏁 Daily trader finished")
        logger.info("="*60)

if __name__ == "__main__":
    main()