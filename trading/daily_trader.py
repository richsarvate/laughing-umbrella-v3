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
    force = '--force' in sys.argv
    debug = '--debug' in sys.argv
    
    logger = setup_logging()
    
    try:
        logger.info("=" * 60)
        logger.info(f"Daily transformer trader starting | timestamp={datetime.now().isoformat()}")
        if debug:
            logger.info("DEBUG MODE ENABLED | trades_will_not_execute=true")
        logger.info("=" * 60)
        
        # Execute trade (credentials loaded from config)
        trader = AlpacaTrader()
        
        # Check if market is open (unless --force flag is used)
        clock = trader.api.get_clock()
        if not clock.is_open and not force:
            logger.info(f"Market closed | date={datetime.now().strftime('%Y-%m-%d')} | action=exit")
            return
        
        if not clock.is_open and force:
            logger.warning("Force mode enabled | market_status=closed | execution=forced")
        
        trader.execute_daily_trade(debug_mode=debug)
        
        logger.info("Daily trading execution completed | status=success")
        
    except Exception as e:
        logger.error(f"Daily trading failed | error={e}")
        
    finally:
        logger.info(f"Daily trader finished | timestamp={datetime.now().isoformat()}")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()