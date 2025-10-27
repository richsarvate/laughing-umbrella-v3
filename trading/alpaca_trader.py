"""
Simple Alpaca paper trading integration.
Uses existing trained transformer model for live trading decisions.
"""

import os
import sys
import logging
from datetime import datetime
import alpaca_trade_api as tradeapi

# Import from core directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
from training_system import TrainingSystem

# Import trading configuration
from config import ALPACA_CONFIG, TRADING_CONFIG

logger = logging.getLogger(__name__)

class AlpacaTrader:
    """Minimal Alpaca paper trading integration."""
    
    def __init__(self):
        # Alpaca paper trading API using config
        self.api = tradeapi.REST(
            ALPACA_CONFIG['api_key'],
            ALPACA_CONFIG['secret_key'],
            base_url=ALPACA_CONFIG['base_url'],
            api_version=ALPACA_CONFIG['api_version']
        )
        
        self.cash_deployment_pct = TRADING_CONFIG['cash_deployment_pct']
        
        # Initialize trading system (model loaded on first prediction)
        print("Initializing transformer trading system...")
        self.trading_system = TrainingSystem()
        print("✅ Trading system ready")
    
    def execute_daily_trade(self, debug_mode=False):
        """Execute one daily trade based on transformer prediction."""
        logger.info(f"Trade execution started | date={datetime.now().strftime('%Y-%m-%d')} | time={datetime.now().strftime('%H:%M:%S')} | debug_mode={debug_mode}")
        
        # Get model decision
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            action, target_stock, top3_choices = self.trading_system.predict_action(today)
            decision = {'action': action, 'target_stock': target_stock}
            
            # Log top 3 predictions
            logger.info(f"Model prediction | action={action} | target_stock={target_stock if target_stock else 'N/A'}")
            logger.info("Top 3 model choices:")
            for i, (choice_action, choice_stock, choice_prob) in enumerate(top3_choices, 1):
                stock_str = choice_stock if choice_stock else choice_action
                logger.info(f"  #{i}: {stock_str} - confidence={choice_prob:.4f} ({choice_prob*100:.2f}%)")
        except Exception as e:
            logger.error(f"Model prediction failed | error={e}")
            return
        
        # Execute based on decision (skip if debug mode)
        if debug_mode:
            logger.info(f"DEBUG MODE | would_execute={decision['action']} | target={target_stock if target_stock else 'N/A'} | skipping_actual_execution=true")
        else:
            if decision['action'] == 'CASH':
                self._go_to_cash()
            elif decision['action'] == 'HOLD':
                logger.info("Trade action | action=HOLD | description=no_action_needed")
            elif decision['action'] == 'SWITCH':
                self._switch_to_stock(decision['target_stock'])
        
        # Show results
        self._print_portfolio()
    
    def _go_to_cash(self):
        """Sell all positions and hold cash."""
        logger.info("Trade action | action=CASH | description=selling_all_positions")
        
        try:
            positions = self.api.list_positions()
            for position in positions:
                if float(position.qty) > 0:
                    logger.info(f"Order submitted | side=sell | symbol={position.symbol} | qty={position.qty} | type=market")
                    self.api.submit_order(
                        symbol=position.symbol,
                        qty=position.qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
            logger.info(f"All positions sold | count={len(positions)}")
        except Exception as e:
            logger.error(f"Error selling positions | error={e}")
    
    def _switch_to_stock(self, target_stock: str):
        """Switch to target stock (sell current, buy target)."""
        logger.info(f"Trade action | action=SWITCH | target_stock={target_stock}")
        
        try:
            # Sell all current positions
            positions = self.api.list_positions()
            for position in positions:
                if float(position.qty) > 0:
                    logger.info(f"Order submitted | side=sell | symbol={position.symbol} | qty={position.qty} | type=market")
                    self.api.submit_order(
                        symbol=position.symbol,
                        qty=position.qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
            
            # Get buying power
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Get current price
            latest_trade = self.api.get_latest_trade(target_stock)
            current_price = latest_trade.price
            
            # Calculate shares to buy
            shares_to_buy = int((buying_power * self.cash_deployment_pct) / current_price)
            
            if shares_to_buy > 0:
                logger.info(f"Order submitted | side=buy | symbol={target_stock} | qty={shares_to_buy} | price={current_price:.2f} | type=market")
                self.api.submit_order(
                    symbol=target_stock,
                    qty=shares_to_buy,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Switch completed | symbol={target_stock} | status=success")
            else:
                logger.error(f"Insufficient buying power | buying_power={buying_power:.2f}")
                
        except Exception as e:
            logger.error(f"Switch failed | target_stock={target_stock} | error={e}")
    
    def _print_portfolio(self):
        """Print current portfolio status."""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            logger.info(f"Portfolio status | equity={float(account.equity):.2f} | cash={float(account.cash):.2f} | position_count={len(positions)}")
            
            if positions:
                total_unrealized_pl = 0
                for pos in positions:
                    pnl = float(pos.unrealized_pl)
                    pnl_pct = float(pos.unrealized_plpc) * 100
                    total_unrealized_pl += pnl
                    logger.info(f"Position | symbol={pos.symbol} | qty={pos.qty} | unrealized_pl={pnl:.2f} | unrealized_pl_pct={pnl_pct:.2f}")
                logger.info(f"Total unrealized P&L | amount={total_unrealized_pl:.2f}")
            else:
                logger.info("Portfolio status | position=CASH")
                
        except Exception as e:
            logger.error(f"Error retrieving portfolio status | error={e}")