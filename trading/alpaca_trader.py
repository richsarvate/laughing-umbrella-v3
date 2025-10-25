"""
Simple Alpaca paper trading integration.
Uses existing trained transformer model for live trading decisions.
"""

import os
import sys
from datetime import datetime
import alpaca_trade_api as tradeapi

# Import from core directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
from training_system import TrainingSystem

# Import trading configuration
from config import ALPACA_CONFIG, TRADING_CONFIG

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
    
    def execute_daily_trade(self):
        """Execute one daily trade based on transformer prediction."""
        print(f"\n🤖 DAILY TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        
        # Get model decision
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            action, target_stock = self.trading_system.predict_action(today)
            decision = {'action': action, 'target_stock': target_stock}
            print(f"🧠 Model Decision: {decision}")
        except Exception as e:
            print(f"❌ Error getting model decision: {e}")
            return
        
        # Execute based on decision
        if decision['action'] == 'CASH':
            self._go_to_cash()
        elif decision['action'] == 'HOLD':
            print("📍 HOLD - No action needed")
        elif decision['action'] == 'SWITCH':
            self._switch_to_stock(decision['target_stock'])
        
        # Show results
        self._print_portfolio()
    
    def _go_to_cash(self):
        """Sell all positions and hold cash."""
        print("💰 Going to CASH - Selling all positions")
        
        try:
            positions = self.api.list_positions()
            for position in positions:
                if float(position.qty) > 0:
                    print(f"   📤 Selling {position.qty} shares of {position.symbol}")
                    self.api.submit_order(
                        symbol=position.symbol,
                        qty=position.qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
            print("✅ All positions sold")
        except Exception as e:
            print(f"❌ Error selling positions: {e}")
    
    def _switch_to_stock(self, target_stock: str):
        """Switch to target stock (sell current, buy target)."""
        print(f"🔄 SWITCH to {target_stock}")
        
        try:
            # Sell all current positions
            positions = self.api.list_positions()
            for position in positions:
                if float(position.qty) > 0:
                    print(f"   📤 Selling {position.qty} shares of {position.symbol}")
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
                print(f"   📥 Buying {shares_to_buy} shares of {target_stock} at ${current_price:.2f}")
                self.api.submit_order(
                    symbol=target_stock,
                    qty=shares_to_buy,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                print(f"✅ Switched to {target_stock}")
            else:
                print("❌ Insufficient buying power")
                
        except Exception as e:
            print(f"❌ Error switching to {target_stock}: {e}")
    
    def _print_portfolio(self):
        """Print current portfolio status."""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            print(f"\n📊 PORTFOLIO STATUS")
            print(f"   💵 Equity: ${float(account.equity):,.2f}")
            print(f"   💰 Cash: ${float(account.cash):,.2f}")
            print(f"   📈 Day P&L: ${float(account.unrealized_pl):,.2f}")
            
            if positions:
                print(f"   📍 Positions:")
                for pos in positions:
                    pnl = float(pos.unrealized_pl)
                    pnl_pct = float(pos.unrealized_plpc) * 100
                    print(f"      {pos.symbol}: {pos.qty} shares, P&L: ${pnl:.2f} ({pnl_pct:.1f}%)")
            else:
                print(f"   📍 Positions: CASH")
                
        except Exception as e:
            print(f"❌ Error getting portfolio status: {e}")