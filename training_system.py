"""
Training system for transformer stock trader.
Handles label generation, model training, and prediction logic.
"""

import pickle
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_processor import MarketDataProcessor
from model import TransformerStockTrader


class EnhancedProfitLoss(nn.Module):
    """Loss function that directly maximizes returns with asymmetric risk penalties."""
    
    def __init__(self, loss_penalty_factor: float = 2.0, risk_penalty: float = 0.01):
        super().__init__()
        self.loss_penalty_factor = loss_penalty_factor  # Punish losses more than reward gains
        self.risk_penalty = risk_penalty  # Small penalty for cash to encourage risk-taking
        
    def forward(self, predictions: torch.Tensor, future_returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate profit-based loss for model predictions using differentiable approach.
        
        Args:
            predictions: [batch_size, 102] - model logits for all choices
            future_returns: [batch_size, 100] - actual 5-day returns for each stock
        
        Returns:
            loss: Scalar loss value (negative expected return)
        """
        batch_size = predictions.shape[0]
        
        # Convert logits to probabilities (differentiable)
        action_probs = F.softmax(predictions, dim=-1)  # [batch_size, 102]
        
        # Create expanded returns tensor including HOLD and CASH options
        # [batch_size, 102] = [HOLD, CASH, stock_0, stock_1, ..., stock_99]
        expanded_returns = torch.zeros_like(action_probs, device=predictions.device)
        
        # Set returns for each action type
        expanded_returns[:, 0] = 0.0  # HOLD return = 0%
        expanded_returns[:, 1] = -self.risk_penalty  # CASH return = small penalty
        expanded_returns[:, 2:] = future_returns  # Stock returns
        
        # Calculate expected return using probability weighting (differentiable)
        expected_returns = torch.sum(action_probs * expanded_returns, dim=-1)
        
        # Apply asymmetric penalty for negative expected returns
        enhanced_returns = torch.where(
            expected_returns >= 0,
            expected_returns,  # Reward positive expected returns
            self.loss_penalty_factor * expected_returns  # Punish negative expected returns more
        )
        
        # Loss = negative expected return (maximize return = minimize negative return)
        loss = -enhanced_returns.mean()
        
        return loss


class TrainingSystem:
    """Orchestrates training and prediction for the stock trading model."""
    
    def __init__(self):
        self.data_processor = MarketDataProcessor(lookback_days=30)
        # Use actual number of successfully downloaded stocks
        actual_num_stocks = len(self.data_processor.sp500_tickers)
        self.model = TransformerStockTrader(num_stocks=actual_num_stocks)
        self.current_position = None  # Track current stock holding
        
    def calculate_future_returns(self, raw_market_data, lookahead_days: int = 5) -> np.ndarray:
        """Calculate actual future returns for all stocks for profit-based training."""
        num_days = len(raw_market_data)
        num_stocks = len(self.data_processor.sp500_tickers)
        
        # Initialize returns matrix: [days, stocks]
        future_returns = np.zeros((num_days - lookahead_days, num_stocks))
        
        for i, ticker in enumerate(self.data_processor.sp500_tickers):
            try:
                # Get price series for this stock
                if ticker in raw_market_data.columns.get_level_values(0):
                    close_prices = raw_market_data[ticker]['Close'].values
                    
                    # Calculate forward returns for each day
                    for day in range(len(close_prices) - lookahead_days):
                        current_price = close_prices[day]
                        future_price = close_prices[day + lookahead_days]
                        
                        if current_price > 0 and not np.isnan(current_price) and not np.isnan(future_price):
                            # Calculate percentage return
                            return_pct = (future_price - current_price) / current_price
                            future_returns[day, i] = return_pct
                        else:
                            future_returns[day, i] = 0.0  # No return if invalid prices
                
            except Exception as e:
                print(f"Warning: Could not calculate returns for {ticker}: {e}")
                future_returns[:, i] = 0.0  # Set returns to 0 for problematic stocks
        
        # Clean up extreme values (cap at +/- 50% daily return)
        future_returns = np.clip(future_returns, -0.5, 0.5)
        
        return future_returns
    
    def train_model(self, start_date: str = "2010-01-01", end_date: str = "2024-01-01"):
        """Train the transformer on historical market data using profit optimization."""
        print("Training transformer stock trader with profit maximization...")
        
        # Download and process training data
        raw_market_data = self.data_processor.download_market_data(start_date, end_date)
        market_features = self.data_processor.extract_anonymous_features(raw_market_data)
        
        # Calculate actual future returns for all stocks
        future_returns = self.calculate_future_returns(raw_market_data)
        
        # Prepare training sequences (30-day windows)
        sequence_length = 30
        training_sequences = []
        training_returns = []
        
        for i in range(sequence_length, len(market_features) - 5):  # Leave room for future returns
            sequence = market_features[i-sequence_length:i]
            returns = future_returns[i-sequence_length]  # Future returns for this day
            training_sequences.append(sequence)
            training_returns.append(returns)
        
        # Convert to tensors
        X_train = torch.FloatTensor(np.array(training_sequences))
        y_train = torch.FloatTensor(np.array(training_returns))
        
        print(f"Training on {len(X_train)} sequences with {len(self.data_processor.sp500_tickers)} stocks")
        
        # Training loop with profit-based loss (optimized for 500 stocks)
        optimizer = optim.Adam(self.model.parameters(), lr=8e-5)  # Slightly lower LR for stability
        loss_function = EnhancedProfitLoss(loss_penalty_factor=2.0, risk_penalty=0.005)  # Lower risk penalty
        
        self.model.train()
        num_epochs = 150  # Fewer epochs initially due to larger complexity
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass - returns decision logits
            decision_logits = self.model(X_train)
            
            # Calculate profit-based loss
            loss = loss_function(decision_logits, y_train)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                avg_return = -loss.item()  # Convert loss back to return for readability
                print(f"Epoch {epoch}, Average Return: {avg_return:.4f} ({avg_return*100:.2f}%)")
        
        # Save trained model
        torch.save(self.model.state_dict(), 'trained_stock_trader.pth')
        with open('feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.data_processor.feature_scaler, f)
        
        final_return = -loss.item()
        print(f"Model training completed! Final average return: {final_return:.4f} ({final_return*100:.2f}%)")
    
    def predict_action(self, target_date: str) -> Tuple[str, Optional[str]]:
        """Make unified trading decision for a specific date."""
        # Load trained model
        self.model.load_state_dict(torch.load('trained_stock_trader.pth'))
        with open('feature_scaler.pkl', 'rb') as f:
            self.data_processor.feature_scaler = pickle.load(f)
        
        # Get recent market data leading up to prediction date
        start_date = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
        raw_market_data = self.data_processor.download_market_data(start_date, target_date)
        market_features = self.data_processor.extract_anonymous_features(raw_market_data)
        
        # Use last 30 days as input sequence
        input_sequence = market_features[-30:].reshape(1, 30, -1, 3)
        input_tensor = torch.FloatTensor(input_sequence)
        
        # Make unified prediction
        self.model.eval()
        with torch.no_grad():
            decision_logits = self.model(input_tensor)
            decision_probabilities = torch.softmax(decision_logits, dim=1)
            predicted_choice = torch.argmax(decision_probabilities, dim=1).item()
            confidence = torch.max(decision_probabilities).item()
        
        # Decode unified prediction
        if predicted_choice == 0:
            action_name = "HOLD"
            target_stock = None
        elif predicted_choice == 1:
            action_name = "CASH"
            target_stock = None
        else:
            # Model chose specific stock (index 2+ maps to stock)
            stock_index = predicted_choice - 2
            action_name = "SWITCH"
            target_stock = self.data_processor.sp500_tickers[stock_index]
        
        print(f"Action: {action_name} (confidence: {confidence:.2f})")
        if target_stock:
            print(f"Model selected: {target_stock}")
        
        return action_name, target_stock