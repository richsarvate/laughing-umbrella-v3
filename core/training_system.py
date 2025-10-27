"""
Training system for transformer stock trader.
Handles label generation, model training, and prediction logic.
"""

import os
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
        training_shuffle_indices = []  # Track shuffle indices to unshuffle predictions
        
        for i in range(sequence_length, len(market_features) - 5):  # Leave room for future returns
            sequence = market_features[i-sequence_length:i]
            returns = future_returns[i-sequence_length]  # Future returns for this day
            
            # Generate random shuffle indices for this sample to prevent position learning
            shuffle_indices = np.random.permutation(len(self.data_processor.sp500_tickers))
            
            # Shuffle both features and returns using the same permutation
            shuffled_sequence = sequence[:, shuffle_indices, :]  # Shuffle stock dimension
            shuffled_returns = returns[shuffle_indices]  # Shuffle corresponding returns
            
            training_sequences.append(shuffled_sequence)
            training_returns.append(shuffled_returns)
            training_shuffle_indices.append(shuffle_indices)
        
        # Convert to tensors
        X_train = torch.FloatTensor(np.array(training_sequences))
        y_train = torch.FloatTensor(np.array(training_returns))
        shuffle_indices_train = np.array(training_shuffle_indices)
        
        print(f"Training on {len(X_train)} sequences with {len(self.data_processor.sp500_tickers)} stocks")
        
        # Training loop with profit-based loss and position shuffling
        optimizer = optim.Adam(self.model.parameters(), lr=8e-5)  # Slightly lower LR for stability
        loss_function = EnhancedProfitLoss(loss_penalty_factor=2.0, risk_penalty=0.005)  # Lower risk penalty
        
        self.model.train()
        num_epochs = 150  # Fewer epochs initially due to larger complexity
        batch_size = 32  # Use batches for more efficient shuffling
        
        for epoch in range(num_epochs):
            # Shuffle training data each epoch for additional randomness
            epoch_indices = np.random.permutation(len(X_train))
            
            total_loss = 0.0
            num_batches = 0
            
            # Process in batches with fresh shuffling for each batch
            for batch_start in range(0, len(X_train), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train))
                batch_indices = epoch_indices[batch_start:batch_end]
                
                # Get batch data
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Apply fresh random shuffling to each batch to prevent position learning
                batch_shuffle_indices = np.random.permutation(len(self.data_processor.sp500_tickers))
                X_batch_shuffled = X_batch[:, :, batch_shuffle_indices, :]  # Shuffle stock positions
                y_batch_shuffled = y_batch[:, batch_shuffle_indices]  # Shuffle corresponding returns
                
                optimizer.zero_grad()
                
                # Forward pass with shuffled positions
                decision_logits = self.model(X_batch_shuffled)
                
                # Calculate loss with shuffled returns
                loss = loss_function(decision_logits, y_batch_shuffled)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = total_loss / num_batches
                avg_return = -avg_loss  # Convert loss back to return for readability
                print(f"Epoch {epoch}, Average Return: {avg_return:.4f} ({avg_return*100:.2f}%)")
        
        # Calculate final average loss across all batches in the last epoch
        final_avg_loss = total_loss / num_batches
        final_return = -final_avg_loss  # Convert to return for readability
        
        # Save trained model
        torch.save(self.model.state_dict(), 'trained_stock_trader.pth')
        with open('feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.data_processor.feature_scaler, f)
        
        print(f"Model training completed! Final average return: {final_return:.4f} ({final_return*100:.2f}%)")
    
    def predict_action(self, target_date: str, model_path: str = None, scaler_path: str = None) -> Tuple[str, Optional[str]]:
        """Make unified trading decision for a specific date."""
        # Use provided paths or default to models directory
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'trained_stock_trader.pth')
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'feature_scaler.pkl')
            
        # Load trained model
        self.model.load_state_dict(torch.load(model_path))
        with open(scaler_path, 'rb') as f:
            self.data_processor.feature_scaler = pickle.load(f)
        
        # Get recent market data leading up to prediction date
        start_date = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
        raw_market_data = self.data_processor.download_market_data(start_date, target_date)
        market_features = self.data_processor.extract_anonymous_features(raw_market_data)
        
        # Use last 30 days as input sequence
        input_sequence = market_features[-30:].reshape(1, 30, -1, 3)
        input_tensor = torch.FloatTensor(input_sequence)
        
        # Make unified prediction with multiple shuffles to reduce position bias
        self.model.eval()
        with torch.no_grad():
            # Run multiple predictions with different random shuffles to average out position bias
            num_prediction_shuffles = 10
            all_decision_logits = []
            
            for _ in range(num_prediction_shuffles):
                # Apply random shuffling during prediction to maintain position-agnostic behavior
                shuffle_indices = np.random.permutation(len(self.data_processor.sp500_tickers))
                shuffled_input = input_tensor[:, :, shuffle_indices, :]
                
                # Get prediction for this shuffle
                shuffled_logits = self.model(shuffled_input)
                
                # Un-shuffle the predictions to restore original stock order
                # Only un-shuffle stock predictions (indices 2 and onwards)
                unshuffled_logits = shuffled_logits.clone()
                stock_logits_shuffled = shuffled_logits[:, 2:]  # Extract stock predictions
                stock_logits_unshuffled = torch.zeros_like(stock_logits_shuffled)
                
                # Reverse the shuffle to get original order
                reverse_indices = np.argsort(shuffle_indices)
                stock_logits_unshuffled[:, reverse_indices] = stock_logits_shuffled
                
                # Combine HOLD/CASH (unchanged) with unshuffled stock predictions
                unshuffled_logits[:, 2:] = stock_logits_unshuffled
                all_decision_logits.append(unshuffled_logits)
            
            # Average predictions across all shuffles for robust decision making
            averaged_logits = torch.stack(all_decision_logits).mean(dim=0)
            decision_probabilities = torch.softmax(averaged_logits, dim=1)
            
            # Get top 3 predictions from averaged results
            top3_probs, top3_indices = torch.topk(decision_probabilities, k=3, dim=1)
            top3_probs = top3_probs[0].tolist()  # Convert to list
            top3_indices = top3_indices[0].tolist()
            
            predicted_choice = top3_indices[0]
            confidence = top3_probs[0]
        
        # Decode top 3 predictions
        top3_choices = []
        for idx, prob in zip(top3_indices, top3_probs):
            if idx == 0:
                choice_name = "HOLD"
                choice_stock = None
            elif idx == 1:
                choice_name = "CASH"
                choice_stock = None
            else:
                choice_name = "SWITCH"
                choice_stock = self.data_processor.sp500_tickers[idx - 2]
            top3_choices.append((choice_name, choice_stock, prob))
        
        # Decode unified prediction (top choice)
        action_name, target_stock, _ = top3_choices[0]
        
        print(f"Action: {action_name} (confidence: {confidence:.2f})")
        if target_stock:
            print(f"Model selected: {target_stock}")
        
        return action_name, target_stock, top3_choices