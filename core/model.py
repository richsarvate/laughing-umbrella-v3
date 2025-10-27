"""
Transformer model architecture for stock trading decisions.
Minimal transformer design using GPT-2 backbone for market pattern recognition.
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class TransformerStockTrader(nn.Module):
    """Position-agnostic transformer model for unified stock trading decisions.
    
    This model is designed to be completely position-invariant:
    - No positional embeddings for stocks (only for time sequence)
    - Input processing treats all stocks identically
    - Training uses random position shuffling to prevent position learning
    """
    
    def __init__(self, num_stocks: int = 500, features_per_stock: int = 3, hidden_size: int = 256):
        super().__init__()
        self.num_stocks = num_stocks
        self.features_per_stock = features_per_stock
        
        # Position-agnostic input projection: [stocks * features] -> hidden_size
        # This treats all stock positions identically during processing
        self.input_projection = nn.Sequential(
            nn.Dropout(0.15),  # Input dropout for noisy financial data
            nn.Linear(num_stocks * features_per_stock, hidden_size),
            nn.LayerNorm(hidden_size)  # Add layer norm for stability with shuffled inputs
        )
        
        # Minimal transformer config (2 layers, 4 attention heads)
        # Only temporal positions are used - no stock position embeddings
        transformer_config = GPT2Config(
            vocab_size=1,  # Not used for our case
            n_positions=512,  # Only for time sequence, not stock positions
            n_embd=hidden_size,
            n_layer=2,
            n_head=4,
            resid_pdrop=0.2,
            attn_pdrop=0.1,
        )
        self.transformer_backbone = GPT2Model(transformer_config)
        
        # Position-agnostic decision head: HOLD(0) + CASH(1) + all stocks(2+)
        num_total_choices = 2 + num_stocks  # HOLD, CASH, + 500 stocks
        self.unified_decision_head = nn.Sequential(
            nn.Dropout(0.2),       # Input dropout to decision head
            nn.Linear(hidden_size, 128),  # Slightly larger intermediate layer
            nn.ReLU(),
            nn.LayerNorm(128),     # Additional normalization for stability
            nn.Dropout(0.25),      # Increased dropout for 502 choices
            nn.Linear(128, num_total_choices)
        )
        
    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        """
        Position-agnostic forward pass through transformer.
        
        Args:
            market_features: [batch_size, sequence_length, num_stocks, features_per_stock]
                            Note: Stock positions are shuffled during training to prevent 
                            position-based learning
        Returns:
            decision_logits: [batch_size, 502] - logits for HOLD/CASH/all stocks
                           Stock logits correspond to shuffled positions during training
        """
        batch_size, seq_len, num_stocks, features = market_features.shape
        
        # Flatten stock features for each timestep (position order doesn't matter)
        # During training, positions are randomly shuffled
        # Use reshape instead of view to handle non-contiguous tensors from shuffling
        flattened_features = market_features.reshape(batch_size, seq_len, -1)
        
        # Project to hidden dimension (treats all positions identically)
        projected_input = self.input_projection(flattened_features)
        
        # Pass through transformer backbone
        # Only temporal attention is used - no positional bias for stocks
        transformer_output = self.transformer_backbone(inputs_embeds=projected_input)
        final_hidden_state = transformer_output.last_hidden_state[:, -1, :]  # [batch, hidden]
        
        # Make unified decision: HOLD/CASH or stock choice (position-agnostic)
        decision_logits = self.unified_decision_head(final_hidden_state)
        
        return decision_logits