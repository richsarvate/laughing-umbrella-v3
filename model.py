"""
Transformer model for stock return prediction.
"""
import torch
import torch.nn as nn

class StockTransformer(nn.Module):
    """2-layer Transformer encoder."""
    
    def __init__(self, input_dim=26, hidden_dim=128, num_heads=2, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_dim)
        Returns:
            (batch_size,) - predicted returns
        """
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Mean pooling
        return self.head(x).squeeze(-1)
