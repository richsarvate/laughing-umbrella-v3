""""""

Transformer model for stock ranking prediction.Transformer model for stock return prediction.

""""""

import torchimport torch

import torch.nn as nnimport torch.nn as nn



class StockTransformer(nn.Module):class StockTransformer(nn.Module):

    """2-layer Transformer encoder with mean pooling for ranking."""    """2-layer Transformer encoder with mean pooling."""

        

    def __init__(self, input_dim=25, hidden_dim=128, num_heads=2, num_layers=2):    def __init__(self, input_dim=25, hidden_dim=128, num_heads=2, num_layers=2):

        super().__init__()        super().__init__()

                

        self.input_proj = nn.Linear(input_dim, hidden_dim)        self.input_proj = nn.Linear(input_dim, hidden_dim)

                

        encoder_layer = nn.TransformerEncoderLayer(        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_dim,            d_model=hidden_dim,

            nhead=num_heads,            nhead=num_heads,

            dim_feedforward=hidden_dim * 4,            dim_feedforward=hidden_dim * 4,

            dropout=0.1,            dropout=0.1,

            batch_first=True            batch_first=True

        )        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                

        self.head = nn.Sequential(        self.head = nn.Sequential(

            nn.Linear(hidden_dim, 64),            nn.Linear(hidden_dim, 64),

            nn.ReLU(),            nn.ReLU(),

            nn.Dropout(0.3),            nn.Dropout(0.3),

            nn.Linear(64, 1)            nn.Linear(64, 1)

        )        )

        

    def forward(self, x):    def forward(self, x):

        """        """

        Args:        Args:

            x: (batch_size, seq_length, input_dim) e.g. (64, 60, 25)            x: (batch_size, seq_length, input_dim) e.g. (64, 60, 25)

        Returns:        Returns:

            (batch_size,) - ranking scores            (batch_size,) - predicted 5-day returns

        """        """

        x = self.input_proj(x)        x = self.input_proj(x)

        x = self.transformer(x)        x = self.transformer(x)

        x = x.mean(dim=1)  # Mean pooling over sequence        x = x.mean(dim=1)  # Mean pooling over sequence

        return self.head(x).squeeze(-1)        return self.head(x).squeeze(-1)

