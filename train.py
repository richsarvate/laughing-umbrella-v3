"""
Step 3: Train the model on preprocessed data (2010-2023).
Validate on 2024.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from model import StockTransformer

def load_sequences(processed_dir, start_date, end_date, seq_length=60):
    """Load all 60-day sequences from preprocessed data."""
    all_sequences = []
    all_targets = []
    
    processed_files = list(processed_dir.glob("*.parquet"))
    
    for file in processed_files:
        import pandas as pd
        df = pd.read_parquet(file)
        
        # Filter date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_filtered = df[mask]
        
        if len(df_filtered) < seq_length + 5:
            continue
        
        # Extract features and targets
        feature_cols = [col for col in df_filtered.columns if col != 'target_5d_return']
        features = df_filtered[feature_cols].values
        targets = df_filtered['target_5d_return'].values
        
        # Create sliding windows
        for i in range(len(features) - seq_length):
            seq = features[i:i+seq_length]
            target = targets[i+seq_length-1]  # Target at end of sequence
            
            if not np.isnan(seq).any() and not np.isnan(target):
                all_sequences.append(seq)
                all_targets.append(target)
    
    return np.array(all_sequences), np.array(all_targets)

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train the model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Log every epoch
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_best.pth')
            print(f"  → New best model saved (val_loss: {val_loss:.6f})")

def main():
    print("="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    processed_dir = Path('data/processed')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading training data (2010-2023)...")
    X_train, y_train = load_sequences(processed_dir, '2010-01-01', '2023-12-31')
    
    print("Loading validation data (2024)...")
    X_val, y_val = load_sequences(processed_dir, '2024-01-01', '2024-12-31')
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(X_train):,} sequences")
    print(f"  Val: {len(X_val):,} sequences")
    print(f"  Sequence shape: {X_train[0].shape}")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    print("\nInitializing model...")
    model = StockTransformer(input_dim=26, hidden_dim=128, num_heads=2, num_layers=2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining...")
    train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device=device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("Model saved to: model_best.pth")
    print("\nNext step: Run 'python backtest.py'")

if __name__ == '__main__':
    main()
