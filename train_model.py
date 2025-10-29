"""
Step 3: Train the model using pairwise ranking loss (2010-2023), validate on 2024.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from model import StockTransformer
import random

def load_stock_data(processed_dir, start_date, end_date):
    """Load all stock data for a date range, grouped by date."""
    stock_data = {}
    
    for file in processed_dir.glob("*.parquet"):
        ticker = file.stem
        df = pd.read_parquet(file)
        
        # Filter date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_filtered = df[mask]
        
        if len(df_filtered) >= 60:
            stock_data[ticker] = df_filtered
    
    return stock_data

def create_pairwise_dataset(stock_data, num_pairs=100000, seq_length=60):
    """Create pairwise ranking dataset."""
    # Group stocks by date
    date_stocks = {}
    for ticker, df in stock_data.items():
        feature_cols = [col for col in df.columns if col != 'target_5d_return']
        for date in df.index[seq_length-1:]:  # Need 60 days before this date
            if date not in date_stocks:
                date_stocks[date] = []
            
            # Get 60-day sequence ending at this date
            idx = df.index.get_loc(date)
            if idx >= seq_length - 1:
                seq = df.iloc[idx-seq_length+1:idx+1][feature_cols].values
                target = df.loc[date, 'target_5d_return']
                
                if not np.isnan(seq).any() and not np.isnan(target):
                    date_stocks[date].append({
                        'ticker': ticker,
                        'sequence': seq,
                        'target': target
                    })
    
    # Create pairs
    pairs = []
    dates_with_stocks = [(date, stocks) for date, stocks in date_stocks.items() if len(stocks) >= 2]
    
    for _ in range(num_pairs):
        # Random date
        date, stocks = random.choice(dates_with_stocks)
        
        # Random pair from this date
        stock_a, stock_b = random.sample(stocks, 2)
        
        # Determine which performed better
        if stock_a['target'] > stock_b['target']:
            pairs.append((stock_a['sequence'], stock_b['sequence'], 1))  # A > B
        else:
            pairs.append((stock_a['sequence'], stock_b['sequence'], -1))  # B > A
    
    return pairs

def train_model(model, train_pairs, val_pairs, epochs=20, lr=0.0005, device='cpu'):
    """Train the model with pairwise ranking loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MarginRankingLoss(margin=0.1)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        
        # Shuffle pairs
        random.shuffle(train_pairs)
        
        # Mini-batches
        batch_size = 64
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            
            seq_a = torch.FloatTensor([p[0] for p in batch]).to(device)
            seq_b = torch.FloatTensor([p[1] for p in batch]).to(device)
            labels = torch.FloatTensor([p[2] for p in batch]).to(device)
            
            optimizer.zero_grad()
            score_a = model(seq_a)
            score_b = model(seq_b)
            
            loss = criterion(score_a, score_b, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Count correct rankings
            pred_order = (score_a > score_b).float() * 2 - 1  # Convert to -1/1
            train_correct += (pred_order == labels).sum().item()
        
        train_loss /= (len(train_pairs) / batch_size)
        train_acc = train_correct / len(train_pairs) * 100
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for i in range(0, len(val_pairs), batch_size):
                batch = val_pairs[i:i+batch_size]
                
                seq_a = torch.FloatTensor([p[0] for p in batch]).to(device)
                seq_b = torch.FloatTensor([p[1] for p in batch]).to(device)
                labels = torch.FloatTensor([p[2] for p in batch]).to(device)
                
                score_a = model(seq_a)
                score_b = model(seq_b)
                
                loss = criterion(score_a, score_b, labels)
                val_loss += loss.item()
                
                pred_order = (score_a > score_b).float() * 2 - 1
                val_correct += (pred_order == labels).sum().item()
        
        val_loss /= (len(val_pairs) / batch_size)
        val_acc = val_correct / len(val_pairs) * 100
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model_best.pth')
            print(f"  → New best model saved (val_acc: {val_acc:.1f}%)")

def main():
    print("="*60)
    print("TRAINING MODEL - PAIRWISE RANKING")
    print("="*60)
    
    processed_dir = Path('data/processed')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading training data (2010-2023)...")
    train_data = load_stock_data(processed_dir, '2010-01-01', '2023-12-31')
    print(f"Train stocks: {len(train_data)}")
    
    print("Loading validation data (2024)...")
    val_data = load_stock_data(processed_dir, '2024-01-01', '2024-12-31')
    print(f"Val stocks: {len(val_data)}")
    
    # Create pairwise datasets
    print("\nCreating pairwise training dataset...")
    train_pairs = create_pairwise_dataset(train_data, num_pairs=100000)
    print(f"Train pairs: {len(train_pairs):,}")
    
    print("Creating pairwise validation dataset...")
    val_pairs = create_pairwise_dataset(val_data, num_pairs=10000)
    print(f"Val pairs: {len(val_pairs):,}")
    
    # Get input dimension
    sample_seq = train_pairs[0][0]
    input_dim = sample_seq.shape[1]
    print(f"Sequence shape: (60, {input_dim})")
    
    # Initialize model
    print("\nInitializing model...")
    model = StockTransformer(input_dim=input_dim, hidden_dim=128, num_heads=2, num_layers=2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining...")
    train_model(model, train_pairs, val_pairs, epochs=20, lr=0.0005, device=device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("Model saved to: model_best.pth")
    print("\nNext step: Run 'python3 backtest.py'")

if __name__ == '__main__':
    main()
