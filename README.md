# Stock Trader Model# Stock Trader Model# Stock Trader Model# Stock Trading Transformer# 



Simple transformer-based stock selection using technical analysis.



## PipelineSimple transformer-based stock selection model.A transformer that learns market patterns without cheating.



1. **Download** - Get all S&P 500 data (2010-2025)

2. **Preprocess** - Calculate 26 features, normalize, lag

3. **Train** - Train transformer on 2010-2023, validate on 2024## StructureSimple transformer-based stock selection model.

4. **Backtest** - Test on 2025 with long-short strategy

```

## Setup

data/## The Problem

```bash

pip install -r requirements.txt  data_loader.py      # Download S&P 500 data

```

features/## StructureMost trading models cheat by memorizing:

## Usage

  engineer.py         # Calculate 26 technical features

### Step 1: Download Data

```bashmodel/```- "Apple is always first in the data"

python download.py

```  transformer.py      # 2-layer transformer architecture

Downloads raw OHLCV data for all S&P 500 stocks. Saves to `data/raw/`.

download_data.py      # One-time data downloaddata/- "Position 50 always has Tesla" 

### Step 2: Preprocess

```bashtrain.py              # Training pipeline

python preprocess.py

```backtest.py           # 2025 backtesting  data_loader.py      # Download S&P 500 data- Company names and ticker symbols## 🎯 How Training Works## 🎯 How Training Works

Calculates 26 technical features, normalizes (rolling z-score), lags by 1 day. Saves to `data/processed/`.

```

### Step 3: Train

```bashfeatures/

python train.py

```## Setup

Trains transformer on 2010-2023 data, validates on 2024. Saves best model to `model_best.pth`.

```bash  engineer.py         # Calculate 26 technical features## Our Solution

### Step 4: Backtest

```bashpip install -r requirements.txt

python backtest.py

``````model/**Position Shuffling**: Randomly shuffle stock order every batch

Tests on 2025 data. Runs 3 cycles: predict all stocks → rank → long top 3, short bottom 3 → hold 5 days.



## Model

## Usage  transformer.py      # 2-layer transformer architecture

- **Architecture**: 2-layer transformer (128-dim, 2 heads)

- **Input**: 60-day sequences × 26 features

- **Output**: Predicted 5-day return

- **Loss**: MSE### 1. Download Data (First Time Only)train.py              # Training pipeline## Training

- **Features**: Price, volume, momentum, volatility (no ticker symbols)

```bash

## Design

python download_data.pybacktest.py           # 2025 backtestingData: Takes 30-day windows of stock market data and extracts technical indicators (momentum, volatility, RSI) for each stock.

See `DESIGN.md` for detailed specifications.

```

Downloads and caches all S&P 500 data from 2010-2025. Run this once before training.```



### 2. TrainTargets: For each day, calculates which stocks will actually be profitable in the next 5 days based on real price movements.

```bash

python train.py## Setup

```

Trains on 2010-2023, validates on 2024. Loads data from cache only.```bashLoss Function: Directly optimizes for profit - if the model predicts a stock will go up and it actually does, that reduces the loss.



### 3. Backtestpip install -r requirements.txt

```bash

python backtest.py```Training Process: Feeds thousands of these 30-day sequences through the transformer for 150 epochs

```

Tests on 2025 with long-short strategy (top 3 long, bottom 3 short). Loads data from cache only.



## Model## UsageEach training sequence contains:

- **Input**: 60-day sequences × 26 features per stock

- **Architecture**: 2-layer transformer, 128-dim hidden, 2 attention heads1. Time dimension: 30 consecutive trading days

- **Output**: Predicted 5-day return

- **Features**: Price, volume, momentum, volatility (no ticker symbols)### Train2. Stock dimension: 308 S&P 500 stocks (in shuffled order)

- **Normalization**: Per-stock rolling z-score (252-day window)

```bash3. Feature dimension: 3 technical indicators per stock:

## Design

See `DESIGN.md` for detailed specifications.python train.pya. Momentum: 5-day price change percentage


```b. Volatility: 20-day rolling standard deviation of returns

Trains on 2010-2023, validates on 2024.c. RSI: 14-day Relative Strength Index (0-1 normalized)

### Backtest
```bash
python backtest.py
```
Tests on 2025 with long-short strategy (top 3 long, bottom 3 short).

## Model
- **Input**: 60-day sequences × 26 features per stock
- **Architecture**: 2-layer transformer, 128-dim hidden, 2 attention heads
- **Output**: Predicted 5-day return
- **Features**: Price, volume, momentum, volatility (no ticker symbols)
- **Normalization**: Per-stock rolling z-score (252-day window)

## Design
See `DESIGN.md` for detailed specifications.
