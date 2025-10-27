# Stock Trading Transformer# 🧠 Position-Agnostic Stock Trading Transformer# 🧠 Position-Agnostic Stock Trading Transformer



A transformer that learns market patterns without cheating.



## The Problem**A transformer that learns pure market patterns without position bias****A transformer that learns pure market patterns without position bias**

Most trading models cheat by memorizing:

- "Apple is always first in the data"

- "Position 50 always has Tesla" 

- Company names and ticker symbols## 🎯 How Training Works## 🎯 How Training Works



## Our Solution

**Position Shuffling**: Randomly shuffle stock order every batch

**Anonymous Data**: Only use technical patterns (momentum, volatility, RSI)### 1. **Position Shuffling**### 1. **Position Shuffling**



## ResultsThe model never sees stocks in the same order twice:The model never sees stocks in the same order twice:

```

Trained on 2014-2024 data (308 stocks)- Each training batch randomly shuffles all stock positions- Each training batch randomly shuffles all stock positions

Epoch 0:   -0.35% returns

Epoch 140: +0.61% returns- Prevents learning "Apple is always first" or "Tesla is position 50"- Prevents learning "Apple is always first" or "Tesla is position 50"

```

- Forces the model to focus on technical patterns, not arrangement- Forces the model to focus on technical patterns, not arrangement

## Usage

```bash

python train_model_2014_2024.py

```### 2. **Anonymous Features**### 2. **Anonymous Features**



The model learns genuine market patterns, not data quirks.No ticker symbols reach the model:No ticker symbols reach the model:

- Only technical indicators: momentum, volatility, RSI  - Only technical indicators: momentum, volatility, RSI  

- Stocks become `STOCK_000`, `STOCK_001`, etc. internally- Stocks become `STOCK_000`, `STOCK_001`, etc. internally

- Model must learn from price patterns alone- Model must learn from price patterns alone



### 3. **Profit Optimization**### 3. **Profit Optimization**

Directly maximizes trading returns:Directly maximizes trading returns:

```python```python

# Model learns: "Which choice will make the most money?"# Model learns: "Which choice will make the most money?"

loss = -expected_return  # Minimize negative return = maximize profitloss = -expected_return  # Minimize negative return = maximize profit

``````



## 🚀 Training Command## 🚀 Training Command



```bash```bash

python train_model_2014_2024.pypython train_model_2014_2024.py

``````



This trains on 10 years (2014-2024) with position shuffling enabled.This trains on 10 years (2014-2024) with position shuffling enabled.



## 📊 Recent Results (2014-2024 Training)## 📊 Recent Results (2014-2024 Training)



``````

Training on 2,481 sequences with 308 stocksTraining on 2,481 sequences with 308 stocks

Epoch 0:   -0.35% (learning from random)Epoch 0:   -0.35% (learning from random)

Epoch 90:  +0.61% (found profitable patterns) Epoch 90:  +0.61% (found profitable patterns) 

Epoch 140: +0.61% (stable performance)Epoch 140: +0.61% (stable performance)

``````



Model successfully learned to generate positive expected returns!Model successfully learned to generate positive expected returns!



## 🏗️ Key Files## 🏗️ Key Files



- **`core/training_system.py`** - Position shuffling & profit optimization- **`core/training_system.py`** - Position shuffling & profit optimization

- **`core/data_processor.py`** - Anonymous feature extraction  - **`core/data_processor.py`** - Anonymous feature extraction  

- **`core/model.py`** - Position-agnostic transformer architecture- **`core/model.py`** - Position-agnostic transformer architecture



## 🔬 Why This Approach Works## 🔬 Why This Approach Works



**Traditional Problem**: Models memorize "AAPL is always profitable" or "position 1 stocks perform better"**Traditional Problem**: Models memorize "AAPL is always profitable" or "position 1 stocks perform better"



**Our Solution**: **Our Solution**: 

- ✅ Random position shuffling prevents position learning- ✅ Random position shuffling prevents position learning

- ✅ Anonymous features prevent ticker memorization  - ✅ Anonymous features prevent ticker memorization  

- ✅ Technical patterns generalize across all stocks- ✅ Technical patterns generalize across all stocks



**Result**: A model that finds genuine market patterns, not data artifacts.**Result**: A model that finds genuine market patterns, not data artifacts.



------



*Simple, robust, and focused on real pattern learning.**Simple, robust, and focused on real pattern learning.*