# 🧠 Position-Agnostic Stock Trading Transformer

**A transformer that learns pure market patterns without position bias**

## 🎯 How Training Works

### 1. **Position Shuffling**
The model never sees stocks in the same order twice:
- Each training batch randomly shuffles all stock positions
- Prevents learning "Apple is always first" or "Tesla is position 50"
- Forces the model to focus on technical patterns, not arrangement

### 2. **Anonymous Features**
No ticker symbols reach the model:
- Only technical indicators: momentum, volatility, RSI  
- Stocks become `STOCK_000`, `STOCK_001`, etc. internally
- Model must learn from price patterns alone

### 3. **Profit Optimization**
Directly maximizes trading returns:
```python
# Model learns: "Which choice will make the most money?"
loss = -expected_return  # Minimize negative return = maximize profit
```

## 🚀 Training Command

```bash
python train_model_2014_2024.py
```

This trains on 10 years (2014-2024) with position shuffling enabled.

## 📊 Recent Results (2014-2024 Training)

```
Training on 2,481 sequences with 308 stocks
Epoch 0:   -0.35% (learning from random)
Epoch 90:  +0.61% (found profitable patterns) 
Epoch 140: +0.61% (stable performance)
```

Model successfully learned to generate positive expected returns!

## 🏗️ Key Files

- **`core/training_system.py`** - Position shuffling & profit optimization
- **`core/data_processor.py`** - Anonymous feature extraction  
- **`core/model.py`** - Position-agnostic transformer architecture

## 🔬 Why This Approach Works

**Traditional Problem**: Models memorize "AAPL is always profitable" or "position 1 stocks perform better"

**Our Solution**: 
- ✅ Random position shuffling prevents position learning
- ✅ Anonymous features prevent ticker memorization  
- ✅ Technical patterns generalize across all stocks

**Result**: A model that finds genuine market patterns, not data artifacts.

---

*Simple, robust, and focused on real pattern learning.*