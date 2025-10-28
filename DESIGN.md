# Stock Trader Model

## EMBEDDINGS
**Core Price Fields**  
Open  
High  
Low  
Close  
Volume  
Adjusted Close  
NO TICKER SYMBOL  

---

**Derived / Technical Features**  
Daily Return  
5-Day Return  
10-Day Return  
20-Day Return  
SMA_5 (5-day Simple Moving Average)  
SMA_10  
SMA_20  
EMA_10 (10-day Exponential Moving Average)  
Close/SMA_10  
SMA_5/SMA_20  
Rolling Volatility_10  
Rolling Volatility_20  
ATR_14 (Average True Range)  
RSI_14 (Relative Strength Index)  
MACD  
MACD_Signal  
Momentum_10  
Rate_of_Change_10  
Volume_ZScore_10  
Volume/SMA_Volume_10  

---

**Market Context Fields**  
SP500_Return (from ^GSPC)  
VIX_Level (from ^VIX)  
Sector_Avg_Return (optional)  

---

**Macro Fields (from FRED)**  
Interest_Rate  
Inflation_Rate  
Unemployment_Rate  

---

## DATA PREPROCESSING
Normalize features per stock (rolling z-score).  
Lag features by one day.  

---

## ARCHITECTURE
A 2-layer Transformer encoder (2 attention heads, 128-dim hidden size) with mean pooling and a small MLP head that maps the 60-day × 32-feature sequence to a single scalar predicted next-week return.

---

## TRAINING
Each training sample is one independent 60-day sequence (for one stock at one time).  

The model sees each sequence separately and learns to predict that window’s 5 day return.  

Sequences are not connected or related during training — no memory, no shared context between samples.  

Use sliding 60-day windows for samples.  

Train/validate by time split, not random.  
Train on data from 2010–2023  
Validate on 2024  

---

## TEST/BACKTEST
Backtest on 2025  

Use the same model for all stocks. For each stock:  
Take its last 60 days × 32 features (normalized).  
Feed that into the model → get 1 predicted return.  
Input per stock: last 60 days × 32 features → shape (1, 60, 32)  
Model output: one scalar = predicted next-week return  

**Loop:**  
Repeat for every stock in the s&p.  
Collect all predictions → rank them.  
Buy the top 3 stocks, short the bottom 3 ones.  
Hold for 5 days → measure portfolio return.  

Stop after 3 cycles  
Report return for the whole run.  
