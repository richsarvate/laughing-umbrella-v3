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

## DATA DOWNLOADER

Create a data downloader that downloads all the data needed one time only. 2010-present. All other components that need data will reference this cache. Other components will not do any downloading

## DATA PREPROCESSING

After data is downloaded a separate script will preprocess the data (for the entire 2010-present)

1. add all calculated fields
2. Normalize features per stock (rolling z-score). Rolling z-score window of 252 trading days
3. Lag ALL features to prevent lookahead bias

---

## ARCHITECTURE
A 2-layer Transformer encoder (2 attention heads, 128-dim hidden size) with mean pooling and a small MLP head that maps the 60-day × 26-feature sequence to a single scalar predicted next-week return.

---

## TRAINING
Each training sample is one independent 60-day sequence (for one stock at one time).  

The model sees each sequence separately and learns to predict that window’s 5 day return.  

Sequences are not connected or related during training — no memory, no shared context between samples.  

Use sliding 60-day windows for samples.

Train/validate by time split, not random.  
Train on data from 2010–2023  
Validate on 2024 

## LOSS FUNCTION
MSE (Mean Squared Error) - standard for regression

---

## TEST/BACKTEST
Download data for 2025 one time and save it in a folder for future use

Backtest on 2025  

Use the same model for all stocks. For each stock:  
Take its last 60 days × 26 features (normalized).  
Feed that into the model → get 1 predicted return.  
Input per stock: last 60 days × 26 features → shape (1, 60, 26)  
Model output: one scalar = predicted next-week return  

**Loop:**  
Repeat for every stock in the s&p.  
Collect all predictions → rank them.  
Buy the top 3 stocks, short the bottom 3 ones.  
Hold for 5 days → measure portfolio return.  

Stop after 3 cycles  
Report return for the whole run.  

## TODO ONLY ONCE REST OF PROJECT IS DONE

## PREDICTION SCRIPT
1. download todays data and last 60 days from yfinance
2. preprocess data
3. say which 3 stocks to buy and which 3 to short
