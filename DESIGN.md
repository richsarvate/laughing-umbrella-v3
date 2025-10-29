# Stock Trader Model

## EMBEDDINGS
**Core Price Fields (5)**  
Open  
High  
Low  
Close  
Volume  
NO TICKER SYMBOL  

---

**Derived / Technical Features (20)**  
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

Create a data downloader that downloads all the data needed one time only. 2010-present. Present means today, whatever that is.

Use current S&P 500 constituents.

Make sure issues like columns with 2 names get cleaned up into 1 col.

All other components that need data will reference this cache. Other components will not do any downloading

## DATA PREPROCESSING

After data is downloaded a separate script will preprocess the data (for the entire 2010-present)

1. add all calculated fields
2. Normalize features with Cross-Sectional Z-Score. Normalize across only stocks with valid data on that day. After cross-sectional normalization data should be saved as one file per stock
3. Lag ALL features by 1 day to prevent lookahead bias, but NOT the target (target is the actual future return we're predicting)

**IMPORTANT:** Only normalize the 25 features cross-sectionally. The target (5-day forward return) should remain as raw percentage returns and NOT be normalized. This allows the model to predict actual returns instead of normalized z-scores.

---

## ARCHITECTURE
A 2-layer Transformer encoder (2 attention heads, 128-dim hidden size) with mean pooling and a small MLP head that maps the 60-day × 25-feature sequence to a single scalar ranking score.

Dropout: 0.3 (forces model to use diverse features, prevents collapse)

---

## TRAINING - PAIRWISE RANKING APPROACH

**Training samples:** Pairs of stocks from the same date.

**Data loading:** Group all stocks by date, then randomly sample pairs from same dates. Create ~100K pairs per epoch from the available stock-date combinations.

For each training batch:
1. Sample 2 random stocks from the same date
2. Check which stock ACTUALLY performed better (based on 5-day return)
3. Model predicts a ranking score for each stock
4. Loss penalizes if predicted order is wrong

**Example:**
- Date: 2020-01-15
- Stock A actual return: +2%, Stock B actual return: -1%
- Model predicts: A score = 0.3, B score = 0.8 (WRONG ORDER)
- Loss = max(0, score_B - score_A + margin) → penalty

This forces the model to learn RELATIVE performance (which stock beats which), not absolute returns. Can't collapse to predicting same score for everything because that would rank randomly.

Train/validate by time split, not random.  
Train on data from 2010–2023  
Validate on 2024 

**Hyperparameters:**
- Learning rate: 0.0005 (balanced for ranking loss)
- Weight decay: 0.01 (regularization)
- Epochs: 20
- Dropout: 0.3
- Margin: 0.1 (for ranking loss)

when training, run in the background without buffering:
nohup python3 -u train_model.py > training_new.log 2>&1 &

## LOSS FUNCTION
Pairwise Ranking Loss (MarginRankingLoss) - learns to rank stocks by predicted performance, not predict exact returns. This prevents model collapse to constant predictions.

**Validation metric:** Percentage of stock pairs correctly ranked by predicted scores (ranking accuracy).

---

## TEST/BACKTEST
Download data for 2025 one time and save it in a folder for future use

Backtest on 2025  

Use the same model for all stocks. For each stock:  
Take its last 60 days × 25 features (normalized).  
Feed that into the model → get 1 ranking score.  
Input per stock: last 60 days × 25 features → shape (1, 60, 25)  
Model output: one scalar ranking score (higher = expected better performance)

**Loop:**  
Repeat for every stock in the s&p.  
Collect all ranking scores → sort them (high to low).  
Buy the top 3 stocks (highest scores), short the bottom 3 (lowest scores).  
Hold for 5 days → measure portfolio return.  

Stop after 3 cycles  
Report return for the whole run.

**Note:** The model outputs relative ranking scores, not absolute return predictions. Scores only matter for ordering stocks relative to each other.  

## TODO ONLY ONCE REST OF PROJECT IS DONE

## PREDICTION SCRIPT
1. download todays data and last 60 days from yfinance
2. preprocess data
3. say which 3 stocks to buy and which 3 to short
