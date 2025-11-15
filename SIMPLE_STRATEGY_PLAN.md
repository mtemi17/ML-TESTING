# SIMPLE STRATEGY - BACK TO BASICS

## ✅ WHAT YOU WANT (Simple & Clear)

### Strategy Rules:
1. **Sessions:** 03:00 (Asian), 10:00 (London), 16:30 (NY)
2. **Range:** First 15-minute candle (high/low)
3. **Entry:** Close above range (bullish candle) OR below range (bearish candle)
4. **Window:** 3 hours after session start
5. **SL:** Opposite range end
6. **TP:** 2R

### What You Had Before:
- EMA 200 (1H) - buy above, sell below
- EMA 200 (5M) - buy above, sell below
- **Problem:** EMA conflict (1H vs 5M)
- **Works:** When market moving
- **Loses:** When consolidating

### What You Want:
- Train model with **EMAs, RSI, Volume, ATR, and many datapoints**
- Model helps execute trades in your favor
- Use Gold (XAUUSD) and GBPJPY data
- Create EA based on model findings

---

## 🎯 WHAT I'M DOING NOW

### Step 1: Create Training Script ✅
- `train_simple_strategy_model.py` - Runs simple strategy
- Collects ALL features (EMAs, RSI, Volume, ATR, etc.)
- Labels trades as win/loss
- Trains model

### Step 2: Train Model (In Progress)
- Using Gold and GBPJPY data
- All features from your strategy
- Model learns what works

### Step 3: Create Simple EA
- Based on model findings
- Uses same simple strategy
- Applies model's learned filters

---

## 📊 FEATURES BEING COLLECTED

### Price & Range:
- EntryPrice, SL, TP, Risk
- RangeHigh, RangeLow, RangeWidth, RangeSizePct
- BreakoutDistance

### EMAs:
- EMA_9_5M, EMA_21_5M, EMA_50_5M, EMA_200_5M
- EMA_200_1H
- EMA relationships (9>21, 21>50, Price>EMA200)

### ATR:
- ATR, ATR_Ratio, ATR_Pct

### RSI:
- RSI (14 period)

### Volume:
- Volume, Volume_Ratio

### Candle:
- Candle_Body, Candle_Body_Pct
- Candle_Upper_Wick, Candle_Lower_Wick
- Is_Bullish, Is_Bearish

### Scores:
- Trend_Score, Consolidation_Score
- Is_Consolidating, Is_Tight_Range

---

## 🚀 NEXT STEPS

1. **Run training script** (currently running)
2. **Review model results** (feature importance, accuracy)
3. **Create simple EA** based on model findings
4. **Test EA** in Strategy Tester

---

**Status:** Training script created, running now...

