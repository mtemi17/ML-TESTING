# WHAT THE TRAINED ML MODEL ACTUALLY USES

## ✅ CONFIRMED: ML MODEL FEATURES (From ml_service.py)

### NUMERIC FEATURES (29 features):
1. EntryPrice
2. SL (Stop Loss)
3. TP (Take Profit)
4. Risk
5. BreakoutDistance
6. BreakoutBodyPct
7. BreakoutAtrMultiple
8. RangeWidth
9. RangeMid
10. EntryOffset
11. RangeAtrRatio
12. **PriceAboveEMA200_5M** ← EMA 200 (5M) as FEATURE
13. ATR_Value
14. **EMA_9_5M** ← EMA 9 as FEATURE
15. **EMA_21_5M** ← EMA 21 as FEATURE
16. **EMA_50_5M** ← EMA 50 as FEATURE
17. **EMA_200_5M** ← EMA 200 (5M) as FEATURE
18. **EMA_200_1H** ← EMA 200 (1H) as FEATURE
19. ATR
20. ATR_Pct
21. ATR_Ratio
22. Consolidation_Score
23. Trend_Score
24. Is_Consolidating
25. Is_Tight_Range
26. **Price_Above_EMA200_5M** ← Binary feature
27. **Price_Above_EMA200_1H** ← Binary feature

### CATEGORICAL FEATURES (6 features):
1. EntryType
2. Type (BUY/SELL)
3. WindowType (0300, 1000, 1630)
4. WindowID
5. Mode
6. Market

---

## ❌ WHAT THE MODEL DOES NOT USE:

- **NO RSI** ❌
- **NO WMA** ❌
- **NO Stochastic** ❌
- **NO MACD** ❌
- **NO Bollinger Bands** ❌

**The model ONLY uses:**
- EMAs (9, 21, 50, 200 on 5M and 1H)
- ATR (14 period)
- Consolidation/Trend scores
- Breakout metrics
- Range metrics

---

## 🔍 KEY DIFFERENCE: EMA 200 IN MODEL vs EA

### In ML Model:
- **EMA_200_1H** is a **FEATURE** (input to model)
- Model uses it to **PREDICT** win probability
- Model considers it along with 28 other features
- Model outputs a **probability** (0-1)

### In EAAI_Full.mq5:
- **EMA 200 filter** is a **HARD FILTER** (blocks trades)
- If price < EMA 200 (LONG) or price > EMA 200 (SHORT) → **BLOCKED**
- No ML model prediction
- Just a simple rule: "Don't trade if price is on wrong side of EMA 200"

---

## 🚨 THE PROBLEM:

### **EAAI_Full.mq5 is NOT using the ML model!**

1. **We trained an ML model** on breakout data with 29 numeric + 6 categorical features
2. **The ML model predicts win probability** based on all features
3. **But EAAI_Full.mq5 is NOT calling the ML model**
4. **Instead, it's using hard-coded filters** that block trades

### Why?
- The user requested a "self-contained" EA without ML dependencies
- So we removed the ML model integration
- We replaced it with hard-coded filters based on the model's insights
- But the filters are **too strict** and are blocking trades

---

## ✅ WHAT WE SHOULD DO:

### Option 1: Use the ML Model (Recommended)
1. **Integrate ML model** into EAAI_Full.mq5
2. **Call ml_service.py** via WebRequest
3. **Get win probability** from model
4. **Only take trades** with probability > threshold (e.g., 0.6)

### Option 2: Use Model's Feature Importance
1. **Keep EA self-contained** (no ML dependencies)
2. **Use the model's feature importance** to guide filters
3. **Remove EMA 200 hard filter** (it's too restrictive)
4. **Use Trend_Score, Consolidation_Score, ATR_Ratio** instead

### Option 3: Hybrid Approach
1. **Use ML model for prediction** (if available)
2. **Fall back to hard filters** (if ML service unavailable)
3. **Best of both worlds**

---

## 📊 MODEL FEATURE IMPORTANCE (From ml_model_summary.md)

Top 15 Most Important Features:
1. **Risk** (11.67%) - Most important!
2. **ATR_Ratio** (10.20%)
3. **EMA_200_1H** (8.92%) ← Used as FEATURE, not filter!
4. **RangeSizePct** (8.29%)
5. **ATR_Pct** (8.00%)
6. **EMA_21_5M** (7.83%)
7. **ATR** (7.79%)
8. **EMA_9_5M** (7.61%)
9. **EMA_50_5M** (7.34%)
10. **RangeSize** (6.92%)
11. **EntryHour** (3.89%)
12. **EntryDayOfWeek** (3.32%)
13. **Trend_Score** (2.31%)
14. **WindowType** (1.41%)
15. **EMA_21_Above_50** (1.01%)

---

## 🎯 KEY INSIGHTS:

1. **EMA_200_1H is used as a FEATURE, not a filter**
   - Model considers it along with other features
   - Model predicts win probability
   - Model doesn't just block trades

2. **Risk is the most important feature (11.67%)**
   - But we're not using it in filters!
   - We're using fixed risk (1% of account)
   - But the model learned that risk amount matters

3. **ATR_Ratio is second most important (10.20%)**
   - We have this in filters
   - But it's part of strict filters only

4. **Trend_Score is less important (2.31%)**
   - But we're using it as a strict filter
   - Model says it's less important than ATR_Ratio

---

## ✅ RECOMMENDATION:

### **USE THE ML MODEL!**

The ML model was trained on:
- ✅ Breakout data
- ✅ EMA features (9, 21, 50, 200)
- ✅ ATR features
- ✅ Trend/Consolidation scores
- ✅ Breakout metrics

**The model knows what works!**

Instead of hard filters, we should:
1. **Call ML model** for each trade opportunity
2. **Get win probability** (0-1)
3. **Only take trades** with probability > 0.6 (or similar threshold)
4. **Let the model decide** based on all 29+6 features

**This is what we trained the model for!**

