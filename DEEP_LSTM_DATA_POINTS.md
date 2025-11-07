# Deep LSTM Model - Data Points (Features) Breakdown

## 📊 Overview

**Total Features:** 20  
**Sequence Length:** 15 trades (lookback)  
**Data Points per Prediction:** 300 (15 trades × 20 features)  
**Input Shape:** (batch_size, 15, 20)  
**Output Shape:** (batch_size, 1) - Win probability

---

## 📋 Complete Feature List

### 1. **Risk** (Risk Management)
- **Type:** Float
- **Description:** The risk amount (distance from entry to stop loss)
- **Purpose:** Most important feature (11.67% importance in RF model)
- **Range:** Typically 0-50+ dollars

### 2. **WindowType** (Strategy Context)
- **Type:** Integer/String
- **Description:** Which key time window (3:00, 10:00, or 16:30)
- **Purpose:** Identifies which trading window the trade came from
- **Values:** 0, 1, 2, or "0300", "1000", "1630"

### 3. **EMA_9_5M** (Exponential Moving Average)
- **Type:** Float
- **Description:** 9-period EMA on 5-minute timeframe
- **Purpose:** Short-term trend indicator
- **Calculation:** EMA(9) on Close price, 5M chart

### 4. **EMA_21_5M** (Exponential Moving Average)
- **Type:** Float
- **Description:** 21-period EMA on 5-minute timeframe
- **Purpose:** Medium-term trend indicator
- **Calculation:** EMA(21) on Close price, 5M chart

### 5. **EMA_50_5M** (Exponential Moving Average)
- **Type:** Float
- **Description:** 50-period EMA on 5-minute timeframe
- **Purpose:** Longer-term trend indicator
- **Calculation:** EMA(50) on Close price, 5M chart

### 6. **EMA_200_1H** (Exponential Moving Average)
- **Type:** Float
- **Description:** 200-period EMA on 1-hour timeframe
- **Purpose:** Major trend filter (8.92% importance)
- **Calculation:** EMA(200) on Close price, 1H chart, forward-filled to 5M

### 7. **ATR** (Average True Range)
- **Type:** Float
- **Description:** 14-period ATR - measures volatility
- **Purpose:** Volatility indicator (8.00% importance)
- **Calculation:** 14-period exponential moving average of True Range

### 8. **ATR_Pct** (ATR as Percentage)
- **Type:** Float
- **Description:** ATR as percentage of current price
- **Purpose:** Normalized volatility measure
- **Calculation:** (ATR / Close) × 100

### 9. **ATR_Ratio** (ATR Relative Ratio)
- **Type:** Float
- **Description:** Current ATR vs recent average (10.20% importance)
- **Purpose:** Compares current volatility to recent average
- **Calculation:** ATR / ATR_MA(20)

### 10. **Is_Consolidating** (Consolidation Indicator)
- **Type:** Binary (0 or 1)
- **Description:** Whether market is in consolidation (low volatility)
- **Purpose:** Identifies low volatility periods
- **Calculation:** ATR < ATR_MA × 0.7

### 11. **Is_Tight_Range** (Tight Range Indicator)
- **Type:** Binary (0 or 1)
- **Description:** Whether price is in a tight range
- **Purpose:** Identifies range-bound markets
- **Calculation:** Price range < 80% of average range

### 12. **Consolidation_Score** (Combined Consolidation)
- **Type:** Float (0-1)
- **Description:** Combined consolidation metric
- **Purpose:** Overall consolidation strength
- **Calculation:** (Is_Consolidating + Is_Tight_Range) / 2

### 13. **Trend_Score** (Overall Trend Strength)
- **Type:** Float (0-1)
- **Description:** Combined trend alignment score
- **Purpose:** Measures overall trend strength
- **Calculation:** (EMA_9_Above_21 + EMA_21_Above_50 + Price_Above_EMA200_1H) / 3

### 14. **EMA_9_Above_21** (EMA Alignment)
- **Type:** Binary (0 or 1)
- **Description:** Whether EMA 9 is above EMA 21
- **Purpose:** Short-term bullish alignment (7.83% importance)
- **Calculation:** EMA_9_5M > EMA_21_5M

### 15. **EMA_21_Above_50** (EMA Alignment)
- **Type:** Binary (0 or 1)
- **Description:** Whether EMA 21 is above EMA 50
- **Purpose:** Medium-term bullish alignment (7.61% importance)
- **Calculation:** EMA_21_5M > EMA_50_5M

### 16. **Price_Above_EMA200_1H** (Major Trend Filter)
- **Type:** Binary (0 or 1)
- **Description:** Whether price is above 200 EMA on 1H
- **Purpose:** Major trend alignment (8.92% importance)
- **Calculation:** Close > EMA_200_1H

### 17. **EntryHour** (Time Feature)
- **Type:** Integer (0-23)
- **Description:** Hour of day when trade was entered
- **Purpose:** Captures time-based patterns
- **Range:** 0-23

### 18. **EntryDayOfWeek** (Time Feature)
- **Type:** Integer (0-6)
- **Description:** Day of week (Monday=0, Sunday=6)
- **Purpose:** Captures weekly patterns
- **Range:** 0-6

### 19. **RangeSize** (Window Size)
- **Type:** Float
- **Description:** Size of the 15-minute window (High - Low)
- **Purpose:** Measures breakout range size
- **Calculation:** WindowHigh - WindowLow

### 20. **RangeSizePct** (Range as Percentage)
- **Type:** Float
- **Description:** Range size as percentage of entry price (8.29% importance)
- **Purpose:** Normalized range size measure
- **Calculation:** (RangeSize / EntryPrice) × 100

---

## 🎯 Feature Categories

### **Risk Management (1 feature)**
- Risk amount (most important: 11.67%)

### **Moving Averages (7 features)**
- EMA_9_5M, EMA_21_5M, EMA_50_5M, EMA_200_1H
- EMA_9_Above_21, EMA_21_Above_50, Price_Above_EMA200_1H

### **Volatility (3 features)**
- ATR, ATR_Pct, ATR_Ratio (10.20% importance)

### **Consolidation (3 features)**
- Is_Consolidating, Is_Tight_Range, Consolidation_Score

### **Trend (1 feature)**
- Trend_Score (combined trend strength)

### **Time (2 features)**
- EntryHour, EntryDayOfWeek

### **Range (2 features)**
- RangeSize, RangeSizePct (8.29% importance)

### **Strategy Context (1 feature)**
- WindowType (which key time window)

---

## 📐 Data Structure

### **For Each Prediction:**
```
Input: 15 sequences × 20 features = 300 data points
  └─ Sequence 1: [Risk, WindowType, EMA_9_5M, ..., RangeSizePct]
  └─ Sequence 2: [Risk, WindowType, EMA_9_5M, ..., RangeSizePct]
  └─ ...
  └─ Sequence 15: [Risk, WindowType, EMA_9_5M, ..., RangeSizePct]

Output: 1 value (Win probability: 0-1)
```

### **Model Architecture:**
- **Input Layer:** (None, 15, 20)
- **LSTM Layers:** 10 layers processing sequences
- **Dense Layers:** 2 layers for final prediction
- **Output:** Win probability (0-1)

---

## 🔍 Feature Importance (from Random Forest)

Based on Random Forest model analysis:

1. **Risk** - 11.67% (Most Important)
2. **ATR_Ratio** - 10.20%
3. **EMA_200_1H alignment** - 8.92%
4. **RangeSizePct** - 8.29%
5. **ATR_Pct** - 8.00%
6. **EMA_9_Above_21** - 7.83%
7. **EMA_21_Above_50** - 7.61%
8. **Trend_Score** - 2.31%

---

## 💡 How LSTM Uses These Features

### **Sequential Learning:**
- LSTM processes 15 trades in sequence
- Each trade has 20 features
- Model learns patterns across time
- Captures dependencies between trades

### **Feature Relationships:**
- **Temporal:** How features change over 15 trades
- **Cross-feature:** Relationships between different features
- **Pattern Recognition:** Identifies winning trade sequences

### **Example:**
```
Trade 1: Risk=5, EMA_9_Above_21=1, ATR_Ratio=1.2, ...
Trade 2: Risk=6, EMA_9_Above_21=1, ATR_Ratio=1.1, ...
...
Trade 15: Risk=4, EMA_9_Above_21=0, ATR_Ratio=0.9, ...
→ LSTM learns: "When Risk decreases, EMAs align, and ATR drops, what happens?"
```

---

## 📊 Data Preprocessing

1. **Missing Values:** Filled with median
2. **Scaling:** StandardScaler (mean=0, std=1)
3. **Sequence Creation:** 15-trade lookback window
4. **Class Balancing:** Class weights applied during training

---

## 🎯 Summary

**Total Data Points:** 300 per prediction (15 trades × 20 features)

**Key Features:**
- Risk management (Risk)
- Trend indicators (4 EMAs + 3 alignment flags)
- Volatility (3 ATR features)
- Consolidation (3 features)
- Time context (2 features)
- Range size (2 features)
- Strategy context (WindowType)

**Model Capacity:** 1,225,345 parameters to learn patterns from these 20 features across 15 trade sequences.

---

*Last Updated: November 6, 2025*

