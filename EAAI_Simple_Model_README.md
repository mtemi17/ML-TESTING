# EAAI Simple Model EA

## ✅ Simple Breakout EA with Model-Based Filters

This EA implements the **exact simple strategy** you described, with filters based on what the ML model learned.

---

## 🎯 Strategy (Exactly as You Described)

1. **Sessions:** 03:00 (Asian), 10:00 (London), 16:30 (NY)
2. **Range:** First 15-minute candle (high/low)
3. **Entry:** Close above range (bullish candle) OR below range (bearish candle)
4. **Window:** 3 hours after session start
5. **SL:** Opposite range end
6. **TP:** 2R

---

## 📊 Model-Based Filters

Based on the trained model's feature importance:

### Top Features:
1. **ATR** (Most Important - 0.62%)
   - Filter: Min/Max ATR thresholds
   - Default: No filter (set `InpMinATR` and `InpMaxATR` to enable)

2. **Consolidation_Score** (Second - 0.36%)
   - Filter: Max Consolidation Score
   - Default: 0.5 (blocks consolidating markets)

3. **EMA_200_1H** (Fourth - 0.36%)
   - Filter: Price must be above EMA 200 (1H) for LONG, below for SHORT
   - Default: OFF (set `InpUseEMA200Filter = true` to enable)

---

## ⚙️ Input Parameters

### Sessions
- `InpAsianStartHour` / `InpAsianStartMinute` - Asian session start (default: 3:00)
- `InpLondonStartHour` / `InpLondonStartMinute` - London session start (default: 10:00)
- `InpNYStartHour` / `InpNYStartMinute` - NY session start (default: 16:30)
- `InpTradingWindowHours` - Trading window after session start (default: 3 hours)

### Risk Management
- `InpRiskPercent` - Risk % per trade (default: 1.0%)
- `InpRewardToRisk` - Reward-to-risk ratio (default: 2.0)

### Model-Based Filters
- `InpUseModelFilters` - Enable/disable model filters (default: true)
- `InpMinATR` - Minimum ATR (0 = no filter)
- `InpMaxATR` - Maximum ATR (0 = no filter)
- `InpMaxConsolidationScore` - Max consolidation score (default: 0.5)
- `InpUseEMA200Filter` - Use EMA 200 (1H) filter (default: false)

### Position Limits
- `InpMaxTotalPositions` - Max total positions (default: 3)
- `InpMaxPositionsPerSymbol` - Max positions per symbol (default: 1)

---

## 📈 Expected Performance

Based on training data:
- **Baseline Win Rate:** 33.2%
- **Model Test Accuracy:** 59.56%
- **Total P&L (training):** $635.53
- **Trades:** 1,121 (611 Gold + 510 GBPJPY)

---

## 🚀 How to Use

1. **Compile** `EAAI_Simple_Model.mq5` in MT5
2. **Attach** to chart (5M timeframe recommended)
3. **Configure** session times if needed
4. **Set** risk parameters
5. **Enable/disable** model filters as desired
6. **Run** in Strategy Tester or live

---

## 💡 Key Features

- ✅ **Simple Strategy** - Exactly as you described
- ✅ **Candle Confirmation** - Bullish candle for LONG, bearish for SHORT
- ✅ **Model-Based Filters** - Uses what the model learned
- ✅ **Self-Contained** - No WebRequest, no external dependencies
- ✅ **One Trade Per Session** - Prevents overtrading
- ✅ **Position Limits** - Risk management built-in

---

## 📝 Notes

- The EA uses **5M timeframe** for breakout detection
- The EA uses **15M timeframe** for range detection
- The EA uses **1H timeframe** for EMA 200 filter (if enabled)
- Only **one trade per session** is allowed
- Trades are labeled with session name for tracking

---

**Status:** ✅ **READY TO USE**

