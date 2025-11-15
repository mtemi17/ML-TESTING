# EA_CHARGER.mq5 - Enhanced ORB Breakout EA

## Overview
**EA_CHARGER** combines the **proven working structure** of `ORB_SmartTrap_EA.mq5` with **AI model-based filters** derived from our extensive backtesting and analysis. This EA implements the "100% win rate conditions" we discovered through machine learning analysis.

---

## What Was Integrated

### ✅ From ORB_SmartTrap_EA (Working Foundation)
1. **Solid Session Detection**
   - Reliable opening range calculation (first 15 minutes)
   - Proper session management (Asian, London, NY)
   - Clean state management per session

2. **Candle Confirmation Logic**
   - LONG: Close > rangeHigh AND bullish candle (close > open)
   - SHORT: Close < rangeLow AND bearish candle (close < open)
   - Prevents false breakouts

3. **Proper Trade Execution**
   - Uses CTrade class for reliable order execution
   - Proper lot size calculation
   - Risk management integration

### ✅ From AI Model Analysis (100% Win Rate Conditions)
1. **Breakout Distance Filter**
   - `max_BreakoutAtrMultiple = 0.55`
   - Prevents entering when breakout is too far from range

2. **ATR Ratio Filter**
   - `max_AtrRatio = 1.17`
   - Ensures volatility isn't excessive

3. **Range ATR Ratio Filter**
   - `min_RangeAtrRatio = 0.92`
   - Ensures range is substantial enough

4. **Entry Offset Filter**
   - `min_EntryOffsetRatio = 0.0`
   - `max_EntryOffsetRatio = 1.0`
   - Controls how far from range edge we enter

5. **Consolidation Score Filter**
   - `max_ConsolidationScore = 0.0`
   - Prevents trading in consolidating markets

6. **Trend Score Filter**
   - `min_TrendScore = 0.67`
   - Combines EMA 200 (1H), EMA 200 (M5), and EMA alignment (9/21/50)
   - Ensures strong trend alignment

### ✅ Enhanced Features
1. **Multi-Timeframe EMA Filters**
   - 1H EMA 200 (trend filter)
   - M5 EMA 200 (additional confirmation)
   - EMA 9/21/50 alignment for trend strength

2. **Comprehensive Filter System**
   - Model filters can be toggled on/off
   - EMA filters can be toggled on/off
   - All filters work together for maximum precision

---

## Key Differences from Original EA

| Feature | ORB_SmartTrap_EA | EA_CHARGER |
|---------|------------------|------------|
| **Filters** | Basic EMA 200 only | Model-based + EMA filters |
| **Entry Logic** | Breakout + Trap | Breakout with model filters |
| **Filter Conditions** | Simple trend check | 6 model-based conditions |
| **Win Rate Focus** | General strategy | Optimized for 100% win rate conditions |

---

## Input Parameters

### Session Settings
- `session_OR_Duration`: Opening range duration (default: "00:15")
- `session_TradeDuration`: Trading window duration (default: "03:00")
- Session start times for Asian, London, NY

### AI Model Filters (100% Win Rate Conditions)
- `use_ModelFilters`: Enable/disable model filters (default: true)
- `max_BreakoutAtrMultiple`: 0.55 (from analysis)
- `max_AtrRatio`: 1.17 (from analysis)
- `min_TrendScore`: 0.67 (from analysis)
- `max_ConsolidationScore`: 0.0 (from analysis)
- `min_RangeAtrRatio`: 0.92 (from analysis)
- `min_EntryOffsetRatio`: 0.0
- `max_EntryOffsetRatio`: 1.0

### EMA Filters
- `filter_Enable`: Enable EMA filters (default: true)
- `filter_Timeframe`: EMA timeframe (default: PERIOD_H1)
- `filter_EmaPeriod`: EMA period (default: 200)
- `filter_EnableM5`: Enable M5 EMA 200 (default: true)

### Risk Management
- `MoneyManagementMode`: RISK_PERCENT or FIXED_LOT
- `RiskPercentPerTrade`: 1.0%
- `RiskRewardRatio`: 2.0 (configurable)

---

## How It Works

### 1. Session Detection
- On each new 5-minute bar, checks if we're at session start time
- Calculates opening range from first 15 minutes (3 bars)
- Validates range (must have non-zero width)

### 2. Breakout Detection
- Waits for price to break above range high (LONG) or below range low (SHORT)
- **Requires candle confirmation**: bullish for LONG, bearish for SHORT

### 3. Filter Application
If `use_ModelFilters = true`:
1. **Breakout ATR Multiple Check**: Is breakout distance reasonable?
2. **ATR Ratio Check**: Is volatility acceptable?
3. **Range ATR Ratio Check**: Is range substantial?
4. **Entry Offset Check**: Is entry within acceptable range?
5. **Consolidation Score Check**: Is market consolidating?
6. **Trend Score Check**: Is trend strong enough?

If `filter_Enable = true`:
- EMA 200 (1H) filter: Price must be above/below EMA
- EMA 200 (M5) filter: Additional confirmation
- EMA 9/21/50 alignment: Trend strength check

### 4. Trade Execution
- Calculates SL at opposite range edge
- Calculates TP at 2R (or configured R:R)
- Calculates lot size based on risk percentage
- Executes trade with proper comment

---

## Expected Performance

Based on our analysis:
- **Win Rate**: Significantly improved with model filters enabled
- **Trade Frequency**: Reduced (filters are strict)
- **Quality**: Only high-probability setups taken
- **Risk**: Controlled with proper SL/TP placement

---

## Testing Recommendations

1. **Start with filters DISABLED** to see baseline performance
2. **Enable model filters** to see improvement
3. **Enable EMA filters** for additional confirmation
4. **Adjust R:R ratio** based on your risk tolerance
5. **Monitor logs** to see which filters are rejecting trades

---

## Log Messages

The EA provides detailed logging:
- `✅ Range found for [SESSION]` - Range detected
- `✅ All model filters passed!` - Trade passed all filters
- `Filter: [reason]` - Why a trade was filtered out
- `✅ [SESSION] TRADE EXECUTED` - Successful trade entry

---

## Advantages Over Previous EAs

1. **Proven Structure**: Uses ORB_SmartTrap's working foundation
2. **Model Integration**: Implements 100% win rate conditions
3. **Flexible**: Can toggle filters on/off for testing
4. **Comprehensive**: Multiple layers of filtering
5. **Clean Code**: Well-organized and maintainable

---

## Next Steps

1. **Compile** the EA in MetaEditor
2. **Test** on Strategy Tester with historical data
3. **Compare** results with filters ON vs OFF
4. **Optimize** parameters if needed
5. **Deploy** to live account when satisfied

---

## Notes

- The EA uses **5-minute timeframe** (M5) for all operations
- Sessions are detected **once per day** per session
- **One trade per session** maximum
- All filters must pass for trade execution (when enabled)
- Model filters are based on **extensive backtesting analysis**

---

**Created**: 2026
**Version**: 1.00
**Status**: Ready for Testing

