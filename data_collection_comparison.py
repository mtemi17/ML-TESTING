"""
COMPARISON: What Data EA Collects vs What Backtest Uses
"""

print("="*80)
print("DATA COLLECTION COMPARISON: EA vs BACKTEST")
print("="*80)

print("""
CRITICAL DIFFERENCES IN DATA COLLECTION:

1. PRICE DATA AT ENTRY TIME:
   
   EA (Live Trading):
   - Uses: iClose(_Symbol, PERIOD_M5, 1)  ← PREVIOUS bar's close
   - Checks: if(close > range.high) where close = bar[1] close
   - Entry price: close (from bar 1)
   - Current price: SymbolInfoDouble(_Symbol, SYMBOL_BID/ASK)
   
   Backtest (CSV):
   - Uses: row['Close']  ← CURRENT bar's close
   - Checks: if(close_price > range_high) where close_price = current row
   - Entry price: close_price (from current row)
   - Processes bars sequentially

   ⚠ ISSUE: EA checks breakout on PREVIOUS bar, backtest checks on CURRENT bar!

2. INDICATOR VALUES:
   
   EA:
   - ATR: GetATRValue(1)  ← ATR at shift 1 (previous bar)
   - EMA9: GetMAValue(g_ma9Handle, 1)  ← EMA at shift 1
   - EMA21: GetMAValue(g_ma21Handle, 1)  ← EMA at shift 1
   - EMA50: GetMAValue(g_ma50Handle, 1)  ← EMA at shift 1
   - EMA200_1H: GetMAValue(g_ma1hHandle, 0)  ← Current 1H bar
   
   Backtest:
   - ATR: row.get('ATR')  ← ATR calculated on current bar
   - EMA9: row.get('EMA_9_5M')  ← EMA on current bar
   - EMA21: row.get('EMA_21_5M')  ← EMA on current bar
   - EMA50: row.get('EMA_50_5M')  ← EMA on current bar
   - EMA200_1H: row.get('EMA_200_1H')  ← Forward-filled to current bar
   
   ⚠ ISSUE: EA uses indicators from PREVIOUS bar, backtest uses CURRENT bar!

3. RANGE CALCULATION:
   
   EA:
   - Uses: iBarShift() to find exact bars
   - Range high: iHigh(_Symbol, PERIOD_M5, i) for i in [endIndex:startIndex]
   - Range low: iLow(_Symbol, PERIOD_M5, i) for i in [endIndex:startIndex]
   - Window: 03:00-03:10 (3 bars)
   
   Backtest:
   - Uses: df.loc[start_time:end_time] (pandas slicing)
   - Range high: window_data['High'].max()
   - Range low: window_data['Low'].min()
   - Window: 03:00-03:10 (3 bars)
   
   ⚠ ISSUE: Different methods might find different bars!

4. TIMING:
   
   EA:
   - OnTick() fires on NEW bar
   - Checks: if(currentBarTime != lastBarTime)
   - Uses data from bar[1] (previous completed bar)
   - Entry happens on NEW bar formation
   
   Backtest:
   - Processes each bar in sequence
   - Uses data from current bar being processed
   - Entry happens on same bar as breakout detection
   
   ⚠ ISSUE: EA is 1 bar behind backtest!

5. BREAKOUT DETECTION:
   
   EA:
   - double close = iClose(_Symbol, PERIOD_M5, 1);  ← Bar 1 close
   - bool breakoutUp = (close > range.high);
   - Entry: if breakout detected on bar[1], enter on bar[0]
   
   Backtest:
   - close_price = row['Close'];  ← Current bar close
   - bool breakout_up = (close_price > range_high);
   - Entry: if breakout detected, enter immediately
   
   ⚠ ISSUE: EA detects breakout on previous bar, backtest on current!

6. ATR AVERAGE:
   
   EA:
   - GetATRAverage(1, 20)  ← Average of ATR from bar[1] to bar[20]
   - Uses CopyBuffer() to get 20 bars starting from shift 1
   
   Backtest:
   - ATR_Ratio = ATR / ATR.rolling(window=20).mean()
   - Uses rolling window on current bar
   
   ⚠ ISSUE: Different calculation methods!

7. CANDLE DATA:
   
   EA:
   - iHigh(_Symbol, PERIOD_M5, 1)  ← Previous bar high
   - iLow(_Symbol, PERIOD_M5, 1)  ← Previous bar low
   - iOpen(_Symbol, PERIOD_M5, 1)  ← Previous bar open
   - iClose(_Symbol, PERIOD_M5, 1)  ← Previous bar close
   
   Backtest:
   - row['High']  ← Current bar high
   - row['Low']  ← Current bar low
   - row['Open']  ← Current bar open
   - row['Close']  ← Current bar close
   
   ⚠ ISSUE: EA uses previous bar, backtest uses current!

8. BREAKOUT DISTANCE:
   
   EA:
   - breakoutDistance = isLong ? MathMax(price - range.high, 0.0)
   - price = close (from bar 1)
   
   Backtest:
   - breakout_distance = max(close_price - range_high, 0.0)
   - close_price = current bar close
   
   ⚠ ISSUE: Different price reference!

SUMMARY OF THE PROBLEM:

The EA operates on a "lookback" basis - it checks if the PREVIOUS bar
broke out, then enters on the CURRENT bar.

The backtest operates on a "current" basis - it checks if the CURRENT
bar breaks out, then enters immediately.

This means:
- EA sees breakout 1 bar later than backtest
- EA uses indicator values from 1 bar earlier
- EA's entry price might be different (previous close vs current close)
- EA's filters use different data than backtest

This explains why results differ - they're not using the same data!
""")

print("\n" + "="*80)
print("SOLUTION: Make Backtest Match EA's Data Collection")
print("="*80)

print("""
To fix this, backtest should:

1. Use shift(1) for all price data:
   - close_price = row['Close'].shift(1)  ← Previous bar
   - high_price = row['High'].shift(1)
   - low_price = row['Low'].shift(1)

2. Use shift(1) for all indicators:
   - atr = row['ATR'].shift(1)
   - ema9 = row['EMA_9_5M'].shift(1)
   - etc.

3. Check breakout on previous bar:
   - if previous_close > range_high: enter on current bar

4. Match EA's timing:
   - Process bars as if OnTick() fires on new bar
   - Use data from bar[1] for decisions
   - Enter on bar[0] (current bar)

This will make backtest match what EA actually sees!
""")

