#!/usr/bin/env python3
"""
Test EAAI_Simple_Model EA logic on Gold data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING EAAI_SIMPLE_MODEL EA ON GOLD DATA")
print("="*80)

# ============================================================================
# 1. LOAD GOLD DATA
# ============================================================================
print("\n1. Loading Gold data...")
df = pd.read_csv(
    'XAUUSD5.csv',
    header=None,
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
)

# Combine Date and Time
df['DateTime'] = pd.to_datetime(
    df['Date'].astype(str) + ' ' + df['Time'].astype(str),
    format='%Y.%m.%d %H:%M',
    errors='coerce'
)
df = df.dropna(subset=['DateTime'])
df.set_index('DateTime', inplace=True)

print(f"   Loaded {len(df)} rows")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# ============================================================================
# 2. ADD INDICATORS (Same as EA)
# ============================================================================
print("\n2. Adding indicators...")

# ATR (14 period)
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift())
low_close = np.abs(df['Low'] - df['Close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.ewm(span=14, adjust=False).mean()

# ATR Ratio
df['ATR_MA_20'] = df['ATR'].rolling(20).mean()
df['ATR_Ratio'] = df['ATR'] / df['ATR_MA_20'].replace(0, np.nan)

# EMA 200 (1H)
df_1h = df.resample('1H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})
df_1h['EMA_200_1H'] = df_1h['Close'].ewm(span=200, adjust=False).mean()
df['EMA_200_1H'] = df_1h['EMA_200_1H'].reindex(df.index, method='ffill')

# Consolidation Score
df['Is_Consolidating'] = (df['ATR'] < df['ATR_MA_20'] * 0.7).astype(int)
df['Is_Tight_Range'] = ((df['High'] - df['Low']) < (df['High'] - df['Low']).rolling(20).mean() * 0.8).astype(int)
df['Consolidation_Score'] = (df['Is_Consolidating'] + df['Is_Tight_Range']) / 2.0

print("   ✅ Indicators added")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def PassesModelFilters(df, idx, is_long, max_consolidation, use_ema200, min_atr_val, max_atr_val):
    """Check if trade passes model-based filters"""
    if not use_model_filters:
        return True
    
    # Get current values
    atr = df.loc[idx, 'ATR']
    consolidation_score = df.loc[idx, 'Consolidation_Score']
    ema200_1h = df.loc[idx, 'EMA_200_1H']
    price = df.loc[idx, 'Close']
    
    # ATR filter
    if min_atr_val > 0 and atr < min_atr_val:
        return False
    if max_atr_val > 0 and atr > max_atr_val:
        return False
    
    # Consolidation Score filter
    if consolidation_score > max_consolidation:
        return False
    
    # EMA 200 (1H) filter
    if use_ema200:
        if is_long and price < ema200_1h:
            return False
        if not is_long and price > ema200_1h:
            return False
    
    return True

def SimulateTrade(df, entry_idx, direction, entry_price, sl_price, tp_price):
    """Simulate trade outcome"""
    future_data = df.loc[entry_idx:]
    
    for idx, row in future_data.iterrows():
        if direction == 'LONG':
            if row['High'] >= tp_price:
                return {
                    'EntryTime': entry_idx,
                    'ExitTime': idx,
                    'EntryPrice': entry_price,
                    'ExitPrice': tp_price,
                    'SL': sl_price,
                    'TP': tp_price,
                    'Direction': direction,
                    'Status': 'TP_HIT',
                    'P&L': tp_price - entry_price,
                    'R_Multiple': (tp_price - entry_price) / (entry_price - sl_price) if (entry_price - sl_price) > 0 else 0
                }
            if row['Low'] <= sl_price:
                return {
                    'EntryTime': entry_idx,
                    'ExitTime': idx,
                    'EntryPrice': entry_price,
                    'ExitPrice': sl_price,
                    'SL': sl_price,
                    'TP': tp_price,
                    'Direction': direction,
                    'Status': 'SL_HIT',
                    'P&L': sl_price - entry_price,
                    'R_Multiple': (sl_price - entry_price) / (entry_price - sl_price) if (entry_price - sl_price) > 0 else 0
                }
        else:  # SHORT
            if row['Low'] <= tp_price:
                return {
                    'EntryTime': entry_idx,
                    'ExitTime': idx,
                    'EntryPrice': entry_price,
                    'ExitPrice': tp_price,
                    'SL': sl_price,
                    'TP': tp_price,
                    'Direction': direction,
                    'Status': 'TP_HIT',
                    'P&L': entry_price - tp_price,
                    'R_Multiple': (entry_price - tp_price) / (sl_price - entry_price) if (sl_price - entry_price) > 0 else 0
                }
            if row['High'] >= sl_price:
                return {
                    'EntryTime': entry_idx,
                    'ExitTime': idx,
                    'EntryPrice': entry_price,
                    'ExitPrice': sl_price,
                    'SL': sl_price,
                    'TP': tp_price,
                    'Direction': direction,
                    'Status': 'SL_HIT',
                    'P&L': entry_price - sl_price,
                    'R_Multiple': (entry_price - sl_price) / (sl_price - entry_price) if (sl_price - entry_price) > 0 else 0
                }
    
    # Trade still open at end of data
    return None

# ============================================================================
# 3. SIMULATE EA LOGIC
# ============================================================================
print("\n3. Simulating EA logic...")

# Sessions
sessions = [
    {'name': 'ASIAN', 'hour': 3, 'minute': 0},
    {'name': 'LONDON', 'hour': 10, 'minute': 0},
    {'name': 'NEW_YORK', 'hour': 16, 'minute': 30}
]

trading_window_hours = 3
reward_to_risk = 2.0

# Model filter settings (from EA)
use_model_filters = True
max_consolidation_score = 0.5
use_ema200_filter = False
min_atr = 0.0
max_atr = 0.0

trades = []

# Resample to 15M for range detection
df_15m = df.resample('15T').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Get unique dates
unique_dates = sorted(pd.Series(df.index.date).unique())
print(f"   Processing {len(unique_dates)} days...")

trades_count = 0
for date_idx, date in enumerate(unique_dates):
    if date_idx % 50 == 0:
        print(f"   Processing day {date_idx+1}/{len(unique_dates)}... (found {trades_count} trades so far)")
    
    for session in sessions:
        # Session start time
        session_start = pd.Timestamp.combine(date, datetime.min.time().replace(hour=session['hour'], minute=session['minute']))
        
        # Range: First 15M candle (15 minutes from session start)
        range_start = session_start
        range_end = session_start + pd.Timedelta(minutes=15)
        window_end = session_start + pd.Timedelta(hours=trading_window_hours)
        
        # Skip if outside data range
        if range_end > df.index.max() or window_end < df.index.min():
            continue
        
        # Get 15M range
        range_data_15m = df_15m.loc[range_start:range_end]
        if len(range_data_15m) == 0:
            continue
        
        range_high = range_data_15m['High'].max()
        range_low = range_data_15m['Low'].min()
        
        # Trading window: 3 hours after session start (on 5M timeframe)
        window_data = df.loc[range_end:window_end]
        if len(window_data) == 0:
            continue
        
        # Check for breakouts
        for idx, row in window_data.iterrows():
            close = row['Close']
            open_price = row['Open']
            is_bullish = close > open_price
            is_bearish = close < open_price
            
            # BULLISH BREAKOUT: Close above range high AND bullish candle
            if close > range_high and is_bullish:
                # Check model filters
                if PassesModelFilters(df, idx, True, max_consolidation_score, use_ema200_filter, min_atr, max_atr):
                    entry_price = close
                    sl_price = range_low
                    risk = entry_price - sl_price
                    tp_price = entry_price + (risk * reward_to_risk)
                    
                    # Simulate trade outcome
                    outcome = SimulateTrade(df, idx, 'LONG', entry_price, sl_price, tp_price)
                    if outcome:
                        outcome['Session'] = session['name']
                        outcome['RangeHigh'] = range_high
                        outcome['RangeLow'] = range_low
                        trades.append(outcome)
                        trades_count += 1
                        break  # Only one trade per session
            
            # BEARISH BREAKOUT: Close below range low AND bearish candle
            elif close < range_low and is_bearish:
                # Check model filters
                if PassesModelFilters(df, idx, False, max_consolidation_score, use_ema200_filter, min_atr, max_atr):
                    entry_price = close
                    sl_price = range_high
                    risk = sl_price - entry_price
                    tp_price = entry_price - (risk * reward_to_risk)
                    
                    # Simulate trade outcome
                    outcome = SimulateTrade(df, idx, 'SHORT', entry_price, sl_price, tp_price)
                    if outcome:
                        outcome['Session'] = session['name']
                        outcome['RangeHigh'] = range_high
                        outcome['RangeLow'] = range_low
                        trades.append(outcome)
                        trades_count += 1
                        break  # Only one trade per session

# ============================================================================
# 4. ANALYZE RESULTS
# ============================================================================
print(f"\n4. Results:")
print(f"   Total trades: {len(trades)}")

if len(trades) == 0:
    print("   ❌ NO TRADES GENERATED!")
    exit(1)

df_trades = pd.DataFrame(trades)

wins = df_trades[df_trades['Status'] == 'TP_HIT']
losses = df_trades[df_trades['Status'] == 'SL_HIT']

win_rate = len(wins) / len(df_trades) * 100 if len(df_trades) > 0 else 0
total_pnl = df_trades['P&L'].sum()
avg_pnl = df_trades['P&L'].mean()

print(f"\n   Wins: {len(wins)} ({win_rate:.1f}%)")
print(f"   Losses: {len(losses)} ({len(losses)/len(df_trades)*100:.1f}%)")
print(f"   Total P&L: ${total_pnl:.2f}")
print(f"   Avg P&L: ${avg_pnl:.2f}")

# By session
print(f"\n   By Session:")
for session in ['ASIAN', 'LONDON', 'NEW_YORK']:
    session_trades = df_trades[df_trades['Session'] == session]
    if len(session_trades) > 0:
        session_wins = len(session_trades[session_trades['Status'] == 'TP_HIT'])
        session_win_rate = session_wins / len(session_trades) * 100
        session_pnl = session_trades['P&L'].sum()
        print(f"     {session}: {len(session_trades)} trades, {session_win_rate:.1f}% WR, ${session_pnl:.2f} P&L")

# Save results
output_file = 'ea_simple_model_gold_test_results.csv'
df_trades.to_csv(output_file, index=False)
print(f"\n   ✅ Results saved to: {output_file}")

print("\n" + "="*80)
print("✅ TEST COMPLETE!")
print("="*80)

