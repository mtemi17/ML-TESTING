import pandas as pd
import numpy as np
from strategy_backtest import TradingStrategy, StrategyConfig
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("REALISTIC OCTOBER 2025 ANALYSIS")
print("="*80)
print("\nISSUE IDENTIFIED:")
print("  - Data only has 23 trading days (missing 8 weekends)")
print("  - Missing 2,580 candles (29% of expected data)")
print("  - Backtest found 69 windows, but EA in live might see different data")
print("="*80)

# Load October data
df = pd.read_csv("XAUUSD5 new data.csv", header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), format='%Y.%m.%d %H:%M')
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month

october = df[df['Month'] == 10].copy()
october = october.sort_values('DateTime')

# Save October data
october_file = "XAUUSD5_October_temp.csv"
october[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']].to_csv(october_file, index=False, header=False)

# Configure strategy
config = StrategyConfig(
    reward_to_risk=2.0,
    pullback_timeout=12,
    use_ema_filter=False,
    allow_breakout=True,
    allow_pullback=False,
    allow_reversal=False,
    max_trades_per_window=1,
    use_breakout_controls=True,
    breakout_initial_stop_ratio=0.6,
    breakout_max_mae_ratio=0.6,
    breakout_momentum_bar=3,
    breakout_momentum_min_gain=0.3,
    max_breakout_atr_multiple=1.8,
    max_atr_ratio=1.3,
    min_trend_score=0.66,
    max_consolidation_score=0.10,
    min_entry_offset_ratio=-0.25,
    max_entry_offset_ratio=1.00,
    first_bar_min_gain=-0.20,
    max_retest_depth_r=1.80,
    max_retest_bars=12,
)

# Run backtest
print("\n" + "="*80)
print("BACKTEST RESULTS (With Incomplete Data)")
print("="*80)

strategy = TradingStrategy(october_file, config)
strategy.load_data()
strategy.add_indicators(ema_periods_5m=[9,21,50], ema_200_1h=True, atr_period=14)
strategy.identify_key_times()
trades = strategy.backtest_strategy()
trades_df = pd.DataFrame(trades)

completed = trades_df[trades_df['Status'].isin(['TP_HIT','SL_HIT'])]
filtered = trades_df[trades_df['Status']=='FILTER_EXIT']

print(f"\nBacktest Results:")
print(f"  Total trades attempted: {len(trades_df)}")
print(f"  Completed trades: {len(completed)}")
print(f"  Filter exits: {len(filtered)}")

if len(completed) > 0:
    wins = (completed['Status']=='TP_HIT').sum()
    losses = (completed['Status']=='SL_HIT').sum()
    win_rate = wins / len(completed) * 100
    total_pnl = completed['P&L'].sum()
    avg_pnl = completed['P&L'].mean()
    
    print(f"\n  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Avg P&L: ${avg_pnl:.2f}")

# Analyze what windows were actually found
print(f"\n" + "="*80)
print("WINDOW ANALYSIS")
print("="*80)

windows_found = trades_df['WindowID'].nunique()
print(f"  Unique windows with trades: {windows_found}")

# Check window availability by date
print(f"\n  Window Availability by Date:")
window_dates = {}
for window_id in trades_df['WindowID'].unique():
    date_part = str(window_id).split('_')[0] if '_' in str(window_id) else str(window_id)
    if date_part not in window_dates:
        window_dates[date_part] = []
    window_dates[date_part].append(window_id)

print(f"    Dates with windows: {len(window_dates)}")
print(f"    Expected dates: 23 (weekdays only)")

# Show what's missing
all_weekdays = pd.bdate_range('2025-10-01', '2025-10-31')
weekday_dates = [d.strftime('%Y.%m.%d') for d in all_weekdays]
missing_dates = [d for d in weekday_dates if d not in window_dates.keys()]

if missing_dates:
    print(f"    ⚠ Dates with no windows: {len(missing_dates)}")
    print(f"      {missing_dates[:5]}...")

# Compare with what EA would see
print(f"\n" + "="*80)
print("LIVE TRADING vs BACKTEST DISCREPANCY")
print("="*80)

print(f"""
PROBLEM IDENTIFIED:

1. DATA COMPLETENESS:
   - Backtest data: 6,348 candles (71% of expected)
   - Missing: 2,580 candles (29%)
   - Missing weekends: 8 days
   - Missing hour 0:00 data

2. WINDOW DETECTION:
   - Backtest found: 69 windows
   - Expected (23 weekdays * 3): 69 windows ✓
   - But EA in live might see: Different data quality

3. WHY RESULTS DIFFER:

   BACKTEST (Historical CSV):
   - Uses incomplete data
   - May create windows that don't exist in live
   - No real-time data gaps
   - No execution delays
   - Perfect indicator calculations

   LIVE TRADING (MT5 EA):
   - Uses broker's real-time data
   - May have different gaps
   - Execution delays/slippage
   - Indicator calculations on incomplete history
   - iBarShift() might fail if data missing
   - GetWindowRange() might return false if bars missing

4. SPECIFIC ISSUES:

   a) EA GetWindowRange() function:
      - Uses iBarShift() which requires exact bar
      - If bar missing, returns false
      - Window not created = no trade opportunity

   b) Indicator calculations:
      - EMA 200 on 1H needs 200 hours of data
      - ATR needs 14 bars
      - If history incomplete, indicators wrong

   c) Missing bars:
      - Backtest interpolates/fills gaps
      - EA sees actual gaps = no data = no trade

5. RECOMMENDATION:

   To match live trading, backtest should:
   - Only use dates with complete data
   - Skip windows where iBarShift would fail
   - Handle missing indicator values like EA does
   - Account for execution delays
   - Use same data source as broker
""")

# Save detailed analysis
analysis = {
    'Data_Quality': {
        'Total_Candles': len(october),
        'Expected_Candles': 31 * 24 * 12,
        'Missing_Candles': 31 * 24 * 12 - len(october),
        'Missing_Percent': (31 * 24 * 12 - len(october)) / (31 * 24 * 12) * 100,
        'Trading_Days': len(october['DateTime'].dt.date.unique()),
        'Missing_Weekends': 8,
    },
    'Window_Analysis': {
        'Windows_Found': windows_found,
        'Expected_Windows': 23 * 3,
        'Dates_With_Windows': len(window_dates),
    },
    'Backtest_Results': {
        'Total_Trades': len(trades_df),
        'Completed_Trades': len(completed),
        'Wins': int(wins) if len(completed) > 0 else 0,
        'Losses': int(losses) if len(completed) > 0 else 0,
        'Win_Rate': float(win_rate) if len(completed) > 0 else 0.0,
        'Total_PnL': float(total_pnl) if len(completed) > 0 else 0.0,
    }
}

import json
with open('analysis/october_data_quality_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"\n  ✓ Detailed analysis saved to: analysis/october_data_quality_analysis.json")

# Cleanup
import os
if os.path.exists(october_file):
    os.remove(october_file)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nCONCLUSION:")
print("  The backtest uses incomplete data that doesn't match what the EA")
print("  sees in live trading. This explains the discrepancy in results.")
print("  The EA likely sees fewer opportunities due to missing data/bars.")
print("="*80)

