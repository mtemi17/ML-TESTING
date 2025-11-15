"""
Test to verify EA matches backtest results
This script helps identify what to expect from EA testing
"""
import pandas as pd
from strategy_backtest import TradingStrategy, StrategyConfig

print("="*80)
print("EA vs BACKTEST VERIFICATION TEST")
print("="*80)
print("\nThis test will show you what to expect when testing EAAI_Simple.mq5")
print("="*80)

# Use optimized config (StrategyMode=1)
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

# Test on October 2025
print("\n1. Running backtest on October 2025 Gold data...")
strategy = TradingStrategy("XAUUSD5 new data.csv", config)
strategy.load_data()

# Filter to October
october_mask = strategy.df.index.month == 10
strategy.df = strategy.df[october_mask].copy()

strategy.add_indicators(ema_periods_5m=[9,21,50], ema_200_1h=True, atr_period=14)
strategy.identify_key_times()
trades = strategy.backtest_strategy()
trades_df = pd.DataFrame(trades)

completed = trades_df[trades_df['Status'].isin(['TP_HIT','SL_HIT'])]
filtered = trades_df[trades_df['Status']=='FILTER_EXIT']

print(f"\n   Backtest Results:")
print(f"   - Total trades attempted: {len(trades_df)}")
print(f"   - Completed trades: {len(completed)}")
print(f"   - Filter exits: {len(filtered)}")
print(f"   - Wins: {(completed['Status']=='TP_HIT').sum()}")
print(f"   - Losses: {(completed['Status']=='SL_HIT').sum()}")

if len(completed) > 0:
    win_rate = (completed['Status']=='TP_HIT').sum() / len(completed) * 100
    total_pnl = completed['P&L'].sum()
    print(f"   - Win Rate: {win_rate:.1f}%")
    print(f"   - Total P&L: ${total_pnl:.2f}")

# Show detailed trade list
print(f"\n2. Expected EA Trades (October 2025):")
print("="*80)
if len(completed) > 0:
    print(f"\n{'#':<4} {'Entry Time':<20} {'Type':<6} {'Entry':<10} {'SL':<10} {'TP':<10} {'Status':<10} {'P&L':<10}")
    print("-"*80)
    for idx, trade in completed.iterrows():
        print(f"{idx+1:<4} {str(trade['EntryTime']):<20} {trade['Type']:<6} "
              f"{trade['EntryPrice']:<10.2f} {trade['SL']:<10.2f} {trade['TP']:<10.2f} "
              f"{trade['Status']:<10} ${trade['P&L']:<9.2f}")
else:
    print("   No completed trades found")

# Show filter exits
if len(filtered) > 0:
    print(f"\n3. Filtered Trades (Would be rejected by EA filters):")
    print(f"   Total filtered: {len(filtered)}")
    print(f"   Reasons:")
    # Count by filter type (we'd need to track this, but for now just show count)
    print(f"   - Pre-entry filters: {len(filtered)} trades")

print(f"\n4. EA Configuration to Test:")
print("="*80)
print("""
In MT5 Strategy Tester, set EAAI_Simple.mq5 with:

Strategy Mode: 1 (Optimized)
Risk Percent: 1.0
Reward to Risk: 2.0
Max Trades Per Window: 1

Winner Profile Controls:
- Breakout Initial Stop Ratio: 0.6
- Breakout Max MAE Ratio: 0.6
- Breakout Momentum Bar: 3
- Breakout Momentum Min Gain: 0.3

Optimized Filters:
- Max Breakout ATR Multiple: 1.8
- Max ATR Ratio: 1.3
- Min Trend Score: 0.66
- Max Consolidation Score: 0.10
- Min Entry Offset Ratio: -0.25
- Max Entry Offset Ratio: 1.00
- First Bar Min Gain: -0.20
- Max Retest Depth R: 1.80
- Max Retest Bars: 12

Symbol: XAUUSD
Timeframe: M5
Date Range: 2025-10-01 to 2025-10-31
Model: Every tick (for accuracy)
""")

print(f"\n5. Expected Results from EA:")
print("="*80)
print(f"""
You should see approximately:
- {len(completed)} completed trades
- {len(filtered)} trades filtered out
- Win Rate: ~{win_rate:.1f}% (if matches backtest)
- Total P&L: ~${total_pnl:.2f} (if matches backtest)

If results differ significantly, check:
1. Data quality in MT5 (same as CSV?)
2. Indicator calculations match?
3. Window detection working correctly?
4. Filter logic executing properly?
""")

# Save expected trades for comparison
if len(completed) > 0:
    completed.to_csv('analysis/expected_ea_trades_october.csv', index=False)
    print(f"\n✓ Expected trades saved to: analysis/expected_ea_trades_october.csv")
    print("  Use this to compare with EA's actual results in MT5 Strategy Tester")

print("\n" + "="*80)
print("READY FOR EA TESTING")
print("="*80)
print("\nNext Steps:")
print("1. Load EAAI_Simple.mq5 in MT5 Strategy Tester")
print("2. Set configuration as shown above")
print("3. Run backtest on October 2025 data")
print("4. Compare results with expected trades CSV")
print("5. If they match → EA is working correctly!")
print("="*80)

