import pandas as pd
import numpy as np
from strategy_backtest import TradingStrategy, StrategyConfig
import glob

print("="*80)
print("FULL CONFIRMATION DELAY TEST ON ALL AVAILABLE DATA")
print("="*80)

# Find all CSV files
csv_files = glob.glob("*USD*.csv") + glob.glob("*XAU*.csv") + glob.glob("*GOLD*.csv")
csv_files = [f for f in csv_files if 'new data' in f.lower() or 'xauusd' in f.lower()]

if not csv_files:
    csv_files = ['XAUUSD5 new data.csv']  # Default

print(f"\nFound {len(csv_files)} data file(s) to test")
for f in csv_files:
    print(f"  - {f}")

# Base optimized config
base_config = StrategyConfig(
    reward_to_risk=2.0,
    pullback_timeout=12,
    use_ema_filter=False,
    allow_breakout=True,
    allow_pullback=False,
    allow_reversal=False,
    max_trades_per_window=1,
    use_breakout_controls=True,
    breakout_initial_stop_ratio=0.6,
    breakout_max_mae_ratio=1.0,
    breakout_momentum_bar=5,
    breakout_momentum_min_gain=0.2,
    max_breakout_atr_multiple=1.8,
    max_atr_ratio=1.3,
    min_trend_score=0.66,
    max_consolidation_score=0.10,
    min_entry_offset_ratio=-0.25,
    max_entry_offset_ratio=1.00,
    first_bar_min_gain=-0.30,
    max_retest_depth_r=3.00,
    max_retest_bars=20,
)

all_results = []

for csv_file in csv_files:
    print(f"\n{'='*80}")
    print(f"Testing: {csv_file}")
    print(f"{'='*80}")
    
    # Test 1: Immediate Entry
    print("\n1. IMMEDIATE ENTRY (Old Behavior)")
    config_immediate = StrategyConfig(**base_config.__dict__)
    config_immediate.wait_for_confirmation = False
    
    strategy1 = TradingStrategy(csv_file, config_immediate)
    strategy1.load_data()
    strategy1.add_indicators(ema_periods_5m=[9,21,50], ema_200_1h=True, atr_period=14)
    strategy1.identify_key_times()
    trades1 = strategy1.backtest_strategy()
    trades_df1 = pd.DataFrame(trades1)
    
    tp_hits1 = trades_df1[trades_df1['Status'] == 'TP_HIT']
    sl_hits1 = trades_df1[trades_df1['Status'] == 'SL_HIT']
    filter_exits1 = trades_df1[trades_df1['Status'] == 'FILTER_EXIT']
    all_closed1 = trades_df1[trades_df1['Status'].isin(['TP_HIT', 'SL_HIT', 'FILTER_EXIT'])]
    
    total1 = len(all_closed1)
    winners1 = len(tp_hits1)
    losers1 = len(sl_hits1) + len(filter_exits1)
    win_rate1 = winners1 / total1 * 100 if total1 > 0 else 0
    total_pnl1 = all_closed1['P&L'].sum()
    avg_pnl1 = all_closed1['P&L'].mean() if len(all_closed1) > 0 else 0
    
    print(f"   Total Trades: {total1}")
    print(f"   Winners: {winners1}")
    print(f"   Losers: {losers1}")
    print(f"   Win Rate: {win_rate1:.1f}%")
    print(f"   Total P&L: ${total_pnl1:.2f}")
    print(f"   Filter Exits: {len(filter_exits1)}")
    
    # Test 2: Confirmation Delay
    print("\n2. CONFIRMATION DELAY (New Behavior)")
    config_confirmed = StrategyConfig(**base_config.__dict__)
    config_confirmed.wait_for_confirmation = True
    config_confirmed.confirmation_timeout_bars = 5
    
    strategy2 = TradingStrategy(csv_file, config_confirmed)
    strategy2.load_data()
    strategy2.add_indicators(ema_periods_5m=[9,21,50], ema_200_1h=True, atr_period=14)
    strategy2.identify_key_times()
    trades2 = strategy2.backtest_strategy()
    trades_df2 = pd.DataFrame(trades2)
    
    tp_hits2 = trades_df2[trades_df2['Status'] == 'TP_HIT']
    sl_hits2 = trades_df2[trades_df2['Status'] == 'SL_HIT']
    filter_exits2 = trades_df2[trades_df2['Status'] == 'FILTER_EXIT']
    all_closed2 = trades_df2[trades_df2['Status'].isin(['TP_HIT', 'SL_HIT', 'FILTER_EXIT'])]
    
    total2 = len(all_closed2)
    winners2 = len(tp_hits2)
    losers2 = len(sl_hits2) + len(filter_exits2)
    win_rate2 = winners2 / total2 * 100 if total2 > 0 else 0
    total_pnl2 = all_closed2['P&L'].sum()
    avg_pnl2 = all_closed2['P&L'].mean() if len(all_closed2) > 0 else 0
    
    print(f"   Total Trades: {total2}")
    print(f"   Winners: {winners2}")
    print(f"   Losers: {losers2}")
    print(f"   Win Rate: {win_rate2:.1f}%")
    print(f"   Total P&L: ${total_pnl2:.2f}")
    print(f"   Filter Exits: {len(filter_exits2)}")
    
    # Comparison
    print(f"\n3. COMPARISON")
    print(f"   Trade Reduction: {total1 - total2} ({((total1 - total2) / total1 * 100) if total1 > 0 else 0:.1f}%)")
    print(f"   Win Rate Change: {win_rate2 - win_rate1:+.1f}%")
    print(f"   P&L Change: ${total_pnl2 - total_pnl1:+.2f}")
    print(f"   Filter Exit Reduction: {len(filter_exits1) - len(filter_exits2)}")
    
    all_results.append({
        'File': csv_file,
        'Method': 'Immediate',
        'Trades': total1,
        'Winners': winners1,
        'Losers': losers1,
        'Win_Rate': win_rate1,
        'Total_PnL': total_pnl1,
        'Avg_PnL': avg_pnl1,
        'Filter_Exits': len(filter_exits1),
    })
    
    all_results.append({
        'File': csv_file,
        'Method': 'Confirmation',
        'Trades': total2,
        'Winners': winners2,
        'Losers': losers2,
        'Win_Rate': win_rate2,
        'Total_PnL': total_pnl2,
        'Avg_PnL': avg_pnl2,
        'Filter_Exits': len(filter_exits2),
    })

# Summary
print(f"\n{'='*80}")
print("OVERALL SUMMARY")
print(f"{'='*80}")

results_df = pd.DataFrame(all_results)
summary = results_df.groupby('Method').agg({
    'Trades': 'sum',
    'Winners': 'sum',
    'Losers': 'sum',
    'Total_PnL': 'sum',
    'Filter_Exits': 'sum',
}).reset_index()

summary['Win_Rate'] = summary['Winners'] / summary['Trades'] * 100
summary['Avg_PnL'] = summary['Total_PnL'] / summary['Trades']

print("\n" + summary.to_string(index=False))

print(f"\n{'='*80}")
print("KEY FINDINGS")
print(f"{'='*80}")

immediate = summary[summary['Method'] == 'Immediate'].iloc[0]
confirmed = summary[summary['Method'] == 'Confirmation'].iloc[0]

print(f"""
Immediate Entry:
- Total Trades: {immediate['Trades']:.0f}
- Win Rate: {immediate['Win_Rate']:.1f}%
- Total P&L: ${immediate['Total_PnL']:.2f}
- Filter Exits: {immediate['Filter_Exits']:.0f}

Confirmation Delay:
- Total Trades: {confirmed['Trades']:.0f}
- Win Rate: {confirmed['Win_Rate']:.1f}%
- Total P&L: ${confirmed['Total_PnL']:.2f}
- Filter Exits: {confirmed['Filter_Exits']:.0f}

Improvement:
- Trade Reduction: {immediate['Trades'] - confirmed['Trades']:.0f} trades
- Win Rate Change: {confirmed['Win_Rate'] - immediate['Win_Rate']:+.1f}%
- P&L Change: ${confirmed['Total_PnL'] - immediate['Total_PnL']:+.2f}
- Filter Exit Reduction: {immediate['Filter_Exits'] - confirmed['Filter_Exits']:.0f}
""")

# Save results
results_df.to_csv('analysis/confirmation_delay_full_test.csv', index=False)
summary.to_csv('analysis/confirmation_delay_summary.csv', index=False)
print(f"\n✓ Results saved to analysis/ folder")

