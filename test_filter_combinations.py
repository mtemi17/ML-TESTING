import pandas as pd
import numpy as np
from strategy_backtest import TradingStrategy, StrategyConfig

print("="*80)
print("TESTING FILTER COMBINATIONS FOR IMPROVED WIN RATE")
print("="*80)

# Base config
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

# Current config (baseline)
current_config = StrategyConfig(**base_config.__dict__)
current_config.breakout_max_mae_ratio = 0.6

# Option 1: Tighten Pre-Entry
option1_config = StrategyConfig(**base_config.__dict__)
option1_config.breakout_max_mae_ratio = 0.6
option1_config.max_breakout_atr_multiple = 2.0  # Tighter (was 1.8)
option1_config.max_atr_ratio = 1.2  # Tighter (was 1.3)
option1_config.min_trend_score = 0.70  # Tighter (was 0.66)

# Option 2: Relax Post-Entry
option2_config = StrategyConfig(**base_config.__dict__)
option2_config.breakout_max_mae_ratio = 0.8  # Relaxed (was 0.6)
option2_config.max_retest_depth_r = 2.5  # Relaxed (was 1.8)
option2_config.max_retest_bars = 15  # More time (was 12)

# Option 3: Hybrid
option3_config = StrategyConfig(**base_config.__dict__)
option3_config.breakout_max_mae_ratio = 0.7  # Slightly relaxed
option3_config.max_retest_depth_r = 2.2  # Moderately relaxed
option3_config.max_breakout_atr_multiple = 2.0  # Slightly tighter
option3_config.max_atr_ratio = 1.25  # Slightly tighter

configs = {
    'Current (Baseline)': current_config,
    'Option 1: Tighten Pre-Entry': option1_config,
    'Option 2: Relax Post-Entry': option2_config,
    'Option 3: Hybrid (Recommended)': option3_config,
}

results = []

for name, config in configs.items():
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    strategy = TradingStrategy("XAUUSD5 new data.csv", config)
    strategy.load_data()
    
    # Filter to October
    october_mask = strategy.df.index.month == 10
    strategy.df = strategy.df[october_mask].copy()
    
    strategy.add_indicators(ema_periods_5m=[9,21,50], ema_200_1h=True, atr_period=14)
    strategy.identify_key_times()
    trades = strategy.backtest_strategy()
    trades_df = pd.DataFrame(trades)
    
    # Analyze results
    tp_hits = trades_df[trades_df['Status'] == 'TP_HIT']
    sl_hits = trades_df[trades_df['Status'] == 'SL_HIT']
    filter_exits = trades_df[trades_df['Status'] == 'FILTER_EXIT']
    all_closed = trades_df[trades_df['Status'].isin(['TP_HIT', 'SL_HIT', 'FILTER_EXIT'])]
    
    total_trades = len(all_closed)
    winners = len(tp_hits)
    losers = len(sl_hits) + len(filter_exits)
    win_rate = winners / total_trades * 100 if total_trades > 0 else 0
    total_pnl = all_closed['P&L'].sum()
    avg_pnl = all_closed['P&L'].mean() if len(all_closed) > 0 else 0
    
    if len(tp_hits) > 0 and len(filter_exits) > 0:
        avg_win = tp_hits['P&L'].mean()
        avg_loss = abs(filter_exits[filter_exits['P&L'] <= 0]['P&L'].mean()) if len(filter_exits[filter_exits['P&L'] <= 0]) > 0 else 0
        profit_factor = (avg_win * winners) / (avg_loss * losers) if avg_loss > 0 and losers > 0 else float('inf')
    else:
        profit_factor = 0
    
    results.append({
        'Config': name,
        'Total_Trades': total_trades,
        'Winners': winners,
        'Losers': losers,
        'Win_Rate': win_rate,
        'Total_PnL': total_pnl,
        'Avg_PnL': avg_pnl,
        'TP_Hits': len(tp_hits),
        'SL_Hits': len(sl_hits),
        'Filter_Exits': len(filter_exits),
        'Profit_Factor': profit_factor,
    })
    
    print(f"\n   Results:")
    print(f"   - Total Trades: {total_trades}")
    print(f"   - Winners (TP): {winners}")
    print(f"   - Losers (SL + Filters): {losers}")
    print(f"   - Win Rate: {win_rate:.1f}%")
    print(f"   - Total P&L: ${total_pnl:.2f}")
    print(f"   - Avg P&L: ${avg_pnl:.2f}")
    print(f"   - Profit Factor: {profit_factor:.2f}")
    print(f"   - Filter Exits: {len(filter_exits)}")

# Summary comparison
print(f"\n{'='*80}")
print("SUMMARY COMPARISON")
print(f"{'='*80}")

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Find best config
best_win_rate = results_df.loc[results_df['Win_Rate'].idxmax()]
best_pnl = results_df.loc[results_df['Total_PnL'].idxmax()]
best_pf = results_df.loc[results_df['Profit_Factor'].idxmax()]

print(f"\n{'='*80}")
print("BEST CONFIGURATIONS")
print(f"{'='*80}")
print(f"\nHighest Win Rate:")
print(f"  {best_win_rate['Config']}: {best_win_rate['Win_Rate']:.1f}% ({best_win_rate['Winners']}/{best_win_rate['Total_Trades']})")
print(f"  Total P&L: ${best_win_rate['Total_PnL']:.2f}")

print(f"\nHighest Total P&L:")
print(f"  {best_pnl['Config']}: ${best_pnl['Total_PnL']:.2f}")
print(f"  Win Rate: {best_pnl['Win_Rate']:.1f}% ({best_pnl['Winners']}/{best_pnl['Total_Trades']})")

print(f"\nBest Profit Factor:")
print(f"  {best_pf['Config']}: {best_pf['Profit_Factor']:.2f}")
print(f"  Win Rate: {best_pf['Win_Rate']:.1f}%")

# Save results
results_df.to_csv('analysis/filter_combination_test_results.csv', index=False)
print(f"\n✓ Results saved to: analysis/filter_combination_test_results.csv")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")
print(f"""
Based on the tests:

1. Current Baseline: {current_config.breakout_max_mae_ratio} MAE, {current_config.max_retest_depth_r}R retest
   → Win Rate: {results_df[results_df['Config']=='Current (Baseline)']['Win_Rate'].values[0]:.1f}%

2. Option 1 (Tighten Pre-Entry): Stricter entry filters
   → Win Rate: {results_df[results_df['Config']=='Option 1: Tighten Pre-Entry']['Win_Rate'].values[0]:.1f}%

3. Option 2 (Relax Post-Entry): More room for trades
   → Win Rate: {results_df[results_df['Config']=='Option 2: Relax Post-Entry']['Win_Rate'].values[0]:.1f}%

4. Option 3 (Hybrid): Balanced approach
   → Win Rate: {results_df[results_df['Config']=='Option 3: Hybrid (Recommended)']['Win_Rate'].values[0]:.1f}%

The best configuration appears to be the one with highest win rate
while maintaining good total P&L and profit factor.
""")

