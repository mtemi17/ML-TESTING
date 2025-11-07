import pandas as pd
import numpy as np

# Load the backtest results
df = pd.read_csv('backtest_results.csv')

print("="*70)
print("FILTERED STRATEGY ANALYSIS")
print("="*70)
print("\nFilters Applied:")
print("  1. Market is NOT consolidating (Is_Consolidating == 0)")
print("  2. BUY trades: Price > EMA_200_1H")
print("  3. SELL trades: Price < EMA_200_1H")
print("="*70)

# Filter for completed trades only
completed = df[df['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

print(f"\nTotal completed trades: {len(completed)}")

# Apply filters
# 1. Not consolidating
non_consolidating = completed[completed['Is_Consolidating'] == 0].copy()
print(f"Trades during non-consolidation: {len(non_consolidating)}")

# 2. Filter by EMA_200_1H alignment
# For BUY: Price should be above EMA_200_1H
# For SELL: Price should be below EMA_200_1H

buy_trades = non_consolidating[non_consolidating['Type'] == 'BUY'].copy()
sell_trades = non_consolidating[non_consolidating['Type'] == 'SELL'].copy()

# Filter BUY trades: EntryPrice > EMA_200_1H
filtered_buy = buy_trades[buy_trades['EntryPrice'] > buy_trades['EMA_200_1H']].copy()

# Filter SELL trades: EntryPrice < EMA_200_1H
filtered_sell = sell_trades[sell_trades['EntryPrice'] < sell_trades['EMA_200_1H']].copy()

# Combine filtered trades
filtered_trades = pd.concat([filtered_buy, filtered_sell], ignore_index=True)

print(f"\nFiltered BUY trades (Price > EMA_200_1H): {len(filtered_buy)}")
print(f"Filtered SELL trades (Price < EMA_200_1H): {len(filtered_sell)}")
print(f"Total filtered trades: {len(filtered_trades)}")

if len(filtered_trades) > 0:
    print("\n" + "="*70)
    print("FILTERED RESULTS SUMMARY")
    print("="*70)
    
    # Overall statistics
    total_trades = len(filtered_trades)
    winning = filtered_trades[filtered_trades['P&L'] > 0]
    losing = filtered_trades[filtered_trades['P&L'] < 0]
    win_rate = (len(winning) / total_trades) * 100
    
    total_pnl = filtered_trades['P&L'].sum()
    avg_win = winning['P&L'].mean() if len(winning) > 0 else 0
    avg_loss = losing['P&L'].mean() if len(losing) > 0 else 0
    avg_r_multiple = filtered_trades['R_Multiple'].mean()
    
    print(f"\nTotal Trades: {total_trades}")
    print(f"Winning Trades: {len(winning)} ({win_rate:.2f}%)")
    print(f"Losing Trades: {len(losing)}")
    print(f"\nTotal P&L: ${total_pnl:.2f}")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Average R-Multiple: {avg_r_multiple:.2f}")
    
    if avg_loss != 0 and len(losing) > 0:
        profit_factor = abs(avg_win * len(winning) / (avg_loss * len(losing)))
        print(f"Profit Factor: {profit_factor:.2f}")
    
    # Compare with unfiltered results
    print("\n" + "-"*70)
    print("COMPARISON: Filtered vs All Non-Consolidating Trades")
    print("-"*70)
    
    all_non_consolidating_pnl = non_consolidating['P&L'].sum()
    all_non_consolidating_win_rate = (non_consolidating['P&L'] > 0).sum() / len(non_consolidating) * 100
    
    print(f"\nAll Non-Consolidating Trades:")
    print(f"  Total: {len(non_consolidating)}")
    print(f"  Win Rate: {all_non_consolidating_win_rate:.2f}%")
    print(f"  Total P&L: ${all_non_consolidating_pnl:.2f}")
    
    print(f"\nFiltered Trades (with EMA alignment):")
    print(f"  Total: {total_trades}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Total P&L: ${total_pnl:.2f}")
    
    improvement = win_rate - all_non_consolidating_win_rate
    pnl_improvement = total_pnl - (all_non_consolidating_pnl * (total_trades / len(non_consolidating)))
    
    print(f"\nImprovement:")
    print(f"  Win Rate: {improvement:+.2f}%")
    print(f"  P&L per trade: ${(total_pnl/total_trades) - (all_non_consolidating_pnl/len(non_consolidating)):+.2f}")
    
    # Breakdown by trade type
    print("\n" + "-"*70)
    print("BREAKDOWN BY TRADE TYPE")
    print("-"*70)
    
    if len(filtered_buy) > 0:
        buy_winning = filtered_buy[filtered_buy['P&L'] > 0]
        buy_win_rate = (len(buy_winning) / len(filtered_buy)) * 100
        print(f"\nBUY Trades (Price > EMA_200_1H):")
        print(f"  Total: {len(filtered_buy)}")
        print(f"  Win Rate: {buy_win_rate:.2f}%")
        print(f"  Total P&L: ${filtered_buy['P&L'].sum():.2f}")
        print(f"  Avg P&L: ${filtered_buy['P&L'].mean():.2f}")
        print(f"  Avg R-Multiple: {filtered_buy['R_Multiple'].mean():.2f}")
    
    if len(filtered_sell) > 0:
        sell_winning = filtered_sell[filtered_sell['P&L'] > 0]
        sell_win_rate = (len(sell_winning) / len(filtered_sell)) * 100
        print(f"\nSELL Trades (Price < EMA_200_1H):")
        print(f"  Total: {len(filtered_sell)}")
        print(f"  Win Rate: {sell_win_rate:.2f}%")
        print(f"  Total P&L: ${filtered_sell['P&L'].sum():.2f}")
        print(f"  Avg P&L: ${filtered_sell['P&L'].mean():.2f}")
        print(f"  Avg R-Multiple: {filtered_sell['R_Multiple'].mean():.2f}")
    
    # Breakdown by window type
    print("\n" + "-"*70)
    print("BREAKDOWN BY WINDOW TYPE")
    print("-"*70)
    window_stats = filtered_trades.groupby('WindowType').agg({
        'P&L': ['count', 'sum', 'mean'],
        'R_Multiple': 'mean'
    }).round(2)
    print(window_stats)
    
    # Save filtered results
    filtered_trades.to_csv('filtered_backtest_results.csv', index=False)
    print(f"\n" + "="*70)
    print(f"Filtered results saved to: filtered_backtest_results.csv")
    print(f"Total filtered trades: {len(filtered_trades)}")
    
else:
    print("\nNo trades match the filter criteria!")

