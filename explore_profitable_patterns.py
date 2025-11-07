import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE PATTERN EXPLORATION FOR PROFITABILITY")
print("="*80)

# Load data
df_trades = pd.read_csv('backtest_results.csv')
df_data = pd.read_csv('XAUUSD5_with_indicators.csv', index_col=0, parse_dates=True)

completed_trades = df_trades[df_trades['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
completed_trades['EntryTime'] = pd.to_datetime(completed_trades['EntryTime'])
completed_trades['ExitTime'] = pd.to_datetime(completed_trades['ExitTime'])

print(f"\nAnalyzing {len(completed_trades)} completed trades...")

# ============================================================================
# 1. TRADE DURATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. TRADE DURATION ANALYSIS")
print("="*80)

completed_trades['Duration_Minutes'] = (completed_trades['ExitTime'] - completed_trades['EntryTime']).dt.total_seconds() / 60
completed_trades['Duration_Candles'] = completed_trades['Duration_Minutes'] / 5

winning = completed_trades[completed_trades['P&L'] > 0]
losing = completed_trades[completed_trades['P&L'] < 0]

print(f"\nWinning Trades:")
print(f"  Avg Duration: {winning['Duration_Minutes'].mean():.1f} minutes ({winning['Duration_Candles'].mean():.1f} candles)")
print(f"  Median Duration: {winning['Duration_Minutes'].median():.1f} minutes")
print(f"  Min: {winning['Duration_Minutes'].min():.1f} min, Max: {winning['Duration_Minutes'].max():.1f} min")

print(f"\nLosing Trades:")
print(f"  Avg Duration: {losing['Duration_Minutes'].mean():.1f} minutes ({losing['Duration_Candles'].mean():.1f} candles)")
print(f"  Median Duration: {losing['Duration_Minutes'].median():.1f} minutes")
print(f"  Min: {losing['Duration_Minutes'].min():.1f} min, Max: {losing['Duration_Minutes'].max():.1f} min")

# Find optimal duration thresholds
print(f"\nDuration Thresholds for Profitability:")
for threshold in [15, 30, 60, 120, 240]:
    short_trades = completed_trades[completed_trades['Duration_Minutes'] <= threshold]
    if len(short_trades) > 0:
        win_rate = (short_trades['P&L'] > 0).sum() / len(short_trades) * 100
        avg_pnl = short_trades['P&L'].mean()
        print(f"  Trades <= {threshold} min: {len(short_trades)} trades, {win_rate:.2f}% win rate, ${avg_pnl:.2f} avg P&L")

# ============================================================================
# 2. POST-TRADE BEHAVIOR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. POST-TRADE BEHAVIOR ANALYSIS")
print("="*80)
print("Analyzing what happens after losing trades...")

# For losing trades, check what happens after exit
losing_trades_analysis = []

for idx, trade in losing.iterrows():
    exit_time = trade['ExitTime']
    # Look at next 1 hour (12 candles) after exit
    future_data = df_data.loc[exit_time:exit_time + timedelta(hours=1)]
    
    if len(future_data) > 0:
        if trade['Type'] == 'BUY':
            # For losing BUY, price went down - check if it continued down or reversed
            max_price_after = future_data['High'].max()
            min_price_after = future_data['Low'].min()
            final_price = future_data.iloc[-1]['Close']
            
            # Could we have held longer and made profit?
            potential_profit = max_price_after - trade['EntryPrice']
            potential_loss = min_price_after - trade['EntryPrice']
            
        else:  # SELL
            max_price_after = future_data['High'].max()
            min_price_after = future_data['Low'].min()
            final_price = future_data.iloc[-1]['Close']
            
            potential_profit = trade['EntryPrice'] - min_price_after
            potential_loss = trade['EntryPrice'] - max_price_after
        
        losing_trades_analysis.append({
            'TradeID': idx,
            'Type': trade['Type'],
            'EntryPrice': trade['EntryPrice'],
            'ExitPrice': trade['ExitPrice'],
            'P&L': trade['P&L'],
            'MaxPriceAfter': max_price_after,
            'MinPriceAfter': min_price_after,
            'FinalPriceAfter1H': final_price,
            'PotentialProfit': potential_profit,
            'PotentialLoss': potential_loss,
            'WouldHaveWonIfHeld': potential_profit > abs(trade['P&L'])
        })

if losing_trades_analysis:
    post_analysis = pd.DataFrame(losing_trades_analysis)
    
    would_have_won = post_analysis[post_analysis['WouldHaveWonIfHeld']]
    print(f"\nLosing trades that would have been winners if held longer:")
    print(f"  Count: {len(would_have_won)} out of {len(post_analysis)} ({len(would_have_won)/len(post_analysis)*100:.1f}%)")
    if len(would_have_won) > 0:
        print(f"  Average potential profit: ${would_have_won['PotentialProfit'].mean():.2f}")
        print(f"  Average actual loss: ${would_have_won['P&L'].mean():.2f}")
        print(f"  Net improvement: ${(would_have_won['PotentialProfit'].mean() + abs(would_have_won['P&L'].mean())):.2f} per trade")

# ============================================================================
# 3. VOLUME ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. VOLUME ANALYSIS AT ENTRY")
print("="*80)

# Get volume data at entry
entry_volumes = []
for idx, trade in completed_trades.iterrows():
    entry_time = trade['EntryTime']
    if entry_time in df_data.index:
        entry_volumes.append({
            'TradeID': idx,
            'Volume': df_data.loc[entry_time, 'Volume'],
            'Volume_MA_20': df_data.loc[entry_time, 'Volume'] if 'Volume' in df_data.columns else None,
            'P&L': trade['P&L'],
            'Won': trade['P&L'] > 0
        })

if entry_volumes:
    vol_df = pd.DataFrame(entry_volumes)
    
    # Calculate volume moving average
    if 'Volume' in df_data.columns:
        df_data['Volume_MA20'] = df_data['Volume'].rolling(20).mean()
        
        for idx, trade in completed_trades.iterrows():
            entry_time = trade['EntryTime']
            if entry_time in df_data.index:
                vol_df.loc[vol_df['TradeID'] == idx, 'Volume_MA20'] = df_data.loc[entry_time, 'Volume_MA20']
        
        vol_df['Volume_Ratio'] = vol_df['Volume'] / vol_df['Volume_MA20']
        
        print(f"\nVolume Analysis:")
        winning_vol = vol_df[vol_df['Won'] == True]
        losing_vol = vol_df[vol_df['Won'] == False]
        
        if len(winning_vol) > 0 and len(losing_vol) > 0:
            print(f"  Winning trades avg volume ratio: {winning_vol['Volume_Ratio'].mean():.2f}")
            print(f"  Losing trades avg volume ratio: {losing_vol['Volume_Ratio'].mean():.2f}")
            
            # Volume thresholds
            for threshold in [0.8, 1.0, 1.2, 1.5, 2.0]:
                high_vol_trades = completed_trades[vol_df['Volume_Ratio'] >= threshold]
                if len(high_vol_trades) > 0:
                    win_rate = (high_vol_trades['P&L'] > 0).sum() / len(high_vol_trades) * 100
                    print(f"  Volume ratio >= {threshold}: {len(high_vol_trades)} trades, {win_rate:.2f}% win rate")

# ============================================================================
# 4. BREAKOUT CHARACTERISTICS
# ============================================================================
print("\n" + "="*80)
print("4. BREAKOUT CHARACTERISTICS")
print("="*80)

# Analyze breakout strength
breakout_analysis = []

for idx, trade in completed_trades.iterrows():
    entry_time = trade['EntryTime']
    if entry_time in df_data.index:
        entry_row = df_data.loc[entry_time]
        prev_row = df_data.loc[entry_time - timedelta(minutes=5)] if (entry_time - timedelta(minutes=5)) in df_data.index else None
        
        if prev_row is not None:
            if trade['Type'] == 'BUY':
                # How far above the range did we break?
                breakout_distance = trade['EntryPrice'] - trade['WindowHigh']
                breakout_pct = (breakout_distance / trade['WindowHigh']) * 100
            else:  # SELL
                breakout_distance = trade['WindowLow'] - trade['EntryPrice']
                breakout_pct = (breakout_distance / trade['WindowLow']) * 100
            
            # Range size
            range_size = trade['WindowHigh'] - trade['WindowLow']
            range_pct = (range_size / trade['EntryPrice']) * 100
            
            breakout_analysis.append({
                'TradeID': idx,
                'Type': trade['Type'],
                'BreakoutDistance': breakout_distance,
                'BreakoutPct': breakout_pct,
                'RangeSize': range_size,
                'RangePct': range_pct,
                'P&L': trade['P&L'],
                'Won': trade['P&L'] > 0
            })

if breakout_analysis:
    breakout_df = pd.DataFrame(breakout_analysis)
    
    print(f"\nBreakout Characteristics:")
    winning_breakouts = breakout_df[breakout_df['Won'] == True]
    losing_breakouts = breakout_df[breakout_df['Won'] == False]
    
    if len(winning_breakouts) > 0 and len(losing_breakouts) > 0:
        print(f"  Winning trades:")
        print(f"    Avg breakout %: {winning_breakouts['BreakoutPct'].abs().mean():.3f}%")
        print(f"    Avg range size: {winning_breakouts['RangePct'].mean():.3f}%")
        
        print(f"  Losing trades:")
        print(f"    Avg breakout %: {losing_breakouts['BreakoutPct'].abs().mean():.3f}%")
        print(f"    Avg range size: {losing_breakouts['RangePct'].mean():.3f}%")
        
        # Strong vs weak breakouts
        strong_breakouts = completed_trades[breakout_df['BreakoutPct'].abs() >= breakout_df['BreakoutPct'].abs().quantile(0.75)]
        weak_breakouts = completed_trades[breakout_df['BreakoutPct'].abs() <= breakout_df['BreakoutPct'].abs().quantile(0.25)]
        
        if len(strong_breakouts) > 0:
            print(f"\n  Strong breakouts (top 25%): {len(strong_breakouts)} trades")
            print(f"    Win rate: {(strong_breakouts['P&L'] > 0).sum() / len(strong_breakouts) * 100:.2f}%")
            print(f"    Avg P&L: ${strong_breakouts['P&L'].mean():.2f}")
        
        if len(weak_breakouts) > 0:
            print(f"\n  Weak breakouts (bottom 25%): {len(weak_breakouts)} trades")
            print(f"    Win rate: {(weak_breakouts['P&L'] > 0).sum() / len(weak_breakouts) * 100:.2f}%")
            print(f"    Avg P&L: ${weak_breakouts['P&L'].mean():.2f}")

# ============================================================================
# 5. INDICATOR COMBINATIONS
# ============================================================================
print("\n" + "="*80)
print("5. PROFITABLE INDICATOR COMBINATIONS")
print("="*80)

# Test various combinations
combinations = []

# Trend + Consolidation
trend_consolidating = completed_trades[
    (completed_trades['Trend_Score'] >= 0.67) & 
    (completed_trades['Consolidation_Score'] >= 0.5)
]
if len(trend_consolidating) > 0:
    win_rate = (trend_consolidating['P&L'] > 0).sum() / len(trend_consolidating) * 100
    combinations.append({
        'Condition': 'Strong Trend + Consolidation',
        'Count': len(trend_consolidating),
        'WinRate': win_rate,
        'AvgP&L': trend_consolidating['P&L'].mean(),
        'TotalP&L': trend_consolidating['P&L'].sum()
    })

# High ATR + Strong Breakout
if breakout_analysis:
    breakout_df = pd.DataFrame(breakout_analysis)
    high_atr_strong_breakout = completed_trades[
        (completed_trades['ATR_Ratio'] >= 1.2) &
        (breakout_df['BreakoutPct'].abs() >= breakout_df['BreakoutPct'].abs().quantile(0.75))
    ]
    if len(high_atr_strong_breakout) > 0:
        win_rate = (high_atr_strong_breakout['P&L'] > 0).sum() / len(high_atr_strong_breakout) * 100
        combinations.append({
            'Condition': 'High ATR + Strong Breakout',
            'Count': len(high_atr_strong_breakout),
            'WinRate': win_rate,
            'AvgP&L': high_atr_strong_breakout['P&L'].mean(),
            'TotalP&L': high_atr_strong_breakout['P&L'].sum()
        })

# EMA alignment
ema_aligned = completed_trades[
    (completed_trades['EMA_9_Above_21'] == 1) &
    (completed_trades['EMA_21_Above_50'] == 1) &
    (completed_trades['Price_Above_EMA200_1H'] == 1) &
    (completed_trades['Type'] == 'BUY')
]
if len(ema_aligned) > 0:
    win_rate = (ema_aligned['P&L'] > 0).sum() / len(ema_aligned) * 100
    combinations.append({
        'Condition': 'All EMAs Aligned (BUY only)',
        'Count': len(ema_aligned),
        'WinRate': win_rate,
        'AvgP&L': ema_aligned['P&L'].mean(),
        'TotalP&L': ema_aligned['P&L'].sum()
    })

# Short duration + High volume
if entry_volumes:
    vol_df = pd.DataFrame(entry_volumes)
    if 'Volume_Ratio' in vol_df.columns:
        short_high_vol = completed_trades[
            (completed_trades['Duration_Minutes'] <= 30) &
            (vol_df['Volume_Ratio'] >= 1.5)
        ]
        if len(short_high_vol) > 0:
            win_rate = (short_high_vol['P&L'] > 0).sum() / len(short_high_vol) * 100
            combinations.append({
                'Condition': 'Short Duration (<=30min) + High Volume',
                'Count': len(short_high_vol),
                'WinRate': win_rate,
                'AvgP&L': short_high_vol['P&L'].mean(),
                'TotalP&L': short_high_vol['P&L'].sum()
            })

if combinations:
    comb_df = pd.DataFrame(combinations)
    comb_df = comb_df.sort_values('WinRate', ascending=False)
    print("\nTop Profitable Combinations:")
    print(comb_df.to_string(index=False))

# ============================================================================
# 6. SAVE ENHANCED DATASET
# ============================================================================
print("\n" + "="*80)
print("6. SAVING ENHANCED DATASET")
print("="*80)

# Add new features to completed trades
if 'Duration_Minutes' in completed_trades.columns:
    enhanced_trades = completed_trades.copy()
    
    if breakout_analysis:
        breakout_df = pd.DataFrame(breakout_analysis)
        enhanced_trades = enhanced_trades.merge(
            breakout_df[['TradeID', 'BreakoutDistance', 'BreakoutPct', 'RangeSize', 'RangePct']],
            left_index=True,
            right_on='TradeID',
            how='left'
        )
    
    if entry_volumes:
        vol_df = pd.DataFrame(entry_volumes)
        if 'Volume_Ratio' in vol_df.columns:
            enhanced_trades = enhanced_trades.merge(
                vol_df[['TradeID', 'Volume', 'Volume_Ratio']],
                left_index=True,
                right_on='TradeID',
                how='left'
            )
    
    enhanced_trades.to_csv('enhanced_trades_for_ml.csv', index=False)
    print(f"Enhanced dataset saved: enhanced_trades_for_ml.csv")
    print(f"  Total trades: {len(enhanced_trades)}")
    print(f"  New features added: Duration, Breakout characteristics, Volume ratios")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)

