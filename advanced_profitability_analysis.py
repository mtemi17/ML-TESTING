import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADVANCED PROFITABILITY ANALYSIS - WAITING STRATEGIES & PATTERNS")
print("="*80)

# Load data
df_trades = pd.read_csv('backtest_results.csv')
df_data = pd.read_csv('XAUUSD5_with_indicators.csv', index_col=0, parse_dates=True)

completed_trades = df_trades[df_trades['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
completed_trades['EntryTime'] = pd.to_datetime(completed_trades['EntryTime'])
completed_trades['ExitTime'] = pd.to_datetime(completed_trades['ExitTime'])

# ============================================================================
# 1. WAITING STRATEGY: What if we held losing trades longer?
# ============================================================================
print("\n" + "="*80)
print("1. WAITING STRATEGY ANALYSIS")
print("="*80)
print("Simulating: What if we held losing trades until they became profitable?")

waiting_results = []

for idx, trade in completed_trades.iterrows():
    if trade['P&L'] < 0:  # Only analyze losing trades
        entry_time = trade['EntryTime']
        exit_time = trade['ExitTime']
        
        # Look ahead up to 24 hours after original exit
        future_data = df_data.loc[exit_time:exit_time + timedelta(hours=24)]
        
        if len(future_data) > 0:
            if trade['Type'] == 'BUY':
                # Check if price ever reached TP level
                max_price = future_data['High'].max()
                reached_tp = max_price >= trade['TP']
                
                # Find when it would have hit TP
                tp_hit_time = None
                if reached_tp:
                    tp_candles = future_data[future_data['High'] >= trade['TP']]
                    if len(tp_candles) > 0:
                        tp_hit_time = tp_candles.index[0]
                        wait_duration = (tp_hit_time - exit_time).total_seconds() / 60
                    else:
                        wait_duration = None
                else:
                    wait_duration = None
                
                # Best possible outcome
                best_price = future_data['High'].max()
                best_pnl = best_price - trade['EntryPrice']
                best_time = future_data['High'].idxmax()
                best_wait = (best_time - exit_time).total_seconds() / 60 if best_time else None
                
            else:  # SELL
                min_price = future_data['Low'].min()
                reached_tp = min_price <= trade['TP']
                
                tp_hit_time = None
                if reached_tp:
                    tp_candles = future_data[future_data['Low'] <= trade['TP']]
                    if len(tp_candles) > 0:
                        tp_hit_time = tp_candles.index[0]
                        wait_duration = (tp_hit_time - exit_time).total_seconds() / 60
                    else:
                        wait_duration = None
                else:
                    wait_duration = None
                
                best_price = future_data['Low'].min()
                best_pnl = trade['EntryPrice'] - best_price
                best_time = future_data['Low'].idxmin()
                best_wait = (best_time - exit_time).total_seconds() / 60 if best_time else None
            
            waiting_results.append({
                'TradeID': idx,
                'Type': trade['Type'],
                'OriginalP&L': trade['P&L'],
                'ReachedTP': reached_tp,
                'WaitDuration_Min': wait_duration,
                'BestPossibleP&L': best_pnl,
                'BestWaitDuration_Min': best_wait,
                'WouldHaveWon': best_pnl > 0
            })

if waiting_results:
    wait_df = pd.DataFrame(waiting_results)
    
    would_have_won = wait_df[wait_df['WouldHaveWon'] == True]
    reached_tp = wait_df[wait_df['ReachedTP'] == True]
    
    print(f"\nResults:")
    print(f"  Losing trades analyzed: {len(wait_df)}")
    print(f"  Would have become winners: {len(would_have_won)} ({len(would_have_won)/len(wait_df)*100:.1f}%)")
    print(f"  Would have reached TP: {len(reached_tp)} ({len(reached_tp)/len(wait_df)*100:.1f}%)")
    
    if len(would_have_won) > 0:
        print(f"\n  Average wait time for winners: {would_have_won['BestWaitDuration_Min'].mean():.1f} minutes")
        print(f"  Median wait time: {would_have_won['BestWaitDuration_Min'].median():.1f} minutes")
        print(f"  Average improvement: ${(would_have_won['BestPossibleP&L'].mean() + abs(would_have_won['OriginalP&L'].mean())):.2f} per trade")
        
        # Time-based thresholds
        print(f"\n  Wait time analysis:")
        for threshold in [60, 120, 240, 480, 1440]:  # 1h, 2h, 4h, 8h, 24h
            within_threshold = would_have_won[would_have_won['BestWaitDuration_Min'] <= threshold]
            if len(within_threshold) > 0:
                print(f"    Within {threshold} min: {len(within_threshold)} trades, avg P&L: ${within_threshold['BestPossibleP&L'].mean():.2f}")

# ============================================================================
# 2. PRICE ACTION AFTER ENTRY
# ============================================================================
print("\n" + "="*80)
print("2. PRICE ACTION PATTERNS AFTER ENTRY")
print("="*80)

price_action_analysis = []

for idx, trade in completed_trades.iterrows():
    entry_time = trade['EntryTime']
    
    # Look at first 30 minutes after entry
    post_entry = df_data.loc[entry_time:entry_time + timedelta(minutes=30)]
    
    if len(post_entry) > 0:
        if trade['Type'] == 'BUY':
            # How far did price go in our favor initially?
            max_favorable = post_entry['High'].max() - trade['EntryPrice']
            min_adverse = trade['EntryPrice'] - post_entry['Low'].min()
            
            # Did it immediately go against us?
            first_candle = post_entry.iloc[0] if len(post_entry) > 0 else None
            if first_candle is not None:
                first_move = first_candle['Close'] - trade['EntryPrice']
                went_against = first_move < 0
            else:
                first_move = None
                went_against = None
                
        else:  # SELL
            max_favorable = trade['EntryPrice'] - post_entry['Low'].min()
            min_adverse = post_entry['High'].max() - trade['EntryPrice']
            
            first_candle = post_entry.iloc[0] if len(post_entry) > 0 else None
            if first_candle is not None:
                first_move = trade['EntryPrice'] - first_candle['Close']
                went_against = first_move < 0
            else:
                first_move = None
                went_against = None
        
        price_action_analysis.append({
            'TradeID': idx,
            'Type': trade['Type'],
            'MaxFavorable_Min30': max_favorable,
            'MinAdverse_Min30': min_adverse,
            'FirstMove': first_move,
            'WentAgainstFirst': went_against,
            'P&L': trade['P&L'],
            'Won': trade['P&L'] > 0
        })

if price_action_analysis:
    pa_df = pd.DataFrame(price_action_analysis)
    
    print(f"\nPrice Action Patterns:")
    winning = pa_df[pa_df['Won'] == True]
    losing = pa_df[pa_df['Won'] == False]
    
    print(f"\n  Winning trades:")
    print(f"    Avg max favorable move (30min): ${winning['MaxFavorable_Min30'].mean():.2f}")
    print(f"    Avg min adverse move: ${winning['MinAdverse_Min30'].mean():.2f}")
    print(f"    Went against first: {(winning['WentAgainstFirst'] == True).sum()} / {len(winning)}")
    
    print(f"\n  Losing trades:")
    print(f"    Avg max favorable move (30min): ${losing['MaxFavorable_Min30'].mean():.2f}")
    print(f"    Avg min adverse move: ${losing['MinAdverse_Min30'].mean():.2f}")
    print(f"    Went against first: {(losing['WentAgainstFirst'] == True).sum()} / {len(losing)}")
    
    # Pattern: Trades that went against us first
    went_against = completed_trades[pa_df['WentAgainstFirst'] == True]
    if len(went_against) > 0:
        win_rate = (went_against['P&L'] > 0).sum() / len(went_against) * 100
        print(f"\n  Trades that went against us first:")
        print(f"    Count: {len(went_against)}")
        print(f"    Win rate: {win_rate:.2f}%")
        print(f"    Avg P&L: ${went_against['P&L'].mean():.2f}")

# ============================================================================
# 3. TIME-BASED PATTERNS
# ============================================================================
print("\n" + "="*80)
print("3. TIME-BASED PROFITABILITY PATTERNS")
print("="*80)

completed_trades['EntryHour'] = completed_trades['EntryTime'].dt.hour
completed_trades['EntryDayOfWeek'] = completed_trades['EntryTime'].dt.dayofweek

print(f"\nBy Entry Hour:")
hour_stats = []
for hour in range(24):
    hour_trades = completed_trades[completed_trades['EntryHour'] == hour]
    if len(hour_trades) > 0:
        win_rate = (hour_trades['P&L'] > 0).sum() / len(hour_trades) * 100
        hour_stats.append({
            'Hour': hour,
            'Count': len(hour_trades),
            'WinRate': win_rate,
            'AvgP&L': hour_trades['P&L'].mean(),
            'TotalP&L': hour_trades['P&L'].sum()
        })

if hour_stats:
    hour_df = pd.DataFrame(hour_stats).sort_values('WinRate', ascending=False)
    print(f"\nTop 5 Most Profitable Hours:")
    print(hour_df.head(5).to_string(index=False))
    print(f"\nBottom 5 Least Profitable Hours:")
    print(hour_df.tail(5).to_string(index=False))

print(f"\nBy Day of Week:")
dow_stats = []
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for dow in range(7):
    dow_trades = completed_trades[completed_trades['EntryDayOfWeek'] == dow]
    if len(dow_trades) > 0:
        win_rate = (dow_trades['P&L'] > 0).sum() / len(dow_trades) * 100
        dow_stats.append({
            'Day': days[dow],
            'Count': len(dow_trades),
            'WinRate': win_rate,
            'AvgP&L': dow_trades['P&L'].mean(),
            'TotalP&L': dow_trades['P&L'].sum()
        })

if dow_stats:
    dow_df = pd.DataFrame(dow_stats).sort_values('WinRate', ascending=False)
    print(dow_df.to_string(index=False))

# ============================================================================
# 4. RANGE SIZE VS PROFITABILITY
# ============================================================================
print("\n" + "="*80)
print("4. RANGE SIZE ANALYSIS")
print("="*80)

completed_trades['RangeSize'] = completed_trades['WindowHigh'] - completed_trades['WindowLow']
completed_trades['RangeSizePct'] = (completed_trades['RangeSize'] / completed_trades['EntryPrice']) * 100

# Categorize by range size
completed_trades['RangeCategory'] = pd.cut(
    completed_trades['RangeSizePct'],
    bins=[0, 0.1, 0.15, 0.2, 0.3, float('inf')],
    labels=['Very Small (<0.1%)', 'Small (0.1-0.15%)', 'Medium (0.15-0.2%)', 'Large (0.2-0.3%)', 'Very Large (>0.3%)']
)

print(f"\nRange Size Categories:")
range_stats = completed_trades.groupby('RangeCategory').agg({
    'P&L': ['count', 'mean', 'sum'],
    'R_Multiple': 'mean'
}).round(2)
print(range_stats)

# ============================================================================
# 5. COMPREHENSIVE PROFITABLE FILTER
# ============================================================================
print("\n" + "="*80)
print("5. COMPREHENSIVE PROFITABLE FILTER COMBINATION")
print("="*80)

# Combine all profitable patterns
if price_action_analysis:
    pa_df = pd.DataFrame(price_action_analysis)
    
    # Best combination based on findings
    best_filter = completed_trades[
        # EMA alignment
        (completed_trades['EMA_9_Above_21'] == 1) &
        (completed_trades['EMA_21_Above_50'] == 1) &
        (completed_trades['Price_Above_EMA200_1H'] == 1) &
        (completed_trades['Type'] == 'BUY') &
        # Not consolidating
        (completed_trades['Is_Consolidating'] == 0) &
        # Price didn't go against us first
        (pa_df['WentAgainstFirst'] == False) &
        # Reasonable range size
        (completed_trades['RangeSizePct'] >= 0.1) &
        (completed_trades['RangeSizePct'] <= 0.3)
    ]
    
    if len(best_filter) > 0:
        win_rate = (best_filter['P&L'] > 0).sum() / len(best_filter) * 100
        print(f"\nOptimal Filter Combination:")
        print(f"  Conditions:")
        print(f"    - All EMAs aligned (BUY only)")
        print(f"    - Not consolidating")
        print(f"    - Price didn't go against us first")
        print(f"    - Range size: 0.1% - 0.3%")
        print(f"\n  Results:")
        print(f"    Total trades: {len(best_filter)}")
        print(f"    Win rate: {win_rate:.2f}%")
        print(f"    Total P&L: ${best_filter['P&L'].sum():.2f}")
        print(f"    Avg P&L: ${best_filter['P&L'].mean():.2f}")
        print(f"    Avg R-Multiple: {best_filter['R_Multiple'].mean():.2f}")

# ============================================================================
# 6. SAVE ALL FINDINGS
# ============================================================================
print("\n" + "="*80)
print("6. SAVING ENHANCED ANALYSIS")
print("="*80)

# Merge all new features
enhanced = completed_trades.copy()

if waiting_results:
    wait_df = pd.DataFrame(waiting_results)
    enhanced = enhanced.merge(
        wait_df[['TradeID', 'ReachedTP', 'WaitDuration_Min', 'BestPossibleP&L', 'WouldHaveWon']],
        left_index=True,
        right_on='TradeID',
        how='left'
    )

if price_action_analysis:
    pa_df = pd.DataFrame(price_action_analysis)
    enhanced = enhanced.merge(
        pa_df[['TradeID', 'MaxFavorable_Min30', 'MinAdverse_Min30', 'WentAgainstFirst']],
        left_index=True,
        right_on='TradeID',
        how='left'
    )

enhanced.to_csv('advanced_analysis_results.csv', index=False)
print(f"Advanced analysis saved: advanced_analysis_results.csv")
print(f"  Total trades: {len(enhanced)}")
print(f"  New features: Waiting strategy, Price action, Time patterns, Range analysis")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

