import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FINDING OPTIMAL EMA CONFIGURATION")
print("="*80)
print("Analyzing all trades to determine best EMA periods and combinations")
print("="*80)

# ============================================================================
# 1. LOAD ALL DATA
# ============================================================================
print("\n1. Loading all trade data...")

df_original = pd.read_csv('advanced_analysis_results_combined.csv')
df_new = pd.read_csv('new_data_backtest_results.csv')
df_all = pd.concat([df_original, df_new], ignore_index=True)
df_all = df_all[df_all['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

print(f"   Total trades: {len(df_all)}")
print(f"   Wins: {(df_all['P&L'] > 0).sum()} ({((df_all['P&L'] > 0).sum()/len(df_all)*100):.1f}%)")
print(f"   Losses: {(df_all['P&L'] <= 0).sum()} ({((df_all['P&L'] <= 0).sum()/len(df_all)*100):.1f}%)")

# ============================================================================
# 2. LOAD RAW PRICE DATA TO CALCULATE DIFFERENT EMAs
# ============================================================================
print("\n2. Loading raw price data to calculate EMAs...")

# Load all raw data files
raw_files = ['XAUUSD5.csv', 'XAUUSD5 new.csv', 'XAUUSD5 new data.csv']
df_raw_list = []

for file in raw_files:
    try:
        df_temp = pd.read_csv(file, header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_temp['DateTime'] = pd.to_datetime(df_temp['Date'] + ' ' + df_temp['Time'])
        df_temp = df_temp.set_index('DateTime')
        df_raw_list.append(df_temp[['Open', 'High', 'Low', 'Close', 'Volume']])
        print(f"   Loaded: {file} ({len(df_temp)} rows)")
    except Exception as e:
        print(f"   Skipped {file}: {e}")

if len(df_raw_list) > 0:
    df_raw = pd.concat(df_raw_list)
    df_raw = df_raw.sort_index()
    print(f"   Total raw data: {len(df_raw)} rows")
else:
    print("   ERROR: No raw data found")
    exit(1)

# ============================================================================
# 3. TEST DIFFERENT EMA PERIODS
# ============================================================================
print("\n" + "="*80)
print("3. TESTING DIFFERENT EMA PERIODS")
print("="*80)

# Common EMA periods to test
ema_periods_5m = [5, 8, 9, 10, 12, 13, 15, 20, 21, 25, 30, 34, 50, 55, 60, 89, 100, 144, 200]
ema_periods_1h = [50, 100, 150, 200, 250, 300]

print(f"\nTesting {len(ema_periods_5m)} EMA periods on 5M timeframe...")
print(f"Testing {len(ema_periods_1h)} EMA periods on 1H timeframe...")

# Calculate EMAs for all periods
ema_results = {}

for period in ema_periods_5m:
    df_raw[f'EMA_{period}_5M'] = df_raw['Close'].ewm(span=period, adjust=False).mean()

# Resample to 1H for 1H EMAs
df_1h = df_raw.resample('1H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

for period in ema_periods_1h:
    df_1h[f'EMA_{period}_1H'] = df_1h['Close'].ewm(span=period, adjust=False).mean()
    # Forward fill back to 5M
    df_raw[f'EMA_{period}_1H'] = df_1h[f'EMA_{period}_1H'].reindex(df_raw.index, method='ffill')

# ============================================================================
# 4. MATCH EMAs TO TRADES
# ============================================================================
print("\n4. Matching EMAs to trades...")

df_all['EntryTime'] = pd.to_datetime(df_all['EntryTime'])
df_all = df_all.sort_values('EntryTime')

# Merge EMAs with trades
for period in ema_periods_5m:
    df_all = df_all.merge(
        df_raw[['EMA_{}_5M'.format(period)]].reset_index(),
        left_on='EntryTime',
        right_on='DateTime',
        how='left',
        suffixes=('', '_new')
    )
    if f'EMA_{period}_5M_new' in df_all.columns:
        df_all[f'EMA_{period}_5M'] = df_all[f'EMA_{period}_5M_new'].fillna(df_all.get(f'EMA_{period}_5M', 0))
        df_all = df_all.drop(columns=[f'EMA_{period}_5M_new'], errors='ignore')
    df_all = df_all.drop(columns=['DateTime'], errors='ignore')

for period in ema_periods_1h:
    df_all = df_all.merge(
        df_raw[['EMA_{}_1H'.format(period)]].reset_index(),
        left_on='EntryTime',
        right_on='DateTime',
        how='left',
        suffixes=('', '_new')
    )
    if f'EMA_{period}_1H_new' in df_all.columns:
        df_all[f'EMA_{period}_1H'] = df_all[f'EMA_{period}_1H_new'].fillna(df_all.get(f'EMA_{period}_1H', 0))
        df_all = df_all.drop(columns=[f'EMA_{period}_1H_new'], errors='ignore')
    df_all = df_all.drop(columns=['DateTime'], errors='ignore')

# ============================================================================
# 5. ANALYZE EMA PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("5. ANALYZING EMA PERFORMANCE")
print("="*80)

df_wins = df_all[df_all['P&L'] > 0].copy()
df_losses = df_all[df_all['P&L'] <= 0].copy()

ema_analysis = []

print("\nAnalyzing individual EMAs...")

for period in ema_periods_5m:
    ema_col = f'EMA_{period}_5M'
    if ema_col in df_all.columns:
        # Price above/below EMA
        df_all[f'Price_Above_{period}'] = (df_all['EntryPrice'] > df_all[ema_col]).astype(int)
        
        # Win rate when price above EMA
        above_trades = df_all[df_all[f'Price_Above_{period}'] == 1]
        if len(above_trades) > 20:
            winrate_above = (above_trades['P&L'] > 0).sum() / len(above_trades) * 100
            pnl_above = above_trades['P&L'].sum()
        else:
            winrate_above = 0
            pnl_above = 0
        
        # Win rate when price below EMA
        below_trades = df_all[df_all[f'Price_Above_{period}'] == 0]
        if len(below_trades) > 20:
            winrate_below = (below_trades['P&L'] > 0).sum() / len(below_trades) * 100
            pnl_below = below_trades['P&L'].sum()
        else:
            winrate_below = 0
            pnl_below = 0
        
        # Best direction
        if winrate_above > winrate_below:
            best_direction = 'Above'
            best_winrate = winrate_above
            best_pnl = pnl_above
            best_trades = len(above_trades)
        else:
            best_direction = 'Below'
            best_winrate = winrate_below
            best_pnl = pnl_below
            best_trades = len(below_trades)
        
        if best_winrate > 0:
            ema_analysis.append({
                'EMA': f'EMA_{period}_5M',
                'Period': period,
                'Timeframe': '5M',
                'Direction': best_direction,
                'WinRate': best_winrate,
                'TotalP&L': best_pnl,
                'Trades': best_trades,
                'AvgP&L': best_pnl / best_trades if best_trades > 0 else 0
            })

# Same for 1H EMAs
for period in ema_periods_1h:
    ema_col = f'EMA_{period}_1H'
    if ema_col in df_all.columns:
        df_all[f'Price_Above_{period}_1H'] = (df_all['EntryPrice'] > df_all[ema_col]).astype(int)
        
        above_trades = df_all[df_all[f'Price_Above_{period}_1H'] == 1]
        if len(above_trades) > 20:
            winrate_above = (above_trades['P&L'] > 0).sum() / len(above_trades) * 100
            pnl_above = above_trades['P&L'].sum()
        else:
            winrate_above = 0
            pnl_above = 0
        
        below_trades = df_all[df_all[f'Price_Above_{period}_1H'] == 0]
        if len(below_trades) > 20:
            winrate_below = (below_trades['P&L'] > 0).sum() / len(below_trades) * 100
            pnl_below = below_trades['P&L'].sum()
        else:
            winrate_below = 0
            pnl_below = 0
        
        if winrate_above > winrate_below:
            best_direction = 'Above'
            best_winrate = winrate_above
            best_pnl = pnl_above
            best_trades = len(above_trades)
        else:
            best_direction = 'Below'
            best_winrate = winrate_below
            best_pnl = pnl_below
            best_trades = len(below_trades)
        
        if best_winrate > 0:
            ema_analysis.append({
                'EMA': f'EMA_{period}_1H',
                'Period': period,
                'Timeframe': '1H',
                'Direction': best_direction,
                'WinRate': best_winrate,
                'TotalP&L': best_pnl,
                'Trades': best_trades,
                'AvgP&L': best_pnl / best_trades if best_trades > 0 else 0
            })

ema_df = pd.DataFrame(ema_analysis)
ema_df = ema_df.sort_values('WinRate', ascending=False)

print("\nTop 20 Individual EMAs:")
print("-"*100)
print(f"{'EMA':<20} {'Period':<8} {'TF':<4} {'Dir':<6} {'WinRate':<10} {'TotalP&L':<12} {'Trades':<8} {'AvgP&L':<10}")
print("-"*100)
for idx, row in ema_df.head(20).iterrows():
    print(f"{row['EMA']:<20} {int(row['Period']):<8} {row['Timeframe']:<4} {row['Direction']:<6} {row['WinRate']:>9.1f}% ${row['TotalP&L']:>10.2f} {int(row['Trades']):<8} ${row['AvgP&L']:>9.2f}")

# ============================================================================
# 6. TEST EMA COMBINATIONS (2 EMAs)
# ============================================================================
print("\n" + "="*80)
print("6. TESTING EMA COMBINATIONS (2 EMAs)")
print("="*80)

# Get top EMAs
top_emas_5m = ema_df[ema_df['Timeframe'] == '5M'].head(10)['EMA'].tolist()
top_emas_1h = ema_df[ema_df['Timeframe'] == '1H'].head(5)['EMA'].tolist()

print(f"\nTesting combinations of top {len(top_emas_5m)} 5M EMAs and top {len(top_emas_1h)} 1H EMAs...")

combination_results = []

for ema1 in top_emas_5m[:5]:  # Top 5
    period1 = int(ema1.split('_')[1].split('_')[0])
    for ema2 in top_emas_5m[:5]:
        if ema1 >= ema2:  # Avoid duplicates
            continue
        period2 = int(ema2.split('_')[1].split('_')[0])
        
        # Create alignment pattern
        col1 = f'Price_Above_{period1}'
        col2 = f'Price_Above_{period2}'
        
        if col1 in df_all.columns and col2 in df_all.columns:
            # Both above
            both_above = df_all[(df_all[col1] == 1) & (df_all[col2] == 1)]
            if len(both_above) > 10:
                winrate = (both_above['P&L'] > 0).sum() / len(both_above) * 100
                pnl = both_above['P&L'].sum()
                combination_results.append({
                    'Type': '5M_5M',
                    'EMA1': ema1,
                    'EMA2': ema2,
                    'Pattern': 'Both_Above',
                    'WinRate': winrate,
                    'TotalP&L': pnl,
                    'Trades': len(both_above),
                    'AvgP&L': pnl / len(both_above) if len(both_above) > 0 else 0
                })
            
            # EMA1 above, EMA2 below
            ema1_above = df_all[(df_all[col1] == 1) & (df_all[col2] == 0)]
            if len(ema1_above) > 10:
                winrate = (ema1_above['P&L'] > 0).sum() / len(ema1_above) * 100
                pnl = ema1_above['P&L'].sum()
                combination_results.append({
                    'Type': '5M_5M',
                    'EMA1': ema1,
                    'EMA2': ema2,
                    'Pattern': 'EMA1_Above_EMA2_Below',
                    'WinRate': winrate,
                    'TotalP&L': pnl,
                    'Trades': len(ema1_above),
                    'AvgP&L': pnl / len(ema1_above) if len(ema1_above) > 0 else 0
                })

# Test 5M + 1H combinations
for ema_5m in top_emas_5m[:5]:
    period_5m = int(ema_5m.split('_')[1].split('_')[0])
    for ema_1h in top_emas_1h[:3]:
        period_1h = int(ema_1h.split('_')[1].split('_')[0])
        
        col_5m = f'Price_Above_{period_5m}'
        col_1h = f'Price_Above_{period_1h}_1H'
        
        if col_5m in df_all.columns and col_1h in df_all.columns:
            # Both aligned
            both_aligned = df_all[(df_all[col_5m] == 1) & (df_all[col_1h] == 1)]
            if len(both_aligned) > 10:
                winrate = (both_aligned['P&L'] > 0).sum() / len(both_aligned) * 100
                pnl = both_aligned['P&L'].sum()
                combination_results.append({
                    'Type': '5M_1H',
                    'EMA1': ema_5m,
                    'EMA2': ema_1h,
                    'Pattern': 'Both_Above',
                    'WinRate': winrate,
                    'TotalP&L': pnl,
                    'Trades': len(both_aligned),
                    'AvgP&L': pnl / len(both_aligned) if len(both_aligned) > 0 else 0
                })

comb_df = pd.DataFrame(combination_results)
comb_df = comb_df.sort_values('WinRate', ascending=False)

print("\nTop 15 EMA Combinations:")
print("-"*120)
print(f"{'Type':<8} {'EMA1':<20} {'EMA2':<20} {'Pattern':<25} {'WinRate':<10} {'TotalP&L':<12} {'Trades':<8} {'AvgP&L':<10}")
print("-"*120)
for idx, row in comb_df.head(15).iterrows():
    print(f"{row['Type']:<8} {row['EMA1']:<20} {row['EMA2']:<20} {row['Pattern']:<25} {row['WinRate']:>9.1f}% ${row['TotalP&L']:>10.2f} {int(row['Trades']):<8} ${row['AvgP&L']:>9.2f}")

# ============================================================================
# 7. TEST 3-EMA COMBINATIONS
# ============================================================================
print("\n" + "="*80)
print("7. TESTING 3-EMA COMBINATIONS")
print("="*80)

# Test best 2-EMA combinations with a third EMA
top_combos = comb_df.head(10)
three_ema_results = []

for idx, combo in top_combos.iterrows():
    ema1 = combo['EMA1']
    ema2 = combo['EMA2']
    
    # Add a third EMA (1H)
    for ema3 in top_emas_1h[:3]:
        period1 = int(ema1.split('_')[1].split('_')[0])
        period2 = int(ema2.split('_')[1].split('_')[0])
        period3 = int(ema3.split('_')[1].split('_')[0])
        
        col1 = f'Price_Above_{period1}'
        col2 = f'Price_Above_{period2}'
        col3 = f'Price_Above_{period3}_1H'
        
        if all(c in df_all.columns for c in [col1, col2, col3]):
            # All three aligned
            all_aligned = df_all[(df_all[col1] == 1) & (df_all[col2] == 1) & (df_all[col3] == 1)]
            if len(all_aligned) > 5:
                winrate = (all_aligned['P&L'] > 0).sum() / len(all_aligned) * 100
                pnl = all_aligned['P&L'].sum()
                three_ema_results.append({
                    'EMA1': ema1,
                    'EMA2': ema2,
                    'EMA3': ema3,
                    'Pattern': 'All_Above',
                    'WinRate': winrate,
                    'TotalP&L': pnl,
                    'Trades': len(all_aligned),
                    'AvgP&L': pnl / len(all_aligned) if len(all_aligned) > 0 else 0
                })

three_ema_df = pd.DataFrame(three_ema_results)
three_ema_df = three_ema_df.sort_values('WinRate', ascending=False)

if len(three_ema_df) > 0:
    print("\nTop 10 3-EMA Combinations:")
    print("-"*140)
    print(f"{'EMA1':<20} {'EMA2':<20} {'EMA3':<20} {'WinRate':<10} {'TotalP&L':<12} {'Trades':<8} {'AvgP&L':<10}")
    print("-"*140)
    for idx, row in three_ema_df.head(10).iterrows():
        print(f"{row['EMA1']:<20} {row['EMA2']:<20} {row['EMA3']:<20} {row['WinRate']:>9.1f}% ${row['TotalP&L']:>10.2f} {int(row['Trades']):<8} ${row['AvgP&L']:>9.2f}")

# ============================================================================
# 8. FIND OPTIMAL EMA CONFIGURATION
# ============================================================================
print("\n" + "="*80)
print("8. OPTIMAL EMA CONFIGURATION")
print("="*80)

# Best single EMA
best_single = ema_df.iloc[0]
print(f"\n✅ Best Single EMA:")
print(f"   {best_single['EMA']} ({best_single['Timeframe']})")
print(f"   Direction: Price {best_single['Direction']} EMA")
print(f"   Win Rate: {best_single['WinRate']:.2f}%")
print(f"   Total P&L: ${best_single['TotalP&L']:.2f}")
print(f"   Trades: {int(best_single['Trades'])}")

# Best 2-EMA combination
if len(comb_df) > 0:
    best_combo = comb_df.iloc[0]
    print(f"\n✅ Best 2-EMA Combination:")
    print(f"   {best_combo['EMA1']} + {best_combo['EMA2']}")
    print(f"   Pattern: {best_combo['Pattern']}")
    print(f"   Win Rate: {best_combo['WinRate']:.2f}%")
    print(f"   Total P&L: ${best_combo['TotalP&L']:.2f}")
    print(f"   Trades: {int(best_combo['Trades'])}")

# Best 3-EMA combination
if len(three_ema_df) > 0:
    best_three = three_ema_df.iloc[0]
    print(f"\n✅ Best 3-EMA Combination:")
    print(f"   {best_three['EMA1']} + {best_three['EMA2']} + {best_three['EMA3']}")
    print(f"   Pattern: {best_three['Pattern']}")
    print(f"   Win Rate: {best_three['WinRate']:.2f}%")
    print(f"   Total P&L: ${best_three['TotalP&L']:.2f}")
    print(f"   Trades: {int(best_three['Trades'])}")

# ============================================================================
# 9. RECOMMENDED CONFIGURATION
# ============================================================================
print("\n" + "="*80)
print("9. RECOMMENDED EMA CONFIGURATION")
print("="*80)

# Choose best configuration based on win rate and trade count
if len(three_ema_df) > 0 and three_ema_df.iloc[0]['WinRate'] >= 50 and three_ema_df.iloc[0]['Trades'] >= 20:
    best_config = three_ema_df.iloc[0]
    recommended = {
        'type': '3_EMA',
        'ema1': best_config['EMA1'],
        'ema2': best_config['EMA2'],
        'ema3': best_config['EMA3'],
        'pattern': 'All_Above'
    }
elif len(comb_df) > 0 and comb_df.iloc[0]['WinRate'] >= 45 and comb_df.iloc[0]['Trades'] >= 30:
    best_config = comb_df.iloc[0]
    recommended = {
        'type': '2_EMA',
        'ema1': best_config['EMA1'],
        'ema2': best_config['EMA2'],
        'pattern': best_config['Pattern']
    }
else:
    best_config = ema_df.iloc[0]
    recommended = {
        'type': '1_EMA',
        'ema': best_config['EMA'],
        'direction': best_config['Direction']
    }

print(f"\n🎯 Recommended Configuration: {recommended['type']}")
print(f"   EMAs: {recommended}")

# Save results
ema_df.to_csv('optimal_ema_analysis.csv', index=False)
comb_df.to_csv('optimal_ema_combinations.csv', index=False)
if len(three_ema_df) > 0:
    three_ema_df.to_csv('optimal_ema_3combinations.csv', index=False)

# Save recommended configuration
import json
with open('recommended_ema_config.json', 'w') as f:
    json.dump(recommended, f, indent=2)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nFiles saved:")
print("  - optimal_ema_analysis.csv")
print("  - optimal_ema_combinations.csv")
print("  - optimal_ema_3combinations.csv")
print("  - recommended_ema_config.json")

