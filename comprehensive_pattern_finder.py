import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE PATTERN FINDER - FINDING OPTIMAL COMBINATIONS")
print("="*80)

# Load all trades
df_original = pd.read_csv('advanced_analysis_results_combined.csv')
df_new = pd.read_csv('new_data_backtest_results.csv')
df_all = pd.concat([df_original, df_new], ignore_index=True)
df_all = df_all[df_all['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

df_wins = df_all[df_all['P&L'] > 0].copy()
df_losses = df_all[df_all['P&L'] <= 0].copy()

print(f"\nTotal: {len(df_all)} trades | Wins: {len(df_wins)} | Losses: {len(df_losses)}")

# ============================================================================
# CREATE EMA PATTERN
# ============================================================================
df_all['EMA_Pattern'] = (
    df_all['EMA_9_Above_21'].astype(int).astype(str) +
    df_all['EMA_21_Above_50'].astype(int).astype(str) +
    df_all['Price_Above_EMA200_1H'].astype(int).astype(str)
)

# ============================================================================
# FIND OPTIMAL FILTER COMBINATIONS - EXHAUSTIVE SEARCH
# ============================================================================
print("\n" + "="*80)
print("FINDING OPTIMAL FILTER COMBINATIONS")
print("="*80)

best_combinations = []

# Test EMA Patterns
print("\n1. Testing EMA Patterns...")
pattern_analysis = df_all.groupby('EMA_Pattern').agg({
    'P&L': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean(), 'sum']
}).round(2)
pattern_analysis.columns = ['Total', 'Wins', 'WinRate', 'TotalP&L']

for pattern, row in pattern_analysis.iterrows():
    if row['WinRate'] >= 0.40 and row['Total'] >= 20:  # At least 40% win rate, 20+ trades
        best_combinations.append({
            'Type': 'EMA_Pattern',
            'Filter': f'Pattern_{pattern}',
            'Value': pattern,
            'Trades': int(row['Total']),
            'WinRate': row['WinRate'] * 100,
            'TotalP&L': row['TotalP&L'],
            'AvgP&L': row['TotalP&L'] / row['Total']
        })

# Test Risk ranges
print("2. Testing Risk Ranges...")
if 'Risk' in df_all.columns:
    risk_bins = [0, 5, 8, 10, 12, 15, 20, 100]
    df_all['Risk_Bin'] = pd.cut(df_all['Risk'], bins=risk_bins, labels=[f'{risk_bins[i]}-{risk_bins[i+1]}' for i in range(len(risk_bins)-1)])
    
    for risk_bin in df_all['Risk_Bin'].unique():
        if pd.notna(risk_bin):
            filtered = df_all[df_all['Risk_Bin'] == risk_bin]
            if len(filtered) >= 20:
                winrate = (filtered['P&L'] > 0).sum() / len(filtered)
                total_pnl = filtered['P&L'].sum()
                if winrate >= 0.40:
                    best_combinations.append({
                        'Type': 'Risk_Range',
                        'Filter': f'Risk_{risk_bin}',
                        'Value': str(risk_bin),
                        'Trades': len(filtered),
                        'WinRate': winrate * 100,
                        'TotalP&L': total_pnl,
                        'AvgP&L': total_pnl / len(filtered)
                    })

# Test ATR Ratio ranges
print("3. Testing ATR Ratio Ranges...")
if 'ATR_Ratio' in df_all.columns:
    atr_bins = [0, 0.8, 1.0, 1.2, 1.5, 10]
    df_all['ATR_Bin'] = pd.cut(df_all['ATR_Ratio'], bins=atr_bins, labels=[f'{atr_bins[i]}-{atr_bins[i+1]}' for i in range(len(atr_bins)-1)])
    
    for atr_bin in df_all['ATR_Bin'].unique():
        if pd.notna(atr_bin):
            filtered = df_all[df_all['ATR_Bin'] == atr_bin]
            if len(filtered) >= 20:
                winrate = (filtered['P&L'] > 0).sum() / len(filtered)
                total_pnl = filtered['P&L'].sum()
                if winrate >= 0.40:
                    best_combinations.append({
                        'Type': 'ATR_Ratio',
                        'Filter': f'ATR_{atr_bin}',
                        'Value': str(atr_bin),
                        'Trades': len(filtered),
                        'WinRate': winrate * 100,
                        'TotalP&L': total_pnl,
                        'AvgP&L': total_pnl / len(filtered)
                    })

# Test Trend Score
print("4. Testing Trend Score...")
if 'Trend_Score' in df_all.columns:
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        filtered = df_all[df_all['Trend_Score'] >= threshold]
        if len(filtered) >= 20:
            winrate = (filtered['P&L'] > 0).sum() / len(filtered)
            total_pnl = filtered['P&L'].sum()
            if winrate >= 0.40:
                best_combinations.append({
                    'Type': 'Trend_Score',
                    'Filter': f'Trend_>={threshold}',
                    'Value': threshold,
                    'Trades': len(filtered),
                    'WinRate': winrate * 100,
                    'TotalP&L': total_pnl,
                    'AvgP&L': total_pnl / len(filtered)
                })

# Test Range Size
print("5. Testing Range Size...")
if 'RangeSize' in df_all.columns:
    range_bins = [0, 3, 5, 7, 10, 100]
    df_all['Range_Bin'] = pd.cut(df_all['RangeSize'], bins=range_bins, labels=[f'{range_bins[i]}-{range_bins[i+1]}' for i in range(len(range_bins)-1)])
    
    for range_bin in df_all['Range_Bin'].unique():
        if pd.notna(range_bin):
            filtered = df_all[df_all['Range_Bin'] == range_bin]
            if len(filtered) >= 20:
                winrate = (filtered['P&L'] > 0).sum() / len(filtered)
                total_pnl = filtered['P&L'].sum()
                if winrate >= 0.40:
                    best_combinations.append({
                        'Type': 'RangeSize',
                        'Filter': f'Range_{range_bin}',
                        'Value': str(range_bin),
                        'Trades': len(filtered),
                        'WinRate': winrate * 100,
                        'TotalP&L': total_pnl,
                        'AvgP&L': total_pnl / len(filtered)
                    })

# ============================================================================
# TEST 2-FILTER COMBINATIONS
# ============================================================================
print("\n6. Testing 2-Filter Combinations...")

# EMA Pattern + Risk
if 'Risk' in df_all.columns:
    for pattern in ['110', '111']:  # Best patterns
        for risk_thresh in [10, 12, 15]:
            filtered = df_all[
                (df_all['EMA_Pattern'] == pattern) & 
                (df_all['Risk'] >= risk_thresh)
            ]
            if len(filtered) >= 10:
                winrate = (filtered['P&L'] > 0).sum() / len(filtered)
                total_pnl = filtered['P&L'].sum()
                if winrate >= 0.45:  # Higher threshold for combinations
                    best_combinations.append({
                        'Type': 'EMA_Pattern + Risk',
                        'Filter': f'Pattern_{pattern}_Risk>={risk_thresh}',
                        'Value': f'{pattern}, Risk>={risk_thresh}',
                        'Trades': len(filtered),
                        'WinRate': winrate * 100,
                        'TotalP&L': total_pnl,
                        'AvgP&L': total_pnl / len(filtered)
                    })

# EMA Pattern + Trend Score
if 'Trend_Score' in df_all.columns:
    for pattern in ['110', '111']:
        for trend_thresh in [0.6, 0.7, 0.8]:
            filtered = df_all[
                (df_all['EMA_Pattern'] == pattern) & 
                (df_all['Trend_Score'] >= trend_thresh)
            ]
            if len(filtered) >= 10:
                winrate = (filtered['P&L'] > 0).sum() / len(filtered)
                total_pnl = filtered['P&L'].sum()
                if winrate >= 0.45:
                    best_combinations.append({
                        'Type': 'EMA_Pattern + Trend',
                        'Filter': f'Pattern_{pattern}_Trend>={trend_thresh}',
                        'Value': f'{pattern}, Trend>={trend_thresh}',
                        'Trades': len(filtered),
                        'WinRate': winrate * 100,
                        'TotalP&L': total_pnl,
                        'AvgP&L': total_pnl / len(filtered)
                    })

# Risk + Trend
if all(col in df_all.columns for col in ['Risk', 'Trend_Score']):
    for risk_thresh in [10, 12, 15]:
        for trend_thresh in [0.6, 0.7]:
            filtered = df_all[
                (df_all['Risk'] >= risk_thresh) & 
                (df_all['Trend_Score'] >= trend_thresh)
            ]
            if len(filtered) >= 10:
                winrate = (filtered['P&L'] > 0).sum() / len(filtered)
                total_pnl = filtered['P&L'].sum()
                if winrate >= 0.45:
                    best_combinations.append({
                        'Type': 'Risk + Trend',
                        'Filter': f'Risk>={risk_thresh}_Trend>={trend_thresh}',
                        'Value': f'Risk>={risk_thresh}, Trend>={trend_thresh}',
                        'Trades': len(filtered),
                        'WinRate': winrate * 100,
                        'TotalP&L': total_pnl,
                        'AvgP&L': total_pnl / len(filtered)
                    })

# ============================================================================
# TEST 3-FILTER COMBINATIONS
# ============================================================================
print("\n7. Testing 3-Filter Combinations...")

# EMA Pattern + Risk + Trend
if all(col in df_all.columns for col in ['Risk', 'Trend_Score']):
    for pattern in ['110', '111']:
        for risk_thresh in [12, 15]:
            for trend_thresh in [0.6, 0.7]:
                filtered = df_all[
                    (df_all['EMA_Pattern'] == pattern) & 
                    (df_all['Risk'] >= risk_thresh) &
                    (df_all['Trend_Score'] >= trend_thresh)
                ]
                if len(filtered) >= 5:
                    winrate = (filtered['P&L'] > 0).sum() / len(filtered)
                    total_pnl = filtered['P&L'].sum()
                    if winrate >= 0.50:  # 50%+ for 3-filter combos
                        best_combinations.append({
                            'Type': 'EMA + Risk + Trend',
                            'Filter': f'Pattern_{pattern}_Risk>={risk_thresh}_Trend>={trend_thresh}',
                            'Value': f'{pattern}, Risk>={risk_thresh}, Trend>={trend_thresh}',
                            'Trades': len(filtered),
                            'WinRate': winrate * 100,
                            'TotalP&L': total_pnl,
                            'AvgP&L': total_pnl / len(filtered)
                        })

# ============================================================================
# RANK AND DISPLAY BEST COMBINATIONS
# ============================================================================
print("\n" + "="*80)
print("BEST FILTER COMBINATIONS FOUND")
print("="*80)

if len(best_combinations) > 0:
    best_df = pd.DataFrame(best_combinations)
    best_df = best_df.sort_values('WinRate', ascending=False)
    
    print("\nTop 20 Combinations:")
    print("-"*100)
    print(f"{'Rank':<5} {'Type':<25} {'Trades':<8} {'WinRate':<10} {'TotalP&L':<12} {'AvgP&L':<10} {'Filter':<30}")
    print("-"*100)
    
    for idx, row in best_df.head(20).iterrows():
        print(f"{idx+1:<5} {row['Type']:<25} {int(row['Trades']):<8} {row['WinRate']:>9.1f}% ${row['TotalP&L']:>10.2f} ${row['AvgP&L']:>9.2f} {row['Filter']:<30}")
    
    # Save
    best_df.to_csv('optimal_filter_combinations.csv', index=False)
    print(f"\n✅ Saved {len(best_df)} optimal combinations to: optimal_filter_combinations.csv")
    
    # Best overall
    best_overall = best_df.iloc[0]
    print(f"\n🏆 BEST COMBINATION:")
    print(f"   Type: {best_overall['Type']}")
    print(f"   Filter: {best_overall['Filter']}")
    print(f"   Trades: {int(best_overall['Trades'])}")
    print(f"   Win Rate: {best_overall['WinRate']:.2f}%")
    print(f"   Total P&L: ${best_overall['TotalP&L']:.2f}")
    print(f"   Avg P&L: ${best_overall['AvgP&L']:.2f}")
    
    # Test best combination on all data
    print(f"\n📊 Testing Best Combination on ALL Data:")
    if 'EMA_Pattern' in best_overall['Filter']:
        # Parse filter
        if 'Pattern_110' in best_overall['Filter']:
            pattern = '110'
        elif 'Pattern_111' in best_overall['Filter']:
            pattern = '111'
        else:
            pattern = None
        
        if pattern:
            filtered_all = df_all[df_all['EMA_Pattern'] == pattern]
            
            # Add other filters if present
            if 'Risk>=' in best_overall['Filter']:
                risk_val = float(best_overall['Filter'].split('Risk>=')[1].split('_')[0].split(',')[0])
                filtered_all = filtered_all[filtered_all['Risk'] >= risk_val]
            
            if 'Trend>=' in best_overall['Filter']:
                trend_val = float(best_overall['Filter'].split('Trend>=')[1].split(',')[0])
                filtered_all = filtered_all[filtered_all['Trend_Score'] >= trend_val]
            
            if len(filtered_all) > 0:
                all_winrate = (filtered_all['P&L'] > 0).sum() / len(filtered_all) * 100
                all_pnl = filtered_all['P&L'].sum()
                all_avg = filtered_all['P&L'].mean()
                
                print(f"   All Data: {len(filtered_all)} trades")
                print(f"   Win Rate: {all_winrate:.2f}%")
                print(f"   Total P&L: ${all_pnl:.2f}")
                print(f"   Avg P&L: ${all_avg:.2f}")
else:
    print("No optimal combinations found with 40%+ win rate")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

