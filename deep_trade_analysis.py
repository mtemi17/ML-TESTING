import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEEP TRADE ANALYSIS - FINDING REAL PATTERNS")
print("="*80)

# Load all trades
df_original = pd.read_csv('advanced_analysis_results_combined.csv')
df_new = pd.read_csv('new_data_backtest_results.csv')
df_all = pd.concat([df_original, df_new], ignore_index=True)
df_all = df_all[df_all['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

df_wins = df_all[df_all['P&L'] > 0].copy()
df_losses = df_all[df_all['P&L'] <= 0].copy()

print(f"\nTotal trades: {len(df_all)}")
print(f"Wins: {len(df_wins)} ({len(df_wins)/len(df_all)*100:.1f}%)")
print(f"Losses: {len(df_losses)} ({len(df_losses)/len(df_all)*100:.1f}%)")

# ============================================================================
# ANALYZE INDICATOR PATTERNS
# ============================================================================
print("\n" + "="*80)
print("ANALYZING INDICATOR PATTERNS - WINS vs LOSSES")
print("="*80)

indicators = [
    'EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_1H',
    'ATR', 'ATR_Ratio', 'ATR_Pct',
    'Risk', 'RangeSize', 'RangeSizePct',
    'EMA_9_Above_21', 'EMA_21_Above_50', 'Price_Above_EMA200_1H',
    'Trend_Score', 'Consolidation_Score',
    'Is_Consolidating', 'Is_Tight_Range'
]

print("\n" + "-"*100)
print(f"{'Indicator':<30} {'Wins Mean':<15} {'Losses Mean':<15} {'Difference':<15} {'Win% When High':<20}")
print("-"*100)

optimal_filters = {}

for ind in indicators:
    if ind in df_all.columns:
        win_mean = df_wins[ind].mean() if len(df_wins) > 0 else 0
        loss_mean = df_losses[ind].mean() if len(df_losses) > 0 else 0
        
        if pd.isna(win_mean): win_mean = 0
        if pd.isna(loss_mean): loss_mean = 0
        
        diff = win_mean - loss_mean
        
        # Find optimal threshold
        if len(df_all[ind].dropna()) > 0:
            # Test different thresholds
            thresholds = np.percentile(df_all[ind].dropna(), [25, 50, 75, 90])
            best_threshold = None
            best_winrate = 0
            
            for thresh in thresholds:
                if diff > 0:  # Wins have higher values
                    filtered = df_all[df_all[ind] >= thresh]
                else:  # Wins have lower values
                    filtered = df_all[df_all[ind] <= thresh]
                
                if len(filtered) > 10:  # Need at least 10 trades
                    winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
                    if winrate > best_winrate and winrate > 50:  # At least 50% win rate
                        best_winrate = winrate
                        best_threshold = thresh
            
            if best_threshold is not None:
                optimal_filters[ind] = {
                    'threshold': best_threshold,
                    'direction': '>=' if diff > 0 else '<=',
                    'winrate': best_winrate,
                    'trades': len(df_all[df_all[ind] >= best_threshold] if diff > 0 else df_all[df_all[ind] <= best_threshold])
                }
        
        # Calculate win rate when indicator is "high" (above median for wins)
        if diff > 0:
            median_val = df_all[ind].median()
            high_ind = df_all[df_all[ind] >= median_val]
        else:
            median_val = df_all[ind].median()
            high_ind = df_all[df_all[ind] <= median_val]
        
        winrate_high = (high_ind['P&L'] > 0).sum() / len(high_ind) * 100 if len(high_ind) > 0 else 0
        
        print(f"{ind:<30} {win_mean:>14.2f} {loss_mean:>14.2f} {diff:>14.2f} {winrate_high:>19.1f}%")

# ============================================================================
# EMA ALIGNMENT DEEP ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("EMA ALIGNMENT DEEP ANALYSIS")
print("="*80)

# Create detailed EMA patterns
df_all['EMA_Pattern'] = (
    df_all['EMA_9_Above_21'].astype(int).astype(str) +
    df_all['EMA_21_Above_50'].astype(int).astype(str) +
    df_all['Price_Above_EMA200_1H'].astype(int).astype(str)
)

# Calculate EMA distances
if all(col in df_all.columns for col in ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_1H', 'EntryPrice']):
    df_all['EMA9_21_Distance'] = (df_all['EMA_9_5M'] - df_all['EMA_21_5M']) / df_all['EMA_21_5M'] * 100
    df_all['EMA21_50_Distance'] = (df_all['EMA_21_5M'] - df_all['EMA_50_5M']) / df_all['EMA_50_5M'] * 100
    df_all['Price_EMA200_Distance'] = (df_all['EntryPrice'] - df_all['EMA_200_1H']) / df_all['EMA_200_1H'] * 100

pattern_analysis = df_all.groupby('EMA_Pattern').agg({
    'P&L': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean(), 'mean', 'sum']
}).round(2)
pattern_analysis.columns = ['Total', 'Wins', 'WinRate', 'AvgP&L', 'TotalP&L']
pattern_analysis = pattern_analysis.sort_values('WinRate', ascending=False)

print("\nEMA Patterns (Best to Worst):")
print(pattern_analysis.to_string())

# Find best patterns
best_patterns = pattern_analysis[pattern_analysis['WinRate'] >= 0.50]  # 50%+ win rate
print(f"\n✅ Best Patterns (50%+ win rate): {len(best_patterns)}")
for pattern, row in best_patterns.iterrows():
    print(f"   Pattern {pattern}: {row['WinRate']*100:.1f}% win rate, {int(row['Total'])} trades, ${row['TotalP&L']:.2f} P&L")

# ============================================================================
# FIND OPTIMAL FILTER COMBINATIONS
# ============================================================================
print("\n" + "="*80)
print("FINDING OPTIMAL FILTER COMBINATIONS")
print("="*80)

# Test combinations of filters
print("\nTesting filter combinations...")

best_combination = None
best_winrate = 0
best_pnl = 0

# Test EMA pattern filters
for pattern in best_patterns.index[:3]:  # Top 3 patterns
    filtered = df_all[df_all['EMA_Pattern'] == pattern]
    if len(filtered) > 20:
        winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
        pnl = filtered['P&L'].sum()
        if winrate > best_winrate:
            best_winrate = winrate
            best_pnl = pnl
            best_combination = {'type': 'EMA_Pattern', 'value': pattern, 'trades': len(filtered)}

# Test ATR filters
if 'ATR_Ratio' in df_all.columns:
    for percentile in [25, 50, 75]:
        threshold = df_all['ATR_Ratio'].quantile(percentile / 100)
        # Test both directions
        for direction in ['low', 'high']:
            if direction == 'low':
                filtered = df_all[df_all['ATR_Ratio'] <= threshold]
            else:
                filtered = df_all[df_all['ATR_Ratio'] >= threshold]
            
            if len(filtered) > 20:
                winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
                pnl = filtered['P&L'].sum()
                if winrate > best_winrate:
                    best_winrate = winrate
                    best_pnl = pnl
                    best_combination = {'type': 'ATR_Ratio', 'value': threshold, 'direction': direction, 'trades': len(filtered)}

# Test Risk filters
if 'Risk' in df_all.columns:
    for percentile in [25, 50, 75]:
        threshold = df_all['Risk'].quantile(percentile / 100)
        filtered = df_all[df_all['Risk'] >= threshold]  # Higher risk often better
        if len(filtered) > 20:
            winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
            pnl = filtered['P&L'].sum()
            if winrate > best_winrate:
                best_winrate = winrate
                best_pnl = pnl
                best_combination = {'type': 'Risk', 'value': threshold, 'trades': len(filtered)}

# Test Trend Score
if 'Trend_Score' in df_all.columns:
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        filtered = df_all[df_all['Trend_Score'] >= threshold]
        if len(filtered) > 20:
            winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
            pnl = filtered['P&L'].sum()
            if winrate > best_winrate:
                best_winrate = winrate
                best_pnl = pnl
                best_combination = {'type': 'Trend_Score', 'value': threshold, 'trades': len(filtered)}

print(f"\n✅ Best Single Filter:")
if best_combination:
    print(f"   Type: {best_combination['type']}")
    print(f"   Value: {best_combination.get('value', 'N/A')}")
    print(f"   Trades: {best_combination['trades']}")
    print(f"   Win Rate: {best_winrate:.2f}%")
    print(f"   Total P&L: ${best_pnl:.2f}")

# Test combinations
print("\nTesting 2-filter combinations...")
best_combo_2 = None
best_winrate_2 = 0

# EMA Pattern + ATR
if 'ATR_Ratio' in df_all.columns:
    for pattern in best_patterns.index[:2]:
        for atr_thresh in [df_all['ATR_Ratio'].quantile(0.25), df_all['ATR_Ratio'].quantile(0.75)]:
            filtered = df_all[
                (df_all['EMA_Pattern'] == pattern) & 
                (df_all['ATR_Ratio'] >= atr_thresh)
            ]
            if len(filtered) > 10:
                winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
                pnl = filtered['P&L'].sum()
                if winrate > best_winrate_2:
                    best_winrate_2 = winrate
                    best_combo_2 = {
                        'filters': ['EMA_Pattern', 'ATR_Ratio'],
                        'values': [pattern, atr_thresh],
                        'trades': len(filtered),
                        'winrate': winrate,
                        'pnl': pnl
                    }

if best_combo_2:
    print(f"   Best 2-filter combo: {best_combo_2['winrate']:.2f}% win rate, {best_combo_2['trades']} trades, ${best_combo_2['pnl']:.2f} P&L")

# Save optimal filters
optimal_filters_df = pd.DataFrame([best_combination] if best_combination else [])
if len(optimal_filters_df) > 0:
    optimal_filters_df.to_csv('optimal_filters.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

