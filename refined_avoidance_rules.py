import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("REFINED AVOIDANCE RULES - FOCUS ON WORST PATTERNS")
print("="*80)

# Load data
df_all = pd.read_csv('advanced_analysis_results_combined.csv')
df_new = pd.read_csv('new_data_backtest_results.csv')
df_all = pd.concat([df_all, df_new], ignore_index=True)
df_all = df_all[df_all['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

df_wins = df_all[df_all['P&L'] > 0].copy()
df_losses = df_all[df_all['P&L'] <= 0].copy()

print(f"\nTotal trades: {len(df_all)}")
print(f"Wins: {len(df_wins)}, Losses: {len(df_losses)}")

# ============================================================================
# FOCUS ON WORST PATTERNS ONLY
# ============================================================================
print("\n" + "="*80)
print("KEY FINDINGS - WORST PATTERNS TO AVOID")
print("="*80)

# 1. WORST EMA PATTERNS
print("\n1. WORST EMA ALIGNMENT PATTERNS:")
print("-"*80)

# Create EMA pattern
df_all['EMA_Pattern'] = (
    df_all['EMA_9_Above_21'].astype(int).astype(str) +
    df_all['EMA_21_Above_50'].astype(int).astype(str) +
    df_all['Price_Above_EMA200_1H'].astype(int).astype(str)
)

pattern_analysis = df_all.groupby('EMA_Pattern').agg({
    'P&L': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean(), 'mean', 'sum']
}).round(2)

pattern_analysis.columns = ['Total', 'Wins', 'WinRate', 'AvgP&L', 'TotalP&L']
pattern_analysis = pattern_analysis.sort_values('WinRate')

print("\nEMA Patterns (Worst to Best):")
print(pattern_analysis.to_string())

# Identify truly bad patterns (win rate < 30% AND negative avg P&L)
bad_patterns = pattern_analysis[(pattern_analysis['WinRate'] < 0.30) & (pattern_analysis['AvgP&L'] < 0)]
print(f"\n⚠️  CRITICAL: Avoid these EMA patterns (Win Rate < 30% AND Negative P&L):")
for pattern, row in bad_patterns.iterrows():
    pattern_desc = f"EMA9_Above_21={pattern[0]}, EMA21_Above_50={pattern[1]}, Price_Above_200={pattern[2]}"
    print(f"   Pattern {pattern} ({pattern_desc}):")
    print(f"      Win Rate: {row['WinRate']*100:.1f}%, Avg P&L: ${row['AvgP&L']:.2f}, Count: {int(row['Total'])}")

# 2. RISK ANALYSIS - Focus on worst ranges
print("\n" + "="*80)
print("2. RISK RANGE ANALYSIS:")
print("-"*80)

df_all['Risk_Range'] = pd.cut(df_all['Risk'], bins=[0, 5, 10, 15, 20, 100], labels=['0-5', '5-10', '10-15', '15-20', '20+'])
risk_analysis = df_all.groupby('Risk_Range').agg({
    'P&L': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean(), 'mean', 'sum']
}).round(2)
risk_analysis.columns = ['Total', 'Wins', 'WinRate', 'AvgP&L', 'TotalP&L']

print(risk_analysis.to_string())

# Worst risk range (low win rate AND negative total P&L)
worst_risk = risk_analysis[(risk_analysis['WinRate'] < 0.35) & (risk_analysis['TotalP&L'] < 0)]
if len(worst_risk) > 0:
    print(f"\n⚠️  AVOID: Risk range {worst_risk.index[0]} (Win Rate: {worst_risk['WinRate'].iloc[0]*100:.1f}%, Total P&L: ${worst_risk['TotalP&L'].iloc[0]:.2f})")

# 3. ATR RATIO - Worst volatility ranges
print("\n" + "="*80)
print("3. ATR RATIO (Volatility) ANALYSIS:")
print("-"*80)

if 'ATR_Ratio' in df_all.columns:
    df_all['ATR_Range'] = pd.cut(df_all['ATR_Ratio'], bins=[0, 0.8, 1.0, 1.2, 1.5, 10], labels=['<0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5', '>1.5'])
    atr_analysis = df_all.groupby('ATR_Range').agg({
        'P&L': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean(), 'mean', 'sum']
    }).round(2)
    atr_analysis.columns = ['Total', 'Wins', 'WinRate', 'AvgP&L', 'TotalP&L']
    atr_analysis = atr_analysis.sort_values('WinRate')
    
    print(atr_analysis.to_string())
    
    # Worst ATR ranges
    worst_atr = atr_analysis[(atr_analysis['WinRate'] < 0.33) & (atr_analysis['TotalP&L'] < 0)]
    if len(worst_atr) > 0:
        print(f"\n⚠️  AVOID: ATR_Ratio ranges with Win Rate < 33% AND negative P&L:")
        for atr_range, row in worst_atr.iterrows():
            print(f"   {atr_range}: Win Rate {row['WinRate']*100:.1f}%, Total P&L ${row['TotalP&L']:.2f}")

# 4. RANGE SIZE - Too small ranges
print("\n" + "="*80)
print("4. RANGE SIZE ANALYSIS:")
print("-"*80)

if 'RangeSize' in df_all.columns:
    df_all['RangeSize_Category'] = pd.cut(df_all['RangeSize'], bins=[0, 3, 5, 7, 10, 100], labels=['<3', '3-5', '5-7', '7-10', '>10'])
    range_analysis = df_all.groupby('RangeSize_Category').agg({
        'P&L': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean(), 'mean', 'sum']
    }).round(2)
    range_analysis.columns = ['Total', 'Wins', 'WinRate', 'AvgP&L', 'TotalP&L']
    
    print(range_analysis.to_string())
    
    worst_range = range_analysis[(range_analysis['WinRate'] < 0.33) & (range_analysis['TotalP&L'] < 0)]
    if len(worst_range) > 0:
        print(f"\n⚠️  AVOID: RangeSize categories with Win Rate < 33% AND negative P&L:")
        for range_cat, row in worst_range.iterrows():
            print(f"   {range_cat}: Win Rate {row['WinRate']*100:.1f}%, Total P&L ${row['TotalP&L']:.2f}")

# ============================================================================
# CREATE REFINED AVOIDANCE FILTER
# ============================================================================
print("\n" + "="*80)
print("5. APPLYING REFINED AVOIDANCE RULES")
print("="*80)

df_filtered = df_all.copy()
df_filtered['ShouldAvoid'] = False

# Rule 1: Avoid worst EMA patterns (only the truly bad ones)
if len(bad_patterns) > 0:
    worst_pattern_list = bad_patterns.index.tolist()
    df_filtered['ShouldAvoid'] = df_filtered['EMA_Pattern'].isin(worst_pattern_list)
    print(f"\nRule 1: Avoiding {len(worst_pattern_list)} worst EMA patterns")
    print(f"   Patterns: {worst_pattern_list}")

# Rule 2: Avoid worst risk range (only if it's really bad)
if len(worst_risk) > 0:
    worst_risk_range = worst_risk.index[0]
    df_filtered['ShouldAvoid'] = df_filtered['ShouldAvoid'] | (df_filtered['Risk_Range'] == worst_risk_range)
    print(f"\nRule 2: Avoiding risk range {worst_risk_range}")

# Rule 3: Avoid worst ATR ranges (only truly bad ones)
if 'ATR_Range' in df_filtered.columns and len(worst_atr) > 0:
    worst_atr_list = worst_atr.index.tolist()
    df_filtered['ShouldAvoid'] = df_filtered['ShouldAvoid'] | df_filtered['ATR_Range'].isin(worst_atr_list)
    print(f"\nRule 3: Avoiding ATR ranges: {worst_atr_list}")

# Rule 4: Avoid very small ranges (if bad)
if 'RangeSize_Category' in df_filtered.columns and len(worst_range) > 0:
    worst_range_list = worst_range.index.tolist()
    df_filtered['ShouldAvoid'] = df_filtered['ShouldAvoid'] | df_filtered['RangeSize_Category'].isin(worst_range_list)
    print(f"\nRule 4: Avoiding RangeSize categories: {worst_range_list}")

# Apply filter
df_after_avoidance = df_filtered[~df_filtered['ShouldAvoid']].copy()

# Results
avoided_count = df_filtered['ShouldAvoid'].sum()
remaining_trades = len(df_after_avoidance)
remaining_winrate = (df_after_avoidance['P&L'] > 0).sum() / len(df_after_avoidance) * 100 if len(df_after_avoidance) > 0 else 0
remaining_pnl = df_after_avoidance['P&L'].sum() if len(df_after_avoidance) > 0 else 0
original_pnl = df_all['P&L'].sum()
original_winrate = (df_all['P&L'] > 0).sum() / len(df_all) * 100

print(f"\n" + "="*80)
print("RESULTS AFTER AVOIDANCE:")
print("="*80)
print(f"Original trades: {len(df_all)}")
print(f"   Win rate: {original_winrate:.2f}%")
print(f"   Total P&L: ${original_pnl:.2f}")
print(f"\nAvoided trades: {avoided_count} ({avoided_count/len(df_all)*100:.1f}%)")
print(f"\nRemaining trades: {remaining_trades} ({remaining_trades/len(df_all)*100:.1f}%)")
print(f"   Win rate: {remaining_winrate:.2f}%")
print(f"   Total P&L: ${remaining_pnl:.2f}")
print(f"   Avg P&L: ${remaining_pnl/remaining_trades:.2f}" if remaining_trades > 0 else "   Avg P&L: $0.00")

improvement = remaining_pnl - original_pnl
improvement_pct = (improvement / abs(original_pnl) * 100) if original_pnl != 0 else 0

print(f"\nImprovement: ${improvement:.2f} ({improvement_pct:+.1f}%)")
print(f"Win Rate Change: {remaining_winrate - original_winrate:+.2f}%")

# Save
df_after_avoidance.to_csv('trades_refined_avoidance.csv', index=False)
print("\n✅ Filtered trades saved to: trades_refined_avoidance.csv")

# Create summary of avoidance rules
avoidance_summary = {
    'Rule': [],
    'Pattern': [],
    'WinRate': [],
    'AvgP&L': [],
    'Count': []
}

if len(bad_patterns) > 0:
    for pattern, row in bad_patterns.iterrows():
        avoidance_summary['Rule'].append('EMA Pattern')
        avoidance_summary['Pattern'].append(pattern)
        avoidance_summary['WinRate'].append(f"{row['WinRate']*100:.1f}%")
        avoidance_summary['AvgP&L'].append(f"${row['AvgP&L']:.2f}")
        avoidance_summary['Count'].append(int(row['Total']))

if len(worst_risk) > 0:
    for risk_range, row in worst_risk.iterrows():
        avoidance_summary['Rule'].append('Risk Range')
        avoidance_summary['Pattern'].append(risk_range)
        avoidance_summary['WinRate'].append(f"{row['WinRate']*100:.1f}%")
        avoidance_summary['AvgP&L'].append(f"${row['AvgP&L']:.2f}")
        avoidance_summary['Count'].append(int(row['Total']))

avoidance_summary_df = pd.DataFrame(avoidance_summary)
avoidance_summary_df.to_csv('refined_avoidance_rules.csv', index=False)
print("✅ Avoidance rules saved to: refined_avoidance_rules.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

