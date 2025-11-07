import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LOSING TRADES ANALYSIS - FINDING COMMON PATTERNS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading all trades...")

# Load combined data
df_original = pd.read_csv('advanced_analysis_results_combined.csv')
df_new = pd.read_csv('new_data_backtest_results.csv')

df_all = pd.concat([df_original, df_new], ignore_index=True)
df_all = df_all[df_all['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

print(f"   Total trades: {len(df_all)}")
print(f"   Winning trades: {(df_all['P&L'] > 0).sum()}")
print(f"   Losing trades: {(df_all['P&L'] <= 0).sum()}")

# Separate winning and losing
df_wins = df_all[df_all['P&L'] > 0].copy()
df_losses = df_all[df_all['P&L'] <= 0].copy()

print(f"\n   Win rate: {len(df_wins) / len(df_all) * 100:.2f}%")

# ============================================================================
# 2. ANALYZE KEY FEATURES - LOSING VS WINNING
# ============================================================================
print("\n" + "="*80)
print("2. FEATURE COMPARISON: LOSING vs WINNING TRADES")
print("="*80)

# Key features to analyze
key_features = [
    'Risk', 'ATR_Ratio', 'ATR_Pct', 'EMA_200_1H', 'RangeSizePct',
    'EMA_9_Above_21', 'EMA_21_Above_50', 'Price_Above_EMA200_1H',
    'Is_Consolidating', 'Is_Tight_Range', 'Consolidation_Score', 'Trend_Score',
    'EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'ATR', 'RangeSize'
]

print("\n" + "-"*80)
print(f"{'Feature':<25} {'Wins Mean':<15} {'Losses Mean':<15} {'Difference':<15} {'Avoid If':<20}")
print("-"*80)

avoidance_rules = []

for feature in key_features:
    if feature in df_all.columns:
        win_mean = df_wins[feature].mean() if len(df_wins) > 0 else 0
        loss_mean = df_losses[feature].mean() if len(df_losses) > 0 else 0
        
        # Handle NaN
        if pd.isna(win_mean):
            win_mean = 0
        if pd.isna(loss_mean):
            loss_mean = 0
        
        diff = loss_mean - win_mean
        diff_pct = (diff / win_mean * 100) if win_mean != 0 else 0
        
        # Determine avoidance rule
        avoid_rule = ""
        if abs(diff_pct) > 10:  # Significant difference
            if diff > 0:
                avoid_rule = f"{feature} > {loss_mean:.2f}"
            else:
                avoid_rule = f"{feature} < {loss_mean:.2f}"
            avoidance_rules.append({
                'Feature': feature,
                'Win_Mean': win_mean,
                'Loss_Mean': loss_mean,
                'Difference': diff,
                'Diff_Pct': diff_pct,
                'Avoid_Rule': avoid_rule
            })
        
        print(f"{feature:<25} {win_mean:>14.2f} {loss_mean:>14.2f} {diff:>14.2f} {avoid_rule:<20}")

# ============================================================================
# 3. STATISTICAL SIGNIFICANCE TEST
# ============================================================================
print("\n" + "="*80)
print("3. STATISTICAL SIGNIFICANCE (Top Differences)")
print("="*80)

# Sort by absolute difference percentage
avoidance_df = pd.DataFrame(avoidance_rules)
if len(avoidance_df) > 0:
    avoidance_df = avoidance_df.sort_values('Diff_Pct', key=abs, ascending=False)
    
    print("\nTop 10 Features to Avoid:")
    print("-"*80)
    for idx, row in avoidance_df.head(10).iterrows():
        print(f"{idx+1:2d}. {row['Feature']:<25} | Loss Mean: {row['Loss_Mean']:>8.2f} | Win Mean: {row['Win_Mean']:>8.2f} | Diff: {row['Diff_Pct']:>6.1f}%")
        print(f"    → Avoid: {row['Avoid_Rule']}")

# ============================================================================
# 4. COMBINATION PATTERNS
# ============================================================================
print("\n" + "="*80)
print("4. COMBINATION PATTERNS IN LOSING TRADES")
print("="*80)

# Analyze combinations
print("\n4.1 EMA Alignment Patterns:")
print("-"*80)

# EMA alignment combinations
ema_combinations = []
for idx, trade in df_all.iterrows():
    ema9_above_21 = trade.get('EMA_9_Above_21', 0)
    ema21_above_50 = trade.get('EMA_21_Above_50', 0)
    price_above_200 = trade.get('Price_Above_EMA200_1H', 0)
    
    alignment = f"{int(ema9_above_21)}{int(ema21_above_50)}{int(price_above_200)}"
    is_win = 1 if trade['P&L'] > 0 else 0
    
    ema_combinations.append({
        'Alignment': alignment,
        'IsWin': is_win,
        'P&L': trade['P&L']
    })

ema_df = pd.DataFrame(ema_combinations)
ema_summary = ema_df.groupby('Alignment').agg({
    'IsWin': ['count', 'sum', 'mean'],
    'P&L': 'mean'
}).round(3)

ema_summary.columns = ['Total', 'Wins', 'WinRate', 'AvgP&L']
ema_summary = ema_summary.sort_values('WinRate')

print("\nEMA Alignment Patterns (Worst to Best):")
print(ema_summary.to_string())

# Worst patterns
worst_patterns = ema_summary.head(3)
print("\n⚠️  WORST EMA Patterns (Avoid These):")
for pattern, row in worst_patterns.iterrows():
    print(f"   Pattern {pattern}: Win Rate {row['WinRate']*100:.1f}%, Avg P&L ${row['AvgP&L']:.2f}, Count: {int(row['Total'])}")

# ============================================================================
# 5. RISK-BASED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. RISK-BASED ANALYSIS")
print("="*80)

# Risk ranges
df_all['Risk_Range'] = pd.cut(df_all['Risk'], bins=[0, 5, 10, 15, 20, 100], labels=['0-5', '5-10', '10-15', '15-20', '20+'])

risk_analysis = df_all.groupby('Risk_Range').agg({
    'P&L': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean(), 'mean', 'sum']
}).round(2)

risk_analysis.columns = ['Total', 'Wins', 'WinRate', 'AvgP&L', 'TotalP&L']
risk_analysis = risk_analysis.sort_values('WinRate')

print("\nRisk Range Analysis (Worst to Best):")
print(risk_analysis.to_string())

# ============================================================================
# 6. CONSOLIDATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. CONSOLIDATION ANALYSIS")
print("="*80)

if 'Is_Consolidating' in df_all.columns:
    consolidation_analysis = df_all.groupby('Is_Consolidating').agg({
        'P&L': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean(), 'mean']
    }).round(2)
    
    consolidation_analysis.columns = ['Total', 'Wins', 'WinRate', 'AvgP&L']
    
    print("\nConsolidation vs Non-Consolidation:")
    print(consolidation_analysis.to_string())
    
    if len(consolidation_analysis) > 1:
        consolidating_winrate = consolidation_analysis.loc[1, 'WinRate'] if 1 in consolidation_analysis.index else 0
        non_consolidating_winrate = consolidation_analysis.loc[0, 'WinRate'] if 0 in consolidation_analysis.index else 0
        
        if consolidating_winrate < non_consolidating_winrate:
            print(f"\n⚠️  AVOID: Trading during consolidation (Win Rate: {consolidating_winrate*100:.1f}% vs {non_consolidating_winrate*100:.1f}%)")

# ============================================================================
# 7. ATR RATIO ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("7. ATR RATIO ANALYSIS (Volatility)")
print("="*80)

if 'ATR_Ratio' in df_all.columns:
    df_all['ATR_Range'] = pd.cut(df_all['ATR_Ratio'], bins=[0, 0.8, 1.0, 1.2, 1.5, 10], labels=['<0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5', '>1.5'])
    
    atr_analysis = df_all.groupby('ATR_Range').agg({
        'P&L': ['count', lambda x: (x > 0).sum(), lambda x: (x > 0).mean(), 'mean']
    }).round(2)
    
    atr_analysis.columns = ['Total', 'Wins', 'WinRate', 'AvgP&L']
    atr_analysis = atr_analysis.sort_values('WinRate')
    
    print("\nATR Ratio Ranges (Worst to Best):")
    print(atr_analysis.to_string())
    
    worst_atr = atr_analysis.head(1)
    if len(worst_atr) > 0:
        worst_range = worst_atr.index[0]
        worst_winrate = worst_atr['WinRate'].iloc[0]
        print(f"\n⚠️  AVOID: ATR_Ratio in range '{worst_range}' (Win Rate: {worst_winrate*100:.1f}%)")

# ============================================================================
# 8. CREATE AVOIDANCE RULES
# ============================================================================
print("\n" + "="*80)
print("8. AVOIDANCE RULES SUMMARY")
print("="*80)

avoidance_rules_list = []

# From feature analysis
if len(avoidance_df) > 0:
    top_avoid = avoidance_df.head(5)
    for idx, row in top_avoid.iterrows():
        avoidance_rules_list.append({
            'Rule': f"Avoid when {row['Feature']} is in losing range",
            'Details': row['Avoid_Rule'],
            'Impact': f"{abs(row['Diff_Pct']):.1f}% difference"
        })

# From EMA patterns
if len(worst_patterns) > 0:
    for pattern, row in worst_patterns.iterrows():
        pattern_desc = f"EMA9_Above_21={pattern[0]}, EMA21_Above_50={pattern[1]}, Price_Above_200={pattern[2]}"
        avoidance_rules_list.append({
            'Rule': f"Avoid EMA alignment pattern: {pattern_desc}",
            'Details': f"Win Rate: {row['WinRate']*100:.1f}%",
            'Impact': f"Only {row['WinRate']*100:.1f}% win rate"
        })

# From risk analysis
worst_risk = risk_analysis.head(1)
if len(worst_risk) > 0:
    worst_risk_range = worst_risk.index[0]
    worst_risk_winrate = worst_risk['WinRate'].iloc[0]
    avoidance_rules_list.append({
        'Rule': f"Avoid Risk range: {worst_risk_range}",
        'Details': f"Win Rate: {worst_risk_winrate*100:.1f}%",
        'Impact': f"Low win rate in this risk range"
    })

print("\nTop Avoidance Rules:")
print("-"*80)
for i, rule in enumerate(avoidance_rules_list[:10], 1):
    print(f"{i:2d}. {rule['Rule']}")
    print(f"    Details: {rule['Details']}")
    print(f"    Impact: {rule['Impact']}")
    print()

# Save rules
avoidance_rules_df = pd.DataFrame(avoidance_rules_list)
avoidance_rules_df.to_csv('avoidance_rules.csv', index=False)
print("   Rules saved to: avoidance_rules.csv")

# ============================================================================
# 9. TEST AVOIDANCE RULES
# ============================================================================
print("\n" + "="*80)
print("9. TESTING AVOIDANCE RULES")
print("="*80)

# Apply top avoidance rules
df_filtered = df_all.copy()

# Rule 1: Avoid worst EMA patterns
if len(worst_patterns) > 0:
    worst_pattern_list = worst_patterns.index.tolist()[:2]  # Top 2 worst
    
    for idx, trade in df_all.iterrows():
        ema9_above_21 = int(trade.get('EMA_9_Above_21', 0))
        ema21_above_50 = int(trade.get('EMA_21_Above_50', 0))
        price_above_200 = int(trade.get('Price_Above_EMA200_1H', 0))
        pattern = f"{ema9_above_21}{ema21_above_50}{price_above_200}"
        
        if pattern in worst_pattern_list:
            df_filtered.loc[idx, 'ShouldAvoid'] = True
        else:
            df_filtered.loc[idx, 'ShouldAvoid'] = False

# Rule 2: Avoid high risk
if 'Risk' in df_all.columns:
    high_risk_threshold = df_losses['Risk'].quantile(0.75)  # Top 25% of losing trades' risk
    df_filtered['HighRisk'] = df_filtered['Risk'] > high_risk_threshold

# Rule 3: Avoid consolidation if it's bad
if 'Is_Consolidating' in df_all.columns and len(consolidation_analysis) > 1:
    consolidating_winrate = consolidation_analysis.loc[1, 'WinRate'] if 1 in consolidation_analysis.index else 0.5
    if consolidating_winrate < 0.35:  # If consolidating has < 35% win rate
        df_filtered['AvoidConsolidation'] = df_filtered['Is_Consolidating'] == 1
    else:
        df_filtered['AvoidConsolidation'] = False

# Combine rules
df_filtered['AvoidTrade'] = (
    df_filtered.get('ShouldAvoid', False) |
    df_filtered.get('HighRisk', False) |
    df_filtered.get('AvoidConsolidation', False)
)

# Test filtered performance
df_after_avoidance = df_filtered[~df_filtered['AvoidTrade']].copy()

if len(df_after_avoidance) > 0:
    avoided_count = df_filtered['AvoidTrade'].sum()
    remaining_trades = len(df_after_avoidance)
    remaining_winrate = (df_after_avoidance['P&L'] > 0).sum() / len(df_after_avoidance) * 100
    remaining_pnl = df_after_avoidance['P&L'].sum()
    
    print(f"\nAfter Applying Avoidance Rules:")
    print(f"   Original trades: {len(df_all)}")
    print(f"   Avoided trades: {avoided_count} ({avoided_count/len(df_all)*100:.1f}%)")
    print(f"   Remaining trades: {remaining_trades} ({remaining_trades/len(df_all)*100:.1f}%)")
    print(f"   Remaining win rate: {remaining_winrate:.2f}%")
    print(f"   Remaining total P&L: ${remaining_pnl:.2f}")
    print(f"   Original total P&L: ${df_all['P&L'].sum():.2f}")
    print(f"   Improvement: ${remaining_pnl - df_all['P&L'].sum():.2f} ({(remaining_pnl - df_all['P&L'].sum())/abs(df_all['P&L'].sum())*100:.1f}%)")

# Save filtered results
df_after_avoidance.to_csv('trades_after_avoidance.csv', index=False)
print("\n   Filtered trades saved to: trades_after_avoidance.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

