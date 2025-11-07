import pandas as pd
import numpy as np
import joblib
from predict_trade import create_trade_features_from_entry
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)
print("\nComparing: Baseline Strategy vs Random Forest ML vs Deep Learning")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading test data...")
df = pd.read_csv('backtest_results_combined.csv')
completed = df[df['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
completed['EntryTime'] = pd.to_datetime(completed['EntryTime'])

print(f"   Total completed trades: {len(completed)}")

# ============================================================================
# 2. BASELINE STRATEGY (No ML Filter)
# ============================================================================
print("\n2. BASELINE STRATEGY (No ML Filter)")
print("-"*80)

baseline_wins = (completed['P&L'] > 0).sum()
baseline_win_rate = baseline_wins / len(completed) * 100
baseline_total_pnl = completed['P&L'].sum()
baseline_avg_pnl = completed['P&L'].mean()

print(f"   Total trades: {len(completed)}")
print(f"   Winning trades: {baseline_wins} ({baseline_win_rate:.2f}%)")
print(f"   Total P&L: ${baseline_total_pnl:.2f}")
print(f"   Average P&L: ${baseline_avg_pnl:.2f}")

baseline_results = {
    'Model': 'Baseline (No Filter)',
    'Trades': len(completed),
    'WinRate': baseline_win_rate,
    'TotalP&L': baseline_total_pnl,
    'AvgP&L': baseline_avg_pnl,
    'TradesTaken': len(completed),
    'TradesTakenPct': 100.0
}

# ============================================================================
# 3. RANDOM FOREST ML MODEL
# ============================================================================
print("\n3. RANDOM FOREST ML MODEL")
print("-"*80)

try:
    rf_classifier = joblib.load('best_classifier_model.pkl')
    rf_scaler = joblib.load('feature_scaler.pkl')
    rf_feature_cols = pd.read_csv('feature_columns.csv', header=None)[0].tolist()
    
    predictions_rf = []
    for idx, trade in completed.iterrows():
        entry_data = {
            'Type': trade['Type'],
            'Risk': trade.get('Risk', 0),
            'WindowType': str(trade.get('WindowType', '0')),
            'EMA_9_5M': trade.get('EMA_9_5M', 0),
            'EMA_21_5M': trade.get('EMA_21_5M', 0),
            'EMA_50_5M': trade.get('EMA_50_5M', 0),
            'EMA_200_1H': trade.get('EMA_200_1H', 0),
            'ATR': trade.get('ATR', 0),
            'ATR_Pct': trade.get('ATR_Pct', 0),
            'ATR_Ratio': trade.get('ATR_Ratio', 0),
            'Is_Consolidating': trade.get('Is_Consolidating', 0),
            'Is_Tight_Range': trade.get('Is_Tight_Range', 0),
            'Consolidation_Score': trade.get('Consolidation_Score', 0),
            'Trend_Score': trade.get('Trend_Score', 0),
            'EMA_9_Above_21': trade.get('EMA_9_Above_21', 0),
            'EMA_21_Above_50': trade.get('EMA_21_Above_50', 0),
            'Price_Above_EMA200_1H': trade.get('Price_Above_EMA200_1H', 0),
            'RangeSize': trade.get('WindowHigh', 0) - trade.get('WindowLow', 0),
            'RangeSizePct': ((trade.get('WindowHigh', 0) - trade.get('WindowLow', 0)) / trade['EntryPrice'] * 100) if trade['EntryPrice'] > 0 else 0,
            'EntryTime': trade['EntryTime']
        }
        
        features = create_trade_features_from_entry(entry_data)
        
        # Prepare feature vector
        feature_vector = []
        for col in rf_feature_cols:
            if col == '0' or col == 0:
                continue
            if col in features:
                val = features[col]
                if pd.isna(val):
                    val = 0
                feature_vector.append(float(val))
            else:
                feature_vector.append(0.0)
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        pred = rf_classifier.predict(feature_array)[0]
        predictions_rf.append(bool(pred))
    
    completed['RF_Predicted'] = predictions_rf
    rf_filtered = completed[completed['RF_Predicted'] == True]
    
    if len(rf_filtered) > 0:
        rf_wins = (rf_filtered['P&L'] > 0).sum()
        rf_win_rate = rf_wins / len(rf_filtered) * 100
        rf_total_pnl = rf_filtered['P&L'].sum()
        rf_avg_pnl = rf_filtered['P&L'].mean()
        
        print(f"   Total trades taken: {len(rf_filtered)} ({len(rf_filtered)/len(completed)*100:.1f}%)")
        print(f"   Winning trades: {rf_wins} ({rf_win_rate:.2f}%)")
        print(f"   Total P&L: ${rf_total_pnl:.2f}")
        print(f"   Average P&L: ${rf_avg_pnl:.2f}")
        
        rf_results = {
            'Model': 'Random Forest ML',
            'Trades': len(rf_filtered),
            'WinRate': rf_win_rate,
            'TotalP&L': rf_total_pnl,
            'AvgP&L': rf_avg_pnl,
            'TradesTaken': len(rf_filtered),
            'TradesTakenPct': len(rf_filtered)/len(completed)*100,
            'P&LImprovement': rf_total_pnl - baseline_total_pnl,
            'P&LImprovementPct': ((rf_total_pnl - baseline_total_pnl) / abs(baseline_total_pnl)) * 100
        }
    else:
        print("   No trades predicted as wins")
        rf_results = {
            'Model': 'Random Forest ML',
            'Trades': 0,
            'WinRate': 0,
            'TotalP&L': 0,
            'AvgP&L': 0,
            'TradesTaken': 0,
            'TradesTakenPct': 0
        }
        
except Exception as e:
    print(f"   ERROR loading Random Forest model: {e}")
    rf_results = {
        'Model': 'Random Forest ML',
        'Trades': 0,
        'WinRate': 0,
        'TotalP&L': 0,
        'AvgP&L': 0,
        'TradesTaken': 0,
        'TradesTakenPct': 0
    }

# ============================================================================
# 4. DEEP LEARNING MODEL
# ============================================================================
print("\n4. DEEP LEARNING MODEL")
print("-"*80)

try:
    dl_classifier = joblib.load('deep_learning_classifier.pkl')
    dl_scaler = joblib.load('dl_feature_scaler.pkl')
    dl_feature_cols = pd.read_csv('dl_feature_columns.csv', header=None)[0].tolist()
    
    predictions_dl = []
    for idx, trade in completed.iterrows():
        entry_data = {
            'Type': trade['Type'],
            'Risk': trade.get('Risk', 0),
            'WindowType': str(trade.get('WindowType', '0')),
            'EMA_9_5M': trade.get('EMA_9_5M', 0),
            'EMA_21_5M': trade.get('EMA_21_5M', 0),
            'EMA_50_5M': trade.get('EMA_50_5M', 0),
            'EMA_200_1H': trade.get('EMA_200_1H', 0),
            'ATR': trade.get('ATR', 0),
            'ATR_Pct': trade.get('ATR_Pct', 0),
            'ATR_Ratio': trade.get('ATR_Ratio', 0),
            'Is_Consolidating': trade.get('Is_Consolidating', 0),
            'Is_Tight_Range': trade.get('Is_Tight_Range', 0),
            'Consolidation_Score': trade.get('Consolidation_Score', 0),
            'Trend_Score': trade.get('Trend_Score', 0),
            'EMA_9_Above_21': trade.get('EMA_9_Above_21', 0),
            'EMA_21_Above_50': trade.get('EMA_21_Above_50', 0),
            'Price_Above_EMA200_1H': trade.get('Price_Above_EMA200_1H', 0),
            'RangeSize': trade.get('WindowHigh', 0) - trade.get('WindowLow', 0),
            'RangeSizePct': ((trade.get('WindowHigh', 0) - trade.get('WindowLow', 0)) / trade['EntryPrice'] * 100) if trade['EntryPrice'] > 0 else 0,
            'EntryTime': trade['EntryTime']
        }
        
        features = create_trade_features_from_entry(entry_data)
        
        # Prepare feature vector - skip '0' placeholder like RF model
        feature_vector = []
        for col in dl_feature_cols:
            if col == '0' or col == 0:
                continue  # Skip placeholder
            if col in features:
                val = features[col]
                if pd.isna(val):
                    val = 0
                feature_vector.append(float(val))
            else:
                feature_vector.append(0.0)
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        feature_array_scaled = dl_scaler.transform(feature_array)
        pred = dl_classifier.predict(feature_array_scaled)[0]
        predictions_dl.append(bool(pred))
    
    completed['DL_Predicted'] = predictions_dl
    dl_filtered = completed[completed['DL_Predicted'] == True]
    
    if len(dl_filtered) > 0:
        dl_wins = (dl_filtered['P&L'] > 0).sum()
        dl_win_rate = dl_wins / len(dl_filtered) * 100
        dl_total_pnl = dl_filtered['P&L'].sum()
        dl_avg_pnl = dl_filtered['P&L'].mean()
        
        print(f"   Total trades taken: {len(dl_filtered)} ({len(dl_filtered)/len(completed)*100:.1f}%)")
        print(f"   Winning trades: {dl_wins} ({dl_win_rate:.2f}%)")
        print(f"   Total P&L: ${dl_total_pnl:.2f}")
        print(f"   Average P&L: ${dl_avg_pnl:.2f}")
        
        dl_results = {
            'Model': 'Deep Learning',
            'Trades': len(dl_filtered),
            'WinRate': dl_win_rate,
            'TotalP&L': dl_total_pnl,
            'AvgP&L': dl_avg_pnl,
            'TradesTaken': len(dl_filtered),
            'TradesTakenPct': len(dl_filtered)/len(completed)*100,
            'P&LImprovement': dl_total_pnl - baseline_total_pnl,
            'P&LImprovementPct': ((dl_total_pnl - baseline_total_pnl) / abs(baseline_total_pnl)) * 100
        }
    else:
        print("   No trades predicted as wins")
        dl_results = {
            'Model': 'Deep Learning',
            'Trades': 0,
            'WinRate': 0,
            'TotalP&L': 0,
            'AvgP&L': 0,
            'TradesTaken': 0,
            'TradesTakenPct': 0,
            'P&LImprovement': -baseline_total_pnl,
            'P&LImprovementPct': -100.0
        }
        
except Exception as e:
    print(f"   ERROR loading Deep Learning model: {e}")
    dl_results = {
        'Model': 'Deep Learning',
        'Trades': 0,
        'WinRate': 0,
        'TotalP&L': 0,
        'AvgP&L': 0,
        'TradesTaken': 0,
        'TradesTakenPct': 0,
        'P&LImprovement': -baseline_total_pnl,
        'P&LImprovementPct': -100.0
    }

# ============================================================================
# 5. COMPREHENSIVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("5. COMPREHENSIVE COMPARISON TABLE")
print("="*80)

comparison_df = pd.DataFrame([baseline_results, rf_results, dl_results])

# Calculate improvements
comparison_df['WinRateImprovement'] = comparison_df['WinRate'] - baseline_win_rate
comparison_df['P&LImprovement'] = comparison_df['TotalP&L'] - baseline_total_pnl
comparison_df['P&LImprovementPct'] = (comparison_df['P&LImprovement'] / abs(baseline_total_pnl)) * 100

print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# 6. DETAILED METRICS
# ============================================================================
print("\n" + "="*80)
print("6. DETAILED METRICS")
print("="*80)

print("\nWin Rate Comparison:")
print(f"  Baseline:        {baseline_win_rate:.2f}%")
print(f"  Random Forest:   {rf_results['WinRate']:.2f}% ({rf_results['WinRate'] - baseline_win_rate:+.2f}%)")
print(f"  Deep Learning:   {dl_results['WinRate']:.2f}% ({dl_results['WinRate'] - baseline_win_rate:+.2f}%)")

print("\nTotal P&L Comparison:")
print(f"  Baseline:        ${baseline_total_pnl:.2f}")
print(f"  Random Forest:   ${rf_results['TotalP&L']:.2f} ({rf_results['P&LImprovement']:+.2f}, {rf_results['P&LImprovementPct']:+.1f}%)")
print(f"  Deep Learning:   ${dl_results['TotalP&L']:.2f} ({dl_results['P&LImprovement']:+.2f}, {dl_results['P&LImprovementPct']:+.1f}%)")

print("\nTrade Selection:")
print(f"  Baseline:        {baseline_results['TradesTaken']} trades (100.0%)")
print(f"  Random Forest:   {rf_results['TradesTaken']} trades ({rf_results['TradesTakenPct']:.1f}%)")
print(f"  Deep Learning:   {dl_results['TradesTaken']} trades ({dl_results['TradesTakenPct']:.1f}%)")

# ============================================================================
# 7. WINNER DETERMINATION
# ============================================================================
print("\n" + "="*80)
print("7. WINNER ANALYSIS")
print("="*80)

# Best by win rate
best_win_rate = comparison_df.loc[comparison_df['WinRate'].idxmax()]
print(f"\nBest Win Rate: {best_win_rate['Model']} ({best_win_rate['WinRate']:.2f}%)")

# Best by total P&L
best_pnl = comparison_df.loc[comparison_df['TotalP&L'].idxmax()]
print(f"Best Total P&L: {best_pnl['Model']} (${best_pnl['TotalP&L']:.2f})")

# Best by average P&L
best_avg_pnl = comparison_df.loc[comparison_df['AvgP&L'].idxmax()]
print(f"Best Avg P&L: {best_avg_pnl['Model']} (${best_avg_pnl['AvgP&L']:.2f})")

# Overall winner (weighted score)
comparison_df['Score'] = (
    (comparison_df['WinRate'] / 100) * 0.4 +
    (comparison_df['TotalP&L'] / abs(baseline_total_pnl)) * 0.4 +
    (comparison_df['AvgP&L'] / abs(baseline_avg_pnl)) * 0.2
)
best_overall = comparison_df.loc[comparison_df['Score'].idxmax()]
print(f"\nOverall Best Model: {best_overall['Model']} (Score: {best_overall['Score']:.3f})")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("8. SAVING COMPARISON RESULTS")
print("="*80)

comparison_df.to_csv('model_comparison_results.csv', index=False)
print("   Comparison results saved to: model_comparison_results.csv")

# Save detailed predictions
if 'RF_Predicted' in completed.columns and 'DL_Predicted' in completed.columns:
    completed.to_csv('all_model_predictions.csv', index=False)
    print("   Detailed predictions saved to: all_model_predictions.csv")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)

