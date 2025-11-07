import pandas as pd
import numpy as np
import joblib
from predict_trade import create_trade_features_from_entry
import sys
import warnings
warnings.filterwarnings('ignore')

# Import strategy class
sys.path.append('.')
from strategy_backtest import TradingStrategy

print("="*80)
print("USDJPY5 BACKTEST - TESTING ALL THREE MODELS")
print("="*80)

# ============================================================================
# 1. RUN STRATEGY BACKTEST ON USDJPY5
# ============================================================================
print("\n1. Running strategy backtest on USDJPY5 data...")

strategy = TradingStrategy('USDJPY5.csv')

# Load data
strategy.load_data()

# Add indicators
strategy.add_indicators(ema_periods_5m=[9, 21, 50], ema_200_1h=True, atr_period=14)

# Identify key time windows
strategy.identify_key_times()

# Backtest
trades_df = strategy.backtest_strategy()

# Analyze results
strategy.analyze_results(trades_df)

# Save results
strategy.save_results(trades_df, 'usdjpy5_backtest_results.csv')

# Save enhanced data
strategy.df.to_csv('USDJPY5_with_indicators.csv')

print(f"\n   Total trades generated: {len(trades_df)}")

# ============================================================================
# 2. PREPARE DATA FOR MODEL TESTING
# ============================================================================
print("\n2. Preparing data for model testing...")

completed = trades_df[trades_df['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
completed['EntryTime'] = pd.to_datetime(completed['EntryTime'])

print(f"   Completed trades: {len(completed)}")

if len(completed) == 0:
    print("   ERROR: No completed trades to test!")
    exit(1)

# ============================================================================
# 3. BASELINE STRATEGY (NO FILTER)
# ============================================================================
print("\n" + "="*80)
print("3. BASELINE STRATEGY (No ML Filter)")
print("="*80)

baseline_wins = (completed['P&L'] > 0).sum()
baseline_win_rate = baseline_wins / len(completed) * 100
baseline_total_pnl = completed['P&L'].sum()
baseline_avg_pnl = completed['P&L'].mean()

print(f"\nTotal trades: {len(completed)}")
print(f"Winning trades: {baseline_wins} ({baseline_win_rate:.2f}%)")
print(f"Total P&L: ${baseline_total_pnl:.2f}")
print(f"Average P&L: ${baseline_avg_pnl:.2f}")

baseline_results = {
    'Model': 'Baseline (No Filter)',
    'Trades': len(completed),
    'WinRate': baseline_win_rate,
    'TotalP&L': baseline_total_pnl,
    'AvgP&L': baseline_avg_pnl
}

# ============================================================================
# 4. RANDOM FOREST ML MODEL
# ============================================================================
print("\n" + "="*80)
print("4. RANDOM FOREST ML MODEL")
print("="*80)

try:
    rf_classifier = joblib.load('best_classifier_model.pkl')
    rf_scaler = joblib.load('feature_scaler.pkl')
    rf_feature_cols = pd.read_csv('feature_columns.csv', header=None)[0].tolist()
    
    print("   Model loaded successfully")
    
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
        
        print(f"\nTotal trades taken: {len(rf_filtered)} ({len(rf_filtered)/len(completed)*100:.1f}%)")
        print(f"Winning trades: {rf_wins} ({rf_win_rate:.2f}%)")
        print(f"Total P&L: ${rf_total_pnl:.2f}")
        print(f"Average P&L: ${rf_avg_pnl:.2f}")
        
        rf_results = {
            'Model': 'Random Forest ML',
            'Trades': len(rf_filtered),
            'WinRate': rf_win_rate,
            'TotalP&L': rf_total_pnl,
            'AvgP&L': rf_avg_pnl
        }
    else:
        print("\nNo trades predicted as wins")
        rf_results = {
            'Model': 'Random Forest ML',
            'Trades': 0,
            'WinRate': 0,
            'TotalP&L': 0,
            'AvgP&L': 0
        }
        
except Exception as e:
    print(f"   ERROR: {e}")
    rf_results = {
        'Model': 'Random Forest ML',
        'Trades': 0,
        'WinRate': 0,
        'TotalP&L': 0,
        'AvgP&L': 0
    }

# ============================================================================
# 5. DEEP LEARNING MODEL
# ============================================================================
print("\n" + "="*80)
print("5. DEEP LEARNING MODEL")
print("="*80)

try:
    dl_classifier = joblib.load('deep_learning_classifier.pkl')
    dl_scaler = joblib.load('dl_feature_scaler.pkl')
    dl_feature_cols = pd.read_csv('dl_feature_columns.csv', header=None)[0].tolist()
    
    print("   Model loaded successfully")
    
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
        
        # Prepare feature vector
        feature_vector = []
        for col in dl_feature_cols:
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
        
        print(f"\nTotal trades taken: {len(dl_filtered)} ({len(dl_filtered)/len(completed)*100:.1f}%)")
        print(f"Winning trades: {dl_wins} ({dl_win_rate:.2f}%)")
        print(f"Total P&L: ${dl_total_pnl:.2f}")
        print(f"Average P&L: ${dl_avg_pnl:.2f}")
        
        dl_results = {
            'Model': 'Deep Learning',
            'Trades': len(dl_filtered),
            'WinRate': dl_win_rate,
            'TotalP&L': dl_total_pnl,
            'AvgP&L': dl_avg_pnl
        }
    else:
        print("\nNo trades predicted as wins")
        dl_results = {
            'Model': 'Deep Learning',
            'Trades': 0,
            'WinRate': 0,
            'TotalP&L': 0,
            'AvgP&L': 0
        }
        
except Exception as e:
    print(f"   ERROR: {e}")
    dl_results = {
        'Model': 'Deep Learning',
        'Trades': 0,
        'WinRate': 0,
        'TotalP&L': 0,
        'AvgP&L': 0
    }

# ============================================================================
# 6. COMPREHENSIVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("6. COMPREHENSIVE COMPARISON - USDJPY5")
print("="*80)

comparison_df = pd.DataFrame([baseline_results, rf_results, dl_results])

# Calculate improvements
comparison_df['WinRateImprovement'] = comparison_df['WinRate'] - baseline_win_rate
comparison_df['P&LImprovement'] = comparison_df['TotalP&L'] - baseline_total_pnl
comparison_df['P&LImprovementPct'] = (comparison_df['P&LImprovement'] / abs(baseline_total_pnl)) * 100 if baseline_total_pnl != 0 else 0
comparison_df['TradesTakenPct'] = (comparison_df['Trades'] / len(completed)) * 100

print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# 7. DETAILED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("7. DETAILED ANALYSIS")
print("="*80)

print("\nWin Rate Comparison:")
print(f"  Baseline:        {baseline_win_rate:.2f}%")
print(f"  Random Forest:   {rf_results['WinRate']:.2f}% ({rf_results['WinRate'] - baseline_win_rate:+.2f}%)")
print(f"  Deep Learning:   {dl_results['WinRate']:.2f}% ({dl_results['WinRate'] - baseline_win_rate:+.2f}%)")

print("\nTotal P&L Comparison:")
print(f"  Baseline:        ${baseline_total_pnl:.2f}")
print(f"  Random Forest:   ${rf_results['TotalP&L']:.2f} ({rf_results['TotalP&L'] - baseline_total_pnl:+.2f}, {((rf_results['TotalP&L'] - baseline_total_pnl) / abs(baseline_total_pnl) * 100):+.1f}%)")
print(f"  Deep Learning:   ${dl_results['TotalP&L']:.2f} ({dl_results['TotalP&L'] - baseline_total_pnl:+.2f}, {((dl_results['TotalP&L'] - baseline_total_pnl) / abs(baseline_total_pnl) * 100):+.1f}%)")

print("\nTrade Selection:")
print(f"  Baseline:        {baseline_results['Trades']} trades (100.0%)")
print(f"  Random Forest:   {rf_results['Trades']} trades ({rf_results['Trades']/len(completed)*100:.1f}%)")
print(f"  Deep Learning:   {dl_results['Trades']} trades ({dl_results['Trades']/len(completed)*100:.1f}%)")

# ============================================================================
# 8. WINNER DETERMINATION
# ============================================================================
print("\n" + "="*80)
print("8. WINNER ON USDJPY5 DATA")
print("="*80)

best_win_rate = comparison_df.loc[comparison_df['WinRate'].idxmax()]
best_pnl = comparison_df.loc[comparison_df['TotalP&L'].idxmax()]
best_avg_pnl = comparison_df.loc[comparison_df['AvgP&L'].idxmax()]

print(f"\nBest Win Rate: {best_win_rate['Model']} ({best_win_rate['WinRate']:.2f}%)")
print(f"Best Total P&L: {best_pnl['Model']} (${best_pnl['TotalP&L']:.2f})")
print(f"Best Avg P&L: {best_avg_pnl['Model']} (${best_avg_pnl['AvgP&L']:.2f})")

# Overall winner
comparison_df['Score'] = (
    (comparison_df['WinRate'] / 100) * 0.4 +
    (comparison_df['TotalP&L'] / abs(baseline_total_pnl)) * 0.4 +
    (comparison_df['AvgP&L'] / abs(baseline_avg_pnl)) * 0.2
)
best_overall = comparison_df.loc[comparison_df['Score'].idxmax()]
print(f"\nOverall Best Model: {best_overall['Model']} (Score: {best_overall['Score']:.3f})")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("9. SAVING RESULTS")
print("="*80)

comparison_df.to_csv('usdjpy5_model_comparison.csv', index=False)
print("   Comparison saved to: usdjpy5_model_comparison.csv")

if 'RF_Predicted' in completed.columns and 'DL_Predicted' in completed.columns:
    completed.to_csv('usdjpy5_all_predictions.csv', index=False)
    print("   Detailed predictions saved to: usdjpy5_all_predictions.csv")

print("\n" + "="*80)
print("USDJPY5 TESTING COMPLETE!")
print("="*80)

