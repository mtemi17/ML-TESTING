import pandas as pd
import numpy as np
import joblib
from predict_trade import predict_trade, create_trade_features_from_entry

print("="*80)
print("TESTING ML MODEL ON ORIGINAL DATA")
print("="*80)

# Load original backtest results
print("\n1. Loading original backtest results...")
df_original = pd.read_csv('backtest_results.csv')
df_original['EntryTime'] = pd.to_datetime(df_original['EntryTime'])

# Filter completed trades only
completed = df_original[df_original['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
print(f"   Total completed trades: {len(completed)}")

# Load ML models
print("\n2. Loading ML models...")
try:
    classifier = joblib.load('best_classifier_model.pkl')
    regressor = joblib.load('best_regressor_model.pkl')
    print("   Models loaded successfully")
except FileNotFoundError as e:
    print(f"   ERROR: {e}")
    print("   Please run ml_model_training.py first!")
    exit(1)

# ============================================================================
# 3. PREDICT ON ALL TRADES
# ============================================================================
print("\n3. Making predictions on all trades...")

predictions = []
for idx, trade in completed.iterrows():
    # Create entry data dictionary
    entry_data = {
        'Type': trade['Type'],
        'Risk': trade['Risk'],
        'WindowType': str(trade['WindowType']),
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
    
    # Get prediction
    features = create_trade_features_from_entry(entry_data)
    prediction = predict_trade(features)
    
    if 'error' not in prediction:
        predictions.append({
            'TradeID': idx,
            'PredictedWin': prediction['predicted_win'],
            'WinProbability': prediction['win_probability'],
            'PredictedP&L': prediction['predicted_pnl'],
            'Confidence': prediction['confidence'],
            'Recommendation': prediction['recommendation']
        })
    else:
        print(f"   Warning: Error predicting trade {idx}: {prediction['error']}")

predictions_df = pd.DataFrame(predictions)
print(f"   Predictions made: {len(predictions_df)}")

# Merge predictions with original trades
results = completed.merge(predictions_df, left_index=True, right_on='TradeID', how='left')

# ============================================================================
# 4. ANALYZE RESULTS
# ============================================================================
print("\n" + "="*80)
print("4. RESULTS ANALYSIS")
print("="*80)

# Baseline (all trades)
baseline_total = len(results)
baseline_wins = (results['P&L'] > 0).sum()
baseline_win_rate = (baseline_wins / baseline_total) * 100
baseline_total_pnl = results['P&L'].sum()
baseline_avg_pnl = results['P&L'].mean()

print(f"\nBASELINE (All Trades):")
print(f"  Total trades: {baseline_total}")
print(f"  Winning trades: {baseline_wins} ({baseline_win_rate:.2f}%)")
print(f"  Total P&L: ${baseline_total_pnl:.2f}")
print(f"  Average P&L: ${baseline_avg_pnl:.2f}")

# Strategy 1: Only take trades where model predicts win
ml_filtered = results[results['PredictedWin'] == True].copy()
if len(ml_filtered) > 0:
    ml_wins = (ml_filtered['P&L'] > 0).sum()
    ml_win_rate = (ml_wins / len(ml_filtered)) * 100
    ml_total_pnl = ml_filtered['P&L'].sum()
    ml_avg_pnl = ml_filtered['P&L'].mean()
    
    print(f"\nML FILTERED (Predicted Wins Only):")
    print(f"  Total trades taken: {len(ml_filtered)} ({len(ml_filtered)/baseline_total*100:.1f}% of all trades)")
    print(f"  Winning trades: {ml_wins} ({ml_win_rate:.2f}%)")
    print(f"  Total P&L: ${ml_total_pnl:.2f}")
    print(f"  Average P&L: ${ml_avg_pnl:.2f}")
    
    improvement = ml_total_pnl - (baseline_total_pnl * len(ml_filtered) / baseline_total)
    improvement_pct = (improvement / abs(baseline_total_pnl)) * 100 if baseline_total_pnl != 0 else 0
    
    print(f"\n  Improvement: ${improvement:.2f} ({improvement_pct:+.1f}%)")
    print(f"  Win rate improvement: {ml_win_rate - baseline_win_rate:+.2f}%")

# Strategy 2: Only take high confidence trades (probability > 0.6)
high_confidence = results[results['WinProbability'] > 0.6].copy()
if len(high_confidence) > 0:
    hc_wins = (high_confidence['P&L'] > 0).sum()
    hc_win_rate = (hc_wins / len(high_confidence)) * 100
    hc_total_pnl = high_confidence['P&L'].sum()
    hc_avg_pnl = high_confidence['P&L'].mean()
    
    print(f"\nHIGH CONFIDENCE (Probability > 60%):")
    print(f"  Total trades taken: {len(high_confidence)} ({len(high_confidence)/baseline_total*100:.1f}% of all trades)")
    print(f"  Winning trades: {hc_wins} ({hc_win_rate:.2f}%)")
    print(f"  Total P&L: ${hc_total_pnl:.2f}")
    print(f"  Average P&L: ${hc_avg_pnl:.2f}")

# Strategy 3: Only take medium+ confidence trades (probability > 0.5)
medium_confidence = results[results['WinProbability'] > 0.5].copy()
if len(medium_confidence) > 0:
    mc_wins = (medium_confidence['P&L'] > 0).sum()
    mc_win_rate = (mc_wins / len(medium_confidence)) * 100
    mc_total_pnl = medium_confidence['P&L'].sum()
    mc_avg_pnl = medium_confidence['P&L'].mean()
    
    print(f"\nMEDIUM+ CONFIDENCE (Probability > 50%):")
    print(f"  Total trades taken: {len(medium_confidence)} ({len(medium_confidence)/baseline_total*100:.1f}% of all trades)")
    print(f"  Winning trades: {mc_wins} ({mc_win_rate:.2f}%)")
    print(f"  Total P&L: ${mc_total_pnl:.2f}")
    print(f"  Average P&L: ${mc_avg_pnl:.2f}")

# ============================================================================
# 5. CONFUSION MATRIX ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. PREDICTION ACCURACY")
print("="*80)

if len(predictions_df) > 0:
    results_with_pred = results.dropna(subset=['PredictedWin'])
    
    true_positives = ((results_with_pred['PredictedWin'] == True) & (results_with_pred['P&L'] > 0)).sum()
    false_positives = ((results_with_pred['PredictedWin'] == True) & (results_with_pred['P&L'] <= 0)).sum()
    true_negatives = ((results_with_pred['PredictedWin'] == False) & (results_with_pred['P&L'] <= 0)).sum()
    false_negatives = ((results_with_pred['PredictedWin'] == False) & (results_with_pred['P&L'] > 0)).sum()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (Predicted Win, Actually Won):  {true_positives}")
    print(f"  False Positives (Predicted Win, Actually Lost): {false_positives}")
    print(f"  True Negatives (Predicted Loss, Actually Lost): {true_negatives}")
    print(f"  False Negatives (Predicted Loss, Actually Won): {false_negatives}")
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(results_with_pred) if len(results_with_pred) > 0 else 0
    
    print(f"\n  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================================================
# 6. BREAKDOWN BY TRADE TYPE
# ============================================================================
print("\n" + "="*80)
print("6. BREAKDOWN BY TRADE TYPE")
print("="*80)

for trade_type in ['BUY', 'SELL']:
    type_trades = results[results['Type'] == trade_type]
    if len(type_trades) > 0:
        print(f"\n{trade_type} Trades:")
        print(f"  Total: {len(type_trades)}")
        print(f"  Baseline win rate: {(type_trades['P&L'] > 0).sum() / len(type_trades) * 100:.2f}%")
        print(f"  Baseline P&L: ${type_trades['P&L'].sum():.2f}")
        
        ml_type = type_trades[type_trades['PredictedWin'] == True]
        if len(ml_type) > 0:
            print(f"  ML filtered: {len(ml_type)} trades")
            print(f"  ML win rate: {(ml_type['P&L'] > 0).sum() / len(ml_type) * 100:.2f}%")
            print(f"  ML P&L: ${ml_type['P&L'].sum():.2f}")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("7. SAVING RESULTS")
print("="*80)

results.to_csv('ml_test_results.csv', index=False)
print(f"   Results saved to: ml_test_results.csv")

# Summary comparison
summary = {
    'Strategy': ['Baseline (All Trades)', 'ML Filtered (Predicted Wins)', 
                 'High Confidence (>60%)', 'Medium+ Confidence (>50%)'],
    'Trades': [
        baseline_total,
        len(ml_filtered) if len(ml_filtered) > 0 else 0,
        len(high_confidence) if len(high_confidence) > 0 else 0,
        len(medium_confidence) if len(medium_confidence) > 0 else 0
    ],
    'WinRate': [
        baseline_win_rate,
        ml_win_rate if len(ml_filtered) > 0 else 0,
        hc_win_rate if len(high_confidence) > 0 else 0,
        mc_win_rate if len(medium_confidence) > 0 else 0
    ],
    'TotalP&L': [
        baseline_total_pnl,
        ml_total_pnl if len(ml_filtered) > 0 else 0,
        hc_total_pnl if len(high_confidence) > 0 else 0,
        mc_total_pnl if len(medium_confidence) > 0 else 0
    ],
    'AvgP&L': [
        baseline_avg_pnl,
        ml_avg_pnl if len(ml_filtered) > 0 else 0,
        hc_avg_pnl if len(high_confidence) > 0 else 0,
        mc_avg_pnl if len(medium_confidence) > 0 else 0
    ]
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv('ml_strategy_comparison.csv', index=False)
print(f"   Strategy comparison saved to: ml_strategy_comparison.csv")

print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)

