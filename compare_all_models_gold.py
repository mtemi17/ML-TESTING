import pandas as pd
import numpy as np
import joblib
from predict_trade import create_trade_features_from_entry
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON - XAUUSD (GOLD)")
print("="*80)
print("\nComparing: Baseline vs Random Forest ML vs Deep Learning vs LSTM")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading XAUUSD data...")
df = pd.read_csv('backtest_results_combined.csv')
completed = df[df['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
completed['EntryTime'] = pd.to_datetime(completed['EntryTime'])

print(f"   Total completed trades: {len(completed)}")

# ============================================================================
# 2. BASELINE STRATEGY
# ============================================================================
print("\n2. BASELINE STRATEGY (No ML Filter)")
print("-"*80)

baseline_wins = (completed['P&L'] > 0).sum()
baseline_win_rate = baseline_wins / len(completed) * 100
baseline_total_pnl = completed['P&L'].sum()
baseline_avg_pnl = completed['P&L'].mean()

print(f"   Total trades: {len(completed)}")
print(f"   Win rate: {baseline_win_rate:.2f}%")
print(f"   Total P&L: ${baseline_total_pnl:.2f}")
print(f"   Avg P&L: ${baseline_avg_pnl:.2f}")

baseline_results = {
    'Model': 'Baseline',
    'Trades': len(completed),
    'WinRate': baseline_win_rate,
    'TotalP&L': baseline_total_pnl,
    'AvgP&L': baseline_avg_pnl
}

# ============================================================================
# 3. RANDOM FOREST ML MODEL
# ============================================================================
print("\n3. RANDOM FOREST ML MODEL")
print("-"*80)

try:
    rf_classifier = joblib.load('best_classifier_model.pkl')
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
        
        print(f"   Trades taken: {len(rf_filtered)} ({len(rf_filtered)/len(completed)*100:.1f}%)")
        print(f"   Win rate: {rf_win_rate:.2f}%")
        print(f"   Total P&L: ${rf_total_pnl:.2f}")
        print(f"   Avg P&L: ${rf_avg_pnl:.2f}")
        
        rf_results = {
            'Model': 'Random Forest ML',
            'Trades': len(rf_filtered),
            'WinRate': rf_win_rate,
            'TotalP&L': rf_total_pnl,
            'AvgP&L': rf_avg_pnl
        }
    else:
        rf_results = {'Model': 'Random Forest ML', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}
except Exception as e:
    print(f"   ERROR: {e}")
    rf_results = {'Model': 'Random Forest ML', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

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
        
        print(f"   Trades taken: {len(dl_filtered)} ({len(dl_filtered)/len(completed)*100:.1f}%)")
        print(f"   Win rate: {dl_win_rate:.2f}%")
        print(f"   Total P&L: ${dl_total_pnl:.2f}")
        print(f"   Avg P&L: ${dl_avg_pnl:.2f}")
        
        dl_results = {
            'Model': 'Deep Learning',
            'Trades': len(dl_filtered),
            'WinRate': dl_win_rate,
            'TotalP&L': dl_total_pnl,
            'AvgP&L': dl_avg_pnl
        }
    else:
        dl_results = {'Model': 'Deep Learning', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}
except Exception as e:
    print(f"   ERROR: {e}")
    dl_results = {'Model': 'Deep Learning', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

# ============================================================================
# 5. LSTM MODEL
# ============================================================================
print("\n5. LSTM MODEL")
print("-"*80)

try:
    import tensorflow as tf
    from tensorflow import keras
    
    lstm_model = keras.models.load_model('lstm_model_gold.h5')
    lstm_feature_cols_raw = pd.read_csv('lstm_feature_columns.csv', header=None)[0].tolist()
    # Skip '0' placeholder, get actual feature names
    lstm_feature_cols = [col for col in lstm_feature_cols_raw if col != '0' and col != 0]
    lstm_params = pd.read_csv('lstm_model_params.csv', header=None)[0].tolist()
    # Skip first value (index), second is sequence_length, third is n_features
    sequence_length = int(lstm_params[1]) if len(lstm_params) > 1 else 10
    n_features = int(lstm_params[2]) if len(lstm_params) > 2 else len(lstm_feature_cols)
    
    print(f"   Model loaded: Sequence length = {sequence_length}")
    
    # For LSTM, we need to create sequences
    # Sort by entry time
    completed_sorted = completed.sort_values('EntryTime').reset_index(drop=True)
    
    # Prepare features - create them dynamically
    from sklearn.preprocessing import StandardScaler
    
    # Build feature matrix
    X_lstm_list = []
    for idx, trade in completed_sorted.iterrows():
        feature_row = []
        for col in lstm_feature_cols:
            if col in trade.index:
                val = trade[col] if pd.notna(trade[col]) else 0
            else:
                # Calculate missing features
                if col == 'EntryHour':
                    val = trade['EntryTime'].hour if 'EntryTime' in trade else 0
                elif col == 'EntryDayOfWeek':
                    val = trade['EntryTime'].weekday() if 'EntryTime' in trade else 0
                elif col == 'RangeSize':
                    val = trade.get('WindowHigh', 0) - trade.get('WindowLow', 0)
                elif col == 'RangeSizePct':
                    val = ((trade.get('WindowHigh', 0) - trade.get('WindowLow', 0)) / trade['EntryPrice'] * 100) if trade.get('EntryPrice', 0) > 0 else 0
                else:
                    val = 0
            feature_row.append(float(val))
        X_lstm_list.append(feature_row)
    
    X_lstm = np.array(X_lstm_list)
    scaler_lstm = StandardScaler()
    X_lstm_scaled = scaler_lstm.fit_transform(X_lstm)
    
    # Create sequences and predict
    predictions_lstm = []
    for i in range(sequence_length, len(completed_sorted)):
        sequence = X_lstm_scaled[i-sequence_length:i].reshape(1, sequence_length, len(lstm_feature_cols))
        pred_prob = lstm_model.predict(sequence, verbose=0)[0][0]
        pred = pred_prob > 0.4  # Lower threshold for LSTM
        predictions_lstm.append(bool(pred))
    
    # Align predictions with trades (first sequence_length trades have no prediction)
    completed_sorted['LSTM_Predicted'] = [False] * sequence_length + predictions_lstm
    
    # Merge back
    completed = completed.merge(
        completed_sorted[['EntryTime', 'LSTM_Predicted']],
        on='EntryTime',
        how='left',
        suffixes=('', '_lstm')
    )
    completed['LSTM_Predicted'] = completed['LSTM_Predicted'].fillna(False)
    
    lstm_filtered = completed[completed['LSTM_Predicted'] == True]
    
    if len(lstm_filtered) > 0:
        lstm_wins = (lstm_filtered['P&L'] > 0).sum()
        lstm_win_rate = lstm_wins / len(lstm_filtered) * 100
        lstm_total_pnl = lstm_filtered['P&L'].sum()
        lstm_avg_pnl = lstm_filtered['P&L'].mean()
        
        print(f"   Trades taken: {len(lstm_filtered)} ({len(lstm_filtered)/len(completed)*100:.1f}%)")
        print(f"   Win rate: {lstm_win_rate:.2f}%")
        print(f"   Total P&L: ${lstm_total_pnl:.2f}")
        print(f"   Avg P&L: ${lstm_avg_pnl:.2f}")
        
        lstm_results = {
            'Model': 'LSTM',
            'Trades': len(lstm_filtered),
            'WinRate': lstm_win_rate,
            'TotalP&L': lstm_total_pnl,
            'AvgP&L': lstm_avg_pnl
        }
    else:
        lstm_results = {'Model': 'LSTM', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}
        
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    lstm_results = {'Model': 'LSTM', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

# ============================================================================
# 6. COMPREHENSIVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("6. COMPREHENSIVE COMPARISON - ALL 4 MODELS")
print("="*80)

comparison_df = pd.DataFrame([baseline_results, rf_results, dl_results, lstm_results])

# Calculate improvements
comparison_df['WinRateImprovement'] = comparison_df['WinRate'] - baseline_win_rate
comparison_df['P&LImprovement'] = comparison_df['TotalP&L'] - baseline_total_pnl
comparison_df['P&LImprovementPct'] = (comparison_df['P&LImprovement'] / abs(baseline_total_pnl)) * 100 if baseline_total_pnl != 0 else 0
comparison_df['TradesTakenPct'] = (comparison_df['Trades'] / len(completed)) * 100

print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# 7. WINNER ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("7. WINNER ANALYSIS")
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
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("8. SAVING RESULTS")
print("="*80)

comparison_df.to_csv('all_models_comparison_gold.csv', index=False)
print("   Comparison saved to: all_models_comparison_gold.csv")

if 'RF_Predicted' in completed.columns and 'DL_Predicted' in completed.columns:
    completed.to_csv('all_models_predictions_gold.csv', index=False)
    print("   Detailed predictions saved to: all_models_predictions_gold.csv")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)

