import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from predict_trade import create_trade_features_from_entry
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL TESTING ON NEW DATA")
print("="*80)

# Try to import TensorFlow for LSTM models
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ============================================================================
# 1. PROCESS NEW DATA
# ============================================================================
print("\n1. Processing new data...")

# Load and process new raw data
df_new = pd.read_csv('XAUUSD5 new data.csv', header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Combine Date and Time
df_new['DateTime'] = pd.to_datetime(df_new['Date'] + ' ' + df_new['Time'])
df_new = df_new.set_index('DateTime')
df_new = df_new[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"   Loaded {len(df_new)} candles")
print(f"   Date range: {df_new.index[0]} to {df_new.index[-1]}")

# Run strategy backtest on new data
print("\n2. Running strategy backtest on new data...")
import sys
sys.path.append('/home/nyale/Desktop/ML TESTING')

# Import strategy backtest
from strategy_backtest import TradingStrategy

# Initialize strategy with data file path
strategy = TradingStrategy('XAUUSD5 new data.csv')
strategy.load_data()
strategy.add_indicators()
strategy.identify_key_times()
strategy.backtest_strategy()
backtest_results = pd.DataFrame(strategy.trades)

print(f"   Completed trades: {len(backtest_results[backtest_results['Status'].isin(['TP_HIT', 'SL_HIT'])])}")

# Save backtest results
backtest_results.to_csv('new_data_backtest_results.csv', index=False)
print("   Saved: new_data_backtest_results.csv")

# Get completed trades
completed = backtest_results[backtest_results['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
completed['EntryTime'] = pd.to_datetime(completed['EntryTime'])

if len(completed) == 0:
    print("\n   ERROR: No completed trades found in new data!")
    print("   Cannot test models without completed trades.")
    exit(1)

print(f"   Found {len(completed)} completed trades for testing")

# ============================================================================
# 2. BASELINE STRATEGY
# ============================================================================
print("\n" + "="*80)
print("2. BASELINE STRATEGY (No ML Filter)")
print("="*80)

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
print("\n" + "="*80)
print("3. RANDOM FOREST ML MODEL")
print("="*80)

rf_results = {'Model': 'Random Forest ML', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

try:
    rf_classifier = joblib.load('best_classifier_model.pkl')
    rf_feature_cols = pd.read_csv('feature_columns.csv', header=None)[0].tolist()
    
    predictions_rf = []
    for idx, trade in completed.iterrows():
        try:
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
        except Exception as e:
            predictions_rf.append(False)
    
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
        print("   No trades predicted as wins")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. DEEP LEARNING MODEL
# ============================================================================
print("\n" + "="*80)
print("4. DEEP LEARNING MODEL")
print("="*80)

dl_results = {'Model': 'Deep Learning', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

try:
    dl_classifier = joblib.load('deep_learning_classifier.pkl')
    dl_scaler = joblib.load('dl_feature_scaler.pkl')
    dl_feature_cols = pd.read_csv('dl_feature_columns.csv', header=None)[0].tolist()
    
    predictions_dl = []
    for idx, trade in completed.iterrows():
        try:
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
        except Exception as e:
            predictions_dl.append(False)
    
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
        print("   No trades predicted as wins")
except Exception as e:
    print(f"   ERROR: {e}")

# ============================================================================
# 5. SHALLOW LSTM MODEL
# ============================================================================
print("\n" + "="*80)
print("5. SHALLOW LSTM MODEL")
print("="*80)

lstm_shallow_results = {'Model': 'LSTM (Shallow)', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

if TENSORFLOW_AVAILABLE:
    try:
        lstm_model = keras.models.load_model('lstm_model_gold.h5')
        lstm_feature_cols = pd.read_csv('lstm_feature_columns.csv', header=None)[0].tolist()
        lstm_params = pd.read_csv('lstm_model_params.csv', header=None)[0].tolist()
        sequence_length = int(lstm_params[1]) if len(lstm_params) > 1 else 10
        
        # Skip '0' placeholder
        lstm_feature_cols = [col for col in lstm_feature_cols if col != '0' and col != 0]
        
        # Sort by entry time
        completed_sorted = completed.sort_values('EntryTime').reset_index(drop=True)
        
        # Prepare features
        X_lstm_list = []
        for idx, trade in completed_sorted.iterrows():
            feature_row = []
            for col in lstm_feature_cols:
                if col in trade.index:
                    val = trade[col] if pd.notna(trade[col]) else 0
                else:
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
            pred = pred_prob > 0.4  # Lower threshold
            predictions_lstm.append(bool(pred))
        
        # Align predictions
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
            
            lstm_shallow_results = {
                'Model': 'LSTM (Shallow)',
                'Trades': len(lstm_filtered),
                'WinRate': lstm_win_rate,
                'TotalP&L': lstm_total_pnl,
                'AvgP&L': lstm_avg_pnl
            }
        else:
            print("   No trades predicted as wins")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   TensorFlow not available")

# ============================================================================
# 6. DEEP LSTM MODEL
# ============================================================================
print("\n" + "="*80)
print("6. DEEP LSTM MODEL")
print("="*80)

lstm_deep_results = {'Model': 'LSTM (Deep)', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

if TENSORFLOW_AVAILABLE:
    try:
        lstm_deep_model = keras.models.load_model('lstm_deep_model_gold.h5')
        lstm_deep_feature_cols = pd.read_csv('lstm_deep_feature_columns.csv', header=None)[0].tolist()
        lstm_deep_params = pd.read_csv('lstm_deep_model_params.csv', header=None)[0].tolist()
        sequence_length_deep = int(lstm_deep_params[1]) if len(lstm_deep_params) > 1 else 15
        
        # Skip '0' placeholder
        lstm_deep_feature_cols = [col for col in lstm_deep_feature_cols if col != '0' and col != 0]
        
        # Sort by entry time
        completed_sorted = completed.sort_values('EntryTime').reset_index(drop=True)
        
        # Prepare features
        X_lstm_deep_list = []
        for idx, trade in completed_sorted.iterrows():
            feature_row = []
            for col in lstm_deep_feature_cols:
                if col in trade.index:
                    val = trade[col] if pd.notna(trade[col]) else 0
                else:
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
            X_lstm_deep_list.append(feature_row)
        
        X_lstm_deep = np.array(X_lstm_deep_list)
        scaler_lstm_deep = StandardScaler()
        X_lstm_deep_scaled = scaler_lstm_deep.fit_transform(X_lstm_deep)
        
        # Create sequences and predict
        predictions_lstm_deep = []
        for i in range(sequence_length_deep, len(completed_sorted)):
            sequence = X_lstm_deep_scaled[i-sequence_length_deep:i].reshape(1, sequence_length_deep, len(lstm_deep_feature_cols))
            pred_prob = lstm_deep_model.predict(sequence, verbose=0)[0][0]
            pred = pred_prob > 0.3  # Lower threshold for deep LSTM
            predictions_lstm_deep.append(bool(pred))
        
        # Align predictions
        completed_sorted['LSTM_Deep_Predicted'] = [False] * sequence_length_deep + predictions_lstm_deep
        
        # Merge back
        completed = completed.merge(
            completed_sorted[['EntryTime', 'LSTM_Deep_Predicted']],
            on='EntryTime',
            how='left',
            suffixes=('', '_lstm_deep')
        )
        completed['LSTM_Deep_Predicted'] = completed['LSTM_Deep_Predicted'].fillna(False)
        
        lstm_deep_filtered = completed[completed['LSTM_Deep_Predicted'] == True]
        
        if len(lstm_deep_filtered) > 0:
            lstm_deep_wins = (lstm_deep_filtered['P&L'] > 0).sum()
            lstm_deep_win_rate = lstm_deep_wins / len(lstm_deep_filtered) * 100
            lstm_deep_total_pnl = lstm_deep_filtered['P&L'].sum()
            lstm_deep_avg_pnl = lstm_deep_filtered['P&L'].mean()
            
            print(f"   Trades taken: {len(lstm_deep_filtered)} ({len(lstm_deep_filtered)/len(completed)*100:.1f}%)")
            print(f"   Win rate: {lstm_deep_win_rate:.2f}%")
            print(f"   Total P&L: ${lstm_deep_total_pnl:.2f}")
            print(f"   Avg P&L: ${lstm_deep_avg_pnl:.2f}")
            
            lstm_deep_results = {
                'Model': 'LSTM (Deep)',
                'Trades': len(lstm_deep_filtered),
                'WinRate': lstm_deep_win_rate,
                'TotalP&L': lstm_deep_total_pnl,
                'AvgP&L': lstm_deep_avg_pnl
            }
        else:
            print("   No trades predicted as wins")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   TensorFlow not available")

# ============================================================================
# 7. COMPREHENSIVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("7. COMPREHENSIVE COMPARISON - ALL MODELS ON NEW DATA")
print("="*80)

comparison_df = pd.DataFrame([
    baseline_results,
    rf_results,
    dl_results,
    lstm_shallow_results,
    lstm_deep_results
])

# Calculate improvements
comparison_df['WinRateImprovement'] = comparison_df['WinRate'] - baseline_win_rate
comparison_df['P&LImprovement'] = comparison_df['TotalP&L'] - baseline_total_pnl
comparison_df['P&LImprovementPct'] = (comparison_df['P&LImprovement'] / abs(baseline_total_pnl)) * 100 if baseline_total_pnl != 0 else 0
comparison_df['TradesTakenPct'] = (comparison_df['Trades'] / len(completed)) * 100

print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# 8. ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8. DETAILED ANALYSIS")
print("="*80)

best_win_rate = comparison_df.loc[comparison_df['WinRate'].idxmax()]
best_pnl = comparison_df.loc[comparison_df['TotalP&L'].idxmax()]
best_avg_pnl = comparison_df.loc[comparison_df['AvgP&L'].idxmax()]

print(f"\n🏆 Best Win Rate: {best_win_rate['Model']} ({best_win_rate['WinRate']:.2f}%)")
print(f"💰 Best Total P&L: {best_pnl['Model']} (${best_pnl['TotalP&L']:.2f})")
print(f"📈 Best Avg P&L: {best_avg_pnl['Model']} (${best_avg_pnl['AvgP&L']:.2f})")

# Overall winner
comparison_df['Score'] = (
    (comparison_df['WinRate'] / 100) * 0.4 +
    (comparison_df['TotalP&L'] / abs(baseline_total_pnl)) * 0.4 +
    (comparison_df['AvgP&L'] / abs(baseline_avg_pnl)) * 0.2
)
best_overall = comparison_df.loc[comparison_df['Score'].idxmax()]
print(f"\n🎯 Overall Best Model: {best_overall['Model']} (Score: {best_overall['Score']:.3f})")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("9. SAVING RESULTS")
print("="*80)

comparison_df.to_csv('new_data_all_models_comparison.csv', index=False)
print("   Comparison saved to: new_data_all_models_comparison.csv")

if 'RF_Predicted' in completed.columns:
    completed.to_csv('new_data_all_models_predictions.csv', index=False)
    print("   Detailed predictions saved to: new_data_all_models_predictions.csv")

print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)

