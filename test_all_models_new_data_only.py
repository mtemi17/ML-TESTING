import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING ALL MODELS ON NEW DATA ONLY (XAUUSD new data)")
print("="*80)

# Try TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ============================================================================
# 1. LOAD NEW DATA
# ============================================================================
print("\n1. Loading new data...")

df_new = pd.read_csv('new_data_backtest_results.csv')
df_new = df_new[df_new['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
df_new['EntryTime'] = pd.to_datetime(df_new['EntryTime'])

print(f"   Total trades: {len(df_new)}")
print(f"   Wins: {(df_new['P&L'] > 0).sum()} ({((df_new['P&L'] > 0).sum()/len(df_new)*100):.1f}%)")
print(f"   Losses: {(df_new['P&L'] <= 0).sum()} ({((df_new['P&L'] <= 0).sum()/len(df_new)*100):.1f}%)")
print(f"   Total P&L: ${df_new['P&L'].sum():.2f}")
print(f"   Avg P&L: ${df_new['P&L'].mean():.2f}")

baseline_results = {
    'Model': 'Baseline (New Data)',
    'Trades': len(df_new),
    'WinRate': (df_new['P&L'] > 0).sum() / len(df_new) * 100,
    'TotalP&L': df_new['P&L'].sum(),
    'AvgP&L': df_new['P&L'].mean()
}

# ============================================================================
# 2. PREPARE FEATURES FUNCTION
# ============================================================================
def prepare_features_for_model(df, feature_cols, scaler=None):
    """Prepare features for model prediction"""
    X_list = []
    
    for idx, trade in df.iterrows():
        feature_row = []
        for col in feature_cols:
            if col in trade.index:
                val = trade[col] if pd.notna(trade[col]) else 0
            else:
                # Create missing features
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
        X_list.append(feature_row)
    
    X = np.array(X_list)
    
    if scaler:
        X = scaler.transform(X)
    
    return X

# ============================================================================
# 3. TEST RANDOM FOREST (RETRAINED)
# ============================================================================
print("\n" + "="*80)
print("2. RANDOM FOREST ML (RETRAINED)")
print("="*80)

rf_results = {'Model': 'Random Forest ML', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

try:
    rf_model = joblib.load('best_classifier_model_retrained.pkl')
    rf_feature_cols = pd.read_csv('feature_columns_retrained.csv', header=None)[0].tolist()
    
    X_rf = prepare_features_for_model(df_new, rf_feature_cols)
    rf_pred = rf_model.predict(X_rf)
    
    df_new['RF_Predicted'] = rf_pred
    rf_filtered = df_new[df_new['RF_Predicted'] == 1]
    
    if len(rf_filtered) > 0:
        rf_wins = (rf_filtered['P&L'] > 0).sum()
        rf_winrate = rf_wins / len(rf_filtered) * 100
        rf_pnl = rf_filtered['P&L'].sum()
        rf_avg = rf_filtered['P&L'].mean()
        
        print(f"   Trades taken: {len(rf_filtered)} ({len(rf_filtered)/len(df_new)*100:.1f}%)")
        print(f"   Win rate: {rf_winrate:.2f}%")
        print(f"   Total P&L: ${rf_pnl:.2f}")
        print(f"   Avg P&L: ${rf_avg:.2f}")
        
        rf_results = {
            'Model': 'Random Forest ML',
            'Trades': len(rf_filtered),
            'WinRate': rf_winrate,
            'TotalP&L': rf_pnl,
            'AvgP&L': rf_avg
        }
    else:
        print("   No trades predicted as wins")
except Exception as e:
    print(f"   ERROR: {e}")

# ============================================================================
# 4. TEST DEEP LEARNING
# ============================================================================
print("\n" + "="*80)
print("3. DEEP LEARNING MODEL")
print("="*80)

dl_results = {'Model': 'Deep Learning', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

try:
    dl_model = joblib.load('deep_learning_classifier.pkl')
    dl_scaler = joblib.load('dl_feature_scaler.pkl')
    dl_feature_cols = pd.read_csv('dl_feature_columns.csv', header=None)[0].tolist()
    dl_feature_cols = [col for col in dl_feature_cols if col != '0' and col != 0]
    
    X_dl = prepare_features_for_model(df_new, dl_feature_cols, dl_scaler)
    dl_pred = dl_model.predict(X_dl)
    
    df_new['DL_Predicted'] = dl_pred
    dl_filtered = df_new[df_new['DL_Predicted'] == 1]
    
    if len(dl_filtered) > 0:
        dl_wins = (dl_filtered['P&L'] > 0).sum()
        dl_winrate = dl_wins / len(dl_filtered) * 100
        dl_pnl = dl_filtered['P&L'].sum()
        dl_avg = dl_filtered['P&L'].mean()
        
        print(f"   Trades taken: {len(dl_filtered)} ({len(dl_filtered)/len(df_new)*100:.1f}%)")
        print(f"   Win rate: {dl_winrate:.2f}%")
        print(f"   Total P&L: ${dl_pnl:.2f}")
        print(f"   Avg P&L: ${dl_avg:.2f}")
        
        dl_results = {
            'Model': 'Deep Learning',
            'Trades': len(dl_filtered),
            'WinRate': dl_winrate,
            'TotalP&L': dl_pnl,
            'AvgP&L': dl_avg
        }
    else:
        print("   No trades predicted as wins")
except Exception as e:
    print(f"   ERROR: {e}")

# ============================================================================
# 5. TEST OPTIMIZED DEEP MODEL (if available)
# ============================================================================
print("\n" + "="*80)
print("4. OPTIMIZED DEEP MODEL (1000 layers)")
print("="*80)

opt_deep_results = {'Model': 'Optimized Deep (1000L)', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

if TENSORFLOW_AVAILABLE:
    try:
        opt_model = keras.models.load_model('optimized_deep_model.h5')
        opt_scaler = joblib.load('optimized_deep_scaler.pkl')
        opt_threshold = joblib.load('optimized_deep_threshold.pkl')
        opt_feature_cols = pd.read_csv('optimized_deep_features.csv', header=None)[0].tolist()
        
        X_opt = prepare_features_for_model(df_new, opt_feature_cols, opt_scaler)
        opt_probs = opt_model.predict(X_opt, verbose=0).flatten()
        opt_pred = (opt_probs > opt_threshold).astype(int)
        
        df_new['OptDeep_Predicted'] = opt_pred
        opt_filtered = df_new[df_new['OptDeep_Predicted'] == 1]
        
        if len(opt_filtered) > 0:
            opt_wins = (opt_filtered['P&L'] > 0).sum()
            opt_winrate = opt_wins / len(opt_filtered) * 100
            opt_pnl = opt_filtered['P&L'].sum()
            opt_avg = opt_filtered['P&L'].mean()
            
            print(f"   Trades taken: {len(opt_filtered)} ({len(opt_filtered)/len(df_new)*100:.1f}%)")
            print(f"   Win rate: {opt_winrate:.2f}%")
            print(f"   Total P&L: ${opt_pnl:.2f}")
            print(f"   Avg P&L: ${opt_avg:.2f}")
            print(f"   Threshold used: {opt_threshold:.3f}")
            
            opt_deep_results = {
                'Model': 'Optimized Deep (1000L)',
                'Trades': len(opt_filtered),
                'WinRate': opt_winrate,
                'TotalP&L': opt_pnl,
                'AvgP&L': opt_avg
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
# 6. TEST ULTIMATE DEEP MODEL (if available)
# ============================================================================
print("\n" + "="*80)
print("5. ULTIMATE DEEP MODEL (800 layers)")
print("="*80)

ult_deep_results = {'Model': 'Ultimate Deep (800L)', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

if TENSORFLOW_AVAILABLE:
    try:
        ult_model = keras.models.load_model('ultimate_deep_model.h5')
        ult_scaler = joblib.load('ultimate_deep_scaler.pkl')
        ult_threshold = joblib.load('ultimate_deep_threshold.pkl')
        ult_feature_cols = pd.read_csv('ultimate_deep_features.csv', header=None)[0].tolist()
        
        X_ult = prepare_features_for_model(df_new, ult_feature_cols, ult_scaler)
        ult_probs = ult_model.predict(X_ult, verbose=0).flatten()
        ult_pred = (ult_probs > ult_threshold).astype(int)
        
        df_new['UltDeep_Predicted'] = ult_pred
        ult_filtered = df_new[df_new['UltDeep_Predicted'] == 1]
        
        if len(ult_filtered) > 0:
            ult_wins = (ult_filtered['P&L'] > 0).sum()
            ult_winrate = ult_wins / len(ult_filtered) * 100
            ult_pnl = ult_filtered['P&L'].sum()
            ult_avg = ult_filtered['P&L'].mean()
            
            print(f"   Trades taken: {len(ult_filtered)} ({len(ult_filtered)/len(df_new)*100:.1f}%)")
            print(f"   Win rate: {ult_winrate:.2f}%")
            print(f"   Total P&L: ${ult_pnl:.2f}")
            print(f"   Avg P&L: ${ult_avg:.2f}")
            print(f"   Threshold used: {ult_threshold:.3f}")
            
            ult_deep_results = {
                'Model': 'Ultimate Deep (800L)',
                'Trades': len(ult_filtered),
                'WinRate': ult_winrate,
                'TotalP&L': ult_pnl,
                'AvgP&L': ult_avg
            }
        else:
            print("   No trades predicted as wins")
    except Exception as e:
        print(f"   ERROR: {e}")

# ============================================================================
# 7. TEST BEST DEEP MODEL (if available)
# ============================================================================
print("\n" + "="*80)
print("6. BEST DEEP MODEL (75 layers)")
print("="*80)

best_deep_results = {'Model': 'Best Deep (75L)', 'Trades': 0, 'WinRate': 0, 'TotalP&L': 0, 'AvgP&L': 0}

if TENSORFLOW_AVAILABLE:
    try:
        best_model = keras.models.load_model('best_deep_model.h5')
        best_scaler = joblib.load('deep_model_scaler.pkl')
        best_feature_cols = pd.read_csv('deep_model_features.csv', header=None)[0].tolist()
        
        X_best = prepare_features_for_model(df_new, best_feature_cols, best_scaler)
        best_probs = best_model.predict(X_best, verbose=0).flatten()
        
        # Test multiple thresholds
        best_threshold = 0.5
        best_score = 0
        
        for threshold in np.arange(0.3, 0.9, 0.1):
            best_pred = (best_probs > threshold).astype(int)
            if best_pred.sum() > 0:
                filtered = df_new.iloc[np.where(best_pred)[0]]
                winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
                pnl = filtered['P&L'].sum()
                score = winrate * 0.6 + (pnl / 100) * 0.4
                if score > best_score and winrate >= 40:
                    best_score = score
                    best_threshold = threshold
        
        best_pred = (best_probs > best_threshold).astype(int)
        df_new['BestDeep_Predicted'] = best_pred
        best_filtered = df_new[df_new['BestDeep_Predicted'] == 1]
        
        if len(best_filtered) > 0:
            best_wins = (best_filtered['P&L'] > 0).sum()
            best_winrate = best_wins / len(best_filtered) * 100
            best_pnl = best_filtered['P&L'].sum()
            best_avg = best_filtered['P&L'].mean()
            
            print(f"   Trades taken: {len(best_filtered)} ({len(best_filtered)/len(df_new)*100:.1f}%)")
            print(f"   Win rate: {best_winrate:.2f}%")
            print(f"   Total P&L: ${best_pnl:.2f}")
            print(f"   Avg P&L: ${best_avg:.2f}")
            print(f"   Threshold used: {best_threshold:.3f}")
            
            best_deep_results = {
                'Model': 'Best Deep (75L)',
                'Trades': len(best_filtered),
                'WinRate': best_winrate,
                'TotalP&L': best_pnl,
                'AvgP&L': best_avg
            }
        else:
            print("   No trades predicted as wins")
    except Exception as e:
        print(f"   ERROR: {e}")

# ============================================================================
# 8. TEST OPTIMAL FILTER COMBINATIONS (Rule-Based)
# ============================================================================
print("\n" + "="*80)
print("7. OPTIMAL FILTER COMBINATIONS (Rule-Based)")
print("="*80)

# Load optimal combinations
try:
    optimal_filters = pd.read_csv('optimal_filter_combinations.csv')
    
    # Test top 3 combinations
    print("\nTesting top 3 optimal filter combinations:")
    
    filter_results = []
    
    for idx, row in optimal_filters.head(3).iterrows():
        filter_type = row['Type']
        filter_value = row['Value']
        
        # Apply filter
        if filter_type == 'ATR_Ratio':
            if '0-0.8' in str(filter_value):
                filtered = df_new[df_new['ATR_Ratio'] <= 0.8]
        elif filter_type == 'EMA_Pattern + Risk':
            if 'Pattern_110' in row['Filter'] and 'Risk>=' in row['Filter']:
                risk_val = float(row['Filter'].split('Risk>=')[1].split('_')[0])
                filtered = df_new[
                    (df_new['EMA_9_Above_21'].astype(int).astype(str) + 
                     df_new['EMA_21_Above_50'].astype(int).astype(str) + 
                     df_new['Price_Above_EMA200_1H'].astype(int).astype(str) == '110') &
                    (df_new['Risk'] >= risk_val)
                ]
        elif filter_type == 'Risk + Trend':
            if 'Risk>=' in str(filter_value) and 'Trend>=' in str(filter_value):
                risk_val = float(str(filter_value).split('Risk>=')[1].split(',')[0])
                trend_val = float(str(filter_value).split('Trend>=')[1])
                filtered = df_new[
                    (df_new['Risk'] >= risk_val) &
                    (df_new['Trend_Score'] >= trend_val)
                ]
        else:
            continue
        
        if len(filtered) > 0:
            winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
            pnl = filtered['P&L'].sum()
            avg_pnl = filtered['P&L'].mean()
            
            print(f"\n   {row['Filter']}:")
            print(f"     Trades: {len(filtered)} ({len(filtered)/len(df_new)*100:.1f}%)")
            print(f"     Win rate: {winrate:.2f}%")
            print(f"     Total P&L: ${pnl:.2f}")
            print(f"     Avg P&L: ${avg_pnl:.2f}")
            
            filter_results.append({
                'Model': f"Filter: {row['Filter']}",
                'Trades': len(filtered),
                'WinRate': winrate,
                'TotalP&L': pnl,
                'AvgP&L': avg_pnl
            })
            
            # Store best filter result
            if len(filter_results) == 1:
                best_filter_result = {
                    'Model': f"Best Filter: {row['Filter']}",
                    'Trades': len(filtered),
                    'WinRate': winrate,
                    'TotalP&L': pnl,
                    'AvgP&L': avg_pnl
                }
except Exception as e:
    print(f"   ERROR loading filters: {e}")
    best_filter_result = None

# ============================================================================
# 9. COMPREHENSIVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("8. COMPREHENSIVE COMPARISON - ALL MODELS ON NEW DATA")
print("="*80)

all_results = [baseline_results, rf_results, dl_results]

if opt_deep_results['Trades'] > 0:
    all_results.append(opt_deep_results)
if ult_deep_results['Trades'] > 0:
    all_results.append(ult_deep_results)
if best_deep_results['Trades'] > 0:
    all_results.append(best_deep_results)

# Add filter results
if 'filter_results' in locals() and len(filter_results) > 0:
    all_results.extend(filter_results)

comparison_df = pd.DataFrame(all_results)

# Calculate improvements
baseline_pnl = baseline_results['TotalP&L']
baseline_winrate = baseline_results['WinRate']

comparison_df['WinRateImprovement'] = comparison_df['WinRate'] - baseline_winrate
comparison_df['P&LImprovement'] = comparison_df['TotalP&L'] - baseline_pnl
comparison_df['P&LImprovementPct'] = (comparison_df['P&LImprovement'] / abs(baseline_pnl)) * 100 if baseline_pnl != 0 else 0
comparison_df['TradesTakenPct'] = (comparison_df['Trades'] / len(df_new)) * 100

print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# 10. WINNER ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("9. WINNER ANALYSIS")
print("="*80)

best_winrate = comparison_df.loc[comparison_df['WinRate'].idxmax()]
best_pnl = comparison_df.loc[comparison_df['TotalP&L'].idxmax()]
best_avg_pnl = comparison_df.loc[comparison_df['AvgP&L'].idxmax()]

print(f"\n🏆 Best Win Rate: {best_winrate['Model']} ({best_winrate['WinRate']:.2f}%)")
print(f"💰 Best Total P&L: {best_pnl['Model']} (${best_pnl['TotalP&L']:.2f})")
print(f"📈 Best Avg P&L: {best_avg_pnl['Model']} (${best_avg_pnl['AvgP&L']:.2f})")

# Overall score
baseline_avg_pnl = baseline_results['AvgP&L']
comparison_df['Score'] = (
    (comparison_df['WinRate'] / 100) * 0.4 +
    (comparison_df['TotalP&L'] / abs(baseline_pnl)) * 0.4 +
    (comparison_df['AvgP&L'] / abs(baseline_avg_pnl)) * 0.2
)
best_overall = comparison_df.loc[comparison_df['Score'].idxmax()]
print(f"\n🎯 Overall Best: {best_overall['Model']} (Score: {best_overall['Score']:.3f})")

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("10. SAVING RESULTS")
print("="*80)

comparison_df.to_csv('new_data_only_all_models_comparison.csv', index=False)
print("   Comparison saved: new_data_only_all_models_comparison.csv")

if 'RF_Predicted' in df_new.columns:
    df_new.to_csv('new_data_only_all_predictions.csv', index=False)
    print("   Predictions saved: new_data_only_all_predictions.csv")

print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)

