import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("OVERFITTING ANALYSIS & DIAGNOSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")
df_original = pd.read_csv('advanced_analysis_results_combined.csv')
df_original = df_original[df_original['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

df_new = pd.read_csv('new_data_backtest_results.csv')
df_new = df_new[df_new['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

print(f"   Original data: {len(df_original)} trades")
print(f"   New data: {len(df_new)} trades")

# ============================================================================
# 2. COMPARE DATA DISTRIBUTIONS
# ============================================================================
print("\n2. Comparing data distributions...")

# Key features to compare
key_features = ['Risk', 'ATR_Ratio', 'EMA_200_1H', 'RangeSizePct', 'ATR_Pct']

print("\n   Feature Statistics Comparison:")
print("   " + "-"*70)
print(f"   {'Feature':<20} {'Original Mean':<15} {'New Mean':<15} {'Difference':<15}")
print("   " + "-"*70)

for feature in key_features:
    if feature in df_original.columns and feature in df_new.columns:
        orig_mean = df_original[feature].mean()
        new_mean = df_new[feature].mean()
        diff = ((new_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0
        print(f"   {feature:<20} {orig_mean:>14.2f} {new_mean:>14.2f} {diff:>14.1f}%")

# Win rate comparison
orig_win_rate = (df_original['P&L'] > 0).sum() / len(df_original) * 100
new_win_rate = (df_new['P&L'] > 0).sum() / len(df_new) * 100

print(f"\n   Win Rate Comparison:")
print(f"   Original data: {orig_win_rate:.2f}%")
print(f"   New data: {new_win_rate:.2f}%")
print(f"   Difference: {new_win_rate - orig_win_rate:.2f}%")

# ============================================================================
# 3. TEST CURRENT MODEL ON BOTH DATASETS
# ============================================================================
print("\n3. Testing current Random Forest model...")

try:
    rf_model = joblib.load('best_classifier_model.pkl')
    rf_feature_cols = pd.read_csv('feature_columns.csv', header=None)[0].tolist()
    rf_feature_cols = [col for col in rf_feature_cols if col != '0' and col != 0]
    
    # Prepare features
    exclude_cols = ['EntryTime', 'ExitTime', 'Status', 'P&L', 'R_Multiple', 
                    'EntryPrice', 'SL', 'TP', 'WindowHigh', 'WindowLow', 'TradeID',
                    'ExitPrice', 'BestPossibleP&L', 'WouldHaveWon', 'ReachedTP', 
                    'WaitDuration_Min', 'BestWaitDuration_Min', 'TradeID_x', 'TradeID_y',
                    'MinAdverse_Min30', 'MaxFavorable_Min30', 'WentAgainstFirst',
                    'Duration_Minutes', 'Duration_Candles']
    
    def prepare_features(df, feature_cols):
        X = []
        y = []
        for idx, trade in df.iterrows():
            feature_row = []
            for col in feature_cols:
                if col in trade.index:
                    val = trade[col] if pd.notna(trade[col]) else 0
                else:
                    val = 0
                feature_row.append(float(val))
            X.append(feature_row)
            y.append(1 if trade['P&L'] > 0 else 0)
        return np.array(X), np.array(y)
    
    X_orig, y_orig = prepare_features(df_original, rf_feature_cols)
    X_new, y_new = prepare_features(df_new, rf_feature_cols)
    
    # Predictions
    pred_orig = rf_model.predict(X_orig)
    pred_new = rf_model.predict(X_new)
    
    acc_orig = accuracy_score(y_orig, pred_orig)
    acc_new = accuracy_score(y_new, pred_new)
    
    print(f"\n   Model Accuracy:")
    print(f"   Original data: {acc_orig:.4f} ({acc_orig*100:.2f}%)")
    print(f"   New data: {acc_new:.4f} ({acc_new*100:.2f}%)")
    print(f"   Performance drop: {(acc_orig - acc_new)*100:.2f}%")
    
    if acc_orig - acc_new > 0.15:  # More than 15% drop
        print("\n   ⚠️  SIGNIFICANT OVERFITTING DETECTED!")
        print(f"   Model accuracy dropped by {(acc_orig - acc_new)*100:.2f}%")
    
except Exception as e:
    print(f"   ERROR: {e}")

# ============================================================================
# 4. CROSS-VALIDATION ANALYSIS
# ============================================================================
print("\n4. Cross-validation analysis on original data...")

try:
    # Combine data for cross-validation
    X_combined = np.vstack([X_orig, X_new])
    y_combined = np.hstack([y_orig, y_new])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_combined, y_combined, cv=cv, scoring='accuracy')
    
    print(f"\n   Cross-Validation Scores:")
    print(f"   Mean: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"   Std: {cv_scores.std():.4f} ({cv_scores.std()*100:.2f}%)")
    print(f"   Min: {cv_scores.min():.4f} ({cv_scores.min()*100:.2f}%)")
    print(f"   Max: {cv_scores.max():.4f} ({cv_scores.max()*100:.2f}%)")
    
    if cv_scores.std() > 0.1:
        print("\n   ⚠️  High variance in CV scores - model may be unstable")
    
except Exception as e:
    print(f"   ERROR: {e}")

# ============================================================================
# 5. OVERFITTING SOLUTIONS
# ============================================================================
print("\n" + "="*80)
print("5. OVERFITTING SOLUTIONS")
print("="*80)

print("\nRecommended Solutions:")
print("1. Retrain with combined data (original + new)")
print("2. Reduce model complexity (fewer trees, max_depth)")
print("3. Increase regularization (min_samples_split, min_samples_leaf)")
print("4. Feature selection (remove less important features)")
print("5. Use ensemble of simpler models")
print("6. Add more training data")
print("7. Use cross-validation for model selection")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

