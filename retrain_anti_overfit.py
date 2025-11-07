import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RETRAINING WITH ANTI-OVERFITTING MEASURES")
print("="*80)

# ============================================================================
# 1. LOAD AND COMBINE DATA
# ============================================================================
print("\n1. Loading and combining data...")

# Load original data
df_original = pd.read_csv('advanced_analysis_results_combined.csv')
df_original = df_original[df_original['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

# Load new data
df_new = pd.read_csv('new_data_backtest_results.csv')
df_new = df_new[df_new['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

# Combine
df_combined = pd.concat([df_original, df_new], ignore_index=True)
print(f"   Original trades: {len(df_original)}")
print(f"   New trades: {len(df_new)}")
print(f"   Combined trades: {len(df_combined)}")

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================
print("\n2. Preparing features...")

exclude_cols = ['EntryTime', 'ExitTime', 'Status', 'P&L', 'R_Multiple', 
                'EntryPrice', 'SL', 'TP', 'WindowHigh', 'WindowLow', 'TradeID',
                'ExitPrice', 'BestPossibleP&L', 'WouldHaveWon', 'ReachedTP', 
                'WaitDuration_Min', 'BestWaitDuration_Min', 'TradeID_x', 'TradeID_y',
                'MinAdverse_Min30', 'MaxFavorable_Min30', 'WentAgainstFirst',
                'Duration_Minutes', 'Duration_Candles']

# Get feature columns
feature_cols = [col for col in df_combined.columns 
                if col not in exclude_cols and df_combined[col].dtype in ['int64', 'float64']]
feature_cols = [col for col in feature_cols if df_combined[col].notna().sum() > len(df_combined) * 0.5]

print(f"   Selected {len(feature_cols)} features")

# Prepare features
X = df_combined[feature_cols].copy()
X = X.fillna(X.median())

# Prepare targets
y_binary = (df_combined['P&L'] > 0).astype(int)

print(f"   Target: {y_binary.sum()} wins, {len(y_binary) - y_binary.sum()} losses")
print(f"   Win rate: {y_binary.sum() / len(y_binary) * 100:.2f}%")

# ============================================================================
# 3. SPLIT DATA (3 BATCHES)
# ============================================================================
print("\n3. Splitting data into 3 batches...")

# First split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.4, random_state=42, stratify=y_binary
)

# Second split: 20% validation, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Batch 1 (Training):   {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Batch 2 (Validation): {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Batch 3 (Testing):    {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 4. TRAIN ANTI-OVERFITTING MODEL
# ============================================================================
print("\n" + "="*80)
print("4. TRAINING ANTI-OVERFITTING MODEL")
print("="*80)

# Much more aggressive regularization to prevent overfitting
model = RandomForestClassifier(
    n_estimators=50,  # Fewer trees
    max_depth=5,  # Shallow trees (much more restrictive)
    min_samples_split=50,  # Require many samples to split
    min_samples_leaf=25,  # Require many samples in leaf
    max_features='sqrt',  # Use sqrt of features
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

print("\nModel Parameters (Anti-Overfitting):")
print(f"  n_estimators: {model.n_estimators}")
print(f"  max_depth: {model.max_depth}")
print(f"  min_samples_split: {model.min_samples_split}")
print(f"  min_samples_leaf: {model.min_samples_leaf}")
print(f"  max_features: {model.max_features}")

# Cross-validation before training
print("\n5. Cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"   CV Mean Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"   CV Std: {cv_scores.std():.4f} ({cv_scores.std()*100:.2f}%)")

# Train model
print("\n6. Training model...")
model.fit(X_train, y_train)

# ============================================================================
# 5. EVALUATE ON ALL SETS
# ============================================================================
print("\n" + "="*80)
print("7. EVALUATION")
print("="*80)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nAccuracy:")
print(f"   Training:   {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Validation: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Test:       {test_acc:.4f} ({test_acc*100:.2f}%)")

# Check for overfitting
train_val_gap = train_acc - val_acc
val_test_gap = val_acc - test_acc

print(f"\nOverfitting Check:")
print(f"   Train-Val Gap: {train_val_gap:.4f} ({train_val_gap*100:.2f}%)")
print(f"   Val-Test Gap:  {val_test_gap:.4f} ({val_test_gap*100:.2f}%)")

if train_val_gap < 0.05:
    print("   ✅ Good generalization (gap < 5%)")
elif train_val_gap < 0.10:
    print("   ⚠️  Moderate overfitting (gap 5-10%)")
else:
    print("   ❌ Significant overfitting (gap > 10%)")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("\n8. Feature Importance (Top 10):")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + feature_importance.head(10).to_string(index=False))

# ============================================================================
# 7. TEST ON NEW DATA SEPARATELY
# ============================================================================
print("\n" + "="*80)
print("9. TESTING ON NEW DATA")
print("="*80)

# Prepare new data - create missing features
X_new_list = []
for idx, trade in df_new.iterrows():
    feature_row = []
    for col in feature_cols:
        if col in trade.index:
            val = trade[col] if pd.notna(trade[col]) else 0
        else:
            # Create missing features
            if col == 'EntryHour':
                val = pd.to_datetime(trade['EntryTime']).hour if 'EntryTime' in trade else 0
            elif col == 'EntryDayOfWeek':
                val = pd.to_datetime(trade['EntryTime']).weekday() if 'EntryTime' in trade else 0
            elif col == 'RangeSize':
                val = trade.get('WindowHigh', 0) - trade.get('WindowLow', 0)
            elif col == 'RangeSizePct':
                val = ((trade.get('WindowHigh', 0) - trade.get('WindowLow', 0)) / trade['EntryPrice'] * 100) if trade.get('EntryPrice', 0) > 0 else 0
            else:
                val = 0
        feature_row.append(float(val))
    X_new_list.append(feature_row)

X_new = pd.DataFrame(X_new_list, columns=feature_cols)
X_new = X_new.fillna(X_new.median())
y_new = (df_new['P&L'] > 0).astype(int)

new_pred = model.predict(X_new)
new_acc = accuracy_score(y_new, new_pred)

print(f"\nNew Data Performance:")
print(f"   Accuracy: {new_acc:.4f} ({new_acc*100:.2f}%)")
print(f"   Actual Win Rate: {y_new.sum() / len(y_new) * 100:.2f}%")

# Calculate strategy performance
new_predicted_wins = new_pred == 1
if new_predicted_wins.sum() > 0:
    new_filtered = df_new.iloc[np.where(new_predicted_wins)[0]]
    new_wins = (new_filtered['P&L'] > 0).sum()
    new_win_rate = new_wins / len(new_filtered) * 100
    new_total_pnl = new_filtered['P&L'].sum()
    
    print(f"\nStrategy Performance (Filtered Trades):")
    print(f"   Trades taken: {len(new_filtered)} ({len(new_filtered)/len(df_new)*100:.1f}%)")
    print(f"   Win rate: {new_win_rate:.2f}%")
    print(f"   Total P&L: ${new_total_pnl:.2f}")
    print(f"   Avg P&L: ${new_total_pnl/len(new_filtered):.2f}")

# ============================================================================
# 8. SAVE MODEL
# ============================================================================
print("\n" + "="*80)
print("10. SAVING MODEL")
print("="*80)

joblib.dump(model, 'best_classifier_model_retrained.pkl')
pd.Series(feature_cols).to_csv('feature_columns_retrained.csv', index=False)

print("   Model saved: best_classifier_model_retrained.pkl")
print("   Features saved: feature_columns_retrained.csv")

print("\n" + "="*80)
print("RETRAINING COMPLETE!")
print("="*80)

