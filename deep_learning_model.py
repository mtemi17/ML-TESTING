import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEEP LEARNING MODEL TRAINING")
print("="*80)

# Set random seeds for reproducibility
np.random.seed(42)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. Loading combined dataset...")
df = pd.read_csv('advanced_analysis_results_combined.csv')
df = df[df['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
print(f"   Loaded {len(df)} completed trades")

# Exclude columns that leak information
exclude_cols = ['EntryTime', 'ExitTime', 'Status', 'P&L', 'R_Multiple', 
                'EntryPrice', 'SL', 'TP', 'WindowHigh', 'WindowLow', 'TradeID',
                'ExitPrice', 'BestPossibleP&L', 'WouldHaveWon', 'ReachedTP', 
                'WaitDuration_Min', 'BestWaitDuration_Min', 'TradeID_x', 'TradeID_y',
                'MinAdverse_Min30', 'MaxFavorable_Min30', 'WentAgainstFirst',
                'Duration_Minutes', 'Duration_Candles']

# Get feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
feature_cols = [col for col in feature_cols if df[col].notna().sum() > len(df) * 0.5]

print(f"   Selected {len(feature_cols)} features")

# Prepare features
X = df[feature_cols].copy()
X = X.fillna(X.median())

# Prepare targets
y_binary = (df['P&L'] > 0).astype(int)
y_continuous = df['P&L'].values

print(f"   Target: Win/Loss (binary) - {y_binary.sum()} wins, {len(y_binary) - y_binary.sum()} losses")

# ============================================================================
# 2. SPLIT DATA INTO 3 BATCHES
# ============================================================================
print("\n2. Splitting data into 3 batches...")

X_train, X_temp, y_binary_train, y_binary_temp, y_cont_train, y_cont_temp = train_test_split(
    X, y_binary, y_continuous, test_size=0.4, random_state=42, stratify=y_binary
)

X_val, X_test, y_binary_val, y_binary_test, y_cont_val, y_cont_test = train_test_split(
    X_temp, y_binary_temp, y_cont_temp, test_size=0.5, random_state=42, stratify=y_binary_temp
)

print(f"   Batch 1 (Training):   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Batch 2 (Validation): {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Batch 3 (Testing):    {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 3. SCALE FEATURES
# ============================================================================
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4. BUILD DEEP LEARNING MODELS
# ============================================================================
print("\n" + "="*80)
print("4. BUILDING DEEP LEARNING MODELS (Neural Networks)")
print("="*80)

# ============================================================================
# 4.1 CLASSIFICATION MODEL (Win/Loss Prediction)
# ============================================================================
print("\n4.1 Building Classification Neural Network...")
print("   Architecture: 128 -> 64 -> 32 -> 16 -> 1 (with dropout)")

clf_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False
)

print("\n   Training classification model...")
clf_model.fit(X_train_scaled, y_binary_train)

# Evaluate
train_pred = clf_model.predict(X_train_scaled)
val_pred = clf_model.predict(X_val_scaled)
test_pred = clf_model.predict(X_test_scaled)

train_acc = accuracy_score(y_binary_train, train_pred)
val_acc = accuracy_score(y_binary_val, val_pred)
test_acc = accuracy_score(y_binary_test, test_pred)

print(f"   Training Accuracy:   {train_acc:.4f}")
print(f"   Validation Accuracy: {val_acc:.4f}")
print(f"   Test Accuracy:       {test_acc:.4f}")

# ============================================================================
# 4.2 REGRESSION MODEL (P&L Prediction)
# ============================================================================
print("\n4.2 Building Regression Neural Network...")
print("   Architecture: 128 -> 64 -> 32 -> 16 -> 1 (with dropout)")

reg_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False
)

print("\n   Training regression model...")
reg_model.fit(X_train_scaled, y_cont_train)

# Evaluate
val_pred_reg = reg_model.predict(X_val_scaled)
test_pred_reg = reg_model.predict(X_test_scaled)

val_r2 = r2_score(y_cont_val, val_pred_reg)
test_r2 = r2_score(y_cont_test, test_pred_reg)
val_rmse = np.sqrt(mean_squared_error(y_cont_val, val_pred_reg))
test_rmse = np.sqrt(mean_squared_error(y_cont_test, test_pred_reg))

print(f"   Validation R²: {val_r2:.4f}, RMSE: ${val_rmse:.2f}")
print(f"   Test R²:       {test_r2:.4f}, RMSE: ${test_rmse:.2f}")

# ============================================================================
# 5. TEST SET EVALUATION
# ============================================================================
print("\n" + "="*80)
print("5. DETAILED TEST SET EVALUATION")
print("="*80)

# Classification
print("\nClassification Results:")
print(confusion_matrix(y_binary_test, test_pred))
print("\nClassification Report:")
print(classification_report(y_binary_test, test_pred, target_names=['Loss', 'Win']))

# Strategy performance
predicted_wins = X_test[test_pred == 1]
actual_pnl_predicted_wins = y_cont_test[test_pred == 1]

print(f"\nStrategy Performance (Only taking predicted wins):")
print(f"  Total trades taken: {len(predicted_wins)}")
if len(predicted_wins) > 0:
    print(f"  Actual win rate: {(actual_pnl_predicted_wins > 0).sum() / len(actual_pnl_predicted_wins) * 100:.2f}%")
    print(f"  Total P&L: ${actual_pnl_predicted_wins.sum():.2f}")
    print(f"  Avg P&L: ${actual_pnl_predicted_wins.mean():.2f}")

baseline_pnl = y_cont_test.sum()
baseline_win_rate = (y_cont_test > 0).sum() / len(y_cont_test) * 100

print(f"\nBaseline (All trades):")
print(f"  Total trades: {len(y_cont_test)}")
print(f"  Win rate: {baseline_win_rate:.2f}%")
print(f"  Total P&L: ${baseline_pnl:.2f}")
print(f"  Avg P&L: ${y_cont_test.mean():.2f}")

if len(predicted_wins) > 0:
    improvement = actual_pnl_predicted_wins.sum() - (baseline_pnl * len(actual_pnl_predicted_wins) / len(y_cont_test))
    print(f"\nImprovement: ${improvement:.2f} ({improvement/abs(baseline_pnl)*100:.1f}%)")

# ============================================================================
# 6. SAVE MODELS
# ============================================================================
print("\n" + "="*80)
print("6. SAVING MODELS")
print("="*80)

import joblib
joblib.dump(clf_model, 'deep_learning_classifier.pkl')
joblib.dump(reg_model, 'deep_learning_regressor.pkl')
joblib.dump(scaler, 'dl_feature_scaler.pkl')
pd.Series(feature_cols).to_csv('dl_feature_columns.csv', index=False)

print(f"   Classification model saved: deep_learning_classifier.pkl")
print(f"   Regression model saved:      deep_learning_regressor.pkl")
print(f"   Scaler saved:                dl_feature_scaler.pkl")
print(f"   Feature columns saved:       dl_feature_columns.csv")

print("\n" + "="*80)
print("DEEP LEARNING TRAINING COMPLETE!")
print("="*80)

