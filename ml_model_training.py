import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MACHINE LEARNING MODEL TRAINING")
print("="*80)

# Load enhanced dataset
print("\n1. Loading enhanced dataset...")
df = pd.read_csv('advanced_analysis_results.csv')
print(f"   Loaded {len(df)} trades")

# Filter for completed trades only
df = df[df['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
print(f"   Completed trades: {len(df)}")

# ============================================================================
# 2. PREPARE FEATURES AND TARGETS
# ============================================================================
print("\n2. Preparing features and targets...")

# Select feature columns (exclude identifiers and targets)
# Also exclude features that leak information about outcomes (post-entry features)
exclude_cols = ['EntryTime', 'ExitTime', 'Status', 'P&L', 'R_Multiple', 
                'EntryPrice', 'SL', 'TP', 'WindowHigh', 'WindowLow', 'TradeID',
                'ExitPrice', 'BestPossibleP&L', 'WouldHaveWon', 'ReachedTP', 
                'WaitDuration_Min', 'BestWaitDuration_Min', 'TradeID_x', 'TradeID_y',
                'MinAdverse_Min30', 'MaxFavorable_Min30', 'WentAgainstFirst',  # Post-entry features
                'Duration_Minutes', 'Duration_Candles']  # Known only after exit

# Get all numeric features
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

# Remove any columns with too many missing values
feature_cols = [col for col in feature_cols if df[col].notna().sum() > len(df) * 0.5]

print(f"   Selected {len(feature_cols)} features")
print(f"   Features: {', '.join(feature_cols[:10])}...")

# Prepare feature matrix
X = df[feature_cols].copy()

# Handle missing values
X = X.fillna(X.median())

# Encode categorical if any
le_dict = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

# Prepare targets
y_binary = (df['P&L'] > 0).astype(int)  # Win/Loss
y_continuous = df['P&L'].values  # P&L
y_r_multiple = df['R_Multiple'].values  # R-Multiple

print(f"   Target variables:")
print(f"     - Win/Loss (binary): {y_binary.sum()} wins, {len(y_binary) - y_binary.sum()} losses")
print(f"     - P&L (continuous): Mean ${y_continuous.mean():.2f}, Std ${y_continuous.std():.2f}")
print(f"     - R-Multiple (continuous): Mean {y_r_multiple.mean():.2f}, Std {y_r_multiple.std():.2f}")

# ============================================================================
# 3. SPLIT DATA INTO 3 BATCHES
# ============================================================================
print("\n3. Splitting data into 3 batches...")

# First split: 60% train, 40% temp
X_train, X_temp, y_binary_train, y_binary_temp, y_cont_train, y_cont_temp, y_r_train, y_r_temp = train_test_split(
    X, y_binary, y_continuous, y_r_multiple, test_size=0.4, random_state=42, stratify=y_binary
)

# Second split: Split temp into 20% validation, 20% test
X_val, X_test, y_binary_val, y_binary_test, y_cont_val, y_cont_test, y_r_val, y_r_test = train_test_split(
    X_temp, y_binary_temp, y_cont_temp, y_r_temp, test_size=0.5, random_state=42, stratify=y_binary_temp
)

print(f"   Batch 1 (Training):   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Batch 2 (Validation): {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Batch 3 (Testing):    {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 4. TRAIN MULTIPLE MODELS
# ============================================================================
print("\n" + "="*80)
print("4. TRAINING MODELS")
print("="*80)

models = {}
results = {}

# Scale features for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4.1 BINARY CLASSIFICATION (Win/Loss Prediction)
# ============================================================================
print("\n4.1 BINARY CLASSIFICATION MODELS (Win/Loss Prediction)")
print("-"*80)

# Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_binary_train)
models['RF_Classifier'] = rf_clf

# Predictions
y_pred_train = rf_clf.predict(X_train)
y_pred_val = rf_clf.predict(X_val)
y_pred_test = rf_clf.predict(X_test)

train_acc = accuracy_score(y_binary_train, y_pred_train)
val_acc = accuracy_score(y_binary_val, y_pred_val)
test_acc = accuracy_score(y_binary_test, y_pred_test)

results['RF_Classifier'] = {
    'type': 'classification',
    'train_acc': train_acc,
    'val_acc': val_acc,
    'test_acc': test_acc
}

print(f"   Training Accuracy:   {train_acc:.4f}")
print(f"   Validation Accuracy: {val_acc:.4f}")
print(f"   Test Accuracy:       {test_acc:.4f}")

# Gradient Boosting Classifier
print("\nTraining Gradient Boosting Classifier...")
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_clf.fit(X_train, y_binary_train)
models['GB_Classifier'] = gb_clf

y_pred_val = gb_clf.predict(X_val)
y_pred_test = gb_clf.predict(X_test)

val_acc = accuracy_score(y_binary_val, y_pred_val)
test_acc = accuracy_score(y_binary_test, y_pred_test)

results['GB_Classifier'] = {
    'type': 'classification',
    'train_acc': accuracy_score(y_binary_train, gb_clf.predict(X_train)),
    'val_acc': val_acc,
    'test_acc': test_acc
}

print(f"   Validation Accuracy: {val_acc:.4f}")
print(f"   Test Accuracy:       {test_acc:.4f}")

# Logistic Regression
print("\nTraining Logistic Regression...")
lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train_scaled, y_binary_train)
models['LR_Classifier'] = lr_clf

y_pred_val = lr_clf.predict(X_val_scaled)
y_pred_test = lr_clf.predict(X_test_scaled)

val_acc = accuracy_score(y_binary_val, y_pred_val)
test_acc = accuracy_score(y_binary_test, y_pred_test)

results['LR_Classifier'] = {
    'type': 'classification',
    'train_acc': accuracy_score(y_binary_train, lr_clf.predict(X_train_scaled)),
    'val_acc': val_acc,
    'test_acc': test_acc
}

print(f"   Validation Accuracy: {val_acc:.4f}")
print(f"   Test Accuracy:       {test_acc:.4f}")

# ============================================================================
# 4.2 REGRESSION MODELS (P&L Prediction)
# ============================================================================
print("\n" + "="*80)
print("4.2 REGRESSION MODELS (P&L Prediction)")
print("-"*80)

# Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_cont_train)
models['RF_Regressor'] = rf_reg

y_pred_val = rf_reg.predict(X_val)
y_pred_test = rf_reg.predict(X_test)

val_r2 = r2_score(y_cont_val, y_pred_val)
test_r2 = r2_score(y_cont_test, y_pred_test)
val_rmse = np.sqrt(mean_squared_error(y_cont_val, y_pred_val))
test_rmse = np.sqrt(mean_squared_error(y_cont_test, y_pred_test))

results['RF_Regressor'] = {
    'type': 'regression',
    'val_r2': val_r2,
    'test_r2': test_r2,
    'val_rmse': val_rmse,
    'test_rmse': test_rmse
}

print(f"   Validation R²: {val_r2:.4f}, RMSE: ${val_rmse:.2f}")
print(f"   Test R²:       {test_r2:.4f}, RMSE: ${test_rmse:.2f}")

# Gradient Boosting Regressor
print("\nTraining Gradient Boosting Regressor...")
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb_reg.fit(X_train, y_cont_train)
models['GB_Regressor'] = gb_reg

y_pred_val = gb_reg.predict(X_val)
y_pred_test = gb_reg.predict(X_test)

val_r2 = r2_score(y_cont_val, y_pred_val)
test_r2 = r2_score(y_cont_test, y_pred_test)
val_rmse = np.sqrt(mean_squared_error(y_cont_val, y_pred_val))
test_rmse = np.sqrt(mean_squared_error(y_cont_test, y_pred_test))

results['GB_Regressor'] = {
    'type': 'regression',
    'val_r2': val_r2,
    'test_r2': test_r2,
    'val_rmse': val_rmse,
    'test_rmse': test_rmse
}

print(f"   Validation R²: {val_r2:.4f}, RMSE: ${val_rmse:.2f}")
print(f"   Test R²:       {test_r2:.4f}, RMSE: ${test_rmse:.2f}")

# ============================================================================
# 5. MODEL COMPARISON AND SELECTION
# ============================================================================
print("\n" + "="*80)
print("5. MODEL COMPARISON")
print("="*80)

print("\nClassification Models (Win/Loss Prediction):")
print("-"*80)
clf_results = {k: v for k, v in results.items() if v['type'] == 'classification'}
for name, metrics in sorted(clf_results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
    print(f"\n{name}:")
    print(f"  Train Acc: {metrics['train_acc']:.4f}")
    print(f"  Val Acc:   {metrics['val_acc']:.4f}")
    print(f"  Test Acc:  {metrics['test_acc']:.4f}")

print("\nRegression Models (P&L Prediction):")
print("-"*80)
reg_results = {k: v for k, v in results.items() if v['type'] == 'regression'}
for name, metrics in sorted(reg_results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
    print(f"\n{name}:")
    print(f"  Val R²:   {metrics['val_r2']:.4f}, RMSE: ${metrics['val_rmse']:.2f}")
    print(f"  Test R²:  {metrics['test_r2']:.4f}, RMSE: ${metrics['test_rmse']:.2f}")

# Select best models
best_clf = max(clf_results.items(), key=lambda x: x[1]['test_acc'])
best_reg = max(reg_results.items(), key=lambda x: x[1]['test_r2'])

print(f"\n" + "="*80)
print("BEST MODELS SELECTED:")
print("="*80)
print(f"Classification: {best_clf[0]} (Test Acc: {best_clf[1]['test_acc']:.4f})")
print(f"Regression:      {best_reg[0]} (Test R²: {best_reg[1]['test_r2']:.4f})")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("6. FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importance from best models
if 'RF_Classifier' in models:
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': models['RF_Classifier'].feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 15 Most Important Features (Random Forest):")
    print("-"*80)
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(f"\nFeature importance saved to: feature_importance.csv")

# ============================================================================
# 7. DETAILED TEST SET EVALUATION
# ============================================================================
print("\n" + "="*80)
print("7. DETAILED TEST SET EVALUATION")
print("="*80)

# Use best classifier
best_clf_model = models[best_clf[0]]
y_pred_test = best_clf_model.predict(X_test)

print(f"\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_binary_test, y_pred_test))

print(f"\nClassification Report (Test Set):")
print(classification_report(y_binary_test, y_pred_test, target_names=['Loss', 'Win']))

# Calculate profitability if we only take predicted wins
predicted_wins = X_test[y_pred_test == 1]
actual_pnl_predicted_wins = y_cont_test[y_pred_test == 1]

print(f"\nStrategy Performance (Only taking predicted wins):")
print(f"  Total trades taken: {len(predicted_wins)}")
print(f"  Actual win rate: {(actual_pnl_predicted_wins > 0).sum() / len(actual_pnl_predicted_wins) * 100:.2f}%")
print(f"  Total P&L: ${actual_pnl_predicted_wins.sum():.2f}")
print(f"  Avg P&L: ${actual_pnl_predicted_wins.mean():.2f}")

# Compare to baseline (taking all trades)
baseline_pnl = y_cont_test.sum()
baseline_win_rate = (y_cont_test > 0).sum() / len(y_cont_test) * 100

print(f"\nBaseline (All trades):")
print(f"  Total trades: {len(y_cont_test)}")
print(f"  Win rate: {baseline_win_rate:.2f}%")
print(f"  Total P&L: ${baseline_pnl:.2f}")
print(f"  Avg P&L: ${y_cont_test.mean():.2f}")

improvement = actual_pnl_predicted_wins.sum() - (baseline_pnl * len(actual_pnl_predicted_wins) / len(y_cont_test))
print(f"\nImprovement: ${improvement:.2f} ({improvement/abs(baseline_pnl)*100:.1f}%)")

# ============================================================================
# 8. SAVE MODELS
# ============================================================================
print("\n" + "="*80)
print("8. SAVING MODELS")
print("="*80)

# Save best models
joblib.dump(models[best_clf[0]], 'best_classifier_model.pkl')
joblib.dump(models[best_reg[0]], 'best_regressor_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

# Save feature columns
pd.Series(feature_cols).to_csv('feature_columns.csv', index=False)

print(f"   Best Classifier saved: best_classifier_model.pkl")
print(f"   Best Regressor saved:  best_regressor_model.pkl")
print(f"   Scaler saved:          feature_scaler.pkl")
print(f"   Feature columns saved: feature_columns.csv")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

