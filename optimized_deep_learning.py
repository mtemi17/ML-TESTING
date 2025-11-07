import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available")
    exit(1)

print("="*80)
print("OPTIMIZED DEEP LEARNING - USING DISCOVERED PATTERNS")
print("="*80)
print("Target: 80%+ Win Rate | Using Optimal Filter Combinations")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")

df_original = pd.read_csv('advanced_analysis_results_combined.csv')
df_new = pd.read_csv('new_data_backtest_results.csv')
df_all = pd.concat([df_original, df_new], ignore_index=True)
df_all = df_all[df_all['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

print(f"   Total trades: {len(df_all)}")

# ============================================================================
# 2. ADD OPTIMAL FILTER FEATURES
# ============================================================================
print("\n2. Adding optimal filter features based on analysis...")

# EMA Pattern
df_all['EMA_Pattern'] = (
    df_all['EMA_9_Above_21'].astype(int).astype(str) +
    df_all['EMA_21_Above_50'].astype(int).astype(str) +
    df_all['Price_Above_EMA200_1H'].astype(int).astype(str)
)

# Optimal filter flags (based on analysis)
df_all['Optimal_ATR_Low'] = (df_all['ATR_Ratio'] <= 0.8).astype(int)  # 62.2% win rate
df_all['Optimal_EMA_Pattern_110'] = (df_all['EMA_Pattern'] == '110').astype(int)  # 41% win rate
df_all['Optimal_EMA_Pattern_111'] = (df_all['EMA_Pattern'] == '111').astype(int)  # 41% win rate
df_all['Optimal_Risk_High'] = (df_all['Risk'] >= 10).astype(int)  # Better performance
df_all['Optimal_Trend_Strong'] = (df_all['Trend_Score'] >= 0.6).astype(int)  # Better performance
df_all['Optimal_Range_Large'] = (df_all['RangeSize'] >= 10).astype(int)  # 45.9% win rate

# Combined optimal filters
df_all['Optimal_Combo_1'] = (
    (df_all['EMA_Pattern'] == '110') & 
    (df_all['Risk'] >= 10) &
    (df_all['Trend_Score'] >= 0.6)
).astype(int)  # 55% win rate

df_all['Optimal_Combo_2'] = (
    (df_all['Risk'] >= 10) &
    (df_all['Trend_Score'] >= 0.6)
).astype(int)  # 50.9% win rate

df_all['Optimal_Combo_3'] = (
    (df_all['EMA_Pattern'] == '110') & 
    (df_all['Risk'] >= 12) &
    (df_all['Trend_Score'] >= 0.6)
).astype(int)  # 50% win rate

# ============================================================================
# 3. COMPREHENSIVE FEATURES
# ============================================================================
print("\n3. Creating comprehensive features...")

# Basic features
if 'EntryPrice' in df_all.columns:
    df_all['Price'] = df_all['EntryPrice']
    df_all['Price_SMA_20'] = df_all['EntryPrice'].rolling(20).mean()
    df_all['Price_Deviation'] = (df_all['EntryPrice'] - df_all['Price_SMA_20']) / df_all['Price_SMA_20'] * 100

# EMA features
ema_cols = ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_1H']
for ema in ema_cols:
    if ema in df_all.columns:
        df_all[f'{ema}_Distance'] = (df_all['EntryPrice'] - df_all[ema]) / df_all['EntryPrice'] * 100 if 'EntryPrice' in df_all.columns else 0
        df_all[f'{ema}_Slope'] = df_all[ema].diff()

# EMA relationships
if all(col in df_all.columns for col in ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M']):
    df_all['EMA_9_21_Distance'] = (df_all['EMA_9_5M'] - df_all['EMA_21_5M']) / df_all['EMA_21_5M'] * 100
    df_all['EMA_21_50_Distance'] = (df_all['EMA_21_5M'] - df_all['EMA_50_5M']) / df_all['EMA_50_5M'] * 100
    df_all['EMA_Spread'] = (df_all['EMA_9_5M'] - df_all['EMA_50_5M']) / df_all['EMA_50_5M'] * 100

# ATR features
if 'ATR' in df_all.columns:
    df_all['ATR_Distance'] = df_all['ATR'] / df_all['EntryPrice'] * 100 if 'EntryPrice' in df_all.columns else 0
    df_all['ATR_MA_20'] = df_all['ATR'].rolling(20).mean()
    df_all['ATR_vs_MA'] = (df_all['ATR'] - df_all['ATR_MA_20']) / df_all['ATR_MA_20'] * 100

if 'ATR_Ratio' in df_all.columns:
    df_all['ATR_Ratio_Squared'] = df_all['ATR_Ratio'] ** 2
    df_all['ATR_Ratio_Log'] = np.log1p(df_all['ATR_Ratio'])

# Breakout features
if all(col in df_all.columns for col in ['WindowHigh', 'WindowLow', 'EntryPrice']):
    df_all['Breakout_Size'] = df_all['WindowHigh'] - df_all['WindowLow']
    df_all['Breakout_Pct'] = df_all['Breakout_Size'] / df_all['EntryPrice'] * 100
    df_all['Breakout_Momentum'] = df_all['Breakout_Pct'] * df_all.get('ATR_Ratio', 1)

# Risk features
if 'Risk' in df_all.columns:
    df_all['Risk_Pct'] = df_all['Risk'] / df_all['EntryPrice'] * 100 if 'EntryPrice' in df_all.columns else 0
    df_all['Risk_Log'] = np.log1p(df_all['Risk'])
    df_all['Risk_Squared'] = df_all['Risk'] ** 2

# Trend features
if 'Trend_Score' in df_all.columns:
    df_all['Trend_Strength'] = df_all['Trend_Score'] ** 2
    df_all['EMA_Alignment_Score'] = (
        df_all['EMA_9_Above_21'] + 
        df_all['EMA_21_Above_50'] + 
        df_all['Price_Above_EMA200_1H']
    ) / 3

# Range features
if 'RangeSize' in df_all.columns:
    df_all['RangeSize_Pct'] = df_all['RangeSize'] / df_all['EntryPrice'] * 100 if 'EntryPrice' in df_all.columns else 0
    df_all['RangeSize_Log'] = np.log1p(df_all['RangeSize'])

# Time features
if 'EntryTime' in df_all.columns:
    df_all['EntryTime'] = pd.to_datetime(df_all['EntryTime'])
    df_all['Hour'] = df_all['EntryTime'].dt.hour
    df_all['DayOfWeek'] = df_all['EntryTime'].dt.dayofweek
    df_all['Hour_Sin'] = np.sin(2 * np.pi * df_all['Hour'] / 24)
    df_all['Hour_Cos'] = np.cos(2 * np.pi * df_all['Hour'] / 24)

# Window type
if 'WindowType' in df_all.columns:
    window_dummies = pd.get_dummies(df_all['WindowType'], prefix='Window')
    df_all = pd.concat([df_all, window_dummies], axis=1)

# Interactions
if all(col in df_all.columns for col in ['Risk', 'ATR_Ratio', 'Trend_Score']):
    df_all['Risk_ATR'] = df_all['Risk'] * df_all['ATR_Ratio']
    df_all['Risk_Trend'] = df_all['Risk'] * df_all['Trend_Score']
    df_all['ATR_Trend'] = df_all['ATR_Ratio'] * df_all['Trend_Score']
    df_all['Risk_ATR_Trend'] = df_all['Risk'] * df_all['ATR_Ratio'] * df_all['Trend_Score']

# Select features
exclude_cols = ['EntryTime', 'ExitTime', 'Status', 'P&L', 'R_Multiple', 
                'EntryPrice', 'SL', 'TP', 'WindowHigh', 'WindowLow', 'TradeID',
                'ExitPrice', 'BestPossibleP&L', 'WouldHaveWon', 'ReachedTP', 
                'WaitDuration_Min', 'BestWaitDuration_Min', 'TradeID_x', 'TradeID_y',
                'MinAdverse_Min30', 'MaxFavorable_Min30', 'WentAgainstFirst',
                'Duration_Minutes', 'Duration_Candles', 'WindowType', 'Type', 'EMA_Pattern']

feature_cols = [col for col in df_all.columns 
                if col not in exclude_cols and df_all[col].dtype in ['int64', 'float64', 'bool']]
feature_cols = [col for col in feature_cols if df_all[col].notna().sum() > len(df_all) * 0.3]

X = df_all[feature_cols].copy()
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], 0)

y_binary = (df_all['P&L'] > 0).astype(int)

print(f"   Features: {len(feature_cols)}")

# ============================================================================
# 4. DATA SPLIT
# ============================================================================
print("\n4. Splitting data...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.4, random_state=42, stratify=y_binary
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 5. BUILD DEEP MODEL (500-1000 LAYERS)
# ============================================================================
print("\n" + "="*80)
print("5. BUILDING ULTRA-DEEP MODEL (1000 LAYERS)")
print("="*80)

np.random.seed(42)
tf.random.set_seed(42)

model = Sequential()

# Input
model.add(Dense(2048, input_dim=X_train_scaled.shape[1], kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.15))

# Ultra-deep layers - 1000 layers
layer_config = (
    [2048]*100 +  # 100 layers
    [1024]*150 +  # 150 layers
    [512]*200 +   # 200 layers
    [256]*250 +   # 250 layers
    [128]*200 +   # 200 layers
    [64]*80 +     # 80 layers
    [32]*20       # 20 layers
)

for i, size in enumerate(layer_config):
    model.add(Dense(size, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    dropout_rate = 0.15 if i < 300 else (0.1 if i < 600 else 0.05)
    model.add(Dropout(dropout_rate))

# Output
model.add(Dense(1, activation='sigmoid'))

# Compile
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # Very low LR for stability
    loss='binary_crossentropy',
    metrics=['accuracy']
)

total_layers = len(model.layers)
print(f"\n   Total layers: {total_layers}")
print(f"   Total parameters: {model.count_params():,}")

# ============================================================================
# 6. TRAIN
# ============================================================================
print("\n" + "="*80)
print("6. TRAINING")
print("="*80)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=150,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=30,
    min_lr=0.00000001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'optimized_deep_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=2000,
    batch_size=8,  # Very small batch for ultra-deep network
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================================
# 7. TEST MULTIPLE THRESHOLDS
# ============================================================================
print("\n" + "="*80)
print("7. TESTING ADAPTIVE THRESHOLDS")
print("="*80)

val_probs = model.predict(X_val_scaled, verbose=0).flatten()
test_probs = model.predict(X_test_scaled, verbose=0).flatten()

thresholds = np.arange(0.2, 0.95, 0.02)
best_threshold = 0.5
best_score = 0
best_results = None

for threshold in thresholds:
    val_pred = (val_probs > threshold).astype(int)
    
    if val_pred.sum() > 0:
        val_indices = X_val.index
        val_trades = df_all.iloc[val_indices]
        predicted_wins = val_pred == 1
        
        if predicted_wins.sum() > 0:
            filtered = val_trades.iloc[np.where(predicted_wins)[0]]
            winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
            pnl = filtered['P&L'].sum()
            trades = len(filtered)
            
            # Score: prioritize win rate (80% target)
            score = winrate * 0.7 + (pnl / 100) * 0.3
            
            if score > best_score and winrate >= 50:  # At least 50% win rate
                best_score = score
                best_threshold = threshold
                best_results = {
                    'threshold': threshold,
                    'trades': trades,
                    'winrate': winrate,
                    'total_pnl': pnl,
                    'avg_pnl': pnl / trades if trades > 0 else 0
                }

print(f"\n✅ Best Threshold: {best_threshold:.3f}")
if best_results:
    print(f"   Trades: {best_results['trades']}")
    print(f"   Win Rate: {best_results['winrate']:.2f}%")
    print(f"   Total P&L: ${best_results['total_pnl']:.2f}")

# Test set evaluation
test_pred = (test_probs > best_threshold).astype(int)
test_indices = X_test.index
test_trades = df_all.iloc[test_indices]

predicted_wins = test_pred == 1
if predicted_wins.sum() > 0:
    filtered = test_trades.iloc[np.where(predicted_wins)[0]]
    
    winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
    pnl = filtered['P&L'].sum()
    avg_pnl = filtered['P&L'].mean()
    
    print(f"\n📊 Test Set Results:")
    print(f"   Trades: {len(filtered)} ({len(filtered)/len(test_trades)*100:.1f}%)")
    print(f"   Win Rate: {winrate:.2f}%")
    print(f"   Total P&L: ${pnl:.2f}")
    print(f"   Avg P&L: ${avg_pnl:.2f}")

# Save
pd.Series(feature_cols).to_csv('optimized_deep_features.csv', index=False)
import joblib
joblib.dump(scaler, 'optimized_deep_scaler.pkl')
joblib.dump(best_threshold, 'optimized_deep_threshold.pkl')

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

