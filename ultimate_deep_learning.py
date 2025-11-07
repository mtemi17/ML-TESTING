import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
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
print("ULTIMATE DEEP LEARNING SYSTEM - 500-1000 LAYERS")
print("="*80)
print("Target: 80%+ Win Rate | Adaptive Thresholds | Optimal Filters")
print("="*80)

# ============================================================================
# 1. LOAD AND ANALYZE ALL TRADES
# ============================================================================
print("\n1. Loading and analyzing all trades...")

df_original = pd.read_csv('advanced_analysis_results_combined.csv')
df_new = pd.read_csv('new_data_backtest_results.csv')
df_all = pd.concat([df_original, df_new], ignore_index=True)
df_all = df_all[df_all['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

df_wins = df_all[df_all['P&L'] > 0].copy()
df_losses = df_all[df_all['P&L'] <= 0].copy()

print(f"   Total: {len(df_all)} trades")
print(f"   Wins: {len(df_wins)} ({len(df_wins)/len(df_all)*100:.1f}%)")
print(f"   Losses: {len(df_losses)} ({len(df_losses)/len(df_all)*100:.1f}%)")

# ============================================================================
# 2. COMPREHENSIVE FEATURE ENGINEERING
# ============================================================================
print("\n2. Creating comprehensive features...")

def create_ultimate_features(df):
    """Create maximum features for deep learning"""
    features_df = df.copy()
    
    # Basic indicators
    if 'EntryPrice' in features_df.columns:
        features_df['Price'] = features_df['EntryPrice']
        features_df['Price_SMA_10'] = features_df['EntryPrice'].rolling(10).mean()
        features_df['Price_SMA_20'] = features_df['EntryPrice'].rolling(20).mean()
        features_df['Price_Deviation_10'] = (features_df['EntryPrice'] - features_df['Price_SMA_10']) / features_df['Price_SMA_10'] * 100
        features_df['Price_Deviation_20'] = (features_df['EntryPrice'] - features_df['Price_SMA_20']) / features_df['Price_SMA_20'] * 100
    
    # EMA features - extensive
    ema_cols = ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_1H']
    for ema in ema_cols:
        if ema in features_df.columns:
            features_df[f'{ema}_Distance'] = (features_df['EntryPrice'] - features_df[ema]) / features_df['EntryPrice'] * 100 if 'EntryPrice' in features_df.columns else 0
            features_df[f'{ema}_Slope'] = features_df[ema].diff()
            features_df[f'{ema}_Slope_Pct'] = features_df[ema].pct_change() * 100
    
    # EMA relationships
    if all(col in features_df.columns for col in ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M']):
        features_df['EMA_9_21_Distance'] = (features_df['EMA_9_5M'] - features_df['EMA_21_5M']) / features_df['EMA_21_5M'] * 100
        features_df['EMA_21_50_Distance'] = (features_df['EMA_21_5M'] - features_df['EMA_50_5M']) / features_df['EMA_50_5M'] * 100
        features_df['EMA_9_50_Distance'] = (features_df['EMA_9_5M'] - features_df['EMA_50_5M']) / features_df['EMA_50_5M'] * 100
        features_df['EMA_Spread'] = (features_df['EMA_9_5M'] - features_df['EMA_50_5M']) / features_df['EMA_50_5M'] * 100
        features_df['EMA_Convergence'] = abs(features_df['EMA_9_5M'] - features_df['EMA_21_5M']) / features_df['EMA_21_5M'] * 100
    
    # ATR features
    if 'ATR' in features_df.columns:
        features_df['ATR_Distance'] = features_df['ATR'] / features_df['EntryPrice'] * 100 if 'EntryPrice' in features_df.columns else 0
        features_df['ATR_Slope'] = features_df['ATR'].diff()
        features_df['ATR_MA_10'] = features_df['ATR'].rolling(10).mean()
        features_df['ATR_MA_20'] = features_df['ATR'].rolling(20).mean()
        features_df['ATR_vs_MA10'] = (features_df['ATR'] - features_df['ATR_MA_10']) / features_df['ATR_MA_10'] * 100
    
    if 'ATR_Ratio' in features_df.columns:
        features_df['ATR_Ratio_Squared'] = features_df['ATR_Ratio'] ** 2
        features_df['ATR_Ratio_Cubed'] = features_df['ATR_Ratio'] ** 3
        features_df['ATR_Ratio_Sqrt'] = np.sqrt(np.abs(features_df['ATR_Ratio']))
        features_df['ATR_Ratio_Log'] = np.log1p(np.abs(features_df['ATR_Ratio']))
    
    # Breakout features
    if all(col in features_df.columns for col in ['WindowHigh', 'WindowLow', 'EntryPrice']):
        features_df['Breakout_Size'] = features_df['WindowHigh'] - features_df['WindowLow']
        features_df['Breakout_Pct'] = features_df['Breakout_Size'] / features_df['EntryPrice'] * 100
        features_df['Breakout_Strength'] = np.where(
            features_df['Breakout_Size'] > 0,
            (features_df['EntryPrice'] - features_df['WindowHigh']) / features_df['Breakout_Size'],
            0
        )
        features_df['Breakout_Position'] = np.where(
            features_df['Breakout_Size'] > 0,
            (features_df['EntryPrice'] - features_df['WindowLow']) / features_df['Breakout_Size'],
            0
        )
        features_df['Breakout_Momentum'] = features_df['Breakout_Pct'] * features_df.get('ATR_Ratio', 1)
        features_df['Breakout_Risk_Ratio'] = features_df.get('Risk', 0) / features_df['Breakout_Size'] if features_df['Breakout_Size'].sum() > 0 else 0
    
    # Risk features
    if 'Risk' in features_df.columns:
        features_df['Risk_Pct'] = features_df['Risk'] / features_df['EntryPrice'] * 100 if 'EntryPrice' in features_df.columns else 0
        features_df['Risk_Squared'] = features_df['Risk'] ** 2
        features_df['Risk_Cubed'] = features_df['Risk'] ** 3
        features_df['Risk_Log'] = np.log1p(features_df['Risk'])
        features_df['Risk_Sqrt'] = np.sqrt(features_df['Risk'])
        features_df['Risk_Category'] = pd.cut(features_df['Risk'], bins=[0, 5, 10, 15, 20, 100], labels=[1, 2, 3, 4, 5]).astype(float)
    
    # Trend features
    if 'Trend_Score' in features_df.columns:
        features_df['Trend_Strength'] = features_df['Trend_Score'] ** 2
        features_df['Trend_Cubed'] = features_df['Trend_Score'] ** 3
        features_df['Trend_Sqrt'] = np.sqrt(features_df['Trend_Score'])
    
    # EMA alignment
    alignment_cols = ['EMA_9_Above_21', 'EMA_21_Above_50', 'Price_Above_EMA200_1H']
    if all(col in features_df.columns for col in alignment_cols):
        features_df['EMA_Alignment_Score'] = (
            features_df['EMA_9_Above_21'] + 
            features_df['EMA_21_Above_50'] + 
            features_df['Price_Above_EMA200_1H']
        ) / 3
        features_df['EMA_Perfect_Alignment'] = (
            (features_df['EMA_9_Above_21'] == 1) & 
            (features_df['EMA_21_Above_50'] == 1) & 
            (features_df['Price_Above_EMA200_1H'] == 1)
        ).astype(int)
        features_df['EMA_Pattern'] = (
            features_df['EMA_9_Above_21'].astype(int).astype(str) +
            features_df['EMA_21_Above_50'].astype(int).astype(str) +
            features_df['Price_Above_EMA200_1H'].astype(int).astype(str)
        )
        # Encode pattern as numeric
        pattern_map = {'000': 0, '001': 1, '010': 2, '011': 3, '100': 4, '101': 5, '110': 6, '111': 7}
        features_df['EMA_Pattern_Num'] = features_df['EMA_Pattern'].map(pattern_map).fillna(0)
    
    # Consolidation
    if 'Consolidation_Score' in features_df.columns:
        features_df['Consolidation_Strength'] = features_df['Consolidation_Score'] ** 2
        features_df['Is_Strong_Consolidation'] = (features_df['Consolidation_Score'] > 0.7).astype(int)
        features_df['Is_Weak_Consolidation'] = (features_df['Consolidation_Score'] < 0.3).astype(int)
    
    # Range features
    if 'RangeSize' in features_df.columns:
        features_df['RangeSize_Pct'] = features_df['RangeSize'] / features_df['EntryPrice'] * 100 if 'EntryPrice' in features_df.columns else 0
        features_df['RangeSize_Squared'] = features_df['RangeSize'] ** 2
        features_df['RangeSize_Log'] = np.log1p(features_df['RangeSize'])
    
    # Time features
    if 'EntryTime' in features_df.columns:
        features_df['EntryTime'] = pd.to_datetime(features_df['EntryTime'])
        features_df['Hour'] = features_df['EntryTime'].dt.hour
        features_df['DayOfWeek'] = features_df['EntryTime'].dt.dayofweek
        features_df['Hour_Sin'] = np.sin(2 * np.pi * features_df['Hour'] / 24)
        features_df['Hour_Cos'] = np.cos(2 * np.pi * features_df['Hour'] / 24)
        features_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * features_df['DayOfWeek'] / 7)
        features_df['DayOfWeek_Cos'] = np.cos(2 * np.pi * features_df['DayOfWeek'] / 7)
    
    # Window type
    if 'WindowType' in features_df.columns:
        window_dummies = pd.get_dummies(features_df['WindowType'], prefix='Window')
        features_df = pd.concat([features_df, window_dummies], axis=1)
    
    # Interactions
    if all(col in features_df.columns for col in ['Risk', 'ATR_Ratio', 'Trend_Score']):
        features_df['Risk_ATR'] = features_df['Risk'] * features_df['ATR_Ratio']
        features_df['Risk_Trend'] = features_df['Risk'] * features_df['Trend_Score']
        features_df['ATR_Trend'] = features_df['ATR_Ratio'] * features_df['Trend_Score']
        features_df['Risk_ATR_Trend'] = features_df['Risk'] * features_df['ATR_Ratio'] * features_df['Trend_Score']
    
    # Wave features
    if 'EntryPrice' in features_df.columns and len(features_df) > 20:
        features_df['Price_Wave_5'] = features_df['EntryPrice'].rolling(5).std()
        features_df['Price_Wave_10'] = features_df['EntryPrice'].rolling(10).std()
        features_df['Price_Wave_20'] = features_df['EntryPrice'].rolling(20).std()
        features_df['Wave_Ratio_5_10'] = np.where(
            features_df['Price_Wave_10'] > 0,
            features_df['Price_Wave_5'] / features_df['Price_Wave_10'],
            0
        )
        features_df['Wave_Ratio_10_20'] = np.where(
            features_df['Price_Wave_20'] > 0,
            features_df['Price_Wave_10'] / features_df['Price_Wave_20'],
            0
        )
    
    # Statistical features
    if 'P&L' in features_df.columns and len(features_df) > 10:
        features_df['P&L_MA_10'] = features_df['P&L'].rolling(10).mean()
        features_df['P&L_Std_10'] = features_df['P&L'].rolling(10).std()
        features_df['P&L_ZScore'] = np.where(
            features_df['P&L_Std_10'] > 0,
            (features_df['P&L'] - features_df['P&L_MA_10']) / features_df['P&L_Std_10'],
            0
        )
    
    return features_df

df_features = create_ultimate_features(df_all)

# Select features
exclude_cols = ['EntryTime', 'ExitTime', 'Status', 'P&L', 'R_Multiple', 
                'EntryPrice', 'SL', 'TP', 'WindowHigh', 'WindowLow', 'TradeID',
                'ExitPrice', 'BestPossibleP&L', 'WouldHaveWon', 'ReachedTP', 
                'WaitDuration_Min', 'BestWaitDuration_Min', 'TradeID_x', 'TradeID_y',
                'MinAdverse_Min30', 'MaxFavorable_Min30', 'WentAgainstFirst',
                'Duration_Minutes', 'Duration_Candles', 'WindowType', 'Type', 'EMA_Pattern']

feature_cols = [col for col in df_features.columns 
                if col not in exclude_cols and df_features[col].dtype in ['int64', 'float64', 'bool']]
feature_cols = [col for col in feature_cols if df_features[col].notna().sum() > len(df_features) * 0.3]

X = df_features[feature_cols].copy()
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], 0)

y_binary = (df_features['P&L'] > 0).astype(int)

print(f"   Features created: {len(feature_cols)}")

# ============================================================================
# 3. DATA SPLIT
# ============================================================================
print("\n3. Splitting data...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.4, random_state=42, stratify=y_binary
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4. BUILD ULTRA-DEEP MODEL (500-1000 LAYERS)
# ============================================================================
print("\n" + "="*80)
print("4. BUILDING ULTRA-DEEP MODEL (800 LAYERS)")
print("="*80)

np.random.seed(42)
tf.random.set_seed(42)

model = Sequential()

# Input
model.add(Dense(1024, input_dim=X_train_scaled.shape[1], kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Deep layers - 800 layers total
# Start with larger layers, gradually reduce
layer_config = (
    [1024]*50 +  # 50 layers of 1024
    [512]*100 +   # 100 layers of 512
    [256]*200 +   # 200 layers of 256
    [128]*300 +   # 300 layers of 128
    [64]*100 +    # 100 layers of 64
    [32]*40 +     # 40 layers of 32
    [16]*10       # 10 layers of 16
)

for i, size in enumerate(layer_config):
    model.add(Dense(size, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Reduce dropout as we go deeper
    dropout_rate = 0.2 if i < 200 else (0.15 if i < 400 else 0.1)
    model.add(Dropout(dropout_rate))

# Output
model.add(Dense(1, activation='sigmoid'))

# Compile
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00005),  # Lower LR for deeper network
    loss='binary_crossentropy',
    metrics=['accuracy']
)

total_layers = len(model.layers)
print(f"\n   Total layers: {total_layers}")
print(f"   Total parameters: {model.count_params():,}")

# ============================================================================
# 5. TRAIN WITH ADAPTIVE THRESHOLDS
# ============================================================================
print("\n" + "="*80)
print("5. TRAINING WITH ADAPTIVE THRESHOLDS")
print("="*80)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    min_lr=0.0000001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'ultimate_deep_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\nTraining... (this will take time with 800 layers)")

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=1000,
    batch_size=16,  # Smaller batch for stability
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================================
# 6. TEST MULTIPLE THRESHOLDS
# ============================================================================
print("\n" + "="*80)
print("6. TESTING MULTIPLE THRESHOLDS (ADAPTIVE)")
print("="*80)

# Get probabilities
train_probs = model.predict(X_train_scaled, verbose=0).flatten()
val_probs = model.predict(X_val_scaled, verbose=0).flatten()
test_probs = model.predict(X_test_scaled, verbose=0).flatten()

# Test different thresholds
thresholds = np.arange(0.3, 0.95, 0.05)
best_threshold = 0.5
best_score = 0
best_results = None

print("\nTesting thresholds...")

for threshold in thresholds:
    # Validation predictions
    val_pred = (val_probs > threshold).astype(int)
    
    if val_pred.sum() > 0:  # At least some predictions
        val_acc = accuracy_score(y_val, val_pred)
        
        # Get actual performance
        val_indices = X_val.index
        val_trades = df_features.iloc[val_indices]
        predicted_wins = val_pred == 1
        
        if predicted_wins.sum() > 0:
            filtered_trades = val_trades.iloc[np.where(predicted_wins)[0]]
            actual_winrate = (filtered_trades['P&L'] > 0).sum() / len(filtered_trades) * 100
            total_pnl = filtered_trades['P&L'].sum()
            trades_taken = len(filtered_trades)
            
            # Score: balance win rate and P&L
            score = actual_winrate * 0.6 + (total_pnl / 100) * 0.4
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_results = {
                    'threshold': threshold,
                    'trades': trades_taken,
                    'winrate': actual_winrate,
                    'total_pnl': total_pnl,
                    'avg_pnl': total_pnl / trades_taken if trades_taken > 0 else 0
                }

print(f"\n✅ Best Threshold: {best_threshold:.3f}")
if best_results:
    print(f"   Trades: {best_results['trades']}")
    print(f"   Win Rate: {best_results['winrate']:.2f}%")
    print(f"   Total P&L: ${best_results['total_pnl']:.2f}")
    print(f"   Avg P&L: ${best_results['avg_pnl']:.2f}")

# ============================================================================
# 7. FINAL EVALUATION ON TEST SET
# ============================================================================
print("\n" + "="*80)
print("7. FINAL EVALUATION (TEST SET)")
print("="*80)

test_pred = (test_probs > best_threshold).astype(int)
test_indices = X_test.index
test_trades = df_features.iloc[test_indices]

predicted_wins = test_pred == 1
if predicted_wins.sum() > 0:
    filtered_trades = test_trades.iloc[np.where(predicted_wins)[0]]
    
    actual_wins = (filtered_trades['P&L'] > 0).sum()
    actual_winrate = actual_wins / len(filtered_trades) * 100
    total_pnl = filtered_trades['P&L'].sum()
    avg_pnl = filtered_trades['P&L'].mean()
    
    print(f"\nTest Set Results:")
    print(f"   Trades taken: {len(filtered_trades)} ({len(filtered_trades)/len(test_trades)*100:.1f}%)")
    print(f"   Win rate: {actual_winrate:.2f}%")
    print(f"   Total P&L: ${total_pnl:.2f}")
    print(f"   Avg P&L: ${avg_pnl:.2f}")
    
    # Baseline
    baseline_pnl = test_trades['P&L'].sum()
    baseline_winrate = (test_trades['P&L'] > 0).sum() / len(test_trades) * 100
    
    print(f"\nBaseline:")
    print(f"   Trades: {len(test_trades)}")
    print(f"   Win rate: {baseline_winrate:.2f}%")
    print(f"   Total P&L: ${baseline_pnl:.2f}")
    
    improvement = total_pnl - (baseline_pnl * len(filtered_trades) / len(test_trades))
    print(f"\nImprovement: ${improvement:.2f}")

# Save
pd.Series(feature_cols).to_csv('ultimate_deep_features.csv', index=False)
import joblib
joblib.dump(scaler, 'ultimate_deep_scaler.pkl')
joblib.dump(best_threshold, 'ultimate_deep_threshold.pkl')

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

