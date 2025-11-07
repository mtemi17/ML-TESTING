import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
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
print("ADVANCED DEEP LEARNING SYSTEM - BLACK BOX PATTERN RECOGNITION")
print("="*80)
print("Target: 80% Win Rate | Layers: 50-100 | Comprehensive Feature Engineering")
print("="*80)

# ============================================================================
# 1. LOAD AND COMBINE ALL DATA
# ============================================================================
print("\n1. Loading and combining all data...")

df_original = pd.read_csv('advanced_analysis_results_combined.csv')
df_new = pd.read_csv('new_data_backtest_results.csv')

df_all = pd.concat([df_original, df_new], ignore_index=True)
df_all = df_all[df_all['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()

print(f"   Total trades: {len(df_all)}")
print(f"   Wins: {(df_all['P&L'] > 0).sum()}, Losses: {(df_all['P&L'] <= 0).sum()}")

# ============================================================================
# 2. COMPREHENSIVE FEATURE ENGINEERING
# ============================================================================
print("\n2. Comprehensive Feature Engineering - Building Black Box Features...")

def create_advanced_features(df):
    """Create extensive feature set for deep learning"""
    features_df = df.copy()
    
    print("   2.1 Basic Indicators...")
    # Basic price features
    if 'EntryPrice' in features_df.columns:
        features_df['Price_Level'] = features_df['EntryPrice']
        features_df['Price_SMA_20'] = features_df['EntryPrice'].rolling(20).mean()
        features_df['Price_Deviation'] = (features_df['EntryPrice'] - features_df['Price_SMA_20']) / features_df['Price_SMA_20'] * 100
    
    print("   2.2 EMA Features (Multiple Timeframes)...")
    # EMA features
    ema_features = ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_1H']
    for ema in ema_features:
        if ema in features_df.columns:
            # Distance from price
            features_df[f'{ema}_Distance'] = (features_df['EntryPrice'] - features_df[ema]) / features_df['EntryPrice'] * 100
            # EMA slope (rate of change)
            if len(features_df) > 1:
                features_df[f'{ema}_Slope'] = features_df[ema].diff()
    
    # EMA relationships
    if all(col in features_df.columns for col in ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M']):
        features_df['EMA_9_21_Distance'] = (features_df['EMA_9_5M'] - features_df['EMA_21_5M']) / features_df['EMA_21_5M'] * 100
        features_df['EMA_21_50_Distance'] = (features_df['EMA_21_5M'] - features_df['EMA_50_5M']) / features_df['EMA_50_5M'] * 100
        features_df['EMA_Spread'] = (features_df['EMA_9_5M'] - features_df['EMA_50_5M']) / features_df['EMA_50_5M'] * 100
    
    print("   2.3 ATR and Volatility Features...")
    # ATR features
    if 'ATR' in features_df.columns:
        features_df['ATR_Distance'] = features_df['ATR'] / features_df['EntryPrice'] * 100
        if 'Risk' in features_df.columns:
            features_df['ATR_Multiple'] = np.where(
                features_df['ATR'] > 0,
                features_df['Risk'] / features_df['ATR'],
                0
            )
        else:
            features_df['ATR_Multiple'] = 0
        features_df['ATR_Volatility'] = features_df['ATR'].rolling(20).std()
    
    if 'ATR_Ratio' in features_df.columns:
        features_df['ATR_Ratio_Squared'] = features_df['ATR_Ratio'] ** 2
        features_df['ATR_Ratio_Cubed'] = features_df['ATR_Ratio'] ** 3
    
    print("   2.4 Breakout Pattern Features...")
    # Breakout characteristics
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
        
        # Breakout momentum
        features_df['Breakout_Momentum'] = features_df['Breakout_Pct'] * features_df.get('ATR_Ratio', 1)
    
    print("   2.5 Risk and Reward Features...")
    # Risk features
    if 'Risk' in features_df.columns:
        features_df['Risk_Pct'] = features_df['Risk'] / features_df['EntryPrice'] * 100
        features_df['Risk_Category'] = pd.cut(features_df['Risk'], bins=[0, 5, 10, 15, 20, 100], labels=[1, 2, 3, 4, 5])
        features_df['Risk_Squared'] = features_df['Risk'] ** 2
        features_df['Risk_Log'] = np.log1p(features_df['Risk'])
    
    if all(col in features_df.columns for col in ['SL', 'TP', 'EntryPrice']):
        features_df['SL_Distance'] = abs(features_df['EntryPrice'] - features_df['SL']) / features_df['EntryPrice'] * 100
        features_df['TP_Distance'] = abs(features_df['TP'] - features_df['EntryPrice']) / features_df['EntryPrice'] * 100
        features_df['RR_Ratio'] = np.where(
            features_df['SL_Distance'] > 0,
            features_df['TP_Distance'] / features_df['SL_Distance'],
            0
        )
    
    print("   2.6 Trend and Alignment Features...")
    # Trend features
    if 'Trend_Score' in features_df.columns:
        features_df['Trend_Strength'] = features_df['Trend_Score'] ** 2
        features_df['Trend_Direction'] = features_df['Trend_Score'] > 0.5
    
    # EMA alignment score
    alignment_features = ['EMA_9_Above_21', 'EMA_21_Above_50', 'Price_Above_EMA200_1H']
    if all(col in features_df.columns for col in alignment_features):
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
    
    print("   2.7 Consolidation Features...")
    # Consolidation features
    if 'Consolidation_Score' in features_df.columns:
        features_df['Consolidation_Strength'] = features_df['Consolidation_Score'] ** 2
        features_df['Is_Strong_Consolidation'] = (features_df['Consolidation_Score'] > 0.7).astype(int)
        features_df['Is_Weak_Consolidation'] = (features_df['Consolidation_Score'] < 0.3).astype(int)
    
    print("   2.8 Mathematical Transformations...")
    # Mathematical transformations
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in ['Risk', 'ATR', 'Breakout_Size', 'ATR_Ratio']:
        if col in features_df.columns:
            # Square root
            features_df[f'{col}_Sqrt'] = np.sqrt(np.abs(features_df[col]))
            # Logarithm
            features_df[f'{col}_Log'] = np.log1p(np.abs(features_df[col]))
            # Exponential
            features_df[f'{col}_Exp'] = np.exp(np.clip(features_df[col], -5, 5))
    
    print("   2.9 Time-Based Features...")
    # Time features
    if 'EntryTime' in features_df.columns:
        features_df['EntryTime'] = pd.to_datetime(features_df['EntryTime'])
        features_df['Hour'] = features_df['EntryTime'].dt.hour
        features_df['DayOfWeek'] = features_df['EntryTime'].dt.dayofweek
        features_df['DayOfMonth'] = features_df['EntryTime'].dt.day
        features_df['Month'] = features_df['EntryTime'].dt.month
        
        # Cyclical encoding
        features_df['Hour_Sin'] = np.sin(2 * np.pi * features_df['Hour'] / 24)
        features_df['Hour_Cos'] = np.cos(2 * np.pi * features_df['Hour'] / 24)
        features_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * features_df['DayOfWeek'] / 7)
        features_df['DayOfWeek_Cos'] = np.cos(2 * np.pi * features_df['DayOfWeek'] / 7)
    
    print("   2.10 Window Type Features...")
    # Window type encoding
    if 'WindowType' in features_df.columns:
        window_dummies = pd.get_dummies(features_df['WindowType'], prefix='Window')
        features_df = pd.concat([features_df, window_dummies], axis=1)
    
    print("   2.11 Interaction Features...")
    # Feature interactions
    if all(col in features_df.columns for col in ['Risk', 'ATR_Ratio', 'Trend_Score']):
        features_df['Risk_ATR_Interaction'] = features_df['Risk'] * features_df['ATR_Ratio']
        features_df['Risk_Trend_Interaction'] = features_df['Risk'] * features_df['Trend_Score']
        features_df['ATR_Trend_Interaction'] = features_df['ATR_Ratio'] * features_df['Trend_Score']
        features_df['Risk_ATR_Trend'] = features_df['Risk'] * features_df['ATR_Ratio'] * features_df['Trend_Score']
    
    print("   2.12 Wave and Pattern Features...")
    # Wave-like patterns (using rolling statistics)
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
    
    print("   2.13 Statistical Features...")
    # Statistical features
    if 'P&L' in features_df.columns and len(features_df) > 10:
        features_df['P&L_MA_10'] = features_df['P&L'].rolling(10).mean()
        features_df['P&L_Std_10'] = features_df['P&L'].rolling(10).std()
        features_df['P&L_ZScore'] = np.where(
            features_df['P&L_Std_10'] > 0,
            (features_df['P&L'] - features_df['P&L_MA_10']) / features_df['P&L_Std_10'],
            0
        )
    
    print("   2.14 Polynomial Features (Key Interactions)...")
    # Polynomial features for key variables
    key_vars = ['Risk', 'ATR_Ratio', 'Trend_Score', 'Breakout_Pct']
    for var in key_vars:
        if var in features_df.columns:
            features_df[f'{var}_Squared'] = features_df[var] ** 2
            features_df[f'{var}_Cubed'] = features_df[var] ** 3
    
    print("   2.15 Normalized Features...")
    # Normalize key features
    normalize_cols = ['Risk', 'ATR', 'Breakout_Size', 'ATR_Ratio']
    for col in normalize_cols:
        if col in features_df.columns:
            col_mean = features_df[col].mean()
            col_std = features_df[col].std()
            if col_std > 0:
                features_df[f'{col}_Normalized'] = (features_df[col] - col_mean) / col_std
    
    return features_df

# Apply feature engineering
df_features = create_advanced_features(df_all)

# Select numeric features only
exclude_cols = ['EntryTime', 'ExitTime', 'Status', 'P&L', 'R_Multiple', 
                'EntryPrice', 'SL', 'TP', 'WindowHigh', 'WindowLow', 'TradeID',
                'ExitPrice', 'BestPossibleP&L', 'WouldHaveWon', 'ReachedTP', 
                'WaitDuration_Min', 'BestWaitDuration_Min', 'TradeID_x', 'TradeID_y',
                'MinAdverse_Min30', 'MaxFavorable_Min30', 'WentAgainstFirst',
                'Duration_Minutes', 'Duration_Candles', 'WindowType', 'Type']

feature_cols = [col for col in df_features.columns 
                if col not in exclude_cols and df_features[col].dtype in ['int64', 'float64', 'bool']]
feature_cols = [col for col in feature_cols if df_features[col].notna().sum() > len(df_features) * 0.3]

print(f"\n   Total features created: {len(feature_cols)}")

# Prepare data
X = df_features[feature_cols].copy()
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], 0)

y_binary = (df_features['P&L'] > 0).astype(int)

print(f"   Final feature count: {len(feature_cols)}")
print(f"   Target distribution: {y_binary.sum()} wins, {len(y_binary) - y_binary.sum()} losses")

# ============================================================================
# 3. DATA SPLITTING
# ============================================================================
print("\n3. Splitting data...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.4, random_state=42, stratify=y_binary
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Training: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 4. SCALING
# ============================================================================
print("\n4. Scaling features...")

scaler = RobustScaler()  # More robust to outliers
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"   Features scaled: {X_train_scaled.shape[1]}")

# ============================================================================
# 5. BUILD DEEP NEURAL NETWORK (50-100 LAYERS)
# ============================================================================
print("\n" + "="*80)
print("5. BUILDING DEEP NEURAL NETWORK (75 LAYERS)")
print("="*80)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Build very deep network
model = Sequential()

# Input layer
model.add(Dense(512, input_dim=X_train_scaled.shape[1], kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

# Deep layers (75 layers total)
layer_sizes = [512, 512, 256, 256, 256, 128, 128, 128, 128, 128,
               128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
               128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
               128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
               128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
               128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
               128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
               64, 64, 32, 16, 8]

for i, size in enumerate(layer_sizes):
    model.add(Dense(size, kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2 if i < 50 else 0.1))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\n   Total layers: {len(model.layers)}")
print(f"   Total parameters: {model.count_params():,}")

model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=0.000001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_deep_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ============================================================================
# 6. TRAIN MODEL
# ============================================================================
print("\n" + "="*80)
print("6. TRAINING DEEP MODEL")
print("="*80)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=500,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================================
# 7. EVALUATE
# ============================================================================
print("\n" + "="*80)
print("7. EVALUATION")
print("="*80)

# Predictions
train_pred = (model.predict(X_train_scaled, verbose=0) > 0.5).astype(int).flatten()
val_pred = (model.predict(X_val_scaled, verbose=0) > 0.5).astype(int).flatten()
test_pred = (model.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nAccuracy:")
print(f"   Training:   {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Validation: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Test:       {test_acc:.4f} ({test_acc*100:.2f}%)")

# Strategy performance
test_indices = X_test.index
test_trades = df_features.iloc[test_indices]

# Get predicted wins
predicted_wins = test_pred == 1
if predicted_wins.sum() > 0:
    filtered_trades = test_trades.iloc[np.where(predicted_wins)[0]]
    
    actual_wins = (filtered_trades['P&L'] > 0).sum()
    actual_winrate = actual_wins / len(filtered_trades) * 100
    total_pnl = filtered_trades['P&L'].sum()
    avg_pnl = filtered_trades['P&L'].mean()
    
    print(f"\nStrategy Performance (Test Set):")
    print(f"   Trades taken: {len(filtered_trades)} ({len(filtered_trades)/len(test_trades)*100:.1f}%)")
    print(f"   Actual win rate: {actual_winrate:.2f}%")
    print(f"   Total P&L: ${total_pnl:.2f}")
    print(f"   Avg P&L: ${avg_pnl:.2f}")
    
    # Baseline
    baseline_pnl = test_trades['P&L'].sum()
    baseline_winrate = (test_trades['P&L'] > 0).sum() / len(test_trades) * 100
    
    print(f"\nBaseline (All Test Trades):")
    print(f"   Total trades: {len(test_trades)}")
    print(f"   Win rate: {baseline_winrate:.2f}%")
    print(f"   Total P&L: ${baseline_pnl:.2f}")
    
    improvement = total_pnl - (baseline_pnl * len(filtered_trades) / len(test_trades))
    print(f"\nImprovement: ${improvement:.2f}")

# Save
pd.Series(feature_cols).to_csv('deep_model_features.csv', index=False)
joblib.dump(scaler, 'deep_model_scaler.pkl')
print("\n   Model files saved:")
print("   - best_deep_model.h5")
print("   - deep_model_features.csv")
print("   - deep_model_scaler.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

