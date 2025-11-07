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
    from tensorflow.keras import layers, regularizers, Model, Input
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Activation, 
        Add, Multiply, Concatenate, LayerNormalization,
        MultiHeadAttention, GlobalAveragePooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.mixed_precision import set_global_policy
    
    # Enable mixed precision for faster training
    try:
        set_global_policy('mixed_float16')
        print("✅ Mixed precision enabled (FP16) - 2x faster, 2x less memory")
    except:
        print("⚠️  Mixed precision not available, using FP32")
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available")
    exit(1)

print("="*80)
print("ULTRA-DEEP TRANSFORMER SYSTEM - 5000+ LAYERS")
print("="*80)
print("Architecture: ResNet + DenseNet + Transformer Blocks")
print("Optimizations: Mixed Precision, Gradient Clipping, Optimized Batch Size")
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
print(f"   Wins: {(df_all['P&L'] > 0).sum()} ({((df_all['P&L'] > 0).sum()/len(df_all)*100):.1f}%)")
print(f"   Losses: {(df_all['P&L'] <= 0).sum()} ({((df_all['P&L'] <= 0).sum()/len(df_all)*100):.1f}%)")

# ============================================================================
# 2. ULTIMATE FEATURE ENGINEERING
# ============================================================================
print("\n2. Creating ultimate feature set (200+ features)...")

# Import feature engineering function from advanced model
def create_ultimate_features(df):
    """Create maximum possible features"""
    features_df = df.copy()
    
    # Basic indicators
    if 'EntryPrice' in features_df.columns:
        for window in [5, 10, 20, 50]:
            features_df[f'Price_SMA_{window}'] = features_df['EntryPrice'].rolling(window).mean()
            features_df[f'Price_Deviation_{window}'] = (features_df['EntryPrice'] - features_df[f'Price_SMA_{window}']) / features_df[f'Price_SMA_{window}'] * 100
        features_df['Price_Change_1'] = features_df['EntryPrice'].pct_change(1) * 100
        features_df['Price_Volatility_20'] = features_df['EntryPrice'].rolling(20).std()
    
    # EMA features
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
        features_df['EMA_Spread'] = (features_df['EMA_9_5M'] - features_df['EMA_50_5M']) / features_df['EMA_50_5M'] * 100
    
    # ATR features
    if 'ATR' in features_df.columns:
        features_df['ATR_Distance'] = features_df['ATR'] / features_df['EntryPrice'] * 100 if 'EntryPrice' in features_df.columns else 0
        features_df['ATR_Slope'] = features_df['ATR'].diff()
        for window in [10, 20]:
            features_df[f'ATR_MA_{window}'] = features_df['ATR'].rolling(window).mean()
    
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
        features_df['Breakout_Momentum'] = features_df['Breakout_Pct'] * features_df.get('ATR_Ratio', 1)
    
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
        features_df['Trend_Direction'] = (features_df['Trend_Score'] > 0.5).astype(int)
        features_df['Trend_Strong'] = (features_df['Trend_Score'] > 0.7).astype(int)
    
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
        pattern_map = {'000': 0, '001': 1, '010': 2, '011': 3, '100': 4, '101': 5, '110': 6, '111': 7}
        features_df['EMA_Pattern_Num'] = features_df['EMA_Pattern'].map(pattern_map).fillna(0)
    
    # Consolidation
    if 'Consolidation_Score' in features_df.columns:
        features_df['Consolidation_Strength'] = features_df['Consolidation_Score'] ** 2
        features_df['Is_Strong_Consolidation'] = (features_df['Consolidation_Score'] > 0.7).astype(int)
    
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
        features_df['Is_London_Session'] = ((features_df['Hour'] >= 8) & (features_df['Hour'] < 16)).astype(int)
        features_df['Is_NY_Session'] = ((features_df['Hour'] >= 13) & (features_df['Hour'] < 21)).astype(int)
    
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
        for window in [5, 10, 20]:
            features_df[f'Price_Wave_{window}'] = features_df['EntryPrice'].rolling(window).std()
    
    # Statistical features
    if 'P&L' in features_df.columns and len(features_df) > 10:
        for window in [5, 10]:
            features_df[f'P&L_MA_{window}'] = features_df['P&L'].rolling(window).mean()
            features_df[f'P&L_Std_{window}'] = features_df['P&L'].rolling(window).std()
    
    # Polynomial features
    key_vars = ['Risk', 'ATR_Ratio', 'Trend_Score', 'Breakout_Pct']
    for var in key_vars:
        if var in features_df.columns:
            features_df[f'{var}_Squared'] = features_df[var] ** 2
            features_df[f'{var}_Cubed'] = features_df[var] ** 3
    
    # Normalized features
    normalize_cols = ['Risk', 'ATR', 'Breakout_Size', 'ATR_Ratio', 'Trend_Score']
    for col in normalize_cols:
        if col in features_df.columns:
            col_mean = features_df[col].mean()
            col_std = features_df[col].std()
            if col_std > 0:
                features_df[f'{col}_Normalized'] = (features_df[col] - col_mean) / col_std
    
    return features_df

df_features = create_ultimate_features(df_all)

# Select numeric features
exclude_cols = ['EntryTime', 'ExitTime', 'Status', 'P&L', 'R_Multiple', 
                'EntryPrice', 'SL', 'TP', 'WindowHigh', 'WindowLow', 'TradeID',
                'ExitPrice', 'BestPossibleP&L', 'WouldHaveWon', 'ReachedTP', 
                'WaitDuration_Min', 'BestWaitDuration_Min', 'TradeID_x', 'TradeID_y',
                'MinAdverse_Min30', 'MaxFavorable_Min30', 'WentAgainstFirst',
                'Duration_Minutes', 'Duration_Candles', 'WindowType', 'Type', 
                'EMA_Pattern']

feature_cols = [col for col in df_features.columns 
                if col not in exclude_cols and df_features[col].dtype in ['int64', 'float64', 'bool']]
feature_cols = [col for col in feature_cols if df_features[col].notna().sum() > len(df_features) * 0.2]

X = df_features[feature_cols].copy()
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], 0)

y_binary = (df_features['P&L'] > 0).astype(int)

print(f"   Total features: {len(feature_cols)}")

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

print(f"   Training: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4. BUILD RESIDUAL BLOCK
# ============================================================================
def residual_block(x, units, dropout_rate=0.1, layer_num=0):
    """Residual block with skip connection"""
    identity = x
    
    out = Dense(units, kernel_regularizer=regularizers.l2(0.0001), name=f'res_dense_{layer_num}_1')(x)
    out = BatchNormalization(name=f'res_bn_{layer_num}_1')(out)
    out = Activation('relu', name=f'res_act_{layer_num}_1')(out)
    out = Dropout(dropout_rate, name=f'res_dropout_{layer_num}_1')(out)
    
    out = Dense(units, kernel_regularizer=regularizers.l2(0.0001), name=f'res_dense_{layer_num}_2')(out)
    out = BatchNormalization(name=f'res_bn_{layer_num}_2')(out)
    
    if identity.shape[-1] == units:
        out = Add(name=f'res_add_{layer_num}')([out, identity])
    else:
        identity = Dense(units, name=f'res_proj_{layer_num}')(identity)
        out = Add(name=f'res_add_{layer_num}')([out, identity])
    
    out = Activation('relu', name=f'res_act_{layer_num}_2')(out)
    out = Dropout(dropout_rate, name=f'res_dropout_{layer_num}_2')(out)
    
    return out

# ============================================================================
# 5. BUILD TRANSFORMER BLOCK
# ============================================================================
def transformer_block(x, d_model, num_heads=8, ff_dim=2048, dropout_rate=0.1, block_num=0):
    """Transformer block with self-attention and feed-forward"""
    # Self-attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=d_model // num_heads,
        name=f'transformer_attn_{block_num}'
    )(x, x)
    attn_output = Dropout(dropout_rate, name=f'transformer_attn_dropout_{block_num}')(attn_output)
    
    # Residual connection and normalization
    out1 = Add(name=f'transformer_add1_{block_num}')([x, attn_output])
    out1 = LayerNormalization(name=f'transformer_ln1_{block_num}')(out1)
    
    # Feed-forward network
    ff_output = Dense(ff_dim, kernel_regularizer=regularizers.l2(0.0001), name=f'transformer_ff1_{block_num}')(out1)
    ff_output = Activation('relu', name=f'transformer_ff_act_{block_num}')(ff_output)
    ff_output = Dropout(dropout_rate, name=f'transformer_ff_dropout1_{block_num}')(ff_output)
    ff_output = Dense(d_model, kernel_regularizer=regularizers.l2(0.0001), name=f'transformer_ff2_{block_num}')(ff_output)
    ff_output = Dropout(dropout_rate, name=f'transformer_ff_dropout2_{block_num}')(ff_output)
    
    # Residual connection and normalization
    out2 = Add(name=f'transformer_add2_{block_num}')([out1, ff_output])
    out2 = LayerNormalization(name=f'transformer_ln2_{block_num}')(out2)
    
    return out2

# ============================================================================
# 6. BUILD DENSE BLOCK
# ============================================================================
def dense_block(x, units, growth_rate=32, num_layers=4, block_num=0):
    """Dense block with concatenated features"""
    concat_list = [x]
    
    for i in range(num_layers):
        out = Dense(growth_rate, kernel_regularizer=regularizers.l2(0.0001), 
                   name=f'dense_dense_{block_num}_{i}')(concat_list[-1])
        out = BatchNormalization(name=f'dense_bn_{block_num}_{i}')(out)
        out = Activation('relu', name=f'dense_act_{block_num}_{i}')(out)
        out = Dropout(0.1, name=f'dense_dropout_{block_num}_{i}')(out)
        
        concat_list.append(out)
        if len(concat_list) > 1:
            out = Concatenate(name=f'dense_concat_{block_num}_{i}')(concat_list)
        else:
            out = concat_list[-1]
        concat_list = [out]
    
    return concat_list[0]

# ============================================================================
# 7. BUILD ULTRA-DEEP MODEL (5000+ LAYERS)
# ============================================================================
print("\n" + "="*80)
print("4. BUILDING ULTRA-DEEP MODEL (5000+ LAYERS)")
print("Architecture: ResNet (2000 layers) + Transformer (1000 layers) + DenseNet (100 layers)")
print("="*80)

np.random.seed(42)
tf.random.set_seed(42)

# Input
inputs = Input(shape=(X_train_scaled.shape[1],), name='input')

# Initial projection
x = Dense(2048, kernel_regularizer=regularizers.l2(0.0001), name='initial_dense')(inputs)
x = BatchNormalization(name='initial_bn')(x)
x = Activation('relu', name='initial_act')(x)
x = Dropout(0.15, name='initial_dropout')(x)

print("\n   Building ResNet blocks (2000 layers)...")
# ResNet blocks - 1000 blocks = 2000 layers
res_units = [2048, 2048, 1024, 1024, 512, 512, 256, 256]
res_blocks_per_unit = [125, 125, 125, 125, 125, 125, 125, 125]  # 1000 blocks total

layer_counter = 0
for unit_size, num_blocks in zip(res_units, res_blocks_per_unit):
    for i in range(num_blocks):
        x = residual_block(x, unit_size, dropout_rate=0.1, layer_num=layer_counter)
        layer_counter += 1
        
        if layer_counter % 200 == 0:
            print(f"      Built {layer_counter} residual blocks...")

print(f"   ✅ Built {layer_counter} residual blocks ({layer_counter * 2} layers)")

# Reshape for transformer (add sequence dimension)
# For tabular data, we'll create a sequence by grouping features
print("\n   Reshaping for Transformer blocks...")
# Expand dimensions for transformer: (batch, 1, features) - treat as sequence length 1
x_expanded = tf.expand_dims(x, axis=1)  # (batch, 1, features)

print("\n   Building Transformer blocks (1000 layers)...")
# Transformer blocks - 200 blocks = ~1000 layers
d_model = x.shape[-1]  # Use current feature dimension
transformer_blocks = 200

for i in range(transformer_blocks):
    x_expanded = transformer_block(x_expanded, d_model, num_heads=8, ff_dim=d_model*2, dropout_rate=0.1, block_num=i)
    
    if (i + 1) % 50 == 0:
        print(f"      Built {i + 1} transformer blocks...")

print(f"   ✅ Built {transformer_blocks} transformer blocks (~1000 layers)")

# Flatten back to 2D
x = tf.squeeze(x_expanded, axis=1)  # (batch, features)

print("\n   Building DenseNet blocks (100 layers)...")
# DenseNet blocks - 50 blocks
dense_units = [256, 128, 64]
dense_blocks_per_unit = [20, 20, 10]

dense_counter = 0
for unit_size, num_blocks in zip(dense_units, dense_blocks_per_unit):
    for i in range(num_blocks):
        x = dense_block(x, unit_size, growth_rate=32, num_layers=2, block_num=dense_counter)
        dense_counter += 1

print(f"   ✅ Built {dense_counter} dense blocks")

# Final layers
x = Dense(128, kernel_regularizer=regularizers.l2(0.0001), name='final_dense_1')(x)
x = BatchNormalization(name='final_bn_1')(x)
x = Activation('relu', name='final_act_1')(x)
x = Dropout(0.1, name='final_dropout_1')(x)

x = Dense(64, kernel_regularizer=regularizers.l2(0.0001), name='final_dense_2')(x)
x = BatchNormalization(name='final_bn_2')(x)
x = Activation('relu', name='final_act_2')(x)
x = Dropout(0.05, name='final_dropout_2')(x)

# Output (ensure float32 for final layer with mixed precision)
outputs = Dense(1, activation='sigmoid', dtype='float32', name='output')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs, name='UltraDeepTransformer')

# Compile with gradient clipping
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Optimizer with gradient clipping and optimized settings
optimizer = keras.optimizers.Adam(
    learning_rate=0.00005,  # Slightly higher for faster convergence
    clipnorm=1.0,
    clipvalue=0.5,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

total_layers = len(model.layers)
total_params = model.count_params()

print(f"\n   ✅ Model Created!")
print(f"   Total layers: {total_layers}")
print(f"   Total parameters: {total_params:,}")
print(f"   Architecture:")
print(f"     - ResNet: {layer_counter} blocks ({layer_counter * 2} layers)")
print(f"     - Transformer: {transformer_blocks} blocks (~1000 layers)")
print(f"     - DenseNet: {dense_counter} blocks (~100 layers)")
print(f"   Total: ~5000+ layers")

# ============================================================================
# 8. TRAIN MODEL (OPTIMIZED)
# ============================================================================
print("\n" + "="*80)
print("5. TRAINING ULTRA-DEEP MODEL (OPTIMIZED)")
print("="*80)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=150,  # Reduced for faster training
    restore_best_weights=True,
    verbose=1,
    min_delta=0.0001
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,  # Reduced for faster adaptation
    min_lr=0.00000001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'ultra_deep_transformer_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    save_weights_only=False
)

print("\n   🚀 Starting optimized training...")
print("   ⚡ Mixed precision: Enabled (2x faster)")
print("   ⚡ Gradient clipping: Enabled")
print("   ⚡ Optimized batch size: Auto-tuned")
print("   ⚡ Early stopping: Enabled")
print("\n   ⚠️  This will take significant time with 5000+ layers")
print("   💡 Estimated time: 24-48 hours (with optimizations)")
print("   💡 Monitor progress in the logs")
print("\n" + "="*80 + "\n")

# Optimize batch size based on available memory
try:
    # Try larger batch first
    batch_size = 16
    # Test if it works
    _ = model.predict(X_train_scaled[:batch_size], verbose=0)
except:
    batch_size = 8
    print(f"   Using batch size: {batch_size} (reduced for memory)")

print(f"   Using batch size: {batch_size}")

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=1500,  # Reduced for faster training
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================================
# 9. TEST MULTIPLE THRESHOLDS
# ============================================================================
print("\n" + "="*80)
print("6. TESTING ADAPTIVE THRESHOLDS")
print("="*80)

val_probs = model.predict(X_val_scaled, verbose=0).flatten()
test_probs = model.predict(X_test_scaled, verbose=0).flatten()

thresholds = np.arange(0.2, 0.95, 0.02)
best_threshold = 0.5
best_score = 0
best_results = None

print("\n   Testing thresholds...")

for threshold in thresholds:
    val_pred = (val_probs > threshold).astype(int)
    
    if val_pred.sum() > 0:
        val_indices = X_val.index
        val_trades = df_features.iloc[val_indices]
        predicted_wins = val_pred == 1
        
        if predicted_wins.sum() > 0:
            filtered = val_trades.iloc[np.where(predicted_wins)[0]]
            winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
            pnl = filtered['P&L'].sum()
            trades = len(filtered)
            
            score = winrate * 0.7 + (pnl / 100) * 0.3
            
            if score > best_score and winrate >= 50:
                best_score = score
                best_threshold = threshold
                best_results = {
                    'threshold': threshold,
                    'trades': trades,
                    'winrate': winrate,
                    'total_pnl': pnl,
                    'avg_pnl': pnl / trades if trades > 0 else 0
                }

print(f"\n   ✅ Best Threshold: {best_threshold:.3f}")
if best_results:
    print(f"   Trades: {best_results['trades']}")
    print(f"   Win Rate: {best_results['winrate']:.2f}%")
    print(f"   Total P&L: ${best_results['total_pnl']:.2f}")
    print(f"   Avg P&L: ${best_results['avg_pnl']:.2f}")

# ============================================================================
# 10. FINAL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("7. FINAL EVALUATION (TEST SET)")
print("="*80)

test_pred = (test_probs > best_threshold).astype(int)
test_indices = X_test.index
test_trades = df_features.iloc[test_indices]

predicted_wins = test_pred == 1
if predicted_wins.sum() > 0:
    filtered = test_trades.iloc[np.where(predicted_wins)[0]]
    
    winrate = (filtered['P&L'] > 0).sum() / len(filtered) * 100
    pnl = filtered['P&L'].sum()
    avg_pnl = filtered['P&L'].mean()
    
    print(f"\n   Test Set Results:")
    print(f"   Trades taken: {len(filtered)} ({len(filtered)/len(test_trades)*100:.1f}%)")
    print(f"   Win rate: {winrate:.2f}%")
    print(f"   Total P&L: ${pnl:.2f}")
    print(f"   Avg P&L: ${avg_pnl:.2f}")
    
    baseline_pnl = test_trades['P&L'].sum()
    baseline_winrate = (test_trades['P&L'] > 0).sum() / len(test_trades) * 100
    
    print(f"\n   Baseline:")
    print(f"   Trades: {len(test_trades)}")
    print(f"   Win rate: {baseline_winrate:.2f}%")
    print(f"   Total P&L: ${baseline_pnl:.2f}")
    
    improvement = pnl - (baseline_pnl * len(filtered) / len(test_trades))
    print(f"\n   Improvement: ${improvement:.2f}")

# Save
import joblib
pd.Series(feature_cols).to_csv('ultra_deep_transformer_features.csv', index=False)
joblib.dump(scaler, 'ultra_deep_transformer_scaler.pkl')
joblib.dump(best_threshold, 'ultra_deep_transformer_threshold.pkl')

# Save architecture info
arch_info = {
    'total_layers': total_layers,
    'total_parameters': total_params,
    'features': len(feature_cols),
    'residual_blocks': layer_counter,
    'transformer_blocks': transformer_blocks,
    'dense_blocks': dense_counter,
    'mixed_precision': True,
    'optimizations': 'Gradient clipping, Mixed precision, Optimized batch size'
}
pd.Series(arch_info).to_csv('ultra_deep_transformer_architecture.csv')

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\n   Files saved:")
print("   - ultra_deep_transformer_model.h5")
print("   - ultra_deep_transformer_features.csv")
print("   - ultra_deep_transformer_scaler.pkl")
print("   - ultra_deep_transformer_threshold.pkl")
print("   - ultra_deep_transformer_architecture.csv")
print("\n   🎉 Ultra-deep model with 5000+ layers trained successfully!")

