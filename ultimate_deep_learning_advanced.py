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
        Attention, MultiHeadAttention
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available")
    exit(1)

print("="*80)
print("ULTIMATE DEEP LEARNING SYSTEM - ADVANCED ARCHITECTURES")
print("="*80)
print("Target: 80%+ Win Rate | 2000+ Layers | ResNet + DenseNet + Attention")
print("Techniques: Residual Connections | Attention Mechanisms | Multi-Scale Features")
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
# 2. ULTIMATE FEATURE ENGINEERING - MAXIMUM FEATURES
# ============================================================================
print("\n2. Creating ULTIMATE feature set (200+ features)...")

def create_ultimate_features(df):
    """Create maximum possible features for deep learning"""
    features_df = df.copy()
    
    print("   2.1 Basic Indicators & Price Features...")
    if 'EntryPrice' in features_df.columns:
        features_df['Price'] = features_df['EntryPrice']
        # Multiple SMA windows
        for window in [5, 10, 20, 50]:
            features_df[f'Price_SMA_{window}'] = features_df['EntryPrice'].rolling(window).mean()
            features_df[f'Price_Deviation_{window}'] = (features_df['EntryPrice'] - features_df[f'Price_SMA_{window}']) / features_df[f'Price_SMA_{window}'] * 100
        
        # Price momentum
        features_df['Price_Change_1'] = features_df['EntryPrice'].pct_change(1) * 100
        features_df['Price_Change_2'] = features_df['EntryPrice'].pct_change(2) * 100
        features_df['Price_Change_5'] = features_df['EntryPrice'].pct_change(5) * 100
        features_df['Price_Volatility_5'] = features_df['EntryPrice'].rolling(5).std()
        features_df['Price_Volatility_20'] = features_df['EntryPrice'].rolling(20).std()
    
    print("   2.2 EMA Features (Extended)...")
    ema_cols = ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_1H']
    for ema in ema_cols:
        if ema in features_df.columns:
            # Distance from price
            features_df[f'{ema}_Distance'] = (features_df['EntryPrice'] - features_df[ema]) / features_df['EntryPrice'] * 100 if 'EntryPrice' in features_df.columns else 0
            # Slopes
            features_df[f'{ema}_Slope'] = features_df[ema].diff()
            features_df[f'{ema}_Slope_Pct'] = features_df[ema].pct_change() * 100
            features_df[f'{ema}_Slope_2'] = features_df[ema].diff(2)
            # Acceleration
            features_df[f'{ema}_Accel'] = features_df[f'{ema}_Slope'].diff()
            # EMA vs its own MA
            features_df[f'{ema}_vs_SMA_10'] = (features_df[ema] - features_df[ema].rolling(10).mean()) / features_df[ema].rolling(10).mean() * 100
    
    # EMA relationships (comprehensive)
    if all(col in features_df.columns for col in ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M']):
        pairs = [('EMA_9_5M', 'EMA_21_5M'), ('EMA_21_5M', 'EMA_50_5M'), ('EMA_9_5M', 'EMA_50_5M')]
        for ema1, ema2 in pairs:
            features_df[f'{ema1}_{ema2}_Distance'] = (features_df[ema1] - features_df[ema2]) / features_df[ema2] * 100
            features_df[f'{ema1}_{ema2}_Spread'] = abs(features_df[ema1] - features_df[ema2]) / features_df[ema2] * 100
            features_df[f'{ema1}_{ema2}_Ratio'] = features_df[ema1] / features_df[ema2]
        
        # Convergence/divergence
        features_df['EMA_Convergence_9_21'] = abs(features_df['EMA_9_5M'] - features_df['EMA_21_5M']) / features_df['EMA_21_5M'] * 100
        features_df['EMA_Convergence_21_50'] = abs(features_df['EMA_21_5M'] - features_df['EMA_50_5M']) / features_df['EMA_50_5M'] * 100
        features_df['EMA_Triangle_Area'] = abs((features_df['EMA_9_5M'] - features_df['EMA_21_5M']) * (features_df['EMA_21_5M'] - features_df['EMA_50_5M'])) / 2
    
    print("   2.3 ATR & Volatility Features (Extended)...")
    if 'ATR' in features_df.columns:
        features_df['ATR_Distance'] = features_df['ATR'] / features_df['EntryPrice'] * 100 if 'EntryPrice' in features_df.columns else 0
        features_df['ATR_Slope'] = features_df['ATR'].diff()
        features_df['ATR_Slope_Pct'] = features_df['ATR'].pct_change() * 100
        features_df['ATR_Accel'] = features_df['ATR_Slope'].diff()
        
        # ATR moving averages
        for window in [5, 10, 20]:
            features_df[f'ATR_MA_{window}'] = features_df['ATR'].rolling(window).mean()
            features_df[f'ATR_vs_MA_{window}'] = (features_df['ATR'] - features_df[f'ATR_MA_{window}']) / features_df[f'ATR_MA_{window}'] * 100
        
        # ATR percentiles
        features_df['ATR_Percentile_20'] = features_df['ATR'].rolling(20).quantile(0.2)
        features_df['ATR_Percentile_80'] = features_df['ATR'].rolling(20).quantile(0.8)
        features_df['ATR_High_Vol'] = (features_df['ATR'] > features_df['ATR_Percentile_80']).astype(int)
        features_df['ATR_Low_Vol'] = (features_df['ATR'] < features_df['ATR_Percentile_20']).astype(int)
    
    if 'ATR_Ratio' in features_df.columns:
        for transform in ['Squared', 'Cubed', 'Sqrt', 'Log']:
            if transform == 'Squared':
                features_df[f'ATR_Ratio_{transform}'] = features_df['ATR_Ratio'] ** 2
            elif transform == 'Cubed':
                features_df[f'ATR_Ratio_{transform}'] = features_df['ATR_Ratio'] ** 3
            elif transform == 'Sqrt':
                features_df[f'ATR_Ratio_{transform}'] = np.sqrt(np.abs(features_df['ATR_Ratio']))
            elif transform == 'Log':
                features_df[f'ATR_Ratio_{transform}'] = np.log1p(np.abs(features_df['ATR_Ratio']))
    
    print("   2.4 Breakout Features (Extended)...")
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
        
        # Breakout vs price history
        if len(features_df) > 20:
            features_df['Breakout_vs_AvgRange_20'] = features_df['Breakout_Size'] / features_df['Breakout_Size'].rolling(20).mean()
            features_df['Breakout_vs_MaxRange_20'] = features_df['Breakout_Size'] / features_df['Breakout_Size'].rolling(20).max()
    
    print("   2.5 Risk Features (Extended)...")
    if 'Risk' in features_df.columns:
        features_df['Risk_Pct'] = features_df['Risk'] / features_df['EntryPrice'] * 100 if 'EntryPrice' in features_df.columns else 0
        features_df['Risk_Squared'] = features_df['Risk'] ** 2
        features_df['Risk_Cubed'] = features_df['Risk'] ** 3
        features_df['Risk_Log'] = np.log1p(features_df['Risk'])
        features_df['Risk_Sqrt'] = np.sqrt(features_df['Risk'])
        features_df['Risk_Exp'] = np.exp(np.clip(features_df['Risk'] / 10, -5, 5))
        
        # Risk categories
        features_df['Risk_Category'] = pd.cut(features_df['Risk'], bins=[0, 5, 10, 15, 20, 100], labels=[1, 2, 3, 4, 5]).astype(float)
        
        # Risk vs historical
        if len(features_df) > 20:
            features_df['Risk_vs_Avg_20'] = features_df['Risk'] / features_df['Risk'].rolling(20).mean()
            features_df['Risk_vs_Median_20'] = features_df['Risk'] / features_df['Risk'].rolling(20).median()
    
    print("   2.6 Trend & Alignment Features (Extended)...")
    if 'Trend_Score' in features_df.columns:
        for transform in ['Squared', 'Cubed', 'Sqrt', 'Log']:
            if transform == 'Squared':
                features_df[f'Trend_{transform}'] = features_df['Trend_Score'] ** 2
            elif transform == 'Cubed':
                features_df[f'Trend_{transform}'] = features_df['Trend_Score'] ** 3
            elif transform == 'Sqrt':
                features_df[f'Trend_{transform}'] = np.sqrt(features_df['Trend_Score'])
            elif transform == 'Log':
                features_df[f'Trend_{transform}'] = np.log1p(features_df['Trend_Score'])
        
        features_df['Trend_Direction'] = (features_df['Trend_Score'] > 0.5).astype(int)
        features_df['Trend_Strong'] = (features_df['Trend_Score'] > 0.7).astype(int)
        features_df['Trend_Weak'] = (features_df['Trend_Score'] < 0.3).astype(int)
    
    # EMA alignment (comprehensive)
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
        
        # Pattern encoding
        features_df['EMA_Pattern'] = (
            features_df['EMA_9_Above_21'].astype(int).astype(str) +
            features_df['EMA_21_Above_50'].astype(int).astype(str) +
            features_df['Price_Above_EMA200_1H'].astype(int).astype(str)
        )
        pattern_map = {'000': 0, '001': 1, '010': 2, '011': 3, '100': 4, '101': 5, '110': 6, '111': 7}
        features_df['EMA_Pattern_Num'] = features_df['EMA_Pattern'].map(pattern_map).fillna(0)
    
    print("   2.7 Consolidation Features...")
    if 'Consolidation_Score' in features_df.columns:
        features_df['Consolidation_Strength'] = features_df['Consolidation_Score'] ** 2
        features_df['Is_Strong_Consolidation'] = (features_df['Consolidation_Score'] > 0.7).astype(int)
        features_df['Is_Weak_Consolidation'] = (features_df['Consolidation_Score'] < 0.3).astype(int)
        features_df['Consolidation_Log'] = np.log1p(features_df['Consolidation_Score'])
    
    print("   2.8 Range Features...")
    if 'RangeSize' in features_df.columns:
        features_df['RangeSize_Pct'] = features_df['RangeSize'] / features_df['EntryPrice'] * 100 if 'EntryPrice' in features_df.columns else 0
        features_df['RangeSize_Squared'] = features_df['RangeSize'] ** 2
        features_df['RangeSize_Log'] = np.log1p(features_df['RangeSize'])
        features_df['RangeSize_Sqrt'] = np.sqrt(features_df['RangeSize'])
        
        if len(features_df) > 20:
            features_df['RangeSize_vs_Avg_20'] = features_df['RangeSize'] / features_df['RangeSize'].rolling(20).mean()
    
    print("   2.9 Time Features (Extended)...")
    if 'EntryTime' in features_df.columns:
        features_df['EntryTime'] = pd.to_datetime(features_df['EntryTime'])
        features_df['Hour'] = features_df['EntryTime'].dt.hour
        features_df['DayOfWeek'] = features_df['EntryTime'].dt.dayofweek
        features_df['DayOfMonth'] = features_df['EntryTime'].dt.day
        features_df['Month'] = features_df['EntryTime'].dt.month
        features_df['WeekOfYear'] = features_df['EntryTime'].dt.isocalendar().week
        
        # Cyclical encoding (sin/cos)
        features_df['Hour_Sin'] = np.sin(2 * np.pi * features_df['Hour'] / 24)
        features_df['Hour_Cos'] = np.cos(2 * np.pi * features_df['Hour'] / 24)
        features_df['DayOfWeek_Sin'] = np.sin(2 * np.pi * features_df['DayOfWeek'] / 7)
        features_df['DayOfWeek_Cos'] = np.cos(2 * np.pi * features_df['DayOfWeek'] / 7)
        features_df['DayOfMonth_Sin'] = np.sin(2 * np.pi * features_df['DayOfMonth'] / 31)
        features_df['DayOfMonth_Cos'] = np.cos(2 * np.pi * features_df['DayOfMonth'] / 31)
        features_df['Month_Sin'] = np.sin(2 * np.pi * features_df['Month'] / 12)
        features_df['Month_Cos'] = np.cos(2 * np.pi * features_df['Month'] / 12)
        
        # Trading session indicators
        features_df['Is_London_Session'] = ((features_df['Hour'] >= 8) & (features_df['Hour'] < 16)).astype(int)
        features_df['Is_NY_Session'] = ((features_df['Hour'] >= 13) & (features_df['Hour'] < 21)).astype(int)
        features_df['Is_Asian_Session'] = (((features_df['Hour'] >= 0) & (features_df['Hour'] < 8)) | 
                                           ((features_df['Hour'] >= 22) & (features_df['Hour'] < 24))).astype(int)
    
    print("   2.10 Window Type Features...")
    if 'WindowType' in features_df.columns:
        window_dummies = pd.get_dummies(features_df['WindowType'], prefix='Window')
        features_df = pd.concat([features_df, window_dummies], axis=1)
    
    print("   2.11 Interaction Features (Extended)...")
    if all(col in features_df.columns for col in ['Risk', 'ATR_Ratio', 'Trend_Score']):
        # Two-way interactions
        features_df['Risk_ATR'] = features_df['Risk'] * features_df['ATR_Ratio']
        features_df['Risk_Trend'] = features_df['Risk'] * features_df['Trend_Score']
        features_df['ATR_Trend'] = features_df['ATR_Ratio'] * features_df['Trend_Score']
        # Three-way interaction
        features_df['Risk_ATR_Trend'] = features_df['Risk'] * features_df['ATR_Ratio'] * features_df['Trend_Score']
        # Four-way with breakout
        if 'Breakout_Pct' in features_df.columns:
            features_df['Risk_ATR_Trend_Breakout'] = features_df['Risk'] * features_df['ATR_Ratio'] * features_df['Trend_Score'] * features_df['Breakout_Pct']
    
    print("   2.12 Wave & Pattern Features...")
    if 'EntryPrice' in features_df.columns and len(features_df) > 20:
        for window in [5, 10, 20]:
            features_df[f'Price_Wave_{window}'] = features_df['EntryPrice'].rolling(window).std()
        
        # Wave ratios
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
    
    print("   2.13 Statistical Features (Extended)...")
    if 'P&L' in features_df.columns and len(features_df) > 10:
        for window in [5, 10, 20]:
            features_df[f'P&L_MA_{window}'] = features_df['P&L'].rolling(window).mean()
            features_df[f'P&L_Std_{window}'] = features_df['P&L'].rolling(window).std()
            
            # Z-scores
            features_df[f'P&L_ZScore_{window}'] = np.where(
                features_df[f'P&L_Std_{window}'] > 0,
                (features_df['P&L'] - features_df[f'P&L_MA_{window}']) / features_df[f'P&L_Std_{window}'],
                0
            )
        
        # Win streak / loss streak
        features_df['Win'] = (features_df['P&L'] > 0).astype(int)
        features_df['Loss'] = (features_df['P&L'] <= 0).astype(int)
        
        # Recent performance
        features_df['Recent_WinRate_5'] = features_df['Win'].rolling(5).mean()
        features_df['Recent_WinRate_10'] = features_df['Win'].rolling(10).mean()
    
    print("   2.14 Polynomial Features...")
    key_vars = ['Risk', 'ATR_Ratio', 'Trend_Score', 'Breakout_Pct', 'EMA_Alignment_Score']
    for var in key_vars:
        if var in features_df.columns:
            features_df[f'{var}_Squared'] = features_df[var] ** 2
            features_df[f'{var}_Cubed'] = features_df[var] ** 3
    
    print("   2.15 Normalized Features...")
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
                'EMA_Pattern', 'Win', 'Loss']

feature_cols = [col for col in df_features.columns 
                if col not in exclude_cols and df_features[col].dtype in ['int64', 'float64', 'bool']]
feature_cols = [col for col in feature_cols if df_features[col].notna().sum() > len(df_features) * 0.2]

X = df_features[feature_cols].copy()
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], 0)

y_binary = (df_features['P&L'] > 0).astype(int)

print(f"\n   Total features created: {len(feature_cols)}")

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

print(f"   Features scaled: {X_train_scaled.shape[1]}")

# ============================================================================
# 4. BUILD RESIDUAL BLOCK (ResNet-style)
# ============================================================================
def residual_block(x, units, dropout_rate=0.1, layer_num=0):
    """Residual block with skip connection"""
    identity = x
    
    # Main path
    out = Dense(units, kernel_regularizer=regularizers.l2(0.0001), name=f'res_dense_{layer_num}_1')(x)
    out = BatchNormalization(name=f'res_bn_{layer_num}_1')(out)
    out = Activation('relu', name=f'res_act_{layer_num}_1')(out)
    out = Dropout(dropout_rate, name=f'res_dropout_{layer_num}_1')(out)
    
    out = Dense(units, kernel_regularizer=regularizers.l2(0.0001), name=f'res_dense_{layer_num}_2')(out)
    out = BatchNormalization(name=f'res_bn_{layer_num}_2')(out)
    
    # Skip connection (if dimensions match)
    if identity.shape[-1] == units:
        out = Add(name=f'res_add_{layer_num}')([out, identity])
    else:
        # Projection shortcut
        identity = Dense(units, name=f'res_proj_{layer_num}')(identity)
        out = Add(name=f'res_add_{layer_num}')([out, identity])
    
    out = Activation('relu', name=f'res_act_{layer_num}_2')(out)
    out = Dropout(dropout_rate, name=f'res_dropout_{layer_num}_2')(out)
    
    return out

# ============================================================================
# 5. BUILD DENSE BLOCK (DenseNet-style)
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
        # Concatenate all previous outputs
        if len(concat_list) > 1:
            out = Concatenate(name=f'dense_concat_{block_num}_{i}')(concat_list)
        else:
            out = concat_list[-1]
        concat_list = [out]
    
    return concat_list[0]

# ============================================================================
# 6. BUILD ULTRA-DEEP MODEL WITH RESIDUAL + DENSE + ATTENTION
# ============================================================================
print("\n" + "="*80)
print("4. BUILDING ULTRA-DEEP MODEL (2000+ LAYERS)")
print("Architecture: ResNet Blocks + DenseNet Blocks + Attention")
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

# Store for attention (multi-scale features)
attention_features = [x]

print("\n   Building residual blocks...")
# Residual blocks (ResNet-style) - 500 blocks = 2000 layers
res_units = [2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128]
res_blocks_per_unit = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]  # 500 blocks total

layer_counter = 0
for unit_size, num_blocks in zip(res_units, res_blocks_per_unit):
    for i in range(num_blocks):
        x = residual_block(x, unit_size, dropout_rate=0.1, layer_num=layer_counter)
        layer_counter += 1
        
        # Store features at intervals for attention
        if layer_counter % 50 == 0:
            attention_features.append(x)
        
        if layer_counter % 100 == 0:
            print(f"      Built {layer_counter} residual blocks...")

print(f"\n   Built {layer_counter} residual blocks")

# Multi-scale attention
print("\n   Adding attention mechanism...")
# Use attention on multi-scale features
if len(attention_features) > 1:
    # Reshape features for attention - ensure all have same dimensionality
    # Get current feature dimension
    current_dim = attention_features[-1].shape[-1]
    
    # Project all attention features to same dimension and stack
    projected_features = []
    for i, feat in enumerate(attention_features):
        if feat.shape[-1] != current_dim:
            # Project to same dimension
            proj_feat = Dense(current_dim, name=f'attention_proj_{i}')(feat)
        else:
            proj_feat = feat
        # Add sequence dimension
        proj_feat = tf.expand_dims(proj_feat, axis=1)  # (batch, 1, features)
        projected_features.append(proj_feat)
    
    # Concatenate all features along sequence dimension
    multi_scale = Concatenate(axis=1, name='attention_concat')(projected_features)  # (batch, num_scales, features)
    
    # Simple attention: compute weights for each scale
    # Flatten multi-scale features
    multi_scale_flat = tf.reshape(multi_scale, (-1, multi_scale.shape[1] * multi_scale.shape[2]))
    
    # Compute attention weights
    attention_weights = Dense(multi_scale.shape[1], activation='softmax', name='attention_weights')(multi_scale_flat)
    attention_weights = tf.expand_dims(attention_weights, axis=-1)  # (batch, num_scales, 1)
    
    # Apply attention weights
    attended = Multiply(name='attention_apply')([multi_scale, attention_weights])
    attended = tf.reduce_sum(attended, axis=1)  # Sum over scales: (batch, features)
    
    # Combine with main path
    x = Concatenate(name='combine_attention')([x, attended])
    # Project back to original dimension
    x = Dense(current_dim, kernel_regularizer=regularizers.l2(0.0001), name='attention_projection')(x)
    x = BatchNormalization(name='attention_bn')(x)
    x = Activation('relu', name='attention_act')(x)

# Dense blocks (DenseNet-style) - 20 blocks
print("\n   Building dense blocks...")
dense_units = [256, 128, 64]
dense_blocks_per_unit = [10, 5, 5]

for unit_size, num_blocks in zip(dense_units, dense_blocks_per_unit):
    for i in range(num_blocks):
        x = dense_block(x, unit_size, growth_rate=32, num_layers=2, block_num=i)
        layer_counter += 1

print(f"   Built {layer_counter} dense blocks")

# Final layers
x = Dense(128, kernel_regularizer=regularizers.l2(0.0001), name='final_dense_1')(x)
x = BatchNormalization(name='final_bn_1')(x)
x = Activation('relu', name='final_act_1')(x)
x = Dropout(0.1, name='final_dropout_1')(x)

x = Dense(64, kernel_regularizer=regularizers.l2(0.0001), name='final_dense_2')(x)
x = BatchNormalization(name='final_bn_2')(x)
x = Activation('relu', name='final_act_2')(x)
x = Dropout(0.05, name='final_dropout_2')(x)

# Output
outputs = Dense(1, activation='sigmoid', name='output')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs, name='UltimateDeepModel')

# Compile with gradient clipping
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Optimizer with gradient clipping
optimizer = keras.optimizers.Adam(
    learning_rate=0.00001,
    clipnorm=1.0,  # Gradient clipping
    clipvalue=0.5
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
print(f"   Architecture: ResNet (500 blocks) + DenseNet (20 blocks) + Attention")

model.summary()

# ============================================================================
# 7. TRAIN MODEL
# ============================================================================
print("\n" + "="*80)
print("5. TRAINING ULTRA-DEEP MODEL")
print("="*80)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=200,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.0001
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=30,
    min_lr=0.00000001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'ultimate_deep_advanced_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    save_weights_only=False
)

print("\n   Starting training...")
print("   ⚠️  This will take significant time with 2000+ layers")
print("   💡 Monitor progress in the logs")

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=2000,
    batch_size=8,  # Small batch for stability
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================================
# 8. TEST MULTIPLE THRESHOLDS
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
            
            # Score: prioritize win rate (80% target)
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
# 9. FINAL EVALUATION
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
    
    # Baseline
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
pd.Series(feature_cols).to_csv('ultimate_deep_advanced_features.csv', index=False)
joblib.dump(scaler, 'ultimate_deep_advanced_scaler.pkl')
joblib.dump(best_threshold, 'ultimate_deep_advanced_threshold.pkl')

# Save architecture info
arch_info = {
    'total_layers': total_layers,
    'total_parameters': total_params,
    'features': len(feature_cols),
    'residual_blocks': 500,
    'dense_blocks': 20,
    'attention_mechanism': True
}
pd.Series(arch_info).to_csv('ultimate_deep_advanced_architecture.csv')

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\n   Files saved:")
print("   - ultimate_deep_advanced_model.h5")
print("   - ultimate_deep_advanced_features.csv")
print("   - ultimate_deep_advanced_scaler.pkl")
print("   - ultimate_deep_advanced_threshold.pkl")
print("   - ultimate_deep_advanced_architecture.csv")

