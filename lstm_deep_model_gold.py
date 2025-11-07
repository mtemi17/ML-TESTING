import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow/Keras available - using deep LSTM (10-15 layers)")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available")
    exit(1)

print("="*80)
print("DEEP LSTM MODEL FOR XAUUSD (GOLD) - 10-15 LAYERS")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. Loading XAUUSD data...")
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

print(f"   Target: Win/Loss - {y_binary.sum()} wins, {len(y_binary) - y_binary.sum()} losses")

# ============================================================================
# 2. CREATE SEQUENCES FOR LSTM
# ============================================================================
print("\n2. Creating sequences for LSTM...")

# Sort by entry time to create sequence
df_sorted = df.sort_values('EntryTime').reset_index(drop=True)
X_sorted = X.reindex(df_sorted.index)

# Create sequences (look back N trades)
sequence_length = 15  # Increased for deeper model
n_features = len(feature_cols)

def create_sequences(data, targets, seq_length):
    """Create sequences for LSTM"""
    X_seq = []
    y_seq = []
    
    for i in range(seq_length, len(data)):
        X_seq.append(data[i-seq_length:i])
        y_seq.append(targets[i])
    
    return np.array(X_seq), np.array(y_seq)

# Prepare data for sequences
X_scaled = StandardScaler().fit_transform(X_sorted)
y_binary_sorted = y_binary.reindex(df_sorted.index).values

# Create sequences
X_sequences, y_sequences = create_sequences(X_scaled, y_binary_sorted, sequence_length)

print(f"   Created {len(X_sequences)} sequences")
print(f"   Sequence shape: {X_sequences.shape}")
print(f"   Features per timestep: {n_features}")

# ============================================================================
# 3. SPLIT DATA INTO 3 BATCHES
# ============================================================================
print("\n3. Splitting data into 3 batches...")

# First split: 60% train, 40% temp
X_train_seq, X_temp_seq, y_train_seq, y_temp_seq = train_test_split(
    X_sequences, y_sequences, test_size=0.4, random_state=42, stratify=y_sequences
)

# Second split: 20% validation, 20% test
X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(
    X_temp_seq, y_temp_seq, test_size=0.5, random_state=42, stratify=y_temp_seq
)

print(f"   Batch 1 (Training):   {len(X_train_seq)} sequences ({len(X_train_seq)/len(X_sequences)*100:.1f}%)")
print(f"   Batch 2 (Validation): {len(X_val_seq)} sequences ({len(X_val_seq)/len(X_sequences)*100:.1f}%)")
print(f"   Batch 3 (Testing):    {len(X_test_seq)} sequences ({len(X_test_seq)/len(X_sequences)*100:.1f}%)")

# ============================================================================
# 4. BUILD DEEP LSTM MODEL (10-15 LAYERS)
# ============================================================================
print("\n" + "="*80)
print("4. BUILDING DEEP LSTM MODEL (12 LAYERS)")
print("="*80)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Build deep LSTM model with 12 layers (10 LSTM + 2 Dense)
model = Sequential()

# Input layer
model.add(LSTM(256, return_sequences=True, input_shape=(sequence_length, n_features)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Layer 2
model.add(LSTM(192, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Layer 3
model.add(LSTM(160, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# Layer 4
model.add(LSTM(128, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# Layer 5
model.add(LSTM(96, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Layer 6
model.add(LSTM(80, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Layer 7
model.add(LSTM(64, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Layer 8
model.add(LSTM(48, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.15))

# Layer 9
model.add(LSTM(32, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.15))

# Layer 10 - Last LSTM layer (no return_sequences)
model.add(LSTM(24, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# Dense layers
# Layer 11
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# Layer 12 - Output layer
model.add(Dense(1, activation='sigmoid'))

# Handle class imbalance with class weights
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train_seq)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_seq)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR for deeper network
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
model.summary()

# Calculate total layers
total_layers = len(model.layers)
lstm_layers = sum(1 for layer in model.layers if isinstance(layer, LSTM))
dense_layers = sum(1 for layer in model.layers if isinstance(layer, Dense))

print(f"\n   Total layers: {total_layers}")
print(f"   LSTM layers: {lstm_layers}")
print(f"   Dense layers: {dense_layers}")
print(f"   Total parameters: {model.count_params():,}")

# Early stopping with more patience for deeper model
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.0001
)

# Reduce learning rate on plateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

# Train model
print("\nTraining Deep LSTM model...")
print("This may take longer due to the depth of the network...")
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=200,  # More epochs for deeper model
    batch_size=16,  # Smaller batch for stability
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate
train_pred = (model.predict(X_train_seq, verbose=0) > 0.5).astype(int).flatten()
val_pred = (model.predict(X_val_seq, verbose=0) > 0.5).astype(int).flatten()
test_pred = (model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()

train_acc = accuracy_score(y_train_seq, train_pred)
val_acc = accuracy_score(y_val_seq, val_pred)
test_acc = accuracy_score(y_test_seq, test_pred)

print(f"\n   Training Accuracy:   {train_acc:.4f}")
print(f"   Validation Accuracy: {val_acc:.4f}")
print(f"   Test Accuracy:       {test_acc:.4f}")

# Save model
model.save('lstm_deep_model_gold.h5')
print("\n   Model saved: lstm_deep_model_gold.h5")

# ============================================================================
# 5. TEST SET EVALUATION
# ============================================================================
print("\n" + "="*80)
print("5. DETAILED TEST SET EVALUATION")
print("="*80)

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test_seq, test_pred))

print("\nClassification Report (Test Set):")
print(classification_report(y_test_seq, test_pred, target_names=['Loss', 'Win']))

# Calculate strategy performance
test_indices = []
test_start = len(X_train_seq) + len(X_val_seq)
for i in range(len(X_test_seq)):
    test_indices.append(test_start + i + sequence_length)

# Get actual P&L for predicted wins
if len(test_indices) > 0:
    predicted_wins_mask = test_pred == 1
    if predicted_wins_mask.sum() > 0:
        actual_pnl = []
        for idx in np.where(predicted_wins_mask)[0]:
            trade_idx = test_indices[idx]
            if trade_idx < len(df_sorted):
                actual_pnl.append(df_sorted.iloc[trade_idx]['P&L'])
        
        if len(actual_pnl) > 0:
            actual_pnl = np.array(actual_pnl)
            win_rate_actual = (actual_pnl > 0).sum() / len(actual_pnl) * 100
            total_pnl = actual_pnl.sum()
            avg_pnl = actual_pnl.mean()
            
            print(f"\nStrategy Performance (Only taking predicted wins):")
            print(f"  Total trades taken: {len(actual_pnl)}")
            print(f"  Actual win rate: {win_rate_actual:.2f}%")
            print(f"  Total P&L: ${total_pnl:.2f}")
            print(f"  Avg P&L: ${avg_pnl:.2f}")
            
            # Baseline
            baseline_pnl = df_sorted.iloc[test_indices]['P&L'].sum() if len(test_indices) <= len(df_sorted) else 0
            baseline_win_rate = (df_sorted.iloc[test_indices]['P&L'] > 0).sum() / len(test_indices) * 100 if len(test_indices) <= len(df_sorted) else 0
            
            print(f"\nBaseline (All trades in test set):")
            print(f"  Total trades: {len(test_indices)}")
            print(f"  Win rate: {baseline_win_rate:.2f}%")
            print(f"  Total P&L: ${baseline_pnl:.2f}")
            
            if baseline_pnl != 0:
                improvement = total_pnl - (baseline_pnl * len(actual_pnl) / len(test_indices))
                print(f"\nImprovement: ${improvement:.2f} ({improvement/abs(baseline_pnl)*100:.1f}%)")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("6. SAVING RESULTS")
print("="*80)

# Save feature columns and scaler info
pd.Series(feature_cols).to_csv('lstm_deep_feature_columns.csv', index=False)
pd.Series([sequence_length, n_features, total_layers, lstm_layers, dense_layers]).to_csv('lstm_deep_model_params.csv', index=False)

print("   Feature columns saved: lstm_deep_feature_columns.csv")
print("   Model parameters saved: lstm_deep_model_params.csv")

print("\n" + "="*80)
print("DEEP LSTM TRAINING COMPLETE!")
print("="*80)
print(f"\nModel Summary:")
print(f"  - Total Layers: {total_layers}")
print(f"  - LSTM Layers: {lstm_layers}")
print(f"  - Dense Layers: {dense_layers}")
print(f"  - Sequence Length: {sequence_length}")
print(f"  - Parameters: {model.count_params():,}")
print(f"  - Test Accuracy: {test_acc:.4f}")

