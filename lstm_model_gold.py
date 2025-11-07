import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras, fallback to sklearn if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow/Keras available - using deep LSTM")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - using alternative approach")
    # We'll use a simpler approach or create sequences manually

print("="*80)
print("LSTM MODEL FOR XAUUSD (GOLD)")
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

# For LSTM, we need sequential data
# Since we have individual trades, we'll create sequences from the data
# Option 1: Use trade order as sequence
# Option 2: Create sequences from price data (if available)

# Sort by entry time to create sequence
df_sorted = df.sort_values('EntryTime').reset_index(drop=True)
X_sorted = X.reindex(df_sorted.index)

# Create sequences (look back N trades)
sequence_length = 10  # Look back 10 trades
n_features = len(feature_cols)

def create_sequences(data, targets, seq_length):
    """Create sequences for LSTM"""
    X_seq = []
    y_seq = []
    
    for i in range(seq_length, len(data)):
        # data is already a numpy array
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
# 4. BUILD LSTM MODEL
# ============================================================================
print("\n" + "="*80)
print("4. BUILDING LSTM MODEL")
print("="*80)

if TENSORFLOW_AVAILABLE:
    print("\nBuilding LSTM with TensorFlow/Keras...")
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Build LSTM model
    model = Sequential([
        # First LSTM layer
        LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        BatchNormalization(),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Handle class imbalance with class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train_seq)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_seq)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
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
    model.save('lstm_model_gold.h5')
    print("\n   Model saved: lstm_model_gold.h5")
    
else:
    print("\nTensorFlow not available. Using alternative approach...")
    print("Creating simplified LSTM-like model using sklearn...")
    
    # Flatten sequences for traditional ML
    X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_val_flat = X_val_seq.reshape(X_val_seq.shape[0], -1)
    X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    
    # Use MLP as LSTM alternative
    print("Using MLP with sequence data...")
    lstm_model = MLPClassifier(
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
    
    lstm_model.fit(X_train_flat, y_train_seq)
    
    train_pred = lstm_model.predict(X_train_flat)
    val_pred = lstm_model.predict(X_val_flat)
    test_pred = lstm_model.predict(X_test_flat)
    
    train_acc = accuracy_score(y_train_seq, train_pred)
    val_acc = accuracy_score(y_val_seq, val_pred)
    test_acc = accuracy_score(y_test_seq, test_pred)
    
    print(f"\n   Training Accuracy:   {train_acc:.4f}")
    print(f"   Validation Accuracy: {val_acc:.4f}")
    print(f"   Test Accuracy:       {test_acc:.4f}")
    
    # Save model
    import joblib
    joblib.dump(lstm_model, 'lstm_model_gold.pkl')
    print("\n   Model saved: lstm_model_gold.pkl")

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
# Get original trade indices for test set
test_indices = []
test_start = len(X_train_seq) + len(X_val_seq)
for i in range(len(X_test_seq)):
    test_indices.append(test_start + i + sequence_length)

# Get actual P&L for predicted wins
if len(test_indices) > 0:
    predicted_wins_mask = test_pred == 1
    if predicted_wins_mask.sum() > 0:
        # Get corresponding trades from original data
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
pd.Series(feature_cols).to_csv('lstm_feature_columns.csv', index=False)
pd.Series([sequence_length, n_features]).to_csv('lstm_model_params.csv', index=False)

print("   Feature columns saved: lstm_feature_columns.csv")
print("   Model parameters saved: lstm_model_params.csv")

print("\n" + "="*80)
print("LSTM TRAINING COMPLETE!")
print("="*80)

