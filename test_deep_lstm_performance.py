import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEEP LSTM MODEL PERFORMANCE TEST")
print("="*80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('advanced_analysis_results_combined.csv')
df = df[df['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
df_sorted = df.sort_values('EntryTime').reset_index(drop=True)

# Load model
print("2. Loading deep LSTM model...")
model = keras.models.load_model('lstm_deep_model_gold.h5')
lstm_feature_cols = pd.read_csv('lstm_deep_feature_columns.csv', header=None)[0].tolist()
lstm_params = pd.read_csv('lstm_deep_model_params.csv', header=None)[0].tolist()
# Skip index (first value), then sequence_length, n_features, etc.
sequence_length = int(lstm_params[1]) if len(lstm_params) > 1 else 15
n_features = int(lstm_params[2]) if len(lstm_params) > 2 else 20

print(f"   Sequence length: {sequence_length}")
print(f"   Features: {n_features}")

# Prepare features
exclude_cols = ['EntryTime', 'ExitTime', 'Status', 'P&L', 'R_Multiple', 
                'EntryPrice', 'SL', 'TP', 'WindowHigh', 'WindowLow', 'TradeID',
                'ExitPrice', 'BestPossibleP&L', 'WouldHaveWon', 'ReachedTP', 
                'WaitDuration_Min', 'BestWaitDuration_Min', 'TradeID_x', 'TradeID_y',
                'MinAdverse_Min30', 'MaxFavorable_Min30', 'WentAgainstFirst',
                'Duration_Minutes', 'Duration_Candles']

feature_cols = [col for col in df_sorted.columns if col not in exclude_cols and df_sorted[col].dtype in ['int64', 'float64']]
feature_cols = [col for col in feature_cols if df_sorted[col].notna().sum() > len(df_sorted) * 0.5]

X = df_sorted[feature_cols].fillna(df_sorted[feature_cols].median())
X_scaled = StandardScaler().fit_transform(X)

# Create sequences
def create_sequences(data, seq_length):
    X_seq = []
    indices = []
    for i in range(seq_length, len(data)):
        X_seq.append(data[i-seq_length:i])
        indices.append(i)
    return np.array(X_seq), indices

X_sequences, sequence_indices = create_sequences(X_scaled, sequence_length)

print(f"\n3. Created {len(X_sequences)} sequences")

# Get predictions with different thresholds
print("\n4. Testing different prediction thresholds...")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
best_threshold = 0.5
best_score = 0
best_results = None

for threshold in thresholds:
    predictions = (model.predict(X_sequences, verbose=0) > threshold).astype(int).flatten()
    
    # Get actual outcomes
    actual = (df_sorted.iloc[sequence_indices]['P&L'] > 0).astype(int).values
    
    # Calculate metrics
    if predictions.sum() > 0:  # Only if we predict some wins
        predicted_wins_mask = predictions == 1
        win_indices = np.array(sequence_indices)[predicted_wins_mask]
        actual_pnl = df_sorted.iloc[win_indices]['P&L'].values
        
        win_rate = (actual_pnl > 0).sum() / len(actual_pnl) * 100 if len(actual_pnl) > 0 else 0
        total_pnl = actual_pnl.sum() if len(actual_pnl) > 0 else 0
        trades_taken = len(actual_pnl)
        
        # Score: balance between win rate and total P&L
        score = win_rate * 0.5 + (total_pnl / 100) * 0.5
        
        print(f"\n   Threshold {threshold}:")
        print(f"     Trades taken: {trades_taken} ({trades_taken/len(predictions)*100:.1f}%)")
        print(f"     Win rate: {win_rate:.2f}%")
        print(f"     Total P&L: ${total_pnl:.2f}")
        print(f"     Score: {score:.2f}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_results = {
                'threshold': threshold,
                'trades': trades_taken,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'predictions': predictions,
                'indices': sequence_indices
            }

# Use best threshold
print(f"\n5. Best threshold: {best_threshold}")
print(f"   Best score: {best_score:.2f}")

if best_results:
    predictions = best_results['predictions']
    sequence_indices = best_results['indices']
    
    # Get actual P&L for predicted wins
    predicted_wins_mask = predictions == 1
    if predicted_wins_mask.sum() > 0:
        win_indices = np.array(sequence_indices)[predicted_wins_mask]
        actual_pnl = df_sorted.iloc[win_indices]['P&L'].values
        
        print(f"\n6. Strategy Performance:")
        print(f"   Total trades taken: {len(actual_pnl)}")
        print(f"   Win rate: {best_results['win_rate']:.2f}%")
        print(f"   Total P&L: ${best_results['total_pnl']:.2f}")
        print(f"   Avg P&L: ${best_results['total_pnl']/len(actual_pnl):.2f}")
        
        # Baseline
        baseline_pnl = df_sorted.iloc[sequence_indices]['P&L'].sum()
        baseline_win_rate = (df_sorted.iloc[sequence_indices]['P&L'] > 0).sum() / len(sequence_indices) * 100
        
        print(f"\n7. Baseline (All trades):")
        print(f"   Total trades: {len(sequence_indices)}")
        print(f"   Win rate: {baseline_win_rate:.2f}%")
        print(f"   Total P&L: ${baseline_pnl:.2f}")
        
        improvement = best_results['total_pnl'] - (baseline_pnl * len(actual_pnl) / len(sequence_indices))
        print(f"\n8. Improvement: ${improvement:.2f} ({improvement/abs(baseline_pnl)*100:.1f}%)")
        
        # Save results
        results_df = pd.DataFrame({
            'Model': ['Baseline', 'Deep LSTM'],
            'Trades': [len(sequence_indices), len(actual_pnl)],
            'WinRate': [baseline_win_rate, best_results['win_rate']],
            'TotalP&L': [baseline_pnl, best_results['total_pnl']],
            'AvgP&L': [baseline_pnl/len(sequence_indices), best_results['total_pnl']/len(actual_pnl)],
            'Threshold': [None, best_threshold]
        })
        
        results_df.to_csv('deep_lstm_performance.csv', index=False)
        print(f"\n9. Results saved to: deep_lstm_performance.csv")
else:
    print("\n   WARNING: Model predicted all losses at all thresholds!")

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)

