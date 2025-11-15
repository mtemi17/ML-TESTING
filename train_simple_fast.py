#!/usr/bin/env python3
"""
Fast training script for simple breakout strategy
Optimized for large datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING ML MODEL - SIMPLE BREAKOUT STRATEGY")
print("="*80)
print("\nStrategy:")
print("  1. First 15M candle of session (03:00, 10:00, 16:30)")
print("  2. Wait for breakout with candle confirmation")
print("  3. SL at opposite range, TP at 2R")
print("="*80)

# Use existing strategy_backtest to generate trades
print("\n1. Using existing strategy_backtest to generate trades...")
from strategy_backtest import TradingStrategy, StrategyConfig

# Simple config - just breakout with candle confirmation
config = StrategyConfig(
    reward_to_risk=2.0,
    allow_breakout=True,
    allow_pullback=False,
    allow_reversal=False,
    use_ema_filter=False,
    use_breakout_controls=False,
    wait_for_confirmation=False
)

all_trades = []

# Process Gold
print("\n2. Processing Gold (XAUUSD)...")
try:
    strategy_gold = TradingStrategy('XAUUSD5.csv', config)
    strategy_gold.load_data()
    strategy_gold.add_indicators(ema_periods_5m=[9, 21, 50], ema_200_1h=True, atr_period=14)
    
    # Add RSI
    delta = strategy_gold.df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    strategy_gold.df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume ratio
    strategy_gold.df['Volume_MA_20'] = strategy_gold.df['Volume'].rolling(20).mean()
    strategy_gold.df['Volume_Ratio'] = strategy_gold.df['Volume'] / strategy_gold.df['Volume_MA_20'].replace(0, np.nan)
    
    strategy_gold.identify_key_times()
    trades_df_gold = strategy_gold.backtest_strategy()
    
    if trades_df_gold is not None and len(trades_df_gold) > 0:
        trades_list = trades_df_gold.to_dict('records')
        for trade in trades_list:
            trade['Market'] = 'XAUUSD'
        all_trades.extend(trades_list)
        print(f"   ✅ Generated {len(trades_list)} trades")
    else:
        print(f"   ⚠️ No trades generated")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Process GBPJPY
print("\n3. Processing GBPJPY...")
try:
    strategy_gbpjpy = TradingStrategy('GBPJPY5.csv', config)
    strategy_gbpjpy.load_data()
    strategy_gbpjpy.add_indicators(ema_periods_5m=[9, 21, 50], ema_200_1h=True, atr_period=14)
    
    # Add RSI
    delta = strategy_gbpjpy.df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    strategy_gbpjpy.df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume ratio
    strategy_gbpjpy.df['Volume_MA_20'] = strategy_gbpjpy.df['Volume'].rolling(20).mean()
    strategy_gbpjpy.df['Volume_Ratio'] = strategy_gbpjpy.df['Volume'] / strategy_gbpjpy.df['Volume_MA_20'].replace(0, np.nan)
    
    strategy_gbpjpy.identify_key_times()
    trades_df_gbpjpy = strategy_gbpjpy.backtest_strategy()
    
    if trades_df_gbpjpy is not None and len(trades_df_gbpjpy) > 0:
        trades_list = trades_df_gbpjpy.to_dict('records')
        for trade in trades_list:
            trade['Market'] = 'GBPJPY'
        all_trades.extend(trades_list)
        print(f"   ✅ Generated {len(trades_list)} trades")
    else:
        print(f"   ⚠️ No trades generated")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Create dataset
print(f"\n4. Creating training dataset...")
print(f"   Total trades: {len(all_trades)}")

if len(all_trades) == 0:
    print("   ❌ NO TRADES! Check data and strategy.")
    exit(1)

df_trades = pd.DataFrame(all_trades)

# Add Won column
df_trades['Won'] = (df_trades['Status'] == 'TP_HIT').astype(int)

# Summary
wins = df_trades[df_trades['Won'] == 1]
losses = df_trades[df_trades['Won'] == 0]

print(f"\n   Wins: {len(wins)} ({len(wins)/len(df_trades)*100:.1f}%)")
print(f"   Losses: {len(losses)} ({len(losses)/len(df_trades)*100:.1f}%)")
print(f"   Total P&L: ${df_trades['P&L'].sum():.2f}")

# Save
output_file = 'simple_strategy_training_data.csv'
df_trades.to_csv(output_file, index=False)
print(f"\n   ✅ Saved: {output_file}")

# Train model
print(f"\n5. Training model...")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Feature columns (from entry info)
feature_cols = [
    'Risk', 'RangeWidth', 'RangeSizePct', 'BreakoutDistance', 'BreakoutBodyPct',
    'EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_5M', 'EMA_200_1H',
    'ATR', 'ATR_Ratio', 'ATR_Pct',
    'RSI',
    'Volume', 'Volume_Ratio',
    'EMA_9_Above_21', 'EMA_21_Above_50', 'Price_Above_EMA200_5M', 'Price_Above_EMA200_1H',
    'Trend_Score', 'Consolidation_Score', 'Is_Consolidating', 'Is_Tight_Range'
]

categorical_cols = ['Market', 'WindowType', 'Type']

# Get available columns
available_features = [col for col in feature_cols if col in df_trades.columns]
available_categorical = [col for col in categorical_cols if col in df_trades.columns]

print(f"   Using {len(available_features)} numeric features")
print(f"   Using {len(available_categorical)} categorical features")

# Prepare data
df_clean = df_trades.dropna(subset=available_features)

if len(df_clean) == 0:
    print("   ❌ NO VALID DATA!")
    exit(1)

X = df_clean[available_features + available_categorical]
y = df_clean['Won']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline
numeric_transformer = Pipeline([('scaler', RobustScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, available_features),
    ('cat', categorical_transformer, available_categorical)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=10, random_state=42))
])

# Train
print("   Training...")
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\n   Training Accuracy: {train_score*100:.2f}%")
print(f"   Test Accuracy: {test_score*100:.2f}%")

# Save
model_dir = 'simple_strategy_model'
import os
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, f'{model_dir}/model.pkl')
print(f"\n   ✅ Model saved: {model_dir}/model.pkl")

# Feature importance (HistGradientBoostingClassifier uses permutation importance)
print("   Calculating feature importance...")
from sklearn.inspection import permutation_importance

# Get feature names from preprocessor
feature_names_transformed = model.named_steps['preprocessor'].get_feature_names_out()

# Calculate permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
importances = perm_importance.importances_mean

# Ensure same length
min_len = min(len(feature_names_transformed), len(importances))
feature_names_transformed = feature_names_transformed[:min_len]
importances = importances[:min_len]

importance_df = pd.DataFrame({
    'Feature': feature_names_transformed,
    'Importance': importances
}).sort_values('Importance', ascending=False)

importance_df.to_csv(f'{model_dir}/feature_importance.csv', index=False)
print(f"   ✅ Feature importance saved")

print("\n" + "="*80)
print("✅ COMPLETE!")
print("="*80)
print(f"\nDataset: {output_file} ({len(df_trades)} trades)")
print(f"Model: {model_dir}/model.pkl")
print(f"Test Accuracy: {test_score*100:.2f}%")
print("\nTop 10 Features:")
print(importance_df.head(10).to_string(index=False))

