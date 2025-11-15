#!/usr/bin/env python3
"""
Train ML Model on Simple Breakout Strategy

Strategy:
1. First 15-minute candle of each session (03:00, 10:00, 16:30)
2. Get high/low from that 15M candle
3. Switch to 5M timeframe
4. Wait for close above (bullish candle) or below (bearish candle) the range
5. Trade window: 3 hours after session start
6. SL at opposite range end, TP at 2R

Features to collect:
- EMAs (9, 21, 50, 200 on 5M and 1H)
- RSI
- Volume
- ATR
- All breakout metrics
- Candle patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING ML MODEL ON SIMPLE BREAKOUT STRATEGY")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")

data_files = {
    'XAUUSD': 'XAUUSD5.csv',
    'GBPJPY': 'GBPJPY5.csv'
}

all_trades = []

for market_name, file_path in data_files.items():
    print(f"\n   Loading {market_name} from {file_path}...")
    try:
        df = pd.read_csv(
            file_path,
            header=None,
            names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        # Combine Date and Time
        df['DateTime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%Y.%m.%d %H:%M',
            errors='coerce'
        )
        df = df.dropna(subset=['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        print(f"   Loaded {len(df)} rows")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        
        # ========================================================================
        # 2. ADD ALL INDICATORS
        # ========================================================================
        print(f"\n   Adding indicators for {market_name}...")
        
        # EMAs on 5M
        for period in [9, 21, 50, 200]:
            df[f'EMA_{period}_5M'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # EMA 200 on 1H
        df_1h = df.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        df_1h['EMA_200_1H'] = df_1h['Close'].ewm(span=200, adjust=False).mean()
        df['EMA_200_1H'] = df_1h['EMA_200_1H'].reindex(df.index, method='ffill')
        
        # ATR (14 period)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.ewm(span=14, adjust=False).mean()
        
        # ATR Ratio (current ATR vs 20-period average)
        df['ATR_MA_20'] = df['ATR'].rolling(20).mean()
        df['ATR_Ratio'] = df['ATR'] / df['ATR_MA_20'].replace(0, np.nan)
        
        # RSI (14 period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume indicators
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20'].replace(0, np.nan)
        
        # EMA relationships
        df['EMA_9_Above_21'] = (df['EMA_9_5M'] > df['EMA_21_5M']).astype(int)
        df['EMA_21_Above_50'] = (df['EMA_21_5M'] > df['EMA_50_5M']).astype(int)
        df['Price_Above_EMA200_5M'] = (df['Close'] > df['EMA_200_5M']).astype(int)
        df['Price_Above_EMA200_1H'] = (df['Close'] > df['EMA_200_1H']).astype(int)
        
        # Trend Score
        df['Trend_Score'] = (
            df['EMA_9_Above_21'] + 
            df['EMA_21_Above_50'] + 
            df['Price_Above_EMA200_1H']
        ) / 3.0
        
        # Consolidation
        df['Is_Consolidating'] = (df['ATR'] < df['ATR_MA_20'] * 0.7).astype(int)
        df['Is_Tight_Range'] = ((df['High'] - df['Low']) < (df['High'] - df['Low']).rolling(20).mean() * 0.8).astype(int)
        df['Consolidation_Score'] = (df['Is_Consolidating'] + df['Is_Tight_Range']) / 2.0
        
        # Candle patterns
        df['Candle_Body'] = np.abs(df['Close'] - df['Open'])
        df['Candle_Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Candle_Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Candle_Body_Pct'] = (df['Candle_Body'] / df['Close']) * 100
        df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
        df['Is_Bearish'] = (df['Close'] < df['Open']).astype(int)
        
        # ========================================================================
        # 3. IDENTIFY SESSIONS AND RANGES
        # ========================================================================
        print(f"\n   Identifying sessions for {market_name}...")
        
        sessions = [
            {'name': 'ASIAN', 'time': '03:00'},
            {'name': 'LONDON', 'time': '10:00'},
            {'name': 'NEW_YORK', 'time': '16:30'}
        ]
        
        session_ranges = []
        
        for date in df.index.date:
            for session in sessions:
                hour, minute = map(int, session['time'].split(':'))
                session_start = pd.Timestamp.combine(date, datetime.min.time().replace(hour=hour, minute=minute))
                
                if session_start not in df.index:
                    continue
                
                # Get first 15-minute candle (3 x 5M bars)
                range_start = session_start
                range_end = session_start + pd.Timedelta(minutes=15)
                
                range_data = df.loc[range_start:range_end]
                if len(range_data) < 3:
                    continue
                
                range_high = range_data['High'].max()
                range_low = range_data['Low'].min()
                
                # Trading window: 3 hours after session start
                window_start = range_end
                window_end = session_start + pd.Timedelta(hours=3)
                
                session_ranges.append({
                    'session': session['name'],
                    'date': date,
                    'session_start': session_start,
                    'range_start': range_start,
                    'range_end': range_end,
                    'range_high': range_high,
                    'range_low': range_low,
                    'window_start': window_start,
                    'window_end': window_end,
                    'market': market_name
                })
        
        print(f"   Found {len(session_ranges)} session ranges")
        
        # ========================================================================
        # 4. SIMULATE TRADES
        # ========================================================================
        print(f"\n   Simulating trades for {market_name}...")
        
        for sr in session_ranges:
            window_data = df.loc[sr['window_start']:sr['window_end']]
            
            for idx, row in window_data.iterrows():
                close = row['Close']
                open_price = row['Open']
                is_bullish = close > open_price
                is_bearish = close < open_price
                
                # BULLISH BREAKOUT: Close above range high AND bullish candle
                if close > sr['range_high'] and is_bullish:
                    # Calculate entry, SL, TP
                    entry_price = close
                    sl_price = sr['range_low']
                    risk = abs(entry_price - sl_price)
                    tp_price = entry_price + (risk * 2.0)  # 2R
                    
                    # Get all features at entry time
                    features = {
                        'Market': market_name,
                        'Session': sr['session'],
                        'EntryTime': idx,
                        'EntryPrice': entry_price,
                        'SL': sl_price,
                        'TP': tp_price,
                        'Risk': risk,
                        'RangeHigh': sr['range_high'],
                        'RangeLow': sr['range_low'],
                        'RangeWidth': sr['range_high'] - sr['range_low'],
                        'RangeSizePct': ((sr['range_high'] - sr['range_low']) / entry_price) * 100,
                        'BreakoutDistance': close - sr['range_high'],
                        'Direction': 'LONG',
                        # EMAs
                        'EMA_9_5M': row.get('EMA_9_5M', 0),
                        'EMA_21_5M': row.get('EMA_21_5M', 0),
                        'EMA_50_5M': row.get('EMA_50_5M', 0),
                        'EMA_200_5M': row.get('EMA_200_5M', 0),
                        'EMA_200_1H': row.get('EMA_200_1H', 0),
                        # ATR
                        'ATR': row.get('ATR', 0),
                        'ATR_Ratio': row.get('ATR_Ratio', 0),
                        'ATR_Pct': (row.get('ATR', 0) / entry_price) * 100 if entry_price > 0 else 0,
                        # RSI
                        'RSI': row.get('RSI', 50),
                        # Volume
                        'Volume': row.get('Volume', 0),
                        'Volume_Ratio': row.get('Volume_Ratio', 0),
                        # EMA relationships
                        'EMA_9_Above_21': row.get('EMA_9_Above_21', 0),
                        'EMA_21_Above_50': row.get('EMA_21_Above_50', 0),
                        'Price_Above_EMA200_5M': row.get('Price_Above_EMA200_5M', 0),
                        'Price_Above_EMA200_1H': row.get('Price_Above_EMA200_1H', 0),
                        # Scores
                        'Trend_Score': row.get('Trend_Score', 0),
                        'Consolidation_Score': row.get('Consolidation_Score', 0),
                        'Is_Consolidating': row.get('Is_Consolidating', 0),
                        'Is_Tight_Range': row.get('Is_Tight_Range', 0),
                        # Candle
                        'Candle_Body': row.get('Candle_Body', 0),
                        'Candle_Body_Pct': row.get('Candle_Body_Pct', 0),
                        'Candle_Upper_Wick': row.get('Candle_Upper_Wick', 0),
                        'Candle_Lower_Wick': row.get('Candle_Lower_Wick', 0),
                        'Is_Bullish': row.get('Is_Bullish', 0),
                        'Is_Bearish': row.get('Is_Bearish', 0),
                    }
                    
                    # Simulate trade outcome
                    future_data = df.loc[idx:]
                    hit_tp = False
                    hit_sl = False
                    exit_price = None
                    exit_time = None
                    
                    for future_idx, future_row in future_data.iterrows():
                        if future_row['High'] >= tp_price:
                            hit_tp = True
                            exit_price = tp_price
                            exit_time = future_idx
                            break
                        if future_row['Low'] <= sl_price:
                            hit_sl = True
                            exit_price = sl_price
                            exit_time = future_idx
                            break
                    
                    if hit_tp or hit_sl:
                        pnl = (exit_price - entry_price) if hit_tp else (exit_price - entry_price)
                        r_multiple = pnl / risk if risk > 0 else 0
                        
                        features['Status'] = 'TP_HIT' if hit_tp else 'SL_HIT'
                        features['ExitPrice'] = exit_price
                        features['ExitTime'] = exit_time
                        features['P&L'] = pnl
                        features['R_Multiple'] = r_multiple
                        features['Won'] = 1 if hit_tp else 0
                        
                        all_trades.append(features)
                    
                    # Only one trade per session
                    break
                
                # BEARISH BREAKOUT: Close below range low AND bearish candle
                elif close < sr['range_low'] and is_bearish:
                    # Calculate entry, SL, TP
                    entry_price = close
                    sl_price = sr['range_high']
                    risk = abs(entry_price - sl_price)
                    tp_price = entry_price - (risk * 2.0)  # 2R
                    
                    # Get all features at entry time
                    features = {
                        'Market': market_name,
                        'Session': sr['session'],
                        'EntryTime': idx,
                        'EntryPrice': entry_price,
                        'SL': sl_price,
                        'TP': tp_price,
                        'Risk': risk,
                        'RangeHigh': sr['range_high'],
                        'RangeLow': sr['range_low'],
                        'RangeWidth': sr['range_high'] - sr['range_low'],
                        'RangeSizePct': ((sr['range_high'] - sr['range_low']) / entry_price) * 100,
                        'BreakoutDistance': sr['range_low'] - close,
                        'Direction': 'SHORT',
                        # EMAs
                        'EMA_9_5M': row.get('EMA_9_5M', 0),
                        'EMA_21_5M': row.get('EMA_21_5M', 0),
                        'EMA_50_5M': row.get('EMA_50_5M', 0),
                        'EMA_200_5M': row.get('EMA_200_5M', 0),
                        'EMA_200_1H': row.get('EMA_200_1H', 0),
                        # ATR
                        'ATR': row.get('ATR', 0),
                        'ATR_Ratio': row.get('ATR_Ratio', 0),
                        'ATR_Pct': (row.get('ATR', 0) / entry_price) * 100 if entry_price > 0 else 0,
                        # RSI
                        'RSI': row.get('RSI', 50),
                        # Volume
                        'Volume': row.get('Volume', 0),
                        'Volume_Ratio': row.get('Volume_Ratio', 0),
                        # EMA relationships
                        'EMA_9_Above_21': row.get('EMA_9_Above_21', 0),
                        'EMA_21_Above_50': row.get('EMA_21_Above_50', 0),
                        'Price_Above_EMA200_5M': row.get('Price_Above_EMA200_5M', 0),
                        'Price_Above_EMA200_1H': row.get('Price_Above_EMA200_1H', 0),
                        # Scores
                        'Trend_Score': row.get('Trend_Score', 0),
                        'Consolidation_Score': row.get('Consolidation_Score', 0),
                        'Is_Consolidating': row.get('Is_Consolidating', 0),
                        'Is_Tight_Range': row.get('Is_Tight_Range', 0),
                        # Candle
                        'Candle_Body': row.get('Candle_Body', 0),
                        'Candle_Body_Pct': row.get('Candle_Body_Pct', 0),
                        'Candle_Upper_Wick': row.get('Candle_Upper_Wick', 0),
                        'Candle_Lower_Wick': row.get('Candle_Lower_Wick', 0),
                        'Is_Bullish': row.get('Is_Bullish', 0),
                        'Is_Bearish': row.get('Is_Bearish', 0),
                    }
                    
                    # Simulate trade outcome
                    future_data = df.loc[idx:]
                    hit_tp = False
                    hit_sl = False
                    exit_price = None
                    exit_time = None
                    
                    for future_idx, future_row in future_data.iterrows():
                        if future_row['Low'] <= tp_price:
                            hit_tp = True
                            exit_price = tp_price
                            exit_time = future_idx
                            break
                        if future_row['High'] >= sl_price:
                            hit_sl = True
                            exit_price = sl_price
                            exit_time = future_idx
                            break
                    
                    if hit_tp or hit_sl:
                        pnl = (entry_price - exit_price) if hit_tp else (entry_price - exit_price)
                        r_multiple = pnl / risk if risk > 0 else 0
                        
                        features['Status'] = 'TP_HIT' if hit_tp else 'SL_HIT'
                        features['ExitPrice'] = exit_price
                        features['ExitTime'] = exit_time
                        features['P&L'] = pnl
                        features['R_Multiple'] = r_multiple
                        features['Won'] = 1 if hit_tp else 0
                        
                        all_trades.append(features)
                    
                    # Only one trade per session
                    break
        
        print(f"   Generated {len([t for t in all_trades if t.get('Market') == market_name])} trades for {market_name}")
    
    except Exception as e:
        print(f"   ❌ Error processing {market_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# 5. CREATE DATASET
# ============================================================================
print(f"\n\n5. Creating training dataset...")
print(f"   Total trades: {len(all_trades)}")

if len(all_trades) == 0:
    print("   ❌ NO TRADES GENERATED! Check data files and session times.")
    exit(1)

df_trades = pd.DataFrame(all_trades)

# Summary
wins = df_trades[df_trades['Won'] == 1]
losses = df_trades[df_trades['Won'] == 0]

print(f"\n   Wins: {len(wins)} ({len(wins)/len(df_trades)*100:.1f}%)")
print(f"   Losses: {len(losses)} ({len(losses)/len(df_trades)*100:.1f}%)")
print(f"   Total P&L: ${df_trades['P&L'].sum():.2f}")
print(f"   Avg P&L: ${df_trades['P&L'].mean():.2f}")

# Save dataset
output_file = 'simple_strategy_training_data.csv'
df_trades.to_csv(output_file, index=False)
print(f"\n   ✅ Saved training dataset to: {output_file}")

# ============================================================================
# 6. TRAIN MODEL
# ============================================================================
print(f"\n\n6. Training ML model...")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Prepare features
feature_cols = [
    'Risk', 'RangeWidth', 'RangeSizePct', 'BreakoutDistance',
    'EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_5M', 'EMA_200_1H',
    'ATR', 'ATR_Ratio', 'ATR_Pct',
    'RSI',
    'Volume', 'Volume_Ratio',
    'EMA_9_Above_21', 'EMA_21_Above_50', 'Price_Above_EMA200_5M', 'Price_Above_EMA200_1H',
    'Trend_Score', 'Consolidation_Score', 'Is_Consolidating', 'Is_Tight_Range',
    'Candle_Body', 'Candle_Body_Pct', 'Candle_Upper_Wick', 'Candle_Lower_Wick',
    'Is_Bullish', 'Is_Bearish'
]

categorical_cols = ['Market', 'Session', 'Direction']

# Remove rows with missing values
df_clean = df_trades.dropna(subset=feature_cols)

if len(df_clean) == 0:
    print("   ❌ NO VALID DATA AFTER CLEANING!")
    exit(1)

X = df_clean[feature_cols + categorical_cols]
y = df_clean['Won']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, feature_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=10,
        random_state=42
    ))
])

# Train
print("   Training model...")
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\n   Training Accuracy: {train_score*100:.2f}%")
print(f"   Test Accuracy: {test_score*100:.2f}%")

# Feature importance
feature_names = feature_cols + list(model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols))
importances = model.named_steps['classifier'].feature_importances_

# Save model
model_dir = 'simple_strategy_model'
import os
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, f'{model_dir}/simple_strategy_model.pkl')
print(f"\n   ✅ Model saved to: {model_dir}/simple_strategy_model.pkl")

# Save feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

importance_df.to_csv(f'{model_dir}/feature_importance.csv', index=False)
print(f"   ✅ Feature importance saved")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\nDataset: {output_file} ({len(df_trades)} trades)")
print(f"Model: {model_dir}/simple_strategy_model.pkl")
print(f"Test Accuracy: {test_score*100:.2f}%")
print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

