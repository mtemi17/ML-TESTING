import pandas as pd
import numpy as np
import sys
import os

# Import the strategy class
sys.path.append('.')
from strategy_backtest import TradingStrategy

print("="*80)
print("PROCESSING NEW DATA: Oct 26 - Nov 6, 2025")
print("="*80)

# ============================================================================
# 1. RUN STRATEGY BACKTEST ON NEW DATA
# ============================================================================
print("\n1. Running strategy backtest on new filtered data...")

strategy = TradingStrategy('XAUUSD5_new_filtered.csv')

# Load data
strategy.load_data()

# Add indicators
strategy.add_indicators(ema_periods_5m=[9, 21, 50], ema_200_1h=True, atr_period=14)

# Identify key time windows
strategy.identify_key_times()

# Backtest
trades_df = strategy.backtest_strategy()

# Analyze results
strategy.analyze_results(trades_df)

# Save results
strategy.save_results(trades_df, 'backtest_results_new.csv')

# Save enhanced data
strategy.df.to_csv('XAUUSD5_new_with_indicators.csv')
print("\nNew data processed and saved!")

# ============================================================================
# 2. MERGE WITH EXISTING DATA
# ============================================================================
print("\n" + "="*80)
print("2. MERGING WITH EXISTING DATA")
print("="*80)

# Load existing backtest results
try:
    df_existing = pd.read_csv('backtest_results.csv')
    print(f"\nExisting trades: {len(df_existing)}")
except FileNotFoundError:
    print("\nNo existing backtest results found")
    df_existing = pd.DataFrame()

# Load new backtest results
df_new = pd.read_csv('backtest_results_new.csv')
print(f"New trades: {len(df_new)}")

# Merge
if len(df_existing) > 0:
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Remove duplicates based on EntryTime
    df_combined['EntryTime'] = pd.to_datetime(df_combined['EntryTime'])
    df_combined = df_combined.drop_duplicates(subset=['EntryTime'], keep='last')
    
    print(f"Combined total: {len(df_combined)} trades")
    print(f"New unique trades added: {len(df_combined) - len(df_existing)}")
else:
    df_combined = df_new.copy()
    print(f"Using new data only: {len(df_combined)} trades")

# Save combined
df_combined.to_csv('backtest_results_combined.csv', index=False)
print(f"\nCombined data saved to: backtest_results_combined.csv")

# ============================================================================
# 3. PREPARE FOR ML RETRAINING
# ============================================================================
print("\n" + "="*80)
print("3. PREPARING FOR ML RETRAINING")
print("="*80)

# We need to add the advanced analysis features to new trades
# Load advanced analysis script features
try:
    from advanced_profitability_analysis import *
    
    # Load the new data with indicators
    df_data_new = pd.read_csv('XAUUSD5_new_with_indicators.csv', index_col=0, parse_dates=True)
    
    # Add advanced features to new trades
    print("\nAdding advanced analysis features...")
    
    completed_new = df_new[df_new['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
    completed_new['EntryTime'] = pd.to_datetime(completed_new['EntryTime'])
    completed_new['ExitTime'] = pd.to_datetime(completed_new['ExitTime'])
    
    # Add duration
    completed_new['Duration_Minutes'] = (completed_new['ExitTime'] - completed_new['EntryTime']).dt.total_seconds() / 60
    
    # Add range size
    completed_new['RangeSize'] = completed_new['WindowHigh'] - completed_new['WindowLow']
    completed_new['RangeSizePct'] = (completed_new['RangeSize'] / completed_new['EntryPrice']) * 100
    
    # Add time features
    completed_new['EntryHour'] = completed_new['EntryTime'].dt.hour
    completed_new['EntryDayOfWeek'] = completed_new['EntryTime'].dt.dayofweek
    
    # Save enhanced new trades
    completed_new.to_csv('backtest_results_new_enhanced.csv', index=False)
    print(f"Enhanced new trades saved: {len(completed_new)} trades")
    
except Exception as e:
    print(f"Could not add advanced features: {e}")
    print("Will use basic features for ML training")

print("\n" + "="*80)
print("DATA PROCESSING COMPLETE!")
print("="*80)
print("\nNext: Run ML retraining with combined data")

