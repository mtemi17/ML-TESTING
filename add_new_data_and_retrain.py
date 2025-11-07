import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADDING NEW DATA AND RETRAINING ML MODEL")
print("="*80)

# ============================================================================
# 1. LOAD EXISTING DATA
# ============================================================================
print("\n1. Loading existing data...")
try:
    df_existing = pd.read_csv('backtest_results.csv')
    print(f"   Existing trades: {len(df_existing)}")
    print(f"   Date range: {df_existing['EntryTime'].min()} to {df_existing['EntryTime'].max()}")
except FileNotFoundError:
    print("   No existing data found. Will use new data only.")
    df_existing = pd.DataFrame()

# ============================================================================
# 2. LOAD NEW DATA
# ============================================================================
print("\n2. Loading new data...")
print("   Please provide the path to your new CSV file.")
print("   Expected format: Same as backtest_results.csv")
print("   Or provide new raw data file (XAUUSD5.csv format)")

# Check for common new data file names
new_data_files = [
    'new_backtest_results.csv',
    'additional_data.csv',
    'new_XAUUSD5.csv',
    'XAUUSD5_new.csv'
]

new_data_loaded = False
for file_path in new_data_files:
    try:
        print(f"\n   Trying to load: {file_path}...")
        df_new = pd.read_csv(file_path)
        print(f"   ✓ Found and loaded: {file_path}")
        print(f"   New trades: {len(df_new)}")
        new_data_loaded = True
        break
    except FileNotFoundError:
        continue

if not new_data_loaded:
    print("\n   No new data files found in common locations.")
    print("   Please specify the path to your new data file:")
    print("   (You can also manually add it to the script)")
    
    # Try to find any CSV files that might be new data
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'new' in f.lower()]
    if csv_files:
        print(f"\n   Found potential new data files:")
        for f in csv_files:
            print(f"     - {f}")
        print(f"\n   Using first found: {csv_files[0]}")
        df_new = pd.read_csv(csv_files[0])
        new_data_loaded = True
    else:
        print("\n   ERROR: No new data found!")
        print("   Please add your new data file to the ML TESTING folder")
        print("   and name it one of: new_backtest_results.csv, additional_data.csv")
        exit(1)

# ============================================================================
# 3. PROCESS NEW DATA
# ============================================================================
print("\n3. Processing new data...")

# Check if new data is in backtest_results format or raw data format
if 'EntryTime' in df_new.columns and 'P&L' in df_new.columns:
    print("   New data appears to be in backtest_results format")
    df_new_processed = df_new.copy()
    
    # Ensure it has all required columns
    required_cols = ['EntryTime', 'Type', 'EntryPrice', 'P&L', 'Status']
    missing_cols = [col for col in required_cols if col not in df_new_processed.columns]
    if missing_cols:
        print(f"   WARNING: Missing columns: {missing_cols}")
        print("   Attempting to process as raw data...")
        # Will handle below
    else:
        print("   ✓ New data has required columns")
        
elif 'Date' in df_new.columns and 'Time' in df_new.columns:
    print("   New data appears to be raw price data")
    print("   Need to run strategy backtest on new data first...")
    print("   This will be handled in the next step")
    df_new_raw = df_new.copy()
    
else:
    print("   WARNING: Unknown data format!")
    print("   Columns found:", df_new.columns.tolist())
    print("   Attempting to merge anyway...")
    df_new_processed = df_new.copy()

# ============================================================================
# 4. MERGE DATA
# ============================================================================
print("\n4. Merging existing and new data...")

if len(df_existing) > 0:
    # Check for duplicates
    if 'EntryTime' in df_new_processed.columns and 'EntryTime' in df_existing.columns:
        df_new_processed['EntryTime'] = pd.to_datetime(df_new_processed['EntryTime'])
        df_existing['EntryTime'] = pd.to_datetime(df_existing['EntryTime'])
        
        # Remove duplicates based on EntryTime
        before_merge = len(df_existing)
        df_combined = pd.concat([df_existing, df_new_processed], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['EntryTime'], keep='last')
        after_merge = len(df_combined)
        
        print(f"   Existing trades: {before_merge}")
        print(f"   New trades: {len(df_new_processed)}")
        print(f"   Combined (after dedup): {after_merge}")
        print(f"   Total new unique trades: {after_merge - before_merge}")
    else:
        df_combined = pd.concat([df_existing, df_new_processed], ignore_index=True)
        print(f"   Combined total: {len(df_combined)} trades")
else:
    df_combined = df_new_processed.copy()
    print(f"   Using new data only: {len(df_combined)} trades")

# Save combined data
df_combined.to_csv('backtest_results_combined.csv', index=False)
print(f"   Combined data saved to: backtest_results_combined.csv")

# ============================================================================
# 5. CHECK IF WE NEED TO RUN STRATEGY BACKTEST
# ============================================================================
print("\n5. Checking if strategy backtest is needed...")

if 'P&L' not in df_combined.columns or df_combined['P&L'].isna().all():
    print("   New data needs strategy backtest...")
    print("   Please run strategy_backtest.py on the new raw data first")
    print("   Or provide backtest results in the correct format")
else:
    print("   ✓ Data has P&L information, ready for ML training")

# ============================================================================
# 6. PREPARE FOR RETRAINING
# ============================================================================
print("\n6. Preparing data for ML retraining...")

# Filter completed trades
completed = df_combined[df_combined['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
print(f"   Completed trades: {len(completed)}")

if len(completed) < 100:
    print("   WARNING: Less than 100 completed trades!")
    print("   Model may not train well with this amount of data")

# Check data quality
missing_cols = []
required_ml_cols = ['EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_1H', 
                    'ATR', 'ATR_Pct', 'ATR_Ratio', 'Risk', 'WindowType']

for col in required_ml_cols:
    if col not in completed.columns:
        missing_cols.append(col)

if missing_cols:
    print(f"   WARNING: Missing ML features: {missing_cols}")
    print("   Need to run strategy_backtest.py to add indicators")
    print("   Or use enhanced_trades_for_ml.csv if available")
    
    # Try to load enhanced data
    try:
        df_enhanced = pd.read_csv('enhanced_trades_for_ml.csv')
        print("   Found enhanced_trades_for_ml.csv, using that instead")
        completed = df_enhanced[df_enhanced['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
    except FileNotFoundError:
        print("   ERROR: Cannot proceed without required features")
        print("   Please run strategy_backtest.py on new data first")
        exit(1)
else:
    print("   ✓ All required ML features present")

# ============================================================================
# 7. RETRAIN MODEL
# ============================================================================
print("\n" + "="*80)
print("7. RETRAINING ML MODEL WITH COMBINED DATA")
print("="*80)

# Import ML training functions
import sys
sys.path.append('.')

# Update the ml_model_training.py to use combined data
print("\n   Updating ML training script to use combined data...")

# Read the training script
with open('ml_model_training.py', 'r') as f:
    training_script = f.read()

# Modify to use combined data
training_script_modified = training_script.replace(
    "df = pd.read_csv('advanced_analysis_results.csv')",
    "df = pd.read_csv('backtest_results_combined.csv')"
)

# Save modified script temporarily
with open('ml_model_training_combined.py', 'w') as f:
    f.write(training_script_modified)

print("   Running ML training on combined data...")
import subprocess
result = subprocess.run(['python3', 'ml_model_training_combined.py'], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("   ✓ Model retrained successfully!")
    print("\n   Training output:")
    # Show last 30 lines of output
    output_lines = result.stdout.split('\n')
    for line in output_lines[-30:]:
        if line.strip():
            print(f"   {line}")
else:
    print("   ERROR during training:")
    print(result.stderr)
    print("\n   Attempting alternative approach...")
    
    # Try with existing enhanced data if available
    try:
        print("   Using enhanced_trades_for_ml.csv...")
        # We'll need to manually retrain
        print("   Please run ml_model_training.py manually after ensuring")
        print("   all data is properly formatted")

# ============================================================================
# 8. TEST UPDATED MODEL
# ============================================================================
print("\n" + "="*80)
print("8. TESTING UPDATED MODEL")
print("="*80)

print("\n   Testing updated model on combined data...")
result = subprocess.run(['python3', 'test_ml_on_original_data.py'], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("   ✓ Model tested successfully!")
    print("\n   Test results:")
    output_lines = result.stdout.split('\n')
    for line in output_lines[-40:]:
        if line.strip() and 'warn' not in line.lower():
            print(f"   {line}")
else:
    print("   Note: Test script may need updating for combined data")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("9. SUMMARY")
print("="*80)

print(f"\n   Original data: {len(df_existing) if len(df_existing) > 0 else 0} trades")
print(f"   New data: {len(df_new_processed)} trades")
print(f"   Combined total: {len(df_combined)} trades")
print(f"   Completed trades: {len(completed)}")

print("\n   Next steps:")
print("   1. Review the retrained model performance")
print("   2. Compare old vs new model metrics")
print("   3. Test on out-of-sample data if available")
print("   4. Deploy updated model if performance improved")

print("\n" + "="*80)
print("PROCESS COMPLETE!")
print("="*80)

