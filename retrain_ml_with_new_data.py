import pandas as pd
import numpy as np
import sys
import os

# Import ML training
exec(open('ml_model_training.py').read().replace(
    "df = pd.read_csv('advanced_analysis_results.csv')",
    "# Load combined data\n    df_existing = pd.read_csv('advanced_analysis_results.csv')\n    df_new = pd.read_csv('backtest_results_new_enhanced.csv')\n    \n    # Merge\n    df = pd.concat([df_existing, df_new], ignore_index=True)\n    df = df[df['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()\n    print(f'Combined dataset: {len(df)} trades')"
))

