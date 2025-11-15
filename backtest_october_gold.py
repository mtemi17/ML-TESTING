import pandas as pd
import numpy as np
from strategy_backtest import TradingStrategy, StrategyConfig
import joblib
from pathlib import Path
from ml_service import load_artifacts, prepare_sample, NUMERIC_FEATURES, CATEGORICAL_FEATURES, run_inference
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("OCTOBER GOLD BACKTEST: AI OFF vs AI ON")
print("="*80)

# Load ML model
print("\n1. Loading ML model...")
try:
    artifacts = load_artifacts(Path("gradient_model_multi"))
    print("   ✓ ML model loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading ML model: {e}")
    artifacts = None

# Load and filter October data
print("\n2. Loading Gold data...")
df = pd.read_csv("XAUUSD5 new data.csv", header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), format='%Y.%m.%d %H:%M')
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month

# Filter for October (check both 2024 and 2025)
october_data = df[df['Month'] == 10].copy()
print(f"   Total October candles: {len(october_data)}")
if len(october_data) == 0:
    print("   ✗ No October data found!")
    exit(1)

print(f"   Date range: {october_data['DateTime'].min()} to {october_data['DateTime'].max()}")

# Save October data to temp file
october_file = "XAUUSD5_October_temp.csv"
october_data[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']].to_csv(october_file, index=False, header=False)
print(f"   ✓ Saved October data to {october_file}")

# Configure strategy (Optimized mode)
config_optimized = StrategyConfig(
    reward_to_risk=2.0,
    pullback_timeout=12,
    use_ema_filter=False,
    allow_breakout=True,
    allow_pullback=False,
    allow_reversal=False,
    max_trades_per_window=1,
    use_breakout_controls=True,
    breakout_initial_stop_ratio=0.6,
    breakout_max_mae_ratio=0.6,
    breakout_momentum_bar=3,
    breakout_momentum_min_gain=0.3,
    max_breakout_atr_multiple=1.8,
    max_atr_ratio=1.3,
    min_trend_score=0.66,
    max_consolidation_score=0.10,
    min_entry_offset_ratio=-0.25,
    max_entry_offset_ratio=1.00,
    first_bar_min_gain=-0.20,
    max_retest_depth_r=1.80,
    max_retest_bars=12,
)

# === BACKTEST WITH AI OFF ===
print("\n" + "="*80)
print("3. BACKTEST WITH AI OFF (Optimized Strategy Only)")
print("="*80)

strategy_off = TradingStrategy(october_file, config_optimized)
strategy_off.load_data()
strategy_off.add_indicators(ema_periods_5m=[9,21,50], ema_200_1h=True, atr_period=14)
strategy_off.identify_key_times()
trades_off = strategy_off.backtest_strategy()
trades_df_off = pd.DataFrame(trades_off)

completed_off = trades_df_off[trades_df_off['Status'].isin(['TP_HIT','SL_HIT'])]
filtered_off = trades_df_off[trades_df_off['Status']=='FILTER_EXIT']

print(f"\n   Total trades attempted: {len(trades_df_off)}")
print(f"   Completed trades: {len(completed_off)}")
print(f"   Filter exits: {len(filtered_off)}")

if len(completed_off) > 0:
    wins_off = (completed_off['Status']=='TP_HIT').sum()
    losses_off = (completed_off['Status']=='SL_HIT').sum()
    win_rate_off = wins_off / len(completed_off) * 100
    total_pnl_off = completed_off['P&L'].sum()
    avg_pnl_off = completed_off['P&L'].mean()
    avg_r_off = completed_off['R_Multiple'].mean()
    
    print(f"\n   Results:")
    print(f"   - Wins: {wins_off} | Losses: {losses_off}")
    print(f"   - Win Rate: {win_rate_off:.1f}%")
    print(f"   - Total P&L: ${total_pnl_off:.2f}")
    print(f"   - Avg P&L: ${avg_pnl_off:.2f}")
    print(f"   - Avg R-Multiple: {avg_r_off:.2f}R")
else:
    print("   ⚠ No completed trades!")

# === BACKTEST WITH AI ON ===
print("\n" + "="*80)
print("4. BACKTEST WITH AI ON (Optimized Strategy + ML Filter)")
print("="*80)

if artifacts is None:
    print("   ✗ Cannot run AI ON test - ML model not loaded")
    trades_df_on = pd.DataFrame()
else:
    # Re-run backtest but filter trades with ML
    strategy_on = TradingStrategy(october_file, config_optimized)
    strategy_on.load_data()
    strategy_on.add_indicators(ema_periods_5m=[9,21,50], ema_200_1h=True, atr_period=14)
    strategy_on.identify_key_times()
    
    # We need to intercept trades before they're taken and check ML probability
    # Let's modify the approach: run backtest, then filter results by ML predictions
    trades_on = strategy_on.backtest_strategy()
    trades_df_on = pd.DataFrame(trades_on)
    
    # For each trade, get ML prediction
    print("\n   Getting ML predictions for all trades...")
    ml_predictions = []
    ai_threshold = 0.50
    
    for idx, trade in trades_df_on.iterrows():
        try:
            # Build feature dict for ML model
            features = {
                'EntryPrice': trade.get('EntryPrice', 0.0),
                'SL': trade.get('SL', 0.0),
                'TP': trade.get('TP', 0.0),
                'Risk': trade.get('Risk', 0.0),
                'BreakoutDistance': trade.get('BreakoutDistance', 0.0),
                'BreakoutBodyPct': trade.get('BreakoutBodyPct', 0.0),
                'BreakoutAtrMultiple': trade.get('BreakoutAtrMultiple', 0.0),
                'RangeWidth': trade.get('RangeWidth', 0.0),
                'RangeMid': trade.get('RangeMid', 0.0),
                'EntryOffset': trade.get('EntryOffset', 0.0),
                'RangeAtrRatio': trade.get('RangeAtrRatio', 0.0),
                'PriceAboveEMA200_5M': trade.get('PriceAboveEMA200_5M', 0),
                'ATR_Value': trade.get('ATR_Value', 0.0),
                'EMA_9_5M': trade.get('EMA_9_5M', 0.0) if 'EMA_9_5M' in trade else 0.0,
                'EMA_21_5M': trade.get('EMA_21_5M', 0.0) if 'EMA_21_5M' in trade else 0.0,
                'EMA_50_5M': trade.get('EMA_50_5M', 0.0) if 'EMA_50_5M' in trade else 0.0,
                'EMA_200_5M': trade.get('EMA_200_5M', 0.0) if 'EMA_200_5M' in trade else 0.0,
                'EMA_200_1H': trade.get('EMA_200_1H', 0.0) if 'EMA_200_1H' in trade else 0.0,
                'ATR': trade.get('ATR_Value', 0.0),
                'ATR_Pct': 0.0,
                'ATR_Ratio': trade.get('ATR_Ratio_Entry', 0.0),
                'Consolidation_Score': trade.get('Consolidation_Score_Entry', 0.0),
                'Trend_Score': trade.get('Trend_Score_Entry', 0.0),
                'Is_Consolidating': 0,
                'Is_Tight_Range': 0,
                'Price_Above_EMA200_5M': trade.get('PriceAboveEMA200_5M', 0),
                'Price_Above_EMA200_1H': 0,
                'EntryType': trade.get('EntryType', 'BREAKOUT'),
                'Type': trade.get('Type', 'BUY'),
                'WindowType': str(trade.get('WindowType', '0300')),
                'WindowID': str(trade.get('WindowID', '')),
                'Mode': trade.get('EntryType', 'BREAKOUT'),
                'Market': 'XAUUSD',
            }
            
            sample = prepare_sample(artifacts, features, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
            probability = run_inference(artifacts.pipeline, sample)
            ml_predictions.append(probability)
        except Exception as e:
            print(f"   Warning: Error predicting for trade {idx}: {e}")
            ml_predictions.append(0.0)
    
    trades_df_on['ML_Probability'] = ml_predictions
    trades_df_on['AI_Approved'] = trades_df_on['ML_Probability'] >= ai_threshold
    
    # Filter to only AI-approved trades
    ai_approved = trades_df_on[trades_df_on['AI_Approved']].copy()
    completed_on = ai_approved[ai_approved['Status'].isin(['TP_HIT','SL_HIT'])]
    filtered_on = ai_approved[ai_approved['Status']=='FILTER_EXIT']
    rejected = trades_df_on[~trades_df_on['AI_Approved']]
    
    print(f"\n   Total trades attempted: {len(trades_df_on)}")
    print(f"   AI Approved: {len(ai_approved)} ({len(ai_approved)/len(trades_df_on)*100:.1f}%)")
    print(f"   AI Rejected: {len(rejected)} ({len(rejected)/len(trades_df_on)*100:.1f}%)")
    print(f"   Completed trades (AI approved): {len(completed_on)}")
    print(f"   Filter exits (AI approved): {len(filtered_on)}")
    
    if len(completed_on) > 0:
        wins_on = (completed_on['Status']=='TP_HIT').sum()
        losses_on = (completed_on['Status']=='SL_HIT').sum()
        win_rate_on = wins_on / len(completed_on) * 100
        total_pnl_on = completed_on['P&L'].sum()
        avg_pnl_on = completed_on['P&L'].mean()
        avg_r_on = completed_on['R_Multiple'].mean()
        avg_prob_on = completed_on['ML_Probability'].mean()
        
        print(f"\n   Results:")
        print(f"   - Wins: {wins_on} | Losses: {losses_on}")
        print(f"   - Win Rate: {win_rate_on:.1f}%")
        print(f"   - Total P&L: ${total_pnl_on:.2f}")
        print(f"   - Avg P&L: ${avg_pnl_on:.2f}")
        print(f"   - Avg R-Multiple: {avg_r_on:.2f}R")
        print(f"   - Avg ML Probability: {avg_prob_on:.3f}")
    else:
        print("   ⚠ No completed trades!")

# === COMPARISON ===
print("\n" + "="*80)
print("5. COMPARISON: AI OFF vs AI ON")
print("="*80)

if len(completed_off) > 0 and len(completed_on) > 0:
    comparison = pd.DataFrame({
        'Metric': ['Trades Taken', 'Wins', 'Losses', 'Win Rate %', 'Total P&L', 'Avg P&L', 'Avg R-Multiple'],
        'AI OFF': [
            len(completed_off),
            wins_off,
            losses_off,
            f"{win_rate_off:.1f}%",
            f"${total_pnl_off:.2f}",
            f"${avg_pnl_off:.2f}",
            f"{avg_r_off:.2f}R"
        ],
        'AI ON': [
            len(completed_on),
            wins_on,
            losses_on,
            f"{win_rate_on:.1f}%",
            f"${total_pnl_on:.2f}",
            f"${avg_pnl_on:.2f}",
            f"{avg_r_on:.2f}R"
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Calculate improvements
    pnl_diff = total_pnl_on - total_pnl_off
    wr_diff = win_rate_on - win_rate_off
    trade_reduction = len(completed_off) - len(completed_on)
    
    print(f"\n   AI Impact:")
    print(f"   - Trade Reduction: {trade_reduction} trades ({trade_reduction/len(completed_off)*100:.1f}% fewer)")
    print(f"   - Win Rate Change: {wr_diff:+.1f}%")
    print(f"   - P&L Change: ${pnl_diff:+.2f}")
    
    # Save results
    completed_off.to_csv('analysis/gold_october_ai_off.csv', index=False)
    completed_on.to_csv('analysis/gold_october_ai_on.csv', index=False)
    comparison.to_csv('analysis/gold_october_comparison.csv', index=False)
    print("\n   ✓ Results saved to analysis/ folder")
else:
    print("   ⚠ Cannot compare - missing data")

# Cleanup
import os
if os.path.exists(october_file):
    os.remove(october_file)
    print(f"\n   ✓ Cleaned up temp file: {october_file}")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)

