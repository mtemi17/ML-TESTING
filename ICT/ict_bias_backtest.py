"""
ICT Top-Down Bias Backtest
===========================
Backtests trading strategy with ICT multi-timeframe bias filtering
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import sys
import os

# Add parent directory to path to import strategy_backtest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_ict_topdown_bias import ICTTopDownBias


def backtest_with_ict_bias(csv_path: str, use_bias_filter: bool = True):
    """
    Backtest strategy with ICT bias filtering
    """
    print("=" * 70)
    print("ICT TOP-DOWN BIAS BACKTEST")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    df = pd.read_csv(
        csv_path,
        header=None,
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    
    df['DateTime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        format='%Y.%m.%d %H:%M'
    )
    df.set_index('DateTime', inplace=True)
    
    print(f"✅ Loaded {len(df)} candles")
    
    # Initialize ICT Bias System
    print("\n🔧 Initializing ICT Bias System...")
    bias_system = ICTTopDownBias(df)
    bias_system.resample_timeframes()
    
    # Run original strategy backtest (you'll need to import your strategy)
    # For now, we'll create a simple example
    print("\n📊 Running strategy backtest...")
    
    # This is a placeholder - you'll need to integrate with your actual strategy
    # For example, if you have strategy_backtest.py:
    # from strategy_backtest import TradingStrategy, StrategyConfig
    # strategy = TradingStrategy(csv_path)
    # strategy.load_data()
    # strategy.add_indicators()
    # strategy.identify_key_times()
    # trades = strategy.backtest_strategy()
    
    # For demonstration, create sample trades
    print("\n⚠️  NOTE: This is a template. Integrate with your actual strategy backtest.")
    print("   The bias filter will be applied to trades after they are generated.")
    
    # Example: Create sample trades (replace with actual strategy)
    sample_trades = []
    # ... your strategy logic here ...
    
    if not use_bias_filter:
        print("\n📊 Running WITHOUT bias filter (baseline)...")
        # Return all trades
        return pd.DataFrame(sample_trades)
    
    # Apply ICT bias filter
    if len(sample_trades) > 0:
        trades_df = pd.DataFrame(sample_trades)
        filtered_trades = bias_system.filter_trades_by_bias(trades_df)
        
        # Calculate performance
        if len(filtered_trades) > 0:
            print("\n📈 Performance Analysis:")
            print(f"  Total Trades: {len(filtered_trades)}")
            
            if 'P&L' in filtered_trades.columns:
                total_pnl = filtered_trades['P&L'].sum()
                win_rate = (filtered_trades['P&L'] > 0).sum() / len(filtered_trades) * 100
                avg_pnl = filtered_trades['P&L'].mean()
                
                print(f"  Total P&L: ${total_pnl:.2f}")
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  Avg P&L: ${avg_pnl:.2f}")
                
                # Compare with original (if available)
                if len(sample_trades) > len(filtered_trades):
                    original_pnl = trades_df['P&L'].sum() if 'P&L' in trades_df.columns else 0
                    original_wr = (trades_df['P&L'] > 0).sum() / len(trades_df) * 100 if 'P&L' in trades_df.columns else 0
                    
                    print(f"\n📊 Comparison (Original vs Filtered):")
                    print(f"  Original: {len(trades_df)} trades, ${original_pnl:.2f} P&L, {original_wr:.1f}% WR")
                    print(f"  Filtered: {len(filtered_trades)} trades, ${total_pnl:.2f} P&L, {win_rate:.1f}% WR")
                    print(f"  Improvement: ${total_pnl - original_pnl:.2f} P&L, {win_rate - original_wr:.1f}% WR")
        
        return filtered_trades
    
    return pd.DataFrame()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "../XAUUSD_5M.csv"
    
    try:
        # Test with bias filter
        filtered_trades = backtest_with_ict_bias(csv_path, use_bias_filter=True)
        
        # Save results
        if len(filtered_trades) > 0:
            output_path = "ict_bias_backtest_results.csv"
            filtered_trades.to_csv(output_path, index=False)
            print(f"\n✅ Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

