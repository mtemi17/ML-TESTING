"""
Integration Example: ICT Bias + Existing Strategy
==================================================
Shows how to integrate top-down bias filtering with your existing strategy
"""

import pandas as pd
import sys
import os

# Add parent directory to import strategy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_ict_topdown_bias import ICTTopDownBias

# Import your strategy (adjust import as needed)
try:
    from strategy_backtest import TradingStrategy, StrategyConfig
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False
    print("⚠️  strategy_backtest.py not found - using example only")


def run_strategy_with_ict_bias(csv_path: str):
    """
    Run your existing strategy but filter trades by ICT top-down bias
    """
    print("=" * 70)
    print("STRATEGY + ICT TOP-DOWN BIAS")
    print("=" * 70)
    
    # Step 1: Initialize ICT Bias System
    print("\n📊 Step 1: Initializing ICT Bias System...")
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
    
    bias_system = ICTTopDownBias(df)
    bias_system.resample_timeframes()
    print("✅ ICT Bias System ready")
    
    # Step 2: Run your existing strategy
    print("\n📊 Step 2: Running existing strategy...")
    
    if STRATEGY_AVAILABLE:
        # Use your actual strategy
        config = StrategyConfig(
            reward_to_risk=2.0,
            use_ema_filter=True,
            # ... your config ...
        )
        
        strategy = TradingStrategy(csv_path, config)
        strategy.load_data()
        strategy.add_indicators()
        strategy.identify_key_times()
        trades = strategy.backtest_strategy()
        
        print(f"✅ Strategy generated {len(trades)} trades")
    else:
        # Example trades for demonstration
        print("⚠️  Using example trades (strategy_backtest not available)")
        trades = pd.DataFrame([
            {
                'EntryTime': df.index[100],
                'Type': 'BUY',
                'EntryPrice': 2000.0,
                'ExitPrice': 2010.0,
                'P&L': 10.0,
                'Status': 'WIN'
            },
            {
                'EntryTime': df.index[200],
                'Type': 'SELL',
                'EntryPrice': 2010.0,
                'ExitPrice': 2005.0,
                'P&L': 5.0,
                'Status': 'WIN'
            },
        ])
        print(f"✅ Example: {len(trades)} trades")
    
    # Step 3: Filter trades by ICT bias
    print("\n📊 Step 3: Filtering trades by ICT Top-Down Bias...")
    filtered_trades = bias_system.filter_trades_by_bias(trades)
    
    # Step 4: Compare results
    print("\n📊 Step 4: Results Comparison")
    print("=" * 70)
    
    if 'P&L' in trades.columns:
        original_pnl = trades['P&L'].sum()
        original_wr = (trades['P&L'] > 0).sum() / len(trades) * 100 if len(trades) > 0 else 0
        original_count = len(trades)
        
        if len(filtered_trades) > 0:
            filtered_pnl = filtered_trades['P&L'].sum()
            filtered_wr = (filtered_trades['P&L'] > 0).sum() / len(filtered_trades) * 100
            filtered_count = len(filtered_trades)
            
            print(f"\n📈 Original Strategy:")
            print(f"  Trades: {original_count}")
            print(f"  Total P&L: ${original_pnl:.2f}")
            print(f"  Win Rate: {original_wr:.1f}%")
            
            print(f"\n📈 With ICT Bias Filter:")
            print(f"  Trades: {filtered_count}")
            print(f"  Total P&L: ${filtered_pnl:.2f}")
            print(f"  Win Rate: {filtered_wr:.1f}%")
            
            print(f"\n📊 Improvement:")
            print(f"  Trades Filtered: {original_count - filtered_count} ({(1 - filtered_count/original_count)*100:.1f}% reduction)")
            print(f"  P&L Change: ${filtered_pnl - original_pnl:.2f}")
            print(f"  Win Rate Change: {filtered_wr - original_wr:.1f}%")
            
            # Calculate improvement metrics
            if original_count > 0:
                pnl_per_trade_original = original_pnl / original_count
                pnl_per_trade_filtered = filtered_pnl / filtered_count if filtered_count > 0 else 0
                
                print(f"\n💰 Efficiency:")
                print(f"  Original: ${pnl_per_trade_original:.2f} per trade")
                print(f"  Filtered: ${pnl_per_trade_filtered:.2f} per trade")
                print(f"  Improvement: ${pnl_per_trade_filtered - pnl_per_trade_original:.2f} per trade")
        else:
            print("\n⚠️  No trades passed the bias filter!")
            print("   This could mean:")
            print("   - Market bias is not aligned")
            print("   - Bias strength is too low")
            print("   - All trades were counter-trend")
    
    # Step 5: Save results
    if len(filtered_trades) > 0:
        output_file = "ict_bias_filtered_trades.csv"
        filtered_trades.to_csv(output_file, index=False)
        print(f"\n✅ Filtered trades saved to: {output_file}")
    
    return filtered_trades


def analyze_bias_distribution(trades: pd.DataFrame):
    """
    Analyze how trades are distributed across bias conditions
    """
    if 'Combined_Bias' not in trades.columns:
        print("\n⚠️  Bias information not available in trades")
        return
    
    print("\n📊 Bias Distribution Analysis:")
    print("=" * 70)
    
    bias_counts = trades['Combined_Bias'].value_counts()
    print("\nTrades by Bias:")
    for bias, count in bias_counts.items():
        pct = count / len(trades) * 100
        print(f"  {bias}: {count} ({pct:.1f}%)")
    
    if 'Bias_Strength' in trades.columns:
        print(f"\nAverage Bias Strength:")
        for bias in bias_counts.index:
            bias_trades = trades[trades['Combined_Bias'] == bias]
            avg_strength = bias_trades['Bias_Strength'].mean()
            print(f"  {bias}: {avg_strength:.2f}")
    
    if 'P&L' in trades.columns:
        print(f"\nP&L by Bias:")
        for bias in bias_counts.index:
            bias_trades = trades[trades['Combined_Bias'] == bias]
            total_pnl = bias_trades['P&L'].sum()
            win_rate = (bias_trades['P&L'] > 0).sum() / len(bias_trades) * 100
            print(f"  {bias}: ${total_pnl:.2f} P&L, {win_rate:.1f}% WR")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "../XAUUSD_5M.csv"
        print(f"⚠️  Using default path: {csv_path}")
        print("   Usage: python integrate_bias_with_strategy.py <path_to_csv>")
    
    try:
        filtered_trades = run_strategy_with_ict_bias(csv_path)
        
        if len(filtered_trades) > 0:
            analyze_bias_distribution(filtered_trades)
        
        print("\n✅ Integration test completed!")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find data file: {csv_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

