#!/usr/bin/env python3
"""
Test EAAI_Full.mq5 logic with Gold data
Uses the 100% win rate conditions to verify calculations match
"""

import pandas as pd
import sys
from strategy_backtest import TradingStrategy, StrategyConfig

def main():
    print("="*80)
    print("TESTING EAAI_Full.mq5 LOGIC WITH GOLD DATA")
    print("="*80)
    
    # Find Gold data file
    data_file = None
    for filename in ["XAUUSD5.csv", "XAUUSD5 new.csv", "XAUUSD5.csv"]:
        try:
            # Quick check if file exists and has data
            test_df = pd.read_csv(filename, nrows=5, header=None)
            if len(test_df.columns) >= 6:  # Should have Date, Time, Open, High, Low, Close, Volume
                data_file = filename
                break
        except:
            continue
    
    if not data_file:
        print("❌ Could not find suitable Gold data file")
        return
    
    print(f"\n📊 Using data file: {data_file}")
    
    # Configure with 100% win rate conditions
    config = StrategyConfig(
        reward_to_risk=2.0,
        use_breakout_controls=True,
        breakout_initial_stop_ratio=0.6,
        breakout_max_mae_ratio=1.0,
        breakout_momentum_bar=5,
        breakout_momentum_min_gain=0.2,
        max_breakout_atr_multiple=0.55,  # 100% win rate condition
        max_atr_ratio=1.17,  # 100% win rate condition
        min_trend_score=0.67,  # 100% win rate condition
        max_consolidation_score=0.0,  # 100% win rate condition (must be exactly 0.0)
        min_range_atr_ratio=0.92,  # 100% win rate condition
        min_entry_offset_ratio=-0.25,
        max_entry_offset_ratio=1.00,
        first_bar_min_gain=-0.30,
        max_retest_depth_r=3.00,
        max_retest_bars=20,
        wait_for_confirmation=False,  # Immediate entry
        use_ema_filter=False,  # EMA 200 (1H) filter disabled
        allow_breakout=True,  # Enable breakout entries
        allow_pullback=False,  # Disable pullback entries
        allow_reversal=False  # Disable reversal entries
    )
    
    print("\n" + "="*80)
    print("CONFIGURATION (100% WIN RATE CONDITIONS)")
    print("="*80)
    print(f"Max Breakout ATR Multiple: {config.max_breakout_atr_multiple}")
    print(f"Max ATR Ratio: {config.max_atr_ratio}")
    print(f"Min Trend Score: {config.min_trend_score}")
    print(f"Max Consolidation Score: {config.max_consolidation_score} (must be 0.0)")
    print(f"Min Range ATR Ratio: {config.min_range_atr_ratio}")
    print(f"Entry Mode: {'IMMEDIATE' if not config.wait_for_confirmation else 'DELAY'}")
    print(f"EMA 200 (1H) Filter: {'ENABLED' if config.use_ema_filter else 'DISABLED'}")
    
    # Initialize strategy
    print("\n" + "="*80)
    print("INITIALIZING STRATEGY")
    print("="*80)
    strategy = TradingStrategy(data_file, config)
    
    # Load data
    print("\n📊 Loading data...")
    strategy.load_data()
    
    # Add indicators
    print("\n📈 Adding indicators...")
    strategy.add_indicators()
    
    # Identify key time windows
    print("\n🕐 Identifying key time windows...")
    strategy.identify_key_times()
    
    # Run backtest
    print("\n" + "="*80)
    print("RUNNING BACKTEST")
    print("="*80)
    trades_df = strategy.backtest_strategy()
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if trades_df is not None and len(trades_df) > 0:
        
        total_trades = len(trades_df)
        tp_hits = len(trades_df[trades_df['Status'] == 'TP_HIT'])
        sl_hits = len(trades_df[trades_df['Status'] == 'SL_HIT'])
        filter_exits = len(trades_df[trades_df['Status'] == 'FILTER_EXIT'])
        
        total_pnl = trades_df['P&L'].sum()
        avg_r = trades_df['R_Multiple'].mean() if len(trades_df) > 0 else 0
        
        # Win rate (TP vs SL only, excluding filter exits)
        win_rate_tp_sl = (tp_hits / (tp_hits + sl_hits) * 100) if (tp_hits + sl_hits) > 0 else 0
        
        # Win rate (including filter exits as losses)
        win_rate_all = (tp_hits / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   TP Hits: {tp_hits}")
        print(f"   SL Hits: {sl_hits}")
        print(f"   Filter Exits: {filter_exits}")
        print(f"\n💰 PERFORMANCE:")
        print(f"   Total P&L: ${total_pnl:.2f}")
        print(f"   Avg R-Multiple: {avg_r:.2f}")
        print(f"\n📈 WIN RATES:")
        print(f"   Win Rate (TP vs SL only): {win_rate_tp_sl:.2f}%")
        print(f"   Win Rate (All trades, filter exits = losses): {win_rate_all:.2f}%")
        
        # Check consolidation scores
        if 'Consolidation_Score_Entry' in trades_df.columns:
            consolidation_scores = trades_df['Consolidation_Score_Entry'].dropna()
            if len(consolidation_scores) > 0:
                print(f"\n🔍 CONSOLIDATION SCORE VERIFICATION:")
                print(f"   Unique values: {sorted(consolidation_scores.unique())}")
                print(f"   Trades with score = 0.0: {len(consolidation_scores[consolidation_scores == 0.0])}")
                print(f"   Trades with score = 0.5: {len(consolidation_scores[consolidation_scores == 0.5])}")
                print(f"   Trades with score = 1.0: {len(consolidation_scores[consolidation_scores == 1.0])}")
        
        # Check trend scores
        if 'Trend_Score_Entry' in trades_df.columns:
            trend_scores = trades_df['Trend_Score_Entry'].dropna()
            if len(trend_scores) > 0:
                print(f"\n🔍 TREND SCORE VERIFICATION:")
                print(f"   Min: {trend_scores.min():.3f}")
                print(f"   Max: {trend_scores.max():.3f}")
                print(f"   Mean: {trend_scores.mean():.3f}")
                print(f"   Trades with score >= 0.67: {len(trend_scores[trend_scores >= 0.67])}")
        
        # Save results
        output_file = "test_ea_full_gold_results.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        print("\n" + "="*80)
        print("✅ TEST COMPLETE")
        print("="*80)
        print("\n📝 KEY VERIFICATIONS:")
        print("   1. Consolidation Score should only be 0.0, 0.5, or 1.0")
        print("   2. All trades should have Consolidation Score = 0.0 (100% win rate condition)")
        print("   3. All trades should have Trend Score >= 0.67")
        print("   4. All trades should have Breakout ATR Multiple <= 0.55")
        print("   5. All trades should have ATR Ratio <= 1.17")
        print("   6. All trades should have Range ATR Ratio >= 0.92")
        
    else:
        print("❌ No trades generated!")
        print("   This could mean:")
        print("   - No breakouts detected")
        print("   - All breakouts filtered out by strict conditions")
        print("   - Data issues")

if __name__ == "__main__":
    main()

