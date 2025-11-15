"""
ICT Top-Down Bias Detection System
==================================
Implements multi-timeframe analysis:
- Weekly TF: Overall market bias
- 4H TF: Intermediate trend confirmation
- 5M/15M TF: Entry signals (only trade with bias)

Based on ICT Smart Money Concepts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class ICTTopDownBias:
    """
    Multi-timeframe bias detection system for ICT trading
    """
    
    def __init__(self, df_5m: pd.DataFrame):
        """
        Initialize with 5-minute data
        Will resample to higher timeframes internally
        """
        self.df_5m = df_5m.copy()
        self.df_weekly = None
        self.df_4h = None
        self.df_15m = None
        
        # Bias states
        self.weekly_bias = None  # 'BULLISH', 'BEARISH', 'NEUTRAL'
        self.h4_bias = None
        self.bias_strength = 0.0  # 0-1, how strong the bias is
        
    def resample_timeframes(self):
        """Resample 5M data to Weekly, 4H, and 15M timeframes"""
        print("📊 Resampling to multiple timeframes...")
        
        # Weekly timeframe
        self.df_weekly = self.df_5m.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        print(f"  ✅ Weekly: {len(self.df_weekly)} candles")
        
        # 4H timeframe
        self.df_4h = self.df_5m.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        print(f"  ✅ 4H: {len(self.df_4h)} candles")
        
        # 15M timeframe
        self.df_15m = self.df_5m.resample('15T').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        print(f"  ✅ 15M: {len(self.df_15m)} candles")
        
    def calculate_ema(self, df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
        """Calculate EMA on any timeframe"""
        return df[column].ewm(span=period, adjust=False).mean()
    
    def detect_weekly_bias(self, lookback_weeks: int = 8) -> Tuple[str, float]:
        """
        Detect overall market bias on Weekly timeframe
        Uses: Price vs EMA, Market Structure (HH/HL vs LH/LL)
        
        Returns: ('BULLISH'/'BEARISH'/'NEUTRAL', strength 0-1)
        """
        if self.df_weekly is None or len(self.df_weekly) < 20:
            return 'NEUTRAL', 0.0
        
        df = self.df_weekly.tail(lookback_weeks)
        
        # Calculate EMA 21 on Weekly (ICT standard)
        df['EMA_21_W'] = self.calculate_ema(df, 21)
        
        current_price = df['Close'].iloc[-1]
        ema_21 = df['EMA_21_W'].iloc[-1]
        
        # Market Structure Analysis
        highs = df['High'].tail(5).values
        lows = df['Low'].tail(5).values
        
        # Check for Higher Highs / Higher Lows (Bullish) or Lower Highs / Lower Lows (Bearish)
        hh_count = 0  # Higher Highs
        hl_count = 0  # Higher Lows
        lh_count = 0  # Lower Highs
        ll_count = 0  # Lower Lows
        
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                hh_count += 1
            elif highs[i] < highs[i-1]:
                lh_count += 1
                
            if lows[i] > lows[i-1]:
                hl_count += 1
            elif lows[i] < lows[i-1]:
                ll_count += 1
        
        # Calculate bias score
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Price vs EMA (40% weight)
        if current_price > ema_21:
            bullish_score += 0.4
        else:
            bearish_score += 0.4
        
        # Market Structure (60% weight)
        structure_score = (hh_count + hl_count) - (lh_count + ll_count)
        if structure_score > 0:
            bullish_score += 0.6 * min(abs(structure_score) / 4.0, 1.0)
        else:
            bearish_score += 0.6 * min(abs(structure_score) / 4.0, 1.0)
        
        # Determine bias
        if bullish_score > bearish_score + 0.2:
            bias = 'BULLISH'
            strength = bullish_score
        elif bearish_score > bullish_score + 0.2:
            bias = 'BEARISH'
            strength = bearish_score
        else:
            bias = 'NEUTRAL'
            strength = max(bullish_score, bearish_score)
        
        return bias, strength
    
    def detect_4h_bias(self, lookback_candles: int = 20) -> Tuple[str, float]:
        """
        Detect intermediate bias on 4H timeframe
        Must align with Weekly bias for confirmation
        
        Returns: ('BULLISH'/'BEARISH'/'NEUTRAL', strength 0-1)
        """
        if self.df_4h is None or len(self.df_4h) < 20:
            return 'NEUTRAL', 0.0
        
        df = self.df_4h.tail(lookback_candles)
        
        # Calculate EMAs on 4H
        df['EMA_9_4H'] = self.calculate_ema(df, 9)
        df['EMA_21_4H'] = self.calculate_ema(df, 21)
        df['EMA_50_4H'] = self.calculate_ema(df, 50)
        
        current_price = df['Close'].iloc[-1]
        ema_9 = df['EMA_9_4H'].iloc[-1]
        ema_21 = df['EMA_21_4H'].iloc[-1]
        ema_50 = df['EMA_50_4H'].iloc[-1]
        
        # EMA Alignment Score
        bullish_alignment = 0
        bearish_alignment = 0
        
        if current_price > ema_9:
            bullish_alignment += 1
        else:
            bearish_alignment += 1
            
        if ema_9 > ema_21:
            bullish_alignment += 1
        else:
            bearish_alignment += 1
            
        if ema_21 > ema_50:
            bullish_alignment += 1
        else:
            bearish_alignment += 1
        
        # Market Structure on 4H
        highs = df['High'].tail(10).values
        lows = df['Low'].tail(10).values
        
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        lh_count = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        # Calculate bias
        bullish_score = (bullish_alignment / 3.0) * 0.5 + (hh_count + hl_count) / 10.0 * 0.5
        bearish_score = (bearish_alignment / 3.0) * 0.5 + (lh_count + ll_count) / 10.0 * 0.5
        
        if bullish_score > bearish_score + 0.15:
            bias = 'BULLISH'
            strength = bullish_score
        elif bearish_score > bullish_score + 0.15:
            bias = 'BEARISH'
            strength = bearish_score
        else:
            bias = 'NEUTRAL'
            strength = max(bullish_score, bearish_score)
        
        return bias, strength
    
    def get_trade_bias(self, timestamp: pd.Timestamp) -> Dict[str, any]:
        """
        Get overall trade bias for a specific timestamp
        Combines Weekly + 4H analysis
        
        Returns dict with:
        - bias: 'BULLISH', 'BEARISH', 'NEUTRAL'
        - strength: 0-1
        - weekly_bias: Weekly timeframe bias
        - h4_bias: 4H timeframe bias
        - aligned: Whether Weekly and 4H are aligned
        """
        # Get Weekly bias (use most recent weekly candle)
        weekly_bias, weekly_strength = self.detect_weekly_bias()
        
        # Get 4H bias (use most recent 4H candle)
        h4_bias, h4_strength = self.detect_4h_bias()
        
        # Check alignment
        aligned = (weekly_bias == h4_bias) and (weekly_bias != 'NEUTRAL')
        
        # Combined bias (both must agree for strong bias)
        if aligned:
            combined_bias = weekly_bias
            combined_strength = (weekly_strength + h4_strength) / 2.0
        elif weekly_bias != 'NEUTRAL':
            # Weekly takes precedence if 4H is neutral
            combined_bias = weekly_bias
            combined_strength = weekly_strength * 0.7  # Reduced strength
        elif h4_bias != 'NEUTRAL':
            # 4H bias if Weekly is neutral
            combined_bias = h4_bias
            combined_strength = h4_strength * 0.6  # Lower strength
        else:
            combined_bias = 'NEUTRAL'
            combined_strength = 0.0
        
        return {
            'bias': combined_bias,
            'strength': combined_strength,
            'weekly_bias': weekly_bias,
            'weekly_strength': weekly_strength,
            'h4_bias': h4_bias,
            'h4_strength': h4_strength,
            'aligned': aligned
        }
    
    def filter_trades_by_bias(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Filter trades to only include those that align with bias
        
        Rules:
        - LONG trades only if bias is BULLISH
        - SHORT trades only if bias is BEARISH
        - Require minimum bias strength (default 0.5)
        """
        print("\n🎯 Filtering trades by ICT Top-Down Bias...")
        
        filtered_trades = []
        bias_stats = {
            'total': len(trades),
            'bullish_bias': 0,
            'bearish_bias': 0,
            'neutral_bias': 0,
            'aligned_trades': 0,
            'filtered_out': 0
        }
        
        for idx, trade in trades.iterrows():
            entry_time = trade['EntryTime'] if 'EntryTime' in trade else trade.name
            trade_type = trade['Type'] if 'Type' in trade else trade.get('Direction', 'BUY')
            
            # Get bias at entry time
            bias_info = self.get_trade_bias(entry_time)
            
            # Update stats
            if bias_info['bias'] == 'BULLISH':
                bias_stats['bullish_bias'] += 1
            elif bias_info['bias'] == 'BEARISH':
                bias_stats['bearish_bias'] += 1
            else:
                bias_stats['neutral_bias'] += 1
            
            # Filter logic
            is_long = trade_type.upper() in ['BUY', 'LONG']
            bias_ok = False
            
            if bias_info['bias'] == 'BULLISH' and is_long:
                bias_ok = True
            elif bias_info['bias'] == 'BEARISH' and not is_long:
                bias_ok = True
            
            # Require minimum strength
            if bias_info['strength'] < 0.5:
                bias_ok = False
            
            # Require alignment (both Weekly and 4H agree)
            if not bias_info['aligned']:
                bias_ok = False
            
            if bias_ok:
                # Add bias info to trade
                trade_copy = trade.copy()
                trade_copy['Weekly_Bias'] = bias_info['weekly_bias']
                trade_copy['H4_Bias'] = bias_info['h4_bias']
                trade_copy['Combined_Bias'] = bias_info['bias']
                trade_copy['Bias_Strength'] = bias_info['strength']
                trade_copy['Bias_Aligned'] = bias_info['aligned']
                filtered_trades.append(trade_copy)
                bias_stats['aligned_trades'] += 1
            else:
                bias_stats['filtered_out'] += 1
        
        result_df = pd.DataFrame(filtered_trades)
        
        # Print statistics
        print(f"\n📊 Bias Filter Results:")
        print(f"  Total Trades: {bias_stats['total']}")
        print(f"  Bullish Bias Periods: {bias_stats['bullish_bias']}")
        print(f"  Bearish Bias Periods: {bias_stats['bearish_bias']}")
        print(f"  Neutral Bias Periods: {bias_stats['neutral_bias']}")
        print(f"  ✅ Aligned Trades (Passed Filter): {bias_stats['aligned_trades']}")
        print(f"  ❌ Filtered Out: {bias_stats['filtered_out']}")
        print(f"  Filter Rate: {bias_stats['filtered_out']/bias_stats['total']*100:.1f}%")
        
        return result_df


def test_ict_bias_system(csv_path: str):
    """
    Test the ICT Top-Down Bias system
    """
    print("=" * 70)
    print("ICT TOP-DOWN BIAS SYSTEM TEST")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    df = pd.read_csv(
        csv_path,
        header=None,
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    
    # Combine Date and Time
    df['DateTime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        format='%Y.%m.%d %H:%M'
    )
    df.set_index('DateTime', inplace=True)
    
    print(f"✅ Loaded {len(df)} candles (5-minute data)")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Initialize ICT Bias System
    print("\n🔧 Initializing ICT Top-Down Bias System...")
    bias_system = ICTTopDownBias(df)
    bias_system.resample_timeframes()
    
    # Detect current bias
    print("\n📈 Detecting Market Bias...")
    weekly_bias, weekly_strength = bias_system.detect_weekly_bias()
    h4_bias, h4_strength = bias_system.detect_4h_bias()
    
    print(f"\n📊 Current Market Bias:")
    print(f"  Weekly TF: {weekly_bias} (Strength: {weekly_strength:.2f})")
    print(f"  4H TF: {h4_bias} (Strength: {h4_strength:.2f})")
    
    # Get combined bias
    current_time = df.index[-1]
    bias_info = bias_system.get_trade_bias(current_time)
    
    print(f"\n🎯 Combined Trade Bias:")
    print(f"  Bias: {bias_info['bias']}")
    print(f"  Strength: {bias_info['strength']:.2f}")
    print(f"  Aligned: {'✅ YES' if bias_info['aligned'] else '❌ NO'}")
    
    if bias_info['aligned']:
        print(f"\n✅ READY TO TRADE!")
        print(f"   Only take {bias_info['bias']} trades on 5M/15M timeframe")
    else:
        print(f"\n⚠️  WAIT FOR ALIGNMENT")
        print(f"   Weekly and 4H timeframes are not aligned")
    
    return bias_system


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path - adjust as needed
        csv_path = "../XAUUSD_5M.csv"  # Adjust to your data file
    
    try:
        bias_system = test_ict_bias_system(csv_path)
        print("\n✅ ICT Top-Down Bias System test completed!")
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find data file: {csv_path}")
        print("   Usage: python test_ict_topdown_bias.py <path_to_csv>")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

