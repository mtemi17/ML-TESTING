"""
ICT Trading System - Standalone
================================
Complete ICT-based trading system using Smart Money Concepts:
- Top-Down Bias (Weekly → 4H → 5M/15M)
- Order Blocks
- Fair Value Gaps (FVG)
- Liquidity Zones
- Premium/Discount Zones
- Market Structure (BOS/CHoCH)

This is a NEW system, separate from the existing ORB strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ICTConfig:
    """Configuration for ICT Trading System"""
    # Bias settings
    weekly_lookback_weeks: int = 8
    h4_lookback_candles: int = 20
    min_bias_strength: float = 0.3  # Lowered to find more trades
    require_alignment: bool = False  # More lenient - don't require alignment
    
    # Order Block settings
    use_order_blocks: bool = True
    ob_lookback_bars: int = 50
    ob_min_candle_size: float = 0.05  # % of price (lowered for more opportunities)
    
    # FVG settings
    use_fvg: bool = True
    fvg_lookback_bars: int = 20
    fvg_min_gap_size: float = 0.02  # % of price (lowered for more opportunities)
    
    # Liquidity settings
    use_liquidity_zones: bool = True
    liquidity_lookback_bars: int = 50
    
    # Premium/Discount
    use_premium_discount: bool = True
    
    # Risk Management
    risk_reward_ratio: float = 2.0
    risk_percent: float = 1.0
    max_trades_per_day: int = 3


class ICTBiasDetector:
    """Multi-timeframe bias detection"""
    
    def __init__(self, df_5m: pd.DataFrame):
        self.df_5m = df_5m.copy()
        self.df_weekly = None
        self.df_4h = None
        self.df_15m = None
        
    def resample_timeframes(self):
        """Resample to Weekly, 4H, 15M"""
        print("📊 Resampling timeframes...")
        
        self.df_weekly = self.df_5m.resample('W').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        self.df_4h = self.df_5m.resample('4H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        self.df_15m = self.df_5m.resample('15T').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        print(f"  ✅ Weekly: {len(self.df_weekly)} candles")
        print(f"  ✅ 4H: {len(self.df_4h)} candles")
        print(f"  ✅ 15M: {len(self.df_15m)} candles")
    
    def detect_weekly_bias(self, lookback: int = 8) -> Tuple[str, float]:
        """Detect bias on Weekly timeframe"""
        if self.df_weekly is None or len(self.df_weekly) < 20:
            return 'NEUTRAL', 0.0
        
        df = self.df_weekly.tail(lookback)
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        current_price = df['Close'].iloc[-1]
        ema_21 = df['EMA_21'].iloc[-1]
        
        # Market Structure
        highs = df['High'].tail(5).values
        lows = df['Low'].tail(5).values
        
        hh_hl = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        hh_hl += sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        lh_ll = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        lh_ll += sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        bullish_score = 0.4 if current_price > ema_21 else 0.0
        bullish_score += 0.6 * min(hh_hl / 4.0, 1.0) if hh_hl > lh_ll else 0.0
        
        bearish_score = 0.4 if current_price < ema_21 else 0.0
        bearish_score += 0.6 * min(lh_ll / 4.0, 1.0) if lh_ll > hh_hl else 0.0
        
        if bullish_score > bearish_score + 0.2:
            return 'BULLISH', bullish_score
        elif bearish_score > bullish_score + 0.2:
            return 'BEARISH', bearish_score
        return 'NEUTRAL', max(bullish_score, bearish_score)
    
    def detect_4h_bias(self, lookback: int = 20) -> Tuple[str, float]:
        """Detect bias on 4H timeframe"""
        if self.df_4h is None or len(self.df_4h) < 20:
            return 'NEUTRAL', 0.0
        
        df = self.df_4h.tail(lookback)
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        current = df.iloc[-1]
        bullish_alignment = sum([
            current['Close'] > current['EMA_9'],
            current['EMA_9'] > current['EMA_21'],
            current['EMA_21'] > current['EMA_50']
        ])
        
        bearish_alignment = 3 - bullish_alignment
        
        bullish_score = (bullish_alignment / 3.0) * 0.5
        bearish_score = (bearish_alignment / 3.0) * 0.5
        
        # Market structure
        highs = df['High'].tail(10).values
        lows = df['Low'].tail(10).values
        hh_hl = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1] or lows[i] > lows[i-1])
        lh_ll = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1] or lows[i] < lows[i-1])
        
        bullish_score += (hh_hl / 10.0) * 0.5
        bearish_score += (lh_ll / 10.0) * 0.5
        
        if bullish_score > bearish_score + 0.15:
            return 'BULLISH', bullish_score
        elif bearish_score > bullish_score + 0.15:
            return 'BEARISH', bearish_score
        return 'NEUTRAL', max(bullish_score, bearish_score)
    
    def get_bias(self, timestamp: pd.Timestamp, config: ICTConfig) -> Dict:
        """Get combined bias"""
        weekly_bias, weekly_strength = self.detect_weekly_bias(config.weekly_lookback_weeks)
        h4_bias, h4_strength = self.detect_4h_bias(config.h4_lookback_candles)
        
        aligned = (weekly_bias == h4_bias) and (weekly_bias != 'NEUTRAL')
        
        if aligned:
            bias = weekly_bias
            strength = (weekly_strength + h4_strength) / 2.0
        elif weekly_bias != 'NEUTRAL':
            bias = weekly_bias
            strength = weekly_strength * 0.7
        elif h4_bias != 'NEUTRAL':
            bias = h4_bias
            strength = h4_strength * 0.6
        else:
            bias = 'NEUTRAL'
            strength = 0.0
        
        return {
            'bias': bias,
            'strength': strength,
            'weekly_bias': weekly_bias,
            'weekly_strength': weekly_strength,
            'h4_bias': h4_bias,
            'h4_strength': h4_strength,
            'aligned': aligned
        }


class ICTOrderBlockDetector:
    """Detect Order Blocks (institutional order zones)"""
    
    @staticmethod
    def find_order_blocks(df: pd.DataFrame, lookback: int = 50, 
                          min_size_pct: float = 0.1) -> List[Dict]:
        """
        Find Order Blocks:
        - Last bullish candle before strong move up (bullish OB)
        - Last bearish candle before strong move down (bearish OB)
        """
        order_blocks = []
        
        if len(df) < 10:
            return order_blocks
        
        df = df.tail(lookback + 10).copy()
        df['CandleSize'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # Look for Order Blocks: last candle before a strong move
        for i in range(2, len(df) - 2):
            current = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Check for strong move (at least 2x the current candle size)
            move_size = abs(next_candle['Close'] - current['Close']) / current['Close'] * 100
            
            # Bullish Order Block: Last candle before strong move up
            if next_candle['Close'] > current['High'] and move_size >= min_size_pct:
                # The current candle is the Order Block
                order_blocks.append({
                    'type': 'BULLISH_OB',
                    'time': current.name,
                    'high': current['High'],
                    'low': current['Low'],
                    'entry': current['Close'],  # Entry at close of OB
                    'target': next_candle['High']  # Target at next high
                })
            
            # Bearish Order Block: Last candle before strong move down
            elif next_candle['Close'] < current['Low'] and move_size >= min_size_pct:
                order_blocks.append({
                    'type': 'BEARISH_OB',
                    'time': current.name,
                    'high': current['High'],
                    'low': current['Low'],
                    'entry': current['Close'],
                    'target': next_candle['Low']
                })
        
        return order_blocks


class ICTFVDetector:
    """Detect Fair Value Gaps (FVG)"""
    
    @staticmethod
    def find_fvgs(df: pd.DataFrame, lookback: int = 20, 
                  min_gap_pct: float = 0.05) -> List[Dict]:
        """
        Find Fair Value Gaps:
        - Bullish FVG: Gap between previous high and current low
        - Bearish FVG: Gap between previous low and current high
        """
        fvgs = []
        
        if len(df) < 3:
            return fvgs
        
        df = df.tail(lookback + 2).copy()
        
        # Look for FVG: gap between candles
        for i in range(1, len(df)):
            prev = df.iloc[i]  # Previous candle (older)
            curr = df.iloc[i-1]  # Current candle (newer)
            
            # Bullish FVG: Previous high < Current low (gap up)
            if prev['High'] < curr['Low']:
                gap_size = (curr['Low'] - prev['High']) / curr['Close'] * 100
                if gap_size >= min_gap_pct:
                    fvgs.append({
                        'type': 'BULLISH_FVG',
                        'time': curr.name,
                        'top': curr['Low'],
                        'bottom': prev['High'],
                        'mid': (curr['Low'] + prev['High']) / 2.0
                    })
            
            # Bearish FVG: Previous low > Current high (gap down)
            elif prev['Low'] > curr['High']:
                gap_size = (prev['Low'] - curr['High']) / curr['Close'] * 100
                if gap_size >= min_gap_pct:
                    fvgs.append({
                        'type': 'BEARISH_FVG',
                        'time': curr.name,
                        'top': prev['Low'],
                        'bottom': curr['High'],
                        'mid': (prev['Low'] + curr['High']) / 2.0
                    })
        
        return fvgs


class ICTLiquidityDetector:
    """Detect Liquidity Zones (where stops cluster)"""
    
    @staticmethod
    def find_liquidity_zones(df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Find liquidity zones:
        - Above recent swing highs (buy-side liquidity for SHORT targets)
        - Below recent swing lows (sell-side liquidity for LONG targets)
        """
        if len(df) < lookback:
            return {'above': [], 'below': []}
        
        df = df.tail(lookback).copy()
        
        # Find swing highs (liquidity above)
        swing_highs = []
        for i in range(2, len(df) - 2):
            if df.iloc[i]['High'] > df.iloc[i-1]['High'] and \
               df.iloc[i]['High'] > df.iloc[i-2]['High'] and \
               df.iloc[i]['High'] > df.iloc[i+1]['High'] and \
               df.iloc[i]['High'] > df.iloc[i+2]['High']:
                swing_highs.append({
                    'time': df.iloc[i].name,
                    'price': df.iloc[i]['High']
                })
        
        # Find swing lows (liquidity below)
        swing_lows = []
        for i in range(2, len(df) - 2):
            if df.iloc[i]['Low'] < df.iloc[i-1]['Low'] and \
               df.iloc[i]['Low'] < df.iloc[i-2]['Low'] and \
               df.iloc[i]['Low'] < df.iloc[i+1]['Low'] and \
               df.iloc[i]['Low'] < df.iloc[i+2]['Low']:
                swing_lows.append({
                    'time': df.iloc[i].name,
                    'price': df.iloc[i]['Low']
                })
        
        return {
            'above': sorted(swing_highs, key=lambda x: x['price'], reverse=True)[:5],
            'below': sorted(swing_lows, key=lambda x: x['price'])[:5]
        }


class ICTTradingSystem:
    """Complete ICT Trading System"""
    
    def __init__(self, csv_path: str, config: Optional[ICTConfig] = None):
        self.config = config or ICTConfig()
        self.csv_path = csv_path
        self.df_5m = None
        self.bias_detector = None
        self.trades = []
        
    def load_data(self):
        """Load 5-minute data"""
        print("📥 Loading data...")
        self.df_5m = pd.read_csv(
            self.csv_path,
            header=None,
            names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        self.df_5m['DateTime'] = pd.to_datetime(
            self.df_5m['Date'].astype(str) + ' ' + self.df_5m['Time'].astype(str),
            format='%Y.%m.%d %H:%M'
        )
        self.df_5m.set_index('DateTime', inplace=True)
        print(f"✅ Loaded {len(self.df_5m)} candles")
        
        # Initialize bias detector
        self.bias_detector = ICTBiasDetector(self.df_5m)
        self.bias_detector.resample_timeframes()
    
    def find_ict_entries(self, timestamp: pd.Timestamp) -> List[Dict]:
        """Find ICT entry opportunities at given timestamp"""
        entries = []
        
        # Get current bias
        bias_info = self.bias_detector.get_bias(timestamp, self.config)
        
        # Check bias requirements
        if self.config.require_alignment:
            # Must be aligned if required
            if not bias_info['aligned']:
                return entries
        else:
            # If not requiring alignment, just need a non-neutral bias
            if bias_info['bias'] == 'NEUTRAL':
                return entries
        
        # Check minimum strength
        if bias_info['strength'] < self.config.min_bias_strength:
            return entries
        
        # Get data up to this timestamp
        df_up_to = self.df_5m[self.df_5m.index <= timestamp].tail(100)
        if len(df_up_to) < 20:
            return entries
        
        # Find Order Blocks
        if self.config.use_order_blocks:
            order_blocks = ICTOrderBlockDetector.find_order_blocks(
                df_up_to, self.config.ob_lookback_bars, 
                self.config.ob_min_candle_size
            )
            
            for ob in order_blocks:
                # Only take OB that aligns with bias
                if (bias_info['bias'] == 'BULLISH' and ob['type'] == 'BULLISH_OB') or \
                   (bias_info['bias'] == 'BEARISH' and ob['type'] == 'BEARISH_OB'):
                    entries.append({
                        'type': 'ORDER_BLOCK',
                        'direction': 'LONG' if ob['type'] == 'BULLISH_OB' else 'SHORT',
                        'entry': ob['entry'],
                        'stop': ob['low'] if ob['type'] == 'BULLISH_OB' else ob['high'],
                        'target': ob['target'],
                        'bias': bias_info['bias'],
                        'strength': bias_info['strength']
                    })
        
        # Find Fair Value Gaps
        if self.config.use_fvg:
            fvgs = ICTFVDetector.find_fvgs(
                df_up_to, self.config.fvg_lookback_bars,
                self.config.fvg_min_gap_size
            )
            
            for fvg in fvgs:
                # Only take FVG that aligns with bias
                if (bias_info['bias'] == 'BULLISH' and fvg['type'] == 'BULLISH_FVG') or \
                   (bias_info['bias'] == 'BEARISH' and fvg['type'] == 'BEARISH_FVG'):
                    # Entry at FVG midpoint
                    entry = fvg['mid']
                    if fvg['type'] == 'BULLISH_FVG':
                        stop = fvg['bottom']
                        risk = entry - stop
                        target = entry + (risk * self.config.risk_reward_ratio)
                    else:
                        stop = fvg['top']
                        risk = stop - entry
                        target = entry - (risk * self.config.risk_reward_ratio)
                    
                    entries.append({
                        'type': 'FAIR_VALUE_GAP',
                        'direction': 'LONG' if fvg['type'] == 'BULLISH_FVG' else 'SHORT',
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'bias': bias_info['bias'],
                        'strength': bias_info['strength']
                    })
        
        return entries
    
    def backtest(self):
        """Backtest the ICT system"""
        print("\n🚀 Starting ICT Trading System Backtest...")
        print("=" * 70)
        
        if self.df_5m is None:
            self.load_data()
        
        # Scan for entries (every 15 minutes to avoid over-trading)
        scan_times = self.df_5m.resample('15T').first().index
        
        trades_taken = 0
        trades_per_day = {}
        
        for timestamp in scan_times:
            # Check daily limit
            day = timestamp.date()
            if day not in trades_per_day:
                trades_per_day[day] = 0
            
            if trades_per_day[day] >= self.config.max_trades_per_day:
                continue
            
            # Find entries
            entries = self.find_ict_entries(timestamp)
            
            for entry in entries:
                if trades_per_day[day] >= self.config.max_trades_per_day:
                    break
                
                # Execute trade
                trade = self.execute_trade(entry, timestamp)
                if trade:
                    self.trades.append(trade)
                    trades_taken += 1
                    trades_per_day[day] += 1
        
        print(f"\n✅ Backtest complete: {len(self.trades)} trades taken")
        return pd.DataFrame(self.trades)
    
    def execute_trade(self, entry: Dict, entry_time: pd.Timestamp) -> Optional[Dict]:
        """Execute a trade and track P&L"""
        direction = entry['direction']
        entry_price = entry['entry']
        stop_price = entry['stop']
        target_price = entry['target']
        
        # Find exit
        df_after = self.df_5m[self.df_5m.index > entry_time]
        
        if len(df_after) == 0:
            return None
        
        # Check if stop or target hit
        exit_time = None
        exit_price = None
        status = None
        
        for idx, row in df_after.iterrows():
            if direction == 'LONG':
                if row['Low'] <= stop_price:
                    exit_time = idx
                    exit_price = stop_price
                    status = 'LOSS'
                    break
                elif row['High'] >= target_price:
                    exit_time = idx
                    exit_price = target_price
                    status = 'WIN'
                    break
            else:  # SHORT
                if row['High'] >= stop_price:
                    exit_time = idx
                    exit_price = stop_price
                    status = 'LOSS'
                    break
                elif row['Low'] <= target_price:
                    exit_time = idx
                    exit_price = target_price
                    status = 'WIN'
                    break
        
        if exit_time is None:
            # No exit found (end of data)
            return None
        
        # Calculate P&L
        if direction == 'LONG':
            pnl = (exit_price - entry_price) * 100  # Assuming 1 lot = $100 per point
        else:
            pnl = (entry_price - exit_price) * 100
        
        risk = abs(entry_price - stop_price) * 100
        r_multiple = pnl / risk if risk > 0 else 0
        
        return {
            'EntryTime': entry_time,
            'ExitTime': exit_time,
            'Type': direction,
            'EntryPrice': entry_price,
            'ExitPrice': exit_price,
            'StopPrice': stop_price,
            'TargetPrice': target_price,
            'P&L': pnl,
            'R_Multiple': r_multiple,
            'Status': status,
            'EntryType': entry['type'],
            'Bias': entry['bias'],
            'BiasStrength': entry['strength']
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "../XAUUSD_5M.csv"
    
    print("=" * 70)
    print("ICT TRADING SYSTEM - STANDALONE")
    print("=" * 70)
    
    config = ICTConfig(
        use_order_blocks=True,
        use_fvg=True,
        use_liquidity_zones=True,
        use_premium_discount=True,
        min_bias_strength=0.5,
        require_alignment=True
    )
    
    system = ICTTradingSystem(csv_path, config)
    trades = system.backtest()
    
    if len(trades) > 0:
        print("\n📊 Results:")
        print(f"  Total Trades: {len(trades)}")
        print(f"  Wins: {(trades['Status'] == 'WIN').sum()}")
        print(f"  Losses: {(trades['Status'] == 'LOSS').sum()}")
        print(f"  Win Rate: {(trades['Status'] == 'WIN').sum() / len(trades) * 100:.1f}%")
        print(f"  Total P&L: ${trades['P&L'].sum():.2f}")
        print(f"  Avg P&L: ${trades['P&L'].mean():.2f}")
        print(f"  Avg R Multiple: {trades['R_Multiple'].mean():.2f}")
        
        trades.to_csv("ict_trades.csv", index=False)
        print(f"\n✅ Trades saved to: ict_trades.csv")
    else:
        print("\n⚠️  No trades generated")

