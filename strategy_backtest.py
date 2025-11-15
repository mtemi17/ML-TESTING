import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StrategyConfig:
    reward_to_risk: float = 2.0
    pullback_timeout: int = 12  # 5-minute bars
    use_ema_filter: bool = True
    allow_breakout: bool = False
    allow_pullback: bool = True
    allow_reversal: bool = True
    min_range_width: float = 0.0
    consolidation_threshold: float = 0.0
    max_trades_per_window: int = 1
    use_breakout_controls: bool = False
    breakout_max_mae_ratio: float = 0.6  # as fraction of initial risk
    breakout_momentum_bar: int = 3
    breakout_momentum_min_gain: float = 0.3  # in R multiples
    breakout_trail_trigger: float = 0.5  # optional trailing trigger (unused yet)
    breakout_initial_stop_ratio: float = 1.0  # fraction of range risk to use for initial stop
    use_failed_move_filter: bool = False
    failed_bearish_max_offset: float = -6.0   # longs (offset can be negative)
    failed_bullish_require_price_above_ema5m: bool = True
    failed_move_max_consolidation: float = 0.05
    max_breakout_atr_multiple: float = 1.8
    max_atr_ratio: float = 1.3
    min_trend_score: float = 0.66
    max_consolidation_score: float = 0.10
    min_range_atr_ratio: float = 0.0  # Min Range ATR Ratio (100% win rate: 0.92)
    min_entry_offset_ratio: float = -0.25
    max_entry_offset_ratio: float = 1.00
    first_bar_min_gain: float = -0.20
    max_retest_depth_r: float = 1.80
    max_retest_bars: int = 12
    wait_for_confirmation: bool = False  # Wait for post-entry conditions before entering
    confirmation_timeout_bars: int = 5  # Max bars to wait for confirmation
    flip_losers: bool = False  # Identify losers before entry and reverse them (old SL becomes new TP, same 2R for new SL)
    flip_losers_after_sl: bool = False  # Old behavior: reverse trades AFTER they hit SL

class TradingStrategy:
    def __init__(self, csv_path, config: Optional[StrategyConfig] = None):
        """Initialize the trading strategy with data loading"""
        self.csv_path = csv_path
        self.df = None
        self.trades = []
        self.config = config or StrategyConfig()
        
    def load_data(self):
        """Load and preprocess the CSV data"""
        print("Loading data...")
        self.df = pd.read_csv(
            self.csv_path, 
            header=None, 
            names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        # Combine Date and Time into a datetime column
        self.df['DateTime'] = pd.to_datetime(
            self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str),
            format='%Y.%m.%d %H:%M'
        )
        
        # Set DateTime as index for easier time-based operations
        self.df.set_index('DateTime', inplace=True)
        
        print(f"Loaded {len(self.df)} rows of data")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        
    def add_indicators(self, ema_periods_5m=[9, 21, 50], ema_200_1h=True, atr_period=14):
        """Add EMA indicators and ATR to the dataset"""
        print(f"\nAdding indicators...")
        
        # Add 5M timeframe EMAs
        print(f"  Adding 5M timeframe EMAs (periods: {ema_periods_5m})...")
        for period in ema_periods_5m:
            self.df[f'EMA_{period}_5M'] = self.df['Close'].ewm(span=period, adjust=False).mean()
            print(f"    - EMA_{period}_5M added")
        
        # 200-period EMA on 5M timeframe
        if 'EMA_200_5M' not in self.df.columns:
            print("  Adding 200 EMA on 5M timeframe...")
            self.df['EMA_200_5M'] = self.df['Close'].ewm(span=200, adjust=False).mean()
        self.df['Price_Above_EMA200_5M'] = (self.df['Close'] > self.df['EMA_200_5M']).astype(int)
        print("    - EMA_200_5M added")

        # Add 200 EMA on 1H timeframe
        if ema_200_1h:
            print("  Adding 200 EMA on 1H timeframe...")
            # Resample to 1H and calculate EMA_200
            df_1h = self.df.resample('1H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            # Calculate EMA_200 on 1H timeframe
            df_1h['EMA_200_1H'] = df_1h['Close'].ewm(span=200, adjust=False).mean()
            
            # Forward fill the 1H EMA_200 back to 5M timeframe
            self.df['EMA_200_1H'] = df_1h['EMA_200_1H'].reindex(self.df.index, method='ffill')
            print("    - EMA_200_1H added")
        
        # Add ATR (Average True Range)
        print(f"  Adding ATR (period: {atr_period})...")
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = true_range.ewm(span=atr_period, adjust=False).mean()
        print(f"    - ATR_{atr_period} added")
        
        # Add ATR-based indicators for ML features
        self.df['ATR_Pct'] = (self.df['ATR'] / self.df['Close']) * 100  # ATR as percentage of price
        self.df['ATR_Ratio'] = self.df['ATR'] / self.df['ATR'].rolling(window=20).mean()  # Current ATR vs recent average
        
        # Add consolidation detection
        print("  Adding consolidation detection...")
        # Consolidation = low volatility (low ATR relative to recent average)
        # Also check if price is staying in a tight range
        lookback = 20
        self.df['ATR_MA'] = self.df['ATR'].rolling(window=lookback).mean()
        self.df['Is_Consolidating'] = (self.df['ATR'] < self.df['ATR_MA'] * 0.7).astype(int)
        
        # Additional consolidation metric: price range over last N periods
        price_range = self.df['High'].rolling(window=lookback).max() - self.df['Low'].rolling(window=lookback).min()
        self.df['Price_Range_Pct'] = (price_range / self.df['Close']) * 100
        self.df['Is_Tight_Range'] = (self.df['Price_Range_Pct'] < self.df['Price_Range_Pct'].rolling(window=50).mean() * 0.8).astype(int)
        
        # Combined consolidation signal
        self.df['Consolidation_Score'] = (self.df['Is_Consolidating'] + self.df['Is_Tight_Range']) / 2
        
        print("    - Consolidation detection added")
        
        # Add trend indicators based on EMAs
        print("  Adding trend indicators...")
        self.df['EMA_9_Above_21'] = (self.df['EMA_9_5M'] > self.df['EMA_21_5M']).astype(int)
        self.df['EMA_21_Above_50'] = (self.df['EMA_21_5M'] > self.df['EMA_50_5M']).astype(int)
        self.df['Price_Above_EMA200_1H'] = (self.df['Close'] > self.df['EMA_200_1H']).astype(int)
        self.df['Trend_Score'] = (self.df['EMA_9_Above_21'] + self.df['EMA_21_Above_50'] + self.df['Price_Above_EMA200_1H']) / 3
        
        print("    - Trend indicators added")
        
        return self.df
    
    def identify_key_times(self):
        """Identify 15-minute candles at 3:00 AM, 10:00 AM, and 16:30 PM"""
        print("\nIdentifying key time windows...")
        
        target_times = ['03:00', '10:00', '16:30']
        
        self.df['KeyWindow'] = False
        self.df['WindowHigh'] = np.nan
        self.df['WindowLow'] = np.nan
        self.df['WindowType'] = ''
        self.df['WindowID'] = ''
        self.df['WindowStart'] = pd.NaT
        self.df['WindowTradeStart'] = pd.NaT
        self.df['WindowTradeEnd'] = pd.NaT
        self.df['ActiveWindowID'] = ''
        self.df['ActiveWindowType'] = ''
        self.df['RangeHigh'] = np.nan
        self.df['RangeLow'] = np.nan
        
        total_windows = 0
        
        for date, group in self.df.groupby(self.df.index.date):
            for target_time in target_times:
                start_time = pd.to_datetime(f"{date} {target_time}")
                end_time = start_time + pd.Timedelta(minutes=10)
                trade_start = start_time + pd.Timedelta(minutes=15)
                trade_end = start_time + pd.Timedelta(hours=3)
                window_id = f"{date}_{target_time.replace(':', '')}"
                time_label = target_time.replace(':', '')
                
                try:
                    window_data = self.df.loc[start_time:end_time]
                except KeyError:
                    continue
                
                if len(window_data) < 3:
                    continue
                
                window_high = window_data['High'].max()
                window_low = window_data['Low'].min()
                
                self.df.loc[start_time:end_time, 'KeyWindow'] = True
                self.df.loc[start_time:end_time, 'WindowHigh'] = window_high
                self.df.loc[start_time:end_time, 'WindowLow'] = window_low
                self.df.loc[start_time:end_time, 'WindowType'] = time_label
                self.df.loc[start_time:end_time, 'WindowID'] = window_id
                self.df.loc[start_time:end_time, 'WindowStart'] = start_time
                
                trade_mask = (self.df.index >= trade_start) & (self.df.index < trade_end)
                if trade_mask.any():
                    self.df.loc[trade_mask, 'ActiveWindowID'] = window_id
                    self.df.loc[trade_mask, 'ActiveWindowType'] = time_label
                    self.df.loc[trade_mask, 'RangeHigh'] = window_high
                    self.df.loc[trade_mask, 'RangeLow'] = window_low
                    self.df.loc[trade_mask, 'WindowTradeStart'] = trade_start
                    self.df.loc[trade_mask, 'WindowTradeEnd'] = trade_end
                
                total_windows += 1
        
        print(f"Found {total_windows} key windows (15-minute candles)")
        
        return self.df
    
    def backtest_strategy(self):
        """Backtest the opening-range breakout strategy with configurable logic."""
        config = self.config
        print("\n" + "=" * 60)
        print("BACKTESTING OPENING RANGE STRATEGY")
        print("=" * 60)
        print("\nStrategy Overview:")
        print("  - Range defined by first 15-minute candle at 03:00, 10:00, 16:30")
        print("  - Trading window: next 3 hours on 5M timeframe")
        print("  - Optional overlays:")
        print(f"      • Immediate breakout entries: {'ON' if config.allow_breakout else 'OFF'}")
        print(f"      • Pullback confirmation: {'ON' if config.allow_pullback else 'OFF'}")
        print(f"      • EMA(200, 1H) filter: {'ON' if config.use_ema_filter else 'OFF'}")
        print(f"      • Reversal (failed breakout fade): {'ON' if config.allow_reversal else 'OFF'}")
        print(f"      • Reward-to-risk: {config.reward_to_risk:.1f}R")
        print("=" * 60 + "\n")
        
        self.trades = []

        state = "idle"
        setup_info: Optional[Dict[str, Any]] = None
        entry_info: Optional[Dict[str, Any]] = None
        pending_breakout: Optional[Dict[str, Any]] = None  # For confirmation delay

        window_trade_counts = defaultdict(int)
        waiting_states = {
            "waiting_pullback_long",
            "waiting_pullback_short",
            "waiting_reversal_long",
            "waiting_reversal_short",
        }

        feature_cols = [
            'EMA_9_5M', 'EMA_21_5M', 'EMA_50_5M', 'EMA_200_5M', 'EMA_200_1H',
            'ATR', 'ATR_Pct', 'ATR_Ratio', 'Consolidation_Score',
            'Trend_Score', 'Is_Consolidating', 'Is_Tight_Range',
            'Price_Above_EMA200_5M', 'Price_Above_EMA200_1H',
            'ActiveWindowType', 'RangeHigh', 'RangeLow'
        ]

        def calculate_breakout_metrics(direction: str, close_price: float, range_high: float, range_low: float, row_data: pd.Series) -> Dict[str, float]:
            # EA uses previous bar's candle data: iHigh(1), iLow(1), iClose(1), iOpen(1)
            candle_range = row_data.get('High_Prev', row_data['High']) - row_data.get('Low_Prev', row_data['Low'])
            candle_body = abs(row_data.get('Close_Prev', row_data['Close']) - row_data.get('Open_Prev', row_data['Open']))
            body_pct = candle_body / candle_range if candle_range > 0 else np.nan

            if direction == 'LONG':
                distance = max(close_price - range_high, 0.0)
            else:
                distance = max(range_low - close_price, 0.0)

            # EA uses GetATRValue(1) - previous bar's ATR
            atr_value = row_data.get('ATR_Prev', row_data.get('ATR', np.nan))
            breakout_atr_multiple = distance / atr_value if atr_value and atr_value > 0 else np.nan
            range_width = range_high - range_low
            range_mid = (range_high + range_low) / 2
            range_atr_ratio = range_width / atr_value if atr_value and atr_value > 0 else np.nan

            # EA uses GetMAValue(handle, 1) - previous bar's EMA
            ema_5m = row_data.get('EMA_200_5M_Prev', row_data.get('EMA_200_5M', np.nan))
            price_above_ema5m = np.nan
            if not pd.isna(ema_5m):
                price_above_ema5m = 1 if close_price > ema_5m else 0

            # EA uses GetATRAverage(1, 20) - ATR ratio from previous bar
            atr_ratio = row_data.get('ATR_Ratio_Prev', row_data.get('ATR_Ratio', np.nan))
            atr_ma = row_data.get('ATR_MA', np.nan)

            return {
                'breakout_distance': distance,
                'breakout_body_pct': body_pct,
                'breakout_atr_multiple': breakout_atr_multiple,
                'range_width': range_width,
                'range_mid': range_mid,
                'entry_offset': close_price - range_mid,
                'range_atr_ratio': range_atr_ratio,
                'price_above_ema5m': price_above_ema5m,
                'atr_value': atr_value,
                'atr_ratio': atr_ratio,
                'atr_ma': atr_ma,
            }

        def normalise_window_id(value):
            if isinstance(value, str):
                value = value.strip()
                return value if value else None
            if pd.isna(value):
                return None
            return value
        
        # Create shifted columns to match EA's data collection (uses previous bar)
        # EA uses iClose(1), GetATRValue(1), etc. - all from previous completed bar
        self.df['Close_Prev'] = self.df['Close'].shift(1)
        self.df['High_Prev'] = self.df['High'].shift(1)
        self.df['Low_Prev'] = self.df['Low'].shift(1)
        self.df['Open_Prev'] = self.df['Open'].shift(1)
        self.df['ATR_Prev'] = self.df['ATR'].shift(1)
        self.df['ATR_Ratio_Prev'] = self.df['ATR_Ratio'].shift(1)
        self.df['EMA_9_5M_Prev'] = self.df['EMA_9_5M'].shift(1)
        self.df['EMA_21_5M_Prev'] = self.df['EMA_21_5M'].shift(1)
        self.df['EMA_50_5M_Prev'] = self.df['EMA_50_5M'].shift(1)
        self.df['EMA_200_5M_Prev'] = self.df['EMA_200_5M'].shift(1)
        self.df['Trend_Score_Prev'] = self.df['Trend_Score'].shift(1)
        self.df['Consolidation_Score_Prev'] = self.df['Consolidation_Score'].shift(1)
        # EMA_200_1H uses shift 0 in EA (current 1H bar), so keep as is
        
        for idx, row in self.df.iterrows():
            # EA uses previous bar's close for breakout detection: iClose(_Symbol, PERIOD_M5, 1)
            close_price = row.get('Close_Prev', row['Close'])  # Use previous bar, fallback to current
            high_price = row['High']  # Current bar for exit checks
            low_price = row['Low']  # Current bar for exit checks
            ema_value = row.get('EMA_200_1H', np.nan)  # EA uses shift 0 for 1H EMA

            # Manage pending breakout (confirmation delay)
            # KEY: We use ORIGINAL conditions from detection time, not new bar conditions
            # We only wait for momentum confirmation - price moving in our favor
            if pending_breakout is not None and config.wait_for_confirmation:
                pending_breakout['bars_waiting'] = pending_breakout.get('bars_waiting', 0) + 1
                
                # Check timeout
                if pending_breakout['bars_waiting'] > config.confirmation_timeout_bars:
                    pending_breakout = None
                    continue
                
                # Calculate current gain from detected price (using ORIGINAL detected price)
                detected_price = pending_breakout['detected_price']
                risk = pending_breakout['risk']  # ORIGINAL risk from detection
                direction = pending_breakout['direction']
                
                # Only check momentum - price moving in our favor
                # Don't re-check ATR, EMAs, or other conditions (they're frozen from detection time)
                if direction == 'LONG':
                    gain = close_price - detected_price
                else:
                    gain = detected_price - close_price
                
                gain_ratio = gain / risk if risk > 0 else 0.0
                
                # ONLY check momentum - that's it!
                # We already checked all other conditions at detection time
                if not pending_breakout.get('momentum_satisfied', False):
                    if gain_ratio >= config.breakout_momentum_min_gain:
                        pending_breakout['momentum_satisfied'] = True
                        # Momentum confirmed - enter immediately!
                    elif pending_breakout['bars_waiting'] >= config.breakout_momentum_bar:
                        # Timeout - momentum not achieved
                        pending_breakout = None
                        continue
                    else:
                        # Still waiting for momentum
                        continue
                
                # Momentum is satisfied - check if this trade looks like a loser BEFORE entering
                # We use ORIGINAL conditions from detection, not new bar conditions
                if pending_breakout.get('momentum_satisfied', False):
                    # Check if this trade would be a loser using post-entry filters
                    # Simulate what would happen if we entered now
                    is_loser = False
                    if config.flip_losers and config.use_breakout_controls:
                        # Calculate what the first bar gain would be
                        entry_price_sim = float(close_price)
                        old_sl = float(pending_breakout['stop_price'])
                        old_tp = float(pending_breakout['target_price'])
                        old_risk = float(risk)
                        
                        # Check first bar gain (using current bar's close vs detected price)
                        if direction == 'LONG':
                            first_bar_gain = close_price - detected_price
                        else:
                            first_bar_gain = detected_price - close_price
                        first_bar_gain_r = first_bar_gain / old_risk if old_risk > 0 else 0.0
                        
                        # Check retest depth
                        if direction == 'LONG':
                            min_price_sim = min(close_price, low_price, row.get('Low_Prev', low_price))
                            retest_depth_sim = max(pending_breakout.get('range_high', detected_price) - min_price_sim, 0.0)
                        else:
                            max_price_sim = max(close_price, high_price, row.get('High_Prev', high_price))
                            retest_depth_sim = max(max_price_sim - pending_breakout.get('range_low', detected_price), 0.0)
                        retest_depth_r = retest_depth_sim / old_risk if old_risk > 0 else 0.0
                        
                        # Check MAE
                        if direction == 'LONG':
                            adverse_sim = max(detected_price - min(close_price, low_price, row.get('Low_Prev', low_price)), 0.0)
                        else:
                            adverse_sim = max(max(close_price, high_price, row.get('High_Prev', high_price)) - detected_price, 0.0)
                        mae_ratio_sim = adverse_sim / old_risk if old_risk > 0 else 0.0
                        
                        # Identify as loser if it would fail post-entry filters
                        if (first_bar_gain_r < config.first_bar_min_gain or
                            (pending_breakout['bars_waiting'] <= config.max_retest_bars and retest_depth_r > config.max_retest_depth_r) or
                            mae_ratio_sim > config.breakout_max_mae_ratio):
                            is_loser = True
                    
                    # If identified as loser, flip it BEFORE entering
                    if is_loser:
                        # Reverse the trade: old SL becomes new TP, same 2R for new SL
                        old_direction = direction
                        old_sl = float(pending_breakout['stop_price'])
                        old_tp = float(pending_breakout['target_price'])
                        old_entry_sim = float(close_price)  # Entry at current price
                        old_risk = float(risk)
                        
                        # New trade: opposite direction
                        new_direction = 'SHORT' if old_direction == 'LONG' else 'LONG'
                        new_entry_price = float(close_price)
                        
                        if new_direction == 'SHORT':
                            # SHORT: TP at old SL (we know price will go DOWN to old SL), SL above entry
                            # For SHORT: TP below entry (price goes down), SL above entry (price goes up)
                            new_tp = old_sl  # Old SL becomes new TP (price goes DOWN to this level)
                            # Calculate distance to TP (reward)
                            reward_distance = abs(new_entry_price - new_tp)
                            # For 2R, risk = reward / 2, so SL = entry + risk
                            new_risk_distance = reward_distance / config.reward_to_risk
                            new_sl = float(new_entry_price + new_risk_distance)  # SL above entry
                        else:
                            # LONG: TP at old SL (we know price will go UP to old SL), SL below entry
                            # For LONG: TP above entry (price goes up), SL below entry (price goes down)
                            new_tp = old_sl  # Old SL becomes new TP (price goes UP to this level)
                            # Calculate distance to TP (reward)
                            reward_distance = abs(new_tp - new_entry_price)
                            # For 2R, risk = reward / 2, so SL = entry - risk
                            new_risk_distance = reward_distance / config.reward_to_risk
                            new_sl = float(new_entry_price - new_risk_distance)  # SL below entry
                        
                        new_risk = abs(new_entry_price - new_sl)
                        
                        # Create flipped entry
                        entry_info = {
                            'direction': new_direction,
                            'entry_time': idx,
                            'entry_price': new_entry_price,
                            'stop_price': new_sl,
                            'target_price': new_tp,
                            'risk': new_risk,
                            'window_id': pending_breakout['window_id'],
                            'window_type': pending_breakout['window_type'],
                            'entry_type': 'FLIPPED',
                            'bars_in_trade': 0,
                            'momentum_satisfied': True,
                            'max_mae_ratio': 0.0,
                            'max_retest_depth': 0.0,
                            'first_bar_checked': True,
                            'confirmed_before_entry': True,
                            'is_flipped_loser': True,  # Flag: this was flipped because identified as loser
                        }
                        
                        # Copy breakout metrics
                        for key in ['breakout_distance', 'breakout_body_pct', 'breakout_atr_multiple',
                                   'range_width', 'range_mid', 'entry_offset', 'range_atr_ratio',
                                   'price_above_ema5m', 'atr_value', 'atr_ratio', 'entry_offset_ratio',
                                   'trend_score', 'consolidation_score']:
                            if key in pending_breakout:
                                entry_info[key] = pending_breakout[key]
                        
                        # Copy indicator snapshot
                        if 'indicator_snapshot' in pending_breakout:
                            entry_info['indicator_snapshot'] = pending_breakout['indicator_snapshot']
                        
                        # Get range data from pending breakout (already stored)
                        entry_info['range_high'] = float(pending_breakout.get('range_high', detected_price))
                        entry_info['range_low'] = float(pending_breakout.get('range_low', detected_price))
                        
                        if new_direction == 'LONG':
                            entry_info['min_price_since_entry'] = float(close_price)
                            entry_info['max_price_since_entry'] = float(close_price)
                            state = "in_long"
                        else:
                            entry_info['min_price_since_entry'] = float(close_price)
                            entry_info['max_price_since_entry'] = float(close_price)
                            state = "in_short"
                        
                        pending_breakout = None
                        continue
                    else:
                        # Normal trade - enter as planned
                        entry_info = {
                            'direction': direction,
                            'entry_time': idx,
                            'entry_price': float(close_price),  # Enter at current price
                            'stop_price': float(pending_breakout['stop_price']),
                            'target_price': float(pending_breakout['target_price']),
                            'risk': float(risk),
                            'window_id': pending_breakout['window_id'],
                            'window_type': pending_breakout['window_type'],
                            'entry_type': 'BREAKOUT',
                            'bars_in_trade': 0,
                            'momentum_satisfied': True,  # Already confirmed
                            'max_mae_ratio': pending_breakout.get('max_mae_ratio', 0.0),
                            'max_retest_depth': pending_breakout.get('max_retest_depth', 0.0),
                            'first_bar_checked': True,  # Already checked before entry
                            'range_high': float(pending_breakout.get('range_high', close_price)),
                            'range_low': float(pending_breakout.get('range_low', close_price)),
                            'confirmed_before_entry': True,  # Flag: this trade was confirmed before entry
                        }
                        # Copy all breakout metrics
                        for key in ['breakout_distance', 'breakout_body_pct', 'breakout_atr_multiple',
                                   'range_width', 'range_mid', 'entry_offset', 'range_atr_ratio',
                                   'price_above_ema5m', 'atr_value', 'atr_ratio', 'entry_offset_ratio',
                                   'trend_score', 'consolidation_score']:
                            if key in pending_breakout:
                                entry_info[key] = pending_breakout[key]
                        
                        # Copy indicator snapshot
                        if 'indicator_snapshot' in pending_breakout:
                            entry_info['indicator_snapshot'] = pending_breakout['indicator_snapshot']
                        
                        if direction == 'LONG':
                            entry_info['min_price_since_entry'] = float(min(close_price, entry_info.get('range_low', close_price)))
                            entry_info['max_price_since_entry'] = float(close_price)
                            state = "in_long"
                        else:
                            entry_info['min_price_since_entry'] = float(close_price)
                            entry_info['max_price_since_entry'] = float(max(close_price, entry_info.get('range_high', close_price)))
                            state = "in_short"
                        
                        pending_breakout = None
                        continue
            
            # Manage open trades first
            if state in {"in_long", "in_short"} and entry_info is not None:
                entry_info['bars_in_trade'] = entry_info.get('bars_in_trade', 0) + 1

                filter_trigger = False
                if config.use_ema_filter and not pd.isna(ema_value):
                    if state == "in_long" and close_price < ema_value:
                        filter_trigger = True
                    elif state == "in_short" and close_price > ema_value:
                        filter_trigger = True

                control_exit = False
                control_exit_price = None
                if (
                    config.use_breakout_controls
                    and entry_info.get('entry_type') == 'BREAKOUT'
                ):
                    # If trade was confirmed before entry, we already checked these conditions
                    # Only check if conditions worsen significantly AFTER entry
                    confirmed_before = entry_info.get('confirmed_before_entry', False)
                    risk = entry_info.get('risk', 0.0)
                    if risk > 0:
                        if state == "in_long":
                            entry_info['min_price_since_entry'] = min(
                                entry_info.get('min_price_since_entry', entry_info['entry_price']),
                                low_price
                            )
                            adverse = max(entry_info['entry_price'] - low_price, 0.0)
                            gain = close_price - entry_info['entry_price']
                            current_retest_depth = max(
                                entry_info.get('range_high', entry_info['entry_price']) - entry_info['min_price_since_entry'],
                                0.0
                            ) / risk if risk > 0 else 0.0
                        else:
                            entry_info['max_price_since_entry'] = max(
                                entry_info.get('max_price_since_entry', entry_info['entry_price']),
                                high_price
                            )
                            adverse = max(high_price - entry_info['entry_price'], 0.0)
                            gain = entry_info['entry_price'] - close_price
                            current_retest_depth = max(
                                entry_info['max_price_since_entry'] - entry_info.get('range_low', entry_info['entry_price']),
                                0.0
                            ) / risk if risk > 0 else 0.0

                        entry_info['max_retest_depth'] = max(
                            entry_info.get('max_retest_depth', 0.0),
                            current_retest_depth
                        )

                        # Skip first bar check if already confirmed before entry
                        if (
                            not entry_info.get('first_bar_checked', False)
                            and entry_info['bars_in_trade'] == 1
                            and not confirmed_before  # Only check if not already confirmed
                        ):
                            first_bar_gain_r = gain / risk if risk > 0 else 0.0
                            if first_bar_gain_r < config.first_bar_min_gain:
                                control_exit = True
                                control_exit_price = close_price
                            entry_info['first_bar_checked'] = True
                        elif confirmed_before and entry_info['bars_in_trade'] == 1:
                            # Already checked before entry, just mark as checked
                            entry_info['first_bar_checked'] = True

                        # For confirmed trades, we already checked all conditions before entry
                        # Only track metrics, don't exit on them (trust the confirmation)
                        if confirmed_before:
                            # No post-entry filters for confirmed trades - only SL/TP
                            mae_ratio = adverse / risk if risk > 0 else 0.0
                            entry_info['max_mae_ratio'] = max(entry_info.get('max_mae_ratio', 0.0), mae_ratio)
                            entry_info['momentum_satisfied'] = True  # Already confirmed
                        else:
                            # For immediate entry trades, use normal post-entry filters
                            retest_threshold = config.max_retest_depth_r
                            mae_threshold = config.breakout_max_mae_ratio
                            
                            if (
                                not control_exit
                                and entry_info['bars_in_trade'] <= config.max_retest_bars
                                and entry_info['max_retest_depth'] > retest_threshold
                            ):
                                control_exit = True
                                control_exit_price = close_price

                            mae_ratio = adverse / risk if risk > 0 else 0.0
                            entry_info['max_mae_ratio'] = max(entry_info.get('max_mae_ratio', 0.0), mae_ratio)

                            if not control_exit:
                                if mae_ratio > mae_threshold:
                                    control_exit = True
                                    control_exit_price = close_price
                                else:
                                    entry_info.setdefault('momentum_satisfied', False)
                                    if not entry_info['momentum_satisfied']:
                                        gain_ratio = gain / risk if risk > 0 else 0.0
                                        if gain_ratio >= config.breakout_momentum_min_gain:
                                            entry_info['momentum_satisfied'] = True
                                        elif entry_info['bars_in_trade'] >= config.breakout_momentum_bar:
                                            control_exit = True
                                            control_exit_price = close_price

                if control_exit:
                    exit_price = control_exit_price if control_exit_price is not None else close_price
                    result = "FILTER_EXIT"
                elif filter_trigger:
                    exit_price = close_price
                    result = "FILTER_EXIT"
                else:
                    if state == "in_long":
                        stop_hit = low_price <= entry_info['stop_price']
                        target_hit = high_price >= entry_info['target_price']
                        if stop_hit:
                            exit_price = entry_info['stop_price']
                            result = "LOSS"
                        elif target_hit:
                            exit_price = entry_info['target_price']
                            result = "WIN"
                        else:
                            # Trade remains open
                            continue
                    else:  # in_short
                        stop_hit = high_price >= entry_info['stop_price']
                        target_hit = low_price <= entry_info['target_price']
                        if stop_hit:
                            exit_price = entry_info['stop_price']
                            result = "LOSS"
                        elif target_hit:
                            exit_price = entry_info['target_price']
                            result = "WIN"
                        else:
                            continue

                if entry_info['direction'] == 'LONG':
                    pnl = exit_price - entry_info['entry_price']
                else:
                    pnl = entry_info['entry_price'] - exit_price
                risk = entry_info['risk']
                rr_multiple = pnl / risk if risk != 0 else 0.0

                trade_record = {
                    'WindowID': entry_info['window_id'],
                    'WindowType': entry_info['window_type'],
                    'EntryType': entry_info['entry_type'],
                    'Type': 'BUY' if entry_info['direction'] == 'LONG' else 'SELL',
                    'EntryTime': entry_info['entry_time'],
                    'ExitTime': idx,
                    'EntryPrice': entry_info['entry_price'],
                    'ExitPrice': exit_price,
                    'SL': entry_info['stop_price'],
                    'TP': entry_info['target_price'],
                    'Risk': risk,
                    'P&L': pnl,
                    'R_Multiple': rr_multiple,
                    'Status': 'TP_HIT' if result == 'WIN' else ('SL_HIT' if result == 'LOSS' else 'FILTER_EXIT')
                }
                trade_record['BreakoutDistance'] = entry_info.get('breakout_distance')
                trade_record['BreakoutBodyPct'] = entry_info.get('breakout_body_pct')
                trade_record['BreakoutAtrMultiple'] = entry_info.get('breakout_atr_multiple')
                trade_record['RangeWidth'] = entry_info.get('range_width')
                trade_record['RangeMid'] = entry_info.get('range_mid')
                trade_record['EntryOffset'] = entry_info.get('entry_offset')
                trade_record['RangeAtrRatio'] = entry_info.get('range_atr_ratio')
                trade_record['PriceAboveEMA200_5M'] = entry_info.get('price_above_ema5m')
                trade_record['ATR_Value'] = entry_info.get('atr_value')
                trade_record['EntryOffsetRatio'] = entry_info.get('entry_offset_ratio')
                trade_record['ATR_Ratio_Entry'] = entry_info.get('atr_ratio')
                trade_record['Trend_Score_Entry'] = entry_info.get('trend_score')
                trade_record['Consolidation_Score_Entry'] = entry_info.get('consolidation_score')
                trade_record['MaxRetestDepth'] = entry_info.get('max_retest_depth')
                trade_record['MaxMaeRatio'] = entry_info.get('max_mae_ratio')

                indicator_snapshot = entry_info.get('indicator_snapshot', {})
                for col, value in indicator_snapshot.items():
                    trade_record[col] = value

                # Flip losers: if trade hit SL and flip_losers is enabled, reverse the trade
                if result == 'LOSS' and config.flip_losers:
                    # Reverse the trade: assume price continues in the direction it was going (towards SL)
                    # For LONG that hit SL: price went DOWN, so flip to SHORT targeting lower price
                    # For SHORT that hit SL: price went UP, so flip to LONG targeting higher price
                    old_direction = entry_info['direction']
                    old_sl = entry_info['stop_price']
                    old_tp = entry_info['target_price']
                    old_entry = entry_info['entry_price']
                    old_risk = risk
                    old_reward = abs(old_tp - old_entry)
                    
                    # New trade: opposite direction, targeting continuation of the move
                    new_direction = 'SHORT' if old_direction == 'LONG' else 'LONG'
                    new_entry_price = float(exit_price)  # Enter at old SL price
                    
                    if new_direction == 'SHORT':
                        # SHORT: entry at old SL, TP below entry (price goes down), SL above entry (same R)
                        # TP = entry - old_reward (price goes down by same amount as old reward)
                        # SL = entry + old_risk (price goes up by same amount as old risk)
                        new_tp = float(new_entry_price - old_reward)  # Target lower price
                        new_sl = float(new_entry_price + old_risk)  # Stop above entry (same R)
                    else:
                        # LONG: entry at old SL, TP above entry (price goes up), SL below entry (same R)
                        # TP = entry + old_reward (price goes up by same amount as old reward)
                        # SL = entry - old_risk (price goes down by same amount as old risk)
                        new_tp = float(new_entry_price + old_reward)  # Target higher price
                        new_sl = float(new_entry_price - old_risk)  # Stop below entry (same R)
                    
                    new_risk = abs(new_entry_price - new_sl)
                    
                    # Create reversed entry
                    entry_info = {
                        'direction': new_direction,
                        'entry_time': idx,
                        'entry_price': new_entry_price,
                        'stop_price': new_sl,
                        'target_price': new_tp,
                        'risk': new_risk,
                        'window_id': entry_info['window_id'],
                        'window_type': entry_info['window_type'],
                        'entry_type': 'FLIPPED',
                        'bars_in_trade': 0,
                        'momentum_satisfied': True,
                        'max_mae_ratio': 0.0,
                        'max_retest_depth': 0.0,
                        'first_bar_checked': True,
                        'confirmed_before_entry': True,
                        'original_trade_pnl': pnl,
                    }
                    
                    if new_direction == 'LONG':
                        entry_info['min_price_since_entry'] = float(close_price)
                        entry_info['max_price_since_entry'] = float(close_price)
                        state = "in_long"
                    else:
                        entry_info['min_price_since_entry'] = float(close_price)
                        entry_info['max_price_since_entry'] = float(close_price)
                        state = "in_short"
                    
                    # Record the original loss first
                    self.trades.append(trade_record)
                    
                    # Then start the flipped trade
                    continue

                self.trades.append(trade_record)

                window_trade_counts[entry_info['window_id']] += 1
                state = "idle"
                setup_info = None
                entry_info = None
                continue

            # Cancel pending breakout if window changed
            if pending_breakout is not None:
                active_window_id = normalise_window_id(row.get('ActiveWindowID'))
                if active_window_id != pending_breakout.get('window_id'):
                    pending_breakout = None
            
            active_window_id = normalise_window_id(row.get('ActiveWindowID'))
            active_window_type = row.get('ActiveWindowType', '')
            range_high = row.get('RangeHigh', np.nan)
            range_low = row.get('RangeLow', np.nan)
            trade_deadline = row.get('WindowTradeEnd')

            if state in waiting_states and setup_info is not None:
                if normalise_window_id(row.get('ActiveWindowID')) != setup_info['window_id']:
                    state = "idle"
                    setup_info = None
                    continue
                if pd.notna(trade_deadline) and idx >= setup_info['trade_deadline']:
                    state = "idle"
                    setup_info = None
                    continue

            if active_window_id is None:
                continue

            if window_trade_counts[active_window_id] >= config.max_trades_per_window:
                continue
            
            # Skip if we have a pending breakout for this window
            if pending_breakout is not None and pending_breakout.get('window_id') == active_window_id:
                continue

            if pd.isna(range_high) or pd.isna(range_low):
                continue

            range_width = range_high - range_low
            if range_width <= config.min_range_width:
                continue
            if config.consolidation_threshold > 0 and range_width <= config.consolidation_threshold:
                continue

            if config.use_ema_filter:
                if pd.isna(ema_value):
                    continue
                trend_bullish = close_price >= ema_value
                trend_bearish = close_price <= ema_value
            else:
                trend_bullish = True
                trend_bearish = True

            # ORB_SmartTrap logic: Require candle confirmation
            # Get previous bar's open for candle direction check
            open_price = row.get('Open_Prev', row.get('Open', close_price))
            is_bullish_candle = close_price > open_price
            is_bearish_candle = close_price < open_price
            
            # BULLISH BREAKOUT: Close above range high AND bullish candle
            breakout_up = close_price > range_high and is_bullish_candle
            # BEARISH BREAKOUT: Close below range low AND bearish candle
            breakout_down = close_price < range_low and is_bearish_candle

            if state == "idle":
                if config.allow_breakout:
                    def passes_breakout_filters(metrics: Dict[str, Any], atr_ratio_val: float,
                                               trend_score_val: float, consolidation_val: float,
                                               offset_ratio_val: float) -> bool:
                        if not config.use_breakout_controls:
                            return True
                        breakout_multiple = metrics.get('breakout_atr_multiple', np.nan)
                        if (
                            not pd.isna(breakout_multiple)
                            and breakout_multiple > config.max_breakout_atr_multiple
                        ):
                            return False
                        # Range ATR Ratio filter (100% WIN RATE)
                        range_atr_ratio = metrics.get('range_atr_ratio', np.nan)
                        if (
                            not pd.isna(range_atr_ratio)
                            and config.min_range_atr_ratio > 0
                            and range_atr_ratio < config.min_range_atr_ratio
                        ):
                            return False
                        if not pd.isna(atr_ratio_val) and atr_ratio_val > config.max_atr_ratio:
                            return False
                        if not pd.isna(trend_score_val) and trend_score_val < config.min_trend_score:
                            return False
                        if not pd.isna(consolidation_val) and consolidation_val > config.max_consolidation_score:
                            return False
                        if (
                            not pd.isna(offset_ratio_val)
                            and (
                                offset_ratio_val < config.min_entry_offset_ratio
                                or offset_ratio_val > config.max_entry_offset_ratio
                            )
                        ):
                            return False
                        return True

                    def create_entry_info(direction: str, risk: float, stop_price: float, target_price: float,
                                           metrics: Dict[str, Any], atr_ratio_val: float,
                                           trend_score_val: float, consolidation_val: float,
                                           offset_ratio_val: float) -> Dict[str, Any]:
                        # EA captures indicator snapshot from previous bar (shift 1)
                        # But we need to use current bar's values for the snapshot since we're entering now
                        # However, for filtering decisions, we used previous bar's values
                        indicator_snapshot = {}
                        for col in feature_cols:
                            if col in row.index:
                                # Use previous bar values for EMAs, ATR (matching EA's GetMAValue(1), GetATRValue(1))
                                if 'EMA' in col or col == 'ATR' or col == 'ATR_Ratio':
                                    prev_col = f"{col}_Prev"
                                    indicator_snapshot[col] = row.get(prev_col, row.get(col))
                                else:
                                    indicator_snapshot[col] = row.get(col)
                        info = {
                            'direction': 'LONG' if direction == 'LONG' else 'SHORT',
                            'entry_time': idx,
                            # EA enters at previous bar's close price (iClose(1))
                            'entry_price': float(close_price),  # close_price is already from previous bar
                            'stop_price': float(stop_price),
                            'target_price': target_price,
                            'risk': float(risk),
                            'window_id': active_window_id,
                            'window_type': active_window_type,
                            'entry_type': 'BREAKOUT',
                            'indicator_snapshot': indicator_snapshot,
                            'bars_in_trade': 0,
                            'momentum_satisfied': False,
                            'max_mae_ratio': 0.0,
                            'max_retest_depth': 0.0,
                            'first_bar_checked': False,
                            'trend_score': trend_score_val,
                            'consolidation_score': consolidation_val,
                            'atr_ratio': atr_ratio_val,
                            'entry_offset_ratio': offset_ratio_val,
                        }
                        info.update(metrics)
                        info['range_high'] = float(range_high)
                        info['range_low'] = float(range_low)
                        if direction == 'LONG':
                            info['min_price_since_entry'] = float(min(close_price, range_low))
                            info['max_price_since_entry'] = float(close_price)
                        else:
                            info['min_price_since_entry'] = float(close_price)
                            info['max_price_since_entry'] = float(max(close_price, range_high))
                        return info

                    if breakout_up and trend_bullish:
                        raw_risk = close_price - range_low
                        risk = raw_risk * config.breakout_initial_stop_ratio
                        if risk > 0:
                            stop_price = close_price - risk
                            target_price = float(close_price + config.reward_to_risk * risk)
                            breakout_metrics = calculate_breakout_metrics('LONG', close_price, range_high, range_low, row)
                            breakout_metrics['raw_risk'] = float(raw_risk)
                            range_width_value = breakout_metrics.get('range_width', np.nan)
                            offset_ratio = np.nan
                            if range_width_value and range_width_value != 0:
                                offset_ratio = breakout_metrics.get('entry_offset', np.nan) / range_width_value

                            # EA uses GetATRValue(1), GetMAValue(1) - previous bar's indicators
                            atr_ratio_val = row.get('ATR_Ratio_Prev', row.get('ATR_Ratio', np.nan))
                            trend_score_val = row.get('Trend_Score_Prev', row.get('Trend_Score', np.nan))
                            consolidation_val = row.get('Consolidation_Score_Prev', row.get('Consolidation_Score', np.nan))

                            if not passes_breakout_filters(breakout_metrics, atr_ratio_val, trend_score_val, consolidation_val, offset_ratio):
                                pass
                            else:
                                breakout_metrics['entry_offset_ratio'] = offset_ratio
                                
                                if config.wait_for_confirmation:
                                    # Create pending breakout instead of entering
                                    indicator_snapshot = {}
                                    for col in feature_cols:
                                        if col in row.index:
                                            if 'EMA' in col or col == 'ATR' or col == 'ATR_Ratio':
                                                prev_col = f"{col}_Prev"
                                                indicator_snapshot[col] = row.get(prev_col, row.get(col))
                                            else:
                                                indicator_snapshot[col] = row.get(col)
                                    
                                    pending_breakout = {
                                        'direction': 'LONG',
                                        'detected_time': idx,
                                        'detected_price': float(close_price),
                                        'stop_price': float(stop_price),
                                        'target_price': target_price,
                                        'risk': float(risk),
                                        'window_id': active_window_id,
                                        'window_type': active_window_type,
                                        'bars_waiting': 0,
                                        'max_mae_ratio': 0.0,
                                        'momentum_satisfied': False,
                                        'max_retest_depth': 0.0,
                                        'first_bar_checked': False,
                                        'range_high': float(range_high),
                                        'range_low': float(range_low),
                                        'indicator_snapshot': indicator_snapshot,
                                        'trend_score': trend_score_val,
                                        'consolidation_score': consolidation_val,
                                        'atr_ratio': atr_ratio_val,
                                    }
                                    pending_breakout.update(breakout_metrics)
                                else:
                                    # Immediate entry (old behavior)
                                    entry_info = create_entry_info(
                                        'LONG', risk, float(stop_price), target_price,
                                        breakout_metrics, atr_ratio_val, trend_score_val,
                                        consolidation_val, offset_ratio
                                    )
                                    state = "in_long"
                                continue

                    if breakout_down and trend_bearish:
                        raw_risk = range_high - close_price
                        risk = raw_risk * config.breakout_initial_stop_ratio
                        if risk > 0:
                            stop_price = close_price + risk
                            target_price = float(close_price - config.reward_to_risk * risk)
                            breakout_metrics = calculate_breakout_metrics('SHORT', close_price, range_high, range_low, row)
                            breakout_metrics['raw_risk'] = float(raw_risk)
                            range_width_value = breakout_metrics.get('range_width', np.nan)
                            offset_ratio = np.nan
                            if range_width_value and range_width_value != 0:
                                offset_ratio = breakout_metrics.get('entry_offset', np.nan) / range_width_value

                            # EA uses GetATRValue(1), GetMAValue(1) - previous bar's indicators
                            atr_ratio_val = row.get('ATR_Ratio_Prev', row.get('ATR_Ratio', np.nan))
                            trend_score_val = row.get('Trend_Score_Prev', row.get('Trend_Score', np.nan))
                            consolidation_val = row.get('Consolidation_Score_Prev', row.get('Consolidation_Score', np.nan))

                            if not passes_breakout_filters(breakout_metrics, atr_ratio_val, trend_score_val, consolidation_val, offset_ratio):
                                pass
                            else:
                                breakout_metrics['entry_offset_ratio'] = offset_ratio
                                
                                if config.wait_for_confirmation:
                                    # Create pending breakout instead of entering
                                    indicator_snapshot = {}
                                    for col in feature_cols:
                                        if col in row.index:
                                            if 'EMA' in col or col == 'ATR' or col == 'ATR_Ratio':
                                                prev_col = f"{col}_Prev"
                                                indicator_snapshot[col] = row.get(prev_col, row.get(col))
                                            else:
                                                indicator_snapshot[col] = row.get(col)
                                    
                                    pending_breakout = {
                                        'direction': 'SHORT',
                                        'detected_time': idx,
                                        'detected_price': float(close_price),
                                        'stop_price': float(stop_price),
                                        'target_price': target_price,
                                        'risk': float(risk),
                                        'window_id': active_window_id,
                                        'window_type': active_window_type,
                                        'bars_waiting': 0,
                                        'max_mae_ratio': 0.0,
                                        'momentum_satisfied': False,
                                        'max_retest_depth': 0.0,
                                        'first_bar_checked': False,
                                        'range_high': float(range_high),
                                        'range_low': float(range_low),
                                        'indicator_snapshot': indicator_snapshot,
                                        'trend_score': trend_score_val,
                                        'consolidation_score': consolidation_val,
                                        'atr_ratio': atr_ratio_val,
                                    }
                                    pending_breakout.update(breakout_metrics)
                                else:
                                    # Immediate entry (old behavior)
                                    entry_info = create_entry_info(
                                        'SHORT', risk, float(stop_price), target_price,
                                        breakout_metrics, atr_ratio_val, trend_score_val,
                                        consolidation_val, offset_ratio
                                    )
                                    state = "in_short"
                                continue

                if config.allow_pullback:
                    if breakout_up and trend_bullish:
                        breakout_metrics = calculate_breakout_metrics('LONG', close_price, range_high, range_low, row)
                        setup_info = {
                            'state': 'waiting_pullback_long',
                            'window_id': active_window_id,
                            'window_type': active_window_type,
                            'range_high': float(range_high),
                            'range_low': float(range_low),
                            'range_width': float(range_width),
                            'breakout_time': idx,
                            'breakout_close': float(close_price),
                            'bars_waited': 0,
                            'trade_deadline': trade_deadline,
                            'entry_type': 'PULLBACK',
                            'entry_ema': float(ema_value) if not pd.isna(ema_value) else None,
                            'breakout_metrics': breakout_metrics,
                        }
                        state = 'waiting_pullback_long'
                        continue
                    elif breakout_down and trend_bearish:
                        breakout_metrics = calculate_breakout_metrics('SHORT', close_price, range_high, range_low, row)
                        setup_info = {
                            'state': 'waiting_pullback_short',
                            'window_id': active_window_id,
                            'window_type': active_window_type,
                            'range_high': float(range_high),
                            'range_low': float(range_low),
                            'range_width': float(range_width),
                            'breakout_time': idx,
                            'breakout_close': float(close_price),
                            'bars_waited': 0,
                            'trade_deadline': trade_deadline,
                            'entry_type': 'PULLBACK',
                            'entry_ema': float(ema_value) if not pd.isna(ema_value) else None,
                            'breakout_metrics': breakout_metrics,
                        }
                        state = 'waiting_pullback_short'
                        continue

                if config.allow_reversal:
                    if breakout_down and trend_bullish:
                        breakout_metrics = calculate_breakout_metrics('LONG', close_price, range_high, range_low, row)
                        setup_info = {
                            'state': 'waiting_reversal_long',
                            'window_id': active_window_id,
                            'window_type': active_window_type,
                            'range_high': float(range_high),
                            'range_low': float(range_low),
                            'range_width': float(range_width),
                            'breakout_time': idx,
                            'breakout_close': float(close_price),
                            'bars_waited': 0,
                            'trade_deadline': trade_deadline,
                            'entry_type': 'REVERSAL',
                            'first_candle': None,
                            'breakout_metrics': breakout_metrics,
                        }
                        state = 'waiting_reversal_long'
                        continue
                    elif breakout_up and trend_bearish:
                        breakout_metrics = calculate_breakout_metrics('SHORT', close_price, range_high, range_low, row)
                        setup_info = {
                            'state': 'waiting_reversal_short',
                            'window_id': active_window_id,
                            'window_type': active_window_type,
                            'range_high': float(range_high),
                            'range_low': float(range_low),
                            'range_width': float(range_width),
                            'breakout_time': idx,
                            'breakout_close': float(close_price),
                            'bars_waited': 0,
                            'trade_deadline': trade_deadline,
                            'entry_type': 'REVERSAL',
                            'first_candle': None,
                            'breakout_metrics': breakout_metrics,
                        }
                        state = 'waiting_reversal_short'
                        continue

            elif state in {'waiting_pullback_long', 'waiting_pullback_short'} and setup_info is not None:
                setup_info['bars_waited'] += 1
                if setup_info['bars_waited'] > config.pullback_timeout:
                    state = 'idle'
                    setup_info = None
                    continue

                if config.use_ema_filter:
                    if state == 'waiting_pullback_long' and not trend_bullish:
                        state = 'idle'
                        setup_info = None
                        continue
                    if state == 'waiting_pullback_short' and not trend_bearish:
                        state = 'idle'
                        setup_info = None
                        continue

                indicator_snapshot = {col: row.get(col) for col in feature_cols if col in row.index}

                if state == 'waiting_pullback_long':
                    entry_level = setup_info['range_high']
                    if low_price <= entry_level <= high_price:
                        risk = entry_level - setup_info['range_low']
                        if risk <= 0:
                            state = 'idle'
                            setup_info = None
                            continue
                        indicator_snapshot = {col: row.get(col) for col in feature_cols if col in row.index}
                        entry_info = {
                            'direction': 'LONG',
                            'entry_time': idx,
                            'entry_price': float(entry_level),
                            'stop_price': float(setup_info['range_low']),
                            'target_price': float(entry_level + config.reward_to_risk * risk),
                            'risk': float(risk),
                            'window_id': setup_info['window_id'],
                            'window_type': setup_info['window_type'],
                            'entry_type': setup_info['entry_type'],
                            'indicator_snapshot': indicator_snapshot,
                        }
                        if 'breakout_metrics' in setup_info:
                            entry_info.update(setup_info['breakout_metrics'])
                        range_mid = (setup_info['range_high'] + setup_info['range_low']) / 2
                        entry_info['range_mid'] = range_mid
                        entry_info['entry_offset'] = entry_info['entry_price'] - range_mid
                        entry_info['range_width'] = setup_info['range_high'] - setup_info['range_low']
                        entry_info['bars_in_trade'] = 0
                        entry_info['momentum_satisfied'] = True
                        state = 'in_long'
                        setup_info = None
                else:
                    entry_level = setup_info['range_low']
                    if low_price <= entry_level <= high_price:
                        risk = setup_info['range_high'] - entry_level
                        if risk <= 0:
                            state = 'idle'
                            setup_info = None
                            continue
                        indicator_snapshot = {col: row.get(col) for col in feature_cols if col in row.index}
                        entry_info = {
                            'direction': 'SHORT',
                            'entry_time': idx,
                            'entry_price': float(entry_level),
                            'stop_price': float(setup_info['range_high']),
                            'target_price': float(entry_level - config.reward_to_risk * risk),
                            'risk': float(risk),
                            'window_id': setup_info['window_id'],
                            'window_type': setup_info['window_type'],
                            'entry_type': setup_info['entry_type'],
                            'indicator_snapshot': indicator_snapshot,
                        }
                        if 'breakout_metrics' in setup_info:
                            entry_info.update(setup_info['breakout_metrics'])
                        range_mid = (setup_info['range_high'] + setup_info['range_low']) / 2
                        entry_info['range_mid'] = range_mid
                        entry_info['entry_offset'] = entry_info['entry_price'] - range_mid
                        entry_info['range_width'] = setup_info['range_high'] - setup_info['range_low']
                        entry_info['bars_in_trade'] = 0
                        entry_info['momentum_satisfied'] = True
                        state = 'in_short'
                        setup_info = None

            elif state in {'waiting_reversal_long', 'waiting_reversal_short'} and setup_info is not None:
                setup_info['bars_waited'] += 1
                if setup_info['bars_waited'] > config.pullback_timeout:
                    state = 'idle'
                    setup_info = None
                    continue

                indicator_snapshot = {col: row.get(col) for col in feature_cols if col in row.index}
                first_candle = setup_info.get('first_candle')

                if state == 'waiting_reversal_short':
                    if first_candle is None:
                        if close_price < row['Open']:
                            setup_info['first_candle'] = {
                                'open': float(row['Open']),
                                'close': float(close_price),
                                'high': float(high_price),
                                'low': float(low_price),
                            }
                    else:
                        ema_condition = True if not config.use_ema_filter else close_price <= ema_value
                        if close_price < row['Open'] and ema_condition:
                            first_mid = (first_candle['open'] + first_candle['close']) / 2
                            second_mid = (row['Open'] + close_price) / 2
                            entry_price = (first_mid + second_mid) / 2
                            stop_price = first_candle['high']
                            risk = stop_price - entry_price
                            if risk <= 0:
                                state = 'idle'
                                setup_info = None
                                continue
                            indicator_snapshot = {col: row.get(col) for col in feature_cols if col in row.index}
                            entry_info = {
                                'direction': 'SHORT',
                                'entry_time': idx,
                                'entry_price': float(entry_price),
                                'stop_price': float(stop_price),
                                'target_price': float(entry_price - config.reward_to_risk * risk),
                                'risk': float(risk),
                                'window_id': setup_info['window_id'],
                                'window_type': setup_info['window_type'],
                                'entry_type': setup_info['entry_type'],
                                'indicator_snapshot': indicator_snapshot,
                            }
                            if 'breakout_metrics' in setup_info:
                                entry_info.update(setup_info['breakout_metrics'])
                            range_mid = (setup_info['range_high'] + setup_info['range_low']) / 2
                            entry_info['range_mid'] = range_mid
                            entry_info['entry_offset'] = entry_info['entry_price'] - range_mid
                            entry_info['range_width'] = setup_info['range_high'] - setup_info['range_low']
                            entry_info['bars_in_trade'] = 0
                            entry_info['momentum_satisfied'] = True
                            if config.use_failed_move_filter:
                                consolidation = row.get('Consolidation_Score')
                                if pd.notna(consolidation) and consolidation > config.failed_move_max_consolidation:
                                    state = 'idle'
                                    setup_info = None
                                    continue
                                price_above_ema5m = row.get('Price_Above_EMA200_5M')
                                if (
                                    config.failed_bullish_require_price_above_ema5m
                                    and price_above_ema5m != 1
                                ):
                                    state = 'idle'
                                    setup_info = None
                                    continue
                            state = 'in_short'
                            setup_info = None
                        elif close_price >= row['Open']:
                            setup_info['first_candle'] = None
                else:
                    if first_candle is None:
                        if close_price > row['Open']:
                            setup_info['first_candle'] = {
                                'open': float(row['Open']),
                                'close': float(close_price),
                                'high': float(high_price),
                                'low': float(low_price),
                            }
                    else:
                        ema_condition = True if not config.use_ema_filter else close_price >= ema_value
                        if close_price > row['Open'] and ema_condition:
                            first_mid = (first_candle['open'] + first_candle['close']) / 2
                            second_mid = (row['Open'] + close_price) / 2
                            entry_price = (first_mid + second_mid) / 2
                            stop_price = first_candle['low']
                            risk = entry_price - stop_price
                            if risk <= 0:
                                state = 'idle'
                                setup_info = None
                                continue
                            indicator_snapshot = {col: row.get(col) for col in feature_cols if col in row.index}
                            entry_info = {
                                'direction': 'LONG',
                                'entry_time': idx,
                                'entry_price': float(entry_price),
                                'stop_price': float(stop_price),
                                'target_price': float(entry_price + config.reward_to_risk * risk),
                                'risk': float(risk),
                                'window_id': setup_info['window_id'],
                                'window_type': setup_info['window_type'],
                                'entry_type': setup_info['entry_type'],
                                'indicator_snapshot': indicator_snapshot,
                            }
                            if 'breakout_metrics' in setup_info:
                                entry_info.update(setup_info['breakout_metrics'])
                            range_mid = (setup_info['range_high'] + setup_info['range_low']) / 2
                            entry_info['range_mid'] = range_mid
                            entry_info['entry_offset'] = entry_info['entry_price'] - range_mid
                            entry_info['range_width'] = setup_info['range_high'] - setup_info['range_low']
                            entry_info['bars_in_trade'] = 0
                            entry_info['momentum_satisfied'] = True
                            if config.use_failed_move_filter:
                                consolidation = row.get('Consolidation_Score')
                                if pd.notna(consolidation) and consolidation > config.failed_move_max_consolidation:
                                    state = 'idle'
                                    setup_info = None
                                    continue
                                if entry_info['entry_offset'] < config.failed_bearish_max_offset:
                                    state = 'idle'
                                    setup_info = None
                                    continue
                            state = 'in_long'
                            setup_info = None
                        elif close_price <= row['Open']:
                            setup_info['first_candle'] = None
        
        if state in {"in_long", "in_short"} and entry_info is not None:
            last_idx = self.df.index[-1]
            last_price = self.df.iloc[-1]['Close']
            if entry_info['direction'] == 'LONG':
                pnl = last_price - entry_info['entry_price']
            else:
                pnl = entry_info['entry_price'] - last_price
            risk = entry_info['risk']
            rr_multiple = pnl / risk if risk != 0 else 0.0

            trade_record = {
                'WindowID': entry_info['window_id'],
                'WindowType': entry_info['window_type'],
                'EntryType': entry_info['entry_type'],
                'Type': 'BUY' if entry_info['direction'] == 'LONG' else 'SELL',
                'EntryTime': entry_info['entry_time'],
                'ExitTime': last_idx,
                'EntryPrice': entry_info['entry_price'],
                'ExitPrice': last_price,
                'SL': entry_info['stop_price'],
                'TP': entry_info['target_price'],
                'Risk': risk,
                'P&L': pnl,
                'R_Multiple': rr_multiple,
                'Status': 'OPEN_AT_END'
            }
            trade_record['BreakoutDistance'] = entry_info.get('breakout_distance')
            trade_record['BreakoutBodyPct'] = entry_info.get('breakout_body_pct')
            trade_record['BreakoutAtrMultiple'] = entry_info.get('breakout_atr_multiple')
            trade_record['RangeWidth'] = entry_info.get('range_width')
            trade_record['RangeMid'] = entry_info.get('range_mid')
            trade_record['EntryOffset'] = entry_info.get('entry_offset')
            trade_record['RangeAtrRatio'] = entry_info.get('range_atr_ratio')
            trade_record['PriceAboveEMA200_5M'] = entry_info.get('price_above_ema5m')
            trade_record['ATR_Value'] = entry_info.get('atr_value')
            indicator_snapshot = entry_info.get('indicator_snapshot', {})
            for col, value in indicator_snapshot.items():
                trade_record[col] = value
            self.trades.append(trade_record)

        return pd.DataFrame(self.trades)
    
    def analyze_results(self, trades_df):
        """Analyze and display backtest results"""
        if len(trades_df) == 0:
            print("No trades executed!")
            return
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        # Filter completed trades
        completed_trades = trades_df[trades_df['Status'].isin(['TP_HIT', 'SL_HIT'])]
        
        if len(completed_trades) > 0:
            total_trades = len(completed_trades)
            winning_trades = len(completed_trades[completed_trades['P&L'] > 0])
            losing_trades = len(completed_trades[completed_trades['P&L'] < 0])
            win_rate = (winning_trades / total_trades) * 100
            
            total_pnl = completed_trades['P&L'].sum()
            avg_win = completed_trades[completed_trades['P&L'] > 0]['P&L'].mean() if winning_trades > 0 else 0
            avg_loss = completed_trades[completed_trades['P&L'] < 0]['P&L'].mean() if losing_trades > 0 else 0
            
            avg_r_multiple = completed_trades['R_Multiple'].mean()
            
            print(f"\nTotal Completed Trades: {total_trades}")
            print(f"Winning Trades: {winning_trades} ({win_rate:.2f}%)")
            print(f"Losing Trades: {losing_trades}")
            print(f"\nTotal P&L: ${total_pnl:.2f}")
            print(f"Average Win: ${avg_win:.2f}")
            print(f"Average Loss: ${avg_loss:.2f}")
            print(f"Average R-Multiple: {avg_r_multiple:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
                print(f"Profit Factor: {profit_factor:.2f}")
        
        # Show open trades
        open_trades = trades_df[trades_df['Status'] == 'OPEN_AT_END']
        if len(open_trades) > 0:
            print(f"\nOpen Trades at End: {len(open_trades)}")
        
        # Show trades by window type
        print("\n" + "-"*60)
        print("Trades by Window Type:")
        print("-"*60)
        window_stats = trades_df.groupby('WindowType').agg({
            'P&L': ['count', 'sum', 'mean']
        }).round(2)
        print(window_stats)
        
        return trades_df
        
    def display_indicator_stats(self, trades_df):
        """Display statistics about indicators for black box analysis"""
        if len(trades_df) == 0:
            return
        
        print("\n" + "="*60)
        print("INDICATOR STATISTICS (Black Box Analysis)")
        print("="*60)
        
        completed_trades = trades_df[trades_df['Status'].isin(['TP_HIT', 'SL_HIT'])].copy()
        
        if len(completed_trades) == 0:
            print("No completed trades to analyze")
            return
        
        # Separate winning and losing trades
        winning = completed_trades[completed_trades['P&L'] > 0]
        losing = completed_trades[completed_trades['P&L'] < 0]
        
        print(f"\nAnalyzing {len(completed_trades)} completed trades")
        print(f"  Winning: {len(winning)} | Losing: {len(losing)}")
        
        # Indicator columns to analyze
        indicator_cols = ['ATR', 'ATR_Pct', 'ATR_Ratio', 'Consolidation_Score', 
                         'Trend_Score', 'EMA_200_1H', 'Is_Consolidating', 'Is_Tight_Range']
        
        print("\n" + "-"*60)
        print("Indicator Values: Winning vs Losing Trades")
        print("-"*60)
        
        for col in indicator_cols:
            if col in completed_trades.columns:
                win_mean = winning[col].mean() if len(winning) > 0 else None
                loss_mean = losing[col].mean() if len(losing) > 0 else None
                
                if pd.notna(win_mean) and pd.notna(loss_mean):
                    diff = win_mean - loss_mean
                    diff_pct = (diff / abs(loss_mean) * 100) if loss_mean != 0 else 0
                    print(f"\n{col}:")
                    print(f"  Winning trades avg: {win_mean:.4f}")
                    print(f"  Losing trades avg:  {loss_mean:.4f}")
                    print(f"  Difference: {diff:.4f} ({diff_pct:+.2f}%)")
        
        # Consolidation analysis
        print("\n" + "-"*60)
        print("Consolidation Analysis")
        print("-"*60)
        if 'Is_Consolidating' in completed_trades.columns:
            consolidating_trades = completed_trades[completed_trades['Is_Consolidating'] == 1]
            non_consolidating = completed_trades[completed_trades['Is_Consolidating'] == 0]
            
            if len(consolidating_trades) > 0 and len(non_consolidating) > 0:
                consolidating_win_rate = (consolidating_trades['P&L'] > 0).sum() / len(consolidating_trades) * 100
                non_consolidating_win_rate = (non_consolidating['P&L'] > 0).sum() / len(non_consolidating) * 100
                
                print(f"\nTrades during consolidation: {len(consolidating_trades)}")
                print(f"  Win rate: {consolidating_win_rate:.2f}%")
                print(f"  Avg P&L: ${consolidating_trades['P&L'].mean():.2f}")
                
                print(f"\nTrades during non-consolidation: {len(non_consolidating)}")
                print(f"  Win rate: {non_consolidating_win_rate:.2f}%")
                print(f"  Avg P&L: ${non_consolidating['P&L'].mean():.2f}")
        
        # Trend analysis
        print("\n" + "-"*60)
        print("Trend Analysis (Price vs EMA_200_1H)")
        print("-"*60)
        if 'Price_Above_EMA200_1H' in completed_trades.columns:
            above_200 = completed_trades[completed_trades['Price_Above_EMA200_1H'] == 1]
            below_200 = completed_trades[completed_trades['Price_Above_EMA200_1H'] == 0]
            
            if len(above_200) > 0 and len(below_200) > 0:
                above_win_rate = (above_200['P&L'] > 0).sum() / len(above_200) * 100
                below_win_rate = (below_200['P&L'] > 0).sum() / len(below_200) * 100
                
                print(f"\nTrades above EMA_200_1H: {len(above_200)}")
                print(f"  Win rate: {above_win_rate:.2f}%")
                print(f"  Avg P&L: ${above_200['P&L'].mean():.2f}")
                
                print(f"\nTrades below EMA_200_1H: {len(below_200)}")
                print(f"  Win rate: {below_win_rate:.2f}%")
                print(f"  Avg P&L: ${below_200['P&L'].mean():.2f}")

        if 'PriceAboveEMA200_5M' in completed_trades.columns:
            print("\n" + "-"*60)
            print("5M EMA (200) Context")
            print("-"*60)
            above = completed_trades[completed_trades['PriceAboveEMA200_5M'] == 1]
            below = completed_trades[completed_trades['PriceAboveEMA200_5M'] == 0]
            if len(above) > 0:
                print(f"\nTrades with price above EMA200 (5M): {len(above)}")
                print(f"  Win rate: {(above['P&L'] > 0).mean() * 100:.2f}%")
                print(f"  Avg P&L: ${above['P&L'].mean():.2f}")
            if len(below) > 0:
                print(f"\nTrades with price below EMA200 (5M): {len(below)}")
                print(f"  Win rate: {(below['P&L'] > 0).mean() * 100:.2f}%")
                print(f"  Avg P&L: ${below['P&L'].mean():.2f}")

        if 'BreakoutDistance' in completed_trades.columns:
            print("\n" + "-"*60)
            print("Breakout Quality Metrics")
            print("-"*60)
            for metric in ['BreakoutDistance', 'BreakoutBodyPct', 'BreakoutAtrMultiple', 'RangeWidth', 'RangeAtrRatio']:
                if metric in completed_trades.columns:
                    win_avg = completed_trades.loc[completed_trades['P&L'] > 0, metric].mean()
                    loss_avg = completed_trades.loc[completed_trades['P&L'] < 0, metric].mean()
                    print(f"\n{metric}:")
                    print(f"  Winning trades avg: {win_avg:.4f}")
                    print(f"  Losing trades avg:  {loss_avg:.4f}")
 
        return completed_trades
    
    def save_results(self, trades_df, output_path):
        """Save results to CSV"""
        trades_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    config = StrategyConfig(
        reward_to_risk=2.0,
        pullback_timeout=12,
        use_ema_filter=True,
        allow_breakout=False,
        allow_pullback=True,
        allow_reversal=True,
        max_trades_per_window=1,
    )

    strategy = TradingStrategy('/home/nyale/Desktop/ML TESTING/XAUUSD5 new data.csv', config=config)
    
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
    
    # Display indicator statistics
    strategy.display_indicator_stats(trades_df)
    
    # Save results
    strategy.save_results(trades_df, '/home/nyale/Desktop/ML TESTING/window_breakout_results.csv')

    # Save enhanced data with indicators
    strategy.df.to_csv('/home/nyale/Desktop/ML TESTING/XAUUSD5_new_with_indicators.csv')
    print("\nEnhanced dataset with indicators saved to: /home/nyale/Desktop/ML TESTING/XAUUSD5_new_with_indicators.csv")

