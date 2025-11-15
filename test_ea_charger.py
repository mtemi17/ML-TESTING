#!/usr/bin/env python3
"""
Test script for EA_CHARGER.mq5
Simulates the EA's logic to verify it works correctly before MT5 testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EA_CHARGER TEST - Simulating EA Logic")
print("="*80)

# ============================================================================
# CONFIGURATION (matches EA_CHARGER.mq5 inputs)
# ============================================================================
class EAConfig:
    # Session settings
    session_OR_Duration = "00:15"  # 15 minutes
    session_TradeDuration = "03:00"  # 3 hours
    asian_SessionStart = "03:00"
    london_SessionStart = "10:00"
    ny_SessionStart = "16:30"
    
    # Model filters (100% win rate conditions)
    use_ModelFilters = True
    max_BreakoutAtrMultiple = 0.55
    max_AtrRatio = 1.17
    min_TrendScore = 0.67
    max_ConsolidationScore = 0.0
    min_RangeAtrRatio = 0.92
    min_EntryOffsetRatio = 0.0
    max_EntryOffsetRatio = 1.0
    
    # EMA filters
    filter_Enable = True
    filter_EmaPeriod = 200
    
    # Risk
    RiskRewardRatio = 2.0
    RiskPercentPerTrade = 1.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def parse_time(time_str):
    """Parse HH:MM string to time object"""
    parts = time_str.split(':')
    return time(int(parts[0]), int(parts[1]))

def parse_duration(duration_str):
    """Parse HH:MM duration to seconds"""
    parts = duration_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60

def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(window=period).mean()

def calculate_ema(df, period):
    """Calculate EMA"""
    return df['Close'].ewm(span=period, adjust=False).mean()

# ============================================================================
# SESSION STATE CLASS
# ============================================================================
class SessionState:
    def __init__(self, name):
        self.name = name
        self.isRangeFound = False
        self.isTradeTaken = False
        self.orHigh = 0.0
        self.orLow = 0.0
        self.orStartTime = None
        self.orEndTime = None

# ============================================================================
# EA_CHARGER SIMULATOR
# ============================================================================
class EAChargerSimulator:
    def __init__(self, data_file, config):
        self.config = config
        self.df = None
        self.load_data(data_file)
        self.add_indicators()
        
        # Session states
        self.asianState = SessionState("ASIAN")
        self.londonState = SessionState("LONDON")
        self.nyState = SessionState("NY")
        
        # Trades
        self.trades = []
        
    def load_data(self, data_file):
        """Load CSV data file"""
        print(f"\n1. Loading data from {data_file}...")
        try:
            self.df = pd.read_csv(data_file, header=None, 
                                 names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            self.df['DateTime'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'])
            self.df = self.df.set_index('DateTime')
            self.df = self.df.sort_index()
            print(f"   ✓ Loaded {len(self.df)} candles")
            print(f"   Date range: {self.df.index[0]} to {self.df.index[-1]}")
        except Exception as e:
            print(f"   ✗ Error loading data: {e}")
            raise
    
    def add_indicators(self):
        """Add all required indicators"""
        print("\n2. Adding indicators...")
        
        # ATR
        self.df['ATR'] = calculate_atr(self.df, 14)
        self.df['ATR_Avg'] = self.df['ATR'].rolling(window=19).mean()
        
        # EMAs
        self.df['EMA_200_1H'] = None  # Would need 1H data
        self.df['EMA_200_M5'] = calculate_ema(self.df, 200)
        self.df['EMA_9'] = calculate_ema(self.df, 9)
        self.df['EMA_21'] = calculate_ema(self.df, 21)
        self.df['EMA_50'] = calculate_ema(self.df, 50)
        
        print("   ✓ Indicators added")
    
    def find_opening_range(self, session_start_time, state, current_time):
        """Find opening range for a session (simulates FindOpeningRange)"""
        # Parse session start time
        session_hour = session_start_time.hour
        session_minute = session_start_time.minute
        
        # Get current date
        current_date = current_time.date()
        session_datetime = pd.Timestamp.combine(current_date, session_start_time)
        
        # Calculate range end (15 minutes later)
        or_duration = parse_duration(self.config.session_OR_Duration)
        range_end = session_datetime + pd.Timedelta(seconds=or_duration)
        
        # Check if we're past the range formation time
        if current_time < range_end:
            return False
        
        # Find bars in the range
        range_bars = self.df[(self.df.index >= session_datetime) & (self.df.index < range_end)]
        
        if len(range_bars) < 2:
            return False
        
        # Calculate range high and low
        state.orHigh = range_bars['High'].max()
        state.orLow = range_bars['Low'].min()
        state.orStartTime = session_datetime
        state.orEndTime = range_end
        
        # Validate range
        if state.orHigh > state.orLow:
            state.isRangeFound = True
            return True
        
        return False
    
    def calculate_consolidation_score(self, idx):
        """Calculate consolidation score (simulates CalculateConsolidationScore)"""
        if idx < 20:
            return 1.0  # Default to high consolidation if not enough data
        
        # Get last 20 bars
        window = self.df.iloc[idx-19:idx+1]
        
        current_atr = window['ATR'].iloc[-1]
        avg_atr = window['ATR'].iloc[:-1].mean()
        
        is_consolidating = (current_atr < avg_atr * 0.7) if avg_atr > 0 else False
        
        current_range = window['High'].iloc[-1] - window['Low'].iloc[-1]
        avg_range = (window['High'].iloc[:-1] - window['Low'].iloc[:-1]).mean()
        
        is_tight_range = (current_range < avg_range * 0.8) if avg_range > 0 else False
        
        score = ((1.0 if is_consolidating else 0.0) + (1.0 if is_tight_range else 0.0)) / 2.0
        return score
    
    def calculate_trend_score(self, idx, is_long, price):
        """Calculate trend score (simulates CalculateTrendScore)"""
        if idx < 200:
            return 0.5  # Neutral if not enough data
        
        score = 0.0
        components = 0
        
        # EMA 200 M5 filter
        if not pd.isna(self.df['EMA_200_M5'].iloc[idx]):
            ema200 = self.df['EMA_200_M5'].iloc[idx]
            price_above_ema = (price > ema200)
            if (is_long and price_above_ema) or (not is_long and not price_above_ema):
                score += 1.0
            components += 1
        
        # EMA alignment (9, 21, 50)
        if (idx >= 50 and 
            not pd.isna(self.df['EMA_9'].iloc[idx]) and
            not pd.isna(self.df['EMA_21'].iloc[idx]) and
            not pd.isna(self.df['EMA_50'].iloc[idx])):
            
            ema9 = self.df['EMA_9'].iloc[idx]
            ema21 = self.df['EMA_21'].iloc[idx]
            ema50 = self.df['EMA_50'].iloc[idx]
            
            aligned = False
            if is_long:
                aligned = (ema9 > ema21 and ema21 > ema50)
            else:
                aligned = (ema9 < ema21 and ema21 < ema50)
            
            if aligned:
                score += 1.0
            components += 1
        
        if components > 0:
            return score / components
        else:
            return 0.5
    
    def passes_trend_filter(self, idx, is_long, price):
        """Check EMA trend filter (simulates PassesTrendFilter)"""
        if not self.config.filter_Enable:
            return True
        
        allows_trade = True
        
        # Check M5 EMA 200
        if not pd.isna(self.df['EMA_200_M5'].iloc[idx]):
            ema200 = self.df['EMA_200_M5'].iloc[idx]
            if is_long:
                allows_trade = (price > ema200)
            else:
                allows_trade = (price < ema200)
        
        return allows_trade
    
    def passes_model_filters(self, idx, is_long, entry_price, state):
        """Check model-based filters (simulates PassesModelFilters)"""
        if not self.config.use_ModelFilters:
            return True
        
        if idx < 20:
            return False
        
        row = self.df.iloc[idx]
        
        # 1. Breakout ATR Multiple
        if is_long:
            breakout_distance = entry_price - state.orHigh
        else:
            breakout_distance = state.orLow - entry_price
        
        current_atr = row['ATR']
        if current_atr <= 0:
            return False
        
        breakout_atr_multiple = breakout_distance / current_atr
        if breakout_atr_multiple > self.config.max_BreakoutAtrMultiple:
            return False
        
        # 2. ATR Ratio
        avg_atr = self.df['ATR'].iloc[idx-19:idx].mean()
        if avg_atr <= 0:
            return False
        
        atr_ratio = current_atr / avg_atr
        if atr_ratio > self.config.max_AtrRatio:
            return False
        
        # 3. Range ATR Ratio
        range_size = state.orHigh - state.orLow
        range_atr_ratio = range_size / current_atr
        if range_atr_ratio < self.config.min_RangeAtrRatio:
            return False
        
        # 4. Entry Offset Ratio
        if is_long:
            entry_offset = entry_price - state.orHigh
        else:
            entry_offset = state.orLow - entry_price
        
        if range_size <= 0:
            return False
        
        entry_offset_ratio = entry_offset / range_size
        if (entry_offset_ratio < self.config.min_EntryOffsetRatio or 
            entry_offset_ratio > self.config.max_EntryOffsetRatio):
            return False
        
        # 5. Consolidation Score
        consolidation_score = self.calculate_consolidation_score(idx)
        if consolidation_score > self.config.max_ConsolidationScore:
            return False
        
        # 6. Trend Score
        trend_score = self.calculate_trend_score(idx, is_long, entry_price)
        if trend_score < self.config.min_TrendScore:
            return False
        
        return True
    
    def check_for_breakout(self, idx, state, session_type):
        """Check for breakout (simulates CheckForBreakout)"""
        if state.isTradeTaken or not state.isRangeFound:
            return
        
        row = self.df.iloc[idx]
        current_time = row.name
        
        # Check if within trading window
        trade_duration = parse_duration(self.config.session_TradeDuration)
        if current_time > state.orStartTime + pd.Timedelta(seconds=trade_duration):
            return
        
        open_price = row['Open']
        close_price = row['Close']
        
        # Candle confirmation
        is_bullish = close_price > open_price
        is_bearish = close_price < open_price
        
        # BULLISH BREAKOUT
        if close_price > state.orHigh and is_bullish:
            if self.passes_model_filters(idx, True, close_price, state):
                if self.passes_trend_filter(idx, True, close_price):
                    self.execute_trade(idx, 'LONG', close_price, state, session_type)
        
        # BEARISH BREAKOUT
        elif close_price < state.orLow and is_bearish:
            if self.passes_model_filters(idx, False, close_price, state):
                if self.passes_trend_filter(idx, False, close_price):
                    self.execute_trade(idx, 'SHORT', close_price, state, session_type)
    
    def execute_trade(self, idx, direction, entry_price, state, session_type):
        """Execute trade (simulates ExecuteTrade)"""
        if state.isTradeTaken:
            return
        
        # Calculate SL and TP
        if direction == 'LONG':
            stop_loss = state.orLow
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * self.config.RiskRewardRatio)
        else:
            stop_loss = state.orHigh
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * self.config.RiskRewardRatio)
        
        if risk <= 0:
            return
        
        # Check if TP or SL hit
        row = self.df.iloc[idx]
        current_time = row.name
        
        # Simulate trade outcome (simplified - check next 100 bars)
        outcome = None
        exit_time = None
        exit_price = None
        
        for i in range(idx + 1, min(idx + 100, len(self.df))):
            bar = self.df.iloc[i]
            
            if direction == 'LONG':
                if bar['High'] >= take_profit:
                    outcome = 'TP_HIT'
                    exit_price = take_profit
                    exit_time = bar.name
                    break
                elif bar['Low'] <= stop_loss:
                    outcome = 'SL_HIT'
                    exit_price = stop_loss
                    exit_time = bar.name
                    break
            else:
                if bar['Low'] <= take_profit:
                    outcome = 'TP_HIT'
                    exit_price = take_profit
                    exit_time = bar.name
                    break
                elif bar['High'] >= stop_loss:
                    outcome = 'SL_HIT'
                    exit_price = stop_loss
                    exit_time = bar.name
                    break
        
        if outcome:
            pnl = 0
            if direction == 'LONG':
                pnl = (exit_price - entry_price) / risk
            else:
                pnl = (entry_price - exit_price) / risk
            
            trade = {
                'EntryTime': current_time,
                'ExitTime': exit_time,
                'Direction': direction,
                'EntryPrice': entry_price,
                'StopLoss': stop_loss,
                'TakeProfit': take_profit,
                'ExitPrice': exit_price,
                'Status': outcome,
                'R_Multiple': pnl,
                'Session': state.name,
                'Risk': risk
            }
            
            self.trades.append(trade)
            state.isTradeTaken = True
            
            print(f"   ✅ {state.name} {direction} @ {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | {outcome}")
    
    def run_backtest(self):
        """Run the backtest simulation"""
        print("\n3. Running backtest simulation...")
        print("   Processing candles...")
        
        # Get unique dates
        unique_dates = pd.Series(self.df.index.date).unique()
        print(f"   Processing {len(unique_dates)} days...")
        
        ranges_found = 0
        breakouts_checked = 0
        filters_blocked = 0
        
        for date in unique_dates:
            date_df = self.df[self.df.index.date == date]
            
            # Reset session states for new day
            if date != unique_dates[0]:  # Not first day
                self.asianState = SessionState("ASIAN")
                self.londonState = SessionState("LONDON")
                self.nyState = SessionState("NY")
            
            # Process each bar
            for idx, (timestamp, row) in enumerate(date_df.iterrows()):
                current_time = timestamp
                time_obj = current_time.time()
                
                # Check Asian session
                asian_start = parse_time(self.config.asian_SessionStart)
                if (time_obj.hour == asian_start.hour and 
                    time_obj.minute >= asian_start.minute and
                    time_obj.minute < asian_start.minute + 15):
                    if not self.asianState.isRangeFound:
                        self.find_opening_range(asian_start, self.asianState, current_time)
                        if self.asianState.isRangeFound:
                            ranges_found += 1
                
                if self.asianState.isRangeFound and not self.asianState.isTradeTaken:
                    # Get global index
                    global_idx = self.df.index.get_loc(timestamp)
                    self.check_for_breakout(global_idx, self.asianState, "ASIAN")
                    breakouts_checked += 1
                
                # Check London session
                london_start = parse_time(self.config.london_SessionStart)
                if (time_obj.hour == london_start.hour and 
                    time_obj.minute >= london_start.minute and
                    time_obj.minute < london_start.minute + 15):
                    if not self.londonState.isRangeFound:
                        self.find_opening_range(london_start, self.londonState, current_time)
                        if self.londonState.isRangeFound:
                            ranges_found += 1
                
                if self.londonState.isRangeFound and not self.londonState.isTradeTaken:
                    # Get global index
                    global_idx = self.df.index.get_loc(timestamp)
                    self.check_for_breakout(global_idx, self.londonState, "LONDON")
                    breakouts_checked += 1
                
                # Check NY session
                ny_start = parse_time(self.config.ny_SessionStart)
                if (time_obj.hour == ny_start.hour and 
                    time_obj.minute >= ny_start.minute and
                    time_obj.minute < ny_start.minute + 15):
                    if not self.nyState.isRangeFound:
                        self.find_opening_range(ny_start, self.nyState, current_time)
                        if self.nyState.isRangeFound:
                            ranges_found += 1
                
                if self.nyState.isRangeFound and not self.nyState.isTradeTaken:
                    # Get global index
                    global_idx = self.df.index.get_loc(timestamp)
                    self.check_for_breakout(global_idx, self.nyState, "NY")
                    breakouts_checked += 1
        
        print(f"\n   ✓ Backtest complete!")
        print(f"   Ranges found: {ranges_found}")
        print(f"   Breakouts checked: {breakouts_checked}")
        print(f"   Trades executed: {len(self.trades)}")
    
    def analyze_results(self):
        """Analyze backtest results"""
        if len(self.trades) == 0:
            print("\n❌ No trades executed!")
            return
        
        df_trades = pd.DataFrame(self.trades)
        completed = df_trades[df_trades['Status'].isin(['TP_HIT', 'SL_HIT'])]
        
        if len(completed) == 0:
            print("\n⚠️  No completed trades!")
            return
        
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        
        wins = completed[completed['Status'] == 'TP_HIT']
        losses = completed[completed['Status'] == 'SL_HIT']
        
        total_trades = len(completed)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_r = completed['R_Multiple'].mean()
        total_r = completed['R_Multiple'].sum()
        
        print(f"\nTotal Trades: {total_trades}")
        print(f"Wins: {len(wins)} ({len(wins)/total_trades*100:.1f}%)")
        print(f"Losses: {len(losses)} ({len(losses)/total_trades*100:.1f}%)")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average R: {avg_r:.2f}")
        print(f"Total R: {total_r:.2f}")
        
        if len(wins) > 0:
            print(f"Average Win: {wins['R_Multiple'].mean():.2f}R")
        if len(losses) > 0:
            print(f"Average Loss: {losses['R_Multiple'].mean():.2f}R")
        
        # By session
        print("\nBy Session:")
        for session in ['ASIAN', 'LONDON', 'NY']:
            session_trades = completed[completed['Session'] == session]
            if len(session_trades) > 0:
                session_wr = (session_trades['Status'] == 'TP_HIT').sum() / len(session_trades) * 100
                print(f"  {session}: {len(session_trades)} trades, {session_wr:.1f}% win rate")
        
        return df_trades

# ============================================================================
# MAIN TEST
# ============================================================================
if __name__ == "__main__":
    # Try to find available data file
    data_files = ['XAUUSD5.csv', 'GBPJPY5.csv', 'EURJPY5.csv', 'USDJPY5.csv']
    data_file = None
    
    for file in data_files:
        try:
            pd.read_csv(file, nrows=1)
            data_file = file
            print(f"Using data file: {data_file}")
            break
        except:
            continue
    
    if not data_file:
        print("❌ No data file found! Please ensure XAUUSD5.csv or similar exists.")
        exit(1)
    
    # Test 1: With filters DISABLED (to test basic logic)
    print("\n" + "="*80)
    print("TEST 1: BASIC LOGIC (Filters DISABLED)")
    print("="*80)
    config1 = EAConfig()
    config1.use_ModelFilters = False
    config1.filter_Enable = False
    
    simulator1 = EAChargerSimulator(data_file, config1)
    simulator1.run_backtest()
    results1 = simulator1.analyze_results()
    
    # Test 2: With filters ENABLED
    print("\n" + "="*80)
    print("TEST 2: WITH MODEL FILTERS (Filters ENABLED)")
    print("="*80)
    config2 = EAConfig()
    config2.use_ModelFilters = True
    config2.filter_Enable = True
    
    simulator2 = EAChargerSimulator(data_file, config2)
    simulator2.run_backtest()
    results2 = simulator2.analyze_results()
    
    # Save results
    if results1 is not None and len(results1) > 0:
        output_file1 = f'ea_charger_test_NO_FILTERS_{data_file.replace(".csv", "")}.csv'
        results1.to_csv(output_file1, index=False)
        print(f"\n✅ Test 1 results saved to: {output_file1}")
    
    if results2 is not None and len(results2) > 0:
        output_file2 = f'ea_charger_test_WITH_FILTERS_{data_file.replace(".csv", "")}.csv'
        results2.to_csv(output_file2, index=False)
        print(f"✅ Test 2 results saved to: {output_file2}")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)

