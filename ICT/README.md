# ICT Trading System - Standalone

## 🎯 Overview

This is a **COMPLETE NEW ICT TRADING SYSTEM** based on Smart Money Concepts, **separate** from the existing ORB/breakout strategy.

This system uses:
- **Top-Down Bias Detection** (Weekly → 4H → 5M/15M)
- **Order Blocks** (institutional order zones)
- **Fair Value Gaps (FVG)** (price imbalances)
- **Liquidity Zones** (where stops cluster)
- **Premium/Discount Zones** (range-based targets)
- **Market Structure** (BOS/CHoCH)

## 📊 Top-Down Approach

The system analyzes market bias from higher to lower timeframes:

1. **Weekly Timeframe** → Overall market bias
2. **4H Timeframe** → Intermediate trend confirmation  
3. **5M/15M Timeframe** → Entry signals (only trade with bias)

### Philosophy

- **Higher timeframes** determine the overall direction (bias)
- **Lower timeframes** provide entry opportunities
- **Only trade in the direction of the higher timeframe bias**

## 📁 Files

### `ict_trading_system.py` ⭐ **MAIN SYSTEM**
Complete standalone ICT trading system. This is the NEW system that:
- Detects bias on Weekly/4H timeframes
- Finds entry opportunities using Order Blocks and FVG
- Only trades in the direction of higher timeframe bias
- Uses ICT concepts for entries and targets
- **This is separate from your existing strategy!**

### `test_ict_topdown_bias.py`
Bias detection utilities. Implements:

- **Multi-timeframe resampling** (Weekly, 4H, 15M from 5M data)
- **Weekly bias detection** using:
  - Price vs EMA 21
  - Market Structure (HH/HL vs LH/LL)
- **4H bias detection** using:
  - EMA alignment (9, 21, 50)
  - Market structure analysis
- **Combined bias logic**:
  - Both timeframes must align for strong bias
  - Weekly takes precedence if 4H is neutral
  - Requires minimum strength (0.5) to trade

### `ict_bias_backtest.py`
Backtest template that integrates ICT bias filtering with trading strategies.

### `README.md`
This file.

## 🚀 Usage

### Main ICT Trading System

This is the **standalone ICT system** - run this to trade using ICT concepts:

```bash
# Run the complete ICT trading system
python ict_trading_system.py path/to/XAUUSD_5M.csv
```

Or in Python:

```python
from ict_trading_system import ICTTradingSystem, ICTConfig

# Configure the system
config = ICTConfig(
    use_order_blocks=True,      # Use Order Block entries
    use_fvg=True,               # Use Fair Value Gap entries
    min_bias_strength=0.5,      # Minimum bias strength to trade
    require_alignment=True,     # Weekly and 4H must align
    risk_reward_ratio=2.0,      # 2R target
    max_trades_per_day=3        # Max 3 trades per day
)

# Initialize and run
system = ICTTradingSystem('XAUUSD_5M.csv', config)
trades = system.backtest()

# Results
print(f"Total Trades: {len(trades)}")
print(f"Win Rate: {(trades['Status'] == 'WIN').sum() / len(trades) * 100:.1f}%")
print(f"Total P&L: ${trades['P&L'].sum():.2f}")
```

### Bias Detection Only

If you just want to check bias:

```python
from test_ict_topdown_bias import ICTTopDownBias
import pandas as pd

# Load your 5-minute data
df = pd.read_csv('XAUUSD_5M.csv', ...)

# Initialize system
bias_system = ICTTopDownBias(df)
bias_system.resample_timeframes()

# Get current bias
bias_info = bias_system.get_trade_bias(df.index[-1])
print(f"Bias: {bias_info['bias']}")
print(f"Strength: {bias_info['strength']}")
print(f"Aligned: {bias_info['aligned']}")
```

### Command Line

```bash
# Run complete ICT system
python ict_trading_system.py path/to/XAUUSD_5M.csv

# Test bias detection only
python test_ict_topdown_bias.py path/to/XAUUSD_5M.csv
```

## 📈 Bias Detection Logic

### Weekly Bias

1. **Price vs EMA 21** (40% weight)
   - Price > EMA 21 → Bullish
   - Price < EMA 21 → Bearish

2. **Market Structure** (60% weight)
   - Higher Highs + Higher Lows → Bullish
   - Lower Highs + Lower Lows → Bearish

### 4H Bias

1. **EMA Alignment** (50% weight)
   - Price > EMA 9 > EMA 21 > EMA 50 → Bullish
   - Price < EMA 9 < EMA 21 < EMA 50 → Bearish

2. **Market Structure** (50% weight)
   - Recent swing highs/lows analysis

### Combined Bias

- **Strong Bias**: Both Weekly and 4H agree (aligned = True)
- **Weak Bias**: Only one timeframe shows bias
- **Neutral**: No clear bias

## 🎯 Trading Rules

1. **Only take LONG trades** when bias is **BULLISH**
2. **Only take SHORT trades** when bias is **BEARISH**
3. **Require minimum bias strength** of 0.5
4. **Require alignment** between Weekly and 4H (both must agree)

## 🎯 How the ICT System Works

### Entry Signals

1. **Order Blocks**:
   - Bullish OB: Last bearish/bullish candle before strong move up
   - Bearish OB: Last bullish/bearish candle before strong move down
   - Entry at the close of the Order Block candle

2. **Fair Value Gaps (FVG)**:
   - Bullish FVG: Gap between previous high and current low
   - Bearish FVG: Gap between previous low and current high
   - Entry at FVG midpoint

### Bias Filtering

- **Only takes LONG trades** when Weekly + 4H bias is **BULLISH**
- **Only takes SHORT trades** when Weekly + 4H bias is **BEARISH**
- Requires both timeframes to align (require_alignment=True)
- Requires minimum bias strength (default 0.5)

### Targets

- Order Blocks: Target at the high/low of the move after OB
- FVG: Target based on risk-reward ratio (default 2R)
- Liquidity zones can be used for additional targets

## 📊 Expected Benefits

- **Higher win rate** by trading with the trend
- **Better risk management** by avoiding counter-trend trades
- **ICT-aligned** with Smart Money Concepts
- **Multi-timeframe confirmation** reduces false signals

## ⚠️ Notes

- This is a **bias detection system**, not a complete trading strategy
- Integrate with your existing entry/exit logic
- Test thoroughly before live trading
- Bias can change - monitor regularly

## ⚠️ Important Notes

- **This is a NEW standalone system** - not related to your existing ORB strategy
- Uses pure ICT concepts: Order Blocks, FVG, Liquidity Zones
- Top-down approach: Weekly → 4H → 5M/15M
- Only trades when higher timeframes align
- Test thoroughly before live trading

## 🔗 Related Files

- `../EAAI_Full.mq5` - Your existing EA (separate system)
- `../strategy_backtest.py` - Your existing strategy (separate system)

## 📝 TODO

- [x] Add order block detection ✅
- [x] Add Fair Value Gap (FVG) detection ✅
- [x] Add liquidity zone identification ✅
- [ ] Create MQL5 version for MT5
- [ ] Add visualization tools for bias display
- [ ] Add premium/discount zone targeting
- [ ] Add market structure break detection (BOS/CHoCH)

