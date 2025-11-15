# EAAI_Simple.mq5 Testing Guide

## Expected Results (October 2025)

Based on backtest with corrected data collection:

- **Completed Trades**: 9
- **Filtered Trades**: 33
- **Win Rate**: 100% (9 wins, 0 losses)
- **Total P&L**: $161.38
- **Avg P&L**: $17.93 per trade

## EA Configuration for Testing

### Input Parameters

```
Strategy Mode: 1 (Optimized)
Risk Percent: 1.0
Reward to Risk: 2.0
Max Trades Per Window: 1

Winner Profile Controls:
- Breakout Initial Stop Ratio: 0.6
- Breakout Max MAE Ratio: 0.6
- Breakout Momentum Bar: 3
- Breakout Momentum Min Gain: 0.3

Optimized Filters:
- Max Breakout ATR Multiple: 1.8
- Max ATR Ratio: 1.3
- Min Trend Score: 0.66
- Max Consolidation Score: 0.10
- Min Entry Offset Ratio: -0.25
- Max Entry Offset Ratio: 1.00
- First Bar Min Gain: -0.20
- Max Retest Depth R: 1.80
- Max Retest Bars: 12
```

### Strategy Tester Settings

- **Symbol**: XAUUSD
- **Timeframe**: M5
- **Date Range**: 2025-10-01 to 2025-10-31
- **Model**: Every tick (for accuracy)
- **Deposit**: Any (results are in $, not %)
- **Spread**: Use your broker's spread

## Expected Trade List

See `analysis/expected_ea_trades_october.csv` for complete list of expected trades with:
- Entry times
- Entry prices
- Stop loss levels
- Take profit levels
- Expected P&L

## Verification Checklist

After running EA in Strategy Tester:

1. **Trade Count**: Should see ~9 completed trades
2. **Win Rate**: Should be 100% (all TP hits)
3. **Total P&L**: Should be approximately $161.38
4. **Entry Times**: Should match expected times (within 1 bar)
5. **Entry Prices**: Should match expected prices (within spread)

## Troubleshooting

### If EA shows fewer trades:
- Check if data is complete in MT5
- Verify window times are correct (03:00, 10:00, 16:30 server time)
- Check if filters are too strict

### If EA shows more trades:
- Check if filters are working correctly
- Verify Strategy Mode is set to 1 (Optimized)

### If P&L differs:
- Check spread settings
- Verify execution model (Every tick vs other)
- Check if slippage is enabled

### If entry prices differ:
- Normal - EA uses market prices (BID/ASK)
- Backtest uses close prices
- Small differences (< 0.1%) are expected

## Key Differences to Expect

1. **Spread**: EA uses BID/ASK, backtest uses Close
2. **Execution**: EA has real execution delays
3. **Data Quality**: MT5 data might differ slightly from CSV
4. **Timing**: EA might enter 1 tick later than backtest

## Success Criteria

✅ **EA is working correctly if:**
- Trade count is within ±2 trades
- Win rate is similar (90-100%)
- Total P&L is within ±20% of expected
- Entry times match within 1-2 bars

❌ **EA needs fixing if:**
- Trade count differs by >5 trades
- Win rate is <80%
- Total P&L is negative when expected positive
- Entry times are completely different

## Next Steps After Testing

1. Compare EA results with `expected_ea_trades_october.csv`
2. Document any differences
3. If significant differences, check:
   - EA code logic
   - Indicator calculations
   - Filter implementations
   - Window detection

