# Confirmation Delay Feature

## Overview

The EA now has the ability to **wait for post-entry conditions to be met BEFORE entering** a trade, instead of entering immediately and then checking if we should exit.

## How It Works

### Old Behavior (Immediate Entry)
1. Detect breakout
2. Enter trade immediately
3. Check post-entry conditions
4. Exit if conditions not met

### New Behavior (Confirmation Delay)
1. Detect breakout
2. **Wait and monitor** for post-entry conditions
3. **Only enter** if conditions are met
4. Cancel if conditions not met within timeout

## New Input Parameters

```
InpWaitForConfirmation = true     // Enable confirmation delay
InpConfirmationTimeoutBars = 5    // Max bars to wait for confirmation
```

## What Conditions Are Checked Before Entry

While waiting for confirmation, the EA checks:

1. **Momentum**: Price must gain at least `InpBreakoutMomentumMinGain` (0.2R) within `InpBreakoutMomentumBar` (5 bars)
2. **First Bar Gain** (if StrategyMode=1): First bar after detection must gain at least `InpFirstBarMinGain` (-0.30R)
3. **Max MAE**: Maximum adverse excursion must not exceed `InpBreakoutMaxMaeRatio` (1.0R)
4. **Retest Depth** (if StrategyMode=1): Retest into range must not exceed `InpMaxRetestDepthR` (3.0R) within `InpMaxRetestBars` (20 bars)

## Entry Conditions

The trade is entered when:
- ✅ Momentum is confirmed (gain >= 0.2R)
- ✅ First bar gain is acceptable (if StrategyMode=1)
- ✅ All other conditions remain valid
- ✅ Timeout not reached

## Benefits

1. **Higher Quality Entries**: Only enter trades that show immediate momentum
2. **Fewer Filter Exits**: Avoid entering trades that would be closed early
3. **Better Win Rate**: Enter only when conditions are favorable
4. **Reduced Risk**: Cancel bad trades before entering

## Example Flow

```
Bar 0: Breakout detected @ 3865.00
       → Pending breakout created
       → Waiting for confirmation...

Bar 1: Price moves to 3866.50 (+1.50)
       → Gain: 0.3R (above 0.2R threshold)
       → Momentum confirmed! ✅
       → First bar gain: 0.3R (above -0.30R) ✅
       → All conditions met
       → ENTER TRADE @ 3866.50
```

## Configuration

### To Enable Confirmation Delay:
```
InpWaitForConfirmation = true
InpConfirmationTimeoutBars = 5
```

### To Disable (Old Behavior):
```
InpWaitForConfirmation = false
```

## Expected Impact

With confirmation delay enabled:
- **Fewer trades**: Only enter when conditions are met
- **Higher win rate**: Better entry quality
- **Less filter exits**: Avoid bad trades before entry
- **Better P&L**: Focus on high-probability setups

## Testing

Test the EA with:
- `InpWaitForConfirmation = true` (new behavior)
- `InpWaitForConfirmation = false` (old behavior)

Compare results to see which performs better for your market conditions.

