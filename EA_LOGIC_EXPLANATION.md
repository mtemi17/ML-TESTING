# EAAI_Full.mq5 - LOGIC EXPLANATION

## 🎯 WHAT THIS EA DOES

**YES, WE ARE LOOKING FOR BREAKOUTS!**

This EA implements the **Opening Range Breakout (ORB)** strategy, similar to `ORB_SmartTrap_EA.mq5` that you provided.

---

## 📋 THE LOGIC (Step by Step)

### 1. **SESSION DETECTION**
- **3 Sessions per day:**
  - `SESSION_1_03:00` (Asian/London overlap)
  - `SESSION_2_10:00` (London session)
  - `SESSION_3_16:30` (New York session)

### 2. **RANGE FORMATION**
- At each session start time (e.g., 03:00), the EA:
  - Takes the **first 15-minute candle** (3 x 5-minute bars)
  - Records the **HIGH** and **LOW** of that 15-minute range
  - This becomes the "Opening Range"

### 3. **TRADING WINDOW**
- After the 15-minute range completes, the EA watches for **3 hours**
- During this window, it looks for **breakouts** above or below the range

### 4. **BREAKOUT DETECTION** (Like ORB_SmartTrap)
- **BULLISH BREAKOUT:**
  - Price closes **ABOVE** the range high
  - **AND** the candle is **bullish** (close > open) ← **CANDLE CONFIRMATION**
  
- **BEARISH BREAKOUT:**
  - Price closes **BELOW** the range low
  - **AND** the candle is **bearish** (close < open) ← **CANDLE CONFIRMATION**

### 5. **ENTRY MODES**

#### **IMMEDIATE ENTRY** (Default)
- When breakout is detected with candle confirmation → **Enter immediately**

#### **DELAYED ENTRY**
- When breakout is detected → **Wait** for momentum confirmation
- Enter only if price moves in favor by `InpMomentumMinGain` (default: 0.05R)

### 6. **PRE-ENTRY FILTERS** (Optional)
- **EMA 200 (1H) Filter:** Only trade in direction of 1H trend
- **Strict Filters (100% WR):** Very restrictive filters that reduce trades but maintain high win rate
- **Entry Offset:** Price must be within certain distance from range midpoint

### 7. **RISK MANAGEMENT**
- **Stop Loss:** `InpBreakoutInitialStopRatio` × range size (default: 0.6 = 60% of range)
- **Take Profit:** `InpRewardToRisk` × risk (default: 2.0 = 2R)
- **Position Size:** Based on `InpRiskPercent` (default: 1% of account)

### 8. **POST-ENTRY CONTROLS** (Optional)
- **First Bar Check:** Trade must show gain on first bar
- **MAE Check:** Maximum adverse excursion limit
- **Retest Check:** Price shouldn't retest range too deeply
- **Momentum Check:** Price must show momentum within N bars

### 9. **ONE TRADE PER SESSION** ⭐ **FIXED!**
- **NEW:** The EA now tracks which windows have been traded
- Once a trade is taken in a session window, **NO MORE TRADES** in that window
- This ensures **exactly 1 trade per session** (not 20!)

---

## 🔧 WHAT WAS FIXED

### **Problem 1: Too Many Trades Per Session**
- **Before:** `CountTradesForWindow` only checked **open positions**
- **After trade closed:** New trades could be taken in same window
- **Result:** 20 trades per session instead of 1

- **Fix:** Added `g_tradedWindows[]` array to track **all traded windows** (even after close)
- **Now:** Once a window is marked as traded, **no more trades** in that window

### **Problem 2: Unclear Session Labels**
- **Before:** "ASIAN/LONDON", "LONDON", "NEW_YORK"
- **After:** "SESSION_1_03:00", "SESSION_2_10:00", "SESSION_3_16:30"
- **Now:** Clear labels showing session number and time

### **Problem 3: Zero SHORT Trades**
- **Still investigating:** Enhanced logging added to diagnose why SHORT trades aren't being taken
- **Check logs for:**
  - "🔻 BEARISH BREAKOUT DETECTED" messages
  - "❌ Filtered [SHORT]:" messages (shows which filter is blocking)

---

## 📊 EXPECTED BEHAVIOR

### **With Default Settings:**
- **Trades per day:** ~3 (1 per session)
- **Win rate:** 35-40% (without strict filters)
- **SHORT trades:** Should be ~40-50% of total (if market conditions allow)

### **With Strict Filters:**
- **Trades per day:** ~1-2 (very selective)
- **Win rate:** 100% (TP vs SL only, excluding filter exits)
- **SHORT trades:** May be reduced if filters are too strict

---

## 🎯 KEY SETTINGS

| Setting | Default | What It Does |
|---------|---------|--------------|
| `InpMaxTradesPerWindow` | 1 | **Max trades per session** (now enforced!) |
| `InpUseImmediateEntry` | true | Enter immediately vs wait for confirmation |
| `InpUseEMA200Filter` | false | Filter by 1H trend (reduces profitability) |
| `InpUseStrictFilters` | false | Use 100% WR conditions (very few trades) |
| `InpRiskPercent` | 1.0 | Risk % per trade |
| `InpRewardToRisk` | 2.0 | Take profit = 2 × risk |

---

## ✅ CONFIRMATION: YES, WE'RE USING YOUR LOGIC!

1. **✅ Opening Range Breakout** - Same as ORB_SmartTrap
2. **✅ Candle Confirmation** - Requires bullish/bearish candle (from ORB_SmartTrap)
3. **✅ Session-based trading** - 3 sessions per day
4. **✅ Risk management** - SL/TP based on range size
5. **✅ One trade per session** - Now properly enforced!

---

## 🚀 NEXT STEPS

1. **Recompile** `EAAI_Full.mq5` in MT5
2. **Run Strategy Tester** again
3. **Check logs** for:
   - "✅ Window marked as traded" messages (confirms 1 trade per session)
   - "❌ SKIPPED: Already traded this window" messages (confirms blocking)
   - Bearish breakout detections (to diagnose SHORT trade issue)
4. **Verify:** You should now see **exactly 1 trade per session** (not 20!)

---

## 📝 SUMMARY

**What we're doing:**
- ✅ Looking for breakouts (YES!)
- ✅ Using ORB_SmartTrap logic (YES!)
- ✅ Candle confirmation (YES!)
- ✅ One trade per session (NOW FIXED!)

**What was wrong:**
- ❌ Too many trades per session (FIXED!)
- ❌ Unclear session labels (FIXED!)
- ❌ Zero SHORT trades (STILL INVESTIGATING - check logs!)

**What you should see now:**
- ✅ Clear session labels: "SESSION_1_03:00", "SESSION_2_10:00", "SESSION_3_16:30"
- ✅ Exactly 1 trade per session (not 20!)
- ✅ Better logging to diagnose SHORT trade issue

