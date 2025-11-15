//+------------------------------------------------------------------+
//|                                                    EAAI_Full.mq5 |
//|                                  Full Integrated Trading System  |
//|                                    All Features Self-Contained  |
//+------------------------------------------------------------------+
#property copyright "Full AI Trading System"
#property link      ""
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                 |
//+------------------------------------------------------------------+

// === SESSION MANAGEMENT ===
input group "=== SESSION SETTINGS ==="
input bool     InpEnableSession1        = true;     // Enable 03:00 Session
input bool     InpEnableSession2        = true;     // Enable 10:00 Session
input bool     InpEnableSession3        = true;     // Enable 16:30 Session
input string   InpSession1Time          = "03:00";  // Session 1 Time
input string   InpSession2Time          = "10:00";  // Session 2 Time
input string   InpSession3Time          = "16:30";  // Session 3 Time
input int      InpTradingWindowHours    = 3;        // Trading Window (hours after session)

// === RISK MANAGEMENT ===
input group "=== RISK MANAGEMENT ==="
input double   InpRiskPercent           = 1.0;      // Risk % per trade
input double   InpRewardToRisk          = 2.0;      // Reward-to-risk multiple
input int      InpMaxTradesPerWindow    = 1;        // Max trades per window
input int      InpMaxTotalPositions     = 3;       // Max total concurrent positions (GLOBAL LIMIT)
input int      InpMaxPositionsPerSymbol = 1;       // Max positions per symbol/market

// === ENTRY MODES ===
input group "=== ENTRY MODES ==="
input bool     InpUseImmediateEntry     = true;     // TRUE = Immediate Entry, FALSE = Delay Entry
input int      InpConfirmationTimeoutBars = 4;      // Delay: Bars to wait for confirmation
input double   InpMomentumMinGain        = 0.05;     // Delay: Min momentum gain (R)

// === EMA FILTERS ===
input group "=== EMA FILTERS ==="
input bool     InpUseEMA200Filter       = false;    // Use EMA 200 (1H) Filter (WARNING: Reduces profitability!)
input bool     InpUseEMAAlignment       = false;    // Require EMA Alignment (9>21>50)

// === BREAKOUT CONTROLS ===
input group "=== BREAKOUT CONTROLS ==="
input bool     InpUseBreakoutControls   = true;     // Enable Breakout Controls
input double   InpBreakoutInitialStopRatio = 0.6;   // SL = range risk * ratio
input double   InpBreakoutMaxMaeRatio   = 1.0;      // Max MAE as fraction of risk
input int      InpBreakoutMomentumBar   = 5;        // Bars to show momentum
input double   InpBreakoutMomentumMinGain = 0.2;    // Min gain (R) by momentum bar

// === PRE-ENTRY FILTERS (100% WIN RATE CONDITIONS) ===
input group "=== PRE-ENTRY FILTERS (100% WIN RATE) ==="
input bool     InpUseStrictFilters      = false;    // Use strict 100% win rate filters (WARNING: Very few trades!)
input double   InpMaxBreakoutAtrMultiple = 0.55;    // Max distance from range in ATR (only if filters enabled)
input double   InpMaxAtrRatio           = 1.17;      // Max ATR / ATR average (only if filters enabled)
input double   InpMinTrendScore          = 0.67;    // Minimum trend alignment score (only if filters enabled)
input double   InpMaxConsolidationScore  = 0.0;     // Maximum consolidation score (only if filters enabled)
input double   InpMinRangeAtrRatio      = 0.92;     // Min Range ATR Ratio (only if filters enabled)
input double   InpMinEntryOffsetRatio    = -0.25;   // Min normalized offset from range mid
input double   InpMaxEntryOffsetRatio    = 1.00;    // Max normalized offset from range mid

// === POST-ENTRY FILTERS ===
input group "=== POST-ENTRY FILTERS ==="
input double   InpFirstBarMinGain       = -0.30;    // First bar minimum gain (R)
input double   InpMaxRetestDepthR       = 3.00;     // Max retest depth into range (R)
input int      InpMaxRetestBars         = 20;       // Max bars to enforce retest rule


//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                 |
//+------------------------------------------------------------------+

#define MAX_SESSIONS 3

struct SessionConfig
{
   bool   enabled;
   string time;
   int    hour;
   int    minute;
};

SessionConfig g_sessions[MAX_SESSIONS];

int g_ema9Handle    = INVALID_HANDLE;
int g_ema21Handle   = INVALID_HANDLE;
int g_ema50Handle   = INVALID_HANDLE;
int g_ema200_5mHandle = INVALID_HANDLE;
int g_ema200_1hHandle = INVALID_HANDLE;
int g_atrHandle     = INVALID_HANDLE;

datetime g_lastWindowTrade[MAX_SESSIONS];
string g_tradedWindows[];  // Track which windows we've already traded

struct WindowRange
{
   datetime start;
   datetime end;
   double   high;
   double   low;
   string   id;
   int      sessionIndex;
};

struct BreakoutState
{
   bool               active;
   datetime           entry_time;
   double             entry_price;
   double             stop_price;
   double             target_price;
   double             risk;
   int                bars_in_trade;
   double             max_mae;
   bool               momentum_satisfied;
   ENUM_POSITION_TYPE direction;
   string             window_id;
   double             range_high;
   double             range_low;
   double             max_retest_depth;
   bool               first_bar_checked;
   bool               confirmed_before_entry;
   string             entry_type;  // "BREAKOUT", "PULLBACK", "REVERSAL"
};

struct PendingBreakout
{
   bool               active;
   datetime           detected_time;
   double             detected_price;
   double             stop_price;
   double             target_price;
   double             risk;
   int                bars_waiting;
   double             max_mae;
   bool               momentum_satisfied;
   ENUM_POSITION_TYPE direction;
   string             window_id;
   double             range_high;
   double             range_low;
   double             max_retest_depth;
   double             first_bar_gain;
   bool               first_bar_checked;
   string             entry_type;
};

BreakoutState g_breakout;
PendingBreakout g_pending;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize sessions
   g_sessions[0].enabled = InpEnableSession1;
   g_sessions[0].time = InpSession1Time;
   ParseTime(InpSession1Time, g_sessions[0].hour, g_sessions[0].minute);
   
   g_sessions[1].enabled = InpEnableSession2;
   g_sessions[1].time = InpSession2Time;
   ParseTime(InpSession2Time, g_sessions[1].hour, g_sessions[1].minute);
   
   g_sessions[2].enabled = InpEnableSession3;
   g_sessions[2].time = InpSession3Time;
   ParseTime(InpSession3Time, g_sessions[2].hour, g_sessions[2].minute);
   
   // Initialize indicators (5M timeframe - always needed)
   g_ema9Handle = iMA(_Symbol, PERIOD_M5, 9, 0, MODE_EMA, PRICE_CLOSE);
   g_ema21Handle = iMA(_Symbol, PERIOD_M5, 21, 0, MODE_EMA, PRICE_CLOSE);
   g_ema50Handle = iMA(_Symbol, PERIOD_M5, 50, 0, MODE_EMA, PRICE_CLOSE);
   g_ema200_5mHandle = iMA(_Symbol, PERIOD_M5, 200, 0, MODE_EMA, PRICE_CLOSE);
   g_atrHandle = iATR(_Symbol, PERIOD_M5, 14);
   
   // Initialize 1H EMA only if filter is enabled
   if(InpUseEMA200Filter)
   {
      g_ema200_1hHandle = iMA(_Symbol, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE);
      if(g_ema200_1hHandle == INVALID_HANDLE)
      {
         Print("ERROR: Failed to create EMA 200 (1H) indicator");
         return INIT_FAILED;
      }
   }
   
   // Check required indicators
   if(g_ema9Handle == INVALID_HANDLE || g_ema21Handle == INVALID_HANDLE || 
      g_ema50Handle == INVALID_HANDLE || g_ema200_5mHandle == INVALID_HANDLE ||
      g_atrHandle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create required indicators");
      return INIT_FAILED;
   }
   
   // Initialize state
   ZeroMemory(g_breakout);
   ZeroMemory(g_pending);
   ArrayResize(g_tradedWindows, 0);  // Clear traded windows array
   
   for(int i = 0; i < MAX_SESSIONS; i++)
      g_lastWindowTrade[i] = 0;
   
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("EAAI_FULL - INITIALIZATION COMPLETE");
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("STRATEGY: Opening Range Breakout (ORB) with Candle Confirmation");
   Print("LOGIC: Like ORB_SmartTrap - Wait for breakout + candle confirmation");
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("SESSIONS:");
   Print("  ", GetSessionName(0), " (", InpSession1Time, ") - ", InpEnableSession1 ? "ENABLED" : "DISABLED");
   Print("  ", GetSessionName(1), " (", InpSession2Time, ") - ", InpEnableSession2 ? "ENABLED" : "DISABLED");
   Print("  ", GetSessionName(2), " (", InpSession3Time, ") - ", InpEnableSession3 ? "ENABLED" : "DISABLED");
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("ENTRY MODE: ", InpUseImmediateEntry ? "IMMEDIATE" : "DELAYED (", InpConfirmationTimeoutBars, " bars, ", InpMomentumMinGain, "R)");
   Print("MAX TRADES PER SESSION: ", InpMaxTradesPerWindow, " (ONE TRADE PER SESSION!)");
   Print("RISK: ", InpRiskPercent, "% per trade | R:R = ", InpRewardToRisk);
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("FILTERS:");
   Print("  EMA 200 (1H) Filter: ", InpUseEMA200Filter ? "ON ⚠️ (BLOCKS SHORT TRADES!)" : "OFF ✅");
   Print("  Strict Filters (100% WR): ", InpUseStrictFilters ? "ON" : "OFF");
   Print("  Breakout Controls: ", InpUseBreakoutControls ? "ON" : "OFF");
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("POSITION LIMITS: Max Total=", InpMaxTotalPositions, " | Max Per Symbol=", InpMaxPositionsPerSymbol);
   Print("=", StringSubstr("=", 0, 60), "=");
   
   if(InpUseEMA200Filter)
   {
      Print("⚠️ WARNING: EMA 200 Filter is ENABLED!");
      Print("   This will BLOCK SHORT trades when price is above EMA 200 (1H)");
      Print("   If you want SHORT trades, set InpUseEMA200Filter = false");
      Print("=", StringSubstr("=", 0, 60), "=");
   }
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_ema9Handle != INVALID_HANDLE) IndicatorRelease(g_ema9Handle);
   if(g_ema21Handle != INVALID_HANDLE) IndicatorRelease(g_ema21Handle);
   if(g_ema50Handle != INVALID_HANDLE) IndicatorRelease(g_ema50Handle);
   if(g_ema200_5mHandle != INVALID_HANDLE) IndicatorRelease(g_ema200_5mHandle);
   if(g_ema200_1hHandle != INVALID_HANDLE) IndicatorRelease(g_ema200_1hHandle);
   if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Manage existing positions first
   if(g_breakout.active)
   {
      ManageBreakoutPosition();
      return;
   }
   
   // Check pending breakout (delay mode)
   if(g_pending.active)
   {
      CheckPendingBreakout();
      return;
   }
   
   // Look for new breakout opportunities
   // Use a simpler new bar detection for Strategy Tester compatibility
   static datetime lastBarTime = 0;
   datetime currentBarTime[];
   if(CopyTime(_Symbol, PERIOD_M5, 0, 1, currentBarTime) <= 0) return;
   
   // Only check on new bar (when bar time changes)
   if(currentBarTime[0] == lastBarTime) return;
   lastBarTime = currentBarTime[0];
   
   // Debug: Print every 100 bars to show EA is running
   static int tickCount = 0;
   tickCount++;
   if(tickCount % 100 == 0)
   {
      Print("OnTick: Bar ", tickCount, " | Time: ", TimeToString(currentBarTime[0]));
   }
   
   for(int i = 0; i < MAX_SESSIONS; i++)
   {
      if(!g_sessions[i].enabled)
      {
         static bool warnedSessions = false;
         if(!warnedSessions && i == 0)
         {
            Print("OnTick: Session ", i, " (", GetSessionName(i), ") is disabled");
            warnedSessions = true;
         }
         continue;
      }
      
      WindowRange range;
      if(GetWindowRange(i, range))
      {
         // Get close price safely
         double close[];
         if(CopyClose(_Symbol, PERIOD_M5, 1, 1, close) <= 0)
         {
            Print("OnTick [", GetSessionName(i), "]: Failed to get close price");
            continue;
         }
         double closePrice = close[0];
         
         // Debug: Print window info (only once per window)
         static string lastWindowId = "";
         if(range.id != lastWindowId)
         {
            Print("✅ Window detected [", GetSessionName(i), "]: ", range.id, " | High: ", range.high, " | Low: ", range.low, " | Price: ", closePrice);
            lastWindowId = range.id;
         }
         
         // Get current bar's open price for candle confirmation
         double open[];
         ArraySetAsSeries(open, true);
         if(CopyOpen(_Symbol, PERIOD_M5, 1, 1, open) <= 0)
         {
            continue; // Skip if we can't get open price
         }
         double openPrice = open[0];
         
         // Determine candle direction
         bool isBullishCandle = closePrice > openPrice;
         bool isBearishCandle = closePrice < openPrice;
         
         // Check for breakouts WITH CANDLE CONFIRMATION (like ORB_SmartTrap_EA)
         // BULLISH BREAKOUT: Close above OR high AND close above open (bullish candle)
         if(closePrice > range.high && isBullishCandle)
         {
            Print("🔺 Breakout UP detected [", GetSessionName(i), "]: ", closePrice, " > ", range.high, " (Bullish candle confirmed)");
            if(InpUseImmediateEntry)
               AttemptBreakout(range, closePrice, true);
            else
               DetectBreakout(range, closePrice, true);
         }
         // BEARISH BREAKOUT: Close below OR low AND close below open (bearish candle)
         else if(closePrice < range.low)
         {
            // Log bearish breakout detection (even if candle not confirmed)
            static int bearishCount = 0;
            bearishCount++;
            if(bearishCount % 10 == 0 || isBearishCandle)
            {
               Print("🔻 BEARISH BREAKOUT DETECTED [", GetSessionName(i), "]: ", closePrice, " < ", range.low);
               Print("   Candle is bearish: ", isBearishCandle, " | Open: ", openPrice, " | Close: ", closePrice);
            }
            
            if(isBearishCandle)
            {
               Print("✅ BEARISH CANDLE CONFIRMED - Calling AttemptBreakout/DetectBreakout");
               if(InpUseImmediateEntry)
                  AttemptBreakout(range, closePrice, false);
               else
                  DetectBreakout(range, closePrice, false);
            }
            else
            {
               static int rejectedBearish = 0;
               rejectedBearish++;
               if(rejectedBearish % 20 == 0)
               {
                  Print("⚠️ BEARISH BREAKOUT REJECTED: Candle not bearish (Open: ", openPrice, " >= Close: ", closePrice, ")");
               }
            }
         }
         else
         {
            // Price is within range - log occasionally for debugging
            static int inRangeCount = 0;
            inRangeCount++;
            if(inRangeCount % 50 == 0)
            {
               Print("Price in range [", GetSessionName(i), "]: ", closePrice, " (Range: ", range.low, " - ", range.high, ")");
            }
         }
      }
      else
      {
         // Window not found - log occasionally to show EA is checking
         static int noWindowCount = 0;
         noWindowCount++;
         if(noWindowCount % 500 == 0)
         {
            Print("OnTick: No window found for session ", i, " (", GetSessionName(i), ") - this is normal between sessions");
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Parse time string (HH:MM)                                        |
//+------------------------------------------------------------------+
void ParseTime(string timeStr, int &hour, int &minute)
{
   string parts[];
   int count = StringSplit(timeStr, ':', parts);
   if(count >= 2)
   {
      hour = (int)StringToInteger(parts[0]);
      minute = (int)StringToInteger(parts[1]);
   }
   else
   {
      hour = 0;
      minute = 0;
   }
}

//+------------------------------------------------------------------+
//| Get session name                                                  |
//+------------------------------------------------------------------+
string GetSessionName(int sessionIndex)
{
   switch(sessionIndex)
   {
      case 0: return "SESSION_1_03:00";
      case 1: return "SESSION_2_10:00";
      case 2: return "SESSION_3_16:30";
      default: return "UNKNOWN";
   }
}

//+------------------------------------------------------------------+
//| Get window range for session                                     |
//+------------------------------------------------------------------+
bool GetWindowRange(int sessionIndex, WindowRange &range)
{
   if(sessionIndex < 0 || sessionIndex >= MAX_SESSIONS || !g_sessions[sessionIndex].enabled)
   {
      static bool warned = false;
      if(!warned && sessionIndex == 0)
      {
         Print("GetWindowRange [", GetSessionName(sessionIndex), "]: Session disabled or invalid");
         warned = true;
      }
      return false;
   }
   
   datetime currentTime = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(currentTime, dt);
   
   // Find the most recent session that has started
   // Check today's session first
   dt.hour = g_sessions[sessionIndex].hour;
   dt.min = g_sessions[sessionIndex].minute;
   dt.sec = 0;
   datetime sessionStartToday = StructToTime(dt);
   
   // Determine which session to check (today's or yesterday's)
   datetime sessionStart = sessionStartToday;
   bool isTodaySession = true;
   
   if(sessionStartToday > currentTime)
   {
      // Today's session hasn't started yet, check yesterday's
      sessionStart = sessionStartToday - 86400; // Yesterday
      isTodaySession = false;
   }
   
   // Range is first 15-minute candle
   datetime rangeStart = sessionStart;
   datetime rangeEnd = sessionStart + 900; // 15 minutes
   
   // Trading window ends N hours after session
   datetime windowEnd = sessionStart + InpTradingWindowHours * 3600;
   
   // Check if we're in the trading window (after range completes, before window ends)
   if(currentTime < rangeEnd)
   {
      // Before range completes - not ready yet (silently skip, no warning needed)
      return false;
   }
   
   if(currentTime > windowEnd)
   {
      // After window ends - SILENTLY SKIP (no logging needed, this is normal)
      // We don't need to log expired windows - they're expected between sessions
      return false;
   }
   
   // Get range high/low from first 15-minute candle
   int rangeStartBar = iBarShift(_Symbol, PERIOD_M15, rangeStart);
   if(rangeStartBar < 0) 
   {
      // Try to find the bar using CopyTime
      datetime times[];
      if(CopyTime(_Symbol, PERIOD_M15, rangeStart, 1, times) <= 0)
      {
         Print("GetWindowRange [", GetSessionName(sessionIndex), "]: Failed to get M15 time for ", TimeToString(rangeStart));
         return false;
      }
      rangeStartBar = iBarShift(_Symbol, PERIOD_M15, times[0]);
      if(rangeStartBar < 0)
      {
         Print("GetWindowRange [", GetSessionName(sessionIndex), "]: Failed to find M15 bar for ", TimeToString(times[0]));
         return false;
      }
   }
   
   double high[], low[];
   if(CopyHigh(_Symbol, PERIOD_M15, rangeStartBar, 1, high) <= 0)
   {
      Print("GetWindowRange [", GetSessionName(sessionIndex), "]: Failed to get M15 high");
      return false;
   }
   if(CopyLow(_Symbol, PERIOD_M15, rangeStartBar, 1, low) <= 0)
   {
      Print("GetWindowRange [", GetSessionName(sessionIndex), "]: Failed to get M15 low");
      return false;
   }
   
   range.high = high[0];
   range.low = low[0];
   range.start = rangeStart;
   range.end = rangeEnd;
   range.id = StringFormat("S%d_%s", sessionIndex + 1, TimeToString(sessionStart, TIME_DATE));
   range.sessionIndex = sessionIndex;
   
   static string lastWindowId = "";
   if(range.id != lastWindowId)
   {
      Print("✅ GetWindowRange [", GetSessionName(sessionIndex), "]: Window found! ", range.id, " | High: ", range.high, " | Low: ", range.low);
      lastWindowId = range.id;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Count trades for window                                          |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Check if we've already traded this window                        |
//+------------------------------------------------------------------+
bool HasTradedWindow(const string &windowId)
{
   // Check if this window ID is in our traded windows array
   for(int i = 0; i < ArraySize(g_tradedWindows); i++)
   {
      if(g_tradedWindows[i] == windowId)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Mark window as traded                                            |
//+------------------------------------------------------------------+
void MarkWindowAsTraded(const string &windowId)
{
   // Add window ID to traded windows array
   int size = ArraySize(g_tradedWindows);
   ArrayResize(g_tradedWindows, size + 1);
   g_tradedWindows[size] = windowId;
   Print("✅ Window marked as traded: ", windowId);
}

//+------------------------------------------------------------------+
//| Count trades for window (DEPRECATED - use HasTradedWindow)      |
//+------------------------------------------------------------------+
int CountTradesForWindow(const string &windowId)
{
   // NEW LOGIC: Check if we've already traded this window
   if(HasTradedWindow(windowId))
      return InpMaxTradesPerWindow;  // Return max to block new trades
   
   // Also check open positions as backup
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         string comment = PositionGetString(POSITION_COMMENT);
         if(StringFind(comment, windowId) >= 0)
            count++;
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| Count total positions (all symbols)                             |
//+------------------------------------------------------------------+
int CountTotalPositions()
{
   return PositionsTotal();
}

//+------------------------------------------------------------------+
//| Count positions for current symbol                              |
//+------------------------------------------------------------------+
int CountPositionsForSymbol()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Check if we can open new position                               |
//+------------------------------------------------------------------+
bool CanOpenNewPosition()
{
   // Check global position limit
   if(CountTotalPositions() >= InpMaxTotalPositions)
   {
      Print("Cannot open position: Global limit reached (", InpMaxTotalPositions, ")");
      return false;
   }
   
   // Check per-symbol position limit
   if(CountPositionsForSymbol() >= InpMaxPositionsPerSymbol)
   {
      Print("Cannot open position: Symbol limit reached (", InpMaxPositionsPerSymbol, ")");
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Detect breakout (for delay mode)                                |
//+------------------------------------------------------------------+
void DetectBreakout(const WindowRange &range, double price, bool isLong)
{
   if(CountTradesForWindow(range.id) >= InpMaxTradesPerWindow)
      return;
   
   // Check position limits (CRITICAL: Prevents overexposure!)
   if(!CanOpenNewPosition())
      return;
   
   // Pre-entry filters
   if(!PassesPreEntryFilters(range, price, isLong))
      return;
   
   // Calculate risk and targets
   double rangeRisk = isLong ? (price - range.low) : (range.high - price);
   if(rangeRisk <= 0) return;
   
   double risk = rangeRisk * InpBreakoutInitialStopRatio;
   double stopPrice = isLong ? (price - risk) : (price + risk);
   double targetPrice = isLong ? (price + risk * InpRewardToRisk) : (price - risk * InpRewardToRisk);
   
   // Create pending breakout
   ZeroMemory(g_pending);
   g_pending.active = true;
   g_pending.detected_time = TimeCurrent();
   g_pending.detected_price = price;
   g_pending.stop_price = stopPrice;
   g_pending.target_price = targetPrice;
   g_pending.risk = risk;
   g_pending.direction = isLong ? POSITION_TYPE_BUY : POSITION_TYPE_SELL;
   g_pending.window_id = range.id;
   g_pending.range_high = range.high;
   g_pending.range_low = range.low;
   g_pending.bars_waiting = 0;
   g_pending.momentum_satisfied = false;
   g_pending.entry_type = "BREAKOUT";
}

//+------------------------------------------------------------------+
//| Attempt breakout (immediate entry)                              |
//+------------------------------------------------------------------+
void AttemptBreakout(const WindowRange &range, double price, bool isLong)
{
   string direction = isLong ? "LONG" : "SHORT";
   Print("=", StringSubstr("==========", 0, 50), "=");
   Print("🔍 AttemptBreakout called: ", direction, " | Price: ", price, " | Window: ", range.id);
   Print("   Session: ", GetSessionName(range.sessionIndex), " | Range: ", range.low, " - ", range.high);
   
   // CHECK IF WE'VE ALREADY TRADED THIS WINDOW (1 TRADE PER SESSION!)
   if(HasTradedWindow(range.id))
   {
      Print("❌ SKIPPED: Already traded this window! (Window ID: ", range.id, ")");
      Print("   We only take 1 trade per session window!");
      return;
   }
   
   // Also check position count as backup
   if(CountTradesForWindow(range.id) >= InpMaxTradesPerWindow)
   {
      Print("❌ Skipped: Max trades per window reached (", InpMaxTradesPerWindow, ")");
      return;
   }
   
   // Check position limits (CRITICAL: Prevents overexposure!)
   if(!CanOpenNewPosition())
   {
      Print("❌ Skipped: Position limit reached (Max Total: ", InpMaxTotalPositions, ", Max Per Symbol: ", InpMaxPositionsPerSymbol, ")");
      return;
   }
   
   // Pre-entry filters
   Print("🔍 Checking pre-entry filters for ", direction, " trade...");
   if(!PassesPreEntryFilters(range, price, isLong))
   {
      Print("❌ SHORT TRADE FILTERED OUT by pre-entry filters!");
      return;
   }
   Print("✅ All pre-entry filters passed for ", direction, " trade!");
   
   // Calculate risk and targets
   double rangeRisk = isLong ? (price - range.low) : (range.high - price);
   if(rangeRisk <= 0) return;
   
   double risk = rangeRisk * InpBreakoutInitialStopRatio;
   double stopPrice = isLong ? (price - risk) : (price + risk);
   double targetPrice = isLong ? (price + risk * InpRewardToRisk) : (price - risk * InpRewardToRisk);
   
   // Enter immediately
   double volume = CalculatePositionSize(price, stopPrice);
   if(volume <= 0)
   {
      Print("AttemptBreakout: Volume calculation failed (", volume, ") | Price: ", price, " | Stop: ", stopPrice);
      return;
   }
   
   Print("AttemptBreakout: Attempting to open ", isLong ? "LONG" : "SHORT", " | Volume: ", volume, " | Entry: ", price, " | SL: ", stopPrice, " | TP: ", targetPrice);
   
   ENUM_ORDER_TYPE orderType = isLong ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   string tag = StringFormat("BO_%s", range.id);
   
   if(SendMarketOrder(orderType, volume, price, stopPrice, targetPrice, tag))
   {
      Print("✅ TRADE OPENED! ", isLong ? "LONG" : "SHORT", " | Volume: ", volume, " | Entry: ", price);
      
      // MARK THIS WINDOW AS TRADED - CRITICAL FOR 1 TRADE PER SESSION!
      MarkWindowAsTraded(range.id);
      
      // Initialize breakout state
      ZeroMemory(g_breakout);
      g_breakout.active = true;
      g_breakout.entry_time = TimeCurrent();
      g_breakout.entry_price = price;
      g_breakout.stop_price = stopPrice;
      g_breakout.target_price = targetPrice;
      g_breakout.risk = risk;
      g_breakout.direction = isLong ? POSITION_TYPE_BUY : POSITION_TYPE_SELL;
      g_breakout.window_id = range.id;
      g_breakout.range_high = range.high;
      g_breakout.range_low = range.low;
      g_breakout.bars_in_trade = 0;
      g_breakout.max_mae = 0;
      g_breakout.momentum_satisfied = false;
      g_breakout.first_bar_checked = false;
      g_breakout.confirmed_before_entry = false;
      g_breakout.entry_type = "BREAKOUT";
   }
   else
   {
      Print("❌ TRADE FAILED! Order send returned false");
   }
}

//+------------------------------------------------------------------+
//| Check pending breakout (delay mode)                             |
//+------------------------------------------------------------------+
void CheckPendingBreakout()
{
   if(!g_pending.active) return;
   
   g_pending.bars_waiting++;
   
   // Check timeout
   if(g_pending.bars_waiting > InpConfirmationTimeoutBars)
   {
      g_pending.active = false;
      return;
   }
   
   double close[];
   if(CopyClose(_Symbol, PERIOD_M5, 1, 1, close) <= 0)
   {
      g_pending.active = false;
      return;
   }
   double closePrice = close[0];
   double detectedPrice = g_pending.detected_price;
   double risk = g_pending.risk;
   
   // Check momentum
   if(!g_pending.momentum_satisfied)
   {
      double gain = g_pending.direction == POSITION_TYPE_BUY ? 
                    (closePrice - detectedPrice) : (detectedPrice - closePrice);
      double gainR = gain / risk;
      
      if(gainR >= InpMomentumMinGain)
      {
         g_pending.momentum_satisfied = true;
      }
   }
   
   // Enter if momentum confirmed
   if(g_pending.momentum_satisfied)
   {
      // Check position limits before entering (CRITICAL!)
      if(!CanOpenNewPosition())
      {
         g_pending.active = false;
         return;
      }
      
      double volume = CalculatePositionSize(closePrice, g_pending.stop_price);
      if(volume > 0)
      {
         ENUM_ORDER_TYPE orderType = g_pending.direction == POSITION_TYPE_BUY ? 
                                     ORDER_TYPE_BUY : ORDER_TYPE_SELL;
         string tag = StringFormat("BO_%s", g_pending.window_id);
         
         if(SendMarketOrder(orderType, volume, closePrice, g_pending.stop_price, 
                           g_pending.target_price, tag))
         {
            // MARK THIS WINDOW AS TRADED - CRITICAL FOR 1 TRADE PER SESSION!
            MarkWindowAsTraded(g_pending.window_id);
            
            // Transfer to breakout state
            ZeroMemory(g_breakout);
            g_breakout.active = true;
            g_breakout.entry_time = TimeCurrent();
            g_breakout.entry_price = closePrice;
            g_breakout.stop_price = g_pending.stop_price;
            g_breakout.target_price = g_pending.target_price;
            g_breakout.risk = g_pending.risk;
            g_breakout.direction = g_pending.direction;
            g_breakout.window_id = g_pending.window_id;
            g_breakout.range_high = g_pending.range_high;
            g_breakout.range_low = g_pending.range_low;
            g_breakout.bars_in_trade = 0;
            g_breakout.max_mae = 0;
            g_breakout.momentum_satisfied = true;
            g_breakout.first_bar_checked = true;
            g_breakout.confirmed_before_entry = true;
            g_breakout.entry_type = g_pending.entry_type;
            
            g_pending.active = false;
         }
      }
   }
}


//+------------------------------------------------------------------+
//| Check pre-entry filters                                          |
//+------------------------------------------------------------------+
bool PassesPreEntryFilters(const WindowRange &range, double price, bool isLong)
{
   // If breakout controls disabled, skip all filters
   if(!InpUseBreakoutControls) return true;
   
   // EMA 200 (1H) filter (CHECKED FIRST - INDEPENDENT OF STRICT FILTERS)
   // WARNING: This filter blocks SHORT trades when price is above EMA 200!
   // For SHORT trades, you typically want price ABOVE EMA 200 (counter-trend fade)
   // But this filter blocks them - consider disabling if you want SHORT trades
   if(InpUseEMA200Filter && g_ema200_1hHandle != INVALID_HANDLE)
   {
      double ema200_1h = GetMAValue(g_ema200_1hHandle, 1);
      if(ema200_1h > 0)
      {
         if(isLong && price < ema200_1h)
         {
            Print("❌ Filtered [LONG]: Price (", price, ") below EMA 200 (1H) (", ema200_1h, ")");
            return false;
         }
         if(!isLong && price > ema200_1h)
         {
            Print("❌ Filtered [SHORT]: Price (", price, ") above EMA 200 (1H) (", ema200_1h, ")");
            Print("   ⚠️ EMA 200 FILTER IS BLOCKING SHORT TRADES!");
            Print("   💡 TIP: Set InpUseEMA200Filter = false if you want SHORT trades");
            return false;
         }
      }
   }
   
   // If strict filters disabled, only apply basic filters (entry offset)
   if(!InpUseStrictFilters)
   {
      // Only check entry offset filter (basic validation)
      double rangeWidth = range.high - range.low;
      if(rangeWidth > 0)
      {
         double rangeMid = (range.high + range.low) / 2.0;
         double offsetRatio = (price - rangeMid) / rangeWidth;
         if(offsetRatio < InpMinEntryOffsetRatio || offsetRatio > InpMaxEntryOffsetRatio)
         {
            return false;
         }
      }
      return true; // Pass basic filters
   }
   
   // STRICT FILTERS (100% win rate conditions) - only applied if InpUseStrictFilters = true
   
   // EMA Alignment filter
   if(InpUseEMAAlignment && isLong)
   {
      double ema9 = GetMAValue(g_ema9Handle, 1);
      double ema21 = GetMAValue(g_ema21Handle, 1);
      double ema50 = GetMAValue(g_ema50Handle, 1);
      
      if(ema9 > 0 && ema21 > 0 && ema50 > 0)
      {
         if(ema9 <= ema21 || ema21 <= ema50) return false;
      }
   }
   
   // Get ATR and ATR ratio (needed for multiple filters)
   double atr = GetATRValue(1);
   double atrRatio = GetATRRatio(1);
   double rangeWidth = range.high - range.low;
   
   // Breakout distance filter (only if parameter > 0)
   if(InpMaxBreakoutAtrMultiple > 0.0 && atr > 0)
   {
      // Match backtest: max(..., 0.0) ensures distance is never negative
      double breakoutDistance = isLong ? MathMax(price - range.high, 0.0) : MathMax(range.low - price, 0.0);
      double breakoutMultiple = breakoutDistance / atr;
      
      if(breakoutMultiple > InpMaxBreakoutAtrMultiple)
      {
         Print("❌ Filtered [", isLong ? "LONG" : "SHORT", "]: Breakout ATR Multiple too high: ", breakoutMultiple, " > ", InpMaxBreakoutAtrMultiple);
         return false;
      }
   }
   
   // ATR ratio filter (only if parameter > 0)
   if(InpMaxAtrRatio > 0.0 && atrRatio > InpMaxAtrRatio)
   {
      Print("Filtered: ATR Ratio too high: ", atrRatio, " > ", InpMaxAtrRatio);
      return false;
   }
   
   // Trend score filter (only if parameter > 0)
   if(InpMinTrendScore > 0.0)
   {
      double trendScore = CalculateTrendScore(isLong);
      if(trendScore < InpMinTrendScore)
      {
         Print("Filtered: Trend Score too low: ", trendScore, " < ", InpMinTrendScore);
         return false;
      }
   }
   
   // Consolidation filter (only if parameter >= 0, allows 0.0)
   if(InpMaxConsolidationScore >= 0.0)
   {
      double consolidationScore = CalculateConsolidationScore(range);
      if(consolidationScore > InpMaxConsolidationScore)
      {
         Print("Filtered [", GetSessionName(range.sessionIndex), "]: Consolidation Score too high: ", consolidationScore, " > ", InpMaxConsolidationScore);
         return false;
      }
   }
   
   // Range ATR Ratio filter (only if parameter > 0)
   if(InpMinRangeAtrRatio > 0.0 && rangeWidth > 0 && atr > 0)
   {
      double rangeAtrRatio = rangeWidth / atr;
      if(rangeAtrRatio < InpMinRangeAtrRatio)
      {
         Print("Filtered: Range ATR Ratio too low: ", rangeAtrRatio, " < ", InpMinRangeAtrRatio);
         return false;
      }
   }
   
   // Entry offset filter
   double rangeMid = (range.high + range.low) / 2.0;
   if(rangeWidth > 0)
   {
      double offsetRatio = (price - rangeMid) / rangeWidth;
      if(offsetRatio < InpMinEntryOffsetRatio || offsetRatio > InpMaxEntryOffsetRatio)
      {
         Print("Filtered: Entry Offset Ratio out of range: ", offsetRatio, " (min: ", InpMinEntryOffsetRatio, ", max: ", InpMaxEntryOffsetRatio, ")");
         return false;
      }
   }
   
   Print("✅ PASSED ALL PRE-ENTRY FILTERS!");
   return true;
}

//+------------------------------------------------------------------+
//| Manage breakout position                                         |
//+------------------------------------------------------------------+
void ManageBreakoutPosition()
{
   if(!g_breakout.active) return;
   
   g_breakout.bars_in_trade++;
   
   double close[], high[], low[];
   if(CopyClose(_Symbol, PERIOD_M5, 1, 1, close) <= 0) return;
   if(CopyHigh(_Symbol, PERIOD_M5, 1, 1, high) <= 0) return;
   if(CopyLow(_Symbol, PERIOD_M5, 1, 1, low) <= 0) return;
   
   double closePrice = close[0];
   double highPrice = high[0];
   double lowPrice = low[0];
   
   // Check SL/TP
   bool stopHit = false;
   bool targetHit = false;
   
   if(g_breakout.direction == POSITION_TYPE_BUY)
   {
      stopHit = lowPrice <= g_breakout.stop_price;
      targetHit = highPrice >= g_breakout.target_price;
   }
   else
   {
      stopHit = highPrice >= g_breakout.stop_price;
      targetHit = lowPrice <= g_breakout.target_price;
   }
   
   if(stopHit || targetHit)
   {
      CloseBreakoutPosition();
      return;
   }
   
   // Post-entry filters (only if not confirmed before entry)
   if(InpUseBreakoutControls && !g_breakout.confirmed_before_entry)
   {
      double risk = g_breakout.risk;
      if(risk > 0)
      {
         // First bar check
         if(g_breakout.bars_in_trade == 1 && !g_breakout.first_bar_checked)
         {
            double gain = g_breakout.direction == POSITION_TYPE_BUY ?
                         (closePrice - g_breakout.entry_price) : 
                         (g_breakout.entry_price - closePrice);
            double gainR = gain / risk;
            
            if(gainR < InpFirstBarMinGain)
            {
               CloseBreakoutPosition();
               return;
            }
            
            g_breakout.first_bar_checked = true;
         }
         
         // MAE check
         double adverse = g_breakout.direction == POSITION_TYPE_BUY ?
                         (g_breakout.entry_price - lowPrice) : (highPrice - g_breakout.entry_price);
         double maeR = adverse / risk;
         g_breakout.max_mae = MathMax(g_breakout.max_mae, maeR);
         
         if(maeR > InpBreakoutMaxMaeRatio)
         {
            CloseBreakoutPosition();
            return;
         }
         
         // Retest depth check
         if(g_breakout.bars_in_trade <= InpMaxRetestBars)
         {
            double retestDepth = 0;
            if(g_breakout.direction == POSITION_TYPE_BUY)
            {
               double minPrice = MathMin(lowPrice, g_breakout.range_low);
               retestDepth = (g_breakout.range_high - minPrice) / risk;
            }
            else
            {
               double maxPrice = MathMax(highPrice, g_breakout.range_high);
               retestDepth = (maxPrice - g_breakout.range_low) / risk;
            }
            
            g_breakout.max_retest_depth = MathMax(g_breakout.max_retest_depth, retestDepth);
            
            if(retestDepth > InpMaxRetestDepthR)
            {
               CloseBreakoutPosition();
               return;
            }
         }
         
         // Momentum check
         if(!g_breakout.momentum_satisfied)
         {
            double gain = g_breakout.direction == POSITION_TYPE_BUY ?
                         (closePrice - g_breakout.entry_price) : 
                         (g_breakout.entry_price - closePrice);
            double gainR = gain / risk;
            
            if(gainR >= InpBreakoutMomentumMinGain)
            {
               g_breakout.momentum_satisfied = true;
            }
            else if(g_breakout.bars_in_trade >= InpBreakoutMomentumBar)
            {
               CloseBreakoutPosition();
               return;
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close breakout position                                          |
//+------------------------------------------------------------------+
void CloseBreakoutPosition()
{
   if(!g_breakout.active) return;
   
   // Find and close position
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol)
      {
         string comment = PositionGetString(POSITION_COMMENT);
         if(StringFind(comment, g_breakout.window_id) >= 0)
         {
            MqlTradeRequest request;
            MqlTradeResult result;
            ZeroMemory(request);
            ZeroMemory(result);
            
            request.action = TRADE_ACTION_DEAL;
            request.position = ticket;
            request.symbol = _Symbol;
            request.volume = PositionGetDouble(POSITION_VOLUME);
            request.deviation = 10;
            request.type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ?
                          ORDER_TYPE_SELL : ORDER_TYPE_BUY;
            request.type_filling = ORDER_FILLING_FOK;
            request.comment = "EAAI_Close";
            
            if(OrderSend(request, result))
            {
               Print("Position closed: ", g_breakout.entry_type, " | ", g_breakout.window_id);
            }
         }
      }
   }
   
   g_breakout.active = false;
}

//+------------------------------------------------------------------+
//| Calculate position size                                          |
//+------------------------------------------------------------------+
double CalculatePositionSize(double entryPrice, double stopPrice)
{
   double risk = MathAbs(entryPrice - stopPrice);
   if(risk <= 0) return 0;
   
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * InpRiskPercent / 100.0;
   
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   if(tickSize > 0 && tickValue > 0 && point > 0)
   {
      double ticks = risk / point;
      double volume = riskAmount / (ticks * tickValue);
      
      double minVolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double maxVolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
      double volumeStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      
      volume = MathFloor(volume / volumeStep) * volumeStep;
      volume = MathMax(minVolume, MathMin(maxVolume, volume));
      
      return volume;
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Send market order                                                |
//+------------------------------------------------------------------+
bool SendMarketOrder(ENUM_ORDER_TYPE type, double volume, double entryPrice,
                     double stopPrice, double targetPrice, const string &tag)
{
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = volume;
   request.type = type;
   request.price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                                               SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.sl = stopPrice;
   request.tp = targetPrice;
   request.deviation = 10;
   request.magic = 123456;
   request.comment = tag;
   request.type_filling = ORDER_FILLING_FOK;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("Order opened: ", tag, " | Volume: ", volume, " | SL: ", stopPrice, " | TP: ", targetPrice);
         return true;
      }
      else
      {
         Print("Order failed: ", result.retcode, " | ", result.comment);
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Get MA value                                                     |
//+------------------------------------------------------------------+
double GetMAValue(int handle, int shift)
{
   double buffer[];
   ArraySetAsSeries(buffer, true);
   if(CopyBuffer(handle, 0, shift, 1, buffer) > 0)
      return buffer[0];
   return 0;
}

//+------------------------------------------------------------------+
//| Get ATR value                                                    |
//+------------------------------------------------------------------+
double GetATRValue(int shift)
{
   double buffer[];
   ArraySetAsSeries(buffer, true);
   if(CopyBuffer(g_atrHandle, 0, shift, 1, buffer) > 0)
      return buffer[0];
   return 0;
}

//+------------------------------------------------------------------+
//| Get ATR ratio                                                    |
//+------------------------------------------------------------------+
double GetATRRatio(int shift)
{
   double atr = GetATRValue(shift);
   if(atr <= 0) return 0;
   
   // Calculate ATR average (20 period)
   double sum = 0;
   int count = 0;
   for(int i = shift; i < shift + 20 && i < 100; i++)
   {
      double val = GetATRValue(i);
      if(val > 0)
      {
         sum += val;
         count++;
      }
   }
   
   if(count > 0)
   {
      double avg = sum / count;
      return atr / avg;
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Calculate trend score                                            |
//+------------------------------------------------------------------+
// Matches backtest logic: (EMA_9_Above_21 + EMA_21_Above_50 + Price_Above_EMA200_1H) / 3
//+------------------------------------------------------------------+
double CalculateTrendScore(bool isLong)
{
   double ema9 = GetMAValue(g_ema9Handle, 1);
   double ema21 = GetMAValue(g_ema21Handle, 1);
   double ema50 = GetMAValue(g_ema50Handle, 1);
   
   if(ema9 <= 0 || ema21 <= 0 || ema50 <= 0) return 0;
   
   // EMA alignment components (binary: 0 or 1)
   int ema9_above_21 = 0;
   int ema21_above_50 = 0;
   int price_above_ema200_1h = 0;
   
   if(isLong)
   {
      if(ema9 > ema21) ema9_above_21 = 1;
      if(ema21 > ema50) ema21_above_50 = 1;
      
      // Check Price_Above_EMA200_1H (only if handle is valid)
      if(g_ema200_1hHandle != INVALID_HANDLE)
      {
         double ema200_1h = GetMAValue(g_ema200_1hHandle, 0);  // 1H uses shift 0
         double close[];
         ArraySetAsSeries(close, true);
         if(CopyClose(_Symbol, PERIOD_M5, 1, 1, close) > 0 && ema200_1h > 0 && close[0] > ema200_1h)
            price_above_ema200_1h = 1;
      }
   }
   else
   {
      if(ema9 < ema21) ema9_above_21 = 1;
      if(ema21 < ema50) ema21_above_50 = 1;
      
      // For SHORT: Price_Below_EMA200_1H (inverse logic)
      if(g_ema200_1hHandle != INVALID_HANDLE)
      {
         double ema200_1h = GetMAValue(g_ema200_1hHandle, 0);  // 1H uses shift 0
         double close[];
         ArraySetAsSeries(close, true);
         if(CopyClose(_Symbol, PERIOD_M5, 1, 1, close) > 0 && ema200_1h > 0 && close[0] < ema200_1h)
            price_above_ema200_1h = 1;  // For SHORT, we want price below EMA200
      }
   }
   
   // Trend_Score = (EMA_9_Above_21 + EMA_21_Above_50 + Price_Above_EMA200_1H) / 3
   return (ema9_above_21 + ema21_above_50 + price_above_ema200_1h) / 3.0;
}

//+------------------------------------------------------------------+
//| Calculate consolidation score                                    |
//+------------------------------------------------------------------+
// Uses EAAI_Simple's proven calculation that works in Strategy Tester
// Returns 0.0, 0.5, or 1.0
//+------------------------------------------------------------------+
double CalculateConsolidationScore(const WindowRange &range)
{
   double rangeWidth = range.high - range.low;
   double atr = GetATRValue(1);
   
   if(rangeWidth <= 0.0 || atr <= 0.0) return 1.0;  // Default to high consolidation if invalid
   
   // Get ATR ratio (current ATR vs average)
   double atrRatio = GetATRRatio(1);
   if(atrRatio <= 0) return 1.0;
   
   // Simple calculation (proven to work in Strategy Tester)
   // isConsolidating = (atrRatio < 0.8) ? 1 : 0
   // isTightRange = (rangeWidth < atr) ? 1 : 0
   int isConsolidating = (atrRatio < 0.8) ? 1 : 0;
   int isTightRange = (rangeWidth < atr) ? 1 : 0;
   
   // Consolidation_Score = (Is_Consolidating + Is_Tight_Range) / 2
   return (isConsolidating + isTightRange) / 2.0;
}

//+------------------------------------------------------------------+

