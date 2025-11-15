//+------------------------------------------------------------------+
//|                                  EAAI_Simple_Model_FIXED.mq5 |
//|                        Simple Breakout EA - WORKING VERSION     |
//+------------------------------------------------------------------+
#property copyright "Simple Breakout EA"
#property version   "1.00"
#property strict

// === SESSIONS ===
input group "=== SESSIONS ==="
input int      InpAsianHour    = 3;
input int      InpAsianMinute  = 0;
input int      InpLondonHour   = 10;
input int      InpLondonMinute = 0;
input int      InpNYHour       = 16;
input int      InpNYMinute     = 30;
input int      InpWindowHours  = 3;

// === RISK ===
input group "=== RISK ==="
input double   InpRiskPercent  = 1.0;
input double   InpRewardToRisk = 2.0;

// === FILTERS ===
input group "=== FILTERS ==="
input bool     InpUseFilters   = false;  // DISABLE FILTERS FOR TESTING!
input double   InpMaxConsolidation = 0.5;

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                 |
//+------------------------------------------------------------------+
struct Session
{
   datetime rangeStart;
   datetime rangeEnd;
   datetime windowEnd;
   double rangeHigh;
   double rangeLow;
   bool traded;
   string name;
};

Session g_sessions[3];
int g_sessionCount = 0;
int g_atrHandle = INVALID_HANDLE;
datetime g_lastCheckTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                  |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("EAAI SIMPLE MODEL - WORKING VERSION");
   Print("=", StringSubstr("=", 0, 60), "=");
   
   g_atrHandle = iATR(_Symbol, PERIOD_M5, 14);
   if(g_atrHandle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create ATR");
      return INIT_FAILED;
   }
   
   Print("Sessions: ", InpAsianHour, ":", InpAsianMinute, " | ", 
         InpLondonHour, ":", InpLondonMinute, " | ", 
         InpNYHour, ":", InpNYMinute);
   Print("Risk: ", InpRiskPercent, "% | R:R = ", InpRewardToRisk);
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_atrHandle != INVALID_HANDLE)
      IndicatorRelease(g_atrHandle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime currentBar = iTime(_Symbol, PERIOD_M5, 0);
   if(currentBar == g_lastCheckTime)
      return; // Only check on new bar
   g_lastCheckTime = currentBar;
   
   // Count positions for THIS symbol only
   int positionsForSymbol = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol)
         positionsForSymbol++;
   }
   
   if(positionsForSymbol >= 3)
      return;
   
   UpdateSessions();
   
   datetime now = iTime(_Symbol, PERIOD_M5, 0);
   MqlDateTime dt;
   TimeToStruct(now, dt);
   
   for(int i = 0; i < g_sessionCount; i++)
   {
      if(g_sessions[i].traded)
         continue;
      
      // CRITICAL: Skip sessions from different days (cleanup should remove, but double-check)
      MqlDateTime sessionDt;
      TimeToStruct(g_sessions[i].rangeStart, sessionDt);
      if(sessionDt.day != dt.day || sessionDt.mon != dt.mon || sessionDt.year != dt.year)
      {
         // Mark as traded and skip - cleanup will remove it next tick
         g_sessions[i].traded = true;
         continue;
      }
      
      if(now < g_sessions[i].rangeEnd)
      {
         Print("DEBUG: Waiting for range to complete. Now: ", TimeToString(now), " RangeEnd: ", TimeToString(g_sessions[i].rangeEnd));
         continue; // Wait for range to complete
      }
      
      if(now > g_sessions[i].windowEnd)
      {
         // Window expired - mark as traded and skip silently (cleanup will remove it)
         g_sessions[i].traded = true;
         continue; // Window expired
      }
      
      Print("DEBUG: Checking breakout for ", g_sessions[i].name, " | Range: ", g_sessions[i].rangeLow, " - ", g_sessions[i].rangeHigh);
      CheckBreakout(g_sessions[i]);
   }
}

//+------------------------------------------------------------------+
//| Update sessions                                                   |
//+------------------------------------------------------------------+
void UpdateSessions()
{
   datetime now = iTime(_Symbol, PERIOD_M5, 0);
   MqlDateTime dt;
   TimeToStruct(now, dt);
   
   // FIRST: Clean up expired sessions BEFORE checking for new ones
   // Remove ALL sessions from different days or expired windows
   for(int i = g_sessionCount - 1; i >= 0; i--)
   {
      MqlDateTime sessionDt;
      TimeToStruct(g_sessions[i].rangeStart, sessionDt);
      
      // Remove if from different day OR window expired (with 1 hour buffer)
      bool isDifferentDay = (sessionDt.day != dt.day || sessionDt.mon != dt.mon || sessionDt.year != dt.year);
      bool isExpired = (now > g_sessions[i].windowEnd + 3600); // 1 hour past window end
      
      if(isDifferentDay || isExpired)
      {
         // Remove expired session by shifting array
         for(int j = i; j < g_sessionCount - 1; j++)
            g_sessions[j] = g_sessions[j + 1];
         g_sessionCount--;
         
         // Only log if from different day (to reduce spam)
         if(isDifferentDay)
            Print("DEBUG: Removed session from different day - ", g_sessions[i].name, " | Session: ", TimeToString(g_sessions[i].rangeStart), " | Now: ", TimeToString(now));
      }
   }
   
   // THEN: Check for sessions that started earlier today (if we missed them)
   CheckTodaySessions(dt);
   
   // THEN: Check for new sessions starting now
   CheckSession(InpAsianHour, InpAsianMinute, "ASIAN", g_sessionCount);
   CheckSession(InpLondonHour, InpLondonMinute, "LONDON", g_sessionCount);
   CheckSession(InpNYHour, InpNYMinute, "NY", g_sessionCount);
}

//+------------------------------------------------------------------+
//| Check session                                                     |
//+------------------------------------------------------------------+
void CheckSession(int hour, int minute, string name, int &idx)
{
   datetime now = iTime(_Symbol, PERIOD_M5, 0);
   MqlDateTime dt;
   TimeToStruct(now, dt);
   
   // Check if we're at session start time (first 15 minutes)
   if(dt.hour == hour && dt.min >= minute && dt.min < minute + 15)
   {
      // Calculate exact session start
      datetime sessionStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", 
                                                        dt.year, dt.mon, dt.day, hour, minute));
      
      // Check if we already have this session
      bool alreadyExists = false;
      for(int j = 0; j < g_sessionCount; j++)
      {
         if(g_sessions[j].name == name)
         {
            MqlDateTime existingDt;
            TimeToStruct(g_sessions[j].rangeStart, existingDt);
            if(existingDt.day == dt.day && existingDt.mon == dt.mon && existingDt.year == dt.year)
            {
               alreadyExists = true;
               break;
            }
         }
      }
      
      if(alreadyExists)
         return;
      
      // Get range from first 3 bars (15 minutes) starting from session start
      double high = 0, low = DBL_MAX;
      int barsFound = 0;
      
      // Use iBarShift to find the starting bar index
      int startBar = iBarShift(_Symbol, PERIOD_M5, sessionStart);
      if(startBar < 0) startBar = 0; // If not found, use current bar
      
      // Get range from first 3 bars (15 minutes = 3 bars of 5M)
      for(int i = 0; i < 3; i++)
      {
         int barIdx = startBar + i;
         if(barIdx < 0) continue; // Skip if out of bounds
         
         datetime barTime = iTime(_Symbol, PERIOD_M5, barIdx);
         if(barTime >= sessionStart && barTime < sessionStart + 15*60)
         {
            double h = iHigh(_Symbol, PERIOD_M5, barIdx);
            double l = iLow(_Symbol, PERIOD_M5, barIdx);
            if(h > high) high = h;
            if(l < low) low = l;
            barsFound++;
         }
      }
      
      // Fallback: use first 3 bars if exact match not found
      if(barsFound < 2)
      {
         high = 0;
         low = DBL_MAX;
         barsFound = 0;
         for(int i = 0; i < 3; i++)
         {
            double h = iHigh(_Symbol, PERIOD_M5, i);
            double l = iLow(_Symbol, PERIOD_M5, i);
            if(h > high) high = h;
            if(l < low) low = l;
            barsFound++;
         }
      }
      
      // Validate range: must have at least 2 bars and non-zero width
      if(barsFound < 2)
      {
         Print("DEBUG: CheckSession - Not enough bars found (", barsFound, ") for ", name);
         return;
      }
      
      if(low >= high)
      {
         Print("DEBUG: CheckSession - Zero-width range detected for ", name, " (", low, " >= ", high, ")");
         return;
      }
      
      if(low < DBL_MAX && high > 0 && idx < 3)
      {
         g_sessions[idx].rangeStart = sessionStart;
         g_sessions[idx].rangeEnd = sessionStart + 15*60;
         g_sessions[idx].windowEnd = sessionStart + InpWindowHours*3600;
         g_sessions[idx].rangeHigh = high;
         g_sessions[idx].rangeLow = low;
         g_sessions[idx].name = name;
         g_sessions[idx].traded = HasTraded(name);
         
         Print("DEBUG: Session detected - ", name, " | Start: ", TimeToString(sessionStart), 
               " | Range: ", low, " - ", high, " | Window ends: ", TimeToString(g_sessions[idx].windowEnd));
         
         idx++;
      }
   }
}

//+------------------------------------------------------------------+
//| Check for sessions that started earlier today                     |
//+------------------------------------------------------------------+
void CheckTodaySessions(MqlDateTime &dt)
{
   datetime today = StringToTime(StringFormat("%04d.%02d.%02d 00:00", dt.year, dt.mon, dt.day));
   datetime now = iTime(_Symbol, PERIOD_M5, 0);
   
   // Check each session time
   int sessions[3][2] = {{InpAsianHour, InpAsianMinute}, {InpLondonHour, InpLondonMinute}, {InpNYHour, InpNYMinute}};
   string names[3] = {"ASIAN", "LONDON", "NY"};
   
   for(int s = 0; s < 3; s++)
   {
      datetime sessionStart = today + sessions[s][0]*3600 + sessions[s][1]*60;
      datetime rangeEnd = sessionStart + 15*60;
      datetime windowEnd = sessionStart + InpWindowHours*3600;
      
      // CRITICAL: Verify session is from TODAY (not previous day)
      MqlDateTime sessionStartDt;
      TimeToStruct(sessionStart, sessionStartDt);
      if(sessionStartDt.day != dt.day || sessionStartDt.mon != dt.mon || sessionStartDt.year != dt.year)
         continue; // Skip if not from today
      
      // Skip if session hasn't started yet or window expired
      if(now < sessionStart || now > windowEnd)
         continue;
      
      // Check if we already have this session
      bool exists = false;
      for(int i = 0; i < g_sessionCount; i++)
      {
         if(g_sessions[i].name == names[s])
         {
            MqlDateTime existingDt;
            TimeToStruct(g_sessions[i].rangeStart, existingDt);
            if(existingDt.day == dt.day && existingDt.mon == dt.mon && existingDt.year == dt.year)
            {
               exists = true;
               break;
            }
         }
      }
      
      if(exists)
         continue;
      
      // Get range from first 3 bars of session
      double high = 0, low = DBL_MAX;
      int barsFound = 0;
      
      // Use iBarShift to find the starting bar index
      int startBar = iBarShift(_Symbol, PERIOD_M5, sessionStart);
      if(startBar < 0) startBar = 0; // If not found, use current bar
      
      // Get range from first 3 bars (15 minutes = 3 bars of 5M)
      for(int i = 0; i < 3; i++)
      {
         int barIdx = startBar + i;
         if(barIdx < 0) continue; // Skip if out of bounds
         
         datetime barTime = iTime(_Symbol, PERIOD_M5, barIdx);
         if(barTime >= sessionStart && barTime < rangeEnd)
         {
            double h = iHigh(_Symbol, PERIOD_M5, barIdx);
            double l = iLow(_Symbol, PERIOD_M5, barIdx);
            if(h > high) high = h;
            if(l < low) low = l;
            barsFound++;
         }
      }
      
      // Validate range: must have at least 2 bars and non-zero width
      if(barsFound < 2)
      {
         Print("DEBUG: CheckTodaySessions - Not enough bars found (", barsFound, ") for ", names[s]);
         continue;
      }
      
      if(low >= high)
      {
         Print("DEBUG: CheckTodaySessions - Zero-width range detected for ", names[s], " (", low, " >= ", high, ")");
         continue;
      }
      
      if(low < DBL_MAX && high > 0 && g_sessionCount < 3)
      {
         int idx = g_sessionCount;
         g_sessions[idx].rangeStart = sessionStart;
         g_sessions[idx].rangeEnd = rangeEnd;
         g_sessions[idx].windowEnd = windowEnd;
         g_sessions[idx].rangeHigh = high;
         g_sessions[idx].rangeLow = low;
         g_sessions[idx].name = names[s];
         g_sessions[idx].traded = HasTraded(names[s]);
         
         Print("DEBUG: Found existing session - ", names[s], " | Range: ", low, " - ", high);
         g_sessionCount++;
      }
   }
}

//+------------------------------------------------------------------+
//| Check if traded                                                   |
//+------------------------------------------------------------------+
bool HasTraded(string session)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      string comment = PositionGetString(POSITION_COMMENT);
      if(StringFind(comment, session) >= 0)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Check breakout                                                    |
//+------------------------------------------------------------------+
void CheckBreakout(Session &s)
{
   // Safety check: skip zero-width ranges
   if(s.rangeHigh <= s.rangeLow)
   {
      Print("DEBUG: Skipping breakout check - zero-width range: ", s.rangeLow, " - ", s.rangeHigh);
      s.traded = true; // Mark as traded to prevent repeated checks
      return;
   }
   
   double close = iClose(_Symbol, PERIOD_M5, 0);
   double open = iOpen(_Symbol, PERIOD_M5, 0);
   double high = iHigh(_Symbol, PERIOD_M5, 0);
   double low = iLow(_Symbol, PERIOD_M5, 0);
   
   Print("DEBUG: Checking breakout | Close: ", close, " | Range: ", s.rangeLow, " - ", s.rangeHigh);
   
   // LONG: Close above range high AND bullish candle
   if(close > s.rangeHigh && close > open)
   {
      Print("DEBUG: LONG breakout detected! Close: ", close, " > RangeHigh: ", s.rangeHigh, " | Bullish: ", (close > open));
      
      if(!InpUseFilters || PassesFilters(true))
      {
         double entry = close;
         double sl = s.rangeLow;
         double risk = entry - sl;
         double tp = entry + (risk * InpRewardToRisk);
         
         Print("DEBUG: Opening LONG | Entry: ", entry, " | SL: ", sl, " | TP: ", tp);
         
         if(OpenTrade(ORDER_TYPE_BUY, entry, sl, tp, s.name))
         {
            s.traded = true;
            Print("✅ TRADE OPENED: ", s.name, " LONG at ", entry);
         }
      }
      else
      {
         Print("DEBUG: LONG breakout filtered out");
      }
      return;
   }
   
   // SHORT: Close below range low AND bearish candle
   if(close < s.rangeLow && close < open)
   {
      Print("DEBUG: SHORT breakout detected! Close: ", close, " < RangeLow: ", s.rangeLow, " | Bearish: ", (close < open));
      
      if(!InpUseFilters || PassesFilters(false))
      {
         double entry = close;
         double sl = s.rangeHigh;
         double risk = sl - entry;
         double tp = entry - (risk * InpRewardToRisk);
         
         Print("DEBUG: Opening SHORT | Entry: ", entry, " | SL: ", sl, " | TP: ", tp);
         
         if(OpenTrade(ORDER_TYPE_SELL, entry, sl, tp, s.name))
         {
            s.traded = true;
            Print("✅ TRADE OPENED: ", s.name, " SHORT at ", entry);
         }
      }
      else
      {
         Print("DEBUG: SHORT breakout filtered out");
      }
   }
}

//+------------------------------------------------------------------+
//| Check filters                                                     |
//+------------------------------------------------------------------+
bool PassesFilters(bool isLong)
{
   double atr[20];
   if(CopyBuffer(g_atrHandle, 0, 0, 20, atr) < 20)
      return false;
   
   double currentATR = atr[0];
   double avgATR = 0;
   for(int i = 1; i < 20; i++)
      avgATR += atr[i];
   avgATR /= 19.0;
   
   bool consolidating = (currentATR < avgATR * 0.7);
   
   double high[20], low[20];
   for(int i = 0; i < 20; i++)
   {
      high[i] = iHigh(_Symbol, PERIOD_M5, i);
      low[i] = iLow(_Symbol, PERIOD_M5, i);
   }
   
   double currentRange = high[0] - low[0];
   double avgRange = 0;
   for(int i = 1; i < 20; i++)
      avgRange += (high[i] - low[i]);
   avgRange /= 19.0;
   
   bool tightRange = (currentRange < avgRange * 0.8);
   
   double score = ((consolidating ? 1.0 : 0.0) + (tightRange ? 1.0 : 0.0)) / 2.0;
   
   return (score <= InpMaxConsolidation);
}

//+------------------------------------------------------------------+
//| Open trade                                                        |
//+------------------------------------------------------------------+
bool OpenTrade(ENUM_ORDER_TYPE type, double entry, double sl, double tp, string session)
{
   double risk = MathAbs(entry - sl);
   if(risk <= 0) return false;
   
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * InpRiskPercent / 100.0;
   double lotSize = riskAmount / risk;
   
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   
   MqlTradeRequest req = {};
   MqlTradeResult res = {};
   
   req.action = TRADE_ACTION_DEAL;
   req.symbol = _Symbol;
   req.volume = lotSize;
   req.type = type;
   req.price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   req.sl = sl;
   req.tp = tp;
   req.deviation = 10;
   req.magic = 123456;
   req.comment = "Simple_" + session;
   
   if(!OrderSend(req, res))
   {
      Print("ERROR: OrderSend failed - ", res.retcode, " - ", res.comment);
      return false;
   }
   
   Print("OK: ", session, " ", (type == ORDER_TYPE_BUY ? "LONG" : "SHORT"), " at ", entry);
   return true;
}

