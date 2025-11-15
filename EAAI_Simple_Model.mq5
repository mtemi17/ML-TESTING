//+------------------------------------------------------------------+
//|                                          EAAI_Simple_Model.mq5 |
//|                        Simple Breakout EA with Model-Based Filters |
//|                                                                  |
//| Strategy:                                                        |
//|   1. First 15M candle of session (03:00, 10:00, 16:30)          |
//|   2. Wait for breakout with candle confirmation                  |
//|   3. SL at opposite range, TP at 2R                             |
//|                                                                  |
//| Filters (from ML model):                                        |
//|   - ATR (most important)                                         |
//|   - Consolidation_Score                                         |
//|   - EMA_200_1H                                                  |
//+------------------------------------------------------------------+
#property copyright "Simple Breakout EA with Model Filters"
#property version   "1.00"
#property strict

// === SESSIONS ===
input group "=== SESSIONS ==="
input int      InpAsianStartHour    = 3;      // Asian session start hour
input int      InpAsianStartMinute  = 0;      // Asian session start minute
input int      InpLondonStartHour   = 10;     // London session start hour
input int      InpLondonStartMinute = 0;      // London session start minute
input int      InpNYStartHour       = 16;     // NY session start hour
input int      InpNYStartMinute     = 30;     // NY session start minute
input int      InpTradingWindowHours = 3;     // Trading window (hours after session start)

// === RISK MANAGEMENT ===
input group "=== RISK MANAGEMENT ==="
input double   InpRiskPercent       = 1.0;    // Risk % per trade
input double   InpRewardToRisk      = 2.0;    // Reward-to-risk ratio

// === MODEL-BASED FILTERS ===
input group "=== MODEL-BASED FILTERS (from trained model) ==="
input bool     InpUseModelFilters   = true;   // Use model-based filters
input double   InpMinATR            = 0.0;    // Min ATR (0 = no filter)
input double   InpMaxATR            = 0.0;    // Max ATR (0 = no filter)
input double   InpMaxConsolidationScore = 0.5; // Max Consolidation Score (model #2)
input bool     InpUseEMA200Filter   = false;  // Use EMA 200 (1H) filter (model #4)

// === POSITION LIMITS ===
input group "=== POSITION LIMITS ==="
input int      InpMaxTotalPositions = 3;      // Max total positions
input int      InpMaxPositionsPerSymbol = 1;  // Max positions per symbol

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                 |
//+------------------------------------------------------------------+
struct SessionRange
{
   datetime rangeStart;
   datetime rangeEnd;
   datetime windowEnd;
   double rangeHigh;
   double rangeLow;
   bool hasTraded;
   string sessionName;
};

SessionRange g_sessions[3];
int g_sessionCount = 0;

// Indicator handles
int g_atrHandle = INVALID_HANDLE;
int g_ema200_1hHandle = INVALID_HANDLE;

// Forward declarations
void UpdateSessionRanges();
void CheckSession(int startHour, int startMinute, string sessionName, int &sessionIndex);
bool HasTradedSession(string sessionName, datetime rangeStart);
bool GetSessionRange(datetime startTime, datetime endTime, SessionRange &range);
void CheckBreakout(SessionRange &session);
bool PassesModelFilters(bool isLong);
double CalculateConsolidationScore();
bool OpenTrade(ENUM_ORDER_TYPE orderType, double entryPrice, double slPrice, double tpPrice, string sessionName);
bool CanOpenNewPosition();
datetime ConvertStructToTime(MqlDateTime &dt);

//+------------------------------------------------------------------+
//| Expert initialization function                                  |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("EAAI SIMPLE MODEL - Simple Breakout with Model Filters");
   Print("=", StringSubstr("=", 0, 60), "=");
   
   // Initialize indicator handles
   g_atrHandle = iATR(_Symbol, PERIOD_M5, 14);
   if(g_atrHandle == INVALID_HANDLE)
   {
      Print("❌ Failed to create ATR indicator");
      return INIT_FAILED;
   }
   
   if(InpUseEMA200Filter)
   {
      g_ema200_1hHandle = iMA(_Symbol, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE);
      if(g_ema200_1hHandle == INVALID_HANDLE)
      {
         Print("❌ Failed to create EMA 200 (1H) indicator");
         return INIT_FAILED;
      }
   }
   
   Print("SESSIONS:");
   Print("  Asian: ", InpAsianStartHour, ":", StringFormat("%02d", InpAsianStartMinute));
   Print("  London: ", InpLondonStartHour, ":", StringFormat("%02d", InpLondonStartMinute));
   Print("  NY: ", InpNYStartHour, ":", StringFormat("%02d", InpNYStartMinute));
   Print("  Trading Window: ", InpTradingWindowHours, " hours");
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("RISK: ", InpRiskPercent, "% per trade | R:R = ", InpRewardToRisk);
   Print("=", StringSubstr("=", 0, 60), "=");
   Print("MODEL FILTERS:");
   Print("  Use Model Filters: ", InpUseModelFilters ? "ON" : "OFF");
   if(InpUseModelFilters)
   {
      Print("    - ATR: ", InpMinATR > 0 ? StringFormat("%.2f", InpMinATR) : "No min", " - ", InpMaxATR > 0 ? StringFormat("%.2f", InpMaxATR) : "No max");
      Print("    - Max Consolidation Score: ", InpMaxConsolidationScore);
      Print("    - EMA 200 (1H) Filter: ", InpUseEMA200Filter ? "ON" : "OFF");
   }
   Print("=", StringSubstr("=", 0, 60), "=");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_atrHandle != INVALID_HANDLE)
      IndicatorRelease(g_atrHandle);
   if(g_ema200_1hHandle != INVALID_HANDLE)
      IndicatorRelease(g_ema200_1hHandle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if we can open new positions
   if(!CanOpenNewPosition())
      return;
   
   // Update session ranges
   UpdateSessionRanges();
   
   // Check for breakouts in active sessions
   for(int i = 0; i < g_sessionCount; i++)
   {
      if(g_sessions[i].hasTraded)
         continue;
      
      datetime currentTime = iTime(_Symbol, PERIOD_M5, 0);
      
      // Check if we're in the trading window
      if(currentTime < g_sessions[i].rangeEnd || currentTime > g_sessions[i].windowEnd)
         continue;
      
      // Check for breakout
      CheckBreakout(g_sessions[i]);
   }
}

//+------------------------------------------------------------------+
//| Update session ranges                                            |
//+------------------------------------------------------------------+
void UpdateSessionRanges()
{
   datetime currentTime = iTime(_Symbol, PERIOD_M5, 0);
   MqlDateTime dt;
   TimeToStruct(currentTime, dt);
   
   g_sessionCount = 0;
   
   // Check each session
   CheckSession(InpAsianStartHour, InpAsianStartMinute, "ASIAN", g_sessionCount);
   CheckSession(InpLondonStartHour, InpLondonStartMinute, "LONDON", g_sessionCount);
   CheckSession(InpNYStartHour, InpNYStartMinute, "NEW_YORK", g_sessionCount);
}

//+------------------------------------------------------------------+
//| Check if we're in a session range period                         |
//+------------------------------------------------------------------+
void CheckSession(int startHour, int startMinute, string sessionName, int &sessionIndex)
{
   datetime currentTime = iTime(_Symbol, PERIOD_M5, 0);
   MqlDateTime dt;
   TimeToStruct(currentTime, dt);
   
   // Check if current time matches session start
   if(dt.hour == startHour && dt.min >= startMinute && dt.min < startMinute + 15)
   {
      // Get the 15M candle that contains this time
      datetime rangeStart = ConvertStructToTime(dt);
      rangeStart = rangeStart - (dt.min % 15) * 60 - dt.sec; // Round down to 15M boundary
      
      datetime rangeEnd = rangeStart + 15 * 60;
      datetime windowEnd = rangeStart + InpTradingWindowHours * 3600;
      
      if(GetSessionRange(rangeStart, rangeEnd, g_sessions[sessionIndex]))
      {
         g_sessions[sessionIndex].rangeStart = rangeStart;
         g_sessions[sessionIndex].rangeEnd = rangeEnd;
         g_sessions[sessionIndex].windowEnd = windowEnd;
         g_sessions[sessionIndex].sessionName = sessionName;
         g_sessions[sessionIndex].hasTraded = HasTradedSession(sessionName, rangeStart);
         sessionIndex++;
      }
   }
}

//+------------------------------------------------------------------+
//| Check if we've already traded this session                       |
//+------------------------------------------------------------------+
bool HasTradedSession(string sessionName, datetime rangeStart)
{
   // Check open positions
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0)
         continue;
      
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      
      string comment = PositionGetString(POSITION_COMMENT);
      if(StringFind(comment, "SimpleModel_" + sessionName) >= 0)
         return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Get session range (15-minute candle high/low)                   |
//+------------------------------------------------------------------+
bool GetSessionRange(datetime startTime, datetime endTime, SessionRange &range)
{
   // Get 15-minute data - find the candle that matches startTime
   int bars = iBars(_Symbol, PERIOD_M15);
   if(bars < 1)
      return false;
   
   // Find the 15M candle that starts at startTime
   for(int i = 0; i < MathMin(bars, 10); i++)
   {
      datetime candleTime = iTime(_Symbol, PERIOD_M15, i);
      
      if(candleTime == startTime || (candleTime < startTime && candleTime + 15*60 > startTime))
      {
         // Get high/low from this 15M candle
         double high = iHigh(_Symbol, PERIOD_M15, i);
         double low = iLow(_Symbol, PERIOD_M15, i);
         
         range.rangeHigh = high;
         range.rangeLow = low;
         return true;
      }
   }
   
   // Fallback: use current 15M candle if we can't find exact match
   datetime candleTime = iTime(_Symbol, PERIOD_M15, 0);
   if(candleTime >= startTime && candleTime < endTime)
   {
      range.rangeHigh = iHigh(_Symbol, PERIOD_M15, 0);
      range.rangeLow = iLow(_Symbol, PERIOD_M15, 0);
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check for breakout                                               |
//+------------------------------------------------------------------+
void CheckBreakout(SessionRange &session)
{
   double close = iClose(_Symbol, PERIOD_M5, 0);
   double open = iOpen(_Symbol, PERIOD_M5, 0);
   double high = iHigh(_Symbol, PERIOD_M5, 0);
   double low = iLow(_Symbol, PERIOD_M5, 0);
   
   bool isBullishCandle = close > open;
   bool isBearishCandle = close < open;
   
   // BULLISH BREAKOUT: Close above range high AND bullish candle
   if(close > session.rangeHigh && isBullishCandle)
   {
      if(PassesModelFilters(true))
      {
         double entryPrice = close;
         double slPrice = session.rangeLow;
         double risk = entryPrice - slPrice;
         double tpPrice = entryPrice + (risk * InpRewardToRisk);
         
         if(OpenTrade(ORDER_TYPE_BUY, entryPrice, slPrice, tpPrice, session.sessionName))
         {
            session.hasTraded = true;
            Print("✅ [", session.sessionName, "] LONG breakout at ", entryPrice, " | SL: ", slPrice, " | TP: ", tpPrice);
         }
      }
      return;
   }
   
   // BEARISH BREAKOUT: Close below range low AND bearish candle
   if(close < session.rangeLow && isBearishCandle)
   {
      if(PassesModelFilters(false))
      {
         double entryPrice = close;
         double slPrice = session.rangeHigh;
         double risk = slPrice - entryPrice;
         double tpPrice = entryPrice - (risk * InpRewardToRisk);
         
         if(OpenTrade(ORDER_TYPE_SELL, entryPrice, slPrice, tpPrice, session.sessionName))
         {
            session.hasTraded = true;
            Print("✅ [", session.sessionName, "] SHORT breakout at ", entryPrice, " | SL: ", slPrice, " | TP: ", tpPrice);
         }
      }
      return;
   }
}

//+------------------------------------------------------------------+
//| Check model-based filters                                        |
//+------------------------------------------------------------------+
bool PassesModelFilters(bool isLong)
{
   if(!InpUseModelFilters)
      return true;
   
   // Get ATR (Model #1 - Most Important)
   double atr[1];
   if(CopyBuffer(g_atrHandle, 0, 0, 1, atr) <= 0)
      return false;
   
   if(InpMinATR > 0 && atr[0] < InpMinATR)
   {
      Print("❌ Filtered: ATR too low (", atr[0], " < ", InpMinATR, ")");
      return false;
   }
   
   if(InpMaxATR > 0 && atr[0] > InpMaxATR)
   {
      Print("❌ Filtered: ATR too high (", atr[0], " > ", InpMaxATR, ")");
      return false;
   }
   
   // Get Consolidation Score (Model #2)
   double consolidationScore = CalculateConsolidationScore();
   if(consolidationScore > InpMaxConsolidationScore)
   {
      Print("❌ Filtered: Consolidation Score too high (", consolidationScore, " > ", InpMaxConsolidationScore, ")");
      return false;
   }
   
   // EMA 200 (1H) filter (Model #4)
   if(InpUseEMA200Filter && g_ema200_1hHandle != INVALID_HANDLE)
   {
      double ema200[1];
      if(CopyBuffer(g_ema200_1hHandle, 0, 0, 1, ema200) <= 0)
         return false;
      
      double price = iClose(_Symbol, PERIOD_M5, 0);
      
      if(isLong && price < ema200[0])
      {
         Print("❌ Filtered [LONG]: Price below EMA 200 (1H)");
         return false;
      }
      
      if(!isLong && price > ema200[0])
      {
         Print("❌ Filtered [SHORT]: Price above EMA 200 (1H)");
         return false;
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Calculate Consolidation Score                                    |
//+------------------------------------------------------------------+
double CalculateConsolidationScore()
{
   // Simple consolidation detection: compare current ATR to average
   double atr[20];
   if(CopyBuffer(g_atrHandle, 0, 0, 20, atr) < 20)
      return 0.0;
   
   double currentATR = atr[0];
   double avgATR = 0.0;
   for(int i = 1; i < 20; i++)
      avgATR += atr[i];
   avgATR /= 19.0;
   
   // Is consolidating if ATR < 70% of average
   bool isConsolidating = (currentATR < avgATR * 0.7);
   
   // Is tight range: compare current range to average
   double high[20], low[20];
   for(int i = 0; i < 20; i++)
   {
      high[i] = iHigh(_Symbol, PERIOD_M5, i);
      low[i] = iLow(_Symbol, PERIOD_M5, i);
   }
   
   double currentRange = high[0] - low[0];
   double avgRange = 0.0;
   for(int i = 1; i < 20; i++)
      avgRange += (high[i] - low[i]);
   avgRange /= 19.0;
   
   bool isTightRange = (currentRange < avgRange * 0.8);
   
   // Consolidation Score = (Is_Consolidating + Is_Tight_Range) / 2
   double score = ((isConsolidating ? 1.0 : 0.0) + (isTightRange ? 1.0 : 0.0)) / 2.0;
   
   return score;
}

//+------------------------------------------------------------------+
//| Open trade                                                        |
//+------------------------------------------------------------------+
bool OpenTrade(ENUM_ORDER_TYPE orderType, double entryPrice, double slPrice, double tpPrice, string sessionName)
{
   double risk = MathAbs(entryPrice - slPrice);
   if(risk <= 0)
      return false;
   
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * InpRiskPercent / 100.0;
   double lotSize = riskAmount / risk;
   
   // Normalize lot size
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lotSize;
   request.type = orderType;
   request.price = (orderType == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.sl = slPrice;
   request.tp = tpPrice;
   request.deviation = 10;
   request.magic = 123456;
   request.comment = "SimpleModel_" + sessionName;
   
   if(!OrderSend(request, result))
   {
      Print("❌ OrderSend failed: ", result.retcode, " - ", result.comment);
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if we can open new position                                |
//+------------------------------------------------------------------+
bool CanOpenNewPosition()
{
   int totalPositions = 0;
   int symbolPositions = 0;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0)
         continue;
      
      if(PositionGetString(POSITION_SYMBOL) == _Symbol)
         symbolPositions++;
      
      totalPositions++;
   }
   
   if(totalPositions >= InpMaxTotalPositions)
      return false;
   
   if(symbolPositions >= InpMaxPositionsPerSymbol)
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Helper: Convert MqlDateTime to datetime                         |
//+------------------------------------------------------------------+
datetime ConvertStructToTime(MqlDateTime &dt)
{
   return StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, dt.hour, dt.min));
}

