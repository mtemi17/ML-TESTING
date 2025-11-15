#property strict

input double   InpRiskPercent              = 1.0;      // Risk % per trade
input double   InpRewardToRisk             = 2.0;      // Reward-to-risk multiple
input bool     InpUseAiFilter              = true;     // Enable AI gating
input double   InpProbabilityThreshold     = 0.50;     // AI minimum probability
input string   InpAiEndpoint               = "http://127.0.0.1:8001/predict";
input int      InpMaxTradesPerWindow       = 1;        // Max breakout trades per window
input double   InpBreakoutInitialStopRatio = 0.6;      // SL = range risk * ratio
input double   InpBreakoutMaxMaeRatio      = 0.6;      // Max MAE as fraction of risk
input int      InpBreakoutMomentumBar      = 3;        // Bars to show momentum
input double   InpBreakoutMomentumMinGain  = 0.3;      // Min gain (in R) by momentum bar
input double   InpMaxBreakoutAtrMultiple   = 1.8;      // Max distance from range edge in ATR multiples
input double   InpMaxAtrRatio              = 1.3;      // Max ATR / ATR average
input double   InpMinTrendScore            = 0.66;     // Minimum trend alignment score
input double   InpMaxConsolidationScore    = 0.10;     // Maximum consolidation score
input double   InpMinEntryOffsetRatio      = -0.25;    // Min normalized offset from range mid
input double   InpMaxEntryOffsetRatio      = 1.00;     // Max normalized offset from range mid
input double   InpFirstBarMinGain          = -0.20;    // First bar minimum gain (R)
input double   InpMaxRetestDepthR          = 1.80;     // Max retest depth into range (R)
input int      InpMaxRetestBars            = 12;       // Max bars to enforce retest rule

#define WINDOW_COUNT 3
string g_windowTimes[WINDOW_COUNT] = {"03:00", "10:00", "16:30"};

int g_ma5Handle  = INVALID_HANDLE;
int g_ma1hHandle = INVALID_HANDLE;
int g_ma9Handle  = INVALID_HANDLE;
int g_ma21Handle = INVALID_HANDLE;
int g_ma50Handle = INVALID_HANDLE;
int g_atrHandle  = INVALID_HANDLE;

datetime g_lastWindowTrade[WINDOW_COUNT];

struct WindowRange
{
   datetime start;
   datetime end;
  double   high;
   double   low;
   string   id;
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
   double             atr_entry;
   double             atr_average;
   double             breakout_distance;
   double             entry_offset_ratio;
   double             trend_score;
   double             consolidation_score;
   double             max_retest_depth;
   bool               first_bar_checked;
};

BreakoutState g_breakout;

bool   AttemptBreakout(const WindowRange &range, double price, bool isLong);
void   ManageBreakoutPosition();
void   CloseBreakoutPosition();
bool   GetWindowRange(int index, WindowRange &range);
int    CountTradesForWindow(const string &windowId);
double CalculatePositionSize(double entryPrice, double stopPrice);
bool   SendMarketOrder(ENUM_ORDER_TYPE type, double volume, double entryPrice,
                      double stopPrice, double targetPrice, const string &tag,
                      const string &windowId);
bool   QueryAiProbability(const WindowRange &range, const string &entryType,
                          double entryPrice, double stopPrice, double targetPrice,
                          double &probability);
string BuildFeatureJson(const WindowRange &range, const string &entryType,
                        double entryPrice, double stopPrice, double targetPrice);
double ParseProbability(const string &json);
double GetMAValue(int handle, int shift);
double GetATRValue(int shift);
double GetATRAverage(int startShift, int count);

datetime DateOfDay(datetime dt)
{
   MqlDateTime tm;
   TimeToStruct(dt, tm);
   tm.hour = 0;
   tm.min  = 0;
   tm.sec  = 0;
   return StructToTime(tm);
}

int OnInit()
{
   for(int i=0; i<WINDOW_COUNT; i++)
      g_lastWindowTrade[i] = 0;

   if(Period() != PERIOD_M5)
   {
      Print("Please attach the EA to a 5-minute chart.");
      return(INIT_PARAMETERS_INCORRECT);
   }

   g_ma5Handle  = iMA(_Symbol, PERIOD_M5, 200, 0, MODE_EMA, PRICE_CLOSE);
   g_ma1hHandle = iMA(_Symbol, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE);
   g_ma9Handle  = iMA(_Symbol, PERIOD_M5, 9,   0, MODE_EMA, PRICE_CLOSE);
   g_ma21Handle = iMA(_Symbol, PERIOD_M5, 21,  0, MODE_EMA, PRICE_CLOSE);
   g_ma50Handle = iMA(_Symbol, PERIOD_M5, 50,  0, MODE_EMA, PRICE_CLOSE);
   g_atrHandle  = iATR(_Symbol, PERIOD_M5, 14);

   if(g_ma5Handle  == INVALID_HANDLE ||
      g_ma1hHandle == INVALID_HANDLE ||
      g_ma9Handle  == INVALID_HANDLE ||
      g_ma21Handle == INVALID_HANDLE ||
      g_ma50Handle == INVALID_HANDLE ||
      g_atrHandle  == INVALID_HANDLE)
   {
      Print("Failed to create indicator handles.");
      return(INIT_FAILED);
   }

   ZeroMemory(g_breakout);
   g_breakout.active = false;
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(g_ma5Handle  != INVALID_HANDLE) IndicatorRelease(g_ma5Handle);
   if(g_ma1hHandle != INVALID_HANDLE) IndicatorRelease(g_ma1hHandle);
   if(g_ma9Handle  != INVALID_HANDLE) IndicatorRelease(g_ma9Handle);
   if(g_ma21Handle != INVALID_HANDLE) IndicatorRelease(g_ma21Handle);
   if(g_ma50Handle != INVALID_HANDLE) IndicatorRelease(g_ma50Handle);
   if(g_atrHandle  != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
}

void OnTick()
{
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, PERIOD_M5, 0);
   if(currentBarTime == lastBarTime)
      return;
   lastBarTime = currentBarTime;

   ManageBreakoutPosition();

   for(int w=0; w<WINDOW_COUNT; w++)
   {
      WindowRange range;
      if(!GetWindowRange(w, range))
         continue;

      if(g_breakout.active && g_breakout.window_id == range.id)
         continue;

      if(CountTradesForWindow(range.id) >= InpMaxTradesPerWindow)
         continue;

      datetime now = TimeCurrent();
      if(now < range.start || now > range.end)
         continue;

      double close = iClose(_Symbol, PERIOD_M5, 1);
      double rangeWidth = range.high - range.low;
      if(rangeWidth <= 0)
         continue;

      bool breakoutUp   = (close > range.high);
      bool breakoutDown = (close < range.low);

      if(breakoutUp)
      {
         if(AttemptBreakout(range, close, true))
            g_lastWindowTrade[w] = now;
      }
      else if(breakoutDown)
      {
         if(AttemptBreakout(range, close, false))
            g_lastWindowTrade[w] = now;
      }
   }
}

bool AttemptBreakout(const WindowRange &range, double price, bool isLong)
{
   double rawRisk = isLong ? price - range.low : range.high - price;
   if(rawRisk <= 0.0)
      return false;

   double risk      = rawRisk * InpBreakoutInitialStopRatio;
   double stopPrice = isLong ? price - risk : price + risk;
   double target    = isLong ? price + InpRewardToRisk * risk
                             : price - InpRewardToRisk * risk;

   string entryTag = isLong ? "BREAKOUT_LONG" : "BREAKOUT_SHORT";

   double atr     = GetATRValue(1);
   double atrAvg  = GetATRAverage(1, 20);
   double atrRatio = (atrAvg > 0.0) ? atr / atrAvg : 0.0;
   double breakoutDistance = isLong ? MathMax(price - range.high, 0.0)
                                    : MathMax(range.low - price, 0.0);
   double breakoutAtrMultiple = (atr > 0.0) ? breakoutDistance / atr : 0.0;

   if(InpMaxBreakoutAtrMultiple > 0.0 && breakoutAtrMultiple > InpMaxBreakoutAtrMultiple)
   {
      PrintFormat("Breakout rejected: distance %.2f ATR exceeds limit %.2f.",
                  breakoutAtrMultiple, InpMaxBreakoutAtrMultiple);
      return false;
   }

   if(InpMaxAtrRatio > 0.0 && atrRatio > InpMaxAtrRatio)
   {
      PrintFormat("Breakout rejected: ATR ratio %.2f above limit %.2f.",
                  atrRatio, InpMaxAtrRatio);
      return false;
   }

   double ema9       = GetMAValue(g_ma9Handle, 1);
   double ema21      = GetMAValue(g_ma21Handle, 1);
   double ema50      = GetMAValue(g_ma50Handle, 1);
   double ema200_1h  = GetMAValue(g_ma1hHandle, 0);
   int ema9Above21   = (ema9 > ema21) ? 1 : 0;
   int ema21Above50  = (ema21 > ema50) ? 1 : 0;
   int priceAbove200 = (price > ema200_1h) ? 1 : 0;
   double trendScore = (ema9Above21 + ema21Above50 + priceAbove200) / 3.0;

   if(trendScore < InpMinTrendScore)
   {
      PrintFormat("Breakout rejected: trend score %.2f below %.2f.",
                  trendScore, InpMinTrendScore);
      return false;
   }

   double rangeWidth = range.high - range.low;
   double consolidationScore = 0.0;
   if(rangeWidth > 0.0 && atr > 0.0)
   {
      int isConsolidating = (atrRatio < 0.8) ? 1 : 0;
      int isTightRange    = (rangeWidth < atr) ? 1 : 0;
      consolidationScore  = (isConsolidating + isTightRange) / 2.0;
   }

   if(consolidationScore > InpMaxConsolidationScore)
   {
      PrintFormat("Breakout rejected: consolidation score %.2f above %.2f.",
                  consolidationScore, InpMaxConsolidationScore);
      return false;
   }

   double rangeMid   = (range.high + range.low) / 2.0;
   double entryOffset = price - rangeMid;
   double offsetRatio = (rangeWidth != 0.0) ? entryOffset / rangeWidth : 0.0;

   if(rangeWidth > 0.0)
   {
      if(offsetRatio < InpMinEntryOffsetRatio || offsetRatio > InpMaxEntryOffsetRatio)
      {
         PrintFormat("Breakout rejected: entry offset ratio %.2f outside [%.2f, %.2f].",
                     offsetRatio, InpMinEntryOffsetRatio, InpMaxEntryOffsetRatio);
         return false;
      }
   }

   if(InpUseAiFilter)
   {
      double probability = 0.0;
      if(!QueryAiProbability(range, entryTag, price, stopPrice, target, probability))
         return false;
      if(probability < InpProbabilityThreshold)
      {
         PrintFormat("AI probability %.2f below threshold %.2f, skipping breakout.",
                     probability, InpProbabilityThreshold);
         return false;
      }
   }

   double lots = CalculatePositionSize(price, stopPrice);
   if(lots <= 0.0)
      return false;

   ENUM_ORDER_TYPE orderType = isLong ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(!SendMarketOrder(orderType, lots, price, stopPrice, target, entryTag, range.id))
      return false;

   ZeroMemory(g_breakout);
   g_breakout.active              = true;
   g_breakout.entry_time          = iTime(_Symbol, PERIOD_M5, 0);
   g_breakout.entry_price         = price;
   g_breakout.stop_price          = stopPrice;
   g_breakout.target_price        = target;
   g_breakout.risk                = MathAbs(price - stopPrice);
   g_breakout.bars_in_trade       = 0;
   g_breakout.max_mae             = 0.0;
   g_breakout.momentum_satisfied  = false;
   g_breakout.direction           = isLong ? POSITION_TYPE_BUY : POSITION_TYPE_SELL;
   g_breakout.window_id           = range.id;
   g_breakout.range_high          = range.high;
   g_breakout.range_low           = range.low;
   g_breakout.atr_entry           = atr;
   g_breakout.atr_average         = atrAvg;
   g_breakout.breakout_distance   = breakoutDistance;
   g_breakout.entry_offset_ratio  = offsetRatio;
   g_breakout.trend_score         = trendScore;
   g_breakout.consolidation_score = consolidationScore;
   g_breakout.max_retest_depth    = 0.0;
   g_breakout.first_bar_checked   = false;
   Print(isLong ? "Opened breakout long." : "Opened breakout short.");
   return true;
}

void ManageBreakoutPosition()
{
   if(!g_breakout.active)
      return;

   if(!PositionSelect(_Symbol))
   {
      g_breakout.active = false;
      return;
   }

   string comment = PositionGetString(POSITION_COMMENT);
   if(StringFind(comment, g_breakout.window_id) < 0)
   {
      g_breakout.active = false;
      return;
   }

   ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   double priceOpen  = PositionGetDouble(POSITION_PRICE_OPEN);
   double stopLoss   = PositionGetDouble(POSITION_SL);
   double takeProfit = PositionGetDouble(POSITION_TP);
   double currentPrice = (type == POSITION_TYPE_BUY)
                         ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                         : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   g_breakout.bars_in_trade++;

   double adverse = 0.0;
   double gain    = 0.0;
   if(type == POSITION_TYPE_BUY)
   {
      adverse = priceOpen - iLow(_Symbol, PERIOD_M5, 1);
      gain    = currentPrice - priceOpen;
   }
   else
   {
      adverse = iHigh(_Symbol, PERIOD_M5, 1) - priceOpen;
      gain    = priceOpen - currentPrice;
   }

   g_breakout.max_mae = MathMax(g_breakout.max_mae, adverse);
   double mae_ratio   = (g_breakout.risk > 0.0) ? g_breakout.max_mae / g_breakout.risk : 0.0;
   if(mae_ratio > InpBreakoutMaxMaeRatio)
   {
      CloseBreakoutPosition();
      PrintFormat("Breakout closed: MAE ratio %.2f exceeded %.2f.", mae_ratio, InpBreakoutMaxMaeRatio);
      g_breakout.active = false;
      return;
   }

   if(!g_breakout.first_bar_checked && g_breakout.bars_in_trade >= 2)
   {
      double firstClose = iClose(_Symbol, PERIOD_M5, 1);
      double firstGain  = (type == POSITION_TYPE_BUY) ? (firstClose - priceOpen)
                                                      : (priceOpen - firstClose);
      double firstGainR = (g_breakout.risk > 0.0) ? firstGain / g_breakout.risk : 0.0;
      if(firstGainR < InpFirstBarMinGain)
      {
         CloseBreakoutPosition();
         PrintFormat("Breakout closed: first-bar gain %.2fR below %.2fR.",
                     firstGainR, InpFirstBarMinGain);
         g_breakout.active = false;
         return;
      }
      g_breakout.first_bar_checked = true;
   }

   double retestDepth = 0.0;
   if(type == POSITION_TYPE_BUY)
   {
      double barLow0 = iLow(_Symbol, PERIOD_M5, 0);
      double barLow1 = iLow(_Symbol, PERIOD_M5, 1);
      double minPrice = MathMin(MathMin(currentPrice, barLow0), barLow1);
      retestDepth = MathMax(g_breakout.range_high - minPrice, 0.0);
   }
   else
   {
      double barHigh0 = iHigh(_Symbol, PERIOD_M5, 0);
      double barHigh1 = iHigh(_Symbol, PERIOD_M5, 1);
      double maxPrice = MathMax(MathMax(currentPrice, barHigh0), barHigh1);
      retestDepth = MathMax(maxPrice - g_breakout.range_low, 0.0);
   }

   double retestDepthR = (g_breakout.risk > 0.0) ? retestDepth / g_breakout.risk : 0.0;
   g_breakout.max_retest_depth = MathMax(g_breakout.max_retest_depth, retestDepthR);
   if(g_breakout.bars_in_trade <= InpMaxRetestBars && g_breakout.max_retest_depth > InpMaxRetestDepthR)
   {
      CloseBreakoutPosition();
      PrintFormat("Breakout closed: retest depth %.2fR exceeded %.2fR within %d bars.",
                  g_breakout.max_retest_depth, InpMaxRetestDepthR, InpMaxRetestBars);
      g_breakout.active = false;
      return;
   }

   double gain_ratio = (g_breakout.risk > 0.0) ? gain / g_breakout.risk : 0.0;
   if(!g_breakout.momentum_satisfied)
   {
      if(gain_ratio >= InpBreakoutMomentumMinGain)
         g_breakout.momentum_satisfied = true;
      else if(g_breakout.bars_in_trade >= InpBreakoutMomentumBar)
      {
         CloseBreakoutPosition();
         Print("Breakout closed: momentum not achieved.");
         g_breakout.active = false;
         return;
      }
   }

   if((type == POSITION_TYPE_BUY && currentPrice >= takeProfit) ||
      (type == POSITION_TYPE_SELL && currentPrice <= takeProfit))
   {
      Print("Breakout TP reached.");
      g_breakout.active = false;
      return;
   }

   if((type == POSITION_TYPE_BUY && currentPrice <= stopLoss) ||
      (type == POSITION_TYPE_SELL && currentPrice >= stopLoss))
   {
      Print("Breakout SL hit.");
      g_breakout.active = false;
      return;
   }
}

void CloseBreakoutPosition()
{
   if(!PositionSelect(_Symbol))
      return;

   string comment = PositionGetString(POSITION_COMMENT);
   if(StringFind(comment, g_breakout.window_id) < 0)
      return;

   ulong ticket = PositionGetInteger(POSITION_TICKET);
   double volume = PositionGetDouble(POSITION_VOLUME);
   ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   double price = (type == POSITION_TYPE_BUY)
                  ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                  : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action   = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol   = _Symbol;
   request.volume   = volume;
   request.type     = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.price    = price;
   request.deviation= 50;
   request.magic    = 123456;

   if(!OrderSend(request, result))
   {
      PrintFormat("OrderSend close failed: %d", GetLastError());
   }
   else if(result.retcode != TRADE_RETCODE_DONE)
   {
      PrintFormat("OrderSend close retcode: %d", result.retcode);
   }
}

bool GetWindowRange(int index, WindowRange &range)
{
   datetime today = DateOfDay(iTime(_Symbol, PERIOD_M5, 0));
   string timeStr = g_windowTimes[index];
   string dateStr = TimeToString(today, TIME_DATE);
   datetime windowStart = StringToTime(dateStr + " " + timeStr);
   datetime windowEnd   = windowStart + 15*60 - 1;
   datetime tradeEnd    = windowStart + 3*60*60;

   int startIndex = iBarShift(_Symbol, PERIOD_M5, windowStart, true);
   if(startIndex < 0)
      return false;

   int endIndex = iBarShift(_Symbol, PERIOD_M5, windowEnd, true);
   if(endIndex < 0)
      endIndex = 0;

   double highest = -DBL_MAX;
   double lowest  =  DBL_MAX;
   for(int i=endIndex; i<=startIndex; i++)
   {
      highest = MathMax(highest, iHigh(_Symbol, PERIOD_M5, i));
      lowest  = MathMin(lowest,  iLow (_Symbol, PERIOD_M5, i));
   }

   range.start = windowStart + 15*60;
   range.end   = tradeEnd;
   range.high  = highest;
   range.low   = lowest;
   range.id    = StringFormat("%s_%s", dateStr, timeStr);
   return true;
}

int CountTradesForWindow(const string &windowId)
{
   int total = 0;
   datetime fromTime = TimeCurrent() - 86400 * 5;
   HistorySelect(fromTime, TimeCurrent());
   uint deals = HistoryDealsTotal();
   for(uint i=0; i<deals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      string comment = HistoryDealGetString(ticket, DEAL_COMMENT);
      if(StringFind(comment, windowId) >= 0)
         total++;
   }
   return total;
}

double CalculatePositionSize(double entryPrice, double stopPrice)
{
   double riskMoney = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;
   double stopDistance = MathAbs(entryPrice - stopPrice);
   if(stopDistance <= 0.0)
      return 0.0;

   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tickSize <= 0.0 || tickValue <= 0.0)
      return 0.0;

   double valuePerLot = stopDistance / tickSize * tickValue;
   if(valuePerLot <= 0.0)
      return 0.0;

   double lots = riskMoney / valuePerLot;
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(lots, minLot);
   lots = MathMin(lots, maxLot);
   return lots;
}

bool SendMarketOrder(ENUM_ORDER_TYPE type, double volume, double entryPrice,
                     double stopPrice, double targetPrice, const string &tag,
                     const string &windowId)
{
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = _Symbol;
   request.volume   = volume;
   request.deviation= 50;
   request.comment  = StringFormat("%s_%s", tag, windowId);
   request.magic    = 123456;
   request.type     = type;
   request.price    = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                               : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.sl       = stopPrice;
   request.tp       = targetPrice;

   if(!OrderSend(request, result))
   {
      PrintFormat("OrderSend failed: %d", GetLastError());
      return false;
   }

   if(result.retcode != TRADE_RETCODE_DONE)
   {
      PrintFormat("OrderSend retcode: %d", result.retcode);
      return false;
   }

   return true;
}

bool QueryAiProbability(const WindowRange &range, const string &entryType,
                        double entryPrice, double stopPrice, double targetPrice,
                        double &probability)
{
   string json = BuildFeatureJson(range, entryType, entryPrice, stopPrice, targetPrice);
   uchar post[];
   int len = StringLen(json);
   ArrayResize(post, len);
   for(int i=0; i<len; i++)
      post[i] = (uchar)StringGetCharacter(json, i);

   uchar answer[];
   string responseHeaders;
   string headers = "Content-Type: application/json\r\n";
   ResetLastError();
   int status = WebRequest("POST", InpAiEndpoint, headers, 5000, post, answer, responseHeaders);
   if(status == -1)
   {
      PrintFormat("WebRequest failed: %d", GetLastError());
      return false;
   }
   if(status != 200)
   {
      PrintFormat("AI service HTTP status %d", status);
      return false;
   }

   char responseChars[];
   ArrayResize(responseChars, ArraySize(answer));
   for(int i=0; i<ArraySize(answer); i++)
      responseChars[i] = (char)answer[i];
   string response = CharArrayToString(responseChars, 0, ArraySize(responseChars));
   double value = ParseProbability(response);
   if(value < 0)
   {
      Print("Invalid AI response.");
      return false;
   }

   probability = value;
   return true;
}

string BuildFeatureJson(const WindowRange &range, const string &entryType,
                        double entryPrice, double stopPrice, double targetPrice)
{
   double risk = MathAbs(entryPrice - stopPrice);
   string direction = (entryType == "BREAKOUT_LONG") ? "BUY" : "SELL";

   double closePrice = iClose(_Symbol, PERIOD_M5, 1);
   double openPrice  = iOpen (_Symbol, PERIOD_M5, 1);
   double highPrice  = iHigh (_Symbol, PERIOD_M5, 1);
   double lowPrice   = iLow  (_Symbol, PERIOD_M5, 1);

   double rangeWidth = range.high - range.low;
   double rangeMid   = (range.high + range.low) / 2.0;
   double entryOffset = entryPrice - rangeMid;

   double breakoutDistance = (direction == "BUY")
                             ? MathMax(closePrice - range.high, 0.0)
                             : MathMax(range.low - closePrice, 0.0);

   double candleRange = highPrice - lowPrice;
   double candleBody  = MathAbs(closePrice - openPrice);
   double breakoutBodyPct = (candleRange > 0.0) ? candleBody / candleRange : 0.0;

   double atr = GetATRValue(1);
   double atrAvg = GetATRAverage(1, 20);
   double atrRatio = (atrAvg > 0.0) ? atr / atrAvg : 0.0;
   double atrPct   = (closePrice != 0.0) ? atr / closePrice * 100.0 : 0.0;
   double rangeAtrRatio = (atr > 0.0) ? rangeWidth / atr : 0.0;
   double breakoutAtrMultiple = (atr > 0.0) ? breakoutDistance / atr : 0.0;

   double ema9   = GetMAValue(g_ma9Handle, 1);
   double ema21  = GetMAValue(g_ma21Handle, 1);
   double ema50  = GetMAValue(g_ma50Handle, 1);
   double ema200 = GetMAValue(g_ma5Handle, 1);
   double ema200_1h = GetMAValue(g_ma1hHandle, 0);

   int priceAbove200_5  = (closePrice > ema200) ? 1 : 0;
   int priceAbove200_1h = (closePrice > ema200_1h) ? 1 : 0;

   int ema9Above21  = (ema9 > ema21) ? 1 : 0;
   int ema21Above50 = (ema21 > ema50) ? 1 : 0;
   double trendScore = (ema9Above21 + ema21Above50 + priceAbove200_1h) / 3.0;

   int isConsolidating = (atrRatio < 0.8) ? 1 : 0;
   int isTightRange    = (rangeWidth < atr) ? 1 : 0;
   double consolidationScore = (isConsolidating + isTightRange) / 2.0;

   string windowType = range.id;
   int underscorePos = StringFind(range.id, "_");
   if(underscorePos >= 0)
      windowType = StringSubstr(range.id, underscorePos + 1);

   string payload = "{ \"features\": {";
   payload += StringFormat("\"EntryPrice\":%.6f,", entryPrice);
   payload += StringFormat("\"SL\":%.6f,", stopPrice);
   payload += StringFormat("\"TP\":%.6f,", targetPrice);
   payload += StringFormat("\"Risk\":%.6f,", risk);
   payload += StringFormat("\"BreakoutDistance\":%.6f,", breakoutDistance);
   payload += StringFormat("\"BreakoutBodyPct\":%.6f,", breakoutBodyPct);
   payload += StringFormat("\"BreakoutAtrMultiple\":%.6f,", breakoutAtrMultiple);
   payload += StringFormat("\"RangeWidth\":%.6f,", rangeWidth);
   payload += StringFormat("\"RangeMid\":%.6f,", rangeMid);
   payload += StringFormat("\"EntryOffset\":%.6f,", entryOffset);
   payload += StringFormat("\"RangeAtrRatio\":%.6f,", rangeAtrRatio);
   payload += StringFormat("\"PriceAboveEMA200_5M\":%d,", priceAbove200_5);
   payload += StringFormat("\"ATR_Value\":%.6f,", atr);
   payload += StringFormat("\"EMA_9_5M\":%.6f,", ema9);
   payload += StringFormat("\"EMA_21_5M\":%.6f,", ema21);
   payload += StringFormat("\"EMA_50_5M\":%.6f,", ema50);
   payload += StringFormat("\"EMA_200_5M\":%.6f,", ema200);
   payload += StringFormat("\"EMA_200_1H\":%.6f,", ema200_1h);
   payload += StringFormat("\"ATR\":%.6f,", atr);
   payload += StringFormat("\"ATR_Pct\":%.6f,", atrPct);
   payload += StringFormat("\"ATR_Ratio\":%.6f,", atrRatio);
   payload += StringFormat("\"Consolidation_Score\":%.6f,", consolidationScore);
   payload += StringFormat("\"Trend_Score\":%.6f,", trendScore);
   payload += StringFormat("\"Is_Consolidating\":%d,", isConsolidating);
   payload += StringFormat("\"Is_Tight_Range\":%d,", isTightRange);
   payload += StringFormat("\"Price_Above_EMA200_5M\":%d,", priceAbove200_5);
   payload += StringFormat("\"Price_Above_EMA200_1H\":%d,", priceAbove200_1h);
   payload += StringFormat("\"EntryType\":\"%s\",", entryType);
   payload += StringFormat("\"Type\":\"%s\",", direction);
   payload += StringFormat("\"WindowType\":\"%s\",", windowType);
   payload += StringFormat("\"WindowID\":\"%s\",", range.id);
   payload += StringFormat("\"Mode\":\"%s\",", entryType);
   payload += StringFormat("\"Market\":\"%s\"", _Symbol);
   payload += "} }";
   return payload;
}

double ParseProbability(const string &json)
{
   int pos = StringFind(json, "probability");
   if(pos < 0)
      return -1.0;
   pos = StringFind(json, ":", pos);
   if(pos < 0)
      return -1.0;
   int end = StringFind(json, ",", pos);
   if(end < 0)
      end = StringFind(json, "}", pos);
   if(end < 0)
      end = StringLen(json);
   string value = StringSubstr(json, pos+1, end-pos-1);
   StringReplace(value, " ", "");
   StringReplace(value, "\"", "");
   return StringToDouble(value);
}

double GetMAValue(int handle, int shift)
{
   if(handle == INVALID_HANDLE)
      return 0.0;
   double buffer[];
   if(CopyBuffer(handle, 0, shift, 1, buffer) <= 0)
      return 0.0;
   return buffer[0];
}

double GetATRValue(int shift)
{
   if(g_atrHandle == INVALID_HANDLE)
      return 0.0;
   double buffer[];
   if(CopyBuffer(g_atrHandle, 0, shift, 1, buffer) <= 0)
      return 0.0;
   return buffer[0];
}

double GetATRAverage(int startShift, int count)
{
   if(g_atrHandle == INVALID_HANDLE)
      return 0.0;
   double buffer[];
   if(CopyBuffer(g_atrHandle, 0, startShift, count, buffer) <= 0)
      return GetATRValue(startShift);
   double sum = 0.0;
   int n = ArraySize(buffer);
   for(int i=0; i<n; i++)
      sum += buffer[i];
   return (n > 0) ? sum / n : 0.0;
}

