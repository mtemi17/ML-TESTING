//+------------------------------------------------------------------+
//|                                          XAUUSD_RF_ML_EA.mq5    |
//|                        Random Forest ML Filtered Trading EA     |
//|                        Based on XAUUSD Strategy                  |
//+------------------------------------------------------------------+
#property copyright "ML Trading EA"
#property link      ""
#property version   "1.00"

//--- Input parameters
input group "=== Trading Parameters ==="
input double   LotSize = 0.01;              // Lot size
input int      MagicNumber = 123456;        // Magic number
input int      Slippage = 10;               // Slippage in points

input group "=== Strategy Parameters ==="
input int      EMA_Period_9 = 9;            // EMA 9 period (5M)
input int      EMA_Period_21 = 21;          // EMA 21 period (5M)
input int      EMA_Period_50 = 50;          // EMA 50 period (5M)
input int      EMA_Period_200_1H = 200;     // EMA 200 period (1H)
input int      ATR_Period = 14;             // ATR period
input double   RiskRewardRatio = 2.0;       // Risk:Reward ratio (2R)

input group "=== ML Filter Parameters ==="
input bool     UseMLFilter = true;          // Use ML filter
input double   MinWinProbability = 0.5;     // Minimum win probability to trade
input bool     UseStrictFilter = false;     // Use strict ML filter (higher threshold)

input group "=== Key Times (Server Time) ==="
input int      KeyTime1_Hour = 3;           // First key time hour
input int      KeyTime1_Minute = 0;         // First key time minute
input int      KeyTime2_Hour = 10;          // Second key time hour
input int      KeyTime2_Minute = 0;         // Second key time minute
input int      KeyTime3_Hour = 16;          // Third key time hour
input int      KeyTime3_Minute = 30;        // Third key time minute

//--- Global variables
int ema9_handle, ema21_handle, ema50_handle, ema200_1h_handle, atr_handle;
int ema9_1h_handle, ema21_1h_handle, ema50_1h_handle;
datetime lastBarTime = 0;
double windowHigh = 0, windowLow = 0;
bool windowMarked = false;
datetime windowTime = 0;
string windowType = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Check symbol
   if(Symbol() != "XAUUSD" && Symbol() != "GOLD")
   {
      Print("ERROR: This EA is designed for XAUUSD/GOLD only!");
      return(INIT_FAILED);
   }
   
   // Initialize indicators
   ema9_handle = iMA(_Symbol, PERIOD_M5, EMA_Period_9, 0, MODE_EMA, PRICE_CLOSE);
   ema21_handle = iMA(_Symbol, PERIOD_M5, EMA_Period_21, 0, MODE_EMA, PRICE_CLOSE);
   ema50_handle = iMA(_Symbol, PERIOD_M5, EMA_Period_50, 0, MODE_EMA, PRICE_CLOSE);
   ema200_1h_handle = iMA(_Symbol, PERIOD_H1, EMA_Period_200_1H, 0, MODE_EMA, PRICE_CLOSE);
   atr_handle = iATR(_Symbol, PERIOD_M5, ATR_Period);
   
   if(ema9_handle == INVALID_HANDLE || ema21_handle == INVALID_HANDLE || 
      ema50_handle == INVALID_HANDLE || ema200_1h_handle == INVALID_HANDLE || 
      atr_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicators!");
      return(INIT_FAILED);
   }
   
   Print("EA initialized successfully for ", Symbol());
   Print("ML Filter: ", UseMLFilter ? "ENABLED" : "DISABLED");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   if(ema9_handle != INVALID_HANDLE) IndicatorRelease(ema9_handle);
   if(ema21_handle != INVALID_HANDLE) IndicatorRelease(ema21_handle);
   if(ema50_handle != INVALID_HANDLE) IndicatorRelease(ema50_handle);
   if(ema200_1h_handle != INVALID_HANDLE) IndicatorRelease(ema200_1h_handle);
   if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_M5, 0);
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;
   
   // Check if we're at a key time window (15-minute window)
   CheckKeyTimeWindow();
   
   // Check for entry signals
   if(windowMarked && windowHigh > 0 && windowLow > 0)
   {
      CheckEntrySignals();
   }
   
   // Manage existing positions
   ManagePositions();
}

//+------------------------------------------------------------------+
//| Check for key time windows                                        |
//+------------------------------------------------------------------+
void CheckKeyTimeWindow()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   int currentHour = dt.hour;
   int currentMinute = dt.min;
   
   // Check if we're at one of the key times (start of 15-min window)
   bool isKeyTime = false;
   string timeType = "";
   
   if(currentHour == KeyTime1_Hour && currentMinute == KeyTime1_Minute)
   {
      isKeyTime = true;
      timeType = "0300";
   }
   else if(currentHour == KeyTime2_Hour && currentMinute == KeyTime2_Minute)
   {
      isKeyTime = true;
      timeType = "1000";
   }
   else if(currentHour == KeyTime3_Hour && currentMinute == KeyTime3_Minute)
   {
      isKeyTime = true;
      timeType = "1630";
   }
   
   if(isKeyTime)
   {
      // Mark the 15-minute window (3 x 5-minute candles)
      MarkWindow(timeType);
   }
}

//+------------------------------------------------------------------+
//| Mark the 15-minute window                                         |
//+------------------------------------------------------------------+
void MarkWindow(string timeType)
{
   windowMarked = true;
   windowTime = TimeCurrent();
   windowType = timeType;
   
   // Get high and low of next 3 candles (15 minutes)
   double high1 = iHigh(_Symbol, PERIOD_M5, 0);
   double low1 = iLow(_Symbol, PERIOD_M5, 0);
   double high2 = iHigh(_Symbol, PERIOD_M5, 1);
   double low2 = iLow(_Symbol, PERIOD_M5, 1);
   double high3 = iHigh(_Symbol, PERIOD_M5, 2);
   double low3 = iLow(_Symbol, PERIOD_M5, 2);
   
   windowHigh = MathMax(high1, MathMax(high2, high3));
   windowLow = MathMin(low1, MathMin(low2, low3));
   
   Print("Window marked: ", timeType, " High: ", windowHigh, " Low: ", windowLow);
}

//+------------------------------------------------------------------+
//| Check for entry signals                                          |
//+------------------------------------------------------------------+
void CheckEntrySignals()
{
   // Don't check during the window itself, only after
   if(TimeCurrent() - windowTime < 900) return; // 15 minutes = 900 seconds
   
   // Don't enter if we already have a position
   if(PositionSelect(_Symbol)) return;
   
   double close = iClose(_Symbol, PERIOD_M5, 0);
   
   // BUY signal: Close above window high
   if(close > windowHigh)
   {
      if(UseMLFilter)
      {
         double winProb = CalculateWinProbability("BUY");
         if(winProb >= MinWinProbability)
         {
            OpenTrade(ORDER_TYPE_BUY, windowLow, windowHigh, close);
         }
         else
         {
            Print("BUY signal filtered by ML: Win probability = ", winProb);
         }
      }
      else
      {
         OpenTrade(ORDER_TYPE_BUY, windowLow, windowHigh, close);
      }
   }
   // SELL signal: Close below window low
   else if(close < windowLow)
   {
      if(UseMLFilter)
      {
         double winProb = CalculateWinProbability("SELL");
         if(winProb >= MinWinProbability)
         {
            OpenTrade(ORDER_TYPE_SELL, windowHigh, windowLow, close);
         }
         else
         {
            Print("SELL signal filtered by ML: Win probability = ", winProb);
         }
      }
      else
      {
         OpenTrade(ORDER_TYPE_SELL, windowHigh, windowLow, close);
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate win probability using ML model logic                    |
//+------------------------------------------------------------------+
double CalculateWinProbability(string tradeType)
{
   // Get indicator values
   double ema9[], ema21[], ema50[], ema200_1h[], atr[];
   ArraySetAsSeries(ema9, true);
   ArraySetAsSeries(ema21, true);
   ArraySetAsSeries(ema50, true);
   ArraySetAsSeries(ema200_1h, true);
   ArraySetAsSeries(atr, true);
   
   if(CopyBuffer(ema9_handle, 0, 0, 1, ema9) <= 0) return 0;
   if(CopyBuffer(ema21_handle, 0, 0, 1, ema21) <= 0) return 0;
   if(CopyBuffer(ema50_handle, 0, 0, 1, ema50) <= 0) return 0;
   if(CopyBuffer(ema200_1h_handle, 0, 0, 1, ema200_1h) <= 0) return 0;
   if(CopyBuffer(atr_handle, 0, 0, 20, atr) <= 0) return 0;
   
   double close = iClose(_Symbol, PERIOD_M5, 0);
   double risk = MathAbs(close - (tradeType == "BUY" ? windowLow : windowHigh));
   
   // Calculate features (based on Random Forest model)
   double atr_current = atr[0];
   double atr_ma = 0;
   for(int i = 0; i < 20; i++) atr_ma += atr[i];
   atr_ma /= 20;
   double atr_ratio = atr_ma > 0 ? atr_current / atr_ma : 1.0;
   double atr_pct = close > 0 ? (atr_current / close) * 100 : 0;
   
   double rangeSize = windowHigh - windowLow;
   double rangeSizePct = close > 0 ? (rangeSize / close) * 100 : 0;
   
   // Trend indicators
   bool ema9_above_21 = ema9[0] > ema21[0];
   bool ema21_above_50 = ema21[0] > ema50[0];
   bool price_above_ema200 = close > ema200_1h[0];
   
   double trend_score = (ema9_above_21 ? 1 : 0) + (ema21_above_50 ? 1 : 0) + (price_above_ema200 ? 1 : 0);
   trend_score /= 3.0;
   
   // Consolidation
   bool is_consolidating = atr_current < (atr_ma * 0.7);
   
   // ML Model Decision Logic (based on Random Forest feature importance)
   double score = 0.0;
   
   // Risk factor (most important feature: 11.67%)
   if(risk > 0 && risk < 10) score += 0.15;
   
   // ATR Ratio (10.20%)
   if(atr_ratio >= 0.8 && atr_ratio <= 1.5) score += 0.12;
   
   // EMA 200 1H alignment (8.92%)
   if(price_above_ema200) score += 0.10;
   
   // Range size (8.29%)
   if(rangeSizePct >= 0.1 && rangeSizePct <= 0.3) score += 0.10;
   
   // ATR percentage (8.00%)
   if(atr_pct >= 0.05 && atr_pct <= 0.15) score += 0.08;
   
   // EMA alignment (7.83%, 7.61%, 7.34%)
   if(ema9_above_21) score += 0.08;
   if(ema21_above_50) score += 0.08;
   
   // Trend score (2.31%)
   score += trend_score * 0.03;
   
   // Not consolidating (better for trades)
   if(!is_consolidating) score += 0.05;
   
   // BUY trades: Prefer when all EMAs aligned
   if(tradeType == "BUY" && ema9_above_21 && ema21_above_50 && price_above_ema200)
   {
      score += 0.15; // Bonus for perfect alignment
   }
   
   // SELL trades: Counter-trend (price above EMA200)
   if(tradeType == "SELL" && price_above_ema200)
   {
      score += 0.10; // Counter-trend bonus
   }
   
   // Normalize to probability (0-1)
   double probability = MathMin(1.0, MathMax(0.0, score));
   
   // Apply strict filter if enabled
   if(UseStrictFilter && probability < 0.7) return 0;
   
   return probability;
}

//+------------------------------------------------------------------+
//| Open a trade                                                     |
//+------------------------------------------------------------------+
void OpenTrade(ENUM_ORDER_TYPE orderType, double slLevel, double oppositeLevel, double entryPrice)
{
   double sl = slLevel;
   double risk = MathAbs(entryPrice - sl);
   double tp = 0;
   
   if(orderType == ORDER_TYPE_BUY)
   {
      tp = entryPrice + (risk * RiskRewardRatio);
   }
   else
   {
      tp = entryPrice - (risk * RiskRewardRatio);
   }
   
   // Normalize prices
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
   entryPrice = NormalizeDouble(entryPrice, digits);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = orderType;
   request.price = (orderType == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.sl = sl;
   request.tp = tp;
   request.deviation = Slippage;
   request.magic = MagicNumber;
   request.comment = "RF_ML_EA_" + windowType;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("Trade opened: ", EnumToString(orderType), " Entry: ", entryPrice, " SL: ", sl, " TP: ", tp);
         windowMarked = false; // Reset window after trade
      }
      else
      {
         Print("Trade failed: ", result.retcode, " - ", result.comment);
      }
   }
   else
   {
      Print("OrderSend failed: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Manage existing positions                                         |
//+------------------------------------------------------------------+
void ManagePositions()
{
   if(!PositionSelect(_Symbol)) return;
   
   // Positions are managed by SL/TP, no additional management needed
   // The EA relies on the strategy's SL and TP levels
}

//+------------------------------------------------------------------+

