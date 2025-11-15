//+------------------------------------------------------------------+
//|                                                  EA_CHARGER.mq5 |
//|                    ORB Breakout EA with AI Model-Based Filters   |
//|                        Combines ORB_SmartTrap + Model Learnings  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Enhanced ORB EA"
#property version   "1.00"
#property description "Opening Range Breakout EA with AI Model-Based Filters"

#include <Trade\Trade.mqh>

//--- Enums
enum ENUM_MONEY_MANAGEMENT
{
    RISK_PERCENT,
    FIXED_LOT
};

enum ENUM_SESSION_TYPE
{
    SESSION_TYPE_ASIAN,
    SESSION_TYPE_LONDON,
    SESSION_TYPE_NEW_YORK
};

//--- Session State Structure
struct SessionState
{
    bool     isRangeFound;
    bool     isTradeTaken;
    double   orHigh;
    double   orLow;
    datetime orStartTime;
    datetime orEndTime;
    string   sessionName;
};

//+------------------------------------------------------------------+
//| Expert Advisor Inputs                                            |
//+------------------------------------------------------------------+
input group "=== SESSION SETTINGS ==="
input string            session_OR_Duration   = "00:15";
input string            session_TradeDuration = "03:00";
input bool              session_EnableAsian   = true;
input string            asian_SessionStart    = "03:00";
input bool              session_EnableLondon  = true;
input string            london_SessionStart   = "10:00";
input bool              session_EnableNewYork = true;
input string            ny_SessionStart       = "16:30";

input group "=== TRADE SETTINGS ==="
input bool              enable_MarketBreakout = true;
input ulong             MagicNumber           = 13579;
input string            EaComment             = "EA_CHARGER";
input double            RiskRewardRatio       = 2.0;

input group "=== AI MODEL FILTERS (100% Win Rate Conditions) ==="
input bool              use_ModelFilters      = true;
input double            max_BreakoutAtrMultiple = 0.55;  // Max breakout distance / ATR
input double            max_AtrRatio          = 1.17;    // Max current ATR / avg ATR
input double            min_TrendScore        = 0.67;    // Min trend score (0-1)
input double            max_ConsolidationScore = 0.0;   // Max consolidation (0-1)
input double            min_RangeAtrRatio     = 0.92;   // Min range size / ATR
input double            min_EntryOffsetRatio  = 0.0;    // Min entry offset from range
input double            max_EntryOffsetRatio  = 1.0;     // Max entry offset from range

input group "=== EMA FILTERS ==="
input bool              filter_Enable         = true;
input ENUM_TIMEFRAMES   filter_Timeframe      = PERIOD_H1;
input int               filter_EmaPeriod      = 200;
input bool              filter_EnableM5        = true;

input group "=== RISK MANAGEMENT ==="
input ENUM_MONEY_MANAGEMENT MoneyManagementMode = RISK_PERCENT;
input double            RiskPercentPerTrade   = 1.0;
input double            FixedLotSize          = 0.01;
input double            MaxLotSize            = 10.0;

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CTrade      trade;
datetime    todayDate;
datetime    lastBarTime = 0;

SessionState asianState;
SessionState londonState;
SessionState nyState;

int         ema_handle_filter = INVALID_HANDLE;
int         ema_handle_m5     = INVALID_HANDLE;
int         atr_handle        = INVALID_HANDLE;
int         ema9_handle      = INVALID_HANDLE;
int         ema21_handle      = INVALID_HANDLE;
int         ema50_handle      = INVALID_HANDLE;

bool        isFilterActive    = false;
bool        isFilterM5Active  = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber(MagicNumber);
    trade.SetTypeFillingBySymbol(_Symbol);
    trade.SetDeviationInPoints(5);
    
    todayDate = TimeCurrent();
    lastBarTime = iTime(_Symbol, PERIOD_M5, 0);
    
    // Initialize ATR
    atr_handle = iATR(_Symbol, PERIOD_M5, 14);
    if(atr_handle == INVALID_HANDLE)
    {
        Print("ERROR: Failed to create ATR indicator");
        return INIT_FAILED;
    }
    
    // Initialize EMAs for filters
    if(filter_Enable)
    {
        ema_handle_filter = iMA(_Symbol, filter_Timeframe, filter_EmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
        if(ema_handle_filter != INVALID_HANDLE) 
            isFilterActive = true;
        else 
            Print("Failed to create Trend Filter EMA handle. Filter inactive.");
        
        if(filter_EnableM5)
        {
            ema_handle_m5 = iMA(_Symbol, PERIOD_M5, filter_EmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
            if(ema_handle_m5 != INVALID_HANDLE) 
                isFilterM5Active = true;
            else 
                Print("Failed to create M5 EMA 200 handle. M5 filter inactive.");
        }
    }
    
    // Initialize EMAs for trend score calculation
    ema9_handle = iMA(_Symbol, PERIOD_M5, 9, 0, MODE_EMA, PRICE_CLOSE);
    ema21_handle = iMA(_Symbol, PERIOD_M5, 21, 0, MODE_EMA, PRICE_CLOSE);
    ema50_handle = iMA(_Symbol, PERIOD_M5, 50, 0, MODE_EMA, PRICE_CLOSE);
    
    ResetDailyVariables();
    
    Print("=", StringSubstr("=", 0, 60), "=");
    Print("EA_CHARGER v1.0 Initialized");
    Print("Magic Number: ", MagicNumber);
    Print("Model Filters: ", (use_ModelFilters ? "ENABLED" : "DISABLED"));
    Print("=", StringSubstr("=", 0, 60), "=");
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(ema_handle_filter != INVALID_HANDLE) IndicatorRelease(ema_handle_filter);
    if(ema_handle_m5 != INVALID_HANDLE) IndicatorRelease(ema_handle_m5);
    if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
    if(ema9_handle != INVALID_HANDLE) IndicatorRelease(ema9_handle);
    if(ema21_handle != INVALID_HANDLE) IndicatorRelease(ema21_handle);
    if(ema50_handle != INVALID_HANDLE) IndicatorRelease(ema50_handle);
    
    Print("EA_CHARGER Deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    datetime newBarTime = iTime(_Symbol, PERIOD_M5, 0);
    if(newBarTime > lastBarTime)
    {
        lastBarTime = newBarTime;
        OnNewBar();
    }
}

//+------------------------------------------------------------------+
//| New Bar Event Handler                                            |
//+------------------------------------------------------------------+
void OnNewBar()
{
    MqlDateTime current_dt, today_struct;
    TimeToStruct(TimeCurrent(), current_dt);
    TimeToStruct(todayDate, today_struct);
    
    if(current_dt.day != today_struct.day)
    {
        ResetDailyVariables();
        todayDate = TimeCurrent();
    }
    
    long tradeWindowSeconds = ParseDurationToSeconds(session_TradeDuration, 10800);
    
    if(session_EnableAsian) 
    {
        asianState.sessionName = "ASIAN";
        ProcessSession(asianState, asian_SessionStart, tradeWindowSeconds, SESSION_TYPE_ASIAN);
    }
    if(session_EnableLondon) 
    {
        londonState.sessionName = "LONDON";
        ProcessSession(londonState, london_SessionStart, tradeWindowSeconds, SESSION_TYPE_LONDON);
    }
    if(session_EnableNewYork) 
    {
        nyState.sessionName = "NY";
        ProcessSession(nyState, ny_SessionStart, tradeWindowSeconds, SESSION_TYPE_NEW_YORK);
    }
}

//+------------------------------------------------------------------+
//| Main processing logic for a single session                       |
//+------------------------------------------------------------------+
void ProcessSession(SessionState &state, string startTime, long tradeWindow, ENUM_SESSION_TYPE sessionType)
{
    if(state.isTradeTaken) return;
    
    if(!state.isRangeFound)
    {
        if(FindOpeningRange(startTime, state))
        {
            Print("✅ Range found for ", state.sessionName, " | High: ", state.orHigh, " | Low: ", state.orLow);
        }
    }
    
    if(state.isRangeFound && TimeCurrent() < state.orStartTime + tradeWindow)
    {
        CheckForBreakout(state, sessionType);
    }
}

//+------------------------------------------------------------------+
//| Enhanced breakout detection with model-based filters            |
//+------------------------------------------------------------------+
void CheckForBreakout(SessionState &state, ENUM_SESSION_TYPE sessionType)
{
    if(state.isTradeTaken) return;
    
    MqlRates bar[];
    if(CopyRates(_Symbol, PERIOD_M5, 1, 1, bar) < 1) return;
    
    double openPrice = bar[0].open;
    double closePrice = bar[0].close;
    double highPrice = bar[0].high;
    double lowPrice = bar[0].low;
    
    // Candle confirmation (from ORB_SmartTrap)
    bool isBullishCandle = closePrice > openPrice;
    bool isBearishCandle = closePrice < openPrice;
    
    // BULLISH BREAKOUT: Close above OR high AND bullish candle
    if(closePrice > state.orHigh && isBullishCandle)
    {
        if(enable_MarketBreakout)
        {
            // Apply model-based filters
            if(!use_ModelFilters || PassesModelFilters(true, closePrice, state))
            {
                // Apply EMA trend filter
                if(!filter_Enable || PassesTrendFilter(true, closePrice))
                {
                    ExecuteTrade(ORDER_TYPE_BUY, closePrice, state, sessionType);
                }
                else
                {
                    Print("LONG breakout filtered out by EMA trend filter");
                }
            }
            else
            {
                Print("LONG breakout filtered out by model filters");
            }
        }
    }
    // BEARISH BREAKOUT: Close below OR low AND bearish candle
    else if(closePrice < state.orLow && isBearishCandle)
    {
        if(enable_MarketBreakout)
        {
            // Apply model-based filters
            if(!use_ModelFilters || PassesModelFilters(false, closePrice, state))
            {
                // Apply EMA trend filter
                if(!filter_Enable || PassesTrendFilter(false, closePrice))
                {
                    ExecuteTrade(ORDER_TYPE_SELL, closePrice, state, sessionType);
                }
                else
                {
                    Print("SHORT breakout filtered out by EMA trend filter");
                }
            }
            else
            {
                Print("SHORT breakout filtered out by model filters");
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Model-based filters (100% win rate conditions)                   |
//+------------------------------------------------------------------+
bool PassesModelFilters(bool isLong, double entryPrice, SessionState &state)
{
    // Get ATR values
    double atr[20];
    if(CopyBuffer(atr_handle, 0, 0, 20, atr) < 20) return false;
    
    double currentATR = atr[0];
    double avgATR = 0;
    for(int i = 1; i < 20; i++) avgATR += atr[i];
    avgATR /= 19.0;
    
    if(avgATR <= 0) return false;
    
    // 1. Breakout Distance / ATR Multiple
    double breakoutDistance = 0;
    if(isLong)
        breakoutDistance = entryPrice - state.orHigh;
    else
        breakoutDistance = state.orLow - entryPrice;
    
    double breakoutAtrMultiple = breakoutDistance / currentATR;
    if(breakoutAtrMultiple > max_BreakoutAtrMultiple)
    {
        Print("Filter: Breakout ATR Multiple too high: ", breakoutAtrMultiple, " > ", max_BreakoutAtrMultiple);
        return false;
    }
    
    // 2. ATR Ratio (current ATR / avg ATR)
    double atrRatio = currentATR / avgATR;
    if(atrRatio > max_AtrRatio)
    {
        Print("Filter: ATR Ratio too high: ", atrRatio, " > ", max_AtrRatio);
        return false;
    }
    
    // 3. Range ATR Ratio (range size / ATR)
    double rangeSize = state.orHigh - state.orLow;
    double rangeAtrRatio = rangeSize / currentATR;
    if(rangeAtrRatio < min_RangeAtrRatio)
    {
        Print("Filter: Range ATR Ratio too low: ", rangeAtrRatio, " < ", min_RangeAtrRatio);
        return false;
    }
    
    // 4. Entry Offset Ratio
    double entryOffset = 0;
    if(isLong)
        entryOffset = entryPrice - state.orHigh;
    else
        entryOffset = state.orLow - entryPrice;
    
    double entryOffsetRatio = entryOffset / rangeSize;
    if(entryOffsetRatio < min_EntryOffsetRatio || entryOffsetRatio > max_EntryOffsetRatio)
    {
        Print("Filter: Entry Offset Ratio out of range: ", entryOffsetRatio);
        return false;
    }
    
    // 5. Consolidation Score
    double consolidationScore = CalculateConsolidationScore();
    if(consolidationScore > max_ConsolidationScore)
    {
        Print("Filter: Consolidation Score too high: ", consolidationScore, " > ", max_ConsolidationScore);
        return false;
    }
    
    // 6. Trend Score
    double trendScore = CalculateTrendScore(isLong, entryPrice);
    if(trendScore < min_TrendScore)
    {
        Print("Filter: Trend Score too low: ", trendScore, " < ", min_TrendScore);
        return false;
    }
    
    Print("✅ All model filters passed!");
    return true;
}

//+------------------------------------------------------------------+
//| Calculate Consolidation Score                                    |
//+------------------------------------------------------------------+
double CalculateConsolidationScore()
{
    double atr[20];
    if(CopyBuffer(atr_handle, 0, 0, 20, atr) < 20) return 1.0;
    
    double currentATR = atr[0];
    double avgATR = 0;
    for(int i = 1; i < 20; i++) avgATR += atr[i];
    avgATR /= 19.0;
    
    bool isConsolidating = (currentATR < avgATR * 0.7);
    
    double high[20], low[20];
    for(int i = 0; i < 20; i++)
    {
        high[i] = iHigh(_Symbol, PERIOD_M5, i);
        low[i] = iLow(_Symbol, PERIOD_M5, i);
    }
    
    double currentRange = high[0] - low[0];
    double avgRange = 0;
    for(int i = 1; i < 20; i++) avgRange += (high[i] - low[i]);
    avgRange /= 19.0;
    
    bool isTightRange = (currentRange < avgRange * 0.8);
    
    double score = ((isConsolidating ? 1.0 : 0.0) + (isTightRange ? 1.0 : 0.0)) / 2.0;
    return score;
}

//+------------------------------------------------------------------+
//| Calculate Trend Score                                            |
//+------------------------------------------------------------------+
double CalculateTrendScore(bool isLong, double price)
{
    double score = 0.0;
    int components = 0;
    
    // EMA 200 (1H) filter
    if(isFilterActive)
    {
        double ema_buffer[];
        if(CopyBuffer(ema_handle_filter, 0, 0, 1, ema_buffer) > 0)
        {
            bool priceAboveEMA = (price > ema_buffer[0]);
            if(isLong && priceAboveEMA) score += 1.0;
            else if(!isLong && !priceAboveEMA) score += 1.0;
            components++;
        }
    }
    
    // EMA 200 (M5) filter
    if(isFilterM5Active)
    {
        double ema_m5_buffer[];
        if(CopyBuffer(ema_handle_m5, 0, 0, 1, ema_m5_buffer) > 0)
        {
            bool priceAboveEMA = (price > ema_m5_buffer[0]);
            if(isLong && priceAboveEMA) score += 1.0;
            else if(!isLong && !priceAboveEMA) score += 1.0;
            components++;
        }
    }
    
    // EMA alignment (9, 21, 50)
    if(ema9_handle != INVALID_HANDLE && ema21_handle != INVALID_HANDLE && ema50_handle != INVALID_HANDLE)
    {
        double ema9[], ema21[], ema50[];
        if(CopyBuffer(ema9_handle, 0, 0, 1, ema9) > 0 &&
           CopyBuffer(ema21_handle, 0, 0, 1, ema21) > 0 &&
           CopyBuffer(ema50_handle, 0, 0, 1, ema50) > 0)
        {
            bool aligned = false;
            if(isLong)
                aligned = (ema9[0] > ema21[0] && ema21[0] > ema50[0]);
            else
                aligned = (ema9[0] < ema21[0] && ema21[0] < ema50[0]);
            
            if(aligned) score += 1.0;
            components++;
        }
    }
    
    if(components > 0)
        return score / components;
    else
        return 0.5; // Neutral if no components available
}

//+------------------------------------------------------------------+
//| EMA Trend Filter                                                 |
//+------------------------------------------------------------------+
bool PassesTrendFilter(bool isLong, double price)
{
    bool allowsTrade = true;
    
    // Check 1H EMA 200
    if(isFilterActive)
    {
        double ema_buffer[];
        if(CopyBuffer(ema_handle_filter, 0, 0, 1, ema_buffer) > 0)
        {
            if(isLong)
                allowsTrade = (price > ema_buffer[0]);
            else
                allowsTrade = (price < ema_buffer[0]);
        }
    }
    
    // Check M5 EMA 200
    if(allowsTrade && isFilterM5Active)
    {
        double ema_m5_buffer[];
        if(CopyBuffer(ema_handle_m5, 0, 0, 1, ema_m5_buffer) > 0)
        {
            if(isLong)
                allowsTrade = (price > ema_m5_buffer[0]);
            else
                allowsTrade = (price < ema_m5_buffer[0]);
        }
    }
    
    return allowsTrade;
}

//+------------------------------------------------------------------+
//| Execute Trade                                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE orderType, double entryPrice, SessionState &state, ENUM_SESSION_TYPE sessionType)
{
    if(state.isTradeTaken) return;
    
    double stopLoss = (orderType == ORDER_TYPE_BUY) ? state.orLow : state.orHigh;
    double slDistance = MathAbs(entryPrice - stopLoss);
    
    if(slDistance <= 0) return;
    
    double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double actualEntry = (orderType == ORDER_TYPE_BUY) ? askPrice : bidPrice;
    
    double takeProfit = (orderType == ORDER_TYPE_BUY) 
        ? actualEntry + (slDistance * RiskRewardRatio) 
        : actualEntry - (slDistance * RiskRewardRatio);
    
    double lotSize = CalculateLotSize(actualEntry, stopLoss);
    if(lotSize <= 0) return;
    
    string comment = EaComment + "_" + state.sessionName;
    bool result = false;
    
    if(orderType == ORDER_TYPE_BUY)
        result = trade.Buy(lotSize, _Symbol, 0, stopLoss, takeProfit, comment);
    else
        result = trade.Sell(lotSize, _Symbol, 0, stopLoss, takeProfit, comment);
    
    if(result && trade.ResultRetcode() == TRADE_RETCODE_DONE)
    {
        state.isTradeTaken = true;
        
        string msg = StringFormat("✅ %s TRADE EXECUTED | %s @ %.5f | SL: %.5f | TP: %.5f | Lots: %.2f | R:R = %.2f",
            state.sessionName,
            (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"),
            trade.ResultPrice(),
            stopLoss,
            takeProfit,
            lotSize,
            RiskRewardRatio);
        Print(msg);
    }
    else
    {
        PrintFormat("❌ Order execution failed. Error: %d - %s", trade.ResultRetcode(), trade.ResultComment());
    }
}

//+------------------------------------------------------------------+
//| Find Opening Range                                               |
//+------------------------------------------------------------------+
bool FindOpeningRange(string sessionStartString, SessionState &state)
{
    string start_parts[];
    if(StringSplit(sessionStartString, ':', start_parts) != 2) return false;
    
    int start_h = (int)StringToInteger(start_parts[0]);
    int start_m = (int)StringToInteger(start_parts[1]);
    
    MqlDateTime dt_start;
    TimeToStruct(TimeCurrent(), dt_start);
    dt_start.hour = start_h;
    dt_start.min = start_m;
    dt_start.sec = 0;
    
    state.orStartTime = StructToTime(dt_start);
    long durationSeconds = ParseDurationToSeconds(session_OR_Duration, 900);
    state.orEndTime = (datetime)(state.orStartTime + durationSeconds);
    
    if(TimeCurrent() < state.orEndTime) return false;
    
    int bar1 = iBarShift(_Symbol, PERIOD_M5, state.orStartTime);
    int bar2 = iBarShift(_Symbol, PERIOD_M5, state.orEndTime);
    
    if(bar1 < 0 || bar2 < 0) return false;
    
    int startIndex = MathMax(bar1, bar2);
    int endIndex = MathMin(bar1, bar2);
    
    double highs[], lows[];
    int barsInRange = startIndex - endIndex + 1;
    if(barsInRange <= 0) return false;
    
    if(CopyHigh(_Symbol, PERIOD_M5, endIndex, barsInRange, highs) < 0 || 
       CopyLow(_Symbol, PERIOD_M5, endIndex, barsInRange, lows) < 0) return false;
    
    state.orHigh = highs[ArrayMaximum(highs)];
    state.orLow = lows[ArrayMinimum(lows)];
    
    // Validate range
    if(state.orHigh > 0 && state.orLow > 0 && state.orHigh > state.orLow)
    {
        state.isRangeFound = true;
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Calculate Lot Size                                               |
//+------------------------------------------------------------------+
double CalculateLotSize(double entryPrice, double stopLossPrice)
{
    if(MoneyManagementMode == FIXED_LOT) return FixedLotSize;
    
    double lotSize = 0.0;
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double slDistance = MathAbs(entryPrice - stopLossPrice);
    
    if(slDistance <= 0) return 0.0;
    
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    if(tickValue <= 0 || tickSize <= 0) return 0.0;
    
    double riskAmount = accountBalance * (RiskPercentPerTrade / 100.0);
    double lossPerLot = slDistance / tickSize * tickValue;
    
    if(lossPerLot > 0) lotSize = riskAmount / lossPerLot;
    
    double volMin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double volMax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lotSize = MathFloor(lotSize / volStep) * volStep;
    lotSize = MathMax(volMin, lotSize);
    lotSize = MathMin(volMax, lotSize);
    lotSize = MathMin(MaxLotSize, lotSize);
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| Parse Duration to Seconds                                        |
//+------------------------------------------------------------------+
long ParseDurationToSeconds(string duration, long defaultValue)
{
    string parts[];
    if(StringSplit(duration, ':', parts) == 2)
    {
        return(StringToInteger(parts[0]) * 3600 + StringToInteger(parts[1]) * 60);
    }
    return defaultValue;
}

//+------------------------------------------------------------------+
//| Reset Daily Variables                                            |
//+------------------------------------------------------------------+
void ResetDailyVariables()
{
    ZeroMemory(asianState);
    ZeroMemory(londonState);
    ZeroMemory(nyState);
    
    asianState.sessionName = "ASIAN";
    londonState.sessionName = "LONDON";
    nyState.sessionName = "NY";
    
    Print("New day detected. Variables reset.");
}

