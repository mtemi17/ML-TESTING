//+------------------------------------------------------------------+
//|                                                   DUKE_ABUYA.mq5 |
//|                     ORB Breakout EA with Winner Tracking & Learning |
//|                          Tracks winners in Strategy Tester       |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Enhanced ORB EA"
#property version   "1.00"
#property description "Opening Range Breakout EA with Winner Tracking for Strategy Tester"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

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

//--- Trade Data Structure for Winner Tracking
struct TradeFilterData
{
    ulong    ticket;
    double   breakoutAtrMultiple;
    double   atrRatio;
    double   rangeAtrRatio;
    double   entryOffsetRatio;
    double   consolidationScore;
    double   trendScore;
    bool     isWinner;
    double   profit;
};

//+------------------------------------------------------------------+
//| Expert Advisor Inputs                                            |
//+------------------------------------------------------------------+
input group "=== SESSION SETTINGS ==="
input string         session_OR_Duration     = "00:15";
input string         session_TradeDuration = "03:00";
input bool           session_EnableAsian     = true;
input string         asian_SessionStart      = "03:00";
input bool           session_EnableLondon    = true;
input string         london_SessionStart     = "10:00";
input bool           session_EnableNewYork = true;
input string         ny_SessionStart         = "16:30";

input group "=== TRADE SETTINGS ==="
input bool           enable_MarketBreakout = true;
input ulong          MagicNumber           = 13579;
input string         EaComment             = "DUKE_ABUYA";
input double         RiskRewardRatio       = 2.0;

input group "=== AI MODEL FILTERS (Learned from Winners) ==="
input bool           use_ModelFilters        = true;
input double         max_BreakoutAtrMultiple = 0.55;
input double         max_AtrRatio            = 1.17;
input double         min_TrendScore          = 0.67;
input double         max_ConsolidationScore = 0.0;
input double         min_RangeAtrRatio       = 0.92;
input double         min_EntryOffsetRatio    = 0.0;
input double         max_EntryOffsetRatio    = 1.0;

input group "=== EMA FILTERS ==="
input bool           filter_Enable         = true;
input ENUM_TIMEFRAMES  filter_Timeframe      = PERIOD_H1;
input int            filter_EmaPeriod      = 200;
input bool           filter_EnableM5       = true;

input group "=== RISK MANAGEMENT ==="
input ENUM_MONEY_MANAGEMENT MoneyManagementMode = RISK_PERCENT;
input double         RiskPercentPerTrade     = 1.0;
input double         FixedLotSize            = 0.01;
input double         MaxLotSize              = 10.0;

input group "=== WINNER TRACKING SETTINGS ==="
input bool           enable_WinnerTracking = true;
input bool           print_WinnerStats     = true;

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CTrade     trade;
CPositionInfo position;
datetime   todayDate;
datetime   lastBarTime = 0;
SessionState asianState;
SessionState londonState;
SessionState nyState;
int        ema_handle_filter = INVALID_HANDLE;
int        ema_handle_m5     = INVALID_HANDLE;
int        atr_handle        = INVALID_HANDLE;
int        ema9_handle       = INVALID_HANDLE;
int        ema21_handle      = INVALID_HANDLE;
int        ema50_handle      = INVALID_HANDLE;
bool       isFilterActive    = false;
bool       isFilterM5Active  = false;

//--- Winner Tracking Arrays
TradeFilterData tradeHistory[];
int tradeHistorySize = 0;

//--- Global Average Calculators
long   g_filter_attempt_count = 0;
double g_sum_breakoutAtrMultiple = 0.0;
double g_sum_atrRatio = 0.0;
double g_sum_rangeAtrRatio = 0.0;
double g_sum_entryOffsetRatio = 0.0;
double g_sum_consolidationScore = 0.0;
double g_sum_trendScore = 0.0;

//--- Winner Statistics
int    g_total_trades = 0;
int    g_winning_trades = 0;
double g_winner_avg_breakoutAtr = 0.0;
double g_winner_avg_atrRatio = 0.0;
double g_winner_avg_rangeAtr = 0.0;
double g_winner_avg_entryOffset = 0.0;
double g_winner_avg_consolidation = 0.0;
double g_winner_avg_trend = 0.0;
//--- Winner Minimum Values
double g_winner_min_breakoutAtr = 0.0;
double g_winner_min_atrRatio = 0.0;
double g_winner_min_rangeAtr = 0.0;
double g_winner_min_entryOffset = 0.0;
double g_winner_min_consolidation = 0.0;
double g_winner_min_trend = 0.0;

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
    if(ema9_handle == INVALID_HANDLE)
        Print("Warning: Failed to create EMA9 handle");
    
    ema21_handle = iMA(_Symbol, PERIOD_M5, 21, 0, MODE_EMA, PRICE_CLOSE);
    if(ema21_handle == INVALID_HANDLE)
        Print("Warning: Failed to create EMA21 handle");
    
    ema50_handle = iMA(_Symbol, PERIOD_M5, 50, 0, MODE_EMA, PRICE_CLOSE);
    if(ema50_handle == INVALID_HANDLE)
        Print("Warning: Failed to create EMA50 handle");
    
    // Reset counters
    g_filter_attempt_count = 0;
    g_sum_breakoutAtrMultiple = 0.0;
    g_sum_atrRatio = 0.0;
    g_sum_rangeAtrRatio = 0.0;
    g_sum_entryOffsetRatio = 0.0;
    g_sum_consolidationScore = 0.0;
    g_sum_trendScore = 0.0;
    g_total_trades = 0;
    g_winning_trades = 0;
    
    // Initialize winner min values to very high numbers
    g_winner_min_breakoutAtr = 999999.0;
    g_winner_min_atrRatio = 999999.0;
    g_winner_min_rangeAtr = 999999.0;
    g_winner_min_entryOffset = 999999.0;
    g_winner_min_consolidation = 999999.0;
    g_winner_min_trend = 999999.0;
    
    // Initialize trade history array
    ArrayResize(tradeHistory, 0);
    tradeHistorySize = 0;
    
    ResetDailyVariables();
    
    // Verify we have enough history data
    if(iBars(_Symbol, PERIOD_M5) < 50)
    {
        Print("ERROR: Not enough historical data. Need at least 50 M5 bars.");
        return INIT_FAILED;
    }
    
    Print("=", StringSubstr("=", 0, 60), "=");
    Print("DUKE_ABUYA v1.00 Initialized (Winner Tracking Enabled)");
    Print("Magic Number: ", MagicNumber);
    Print("Symbol: ", _Symbol);
    Print("Available Bars: ", iBars(_Symbol, PERIOD_M5));
    Print("Winner Tracking: ", (enable_WinnerTracking ? "ENABLED" : "DISABLED"));
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
    
    // Check all closed trades and update winner stats
    if(enable_WinnerTracking)
    {
        CheckAllClosedTrades();
        PrintWinnerStatistics();
    }
    
    // Print Averages Report
    Print("=", StringSubstr("=", 0, 60), "=");
    Print("DUKE_ABUYA: Filter Averages Report");
    if(g_filter_attempt_count > 0)
    {
        PrintFormat("Total Filter Attempts: %d", g_filter_attempt_count);
        PrintFormat("  [1] Avg Breakout ATR Multiple: %.2f (Your Max: %.2f)", 
                    (g_sum_breakoutAtrMultiple / g_filter_attempt_count), max_BreakoutAtrMultiple);
        PrintFormat("  [2] Avg ATR Ratio:             %.2f (Your Max: %.2f)", 
                    (g_sum_atrRatio / g_filter_attempt_count), max_AtrRatio);
        PrintFormat("  [3] Avg Range ATR Ratio:       %.2f (Your Min: %.2f)", 
                    (g_sum_rangeAtrRatio / g_filter_attempt_count), min_RangeAtrRatio);
        PrintFormat("  [4] Avg Entry Offset Ratio:    %.2f (Your Min: %.2f, Max: %.2f)", 
                    (g_sum_entryOffsetRatio / g_filter_attempt_count), min_EntryOffsetRatio, max_EntryOffsetRatio);
        PrintFormat("  [5] Avg Consolidation Score:   %.2f (Your Max: %.2f)", 
                    (g_sum_consolidationScore / g_filter_attempt_count), max_ConsolidationScore);
        PrintFormat("  [6] Avg Trend Score:           %.2f (Your Min: %.2f)", 
                    (g_sum_trendScore / g_filter_attempt_count), min_TrendScore);
    }
    else
    {
        Print("No filter attempts were recorded.");
    }
    Print("=", StringSubstr("=", 0, 60), "=");
    
    Print("DUKE_ABUYA Deinitialized. Reason: ", reason);
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
    
    // Winner tracking is done at deinit to avoid performance issues
    // CheckClosedPositions() removed from OnTick
}

//+------------------------------------------------------------------+
//| Check for closed positions and update winner stats               |
//+------------------------------------------------------------------+
// CheckClosedPositions() - DISABLED to avoid performance issues
// Winner tracking is now only done at OnDeinit

//+------------------------------------------------------------------+
//| Check all closed trades at deinit                               |
//+------------------------------------------------------------------+
void CheckAllClosedTrades()
{
    if(!enable_WinnerTracking) return;
    if(tradeHistorySize == 0) return;
    
    // Safety check - make sure array is valid
    if(ArraySize(tradeHistory) < tradeHistorySize)
    {
        Print("Error: Trade history array size mismatch");
        return;
    }
    
    // Select history from beginning
    if(!HistorySelect(0, TimeCurrent()))
    {
        Print("Warning: Failed to select history for winner tracking");
        return;
    }
    
    int totalDeals = HistoryDealsTotal();
    if(totalDeals <= 0) return;
    
    // Process each trade in our history
    for(int i = 0; i < tradeHistorySize; i++)
    {
        if(tradeHistory[i].isWinner) continue; // Already processed
        
        // Search through deals to find closing deal for this position
        for(int j = 0; j < totalDeals; j++)
        {
            ulong ticket = HistoryDealGetTicket(j);
            if(ticket == 0) continue;
            
            // Check magic number first (faster filter)
            long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
            if(magic != (long)MagicNumber) continue;
            
            // Check position ID
            long posId = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
            if(posId != (long)tradeHistory[i].ticket) continue;
            
            // Check if it's a closing deal
            long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
            if(entry == DEAL_ENTRY_OUT)
            {
                double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
                tradeHistory[i].profit = profit;
                tradeHistory[i].isWinner = (profit > 0);
                
                if(tradeHistory[i].isWinner)
                {
                    g_winning_trades++;
                    UpdateWinnerAverages(tradeHistory[i]);
                }
                g_total_trades++;
                break; // Found the closing deal, move to next trade
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Update winner averages                                           |
//+------------------------------------------------------------------+
void UpdateWinnerAverages(TradeFilterData &winner)
{
    if(g_winning_trades == 1)
    {
        // First winner - initialize both avg and min
        g_winner_avg_breakoutAtr = winner.breakoutAtrMultiple;
        g_winner_avg_atrRatio = winner.atrRatio;
        g_winner_avg_rangeAtr = winner.rangeAtrRatio;
        g_winner_avg_entryOffset = winner.entryOffsetRatio;
        g_winner_avg_consolidation = winner.consolidationScore;
        g_winner_avg_trend = winner.trendScore;
        
        // Initialize min values
        g_winner_min_breakoutAtr = winner.breakoutAtrMultiple;
        g_winner_min_atrRatio = winner.atrRatio;
        g_winner_min_rangeAtr = winner.rangeAtrRatio;
        g_winner_min_entryOffset = winner.entryOffsetRatio;
        g_winner_min_consolidation = winner.consolidationScore;
        g_winner_min_trend = winner.trendScore;
    }
    else
    {
        // Running average
        double n = (double)g_winning_trades;
        g_winner_avg_breakoutAtr = ((n-1) * g_winner_avg_breakoutAtr + winner.breakoutAtrMultiple) / n;
        g_winner_avg_atrRatio = ((n-1) * g_winner_avg_atrRatio + winner.atrRatio) / n;
        g_winner_avg_rangeAtr = ((n-1) * g_winner_avg_rangeAtr + winner.rangeAtrRatio) / n;
        g_winner_avg_entryOffset = ((n-1) * g_winner_avg_entryOffset + winner.entryOffsetRatio) / n;
        g_winner_avg_consolidation = ((n-1) * g_winner_avg_consolidation + winner.consolidationScore) / n;
        g_winner_avg_trend = ((n-1) * g_winner_avg_trend + winner.trendScore) / n;
        
        // Update minimum values
        if(winner.breakoutAtrMultiple < g_winner_min_breakoutAtr)
            g_winner_min_breakoutAtr = winner.breakoutAtrMultiple;
        if(winner.atrRatio < g_winner_min_atrRatio)
            g_winner_min_atrRatio = winner.atrRatio;
        if(winner.rangeAtrRatio < g_winner_min_rangeAtr)
            g_winner_min_rangeAtr = winner.rangeAtrRatio;
        if(winner.entryOffsetRatio < g_winner_min_entryOffset)
            g_winner_min_entryOffset = winner.entryOffsetRatio;
        if(winner.consolidationScore < g_winner_min_consolidation)
            g_winner_min_consolidation = winner.consolidationScore;
        if(winner.trendScore < g_winner_min_trend)
            g_winner_min_trend = winner.trendScore;
    }
}

//+------------------------------------------------------------------+
//| Print winner statistics                                          |
//+------------------------------------------------------------------+
void PrintWinnerStatistics()
{
    if(g_total_trades == 0) return;
    
    Print("=", StringSubstr("=", 0, 60), "=");
    Print("DUKE_ABUYA: WINNER STATISTICS");
    PrintFormat("Total Trades: %d | Winners: %d | Win Rate: %.1f%%", 
                g_total_trades, g_winning_trades, (g_winning_trades * 100.0 / g_total_trades));
    
    if(g_winning_trades > 0)
    {
        Print("--- WINNER STATISTICS (Use these as your filter values) ---");
        PrintFormat("  [1] Breakout ATR Multiple: Min=%.2f | Avg=%.2f", 
                    g_winner_min_breakoutAtr, g_winner_avg_breakoutAtr);
        PrintFormat("  [2] ATR Ratio:             Min=%.2f | Avg=%.2f", 
                    g_winner_min_atrRatio, g_winner_avg_atrRatio);
        PrintFormat("  [3] Range ATR Ratio:       Min=%.2f | Avg=%.2f", 
                    g_winner_min_rangeAtr, g_winner_avg_rangeAtr);
        PrintFormat("  [4] Entry Offset Ratio:    Min=%.2f | Avg=%.2f", 
                    g_winner_min_entryOffset, g_winner_avg_entryOffset);
        PrintFormat("  [5] Consolidation Score:   Min=%.2f | Avg=%.2f", 
                    g_winner_min_consolidation, g_winner_avg_consolidation);
        PrintFormat("  [6] Trend Score:           Min=%.2f | Avg=%.2f", 
                    g_winner_min_trend, g_winner_avg_trend);
        Print("");
        Print("=== MINIMUM VALUES FOR WINNERS (COPY THESE) ===");
        PrintFormat("  MIN Breakout ATR Multiple: %.2f", g_winner_min_breakoutAtr);
        PrintFormat("  MIN ATR Ratio:             %.2f", g_winner_min_atrRatio);
        PrintFormat("  MIN Range ATR Ratio:       %.2f", g_winner_min_rangeAtr);
        PrintFormat("  MIN Entry Offset Ratio:    %.2f", g_winner_min_entryOffset);
        PrintFormat("  MIN Consolidation Score:   %.2f", g_winner_min_consolidation);
        PrintFormat("  MIN Trend Score:           %.2f", g_winner_min_trend);
        Print("");
        Print("=== RECOMMENDED FILTER SETTINGS ===");
        PrintFormat("  max_BreakoutAtrMultiple = %.2f (use avg or slightly above)", g_winner_avg_breakoutAtr);
        PrintFormat("  max_AtrRatio            = %.2f (use avg or slightly above)", g_winner_avg_atrRatio);
        PrintFormat("  min_RangeAtrRatio       = %.2f (use min or slightly below)", g_winner_min_rangeAtr);
        PrintFormat("  min_EntryOffsetRatio    = %.2f (use min or slightly below)", g_winner_min_entryOffset);
        PrintFormat("  max_ConsolidationScore  = %.2f (use avg or slightly above)", g_winner_avg_consolidation);
        PrintFormat("  min_TrendScore          = %.2f (use min or slightly below)", g_winner_min_trend);
        Print("--- COPY THESE VALUES TO YOUR FILTER INPUTS ---");
    }
    Print("=", StringSubstr("=", 0, 60), "=");
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
//| Enhanced breakout detection with model-based filters             |
//+------------------------------------------------------------------+
void CheckForBreakout(SessionState &state, ENUM_SESSION_TYPE sessionType)
{
    if(state.isTradeTaken) return;
    
    MqlRates bar[];
    if(CopyRates(_Symbol, PERIOD_M5, 1, 1, bar) < 1) return;
    
    double openPrice = bar[0].open;
    double closePrice = bar[0].close;
    
    bool isBullishCandle = closePrice > openPrice;
    bool isBearishCandle = closePrice < openPrice;
    
    // BULLISH BREAKOUT: Close above OR high AND bullish candle
    if(closePrice > state.orHigh && isBullishCandle)
    {
        if(enable_MarketBreakout)
        {
            // Create trade data structure
            TradeFilterData tradeData;
            ZeroMemory(tradeData);
            
            // Apply model-based filters
            if(!use_ModelFilters || PassesModelFilters(true, closePrice, state, tradeData))
            {
                // Apply EMA trend filter
                if(!filter_Enable || PassesTrendFilter(true, closePrice))
                {
                    ExecuteTrade(ORDER_TYPE_BUY, closePrice, state, sessionType, tradeData);
                }
                else
                {
                    Print("LONG breakout filtered out by EMA trend filter");
                }
            }
        }
    }
    // BEARISH BREAKOUT: Close below OR low AND bearish candle
    else if(closePrice < state.orLow && isBearishCandle)
    {
        if(enable_MarketBreakout)
        {
            // Create trade data structure
            TradeFilterData tradeData;
            ZeroMemory(tradeData);
            
            // Apply model-based filters
            if(!use_ModelFilters || PassesModelFilters(false, closePrice, state, tradeData))
            {
                // Apply EMA trend filter
                if(!filter_Enable || PassesTrendFilter(false, closePrice))
                {
                    ExecuteTrade(ORDER_TYPE_SELL, closePrice, state, sessionType, tradeData);
                }
                else
                {
                    Print("SHORT breakout filtered out by EMA trend filter");
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Model-based filters (stores data in tradeData)                   |
//+------------------------------------------------------------------+
bool PassesModelFilters(bool isLong, double entryPrice, SessionState &state, TradeFilterData &tradeData)
{
    // Get ATR data
    double atr[20];
    if(CopyBuffer(atr_handle, 0, 0, 20, atr) < 20) return false;
    
    double currentATR = atr[0];
    double avgATR = 0;
    for(int i = 1; i < 20; i++) avgATR += atr[i];
    avgATR /= 19.0;
    
    if(avgATR <= 0 || currentATR <= 0) return false;
    
    double rangeSize = state.orHigh - state.orLow;
    if(rangeSize <= 0) return false;
    
    // Calculate all filter values
    double breakoutDistance = (isLong) ? (entryPrice - state.orHigh) : (state.orLow - entryPrice);
    double breakoutAtrMultiple = breakoutDistance / currentATR;
    double atrRatio = currentATR / avgATR;
    double rangeAtrRatio = rangeSize / currentATR;
    double entryOffsetRatio = breakoutDistance / rangeSize;
    double consolidationScore = CalculateConsolidationScore();
    double trendScore = CalculateTrendScore(isLong, entryPrice);
    
    // Store values in tradeData
    tradeData.breakoutAtrMultiple = breakoutAtrMultiple;
    tradeData.atrRatio = atrRatio;
    tradeData.rangeAtrRatio = rangeAtrRatio;
    tradeData.entryOffsetRatio = entryOffsetRatio;
    tradeData.consolidationScore = consolidationScore;
    tradeData.trendScore = trendScore;
    
    // Update running averages
    g_filter_attempt_count++;
    g_sum_breakoutAtrMultiple += breakoutAtrMultiple;
    g_sum_atrRatio += atrRatio;
    g_sum_rangeAtrRatio += rangeAtrRatio;
    g_sum_entryOffsetRatio += entryOffsetRatio;
    g_sum_consolidationScore += consolidationScore;
    g_sum_trendScore += trendScore;
    
    // Check all conditions
    bool pass1 = (breakoutAtrMultiple <= max_BreakoutAtrMultiple);
    bool pass2 = (atrRatio <= max_AtrRatio);
    bool pass3 = (rangeAtrRatio >= min_RangeAtrRatio);
    bool pass4 = (entryOffsetRatio >= min_EntryOffsetRatio && entryOffsetRatio <= max_EntryOffsetRatio);
    bool pass5 = (consolidationScore <= max_ConsolidationScore);
    bool pass6 = (trendScore >= min_TrendScore);
    
    if(pass1 && pass2 && pass3 && pass4 && pass5 && pass6)
    {
        Print("✅ All model filters passed!");
        return true;
    }
    
    return false;
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
    
    if(ema9_handle != INVALID_HANDLE && ema21_handle != INVALID_HANDLE && ema50_handle != INVALID_HANDLE)
    {
        double ema9[], ema21[], ema50[];
        if(CopyBuffer(ema9_handle, 0, 0, 1, ema9) > 0 &&
           CopyBuffer(ema21_handle, 0, 0, 1, ema21) > 0 &&
           CopyBuffer(ema50_handle, 0, 0, 1, ema50) > 0)
        {
            bool aligned = false;
            if(isLong) aligned = (ema9[0] > ema21[0] && ema21[0] > ema50[0]);
            else aligned = (ema9[0] < ema21[0] && ema21[0] < ema50[0]);
            
            if(aligned) score += 1.0;
            components++;
        }
    }
    
    if(components > 0) return score / components;
    else return 0.5;
}

//+------------------------------------------------------------------+
//| EMA Trend Filter                                                 |
//+------------------------------------------------------------------+
bool PassesTrendFilter(bool isLong, double price)
{
    bool allowsTrade = true;
    
    if(isFilterActive)
    {
        double ema_buffer[];
        if(CopyBuffer(ema_handle_filter, 0, 0, 1, ema_buffer) > 0)
        {
            if(isLong) allowsTrade = (price > ema_buffer[0]);
            else allowsTrade = (price < ema_buffer[0]);
        }
    }
    
    if(allowsTrade && isFilterM5Active)
    {
        double ema_m5_buffer[];
        if(CopyBuffer(ema_handle_m5, 0, 0, 1, ema_m5_buffer) > 0)
        {
            if(isLong) allowsTrade = (price > ema_m5_buffer[0]);
            else allowsTrade = (price < ema_m5_buffer[0]);
        }
    }
    
    return allowsTrade;
}

//+------------------------------------------------------------------+
//| Execute Trade (stores trade data for winner tracking)            |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE orderType, double entryPrice, SessionState &state, 
                  ENUM_SESSION_TYPE sessionType, TradeFilterData &tradeData)
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
        
        // Store trade data for winner tracking
        if(enable_WinnerTracking)
        {
            // Get position ticket - try multiple methods
            ulong posTicket = 0;
            if(PositionSelect(_Symbol))
            {
                posTicket = PositionGetInteger(POSITION_IDENTIFIER);
            }
            
            // If position select failed, try using order result
            if(posTicket == 0)
            {
                posTicket = trade.ResultOrder();
            }
            
            if(posTicket > 0)
            {
                tradeData.ticket = posTicket;
                tradeData.isWinner = false;
                tradeData.profit = 0.0;
                
                // Add to history array with safety check
                int newSize = tradeHistorySize + 1;
                if(ArrayResize(tradeHistory, newSize) > 0)
                {
                    tradeHistory[tradeHistorySize] = tradeData;
                    tradeHistorySize = newSize;
                }
                else
                {
                    Print("Warning: Failed to resize trade history array");
                }
            }
        }
        
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

//+------------------------------------------------------------------+

