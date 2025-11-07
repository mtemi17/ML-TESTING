# USDJPY5 TEST REPORT - ALL THREE MODELS

## 📊 Test Results Summary

**Test Dataset:** USDJPY5 (386 completed trades)
**Models Tested:** Baseline, Random Forest ML, Deep Learning
**Note:** Models were trained on XAUUSD data, tested on USDJPY

---

## 🎯 RESULTS ON USDJPY5

| Model | Trades | Win Rate | Total P&L | Avg P&L | Performance |
|-------|--------|----------|-----------|---------|-------------|
| **Baseline (No Filter)** | 386 (100%) | **36.27%** | **$5.46** | $0.01 | ✅ Best |
| **Random Forest ML** | 386 (100%) | 36.27% | $5.46 | $0.01 | ⚠️ No Filtering |
| **Deep Learning** | 4 (1.0%) | 25.00% | **-$0.88** | -$0.22 | ❌ Poor |

---

## 🔍 KEY FINDINGS

### 1. Random Forest ML Model
- **Issue:** Predicted ALL 386 trades as wins (100%)
- **Result:** Same performance as baseline (no filtering effect)
- **Analysis:** Model learned XAUUSD-specific patterns that don't apply to USDJPY
- **Conclusion:** Model does NOT generalize to different currency pairs

### 2. Deep Learning Model
- **Very Conservative:** Only selected 4 trades (1.0% of all trades)
- **Poor Performance:** 25% win rate, negative P&L (-$0.88)
- **Analysis:** Model is too conservative and selected losing trades
- **Conclusion:** Model does NOT generalize well to USDJPY

### 3. Baseline Strategy
- **Best Performer:** 36.27% win rate, $5.46 total P&L
- **Consistent:** Similar win rate to XAUUSD baseline (36.51%)
- **Conclusion:** Strategy works across pairs, but ML models are pair-specific

---

## 📈 CROSS-PAIR COMPARISON

### XAUUSD (Training Data)
| Model | Trades | Win Rate | Total P&L |
|-------|--------|----------|-----------|
| Baseline | 545 | 36.5% | $965.92 |
| RF ML | 151 | **87.4%** | **$2,288.65** |
| DL | 46 | 69.6% | $717.25 |

### USDJPY (Test Data)
| Model | Trades | Win Rate | Total P&L |
|-------|--------|----------|-----------|
| Baseline | 386 | 36.3% | **$5.46** |
| RF ML | 386 | 36.3% | $5.46 |
| DL | 4 | 25.0% | -$0.88 |

---

## 💡 CRITICAL INSIGHT

### Models Do NOT Generalize Across Currency Pairs

**Evidence:**
1. Random Forest trained on XAUUSD predicted 100% of USDJPY trades as wins
2. Deep Learning was too conservative and selected losing trades
3. Baseline strategy performed consistently across both pairs

**Why This Happens:**
- Different currency pairs have different volatility characteristics
- Price ranges and movements differ significantly
- Indicator values have different meanings across pairs
- Models learned XAUUSD-specific patterns

---

## ✅ RECOMMENDATIONS

### 1. Pair-Specific Models
- **Train separate models for each currency pair**
- XAUUSD model for Gold trading
- USDJPY model for JPY trading
- Each model learns pair-specific patterns

### 2. Baseline Strategy
- **Works consistently across pairs** (36-37% win rate)
- Can be used as-is for any currency pair
- ML models need retraining for each pair

### 3. Model Validation
- Always test models on out-of-sample data
- Test on different currency pairs to check generalization
- If models don't generalize, train pair-specific models

---

## 📁 FILES CREATED

1. **usdjpy5_backtest_results.csv** - All trades from USDJPY5 backtest
2. **usdjpy5_model_comparison.csv** - Model comparison results
3. **usdjpy5_all_predictions.csv** - Detailed predictions from all models
4. **USDJPY5_with_indicators.csv** - Enhanced USDJPY5 data with indicators

---

## 🎯 CONCLUSION

**On USDJPY5:**
- ✅ **Baseline Strategy** performs best
- ⚠️ **Random Forest ML** doesn't filter (predicts all as wins)
- ❌ **Deep Learning** too conservative, poor performance

**Key Takeaway:**
Models trained on XAUUSD are **pair-specific** and do NOT generalize to USDJPY. For best results, train separate models for each currency pair.

---

*Test completed on 386 USDJPY5 trades using models trained on XAUUSD data*

