# LSTM Model Report - XAUUSD (Gold)

## 📊 Model Overview

**Model Type:** Long Short-Term Memory (LSTM) Neural Network  
**Purpose:** Predict win/loss for XAUUSD trading strategy  
**Data Split:** 3 batches (60% Train, 20% Validation, 20% Test)  
**Sequence Length:** 10 trades (look back 10 previous trades)

---

## 🏗️ Architecture

### LSTM Layers
- **Layer 1:** 128 units, return_sequences=True
- **Layer 2:** 64 units, return_sequences=True  
- **Layer 3:** 32 units, return_sequences=False
- **Dropout:** 0.2-0.3 between layers
- **Batch Normalization:** Applied after LSTM layers

### Dense Layers
- **Dense 1:** 32 units, ReLU activation
- **Dense 2:** 16 units, ReLU activation
- **Output:** 1 unit, Sigmoid activation (binary classification)

### Training Parameters
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss:** Binary Crossentropy
- **Batch Size:** 32
- **Epochs:** 100 (with early stopping)
- **Class Weights:** Balanced (to handle class imbalance)

**Total Parameters:** 140,481

---

## 📈 Training Results

### Model Performance
- **Training Accuracy:** 65.73%
- **Validation Accuracy:** 59.81%
- **Test Accuracy:** 59.81%

### Training History
- **Best Epoch:** 5 (early stopping at epoch 25)
- **Training Loss:** 0.6297 (best)
- **Validation Loss:** 0.6742 (best)

---

## 🎯 Strategy Performance Comparison

### All 4 Models on XAUUSD (545 trades)

| Model | Trades | Win Rate | Total P&L | Avg P&L | Improvement |
|-------|--------|----------|-----------|---------|-------------|
| **Baseline** | 545 (100%) | 36.51% | $965.92 | $1.77 | - |
| **Random Forest ML** | 151 (27.7%) | **87.42%** | **$2,288.65** | $15.16 | **+136.9%** |
| **Deep Learning** | 46 (8.4%) | 69.57% | $717.25 | **$15.59** | -25.7% |
| **LSTM** | 300 (55.0%) | 39.33% | $841.39 | $2.80 | -12.9% |

### Key Findings

1. **Random Forest ML is the clear winner:**
   - Highest win rate: 87.42%
   - Highest total P&L: $2,288.65
   - Best improvement: +136.9% vs baseline

2. **LSTM Performance:**
   - Takes 55% of trades (moderate filtering)
   - Win rate: 39.33% (slightly better than baseline)
   - Total P&L: $841.39 (worse than baseline)
   - **Not as effective as Random Forest**

3. **Deep Learning:**
   - Very conservative (only 8.4% of trades)
   - Good win rate: 69.57%
   - Best average P&L: $15.59
   - Lower total P&L due to fewer trades

---

## 🔍 Analysis

### Why LSTM Underperformed

1. **Sequence Dependency:**
   - LSTM relies on sequential patterns across 10 trades
   - Trading outcomes may not have strong sequential dependencies
   - Each trade is relatively independent

2. **Data Characteristics:**
   - Only 545 trades total
   - After creating sequences (10 lookback), only 535 sequences available
   - Limited data for deep learning model

3. **Feature Engineering:**
   - Random Forest benefits from feature importance analysis
   - LSTM may need different feature representation
   - Time-based features might not be as relevant

4. **Class Imbalance:**
   - 199 wins vs 346 losses (36.5% win rate)
   - Even with class weights, LSTM struggled to learn winning patterns

### LSTM Strengths

- **Sequential Learning:** Can capture temporal patterns
- **Memory:** Remembers previous trade context
- **Flexibility:** Can learn complex non-linear relationships

### LSTM Weaknesses (for this use case)

- **Data Requirements:** Needs more data for optimal performance
- **Overfitting Risk:** Complex model on limited data
- **Interpretability:** Harder to understand than Random Forest
- **Training Time:** Longer training time

---

## 💡 Recommendations

### For LSTM Model

1. **More Data:**
   - Collect more trades (1000+)
   - Longer sequences might help (20-30 trades)

2. **Feature Engineering:**
   - Add more time-based features
   - Market regime indicators
   - Volatility clusters

3. **Architecture Tuning:**
   - Try different sequence lengths
   - Adjust LSTM units
   - Experiment with attention mechanisms

4. **Ensemble Approach:**
   - Combine LSTM with Random Forest
   - Use LSTM for sequence patterns, RF for feature importance

### Overall Recommendation

**Use Random Forest ML as primary model:**
- Best performance (87.42% win rate)
- Highest profitability ($2,288.65)
- Good balance of trades (27.7%)
- Interpretable feature importance

**LSTM could be useful for:**
- Longer-term pattern recognition
- Market regime detection
- When more data is available

---

## 📁 Files Created

1. **lstm_model_gold.h5** - Trained LSTM model (TensorFlow/Keras format)
2. **lstm_feature_columns.csv** - Feature names used by model
3. **lstm_model_params.csv** - Model parameters (sequence_length, n_features)
4. **all_models_comparison_gold.csv** - Complete comparison table
5. **all_models_predictions_gold.csv** - Detailed predictions for all trades

---

## 🚀 Next Steps

1. **Collect More Data:**
   - Extend backtesting period
   - Include more market conditions

2. **Hybrid Approach:**
   - Use Random Forest for filtering
   - Use LSTM for market regime detection
   - Combine predictions

3. **Real-Time Testing:**
   - Test LSTM on live data
   - Monitor performance
   - Adjust threshold dynamically

---

## 📊 Conclusion

The LSTM model was successfully trained and tested, but **Random Forest ML remains the best performing model** for this XAUUSD trading strategy. The LSTM shows promise but needs more data and tuning to match Random Forest's performance.

**Winner: Random Forest ML** 🏆
- 87.42% win rate
- $2,288.65 total P&L
- 136.9% improvement over baseline

---

*Report generated: November 6, 2025*

