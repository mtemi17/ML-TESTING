# Deep LSTM Model Architecture - 10-15 Layers

## 🏗️ Model Architecture

### Layer Structure (34 Total Layers)

**LSTM Layers (10 layers):**
1. LSTM(256) → BatchNorm → Dropout(0.3)
2. LSTM(192) → BatchNorm → Dropout(0.3)
3. LSTM(160) → BatchNorm → Dropout(0.25)
4. LSTM(128) → BatchNorm → Dropout(0.25)
5. LSTM(96) → BatchNorm → Dropout(0.2)
6. LSTM(80) → BatchNorm → Dropout(0.2)
7. LSTM(64) → BatchNorm → Dropout(0.2)
8. LSTM(48) → BatchNorm → Dropout(0.15)
9. LSTM(32) → BatchNorm → Dropout(0.15)
10. LSTM(24) → BatchNorm → Dropout(0.1) [Last LSTM, no return_sequences]

**Dense Layers (2 layers):**
11. Dense(32, ReLU) → BatchNorm → Dropout(0.1)
12. Dense(1, Sigmoid) [Output]

**Total: 34 layers** (10 LSTM + 2 Dense + 10 BatchNorm + 10 Dropout + 2 BatchNorm/Dropout for Dense)

---

## 📊 Model Specifications

### Parameters
- **Total Parameters:** 1,225,345
- **Trainable Parameters:** 1,223,121
- **Non-trainable Parameters:** 2,224 (BatchNorm)
- **Model Size:** 4.67 MB

### Input/Output
- **Input Shape:** (batch_size, 15, 20)
  - Sequence length: 15 trades
  - Features per timestep: 20
- **Output Shape:** (batch_size, 1)
  - Binary classification (Win/Loss probability)

### Architecture Details
- **Sequence Length:** 15 trades (increased from 10)
- **Feature Count:** 20 features
- **Units Progression:** 256 → 192 → 160 → 128 → 96 → 80 → 64 → 48 → 32 → 24 → 32 → 1

---

## 🎯 Training Configuration

### Hyperparameters
- **Optimizer:** Adam
- **Initial Learning Rate:** 0.0005 (lowered for deeper network)
- **Loss Function:** Binary Crossentropy
- **Batch Size:** 16 (smaller for stability)
- **Max Epochs:** 200
- **Class Weights:** Balanced (handles class imbalance)

### Regularization
- **Dropout Rates:** 0.1-0.3 (decreasing through layers)
- **Batch Normalization:** After every LSTM and Dense layer
- **Early Stopping:** Patience=30, min_delta=0.0001
- **Learning Rate Reduction:** Factor=0.5, patience=10

### Data Split
- **Training:** 60% (318 sequences)
- **Validation:** 20% (106 sequences)
- **Testing:** 20% (106 sequences)

---

## 🔄 Why Deep Architecture?

### Advantages
1. **Hierarchical Feature Learning:**
   - Early layers capture simple patterns
   - Middle layers combine patterns
   - Later layers learn complex relationships

2. **Better Representation:**
   - More capacity to learn intricate patterns
   - Can capture long-term dependencies
   - Better at handling non-linear relationships

3. **Feature Abstraction:**
   - Each layer builds on previous representations
   - Progressive complexity increase
   - Better generalization potential

### Challenges
1. **Overfitting Risk:**
   - Mitigated by extensive dropout and batch normalization
   - Early stopping prevents overtraining
   - Class weights help with imbalanced data

2. **Training Time:**
   - Deeper networks take longer to train
   - Requires more computational resources
   - May need more epochs to converge

3. **Vanishing Gradients:**
   - Batch normalization helps
   - Proper initialization
   - Residual connections could be added if needed

---

## 📈 Expected Performance

### Compared to Shallow LSTM (3 layers)
- **More Capacity:** 1.2M vs 140K parameters
- **Better Pattern Recognition:** Can learn more complex relationships
- **Longer Sequences:** 15 vs 10 trades lookback
- **Potentially Better Accuracy:** If patterns exist in the data

### Trade-offs
- **Training Time:** ~10-20x longer
- **Memory Usage:** ~4.67 MB vs 0.55 MB
- **Risk of Overfitting:** Higher, but mitigated by regularization

---

## 🚀 Usage

### Training
```bash
python3 lstm_deep_model_gold.py
```

### Model Files
- **Model:** `lstm_deep_model_gold.h5`
- **Features:** `lstm_deep_feature_columns.csv`
- **Parameters:** `lstm_deep_model_params.csv`

### Loading Model
```python
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('lstm_deep_model_gold.h5')
```

---

## 🔍 Architecture Comparison

| Model | Layers | LSTM Layers | Parameters | Sequence Length |
|-------|--------|-------------|------------|-----------------|
| **Shallow LSTM** | 7 | 3 | 140,481 | 10 |
| **Deep LSTM** | 34 | 10 | 1,225,345 | 15 |

**Deep LSTM has:**
- 4.9x more layers
- 3.3x more LSTM layers
- 8.7x more parameters
- 1.5x longer sequences

---

## 💡 Recommendations

### For Best Performance
1. **More Data:** Deep networks benefit from more training data
2. **Longer Training:** Allow more epochs for convergence
3. **Hyperparameter Tuning:** Adjust learning rate, dropout rates
4. **Ensemble:** Combine with Random Forest for best results

### Monitoring
- Watch validation loss for overfitting
- Monitor training vs validation accuracy gap
- Check if early stopping triggers too early
- Adjust patience if needed

---

## 📝 Notes

- Model uses progressive unit reduction (256→24) for efficiency
- Dropout rates decrease through layers (0.3→0.1)
- Batch normalization after every layer for stability
- Lower learning rate (0.0005) for deeper network stability
- Class weights handle imbalanced win/loss ratio

---

*Architecture designed for XAUUSD trading strategy with 545 trades*

