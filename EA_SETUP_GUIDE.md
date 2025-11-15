# Opening Range EA + AI Integration Guide

Use this checklist tomorrow (or anytime) to rebuild the full pipeline.

---

## 1. Refresh trade data & metrics

```bash
python3 strategy_backtest.py
```
- regenerates `window_breakout_results.csv`
- includes 5 m EMA-200 flags and breakout-quality metrics

## 2. Feature selection (optional sanity check)

```bash
python3 breakout_feature_selection.py
```
- outputs `breakout_feature_importance.csv`
- top drivers: EntryOffset, Risk, ATR metrics, EMA-200

## 3. Train the deep model

```bash
python3 deep_breakout_model.py \
  --layers 60 --units 128 --dropout 0.1 \
  --epochs 120 --batch-size 32 \
  --output deep_breakout_model
```
- saves model, scaler, encoder, feature list to `deep_breakout_model/`

## 4. Evaluate thresholds

```bash
python3 evaluate_deep_breakout.py \
  --trades window_breakout_results.csv \
  --model-dir deep_breakout_model \
  --output deep_breakout_evaluation
```
- `deep_breakout_evaluation/deep_breakout_thresholds.csv` shows win rate & P&L per probability cutoff
- use this to pick EA threshold (e.g. p ≥ 0.5)

## 5. Run the ML scoring service

```bash
pip install fastapi uvicorn tensorflow pandas numpy scipy scikit-learn joblib
python3 ml_service.py
```
- service listens on `http://127.0.0.1:8001/predict`
- converts `.keras` → `.tflite` automatically if needed

Example request:
```bash
curl -X POST http://127.0.0.1:8001/predict \
     -H "Content-Type: application/json" \
     -d '{"features": {"EntryPrice": 2650, "SL": 2645, "TP": 2660, "Risk": 5}}'
```

## 6. MT5 Expert Advisor (manual coding steps)

1. Create `Experts/OpeningRangeBreakoutEA.mq5`.
2. Implement:
   - 15‑minute windows (03:00, 10:00, 16:30) & 3‑hour trade window.
   - Pullback + reversal entries aligned with EMA filters.
   - Risk sizing by % (`RiskPercent` input).
   - One trade per window, 2R stop/target, EMA filter exit.
3. Add `UseMLFilter`:
   - Gather feature vector (match `NUMERIC_FEATURES` + `CATEGORICAL_FEATURES` in `ml_service.py`).
   - Call AI service (file-based or socket). Example payload:
     ```json
     {
       "EntryPrice": 2650,
       "SL": 2645,
       "TP": 2660,
       "Risk": 5,
       "EMA_200_5M": ...,
       "BreakoutDistance": ...,
       "EntryType": "PULLBACK",
       "WindowType": "0300",
       "WindowID": "2025-01-07_0300",
       ...
     }
     ```
   - Only place trade if `prob >= ProbabilityThreshold`.

## 7. Test workflow

1. Backtest EA with `UseMLFilter = false` (should match Python stats).
2. Launch `ml_service.py`.
3. Attach EA on demo chart (`UseMLFilter = true`, set threshold).
4. Verify request/response logs; adjust as needed.

Keep this guide and re-run steps whenever you update the model or strategy. Save this file somewhere safe.


