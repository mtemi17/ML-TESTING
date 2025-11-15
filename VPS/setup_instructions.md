# VPS Deployment Checklist

This folder contains all assets needed to run the Opening-Range EA with the AI probability filter on a VPS.

## Contents

- `Expert/EAAI.mq5` – Expert Advisor source code
- `Python/ml_service.py` – FastAPI scoring service
- `Python/deep_breakout_model/` – model artifacts (scaler, encoder, model specs, metrics)
- `Python/start_ml_service.sh` – helper script to launch the AI service
- `Python/requirements.txt` – Python dependencies
- `Python/README.md` – setup instructions
- `Python/deep_breakout_model/deep_breakout_scaler.pkl`
- `Python/deep_breakout_model/deep_breakout_encoder.pkl`
- `Python/deep_breakout_model/deep_breakout_model.keras`
- `Python/deep_breakout_model/deep_breakout_metrics.csv`

Copy the entire `VPS` directory to your VPS, then follow the instructions in `Python/README.md`.

