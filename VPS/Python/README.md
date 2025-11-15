# Python ML Service Setup (VPS)

## 1. Prerequisites
- Python 3.9+ installed on the VPS
- Internet access to install dependencies

## 2. Installation
```bash
cd /path/to/VPS/Python
chmod +x start_ml_service.sh
./start_ml_service.sh
```
This creates a virtual environment, installs packages from `requirements.txt`, and launches the FastAPI service on port 8001.

## 3. Manual Start (optional)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python ml_service.py
```

## 4. Service Endpoint
The API listens on `http://localhost:8001/predict`.
Sample request:
```bash
curl -X POST http://127.0.0.1:8001/predict \
     -H "Content-Type: application/json" \
     -d '{"features":{"EntryPrice":1900,"SL":1895,"TP":1910,"Risk":5}}'
```

## 5. Keeping the service alive
Run with `screen`/`tmux` or create a systemd service pointing to `ml_service.py`. Ensure the working directory is this Python folder so the model artifacts are found.

