"""Lightweight Python service to score breakout trades with the gradient model.

This script loads the scikit-learn pipeline (RobustScaler + OneHotEncoder +
HistGradientBoostingClassifier) produced during the training step, then exposes
an HTTP API that MT5 can call in real time.

Endpoint:
    POST /predict
    body  : JSON with feature dictionary
    reply : {"probability": 0.63}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

ARTIFACT_DIR = Path("gradient_model_multi")


@dataclass
class Artifacts:
    pipeline: object


class PredictRequest(BaseModel):
    features: Dict[str, float]


def load_artifacts(directory: Path) -> Artifacts:
    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Artifact directory not found: {directory}")

    pipeline_path = directory / "hist_gradient_selector.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline model not found: {pipeline_path}")

    pipeline = joblib.load(pipeline_path)
    return Artifacts(pipeline=pipeline)


def prepare_sample(
    artifacts: Artifacts,
    request_features: Dict[str, float],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    payload = {}
    for col in numeric_cols:
        value = request_features.get(col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            payload[col] = 0.0
        else:
            try:
                payload[col] = float(value)
            except (TypeError, ValueError):
                payload[col] = 0.0

    for col in categorical_cols:
        value = request_features.get(col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            payload[col] = "UNKNOWN"
        else:
            payload[col] = str(value)

    df = pd.DataFrame([payload], columns=numeric_cols + categorical_cols)
    return df


def run_inference(pipeline, sample: pd.DataFrame) -> float:
    proba = pipeline.predict_proba(sample)[0, 1]
    return float(proba)


NUMERIC_FEATURES = [
    "EntryPrice",
    "SL",
    "TP",
    "Risk",
    "BreakoutDistance",
    "BreakoutBodyPct",
    "BreakoutAtrMultiple",
    "RangeWidth",
    "RangeMid",
    "EntryOffset",
    "RangeAtrRatio",
    "PriceAboveEMA200_5M",
    "ATR_Value",
    "EMA_9_5M",
    "EMA_21_5M",
    "EMA_50_5M",
    "EMA_200_5M",
    "EMA_200_1H",
    "ATR",
    "ATR_Pct",
    "ATR_Ratio",
    "Consolidation_Score",
    "Trend_Score",
    "Is_Consolidating",
    "Is_Tight_Range",
    "Price_Above_EMA200_5M",
    "Price_Above_EMA200_1H",
]

CATEGORICAL_FEATURES = ["EntryType", "Type", "WindowType", "WindowID", "Mode", "Market"]


artifacts = load_artifacts(ARTIFACT_DIR)
app = FastAPI()


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        sample = prepare_sample(
            artifacts,
            request.features,
            numeric_cols=NUMERIC_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )
        probability = run_inference(artifacts.pipeline, sample)
        return {"probability": probability}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)


