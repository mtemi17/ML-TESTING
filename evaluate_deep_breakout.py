"""Evaluate the deep breakout model on a trade dataset and compare baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class EvaluationConfig:
    threshold_grid: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RangeWidth"] = df["RangeHigh"] - df["RangeLow"]
    df["RangeMid"] = (df["RangeHigh"] + df["RangeLow"]) / 2
    df["EntryOffset"] = df["EntryPrice"] - df["RangeMid"]
    return df


def load_artifacts(model_dir: Path) -> Dict[str, object]:
    from tensorflow import keras

    artifacts = {}
    artifacts["model"] = keras.models.load_model(model_dir / "deep_breakout_model.keras")
    artifacts["scaler"] = joblib.load(model_dir / "deep_breakout_scaler.pkl")
    artifacts["encoder"] = joblib.load(model_dir / "deep_breakout_encoder.pkl")
    feature_path = model_dir / "deep_breakout_features.csv"
    artifacts["feature_names"] = pd.read_csv(feature_path, header=None)[0].tolist()
    return artifacts


def prepare_matrix(df: pd.DataFrame, artifacts: Dict[str, object]):
    numeric_cols = [
        "EntryPrice",
        "SL",
        "TP",
        "Risk",
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
        "RangeHigh",
        "RangeLow",
        "RangeWidth",
        "RangeMid",
        "EntryOffset",
        "BreakoutDistance",
        "BreakoutBodyPct",
        "BreakoutAtrMultiple",
        "RangeAtrRatio",
        "ATR_Value",
        "PriceAboveEMA200_5M",
        "Price_Above_EMA200_5M",
        "Price_Above_EMA200_1H",
    ]

    categorical_cols = ["EntryType", "Type", "WindowType", "WindowID"]

    for col in categorical_cols + ["WindowID"]:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    df_numeric = df[numeric_cols].copy()
    df_numeric = df_numeric.fillna(df_numeric.median())

    if "PriceAboveEMA200_5M" not in df.columns and "Price_Above_EMA200_5M" in df.columns:
        df["PriceAboveEMA200_5M"] = df["Price_Above_EMA200_5M"]
    if "Price_Above_EMA200_5M" not in df.columns and "PriceAboveEMA200_5M" in df.columns:
        df["Price_Above_EMA200_5M"] = df["PriceAboveEMA200_5M"]

    X_numeric = artifacts["scaler"].transform(df_numeric)

    X_categorical = artifacts["encoder"].transform(df[categorical_cols])

    from scipy import sparse

    X = sparse.hstack([X_numeric, X_categorical]).toarray()
    return X


def evaluate(trades_csv: Path, model_dir: Path, output_dir: Path, config: EvaluationConfig):
    trades_csv = trades_csv.resolve()
    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(trades_csv, parse_dates=["EntryTime", "ExitTime"], infer_datetime_format=True)
    completed = df[df["Status"].isin(["TP_HIT", "SL_HIT"])].copy()
    completed["Target"] = (completed["Status"] == "TP_HIT").astype(int)

    print(f"Loaded {len(completed)} completed trades from {trades_csv.name}")

    completed = engineer_features(completed)
    artifacts = load_artifacts(model_dir)

    X = prepare_matrix(completed, artifacts)
    y_true = completed["Target"].values

    model = artifacts["model"]
    y_probs = model.predict(X, verbose=0).flatten()
    completed["PredictedProb"] = y_probs

    y_pred = (y_probs > 0.5).astype(int)

    print("\nModel Evaluation (threshold=0.5):")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))

    baseline_summary = {
        "TotalTrades": len(completed),
        "WinRate": (completed["Target"].mean()) * 100,
        "TotalPnL": completed["P&L"].sum(),
        "AveragePnL": completed["P&L"].mean(),
    }

    print("\nBaseline Performance:")
    for key, value in baseline_summary.items():
        print(f"  {key}: {value:.3f}")

    threshold_rows: List[Dict[str, object]] = []
    for threshold in config.threshold_grid:
        mask = completed["PredictedProb"] >= threshold
        selected = completed[mask]
        if selected.empty:
            threshold_rows.append({
                "Threshold": threshold,
                "SelectedTrades": 0,
                "WinRate": np.nan,
                "TotalPnL": np.nan,
                "AvgPnL": np.nan,
            })
            continue

        win_rate = selected["Target"].mean() * 100
        total_pnl = selected["P&L"].sum()
        avg_pnl = selected["P&L"].mean()

        threshold_rows.append({
            "Threshold": threshold,
            "SelectedTrades": len(selected),
            "WinRate": win_rate,
            "TotalPnL": total_pnl,
            "AvgPnL": avg_pnl,
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(output_dir / "deep_breakout_thresholds.csv", index=False)
    completed.to_csv(output_dir / "deep_breakout_predictions.csv", index=False)

    print("\nThreshold Analysis:")
    print(threshold_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nDetailed predictions saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deep breakout model")
    parser.add_argument(
        "--trades",
        type=Path,
        default=Path("window_breakout_results.csv"),
        help="Path to trade log CSV",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("deep_breakout_model"),
        help="Directory containing trained model artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deep_breakout_evaluation"),
        help="Directory to store evaluation outputs",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Custom probability thresholds (space separated)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.thresholds:
        config = EvaluationConfig(threshold_grid=tuple(args.thresholds))
    else:
        config = EvaluationConfig()

    evaluate(args.trades, args.model_dir, args.output, config)


if __name__ == "__main__":
    main()


