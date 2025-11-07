"""Compute feature importance for breakout trade dataset using RandomForest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RangeWidth"] = df.get("RangeWidth") if "RangeWidth" in df.columns else df["RangeHigh"] - df["RangeLow"]
    df["RangeMid"] = df.get("RangeMid") if "RangeMid" in df.columns else (df["RangeHigh"] + df["RangeLow"]) / 2
    df["EntryOffset"] = df.get("EntryOffset") if "EntryOffset" in df.columns else df["EntryPrice"] - df["RangeMid"]

    if "PriceAboveEMA200_5M" not in df.columns and "Price_Above_EMA200_5M" in df.columns:
        df["PriceAboveEMA200_5M"] = df["Price_Above_EMA200_5M"]
    if "Price_Above_EMA200_5M" not in df.columns and "PriceAboveEMA200_5M" in df.columns:
        df["Price_Above_EMA200_5M"] = df["PriceAboveEMA200_5M"]

    return df


def prepare_matrix(df: pd.DataFrame):
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

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    df_numeric = df[numeric_cols].copy()
    df_numeric = df_numeric.fillna(df_numeric.median())

    scaler = RobustScaler()
    X_numeric = scaler.fit_transform(df_numeric)

    encoder = OneHotEncoder(handle_unknown="ignore")
    X_categorical = encoder.fit_transform(df[categorical_cols])

    from scipy import sparse

    X = sparse.hstack([X_numeric, X_categorical]).toarray()
    y = (df["Status"] == "TP_HIT").astype(int).values

    feature_names = numeric_cols + encoder.get_feature_names_out(categorical_cols).tolist()

    return X, y, feature_names


def feature_selection(trades_csv: Path, output_csv: Path, n_estimators: int = 500, random_state: int = 42):
    df = pd.read_csv(trades_csv)
    completed = df[df["Status"].isin(["TP_HIT", "SL_HIT"])].copy()
    completed = engineer_features(completed)

    X, y, feature_names = prepare_matrix(completed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    ranking = pd.DataFrame({"feature": feature_names, "importance": importance})
    ranking.sort_values("importance", ascending=False, inplace=True)

    ranking.to_csv(output_csv, index=False)
    print(ranking.head(25).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFeature importances saved to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature selection for breakout trades")
    parser.add_argument(
        "--trades",
        type=Path,
        default=Path("window_breakout_results.csv"),
        help="Path to trade log CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("breakout_feature_importance.csv"),
        help="Where to save feature importance ranking",
    )
    parser.add_argument("--estimators", type=int, default=500, help="Number of trees")
    return parser.parse_args()


def main():
    args = parse_args()
    feature_selection(args.trades, args.output, n_estimators=args.estimators)


if __name__ == "__main__":
    main()


