"""Train a deep neural network (50-100 layers) on breakout window trades."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    reward_to_risk: float = 2.0
    dense_units: int = 128
    hidden_layers: int = 60  # Between 50 and 100 layers
    dropout_rate: float = 0.1
    batch_size: int = 32
    epochs: int = 150


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["EntryTime", "ExitTime"], infer_datetime_format=True)
    df = df[df["Status"].isin(["TP_HIT", "SL_HIT"])].copy()
    df["Target"] = (df["Status"] == "TP_HIT").astype(int)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, List[str], RobustScaler, OneHotEncoder]:
    feature_cols = [
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

    df["RangeWidth"] = df.get("RangeWidth") if "RangeWidth" in df.columns else df["RangeHigh"] - df["RangeLow"]
    df["RangeMid"] = df.get("RangeMid") if "RangeMid" in df.columns else (df["RangeHigh"] + df["RangeLow"]) / 2
    df["EntryOffset"] = df.get("EntryOffset") if "EntryOffset" in df.columns else df["EntryPrice"] - df["RangeMid"]

    if "PriceAboveEMA200_5M" not in df.columns and "Price_Above_EMA200_5M" in df.columns:
        df["PriceAboveEMA200_5M"] = df["Price_Above_EMA200_5M"]
    if "Price_Above_EMA200_5M" not in df.columns and "PriceAboveEMA200_5M" in df.columns:
        df["Price_Above_EMA200_5M"] = df["PriceAboveEMA200_5M"]

    categorical_cols = ["EntryType", "Type", "WindowType", "WindowID"]

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    df_numeric = df[feature_cols].copy()
    df_numeric = df_numeric.fillna(df_numeric.median())

    scaler = RobustScaler()
    X_numeric = scaler.fit_transform(df_numeric)

    encoder = OneHotEncoder(handle_unknown="ignore")
    X_categorical = encoder.fit_transform(df[categorical_cols])

    from scipy import sparse

    X = sparse.hstack([X_numeric, X_categorical]).toarray()
    y = df["Target"].values

    numeric_feature_names = feature_cols
    categorical_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numeric_feature_names + categorical_feature_names

    return X, y, feature_names, scaler, encoder


def build_model(input_dim: int, config: TrainingConfig):
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for this script.") from exc

    tf.random.set_seed(config.random_state)
    np.random.seed(config.random_state)

    inputs = keras.Input(shape=(input_dim,), name="trade_features")
    x = layers.BatchNormalization()(inputs)

    units = config.dense_units
    for i in range(config.hidden_layers):
        x = layers.Dense(units, kernel_initializer="he_normal", name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        if config.dropout_rate > 0:
            x = layers.Dropout(config.dropout_rate, name=f"dropout_{i}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DeepBreakoutModel")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(csv_path: Path, output_dir: Path, config: TrainingConfig):
    csv_path = csv_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trade dataset from {csv_path}")
    df = load_dataset(csv_path)
    print(f"Loaded {len(df)} completed trades")

    X, y, feature_names, scaler, encoder = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = build_model(X.shape[1], config)
    model.summary(print_fn=lambda line: print(line))

    from tensorflow import keras
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir / "deep_breakout_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    y_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_probs > 0.5).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    metrics_path = output_dir / "deep_breakout_metrics.csv"
    metrics_df = pd.DataFrame({
        "epoch": range(1, len(history.history["loss"]) + 1),
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
    })
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Training metrics saved to {metrics_path}")

    import joblib

    joblib.dump(scaler, output_dir / "deep_breakout_scaler.pkl")
    joblib.dump(encoder, output_dir / "deep_breakout_encoder.pkl")
    pd.Series(feature_names).to_csv(output_dir / "deep_breakout_features.csv", index=False)
    print(f"Artifacts saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deep breakout model")
    parser.add_argument(
        "--trades",
        type=Path,
        default=Path("window_breakout_results.csv"),
        help="Path to trade log CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deep_breakout_model"),
        help="Directory to store model artifacts",
    )
    parser.add_argument("--layers", type=int, default=60, help="Number of hidden layers (50-100)")
    parser.add_argument("--units", type=int, default=128, help="Hidden layer width")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate per layer")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainingConfig(
        hidden_layers=max(50, min(100, args.layers)),
        dense_units=args.units,
        dropout_rate=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    train_and_evaluate(args.trades, args.output, config)


if __name__ == "__main__":
    main()


