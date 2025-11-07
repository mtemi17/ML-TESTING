"""Backtest a breakout-pullback strategy on 5-minute XAUUSD data.

The strategy logic:
1. Identify a rolling range over the last `range_lookback` candles.
2. Detect a breakout when price closes outside the range.
3. Wait for price to pull back to the breached edge of the range (entry level).
4. Enter a limit order at the range boundary with stop at the opposite edge.
5. Target is set to achieve a 2R reward-to-risk.

Results are exported to `breakout_pullback_trades.csv` and a quick
performance summary is printed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class StrategyConfig:
    range_lookback: int = 12  # 1 hour window on 5-minute data
    pullback_timeout: int = 12  # cancel setup if no pullback within N candles
    min_range_width: float = 1.0  # in USD; skip too narrow ranges
    tolerance: float = 0.0  # optional buffer above/below range for breakouts
    reward_to_risk: float = 2.0
    consolidation_threshold: float = 5.0  # treat tight ranges as consolidation
    use_ema_filter: bool = True
    allow_pullback: bool = True
    allow_reversal: bool = True
    allow_breakout: bool = False


@dataclass
class TradeRecord:
    direction: str
    breakout_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    result: str
    entry_type: str
    pnl: float
    rr_multiple: float
    range_high: float
    range_low: float
    range_width: float
    breakout_close: float
    ema_value: float


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load XAUUSD 5-minute data and return as a DataFrame with DateTime index."""

    df = pd.read_csv(
        csv_path,
        names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
    )

    timestamp = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%Y.%m.%d %H:%M",
    )

    df = df.drop(columns=["Date", "Time"])
    df.insert(0, "Timestamp", timestamp)
    df = df.set_index("Timestamp")
    df = df.sort_index()

    hourly_close = df["Close"].resample("1h").last()
    ema200_1h = hourly_close.ewm(span=200, adjust=False).mean()
    df["EMA200_1H"] = ema200_1h.reindex(df.index, method="ffill")
    return df


def backtest_breakout_pullback(df: pd.DataFrame, config: StrategyConfig) -> List[TradeRecord]:
    """Run the breakout-pullback backtest under the supplied configuration."""

    trades: List[TradeRecord] = []

    state: str = "idle"
    setup_info: Optional[dict] = None
    entry_info: Optional[dict] = None

    def reset_state() -> None:
        nonlocal state, setup_info, entry_info
        state = "idle"
        setup_info = None
        entry_info = None

    for idx in range(config.range_lookback, len(df)):
        row = df.iloc[idx]
        timestamp = row.name
        ema_value = row.get("EMA200_1H")

        window = df.iloc[idx - config.range_lookback : idx]
        range_high = window["High"].max()
        range_low = window["Low"].min()
        range_width = range_high - range_low

        if range_width < config.min_range_width or range_width <= config.consolidation_threshold:
            reset_state()
            continue

        if config.use_ema_filter and pd.isna(ema_value):
            reset_state()
            continue

        if not config.use_ema_filter and pd.isna(ema_value):
            # If the filter is disabled we still want a numeric value for logging
            ema_value = row["Close"]

        trend_bullish = True if not config.use_ema_filter else row["Close"] >= ema_value
        trend_bearish = True if not config.use_ema_filter else row["Close"] <= ema_value

        breakout_up = row["Close"] > range_high + config.tolerance
        breakout_down = row["Close"] < range_low - config.tolerance

        if state == "idle":
            # Immediate breakout entries (baseline mode)
            if config.allow_breakout:
                if breakout_up:
                    entry_price = float(row["Close"])
                    stop_price = float(range_low)
                    risk = entry_price - stop_price
                    if risk > 0:
                        entry_info = {
                            "direction": "LONG",
                            "entry_time": timestamp,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "target_price": entry_price + config.reward_to_risk * risk,
                            "range_high": float(range_high),
                            "range_low": float(range_low),
                            "range_width": float(range_width),
                            "breakout_time": timestamp,
                            "breakout_close": entry_price,
                            "entry_type": "BREAKOUT",
                            "entry_ema_value": float(ema_value),
                        }
                        state = "in_long"
                        continue
                if breakout_down:
                    entry_price = float(row["Close"])
                    stop_price = float(range_high)
                    risk = stop_price - entry_price
                    if risk > 0:
                        entry_info = {
                            "direction": "SHORT",
                            "entry_time": timestamp,
                            "entry_price": entry_price,
                            "stop_price": stop_price,
                            "target_price": entry_price - config.reward_to_risk * risk,
                            "range_high": float(range_high),
                            "range_low": float(range_low),
                            "range_width": float(range_width),
                            "breakout_time": timestamp,
                            "breakout_close": entry_price,
                            "entry_type": "BREAKOUT",
                            "entry_ema_value": float(ema_value),
                        }
                        state = "in_short"
                        continue

            # Pullback logic
            if config.allow_pullback:
                if breakout_up and trend_bullish:
                    state = "waiting_pullback_long"
                    setup_info = {
                        "direction": "LONG",
                        "breakout_time": timestamp,
                        "breakout_close": float(row["Close"]),
                        "range_high": float(range_high),
                        "range_low": float(range_low),
                        "range_width": float(range_width),
                        "bars_waited": 0,
                        "entry_type": "PULLBACK",
                        "entry_ema": float(ema_value),
                    }
                elif breakout_down and trend_bearish:
                    state = "waiting_pullback_short"
                    setup_info = {
                        "direction": "SHORT",
                        "breakout_time": timestamp,
                        "breakout_close": float(row["Close"]),
                        "range_high": float(range_high),
                        "range_low": float(range_low),
                        "range_width": float(range_width),
                        "bars_waited": 0,
                        "entry_type": "PULLBACK",
                        "entry_ema": float(ema_value),
                    }

            # Reversal logic
            if config.allow_reversal and state == "idle":
                if breakout_down and trend_bullish:
                    state = "waiting_reversal_long"
                    setup_info = {
                        "direction": "LONG",
                        "breakout_time": timestamp,
                        "breakout_close": float(row["Close"]),
                        "range_high": float(range_high),
                        "range_low": float(range_low),
                        "range_width": float(range_width),
                        "bars_waited": 0,
                        "first_candle": None,
                        "entry_type": "REVERSAL",
                    }
                elif breakout_up and trend_bearish:
                    state = "waiting_reversal_short"
                    setup_info = {
                        "direction": "SHORT",
                        "breakout_time": timestamp,
                        "breakout_close": float(row["Close"]),
                        "range_high": float(range_high),
                        "range_low": float(range_low),
                        "range_width": float(range_width),
                        "bars_waited": 0,
                        "first_candle": None,
                        "entry_type": "REVERSAL",
                    }

        elif state in {"waiting_pullback_long", "waiting_pullback_short"}:
            assert setup_info is not None
            setup_info["bars_waited"] += 1

            if setup_info["bars_waited"] > config.pullback_timeout:
                reset_state()
                continue

            if config.use_ema_filter:
                if state == "waiting_pullback_long" and not trend_bullish:
                    reset_state()
                    continue
                if state == "waiting_pullback_short" and not trend_bearish:
                    reset_state()
                    continue

            if state == "waiting_pullback_long":
                entry_level = setup_info["range_high"]
                if row["Low"] <= entry_level <= row["High"]:
                    risk = entry_level - setup_info["range_low"]
                    if risk <= 0:
                        reset_state()
                        continue
                    entry_info = {
                        "direction": "LONG",
                        "entry_time": timestamp,
                        "entry_price": float(entry_level),
                        "stop_price": float(setup_info["range_low"]),
                        "target_price": float(entry_level + config.reward_to_risk * risk),
                        "range_high": setup_info["range_high"],
                        "range_low": setup_info["range_low"],
                        "range_width": setup_info["range_width"],
                        "breakout_time": setup_info["breakout_time"],
                        "breakout_close": setup_info["breakout_close"],
                        "entry_type": setup_info["entry_type"],
                        "entry_ema_value": setup_info.get("entry_ema", float(ema_value)),
                    }
                    state = "in_long"
                    setup_info = None
            else:
                entry_level = setup_info["range_low"]
                if row["Low"] <= entry_level <= row["High"]:
                    risk = setup_info["range_high"] - entry_level
                    if risk <= 0:
                        reset_state()
                        continue
                    entry_info = {
                        "direction": "SHORT",
                        "entry_time": timestamp,
                        "entry_price": float(entry_level),
                        "stop_price": float(setup_info["range_high"]),
                        "target_price": float(entry_level - config.reward_to_risk * risk),
                        "range_high": setup_info["range_high"],
                        "range_low": setup_info["range_low"],
                        "range_width": setup_info["range_width"],
                        "breakout_time": setup_info["breakout_time"],
                        "breakout_close": setup_info["breakout_close"],
                        "entry_type": setup_info["entry_type"],
                        "entry_ema_value": setup_info.get("entry_ema", float(ema_value)),
                    }
                    state = "in_short"
                    setup_info = None

        elif state in {"waiting_reversal_long", "waiting_reversal_short"}:
            assert setup_info is not None
            setup_info["bars_waited"] += 1

            if setup_info["bars_waited"] > config.pullback_timeout:
                reset_state()
                continue

            first_candle = setup_info.get("first_candle")

            if state == "waiting_reversal_short":
                if first_candle is None:
                    if row["Close"] < row["Open"]:
                        setup_info["first_candle"] = {
                            "open": float(row["Open"]),
                            "close": float(row["Close"]),
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                        }
                else:
                    ema_condition = True if not config.use_ema_filter else row["Close"] <= ema_value
                    if row["Close"] < row["Open"] and ema_condition:
                        first_mid = (first_candle["open"] + first_candle["close"]) / 2
                        second_mid = (row["Open"] + row["Close"]) / 2
                        entry_price = (first_mid + second_mid) / 2
                        stop_price = first_candle["high"]
                        risk = stop_price - entry_price
                        if risk <= 0:
                            reset_state()
                            continue
                        target_price = entry_price - config.reward_to_risk * risk
                        entry_info = {
                            "direction": "SHORT",
                            "entry_time": timestamp,
                            "entry_price": float(entry_price),
                            "stop_price": float(stop_price),
                            "target_price": float(target_price),
                            "range_high": setup_info["range_high"],
                            "range_low": setup_info["range_low"],
                            "range_width": setup_info["range_width"],
                            "breakout_time": setup_info["breakout_time"],
                            "breakout_close": setup_info["breakout_close"],
                            "entry_type": setup_info["entry_type"],
                            "entry_ema_value": float(ema_value),
                        }
                        state = "in_short"
                        setup_info = None
                    elif row["Close"] >= row["Open"]:
                        setup_info["first_candle"] = None

            else:  # waiting_reversal_long
                if first_candle is None:
                    if row["Close"] > row["Open"]:
                        setup_info["first_candle"] = {
                            "open": float(row["Open"]),
                            "close": float(row["Close"]),
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                        }
                else:
                    ema_condition = True if not config.use_ema_filter else row["Close"] >= ema_value
                    if row["Close"] > row["Open"] and ema_condition:
                        first_mid = (first_candle["open"] + first_candle["close"]) / 2
                        second_mid = (row["Open"] + row["Close"]) / 2
                        entry_price = (first_mid + second_mid) / 2
                        stop_price = first_candle["low"]
                        risk = entry_price - stop_price
                        if risk <= 0:
                            reset_state()
                            continue
                        target_price = entry_price + config.reward_to_risk * risk
                        entry_info = {
                            "direction": "LONG",
                            "entry_time": timestamp,
                            "entry_price": float(entry_price),
                            "stop_price": float(stop_price),
                            "target_price": float(target_price),
                            "range_high": setup_info["range_high"],
                            "range_low": setup_info["range_low"],
                            "range_width": setup_info["range_width"],
                            "breakout_time": setup_info["breakout_time"],
                            "breakout_close": setup_info["breakout_close"],
                            "entry_type": setup_info["entry_type"],
                            "entry_ema_value": float(ema_value),
                        }
                        state = "in_long"
                        setup_info = None
                    elif row["Close"] <= row["Open"]:
                        setup_info["first_candle"] = None

        elif state in {"in_long", "in_short"}:
            assert entry_info is not None

            filter_trigger = False
            if config.use_ema_filter:
                if state == "in_long" and not trend_bullish:
                    filter_trigger = True
                if state == "in_short" and not trend_bearish:
                    filter_trigger = True

            if filter_trigger:
                exit_price = float(row["Close"])
                result = "FILTER_EXIT"
            else:
                if state == "in_long":
                    stop_hit = row["Low"] <= entry_info["stop_price"]
                    target_hit = row["High"] >= entry_info["target_price"]

                    if stop_hit:
                        exit_price = entry_info["stop_price"]
                        result = "LOSS"
                    elif target_hit:
                        exit_price = entry_info["target_price"]
                        result = "WIN"
                    else:
                        continue

                else:  # in_short
                    stop_hit = row["High"] >= entry_info["stop_price"]
                    target_hit = row["Low"] <= entry_info["target_price"]

                    if stop_hit:
                        exit_price = entry_info["stop_price"]
                        result = "LOSS"
                    elif target_hit:
                        exit_price = entry_info["target_price"]
                        result = "WIN"
                    else:
                        continue

            direction = entry_info["direction"]
            if direction == "LONG":
                pnl = exit_price - entry_info["entry_price"]
                risk = entry_info["entry_price"] - entry_info["stop_price"]
            else:
                pnl = entry_info["entry_price"] - exit_price
                risk = entry_info["stop_price"] - entry_info["entry_price"]

            rr_multiple = pnl / risk if risk != 0 else 0.0

            trades.append(
                TradeRecord(
                    direction=direction,
                    breakout_time=entry_info.get("breakout_time", timestamp),
                    entry_time=entry_info["entry_time"],
                    exit_time=timestamp,
                    entry_price=float(entry_info["entry_price"]),
                    stop_price=float(entry_info["stop_price"]),
                    target_price=float(entry_info["target_price"]),
                    exit_price=float(exit_price),
                    result=result,
                    entry_type=entry_info.get("entry_type", "UNKNOWN"),
                    pnl=float(pnl),
                    rr_multiple=float(rr_multiple),
                    range_high=float(entry_info.get("range_high", range_high)),
                    range_low=float(entry_info.get("range_low", range_low)),
                    range_width=float(entry_info.get("range_width", range_width)),
                    breakout_close=float(entry_info.get("breakout_close", row["Close"])),
                    ema_value=float(entry_info.get("entry_ema_value", ema_value)),
                )
            )

            reset_state()

    return trades


def trades_to_dataframe(trades: List[TradeRecord]) -> pd.DataFrame:
    return pd.DataFrame([trade.__dict__ for trade in trades])


def summarize_trades(df_trades: pd.DataFrame) -> None:
    if df_trades.empty:
        print("No trades were generated with the current configuration.")
        return

    wins = (df_trades["result"] == "WIN").sum()
    losses = (df_trades["result"] == "LOSS").sum()
    filter_exits = (df_trades["result"] == "FILTER_EXIT").sum()
    total = len(df_trades)
    counted = wins + losses

    print("\n=== Breakout Pullback Strategy Summary ===")
    print(f"Total trades: {total}")
    print(f"Wins: {wins} ({wins / total * 100:.1f}%)")
    print(f"Losses: {losses} ({losses / total * 100:.1f}%)")
    if filter_exits:
        print(f"Filter exits: {filter_exits} ({filter_exits / total * 100:.1f}%)")
    if counted:
        print(f"Win rate (excluding filter exits): {wins / counted * 100:.1f}%")
    print(f"Average P&L (USD): {df_trades['pnl'].mean():.2f}")
    print(f"Total P&L (USD): {df_trades['pnl'].sum():.2f}")
    print(f"Average R multiple: {df_trades['rr_multiple'].mean():.2f}")
    print(f"Median range width: {df_trades['range_width'].median():.2f}")
    entry_counts = df_trades["entry_type"].value_counts()
    if not entry_counts.empty:
        print("Entries by type:")
        for entry_type, count in entry_counts.items():
            share = count / total * 100
            print(f"  {entry_type}: {count} ({share:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breakout pullback backtest")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("XAUUSD5 new data.csv"),
        help="Path to the 5-minute OHLC CSV",
    )
    parser.add_argument("--range-lookback", type=int, default=12)
    parser.add_argument("--pullback-timeout", type=int, default=12)
    parser.add_argument("--min-range-width", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--reward-to-risk", type=float, default=2.0)
    parser.add_argument("--consolidation-threshold", type=float, default=5.0)
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "pullback",
            "pullback_ema",
            "pullback_ema_reversal",
            "reversal_only",
        ],
        default="pullback_ema_reversal",
        help="Preset configuration for strategy logic",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("breakout_pullback_trades.csv"),
        help="CSV file for trade log",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = StrategyConfig(
        range_lookback=args.range_lookback,
        pullback_timeout=args.pullback_timeout,
        min_range_width=args.min_range_width,
        tolerance=args.tolerance,
        reward_to_risk=args.reward_to_risk,
        consolidation_threshold=args.consolidation_threshold,
    )

    if args.mode == "baseline":
        config.use_ema_filter = False
        config.allow_pullback = False
        config.allow_reversal = False
        config.allow_breakout = True
    elif args.mode == "pullback":
        config.use_ema_filter = False
        config.allow_pullback = True
        config.allow_reversal = False
        config.allow_breakout = False
    elif args.mode == "pullback_ema":
        config.use_ema_filter = True
        config.allow_pullback = True
        config.allow_reversal = False
        config.allow_breakout = False
    elif args.mode == "pullback_ema_reversal":
        config.use_ema_filter = True
        config.allow_pullback = True
        config.allow_reversal = True
        config.allow_breakout = False
    elif args.mode == "reversal_only":
        config.use_ema_filter = True
        config.allow_pullback = False
        config.allow_reversal = True
        config.allow_breakout = False

    print("Loading data...")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}")
    print(f"Mode: {args.mode}")

    print("Running backtest...")
    trades = backtest_breakout_pullback(df, config)
    trades_df = trades_to_dataframe(trades)

    trades_df.to_csv(args.output, index=False)
    summarize_trades(trades_df)
    print(f"\nTrade log saved to {args.output}")


if __name__ == "__main__":
    main()


