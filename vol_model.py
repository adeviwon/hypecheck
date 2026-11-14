"""Compare next-day price range forecasts on the complete five-coin panel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import ROOT, split_dates


def date_block_interval(dates, delta):
    series = pd.DataFrame({"date": dates, "delta": delta}).groupby("date").delta.mean().to_numpy()
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(1000):
        starts = rng.integers(0, len(series), size=int(np.ceil(len(series) / 7)))
        sample = np.concatenate([series[(start + np.arange(7)) % len(series)] for start in starts])[:len(series)]
        draws.append(float(sample.mean()))
    return {"delta": float(series.mean()), "interval95": np.quantile(draws, [0.025, 0.975]).tolist()}


def evaluate_volatility(frame):
    frame = frame.sort_values(["coin", "date"]).copy()
    if frame.duplicated(["coin", "date"]).any():
        raise ValueError("Duplicate coin-days")
    frame["target_range"] = (frame.high - frame.low) / frame.open
    frame["previous_range"] = frame.groupby("coin").target_range.shift(1)
    for days in (3, 7):
        frame[f"count_{days}d"] = frame.groupby("coin").post_count.transform(
            lambda count: count.rolling(days, min_periods=days).sum()
        )
    if not frame.groupby("coin").date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Missing calendar days in the market panel")
    train, _, test = split_dates(frame)
    counts = ["post_count", "count_3d", "count_7d"]
    market = ["previous_range", "vol7", "hourly_vol1"]
    train = train.dropna(subset=market + counts)
    actual = test.target_range.to_numpy()
    predictions = {
        "Training median": np.full(len(test), train.target_range.median()),
        "Previous daily range": test.previous_range.to_numpy(),
    }
    for name, columns in (("Market only", market), ("Market + post count", market + counts)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train[columns], train.target_range)
        predictions[name] = model.predict(test[columns])
    rows = {}
    for name, predicted in predictions.items():
        rho = spearmanr(actual, predicted).statistic if np.ptp(predicted) else None
        rows[name] = {"mae": float(np.abs(actual - predicted).mean()),
                      "spearman": None if rho is None else float(rho)}
    delta = np.abs(actual - predictions["Market + post count"]) - np.abs(actual - predictions["Market only"])
    return {
        "target": "(high - low) / open during the UTC day after the feature window",
        "test_coin_days": len(test),
        "test_dates": [test.date.min().date().isoformat(), test.date.max().date().isoformat()],
        "with_observed_posts": int(test.post_count.gt(0).sum()),
        "metrics": rows,
        "post_count_minus_market_mae": date_block_interval(test.date, delta),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/features.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.features)
    frame["date"] = pd.to_datetime(frame.date, utc=True)
    print(json.dumps(evaluate_volatility(frame), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
