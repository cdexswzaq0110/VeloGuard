from __future__ import annotations

import math

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = ["station_id", "bucket_of_day", "day_of_week", "month"]
NUMERIC_COLUMNS = ["is_weekend", "time_sin", "time_cos", "dow_sin", "dow_cos", "latitude", "longitude"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
TARGET_COLUMNS = ["target_departures", "target_arrivals"]


def _forward_sum(values: pd.Series, steps: int) -> pd.Series:
    future_values = [values.shift(-offset) for offset in range(1, steps + 1)]
    return pd.concat(future_values, axis=1).sum(axis=1, min_count=steps)


def add_calendar_features(frame: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"])
    buckets_per_hour = 60 // bucket_minutes
    result["bucket_of_day"] = timestamp.dt.hour * buckets_per_hour + timestamp.dt.minute // bucket_minutes
    result["day_of_week"] = timestamp.dt.dayofweek
    result["month"] = timestamp.dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")
    day_angle = 2 * math.pi * result["bucket_of_day"] / (24 * buckets_per_hour)
    week_angle = 2 * math.pi * result["day_of_week"] / 7
    result["time_sin"] = np.sin(day_angle)
    result["time_cos"] = np.cos(day_angle)
    result["dow_sin"] = np.sin(week_angle)
    result["dow_cos"] = np.cos(week_angle)
    return result


def build_training_frame(flow: pd.DataFrame, *, bucket_minutes: int, horizon_minutes: int) -> pd.DataFrame:
    if horizon_minutes <= 0 or horizon_minutes % bucket_minutes:
        raise ValueError("horizon_minutes must be a positive multiple of bucket_minutes")
    required = {"station_id", "timestamp", "departures", "arrivals", "latitude", "longitude"}
    missing = required - set(flow.columns)
    if missing:
        raise ValueError(f"Flow data is missing columns: {sorted(missing)}")

    steps = horizon_minutes // bucket_minutes
    frame = flow.sort_values(["station_id", "timestamp"]).copy()
    frame["target_departures"] = frame.groupby("station_id", sort=False)["departures"].transform(
        lambda series: _forward_sum(series, steps)
    )
    frame["target_arrivals"] = frame.groupby("station_id", sort=False)["arrivals"].transform(
        lambda series: _forward_sum(series, steps)
    )
    frame = add_calendar_features(frame, bucket_minutes)
    frame = frame.dropna(subset=TARGET_COLUMNS + ["latitude", "longitude"]).copy()
    frame[CATEGORICAL_COLUMNS] = frame[CATEGORICAL_COLUMNS].astype("string")
    return frame.sort_values(["timestamp", "station_id"], ignore_index=True)


def make_live_feature_frame(stations: pd.DataFrame, *, as_of: pd.Timestamp, bucket_minutes: int) -> pd.DataFrame:
    required = {"station_id", "latitude", "longitude"}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"Station snapshot is missing columns: {sorted(missing)}")
    frame = stations.copy()
    frame["timestamp"] = pd.Timestamp(as_of).floor(f"{bucket_minutes}min")
    frame = add_calendar_features(frame, bucket_minutes)
    frame[CATEGORICAL_COLUMNS] = frame[CATEGORICAL_COLUMNS].astype("string")
    return frame


def rolling_time_splits(
    frame: pd.DataFrame,
    *,
    test_days: int,
    validation_days: int,
    max_folds: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, pd.Timestamp]:
    timestamps = pd.to_datetime(frame["timestamp"])
    first = timestamps.min().floor("D")
    last = timestamps.max().ceil("D")
    test_start = last - pd.Timedelta(days=test_days)
    if test_start <= first:
        raise ValueError("Not enough history for the requested frozen test window")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for offset in range(max_folds, 0, -1):
        valid_start = test_start - pd.Timedelta(days=validation_days * offset)
        valid_end = valid_start + pd.Timedelta(days=validation_days)
        train_mask = timestamps < valid_start
        valid_mask = (timestamps >= valid_start) & (timestamps < valid_end)
        if train_mask.any() and valid_mask.any() and valid_start - first >= pd.Timedelta(days=7):
            folds.append((np.flatnonzero(train_mask), np.flatnonzero(valid_mask)))
    if not folds:
        raise ValueError("Not enough history to create a rolling validation fold")
    test_index = np.flatnonzero(timestamps >= test_start)
    if not len(test_index):
        raise ValueError("Frozen test window is empty")
    return folds, test_index, test_start
