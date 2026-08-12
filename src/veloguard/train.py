from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import resolve_path
from .features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMNS,
    build_training_frame,
    rolling_time_splits,
)
from .replay import simulate_rebalancing_policy


def fit_baseline(frame: pd.DataFrame) -> dict:
    model: dict = {"known_stations": sorted(frame["station_id"].astype(str).unique().tolist())}
    for target in TARGET_COLUMNS:
        model[target] = {
            "primary": frame.groupby(["station_id", "day_of_week", "bucket_of_day"])[target].median().to_dict(),
            "station_bucket": frame.groupby(["station_id", "bucket_of_day"])[target].median().to_dict(),
            "bucket": frame.groupby("bucket_of_day")[target].median().to_dict(),
            "global": float(frame[target].median()),
        }
    return model


def predict_baseline(model: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    for target in TARGET_COLUMNS:
        levels = model[target]
        values = []
        for row in frame[["station_id", "day_of_week", "bucket_of_day"]].itertuples(index=False):
            station_id, day_of_week, bucket = str(row.station_id), str(row.day_of_week), str(row.bucket_of_day)
            value = levels["primary"].get((station_id, day_of_week, bucket))
            if value is None:
                value = levels["station_bucket"].get((station_id, bucket))
            if value is None:
                value = levels["bucket"].get(bucket, levels["global"])
            values.append(value)
        predictions.append(np.asarray(values, dtype=float))
    return predictions[0], predictions[1]


def make_candidate_pipeline(*, random_seed: int) -> Pipeline:
    categorical = Pipeline(
        [
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            )
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical, CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ],
        sparse_threshold=0,
    )
    categorical_mask = [True] * len(CATEGORICAL_COLUMNS) + [False] * len(NUMERIC_COLUMNS)
    estimator = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        random_state=random_seed,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def fit_candidate(frame: pd.DataFrame, *, random_seed: int) -> dict:
    models = {}
    for target in TARGET_COLUMNS:
        pipeline = make_candidate_pipeline(random_seed=random_seed)
        pipeline.fit(frame[FEATURE_COLUMNS], frame[target])
        models[target] = pipeline
    return models


def predict_candidate(models: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    departures = np.clip(models["target_departures"].predict(frame[FEATURE_COLUMNS]), 0, None)
    arrivals = np.clip(models["target_arrivals"].predict(frame[FEATURE_COLUMNS]), 0, None)
    return departures, arrivals


def evaluate_predictions(frame: pd.DataFrame, departures: np.ndarray, arrivals: np.ndarray) -> dict:
    actual_departures = frame["target_departures"].to_numpy(dtype=float)
    actual_arrivals = frame["target_arrivals"].to_numpy(dtype=float)
    dep_mae = mean_absolute_error(actual_departures, departures)
    arr_mae = mean_absolute_error(actual_arrivals, arrivals)
    observed_imbalance = np.abs(actual_departures - actual_arrivals)
    predicted_imbalance = np.abs(departures - arrivals)
    k = max(1, math.ceil(len(frame) * 0.10))
    actual_top = set(np.argpartition(observed_imbalance, -k)[-k:])
    predicted_top = set(np.argpartition(predicted_imbalance, -k)[-k:])
    top_k_recall = len(actual_top & predicted_top) / k
    result = {
        "rows": len(frame),
        "departures_mae": dep_mae,
        "arrivals_mae": arr_mae,
        "combined_mae": (dep_mae + arr_mae) / 2,
        "departures_poisson_deviance": mean_poisson_deviance(actual_departures, np.clip(departures, 1e-6, None)),
        "arrivals_poisson_deviance": mean_poisson_deviance(actual_arrivals, np.clip(arrivals, 1e-6, None)),
        "top_10pct_imbalance_recall": top_k_recall,
    }
    peak = frame["bucket_of_day"].astype(int).between(28, 39) | frame["bucket_of_day"].astype(int).between(64, 79)
    if peak.any() and (~peak).any():
        result["peak_combined_mae"] = (
            mean_absolute_error(actual_departures[peak], departures[peak])
            + mean_absolute_error(actual_arrivals[peak], arrivals[peak])
        ) / 2
        result["offpeak_combined_mae"] = (
            mean_absolute_error(actual_departures[~peak], departures[~peak])
            + mean_absolute_error(actual_arrivals[~peak], arrivals[~peak])
        ) / 2
    return result


def _fold_evaluation(frame: pd.DataFrame, train_index: np.ndarray, valid_index: np.ndarray, seed: int) -> dict:
    train_frame = frame.iloc[train_index]
    valid_frame = frame.iloc[valid_index]
    started = time.perf_counter()
    baseline = fit_baseline(train_frame)
    baseline_metrics = evaluate_predictions(valid_frame, *predict_baseline(baseline, valid_frame))
    candidate = fit_candidate(train_frame, random_seed=seed)
    candidate_metrics = evaluate_predictions(valid_frame, *predict_candidate(candidate, valid_frame))
    improvement = (baseline_metrics["combined_mae"] - candidate_metrics["combined_mae"]) / max(
        baseline_metrics["combined_mae"], 1e-12
    )
    return {
        "train_start": train_frame["timestamp"].min().isoformat(),
        "train_end": train_frame["timestamp"].max().isoformat(),
        "valid_start": valid_frame["timestamp"].min().isoformat(),
        "valid_end": valid_frame["timestamp"].max().isoformat(),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "relative_improvement": improvement,
        "candidate_won": bool(candidate_metrics["combined_mae"] < baseline_metrics["combined_mae"]),
        "fit_seconds": time.perf_counter() - started,
    }


def _quantile_higher(values: np.ndarray, coverage: float) -> float:
    finite_sample_level = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(values, finite_sample_level, method="higher"))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def train_and_evaluate(
    flow: pd.DataFrame,
    config: dict,
    *,
    artifact_path: Path | None = None,
    report_path: Path | None = None,
    data_mode: str = "official",
) -> tuple[dict, dict]:
    data_settings = config["data"]
    training = config["training"]
    frame = build_training_frame(
        flow,
        bucket_minutes=data_settings["bucket_minutes"],
        horizon_minutes=data_settings["horizon_minutes"],
    )
    folds, test_index, test_start = rolling_time_splits(
        frame,
        test_days=training["test_days"],
        validation_days=training["validation_days"],
        max_folds=training["max_folds"],
    )
    fold_results = [
        _fold_evaluation(frame, train_index, valid_index, training["random_seed"] + fold_number)
        for fold_number, (train_index, valid_index) in enumerate(folds)
    ]
    mean_improvement = float(np.mean([fold["relative_improvement"] for fold in fold_results]))
    wins = sum(fold["candidate_won"] for fold in fold_results)
    required_wins = math.ceil(len(fold_results) / 2)
    champion = (
        "candidate"
        if mean_improvement >= training["min_relative_improvement"] and wins >= required_wins
        else "baseline"
    )

    calibration_start = test_start - pd.Timedelta(days=training["validation_days"])
    fit_frame = frame.loc[frame["timestamp"] < calibration_start]
    calibration_frame = frame.loc[(frame["timestamp"] >= calibration_start) & (frame["timestamp"] < test_start)]
    test_frame = frame.iloc[test_index]
    if fit_frame.empty or calibration_frame.empty:
        raise ValueError("Final fit or calibration window is empty")

    baseline = fit_baseline(fit_frame)
    candidate = fit_candidate(fit_frame, random_seed=training["random_seed"])
    predictor = predict_candidate if champion == "candidate" else predict_baseline
    selected = candidate if champion == "candidate" else baseline
    calibration_residuals = {}
    for policy_name, policy_model, policy_predictor in (
        ("baseline", baseline, predict_baseline),
        ("candidate", candidate, predict_candidate),
    ):
        calibration_predictions = policy_predictor(policy_model, calibration_frame)
        calibration_residuals[policy_name] = {
            "departures": _quantile_higher(
                np.abs(calibration_frame["target_departures"].to_numpy() - calibration_predictions[0]),
                training["coverage"],
            ),
            "arrivals": _quantile_higher(
                np.abs(calibration_frame["target_arrivals"].to_numpy() - calibration_predictions[1]),
                training["coverage"],
            ),
        }
    residuals = calibration_residuals[champion]
    baseline_test_predictions = predict_baseline(baseline, test_frame)
    candidate_test_predictions = predict_candidate(candidate, test_frame)
    baseline_test_metrics = evaluate_predictions(test_frame, *baseline_test_predictions)
    candidate_test_metrics = evaluate_predictions(test_frame, *candidate_test_predictions)
    test_predictions = candidate_test_predictions if champion == "candidate" else baseline_test_predictions
    frozen_metrics = evaluate_predictions(test_frame, *test_predictions)
    dep_covered = np.abs(test_frame["target_departures"].to_numpy() - test_predictions[0]) <= residuals["departures"]
    arr_covered = np.abs(test_frame["target_arrivals"].to_numpy() - test_predictions[1]) <= residuals["arrivals"]
    frozen_metrics["departures_interval_coverage"] = float(dep_covered.mean())
    frozen_metrics["arrivals_interval_coverage"] = float(arr_covered.mean())
    frozen_metrics["baseline_combined_mae"] = baseline_test_metrics["combined_mae"]
    frozen_metrics["candidate_combined_mae"] = candidate_test_metrics["combined_mae"]
    frozen_metrics["candidate_relative_improvement"] = (
        baseline_test_metrics["combined_mae"] - candidate_test_metrics["combined_mae"]
    ) / max(baseline_test_metrics["combined_mae"], 1e-12)
    simulation = config.get(
        "simulation", {"station_capacity": 30, "planning_interval_minutes": data_settings["horizon_minutes"]}
    )
    baseline_replay = simulate_rebalancing_policy(
        test_frame,
        *baseline_test_predictions,
        departure_error=calibration_residuals["baseline"]["departures"],
        arrival_error=calibration_residuals["baseline"]["arrivals"],
        planner_settings=config["planner"],
        bucket_minutes=data_settings["bucket_minutes"],
        planning_interval_minutes=simulation["planning_interval_minutes"],
        station_capacity=simulation["station_capacity"],
        policy_name="baseline",
    )
    candidate_replay = simulate_rebalancing_policy(
        test_frame,
        *candidate_test_predictions,
        departure_error=calibration_residuals["candidate"]["departures"],
        arrival_error=calibration_residuals["candidate"]["arrivals"],
        planner_settings=config["planner"],
        bucket_minutes=data_settings["bucket_minutes"],
        planning_interval_minutes=simulation["planning_interval_minutes"],
        station_capacity=simulation["station_capacity"],
        policy_name="candidate",
    )
    baseline_decision = baseline_replay["with_rebalancing"]
    candidate_decision = candidate_replay["with_rebalancing"]
    candidate_dominates = (
        candidate_decision["service_failures"] <= baseline_decision["service_failures"]
        and candidate_decision["bikes_moved"] <= baseline_decision["bikes_moved"]
        and (
            candidate_decision["service_failures"] < baseline_decision["service_failures"]
            or candidate_decision["bikes_moved"] < baseline_decision["bikes_moved"]
        )
    )
    release_status = (
        "approved_for_decision_support"
        if champion == "baseline" or candidate_dominates
        else "shadow_mode_only"
    )

    version_payload = {
        "target_definition": "next-exclusive-buckets-v1",
        "champion": champion,
        "release_status": release_status,
        "data_mode": data_mode,
        "trained_through": str(fit_frame["timestamp"].max()),
        "stations": baseline["known_stations"],
        "config": {"data": data_settings, "training": training},
    }
    model_version = hashlib.sha256(json.dumps(version_payload, sort_keys=True).encode()).hexdigest()[:12]
    station_catalog = (
        frame.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "last"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
        )
        .to_dict(orient="records")
    )
    bundle = {
        "model_version": model_version,
        "target_definition": "next-exclusive-buckets-v1",
        "data_mode": data_mode,
        "champion": champion,
        "release_status": release_status,
        "baseline": baseline,
        "candidate": candidate,
        "residual_quantiles": residuals,
        "calibration_residuals_by_policy": calibration_residuals,
        "coverage_target": training["coverage"],
        "bucket_minutes": data_settings["bucket_minutes"],
        "horizon_minutes": data_settings["horizon_minutes"],
        "trained_through": fit_frame["timestamp"].max().isoformat(),
        "known_stations": baseline["known_stations"],
        "station_catalog": station_catalog,
    }
    report = {
        "project": config["project_name"],
        "model_version": model_version,
        "data_mode": data_mode,
        "target_definition": "next-exclusive-buckets-v1",
        "rows": len(frame),
        "stations": int(frame["station_id"].nunique()),
        "time_range": [frame["timestamp"].min().isoformat(), frame["timestamp"].max().isoformat()],
        "rolling_folds": fold_results,
        "promotion": {
            "champion": champion,
            "mean_relative_improvement": mean_improvement,
            "candidate_wins": wins,
            "required_wins": required_wins,
            "minimum_relative_improvement": training["min_relative_improvement"],
        },
        "final_windows": {
            "fit_end": fit_frame["timestamp"].max().isoformat(),
            "calibration_start": calibration_frame["timestamp"].min().isoformat(),
            "calibration_end": calibration_frame["timestamp"].max().isoformat(),
            "test_start": test_frame["timestamp"].min().isoformat(),
            "test_end": test_frame["timestamp"].max().isoformat(),
        },
        "residual_quantiles": residuals,
        "frozen_test": frozen_metrics,
        "decision_replay": {
            "selected_policy": champion,
            "release_status": release_status,
            "candidate_dominates_baseline": candidate_dominates,
            "release_rule": "Candidate must be no worse on service failures and bike moves, and strictly better on at least one.",
            "baseline": baseline_replay,
            "candidate": candidate_replay,
        },
    }

    artifact_path = artifact_path or resolve_path(config["paths"]["artifact"])
    report_path = report_path or resolve_path(config["paths"]["report"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    report_path.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    return bundle, _json_ready(report)
