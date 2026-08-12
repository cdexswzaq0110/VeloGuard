from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd

from .features import make_live_feature_frame
from .train import predict_baseline, predict_candidate
from .planner import haversine_km


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    required = {"model_version", "champion", "baseline", "candidate", "residual_quantiles"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model artifact is missing fields: {sorted(missing)}")
    return bundle


def _normalize_station_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def align_live_stations(
    bundle: dict,
    snapshot: pd.DataFrame,
    *,
    max_coordinate_distance_km: float = 0.15,
    max_name_distance_km: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    catalog = pd.DataFrame(bundle.get("station_catalog", []))
    if catalog.empty:
        raise ValueError("Model artifact has no station catalog")
    catalog["station_id"] = catalog["station_id"].astype(str)
    catalog["normalized_name"] = catalog["station_name"].map(_normalize_station_name)
    live = snapshot.copy()
    live["station_id"] = live["station_id"].astype(str)
    live["normalized_name"] = live["station_name"].map(_normalize_station_name)

    known_ids = set(catalog["station_id"])
    unique_names = catalog.groupby("normalized_name").filter(lambda group: len(group) == 1)
    name_to_id = dict(zip(unique_names["normalized_name"], unique_names["station_id"]))
    catalog_by_id = catalog.set_index("station_id")
    assignments = []
    used_model_ids: set[str] = set()
    for row in live.sort_values("station_id").itertuples(index=False):
        live_id = str(row.station_id)
        model_id = None
        method = None
        distance_km = None
        if live_id in known_ids and live_id not in used_model_ids:
            model_id, method, distance_km = live_id, "station_id", 0.0
        else:
            name_match = name_to_id.get(row.normalized_name)
            if name_match and name_match not in used_model_ids:
                historical = catalog_by_id.loc[name_match]
                name_distance_km = haversine_km(
                    float(row.latitude), float(row.longitude), float(historical.latitude), float(historical.longitude)
                )
                if name_distance_km <= max_name_distance_km:
                    model_id, method, distance_km = name_match, "normalized_name", name_distance_km
            if model_id is None:
                candidates = []
                for historical in catalog.itertuples(index=False):
                    historical_id = str(historical.station_id)
                    if historical_id in used_model_ids:
                        continue
                    distance = haversine_km(
                        float(row.latitude),
                        float(row.longitude),
                        float(historical.latitude),
                        float(historical.longitude),
                    )
                    if distance <= max_coordinate_distance_km:
                        candidates.append((distance, historical_id))
                if candidates:
                    distance_km, model_id = min(candidates, key=lambda item: (item[0], item[1]))
                    method = "coordinate"
        if model_id is not None:
            used_model_ids.add(model_id)
            assignments.append(
                {
                    "live_station_id": live_id,
                    "model_station_id": model_id,
                    "alignment_method": method,
                    "alignment_distance_km": round(float(distance_km), 4),
                }
            )

    assignment_frame = pd.DataFrame(assignments)
    if assignment_frame.empty:
        return live.iloc[0:0], {
            "system_station_count": len(live),
            "covered_station_count": 0,
            "coverage": 0.0,
            "alignment_methods": {},
        }
    aligned = live.merge(assignment_frame, left_on="station_id", right_on="live_station_id", validate="one_to_one")
    aligned["station_id"] = aligned["model_station_id"]
    method_counts = aligned["alignment_method"].value_counts().to_dict()
    return aligned, {
        "system_station_count": len(live),
        "covered_station_count": len(aligned),
        "coverage": len(aligned) / max(len(live), 1),
        "alignment_methods": {str(key): int(value) for key, value in method_counts.items()},
    }


def predict_flows(bundle: dict, feature_frame: pd.DataFrame) -> pd.DataFrame:
    baseline_departures, baseline_arrivals = predict_baseline(bundle["baseline"], feature_frame)
    known = feature_frame["station_id"].astype(str).isin(bundle["known_stations"]).to_numpy()
    if bundle["champion"] == "candidate":
        departures, arrivals = predict_candidate(bundle["candidate"], feature_frame)
        departures = np.where(known, departures, baseline_departures)
        arrivals = np.where(known, arrivals, baseline_arrivals)
    else:
        departures, arrivals = baseline_departures, baseline_arrivals

    dep_error = bundle["residual_quantiles"]["departures"]
    arr_error = bundle["residual_quantiles"]["arrivals"]
    return pd.DataFrame(
        {
            "station_id": feature_frame["station_id"].astype(str).to_numpy(),
            "predicted_departures": departures,
            "predicted_arrivals": arrivals,
            "departures_lower": np.clip(departures - dep_error, 0, None),
            "departures_upper": departures + dep_error,
            "arrivals_lower": np.clip(arrivals - arr_error, 0, None),
            "arrivals_upper": arrivals + arr_error,
            "known_station": known,
        }
    )


def forecast_snapshot(bundle: dict, stations: pd.DataFrame, *, as_of: pd.Timestamp) -> pd.DataFrame:
    features = make_live_feature_frame(stations, as_of=as_of, bucket_minutes=bundle["bucket_minutes"])
    predictions = predict_flows(bundle, features)
    snapshot = stations.copy()
    snapshot["station_id"] = snapshot["station_id"].astype(str)
    result = snapshot.merge(predictions, on="station_id", how="left", validate="one_to_one")

    current_bikes = result["bikes_available"].astype(float)
    result["projected_bikes"] = current_bikes + result["predicted_arrivals"] - result["predicted_departures"]
    result["projected_bikes_lower"] = current_bikes + result["arrivals_lower"] - result["departures_upper"]
    result["projected_bikes_upper"] = current_bikes + result["arrivals_upper"] - result["departures_lower"]
    result["model_version"] = bundle["model_version"]
    result["data_mode"] = bundle["data_mode"]
    result["horizon_minutes"] = bundle["horizon_minutes"]
    if "live_station_id" in result:
        result["model_station_id"] = result["station_id"]
        result["station_id"] = result["live_station_id"]
    return result


def score_risks(
    forecast: pd.DataFrame,
    *,
    safety_bikes: int,
    safety_docks: int,
) -> pd.DataFrame:
    result = forecast.copy()
    capacity = result["capacity"].clip(lower=1).astype(float)
    empty_margin = safety_bikes - result["projected_bikes_lower"]
    full_threshold = capacity - safety_docks
    full_margin = result["projected_bikes_upper"] - full_threshold
    result["empty_risk_score"] = np.clip(empty_margin / max(safety_bikes, 1), 0, 1)
    result["full_risk_score"] = np.clip(full_margin / max(safety_docks, 1), 0, 1)
    result["risk_score"] = result[["empty_risk_score", "full_risk_score"]].max(axis=1)
    result["risk_type"] = np.where(
        result["empty_risk_score"] >= result["full_risk_score"], "empty", "full"
    )
    result.loc[result["risk_score"] == 0, "risk_type"] = "balanced"
    result["risk_level"] = np.select(
        [result["risk_score"] >= 0.75, result["risk_score"] >= 0.25],
        ["high", "medium"],
        default="low",
    )
    return result.sort_values(["risk_score", "station_id"], ascending=[False, True], ignore_index=True)
