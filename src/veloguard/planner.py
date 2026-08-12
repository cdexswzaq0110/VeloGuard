from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone

import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def create_rebalancing_plan(
    risks: pd.DataFrame,
    *,
    snapshot_at: str,
    model_version: str,
    idempotency_key: str,
    safety_bikes: int,
    safety_docks: int,
    target_fill_ratio: float,
    max_total_moves: int,
    max_route_km: float,
) -> dict:
    if not idempotency_key.strip():
        raise ValueError("Idempotency-Key must not be empty")
    if not 0 < target_fill_ratio < 1:
        raise ValueError("target_fill_ratio must be between 0 and 1")
    if max_total_moves < 0 or max_route_km <= 0:
        raise ValueError("Planner limits must be positive")

    required = {
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "capacity",
        "bikes_available",
        "docks_available",
        "projected_bikes",
        "empty_risk_score",
        "full_risk_score",
    }
    missing = required - set(risks.columns)
    if missing:
        raise ValueError(f"Risk frame is missing columns: {sorted(missing)}")

    stations = risks.copy()
    if {"is_installed", "is_renting", "is_returning"}.issubset(stations.columns):
        stations = stations.loc[
            stations["is_installed"].astype(bool)
            & stations["is_renting"].astype(bool)
            & stations["is_returning"].astype(bool)
        ]
    stations["target_bikes"] = stations["capacity"] * target_fill_ratio
    stations["projected_lower"] = stations.get("projected_bikes_lower", stations["projected_bikes"])
    stations["projected_upper"] = stations.get("projected_bikes_upper", stations["projected_bikes"])
    stations["source_units"] = stations.apply(
        lambda row: 0
        if float(row["full_risk_score"]) <= 0
        else max(
            0,
            min(
                int(row["bikes_available"]) - safety_bikes,
                math.floor(float(row["projected_lower"]) - float(row["target_bikes"])),
            ),
        ),
        axis=1,
    )
    stations["destination_units"] = stations.apply(
        lambda row: 0
        if float(row["empty_risk_score"]) <= 0
        else max(
            0,
            min(
                int(row["docks_available"]) - safety_docks,
                math.floor(
                    float(row["capacity"]) - safety_docks - float(row["projected_upper"])
                ),
                math.ceil(float(row["target_bikes"]) - float(row["projected_lower"])),
            ),
        ),
        axis=1,
    )

    sources = stations.loc[stations["source_units"] > 0].copy()
    destinations = stations.loc[stations["destination_units"] > 0].copy()
    sources = sources.sort_values(["full_risk_score", "source_units", "station_id"], ascending=[False, False, True])
    destinations = destinations.sort_values(
        ["empty_risk_score", "destination_units", "station_id"], ascending=[False, False, True]
    )

    remaining_sources = {str(row.station_id): int(row.source_units) for row in sources.itertuples()}
    moves = []
    total_units = 0
    for destination in destinations.itertuples():
        needed = int(destination.destination_units)
        while needed > 0 and total_units < max_total_moves:
            candidates = []
            for source in sources.itertuples():
                available = remaining_sources[str(source.station_id)]
                if available <= 0 or str(source.station_id) == str(destination.station_id):
                    continue
                distance = haversine_km(
                    float(source.latitude),
                    float(source.longitude),
                    float(destination.latitude),
                    float(destination.longitude),
                )
                if distance <= max_route_km:
                    candidates.append((distance, str(source.station_id), source, available))
            if not candidates:
                break
            distance, source_id, source, available = min(candidates, key=lambda item: (item[0], item[1]))
            units = min(available, needed, max_total_moves - total_units)
            remaining_sources[source_id] -= units
            needed -= units
            total_units += units
            moves.append(
                {
                    "from_station_id": source_id,
                    "from_station_name": str(source.station_name),
                    "to_station_id": str(destination.station_id),
                    "to_station_name": str(destination.station_name),
                    "bikes": units,
                    "distance_km": round(distance, 3),
                    "destination_empty_risk": round(float(destination.empty_risk_score), 4),
                }
            )

    requested_units = int(destinations["destination_units"].sum()) if not destinations.empty else 0
    plan_id = hashlib.sha256(
        f"{idempotency_key}|{snapshot_at}|{model_version}|{max_total_moves}".encode()
    ).hexdigest()[:16]
    return {
        "plan_id": plan_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_at": snapshot_at,
        "model_version": model_version,
        "algorithm": "capacity-safe-nearest-source-v1",
        "total_bikes_moved": total_units,
        "requested_bikes": requested_units,
        "unfilled_bikes": max(0, requested_units - total_units),
        "max_total_moves": max_total_moves,
        "moves": moves,
        "limitations": "Greedy one-hop plan; routing order and truck fleet scheduling are out of scope.",
    }
