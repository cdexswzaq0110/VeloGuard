from __future__ import annotations

import numpy as np
import pandas as pd

from .planner import create_rebalancing_plan


def _serve_bucket(inventory: int, capacity: int, departures: int, arrivals: int) -> tuple[int, int, int]:
    served_departures = min(inventory, departures)
    unmet_departures = departures - served_departures
    inventory -= served_departures
    accepted_arrivals = min(capacity - inventory, arrivals)
    rejected_arrivals = arrivals - accepted_arrivals
    inventory += accepted_arrivals
    return inventory, unmet_departures, rejected_arrivals


def simulate_rebalancing_policy(
    test_frame: pd.DataFrame,
    predicted_departures: np.ndarray,
    predicted_arrivals: np.ndarray,
    *,
    departure_error: float,
    arrival_error: float,
    planner_settings: dict,
    bucket_minutes: int,
    planning_interval_minutes: int,
    station_capacity: int,
    policy_name: str,
) -> dict:
    if planning_interval_minutes % bucket_minutes:
        raise ValueError("planning_interval_minutes must be a multiple of bucket_minutes")
    frame = test_frame.copy()
    frame["predicted_departures"] = predicted_departures
    frame["predicted_arrivals"] = predicted_arrivals
    frame = frame.sort_values(["timestamp", "station_id"], ignore_index=True)
    station_ids = sorted(frame["station_id"].astype(str).unique())
    initial_inventory = round(station_capacity * planner_settings["target_fill_ratio"])
    no_action_inventory = {station_id: initial_inventory for station_id in station_ids}
    policy_inventory = no_action_inventory.copy()
    no_action_unmet = no_action_rejected = policy_unmet = policy_rejected = 0
    bikes_moved = 0
    move_distance_km = 0.0
    plans_created = 0
    planning_every = planning_interval_minutes // bucket_minutes

    for step, (timestamp, bucket) in enumerate(frame.groupby("timestamp", sort=True)):
        if step % planning_every == 0:
            risk_rows = []
            for row in bucket.itertuples(index=False):
                station_id = str(row.station_id)
                bikes = policy_inventory[station_id]
                projected = bikes + float(row.predicted_arrivals) - float(row.predicted_departures)
                lower = bikes + max(0.0, float(row.predicted_arrivals) - arrival_error) - (
                    float(row.predicted_departures) + departure_error
                )
                upper = bikes + float(row.predicted_arrivals) + arrival_error - max(
                    0.0, float(row.predicted_departures) - departure_error
                )
                empty_score = float(np.clip((planner_settings["safety_bikes"] - lower) / max(planner_settings["safety_bikes"], 1), 0, 1))
                full_score = float(
                    np.clip(
                        (upper - (station_capacity - planner_settings["safety_docks"]))
                        / max(planner_settings["safety_docks"], 1),
                        0,
                        1,
                    )
                )
                risk_rows.append(
                    {
                        "station_id": station_id,
                        "station_name": getattr(row, "station_name", station_id),
                        "latitude": float(row.latitude),
                        "longitude": float(row.longitude),
                        "capacity": station_capacity,
                        "bikes_available": bikes,
                        "docks_available": station_capacity - bikes,
                        "projected_bikes": projected,
                        "projected_bikes_lower": lower,
                        "projected_bikes_upper": upper,
                        "empty_risk_score": empty_score,
                        "full_risk_score": full_score,
                        "is_installed": 1,
                        "is_renting": 1,
                        "is_returning": 1,
                    }
                )
            plan = create_rebalancing_plan(
                pd.DataFrame(risk_rows),
                snapshot_at=pd.Timestamp(timestamp).isoformat(),
                model_version=policy_name,
                idempotency_key=f"{policy_name}-{pd.Timestamp(timestamp).isoformat()}",
                **planner_settings,
            )
            plans_created += 1
            bikes_moved += plan["total_bikes_moved"]
            for move in plan["moves"]:
                source = move["from_station_id"]
                destination = move["to_station_id"]
                units = move["bikes"]
                policy_inventory[source] -= units
                policy_inventory[destination] += units
                move_distance_km += move["distance_km"] * units

        for row in bucket.itertuples(index=False):
            station_id = str(row.station_id)
            departures = int(row.departures)
            arrivals = int(row.arrivals)
            no_action_inventory[station_id], unmet, rejected = _serve_bucket(
                no_action_inventory[station_id], station_capacity, departures, arrivals
            )
            no_action_unmet += unmet
            no_action_rejected += rejected
            policy_inventory[station_id], unmet, rejected = _serve_bucket(
                policy_inventory[station_id], station_capacity, departures, arrivals
            )
            policy_unmet += unmet
            policy_rejected += rejected

    no_action_failures = no_action_unmet + no_action_rejected
    policy_failures = policy_unmet + policy_rejected
    return {
        "policy": policy_name,
        "assumed_station_capacity": station_capacity,
        "planning_interval_minutes": planning_interval_minutes,
        "no_action": {
            "unmet_departures": no_action_unmet,
            "rejected_arrivals": no_action_rejected,
            "service_failures": no_action_failures,
        },
        "with_rebalancing": {
            "unmet_departures": policy_unmet,
            "rejected_arrivals": policy_rejected,
            "service_failures": policy_failures,
            "relative_failure_reduction": (no_action_failures - policy_failures) / max(no_action_failures, 1),
            "bikes_moved": bikes_moved,
            "bike_km": round(move_distance_km, 3),
            "plans_created": plans_created,
        },
        "simulation_limitations": [
            "All stations use one assumed capacity and begin half full.",
            "Trips inside each 15-minute bucket are processed as departures then arrivals.",
            "Truck routing, loading time, traffic and operator behavior are not modeled.",
        ],
    }
