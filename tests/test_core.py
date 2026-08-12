from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

from veloguard.api import create_app
from veloguard.data import generate_demo_flow
from veloguard.features import build_training_frame, rolling_time_splits
from veloguard.forecast import align_live_stations
from veloguard.live import latest_snapshot, seed_demo_snapshot
from veloguard.planner import create_rebalancing_plan
from veloguard.train import train_and_evaluate


def test_station_alignment_prefers_identity_and_rejects_distant_name_collision():
    bundle = {
        "station_catalog": [
            {"station_id": "historic-a", "station_name": "Main & 1st", "latitude": 40.70, "longitude": -74.00},
            {"station_id": "historic-b", "station_name": "Park", "latitude": 40.71, "longitude": -74.01},
        ]
    }
    live = pd.DataFrame(
        [
            {"station_id": "historic-b", "station_name": "Renamed", "latitude": 40.71, "longitude": -74.01},
            {"station_id": "new-a", "station_name": "MAIN & 1ST", "latitude": 40.7001, "longitude": -74.0001},
            {"station_id": "collision", "station_name": "Main & 1st", "latitude": 41.20, "longitude": -73.40},
        ]
    )
    aligned, metadata = align_live_stations(bundle, live)
    assert set(aligned["live_station_id"]) == {"historic-b", "new-a"}
    assert metadata["alignment_methods"] == {"station_id": 1, "normalized_name": 1}


def test_forward_horizon_and_rolling_split_are_time_ordered():
    timestamps = pd.date_range("2026-01-01", periods=4, freq="15min")
    flow = pd.DataFrame(
        {
            "station_id": ["A"] * 4,
            "timestamp": timestamps,
            "departures": [1, 2, 3, 4],
            "arrivals": [0, 1, 0, 1],
            "latitude": [40.7] * 4,
            "longitude": [-74.0] * 4,
        }
    )
    frame = build_training_frame(flow, bucket_minutes=15, horizon_minutes=30)
    assert frame["target_departures"].tolist() == [5.0, 7.0]
    assert frame["target_arrivals"].tolist() == [1.0, 1.0]

    long_frame = build_training_frame(generate_demo_flow(days=24, station_count=2), bucket_minutes=15, horizon_minutes=30)
    folds, test_index, test_start = rolling_time_splits(
        long_frame, test_days=4, validation_days=2, max_folds=2
    )
    for train_index, valid_index in folds:
        assert long_frame.iloc[train_index]["timestamp"].max() < long_frame.iloc[valid_index]["timestamp"].min()
    assert long_frame.iloc[test_index]["timestamp"].min() >= test_start


def test_planner_respects_inventory_capacity_distance_and_budget():
    risks = pd.DataFrame(
        [
            {
                "station_id": "source",
                "station_name": "Source",
                "latitude": 40.70,
                "longitude": -74.00,
                "capacity": 20,
                "bikes_available": 18,
                "docks_available": 2,
                "projected_bikes": 18.0,
                "projected_bikes_lower": 16.0,
                "projected_bikes_upper": 19.0,
                "empty_risk_score": 0.0,
                "full_risk_score": 1.0,
                "is_installed": 1,
                "is_renting": 1,
                "is_returning": 1,
            },
            {
                "station_id": "destination",
                "station_name": "Destination",
                "latitude": 40.705,
                "longitude": -74.005,
                "capacity": 20,
                "bikes_available": 2,
                "docks_available": 18,
                "projected_bikes": 1.0,
                "projected_bikes_lower": 0.0,
                "projected_bikes_upper": 2.0,
                "empty_risk_score": 1.0,
                "full_risk_score": 0.0,
                "is_installed": 1,
                "is_renting": 1,
                "is_returning": 1,
            },
        ]
    )
    kwargs = dict(
        snapshot_at="2026-01-01T00:00:00+00:00",
        model_version="model-1",
        idempotency_key="same-request",
        safety_bikes=2,
        safety_docks=2,
        target_fill_ratio=0.5,
        max_total_moves=5,
        max_route_km=4.0,
    )
    first = create_rebalancing_plan(risks, **kwargs)
    second = create_rebalancing_plan(risks, **kwargs)
    assert first["plan_id"] == second["plan_id"]
    assert first["total_bikes_moved"] == 5
    assert sum(move["bikes"] for move in first["moves"]) == 5
    assert all(move["distance_km"] <= 4.0 for move in first["moves"])
    assert 18 - first["total_bikes_moved"] >= 2
    assert 2 + first["total_bikes_moved"] <= 20 - 2

    uncertain = risks.copy()
    uncertain.loc[uncertain["station_id"] == "source", "projected_bikes_lower"] = 9.0
    uncertain.loc[uncertain["station_id"] == "destination", "projected_bikes_upper"] = 19.0
    blocked = create_rebalancing_plan(uncertain, **{**kwargs, "idempotency_key": "uncertain-request"})
    assert blocked["total_bikes_moved"] == 0


def test_end_to_end_model_snapshot_api_and_idempotent_plan(tmp_path):
    flow = generate_demo_flow(days=22, station_count=5, seed=7)
    config = {
        "project_name": "veloguard-test",
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_flow": str(tmp_path / "flow.parquet"),
            "artifact": str(tmp_path / "model.joblib"),
            "report": str(tmp_path / "report.json"),
            "database": str(tmp_path / "veloguard.sqlite3"),
        },
        "data": {
            "months": ["2024-01"],
            "url_template": "unused",
            "bucket_minutes": 15,
            "horizon_minutes": 30,
            "max_stations": 5,
            "selection_fraction": 0.6,
            "csv_chunk_rows": 1000,
        },
        "training": {
            "test_days": 4,
            "validation_days": 2,
            "max_folds": 2,
            "min_relative_improvement": 0.02,
            "coverage": 0.9,
            "random_seed": 7,
        },
        "live": {
            "discovery_url": "unused",
            "max_feed_age_seconds": 600,
            "request_timeout_seconds": 1,
        },
        "planner": {
            "safety_bikes": 2,
            "safety_docks": 2,
            "target_fill_ratio": 0.5,
            "max_total_moves": 12,
            "max_route_km": 4.0,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    bundle, report = train_and_evaluate(
        flow,
        config,
        artifact_path=tmp_path / "model.joblib",
        report_path=tmp_path / "report.json",
        data_mode="synthetic-test",
    )
    assert bundle["model_version"] == report["model_version"]
    assert bundle["release_status"] in {"approved_for_decision_support", "shadow_mode_only"}
    decision = report["decision_replay"]
    baseline_decision = decision["baseline"]["with_rebalancing"]
    candidate_decision = decision["candidate"]["with_rebalancing"]
    expected_dominance = (
        candidate_decision["service_failures"] <= baseline_decision["service_failures"]
        and candidate_decision["bikes_moved"] <= baseline_decision["bikes_moved"]
        and (
            candidate_decision["service_failures"] < baseline_decision["service_failures"]
            or candidate_decision["bikes_moved"] < baseline_decision["bikes_moved"]
        )
    )
    assert decision["candidate_dominates_baseline"] is expected_dominance
    assert 0 <= report["frozen_test"]["departures_interval_coverage"] <= 1

    seed_demo_snapshot(tmp_path / "veloguard.sqlite3", flow, seed=7)
    snapshot, metadata = latest_snapshot(tmp_path / "veloguard.sqlite3")
    assert len(snapshot) == 5
    assert metadata["feed_age_seconds"] < 10

    client = TestClient(create_app(config_path))
    assert client.get("/healthz").status_code == 200
    risks = client.get("/v1/stations/risks").json()
    assert risks["station_count"] == 5
    assert risks["release_status"] == bundle["release_status"]
    summary = client.get("/v1/model/summary")
    assert summary.status_code == 200
    assert summary.json()["model_version"] == bundle["model_version"]
    headers = {"Idempotency-Key": "fixed-test-key"}
    first = client.post("/v1/rebalance-plans", headers=headers, json={"max_total_moves": 8})
    second = client.post("/v1/rebalance-plans", headers=headers, json={"max_total_moves": 8})
    assert first.status_code == 200
    assert first.json()["plan_id"] == second.json()["plan_id"]
    assert first.json()["release_status"] == bundle["release_status"]
