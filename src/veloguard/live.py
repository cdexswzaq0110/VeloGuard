from __future__ import annotations

import json
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _connect(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_database(database_path: str | Path) -> None:
    with _connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS station_snapshots (
                observed_at TEXT NOT NULL,
                source_updated_at TEXT NOT NULL,
                station_id TEXT NOT NULL,
                station_name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                capacity INTEGER NOT NULL CHECK (capacity >= 0),
                bikes_available INTEGER NOT NULL CHECK (bikes_available >= 0),
                docks_available INTEGER NOT NULL CHECK (docks_available >= 0),
                is_installed INTEGER NOT NULL,
                is_renting INTEGER NOT NULL,
                is_returning INTEGER NOT NULL,
                PRIMARY KEY (observed_at, station_id)
            );
            CREATE INDEX IF NOT EXISTS idx_station_snapshots_latest
                ON station_snapshots(observed_at DESC);

            CREATE TABLE IF NOT EXISTS rebalancing_plans (
                plan_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                snapshot_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )


def _fetch_json(url: str, *, timeout: int) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "VeloGuard/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"GBFS request returned HTTP {response.status}: {url}")
        return json.load(response)


def discover_feeds(discovery_url: str, *, timeout: int) -> dict[str, str]:
    payload = _fetch_json(discovery_url, timeout=timeout)
    data = payload.get("data", {})
    if "feeds" in data:
        feeds = data["feeds"]
    else:
        language = "en" if "en" in data else next(iter(data), None)
        if language is None:
            raise ValueError("GBFS discovery document has no language or feed list")
        feeds = data[language].get("feeds", [])
    result = {item["name"]: item["url"] for item in feeds if "name" in item and "url" in item}
    required = {"station_information", "station_status"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"GBFS discovery document is missing feeds: {sorted(missing)}")
    return result


def fetch_station_snapshot(discovery_url: str, *, timeout: int = 20) -> tuple[pd.DataFrame, dict]:
    feeds = discover_feeds(discovery_url, timeout=timeout)
    information_payload = _fetch_json(feeds["station_information"], timeout=timeout)
    status_payload = _fetch_json(feeds["station_status"], timeout=timeout)
    information_rows = information_payload.get("data", {}).get("stations", [])
    status_rows = status_payload.get("data", {}).get("stations", [])
    if not information_rows or not status_rows:
        raise ValueError("GBFS station feeds returned no stations")

    information = pd.DataFrame(information_rows)
    status = pd.DataFrame(status_rows)
    information = information.rename(columns={"name": "station_name", "lat": "latitude", "lon": "longitude"})
    status = status.rename(
        columns={
            "num_bikes_available": "bikes_available",
            "num_vehicles_available": "vehicles_available",
            "num_docks_available": "docks_available",
        }
    )
    if "bikes_available" not in status and "vehicles_available" in status:
        status["bikes_available"] = status["vehicles_available"]
    required_information = {"station_id", "station_name", "latitude", "longitude"}
    required_status = {"station_id", "bikes_available"}
    if required_information - set(information) or required_status - set(status):
        raise ValueError("GBFS station feeds do not satisfy the required schema")

    keep_information = [column for column in ["station_id", "station_name", "latitude", "longitude", "capacity"] if column in information]
    keep_status = [
        column
        for column in [
            "station_id",
            "bikes_available",
            "docks_available",
            "is_installed",
            "is_renting",
            "is_returning",
        ]
        if column in status
    ]
    snapshot = information[keep_information].merge(status[keep_status], on="station_id", validate="one_to_one")
    snapshot["docks_available"] = snapshot.get("docks_available", 0)
    if "capacity" not in snapshot:
        snapshot["capacity"] = snapshot["bikes_available"] + snapshot["docks_available"]
    snapshot["capacity"] = snapshot["capacity"].fillna(snapshot["bikes_available"] + snapshot["docks_available"])
    for column in ("is_installed", "is_renting", "is_returning"):
        if column not in snapshot:
            snapshot[column] = 1
    numeric = ["latitude", "longitude", "capacity", "bikes_available", "docks_available"]
    snapshot[numeric] = snapshot[numeric].apply(pd.to_numeric, errors="coerce")
    snapshot = snapshot.dropna(subset=["station_id", "latitude", "longitude", "capacity", "bikes_available"])
    snapshot["station_id"] = snapshot["station_id"].astype(str)
    snapshot[["capacity", "bikes_available", "docks_available"]] = snapshot[
        ["capacity", "bikes_available", "docks_available"]
    ].clip(lower=0).astype(int)

    source_epoch = int(status_payload.get("last_updated") or information_payload.get("last_updated") or 0)
    if source_epoch <= 0:
        raise ValueError("GBFS feed does not provide a valid last_updated timestamp")
    source_updated = datetime.fromtimestamp(source_epoch, tz=timezone.utc)
    observed = datetime.now(timezone.utc)
    metadata = {
        "observed_at": observed.isoformat(),
        "source_updated_at": source_updated.isoformat(),
        "feed_age_seconds": max(0.0, (observed - source_updated).total_seconds()),
        "station_count": len(snapshot),
        "discovery_url": discovery_url,
    }
    return snapshot, metadata


def store_snapshot(database_path: str | Path, snapshot: pd.DataFrame, metadata: dict) -> int:
    init_database(database_path)
    rows = []
    for station in snapshot.itertuples(index=False):
        rows.append(
            (
                metadata["observed_at"],
                metadata["source_updated_at"],
                str(station.station_id),
                str(station.station_name),
                float(station.latitude),
                float(station.longitude),
                int(station.capacity),
                int(station.bikes_available),
                int(station.docks_available),
                int(bool(station.is_installed)),
                int(bool(station.is_renting)),
                int(bool(station.is_returning)),
            )
        )
    with _connect(database_path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO station_snapshots (
                observed_at, source_updated_at, station_id, station_name, latitude, longitude,
                capacity, bikes_available, docks_available, is_installed, is_renting, is_returning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return connection.total_changes - before


def collect_snapshot(database_path: str | Path, discovery_url: str, *, timeout: int = 20) -> dict:
    snapshot, metadata = fetch_station_snapshot(discovery_url, timeout=timeout)
    metadata["inserted_rows"] = store_snapshot(database_path, snapshot, metadata)
    return metadata


def latest_snapshot(database_path: str | Path) -> tuple[pd.DataFrame, dict]:
    init_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT observed_at, source_updated_at FROM station_snapshots ORDER BY observed_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise LookupError("No station snapshot is stored; run the collect command first")
        snapshot = pd.read_sql_query(
            "SELECT * FROM station_snapshots WHERE observed_at = ? ORDER BY station_id",
            connection,
            params=[row["observed_at"]],
        )
    source_updated = pd.Timestamp(row["source_updated_at"])
    now = pd.Timestamp.now(tz="UTC")
    if source_updated.tzinfo is None:
        source_updated = source_updated.tz_localize("UTC")
    metadata = {
        "observed_at": row["observed_at"],
        "source_updated_at": row["source_updated_at"],
        "feed_age_seconds": max(0.0, (now - source_updated).total_seconds()),
        "station_count": len(snapshot),
    }
    return snapshot, metadata


def latest_snapshot_metadata(database_path: str | Path) -> dict:
    init_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT observed_at, source_updated_at, COUNT(*) AS station_count "
            "FROM station_snapshots WHERE observed_at = (SELECT MAX(observed_at) FROM station_snapshots)"
        ).fetchone()
    if row is None or row["observed_at"] is None:
        raise LookupError("No station snapshot is stored; run the collect command first")
    source_updated = pd.Timestamp(row["source_updated_at"])
    if source_updated.tzinfo is None:
        source_updated = source_updated.tz_localize("UTC")
    return {
        "observed_at": row["observed_at"],
        "source_updated_at": row["source_updated_at"],
        "feed_age_seconds": max(0.0, (pd.Timestamp.now(tz="UTC") - source_updated).total_seconds()),
        "station_count": int(row["station_count"]),
    }


def seed_demo_snapshot(database_path: str | Path, flow: pd.DataFrame, *, seed: int = 42) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    stations = flow.groupby("station_id", as_index=False).agg(
        station_name=("station_name", "last"),
        latitude=("latitude", "last"),
        longitude=("longitude", "last"),
    )
    stations["capacity"] = rng.integers(18, 36, size=len(stations))
    stations["bikes_available"] = [int(rng.integers(2, capacity - 1)) for capacity in stations["capacity"]]
    stations["docks_available"] = stations["capacity"] - stations["bikes_available"]
    stations[["is_installed", "is_renting", "is_returning"]] = 1
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "observed_at": now,
        "source_updated_at": now,
        "feed_age_seconds": 0.0,
        "station_count": len(stations),
        "discovery_url": "synthetic-demo",
    }
    metadata["inserted_rows"] = store_snapshot(database_path, stations, metadata)
    return metadata


def save_plan(database_path: str | Path, plan: dict, *, idempotency_key: str) -> dict:
    init_database(database_path)
    with _connect(database_path) as connection:
        existing = connection.execute(
            "SELECT payload_json FROM rebalancing_plans WHERE idempotency_key = ?", [idempotency_key]
        ).fetchone()
        if existing:
            return json.loads(existing["payload_json"])
        connection.execute(
            """
            INSERT INTO rebalancing_plans (
                plan_id, idempotency_key, created_at, snapshot_at, model_version, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                plan["plan_id"],
                idempotency_key,
                plan["created_at"],
                plan["snapshot_at"],
                plan["model_version"],
                json.dumps(plan),
            ),
        )
    return plan


def get_plan(database_path: str | Path, plan_id: str) -> dict:
    init_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT payload_json FROM rebalancing_plans WHERE plan_id = ?", [plan_id]
        ).fetchone()
    if row is None:
        raise LookupError(f"Unknown rebalancing plan: {plan_id}")
    return json.loads(row["payload_json"])
