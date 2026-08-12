from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd


CURRENT_COLUMNS = (
    "ride_id",
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
    "start_station_name",
    "end_station_name",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
)

ALIASES = {
    "tripduration": "trip_duration",
    "starttime": "started_at",
    "stoptime": "ended_at",
    "start station id": "start_station_id",
    "end station id": "end_station_id",
    "start station name": "start_station_name",
    "end station name": "end_station_name",
    "start station latitude": "start_lat",
    "start station longitude": "start_lng",
    "end station latitude": "end_lat",
    "end station longitude": "end_lng",
    "bikeid": "ride_id",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else project_root() / "config" / "default.json"
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_month(
    month: str,
    raw_dir: Path,
    url_template: str,
    *,
    force: bool = False,
) -> dict:
    yyyymm = month.replace("-", "")
    url = url_template.format(month=month, yyyymm=yyyymm)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{yyyymm}-citibike-tripdata.zip"

    if not destination.exists() or force:
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "VeloGuard/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    if not zipfile.is_zipfile(destination):
        raise ValueError(f"Downloaded file is not a ZIP archive: {destination}")

    return {
        "month": month,
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def download_configured_months(config: dict, *, force: bool = False) -> list[Path]:
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    records = [
        download_month(month, raw_dir, config["data"]["url_template"], force=force)
        for month in config["data"]["months"]
    ]
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return [Path(record["path"]) for record in records]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = {column: str(column).strip().lower() for column in frame.columns}
    frame = frame.rename(columns=normalized).rename(columns=ALIASES)
    missing = set(CURRENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Trip data is missing required columns: {sorted(missing)}")
    return frame[list(CURRENT_COLUMNS)].copy()


def _clean_trip_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_columns(frame)
    frame["started_at"] = pd.to_datetime(frame["started_at"], errors="coerce")
    frame["ended_at"] = pd.to_datetime(frame["ended_at"], errors="coerce")
    for column in ("start_station_id", "end_station_id"):
        frame[column] = frame[column].astype("string").str.replace(r"\.0$", "", regex=True)
    for column in ("start_lat", "start_lng", "end_lat", "end_lng"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = (
        frame["ride_id"].notna()
        & frame["started_at"].notna()
        & frame["ended_at"].notna()
        & frame["start_station_id"].notna()
        & frame["end_station_id"].notna()
        & (frame["ended_at"] >= frame["started_at"])
    )
    return frame.loc[valid]


def iter_trip_chunks(
    zip_paths: list[Path],
    *,
    chunk_rows: int = 250_000,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    remaining = max_rows
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            members = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
            if not members:
                raise ValueError(f"No CSV files found in {zip_path}")
            for member in members:
                with archive.open(member) as source:
                    for chunk in pd.read_csv(source, chunksize=chunk_rows, low_memory=False):
                        if remaining is not None:
                            if remaining <= 0:
                                return
                            chunk = chunk.iloc[:remaining]
                            remaining -= len(chunk)
                        cleaned = _clean_trip_chunk(chunk)
                        if not cleaned.empty:
                            yield cleaned


def _station_metadata(chunk: pd.DataFrame) -> pd.DataFrame:
    starts = chunk[["start_station_id", "start_station_name", "start_lat", "start_lng"]].rename(
        columns={
            "start_station_id": "station_id",
            "start_station_name": "station_name",
            "start_lat": "latitude",
            "start_lng": "longitude",
        }
    )
    ends = chunk[["end_station_id", "end_station_name", "end_lat", "end_lng"]].rename(
        columns={
            "end_station_id": "station_id",
            "end_station_name": "station_name",
            "end_lat": "latitude",
            "end_lng": "longitude",
        }
    )
    metadata = pd.concat([starts, ends], ignore_index=True).dropna(subset=["station_id"])
    return metadata.groupby("station_id", as_index=False).agg(
        station_name=("station_name", "last"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
    )


def aggregate_trip_zips(
    zip_paths: list[Path],
    *,
    bucket_minutes: int,
    max_stations: int,
    selection_fraction: float = 0.6,
    chunk_rows: int = 250_000,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    if 60 % bucket_minutes:
        raise ValueError("bucket_minutes must divide one hour")
    if not 0 < selection_fraction <= 1:
        raise ValueError("selection_fraction must be in (0, 1]")

    frequency = f"{bucket_minutes}min"
    flow_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    input_rows = 0
    valid_rows = 0

    for chunk in iter_trip_chunks(zip_paths, chunk_rows=chunk_rows, max_rows=max_rows):
        input_rows += len(chunk)
        valid_rows += len(chunk)
        departures = (
            chunk.assign(timestamp=chunk["started_at"].dt.floor(frequency))
            .groupby(["start_station_id", "timestamp"], as_index=False)
            .size()
            .rename(columns={"start_station_id": "station_id", "size": "departures"})
        )
        arrivals = (
            chunk.assign(timestamp=chunk["ended_at"].dt.floor(frequency))
            .groupby(["end_station_id", "timestamp"], as_index=False)
            .size()
            .rename(columns={"end_station_id": "station_id", "size": "arrivals"})
        )
        flow_parts.append(departures.merge(arrivals, how="outer", on=["station_id", "timestamp"]))
        metadata_parts.append(_station_metadata(chunk))

    if not flow_parts:
        raise ValueError("No valid trip rows were found")

    observed = pd.concat(flow_parts, ignore_index=True).fillna({"departures": 0, "arrivals": 0})
    observed = observed.groupby(["station_id", "timestamp"], as_index=False)[["departures", "arrivals"]].sum()
    observed[["departures", "arrivals"]] = observed[["departures", "arrivals"]].astype("int32")

    start = observed["timestamp"].min()
    end = observed["timestamp"].max()
    selection_end = start + (end - start) * selection_fraction
    selection = observed.loc[observed["timestamp"] <= selection_end].copy()
    activity = selection.assign(activity=selection["departures"] + selection["arrivals"])
    top_ids = (
        activity.groupby("station_id")["activity"]
        .sum()
        .sort_values(ascending=False)
        .head(max_stations)
        .index.astype(str)
    )

    timestamps = pd.date_range(start.floor(frequency), end.ceil(frequency), freq=frequency)
    index = pd.MultiIndex.from_product([top_ids, timestamps], names=["station_id", "timestamp"])
    flow = (
        observed.loc[observed["station_id"].astype(str).isin(top_ids)]
        .assign(station_id=lambda x: x["station_id"].astype(str))
        .set_index(["station_id", "timestamp"])
        .reindex(index, fill_value=0)
        .reset_index()
    )

    metadata = pd.concat(metadata_parts, ignore_index=True).groupby("station_id", as_index=False).agg(
        station_name=("station_name", "last"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
    )
    metadata["station_id"] = metadata["station_id"].astype(str)
    flow = flow.merge(metadata, how="left", on="station_id")
    flow = flow.sort_values(["timestamp", "station_id"], ignore_index=True)

    report = {
        "source_files": [str(path) for path in zip_paths],
        "valid_trip_rows": valid_rows,
        "flow_rows": len(flow),
        "stations": int(flow["station_id"].nunique()),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "selection_end": selection_end.isoformat(),
        "bucket_minutes": bucket_minutes,
        "max_rows_requested": max_rows,
    }
    return flow, report


def prepare_configured_data(config: dict, *, max_rows: int | None = None) -> tuple[Path, dict]:
    zip_paths = download_configured_months(config)
    settings = config["data"]
    flow, report = aggregate_trip_zips(
        zip_paths,
        bucket_minutes=settings["bucket_minutes"],
        max_stations=settings["max_stations"],
        selection_fraction=settings["selection_fraction"],
        chunk_rows=settings["csv_chunk_rows"],
        max_rows=max_rows,
    )
    destination = resolve_path(config["paths"]["processed_flow"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    flow.to_parquet(destination, index=False)
    destination.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination, report


def generate_demo_flow(
    *,
    days: int = 28,
    station_count: int = 12,
    bucket_minutes: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    periods = days * 24 * (60 // bucket_minutes)
    timestamps = pd.date_range("2026-01-01", periods=periods, freq=f"{bucket_minutes}min")
    records = []
    for station in range(station_count):
        latitude = 40.70 + (station % 4) * 0.012
        longitude = -74.01 + (station // 4) * 0.014
        station_bias = 0.6 + station / station_count
        for timestamp in timestamps:
            hour = timestamp.hour + timestamp.minute / 60
            morning = max(0.0, 1 - abs(hour - 8.5) / 2.5)
            evening = max(0.0, 1 - abs(hour - 17.5) / 3.0)
            weekend = timestamp.dayofweek >= 5
            outbound = station_bias * (0.5 + 3.0 * morning + 1.4 * evening) * (0.75 if weekend else 1.0)
            inbound = (2.0 - station_bias) * (0.5 + 1.3 * morning + 3.0 * evening) * (0.8 if weekend else 1.0)
            records.append(
                {
                    "station_id": f"demo-{station:02d}",
                    "timestamp": timestamp,
                    "departures": int(rng.poisson(outbound)),
                    "arrivals": int(rng.poisson(inbound)),
                    "station_name": f"Demo Station {station + 1}",
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
    return pd.DataFrame.from_records(records)
