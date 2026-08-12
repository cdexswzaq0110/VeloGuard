from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import load_config, project_root, resolve_path
from .forecast import align_live_stations, forecast_snapshot, load_bundle, score_risks
from .live import collect_snapshot, get_plan, latest_snapshot, latest_snapshot_metadata, save_plan
from .planner import create_rebalancing_plan


class PlanRequest(BaseModel):
    max_total_moves: int | None = Field(default=None, ge=0, le=500)


@lru_cache(maxsize=4)
def _cached_bundle(path: str, modified_ns: int) -> dict:
    del modified_ns
    return load_bundle(path)


def _bundle(config: dict) -> dict:
    artifact = resolve_path(config["paths"]["artifact"])
    if not artifact.exists():
        raise HTTPException(status_code=503, detail="Model artifact is missing; run 'veloguard demo' or 'veloguard train'")
    return _cached_bundle(str(artifact), artifact.stat().st_mtime_ns)


@lru_cache(maxsize=8)
def _cached_risk_calculation(
    database_path: str,
    artifact_path: str,
    artifact_modified_ns: int,
    observed_at: str,
    safety_bikes: int,
    safety_docks: int,
) -> tuple[pd.DataFrame, dict, dict]:
    del observed_at
    snapshot, snapshot_metadata = latest_snapshot(database_path)
    bundle = _cached_bundle(artifact_path, artifact_modified_ns)
    try:
        snapshot, alignment = align_live_stations(bundle, snapshot)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if snapshot.empty:
        raise HTTPException(status_code=503, detail="The live feed has no stations covered by this model artifact")
    snapshot_metadata.update(alignment)
    as_of = pd.Timestamp(snapshot_metadata["source_updated_at"])
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert("America/New_York").tz_localize(None)
    forecast = forecast_snapshot(bundle, snapshot, as_of=as_of)
    risks = score_risks(forecast, safety_bikes=safety_bikes, safety_docks=safety_docks)
    return risks, snapshot_metadata, bundle


def _risk_frame(config: dict, *, allow_stale: bool) -> tuple[pd.DataFrame, dict, dict]:
    database = resolve_path(config["paths"]["database"])
    try:
        current_metadata = latest_snapshot_metadata(database)
    except LookupError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    max_age = config["live"]["max_feed_age_seconds"]
    if current_metadata["feed_age_seconds"] > max_age and not allow_stale:
        raise HTTPException(
            status_code=503,
            detail=f"Latest GBFS snapshot is stale ({current_metadata['feed_age_seconds']:.0f}s > {max_age}s)",
        )
    planner = config["planner"]
    artifact = resolve_path(config["paths"]["artifact"])
    if not artifact.exists():
        raise HTTPException(status_code=503, detail="Model artifact is missing; run 'veloguard demo' or 'veloguard train'")
    risks, snapshot_metadata, bundle = _cached_risk_calculation(
        str(database),
        str(artifact),
        artifact.stat().st_mtime_ns,
        current_metadata["observed_at"],
        planner["safety_bikes"],
        planner["safety_docks"],
    )
    snapshot_metadata["feed_age_seconds"] = current_metadata["feed_age_seconds"]
    return risks, snapshot_metadata, bundle


def _records(frame: pd.DataFrame) -> list[dict]:
    safe = frame.astype(object).where(pd.notna(frame), None)
    return safe.to_dict(orient="records")


def create_app(config_path: str | Path | None = None) -> FastAPI:
    config = load_config(config_path)
    app = FastAPI(
        title="VeloGuard API",
        version="1.0.0",
        description="Forecast bike-share demand risk and generate capacity-safe rebalancing plans.",
    )
    web_root = project_root() / "web"
    assets = web_root / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(web_root / "index.html")

    @app.get("/healthz")
    def health() -> dict:
        artifact = resolve_path(config["paths"]["artifact"])
        database = resolve_path(config["paths"]["database"])
        result = {"status": "ok", "model_ready": artifact.exists(), "snapshot_ready": False}
        try:
            metadata = latest_snapshot_metadata(database)
            result["snapshot_ready"] = True
            result["feed_age_seconds"] = round(metadata["feed_age_seconds"], 1)
            if metadata["feed_age_seconds"] > config["live"]["max_feed_age_seconds"]:
                result["status"] = "degraded"
        except LookupError:
            result["status"] = "degraded"
        return result

    @app.post("/v1/snapshots/collect")
    def collect() -> dict:
        try:
            return collect_snapshot(
                resolve_path(config["paths"]["database"]),
                config["live"]["discovery_url"],
                timeout=config["live"]["request_timeout_seconds"],
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=f"GBFS collection failed: {error}") from error

    @app.get("/v1/stations/risks")
    def station_risks(
        limit: int = Query(default=200, ge=1, le=3000),
        allow_stale: bool = Query(default=False),
    ) -> dict:
        risks, metadata, bundle = _risk_frame(config, allow_stale=allow_stale)
        high_risk = int((risks["risk_level"] == "high").sum())
        coverage = metadata["covered_station_count"] / max(metadata["system_station_count"], 1)
        return {
            "as_of": metadata["source_updated_at"],
            "feed_age_seconds": round(metadata["feed_age_seconds"], 1),
            "max_feed_age_seconds": config["live"]["max_feed_age_seconds"],
            "feed_is_stale": metadata["feed_age_seconds"] > config["live"]["max_feed_age_seconds"],
            "model_version": bundle["model_version"],
            "champion": bundle["champion"],
            "release_status": bundle.get("release_status", "unknown"),
            "data_mode": bundle["data_mode"],
            "horizon_minutes": bundle["horizon_minutes"],
            "station_count": len(risks),
            "system_station_count": metadata["system_station_count"],
            "high_risk_stations": high_risk,
            "model_station_coverage": round(coverage, 4),
            "alignment_methods": metadata["alignment_methods"],
            "stations": _records(risks.head(limit)),
        }

    @app.get("/v1/model/summary")
    def model_summary() -> dict:
        report_path = resolve_path(config["paths"]["report"])
        if not report_path.exists():
            raise HTTPException(status_code=503, detail="Model report is missing; run 'veloguard demo' or 'veloguard train'")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=503, detail=f"Model report is unreadable: {error}") from error
        return {
            "model_version": report.get("model_version"),
            "data_mode": report.get("data_mode"),
            "target_definition": report.get("target_definition"),
            "rows": report.get("rows"),
            "stations": report.get("stations"),
            "time_range": report.get("time_range"),
            "promotion": report.get("promotion", {}),
            "frozen_test": report.get("frozen_test", {}),
            "decision_replay": report.get("decision_replay", {}),
            "final_windows": report.get("final_windows", {}),
        }

    @app.post("/v1/rebalance-plans")
    def create_plan(
        request: PlanRequest,
        idempotency_key: str = Header(min_length=8, max_length=128, alias="Idempotency-Key"),
        allow_stale: bool = Query(default=False),
    ) -> dict:
        risks, metadata, bundle = _risk_frame(config, allow_stale=allow_stale)
        settings = config["planner"].copy()
        if request.max_total_moves is not None:
            settings["max_total_moves"] = request.max_total_moves
        plan = create_rebalancing_plan(
            risks,
            snapshot_at=metadata["source_updated_at"],
            model_version=bundle["model_version"],
            idempotency_key=idempotency_key,
            **settings,
        )
        plan["release_status"] = bundle.get("release_status", "unknown")
        return save_plan(
            resolve_path(config["paths"]["database"]),
            plan,
            idempotency_key=idempotency_key,
        )

    @app.get("/v1/rebalance-plans/{plan_id}")
    def read_plan(plan_id: str) -> dict:
        try:
            return get_plan(resolve_path(config["paths"]["database"]), plan_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return app


app = create_app()
