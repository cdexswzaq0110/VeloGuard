from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import (
    generate_demo_flow,
    load_config,
    prepare_configured_data,
    resolve_path,
)
from .live import collect_snapshot, seed_demo_snapshot
from .train import train_and_evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veloguard", description="Bike-share forecasting and rebalancing system")
    parser.add_argument("--config", default=None, help="Path to a JSON config file")
    subcommands = parser.add_subparsers(dest="command", required=True)

    backfill = subcommands.add_parser("backfill", help="Download and aggregate configured official trip files")
    backfill.add_argument("--max-rows", type=int, default=None, help="Optional smoke-run row limit")

    subcommands.add_parser("train", help="Train and evaluate from the processed flow table")
    subcommands.add_parser("collect", help="Collect one live GBFS station snapshot")

    demo = subcommands.add_parser("demo", help="Create a complete synthetic demo without network access")
    demo.add_argument("--days", type=int, default=28)
    demo.add_argument("--stations", type=int, default=12)

    pipeline = subcommands.add_parser("pipeline", help="Backfill, train and collect a live snapshot")
    pipeline.add_argument("--max-rows", type=int, default=None, help="Optional smoke-run row limit")

    serve = subcommands.add_parser("serve", help="Start the API and dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)

    if args.command == "backfill":
        path, report = prepare_configured_data(config, max_rows=args.max_rows)
        _print({"processed_flow": str(path), **report})
        return

    if args.command == "train":
        flow_path = resolve_path(config["paths"]["processed_flow"])
        if not flow_path.exists():
            raise SystemExit("Processed flow file is missing; run 'veloguard backfill' first")
        _, report = train_and_evaluate(pd.read_parquet(flow_path), config, data_mode="official")
        _print(report)
        return

    if args.command == "collect":
        _print(
            collect_snapshot(
                resolve_path(config["paths"]["database"]),
                config["live"]["discovery_url"],
                timeout=config["live"]["request_timeout_seconds"],
            )
        )
        return

    if args.command == "demo":
        flow = generate_demo_flow(
            days=args.days,
            station_count=args.stations,
            bucket_minutes=config["data"]["bucket_minutes"],
            seed=config["training"]["random_seed"],
        )
        flow_path = resolve_path(config["paths"]["processed_flow"])
        flow_path.parent.mkdir(parents=True, exist_ok=True)
        flow.to_parquet(flow_path, index=False)
        _, report = train_and_evaluate(flow, config, data_mode="synthetic-demo")
        snapshot = seed_demo_snapshot(
            resolve_path(config["paths"]["database"]), flow, seed=config["training"]["random_seed"]
        )
        _print({"training": report, "snapshot": snapshot})
        return

    if args.command == "pipeline":
        flow_path, data_report = prepare_configured_data(config, max_rows=args.max_rows)
        _, training_report = train_and_evaluate(pd.read_parquet(flow_path), config, data_mode="official")
        snapshot_report = collect_snapshot(
            resolve_path(config["paths"]["database"]),
            config["live"]["discovery_url"],
            timeout=config["live"]["request_timeout_seconds"],
        )
        _print({"data": data_report, "training": training_report, "snapshot": snapshot_report})
        return

    if args.command == "serve":
        import uvicorn
        from .api import create_app

        uvicorn.run(create_app(args.config), host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
