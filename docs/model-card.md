# VeloGuard Model Card

## Summary

VeloGuard forecasts Citi Bike station departures and arrivals for a 30-minute horizon. It uses live inventory plus calibrated prediction errors to rank empty/full risk. The artifact contains a seasonal-median baseline, two Poisson histogram gradient-boosting models, calibration residuals, a historical station catalog, and model/version metadata.

This artifact is decision support only. It must not autonomously dispatch workers or modify transportation infrastructure.

## Intended use

- Rank modeled stations for near-term operational review.
- Generate bounded transfer suggestions for a human operator.
- Demonstrate leakage-resistant ML evaluation, model promotion, live-data alignment, and delayed decision evaluation.

Not intended for rider safety, pricing, personnel decisions, full-network production operation, or causal claims about service improvement.

## Training data

- Source: official January 2024 Citi Bike trip archive.
- Valid trip rows: 1,881,977.
- Aggregation: departure and arrival counts per station every 15 minutes.
- Scope: 40 highest-activity stations, ranked using only the first 60% of the observed time range.
- Aggregated rows: 120,680; supervised rows: 120,600 after constructing a strictly future two-bucket horizon.
- Time range: 2023-12-31 13:45 through 2024-01-31 23:45 local timestamps.
- Artifact version from the verified run: `f19de84ec919`.
- Release status: `shadow_mode_only`.

Source files are downloaded atomically, checked as ZIP, and recorded with SHA-256. The final archive is not committed because it is large and publicly reproducible.

## Target and features

- `target_departures`: departure counts in the next two 15-minute buckets.
- `target_arrivals`: arrival counts in the next two 15-minute buckets.
- Features: station ID, day of week, time bucket, hour, minute, weekend, circular time-of-day, and circular day-of-week.

All features are known at prediction time. No current/future target value is used as a feature.

## Evaluation protocol

1. Three expanding rolling training folds, each followed by a three-day validation window.
2. Promotion gate: at least 2% mean combined-MAE improvement and wins in at least two folds.
3. Separate three-day calibration window.
4. Final untouched five-day test window: 2024-01-27 through 2024-01-31.
5. Fixed random seed 42.

The candidate won all three validation folds. Mean rolling improvement was 10.57%.

## Frozen-test performance

| Metric | Candidate | Baseline / target |
|---|---:|---:|
| Combined MAE | 1.8231 | 1.9233 baseline |
| Relative MAE improvement | 5.21% | ≥2% promotion threshold |
| Top-10% imbalance recall | 40.12% | — |
| Peak combined MAE | 2.7889 | — |
| Off-peak combined MAE | 1.4231 | — |
| Departure interval coverage | 88.72% | 90% target |
| Arrival interval coverage | 87.70% | 90% target |

The two interval-coverages missed target. VeloGuard surfaces this rather than hiding it. The likely causes are symmetric global residual bands, non-stationary demand, and conditional error differences at high-volume periods. A production model should calibrate by station/volume regime and monitor conditional coverage.

## Decision replay

The frozen test was replayed with an initial half-full inventory, common station capacity 30, 30-minute planning cadence, and actual subsequent trip counts.

| Policy | Service failures | Reduction vs no action | Bikes moved | Bike-km |
|---|---:|---:|---:|---:|
| No action | 4,280 | — | 0 | 0 |
| Baseline decisions | 1,557 | 63.62% | 3,467 | 6,578.547 |
| Candidate decisions | 1,652 | 61.40% | 3,268 | 6,103.975 |

Candidate decisions moved 199 fewer bikes and 474.572 fewer bike-km, but caused 95 more simulated failures than baseline decisions. Because neither policy dominates the other and no stakeholder-approved cost weight exists, the forecast candidate remains `shadow_mode_only`. This is the central lesson of the project: a better forecast metric does not automatically justify a decision-policy release.

This is not a real-world or causal result. The simulator assumes every station has capacity 30 and begins half full; processes departures before arrivals within a bucket; and omits trucks, routes, traffic, loading time, existing rebalancing, and human behavior.

## Live applicability

In the verified GBFS collection, 39 trained stations aligned within a 2,509-station system: 32 by guarded normalized name and 7 by coordinate. This is 1.55% network coverage and is intentionally displayed by the API/UI.

Unmatched stations are excluded; they are not silently passed through the model. Production requires expanding the training catalog and governing station identity with effective dates.

## Risk construction

- Prediction intervals are created from calibration absolute-error quantiles.
- Conservative lower/upper projected inventory is compared with configurable safety-bike and safety-dock buffers.
- Risk is a bounded operational score, not a calibrated probability of failure.

The UI says “risk score,” never “probability.”

## Limitations and ethical considerations

- One winter month cannot represent seasonal, weather, event, or long-term network changes.
- Historical trip data contains only completed rides; unmet demand is unobserved.
- The model may prioritize high-volume areas and underserve low-volume neighborhoods if coverage expansion optimizes only aggregate metrics.
- Station names/coordinates can change and create identity errors.
- A dispatcher may over-trust precise-looking forecasts; stale, scope, version, and uncertainty must remain visible.
- Operational experiments affect riders and workers and require human oversight, safety review, and fair service metrics across neighborhoods.

## Monitoring and retraining

Track:

- feed age, station-count changes, schema errors, alignment rate and method;
- MAE/Poisson deviance by station, hour, weekday, and volume;
- interval coverage overall and by segment;
- risk precision/recall after labels mature;
- plan acceptance, unmatched needs, bike moves, bike-km, and downstream service failures;
- drift in station catalog, trip volume, and residual distribution.

Retrain only with a time-based evaluation and the same two-stage gate. A new candidate may become the forecast champion after rolling validation, but may exit shadow mode only when it passes the decision-dominance rule or a stakeholder-approved operational objective; otherwise retain human review and the baseline comparison.

## Reproducibility

```bash
# Linux / WSL
python -m pip install -r requirements.txt
veloguard pipeline
python -m pytest -q
```

The verified archive is about 369 MB. Exact values may differ if the upstream source archive changes; its generated manifest records URL, byte size, and SHA-256.
