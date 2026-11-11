# Project release review — 11 November 2026

Changes are local, uncommitted and not deployed. The current workspace is AutoCar-Sim; CASS and PAQS remain independent repositories under `.project-reviews/` (ignored by the parent repository). Existing user media and original checkpoints were preserved.

## AutoCar-Sim

Implemented frame-aware RGB/depth queue reads, real sensor frame IDs and simulator timestamps, recording identifiers, cleanup of owned traffic actors, map-load handling, and refusal to overwrite collection outputs. Autopilot labels explicitly represent the control chosen from that observation for the next tick. Manual collection pairs images with cached frame snapshots; control reads remain approximate because the external manual-control client does not supply atomic per-frame labels. It is labelled accordingly in telemetry.

Training now uses whole recordings when at least three exist, or chronological partitions separated by an explicit frame embargo for legacy single recordings. A fingerprinted `splits.json` travels with each model. Standalone evaluation defaults to those held-out indices and rejects changed datasets; `--external-dataset` is an explicit separate-data mode. Missing depth/RGB files fail with actionable errors. Brightness augmentation changes RGB intensity without corrupting depth. Tuning ranks validation MAE, not test MAE. Legacy dashboard metrics and illustrative DQN values are labelled.

Verification:

- Full TensorFlow suite: **13 passed**.
- Live collection: **20 consecutive RGB/depth pairs**, loaded as `(20,66,200,4)` and checked for consecutive sensor frames. Initial map-reload timeout was fixed and the rerun passed.
- One-epoch synthetic pipeline: saved split and standalone evaluation both used **45 test samples**, with identical MAE/MSE. This validates plumbing, not driving quality.
- Six live CARLA evaluations of the original checkpoints: three seeds/spawn points each, 300 frames per run, no traffic, Town10HD_Opt. Baseline completed all three without collision flags. Augmented collided and stalled in all three.

| Model | Seed/spawn | Distance m | Average km/h | Collision-flagged frames |
|---|---|---:|---:|---:|
| Baseline | 42 / 0 | 92.97 | 22.37 | 0 |
| Baseline | 43 / 1 | 93.03 | 22.39 | 0 |
| Baseline | 44 / 2 | 93.07 | 22.39 | 0 |
| Augmented | 42 / 0 | 32.47 | 7.83 | 179 |
| Augmented | 43 / 1 | 47.09 | 11.33 | 130 |
| Augmented | 44 / 2 | 29.97 | 7.22 | 183 |

Collision-flagged frames count repeated contact, not distinct crashes. The map/weather suite is not exhaustive, weather was not recorded in this benchmark, and original checkpoint training-split provenance is unknown. The augmented checkpoint is **not ready to showcase as robust autonomous driving**. Baseline is the better candidate for a limited, accurately described demonstration; wider traffic/weather validation and new independent-recording training remain necessary.

Raw benchmark: `docs/validation/carla-20261111.json`. Screenshot: `docs/images/release-dashboard.png`.

## CASS

Added SQL AST validation, read-only execution, bounded result rows and query timeouts; fixed interpolated table names and parameters. Data routes require a bearer token unless anonymous local development is explicitly enabled. The frontend sends credentials in headers through fetch-based SSE, handles split event chunks and premature termination, and uses configurable backend routing. Production build dependencies were upgraded; setup commands, documented configuration and attribution were corrected.

Verification: **13 backend tests passed**, including an actual disposable PostgreSQL 18 cluster with a restricted role. Write denial, row limits, timeout recovery, authentication, streaming failures and database-value serialization were exercised. **3 browser-client stream tests passed**; lint and production build passed. Dependency audit reported zero vulnerabilities after upgrades. Browser UI executed a real count query against seeded data. The installed `qwen3:4b` answered **3/3 reference SQL questions**; the fallback `sqlcoder:7b` is not installed and was not validated. Results: `.project-reviews/CASS/docs/local-model-validation.json`.

The old README credited Cat Communist Party and local commits identify Rayan Rahman; those facts are preserved. It advertised MIT but no LICENSE file is present. This update does not invent a license. Production hosting, rate/concurrency policy and license clarification remain release work. Test database contents are disposable and separate from user databases.

## PAQS — revised scope: live API project

Per the user's follow-up, the default project now uses current Open-Meteo weather and CAMS air-quality forecast APIs, without old datasets or API keys. Ten Pune coordinates, 1–72 hour API horizons, a live Streamlit dashboard, CSV export, India-time display, source attribution and freshness metadata are implemented. Requests cache for 15 minutes; the open dashboard checks every 15 minutes. Outages may use cache for up to two hours with an explicit stale label, then return unavailable. Missing values stay missing. No random data is substituted.

These are regional provider forecasts on the **US EPA AQI scale**, not Indian CPCB station measurements or predictions by the original project models. Nearby locations may share the ~45 km CAMS global grid. Public provider access is for non-commercial use, subject to provider terms/rate limits and availability; it is not a guarantee of perpetual free hosting or service.

A real 24-hour Pune weather/air-quality response and the rendered dashboard were verified. Tests cover source labels, cache hits/expiry, outages, null values, parameter validation and the live endpoint. Original ML code is retained with chronological multi-station splitting, gap-safe windows, causal cleaning, corrected ARIMA fallback, and optional station/scaler/checkpoint inference bundles. Old model scores have been removed from the live dashboard.

Validation history: the broad suite initially produced **211 passes, 5 Parquet failures and 1 disabled network test**. All five failures traced to PyArrow requiring NumPy 2 while the installed PyTorch uses NumPy 1. Pinning PyArrow 19.0.1 resolved them. The affected/new suites then passed **84 tests**, with the same network test skipped. Previously passing unaffected ML tests were not unnecessarily repeated. An additional one-epoch simulated bundle-training smoke test completed and exported both models over 25 identical held-out windows; its scores are not real-world performance claims. Original real-data model comparisons were superseded by the requested API conversion and are not claimed reproduced.

Screenshot: `.project-reviews/pune-air-quality-sentinel/docs/screenshots/live-api.png`.

## Reproducibility and handoff

- Tested dependency versions and setup instructions are checked into each repository. PAQS separates the lightweight live app from optional research dependencies.
- AutoCar CI uses tested versions; CASS CI includes a PostgreSQL service plus frontend checks; PAQS CI includes API and ML tests. These workflow files have not yet run on GitHub because changes have not been pushed.
- Screenshots of all three UIs were captured and inspected locally. Existing demo recordings remain intact.
- No commits, pushes, production deployment, or portfolio build have been performed.

The future 3D portfolio still follows project stabilization. Browser-playable AutoCar-Sim requires a browser implementation or streamed native CARLA; it is not implemented by these release repairs.
