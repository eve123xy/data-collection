# Inference-side collection

The serving-campaign pipeline used to collect the inference power traces:
prompt dataset preparation, GPU instance provisioning (vast.ai / Lambda), vLLM
server launch, DCGM telemetry capture, and run upload. Released for
double-blind review.

## Structure

- `scripts/`
  - `pins.py`, `common.py` — pinned constants (dataset revisions, the
    campaign container image, notification endpoint) and small shared
    helpers.
  - `dataset/` — builds and publishes the prompt pools (ShareGPT pairs,
    BurstGPT replay windows, fixed-length ISL/OSL prompts).
  - `cells/` — one run ("cell") lifecycle: provisioning, launch, status,
    tracker sync, watchdog, and notifications.
  - `workload/` — the request-driving client against the vLLM server and its
    metrics scraper.
  - `telemetry/` — DCGM power/clock capture, per-run summarization and
    plotting.
  - `upload/` — pushes a completed run to the shared run store.
- `tools/` — one-off maintenance scripts (audits, backfills, guards).
- `tests/` — unit tests for the modules above.
- `run_plans/` — one Markdown plan per provisioning batch, written before
  dispatch and appended with the outcome.
- `docker/` — the pinned campaign container image definition.
- `pyproject.toml` — dependencies.

## Anonymization note

This repository is released for double-blind review. Hugging Face dataset/run
repository owners, a personal notification endpoint, and a Docker Hub account
have been replaced by placeholders (`<hf-owner>`, `<notify-topic>`,
`<dockerhub-account>`). Set `HF_TOKEN`, `VAST_API_KEY`, `LAMBDA_API_KEY`, and
the other variables in `scripts/.env_example` in the environment before
running any script.

## Scope

This is the active collection pipeline only. The source repository also
contains campaign planning documents, a prompt-dataset justification analysis
(notebooks and figures), a run-tracker spreadsheet, and a read-only vendored
copy of an earlier, separate pipeline kept for reference; none of those are
included here.
