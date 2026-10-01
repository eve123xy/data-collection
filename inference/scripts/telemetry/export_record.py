"""Projections of run_meta.json for the data-management team.

run_meta.json stays the single canonical home for every fact (PIPELINE_POSTMORTEM
R4); these are derived views of it, never a second hand-maintained truth.

Their instruction on missing values is explicit: "If a value was not recorded or
does not apply, leave it blank/null rather than estimating it."
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import re
from collections import defaultdict

TIMELINE_COLUMNS = ["time_relative_s", "requests_arrived", "active_requests",
                    "mean_prompt_tokens", "mean_output_tokens"]


def _flag_value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def build_run_record(cell_row, run_meta, agg):
    """The data team's field vocabulary, derived mechanically."""
    w = cell_row["workload"]
    args = cell_row.get("vllm_args") or []

    gpu_freq = None
    for cmd in cell_row.get("pre_launch") or []:
        m = re.search(r"-lgc\s+(\d+)", cmd)
        if m:
            gpu_freq = int(m.group(1))

    kv = _flag_value(args, "--kv-cache-dtype")
    server = run_meta.get("server") or {}

    return {
        "cell_id": cell_row["cell_id"],
        "model": cell_row["model"],
        "gpu_type": cell_row["gpu"],
        "tp_size": cell_row["tp"],
        "kv_cache_quant": None if kv in (None, "auto") else kv,
        "weight_quant": _flag_value(args, "--quantization"),
        "gpu_frequency": gpu_freq,
        "concurrency": w.get("in_flight"),
        "arrival_rate": w.get("rate"),
        "compression_factor": w.get("scale"),
        "workload_pattern": w["mode"],
        "thinking_mode": w.get("thinking"),
        "serving_engine": f"vLLM {run_meta.get('vllm_version')}",
        "window_duration_s": w.get("duration"),
        "requests_total": run_meta.get("requests_total"),
        "prompt_tokens_total": server.get("prompt_tokens"),
        "generation_tokens_total": server.get("generation_tokens"),
        "mean_power_w": agg.get("mean_power_w"),
        "energy_j": agg.get("energy_j"),
        "n_gpus": agg.get("n_gpus"),
    }


def build_timeline(requests, metrics_samples, t0, t1, warmup_s):
    """1 s bins on the relative-time axis shared with the GPU telemetry.

    Warm-up bins are negative, the window runs 0 -> duration. Binning at 1 s lets
    active_requests come from vLLM's own num_requests_running gauge rather than
    being reconstructed from request records.
    """
    lo = int(-warmup_s)
    hi = int(round(t1 - t0))
    arrivals = defaultdict(list)
    for r in requests:
        b = int((r["send_ts"] - t0) // 1)
        arrivals[b].append(r)

    running = {}
    for s in metrics_samples:
        if "ts" not in s:
            continue
        running[int((s["ts"] - t0) // 1)] = s.get("vllm:num_requests_running")

    out = []
    for b in range(lo, hi):
        rs = arrivals.get(b, [])
        out.append({
            "time_relative_s": float(b),
            "requests_arrived": len(rs),
            "active_requests": running.get(b),
            "mean_prompt_tokens": (sum(r["prompt_tokens"] for r in rs) / len(rs)
                                   if rs else None),
            "mean_output_tokens": (sum(r["completion_tokens"] for r in rs) / len(rs)
                                   if rs else None),
        })
    return out
