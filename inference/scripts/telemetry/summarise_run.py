"""Slice the capture to the measurement window, aggregate it, and gate it.

    python scripts/summarise_run.py --cell-id <id> --out /data/run

Aggregation is grouped by gpu_id. OPERATIONAL_LEARNINGS section 3.3 calls the
ungrouped form "a gate on running any multi-GPU sweep at all": ungrouped, a
multi-GPU run produces an invalid energy-per-token - the headline metric - with
no error and a plausible magnitude, and the defect scales with GPU count, so it
corrupts exactly the tensor-parallel comparison the sweep exists to make.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import csv
import json
import re
import sys
from collections import defaultdict

import pins
from pathlib import Path


def load_capture(path):
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                if k in ("entity_type", "gpu_id"):
                    row[k] = v
                elif v == "" or v is None:
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = None
            rows.append(row)
    return rows


def slice_window(rows, t0, t1):
    return [r for r in rows if r["ts"] is not None and t0 <= r["ts"] <= t1]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def per_gpu(rows):
    """Per-card statistics. Never mix cards before this point."""
    by = defaultdict(list)
    for r in rows:
        by[r["gpu_id"]].append(r)

    out = {}
    for gid, rs in by.items():
        rs = sorted(rs, key=lambda r: r["ts"])
        power = [r.get("power_w") for r in rs]
        clocks = [r.get("sm_clock_mhz") for r in rs]
        # Memory clock has been captured per sample all along (DCGM field 101)
        # but was never surfaced. It is a direct lever on power and on decode
        # throughput, so an unnoticed change in it is indistinguishable from a
        # real effect. On 2026-09-07 a box was left with its memory clock pinned
        # at 1593 instead of 2619 by a deferred lock that did not clear; under a
        # heavy matmul it drew 123 W where an H100 should draw 400-700 W. Nothing
        # in the pipeline would have caught that. Summarised here so a throttled
        # card is visible rather than published.
        mclocks = [r.get("mem_clock_mhz") for r in rs]
        mvals = [c for c in mclocks if c is not None]
        out[gid] = {
            "samples": len(rs),
            "mean_power_w": _mean(power),
            "peak_power_w": max((p for p in power if p is not None), default=None),
            "energy_j": None,        # filled by aggregate(), bracketed - see below
            "mean_sm_clock_mhz": _mean(clocks),
            "mean_mem_clock_mhz": _mean(mclocks),
            "min_mem_clock_mhz": min(mvals, default=None),
            "max_mem_clock_mhz": max(mvals, default=None),
            "sm_clocks": [c for c in clocks if c is not None],
            "powers": [p for p in power if p is not None],
            "max_gap_s": max((b["ts"] - a["ts"] for a, b in zip(rs, rs[1:])), default=0.0),
        }
    return out


def _bracketed_energy(rows, gid, t0, t1):
    """Energy for one GPU from the counter samples either side of the window."""
    s = sorted((r for r in rows
                if r["gpu_id"] == gid and r.get("total_energy_mj") is not None),
               key=lambda r: r["ts"])
    if len(s) < 2:
        return None
    before = [r for r in s if r["ts"] <= t0]
    after = [r for r in s if r["ts"] >= t1]
    lo = before[-1] if before else s[0]
    hi = after[0] if after else s[-1]
    return (hi["total_energy_mj"] - lo["total_energy_mj"]) / 1000.0


def aggregate(rows, t0, t1):
    """Window totals. Sums are over per-GPU values, never over raw rows."""
    win = slice_window(rows, t0, t1)
    if not win:
        return {"n_gpus": 0, "samples": 0, "mean_power_w": None,
                "energy_j": None, "max_sample_gap_s": None, "per_gpu": {},
                "_per_gpu_full": {}}
    g = per_gpu(win)
    # Energy is bracketed across the window, not differenced between interior
    # samples: an interior delta silently omits everything accumulated in the
    # first and last sample interval.
    for gid, gv in g.items():
        gv["energy_j"] = _bracketed_energy(rows, gid, t0, t1)
    powers = [v["mean_power_w"] for v in g.values() if v["mean_power_w"] is not None]
    energies = [v["energy_j"] for v in g.values() if v["energy_j"] is not None]
    return {
        "n_gpus": len(g),
        "samples": len(win),
        "mean_power_w": sum(powers) if powers else None,          # summed per GPU
        "peak_power_w": sum(v["peak_power_w"] for v in g.values()
                            if v["peak_power_w"] is not None) or None,
        "energy_j": sum(energies) if energies else None,          # summed per GPU
        "mean_power_per_gpu_w": (sum(powers) / len(powers)) if powers else None,
        "max_sample_gap_s": max(v["max_gap_s"] for v in g.values()),
        "window_s": t1 - t0,
        "per_gpu": {k: {kk: vv for kk, vv in v.items()
                        if kk not in ("sm_clocks", "powers")}
                    for k, v in g.items()},
        "_per_gpu_full": g,
    }


# A card drawing less than this WHILE SERVING is idle, not a result. The archive
# paired working runs with hung-run captures reading 115-145 W against a real
# 599 W - 19 of 93 cells, and it inverted a tensor-parallel conclusion that
# would have shipped (OPERATIONAL_LEARNINGS section 3.1 corollary).
IDLE_CARD_W = 150.0
ENERGY_TOLERANCE = 0.02      # counter vs integral(P dt)
CLOCK_TOLERANCE_MHZ = 15.0
CLOCK_MIN_HELD_FRAC = 0.99
MAX_SAMPLE_GAP_S = 0.5
TOKEN_TOLERANCE = 0.05


def _locked_clock(cell_row):
    for cmd in cell_row.get("pre_launch") or []:
        m = re.search(r"-lgc\s+(\d+)", cmd)
        if m:
            return float(m.group(1))
    return None


def run_gates(agg, cell_row, client_tokens=None, server_tokens=None):
    """Return (fatal, flags). Fatal means the cell's numbers must not be used."""
    fatal, flags = [], []
    cid = cell_row["cell_id"]

    if agg.get("samples", 0) == 0:
        return [f"{cid}: no telemetry samples inside the measurement window"], []

    per_gpu_w = agg.get("mean_power_per_gpu_w")
    if per_gpu_w is not None and per_gpu_w < IDLE_CARD_W:
        fatal.append(f"{cid}: mean power {per_gpu_w:.1f} W per GPU is below "
                     f"{IDLE_CARD_W:.0f} W - that is an idle card, not a result")

    energy, power, window = agg.get("energy_j"), agg.get("mean_power_w"), agg.get("window_s")
    if energy and power and window:
        integral = power * window
        rel = abs(energy - integral) / max(integral, 1.0)
        if rel > ENERGY_TOLERANCE:
            fatal.append(f"{cid}: energy counter {energy:.0f} J disagrees with "
                         f"integral(P dt) {integral:.0f} J by {100 * rel:.1f}% "
                         f"(tolerance {100 * ENERGY_TOLERANCE:.0f}%)")

    lock = _locked_clock(cell_row)
    if lock is not None:
        for gid, g in agg.get("_per_gpu_full", {}).items():
            clocks = g.get("sm_clocks") or []
            if not clocks:
                continue
            held = sum(1 for c in clocks if abs(c - lock) <= CLOCK_TOLERANCE_MHZ)
            frac = held / len(clocks)
            if frac < CLOCK_MIN_HELD_FRAC:
                fatal.append(f"{cid}: GPU {gid} SM clock held {lock:.0f} MHz for "
                             f"only {100 * frac:.1f}% of samples - the control "
                             f"variable did not bind")

    gap = agg.get("max_sample_gap_s")
    if gap is not None and gap > MAX_SAMPLE_GAP_S:
        flags.append(f"{cid}: DCGM sample gap {gap:.2f}s exceeds "
                     f"{MAX_SAMPLE_GAP_S}s - multiplexing or a stalled capture")

    if client_tokens and server_tokens:
        rel = abs(client_tokens - server_tokens) / max(server_tokens, 1)
        if rel > TOKEN_TOLERANCE:
            flags.append(f"{cid}: client tokens {client_tokens:,} vs server "
                         f"{server_tokens:,} differ by {100 * rel:.1f}%")

    return fatal, flags


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return float(s[min(len(s) - 1, int(round(q / 100 * len(s) + 0.5)) - 1)])


def power_percentiles(agg):
    """Percentiles over the pooled per-GPU power samples."""
    pooled = []
    for g in agg.get("_per_gpu_full", {}).values():
        pooled.extend(g.get("powers") or [])
    return {"p50_power_w": _pct(pooled, 50), "p95_power_w": _pct(pooled, 95),
            "p99_power_w": _pct(pooled, 99),
            "max_power_w": max(pooled) if pooled else None}


def ramp_rates(rows, t0, t1):
    """dP/dt distribution from the 100 ms series, computed within each GPU."""
    by = defaultdict(list)
    for r in slice_window(rows, t0, t1):
        if r.get("power_w") is not None:
            by[r["gpu_id"]].append((r["ts"], r["power_w"]))
    slopes = []
    for series in by.values():
        series.sort()
        for (ta, pa), (tb, pb) in zip(series, series[1:]):
            dt = tb - ta
            if dt > 0:
                slopes.append((pb - pa) / dt)
    if not slopes:
        return {"max_abs_dpdt_w_per_s": None, "p95_abs_dpdt_w_per_s": None}
    absl = [abs(x) for x in slopes]
    return {"max_abs_dpdt_w_per_s": max(absl),
            "p95_abs_dpdt_w_per_s": _pct(absl, 95)}


def derive_metrics(agg, idle_agg, server_tokens):
    """Headline quantities. Any denominator we did not measure stays None."""
    energy = agg.get("energy_j")
    idle_w = idle_agg.get("mean_power_w") if idle_agg else None
    window = agg.get("window_s")
    out = {
        "energy_j": energy,
        "mean_power_w": agg.get("mean_power_w"),
        "mean_power_per_gpu_w": agg.get("mean_power_per_gpu_w"),
        "idle_power_w": idle_w,
        "energy_above_idle_j": (energy - idle_w * window)
                               if (energy and idle_w and window) else None,
        "j_per_token": (energy / server_tokens)
                       if (energy and server_tokens) else None,
        "n_gpus": agg.get("n_gpus"),
        "samples": agg.get("samples"),
    }
    out.update(power_percentiles(agg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--out", default="/data/run")
    ap.add_argument("--manifest", default="plan/cells_v1.json")
    args = ap.parse_args()
    out = Path(args.out)

    meta_path = out / "run_meta.json"
    if not meta_path.exists():
        sys.exit(f"[FATAL] {meta_path} missing - run the workload first")
    run_meta = json.loads(meta_path.read_text())
    t0 = run_meta["window_start_ts_ns"] / 1e9
    t1 = run_meta["window_end_ts_ns"] / 1e9

    cells = json.loads(Path(args.manifest).read_text())
    cell_row = next((c for c in cells if c["cell_id"] == args.cell_id), None)
    if cell_row is None:
        sys.exit(f"[FATAL] {args.cell_id} not in {args.manifest}")

    # Files are matched to the run by cell_id, never by sort order (section 3.1).
    caps = sorted(out.glob(f"dcgm_{args.cell_id}_gpu*.csv"))
    if not caps:
        sys.exit(f"[FATAL] no capture matching dcgm_{args.cell_id}_gpu*.csv")
    rows = [r for p in caps for r in load_capture(p)]

    idle_caps = sorted(out.glob(f"dcgm_idle_{args.cell_id}_gpu*.csv"))
    idle_rows = [r for p in idle_caps for r in load_capture(p)]
    idle_agg = None
    if idle_rows:
        its = [r["ts"] for r in idle_rows if r["ts"] is not None]
        idle_agg = aggregate(idle_rows, min(its), max(its))

    agg = aggregate(rows, t0, t1)
    server_tokens = (run_meta.get("server") or {}).get("generation_tokens")
    derived = derive_metrics(agg, idle_agg, server_tokens)
    derived.update(ramp_rates(rows, t0, t1))

    fatal, flags = run_gates(
        agg, cell_row,
        client_tokens=(run_meta.get("attribution") or {}).get("tokens_in_window"),
        server_tokens=server_tokens)

    # Record which field set this cell actually carried, so an Ampere cell's
    # absent DCP columns are a stated property of the run rather than something
    # a later reader has to infer from a hole in the data.
    facts_p = out / "instance_facts.json"
    cap = None
    if facts_p.exists():
        try:
            cap = json.loads(facts_p.read_text()).get("compute_cap")
        except Exception:
            pass
    from dcgm_fields import fields_for
    _, expected_names, _ = fields_for(cap)

    summary = {"cell_id": args.cell_id, "window": {"t0": t0, "t1": t1},
               "compute_cap": cap,
               "dcgm_fields_present": expected_names,
               "telemetry": {k: v for k, v in agg.items() if not k.startswith("_")},
               "derived": derived, "fatal": fatal, "flags": flags}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    # timeline_<cell>.csv and run_record.json are deliverables in every run plan
    # ("for the data team"), and export_record.py builds both -- but NOTHING EVER
    # CALLED IT. build_timeline and build_run_record were dead code, so the
    # timeline CSV was never written, _read_timeline returned [], and the
    # arrivals/s and active-reqs panels of every power_profile plot were blank.
    # The data was always there in requests_<cell>.jsonl and vllm_metrics.jsonl;
    # nothing assembled it.
    try:
        from export_record import build_run_record, build_timeline
        reqs = []
        rp = out / f"requests_{args.cell_id}.jsonl"
        if rp.exists():
            reqs = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
        samples = []
        mp = out / "vllm_metrics.jsonl"
        if mp.exists():
            samples = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]

        (out / "run_record.json").write_text(
            json.dumps(build_run_record(cell_row, run_meta, agg), indent=1) + "\n")

        tl = build_timeline(reqs, samples, t0, t1, pins.WARMUP_S)
        tl_path = out / f"timeline_{args.cell_id}.csv"
        with open(tl_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tl[0].keys()) if tl else
                               ["time_relative_s", "requests_arrived",
                                "active_requests", "mean_prompt_tokens",
                                "mean_output_tokens"])
            w.writeheader()
            w.writerows(tl)
        print(f"[OK] run_record.json and {tl_path.name} "
              f"({len(tl)} bins from {len(reqs)} requests, {len(samples)} samples)")
    except Exception as e:
        # Never let a reporting artifact take down a completed measurement.
        print(f"[WARN] could not write run_record/timeline ({type(e).__name__}: {e})")

    try:
        from plot_run import plot_run
        fig = plot_run(out, cell_row, run_meta,
                       out / f"power_profile_{args.cell_id}.png")
        print(f"[OK] figure -> {fig}")
    except Exception as e:
        # A missing figure must never cost a cell its data.
        print(f"[WARN] could not render the power profile: {type(e).__name__}: {e}")

    for f in flags:
        print(f"[FLAG] {f}")
    if fatal:
        for f in fatal:
            print(f"[FATAL] {f}")
        sys.exit(1)
    print(f"[OK] {derived['energy_j']:.0f} J, {derived['mean_power_w']:.0f} W "
          f"across {agg['n_gpus']} GPU(s) -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
