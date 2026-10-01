"""The per-cell power profile figure.

Stacked panels on ONE shared time axis. Never a second y-scale: a dual-axis
chart lets any correlation be implied by choosing the scaling, and two measures
of different scale get two panels instead.

Colours are the validated categorical order (light mode), which passes the
lightness, chroma, CVD-separation and normal-vision checks on the adjacent
pairlist that line charts use. Three of the eight fall below 3:1 contrast on a
light surface, so identity never rests on colour alone: every figure carries a
legend, and cells with four or fewer GPUs are direct-labelled too.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display on a rented box
import matplotlib.pyplot as plt

from summarise_run import IDLE_CARD_W, load_capture

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TOTAL_INK = "#1a1a19"            # the aggregate is not one identity among peers
GRID_INK = "#d9d8d2"
WARMUP_INK = "#f0efe9"
MUTED_INK = "#8a8981"


def panel_count(cell_row):
    """3 normally; 4 on GPU Frequency cells, where SM clock gets its own panel
    rather than a right-hand axis."""
    return 4 if cell_row["sheet"] == "GPU Frequency" else 3


def _locked_clock(cell_row):
    for cmd in cell_row.get("pre_launch") or []:
        m = re.search(r"-lgc\s+(\d+)", cmd)
        if m:
            return float(m.group(1))
    return None


def _read_timeline(run_dir, cell_id):
    p = Path(run_dir) / f"timeline_{cell_id}.csv"
    if not p.exists():
        return []
    with open(p, newline="") as fh:
        return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def plot_run(run_dir, cell_row, run_meta, out_path):
    run_dir, cid = Path(run_dir), cell_row["cell_id"]
    t0 = run_meta["window_start_ts_ns"] / 1e9
    t1 = run_meta["window_end_ts_ns"] / 1e9

    by_gpu = {}
    for p in sorted(run_dir.glob(f"dcgm_{cid}_gpu*.csv")):
        rows = [r for r in load_capture(p) if r.get("ts") is not None]
        if rows:
            by_gpu[rows[0]["gpu_id"]] = sorted(rows, key=lambda r: r["ts"])

    n = panel_count(cell_row)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n), sharex=True,
                             gridspec_kw={"height_ratios": [2.2] + [1] * (n - 1)})
    for ax in axes:
        ax.grid(True, color=GRID_INK, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # --- panel 1: power, PER GPU ---
    # Per-GPU only, deliberately. Total (~2.4 kW on a TP4 cell) against per-GPU
    # (~600 W) on one axis squashes the per-GPU lines into the bottom fifth and
    # hides exactly the imbalance this panel exists to reveal. The total is in
    # the title, and in summary.json; what you need to SEE is the spread.
    ax, lo, hi = axes[0], None, None
    win_end = t1 - t0
    labels = []                 # (y_at_window_end, gid, colour)
    for i, (gid, rows) in enumerate(sorted(by_gpu.items())):
        t = [r["ts"] - t0 for r in rows]
        p = [r.get("power_w") for r in rows]
        lo = min(t) if lo is None else min(lo, min(t))
        hi = max(t) if hi is None else max(hi, max(t))
        colour = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(t, p, lw=1.3, color=colour, label=f"GPU {gid}")
        # Label at the WINDOW END, not the capture end: during drain every card
        # converges and the labels pile up on each other.
        inwin = [(tt, pp) for tt, pp in zip(t, p) if pp is not None and tt <= win_end]
        if inwin:
            labels.append([inwin[-1][1], gid, colour])

    if labels and len(labels) <= 4:
        # Nudge apart any labels closer than 4% of the y-range. Cards in a
        # healthy TP cell sit within a few watts of each other, so unnudged
        # labels overprint precisely when the trace looks correct.
        span = max(l[0] for l in labels) - min(l[0] for l in labels)
        floor_gap = max(span, 1.0) * 0.22
        labels.sort()
        for a, b in zip(labels, labels[1:]):
            if b[0] - a[0] < floor_gap:
                b[0] = a[0] + floor_gap
        for y, gid, colour in labels:
            ax.annotate(f"GPU {gid}", (win_end, y), xytext=(6, 0),
                        textcoords="offset points", fontsize=7, va="center",
                        color=colour, annotation_clip=False)

    per_gpu_mean = None
    if by_gpu:
        means = []
        for rows in by_gpu.values():
            vals = [r["power_w"] for r in rows
                    if r.get("power_w") is not None and t0 <= r["ts"] <= t1]
            if vals:
                means.append(sum(vals) / len(vals))
        if means:
            per_gpu_mean = sum(means) / len(means)
            ax.axhline(per_gpu_mean, color=TOTAL_INK, lw=0.9, ls="--", alpha=0.7)
            ax.annotate(f"mean {per_gpu_mean:.0f} W/GPU", (0, per_gpu_mean),
                        xytext=(6, 4), textcoords="offset points", fontsize=7,
                        color=TOTAL_INK)

    idle_caps = sorted(run_dir.glob(f"dcgm_idle_{cid}_gpu*.csv"))
    if idle_caps:
        vals = [r["power_w"] for p in idle_caps for r in load_capture(p)
                if r.get("power_w") is not None]
        if vals:
            floor = sum(vals) / len(vals)          # per GPU, like the gate
            ax.axhline(floor, color=MUTED_INK, lw=0.9, ls=":")
            ax.annotate(f"idle floor {floor:.0f} W/GPU", (0, floor), xytext=(6, 4),
                        textcoords="offset points", fontsize=7, color=MUTED_INK)

    # The gate that fails a cell is per-GPU, so show the same threshold here.
    ax.axhspan(0, IDLE_CARD_W, color="#f7e9e9", zorder=0)
    ax.annotate(f"below {IDLE_CARD_W:.0f} W/GPU = idle card, run fails",
                (win_end, IDLE_CARD_W), xytext=(-6, -11),
                textcoords="offset points", fontsize=6.5, color="#a33", ha="right")

    ax.set_ylabel("power (W per GPU)")
    total_txt = (f"   ·   mean total {per_gpu_mean * len(by_gpu):.0f} W"
                 if per_gpu_mean else "")
    ax.set_title(f"{cid}   ·   {len(by_gpu)}x {cell_row['gpu']}   ·   "
                 f"{cell_row['model']}{total_txt}", fontsize=10, loc="left")
    if len(by_gpu) > 1:
        ax.legend(frameon=False, fontsize=7, ncol=min(len(by_gpu), 5), loc="lower right")

    # --- panels 2-3: load ---
    tl = _read_timeline(run_dir, cid)
    ts_rel = [r["time_relative_s"] for r in tl]
    axes[1].bar(ts_rel, [r["requests_arrived"] or 0 for r in tl],
                width=1.0, color=SERIES_COLORS[0])
    axes[1].set_ylabel("arrivals / s")
    axes[2].plot(ts_rel, [r["active_requests"] for r in tl], lw=1.4,
                 color=SERIES_COLORS[1])
    axes[2].set_ylabel("active reqs")

    # --- panel 4: SM clock, GPU Frequency cells only ---
    if n == 4:
        for i, (gid, rows) in enumerate(sorted(by_gpu.items())):
            axes[3].plot([r["ts"] - t0 for r in rows],
                         [r.get("sm_clock_mhz") for r in rows], lw=1.2,
                         color=SERIES_COLORS[i % len(SERIES_COLORS)])
        lock = _locked_clock(cell_row)
        if lock:
            axes[3].axhline(lock, color=TOTAL_INK, lw=0.9, ls="--")
            if lo is not None:
                axes[3].annotate(f"lock {lock:.0f} MHz", (lo, lock), xytext=(4, 3),
                                 textcoords="offset points", fontsize=7)
        axes[3].set_ylabel("SM clock (MHz)")

    for ax in axes:
        if lo is not None and lo < 0:
            ax.axvspan(lo, 0, color=WARMUP_INK, zorder=0)
        if hi is not None and hi > (t1 - t0):
            ax.axvspan(t1 - t0, hi, color=WARMUP_INK, zorder=0)
    axes[-1].set_xlabel("time relative to window start (s)   ·   "
                        "shaded = warm-up and drain, excluded from the window")

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
