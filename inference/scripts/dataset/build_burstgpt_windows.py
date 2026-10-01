"""LOCAL, run once. Derive the three BurstGPT windows and freeze replay files.

Needs the trace and both ShareGPT pools locally.

    python scripts/build_burstgpt_windows.py \
        --trace data/burstgpt/BurstGPT_without_fails_3.csv --publish
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import bisect
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import pins
from common import count_rows, publish, read_jsonl, sha256_file, write_jsonl, write_meta


# ---------- window derivation ----------

def _rolling_sum(counts, width):
    """Sum over [t, t+width) for every t."""
    c = np.concatenate([[0.0], np.cumsum(counts, dtype=np.float64)])
    return c[width:] - c[:-width]


def window_stats(timestamps, response_tokens, window_s):
    """Per-start-second stats for every window. Index i = window starting at second i."""
    if window_s < 60:
        sys.exit("[FATAL] window_s must be >= 60 to measure per-minute peaks")
    sec = np.asarray(timestamps, dtype=np.float64).astype(np.int64)
    tok = np.asarray(response_tokens, dtype=np.float64)
    span = int(sec.max()) + 1

    reqs = np.bincount(sec, minlength=span).astype(np.float64)
    toks = np.bincount(sec, weights=tok, minlength=span).astype(np.float64)

    request_count = _rolling_sum(reqs, window_s)          # len span-window_s+1
    decode_work = _rolling_sum(toks, window_s)

    # Peak requests in any 60 s sub-window lying wholly inside each window.
    # Those start at s in [j, j+window_s-60], so there are window_s-59 of them;
    # rolling at that width keeps `peak` the same length as `request_count`.
    per_min = _rolling_sum(reqs, 60)                      # len span-59
    width = window_s - 59
    peak = pd.Series(per_min).rolling(width, min_periods=width).max().to_numpy()
    peak = peak[width - 1:]

    mean_per_min = request_count / (window_s / 60.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        burstiness = np.nan_to_num(np.where(mean_per_min > 0, peak / mean_per_min, 0.0))

    return {"request_count": request_count, "decode_work": decode_work,
            "burstiness": burstiness, "window_s": window_s}


def derive_window(stats, criterion, min_rate=None):
    """Return (offset_s, req_per_s) for the argmax window under `criterion`.

    `min_rate` restricts the search to windows sustaining at least that many
    req/s. Burstiness needs it: unconstrained, its argmax is a near-empty
    window at a traffic gap that scores the ceiling.
    """
    score = stats[criterion]
    if min_rate is not None:
        rate_all = stats["request_count"] / stats["window_s"]
        score = np.where(rate_all >= min_rate, score, -np.inf)
    offset = int(np.argmax(score))
    rate = float(stats["request_count"][offset]) / stats["window_s"]
    return offset, rate


def measure_at(stats, offset_s):
    """Measured (req_per_s, burstiness, decode_work) at a given start second."""
    return (float(stats["request_count"][offset_s]) / stats["window_s"],
            float(stats["burstiness"][offset_s]),
            float(stats["decode_work"][offset_s]))


def check_window(name, stats):
    """Validate a window against what the archive recorded.

    Always: the measured arrival rate at the recorded offset must match.
    Additionally, where the selection criterion is known, re-deriving it must
    land on the recorded offset - that is what proves the criterion.
    """
    exp = pins.WINDOWS[name]
    offset = exp["offset_s"]
    if offset >= len(stats["request_count"]):
        sys.exit(f"[FATAL] {name}: recorded offset {offset:,} is past the end "
                 f"of the trace - wrong release?")

    rate, burst, work = measure_at(stats, offset)
    if abs(rate - exp["req_per_s"]) / exp["req_per_s"] > pins.RATE_TOLERANCE:
        sys.exit(f"[FATAL] {name}: measured {rate:.3f} req/s at recorded offset "
                 f"{offset:,}, but the archive recorded {exp['req_per_s']:.3f}")

    if exp["criterion"]:
        floor = pins.BURSTINESS_MIN_RATE if exp["criterion"] == "burstiness" else None
        derived, _ = derive_window(stats, exp["criterion"], min_rate=floor)
        # For a shifted window the argmax is checked against where the argmax
        # is expected, not against where we slice.
        expect_argmax = exp.get("argmax_offset_s", offset)
        drift = abs(derived - expect_argmax)
        if drift > pins.OFFSET_TOLERANCE_S:
            sys.exit(f"[FATAL] {name}: argmax of '{exp['criterion']}' is "
                     f"{derived:,}, {drift:,}s from the recorded "
                     f"{expect_argmax:,}. "
                     f"The criterion is wrong - stop and confirm it before "
                     f"publishing.")
        note = (f" (min rate {pins.BURSTINESS_MIN_RATE} req/s)"
                if exp["criterion"] == "burstiness" else "")
        shift = exp.get("shift_s", 0)
        where = (f"slicing at {offset:,} (argmax +{shift}s)" if shift
                 else "slicing at recorded")
        print(f"[OK] {name}: '{exp['criterion']}'{note} argmax is {derived:,}, "
              f"{drift}s from expected {expect_argmax:,}; {where}. "
              f"{rate:.3f} req/s, burstiness {burst:.1f}")
    else:
        print(f"[OK] {name}: offset {offset:,} taken as recorded; measured "
              f"{rate:.3f} req/s, burstiness {burst:.1f}")
    return offset, rate, burst, work


# ---------- prompt matching ----------

class PromptMatcher:
    """Nearest-neighbour by prompt_len ONLY.

    The trace's Response tokens are out of scope under free generation
    (run_plan_settings_v2.md 7), so output_len must never steer which prompt
    text gets picked. Ties break randomly from a seeded RNG.
    """

    def __init__(self, pool, seed):
        self.rng = random.Random(seed)
        self.by_len = {}
        for p in pool:
            self.by_len.setdefault(int(p["prompt_len"]), []).append(p)
        self.lens = sorted(self.by_len)
        self.min_len = self.lens[0]
        # Draw without replacement within a length bucket, so two requests
        # wanting the same length get different prompts while the length match
        # stays exact. Prefix caching is on, and gratuitous repeats would show
        # up as cache hits that inflate throughput.
        self._cursor = {}
        for L, bucket in self.by_len.items():
            self.rng.shuffle(bucket)
            self._cursor[L] = 0

    def match(self, prompt_len):
        pos = bisect.bisect_left(self.lens, prompt_len)
        if pos == 0:
            best = self.lens[0]
        elif pos == len(self.lens):
            best = self.lens[-1]
        else:
            before, after = self.lens[pos - 1], self.lens[pos]
            d_b, d_a = prompt_len - before, after - prompt_len
            best = before if d_b < d_a else after if d_a < d_b else self.rng.choice((before, after))
        bucket = self.by_len[best]
        i = self._cursor[best]
        self._cursor[best] = (i + 1) % len(bucket)   # wraps only if exhausted
        return bucket[i]


def build_replay(rows, matcher):
    """Trace rows -> frozen replay list. Zero-token rows are CLAMPED, not dropped:
    dropping them would quietly cut the window's arrival rate by ~4.5%."""
    replay, clamped, prev = [], 0, None
    for seq, row in enumerate(rows):
        ts, trace_len = float(row["timestamp"]), int(row["prompt_len"])
        if trace_len <= 0:
            clamped += 1
            trace_len = matcher.min_len
        picked = matcher.match(trace_len)
        replay.append({
            "seq": seq,
            "arrival_delta_s": 0.0 if prev is None else round(ts - prev, 6),
            "prompt_text": picked["prompt"],
            "trace_prompt_len": int(row["prompt_len"]),
            "matched_prompt_len": int(picked["prompt_len"]),
        })
        prev = ts
    return replay, clamped


def slice_window(df, offset_s, window_s):
    """Half-open [offset, offset+window), rebased so the first arrival is t=0."""
    sel = df[(df["timestamp"] >= offset_s) & (df["timestamp"] < offset_s + window_s)]
    if sel.empty:
        sys.exit(f"[FATAL] no rows in window [{offset_s}, {offset_s + window_s})")
    return [{"timestamp": float(t) - offset_s, "prompt_len": int(p)}
            for t, p in zip(sel["timestamp"], sel["prompt_len"])]


# ---------- main ----------

def load_trace(path):
    df = pd.read_csv(path, usecols=["Timestamp", "Request tokens", "Response tokens"])
    df.columns = ["timestamp", "prompt_len", "response_len"]
    if len(df) != pins.BURSTGPT_TRACE_ROWS:
        sys.exit(f"[FATAL] trace has {len(df):,} rows, expected "
                 f"{pins.BURSTGPT_TRACE_ROWS:,} - wrong release?")
    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--pool-dir", default="data/sharegpt_pairs")
    ap.add_argument("--out", default="build_out/burstgpt")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    df = load_trace(args.trace)
    stats = window_stats(df["timestamp"].to_numpy(),
                         df["response_len"].to_numpy(), pins.WINDOW_S)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = []

    # Derive and slice every window first (cheap), so each 1.1 GB pool is
    # parsed once rather than once per window.
    windows = {}
    for name in pins.WINDOWS:
        offset, rate, burstiness, decode_work = check_window(name, stats)
        windows[name] = {"offset": offset, "rate": rate, "burstiness": burstiness,
                         "decode_work": decode_work,
                         "rows": slice_window(df, offset, pins.WINDOW_S)}

    for family, pool_repo in pins.POOLS.items():
        pool_path = Path(args.pool_dir) / f"sharegpt_pairs_{family}.jsonl"
        print(f"[..] loading pool {pool_path.name}")
        pool = read_jsonl(pool_path)
        for name, w in windows.items():
            replay, clamped = build_replay(w["rows"], PromptMatcher(pool, pins.SHUFFLE_SEED))
            out = out_dir / f"{name}-{family}.jsonl"
            write_jsonl(out, replay)
            meta = {
                "artifact": f"burstgpt/{name}-{family}.jsonl",
                "window": name, "family": family, "pool": pool_repo,
                "criterion": pins.WINDOWS[name]["criterion"],
                "argmax_offset_s": pins.WINDOWS[name].get("argmax_offset_s"),
                "shift_s": pins.WINDOWS[name].get("shift_s", 0),
                "shift_reason": pins.WINDOWS[name].get("shift_reason"),
                "offset_s": w["offset"], "req_per_s": w["rate"],
                "burstiness_peak_min_over_mean_min": round(w["burstiness"], 3),
                "trace_decode_tokens": int(w["decode_work"]),
                "window_s": pins.WINDOW_S, "scale_c": 1.0,
                "clamped_rows": clamped,
                "distinct_prompts": len({r["prompt_text"] for r in replay}),
                "rows": count_rows(out), "sha256": sha256_file(out),
                "built_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_meta(out_dir / f"{name}-{family}.meta.json", meta)
            metas.append(meta)
            print(f"     {out.name}: {meta['rows']:,} req, {clamped} clamped, "
                  f"{meta['distinct_prompts']:,} distinct prompts")
        del pool

    write_meta(out_dir / "windows.meta.json", {"windows": metas})
    if args.publish:
        publish(pins.DATASETS_REPO, out_dir.parent)


if __name__ == "__main__":
    main()
