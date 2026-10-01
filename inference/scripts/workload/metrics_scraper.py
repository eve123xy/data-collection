"""ON INSTANCE. Sample vLLM's /metrics on a fixed cadence, in its OWN PROCESS.

This exists because the scrape used to be an asyncio task sharing the event loop
with the request drivers. Python's asyncio is single-threaded and cooperative:
nothing preempts a running coroutine. At ~99 concurrent streaming requests the
loop was saturated handling chunk callbacks, so the scraper's sleep expired on
time but it was not SCHEDULED again for up to 118 seconds.

Measured on burstgpt_h200_qwen3-8b_burst_unbounded, 2026-09-03: scrape interval
p50 1.013 s, max 118.277 s, five gaps >2 s all inside the window and all at ~99
concurrent. 11.2% of the window was missing from the average, and
achieved_batch_mean came out 1.992 against an actual ~12.6 -- 6.3x low, on a
field that is a law-fit covariate.

The proof of mechanism is that dcgm_capture.py, sampling the same box over the
same window at 10 Hz, lost nothing -- because it is a separate process. So the
scrape becomes one too.

Two behaviours differ from the old loop, both deliberate:

1. **Fixed wall-clock cadence.** The old code did `await sleep(interval)` AFTER
   each request, so the period was request-time + interval and drift compounded
   under load. This schedules on absolute deadlines, so a slow scrape steals
   from its own slot rather than shifting every later sample.
2. **Gaps are recorded, not silently absorbed.** If a deadline is missed the
   row carries `late_s`, so a starved scrape is visible in the data instead of
   having to be reconstructed afterwards.

    python scripts/workload/metrics_scraper.py --base-url http://127.0.0.1:8000 \
        --out /data/run/vllm_metrics.jsonl --interval 1.0
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/
sys.path.insert(0, str(_Path(__file__).resolve().parent))       # workload/

import pins                                    # noqa: E402
from vllm_metrics import parse_prometheus      # noqa: E402


def scrape_once(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=pins.SCRAPE_INTERVAL_S)
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop after this long; 0 means run until killed")
    a = ap.parse_args()

    url = a.base_url.rstrip("/") + pins.METRICS_PATH
    deadline_end = time.time() + a.max_seconds if a.max_seconds else None
    nxt = time.time()

    with open(a.out, "w", encoding="utf-8") as fh:
        while True:
            if deadline_end and time.time() >= deadline_end:
                break
            scheduled = nxt
            try:
                row = parse_prometheus(scrape_once(url))
            except Exception as e:
                row = {"error": str(e)}
            now = time.time()
            row["ts"] = now
            # How late this sample was against its slot. Zero in normal running;
            # non-zero is the signature that something starved the scraper.
            row["late_s"] = round(now - scheduled, 4)
            fh.write(json.dumps(row) + "\n")
            fh.flush()

            nxt += a.interval
            if nxt <= time.time():
                # Missed one or more whole slots: resynchronise to the next
                # future slot rather than firing a burst of catch-up scrapes.
                missed = int((time.time() - nxt) // a.interval) + 1
                nxt += missed * a.interval
            time.sleep(max(0.0, nxt - time.time()))


if __name__ == "__main__":
    main()
