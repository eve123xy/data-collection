"""Drive a vLLM server for one cell and record everything.

    python scripts/run_workload.py --mode closed  --in-flight 32 --duration 600 \
        --model Qwen/Qwen3-8B --family qwen3 --osl forced --cell-id kv_h100_32b_bf16
    python scripts/run_workload.py --mode replay  --replay /data/replay.jsonl ...
    python scripts/run_workload.py --mode poisson --rate 2.0 --duration 1200 ...
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

import pins

_WRAP = {}   # filled by run_closed only when --allow-wrap actually recycles
from common import read_jsonl, write_meta
import vllm_metrics as vm            # metric NAMES live in one place there
from vllm_metrics import (cross_check, parse_log_line, parse_prometheus,
                          parse_startup)
from workload import (mark_window, poisson_schedule, replay_schedule,
                      run_closed, run_open)


def _pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[min(len(s) - 1, int(round(p / 100 * len(s) + 0.5)) - 1)])


def summarise(samples, t0, t1):
    """Window statistics from the /metrics time series.

    Gauges (running, waiting, KV usage) are averaged over samples INSIDE the
    window. Counters (tokens, preemptions, prefix cache) are differenced across
    samples that BRACKET it - the last at or before t0 and the first at or after
    t1 - because a counter delta between two interior samples silently omits
    everything generated in the first and last scrape interval.
    """
    win = [s for s in samples if t0 <= s.get("ts", -1) <= t1]
    if not win:
        return {"achieved_batch_mean": 0.0, "achieved_batch_p95": 0.0,
                "achieved_batch_max": 0.0, "waiting_p95": 0.0, "kv_usage_p95": None,
                "preemptions": 0, "generation_tokens": 0, "prompt_tokens": 0,
                "prefix_hit_pct": None, "samples": 0}

    run = [s.get(vm.M_RUNNING, 0.0) for s in win]
    wait = [s.get(vm.M_WAITING, 0.0) for s in win]
    # None, not 0.0, when the metric is absent: a fabricated zero here is
    # indistinguishable from a KV cache that genuinely sat empty.
    kv = [100.0 * v for v in (s.get(vm.M_KV_USAGE) for s in win) if v is not None]

    ordered = sorted((s for s in samples if "ts" in s), key=lambda s: s["ts"])
    before = [s for s in ordered if s["ts"] <= t0]
    after = [s for s in ordered if s["ts"] >= t1]
    lo = before[-1] if before else win[0]
    hi = after[0] if after else win[-1]

    def delta(key):
        return (hi.get(key, 0.0) or 0.0) - (lo.get(key, 0.0) or 0.0)

    q = delta(vm.M_PREFIX_Q)
    h = delta(vm.M_PREFIX_H)

    return {
        "achieved_batch_mean": round(sum(run) / len(run), 3),
        "achieved_batch_p95": _pct(run, 95),
        "achieved_batch_max": max(run),
        "waiting_p95": _pct(wait, 95),
        "kv_usage_p95": round(_pct(kv, 95), 3) if kv else None,
        "preemptions": int(delta(vm.M_PREEMPT)),
        "generation_tokens": int(delta(vm.M_GEN_TOK)),
        "prompt_tokens": int(delta(vm.M_PROMPT_TOK)),
        "prefix_hit_pct": round(100.0 * h / q, 4) if q > 0 else None,
        "samples": len(win),
        "counter_bracket_s": round(hi["ts"] - lo["ts"], 3) if "ts" in lo and "ts" in hi else None,
    }


async def scrape_loop(client, base_url, out_path, stop_event, interval):
    """Sample /metrics on a fixed cadence until told to stop."""
    samples = []
    with open(out_path, "w", encoding="utf-8") as fh:
        while not stop_event.is_set():
            try:
                r = await client.get(base_url + pins.METRICS_PATH, timeout=5.0)
                row = parse_prometheus(r.text)
                row["ts"] = time.time()
                samples.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            except Exception as e:
                fh.write(json.dumps({"ts": time.time(), "error": str(e)}) + "\n")
                fh.flush()
            await asyncio.sleep(interval)
    return samples


def _git_commit():
    """Commit of the RUNNING code. See record_run._git_commit -- an instance has
    no .git, so this returned None and run_meta.git_commit was silently null."""
    stamp = Path(__file__).resolve().parents[1] / "CAMPAIGN_COMMIT"
    try:
        v = stamp.read_text().strip()
        if v:
            return v
    except Exception:
        pass
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def build_flags(args, stats, check, records):
    """Run-level QC flags. A flagged run still produces data; it is not fatal."""
    flags = []
    # prefix_hit_pct is RECORDED but no longer flagged. The 2.0% threshold
    # assumed that distinct prompts imply a near-zero hit rate, and that is
    # simply wrong: every request carries the same Qwen3 chat-template preamble,
    # which prefix caching serves from cache after the first. The observed rate
    # is 15-18% on every cell, structurally, so the flag fired on all of them and
    # would have fired on all 248 -- a status column where every row says the
    # same thing carries no information, and it buries the cells that are
    # genuinely interesting.
    #
    # Nothing is lost: prefix_hit_pct stays in run_meta.json for every cell, so
    # any post-hoc check on prompt reuse remains possible from the recorded data.
    # This is deliberately NOT a raised threshold -- moving a number so runs pass
    # is the one edit the freeze forbids. The measurement is unchanged; only the
    # decision to shout about it is removed.
    if args.mode == "closed" and args.in_flight == 32 and stats["preemptions"] > 0:
        flags.append(f"{stats['preemptions']} preemptions at batch 32 - "
                     "misconfiguration")
    # logging_metrics.md sanity check 3
    if args.mode == "closed" and stats["samples"]:
        if stats["achieved_batch_max"] > args.in_flight:
            flags.append(f"achieved batch max {stats['achieved_batch_max']} exceeds "
                         f"configured cap {args.in_flight}")
        if stats["achieved_batch_mean"] < 0.9 * args.in_flight:
            flags.append(f"achieved batch mean {stats['achieved_batch_mean']} is "
                         f"below 0.9x cap {args.in_flight} - cell did not saturate, "
                         f"or the KV cache bound it below the cap")
    # getattr, not args.client_pool: _amain sets it, but build_flags is also
    # called directly (tests, replays) and a missing attribute must not take
    # down flag generation for a completed cell.
    pool = getattr(args, "client_pool", None)
    if pool and (stats.get("achieved_batch_max") or 0) >= 0.9 * pool:
        flags.append(
            f"achieved batch max {stats['achieved_batch_max']} is within 10% of "
            f"the CLIENT connection pool ({pool}) - offered load may "
            f"have been capped by the client, not the server")
    if check.get("missing_metrics"):
        # Loud by name. A renamed metric previously surfaced only as a vague
        # "the two sources disagree", which points at the wrong problem.
        flags.append("/metrics MISSING pinned name(s): "
                     + ", ".join(check["missing_metrics"]))
    if check.get("absent"):
        flags.append(f"/metrics or log has no data for {check['absent']}")
    if check["agree"] is False and check["diverged"]:
        flags.append(f"/metrics and log disagree on {check['diverged']}")
    client_tok = sum(r["completion_tokens"] for r in records)
    if stats["generation_tokens"] and abs(client_tok - stats["generation_tokens"]) \
            / max(stats["generation_tokens"], 1) > 0.05:
        flags.append(f"client tokens {client_tok} vs server "
                     f"{stats['generation_tokens']} differ >5%")
    return flags


def _raise_fd_limit(need):
    """Best-effort raise of the open-file soft limit.

    A large connection pool is useless if the process cannot open the sockets:
    the default soft limit is often 1024, and hitting it would reintroduce a
    silent client-side ceiling by another route.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = min(hard, max(soft, need * 4 + 256))
        if want > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except Exception as e:
        print(f"[WARN] could not raise the fd limit ({type(e).__name__}: {e})")


async def _amain(args):
    if args.mode == "closed" and args.in_flight is None:
        sys.exit("[FATAL] closed-loop cell with no --in-flight: the in-flight cap "
                 "IS the independent variable for these cells, and running without "
                 "one would silently measure something else")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")

    # --prompts defaults to a truthy path, so this used to be read even in
    # replay mode -- where the result is discarded two lines later. A replay
    # cell on a CLEAN box therefore died with FileNotFoundError before it could
    # start, and only worked at all if a Bucket-S cell had run there first and
    # left /data/prompts.jsonl behind. Provably inert to reorder: in replay mode
    # the value was overwritten unconditionally.
    replay = read_jsonl(args.replay) if args.replay else None
    if replay is not None:
        prompts = [{"prompt_id": r["seq"], "text": r["prompt_text"]} for r in replay]
    else:
        prompts = read_jsonl(args.prompts) if args.prompts else None
    if not prompts:
        sys.exit("[FATAL] no prompts: pass --prompts or --replay")

    cfg = {"osl_kind": args.osl, "thinking": args.thinking, "family": args.family,
           "forced_osl": args.forced_osl}
    duration = args.duration

    # httpx.AsyncClient defaults to max_connections=100. With no limits= set,
    # an "unbounded" cell was capped at 100 CLIENT-SIDE: `running` plateaued at
    # exactly 100.0 while `num_requests_waiting` stayed 0, because the excess
    # queued in our own connection pool and never reached vLLM. Server-side
    # queueing and admission -- the entire phenomenon the open-loop sheets
    # exist to measure -- was therefore never observed, and TTFT absorbed the
    # pool wait (13.4 s mean) while sched_delay_s stayed at 0.18 s.
    #
    # Bounded cells get a pool comfortably above their own in-flight cap, so the
    # pool can never be the binding constraint; unbounded cells get
    # pins.CLIENT_MAX_CONNECTIONS. The limit in force is RECORDED, and
    # build_flags raises a flag if achieved concurrency approaches it -- a
    # ceiling that is reached must be visible, never silent again.
    pool = (pins.CLIENT_MAX_CONNECTIONS if args.in_flight is None
            else max(2 * args.in_flight, 64))
    _raise_fd_limit(pool)
    limits = httpx.Limits(max_connections=pool,
                          max_keepalive_connections=pool)
    async with httpx.AsyncClient(base_url=base, limits=limits) as client:
        try:
            probe = await client.get(base + pins.METRICS_PATH, timeout=10.0)
            probe.raise_for_status()
        except Exception as e:
            sys.exit(f"[FATAL] /metrics unavailable at start: {e}")

        # The scrape runs in a SEPARATE PROCESS, not as a task on this event
        # loop. As a coroutine it shared the loop with the request drivers, and
        # asyncio never preempts a running coroutine -- at ~99 concurrent
        # streaming requests the loop was saturated by chunk callbacks and the
        # scraper went 118 s without being scheduled, putting achieved_batch_mean
        # 6.3x low on a law-fit covariate. dcgm_capture.py sampled the same box
        # over the same window and lost nothing, because it is a separate
        # process. See metrics_scraper.py.
        metrics_path = out / "vllm_metrics.jsonl"
        scraper_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve().parent / "metrics_scraper.py"),
             "--base-url", base, "--out", str(metrics_path),
             "--interval", str(pins.SCRAPE_INTERVAL_S)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # --- warm-up, discarded ---
        print(f"[..] warm-up {pins.WARMUP_S}s")
        warm_ts = time.time()
        if args.mode == "closed":
            await run_closed(client, base, args.model, prompts, cfg,
                             args.in_flight, pins.WARMUP_S, allow_wrap=args.allow_wrap)
        else:
            sched = ([o for o in replay_schedule(replay, args.scale)
                      if o <= pins.WARMUP_S]
                     if args.mode == "replay"
                     else poisson_schedule(args.rate, pins.WARMUP_S, args.seed))
            await run_open(client, base, args.model, prompts, cfg, sched, pins.WARMUP_S)
        warmup_end_ts = time.time()

        # --- measured window ---
        print(f"[..] measuring {duration}s")
        t0 = time.time()
        if args.mode == "closed":
            records = await run_closed(client, base, args.model, prompts, cfg,
                                       args.in_flight, duration,
                                       allow_wrap=args.allow_wrap, wrap_out=_WRAP)
        else:
            sched = (replay_schedule(replay, args.scale) if args.mode == "replay"
                     else poisson_schedule(args.rate, duration, args.seed))
            records = await run_open(client, base, args.model, prompts, cfg,
                                     sched, duration)
        t1 = time.time()

        # Take at least one scrape AFTER t1 so the counter deltas have a
        # bracketing sample on the high side. Without it the last sample lands
        # before t1 and every counter silently under-counts the final interval.
        await asyncio.sleep(pins.SCRAPE_INTERVAL_S * 2)
        scraper_proc.terminate()
        try:
            scraper_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            scraper_proc.kill()
            scraper_proc.wait(timeout=10)
        samples = [json.loads(l) for l in
                   metrics_path.read_text().splitlines() if l.strip()]
        if not samples:
            err = (scraper_proc.stderr.read() or b"").decode()[:300]
            sys.exit(f"[FATAL] metrics scraper produced no samples: {err}")
        # A starved scrape is now visible rather than reconstructible: late_s is
        # how far each sample slipped past its slot.
        late = [s_.get("late_s", 0.0) for s_ in samples]
        worst = max(late) if late else 0.0
        if worst > pins.SCRAPE_INTERVAL_S:
            print(f"[WARN] scrape slipped its slot by up to {worst:.1f}s "
                  f"({sum(1 for l in late if l > pins.SCRAPE_INTERVAL_S)} of "
                  f"{len(samples)} samples)")

    wrap_info = _WRAP
    attribution = mark_window(records, t0, t1)
    stats = summarise(samples, t0, t1)

    log_rows = []
    stale_log = None
    startup = {"engine_args_raw": None, "gpu_blocks": None, "kv_cache_tokens": None}
    if args.server_log and Path(args.server_log).exists():
        # A leftover log from a PREVIOUS cell on the same instance parses fine
        # and yields wrong-but-plausible provenance -- on 2026-09-02 a stale log
        # reported kv_cache_tokens=799296 for a server whose real value was
        # 275600. Refuse to trust a log older than the run.
        log_mtime = Path(args.server_log).stat().st_mtime
        if log_mtime < t0:
            stale_log = (f"server log {args.server_log} is STALE (mtime "
                         f"{log_mtime:.0f} predates window start {t0:.0f}) - "
                         f"engine args and kv_cache_tokens NOT taken from it")
            log_text = ""
        else:
            log_text = Path(args.server_log).read_text(errors="ignore")
        startup = parse_startup(log_text)
        for line in log_text.splitlines():
            row = parse_log_line(line)
            if row:
                log_rows.append(row)
    check = cross_check([s for s in samples if t0 <= s.get("ts", 0) <= t1], log_rows)

    with open(out / f"requests_{args.cell_id}.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    args.client_pool = pool
    flags = build_flags(args, stats, check, records)
    if stale_log:
        flags.append(stale_log)

    write_meta(out / "run_meta.json", {
        "cell_id": args.cell_id, "release_tag": "v1",
        "git_commit": _git_commit(),
        "vllm_version": pins.VLLM_VERSION, "vllm_image": pins.VLLM_IMAGE,
        "mode": args.mode, "model": args.model, "family": args.family,
        "osl_kind": args.osl, "thinking": args.thinking,
        "reasoning_parser": pins.REASONING_PARSER if args.thinking else None,
        "in_flight": args.in_flight, "rate": args.rate, "scale": args.scale,
        "forced_osl": args.forced_osl,
        # The client-side pool that was actually in force. Recorded because an
        # unrecorded ceiling is exactly how "unbounded" quietly meant 100.
        "client_max_connections": pool,
        "seed": args.seed,
        "sampling": pins.SAMPLING, # The ISL_OSL grid overrides the bucket default per cell, so recording
        # pins.OSL[osl_kind] wrote 512 on every grid row regardless of the
        # OSL that actually applied. forced_osl was right and realized
        # completion_tokens confirmed it, but max_tokens contradicted both.
        "max_tokens": args.forced_osl or pins.OSL[args.osl],
        "max_model_len": pins.MAX_MODEL_LEN[args.osl],
        "engine_args_raw": startup["engine_args_raw"],
        **({"prompt_pool_wrap": wrap_info} if wrap_info else {}),
        "kv_cache_gpu_blocks": startup["gpu_blocks"],
        "kv_cache_tokens": startup["kv_cache_tokens"],
        "client_server_colocated": args.base_url.startswith(
            ("http://127.0.0.1", "http://localhost")),
        "warmup_start_ts_ns": int(warm_ts * 1e9),
        "warmup_end_ts_ns": int(warmup_end_ts * 1e9),
        "window_start_ts_ns": int(t0 * 1e9), "window_end_ts_ns": int(t1 * 1e9),
        "dataset_repo": pins.DATASETS_REPO,
        "dataset_revision": pins.DATASETS_REVISION,
        "distinct_prompts_available": len(prompts),
        "requests_total": len(records),
        "requests_errored": sum(1 for r in records if r["error"]),
        "finish_reasons": {str(k): sum(1 for r in records if r["finish_reason"] == k)
                           for k in {r["finish_reason"] for r in records}},
        "attribution": attribution, "server": stats, "cross_check": check,
        "flags": flags,
    })

    print(f"[OK] {len(records):,} requests, {attribution['tokens_in_window']:,} "
          f"tokens in window, achieved batch {stats['achieved_batch_mean']}")
    for f in flags:
        print(f"[FLAG] {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["closed", "replay", "poisson"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=sorted(pins.POOLS))
    ap.add_argument("--osl", required=True, choices=sorted(pins.OSL))
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--prompts", default="/data/prompts.jsonl")
    ap.add_argument("--replay")
    # DEFAULT None, NOT 32. cell.py emits --in-flight only when the manifest's
    # in_flight is non-null, so an UNBOUNDED cell never passes it -- and with a
    # default of 32 the "unbounded" branch below was unreachable, handing those
    # cells a pool of max(2*32, 64) = 64. That is TIGHTER than the httpx default
    # of 100 the pool fix was written to remove: burstgpt then clipped flat at
    # exactly 64 concurrent with num_requests_waiting still identically zero.
    # Every other use of args.in_flight is guarded by mode == "closed", and a
    # closed cell always receives the flag explicitly, so None reaches only the
    # code that wants it.
    ap.add_argument("--in-flight", type=int, default=None)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="BurstGPT compression factor c: arrival gaps are "
                         "divided by it. The whole replay file is always used, "
                         "so wall clock is window_s / c.")
    ap.add_argument("--duration", type=int, default=pins.DURATION_S["knob"])
    ap.add_argument("--seed", type=int, default=pins.SHUFFLE_SEED)
    # The ISL_OSL grid sweeps OSL per cell; every other sheet uses the bucket
    # default from pins.OSL.
    ap.add_argument("--forced-osl", type=int, default=None)
    ap.add_argument("--allow-wrap", action="store_true",
                    help="closed-loop only: recycle the prompt pool instead of "
                         "erroring when it is exhausted. OFF by default and "
                         "opt-in per cell; a wrapped pool inflates prefix-cache "
                         "hits, so the run record carries the wrap factor.")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--server-log", default="/data/vllm_server.log")
    ap.add_argument("--out", default="/data/run")
    args = ap.parse_args()
    if args.mode == "replay" and not args.replay:
        sys.exit("[FATAL] --mode replay needs --replay")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
