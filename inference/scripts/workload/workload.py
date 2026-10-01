"""One streaming chat request -> one per-request record, plus the dispatchers.

Streaming is mandatory: TTFT and inter-token latency are only observable from
chunk arrival times. Token counts come from vLLM's own usage block, never from
a client-side tokenizer that could disagree with the server.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import asyncio
import hashlib
import json
import random
import time

import pins


def build_payload(model, prompt, osl_kind, thinking, family, forced_osl=None):
    """Chat-completions payload for one request.

    `chat_template_kwargs` is sent for Qwen3 in BOTH directions: the template
    defaults enable_thinking to true, so omitting it gives the opposite of the
    settings contract's "OFF by default".
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        # forced_osl, when the manifest carries one, OVERRIDES the bucket default.
        # The ISL_OSL grid sweeps OSL across 128/512/2048 per cell and records it
        # as workload.forced_osl -- but nothing read it, so every grid cell would
        # have run at pins.OSL["forced"] = 512. Twelve of the fifteen would have
        # carried the wrong output length, and a grid that does not vary its grid
        # yields five copies of one point. run_plan_settings_v2 §1: "max_tokens =
        # the forced OSL (512 unless the ISL/OSL grid says otherwise)".
        "max_tokens": forced_osl if forced_osl else pins.OSL[osl_kind],
        "ignore_eos": osl_kind == "forced",
        **pins.SAMPLING,
    }
    if family == "qwen3":
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    return payload


def _delta_text(choice_delta):
    """vLLM's reasoning delta field name is version-dependent; take whichever."""
    return ((choice_delta.get("content") or "")
            + (choice_delta.get("reasoning_content") or "")
            + (choice_delta.get("reasoning") or ""))


async def fire_one(client, base_url, model, prompt_row, cfg, scheduled_ts):
    """Send one request, stream it, and return the record. Never raises."""
    payload = build_payload(model, prompt_row["text"], cfg["osl_kind"],
                            cfg["thinking"], cfg["family"],
                            cfg.get("forced_osl"))

    send_ts = time.time()
    send_perf = time.perf_counter()
    token_perf = []
    finish_reason = None
    prompt_tokens = completion_tokens = 0
    status = None
    error = None

    try:
        async with client.stream("POST", base_url + pins.CHAT_ENDPOINT,
                                 json=payload, timeout=None) as resp:
            status = resp.status_code
            if status != 200:
                body = await resp.aread()
                error = f"HTTP {status}: {body[:200]!r}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    for ch in obj.get("choices") or []:
                        if _delta_text(ch.get("delta") or {}):
                            token_perf.append(time.perf_counter())
                        if ch.get("finish_reason"):
                            finish_reason = ch["finish_reason"]
                    if obj.get("usage"):
                        prompt_tokens = obj["usage"].get("prompt_tokens", 0)
                        completion_tokens = obj["usage"].get("completion_tokens", 0)
    except Exception as e:                     # network, timeout, malformed stream
        error = f"{type(e).__name__}: {e}"

    end_perf = time.perf_counter()
    end_ts = time.time()
    ttft = (token_perf[0] - send_perf) if token_perf else None
    itl_ms = [round(1000 * (b - a), 4) for a, b in zip(token_perf, token_perf[1:])]
    e2e = end_perf - send_perf

    return {
        "request_id": f"{prompt_row['prompt_id']}-{send_ts:.6f}",
        "prompt_id": prompt_row["prompt_id"],
        "prompt_hash": hashlib.sha256(prompt_row["text"].encode()).hexdigest()[:16],
        "scheduled_ts": scheduled_ts,
        "send_ts": send_ts,
        "sched_delay_s": round(send_ts - scheduled_ts, 6) if scheduled_ts else 0.0,
        "first_token_ts": (send_ts + ttft) if ttft is not None else None,
        "end_ts": end_ts,
        "ttft_s": round(ttft, 6) if ttft is not None else None,
        "e2e_s": round(e2e, 6),
        "tpot_s": round((e2e - ttft) / max(completion_tokens - 1, 1), 6)
                  if ttft is not None and completion_tokens > 1 else None,
        "itl_ms": itl_ms,
        "n_chunks": len(token_perf),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "http_status": status,
        "retries": 0,
        "error": error,
        "sampling_echo": {k: payload[k] for k in
                          ("temperature", "top_p", "max_tokens", "ignore_eos")},
        "thinking": payload.get("chat_template_kwargs", {}).get("enable_thinking"),
        "in_window": None,          # filled by the window pass
    }


def mark_window(records, t0, t1):
    """Attribute records to the measurement window [t0, t1].

    Two different rules, deliberately:
      - TOKENS are counted by the wall-clock time of the chunk that carried
        them, so a request that straddles t1 contributes the part generated
        inside the window.
      - PER-REQUEST SLOs use only requests wholly inside, because a truncated
        request has no meaningful e2e latency.
    """
    tokens = 0
    inside = straddling = 0

    for r in records:
        whole = (r["send_ts"] >= t0 and r["end_ts"] <= t1)
        r["in_window"] = whole
        if whole:
            inside += 1
        elif r["end_ts"] > t1 >= r["send_ts"]:
            straddling += 1

        if r.get("first_token_ts") is None:
            continue

        # Reconstruct each chunk's wall-clock time from first_token_ts + cumulative ITL.
        t = r["first_token_ts"]
        if t0 <= t <= t1:
            tokens += 1
        for gap_ms in r["itl_ms"]:
            t += gap_ms / 1000.0
            if t > t1:
                break
            if t >= t0:
                tokens += 1

    return {"tokens_in_window": tokens,
            "requests_in_window": inside,
            "requests_straddling": straddling}


def poisson_schedule(rate, horizon_s, seed):
    """Arrival offsets from an exponential inter-arrival process."""
    rng = random.Random(seed)
    out, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t > horizon_s:
            return out
        out.append(round(t, 6))


def replay_schedule(replay_rows, scale=1.0):
    """Cumulative arrival offsets from a replay file's inter-arrival deltas.

    `scale` is the compression factor c: gaps are divided by it, so c=2 replays
    at twice the trace's own rate and c=0.5 at half. The whole file is always
    replayed - duration follows from c rather than truncating the trace, which
    is what keeps content identical across a rate ladder.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    out, t = [], 0.0
    for row in replay_rows:
        t += float(row["arrival_delta_s"]) / scale
        out.append(round(t, 6))
    return out


async def run_closed(client, base_url, model, prompts, cfg, in_flight, total_s,
                     allow_wrap=False, wrap_out=None):
    """Hold `in_flight` requests outstanding, replacing each on completion.

    `allow_wrap` defaults False, which is the frozen behaviour: exhausting the
    pool is a hard error, because with prefix caching ON a recycled prompt is
    served from cache and the cell measures the cache instead of the model.

    It is opt-in per cell for the one case where the corpus physically cannot
    cover the window -- ISL 2048 x OSL 128 on the fastest model needs ~8,510
    distinct 2048-token prompts and only 6,831 exist in total. Operator
    decision 2026-09-06: run it wrapped and report the dataset-capacity limit
    alongside, rather than leave the grid point empty. When wrapped, the run
    record carries the wrap factor and prefix_hit_pct so the contamination is
    visible rather than implied.
    """
    records = []
    idx = 0
    wrapped = 0
    stop = asyncio.get_running_loop().time() + total_s
    pending = set()

    def more():
        nonlocal idx
        if idx >= len(prompts):
            if allow_wrap:
                nonlocal wrapped
                wrapped += 1
                idx = 0
            else:
                raise RuntimeError("prompt pool exhausted - the dataset spec "
                                   "asserts no cell wraps, so this run is invalid")
        row = prompts[idx]
        idx += 1
        return asyncio.create_task(fire_one(client, base_url, model, row, cfg, None))

    while len(pending) < in_flight:
        pending.add(more())

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            records.append(t.result())
        if asyncio.get_running_loop().time() < stop:
            for _ in done:
                pending.add(more())

    # The wrap count is reported through `wrap_out`, NOT by appending a record.
    # Appending a metadata dict to `records` crashed mark_window, which reads
    # r["send_ts"] on every entry -- so --allow-wrap failed 100% of the time the
    # pool actually wrapped, which is the only case it exists for.
    if wrapped and wrap_out is not None:
        wrap_out["pool_wrapped"] = wrapped
        wrap_out["pool_size"] = len(prompts)
    return records


async def run_open(client, base_url, model, prompts, cfg, schedule, total_s):
    """Fire at fixed offsets regardless of how backed up the server is.

    Every request is its own task, so dispatch never waits on a response. If it
    did, we would be measuring served load and calling it offered load.
    """
    if len(schedule) > len(prompts):
        raise RuntimeError(f"prompt pool exhausted: schedule wants "
                           f"{len(schedule):,} requests, pool has {len(prompts):,}")

    loop = asyncio.get_running_loop()
    origin = loop.time()
    tasks = []

    async def at(offset, row):
        due = origin + offset
        delay = due - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        return await fire_one(client, base_url, model, row, cfg,
                              scheduled_ts=time.time() - (loop.time() - due))

    for offset, row in zip(schedule, prompts):
        if offset > total_s:
            break
        tasks.append(asyncio.create_task(at(offset, row)))

    return list(await asyncio.gather(*tasks))
