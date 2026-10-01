import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("", "dataset", "workload", "telemetry", "cells", "upload"):
    sys.path.insert(0, str(ROOT / "scripts" / _sub))

import pins
from workload import (build_payload, fire_one, mark_window, poisson_schedule,
                      replay_schedule, run_closed, run_open)


# --- pinned constants ---

def test_sampling_matches_the_settings_contract():
    assert pins.SAMPLING["temperature"] == 0.7
    assert pins.SAMPLING["top_p"] == 0.9
    assert "top_k" not in pins.SAMPLING          # top_k off means absent, not 0


def test_endpoint_is_chat_never_raw_completions():
    assert pins.CHAT_ENDPOINT == "/v1/chat/completions"


def test_thinking_caps_leave_room_for_the_cap_to_bind():
    # max_model_len must exceed the longest ISL plus the full output cap,
    # or generation stops at the context boundary instead of the cap.
    assert pins.MAX_MODEL_LEN["thinking"] >= 2048 + pins.OSL["thinking"]


def test_window_durations_match_the_plan():
    assert pins.WARMUP_S == 60
    assert pins.DURATION_S["knob"] == 600
    assert pins.DURATION_S["open"] == 1200


def test_vllm_image_is_pinned_not_latest():
    assert pins.VLLM_IMAGE.endswith(f"v{pins.VLLM_VERSION}")
    assert "latest" not in pins.VLLM_IMAGE


# --- payload ---

def test_payload_sets_thinking_explicitly_for_qwen3_in_both_directions():
    on = build_payload("Qwen/Qwen3-8B", "hi", "free", thinking=True, family="qwen3")
    off = build_payload("Qwen/Qwen3-8B", "hi", "free", thinking=False, family="qwen3")
    assert on["chat_template_kwargs"] == {"enable_thinking": True}
    assert off["chat_template_kwargs"] == {"enable_thinking": False}


def test_payload_omits_thinking_kwarg_for_llama():
    p = build_payload("meta-llama/Llama-3.1-70B", "hi", "free",
                      thinking=False, family="llama31")
    assert "chat_template_kwargs" not in p


def test_payload_forces_osl_with_ignore_eos():
    p = build_payload("Qwen/Qwen3-8B", "hi", "forced", thinking=False, family="qwen3")
    assert p["ignore_eos"] is True and p["max_tokens"] == 512
    f = build_payload("Qwen/Qwen3-8B", "hi", "free", thinking=False, family="qwen3")
    assert f["ignore_eos"] is False and f["max_tokens"] == 2048


def test_payload_requests_usage_and_uses_messages_not_prompt():
    p = build_payload("Qwen/Qwen3-8B", "hi", "free", thinking=False, family="qwen3")
    assert p["stream"] is True
    assert p["stream_options"] == {"include_usage": True}
    assert p["messages"] == [{"role": "user", "content": "hi"}]
    assert "prompt" not in p


# --- request path ---

def _sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)


async def test_fire_one_builds_a_record_with_itl_and_usage():
    async def handler(request):
        payload = (_sse([{"choices": [{"delta": {"content": "a"}}]}])
                   + _sse([{"choices": [{"delta": {"reasoning_content": "b"}}]}])
                   + _sse([{"choices": [{"delta": {"content": "c"},
                                         "finish_reason": "stop"}]}])
                   + _sse([{"choices": [], "usage": {"prompt_tokens": 7,
                                                     "completion_tokens": 3}}])
                   + "data: [DONE]\n\n")
        return httpx.Response(200, text=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as c:
        rec = await fire_one(c, "http://x", "Qwen/Qwen3-8B",
                             {"prompt_id": 5, "text": "hi"},
                             {"osl_kind": "free", "thinking": False, "family": "qwen3"},
                             scheduled_ts=1.0)
    assert rec["prompt_id"] == 5
    assert rec["prompt_tokens"] == 7 and rec["completion_tokens"] == 3
    assert len(rec["itl_ms"]) == 2          # 3 token chunks -> 2 gaps
    assert rec["ttft_s"] >= 0
    assert rec["http_status"] == 200
    assert rec["error"] is None
    assert rec["thinking"] is False


async def test_fire_one_records_an_error_without_raising():
    async def handler(request):
        return httpx.Response(500, text="boom")
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as c:
        rec = await fire_one(c, "http://x", "m", {"prompt_id": 1, "text": "hi"},
                             {"osl_kind": "free", "thinking": False, "family": "qwen3"},
                             scheduled_ts=0.0)
    assert rec["http_status"] == 500
    assert rec["error"] is not None
    assert rec["completion_tokens"] == 0


# --- window attribution ---

def _rec(send, first, end, itl_ms, out_tokens):
    return {"send_ts": send, "first_token_ts": first, "end_ts": end,
            "itl_ms": itl_ms, "completion_tokens": out_tokens, "in_window": None}


def test_request_wholly_inside_counts_for_both():
    r = _rec(10.0, 10.1, 10.5, [100.0, 100.0], 3)
    out = mark_window([r], t0=0.0, t1=100.0)
    assert r["in_window"] is True
    assert out["requests_in_window"] == 1
    assert out["tokens_in_window"] == 3


def test_request_straddling_the_end_counts_tokens_but_not_slos():
    # first token at t=9.9, gaps of 100ms -> chunks at 9.9, 10.0, 10.1
    r = _rec(9.8, 9.9, 10.2, [100.0, 100.0], 3)
    out = mark_window([r], t0=0.0, t1=10.0)
    assert r["in_window"] is False              # ends after t1
    assert out["requests_in_window"] == 0
    assert out["requests_straddling"] == 1
    assert out["tokens_in_window"] == 2         # chunks at 9.9 and 10.0 only


def test_request_entirely_before_t0_is_excluded():
    r = _rec(1.0, 1.1, 1.5, [100.0], 2)
    out = mark_window([r], t0=5.0, t1=10.0)
    assert r["in_window"] is False
    assert out["tokens_in_window"] == 0


def test_errored_request_contributes_no_tokens():
    r = _rec(10.0, None, 10.5, [], 0)
    out = mark_window([r], t0=0.0, t1=100.0)
    assert out["tokens_in_window"] == 0


# --- schedules and dispatchers ---

def test_poisson_schedule_is_seeded_and_hits_the_target_rate():
    a = poisson_schedule(rate=5.0, horizon_s=200.0, seed=1)
    b = poisson_schedule(rate=5.0, horizon_s=200.0, seed=1)
    assert a == b
    assert a != poisson_schedule(rate=5.0, horizon_s=200.0, seed=2)
    assert a == sorted(a) and a[-1] <= 200.0
    assert 0.8 < (len(a) / 200.0) / 5.0 < 1.2      # within 20% of target


def test_replay_schedule_accumulates_arrival_deltas():
    rows = [{"arrival_delta_s": 0.0}, {"arrival_delta_s": 0.5},
            {"arrival_delta_s": 2.5}]
    assert replay_schedule(rows) == [0.0, 0.5, 3.0]


def test_compression_factor_scales_the_whole_schedule_uniformly():
    """c divides every gap, so the arrival SHAPE is preserved exactly and only
    the time axis changes - that is what makes a rate ladder comparable."""
    rows = [{"arrival_delta_s": 0.0}, {"arrival_delta_s": 0.5},
            {"arrival_delta_s": 2.5}]
    assert replay_schedule(rows, scale=2.0) == [0.0, 0.25, 1.5]     # 2x faster
    assert replay_schedule(rows, scale=0.5) == [0.0, 1.0, 6.0]      # 2x slower
    base = replay_schedule(rows)
    for c in (0.25, 0.5, 2.0):
        scaled = replay_schedule(rows, scale=c)
        assert scaled[-1] == pytest.approx(base[-1] / c)


def test_replay_schedule_rejects_a_non_positive_scale():
    with pytest.raises(ValueError, match="positive"):
        replay_schedule([{"arrival_delta_s": 1.0}], scale=0)


def _client(delay=0.0):
    async def handler(request):
        if delay:
            await asyncio.sleep(delay)
        body = (_sse([{"choices": [{"delta": {"content": "a"}}]}])
                + _sse([{"choices": [], "usage": {"prompt_tokens": 5,
                                                  "completion_tokens": 1}}])
                + "data: [DONE]\n\n")
        return httpx.Response(200, text=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                             base_url="http://x")


CFG = {"osl_kind": "free", "thinking": False, "family": "qwen3"}
PROMPTS = [{"prompt_id": i, "text": f"p{i}"} for i in range(500)]


async def test_closed_loop_holds_the_pool_and_stops_at_the_deadline():
    async with _client(delay=0.02) as c:
        recs = await run_closed(c, "http://x", "m", PROMPTS, CFG,
                                in_flight=4, total_s=0.4)
    assert len(recs) > 4
    assert len({r["prompt_id"] for r in recs}) == len(recs)   # never repeats


async def test_open_loop_fires_on_schedule_even_when_the_server_is_slow():
    """The point of asyncio: a backend slower than the arrival gap must not
    delay dispatch, or we stop measuring offered load."""
    async with _client(delay=0.30) as c:          # far slower than the 0.02s gap
        schedule = [i * 0.02 for i in range(20)]
        recs = await run_open(c, "http://x", "m", PROMPTS, CFG,
                              schedule=schedule, total_s=1.5)
    assert len(recs) == 20
    assert max(r["sched_delay_s"] for r in recs) < 0.10


async def test_prompt_exhaustion_raises():
    async with _client() as c:
        with pytest.raises(RuntimeError, match="exhausted"):
            await run_open(c, "http://x", "m", PROMPTS[:3], CFG,
                           schedule=[0.0, 0.01, 0.02, 0.03], total_s=1.0)


# --- window summary ---

from run_workload import summarise, build_flags


def test_summarise_reports_achieved_batch_and_derived_counters():
    samples = [
        {"ts": 1.0, "vllm:num_requests_running": 10, "vllm:num_requests_waiting": 0,
         "vllm:gpu_cache_usage_perc": 0.10, "vllm:num_preemptions_total": 0,
         "vllm:prefix_cache_queries_total": 100, "vllm:prefix_cache_hits_total": 1,
         "vllm:generation_tokens_total": 1000},
        {"ts": 2.0, "vllm:num_requests_running": 30, "vllm:num_requests_waiting": 5,
         "vllm:gpu_cache_usage_perc": 0.50, "vllm:num_preemptions_total": 2,
         "vllm:prefix_cache_queries_total": 300, "vllm:prefix_cache_hits_total": 4,
         "vllm:generation_tokens_total": 5000},
    ]
    s = summarise(samples, t0=0.5, t1=2.5)
    assert s["achieved_batch_mean"] == 20.0
    assert s["achieved_batch_max"] == 30.0
    assert s["waiting_p95"] >= 0
    assert s["preemptions"] == 2                       # counter delta
    assert s["generation_tokens"] == 4000              # counter delta
    assert s["prefix_hit_pct"] == pytest.approx(1.5)   # (4-1)/(300-100)


def test_counter_deltas_bracket_the_window_rather_than_sampling_inside_it():
    """A counter delta between two INTERIOR samples omits everything generated
    in the first and last scrape interval. Bracketing captures the whole window."""
    samples = [
        {"ts": 0.0, "vllm:generation_tokens_total": 0},      # before t0
        {"ts": 1.5, "vllm:generation_tokens_total": 500},
        {"ts": 2.5, "vllm:generation_tokens_total": 900},
        {"ts": 4.0, "vllm:generation_tokens_total": 1400},   # after t1
    ]
    s = summarise(samples, t0=1.0, t1=3.0)
    assert s["generation_tokens"] == 1400        # bracketed 0.0 -> 4.0
    # interior-only would have given 900-500 = 400, losing 71%


def test_counter_bracket_falls_back_safely_when_no_sample_follows_t1():
    """If the scraper stopped before t1 there is no high bracket; fall back to
    the last interior sample rather than crashing, and record the span used."""
    samples = [{"ts": 1.0, "vllm:generation_tokens_total": 100},
               {"ts": 2.0, "vllm:generation_tokens_total": 300}]
    s = summarise(samples, t0=0.5, t1=9.0)
    assert s["generation_tokens"] == 200
    assert s["counter_bracket_s"] == 1.0        # 1.0 -> 2.0, not the full window


def test_summarise_ignores_samples_outside_the_window():
    samples = [{"ts": 0.0, "vllm:num_requests_running": 999},
               {"ts": 5.0, "vllm:num_requests_running": 10}]
    assert summarise(samples, t0=4.0, t1=6.0)["achieved_batch_max"] == 10.0


class _Args:
    mode = "closed"
    in_flight = 32


def test_prefix_cache_hit_rate_is_recorded_but_never_flags():
    """Retired 2026-09-03. The 2.0% guard assumed distinct prompts imply a
    near-zero hit rate, but every request carries the same chat-template
    preamble, so the real rate is 15-18% on EVERY cell -- one cell drew 4,676
    distinct prompt_ids and still hit 15.5%. Left in place it would have marked
    all 248 cells "Flagged", and a status column where every row says the same
    thing carries no information. The measurement is still recorded; only the
    decision to shout about it was removed."""
    stats = {"prefix_hit_pct": 18.0, "preemptions": 0, "achieved_batch_max": 30,
             "achieved_batch_mean": 30.0, "generation_tokens": 100, "samples": 5}
    flags = build_flags(_Args(), stats, {"agree": True, "diverged": []},
                        [{"completion_tokens": 100, "error": None}])
    assert not any("prefix_hit_pct" in f for f in flags)


def test_flags_fire_when_achieved_batch_never_reaches_the_cap():
    stats = {"prefix_hit_pct": 0.0, "preemptions": 0, "achieved_batch_max": 12,
             "achieved_batch_mean": 11.0, "generation_tokens": 100, "samples": 5}
    flags = build_flags(_Args(), stats, {"agree": True, "diverged": []},
                        [{"completion_tokens": 100, "error": None}])
    assert any("below 0.9x cap" in f for f in flags)


def test_flags_fire_on_preemption_at_batch_32():
    stats = {"prefix_hit_pct": 0.0, "preemptions": 7, "achieved_batch_max": 32,
             "achieved_batch_mean": 31.0, "generation_tokens": 100, "samples": 5}
    flags = build_flags(_Args(), stats, {"agree": True, "diverged": []},
                        [{"completion_tokens": 100, "error": None}])
    assert any("preemptions at batch 32" in f for f in flags)


def test_flags_fire_when_client_and_server_token_counts_disagree():
    stats = {"prefix_hit_pct": 0.0, "preemptions": 0, "achieved_batch_max": 32,
             "achieved_batch_mean": 31.0, "generation_tokens": 1000, "samples": 5}
    flags = build_flags(_Args(), stats, {"agree": True, "diverged": []},
                        [{"completion_tokens": 500, "error": None}])
    assert any("differ >5%" in f for f in flags)


def test_a_clean_run_produces_no_flags():
    stats = {"prefix_hit_pct": 0.1, "preemptions": 0, "achieved_batch_max": 32,
             "achieved_batch_mean": 31.5, "generation_tokens": 1000, "samples": 5}
    flags = build_flags(_Args(), stats, {"agree": True, "diverged": []},
                        [{"completion_tokens": 1000, "error": None}])
    assert flags == []
