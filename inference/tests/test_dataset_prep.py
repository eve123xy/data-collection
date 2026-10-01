import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("", "dataset", "workload", "telemetry", "cells", "upload"):
    sys.path.insert(0, str(ROOT / "scripts" / _sub))

from build_smallprompts import shuffle_prompts
from build_burstgpt_windows import (
    PromptMatcher, build_replay, derive_window, window_stats,
)

POOL = [{"prompt": "short", "prompt_len": 10},
        {"prompt": "mid", "prompt_len": 100},
        {"prompt": "long", "prompt_len": 1000}]


# --- shuffle ---

def test_shuffle_is_deterministic_and_seed_sensitive():
    texts = [f"p{i}" for i in range(50)]
    assert shuffle_prompts(texts, 7) == shuffle_prompts(texts, 7)
    assert shuffle_prompts(texts, 7) != shuffle_prompts(texts, 8)


def test_prompt_id_is_the_original_index_not_the_shuffled_position():
    texts = ["zero", "one", "two", "three", "four", "five"]
    for r in shuffle_prompts(texts, 7):
        assert texts[r["prompt_id"]] == r["text"]


def test_shuffle_keeps_every_prompt_exactly_once():
    texts = [f"p{i}" for i in range(50)]
    rows = shuffle_prompts(texts, 7)
    assert sorted(r["prompt_id"] for r in rows) == list(range(50))


# --- window derivation ---

def test_request_count_finds_the_densest_window():
    ts = np.concatenate([np.arange(0, 500, 50.0), np.arange(500, 600, 1.0)])
    stats = window_stats(ts, np.ones_like(ts), 100)
    offset, rate = derive_window(stats, "request_count")
    assert 400 < offset <= 500
    assert rate == pytest.approx(1.0, rel=0.2)


def test_decode_work_ignores_request_count():
    # dense-but-trivial burst at t=0, sparse-but-heavy at t=1000
    ts = np.concatenate([np.arange(0, 100, 1.0), np.arange(1000, 1100, 25.0)])
    tok = np.concatenate([np.ones(100), np.full(4, 10_000.0)])
    stats = window_stats(ts, tok, 100)
    assert derive_window(stats, "request_count")[0] < 500
    assert derive_window(stats, "decode_work")[0] >= 900


def test_burstiness_is_higher_for_a_spike_at_equal_request_count():
    """Compare two windows with IDENTICAL totals: one flat, one spiked.

    Not tested via argmax: burstiness = peak-minute/mean-minute is capped at
    W/60 and a window straddling a traffic gap (all its requests in one
    minute) hits that ceiling, so the global argmax lands on near-empty
    windows rather than bursty ones. That degeneracy is why `burst` is taken
    as a recorded offset rather than re-derived.
    """
    flat = np.arange(0, 1200, 1.0)                        # 1200 reqs, 1/s
    spread = np.arange(5000, 6200, 2.0)                   # 600 reqs, 0.5/s
    spike = np.full(600, 5600.5)                          # 600 reqs in one second
    tail = np.array([6200.0])   # extends the span; excluded from [5000, 6200)
    ts = np.sort(np.concatenate([flat, spread, spike, tail]))
    stats = window_stats(ts, np.ones_like(ts), 1200)

    assert stats["request_count"][0] == stats["request_count"][5000] == 1200
    assert stats["burstiness"][0] == pytest.approx(1.0)
    assert stats["burstiness"][5000] > 8.0


def test_min_rate_floor_rejects_the_degenerate_burstiness_argmax():
    """Without a floor, a near-empty window at a traffic gap wins on
    burstiness by putting all its traffic in one minute. With a floor, the
    genuinely bursty window wins. This is why `burst` carries a min-rate."""
    sparse = np.arange(0, 60, 1.0)                        # 60 reqs then silence
    dense = np.arange(5000, 6200, 0.5)                    # 2400 reqs, 2/s
    spike = np.full(1200, 5600.5)                         # 1200 reqs in one second
    tail = np.array([6200.0])
    ts = np.sort(np.concatenate([sparse, dense, spike, tail]))
    stats = window_stats(ts, np.ones_like(ts), 1200)

    loose, _ = derive_window(stats, "burstiness")
    assert loose < 1000                                    # the sparse decoy wins

    strict, rate = derive_window(stats, "burstiness", min_rate=1.0)
    assert strict <= 5600 < strict + 1200                  # the real spike wins
    assert rate >= 1.0


def test_burstiness_ceiling_is_window_minutes():
    """All traffic inside a single minute scores the theoretical maximum."""
    ts = np.concatenate([np.arange(0, 60, 0.1),           # 600 reqs in 60 s
                         [1200.0]])                       # span extender, outside [0, 1200)
    stats = window_stats(ts, np.ones_like(ts), 1200)
    assert stats["burstiness"][0] == pytest.approx(1200 / 60)


def test_window_stats_arrays_are_all_the_same_length():
    ts = np.arange(0, 5000, 3.0)
    stats = window_stats(ts, np.ones_like(ts), 1200)
    n = len(stats["request_count"])
    assert len(stats["decode_work"]) == n
    assert len(stats["burstiness"]) == n


# --- matching and clamping ---

def test_match_picks_nearest_prompt_len():
    m = PromptMatcher(POOL, seed=1)
    assert m.match(12)["prompt"] == "short"
    assert m.match(90)["prompt"] == "mid"
    assert m.match(5000)["prompt"] == "long"


def test_match_ignores_output_len():
    pool = [{"prompt": "a", "prompt_len": 50, "output_len": 1},
            {"prompt": "b", "prompt_len": 50, "output_len": 9999}]
    m = PromptMatcher(pool, seed=3)
    assert {m.match(50)["prompt"] for _ in range(50)} == {"a", "b"}


def test_match_draws_without_replacement_within_a_length():
    """Distinct requests at the same length get distinct prompts until the
    bucket is exhausted, so trace replay does not manufacture cache hits."""
    pool = [{"prompt": f"p{i}", "prompt_len": 50} for i in range(5)]
    m = PromptMatcher(pool, seed=3)
    assert len({m.match(50)["prompt"] for _ in range(5)}) == 5   # all distinct
    m2 = PromptMatcher(pool, seed=3)
    got = [m2.match(50)["prompt"] for _ in range(7)]
    assert len(set(got)) == 5                                    # then wraps


def test_zero_token_rows_are_clamped_not_dropped():
    rows = [{"timestamp": 0.0, "prompt_len": 100},
            {"timestamp": 1.0, "prompt_len": 0},
            {"timestamp": 3.5, "prompt_len": 1000}]
    replay, clamped = build_replay(rows, PromptMatcher(POOL, seed=1))
    assert len(replay) == 3          # arrival count preserved
    assert clamped == 1
    assert replay[1]["matched_prompt_len"] == 10


def test_replay_emits_inter_arrival_deltas():
    rows = [{"timestamp": 10.0, "prompt_len": 100},
            {"timestamp": 10.5, "prompt_len": 100},
            {"timestamp": 13.0, "prompt_len": 100}]
    replay, _ = build_replay(rows, PromptMatcher(POOL, seed=1))
    assert [r["arrival_delta_s"] for r in replay] == [0.0, 0.5, 2.5]
    assert [r["seq"] for r in replay] == [0, 1, 2]
