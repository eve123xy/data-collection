import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for _sub in ("", "dataset", "workload", "telemetry", "cells", "upload"):
    sys.path.insert(0, str(ROOT / "scripts" / _sub))

from vllm_metrics import (parse_prometheus, parse_log_line, cross_check,
                          parse_startup)

SAMPLE = """\
# HELP vllm:num_requests_running Number running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen3-8B"} 12.0
vllm:num_requests_waiting{model_name="Qwen/Qwen3-8B"} 3.0
vllm:kv_cache_usage_perc{model_name="Qwen/Qwen3-8B"} 0.42
vllm:num_preemptions_total{model_name="Qwen/Qwen3-8B"} 0.0
vllm:prefix_cache_queries_total{model_name="Qwen/Qwen3-8B"} 200.0
vllm:prefix_cache_hits_total{model_name="Qwen/Qwen3-8B"} 2.0
vllm:generation_tokens_total{model_name="Qwen/Qwen3-8B"} 51200.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="Q"} 5.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="Q"} 9.0
vllm:time_to_first_token_seconds_count{model_name="Q"} 9.0
"""


def test_parse_prometheus_reads_gauges_and_counters():
    m = parse_prometheus(SAMPLE)
    assert m["vllm:num_requests_running"] == 12.0
    assert m["vllm:num_requests_waiting"] == 3.0
    assert m["vllm:kv_cache_usage_perc"] == 0.42
    assert m["vllm:generation_tokens_total"] == 51200.0


def test_parse_prometheus_keeps_histogram_buckets_separate():
    m = parse_prometheus(SAMPLE)
    assert m['vllm:time_to_first_token_seconds_bucket{le="0.1"}'] == 5.0
    assert m['vllm:time_to_first_token_seconds_bucket{le="+Inf"}'] == 9.0
    assert m["vllm:time_to_first_token_seconds_count"] == 9.0


def test_parse_prometheus_ignores_comments():
    assert not any(k.startswith("#") for k in parse_prometheus(SAMPLE))


def test_parse_log_line_extracts_the_stats_fields():
    line = ("INFO 08-30 12:00:00 metrics.py:123] Avg prompt throughput: 0.0 tokens/s, "
            "Avg generation throughput: 812.5 tokens/s, Running: 12 reqs, "
            "Waiting: 3 reqs, GPU KV cache usage: 42.0%, Prefix cache hit rate: 1.0%")
    got = parse_log_line(line)
    assert got == {"running": 12, "waiting": 3,
                   "kv_usage_pct": 42.0, "prefix_hit_pct": 1.0}


def test_parse_log_line_returns_none_for_other_lines():
    assert parse_log_line("INFO something else entirely") is None


def test_parse_startup_captures_engine_args_verbatim():
    log = (
        "INFO 08-30 11:59:00 llm_engine.py:1] Initializing an LLM engine (v0.28.0) "
        "with config: model='Qwen/Qwen3-8B', dtype=torch.bfloat16, max_num_seqs=32, "
        "max_model_len=2048, enable_prefix_caching=True, kv_cache_dtype=auto\n"
        "INFO 08-30 11:59:30 worker.py:2] # GPU blocks: 24576, # CPU blocks: 2048\n"
    )
    got = parse_startup(log)
    assert "max_num_seqs=32" in got["engine_args_raw"]
    assert got["gpu_blocks"] == 24576


def test_parse_startup_reads_kv_cache_size_in_the_newer_format():
    log = "INFO 08-30 11:59:30 core.py:9] GPU KV cache size: 393,216 tokens\n"
    assert parse_startup(log)["kv_cache_tokens"] == 393216


def test_parse_startup_returns_empty_fields_when_absent():
    got = parse_startup("INFO nothing useful here\n")
    assert got["engine_args_raw"] is None
    assert got["gpu_blocks"] is None


def test_cross_check_passes_when_sources_agree():
    samples = [{"vllm:num_requests_running": 12.0, "vllm:num_requests_waiting": 3.0,
                "vllm:kv_cache_usage_perc": 0.42}]
    logs = [{"running": 12, "waiting": 3, "kv_usage_pct": 42.0, "prefix_hit_pct": 1.0}]
    assert cross_check(samples, logs)["agree"] is True


def test_cross_check_flags_a_real_divergence():
    samples = [{"vllm:num_requests_running": 12.0, "vllm:num_requests_waiting": 3.0,
                "vllm:kv_cache_usage_perc": 0.42}]
    logs = [{"running": 40, "waiting": 3, "kv_usage_pct": 42.0, "prefix_hit_pct": 1.0}]
    out = cross_check(samples, logs)
    assert out["agree"] is False
    assert "running" in out["diverged"]
