"""Parse vLLM's /metrics endpoint and its stdout stats lines, and check they agree.

Both sources are collected because each carries something the other does not:
/metrics has counters and histograms the log never prints; the log is
human-readable and is captured anyway. Requiring them to AGREE is what turns
two sources into a check rather than two guesses.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import re

_LOG_RE = re.compile(
    r"Running:\s*(?P<running>\d+)\s*reqs.*?"
    r"Waiting:\s*(?P<waiting>\d+)\s*reqs.*?"
    r"GPU KV cache usage:\s*(?P<kv>[\d.]+)%"
    r"(?:.*?Prefix cache hit rate:\s*(?P<hit>[\d.]+)%)?",
    re.S,
)

_ENGINE_RE = re.compile(r"with config:\s*(?P<args>.+)")
_BLOCKS_RE = re.compile(r"#\s*GPU blocks:\s*(?P<n>[\d,]+)")
_KVTOK_RE = re.compile(r"GPU KV cache size:\s*(?P<n>[\d,]+)\s*tokens")


def parse_prometheus(text):
    """name -> value. Histogram buckets keep their `le` label; others drop labels."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_part, _, value = line.rpartition(" ")
        if not name_part:
            continue
        try:
            v = float(value)
        except ValueError:
            continue
        if "{" in name_part:
            base, _, labels = name_part.partition("{")
            labels = labels.rstrip("}")
            le = None
            for kv in labels.split(","):
                k, _, val = kv.partition("=")
                if k.strip() == "le":
                    le = val.strip().strip('"')
            key = f'{base}{{le="{le}"}}' if le is not None else base
        else:
            key = name_part
        out[key] = v
    return out


def parse_log_line(line):
    """Extract one vLLM stats line, or None if this is not one."""
    m = _LOG_RE.search(line)
    if not m:
        return None
    hit = m.group("hit")
    return {"running": int(m.group("running")),
            "waiting": int(m.group("waiting")),
            "kv_usage_pct": float(m.group("kv")),
            "prefix_hit_pct": float(hit) if hit is not None else None}


def parse_startup(log_text):
    """Pull the engine config and the allocated KV cache out of the startup log.

    The KV cache size as allocated is the memory ceiling for the cell. It is what
    distinguishes a KV-bound cell from a max_num_seqs-bound one - the archive's
    17-of-90 defect, where a row's concurrency label did not describe the regime
    it actually ran in.
    """
    def _int(m):
        return int(m.group("n").replace(",", "")) if m else None

    args = _ENGINE_RE.search(log_text)
    return {"engine_args_raw": args.group("args").strip() if args else None,
            "gpu_blocks": _int(_BLOCKS_RE.search(log_text)),
            "kv_cache_tokens": _int(_KVTOK_RE.search(log_text))}


# The /metrics names this module depends on, in ONE place. vLLM renames these
# between versions: 0.28.0 renamed vllm:gpu_cache_usage_perc (the 0.21.0 name
# the archive used) to vllm:kv_cache_usage_perc. That rename cost a whole smoke
# cell, because the old name simply returned nothing and every consumer turned
# "no samples" into 0.0 -- a plausible-looking measured zero rather than an
# absence. Anything added here must also be handled by missing_metrics().
M_RUNNING = "vllm:num_requests_running"
M_WAITING = "vllm:num_requests_waiting"
M_KV_USAGE = "vllm:kv_cache_usage_perc"       # 0.21.0: vllm:gpu_cache_usage_perc
M_PREFIX_Q = "vllm:prefix_cache_queries_total"
M_PREFIX_H = "vllm:prefix_cache_hits_total"
M_PREEMPT = "vllm:num_preemptions_total"
M_GEN_TOK = "vllm:generation_tokens_total"
M_PROMPT_TOK = "vllm:prompt_tokens_total"

REQUIRED_METRICS = [M_RUNNING, M_WAITING, M_KV_USAGE, M_PREFIX_Q, M_PREFIX_H,
                    M_PREEMPT, M_GEN_TOK, M_PROMPT_TOK]


def missing_metrics(samples):
    """Pinned metric names that never appeared in ANY sample.

    A renamed or removed metric must be loud. Silence here is the failure mode
    that produced a null kv_usage column for a whole cell while every gate
    passed.
    """
    return [m for m in REQUIRED_METRICS
            if not any(s.get(m) is not None for s in samples)]


def _mean(xs):
    """Mean, or None when there is nothing to average.

    Deliberately NOT 0.0. An absent metric and a metric that genuinely measured
    zero must not be indistinguishable downstream.
    """
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def cross_check(samples, log_rows, tol=0.15):
    """Compare the two sources on the fields they share.

    Compares window means rather than instants: the log emits every ~10s and
    /metrics every 1s, so they never sample the same moment.
    """
    if not samples or not log_rows:
        return {"agree": None, "reason": "one source empty", "diverged": [],
                "absent": [], "missing_metrics": missing_metrics(samples),
                "metrics": {}, "log": {}}

    kv = _mean([s.get(M_KV_USAGE) for s in samples])
    a = {"running": _mean([s.get(M_RUNNING) for s in samples]),
         "waiting": _mean([s.get(M_WAITING) for s in samples]),
         "kv_usage_pct": None if kv is None else 100.0 * kv}
    b = {"running": _mean([r.get("running") for r in log_rows]),
         "waiting": _mean([r.get("waiting") for r in log_rows]),
         "kv_usage_pct": _mean([r.get("kv_usage_pct") for r in log_rows])}

    # A field absent from either source is reported as absent, never compared.
    # Comparing against a stand-in zero is how a rename shows up as "the two
    # sources disagree" instead of "the metric no longer exists".
    diverged, absent = [], []
    for k in a:
        if a[k] is None or b[k] is None:
            absent.append(k)
            continue
        scale = max(abs(a[k]), abs(b[k]), 1.0)
        if abs(a[k] - b[k]) / scale > tol:
            diverged.append(k)
    return {"agree": (not diverged) and (not absent), "diverged": diverged,
            "absent": absent, "missing_metrics": missing_metrics(samples),
            "metrics": a, "log": b, "tolerance": tol}
