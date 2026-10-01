"""LOCAL, post hoc. Re-derive the client/server token gate without touching the instrument.

    uv run python tools/recompute_token_attribution.py              # dry run
    uv run python tools/recompute_token_attribution.py --commit     # write corrections

`workload.mark_window()` attributes tokens to the measurement window by counting
ONE TOKEN PER `itl_ms` ENTRY. But `itl_ms` holds one entry per *streaming chunk*,
and vLLM packs several tokens into a chunk whenever generation outruns the send
loop -- so `attribution.tokens_in_window` is a chunk count wearing a token name
(OPERATIONAL_LEARNINGS 2.24). Verified exactly, to the token:
`tokens_in_window == sum(n_chunks) == sum(len(itl_ms) + 1)`.

The gap IS the tokens-per-chunk ratio, which rises as decode slows. That is why
only the slowest model's cells crossed the gate's 5% tolerance while every other
cell carries the same defect at 0.28-4.76%.

**The instrument is NOT changed.** All 248 cells run identical measurement code,
which is the whole point of freezing it; a mid-campaign edit would put part of
the difference between cell 3 and cell 200 into the code rather than the knob.
So the correction is computed here, after the fact, from artifacts already
retained, and written as a SEPARATE derived record. `run_meta.json` keeps the
raw value the instrument produced.

No measured quantity moves. `j_per_token` divides energy by vLLM's own
`server_tokens`, never by this field, so power, energy, throughput and J/token
are untouched and no cell is re-run. Only the FLAG STATUS of affected cells
changes -- and only where this was the sole reason a cell was flagged.

Correction, per request: count the chunks falling in the window exactly as
mark_window does, then weight them by that request's own tokens-per-chunk
ratio (`completion_tokens / n_chunks`). For a request wholly inside the window
that returns its exact `completion_tokens`; for one straddling the edge it
apportions by the same rule the original intended.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "cells", "upload", "workload", "telemetry"):
    sys.path.insert(0, str(ROOT / "scripts" / sub))

import pins                                       # noqa: E402
from common import load_env, token                # noqa: E402
from status import fetch_records                  # noqa: E402

TOKEN_TOLERANCE = 0.05          # the gate's own tolerance, unchanged


def corrected_tokens(records, t0, t1):
    """tokens_in_window, weighting each in-window chunk by its token payload."""
    total = 0.0
    for r in records:
        if r.get("first_token_ts") is None:
            continue
        chunks = (r.get("n_chunks") or (len(r.get("itl_ms") or []) + 1))
        toks = r.get("completion_tokens") or 0
        per_chunk = (toks / chunks) if chunks else 0.0

        in_win = 0
        t = r["first_token_ts"]
        if t0 <= t <= t1:
            in_win += 1
        for gap_ms in (r.get("itl_ms") or []):
            t += gap_ms / 1000.0
            if t > t1:
                break
            if t >= t0:
                in_win += 1
        total += in_win * per_chunk
    return int(round(total))


def split_flags(flags):
    """(token-gate flags, every other flag) from a recorded flag list."""
    flags = flags if isinstance(flags, list) else (
        [f.strip() for f in str(flags or "").split("|") if f.strip()])
    tok = [f for f in flags if "client tokens" in f]
    return tok, [f for f in flags if "client tokens" not in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--out", default="docs/TOKEN_ATTRIBUTION_CORRECTION.json")
    a = ap.parse_args()
    load_env()
    from huggingface_hub import hf_hub_download

    rows, changed = [], []
    recs = [r for r in fetch_records() if r.get("artifacts")]
    print(f"{len(recs)} recorded cell(s) with artifacts\n")

    for r in recs:
        cid, base = r["cell_id"], r["artifacts"].rstrip("/")
        try:
            mp = hf_hub_download(pins.RUNS_REPO, f"{base}/run_meta.json",
                                 repo_type="dataset", token=token())
            m = json.load(open(mp))
            rp = hf_hub_download(pins.RUNS_REPO, f"{base}/requests_{cid}.jsonl",
                                 repo_type="dataset", token=token())
            reqs = [json.loads(l) for l in open(rp) if l.strip()]
        except Exception as e:
            print(f"[--] {cid}: {type(e).__name__} - skipped")
            continue

        t0 = (m.get("window_start_ts_ns") or 0) / 1e9
        t1 = (m.get("window_end_ts_ns") or 0) / 1e9
        if not (t0 and t1):
            print(f"[--] {cid}: no window timestamps - skipped")
            continue

        recorded = (m.get("attribution") or {}).get("tokens_in_window")
        server = (m.get("server") or {}).get("generation_tokens")
        fixed = corrected_tokens(reqs, t0, t1)
        if not server:
            continue

        old_rel = abs((recorded or 0) - server) / server
        new_rel = abs(fixed - server) / server
        tok_flags, other = split_flags(r.get("flags"))

        # Re-derive the outcome the gate WOULD have reached with real tokens.
        would_flag = new_rel > TOKEN_TOLERANCE
        new_flags = other + ([f"{cid}: client tokens {fixed:,} vs server "
                              f"{server:,} differ by {100 * new_rel:.1f}%"]
                             if would_flag else [])
        new_outcome = ("run_failed" if r["outcome"] == "run_failed"
                       else ("flagged" if new_flags else "ok"))

        row = {"cell_id": cid, "outcome_recorded": r["outcome"],
               "outcome_corrected": new_outcome,
               "tokens_in_window_recorded": recorded,
               "tokens_in_window_corrected": fixed,
               "server_generation_tokens": server,
               "gap_recorded_pct": round(100 * old_rel, 3),
               "gap_corrected_pct": round(100 * new_rel, 3),
               "flags_corrected": new_flags,
               "flags_dropped": tok_flags}
        rows.append(row)
        if new_outcome != r["outcome"] or tok_flags:
            changed.append(row)
            print(f"[~] {cid}\n     gap {100*old_rel:.2f}% -> {100*new_rel:.2f}%   "
                  f"{r['outcome']} -> {new_outcome}")

    out = Path(a.out)
    payload = {"note": "Post hoc correction. Instrument unchanged; run_meta.json "
                       "retains the raw value. See OPERATIONAL_LEARNINGS 2.24.",
               "tolerance": TOKEN_TOLERANCE, "cells": rows}
    print(f"\n{len(rows)} cell(s) recomputed, {len(changed)} affected")
    if not a.commit:
        print("dry run - nothing written. Re-run with --commit.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
