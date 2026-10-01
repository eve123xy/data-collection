"""LOCAL, run once. Freeze one prompt artifact per exact ISL and publish them.

    uv run python scripts/dataset/build_isl_prompts.py
    uv run python scripts/dataset/build_isl_prompts.py --publish

The ISL_OSL grid sweeps input length across 128 / 512 / 2048 tokens, and
run_plan_settings_v2 §1 requires "bucket M/L truncated to exact ISL". Neither
half of that was happening: every grid cell pointed at bucket-S, whose prompts
are 129-384 tokens and which therefore CANNOT supply 512 or 2048, and nothing
truncated anything. All 15 cells would have run at whatever bucket-S offered.

Truncation is by TOKEN, not character, and uses the model's own tokenizer, so
"ISL 512" means 512 tokens to the server rather than an approximation. Measured
bucket ranges (n_tokens, from each dataset's own field):

    S    116,981 rows    129 - 384
    M     38,233 rows    385 - 768
    L      5,879 rows    769 - 1536
    XL     2,870 rows   1537 - 3070
    XXL    3,563 rows   3073 - 65293

so a prompt can only be truncated DOWN to a target, never padded up:

    ISL  128  <- S, M, L        (any prompt >= 128 tokens)
    ISL  512  <- M, L           (M's shorter half is excluded by the filter)
    ISL 2048  <- XL, XXL

The pool is shuffled once with pins.SHUFFLE_SEED so any prefix is a
representative sample -- a slow cell reads 300 rows and a fast one 4,000, and
they must see the same distribution.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/

import argparse
import random
from pathlib import Path

import pins
from common import publish, sha256_file, token, write_jsonl, write_meta

# target ISL -> source buckets, in preference order
SOURCES = {
    128:  [("SmallPromptsDataset", "prompts_S.jsonl"),
           ("MediumPromptsDataset", "prompts_M.jsonl")],
    512:  [("MediumPromptsDataset", "prompts_M.jsonl"),
           ("LargePromptsDataset", "prompts_L.jsonl")],
    2048: [("ExtraLargePromptsDataset", "prompts_XL.jsonl"),
           ("HugePromptsDataset", "prompts_XXL.jsonl")],
}
OWNER = "<hf-owner>"
TOKENIZER = "Qwen/Qwen3-8B"   # one tokenizer across the Qwen3 family


def build_one(isl, tokenizer, tok_hf, verify=200):
    """Rows of exactly `isl` tokens, drawn from prompts long enough to trim."""
    import json
    from huggingface_hub import hf_hub_download

    texts = []
    for repo, fname in SOURCES[isl]:
        p = hf_hub_download(repo_id=f"{OWNER}/{repo}", filename=fname,
                            repo_type="dataset", token=tok_hf)
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            # n_tokens is the dataset's own count; the real check is the
            # re-tokenisation below, this just avoids tokenising the whole pool
            if r.get("n_tokens", 0) >= isl:
                texts.append(r["prompt"])

    rows, dropped = [], 0
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if len(ids) < isl:
            dropped += 1          # n_tokens disagreed with our tokenizer
            continue
        rows.append(tokenizer.decode(ids[:isl], skip_special_tokens=True))

    random.Random(pins.SHUFFLE_SEED).shuffle(rows)
    out = [{"prompt_id": i, "text": t} for i, t in enumerate(rows)]

    # Verify a sample really re-tokenises to the target. Decode/re-encode is not
    # guaranteed to round-trip exactly, so this is measured rather than assumed.
    exact = off = 0
    for r in out[:verify]:
        n = len(tokenizer(r["text"], add_special_tokens=False)["input_ids"])
        if n == isl:
            exact += 1
        else:
            off += 1
    return out, {"candidates": len(texts), "dropped_short": dropped,
                 "rows": len(out), "verified": min(verify, len(out)),
                 "verified_exact": exact, "verified_off_by_any": off}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build_out/isl_publish")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok_hf = token()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, token=tok_hf)

    for isl in sorted(SOURCES):
        rows, stats = build_one(isl, tokenizer, tok_hf)
        if not rows:
            sys.exit(f"[FATAL] ISL {isl}: no prompt long enough to truncate")
        d = Path(a.out) / "islprompts" / f"isl{isl}"
        d.mkdir(parents=True, exist_ok=True)
        out = d / "prompts.jsonl"
        write_jsonl(out, rows)
        write_meta(d / "prompts.meta.json", {
            "artifact": f"islprompts/isl{isl}/prompts.jsonl",
            "target_isl_tokens": isl,
            "tokenizer": TOKENIZER,
            "sources": [f"{OWNER}/{r}" for r, _ in SOURCES[isl]],
            "seed": pins.SHUFFLE_SEED,
            "sha256": sha256_file(out),
            **stats,
        })
        pct = 100.0 * stats["verified_exact"] / max(1, stats["verified"])
        print(f"[OK] ISL {isl:>4}: {stats['rows']:>6,} rows from "
              f"{stats['candidates']:>6,} candidates "
              f"({stats['dropped_short']} too short) - "
              f"{pct:.1f}% of a {stats['verified']}-row sample re-tokenise to "
              f"exactly {isl}")
    if a.publish:
        # one commit for all three, so the repo gets islprompts/isl<N>/...
        rev = publish(pins.DATASETS_REPO, Path(a.out))
        print(f"[OK] set pins.ISLPROMPTS_REVISION = \"{rev}\"")


if __name__ == "__main__":
    main()
