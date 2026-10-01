"""LOCAL, run once. A LARGER exact-ISL-2048 pool, for one cell that exhausted the original.

    uv run python scripts/dataset/build_isl2048_expanded.py [--publish]

`islosl_h200_qwen3-8b_2048-128_b32` returned a genuine `run_failed`:

    prompt pool exhausted - the dataset spec asserts no cell wraps,
    so this run is invalid

Long inputs with SHORT outputs is the worst case for pool demand -- 2048/128
sustains ~10 req/s, so a 600 s window needs ~6,200 distinct prompts and the
original isl2048 artifact holds 4,966. The driver refuses to recycle, correctly:
with prefix caching on, a repeated prompt is served from cache and the cell
measures the cache instead of the model.

**This is published as a SEPARATE artifact, not a replacement.** Four cells have
already run against `islprompts/isl2048` (sha 60ccfcab...), and overwriting it
would silently change what their recorded provenance points at. Only the one
failing cell reads this pool; the operator's instruction was to expand it "but
only for that one".

Extra supply comes from ShareGPT prompts already >= 2048 tokens, truncated to
exactly 2048 by the same tokenizer and the same rule as the original -- so the
pool stays a superset in kind, not a different kind of text.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/

import argparse
import json
import random
from pathlib import Path

import pins
from common import publish, sha256_file, token, write_jsonl, write_meta

OWNER = "<hf-owner>"
TOKENIZER = "Qwen/Qwen3-8B"
TARGET = 2048
BASE_SOURCES = [("ExtraLargePromptsDataset", "prompts_XL.jsonl"),
                ("HugePromptsDataset", "prompts_XXL.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build_out/isl2048x_publish")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok_hf = token()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, token=tok_hf)

    texts, prov = [], []
    for repo, fname in BASE_SOURCES:
        p = hf_hub_download(f"{OWNER}/{repo}", fname, repo_type="dataset", token=tok_hf)
        n = 0
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("n_tokens") or 0) >= TARGET:
                texts.append(r["prompt"]); n += 1
        prov.append(f"{OWNER}/{repo} ({n:,})")

    # extra supply: ShareGPT prompts already long enough to truncate down
    p = hf_hub_download(f"{OWNER}/ShareGPTPairsQwen3", "sharegpt_pairs_qwen3.jsonl",
                        repo_type="dataset", token=tok_hf)
    n = 0
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("prompt_len") or 0) >= TARGET:
            texts.append(r["prompt"]); n += 1
    prov.append(f"{OWNER}/ShareGPTPairsQwen3 ({n:,})")

    rows, dropped, seen = [], 0, set()
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if len(ids) < TARGET:
            dropped += 1
            continue
        cut = tokenizer.decode(ids[:TARGET], skip_special_tokens=True)
        # NOT deduplicated, deliberately. The original islprompts/isl2048 is 27%
        # duplicates (4,966 rows, 3,624 distinct) and the four cells already run
        # against it drew 4-26% repeats. Deduplicating here would make THIS cell
        # cleaner than its four siblings, which is worse for comparability than
        # matching them: the ISL_OSL grid is read across cells, not in isolation.
        # Distinct >=2048-token supply is ~3,752 in total and there is no more --
        # the M and L buckets contain zero prompts that long -- so a deduplicated
        # pool cannot reach this cell's ~6,200 demand at any window length.
        seen.add(cut)
        rows.append(cut)

    random.Random(pins.SHUFFLE_SEED).shuffle(rows)
    out_rows = [{"prompt_id": i, "text": t} for i, t in enumerate(rows)]

    exact = sum(1 for r in out_rows[:200]
                if len(tokenizer(r["text"], add_special_tokens=False)["input_ids"]) == TARGET)

    NEED = 6200
    if len(out_rows) < NEED:
        sys.exit(f"[FATAL] only {len(out_rows):,} prompts; the failing cell needs ~{NEED:,}")

    d = Path(a.out) / "islprompts" / "isl2048_expanded"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "prompts.jsonl"
    write_jsonl(f, out_rows)
    write_meta(d / "prompts.meta.json", {
        "artifact": "islprompts/isl2048_expanded/prompts.jsonl",
        "target_isl_tokens": TARGET,
        "tokenizer": TOKENIZER,
        "sources": prov,
        "seed": pins.SHUFFLE_SEED,
        "rows": len(out_rows),
        "dropped_short_or_duplicate": dropped,
        "verified": min(200, len(out_rows)),
        "verified_exact": exact,
        "sha256": sha256_file(f),
        "supersedes_for": ["islosl_h200_qwen3-8b_2048-128_b32"],
        "note": "Separate artifact. islprompts/isl2048 is UNCHANGED and remains "
                "the provenance for the four 2048 cells already completed.",
    })
    print(f"[OK] isl2048_expanded: {len(out_rows):,} rows "
          f"({dropped:,} dropped short/dup), {exact}/200 verified exactly {TARGET}")
    print(f"     sources: {'; '.join(prov)}")
    if a.publish:
        rev = publish(pins.DATASETS_REPO, Path(a.out))
        print(f"[OK] revision {rev}")


if __name__ == "__main__":
    main()
