"""LOCAL, run once. Freeze one prompt artifact per free-generation SOURCE.

    uv run --with pyarrow python scripts/dataset/build_source_prompts.py
    uv run --with pyarrow python scripts/dataset/build_source_prompts.py --publish

ISL_OSL blocks 2 and 3 sweep the *demand* side: real prompts from three domains,
`max_tokens` 2048, no `ignore_eos`, so the model decides when to stop and the
joint ISL/OSL distribution is the measurement. run_plan_settings_v2 §1 says
"source datasets as-is" -- so unlike the ISL grid, nothing here is truncated.

Twelve cells were blocked because these artifacts did not exist and `cell.py`
had no way to resolve `workload.source`. Without both, a cell would have served
generic bucket-S prompts while being recorded under `math-aime` -- which is
exactly the mislabelling that got them blocked in the first place.

**Pool size is a correctness constraint, not a convenience.** A free-generation
cell at batch 32 over a 600 s window issues several thousand requests, and the
driver treats a short pool as a hard error rather than silently recycling. That
is deliberate: `enable_prefix_caching` is ON, so a repeated prompt would be
served from cache and the cell would measure the cache instead of the model.
Every pool here must therefore exceed the largest request count any cell will
ask of it, with margin.

    chat-sharegpt        ShareGPTPairsQwen3            382,424 available
    code-instructcoder   likaixin/InstructCoder        108,391 available
    math-comp            AIME 1983-2024 + NuminaMath   ~185,000 available

**On `math-comp`.** The manifest calls this source `math-aime`, and AIME alone
CANNOT support it: every AIME problem ever set between 1983 and 2024 is 933
prompts, roughly a seventh of one cell's demand. Competition mathematics is
intrinsically a small corpus. On the operator's decision (2026-09-04) the source
is widened to competition maths as a class, keeping AIME as its core: all 933
AIME problems first, then the competition subset of NuminaMath-CoT
(`amc_aime`, `aops_forum`, `olympiads`). The label changes with the content --
these cells are competition maths, not AIME, and metadata must say so.

Prompts are deduplicated before shuffling: a duplicate is a prefix-cache hit
waiting to happen, which is the same contamination the pool-size rule exists to
prevent.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/

import argparse
import hashlib
import random
from pathlib import Path

import pins
from common import publish, sha256_file, token, write_jsonl, write_meta

OWNER = "<hf-owner>"
MAX_ROWS = 200_000          # plenty over the largest cell demand; keeps the artifact small
MIN_CHARS = 16              # a 3-character "prompt" is not a workload


def _clean(texts):
    """Deduplicate, drop stubs, keep order stable for the seeded shuffle."""
    seen, out = set(), []
    for t in texts:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if len(t) < MIN_CHARS:
            continue
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out


def build_chat_sharegpt(tok):
    from huggingface_hub import hf_hub_download
    import json
    p = hf_hub_download(f"{OWNER}/ShareGPTPairsQwen3", "sharegpt_pairs_qwen3.jsonl",
                        repo_type="dataset", token=tok)
    texts = []
    for line in open(p, encoding="utf-8"):
        if line.strip():
            texts.append(json.loads(line).get("prompt"))
    return texts, [f"{OWNER}/ShareGPTPairsQwen3"]


def build_code_instructcoder(tok):
    from huggingface_hub import hf_hub_download
    import json
    texts = []
    for fname in ("train.json", "valid.json"):
        d = json.load(open(hf_hub_download("likaixin/InstructCoder", fname,
                                           repo_type="dataset", token=tok)))
        for r in d:
            instr, inp = (r.get("instruction") or ""), (r.get("input") or "")
            # InstructCoder splits the ask from the code it edits; a prompt that
            # dropped `input` would be an instruction with nothing to act on.
            texts.append(f"{instr}\n\n{inp}".strip() if inp else instr)
    return texts, ["likaixin/InstructCoder"]


def build_math_comp(tok):
    from huggingface_hub import hf_hub_download
    import csv
    import pandas as pd
    texts, srcs = [], []

    p = hf_hub_download("di-zhang-fdu/AIME_1983_2024", "AIME_Dataset_1983_2024.csv",
                        repo_type="dataset", token=tok)
    aime = [r["Question"] for r in csv.DictReader(open(p, encoding="utf-8"))]
    texts += aime                                   # AIME first: it is the core
    srcs.append("di-zhang-fdu/AIME_1983_2024")

    keep = {"amc_aime", "aops_forum", "olympiads"}
    for i in range(5):
        f = hf_hub_download("AI-MO/NuminaMath-CoT",
                            f"data/train-0000{i}-of-00005.parquet",
                            repo_type="dataset", token=tok)
        df = pd.read_parquet(f, columns=["problem", "source"])
        texts += df[df["source"].isin(keep)]["problem"].tolist()
    srcs.append("AI-MO/NuminaMath-CoT[amc_aime,aops_forum,olympiads]")
    return texts, srcs


BUILDERS = {"chat-sharegpt": build_chat_sharegpt,
            "code-instructcoder": build_code_instructcoder,
            "math-comp": build_math_comp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build_out/source_publish")
    ap.add_argument("--only", default=None)
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    tok = token()

    for name, fn in BUILDERS.items():
        if a.only and a.only != name:
            continue
        raw, srcs = fn(tok)
        rows = _clean(raw)
        random.Random(pins.SHUFFLE_SEED).shuffle(rows)
        rows = rows[:MAX_ROWS]
        if len(rows) < 20_000:
            sys.exit(f"[FATAL] {name}: only {len(rows):,} usable prompts - a "
                     f"free-generation cell can exhaust that mid-window")
        out_rows = [{"prompt_id": i, "text": t} for i, t in enumerate(rows)]

        d = Path(a.out) / "sourceprompts" / name
        d.mkdir(parents=True, exist_ok=True)
        out = d / "prompts.jsonl"
        write_jsonl(out, out_rows)
        write_meta(d / "prompts.meta.json", {
            "artifact": f"sourceprompts/{name}/prompts.jsonl",
            "source_name": name,
            "sources": srcs,
            "seed": pins.SHUFFLE_SEED,
            "raw_candidates": len(raw),
            "after_dedupe_and_filter": len(_clean(raw)),
            "rows": len(out_rows),
            "truncated": False,       # §1: free-gen sources are used as-is
            "sha256": sha256_file(out),
        })
        print(f"[OK] {name:<20} {len(out_rows):>7,} rows from {len(raw):>7,} raw "
              f"({len(raw) - len(_clean(raw)):,} dropped as dup/stub)")

    if a.publish:
        rev = publish(pins.DATASETS_REPO, Path(a.out))
        print(f"[OK] set pins.SOURCEPROMPTS_REVISION = \"{rev}\"")


if __name__ == "__main__":
    main()
