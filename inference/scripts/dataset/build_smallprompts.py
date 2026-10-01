"""LOCAL, run once. Freeze bucket-S into a shuffled artifact and publish it.

    python scripts/build_smallprompts.py --publish
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pins
from common import (count_rows, publish, read_jsonl, sha256_file, token,
                    write_jsonl, write_meta)


def shuffle_prompts(texts, seed):
    """Shuffle the whole pool once so ANY prefix is a representative sample:
    a slow cell reads 2k rows and a fast one 18k, and they must see the same
    length distribution."""
    rows = [{"prompt_id": i, "text": t} for i, t in enumerate(texts)]
    random.Random(seed).shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build_out/smallprompts")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    tok = token()
    revision = HfApi().dataset_info(pins.SMALLPROMPTS_SOURCE, token=tok).sha
    src = hf_hub_download(repo_id=pins.SMALLPROMPTS_SOURCE,
                          filename=pins.SMALLPROMPTS_FILE, revision=revision,
                          repo_type="dataset", token=tok)
    texts = [r["prompt"] for r in read_jsonl(src)]

    if len(texts) != pins.SMALLPROMPTS_ROWS:
        sys.exit(f"[FATAL] source has {len(texts):,} rows, "
                 f"expected {pins.SMALLPROMPTS_ROWS:,}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "prompts.jsonl"
    write_jsonl(out, shuffle_prompts(texts, pins.SHUFFLE_SEED))

    write_meta(out_dir / "prompts.meta.json", {
        "artifact": "smallprompts/prompts.jsonl",
        "source": pins.SMALLPROMPTS_SOURCE,
        "source_revision": revision,
        "source_file": pins.SMALLPROMPTS_FILE,
        "seed": pins.SHUFFLE_SEED,
        "rows": count_rows(out),
        "sha256": sha256_file(out),
        "built_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[OK] {count_rows(out):,} rows -> {out}")

    if args.publish:
        publish(pins.DATASETS_REPO, out_dir.parent)


if __name__ == "__main__":
    main()
