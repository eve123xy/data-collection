"""ON INSTANCE. Fetch the exact-ISL prompt artifact a cell needs.

    python scripts/dataset/prepare_isl_prompts.py --isl 512

The ISL_OSL grid sweeps input length across 128 / 512 / 2048 tokens and
run_plan_settings_v2 §1 requires prompts truncated to exactly that length.
Bucket-S is 129-384 tokens and physically cannot supply 512 or 2048, so these
cells read a separate frozen artifact built by build_isl_prompts.py.

Pinned at pins.ISLPROMPTS_REVISION, which is deliberately SEPARATE from
DATASETS_REVISION so publishing these could not move the revision the bucket-S
artifact is pinned at.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/

import argparse
import shutil
from pathlib import Path

import pins
from common import download, read_meta, verify

DEST = Path("/data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=int, required=True, choices=sorted(pins.ISLPROMPTS_ROWS))
    ap.add_argument("--expanded", action="store_true",
                    help="ISL 2048 only: fetch the larger pool built for the one "
                         "cell that exhausted the original. Writes the SAME "
                         "/data path, so cell.py needs no change.")
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)

    out = DEST / f"prompts_isl{a.isl}.jsonl"
    meta_out = DEST / f"prompts_isl{a.isl}.meta.json"
    if out.exists() and meta_out.exists():
        m = read_meta(meta_out)
        verify(out, m["sha256"], m["rows"])
        print(f"[OK] already present and verified: {out}")
        return

    # Stage into a per-ISL subdir, NEVER straight into /data. download() names
    # the file after its basename, so both this and prepare_smallprompts.py
    # resolve to /data/prompts.jsonl -- and the rename below used to move the
    # bucket-S artifact out from under the next bucket-S cell. That cost a cell
    # start on 2026-09-03 (FileNotFoundError before the server was queried, so
    # nothing was measured or recorded). See OPERATIONAL_LEARNINGS 2.20.
    stage = DEST / f".stage_isl{a.isl}"
    stage.mkdir(parents=True, exist_ok=True)
    if a.expanded:
        if a.isl != 2048:
            sys.exit("[FATAL] --expanded exists only for ISL 2048")
        src, rev, want = ("islprompts/isl2048_expanded",
                          pins.ISL2048_EXPANDED_REVISION, pins.ISL2048_EXPANDED_ROWS)
    else:
        src, rev, want = f"islprompts/isl{a.isl}", pins.ISLPROMPTS_REVISION, None
    got_meta = download(pins.DATASETS_REPO, f"{src}/prompts.meta.json", rev, stage)
    got = download(pins.DATASETS_REPO, f"{src}/prompts.jsonl", rev, stage)
    Path(got).replace(out)
    Path(got_meta).replace(meta_out)
    shutil.rmtree(stage, ignore_errors=True)
    m = read_meta(meta_out)
    verify(out, m["sha256"], m["rows"])
    if a.expanded and m["rows"] != pins.ISL2048_EXPANDED_ROWS:
        sys.exit(f"[FATAL] expanded pool has {m['rows']:,} rows, pins expects "
                 f"{pins.ISL2048_EXPANDED_ROWS:,}")
    if m["target_isl_tokens"] != a.isl:
        sys.exit(f"[FATAL] artifact says ISL {m['target_isl_tokens']}, asked for {a.isl}")
    print(f"[OK] {m['rows']:,} prompts of exactly {a.isl} tokens verified at {out}")


if __name__ == "__main__":
    main()
