"""ON INSTANCE. Fetch the free-generation SOURCE prompt artifact a cell needs.

    python scripts/dataset/prepare_source_prompts.py --source math-aime

Takes the MANIFEST's source label and resolves it through pins.SOURCE_ARTIFACT,
so a cell asks for what it is called and gets what it was decided to be served:
`math-aime` resolves to the `math-comp` artifact (see the note in pins.py --
AIME alone is 933 prompts, a fraction of one cell's demand).

Stages into a per-source subdirectory and only then moves the files into place.
common.download() names its output after the source basename, so downloading
straight into /data would collide with the bucket-S artifact at
/data/prompts.jsonl -- that collision already cost a cell start once
(OPERATIONAL_LEARNINGS 2.20).
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
    ap.add_argument("--source", required=True, choices=sorted(pins.SOURCE_ARTIFACT))
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)

    art = pins.SOURCE_ARTIFACT[a.source]
    out = DEST / f"prompts_src_{art}.jsonl"
    meta_out = DEST / f"prompts_src_{art}.meta.json"
    if out.exists() and meta_out.exists():
        m = read_meta(meta_out)
        verify(out, m["sha256"], m["rows"])
        print(f"[OK] already present and verified: {out}")
        return

    stage = DEST / f".stage_src_{art}"
    stage.mkdir(parents=True, exist_ok=True)
    src = f"sourceprompts/{art}"
    got_meta = download(pins.DATASETS_REPO, f"{src}/prompts.meta.json",
                        pins.SOURCEPROMPTS_REVISION, stage)
    got = download(pins.DATASETS_REPO, f"{src}/prompts.jsonl",
                   pins.SOURCEPROMPTS_REVISION, stage)
    Path(got).replace(out)
    Path(got_meta).replace(meta_out)
    shutil.rmtree(stage, ignore_errors=True)

    m = read_meta(meta_out)
    verify(out, m["sha256"], m["rows"])
    if m["source_name"] != art:
        sys.exit(f"[FATAL] artifact says '{m['source_name']}', asked for '{art}'")
    expect = pins.SOURCEPROMPTS_ROWS.get(art)
    if expect and m["rows"] != expect:
        sys.exit(f"[FATAL] {art}: {m['rows']:,} rows, pins expects {expect:,}")
    print(f"[OK] {m['rows']:,} '{art}' prompts verified at {out}"
          + (f"  (manifest label '{a.source}')" if a.source != art else ""))


if __name__ == "__main__":
    main()
