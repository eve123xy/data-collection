"""ON INSTANCE. No arguments, so it cannot be misinvoked.

    python scripts/prepare_smallprompts.py
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

from pathlib import Path

import pins
from common import download, read_meta, verify

DEST = Path("/data")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    out, meta_out = DEST / "prompts.jsonl", DEST / "prompts.meta.json"

    if out.exists() and meta_out.exists():
        m = read_meta(meta_out)
        verify(out, m["sha256"], m["rows"])
        print(f"[OK] already present and verified: {out}")
        return

    download(pins.DATASETS_REPO, "smallprompts/prompts.meta.json",
             pins.DATASETS_REVISION, DEST)
    got = download(pins.DATASETS_REPO, "smallprompts/prompts.jsonl",
                   pins.DATASETS_REVISION, DEST)
    m = read_meta(meta_out)
    verify(got, m["sha256"], m["rows"])
    print(f"[OK] {m['rows']:,} prompts verified at {out}")


if __name__ == "__main__":
    main()
