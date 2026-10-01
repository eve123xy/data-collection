"""ON INSTANCE.

    python scripts/prepare_burstgpt.py --window work --family qwen3
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import sys
from pathlib import Path

import pins
from common import count_rows, download, read_meta, verify

DEST = Path("/data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, choices=sorted(pins.WINDOWS))
    ap.add_argument("--family", required=True, choices=sorted(pins.POOLS))
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    out, meta_out = DEST / "replay.jsonl", DEST / "replay.meta.json"

    if out.exists() and meta_out.exists():
        m = read_meta(meta_out)
        if m.get("window") == args.window and m.get("family") == args.family:
            verify(out, m["sha256"], m["rows"])
            print(f"[OK] already present and verified: {out}")
            return

    stem = f"{args.window}-{args.family}"
    download(pins.DATASETS_REPO, f"burstgpt/{stem}.meta.json",
             pins.DATASETS_REVISION, DEST)
    got = download(pins.DATASETS_REPO, f"burstgpt/{stem}.jsonl",
                   pins.DATASETS_REVISION, DEST)
    m = read_meta(DEST / f"{stem}.meta.json")
    verify(got, m["sha256"], m["rows"])

    # the replay must actually carry the arrival rate its window claims
    actual = count_rows(got) / m["window_s"]
    if abs(actual - m["req_per_s"]) / m["req_per_s"] > pins.RATE_TOLERANCE:
        sys.exit(f"[FATAL] measured {actual:.3f} req/s != expected "
                 f"{m['req_per_s']:.3f} req/s for window {args.window}")

    got.replace(out)
    (DEST / f"{stem}.meta.json").replace(meta_out)
    print(f"[OK] {m['rows']:,} requests, {actual:.3f} req/s verified at {out}")


if __name__ == "__main__":
    main()
