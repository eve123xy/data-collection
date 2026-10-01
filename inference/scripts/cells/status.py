"""LOCAL. The work queue: what has run, what is next.

    python scripts/status.py
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import json
import sys
from collections import Counter
from pathlib import Path

import pins
from common import token

MANIFEST = Path(__file__).resolve().parents[2] / "plan" / "cells_v1.json"
DONE = ("ok", "flagged", "skipped")


def summarise_records(cells, records):
    by_id = {r["cell_id"]: r for r in records}
    counts = Counter()
    flagged = []
    for c in cells:
        if c.get("blocked"):
            counts["blocked"] += 1
            continue
        rec = by_id.get(c["cell_id"])
        if rec is None:
            counts["pending"] += 1
            continue
        counts[rec["outcome"]] += 1
        if rec["outcome"] == "flagged":
            flagged.append((c["cell_id"], rec.get("flags", [])))

    nxt = None
    for c in sorted(cells, key=lambda c: c["order"]):
        if c.get("blocked"):
            continue
        rec = by_id.get(c["cell_id"])
        if rec is None or rec["outcome"] not in DONE:
            nxt = c["cell_id"]
            break
    return {"counts": dict(counts), "next": nxt, "flagged": flagged,
            "blocked": [c["cell_id"] for c in cells if c.get("blocked")]}


def fetch_records():
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    try:
        files = [f for f in api.list_repo_files(pins.RUNS_REPO, repo_type="dataset",
                                                token=token())
                 if f.startswith("runs/") and f.endswith(".json")]
    except Exception as e:
        print(f"[WARN] runs repo unreadable ({type(e).__name__}); "
              f"treating every cell as pending")
        return []
    out = []
    for f in files:
        p = hf_hub_download(repo_id=pins.RUNS_REPO, filename=f,
                            repo_type="dataset", token=token())
        out.append(json.loads(Path(p).read_text()))
    return out


def main():
    if not MANIFEST.exists():
        sys.exit(f"[FATAL] {MANIFEST} missing - run tools/build_cells.py")
    cells = json.loads(MANIFEST.read_text())
    s = summarise_records(cells, fetch_records())
    print(f"{len(cells)} cells: " + ", ".join(
        f"{v} {k}" for k, v in sorted(s['counts'].items())))
    print(f"next: {s['next']}")
    if s["flagged"]:
        print("flagged:")
        for cid, flags in s["flagged"]:
            print(f"  {cid}  {'; '.join(flags)}")


if __name__ == "__main__":
    main()
