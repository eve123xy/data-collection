"""LOCAL. Fetch a cell's artifacts and verify them against checksums.json.

    python scripts/upload/pull_runs.py --cell-id <id>
    python scripts/upload/pull_runs.py --all

Fetches from HuggingFace rather than Drive: one API, already authenticated, same
content. A file whose sha256 does not match is reported and NOT written over a
good local copy.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import hashlib
import json
from pathlib import Path

import pins
from upload_run import CHECKSUMS_NAME, dest_prefix


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def verify_local(run_dir, manifest):
    """Names that are missing locally or whose contents do not match."""
    bad = []
    for name, meta in manifest.get("files", {}).items():
        p = Path(run_dir) / name
        if not p.exists() or p.stat().st_size != meta["size"]:
            bad.append(name)
            continue
        if _sha256_bytes(p.read_bytes()) != meta["sha256"]:
            bad.append(name)
    return sorted(bad)


def pull(cell_row, dest_root, fetch):
    """Fetch every file listed in the remote checksums.json and verify it.

    `fetch(repo_path) -> bytes` is injected so this is testable without network.
    """
    prefix = dest_prefix(cell_row)
    out = Path(dest_root) / prefix
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(fetch(f"{prefix}/{CHECKSUMS_NAME}").decode())
    (out / CHECKSUMS_NAME).write_text(json.dumps(manifest, indent=1) + "\n")

    for name, meta in manifest.get("files", {}).items():
        body = fetch(f"{prefix}/{name}")
        if _sha256_bytes(body) != meta["sha256"]:
            print(f"[WARN] {name}: sha256 mismatch, not written")
            continue
        (out / name).write_bytes(body)

    bad = verify_local(out, manifest)
    incomplete = manifest.get("incomplete") or []
    return {"ok": not bad and not incomplete, "bad": bad,
            "incomplete": incomplete, "dir": str(out)}


def _fetch(repo_path):
    from huggingface_hub import hf_hub_download
    from common import token
    p = hf_hub_download(repo_id=pins.RUNS_REPO, filename=repo_path,
                        repo_type="dataset", token=token())
    return Path(p).read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dest", default="pulled")
    ap.add_argument("--manifest", default="plan/cells_v1.json")
    args = ap.parse_args()

    cells = json.loads(Path(args.manifest).read_text())
    if args.all:
        targets = cells
    elif args.cell_id:
        targets = [c for c in cells if c["cell_id"] == args.cell_id]
        if not targets:
            sys.exit(f"[FATAL] {args.cell_id} not in {args.manifest}")
    else:
        sys.exit("[FATAL] pass --cell-id or --all")

    failures = 0
    for c in targets:
        try:
            res = pull(c, args.dest, _fetch)
        except Exception as e:
            print(f"[SKIP] {c['cell_id']}: {type(e).__name__}")
            continue
        print(f"[{'OK' if res['ok'] else 'INCOMPLETE'}] {c['cell_id']} -> {res['dir']}")
        for n in res["bad"]:
            print(f"    corrupt: {n}")
        for n in res["incomplete"]:
            print(f"    never uploaded: {n}")
        failures += 0 if res["ok"] else 1

    if failures:
        sys.exit(f"[FATAL] {failures} cell(s) incomplete or corrupt")


if __name__ == "__main__":
    main()
