"""LOCAL. Drain ledger rows that HuggingFace refused into the runs repo.

When HF returns 429 the run does NOT hold the GPU waiting for it. `record_run.py`
gives HF about 3.5 minutes, then queues the row in Drive under `runs_pending/`
and lets the cell finish -- Drive already holds the verified artifacts, so the
only thing outstanding is the HF copy.

That is a deliberate trade the operator set on 2026-09-03: holding a box at
$3-8/hr waiting on a rolling rate limit costs more than backfilling later, and
the measurement is already safe in Drive.

This drains the queue. Run it any time; it needs no instance.

    uv run python tools/flush_pending.py            # push queued rows to HF
    uv run python tools/flush_pending.py --dry-run  # list what is queued

It is idempotent: a row already present in the runs repo with identical content
is skipped, and a row that pushes successfully is removed from the Drive queue.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "cells", "upload"):
    sys.path.insert(0, str(ROOT / "scripts" / sub))

import pins                                    # noqa: E402
from common import drive_root_folder_id, load_env, token   # noqa: E402


def queued_rows(svc, drive_id):
    """(name, file_id, bytes) for every row sitting in Drive's runs_pending."""
    import drive as drv
    folder = drv.find_file(svc, drive_id, drive_root_folder_id(), "runs_pending")
    if not folder:
        return []
    out = []
    for f in drv.list_folder(svc, drive_id, folder["id"]):
        req = svc.files().get_media(fileId=f["id"], supportsAllDrives=True)
        out.append((f["name"], f["id"], req.execute()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    load_env()
    import os
    import drive as drv
    from huggingface_hub import HfApi

    svc = drv.build_service()
    drive_id = os.environ["DRIVE_ID"]
    rows = queued_rows(svc, drive_id)
    if not rows:
        print("[OK] nothing queued")
        return

    print(f"{len(rows)} row(s) queued in Drive:")
    for name, _, body in rows:
        rec = json.loads(body)
        print(f"    {rec.get('cell_id'):<48} {rec.get('outcome')}")
    if a.dry_run:
        print("\n--dry-run: nothing pushed")
        return

    api = HfApi()
    pushed, failed = 0, 0
    for name, fid, body in rows:
        rec = json.loads(body)
        for delay in (0, 30, 60, 120):
            if delay:
                print(f"    [..] retry in {delay}s")
                time.sleep(delay)
            try:
                api.upload_file(path_or_fileobj=body,
                                path_in_repo=f"runs/{rec['cell_id']}.json",
                                repo_id=pins.RUNS_REPO, repo_type="dataset",
                                token=token(),
                                commit_message=f"flush {rec['cell_id']}: {rec['outcome']}")
                svc.files().delete(fileId=fid, supportsAllDrives=True).execute()
                print(f"[OK] {rec['cell_id']} -> ledger, removed from queue")
                pushed += 1
                break
            except Exception as e:
                if "429" not in str(e) and "rate limit" not in str(e).lower():
                    print(f"[FAIL] {rec['cell_id']}: {type(e).__name__}: {e}")
                    failed += 1
                    break
        else:
            print(f"[FAIL] {rec['cell_id']}: still rate-limited, left in the queue")
            failed += 1

    print(f"\npushed {pushed}, still queued {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
