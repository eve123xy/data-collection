"""LOCAL, read-only. Does every recorded run actually exist on BOTH sinks?

The ledger says a cell ran. That is a claim, not evidence. This checks the
claim against what is actually stored in the HuggingFace runs repo AND the
Google Shared Drive, file by file, and reports every disagreement.

It is deliberately independent of upload_run.py's own verification: that runs
on the instance, at upload time, against the directory it just wrote. This runs
later, from a different machine, against what the sinks actually hold now.

    uv run python tools/audit_sinks.py            # every recorded cell
    uv run python tools/audit_sinks.py --sheet "GPU Frequency"
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "cells"))
sys.path.insert(0, str(ROOT / "scripts" / "upload"))

import pins                                        # noqa: E402
from common import load_env, drive_root_folder_id  # noqa: E402


def dest_prefix(row):
    """Must match upload_run.dest_prefix exactly."""
    return (f"artifacts/{row['sheet']}/{row['gpu']}/"
            f"{row['model'].split('/')[-1]}/{row['cell_id']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet")
    ap.add_argument("--deep", action="store_true",
                    help="also compare file SIZES across sinks - matching names "
                         "do not prove matching content")
    a = ap.parse_args()
    load_env()

    cells = {c["cell_id"]: c for c in json.loads((ROOT / "plan" / "cells_v1.json").read_text())}
    from status import fetch_records
    records = {r["cell_id"]: r for r in fetch_records() if r.get("cell_id")}

    # ---- HF side: one listing for the whole repo ----
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    hf_files, hf_size = set(), {}
    for t in api.list_repo_tree(pins.RUNS_REPO, repo_type="dataset", recursive=True):
        if getattr(t, "size", None) is not None:      # blobs only, not folders
            hf_files.add(t.path)
            hf_size[t.path] = t.size

    # ---- Drive side ----
    import drive as drv
    svc = drv.build_service()
    drive_id = os.environ["DRIVE_ID"]
    root = drive_root_folder_id()

    folder_cache = {}

    def drive_children(parts):
        """Folder id for a path, walking and caching. None if any part missing."""
        key = tuple(parts)
        if key in folder_cache:
            return folder_cache[key]
        fid = root
        for p in parts:
            got = None
            for f in drv.list_folder(svc, drive_id, fid):
                if f["name"] == p and f["mimeType"].endswith("folder"):
                    got = f["id"]
                    break
            if got is None:
                folder_cache[key] = None
                return None
            fid = got
        folder_cache[key] = fid
        return fid

    targets = [cid for cid in records
               if cid in cells and (not a.sheet or cells[cid]["sheet"] == a.sheet)]
    targets.sort()

    print(f"auditing {len(targets)} recorded cell(s) against BOTH sinks\n")
    problems, ok_both = [], 0

    for cid in targets:
        row, rec = cells[cid], records[cid]
        prefix = dest_prefix(row)
        hf = {f[len(prefix) + 1:] for f in hf_files if f.startswith(prefix + "/")}

        fid = drive_children(prefix.split("/"))
        gd, gd_size = set(), {}
        if fid:
            for f in drv.list_folder(svc, drive_id, fid):
                if f["mimeType"].endswith("folder"):
                    continue
                gd.add(f["name"])
                if f.get("size") is not None:
                    gd_size[f["name"]] = int(f["size"])

        outcome = rec.get("outcome")
        if not hf and not gd:
            problems.append((cid, outcome, "NOTHING on either sink", len(hf), len(gd)))
        elif not gd:
            problems.append((cid, outcome, "MISSING FROM DRIVE entirely", len(hf), 0))
        elif not hf:
            problems.append((cid, outcome, "MISSING FROM HF entirely", 0, len(gd)))
        elif hf != gd:
            only_hf = sorted(hf - gd)
            only_gd = sorted(gd - hf)
            detail = []
            if only_hf:
                detail.append(f"HF-only: {', '.join(only_hf[:3])}"
                              + (f" +{len(only_hf)-3}" if len(only_hf) > 3 else ""))
            if only_gd:
                detail.append(f"Drive-only: {', '.join(only_gd[:3])}"
                              + (f" +{len(only_gd)-3}" if len(only_gd) > 3 else ""))
            problems.append((cid, outcome, "; ".join(detail), len(hf), len(gd)))
        elif a.deep:
            # Names matching does not mean content matches: a truncated or
            # half-written upload keeps its name. Compare sizes.
            bad = []
            for n in sorted(hf & gd):
                h, g = hf_size.get(f"{prefix}/{n}"), gd_size.get(n)
                if h is not None and g is not None and h != g:
                    bad.append(f"{n} (hf {h} vs drive {g} bytes)")
            if bad:
                problems.append((cid, outcome, "SIZE MISMATCH: " + "; ".join(bad[:3]),
                                 len(hf), len(gd)))
            else:
                ok_both += 1
        else:
            ok_both += 1

    print(f"  {ok_both} cell(s): identical file sets on HF and Drive")
    if problems:
        print(f"  {len(problems)} cell(s) with a discrepancy:\n")
        for cid, outcome, what, nh, ng in problems:
            print(f"  {cid}")
            print(f"      outcome={outcome}  hf={nh} files  drive={ng} files")
            print(f"      {what}")
    else:
        print("\n  No discrepancies. Every recorded cell is on both sinks with "
              "the same files.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
