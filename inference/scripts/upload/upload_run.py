"""ON INSTANCE, after summarise_run. Push a cell's artifacts to both sinks.

    python scripts/upload/upload_run.py --cell-id <id> --out /data/run

Uploading is decoupled from run success (PIPELINE_POSTMORTEM R14): a cell whose
gates failed still uploads, because the failure is evidence. record_run.py
carries the outcome separately.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import hashlib
import json
from pathlib import Path

import pins

CHECKSUMS_NAME = "checksums.json"


def dest_prefix(cell_row):
    """artifacts/<sheet>/<gpu>/<model>/<cell_id>, identical on both sinks.

    The model leaf is the id's last segment: a '/' in 'Qwen/Qwen3-32B' would
    otherwise create an extra folder level that neither sink expects.
    """
    model = cell_row["model"].split("/")[-1]
    return f"artifacts/{cell_row['sheet']}/{cell_row['gpu']}/{model}/{cell_row['cell_id']}"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_checksums(run_dir):
    """sha256 and byte size for every file in the run directory."""
    out = {}
    for p in sorted(Path(run_dir).iterdir()):
        if p.is_file() and p.name != CHECKSUMS_NAME:
            out[p.name] = {"sha256": _sha256(p), "size": p.stat().st_size}
    return out


def write_checksums(run_dir, cell_row):
    """Write checksums.json BEFORE upload; it ships with the set and is what
    the local pull and the data team's import manifest verify against."""
    manifest = {"cell_id": cell_row["cell_id"], "dest": dest_prefix(cell_row),
                "files": file_checksums(run_dir), "incomplete": []}
    (Path(run_dir) / CHECKSUMS_NAME).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def verify_remote(manifest, remote_listing):
    """Names that did not arrive, or arrived at the wrong size."""
    have = {f["name"]: int(f.get("size") or 0) for f in remote_listing}
    missing = []
    for name, meta in manifest["files"].items():
        if name not in have or have[name] != meta["size"]:
            missing.append(name)
    return sorted(missing)


def mark_incomplete(run_dir, manifest, missing):
    """Record the gap in checksums.json so a partial set is data, not an
    absence a reader has to notice (R13)."""
    manifest = dict(manifest, incomplete=sorted(missing))
    (Path(run_dir) / CHECKSUMS_NAME).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def upload(run_dir, cell_row, hf_put, hf_put_many, drive_put, drive_list):
    """Checksum, send to both sinks, verify, retry the missing, report.

    Sinks are injected so this is testable without a network. hf_put takes
    (local_path, repo_path); drive_put takes (local_path); drive_list returns
    the destination folder's current contents.
    """
    run_dir = Path(run_dir)
    manifest = write_checksums(run_dir, cell_row)
    prefix = manifest["dest"]

    names = list(manifest["files"]) + [CHECKSUMS_NAME]
    for attempt in (1, 2):
        # DRIVE FIRST, AND INDEPENDENTLY OF HF. These used to run as
        # hf_put(p); drive_put(p) inside one loop, so a HuggingFace failure on
        # the first file aborted the whole upload and DRIVE NEVER GOT WRITTEN
        # EITHER. On 2026-09-03 an HF 429 therefore left cells in NEITHER sink
        # rather than one -- the opposite of what two sinks are for. Drive has
        # never rate-limited us, so it goes first and its success no longer
        # depends on HF's.
        drive_err = None
        for name in names:
            try:
                drive_put(run_dir / name)
            except Exception as e:
                drive_err = drive_err or e
                print(f"[WARN] drive {name}: {type(e).__name__}: {e}")
        hf_ok = True
        try:
            hf_put_many(run_dir, prefix, names)
        except Exception as e:
            hf_ok = False
            print(f"[WARN] hf upload gave up: {type(e).__name__}: {e}")
            print("[..] Drive is authoritative for this cell; HF will be "
                  "backfilled by tools/flush_pending.py")
        missing = verify_remote(manifest, drive_list())
        if not missing:
            # Drive holds a verified copy. hf_pending says whether HF still
            # owes this cell, so tools/flush_pending.py can backfill it later
            # WITHOUT the instance being alive.
            return {"ok": True, "missing": [], "dest": prefix,
                    "hf_pending": bool(drive_err) or not hf_ok,
                    "files": len(manifest["files"]) + 1}
        # R9: retry only what did not arrive, rather than re-sending everything
        names = missing
        print(f"[WARN] attempt {attempt}: {len(missing)} file(s) missing, retrying")

    mark_incomplete(run_dir, manifest, missing)
    drive_put(run_dir / CHECKSUMS_NAME)
    hf_put(run_dir / CHECKSUMS_NAME, f"{prefix}/{CHECKSUMS_NAME}")
    return {"ok": False, "missing": missing, "dest": prefix,
            "files": len(manifest["files"]) + 1}


def _sinks(cell_row):
    """Real HF and Drive sinks. Imported lazily so the tests never need them."""
    import os
    from huggingface_hub import HfApi
    import drive as drv
    from common import drive_root_folder_id, token

    api, tok = HfApi(), token()

    def hf_put(path, repo_path):
        api.upload_file(path_or_fileobj=str(path), path_in_repo=repo_path,
                        repo_id=pins.RUNS_REPO, repo_type="dataset", token=tok,
                        commit_message=f"upload {repo_path}")

    def hf_put_many(run_dir, prefix, names):
        """ONE commit for the whole cell, with short backoff on 429.

        upload_file commits per call, so a 12-file cell burned 12 of the
        repo's 128 commits/hour and record_run.py added a 13th. A single
        10-cell agent needs ~130 and exceeds the ceiling BY ITSELF, before any
        concurrency -- which is exactly what happened with three agents live.
        upload_folder sends the set as one commit, taking a cell from 13 to 2.

        The backoff is short on purpose. The 429 text advertises an hour, but
        measured behaviour is a rolling window that frees continuously: one
        cell uploaded cleanly between two failures of another, and a blocked
        cell went through 20 minutes later. Retrying every 60-120s drains a
        backlog; waiting out the advertised hour wastes it.
        """
        import time
        allow = [f"{n}" for n in names]
        # ~3.5 minutes of HF budget, then give up and let the caller proceed
        # Drive-only. Holding a GPU at $3-8/hr waiting on a rate limit costs
        # more than backfilling HF later from Drive, and Drive is the sink that
        # verifies. The operator set this policy on 2026-09-03.
        for i, delay in enumerate((0, 30, 60, 120)):
            if delay:
                print(f"[..] hf retry in {delay}s (attempt {i+1})")
                time.sleep(delay)
            try:
                api.upload_folder(folder_path=str(run_dir), path_in_repo=prefix,
                                  repo_id=pins.RUNS_REPO, repo_type="dataset",
                                  token=tok, allow_patterns=allow,
                                  commit_message=f"upload {prefix} ({len(names)} files)")
                return
            except Exception as e:
                if "429" not in str(e) and "rate limit" not in str(e).lower():
                    raise
                last = e
        raise last

    svc = drv.build_service()
    drive_id = os.environ["DRIVE_ID"]
    folder = drv.ensure_path(svc, drive_id, drive_root_folder_id(),
                             dest_prefix(cell_row).split("/"))

    def drive_put(path):
        drv.put_file(svc, drive_id, folder, path)

    def drive_list():
        return drv.list_folder(svc, drive_id, folder)

    return hf_put, hf_put_many, drive_put, drive_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--out", default="/data/run")
    ap.add_argument("--manifest", default="plan/cells_v1.json")
    args = ap.parse_args()

    cells = json.loads(Path(args.manifest).read_text())
    cell_row = next((c for c in cells if c["cell_id"] == args.cell_id), None)
    if cell_row is None:
        sys.exit(f"[FATAL] {args.cell_id} not in {args.manifest}")

    run_dir = Path(args.out)
    if not run_dir.exists() or not any(p.is_file() for p in run_dir.iterdir()):
        sys.exit(f"[FATAL] {run_dir} has no artifacts to upload")

    res = upload(run_dir, cell_row, *_sinks(cell_row))
    print(f"[{'OK' if res['ok'] else 'FAIL'}] {res['files']} file(s) -> {res['dest']}")
    if not res["ok"]:
        print(f"[FATAL] {len(res['missing'])} file(s) never arrived: "
              + ", ".join(res["missing"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
