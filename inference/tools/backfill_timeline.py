"""LOCAL ONLY, run once. Rebuild timeline_<cell>.csv and the power profile for
cells that already ran.

Why this exists: `export_record.py` builds `timeline_<cell>.csv` and
`run_record.json`, and until commit ea860d7 **nothing ever called it** -- no
main, no __main__ guard, no importer. So the timeline was never written,
`plot_run._read_timeline` returned [], and the `arrivals / s` and `active reqs`
panels of every power profile rendered as empty axes.

No measurement was lost. `requests_<cell>.jsonl` and `vllm_metrics.jsonl` were
captured and uploaded for every cell; only the assembly step was missing. This
reconstructs the two artifacts from those files.

It deliberately does NOT touch `summary.json`, the ledger, or any recorded
value. It reads the uploaded artifacts, writes the timeline, and re-renders the
figure. By default nothing is uploaded -- inspect the output, then decide.

    uv run python tools/backfill_timeline.py                 # all ledger cells
    uv run python tools/backfill_timeline.py --cell-id <id>  # just one
    uv run python tools/backfill_timeline.py --upload        # send them

--upload writes the rebuilt figure as power_profile_<cell>_v2.png ALONGSIDE the
original, never over it: the original shipped with a verified checksums.json,
and overwriting it would leave the stored set disagreeing with its own manifest.
timeline_<cell>.csv and run_record.json are new filenames, so they simply land.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "cells", "telemetry", "workload", "upload"):
    sys.path.insert(0, str(ROOT / "scripts" / sub))

import pins                                          # noqa: E402
from common import load_env                          # noqa: E402
from export_record import build_run_record, build_timeline   # noqa: E402
from plot_run import plot_run                        # noqa: E402
from status import fetch_records                     # noqa: E402
from summarise_run import aggregate, load_capture    # noqa: E402

OUT_ROOT = ROOT / "build_out" / "backfill"


def artifact_dir_for(api, cell_id):
    """Every uploaded file for one cell, pulled into a local directory."""
    from huggingface_hub import hf_hub_download
    files = [f for f in api.list_repo_files(pins.RUNS_REPO, repo_type="dataset")
             if f"/{cell_id}/" in f or f.endswith(f"/{cell_id}")]
    files = [f for f in files if not f.startswith("runs/")]
    if not files:
        return None, []
    dest = OUT_ROOT / cell_id
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for f in files:
        p = hf_hub_download(repo_id=pins.RUNS_REPO, filename=f,
                            repo_type="dataset")
        target = dest / Path(f).name
        target.write_bytes(Path(p).read_bytes())
        got.append(target.name)
    return dest, got


def backfill(cell_id, cell_row, api):
    dest, got = artifact_dir_for(api, cell_id)
    if dest is None:
        return f"{cell_id}: no artifacts found in {pins.RUNS_REPO}"

    meta_p = dest / "run_meta.json"
    if not meta_p.exists():
        return f"{cell_id}: run_meta.json missing from the upload"
    run_meta = json.loads(meta_p.read_text())
    t0 = run_meta["window_start_ts_ns"] / 1e9
    t1 = run_meta["window_end_ts_ns"] / 1e9

    reqs = []
    rp = dest / f"requests_{cell_id}.jsonl"
    if rp.exists():
        reqs = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
    samples = []
    mp = dest / "vllm_metrics.jsonl"
    if mp.exists():
        samples = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
    if not reqs and not samples:
        return f"{cell_id}: neither requests nor metrics uploaded - cannot rebuild"

    tl = build_timeline(reqs, samples, t0, t1, pins.WARMUP_S)
    tl_path = dest / f"timeline_{cell_id}.csv"
    with open(tl_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tl[0].keys()))
        w.writeheader()
        w.writerows(tl)

    # run_record.json was never produced either, for the same reason.
    caps = sorted(dest.glob(f"dcgm_{cell_id}_gpu*.csv"))
    rows = [r for p in caps for r in load_capture(p)]
    if rows:
        agg = aggregate(rows, t0, t1)
        (dest / "run_record.json").write_text(
            json.dumps(build_run_record(cell_row, run_meta, agg), indent=1) + "\n")

    fig = plot_run(dest, cell_row, run_meta,
                   dest / f"power_profile_{cell_id}.png")

    arrivals = sum(r["requests_arrived"] for r in tl)
    active = [r["active_requests"] for r in tl if r["active_requests"] is not None]
    return (f"{cell_id}: {len(tl)} bins, {arrivals} arrivals binned, "
            f"{len(active)} bins with active_requests "
            f"(peak {max(active) if active else 0:.0f}) -> {fig}")


def upload_alongside(cell_id, dest, api):
    """Upload the rebuilt artifacts NEXT TO the originals, never over them.

    The original power_profile_<cell>.png shipped with a verified
    checksums.json, so overwriting it would leave the stored set disagreeing
    with its own manifest. `_v2` keeps the seam visible: the original stays as
    the artifact that run produced, and the rebuilt one is plainly a later
    reconstruction.
    """
    files = [f for f in api.list_repo_files(pins.RUNS_REPO, repo_type="dataset")
             if f.endswith(f"/{cell_id}/summary.json")]
    if not files:
        return f"{cell_id}: cannot locate the artifact folder"
    folder = files[0].rsplit("/", 1)[0]
    sent = []
    for local, remote in ((dest / f"power_profile_{cell_id}.png",
                           f"power_profile_{cell_id}_v2.png"),
                          (dest / f"timeline_{cell_id}.csv",
                           f"timeline_{cell_id}.csv"),
                          (dest / "run_record.json", "run_record.json")):
        if not local.exists():
            continue
        api.upload_file(path_or_fileobj=str(local),
                        path_in_repo=f"{folder}/{remote}",
                        repo_id=pins.RUNS_REPO, repo_type="dataset",
                        commit_message=f"backfill {remote} for {cell_id}")
        sent.append(remote)
    return f"{cell_id}: uploaded {', '.join(sent)} -> {folder}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id")
    ap.add_argument("--upload", action="store_true",
                    help="upload the rebuilt artifacts ALONGSIDE the originals")
    a = ap.parse_args()
    load_env()
    from huggingface_hub import HfApi
    import os
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    cells = json.loads((ROOT / "plan" / "cells_v1.json").read_text())
    by_id = {c["cell_id"]: c for c in cells}

    ids = [a.cell_id] if a.cell_id else [r["cell_id"] for r in fetch_records()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for cid in sorted(ids):
        row = by_id.get(cid)
        if row is None:
            print(f"[SKIP] {cid} not in the manifest")
            continue
        try:
            print("[OK] " + backfill(cid, row, api))
            if a.upload:
                print("[OK] " + upload_alongside(cid, OUT_ROOT / cid, api))
        except Exception as e:
            print(f"[FAIL] {cid}: {type(e).__name__}: {e}")
    tail = ("uploaded alongside the originals as _v2" if a.upload
            else "NOTHING UPLOADED - pass --upload to send them")
    print(f"\nfigures under {OUT_ROOT} - {tail}")


if __name__ == "__main__":
    main()
