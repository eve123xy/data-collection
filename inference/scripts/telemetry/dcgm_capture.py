"""ON INSTANCE. Capture GPU telemetry via `dcgmi dmon` to one CSV per GPU.

    python scripts/dcgm_capture.py --cell-id <id> --out /data/run
    python scripts/dcgm_capture.py --cell-id <id> --out /data/run --idle 30

Files are named by cell_id, and the output directory is cleared first. Together
those make the archive's failure mode unrepresentable: it selected a capture
with sorted(glob("dcgm_*.csv"))[0] from a directory retrieval never cleared, and
paired a working run's tokens with a hung run's power trace - 117 W instead of
599 W, one cell off by 106x (OPERATIONAL_LEARNINGS section 3.1).
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dcgm_fields import (ALL_FIELDS, FIELD_IDS, FIELD_NAMES,
                         compute_cap_of_gpu0, fields_for, parse_dmon_line)

CSV_COLUMNS = ["ts", "entity_type", "gpu_id"] + FIELD_NAMES


def clear_output_dir(out_dir, cell_id):
    """Remove every artifact from a previous run in this directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
    print(f"[OK] cleared {out_dir} for {cell_id}")


def capture_stream(lines, out_dir, field_names, stem):
    """Consume dmon lines, writing one CSV per gpu_id. Returns rows per GPU."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["ts", "entity_type", "gpu_id"] + field_names
    handles, writers, counts = {}, {}, {}
    try:
        for line in lines:
            row = parse_dmon_line(line, field_names)
            if row is None:
                continue
            gid = row["gpu_id"]
            if gid not in writers:
                fh = open(out_dir / f"dcgm_{stem}_gpu{gid}.csv", "w", newline="")
                handles[gid] = fh
                writers[gid] = csv.DictWriter(fh, fieldnames=columns)
                writers[gid].writeheader()
                counts[gid] = 0
            row["ts"] = f"{time.time():.6f}"
            writers[gid].writerow(row)
            counts[gid] += 1
            handles[gid].flush()
    finally:
        for fh in handles.values():
            fh.close()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--out", default="/data/run")
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--idle", type=int, default=0,
                    help="capture this many seconds of idle baseline, then exit")
    ap.add_argument("--clear", action="store_true",
                    help="clear the output directory first (do this once per cell)")
    args = ap.parse_args()

    if not shutil.which("dcgmi"):
        sys.exit("[FATAL] dcgmi not on PATH - is the campaign image in use?")
    if args.clear:
        clear_output_dir(args.out, args.cell_id)

    stem = f"idle_{args.cell_id}" if args.idle else args.cell_id
    # Must match bootstrap's architecture-conditional set. dmon takes ONE list,
    # so a single unwatchable field kills the entire stream -- an A100 asking
    # for the DCP fields writes zero rows, not partial columns.
    _, field_names, field_ids = fields_for(compute_cap_of_gpu0())
    cmd = ["dcgmi", "dmon", "-e", ",".join(field_ids), "-d", str(args.interval_ms)]
    print(f"[..] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    deadline = time.time() + args.idle if args.idle else None
    lines = iter(proc.stdout.readline, "")
    if deadline:
        def bounded():
            for l in lines:
                if time.time() > deadline:
                    return
                yield l
        counts = capture_stream(bounded(), args.out, field_names, stem)
    else:
        counts = capture_stream(lines, args.out, field_names, stem)
    proc.terminate()
    print(f"[OK] captured {sum(counts.values()):,} rows across "
          f"{len(counts)} GPU(s) -> {args.out}")


if __name__ == "__main__":
    main()
