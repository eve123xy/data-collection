"""LOCAL. Upload and record cells whose sinks refused while the box was alive.

When HuggingFace rate-limits a run, agents pull the finished artifacts to local
disk before the instance is destroyed. Those cells are fully measured and
gate-passed; all they lack is a copy in the sinks and a row in the ledger.

This replays them from those local directories. No instance is needed.

    uv run python tools/replay_backups.py --dir <backup_dir>            # dry run
    uv run python tools/replay_backups.py --dir <backup_dir> --commit   # do it

Each subdirectory of <backup_dir> is one cell, named by cell_id, holding the
run directory as it was on the box. The outcome and flags are read from that
cell's own summary.json rather than retyped, so the ledger row cannot disagree
with the artifacts it points at.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "cells", "upload"):
    sys.path.insert(0, str(ROOT / "scripts" / sub))

import pins                                          # noqa: E402
from common import load_env                          # noqa: E402
from status import fetch_records                     # noqa: E402


def outcome_of(run_dir):
    """(outcome, flags) from the cell's own summary.json and run_meta.json."""
    summary = json.loads((run_dir / "summary.json").read_text())
    flags = list(summary.get("flags") or [])
    meta_p = run_dir / "run_meta.json"
    if meta_p.exists():
        flags += list(json.loads(meta_p.read_text()).get("flags") or [])
    if summary.get("fatal"):
        return "run_failed", list(summary["fatal"])
    return ("flagged" if flags else "ok"), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    load_env()

    have = {r["cell_id"]: r["outcome"] for r in fetch_records()}
    root = Path(a.dir)
    cells = sorted(d for d in root.iterdir() if d.is_dir())
    print(f"{len(cells)} backed-up cell(s) in {root}\n")

    todo = []
    for d in cells:
        try:
            out, flags = outcome_of(d)
        except Exception as e:
            print(f"  {d.name:<48} UNREADABLE ({type(e).__name__}) - skipping")
            continue
        state = have.get(d.name)
        need = state is None or state == "upload_failed"
        print(f"  {d.name:<48} summary={out:<10} ledger={state or 'MISSING':<14}"
              f"{'  -> REPLAY' if need else '  (already recorded)'}")
        if need:
            todo.append((d, out, flags))

    if not todo:
        print("\nnothing to replay")
        return
    if not a.commit:
        print(f"\n{len(todo)} cell(s) would be replayed. Re-run with --commit.")
        return

    ok = fail = 0
    for d, out, flags in todo:
        print(f"\n=== {d.name} ===")
        up = subprocess.run(
            [sys.executable, str(ROOT / "scripts/upload/upload_run.py"),
             "--cell-id", d.name, "--out", str(d)],
            capture_output=True, text=True, cwd=str(ROOT))
        print((up.stdout or up.stderr).strip()[-400:])
        if up.returncode != 0:
            print(f"[FAIL] upload {d.name}")
            fail += 1
            continue
        dest = None
        for line in (up.stdout or "").splitlines():
            if "->" in line and "artifacts/" in line:
                dest = line.split("->", 1)[1].strip()
        cmd = [sys.executable, str(ROOT / "scripts/cells/record_run.py"),
               "--cell-id", d.name, "--outcome", out]
        if dest:
            cmd += ["--artifacts", dest]
        if flags:
            cmd += ["--flags", " | ".join(flags)]
        rec = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        print((rec.stdout or rec.stderr).strip()[-300:])
        if rec.returncode == 0:
            ok += 1
        else:
            fail += 1
    print(f"\nreplayed {ok}, failed {fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
