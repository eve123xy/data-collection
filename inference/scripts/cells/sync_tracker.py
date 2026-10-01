"""Project the run ledger into a status copy of the tracker, in Drive.

    python scripts/cells/sync_tracker.py                 # regenerate from the ledger
    python scripts/cells/sync_tracker.py --set <cell_id> --status Running
    python scripts/cells/sync_tracker.py --dry-run

The ledger (runs/<cell_id>.json) is canonical; this sheet is a projection of it,
written by ONE writer - the orchestrator - after each result. That is what keeps
the two from disagreeing the first time a run succeeds and its upload fails.

The operator's own tracker is NEVER opened for writing. openpyxl does not
round-trip everything an xlsx can hold - charts, images and some conditional
formatting can be dropped silently - so this maintains a separate status copy
beside it and regenerates that.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import io
import json
from pathlib import Path

import openpyxl

import pins
from common import drive_root_folder_id, load_env

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "plan" / "New_Manual_Tracker_v2 (1).xlsx"
MANIFEST = ROOT / "plan" / "cells_v1.json"
LOCAL_COPY = ROOT / "plan" / "tracker_run_status.xlsx"
DRIVE_NAME = "New_Manual_Tracker_v2 - run status.xlsx"

# What each ledger outcome writes into the grid cell. Deliberately avoids "Done"
# and "Ran": the tracker already uses those for the archive's completed cells.
OUTCOME_TEXT = {"ok": "Completed", "flagged": "Flagged",
                "run_failed": "FAILED", "upload_failed": "FAILED upload",
                "skipped": "Skipped", "running": "Running"}


def ledger_records(runs_dir):
    """Ledger records, keyed by cell_id.

    THE LEDGER LIVES IN THE HF RUNS REPO, not on this machine. This used to read
    only a local runs/ directory that nothing ever populates -- record_run.py
    uploads straight to HF -- so the tracker silently reported "0 status cells"
    no matter how many cells had landed. It is the same source status.py reads;
    a local runs/ dir, if one exists from pull_runs.py, is merged on top.
    """
    out = {}
    try:
        sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from status import fetch_records
        for r in fetch_records():
            if r.get("cell_id"):
                out[r["cell_id"]] = r
    except Exception as e:
        print(f"[WARN] could not read the HF ledger ({type(e).__name__}: {e}); "
              f"falling back to local {runs_dir}/ only")
    d = Path(runs_dir)
    if d.exists():
        for p in sorted(d.glob("*.json")):
            try:
                r = json.loads(p.read_text())
                out[r["cell_id"]] = r
            except Exception:
                pass
    return out


# Columns the campaign added after the operator built the tracker. apply_status
# writes a run outcome into each cell's tracker_ref coordinate, but a coordinate
# in a column the sheet has no header for reads as a status floating in empty
# space -- so the header and a "To Run" placeholder are stamped first.
#
# Only ever written into the STATUS COPY. Every column here is empty in the
# operator's sheet; tools/build_cells.py refuses to synthesise into an occupied
# one (_assert_column_free).
NEW_COLUMNS = [
    {"sheet": "GPU Frequency", "col": 9, "header": "345",
     "header_rows": [2, 11, 20], "style_from_col": 3,
     "data_rows": [3, 4, 5, 6, 12, 13, 14, 15, 21, 22, 23, 24],
     "why": "Added 2026-09-07 so the SM-clock axis spans the full supported "
            "range. 345 MHz is the H100 SXM5 hardware floor; the original "
            "1980/1600/1200/900/600 ladder left the bottom 255 MHz unsampled. "
            "J/token vs clock is expected to be U-shaped, so a ladder stopping "
            "at 600 could miss the minimum and report the right half of a U as "
            "a clean monotonic trend."},
    {"sheet": "BurstGPT", "col": 8, "header": "TP4 (capacity)",
     "header_rows": [2, 9], "style_from_col": 3,
     "data_rows": [3, 4, 5, 6, 10, 11, 12, 13],
     "blank_text": "n/a - TP2 served this trace",
     "why": "Added 2026-09-05. The TP2 cells in this column's rows offer more "
            "load than a TP2 server can serve (11.66 and 10.28 req/s against a "
            "measured 7.2 for Qwen3-32B and 6.6 for Llama-70B), so the client "
            "backlog diverges. The TP2 rows are RETAINED and reported as the "
            "capacity result; these rows were ADDED to reach the offered load."},
    {"sheet": "BurstGPT", "col": 9, "header": "TP8 (capacity)",
     "header_rows": [2, 9], "style_from_col": 3,
     "data_rows": [3, 4, 5, 6, 10, 11, 12, 13],
     "blank_text": "n/a - TP2 served this trace",
     "why": "Added 2026-09-05, same reason as TP4 (capacity). TP8 returns "
            "despite being cut from the v1 TP sweep, for a different purpose: "
            "not a point on the TP axis, but the hardware needed to make a "
            "fixed offered load servable."},
]


def stamp_new_columns(wb, cells):
    """Header + 'To Run' placeholder for columns the operator's sheet lacks.

    Refuses to overwrite anything. A manifest cell whose grid coordinate is
    blank gets 'To Run' so the column shows the cells exist before any of them
    has run; apply_status then overwrites that with the real outcome.
    """
    from copy import copy as _copy
    n_hdr = n_new = 0
    for spec in NEW_COLUMNS:
        ws = wb[spec["sheet"]]
        for r in spec["header_rows"]:
            c = ws.cell(r, spec["col"])
            if c.value is not None:
                continue
            c.value = spec["header"]
            src = ws.cell(r, spec["style_from_col"])
            c._style = _copy(src._style)
            c.comment = openpyxl.comments.Comment(spec["why"], "campaign")
            n_hdr += 1
    for cell in cells:
        ref = cell["tracker_ref"]
        ws = wb[ref["sheet"]]
        c = ws.cell(ref["row"], ref["col"])
        if c.value is None:
            c.value = "To Run"
            spec = next((s for s in NEW_COLUMNS if s["sheet"] == ref["sheet"]
                         and s["col"] == ref["col"]), None)
            if spec:
                c._style = _copy(ws.cell(ref["row"], spec["style_from_col"])._style)
            n_new += 1
    # A blank in an added column is ambiguous: no cell, or a cell nobody ran?
    # Say which. Mirrors an "N/A (...)" from the reference column when the model
    # does not fit at all, otherwise the column's own reason.
    n_na = 0
    for spec in NEW_COLUMNS:
        ws = wb[spec["sheet"]]
        for r in spec.get("data_rows", []):
            c = ws.cell(r, spec["col"])
            if c.value is not None:
                continue
            ref = ws.cell(r, spec["style_from_col"]).value
            text = (str(ref) if isinstance(ref, str) and ref.startswith("N/A")
                    else spec.get("blank_text"))
            if not text:
                continue
            c.value = text
            c._style = _copy(ws.cell(r, spec["style_from_col"])._style)
            n_na += 1
    return n_hdr, n_new, n_na


def apply_status(wb, cells, records, overrides=None):
    """Write status into each cell's tracker_ref coordinate. Returns a summary."""
    overrides = overrides or {}
    counts, written = {}, 0
    for c in cells:
        cid = c["cell_id"]
        rec = records.get(cid)
        status = overrides.get(cid) or (
            OUTCOME_TEXT.get(rec["outcome"], rec["outcome"]) if rec else None)
        if not status:
            continue
        ref = c["tracker_ref"]
        ws = wb[ref["sheet"]]
        note = ""
        if rec and rec.get("outcome") in ("run_failed", "upload_failed"):
            reason = (rec.get("flags") or [rec.get("note") or ""])[0]
            note = f": {reason}" if reason else ""
        ws.cell(ref["row"], ref["col"]).value = f"{status}{note}"[:250]
        counts[status] = counts.get(status, 0) + 1
        written += 1
    return written, counts


# Which code path each sheet sits on. A smoke group validates ONE path, so each
# path is smoked and pinned separately before the sheets on it mass-launch.
SMOKE_PATHS = [
    {"path": "Closed-loop, TP1, single GPU",
     "sheets": "KV Quant, Concurrency, Weight Quant, ISL_OSL block 1",
     "first_exercises": "forced OSL via ignore_eos; server-side --max-num-seqs",
     "run_plan": "run_plans/smoke-kvquant-h100-2x2.md",
     "cells": ["kvquant_h100_qwen3-8b_bf16_b32",
               "kvquant_h100_qwen3-8b_fp8-e4m3_b32",
               "kvquant_h100_qwen3-14b_bf16_b32",
               "kvquant_h100_qwen3-14b_fp8-e4m3_b32"]},
    {"path": "Multi-GPU, TP>=2",
     "sheets": "TP Number; the TP2 rows on KV Quant and Weight Quant",
     "first_exercises": "tensor-parallel launch; nvlink_tx/rx become meaningful; "
                        "per-GPU telemetry and power summation across 2 devices",
     "run_plan": "run_plans/smoke-tp2-h100-multigpu.md",
     "cells": ["tp_h100_qwen3-14b_tp2_b32"]},
    {"path": "closed/free - free generation",
     "sheets": "ISL_OSL free blocks (9 cells)",
     "first_exercises": "ignore_eos=False, max_tokens 2048; realized token "
                        "counts as the length covariates",
     "run_plan": "run_plans/smoke-drivers-h200-openloop.md",
     "cells": ["islosl_h200_qwen3-8b_chat-sharegpt_b32"]},
    {"path": "closed/thinking - thinking mode",
     "sheets": "ISL_OSL thinking (6), Poisson thinking (3)",
     "first_exercises": "MAX_MODEL_LEN 16384; the streaming reasoning delta "
                        "field - unanswerable on any KV Quant cell",
     "run_plan": "run_plans/smoke-drivers-h200-openloop.md",
     "cells": ["islosl_h200_qwen3-8b_chat-sharegpt"]},
    {"path": "poisson - arrival process",
     "sheets": "Poisson (21)",
     "first_exercises": "unbounded in-flight; queueing, admission, saturation",
     "run_plan": "run_plans/smoke-drivers-h200-openloop.md",
     "cells": ["poisson_h200_qwen3-8b_r0.25_unbounded"]},
    {"path": "replay - BurstGPT trace",
     "sheets": "BurstGPT (33); the driver half of GPU Frequency (36)",
     "first_exercises": "trace replay against wall-clock arrivals",
     "run_plan": "run_plans/smoke-drivers-h200-openloop.md",
     "cells": ["burstgpt_h200_qwen3-8b_burst_unbounded"]},
    {"path": "Lambda + SM clock control",
     "sheets": "GPU Frequency (all 54)",
     "first_exercises": "different provider; nvidia-smi -lgc clock pinning",
     "run_plan": "", "cells": []},
]

SMOKE_TAB = "Smoke Runs"


def write_smoke_tab(wb, records):
    """(Re)build the Smoke Runs tab in the STATUS COPY from the ledger.

    Self-updating rather than hand-maintained: status and the pinned commit are
    read from runs/<cell_id>.json, so the tab cannot drift from what actually
    ran. Written only to the status copy -- the operator's tracker is never
    opened for writing.
    """
    if SMOKE_TAB in wb.sheetnames:
        del wb[SMOKE_TAB]
    ws = wb.create_sheet(SMOKE_TAB)
    headers = ["Path", "Sheets it unlocks", "What it first exercises",
               "Smoke cells", "Done", "Outcomes", "Pinned commit",
               "Instrument pinned?", "Run plan"]
    ws.append(headers)
    for c in ws[1]:
        c.font = openpyxl.styles.Font(bold=True)

    for spec in SMOKE_PATHS:
        recs = [records.get(c) for c in spec["cells"]]
        got = [r for r in recs if r]
        outcomes = sorted({r["outcome"] for r in got}) or ["-"]
        commits = sorted({(r.get("git_commit") or "")[:12] for r in got if r.get("git_commit")})
        n = len(spec["cells"])
        done = f"{len(got)}/{n}" if n else "not started"
        # A path is pinned only when every smoke cell landed ok on ONE commit.
        pinned = ("no - not started" if not n else
                  "no - smoke incomplete" if len(got) < n else
                  "no - smoke did not all pass" if outcomes != ["ok"] else
                  "no - cells span >1 commit" if len(commits) > 1 else
                  f"YES @ {commits[0]}" if commits else "no - commit not recorded")
        ws.append([spec["path"], spec["sheets"], spec["first_exercises"],
                   "\n".join(spec["cells"]) or "-", done, ", ".join(outcomes),
                   ", ".join(commits) or "-", pinned, spec["run_plan"] or "-"])

    for col, width in zip("ABCDEFGHI", (30, 42, 52, 40, 12, 14, 28, 26, 38)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=1):
        for c in row:
            c.alignment = openpyxl.styles.Alignment(vertical="top", wrap_text=True)
    return len(SMOKE_PATHS)


def upload_copy(local_path):
    """Create or update the status copy in Drive. Never touches the original."""
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "upload"))
    import drive as drv
    load_env()
    import os
    svc = drv.build_service()
    drive_id = os.environ["DRIVE_ID"]
    folder = drive_root_folder_id()
    hit = drv.find_file(svc, drive_id, folder, DRIVE_NAME)
    tmp = local_path.parent / DRIVE_NAME
    tmp.write_bytes(local_path.read_bytes())
    try:
        drv.put_file(svc, drive_id, folder, tmp)
    finally:
        tmp.unlink(missing_ok=True)
    return "updated" if hit else "created"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    # Comma-separated: the sheet is regenerated from the ledger on every run,
    # so a one-cell --set meant only the most recent marker survived and a
    # dispatched batch showed one Running cell out of six.
    ap.add_argument("--set", dest="cell_id",
                    help="cell_id, or a comma-separated list")
    ap.add_argument("--status", default="Running")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()

    if not SOURCE.exists():
        sys.exit(f"[FATAL] {SOURCE} missing")
    cells = json.loads(MANIFEST.read_text())
    records = ledger_records(a.runs)
    overrides = {c: a.status for c in (a.cell_id or "").split(",") if c}

    wb = openpyxl.load_workbook(SOURCE)          # read-only intent: never saved back
    n_hdr, n_new, n_na = stamp_new_columns(wb, cells)
    written, counts = apply_status(wb, cells, records, overrides)

    if a.dry_run:
        print(f"[dry-run] would add {n_hdr} column header(s) and {n_new} "
              f"'To Run' placeholder(s), {n_na} n/a marker(s), then write {written} status "
              f"cell(s): {counts}")
        return

    n_smoke = write_smoke_tab(wb, records)
    wb.save(LOCAL_COPY)
    # A silent "0 status cells" while the ledger holds records is how the tracker
    # sat empty for a whole session and still printed [OK]. Make it loud.
    if records and not written:
        print(f"[FAIL] ledger holds {len(records)} record(s) but ZERO status "
              f"cells were written - the tracker does not reflect the ledger. "
              f"Check that cell_ids in runs/*.json match plan/cells_v1.json.")
        sys.exit(2)
    print(f"[OK] {written} status cell(s) from {len(records)} ledger record(s) "
          f"-> {LOCAL_COPY.name}  {counts}")
    print(f"[OK] '{SMOKE_TAB}' tab rebuilt: {n_smoke} code path(s)")
    print(f"[OK] {n_hdr} added-column header(s), {n_new} 'To Run' "
          f"placeholder(s), {n_na} n/a marker(s)")
    if not a.no_upload:
        try:
            what = upload_copy(LOCAL_COPY)
            print(f"[OK] {what} '{DRIVE_NAME}' in Drive")
        except Exception as e:
            print(f"[WARN] Drive upload failed ({type(e).__name__}: {e}) - "
                  f"the local copy at {LOCAL_COPY} is still correct")


if __name__ == "__main__":
    main()
