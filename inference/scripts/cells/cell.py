"""Print the exact commands for one cell. The agent copies; it never composes.

    python scripts/cell.py --cell-id <id> --print launch
    python scripts/cell.py --cell-id <id> --verify-gpu
    python scripts/cell.py --next
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pins

MANIFEST = Path(__file__).resolve().parents[2] / "plan" / "cells_v1.json"


def load_cells():
    if not MANIFEST.exists():
        sys.exit(f"[FATAL] {MANIFEST} missing - run tools/build_cells.py")
    return json.loads(MANIFEST.read_text())


def get(cell_id):
    for r in load_cells():
        if r["cell_id"] == cell_id:
            return r
    sys.exit(f"[FATAL] unknown cell_id {cell_id!r}")


def launch_cmd(row):
    # --generation-config vllm: WITHOUT this, vLLM loads the model's own
    # generation_config.json as SERVER-SIDE DEFAULTS. Qwen3 ships
    # {top_k: 20, temperature: 0.6, top_p: 0.95}. Our per-request temperature
    # and top_p override theirs, but pins.SAMPLING sends no top_k -- so top_k=20
    # was silently applied to every request, and the sampling policy in effect
    # was not the one the settings contract states. Found on the first smoke
    # cell, 2026-09-02.
    args = list(row["vllm_args"]) + ["--generation-config", "vllm"]
    return "vllm serve " + " ".join(shlex.quote(a) for a in args)


def workload_cmd(row):
    w = row["workload"]
    parts = ["python", "scripts/workload/run_workload.py",
             "--mode", w["mode"], "--model", row["model"],
             "--family", row["family"], "--osl", w["osl"],
             "--cell-id", row["cell_id"], "--duration", str(w["duration"])]
    if w["in_flight"] is not None:
        parts += ["--in-flight", str(w["in_flight"])]
    if w.get("forced_osl"):
        parts += ["--forced-osl", str(w["forced_osl"])]
    if w.get("rate"):
        parts += ["--rate", str(w["rate"])]
    if w.get("scale"):
        parts += ["--scale", str(w["scale"])]
    if w["thinking"]:
        parts += ["--thinking"]
    if row["dataset"]["kind"] == "burstgpt":
        parts += ["--replay", "/data/replay.jsonl"]
    elif w.get("forced_isl"):
        # The ISL_OSL grid needs prompts of EXACTLY forced_isl tokens, not
        # bucket-S as-is. Bucket-S is 129-384 tokens and cannot supply 512 or
        # 2048 at all, so these cells previously ran at whatever it offered --
        # a grid that did not vary its grid. prepare_isl_prompts.py fetches the
        # matching frozen artifact to this path.
        parts += ["--prompts", f"/data/prompts_isl{w['forced_isl']}.jsonl"]
    elif w.get("source"):
        # ISL_OSL blocks 2-3 sweep the DEMAND side: real prompts from three
        # domains, used as-is (run_plan_settings_v2 §1), with the model choosing
        # when to stop. Until this branch existed, `workload.source` was read by
        # nothing and --prompts fell through to bucket-S, so these cells would
        # have served generic prompts while being recorded under their source
        # name. That is why all 12 were blocked.
        #
        # Added 2026-09-04 with the operator's explicit approval, deliberately
        # scoped: it fires ONLY when a cell carries `source`, so every cell
        # without one resolves through exactly the code path it always did.
        art = pins.SOURCE_ARTIFACT[w["source"]]
        parts += ["--prompts", f"/data/prompts_src_{art}.jsonl"]
    return " ".join(shlex.quote(p) for p in parts)


def check_gpu(row, smi_text):
    """Compare `nvidia-smi --query-gpu=name,memory.total` output to the row."""
    lines = [l for l in smi_text.strip().splitlines() if l.strip()]
    errs = []
    if len(lines) != row["expected_gpus"]:
        errs.append(f"expected {row['expected_gpus']} GPU(s), nvidia-smi "
                    f"reports {len(lines)}")
    want = pins.GPUS[row["gpu"]]
    for l in lines:
        if want["name"].split()[0] not in l:
            errs.append(f"expected {want['name']} ({want['variant']}), got {l.strip()!r}")
        mem = [int(t) for t in l.replace(",", " ").split() if t.isdigit()]
        if mem and mem[-1] < want["mem_gb"] * 900:      # MiB, 10% tolerance
            errs.append(f"expected ~{want['mem_gb']}GB, got {mem[-1]} MiB in {l.strip()!r}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id")
    ap.add_argument("--print", dest="what",
                    choices=["launch", "workload", "pre", "post", "json"])
    ap.add_argument("--verify-gpu", action="store_true")
    ap.add_argument("--next", action="store_true")
    args = ap.parse_args()

    if args.next:
        print(load_cells()[0]["cell_id"])
        return

    if not args.cell_id:
        sys.exit("[FATAL] --cell-id is required")
    row = get(args.cell_id)

    if args.verify_gpu:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True)
        errs = check_gpu(row, out)
        if errs:
            for e in errs:
                print("[FATAL]", e)
            sys.exit(1)
        print(f"[OK] GPU matches: {row['expected_gpus']}x {row['gpu']}")
        return

    print({"launch": launch_cmd, "workload": workload_cmd,
           "pre": lambda r: "\n".join(r["pre_launch"]),
           "post": lambda r: "\n".join(r["post_run"]),
           "json": lambda r: json.dumps(r, indent=1)}[args.what](row))


if __name__ == "__main__":
    main()
