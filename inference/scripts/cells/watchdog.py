"""Liveness and schedule check for a live provision.

    python scripts/cells/watchdog.py --instance <iid>
    python scripts/cells/watchdog.py --instance <iid> --probe   # also ssh in

Exists so the orchestrator does not poll by reading logs and forming a judgement
about whether a run "looks healthy". OPERATIONAL_LEARNINGS section 3 is entirely
about runs that completed successfully and produced wrong numbers - eyeballing
is the instrument that failed there. This reports facts; the gates decide
validity.

Exit code is 0 when nothing needs attention, 1 when something does.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import json
import subprocess
import time
from pathlib import Path

import pins

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plan" / "cells_v1.json"
PROVISIONS = ROOT / "plan" / "provisions.jsonl"

# Per-cell wall-clock budget: warm-up + window + drain, plus a fixed allowance
# for image pull, model load and vLLM startup.
STARTUP_BUDGET_S = 20 * 60


def _provision(instance_id):
    if not PROVISIONS.exists():
        return None
    for line in PROVISIONS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if str(r["instance"]) == str(instance_id):
                return r
    return None


def expected_seconds(cell_ids):
    cells = {c["cell_id"]: c for c in json.loads(MANIFEST.read_text())}
    total = 0
    for cid in cell_ids:
        c = cells.get(cid)
        if c:
            total += pins.WARMUP_S + c["workload"]["duration"] + 120   # drain
    return total + STARTUP_BUDGET_S


def check(instance_id, probe=False):
    problems = []
    rec = _provision(instance_id)
    if rec is None:
        problems.append(f"instance {instance_id} is not in provisions.jsonl - "
                        f"launched outside provision.py, so it is untracked")

    out = subprocess.run(["vastai", "show", "instance", str(instance_id), "--raw"],
                         capture_output=True, text=True, timeout=120)
    try:
        inst = json.loads(out.stdout)
    except Exception:
        inst = {}
    # `vastai show instance <id> --raw` returns the instance object ITSELF, not
    # a wrapper with an "instances" key -- that shape belongs to `show
    # instances` (plural). Testing for it meant the watchdog reported EVERY live
    # instance as "not found or preempted", which is the one alarm an
    # orchestrator must never cry wolf on: acting on it would mean destroying a
    # box mid-cell. Key off the id the API actually returns.
    if not inst or inst.get("id") is None:
        problems.append(f"instance {instance_id} not found - it may have been "
                        f"destroyed or preempted")
        _report(instance_id, None, rec, problems)
        return problems

    status = inst.get("actual_status")
    dph = float(inst.get("dph_total") or 0)
    print(f"instance {instance_id}: {status}  ${dph:.3f}/hr  label={inst.get('label')}")

    if status != "running":
        problems.append(f"status is {status!r}, not running")

    if rec:
        elapsed = time.time() - rec["launched_ts"]
        budget = expected_seconds(rec["cells"])
        print(f"  elapsed {elapsed/60:.1f} min of ~{budget/60:.0f} min budget "
              f"for {len(rec['cells'])} cell(s)")
        print(f"  spent so far: ${dph * elapsed / 3600:.2f}")
        if elapsed > budget:
            problems.append(f"{elapsed/60:.0f} min elapsed exceeds the "
                            f"{budget/60:.0f} min budget")
        if elapsed / 3600 > pins.MAX_INSTANCE_HOURS:
            problems.append(f"{elapsed/3600:.1f}h exceeds the "
                            f"{pins.MAX_INSTANCE_HOURS}h hard cap - destroy it")

    if probe and status == "running":
        problems += _probe_phase(instance_id, rec)

    _report(instance_id, status, rec, problems)
    return problems


def _probe_phase(instance_id, rec):
    """Ask the box which artifacts exist, so 'which phase' is a fact not a guess."""
    url = subprocess.run(["vastai", "ssh-url", str(instance_id)],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    if not url.startswith("ssh://"):
        return [f"no ssh url for {instance_id}: {url[:80]}"]
    hostport = url[len("ssh://"):]
    user_host, _, port = hostport.partition(":")
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
           "-p", port or "22", user_host, "ls -1 /data/run 2>/dev/null | head -40"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    files = [f for f in r.stdout.split() if f]
    if not files:
        return ["ssh reachable but /data/run is empty - nothing has started"]
    phase = ("uploaded"   if any(f == "checksums.json" for f in files) else
             "summarised" if any(f == "summary.json" for f in files) else
             "workload"   if any(f.startswith("requests_") for f in files) else
             "capturing"  if any(f.startswith("dcgm_") for f in files) else
             "bootstrapped" if "instance_facts.json" in files else "starting")
    print(f"  phase: {phase}  ({len(files)} artifact(s) in /data/run)")
    return []


def _report(instance_id, status, rec, problems):
    if problems:
        print()
        for p in problems:
            print(f"[ATTENTION] {p}")
    else:
        print("  [OK] nothing needs attention")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--probe", action="store_true",
                    help="ssh in and report which artifacts exist")
    a = ap.parse_args()
    sys.exit(1 if check(a.instance, a.probe) else 0)


if __name__ == "__main__":
    main()
