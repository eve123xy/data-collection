"""LOCAL. One line per Lambda instance that needs attention. For Monitor.

Lambda has no `label` field and no wall-clock cap of its own -- an instance runs
until someone terminates it, and `end_date` is not a thing. So the only guard is
this one, and it has to do what sweep() does on vast plus what nothing did there
either: catch the box that is up and idle.

That distinction is the point (see fleet_guard / 2.40):
  OVER_CAP+IDLE  past MAX_INSTANCE_HOURS and running nothing -> terminate
  IDLE           up over 30 min with no vLLM and no workload -> an agent
                 probably stopped between measuring and uploading
  UNSAFE         past the cap but still busy -> do NOT terminate blindly; a box
                 was destroyed mid-work on a stale read and lost a staged cell

Silent while healthy, so every line printed is actionable.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pins                                   # noqa: E402
from common import load_env                   # noqa: E402

API = "https://cloud.lambda.ai/api/v1"
STATE = ROOT / "plan" / "lambda_instances.jsonl"
KEY = str(Path.home() / ".ssh" / "campaign_ed25519")
PREFIX = "llmpl-"


def _api(path):
    load_env()
    k = os.environ.get("LAMBDA_API_KEY", "")
    try:
        out = subprocess.run(["curl", "-s", "-u", f"{k}:", f"{API}{path}"],
                             capture_output=True, text=True, timeout=60).stdout
        return json.loads(out or "{}").get("data", [])
    except Exception:
        return []


def _probe(ip):
    """(busy, idle_minutes) for a box. (None, None) if unreachable.

    Two independent signals, because each one alone has now failed:

    * campaign processes. The original probe was
      `pgrep -c -f 'vllm serve|run_workload.py'` over ssh, which MATCHES ITSELF:
      the remote shell's own command line contains the pattern, so the count was
      never zero and the IDLE branch could not fire. A box sat idle for two
      hours at $4.29/hr while this guard reported healthy. Bracketing one
      character of each alternative stops the self-match, and `ps | grep` is
      used rather than pgrep because pgrep did not reliably see processes
      inside the campaign container.

    * artifact freshness. Processes are the wrong question when an agent stops
      BETWEEN measuring and uploading -- that is the failure this guard exists
      for, and it leaves no process behind. If nothing has been written to
      /data/run for a while, the box is doing nothing regardless of what is or
      is not in the process table.
    """
    cmd = ("ps -eo args --no-headers | grep -cE 'vllm serv[e]|run_workloa[d].py|"
           "dcgm_captur[e]' || true; "
           "find /data/run -type f -newermt '-20 minutes' 2>/dev/null | wc -l")
    try:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-i", KEY,
             f"ubuntu@{ip}", cmd],
            capture_output=True, text=True, timeout=60)
        nums = [int(x) for x in (r.stdout or "").split() if x.isdigit()]
        if len(nums) < 2:
            return None, None
        procs, fresh = nums[0], nums[1]
        return (procs > 0 or fresh > 0), fresh
    except Exception:
        return None, None


def main():
    launched = {}
    if STATE.exists():
        for l in STATE.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                launched[r["instance"]] = r
    for i in _api("/instances"):
        name = str(i.get("name") or "")
        if not name.startswith(PREFIX):
            continue                    # shared team account: not ours
        if i.get("status") != "active":
            continue
        iid, ip = i.get("id"), i.get("ip")
        rec = launched.get(iid)
        hrs = (time.time() - rec["launched_ts"]) / 3600 if rec else None
        dph = i.get("instance_type", {}).get("price_cents_per_hour", 0) / 100
        busy, fresh = _probe(ip)
        over = hrs is not None and hrs > pins.MAX_INSTANCE_HOURS
        age = f"{hrs:.1f}h" if hrs is not None else "age?"
        if over and busy is False:
            print(f"OVER_CAP+IDLE {iid[:12]} {name} {age} ${dph:.2f}/hr -> terminate now")
        elif over:
            print(f"UNSAFE {iid[:12]} {name} {age} ${dph:.2f}/hr still busy"
                  f"(unreachable={busy is None}) -> check before terminating")
        elif busy is False and (hrs is None or hrs > 0.5):
            print(f"IDLE {iid[:12]} {name} {age} ${dph:.2f}/hr no campaign process "
                  f"and nothing written to /data/run in 20 min "
                  f"-> agent may have stopped between measuring and uploading")
        elif busy is None:
            print(f"UNREACHABLE {iid[:12]} {name} {age} ${dph:.2f}/hr "
                  f"-> ssh probe failed; cannot tell if it is working")


if __name__ == "__main__":
    main()
