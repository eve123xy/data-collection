"""LOCAL. One line per instance that needs attention. Designed for Monitor.

Prints nothing while the fleet is healthy, so every line it emits is actionable.
Three conditions, each of which has cost real money in this campaign:

  OVER_CAP  past pins.MAX_INSTANCE_HOURS. sweep() exists and finds these, but
            nothing ever invoked it -- two boxes ran 8.4h and 7.7h against a
            4h cap and wasted $48.89 (2.44).
  IDLE      no vLLM and no workload process. An agent that ends its turn after
            measuring leaves the box billing with a finished cell unuploaded;
            one sat at 0% GPU for ~4 hours holding a recorded-nowhere result.
  UNSAFE    over cap BUT still working. Do NOT destroy these blindly: a box was
            destroyed mid-work on a stale ledger read, losing a staged cell.

The distinction between IDLE and UNSAFE is the whole point. "Past the cap" is
not sufficient grounds to destroy; "past the cap AND doing nothing" is.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "cells"):
    sys.path.insert(0, str(ROOT / "scripts" / sub))

import pins                                      # noqa: E402
from common import load_env                      # noqa: E402

PROV = ROOT / "plan" / "provisions.jsonl"
KEY = str(Path.home() / ".ssh" / "campaign_ed25519")


def _vast(*a):
    try:
        out = subprocess.run(["vastai", *a, "--raw"], capture_output=True,
                             text=True, timeout=120).stdout
        return json.loads(out or "[]")
    except Exception:
        return []


def _busy(iid):
    """True if the box is running a server or workload. None if unreachable."""
    try:
        url = subprocess.run(["vastai", "ssh-url", str(iid)], capture_output=True,
                             text=True, timeout=60).stdout.strip()
        hp = url[len("ssh://"):].split("@")[-1]
        host, _, port = hp.partition(":")
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-i", KEY,
             "-p", port or "22", f"root@{host}",
             "pgrep -c -f 'vllm serve|run_workload.py' || true"],
            capture_output=True, text=True, timeout=60)
        n = (r.stdout or "0").strip().splitlines()[-1:] or ["0"]
        return int(n[0] or 0) > 0
    except Exception:
        return None


def main():
    load_env()
    recs = ({json.loads(l)["instance"]: json.loads(l)
             for l in PROV.read_text().splitlines() if l.strip()}
            if PROV.exists() else {})
    for i in _vast("show", "instances"):
        label = str(i.get("label") or "")
        if not label.startswith(pins.INSTANCE_LABEL_PREFIX):
            continue                       # team account: not ours, leave alone
        if i.get("actual_status") in ("exited", "stopped", None):
            continue
        iid = i["id"]
        r = recs.get(iid)
        hrs = (time.time() - r["launched_ts"]) / 3600 if r else 0.0
        dph = float(i.get("dph_total") or 0)
        over = hrs > pins.MAX_INSTANCE_HOURS
        busy = _busy(iid)
        if over and busy is False:
            print(f"OVER_CAP+IDLE {iid} {hrs:.1f}h ${dph:.2f}/hr "
                  f"cells={','.join((r or {}).get('cells', []))[:90]} -> destroy now")
        elif over:
            print(f"UNSAFE {iid} {hrs:.1f}h ${dph:.2f}/hr still busy(or unreachable={busy is None})"
                  f" -> check before destroying")
        elif busy is False and hrs > 0.5:
            print(f"IDLE {iid} {hrs:.1f}h ${dph:.2f}/hr no server/workload "
                  f"-> agent may have stopped; check for unuploaded cells")


if __name__ == "__main__":
    main()
