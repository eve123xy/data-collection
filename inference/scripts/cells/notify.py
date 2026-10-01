"""ntfy.sh notifications for campaign events.

    python scripts/cells/notify.py provisioned --cells "a,b,c" --extra "offer 123"
    python scripts/cells/notify.py completed   --cell <id>
    python scripts/cells/notify.py flagged     --cell <id> --reason "idle baseline dirty"
    python scripts/cells/notify.py failed      --cell <id> --reason "idle card"
    python scripts/cells/notify.py step        --cell <id> --extra "vLLM ready"

Every message is sent only AFTER the thing it describes has been verified, never
on intent. A notification that a run completed, sent before the gates ran, is
worse than no notification.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import subprocess

import pins


def send(message):
    """Post one line to the campaign topic. Never fatal - a lost notification
    must not cost a run."""
    try:
        subprocess.run(["curl", "-s", "-d", message, pins.NTFY_TOPIC],
                       capture_output=True, timeout=15)
        print(f"[notify] {message}")
        return True
    except Exception as e:
        print(f"[WARN] notification failed ({type(e).__name__}): {message}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event", choices=["provisioned", "completed", "flagged", "failed",
                                      "step", "destroyed", "alert"])
    ap.add_argument("--cell", default="")
    ap.add_argument("--cells", default="")
    ap.add_argument("--reason", default="")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()

    msg = {
        "provisioned": f"Instance Provisioned for {a.cells or a.cell}",
        "completed":   f"Cell Run Completed {a.cell}",
        # `flagged` exists because the ledger has three outcomes and this channel
        # had two. Sending `failed` for a flagged cell reads as "FAILED Cell Run"
        # to anyone who only sees the notification, which overstates a cell whose
        # gates passed -- and a channel that overstates is one people stop
        # trusting in the direction that matters.
        "flagged":     f"Cell Run Completed WITH FLAGS {a.cell}: {a.reason}",
        "failed":      f"FAILED Cell Run {a.cell} REASON FOR FAIL: {a.reason}",
        "step":        f"Step {a.cell}: {a.extra}",
        "destroyed":   f"Instance destroyed for {a.cells or a.cell}",
        "alert":       f"ALERT {a.cell}: {a.reason}",
    }[a.event]
    if a.extra and a.event not in ("step",):
        msg += f" | {a.extra}"
    send(msg)


if __name__ == "__main__":
    main()
