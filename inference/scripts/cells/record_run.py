"""ON INSTANCE, last step of a cell. Record what happened.

    python scripts/record_run.py --cell-id <id> --outcome ok \
        --artifacts kvquant/h100/... --flags ""

Recording FAILURE is the point: a cell that ran and failed to upload must be
distinguishable from one that never ran, which is what inferring completion
from artifact presence cannot do (PIPELINE_POSTMORTEM R12).
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
import sys
from datetime import datetime, timezone

import pins
from common import token

OUTCOMES = ("ok", "flagged", "run_failed", "upload_failed", "skipped")


def _git_commit():
    """The commit the RUNNING code came from.

    On an instance there is no .git, so `git rev-parse` fails and this used to
    return None -- losing per-run code provenance silently, which is exactly
    what the smoke/pin workflow depends on. push-creds now writes
    scripts/CAMPAIGN_COMMIT when it delivers the tree; prefer that, and fall
    back to git locally.
    """
    stamp = Path(__file__).resolve().parents[1] / "CAMPAIGN_COMMIT"
    try:
        v = stamp.read_text().strip()
        if v:
            return v
    except Exception:
        pass
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def build_record(cell_id, outcome, instance=None, flags=None,
                 artifacts=None, note=None, gpu_observed=None):
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    return {
        "cell_id": cell_id, "outcome": outcome,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "instance": instance or os.environ.get("INSTANCE_ID") or socket.gethostname(),
        "gpu_observed": gpu_observed,
        "vllm_version": pins.VLLM_VERSION,
        "git_commit": _git_commit(),
        "dataset_revision": pins.DATASETS_REVISION,
        "flags": flags or [], "artifacts": artifacts, "note": note,
    }


def push(record):
    """Write runs/<cell_id>.json to the runs repo, and to Drive when configured."""
    from huggingface_hub import HfApi
    import time
    body = json.dumps(record, indent=1).encode()
    # Retry on 429 rather than losing the ledger row. The ledger lives in the
    # SAME repo as the artifacts, so it shares the 128 commits/hour ceiling --
    # on 2026-09-03 six cells ran clean and had no ledger row because
    # record_run was the commit that tipped over. An unrecorded cell is
    # indistinguishable from one that never ran, which is the failure
    # PIPELINE_POSTMORTEM R6 is about.
    #
    # Short waits on purpose: the limit is a rolling window that frees
    # continuously, not an hourly bucket. One cell uploaded cleanly between two
    # failures of another, and a blocked cell went through 20 minutes later.
    last = None
    for delay in (0, 30, 60, 120):
        if delay:
            print(f"[..] ledger write rate-limited, retrying in {delay}s")
            time.sleep(delay)
        try:
            HfApi().upload_file(
                path_or_fileobj=body, path_in_repo=f"runs/{record['cell_id']}.json",
                repo_id=pins.RUNS_REPO, repo_type="dataset", token=token(),
                commit_message=f"record {record['cell_id']}: {record['outcome']}")
            break
        except Exception as e:
            last = e
            if "429" not in str(e) and "rate limit" not in str(e).lower():
                raise
    else:
        # HF is still refusing after ~3.5 minutes. Do NOT lose the record and do
        # NOT hold the instance: queue it locally and let the run continue.
        # An unrecorded cell is indistinguishable from one that never ran
        # (PIPELINE_POSTMORTEM R6), which is how a completed cell gets paid for
        # twice -- so the row must survive even when its sink will not take it.
        # Queue it in DRIVE, not on the instance. A pending row written to
        # /opt/campaign dies with the box, which would defeat the whole point.
        # Drive is the sink that has never rate-limited us, so it becomes the
        # durable store for rows HF will not yet take.
        tmp = Path("/tmp") / f"{record['cell_id']}.json"
        tmp.write_bytes(body)
        try:
            import sys as _s
            _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "upload"))
            import drive as drv
            from common import drive_root_folder_id
            svc = drv.build_service()
            did = os.environ["DRIVE_ID"]
            folder = drv.ensure_path(svc, did, drive_root_folder_id(), ["runs_pending"])
            drv.put_file(svc, did, folder, tmp)
            print(f"[WARN] HF ledger write rate-limited; row queued in Drive at "
                  f"runs_pending/{record['cell_id']}.json")
            print("[..] drain with: uv run python tools/flush_pending.py")
        except Exception as e:
            print(f"[FATAL] HF refused the ledger row AND Drive queueing failed "
                  f"({type(e).__name__}: {e}). The cell is measured and uploaded "
                  f"but UNRECORDED - copy {tmp} off this box before it dies.")
            raise
        finally:
            tmp.unlink(missing_ok=True)
        return
    print(f"[OK] recorded {record['cell_id']} -> {pins.RUNS_REPO}")

    if os.environ.get("DRIVE_ROOT_FOLDER_ID"):
        sys.exit("[FATAL] Drive sink is declared but not implemented yet - "
                 "unset DRIVE_ROOT_FOLDER_ID or wait for the uploader subsystem")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--outcome", required=True, choices=OUTCOMES)
    ap.add_argument("--artifacts")
    ap.add_argument("--note")
    ap.add_argument("--gpu-observed")
    ap.add_argument("--flags", default="")
    args = ap.parse_args()
    flags = [f for f in args.flags.split("|") if f]
    push(build_record(args.cell_id, args.outcome, flags=flags,
                      artifacts=args.artifacts, note=args.note,
                      gpu_observed=args.gpu_observed))


if __name__ == "__main__":
    main()
