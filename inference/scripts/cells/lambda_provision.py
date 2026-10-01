"""LOCAL. Provision, prepare and tear down Lambda Cloud boxes.

Lambda is the GPU Frequency sheet's provider because vast.ai blocks
`nvidia-smi -lgc`. It differs from vast in ways that have already cost money, so
each one is handled here rather than remembered:

  * It is a bare VM, user `ubuntu`, not a container. Docker needs sudo.
  * There is no `label` field and no wall-clock cap of its own -- an instance
    runs until someone terminates it. `tools/lambda_guard.py` is the only guard.
  * The account is SHARED with the team (14 keys). Only instances whose name
    starts with llmpl- are ever touched.
  * FABRIC STATE MUST BE CHECKED BEFORE USE. On 2026-09-07 a box came up with
    an H100 passed through from an HGX baseboard WITHOUT its NVSwitches;
    fabricmanager waited forever for a registration that could never happen and
    every CUDA call died with `error 802: system not yet initialized`. nvidia-smi
    looked perfectly healthy. That box billed for ~20 minutes and ran nothing.

    usage: lambda_provision.py launch --cells a,b,c
           lambda_provision.py setup --instance <id>
           lambda_provision.py check --instance <id>
           lambda_provision.py terminate --instance <id>
           lambda_provision.py list
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import pins                                        # noqa: E402
from common import load_env                        # noqa: E402

API = "https://cloud.lambda.ai/api/v1"
STATE = ROOT / "plan" / "lambda_instances.jsonl"
KEY = str(Path.home() / ".ssh" / "campaign_ed25519")
PREFIX = "llmpl-"
INSTANCE_TYPE = "gpu_1x_h100_sxm5"      # the pin: H100 SXM5, NOT the PCIe part
SSH_KEY_NAME = "llmpl-campaign"


def _curl(method, path, body=None):
    load_env()
    k = os.environ["LAMBDA_API_KEY"]
    cmd = ["curl", "-s", "-u", f"{k}:", f"{API}{path}", "--max-time", "90"]
    if body is not None:
        cmd += ["-X", method, "-H", "Content-Type: application/json", "-d", json.dumps(body)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    try:
        return json.loads(out or "{}")
    except Exception:
        return {"_raw": out[:400]}


def _ssh(ip, cmd, timeout=300):
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", "-i", KEY,
         f"ubuntu@{ip}", cmd], capture_output=True, text=True, timeout=timeout)


def capacity():
    d = _curl("GET", "/instance-types").get("data", {}).get(INSTANCE_TYPE, {})
    return [r["name"] for r in d.get("regions_with_capacity_available", [])]


def check(ip):
    """Refuse a box whose GPU fabric never initialised. Returns (ok, report)."""
    r = _ssh(ip, "nvidia-smi -q | grep -A3 -i 'Fabric' | head -6; echo '---'; "
                 "nvidia-smi --query-gpu=name,clocks.current.memory,clocks.max.memory "
                 "--format=csv,noheader; echo '---'; "
                 "python3 -c 'import ctypes;ctypes.CDLL(\"libcuda.so.1\").cuInit(0)' "
                 "2>&1 | tail -1 || true")
    txt = r.stdout or ""
    bad = "In Progress" in txt          # fabric never completed -> CUDA error 802
    return (not bad), txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["launch", "setup", "check", "terminate", "list"])
    ap.add_argument("--instance")
    ap.add_argument("--cells", default="")
    ap.add_argument("--region")
    a = ap.parse_args()

    if a.action == "list":
        for i in _curl("GET", "/instances").get("data", []):
            mine = str(i.get("name") or "").startswith(PREFIX)
            print(f"  {i.get('id')[:12]} {i.get('name'):<28} {i.get('status'):<10} "
                  f"{i.get('ip') or '-':<16} "
                  f"${i.get('instance_type', {}).get('price_cents_per_hour', 0)/100:.2f}/hr"
                  f"{'' if mine else '   (NOT OURS - leave alone)'}")
        return

    if a.action == "launch":
        regs = capacity()
        if not regs:
            sys.exit(f"[FATAL] no {INSTANCE_TYPE} capacity in any region right now")
        region = a.region or regs[0]
        cells = [c for c in a.cells.split(",") if c]
        r = _curl("POST", "/instance-operations/launch", {
            "region_name": region, "instance_type_name": INSTANCE_TYPE,
            "ssh_key_names": [SSH_KEY_NAME],
            "name": f"{PREFIX}gpufreq-{int(time.time()) % 100000}"})
        if "data" not in r:
            sys.exit(f"[FATAL] launch failed: {json.dumps(r)[:300]}")
        iid = r["data"]["instance_ids"][0]
        with open(STATE, "a") as fh:
            fh.write(json.dumps({"instance": iid, "region": region,
                                 "cells": cells, "launched_ts": time.time()}) + "\n")
        print(f"[OK] launched {iid} in {region} for {len(cells)} cell(s)")
        return

    if a.action in ("check", "setup", "terminate"):
        inst = _curl("GET", f"/instances/{a.instance}").get("data", {})
        ip = inst.get("ip")
        if a.action == "terminate":
            r = _curl("POST", "/instance-operations/terminate",
                      {"instance_ids": [a.instance]})
            print("[OK] terminated" if "data" in r else f"[FAIL] {json.dumps(r)[:200]}")
            return
        if not ip:
            sys.exit(f"[FATAL] {a.instance} has no ip yet (status {inst.get('status')})")
        ok, report = check(ip)
        print(report.strip())
        if not ok:
            sys.exit("[FATAL] GPU fabric state is 'In Progress' - this box cannot run "
                     "CUDA (error 802) and never will. Terminate it and relaunch; it is "
                     "a Lambda-side host fault, not something a reboot fixes.")
        print("[OK] fabric healthy")
        if a.action == "check":
            return

        # ---- setup ----
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
        _ssh(ip, "sudo mkdir -p /opt/campaign/plan /data/run && "
                 "sudo chown -R ubuntu:ubuntu /opt/campaign /data")
        tar = subprocess.Popen(
            ["tar", "czf", "-", "-C", str(ROOT / "scripts"), "--exclude", ".env",
             "--exclude", "*.json", "--exclude", "__pycache__", "--exclude", "._*",
             "--exclude", ".DS_Store", "."],
            stdout=subprocess.PIPE, env={**os.environ, "COPYFILE_DISABLE": "1"})
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes",
                        "-i", KEY, f"ubuntu@{ip}",
                        "rm -rf /opt/campaign/scripts && mkdir -p /opt/campaign/scripts "
                        "&& tar xzf - -C /opt/campaign/scripts"],
                       stdin=tar.stdout, capture_output=True)
        tar.wait()
        load_env()
        cred = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        for src, dst in ((ROOT / "scripts" / ".env", "/opt/campaign/scripts/.env"),
                         (cred, f"/opt/campaign/scripts/{cred.name}"),
                         # cell.py/status.py resolve parents[2]/plan/cells_v1.json --
                         # delivering it to /opt/campaign/cells_v1.json made every one
                         # of them fail on 2026-09-07.
                         (ROOT / "plan" / "cells_v1.json", "/opt/campaign/plan/cells_v1.json")):
            subprocess.run(["scp", "-o", "StrictHostKeyChecking=no",
                            "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes",
                            "-i", KEY, "-q", str(src), f"ubuntu@{ip}:{dst}"],
                           capture_output=True)
        # camp.sh at a stable top-level path: it is the documented entry point
        # in every dispatch prompt, and the image's ENTRYPOINT is `vllm`, so a
        # bare `docker run` parses the command as vllm arguments.
        _ssh(ip, "cp /opt/campaign/scripts/cells/camp.sh /opt/campaign/camp.sh "
                 "&& chmod +x /opt/campaign/camp.sh")
        _ssh(ip, f"cd /opt/campaign && "
                 f"sed -i 's|^GOOGLE_APPLICATION_CREDENTIALS=.*|"
                 f"GOOGLE_APPLICATION_CREDENTIALS=/opt/campaign/scripts/{cred.name}|' "
                 f"scripts/.env && echo 'CAMPAIGN_COMMIT={commit}' >> scripts/.env")
        _ssh(ip, "set -a; . /opt/campaign/scripts/.env; set +a; "
                 "echo \"$DOCKERHUB_RO_TOKEN\" | sudo docker login -u \"$DOCKERHUB_USERNAME\" "
                 f"--password-stdin >/dev/null 2>&1; nohup sudo docker pull {pins.CAMPAIGN_IMAGE} "
                 "> /tmp/pull.log 2>&1 &")
        v = _ssh(ip, "echo scripts=$(find /opt/campaign/scripts -name '*.py' | wc -l) "
                     "manifest=$([ -f /opt/campaign/plan/cells_v1.json ] && echo present) "
                     "cred=$([ -f /opt/campaign/scripts/*.json ] && echo present)")
        print(v.stdout.strip())
        print(f"[OK] {a.instance} prepared at {ip}, commit {commit[:12]}, image pulling")


if __name__ == "__main__":
    main()
